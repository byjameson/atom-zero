"""Deterministic document version diff.

RAG indexes change when a document changes. This module compares two text
versions at block level (paragraphs separated by blank lines), reports added,
removed and changed blocks, and exposes a similarity ratio. It is stdlib-only,
makes no model calls and does not modify the document memory.

    from atomzero.version_diff import audit_document_diff
    report = audit_document_diff(old_text, new_text)
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from pathlib import Path

__all__ = [
    "audit_document_diff",
    "diff_blocks",
    "split_blocks",
]

_SPACE_RE = re.compile(r"[ \t]+")
_BLANK_LINE_RE = re.compile(r"\n\s*\n+")


def split_blocks(text: str) -> list[str]:
    """Metni paragraf bloklarına ayırır ve boşlukları normalize eder."""
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    blocks = []
    for raw in _BLANK_LINE_RE.split(normalized):
        block = "\n".join(
            _SPACE_RE.sub(" ", line).strip() for line in raw.splitlines()
        ).strip()
        if block:
            blocks.append(block)
    return blocks


def diff_blocks(old_blocks: list[str], new_blocks: list[str]) -> dict:
    """İki blok listesini ekleme/silme/değişiklik olarak karşılaştırır."""
    matcher = difflib.SequenceMatcher(
        a=old_blocks, b=new_blocks, autojunk=False)
    added: list[str] = []
    removed: list[str] = []
    changed: list[dict] = []
    unchanged = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            unchanged += i2 - i1
        elif tag == "insert":
            added.extend(new_blocks[j1:j2])
        elif tag == "delete":
            removed.extend(old_blocks[i1:i2])
        elif tag == "replace":
            old_part = old_blocks[i1:i2]
            new_part = new_blocks[j1:j2]
            for index in range(min(len(old_part), len(new_part))):
                changed.append({
                    "old": old_part[index],
                    "new": new_part[index],
                })
            removed.extend(old_part[len(new_part):])
            added.extend(new_part[len(old_part):])
    return {
        "added": added,
        "removed": removed,
        "changed": changed,
        "unchanged": unchanged,
        "similarity": round(matcher.ratio(), 4),
    }


def audit_document_diff(
    old_text: str,
    new_text: str,
    *,
    old_label: str = "old",
    new_label: str = "new",
) -> dict:
    """İki metin sürümü için deterministik bir fark raporu üretir."""
    old_blocks = split_blocks(old_text)
    new_blocks = split_blocks(new_text)
    diff = diff_blocks(old_blocks, new_blocks)
    changed = bool(diff["added"] or diff["removed"] or diff["changed"])
    return {
        "old_label": str(old_label or "old"),
        "new_label": str(new_label or "new"),
        "changes": diff,
        "summary": {
            "old_blocks": len(old_blocks),
            "new_blocks": len(new_blocks),
            "added": len(diff["added"]),
            "removed": len(diff["removed"]),
            "changed": len(diff["changed"]),
            "unchanged": diff["unchanged"],
            "similarity": diff["similarity"],
            "changed": changed,
        },
        "weights_changed": False,
        "method": "deterministic_block_diff_v1",
    }


def _render(report: dict) -> str:
    summary = report["summary"]
    lines = [
        f"Document diff: {report['old_label']} -> {report['new_label']}",
        f"similarity {summary['similarity']:.3f}; "
        f"+{summary['added']} -{summary['removed']} "
        f"~{summary['changed']} = {summary['unchanged']}",
    ]
    for block in report["changes"]["added"]:
        lines.append(f"+ {block[:160]}")
    for block in report["changes"]["removed"]:
        lines.append(f"- {block[:160]}")
    for change in report["changes"]["changed"]:
        lines.append(f"~ {change['old'][:160]}")
        lines.append(f"  -> {change['new'][:160]}")
    return "\n".join(lines)


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8-sig")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", required=True, help="old text file")
    parser.add_argument("--new", required=True, help="new text file")
    parser.add_argument("--old-label", default="old")
    parser.add_argument("--new-label", default="new")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        old_text = _read(args.old)
        new_text = _read(args.new)
    except OSError as exc:
        print(f"input error: {exc}", file=sys.stderr)
        return 2
    report = audit_document_diff(
        old_text, new_text,
        old_label=args.old_label, new_label=args.new_label)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(_render(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
