"""Compare two atomzero-self-code reports.

The diff answers one question: did a change resolve findings, introduce new
ones, or merely move code around? It is deterministic, local and read-only.

    atomzero-self-code-diff before.json after.json --json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

__all__ = ["compare_reports", "main"]

_SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}
_FIELDS = ("finding_id", "kind", "severity", "path", "line", "message")


def _validate_report(report, label: str) -> list[dict]:
    if not isinstance(report, dict):
        raise ValueError(f"{label} report must be an object")
    findings = report.get("findings")
    if findings is None:
        findings = []
    if not isinstance(findings, list):
        raise ValueError(f"{label}.findings must be a list")
    for index, finding in enumerate(findings):
        if not isinstance(finding, dict):
            raise ValueError(f"{label}.findings[{index}] must be an object")
        for field in ("kind", "path", "message"):
            if not isinstance(finding.get(field), str):
                raise ValueError(
                    f"{label}.findings[{index}].{field} must be a string")
        line = finding.get("line", 0)
        if isinstance(line, bool) or not isinstance(line, int) or line < 0:
            raise ValueError(
                f"{label}.findings[{index}].line must be a non-negative "
                "integer")
    return findings


def _identity(finding: dict) -> tuple[str, str, str, str]:
    return (finding["kind"], str(finding.get("severity") or ""),
            finding["path"], finding["message"])


def _simple(finding: dict) -> dict:
    return {field: finding.get(field) for field in _FIELDS}


def _sort(items: list[dict]) -> list[dict]:
    return sorted(
        items,
        key=lambda item: (
            _SEVERITY_ORDER.get(str(item.get("severity")), 3),
            str(item.get("kind")), str(item.get("path")),
            int(item.get("line", 0))))


def _pair(old: list[dict], new: list[dict]):
    """Pair findings by stable id first, then by kind/path/message/severity."""
    persisting = []
    remaining_old = list(old)
    remaining_new = list(new)

    for new_finding in list(remaining_new):
        identifier = str(new_finding.get("finding_id") or "").strip()
        if not identifier:
            continue
        match = next(
            (item for item in remaining_old
             if str(item.get("finding_id") or "").strip() == identifier),
            None)
        if match is None:
            continue
        persisting.append(_simple(new_finding))
        remaining_old.remove(match)
        remaining_new.remove(new_finding)

    groups: dict[tuple, tuple[list[dict], list[dict]]] = {}
    for item in remaining_old:
        groups.setdefault(_identity(item), ([], []))[0].append(item)
    for item in remaining_new:
        groups.setdefault(_identity(item), ([], []))[1].append(item)
    for old_group, new_group in groups.values():
        old_group.sort(key=lambda item: int(item.get("line", 0)))
        new_group.sort(key=lambda item: int(item.get("line", 0)))
        for old_finding, new_finding in zip(old_group, new_group):
            persisting.append(_simple(new_finding))
            remaining_old.remove(old_finding)
            remaining_new.remove(new_finding)

    return (persisting, [_simple(item) for item in remaining_old],
            [_simple(item) for item in remaining_new])


def compare_reports(before, after) -> dict:
    """Compare two self-code reports and classify every finding."""
    old = _validate_report(before, "before")
    new = _validate_report(after, "after")
    persisting, resolved, introduced = _pair(old, new)
    summary = {
        "before": len(old),
        "after": len(new),
        "resolved": len(resolved),
        "introduced": len(introduced),
        "persisting": len(persisting),
        "new_high": sum(
            1 for item in introduced if item.get("severity") == "high"),
        "new_medium": sum(
            1 for item in introduced if item.get("severity") == "medium"),
        "resolved_high": sum(
            1 for item in resolved if item.get("severity") == "high"),
        "resolved_medium": sum(
            1 for item in resolved if item.get("severity") == "medium"),
    }
    return {
        "resolved": _sort(resolved),
        "introduced": _sort(introduced),
        "persisting": _sort(persisting),
        "summary": summary,
        "ok": summary["new_high"] == 0 and summary["new_medium"] == 0,
        "auto_apply": False,
        "weights_changed": False,
        "method": "deterministic_self_code_diff_v1",
    }


def _load(path: str) -> dict:
    try:
        raw = sys.stdin.read() if path == "-" else Path(path).read_text(
            encoding="utf-8-sig")
        data = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before")
    parser.add_argument("after")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = compare_reports(_load(args.before), _load(args.after))
    except ValueError as exc:
        print(f"input error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        summary = result["summary"]
        print(
            f"Self-code diff: {summary['resolved']} resolved, "
            f"{summary['introduced']} introduced, "
            f"{summary['persisting']} persisting")
        for item in result["introduced"]:
            print(f"+ {item['severity']} {item['kind']} "
                  f"{item['path']}:{item['line']}")
        for item in result["resolved"]:
            print(f"- {item['severity']} {item['kind']} "
                  f"{item['path']}:{item['line']}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
