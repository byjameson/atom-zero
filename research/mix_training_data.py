"""Build a deterministic copy + content/replay training mix.

Full fine-tuning on copy-only data can improve copy exact match while
forgetting assistant content behaviour. This tool combines one or more copy
curriculum files with zero or more content/replay files, deduplicates the
result, and writes a JSONL training file plus a content-free manifest.

Rows use the repository schema::

    {"user": "...", "assistant": "...", "family": "...", "keys": ["..."]}

``family`` and ``keys`` are optional. Malformed rows are skipped and counted.
Duplicates are detected on the normalized ``(user, assistant)`` pair; the first
occurrence wins.

Weights are sampling multipliers, not target fractions. Each file requests
``round(weight * unique_rows_in_file)`` rows, capped at the rows it has. If a
file is exhausted, remaining rows are filled from other files with spare rows
and the manifest reports the fill. Rows are never duplicated to satisfy a
weight, so a source can never contribute more than its unique row count.

Leakage guard: any input file whose name contains ``eval`` (case-insensitive),
any path with an ``eval`` / ``eval_frozen`` directory component, or any path
under ``data/eval_frozen/`` is refused **before it is opened**. Eval rows are
never read and never mixed into training data.

Usage::

    python research/mix_training_data.py \
        --copy data/copy_curriculum_v1/train.jsonl \
        --content data/content_v1/train.jsonl \
        --copy-weight 1.0 --content-weight 1.0 \
        --max-rows 120000 --seed 13 --shuffle \
        --out data/mixed_replay_v1/train.jsonl \
        --manifest data/mixed_replay_v1/manifest.json

With no ``--content`` files the tool is a copy-only dedupe/cleanup pass.
The output is still a research training file: it is not promoted by this tool
and must pass retention evaluation plus the checkpoint gate before use.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "atomzero.mix_training_data.v1"
DEFAULT_SEED = 13
DEFAULT_WEIGHT = 1.0
ROLES = ("copy", "content")

__all__ = [
    "DEFAULT_SEED",
    "DEFAULT_WEIGHT",
    "MixError",
    "SCHEMA",
    "assert_safe_input",
    "main",
    "mix_rows",
    "normalize_pair",
    "read_source",
    "run_mix",
]


class MixError(ValueError):
    """Invalid mix request; the CLI prints the message and exits non-zero."""


@dataclass
class _Source:
    role: str
    path: Path
    weight: float = 0.0
    rows: list[dict[str, Any]] = field(default_factory=list)
    rows_read: int = 0
    skipped: int = 0
    blank_lines: int = 0
    duplicates: int = 0
    available: int = 0
    requested: int = 0
    selected: int = 0
    trimmed: int = 0
    filled: int = 0
    exhausted: bool = False

    @property
    def final_rows(self) -> int:
        return self.selected + self.filled


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    if not value.is_absolute():
        value = ROOT / value
    return value.resolve()


def _display(path: str | Path) -> str:
    value = _resolve(path)
    try:
        return value.relative_to(ROOT).as_posix()
    except (OSError, ValueError):
        return value.as_posix()


def _eval_reason(path: Path) -> str | None:
    """Return why ``path`` looks like an eval path, or ``None``."""

    parts = [part.lower() for part in path.parts]
    if "eval" in path.name.lower():
        return "file name contains 'eval'"
    if "eval_frozen" in parts or "eval" in parts:
        return "path contains an 'eval' directory"
    for index in range(len(parts) - 1):
        if parts[index] == "data" and parts[index + 1] == "eval_frozen":
            return "path is under data/eval_frozen"
    return None


def assert_safe_input(path: str | Path, *, role: str = "input") -> Path:
    """Resolve and validate a training input without opening it.

    Raises :class:`MixError` for eval-looking paths. Call this for every input
    before reading any file so a later eval path cannot be read first.
    """

    resolved = _resolve(path)
    reason = _eval_reason(resolved)
    if reason is not None:
        raise MixError(
            f"refusing eval {role} {_display(resolved)}: {reason}. "
            "Training mixes may not read eval files; use eval rows only in "
            "retention/checkpoint evaluation.")
    return resolved


def _normalize_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split())


def normalize_pair(user: str, assistant: str) -> tuple[str, str]:
    """Normalize a ``(user, assistant)`` pair for deduplication.

    Whitespace is collapsed and Unicode is NFKC-normalized. Case and
    punctuation are preserved because copy examples can legitimately differ by
    case.
    """

    return _normalize_text(str(user)), _normalize_text(str(assistant))


def _clean_row(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, Mapping):
        return None
    user = raw.get("user")
    assistant = raw.get("assistant")
    if not isinstance(user, str) or not isinstance(assistant, str):
        return None
    if not user.strip() or not assistant.strip():
        return None
    cleaned: dict[str, Any] = {"user": user, "assistant": assistant.strip()}
    if "family" in raw:
        family = raw["family"]
        if not isinstance(family, str) or not family.strip():
            return None
        cleaned["family"] = family.strip()
    if "keys" in raw:
        keys = raw["keys"]
        if (not isinstance(keys, (list, tuple))
                or not all(isinstance(key, str) for key in keys)):
            return None
        cleaned["keys"] = list(keys)
    return cleaned


def read_source(path: str | Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Read one JSONL source, skipping and counting malformed rows.

    The path is leakage-checked first. Invalid JSON, non-objects, missing or
    empty ``user`` / ``assistant``, and invalid optional fields are counted as
    skipped rather than raised.
    """

    resolved = assert_safe_input(path)
    text = resolved.read_text(encoding="utf-8-sig")
    rows: list[dict[str, Any]] = []
    counts = {"rows_read": 0, "skipped": 0, "blank_lines": 0}
    for line in text.splitlines():
        if not line.strip():
            counts["blank_lines"] += 1
            continue
        counts["rows_read"] += 1
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            counts["skipped"] += 1
            continue
        cleaned = _clean_row(raw)
        if cleaned is None:
            counts["skipped"] += 1
            continue
        rows.append(cleaned)
    return rows, counts


def _weight(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MixError(f"{name} must be a number")
    weight = float(value)
    if not math.isfinite(weight):
        raise MixError(f"{name} must be finite")
    if weight < 0.0:
        raise MixError(f"{name} must be >= 0")
    return weight


def _max_rows(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise MixError("max_rows must be an int or None")
    if value < 0:
        raise MixError("max_rows must be >= 0")
    return value


def _seed(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise MixError("seed must be an int")
    return value


def _paths(value: str | Path | Sequence[str | Path] | None,
           name: str) -> list[str | Path]:
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        items = [value]
    else:
        items = list(value)
    if any(not isinstance(item, (str, Path)) for item in items):
        raise MixError(f"{name} paths must be strings or Path objects")
    return items


def _round_half_up(value: float) -> int:
    return int(math.floor(value + 0.5))


def _allocate_proportional(counts: Sequence[int], target: int) -> list[int]:
    """Allocate ``target`` rows across counts, preserving ratios."""

    total = sum(counts)
    if target >= total:
        return list(counts)
    if target <= 0:
        return [0] * len(counts)
    raw = [target * count / total for count in counts]
    allocated = [int(math.floor(value)) for value in raw]
    remaining = target - sum(allocated)
    order = sorted(
        range(len(counts)),
        key=lambda index: (raw[index] - allocated[index], -index),
        reverse=True)
    for index in order[:remaining]:
        allocated[index] += 1
    return allocated


def _dedupe(sources: Sequence[_Source]) -> int:
    seen: set[tuple[str, str]] = set()
    duplicates = 0
    for source in sources:
        kept: list[dict[str, Any]] = []
        for row in source.rows:
            key = normalize_pair(row["user"], row["assistant"])
            if key in seen:
                source.duplicates += 1
                duplicates += 1
                continue
            seen.add(key)
            kept.append(row)
        source.rows = kept
        source.available = len(kept)
    return duplicates


def _fill_shortfall(sources: Sequence[_Source], needed: int) -> int:
    """Round-robin fill rows from sources that still have spare rows."""

    if needed <= 0:
        return 0
    remaining = needed
    while remaining > 0:
        progressed = False
        for source in sources:
            if remaining == 0:
                break
            spare = source.available - source.selected - source.filled
            if source.weight > 0.0 and spare > 0:
                source.filled += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break
    return remaining


def _source_payload(source: _Source) -> dict[str, Any]:
    return {
        "role": source.role,
        "path": _display(source.path),
        "rows_read": source.rows_read,
        "skipped": source.skipped,
        "blank_lines": source.blank_lines,
        "duplicates": source.duplicates,
        "available": source.available,
        "requested": source.requested,
        "selected": source.selected,
        "trimmed": source.trimmed,
        "filled": source.filled,
        "final_rows": source.final_rows,
        "exhausted": source.exhausted,
    }


def mix_rows(copy_paths: str | Path | Sequence[str | Path],
             content_paths: str | Path | Sequence[str | Path] | None = None,
             *, copy_weight: float = DEFAULT_WEIGHT,
             content_weight: float = DEFAULT_WEIGHT,
             max_rows: int | None = None,
             seed: int = DEFAULT_SEED,
             shuffle: bool = False) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load, deduplicate and mix copy/content rows.

    Returns ``(rows, manifest)``. The manifest has no prompt, answer, key or
    value text and no output hash yet; :func:`run_mix` adds the hash after the
    output is written.
    """

    copy_items = _paths(copy_paths, "copy")
    content_items = _paths(content_paths, "content")
    if not copy_items:
        raise MixError("at least one --copy file is required")
    copy_weight = _weight(copy_weight, "copy_weight")
    content_weight = _weight(content_weight, "content_weight")
    max_rows = _max_rows(max_rows)
    seed = _seed(seed)

    inputs = ([("copy", item) for item in copy_items]
              + [("content", item) for item in content_items])
    # Validate every path before opening any of them.
    resolved_inputs = [
        (role, assert_safe_input(item, role=role)) for role, item in inputs
    ]

    sources: list[_Source] = []
    for role, path in resolved_inputs:
        rows, counts = read_source(path)
        sources.append(_Source(
            role=role,
            path=path,
            rows=rows,
            rows_read=counts["rows_read"],
            skipped=counts["skipped"],
            blank_lines=counts["blank_lines"],
        ))

    total_duplicates = _dedupe(sources)
    weights = {"copy": copy_weight, "content": content_weight}
    for source in sources:
        source.weight = weights[source.role]
        source.requested = _round_half_up(
            weights[source.role] * source.available)
        source.selected = min(source.requested, source.available)
        source.exhausted = source.requested > source.available

    selected_total = sum(source.selected for source in sources)
    if max_rows is not None and selected_total > max_rows:
        allocations = _allocate_proportional(
            [source.selected for source in sources], max_rows)
        for source, allocation in zip(sources, allocations):
            source.trimmed = source.selected - allocation
            source.selected = allocation

    desired_total = sum(source.requested for source in sources)
    if max_rows is not None:
        desired_total = min(desired_total, max_rows)
    unfilled = _fill_shortfall(
        sources, max(0, desired_total - sum(s.selected for s in sources)))

    rows: list[dict[str, Any]] = []
    for source in sources:
        start = source.selected
        stop = start + source.filled
        rows.extend(source.rows[:start])
        rows.extend(source.rows[start:stop])
    if shuffle:
        random.Random(seed).shuffle(rows)
    if max_rows is not None and len(rows) > max_rows:
        rows = rows[:max_rows]

    role_totals: dict[str, dict[str, int]] = {}
    for role in ROLES:
        role_sources = [source for source in sources if source.role == role]
        role_totals[role] = {
            "sources": len(role_sources),
            "rows_read": sum(source.rows_read for source in role_sources),
            "skipped": sum(source.skipped for source in role_sources),
            "duplicates": sum(source.duplicates for source in role_sources),
            "available": sum(source.available for source in role_sources),
            "requested": sum(source.requested for source in role_sources),
            "selected": sum(source.selected for source in role_sources),
            "filled": sum(source.filled for source in role_sources),
            "final_rows": sum(source.final_rows for source in role_sources),
        }

    final_counts = {
        role: sum(source.final_rows for source in sources
                  if source.role == role)
        for role in ROLES
    }
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "copy_weight": copy_weight,
        "content_weight": content_weight,
        "weights": weights,
        "seed": seed,
        "shuffle": bool(shuffle),
        "max_rows": max_rows,
        "sources": [_source_payload(source) for source in sources],
        "roles": role_totals,
        "final_counts": final_counts,
        "final_rows": len(rows),
        "duplicates": total_duplicates,
        "skipped": sum(source.skipped for source in sources),
        "filled_rows": sum(source.filled for source in sources),
        "unfilled_rows": unfilled,
        "totals": {
            "rows_read": sum(source.rows_read for source in sources),
            "skipped": sum(source.skipped for source in sources),
            "duplicates": total_duplicates,
            "available": sum(source.available for source in sources),
            "requested": sum(source.requested for source in sources),
            "selected": sum(source.selected for source in sources),
            "filled": sum(source.filled for source in sources),
            "final_rows": len(rows),
            "unfilled": unfilled,
        },
        "eval": {
            "rows_used": 0,
            "note": ("No eval rows were read or used. Eval-looking input "
                     "paths are refused before opening."),
        },
        "no_eval_rows_used": True,
    }
    return rows, manifest


def _encode_rows(rows: Sequence[Mapping[str, Any]]) -> bytes:
    payload = "".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    return payload.encode("utf-8")


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = _resolve(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
        + "\n",
        encoding="utf-8")


def run_mix(copy_paths: str | Path | Sequence[str | Path],
            content_paths: str | Path | Sequence[str | Path] | None,
            out: str | Path, *, copy_weight: float = DEFAULT_WEIGHT,
            content_weight: float = DEFAULT_WEIGHT,
            max_rows: int | None = None, seed: int = DEFAULT_SEED,
            shuffle: bool = False,
            manifest_path: str | Path | None = None) -> dict[str, Any]:
    """Mix rows, write ``out`` and optionally write the manifest."""

    rows, manifest = mix_rows(
        copy_paths, content_paths,
        copy_weight=copy_weight, content_weight=content_weight,
        max_rows=max_rows, seed=seed, shuffle=shuffle)

    out_path = _resolve(out)
    input_paths = {_resolve(item) for item in
                   _paths(copy_paths, "copy") + _paths(content_paths, "content")}
    if out_path in input_paths:
        raise MixError("--out must not overwrite an input file")
    data = _encode_rows(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(data)
    sha256 = hashlib.sha256(data).hexdigest()
    manifest["output"] = {"path": _display(out_path), "sha256": sha256}
    manifest["output_sha256"] = sha256
    if manifest_path is not None:
        manifest["manifest_path"] = _display(manifest_path)
        _write_json(manifest_path, manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--copy", nargs="+", required=True,
        help="one or more copy curriculum JSONL files")
    parser.add_argument(
        "--content", nargs="*", default=[],
        help="zero or more content/replay JSONL files")
    parser.add_argument("--out", required=True, help="output JSONL path")
    parser.add_argument(
        "--copy-weight", type=float, default=DEFAULT_WEIGHT,
        help="copy sampling multiplier (default: 1.0)")
    parser.add_argument(
        "--content-weight", type=float, default=DEFAULT_WEIGHT,
        help="content/replay sampling multiplier (default: 1.0)")
    parser.add_argument(
        "--max-rows", type=int, default=None,
        help="cap the final row count, preserving the planned ratios")
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"shuffle/selection seed (default: {DEFAULT_SEED})")
    parser.add_argument(
        "--shuffle", action="store_true",
        help="shuffle the final rows deterministically with --seed")
    parser.add_argument(
        "--manifest", default=None,
        help="write the content-free JSON manifest here")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        manifest = run_mix(
            args.copy, args.content, args.out,
            copy_weight=args.copy_weight,
            content_weight=args.content_weight,
            max_rows=args.max_rows,
            seed=args.seed,
            shuffle=args.shuffle,
            manifest_path=args.manifest)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    summary = {
        "output": manifest["output"],
        "final_rows": manifest["final_rows"],
        "copy_rows": manifest["final_counts"]["copy"],
        "content_rows": manifest["final_counts"]["content"],
        "duplicates": manifest["duplicates"],
        "skipped": manifest["skipped"],
        "filled_rows": manifest["filled_rows"],
        "unfilled_rows": manifest["unfilled_rows"],
        "manifest": manifest.get("manifest_path"),
    }
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
