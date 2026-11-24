"""Benchmark the deterministic runtime copy tool.

This benchmark measures the tool, not the model: it reads a staged copy eval
JSONL file, extracts the pairs from each prompt with
``atomzero.copy_tool.extract_pairs``, renders the requested format and parses
the result back. The summary contains counts and rates only; prompts, keys and
values never appear in it.

A row is valid when it has a prompt, a supported format and a parseable
``pairs`` list, and the prompt pairs equal the metadata pairs. Invalid rows are
counted and excluded from the rates.

Default dataset (gitignored)::

    data/multi_short_v1/stage_4short/eval.jsonl

Usage::

    python -m research.benchmark_copy_tool
    python -m research.benchmark_copy_tool --dataset path/to/eval.jsonl --json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from atomzero.copy_tool import (
    copy_pairs,
    extract_pairs,
    normalize_format,
    parse_rendered,
)

METHOD = "deterministic_runtime_copy_tool_benchmark_v1"
DEFAULT_DATASET = Path("data/multi_short_v1/stage_4short/eval.jsonl")


def _coerce_pairs(value: Any) -> list[tuple[str, str]]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("missing or empty pairs metadata")
    pairs: list[tuple[str, str]] = []
    seen: set[str] = set()
    for index, pair in enumerate(value, start=1):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"pair {index} is not a two-item list")
        key = str(pair[0]).strip()
        raw_value = pair[1]
        text = raw_value if isinstance(raw_value, str) else str(raw_value)
        if not key:
            raise ValueError(f"pair {index} has an empty key")
        if not text:
            raise ValueError(f"pair {index} has an empty value")
        if key in seen:
            raise ValueError(f"duplicate key in pairs metadata: {key!r}")
        seen.add(key)
        pairs.append((key, text))
    return pairs


def evaluate_row(row: Any) -> dict[str, Any]:
    """Evaluate one eval row. Return flags and a category, never content."""
    result: dict[str, Any] = {
        "valid": False,
        "error": None,
        "format": None,
        "exact": False,
        "keys_match": False,
        "values_match": False,
    }
    if not isinstance(row, Mapping):
        result["error"] = "invalid_row"
        return result

    raw_format = row.get("format")
    try:
        fmt = normalize_format(raw_format)
    except ValueError:
        result["error"] = "unsupported_format"
        return result
    result["format"] = fmt

    prompt = row.get("user")
    if not isinstance(prompt, str) or not prompt.strip():
        result["error"] = "missing_prompt"
        return result

    try:
        prompt_pairs = extract_pairs(prompt)
    except ValueError:
        result["error"] = "malformed_prompt"
        return result

    try:
        expected_pairs = _coerce_pairs(row.get("pairs"))
    except ValueError:
        result["error"] = "invalid_pairs_metadata"
        return result

    if prompt_pairs != expected_pairs:
        result["error"] = "prompt_metadata_mismatch"
        return result

    try:
        rendered = copy_pairs(prompt, fmt=fmt)
        parsed = parse_rendered(rendered, fmt=fmt)
    except ValueError:
        result["error"] = "tool_error"
        return result

    expected = dict(expected_pairs)
    parsed_text = {str(key): str(value) for key, value in parsed.items()}
    keys_match = (
        set(parsed_text) == set(expected)
        and len(parsed_text) == len(expected)
    )
    values_match = keys_match and all(
        parsed_text[key] == expected[key] for key in expected)
    result.update({
        "valid": True,
        "error": None,
        "keys_match": keys_match,
        "values_match": values_match,
        "exact": keys_match and values_match,
    })
    return result


def _rate(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round(count / total, 6)


def summarize_rows(
    rows: Sequence[Any], *, dataset: str = "<memory>",
) -> dict[str, Any]:
    """Build a content-free summary for evaluated rows."""
    evaluations = [evaluate_row(row) for row in rows]
    valid = [item for item in evaluations if item["valid"]]
    exact = sum(1 for item in valid if item["exact"])
    key_copy = sum(1 for item in valid if item["keys_match"])
    value_copy = sum(1 for item in valid if item["values_match"])
    formats = Counter(item["format"] for item in valid)
    invalid_reasons = Counter(
        item["error"] or "unknown"
        for item in evaluations if not item["valid"])
    if not valid:
        status = "no_valid_rows"
    elif exact == len(valid):
        status = "exact"
    else:
        status = "mismatch"
    return {
        "tool": "atomzero-copy",
        "method": METHOD,
        "dataset": str(dataset),
        "rows_total": len(evaluations),
        "rows_valid": len(valid),
        "rows_invalid": len(evaluations) - len(valid),
        "rows_exact": exact,
        "rows_key_copy": key_copy,
        "rows_value_copy": value_copy,
        "exact_rate": _rate(exact, len(valid)),
        "key_copy_rate": _rate(key_copy, len(valid)),
        "value_copy_rate": _rate(value_copy, len(valid)),
        "formats": {name: formats.get(name, 0) for name in sorted(formats)},
        "invalid_reasons": {
            name: invalid_reasons.get(name, 0)
            for name in sorted(invalid_reasons)
        },
        "status": status,
        "model_used": False,
        "weights_changed": False,
        "content_free": True,
    }


def load_rows(path: str | Path) -> list[Any]:
    """Read JSONL rows. Malformed lines become invalid rows, not crashes."""
    rows: list[Any] = []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                rows.append(None)
    return rows


def run_benchmark(
    dataset_path: str | Path = DEFAULT_DATASET,
    *,
    limit: int | None = None,
) -> dict[str, Any]:
    """Load ``dataset_path`` and return the content-free summary."""
    path = Path(dataset_path)
    rows = load_rows(path)
    if limit is not None:
        if limit < 0:
            raise ValueError("limit must be non-negative")
        rows = rows[:limit]
    return summarize_rows(rows, dataset=str(path))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dataset", default=str(DEFAULT_DATASET),
        help=f"staged eval JSONL (default: {DEFAULT_DATASET})")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="evaluate only the first N rows")
    parser.add_argument(
        "--out", default=None,
        help="also write the summary JSON to this path")
    parser.add_argument(
        "--fail-under", type=float, default=0.0,
        help="exit 1 if exact_rate is below this threshold (default: 0.0)")
    parser.add_argument(
        "--json", action="store_true",
        help="accepted for symmetry; output is always JSON")
    args = parser.parse_args(argv)

    try:
        summary = run_benchmark(args.dataset, limit=args.limit)
    except OSError as exc:
        print(f"error: could not read dataset: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    body = json.dumps(summary, indent=2, sort_keys=True)
    print(body)
    if args.out:
        try:
            Path(args.out).write_text(body + "\n", encoding="utf-8")
        except OSError as exc:
            print(f"error: could not write summary: {exc}", file=sys.stderr)
            return 2
    if summary["exact_rate"] < args.fail_under:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
