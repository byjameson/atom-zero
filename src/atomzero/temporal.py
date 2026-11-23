"""Deterministic as-of / validity filtering for retrieval hits.

Freshness looks at a date written in the text. Temporal validity is stronger:
a hit may carry an explicit ``valid_from`` and ``valid_until`` window. This
module filters a local hit list for an "as of" date and reports why a hit was
kept or excluded. It makes no model calls and does not persist input.

    from atomzero.temporal import filter_as_of
    report = filter_as_of(hits, as_of="2026-11-23")
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path

__all__ = [
    "assess_validity",
    "filter_as_of",
    "parse_date",
]

_STATUSES = ("valid", "expired", "not_yet", "undated")


def parse_date(value, field: str = "date") -> date:
    """Bir date/datetime veya ISO tarih dizesini date'e çevirir."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str) and value.strip():
        text = value.strip()
        try:
            return date.fromisoformat(text)
        except ValueError:
            try:
                return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
            except ValueError as exc:
                raise ValueError(f"{field} must be an ISO date") from exc
    raise ValueError(f"{field} must be an ISO date")


def _labels(hit: dict, index: int) -> tuple[str, str]:
    source = str(hit.get("source") or hit.get("title") or f"hit-{index}")
    title = str(hit.get("title") or "")
    return source, title


def assess_validity(hit: dict, as_of=None) -> dict:
    """Bir hit'in geçerlilik penceresini verilen tarihe göre değerlendirir."""
    if not isinstance(hit, dict):
        raise ValueError("hit must be an object")
    moment = parse_date(as_of, "as_of") if as_of is not None else date.today()
    source, title = _labels(hit, 0)
    from_value = hit.get("valid_from")
    until_value = hit.get("valid_until")
    valid_from = parse_date(from_value, "valid_from") if from_value else None
    valid_until = parse_date(until_value, "valid_until") if until_value else None
    if valid_from is None and valid_until is None:
        status = "undated"
    elif valid_from is not None and moment < valid_from:
        status = "not_yet"
    elif valid_until is not None and moment > valid_until:
        status = "expired"
    else:
        status = "valid"
    return {
        "source": source,
        "title": title,
        "as_of": moment.isoformat(),
        "valid_from": valid_from.isoformat() if valid_from else None,
        "valid_until": valid_until.isoformat() if valid_until else None,
        "status": status,
        "days_remaining": (
            (valid_until - moment).days if valid_until is not None else None),
        "days_until_start": (
            (valid_from - moment).days if valid_from is not None else None),
    }


def filter_as_of(hits, as_of=None, keep_undated: bool = True) -> dict:
    """Hit listesini as-of tarihine göre filtreler ve rapor döndürür."""
    if not isinstance(hits, list):
        raise ValueError("hits must be a list")
    moment = parse_date(as_of, "as_of") if as_of is not None else date.today()
    assessed = []
    for index, hit in enumerate(hits):
        if not isinstance(hit, dict):
            raise ValueError(f"hit {index} must be an object")
        entry = assess_validity(hit, moment)
        entry["source"], entry["title"] = _labels(hit, index)
        assessed.append(entry)
    included = []
    excluded = []
    for entry in assessed:
        keep = entry["status"] == "valid" or (
            entry["status"] == "undated" and keep_undated)
        (included if keep else excluded).append(entry)
    counts = {status: 0 for status in _STATUSES}
    for entry in assessed:
        counts[entry["status"]] += 1
    return {
        "as_of": moment.isoformat(),
        "keep_undated": bool(keep_undated),
        "included": included,
        "excluded": excluded,
        "summary": {
            "total": len(assessed),
            "included": len(included),
            "excluded": len(excluded),
            "status_counts": counts,
        },
        "weights_changed": False,
        "method": "deterministic_as_of_validity_v1",
    }


def _render(report: dict) -> str:
    summary = report["summary"]
    lines = [
        f"As of {report['as_of']}: {summary['included']} included, "
        f"{summary['excluded']} excluded",
    ]
    for entry in report["included"]:
        lines.append(f"+ [{entry['status']}] {entry['source']}")
    for entry in report["excluded"]:
        lines.append(f"- [{entry['status']}] {entry['source']}")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, help="JSON hit list")
    parser.add_argument("--as-of", default=None, help="ISO date")
    parser.add_argument("--exclude-undated", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.file:
            payload = json.loads(args.file.read_text(encoding="utf-8-sig"))
        else:
            raw = sys.stdin.read()
            payload = json.loads(raw) if raw.strip() else []
    except (OSError, json.JSONDecodeError) as exc:
        print(f"input error: {exc}", file=sys.stderr)
        return 2
    if isinstance(payload, dict):
        payload = payload.get("hits", [])
    if not isinstance(payload, list) or not payload:
        print("no hits", file=sys.stderr)
        return 2
    try:
        report = filter_as_of(
            payload, as_of=args.as_of,
            keep_undated=not args.exclude_undated)
    except ValueError as exc:
        print(f"input error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(_render(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
