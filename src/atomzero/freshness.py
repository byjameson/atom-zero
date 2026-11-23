"""Deterministic source-freshness assessment for local retrieval hits.

The module reads calendar dates from hit text and compares the latest date that
is not in the future with a caller-supplied ``now``. It is an evidence signal,
not a truth verdict: a recent date does not make a claim correct, and an
undated source is not silently treated as fresh.

No model is called, no input is persisted and no retrieval weight is changed.

    from atomzero.freshness import assess_hit, audit_freshness
    item = assess_hit({"source": "note.md", "chunk": "Updated 2025-12-01."})
    report = audit_freshness([hit], now="2026-01-01")
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date, datetime
from pathlib import Path

from atomzero.contradictions import date_matches

__all__ = ["assess_hit", "audit_freshness", "main"]

METHOD = "deterministic_source_freshness_v1"
DEFAULT_MAX_AGE_DAYS = 365
DEFAULT_FRESH_DAYS = 90
STATUSES = ("fresh", "aging", "stale", "future", "undated")
TEXT_KEYS = ("chunk", "snippet", "title")


def _resolve_now(now=None) -> date:
    """Return ``now`` as a local calendar date."""
    if now is None:
        return date.today()
    if isinstance(now, datetime):
        return now.date()
    if isinstance(now, date):
        return now
    if isinstance(now, str):
        value = now.strip()
        if value:
            try:
                return date.fromisoformat(value)
            except ValueError:
                pass
            candidate = (
                value[:-1] + "+00:00"
                if value.endswith(("Z", "z"))
                else value
            )
            try:
                return datetime.fromisoformat(candidate).date()
            except ValueError:
                pass
    raise ValueError(
        f"now must be a date or ISO date/datetime string: {now!r}")


def _thresholds(max_age_days, fresh_days) -> tuple[float, float]:
    """Validate ``max_age_days >= fresh_days >= 0``."""
    values = {}
    for name, value in (
            ("max_age_days", max_age_days), ("fresh_days", fresh_days)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a non-negative number")
        number = float(value)
        if not math.isfinite(number) or number < 0:
            raise ValueError(f"{name} must be a non-negative number")
        values[name] = number
    if values["max_age_days"] < values["fresh_days"]:
        raise ValueError(
            "max_age_days must be greater than or equal to fresh_days")
    return values["max_age_days"], values["fresh_days"]


def _field_text(hit: dict, key: str) -> str:
    value = hit.get(key)
    if isinstance(value, str):
        return value.strip()
    return ""


def _labels(hit: dict, index: int) -> tuple[str, str]:
    """Return stable source/title labels with mutual fallback."""
    raw_source = _field_text(hit, "source")
    raw_title = _field_text(hit, "title")
    source = raw_source or raw_title or f"hit-{index}"
    title = raw_title or raw_source
    return source, title


def _date_candidates(hit: dict) -> list[tuple[date, str, dict]]:
    """Collect valid calendar dates from chunk, snippet and title text."""
    candidates: list[tuple[date, str, dict]] = []
    for key in TEXT_KEYS:
        text = hit.get(key)
        if not isinstance(text, str) or not text:
            continue
        for match in date_matches(text):
            try:
                parsed = date.fromisoformat(match["date"])
            except (TypeError, ValueError):
                # date_matches accepts day 1..31; skip impossible calendar
                # dates such as 2026-02-31 instead of inventing a date.
                continue
            candidates.append((parsed, key, match))
    return candidates


def _assess(hit: dict, index: int, today: date,
            max_age_days: float, fresh_days: float) -> dict:
    source, title = _labels(hit, index)
    candidates = _date_candidates(hit)
    past = [item for item in candidates if item[0] <= today]
    future = [item for item in candidates if item[0] > today]

    chosen = None
    age_days = None
    if past:
        # Latest past date wins; ties keep the first deterministic field/span.
        chosen = max(past, key=lambda item: item[0])
        age_days = (today - chosen[0]).days
        if age_days <= fresh_days:
            status = "fresh"
        elif age_days <= max_age_days:
            status = "aging"
        else:
            status = "stale"
    elif future:
        # Only future dates: report the nearest one as a signal, not as age.
        chosen = min(future, key=lambda item: item[0])
        age_days = (today - chosen[0]).days
        status = "future"
    else:
        status = "undated"

    match = chosen[2] if chosen is not None else None
    return {
        "source": source,
        "title": title,
        "date": chosen[0].isoformat() if chosen is not None else None,
        "age_days": age_days,
        "status": status,
        "span": [match["start"], match["end"]] if match else None,
        "precision": match["precision"] if match else None,
    }


def assess_hit(hit, now=None, max_age_days=DEFAULT_MAX_AGE_DAYS,
               fresh_days=DEFAULT_FRESH_DAYS) -> dict:
    """Assess one retrieval hit against ``now``.

    Dates are read from ``chunk``, ``snippet`` and ``title`` text with
    :func:`atomzero.contradictions.date_matches`. The latest date not later
    than ``now`` is used. If every date is in the future, the nearest future
    date is reported with status ``future`` and a negative ``age_days``.
    ``span`` is the ``[start, end]`` offset of the chosen date inside the text
    field where it was found.
    """
    if not isinstance(hit, dict):
        raise ValueError("hit must be a mapping")
    max_age, fresh = _thresholds(max_age_days, fresh_days)
    today = _resolve_now(now)
    return _assess(hit, 0, today, max_age, fresh)


def audit_freshness(hits, now=None, max_age_days=DEFAULT_MAX_AGE_DAYS,
                    fresh_days=DEFAULT_FRESH_DAYS) -> dict:
    """Assess a list of hits and return a deterministic freshness report."""
    if not isinstance(hits, list):
        raise ValueError("hits must be a list")
    for index, hit in enumerate(hits):
        if not isinstance(hit, dict):
            raise ValueError(f"hit {index} must be a mapping")
    max_age, fresh = _thresholds(max_age_days, fresh_days)
    today = _resolve_now(now)

    assessed = [
        _assess(hit, index, today, max_age, fresh)
        for index, hit in enumerate(hits)
    ]
    status_counts = {status: 0 for status in STATUSES}
    sources = set()
    stale_sources = set()
    undated_sources = set()
    dates = []
    for item in assessed:
        status_counts[item["status"]] += 1
        sources.add(item["source"])
        if item["date"] is not None:
            dates.append(item["date"])
        if item["status"] == "stale":
            stale_sources.add(item["source"])
        elif item["status"] == "undated":
            undated_sources.add(item["source"])

    summary = {
        "hit_count": len(assessed),
        "status_counts": status_counts,
        "source_count": len(sources),
        "oldest_date": min(dates) if dates else None,
        "newest_date": max(dates) if dates else None,
        "stale_sources": sorted(stale_sources),
        "undated_sources": sorted(undated_sources),
    }
    return {
        "hits": assessed,
        "summary": summary,
        "weights_changed": False,
        "method": METHOD,
    }


def _render(report: dict) -> str:
    lines = []
    for item in report["hits"]:
        title = (
            f" | {item['title']}"
            if item["title"] and item["title"] != item["source"]
            else ""
        )
        age = (
            f"{item['age_days']}d"
            if item["age_days"] is not None
            else "-"
        )
        lines.append(
            f"- [{item['status']}] {item['source']}{title} | "
            f"date={item['date'] or '-'} | age={age}")
    summary = report["summary"]
    counts = summary["status_counts"]
    stale = ",".join(summary["stale_sources"]) or "-"
    undated = ",".join(summary["undated_sources"]) or "-"
    lines.append(
        f"Freshness: {summary['hit_count']} hits; "
        f"fresh={counts['fresh']}, aging={counts['aging']}, "
        f"stale={counts['stale']}, future={counts['future']}, "
        f"undated={counts['undated']}; sources={summary['source_count']}; "
        f"oldest={summary['oldest_date'] or '-'}; "
        f"newest={summary['newest_date'] or '-'}; "
        f"stale_sources={stale}; undated_sources={undated}")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="atomzero-freshness",
        description="Deterministic source-freshness audit for JSON hits.")
    parser.add_argument("--file", type=Path,
                        help="JSON hit list or {hits: [...]} object")
    parser.add_argument("--now", default=None,
                        help="assessment date (YYYY-MM-DD or ISO datetime)")
    parser.add_argument("--max-age-days", type=int,
                        default=DEFAULT_MAX_AGE_DAYS)
    parser.add_argument("--fresh-days", type=int, default=DEFAULT_FRESH_DAYS)
    parser.add_argument("--json", action="store_true",
                        help="print the full JSON report")
    args = parser.parse_args(argv)

    try:
        if args.file:
            payload = json.loads(
                args.file.read_text(encoding="utf-8-sig"))
        else:
            raw = sys.stdin.read()
            payload = json.loads(raw) if raw.strip() else []
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        print(f"input error: {exc}", file=sys.stderr)
        return 2

    if isinstance(payload, dict):
        payload = payload.get("hits", [])
    if not isinstance(payload, list):
        print("input error: hits must be a JSON list", file=sys.stderr)
        return 2
    if not payload:
        print("no hits", file=sys.stderr)
        return 2

    try:
        report = audit_freshness(
            payload, now=args.now, max_age_days=args.max_age_days,
            fresh_days=args.fresh_days)
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
