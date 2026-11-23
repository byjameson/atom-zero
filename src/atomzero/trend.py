"""Content-free trend view over improvement decisions and measured runs.

The planner says what should improve; the scoreboard says whether it did. This
module combines both without copying text, test output or gate output. It is a
read-only reporting layer.

    atomzero-trend --ledger data/improvement_ledger.jsonl \
      --scoreboard data/scoreboard.jsonl --json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from atomzero.improvement import summarize_ledger
from atomzero.scoreboard import load_jsonl as load_scoreboard
from atomzero.scoreboard import summarize_runs

__all__ = ["build_trend", "main"]


def _load_ledger(path) -> list[dict]:
    target = Path(path)
    if not target.exists():
        return []
    entries = []
    for number, line in enumerate(
            target.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{target}:{number}: invalid JSON") from exc
        if not isinstance(entry, dict):
            raise ValueError(f"{target}:{number}: must be a JSON object")
        entries.append(entry)
    return entries


def build_trend(ledger, runs) -> dict:
    """Combine decision counts and measured deltas into a trend report."""
    decisions = summarize_ledger(ledger)
    scoreboard = summarize_runs(runs)
    alerts = []
    if scoreboard["latest"] is not None:
        latest = scoreboard["latest"]
        delta = scoreboard["delta"]
        if not scoreboard["green"]:
            alerts.append("tests_failed")
        if delta.get("tests_passed", 0) < 0:
            alerts.append("tests_passed_decreased")
        if delta.get("benchmark_passed", 0) < 0:
            alerts.append("benchmark_passed_decreased")
        if delta.get("aisc_passed", 0) < 0:
            alerts.append("aisc_passed_decreased")
        if latest.get("benchmark_total") and (
                latest.get("benchmark_passed", 0)
                < latest.get("benchmark_total", 0)):
            alerts.append("benchmark_incomplete")
    return {
        "decisions": decisions,
        "scoreboard": scoreboard,
        "alerts": sorted(set(alerts)),
        "ok": not alerts,
        "weights_changed": False,
        "method": "content_free_trend_v1",
    }


def _render(report: dict) -> str:
    scoreboard = report["scoreboard"]
    decisions = report["decisions"]
    lines = [
        f"Trend: {scoreboard['runs']} measured run(s), "
        f"{decisions['total']} decision(s)",
    ]
    if scoreboard["latest"]:
        latest = scoreboard["latest"]
        lines.append(
            f"- latest {latest['commit']}: {latest['tests_passed']} tests, "
            f"benchmark {latest['benchmark_passed']}/"
            f"{latest['benchmark_total']}, AISC "
            f"{latest['aisc_passed']}/{latest['aisc_total']}")
        if scoreboard["delta"]:
            lines.append("- delta: " + ", ".join(
                f"{key}={value:+d}"
                for key, value in sorted(scoreboard["delta"].items())))
    lines.append("- decisions: " + ", ".join(
        f"{key}={value}" for key, value in sorted(
            decisions["decisions"].items())) if decisions["decisions"] else
        "- decisions: none")
    lines.append("- alerts: " + (", ".join(report["alerts"])
                                 if report["alerts"] else "none"))
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path,
                        default=Path("data/improvement_ledger.jsonl"))
    parser.add_argument("--scoreboard", type=Path,
                        default=Path("data/scoreboard.jsonl"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = build_trend(
            _load_ledger(args.ledger), load_scoreboard(args.scoreboard))
    except (OSError, ValueError) as exc:
        print(f"input error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(_render(report))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
