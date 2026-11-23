"""One content-free report for the growth ladder.

Combines three local evidence sources:

- the escalation/growth ledger (local/teacher/agent/refuse counts and
  verified teacher answers),
- the measured scoreboard (tests, benchmark, AISC), and
- an optional self-code diff (resolved/introduced findings).

It never calls a network, an agent or a trainer, and it never changes weights.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from atomzero.escalation import summarize_growth
from atomzero.scoreboard import summarize_runs

__all__ = ["build_report", "main"]

METHOD = "content_free_growth_report_v1"


def _load_jsonl(path: str) -> list[dict]:
    if path == "-":
        raw = sys.stdin.read()
    else:
        raw = Path(path).expanduser().read_text(
            encoding="utf-8-sig", errors="replace")
    entries = []
    for number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{number}: invalid JSON") from exc
        if not isinstance(item, dict):
            raise ValueError(f"{path}:{number}: must be a JSON object")
        entries.append(item)
    return entries


def _load_json(path: str) -> dict:
    if path == "-":
        raw = sys.stdin.read()
    else:
        raw = Path(path).expanduser().read_text(
            encoding="utf-8-sig", errors="replace")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path}: must be a JSON object")
    return data


def build_report(*, escalation_ledger=None, scoreboard=None,
                 self_code_diff=None, verification_ledger=None) -> dict:
    """Build one deterministic, content-free growth report."""
    if escalation_ledger is None:
        escalation_ledger = []
    if scoreboard is None:
        scoreboard = []
    if verification_ledger is None:
        verification_ledger = []
    if not isinstance(escalation_ledger, list):
        raise ValueError("escalation_ledger must be a list")
    if not isinstance(scoreboard, list):
        raise ValueError("scoreboard must be a list")
    if not isinstance(verification_ledger, list):
        raise ValueError("verification_ledger must be a list")
    for index, event in enumerate(verification_ledger):
        if not isinstance(event, dict):
            raise ValueError(
                f"verification_ledger[{index}] must be an object")
    if self_code_diff is not None and not isinstance(self_code_diff, dict):
        raise ValueError("self_code_diff must be an object")

    growth = summarize_growth(escalation_ledger)
    measured = summarize_runs(scoreboard)
    verified_notes = sum(
        1 for event in verification_ledger if event.get("verified") is True)
    promotable_notes = sum(
        1 for event in verification_ledger if event.get("promotable") is True)
    reasons = []
    if not measured["runs"]:
        reasons.append("no_scoreboard")
    elif not measured["green"]:
        reasons.append("tests_failed")
    code = None
    if self_code_diff is not None:
        summary = self_code_diff.get("summary")
        if summary is not None and not isinstance(summary, dict):
            raise ValueError("self_code_diff.summary must be an object")
        summary = summary or {}
        code = {
            "ok": self_code_diff.get("ok") is True,
            "resolved": int(summary.get("resolved", 0) or 0),
            "introduced": int(summary.get("introduced", 0) or 0),
            "persisting": int(summary.get("persisting", 0) or 0),
            "new_high": int(summary.get("new_high", 0) or 0),
            "new_medium": int(summary.get("new_medium", 0) or 0),
        }
        if not code["ok"]:
            reasons.append("self_code_regression")

    latest = measured["latest"] or {}
    report = {
        "ok": not reasons,
        "reasons": reasons,
        "escalation": growth,
        "verification": {
            "total": len(verification_ledger),
            "verified": verified_notes,
            "promotable": promotable_notes,
        },
        "scoreboard": {
            "runs": measured["runs"],
            "green": measured["green"],
            "latest": {
                key: latest.get(key) for key in (
                    "commit", "tests_passed", "tests_failed",
                    "benchmark_passed", "benchmark_total",
                    "aisc_passed", "aisc_total")
            } if latest else None,
            "delta": measured["delta"],
        },
        "self_code": code,
        "capability": {
            "verified_teacher_answers": growth["verified_teacher_answers"],
            "verified_teacher_notes": verified_notes,
            "promotable_teacher_notes": promotable_notes,
            "adapter_candidates": growth["adapter_candidates"],
            "agent_handoffs": growth["agent_handoffs"],
            "tests_passed": latest.get("tests_passed", 0),
            "benchmark_passed": latest.get("benchmark_passed", 0),
            "aisc_passed": latest.get("aisc_passed", 0),
            "code_findings_resolved": code["resolved"] if code else 0,
        },
        "parameter_growth": 0,
        "parameter_growth_state": "unchanged",
        "weights_changed": False,
        "auto_apply": False,
        "method": METHOD,
    }
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--escalation-jsonl", default="")
    parser.add_argument("--scoreboard-jsonl", default="")
    parser.add_argument("--verification-jsonl", default="")
    parser.add_argument("--self-code-diff-json", default="")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    paths = [args.escalation_jsonl, args.scoreboard_jsonl,
             args.verification_jsonl, args.self_code_diff_json]
    if not any(paths):
        print("input error: at least one source is required", file=sys.stderr)
        return 2
    if sum(1 for path in paths if path == "-") > 1:
        print("input error: only one source may read stdin", file=sys.stderr)
        return 2
    try:
        report = build_report(
            escalation_ledger=(_load_jsonl(args.escalation_jsonl)
                               if args.escalation_jsonl else []),
            scoreboard=(_load_jsonl(args.scoreboard_jsonl)
                        if args.scoreboard_jsonl else []),
            verification_ledger=(_load_jsonl(args.verification_jsonl)
                                 if args.verification_jsonl else []),
            self_code_diff=(_load_json(args.self_code_diff_json)
                            if args.self_code_diff_json else None))
    except (OSError, ValueError) as exc:
        print(f"input error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        cap = report["capability"]
        print(f"Growth report: {'OK' if report['ok'] else 'BLOCKED'}")
        if report["reasons"]:
            print("reasons: " + ", ".join(report["reasons"]))
        print(f"- verified teacher answers: {cap['verified_teacher_answers']}")
        print(f"- verified teacher notes: {cap['verified_teacher_notes']}")
        print(f"- promotable teacher notes: {cap['promotable_teacher_notes']}")
        print(f"- adapter candidates: {cap['adapter_candidates']}")
        print(f"- tests passed: {cap['tests_passed']}")
        print(f"- code findings resolved: {cap['code_findings_resolved']}")
        print("- parameter growth: 0 (unchanged)")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
