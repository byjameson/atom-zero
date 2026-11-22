"""Deterministic regression benchmark for the daily brief and goal planner.

Run:

    python benchmarks/briefing_regression.py
    python benchmarks/briefing_regression.py --json
"""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

from atomzero.briefing import build_briefing, load_session, plan_goal
from atomzero.doc_memory import DocMemory
from atomzero.knowledge_gaps import KnowledgeGapStore


def run(output: str | Path | None = None) -> dict:
    checks = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    with tempfile.TemporaryDirectory(prefix="atomzero-brief-") as folder:
        docs = DocMemory(Path(folder) / "docs.db")
        try:
            docs.add_text("notes/launch.txt",
                          "The launch code is L-7781.", title="Launch")
            docs.add_text("notes/warranty.txt",
                          "The warranty period is 24 months. "
                          "Claims need a receipt.", title="Warranty")
            gaps = KnowledgeGapStore(Path(folder) / "gaps.json")
            gaps.add("What is the warranty claim deadline?")
            session = {"tasks": [
                {"id": 1, "text": "review warranty page", "status": "open"},
                {"id": 2, "text": "old task", "status": "done"},
            ]}
            brief = build_briefing(
                session, gaps, docs, now="2026-11-22 09:00")
            check(
                "brief_counts_and_focus",
                brief["open_task_count"] == 1
                and brief["pending_gap_count"] == 1
                and "review warranty page" in brief["focus"],
                brief["focus"])
            check(
                "brief_index_and_graph",
                "Index: 2 chunk(s) / 2 source(s)" in brief["text"]
                and "L-7781" in {
                    node["name"] for node in brief["graph"]["entities"]},
                brief["text"].splitlines()[-4] if brief["text"] else "")
            check(
                "brief_actions_and_no_weights",
                len(brief["next_actions"]) >= 3
                and brief["weights_changed"] is False,
                str(brief["next_actions"][:3]))

            plan = plan_goal("prepare the warranty page", docs, gaps, session)
            check(
                "plan_context",
                plan["context"]
                and plan["context"][0]["source"] == "notes/warranty.txt"
                and "24 months" in plan["context"][0]["snippet"],
                plan["context"][0]["citation"] if plan["context"] else "")
            check(
                "plan_suggestions",
                len(plan["suggested_tasks"]) == 3
                and "pending_gaps" in plan
                and plan["weights_changed"] is False,
                str(plan["suggested_tasks"]))
            check(
                "plan_does_not_mutate_session",
                len(session["tasks"]) == 2,
                f"{len(session['tasks'])} task(s)")

            empty_plan = plan_goal("invent a new product", None, None, {})
            check(
                "empty_plan_is_honest",
                empty_plan["context"] == []
                and any("Find one local source" in suggestion
                        for suggestion in empty_plan["suggested_tasks"])
                and empty_plan["confidence"]["label"] == "none",
                str(empty_plan["suggested_tasks"]))

            empty_brief = build_briefing({}, None, None, now="2026-11-22")
            check(
                "empty_brief_is_honest",
                "add a local document" in empty_brief["focus"]
                and empty_brief["documents"]["chunks"] == 0
                and empty_brief["text"].startswith("Daily brief"),
                empty_brief["focus"])

            check(
                "missing_session_is_empty",
                load_session(Path(folder) / "missing.json") == {},
                "unexpected session data")
        finally:
            docs.close()

    failed = sum(1 for item in checks if not item["ok"])
    payload = {
        "benchmark": "atomzero-briefing-regression",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "checks": checks,
        "passed": len(checks) - failed,
        "failed": failed,
        "ok": failed == 0,
    }
    if output:
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", default="benchmarks/results/briefing_regression.json")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    payload = run(args.out)
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        for item in payload["checks"]:
            status = "OK" if item["ok"] else "FAIL"
            print(f"[{status}] {item['name']}: {item['detail']}")
        print(f"summary: {payload['passed']} passed, "
              f"{payload['failed']} failed")
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
