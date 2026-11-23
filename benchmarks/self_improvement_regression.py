"""Deterministic regression benchmark for the self-improvement loop.

Checks the planning, human gate, content-free ledger, proactive reminder,
evaluation allowlist and the upstream claim/temporal signals. No model calls,
no network and no writes outside a temporary ledger.

Run:

    python benchmarks/self_improvement_regression.py
"""
from __future__ import annotations

import argparse
import json
import tempfile
from datetime import datetime
from pathlib import Path

from atomzero.claims import audit_answer
from atomzero.consensus import audit_consensus
from atomzero.evaluation import parse_check
from atomzero.improvement import (
    build_plan,
    collect_signals,
    plan_from_inputs,
    record_decision,
    record_gated_decision,
    summarize_ledger,
    verify_ledger,
)
from atomzero.proactive import (
    build_candidate,
    configure,
    note_improvement_plan,
    should_initiate,
)
from atomzero.scoreboard import record_run
from atomzero.temporal import filter_as_of
from atomzero.trend import build_trend

NOW = datetime(2026, 11, 23, 12, 0, 0)


def _plan():
    return plan_from_inputs(
        claim_audit={"unsupported": 2, "conflict_count": 1,
                     "single_source": 1},
        gaps=[{"id": "g1", "query": "local policy", "status": "open"}],
    )


def check_plan_priority_and_human_gate():
    plan = _plan()
    kinds = [action["kind"] for action in plan["actions"]]
    assert kinds[0] == "unsupported_claims"
    assert "conflicts" in kinds and "knowledge_gaps" in kinds
    assert all(action["requires_human"] is True for action in plan["actions"])
    assert all(action["acceptance"] for action in plan["actions"])


def check_plan_does_not_auto_apply():
    plan = _plan()
    assert plan["auto_apply"] is False
    assert plan["weights_changed"] is False
    assert plan["method"] == "deterministic_human_gated_improvement_v1"


def check_decision_is_hash_only():
    plan = _plan()
    ledger = []
    record = record_decision(
        ledger, plan["actions"][0], "approve",
        note="private review note", now=NOW)
    assert len(record["note_id"]) == 16
    assert "private review note" not in json.dumps(record)
    assert verify_ledger(ledger)["ok"] is True


def check_gated_approval():
    gate = {
        "ok": True,
        "summary": {"total": 2, "passed": 2, "failed": 0},
        "method": "allowlisted_evaluation_gate_v1",
    }
    ledger = []
    record = record_gated_decision(
        ledger, _plan()["actions"][0], "approve", gate, now=NOW)
    assert record["gate_ok"] is True
    assert verify_ledger(ledger)["ok"] is True
    try:
        record_gated_decision(
            [], _plan()["actions"][0], "approve",
            {"ok": False,
             "summary": {"total": 2, "passed": 1, "failed": 1},
             "method": "allowlisted_evaluation_gate_v1"},
            now=NOW)
    except ValueError:
        pass
    else:
        raise AssertionError("failed gate must not approve")


def check_tampered_ledger_is_rejected():
    plan = _plan()
    ledger = []
    record_decision(ledger, plan["actions"][0], "approve", now=NOW)
    ledger[0]["text"] = "leak"
    result = verify_ledger(ledger)
    assert result["ok"] is False


def check_ledger_summary():
    plan = _plan()
    ledger = []
    record_decision(ledger, plan["actions"][0], "approve", now=NOW)
    record_decision(ledger, plan["actions"][0], "reject", now=NOW)
    summary = summarize_ledger(ledger)
    assert summary["total"] == 2
    assert summary["decisions"] == {"approve": 1, "reject": 1}


def check_aisc_coverage_signal():
    signals = collect_signals(aisc_coverage={
        "summary": {
            "specs_with_missing_evidence": 1,
            "specs_with_unverified_components": 1,
            "errors": 0,
        }})
    assert signals[0]["kind"] == "aisc_coverage_gaps"
    plan = build_plan(signals)
    assert plan["actions"][0]["kind"] == "aisc_coverage_gaps"


def check_proactive_improvement_reminder():
    session = {"proactive": {"enabled": True, "min_interval_minutes": 0,
                             "max_per_day": 5, "quiet_start": "00:00",
                             "quiet_end": "00:00"}}
    plan = _plan()
    entry = note_improvement_plan(session, plan, now=NOW)
    assert entry["actions"] == len(plan["actions"])
    assert "private" not in json.dumps(entry)
    decision = should_initiate(session, now=NOW)
    assert decision["due"] is True
    assert decision["reason"] == "improvement_plan"
    candidate = build_candidate(session, now=NOW)
    assert candidate["actions"] == [
        "show improvements", "run evaluation gate"]


def check_evaluation_allowlist():
    assert parse_check("pytest:tests/test_claims.py")["kind"] == "pytest"
    try:
        parse_check("shell:dir")
    except ValueError:
        pass
    else:
        raise AssertionError("shell check should be rejected")


def check_end_to_end_loop():
    plan = _plan()
    action = plan["actions"][0]
    gate = {
        "ok": True,
        "summary": {"total": 2, "passed": 2, "failed": 0},
        "method": "allowlisted_evaluation_gate_v1",
    }
    ledger = []
    record_gated_decision(ledger, action, "approve", gate, now=NOW)
    runs = []
    record_run(runs, commit="514632e", tests_passed=755, tests_failed=0,
               benchmark_passed=11, benchmark_total=11,
               aisc_passed=16, aisc_total=18, now=NOW)
    report = build_trend(ledger, runs)
    assert report["ok"] is True
    assert report["decisions"]["total"] == 1
    assert report["scoreboard"]["latest"]["commit"] == "514632e"
    assert report["weights_changed"] is False


def check_upstream_audits_are_content_free():
    claims = audit_answer(
        "The battery lasts 8 hours.",
        [{"source": "a.txt", "chunk": "The battery lasts 8 hours."}])
    assert claims["weights_changed"] is False
    consensus = audit_consensus([
        {"source": "a.txt", "chunk": "The battery lasts 8 hours."},
        {"source": "b.txt", "chunk": "The battery lasts 8 hours."},
    ])
    assert consensus["summary"]["consensus_clusters"] == 1
    assert consensus["weights_changed"] is False


def check_temporal_filter_is_deterministic():
    hits = [
        {"source": "old", "valid_until": "2025-12-31"},
        {"source": "new", "valid_from": "2026-01-01"},
        {"source": "undated"},
    ]
    first = filter_as_of(hits, "2026-11-23")
    second = filter_as_of(hits, "2026-11-23")
    assert first == second
    assert first["summary"]["status_counts"]["expired"] == 1
    assert first["weights_changed"] is False


CHECKS = [
    check_plan_priority_and_human_gate,
    check_plan_does_not_auto_apply,
    check_decision_is_hash_only,
    check_gated_approval,
    check_tampered_ledger_is_rejected,
    check_ledger_summary,
    check_aisc_coverage_signal,
    check_proactive_improvement_reminder,
    check_end_to_end_loop,
    check_evaluation_allowlist,
    check_upstream_audits_are_content_free,
    check_temporal_filter_is_deterministic,
]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    results = []
    for check in CHECKS:
        try:
            check()
            results.append({"name": check.__name__, "ok": True})
        except Exception as exc:  # noqa: BLE001 - the benchmark records failures
            results.append({
                "name": check.__name__, "ok": False,
                "error": f"{type(exc).__name__}: {exc}"})
    passed = sum(1 for item in results if item["ok"])
    report = {
        "checks": results,
        "passed": passed,
        "total": len(results),
        "ok": passed == len(results),
    }
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        for item in results:
            line = f"[{'PASS' if item['ok'] else 'FAIL'}] {item['name']}"
            if not item["ok"]:
                line += f": {item['error']}"
            print(line)
        print(f"summary: {passed}/{len(results)} checks passed")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
