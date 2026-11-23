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
from atomzero.adapter_candidates import (
    build_candidate as build_adapter_candidate,
    evaluate_candidate as evaluate_adapter_candidate,
)
from atomzero.escalation import decide, record_escalation, summarize_growth
from atomzero.evaluation import parse_check
from atomzero.growth_report import build_report
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
from atomzero.scoreboard import (
    parse_aisc_summary,
    parse_benchmark_summary,
    parse_pytest_summary,
    record_run,
)
from atomzero.self_code import audit_report
from atomzero.self_code_diff import compare_reports
from atomzero.teacher_verification import verification_event, verify_answer
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


def check_self_code_audit_and_plan():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "src").mkdir()
        (root / "src" / "mod.py").write_text(
            "def f():\n    try:\n        return 1\n"
            "    except Exception:\n        return None\n", encoding="utf-8")
        report = audit_report(root, propose=True)
        assert report["summary"]["medium"] == 1
        assert report["auto_apply"] is False
        assert report["weights_changed"] is False
        assert report["proposals"][0]["requires_human"] is True
        plan = plan_from_inputs(self_code=report)
        assert plan["actions"][0]["kind"] == "self_code_medium"


def check_self_code_diff_is_a_gate():
    finding = {
        "finding_id": "sc-1", "kind": "broad_exception", "severity": "medium",
        "path": "src/mod.py", "line": 4, "message": "broad",
    }
    before = {"findings": [finding]}
    fixed = compare_reports(before, {"findings": []})
    assert fixed["ok"] is True
    assert fixed["summary"]["resolved"] == 1
    regressed = compare_reports({"findings": []}, before)
    assert regressed["ok"] is False
    assert regressed["summary"]["new_medium"] == 1


def check_history_adjusted_priorities():
    finding = {"finding_id": "sc-1", "severity": "medium"}
    approved = collect_signals(
        self_code={"findings": [finding]},
        history=[{"kind": "self_code_medium", "decision": "approve"}] * 2)
    rejected = collect_signals(
        self_code={"findings": [finding]},
        history=[{"kind": "self_code_medium", "decision": "reject"}] * 2)
    assert approved[0]["kind"] == "self_code_medium"
    assert approved[0]["priority"] == 77
    assert rejected[0]["priority"] == 67


def check_escalation_router_and_growth():
    local = decide(
        {"kind": "answer", "confidence": 0.9, "privacy": "public"},
        policy={})
    assert local["route"] == "local"
    assert local["external_call"] is False
    teacher = decide(
        {"kind": "answer", "confidence": 0.2, "privacy": "public"},
        policy={"teacher_enabled": True, "allow_external": True,
                "max_external_calls": 1})
    assert teacher["route"] == "teacher"
    secret = decide(
        {"kind": "answer", "confidence": 0.2, "privacy": "secret"},
        policy={})
    assert secret["route"] == "refuse"
    assert secret["reason_code"] == "privacy_secret"
    agent = decide(
        {"kind": "code", "confidence": 0.1, "privacy": "public"},
        policy={"agent_enabled": True, "max_agent_calls": 1})
    assert agent["route"] == "agent"
    assert agent["requires_human"] is True
    ledger = []
    record_escalation(ledger, teacher, outcome="success", verified=True)
    summary = summarize_growth(ledger)
    assert summary["verified_teacher_answers"] == 1
    assert summary["adapter_candidates"] == 1
    assert summary["parameter_growth"] == 0
    assert summary["weights_changed"] is False


def check_scoreboard_output_parsers():
    assert parse_pytest_summary(
        "2 failed, 855 passed, 1 warning in 12.3s") == {
            "tests_passed": 855, "tests_failed": 2}
    assert parse_benchmark_summary(
        "summary: 13/13 checks passed") == {
            "benchmark_passed": 13, "benchmark_total": 13}
    assert parse_aisc_summary(
        "summary: 22 specs, 20 pass, 2 fail, 0 error") == {
            "aisc_passed": 20, "aisc_total": 22}


def check_growth_report_combines_layers():
    runs = []
    record_run(
        runs, commit="aaaaaaa", tests_passed=985, tests_failed=0,
        benchmark_passed=18, benchmark_total=18,
        aisc_passed=22, aisc_total=24, now="2026-11-24T12:00:00Z")
    report = build_report(
        scoreboard=runs,
        verification_ledger=[{"verified": True, "promotable": True}],
        self_code_diff={"ok": True, "summary": {
            "resolved": 1, "introduced": 0, "persisting": 0,
            "new_high": 0, "new_medium": 0}})
    assert report["ok"] is True
    assert report["verification"]["promotable"] == 1
    assert report["capability"]["code_findings_resolved"] == 1
    assert report["parameter_growth"] == 0
    assert report["weights_changed"] is False
    blocked = build_report(scoreboard=[])
    assert blocked["ok"] is False
    assert "no_scoreboard" in blocked["reasons"]


def check_teacher_verification_is_content_free():
    hits = [{"chunk": "The launch code is ORBIT-42.",
             "source": "doc://policy"}]
    result = verify_answer("The launch code is ORBIT-42.", hits)
    assert result["status"] == "verified"
    assert result["verified"] is True
    event = verification_event(result)
    assert event["promotable"] is True
    assert event["auto_apply"] is False
    assert event["weights_changed"] is False
    assert not ({"answer", "text", "query", "source", "content"} & set(event))
    bad = verify_answer("The launch code is ZEBRA-99.", hits)
    assert bad["status"] == "unverified"
    assert bad["verified"] is False


def check_adapter_candidate_gate():
    examples = [
        {"user": f"q{index}", "assistant": f"a{index}",
         "verification": {"verified": True, "promotable": True}}
        for index in range(4)
    ]
    candidate = build_adapter_candidate(examples, eval_ratio=0.25)
    assert candidate["metadata"]["eval_examples"] == 1
    passed = evaluate_adapter_candidate(candidate, {
        "examples": 1,
        "baseline": {"loss": 1.2, "ppl": 3.3},
        "adapter": {"loss": 1.0, "ppl": 3.0},
    })
    assert passed["status"] == "pass"
    assert passed["promotable"] is True
    assert passed["auto_apply"] is False
    assert passed["weights_changed"] is False
    failed = evaluate_adapter_candidate(candidate, {
        "examples": 2,
        "baseline": {"loss": 1.2, "ppl": 3.3},
        "adapter": {"loss": 1.0, "ppl": 3.0},
    })
    assert failed["status"] == "fail"


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
    check_self_code_audit_and_plan,
    check_self_code_diff_is_a_gate,
    check_history_adjusted_priorities,
    check_escalation_router_and_growth,
    check_scoreboard_output_parsers,
    check_growth_report_combines_layers,
    check_teacher_verification_is_content_free,
    check_adapter_candidate_gate,
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
