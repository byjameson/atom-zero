import copy
import hashlib
import io
import json
import sys
from datetime import datetime, timezone

import pytest

from atomzero.improvement import (
    METHOD,
    SIGNAL_PRIORITIES,
    VALID_DECISIONS,
    build_plan,
    collect_signals,
    main,
    plan_from_inputs,
    record_decision,
    record_gated_decision,
    summarize_ledger,
    verify_ledger,
)

NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
ACTION = {
    "action_id": "imp-01-unsupported_claims",
    "kind": "unsupported_claims",
}
ALL_KINDS = [
    "unsupported_claims",
    "conflicts",
    "knowledge_gaps",
    "single_source",
    "stale_sources",
    "undated_sources",
    "weak_sources",
    "pending_feedback",
]


def _write(tmp_path, name, payload):
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _full_inputs():
    return {
        "claim_audit": {
            "unsupported": 2,
            "conflict_count": 1,
            "single_source": 3,
        },
        "gaps": [
            {"id": "g0001", "query": "What is Q1?", "status": "pending",
             "count": 2},
            {"id": "g0002", "query": "What is Q2?", "status": "closed",
             "count": 99},
        ],
        "feedback": [
            {"id": "f1", "status": "pending"},
            {"id": "f2", "status": "rejected"},
            {"id": "f3", "status": "approved"},
            {"id": "f4", "status": "exported"},
        ],
        "freshness": {
            "summary": {
                "status_counts": {
                    "fresh": 5, "aging": 0, "stale": 2, "future": 0,
                    "undated": 1,
                },
                "single_source_clusters": 3,
                "conflict_count": 1,
            },
        },
        "reliability": {
            "sources": [
                {"source_id": "a" * 16, "status": "weak",
                 "observations": 4},
                {"source_id": "b" * 16, "status": "reliable",
                 "observations": 1},
                {"source_id": "c" * 16, "status": "reliable",
                 "observations": 3},
            ],
        },
    }


def _by_kind(signals):
    return {signal["kind"]: signal for signal in signals}


def test_collect_signals_covers_each_kind_and_priority_order():
    signals = collect_signals(**_full_inputs())
    assert [signal["kind"] for signal in signals] == ALL_KINDS
    priorities = [signal["priority"] for signal in signals]
    assert priorities == sorted(priorities, reverse=True)
    assert priorities == [SIGNAL_PRIORITIES[kind] for kind in ALL_KINDS]
    assert all(
        set(signal) == {
            "kind", "priority", "count", "title", "detail", "refs"}
        for signal in signals)
    by_kind = _by_kind(signals)
    assert by_kind["unsupported_claims"]["count"] == 2
    assert by_kind["conflicts"]["count"] == 1
    assert by_kind["knowledge_gaps"]["count"] == 1
    assert by_kind["single_source"]["count"] == 3
    assert by_kind["stale_sources"]["count"] == 2
    assert by_kind["undated_sources"]["count"] == 1
    assert by_kind["weak_sources"]["count"] == 2
    assert by_kind["pending_feedback"]["count"] == 2


def test_zero_count_signals_are_excluded():
    signals = collect_signals(
        claim_audit={"unsupported": 0, "conflict_count": 0,
                     "single_source": 0},
        gaps=[],
        feedback=[],
        freshness={"status_counts": {"stale": 0, "undated": 0},
                   "single_source_clusters": 0, "conflict_count": 0},
        reliability={"sources": []},
    )
    assert signals == []


def test_claim_audit_signal_references_stay_local_and_content_free():
    signals = collect_signals(claim_audit={
        "unsupported": 1, "conflict_count": 0, "single_source": 0})
    assert [signal["kind"] for signal in signals] == ["unsupported_claims"]
    signal = signals[0]
    assert signal["refs"] == ["claim_audit"]
    assert "claim" in signal["detail"].lower()
    assert set(signal["refs"]) == {"claim_audit"}


def test_gaps_filter_closed_statuses_and_keep_top_gap_reference():
    signals = collect_signals(gaps=[
        {"id": "g0001", "query": "low count", "status": "done",
         "count": 50},
        {"id": "g0002", "query": "small pending", "status": "pending",
         "count": 1},
        {"id": "g0003", "query": "top pending", "status": "proposed",
         "count": 9},
        {"id": "g0004", "query": "dismissed", "status": "dismissed",
         "count": 40},
        {"id": "g0005", "query": "closed", "status": "closed",
         "count": 30},
    ])
    assert len(signals) == 1
    signal = signals[0]
    assert signal["kind"] == "knowledge_gaps"
    assert signal["count"] == 2
    assert signal["refs"] == ["gap:g0003", "query:top pending"]


def test_feedback_counts_everything_not_approved_or_exported():
    signals = collect_signals(feedback=[
        {"id": "f1", "status": "pending"},
        {"id": "f2", "status": "rejected"},
        {"id": "f3", "status": "approved"},
        {"id": "f4", "status": "exported"},
        {"id": "f5", "status": "dismissed"},
    ])
    assert len(signals) == 1
    signal = signals[0]
    assert signal["kind"] == "pending_feedback"
    assert signal["count"] == 3
    assert signal["refs"] == [
        "feedback:f1", "feedback:f2", "feedback:f5"]


def test_freshness_flat_and_nested_summaries_are_supported():
    nested = collect_signals(freshness={
        "summary": {"status_counts": {"stale": 2, "undated": 3},
                    "single_source_clusters": 4, "conflict_count": 5}})
    flat = collect_signals(freshness={
        "status_counts": {"stale": 2, "undated": 3},
        "single_source_clusters": 4, "conflict_count": 5})
    assert nested == flat
    by_kind = _by_kind(nested)
    assert by_kind["stale_sources"]["count"] == 2
    assert by_kind["undated_sources"]["count"] == 3
    assert by_kind["single_source"]["count"] == 4
    assert by_kind["conflicts"]["count"] == 5


def test_claim_and_freshness_views_are_not_double_counted():
    signals = collect_signals(
        claim_audit={"conflict_count": 2, "single_source": 7},
        freshness={"conflict_count": 5, "single_source_clusters": 3},
    )
    by_kind = _by_kind(signals)
    assert by_kind["conflicts"]["count"] == 5
    assert by_kind["single_source"]["count"] == 7


def test_reliability_counts_weak_or_low_observation_sources():
    signals = collect_signals(reliability={"sources": [
        {"source_id": "a" * 16, "status": "weak", "observations": 5},
        {"source_id": "b" * 16, "status": "reliable", "observations": 1},
        {"source_id": "c" * 16, "status": "mixed", "observations": 0},
        {"source_id": "d" * 16, "status": "reliable", "observations": 3},
    ]})
    assert len(signals) == 1
    signal = signals[0]
    assert signal["kind"] == "weak_sources"
    assert signal["count"] == 3
    assert signal["refs"] == [
        "source:" + "a" * 16,
        "source:" + "b" * 16,
        "source:" + "c" * 16,
    ]


def test_collect_signals_is_deterministic():
    first = collect_signals(**_full_inputs())
    second = collect_signals(**_full_inputs())
    assert first == second


@pytest.mark.parametrize(
    "kwargs",
    [
        {"claim_audit": []},
        {"gaps": {}},
        {"feedback": {}},
        {"freshness": []},
        {"reliability": []},
        {"claim_audit": {"unsupported": -1}},
        {"claim_audit": {"unsupported": True}},
        {"freshness": {"status_counts": {"stale": "many"}}},
        {"reliability": {"sources": [{"observations": -1}]}},
        {"gaps": ["not-a-mapping"]},
        {"feedback": ["not-a-mapping"]},
    ],
)
def test_collect_signals_rejects_invalid_input(kwargs):
    with pytest.raises(ValueError):
        collect_signals(**kwargs)


def test_build_plan_sorts_limits_and_adds_human_gates():
    plan = build_plan(collect_signals(**_full_inputs()), limit=3)
    assert [action["kind"] for action in plan["actions"]] == ALL_KINDS[:3]
    assert all(
        action["requires_human"] is True for action in plan["actions"])
    assert all(
        action["acceptance"].endswith(
            "run the relevant pytest/AISC regression before accepting")
        for action in plan["actions"])
    assert (
        "add a supporting source and rerun claim audit"
        in plan["actions"][0]["acceptance"])
    assert plan["summary"] == {
        "conflicts": 1,
        "knowledge_gaps": 1,
        "unsupported_claims": 1,
    }
    assert all(
        isinstance(action["priority"], int) for action in plan["actions"])
    assert plan["auto_apply"] is False
    assert plan["weights_changed"] is False
    assert plan["method"] == METHOD


def test_build_plan_acceptance_gate_is_present_for_every_kind():
    plan = build_plan(collect_signals(**_full_inputs()), limit=20)
    assert [action["kind"] for action in plan["actions"]] == ALL_KINDS
    acceptances = {
        action["kind"]: action["acceptance"] for action in plan["actions"]}
    assert "second source" in acceptances["single_source"]
    assert "freshness audit" in acceptances["stale_sources"]
    assert "dated" in acceptances["undated_sources"]
    assert "observations" in acceptances["weak_sources"]
    assert "approve or reject" in acceptances["pending_feedback"]
    assert all(
        "pytest/AISC regression" in value
        for value in acceptances.values())


@pytest.mark.parametrize("limit", [0, -1, 21, True, 5.0, "5", None])
def test_build_plan_rejects_invalid_limit(limit):
    with pytest.raises(ValueError):
        build_plan([], limit=limit)


@pytest.mark.parametrize("limit", [1, 5, 20])
def test_build_plan_accepts_valid_limit(limit):
    plan = build_plan(collect_signals(**_full_inputs()), limit=limit)
    assert len(plan["actions"]) == min(limit, len(ALL_KINDS))


@pytest.mark.parametrize(
    "signals",
    [
        "not-a-list",
        [None],
        [{}],
        [{"kind": 3, "priority": 1}],
        [{"kind": "x", "priority": True}],
        [{"kind": "x", "priority": float("nan")}],
        [{"kind": "x", "priority": 1, "count": -1}],
    ],
)
def test_build_plan_rejects_invalid_signals(signals):
    with pytest.raises(ValueError):
        build_plan(signals, limit=1)


def test_build_plan_is_deterministic_and_flags_stay_false():
    signals = collect_signals(**_full_inputs())
    first = build_plan(signals, limit=5)
    second = build_plan(signals, limit=5)
    assert first == second
    assert first["auto_apply"] is False
    assert first["weights_changed"] is False


def test_build_plan_defaults_for_minimal_signals_and_unknown_kind():
    plan = build_plan([{"kind": "custom_kind", "priority": 1}], limit=1)
    action = plan["actions"][0]
    assert action["title"] == "Custom kind"
    assert action["requires_human"] is True
    assert action["acceptance"] == (
        "run the relevant pytest/AISC regression before accepting")


def test_plan_from_inputs_composes_collect_and_plan():
    inputs = _full_inputs()
    expected = build_plan(collect_signals(**inputs), limit=2)
    assert plan_from_inputs(limit=2, **inputs) == expected


def test_plan_from_inputs_rejects_unknown_input():
    with pytest.raises(ValueError):
        plan_from_inputs(weights="not-allowed")


def test_record_decision_appends_content_free_hashed_record():
    ledger = []
    record = record_decision(
        ledger, ACTION, "approve", note="  add a source  ", now=NOW)
    assert ledger == [record]
    assert record == {
        "version": 1,
        "at": "2026-01-01T12:00:00Z",
        "action_id": "imp-01-unsupported_claims",
        "kind": "unsupported_claims",
        "decision": "approve",
        "note_id": hashlib.sha256(
            b"add a source").hexdigest()[:16],
    }
    assert "add a source" not in json.dumps(record)
    assert record["note_id"] != "add a source"


def test_record_decision_empty_note_and_defer():
    ledger = []
    record = record_decision(ledger, ACTION, "defer", now=NOW)
    assert record["note_id"] == ""
    assert record["decision"] == "defer"


@pytest.mark.parametrize(
    "decision", ["accept", "", "APPROVE", None, 1, [], {}])
def test_record_decision_rejects_invalid_decision(decision):
    with pytest.raises(ValueError):
        record_decision([], ACTION, decision, now=NOW)


@pytest.mark.parametrize(
    "ledger,action,note",
    [
        ("not-a-list", ACTION, ""),
        ([], "not-a-mapping", ""),
        ([], {}, ""),
        ([], {"action_id": "x"}, ""),
        ([], {"kind": "x"}, ""),
        ([], ACTION, 5),
        ([], ACTION, b"bytes"),
    ],
)
def test_record_decision_rejects_invalid_arguments(ledger, action, note):
    with pytest.raises(ValueError):
        record_decision(ledger, action, "approve", note=note, now=NOW)


def test_actions_carry_allowlisted_next_steps():
    plan = plan_from_inputs(
        gaps=[{"id": "g1", "query": "local policy", "status": "open"}],
        feedback=[{"id": "f1", "status": "pending"}],
    )
    by_kind = {action["kind"]: action for action in plan["actions"]}
    assert "/v1/learning/gaps/propose" in by_kind["knowledge_gaps"]["next_step"]
    assert "feedback queue" in by_kind["pending_feedback"]["next_step"]
    assert by_kind["knowledge_gaps"]["requires_human"] is True
    assert by_kind["knowledge_gaps"]["suggested_checks"] == [
        "pytest:tests/test_improvement.py"]
    weak = plan_from_inputs(reliability={
        "sources": [{"source_id": "abc", "status": "weak",
                     "observations": 1}]})
    assert "source reliability" in weak["actions"][0]["next_step"]
    assert weak["actions"][0]["suggested_checks"] == [
        "pytest:tests/test_reliability.py"]


def test_aisc_coverage_signal_and_action():
    signals = collect_signals(aisc_coverage={
        "summary": {
            "specs_with_missing_evidence": 1,
            "specs_with_unverified_components": 2,
            "errors": 0,
        }})
    assert len(signals) == 1
    signal = signals[0]
    assert signal["kind"] == "aisc_coverage_gaps"
    assert signal["count"] == 3
    plan = build_plan(signals)
    assert plan["actions"][0]["kind"] == "aisc_coverage_gaps"
    assert "AISC coverage" in plan["actions"][0]["acceptance"]
    assert plan["actions"][0]["suggested_checks"] == [
        "pytest:tests/test_aisc_coverage.py",
        "aisc:docs/aisc/atomzero-read.toml",
    ]


GATE_PASS = {
    "ok": True,
    "summary": {"total": 2, "passed": 2, "failed": 0},
    "method": "allowlisted_evaluation_gate_v1",
}
GATE_FAIL = {
    "ok": False,
    "summary": {"total": 2, "passed": 1, "failed": 1},
    "method": "allowlisted_evaluation_gate_v1",
}


def test_gated_approval_requires_passing_gate():
    ledger = []
    record = record_gated_decision(
        ledger, ACTION, "approve", GATE_PASS, note="reviewed", now=NOW)
    assert record["gate_ok"] is True
    assert record["gate_total"] == 2
    assert verify_ledger(ledger)["ok"] is True
    with pytest.raises(ValueError, match="passing evaluation gate"):
        record_gated_decision([], ACTION, "approve", GATE_FAIL, now=NOW)


def test_reject_can_carry_failed_gate():
    ledger = []
    record = record_gated_decision(ledger, ACTION, "reject", GATE_FAIL, now=NOW)
    assert record["gate_ok"] is False
    assert verify_ledger(ledger)["ok"] is True


def test_gate_summary_is_validated():
    with pytest.raises(ValueError):
        record_gated_decision([], ACTION, "approve", {
            "ok": True, "summary": {"total": 2, "passed": 1, "failed": 0},
            "method": "gate"}, now=NOW)
    with pytest.raises(ValueError):
        record_gated_decision([], ACTION, "approve", {
            "ok": True, "summary": {"total": 1, "passed": 1, "failed": 0}},
            now=NOW)


def test_verify_ledger_rejects_tampered_gate():
    ledger = []
    record_gated_decision(ledger, ACTION, "approve", GATE_PASS, now=NOW)
    ledger[0]["gate_ok"] = False
    assert verify_ledger(ledger)["ok"] is False


def test_scoreboard_regression_is_the_highest_priority_signal():
    signals = collect_signals(scoreboard={
        "latest": {"tests_failed": 1},
        "delta": {"tests_passed": -3, "benchmark_passed": -1},
    })
    assert signals[0]["kind"] == "scoreboard_regression"
    assert signals[0]["count"] == 3
    plan = build_plan(signals)
    action = plan["actions"][0]
    assert action["kind"] == "scoreboard_regression"
    assert "regressed metric" in action["acceptance"]
    assert action["suggested_checks"] == [
        "pytest:tests/test_scoreboard.py",
        "pytest:tests/test_improvement.py",
    ]


def test_calibration_drift_signal():
    signals = collect_signals(drift={
        "alerts": ["brier_increased", "ece_increased"]})
    assert signals[0]["kind"] == "calibration_drift"
    assert signals[0]["count"] == 2
    plan = build_plan(signals)
    action = plan["actions"][0]
    assert action["kind"] == "calibration_drift"
    assert "recalibrate" in action["acceptance"]
    assert action["suggested_checks"] == [
        "pytest:tests/test_drift.py",
        "pytest:tests/test_answerability.py",
    ]


def test_scoreboard_without_regression_adds_no_signal():
    assert collect_signals(scoreboard={
        "latest": {"tests_failed": 0},
        "delta": {"tests_passed": 2, "benchmark_passed": 1},
    }) == []


def test_summarize_ledger_is_content_free():
    ledger = []
    record_decision(ledger, ACTION, "approve", note="private note", now=NOW)
    record_decision(ledger, ACTION, "reject", now=NOW)
    summary = summarize_ledger(ledger)
    assert summary["total"] == 2
    assert summary["decisions"] == {"approve": 1, "reject": 1}
    assert summary["kinds"] == {"unsupported_claims": 2}
    assert summary["weights_changed"] is False
    assert "private note" not in json.dumps(summary)


def test_summarize_ledger_rejects_invalid_ledger():
    with pytest.raises(ValueError):
        summarize_ledger([{"text": "leak"}])


def test_verify_ledger_accepts_records_without_note_text():
    ledger = []
    for decision in sorted(VALID_DECISIONS):
        record_decision(
            ledger, ACTION, decision,
            note="private review note", now=NOW)
    result = verify_ledger(ledger)
    assert result == {"ok": True, "entries": 3, "errors": []}


@pytest.mark.parametrize(
    "forbidden",
    ["text", "query", "answer", "chunk", "content", "note", "session",
     "source"],
)
def test_verify_ledger_rejects_content_keys(forbidden):
    ledger = []
    record_decision(ledger, ACTION, "approve", now=NOW)
    bad = copy.deepcopy(ledger)
    bad[0][forbidden] = "private text"
    result = verify_ledger(bad)
    assert result["ok"] is False
    assert any(forbidden in error for error in result["errors"])


@pytest.mark.parametrize(
    "mutate",
    [
        lambda entry: entry.update(decision="accept"),
        lambda entry: entry.update(decision=[]),
        lambda entry: entry.update(version=2),
        lambda entry: entry.update(note_id="not hex!"),
        lambda entry: entry.update(action_id=""),
        lambda entry: entry.update(kind=""),
        lambda entry: entry.update(at=""),
        lambda entry: entry.pop("note_id"),
        lambda entry: entry.update(extra="value"),
    ],
)
def test_verify_ledger_rejects_malformed_records(mutate):
    ledger = []
    record_decision(ledger, ACTION, "approve", now=NOW)
    bad = copy.deepcopy(ledger)
    mutate(bad[0])
    result = verify_ledger(bad)
    assert result["ok"] is False
    assert result["entries"] == 1
    assert result["errors"]


def test_verify_ledger_rejects_non_list_and_non_mapping_entries():
    assert verify_ledger({})["ok"] is False
    assert verify_ledger({})["entries"] == 0
    result = verify_ledger(["not-a-mapping"])
    assert result["ok"] is False
    assert result["entries"] == 1


def test_cli_json_with_files(tmp_path, capsys):
    claims = _write(tmp_path, "claims.json", {
        "unsupported": 2, "conflict_count": 1, "single_source": 0})
    gaps = _write(tmp_path, "gaps.json", [
        {"id": "g1", "query": "Q", "status": "pending", "count": 1}])
    code = main([
        "--claims-json", str(claims),
        "--gaps-json", str(gaps),
        "--limit", "1",
        "--json",
    ])
    captured = capsys.readouterr()
    assert code == 0
    plan = json.loads(captured.out)
    assert len(plan["actions"]) == 1
    assert plan["actions"][0]["kind"] == "unsupported_claims"
    assert plan["auto_apply"] is False
    assert plan["weights_changed"] is False
    assert captured.err == ""


def test_cli_all_input_files(tmp_path, capsys):
    inputs = _full_inputs()
    paths = {
        "claims": _write(tmp_path, "claims.json", inputs["claim_audit"]),
        "gaps": _write(tmp_path, "gaps.json", inputs["gaps"]),
        "feedback": _write(
            tmp_path, "feedback.json", inputs["feedback"]),
        "freshness": _write(
            tmp_path, "freshness.json", inputs["freshness"]),
        "reliability": _write(
            tmp_path, "reliability.json", inputs["reliability"]),
    }
    code = main([
        "--claims-json", str(paths["claims"]),
        "--gaps-json", str(paths["gaps"]),
        "--feedback-json", str(paths["feedback"]),
        "--freshness-json", str(paths["freshness"]),
        "--reliability-json", str(paths["reliability"]),
        "--limit", "8",
        "--json",
    ])
    captured = capsys.readouterr()
    assert code == 0
    plan = json.loads(captured.out)
    assert [action["kind"] for action in plan["actions"]] == ALL_KINDS


def test_cli_reads_one_stdin_payload(monkeypatch, capsys):
    payload = {
        "claims": {"unsupported": 1, "conflict_count": 0,
                   "single_source": 0},
        "gaps": [{"id": "g1", "query": "Q", "status": "pending"}],
    }
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))
    code = main(["--json"])
    captured = capsys.readouterr()
    assert code == 0
    plan = json.loads(captured.out)
    assert {action["kind"] for action in plan["actions"]} == {
        "unsupported_claims", "knowledge_gaps"}


def test_cli_gaps_file_can_be_stdin(monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps([
        {"id": "g1", "query": "Q", "status": "pending"}])))
    code = main(["--gaps-json", "-", "--json"])
    captured = capsys.readouterr()
    assert code == 0
    assert json.loads(captured.out)["actions"][0]["kind"] == "knowledge_gaps"


def test_cli_empty_stdin_exits_2(monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    code = main(["--json"])
    captured = capsys.readouterr()
    assert code == 2
    assert captured.out == ""
    assert "no input payload" in captured.err


def test_cli_empty_object_stdin_exits_2(monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", io.StringIO("{}"))
    code = main(["--json"])
    captured = capsys.readouterr()
    assert code == 2
    assert "empty input" in captured.err


def test_cli_unknown_stdin_key_exits_2(monkeypatch, capsys):
    monkeypatch.setattr(
        sys, "stdin", io.StringIO(json.dumps({"weights": []})))
    code = main(["--json"])
    captured = capsys.readouterr()
    assert code == 2
    assert "unknown input key" in captured.err


def test_cli_invalid_json_exits_2(tmp_path, capsys):
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    code = main(["--claims-json", str(path), "--json"])
    captured = capsys.readouterr()
    assert code == 2
    assert captured.out == ""
    assert captured.err.startswith("error:")


def test_cli_missing_file_exits_2(tmp_path, capsys):
    code = main(["--gaps-json", str(tmp_path / "missing.json"), "--json"])
    captured = capsys.readouterr()
    assert code == 2
    assert captured.err.startswith("error:")


def test_cli_invalid_limit_exits_2(tmp_path, capsys):
    claims = _write(tmp_path, "claims.json", {"unsupported": 1})
    code = main(["--claims-json", str(claims), "--limit", "0", "--json"])
    captured = capsys.readouterr()
    assert code == 2
    assert "limit must be an integer" in captured.err


def test_cli_zero_signals_is_valid_and_exits_0(tmp_path, capsys):
    claims = _write(tmp_path, "claims.json", {
        "unsupported": 0, "conflict_count": 0, "single_source": 0})
    code = main(["--claims-json", str(claims), "--json"])
    captured = capsys.readouterr()
    assert code == 0
    plan = json.loads(captured.out)
    assert plan["actions"] == []
    assert plan["summary"] == {}


def test_cli_human_output_includes_gate_and_flags(tmp_path, capsys):
    gaps = _write(tmp_path, "gaps.json", [
        {"id": "g1", "query": "Q", "status": "pending"}])
    code = main(["--gaps-json", str(gaps), "--limit", "1"])
    captured = capsys.readouterr()
    assert code == 0
    assert "auto_apply: false" in captured.out
    assert "weights_changed: false" in captured.out
    assert "requires human approval: true" in captured.out
    assert "acceptance:" in captured.out
