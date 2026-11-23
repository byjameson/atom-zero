import io
import json
import sys
from datetime import datetime

import pytest

from atomzero.escalation import (
    FORBIDDEN_KEYS,
    GROWTH_METHOD,
    METHOD,
    decide,
    main,
    record_escalation,
    summarize_growth,
)

NOW = datetime(2026, 11, 24, 12, 0, 0)
NOW_TEXT = "2026-11-24T12:00:00Z"
TEACHER_POLICY = {
    "teacher_enabled": True,
    "allow_external": True,
    "max_external_calls": 2,
}


def task(**overrides):
    data = {"kind": "answer", "confidence": 0.9, "privacy": "public"}
    data.update(overrides)
    return data


def policy(**overrides):
    data = dict(TEACHER_POLICY)
    data.update(overrides)
    return data


def route_decision(route, reason_code):
    return {"route": route, "reason_code": reason_code}


def write_json(tmp_path, name, value):
    path = tmp_path / name
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_local_confident_route():
    decision = decide(task(confidence=0.84), policy={})
    assert decision["route"] == "local"
    assert decision["reason_code"] == "local_confident"
    assert decision["confidence_bucket"] == "high"
    assert decision["requires_human"] is False
    assert decision["external_call"] is False
    assert decision["auto_apply"] is False
    assert decision["weights_changed"] is False
    assert decision["method"] == METHOD
    assert decision["budget"] == {
        "external_calls": 0,
        "external_limit": 0,
        "external_remaining": 0,
        "agent_calls": 0,
        "agent_limit": 0,
        "agent_remaining": 0,
    }


def test_min_local_confidence_is_inclusive():
    decision = decide(task(confidence=0.55),
                      policy={"min_local_confidence": 0.55})
    assert decision["route"] == "local"
    assert decision["confidence_bucket"] == "medium"


def test_low_confidence_teacher_route():
    decision = decide(task(confidence=0.3), policy=policy())
    assert decision["route"] == "teacher"
    assert decision["reason_code"] == "low_confidence_or_freshness"
    assert decision["confidence_bucket"] == "low"
    assert decision["requires_human"] is False
    assert decision["external_call"] is False
    assert decision["budget"]["external_calls"] == 0
    assert decision["budget"]["external_limit"] == 2
    assert decision["budget"]["external_remaining"] == 2


def test_freshness_forces_teacher_even_when_confident():
    decision = decide(
        task(confidence=0.98, requires_freshness=True), policy=policy())
    assert decision["route"] == "teacher"
    assert decision["reason_code"] == "low_confidence_or_freshness"
    assert decision["confidence_bucket"] == "high"


def test_secret_privacy_refuses_before_other_routes():
    decision = decide(
        task(kind="code", confidence=0.99, privacy="secret"),
        policy=policy(agent_enabled=True, max_agent_calls=5))
    assert decision["route"] == "refuse"
    assert decision["reason_code"] == "privacy_secret"
    assert decision["requires_human"] is False
    assert decision["external_call"] is False


def test_private_privacy_blocks_teacher():
    decision = decide(task(confidence=0.2, privacy="private"),
                      policy=policy())
    assert decision["route"] == "refuse"
    assert decision["reason_code"] == "privacy_private"


def test_private_privacy_without_external_route_is_generic_refusal():
    decision = decide(task(confidence=0.2, privacy="private"), policy={})
    assert decision["route"] == "refuse"
    assert decision["reason_code"] == "no_available_route"


def test_code_agent_handoff_requires_human_gate():
    decision = decide(
        task(kind="code", confidence=0.99),
        policy={"agent_enabled": True, "max_agent_calls": 1})
    assert decision["route"] == "agent"
    assert decision["reason_code"] == "bounded_code_handoff"
    assert decision["requires_human"] is True
    assert decision["external_call"] is False
    assert decision["auto_apply"] is False
    assert decision["weights_changed"] is False
    assert decision["budget"]["agent_remaining"] == 1


def test_agent_budget_exhausted_falls_through_to_teacher():
    history = [{"route": "agent"}]
    decision = decide(
        task(kind="code", confidence=0.2),
        policy=policy(agent_enabled=True, max_agent_calls=1),
        history=history)
    assert decision["route"] == "teacher"
    assert decision["budget"]["agent_calls"] == 1
    assert decision["budget"]["agent_remaining"] == 0


def test_teacher_budget_exhausted_refuses():
    history = [{"route": "teacher"}, {"route": "teacher"}]
    decision = decide(
        task(confidence=0.2), policy=policy(), history=history)
    assert decision["route"] == "refuse"
    assert decision["reason_code"] == "no_available_route"
    assert decision["budget"]["external_calls"] == 2
    assert decision["budget"]["external_remaining"] == 0


def test_history_only_counts_teacher_and_agent_routes():
    history = [
        {"route": "local"},
        {"route": "refuse"},
        {"route": "teacher", "outcome": "failure"},
        {"route": "agent", "outcome": "success"},
    ]
    decision = decide(
        task(confidence=0.2),
        policy=policy(agent_enabled=True, max_agent_calls=3),
        history=history)
    assert decision["budget"]["external_calls"] == 1
    assert decision["budget"]["agent_calls"] == 1
    assert decision["budget"]["external_remaining"] == 1
    assert decision["budget"]["agent_remaining"] == 2


@pytest.mark.parametrize("bad_task, message", [
    (None, "task must be a mapping"),
    ("plain text", "task must be a mapping"),
    ({}, "task.kind must be one of"),
    ({"kind": "other", "confidence": 0.5, "privacy": "public"},
     "task.kind must be one of"),
    ({"kind": "answer", "privacy": "public"}, "task.confidence is required"),
    ({"kind": "answer", "confidence": 1.5, "privacy": "public"},
     "task.confidence must be a number"),
    ({"kind": "answer", "confidence": -0.1, "privacy": "public"},
     "task.confidence must be a number"),
    ({"kind": "answer", "confidence": True, "privacy": "public"},
     "task.confidence must be a number"),
    ({"kind": "answer", "confidence": float("nan"), "privacy": "public"},
     "task.confidence must be a number"),
    ({"kind": "answer", "confidence": 0.5, "privacy": "world"},
     "task.privacy must be one of"),
    ({"kind": "answer", "confidence": 0.5, "privacy": "public",
      "requires_freshness": "yes"}, "task.requires_freshness must be a boolean"),
    ({"kind": "answer", "confidence": 0.5, "privacy": "public",
      "complexity": "extreme"}, "task.complexity must be one of"),
    ({"kind": "answer", "confidence": 0.5, "privacy": "public",
      "query": "private question"}, "content key"),
])
def test_malformed_tasks_raise_value_error(bad_task, message):
    with pytest.raises(ValueError, match=message):
        decide(bad_task, policy={})


@pytest.mark.parametrize("bad_policy, message", [
    (None, "policy must be a mapping"),
    ({"teacher_enabled": "yes"}, "policy.teacher_enabled must be a boolean"),
    ({"allow_external": 1}, "policy.allow_external must be a boolean"),
    ({"max_external_calls": -1},
     "policy.max_external_calls must be a non-negative integer"),
    ({"max_agent_calls": 1.5},
     "policy.max_agent_calls must be a non-negative integer"),
    ({"min_local_confidence": 1.2},
     "policy.min_local_confidence must be a number"),
    ({"unknown_setting": True}, "policy has unknown key"),
    ({"answer": "private answer"}, "policy content key"),
])
def test_malformed_policies_raise_value_error(bad_policy, message):
    with pytest.raises(ValueError, match=message):
        decide(task(), policy=bad_policy)


@pytest.mark.parametrize("bad_history, message", [
    ("not a list", "history must be a list"),
    ({}, "history must be a list"),
    ([None], "history\\[0\\] must be a mapping"),
    ([{"route": "unknown"}], "history\\[0\\].route must be one of"),
    ([{"route": "teacher", "query": "private"}], "content key"),
])
def test_malformed_history_raises_value_error(bad_history, message):
    with pytest.raises(ValueError, match=message):
        decide(task(), policy=policy(), history=bad_history)


def test_default_policy_is_not_mutated():
    supplied = {}
    decide(task(), policy=supplied)
    assert supplied == {}


def test_decision_is_deterministic_and_content_free():
    first = decide(task(confidence=0.2), policy=policy())
    second = decide(task(confidence=0.2), policy=policy())
    assert first == second
    assert set(first) == {
        "route", "reason_code", "confidence_bucket", "requires_human",
        "external_call", "auto_apply", "weights_changed", "budget", "method",
    }
    rendered = json.dumps(first)
    for key in FORBIDDEN_KEYS:
        assert f'"{key}"' not in rendered
    assert "private" not in rendered


def test_record_escalation_and_summarize_growth():
    ledger = []
    local = decide(task(confidence=0.9), policy={})
    local_event = record_escalation(ledger, local, outcome="success", now=NOW)
    assert local_event["growth_event"] == ""
    assert local_event["at"] == NOW_TEXT
    assert local_event["version"] == 1
    assert local_event["method"] == GROWTH_METHOD

    teacher = decide(task(confidence=0.2), policy=policy())
    verified = record_escalation(
        ledger, teacher, outcome="success", verified=True, now=NOW)
    assert verified["growth_event"] == "verified_teacher"
    record_escalation(ledger, teacher, outcome="failure", verified=True, now=NOW)
    record_escalation(ledger, teacher, outcome="success", verified=False, now=NOW)

    agent = decide(
        task(kind="code", confidence=0.2),
        policy={"agent_enabled": True, "max_agent_calls": 1})
    record_escalation(ledger, agent, outcome="success", now=NOW)

    refused = decide(task(privacy="secret"), policy={})
    record_escalation(ledger, refused, outcome="failure", now=NOW)

    summary = summarize_growth(ledger)
    assert summary["total"] == 6
    assert summary["routes"] == {
        "local": 1, "teacher": 3, "agent": 1, "refuse": 1}
    assert summary["verified_teacher_answers"] == 1
    assert summary["adapter_candidates"] == 1
    assert summary["external_calls"] == 3
    assert summary["agent_handoffs"] == 1
    assert summary["refusals"] == 1
    assert summary["parameter_growth"] == 0
    assert summary["parameter_growth_state"] == "unchanged"
    assert summary["weights_changed"] is False
    assert summary["auto_apply"] is False
    assert summary["last_at"] == NOW_TEXT
    assert summary["method"] == GROWTH_METHOD


def test_adapter_candidates_count_only_verified_teacher_events():
    ledger = []
    teacher = decide(task(confidence=0.1), policy=policy())
    agent = decide(
        task(kind="code", confidence=0.1),
        policy={"agent_enabled": True, "max_agent_calls": 1})
    record_escalation(ledger, teacher, outcome="success", verified=True, now=NOW)
    record_escalation(ledger, teacher, outcome="success", verified=True, now=NOW)
    record_escalation(ledger, teacher, outcome="failure", verified=True, now=NOW)
    record_escalation(ledger, teacher, outcome="success", verified=False, now=NOW)
    record_escalation(ledger, agent, outcome="success", verified=True, now=NOW)
    summary = summarize_growth(ledger)
    assert summary["verified_teacher_answers"] == 2
    assert summary["adapter_candidates"] == 2
    assert summary["external_calls"] == 4
    assert summary["agent_handoffs"] == 1


def test_parameter_growth_is_always_unchanged():
    empty = summarize_growth([])
    assert empty["parameter_growth"] == 0
    assert empty["parameter_growth_state"] == "unchanged"
    assert empty["weights_changed"] is False

    ledger = []
    record_escalation(
        ledger, decide(task(confidence=0.2), policy=policy()),
        outcome="success", verified=True, now=NOW)
    summary = summarize_growth(ledger)
    assert summary["parameter_growth"] == 0
    assert summary["parameter_growth_state"] == "unchanged"
    assert summary["weights_changed"] is False


@pytest.mark.parametrize("ledger, message", [
    ("not a list", "ledger must be a list"),
    ([None], "ledger\\[0\\] must be a mapping"),
    ([{"version": 2, "at": NOW_TEXT, "route": "local",
       "reason_code": "local_confident", "outcome": "success",
       "verified": False}], "version must be 1"),
    ([{"version": 1, "at": NOW_TEXT, "route": "local",
       "reason_code": "local_confident", "outcome": "success"}],
     "verified must be a boolean"),
    ([{"version": 1, "at": NOW_TEXT, "route": "local",
       "reason_code": "local_confident", "outcome": "wrong",
       "verified": False}], "outcome must be one of"),
    ([{"version": 1, "at": NOW_TEXT, "route": "local",
       "reason_code": "privacy_secret", "outcome": "success",
       "verified": False}], "is not valid for route"),
    ([{"version": 1, "at": NOW_TEXT, "route": "teacher",
       "reason_code": "low_confidence_or_freshness", "outcome": "success",
       "verified": True, "growth_event": ""}], "growth_event does not match"),
    ([{"version": 1, "at": NOW_TEXT, "route": "teacher",
       "reason_code": "low_confidence_or_freshness", "outcome": "success",
       "verified": True, "text": "leak"}], "content key"),
])
def test_malformed_ledgers_are_rejected(ledger, message):
    with pytest.raises(ValueError, match=message):
        summarize_growth(ledger)


def test_record_escalation_validates_decision_and_outcome():
    with pytest.raises(ValueError, match="decision must be a mapping"):
        record_escalation([], "teacher", outcome="success")
    with pytest.raises(ValueError, match="decision.route must be one of"):
        record_escalation(
            [], route_decision("nowhere", "local_confident"),
            outcome="success")
    with pytest.raises(ValueError, match="decision.reason_code must be one of"):
        record_escalation(
            [], route_decision("teacher", "made_up"), outcome="success")
    with pytest.raises(ValueError, match="is not valid for route"):
        record_escalation(
            [], route_decision("local", "privacy_secret"),
            outcome="success")
    with pytest.raises(ValueError, match="decision.external_call must be false"):
        record_escalation(
            [], {"route": "teacher",
                 "reason_code": "low_confidence_or_freshness",
                 "external_call": True},
            outcome="success")
    with pytest.raises(ValueError, match="outcome must be one of"):
        record_escalation(
            [], route_decision("teacher", "low_confidence_or_freshness"),
            outcome="ok")
    with pytest.raises(ValueError, match="verified must be a boolean"):
        record_escalation(
            [], route_decision("teacher", "low_confidence_or_freshness"),
            outcome="success", verified=1)
    with pytest.raises(ValueError, match="content key"):
        record_escalation(
            [], {"route": "teacher",
                 "reason_code": "low_confidence_or_freshness",
                 "answer": "leak"},
            outcome="success")


def test_record_escalation_rejects_a_corrupt_ledger():
    ledger = [{"version": 1, "at": NOW_TEXT, "route": "local",
               "reason_code": "local_confident", "outcome": "success",
               "verified": False, "text": "leak"}]
    with pytest.raises(ValueError, match="content key"):
        record_escalation(
            ledger, route_decision("local", "local_confident"),
            outcome="success")
    assert len(ledger) == 1


def test_cli_json_decision(tmp_path, capsys):
    task_path = write_json(tmp_path, "task.json", task(confidence=0.2))
    policy_path = write_json(tmp_path, "policy.json", policy())
    code = main([
        "--task-json", str(task_path),
        "--policy-json", str(policy_path),
        "--json",
    ])
    captured = capsys.readouterr()
    assert code == 0
    assert captured.err == ""
    decision = json.loads(captured.out)
    assert decision["route"] == "teacher"
    assert decision["external_call"] is False
    assert decision["auto_apply"] is False
    assert decision["weights_changed"] is False


def test_cli_reads_task_from_stdin(monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(task())))
    code = main(["--task-json", "-", "--json"])
    captured = capsys.readouterr()
    assert code == 0
    assert captured.err == ""
    assert json.loads(captured.out)["route"] == "local"


def test_cli_reads_stdin_only_once(monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(task())))
    code = main([
        "--task-json", "-", "--policy-json", "-", "--json",
    ])
    captured = capsys.readouterr()
    assert code == 2
    assert captured.out == ""
    assert "stdin can only be read once" in captured.err


def test_cli_summary_json(tmp_path, capsys):
    ledger = []
    teacher = decide(task(confidence=0.2), policy=policy())
    record_escalation(
        ledger, teacher, outcome="success", verified=True, now=NOW)
    record_escalation(ledger, teacher, outcome="failure", now=NOW)
    path = write_json(tmp_path, "ledger.json", ledger)
    code = main(["--summary-json", str(path)])
    captured = capsys.readouterr()
    assert code == 0
    assert captured.err == ""
    summary = json.loads(captured.out)
    assert summary["verified_teacher_answers"] == 1
    assert summary["adapter_candidates"] == 1
    assert summary["external_calls"] == 2
    assert summary["parameter_growth"] == 0
    assert summary["parameter_growth_state"] == "unchanged"


def test_cli_summary_rejects_decision_inputs(tmp_path, capsys):
    task_path = write_json(tmp_path, "task.json", task())
    ledger_path = write_json(tmp_path, "ledger.json", [])
    code = main([
        "--summary-json", str(ledger_path),
        "--task-json", str(task_path),
    ])
    captured = capsys.readouterr()
    assert code == 2
    assert captured.out == ""
    assert "--summary-json cannot be combined" in captured.err


def test_cli_requires_task_json(capsys):
    code = main([])
    captured = capsys.readouterr()
    assert code == 2
    assert captured.out == ""
    assert "--task-json is required" in captured.err


def test_cli_bad_inputs_exit_2(tmp_path, capsys):
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{not json", encoding="utf-8")
    assert main(["--task-json", str(invalid), "--json"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("error:")

    missing = tmp_path / "missing.json"
    assert main(["--task-json", str(missing), "--json"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("error:")

    as_list = write_json(tmp_path, "list.json", [])
    assert main(["--task-json", str(as_list), "--json"]) == 2
    captured = capsys.readouterr()
    assert "task must be a mapping" in captured.err


def test_cli_human_output(tmp_path, capsys):
    task_path = write_json(tmp_path, "task.json", task())
    code = main(["--task-json", str(task_path)])
    captured = capsys.readouterr()
    assert code == 0
    assert "route: local" in captured.out
    assert "reason: local_confident" in captured.out
    assert "external_call: false" in captured.out
    assert "auto_apply: false" in captured.out
    assert "weights_changed: false" in captured.out
