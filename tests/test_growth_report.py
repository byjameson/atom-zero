import json

import pytest

from atomzero.growth_report import build_report, main
from atomzero.scoreboard import record_run


def _event(route="teacher", verified=True):
    return {
        "route": route,
        "reason_code": "low_confidence_or_freshness",
        "outcome": "success",
        "verified": verified,
        "growth_event": "verified_teacher" if verified else "",
        "at": "2026-11-24T12:00:00Z",
        "version": 1,
        "method": "deterministic_escalation_growth_v1",
    }


def _run(commit="aaaaaaa", passed=985, failed=0):
    runs = []
    record_run(
        runs, commit=commit, tests_passed=passed, tests_failed=failed,
        benchmark_passed=17, benchmark_total=17,
        aisc_passed=22, aisc_total=24, now="2026-11-24T12:00:00Z")
    return runs[0]


def test_report_combines_sources():
    report = build_report(
        escalation_ledger=[_event()],
        scoreboard=[_run()],
        self_code_diff={"ok": True, "summary": {
            "resolved": 2, "introduced": 0, "persisting": 1,
            "new_high": 0, "new_medium": 0}})
    assert report["ok"] is True
    assert report["reasons"] == []
    assert report["capability"]["verified_teacher_answers"] == 1
    assert report["capability"]["adapter_candidates"] == 1
    assert report["capability"]["tests_passed"] == 985
    assert report["capability"]["code_findings_resolved"] == 2
    assert report["self_code"]["resolved"] == 2
    assert report["parameter_growth"] == 0
    assert report["parameter_growth_state"] == "unchanged"


def test_no_scoreboard_blocks():
    report = build_report(escalation_ledger=[])
    assert report["ok"] is False
    assert "no_scoreboard" in report["reasons"]
    assert report["scoreboard"]["latest"] is None


def test_failed_tests_block():
    report = build_report(scoreboard=[_run(failed=2)])
    assert report["ok"] is False
    assert "tests_failed" in report["reasons"]


def test_self_code_regression_blocks():
    report = build_report(
        scoreboard=[_run()],
        self_code_diff={"ok": False, "summary": {
            "resolved": 0, "introduced": 1, "persisting": 0,
            "new_high": 0, "new_medium": 1}})
    assert report["ok"] is False
    assert "self_code_regression" in report["reasons"]
    assert report["self_code"]["new_medium"] == 1


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        build_report(escalation_ledger={})
    with pytest.raises(ValueError):
        build_report(scoreboard={})
    with pytest.raises(ValueError):
        build_report(self_code_diff=[])


def test_cli_json_and_exit(tmp_path, capsys):
    scoreboard = tmp_path / "scoreboard.jsonl"
    scoreboard.write_text(json.dumps(_run()) + "\n", encoding="utf-8")
    assert main(["--scoreboard-jsonl", str(scoreboard), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["capability"]["tests_passed"] == 985
    assert payload["weights_changed"] is False

    scoreboard.write_text(json.dumps(_run(failed=1)) + "\n", encoding="utf-8")
    assert main(["--scoreboard-jsonl", str(scoreboard)]) == 1


def test_cli_rejects_two_stdin_sources():
    assert main(["--escalation-jsonl", "-", "--scoreboard-jsonl", "-"]) == 2
