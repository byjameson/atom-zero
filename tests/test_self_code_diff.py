import json

import pytest

from atomzero.self_code_diff import compare_reports, main


def _finding(finding_id, kind="broad_exception", severity="medium",
             path="src/atomzero/mod.py", line=10, message="broad"):
    return {
        "finding_id": finding_id,
        "kind": kind,
        "severity": severity,
        "path": path,
        "line": line,
        "message": message,
        "evidence": "",
    }


def _report(*findings):
    return {"findings": list(findings), "auto_apply": False,
            "weights_changed": False}


def test_resolved_introduced_and_persisting():
    before = _report(
        _finding("sc-1", message="first"),
        _finding("sc-2", line=20, message="second"))
    after = _report(
        _finding("sc-2", line=22, message="second"),
        _finding("sc-3", line=30, message="third"))
    result = compare_reports(before, after)
    assert [item["finding_id"] for item in result["resolved"]] == ["sc-1"]
    assert [item["finding_id"] for item in result["introduced"]] == ["sc-3"]
    assert [item["finding_id"] for item in result["persisting"]] == ["sc-2"]
    assert result["summary"]["before"] == 2
    assert result["summary"]["after"] == 2
    assert result["summary"]["resolved"] == 1
    assert result["summary"]["introduced"] == 1
    assert result["summary"]["persisting"] == 1
    assert result["summary"]["new_medium"] == 1
    assert result["ok"] is False


def test_moved_finding_is_persisting_not_noise():
    before = _report(_finding("sc-1", line=5))
    after = _report(_finding("sc-1", line=105))
    result = compare_reports(before, after)
    assert result["summary"]["resolved"] == 0
    assert result["summary"]["introduced"] == 0
    assert result["summary"]["persisting"] == 1
    assert result["persisting"][0]["line"] == 105


def test_duplicate_findings_are_paired_by_line():
    before = _report(_finding("sc-1", line=5), _finding("sc-2", line=9))
    after = _report(_finding("sc-3", line=6))
    result = compare_reports(before, after)
    assert result["summary"]["persisting"] == 1
    assert result["summary"]["resolved"] == 1
    assert result["summary"]["introduced"] == 0
    assert result["persisting"][0]["line"] == 6


def test_new_medium_or_high_fails_the_diff():
    high = compare_reports(
        _report(), _report(_finding("sc-1", severity="high")))
    assert high["ok"] is False
    assert high["summary"]["new_high"] == 1
    medium = compare_reports(
        _report(), _report(_finding("sc-2", severity="medium")))
    assert medium["ok"] is False
    assert medium["summary"]["new_medium"] == 1


def test_new_low_only_is_ok():
    result = compare_reports(
        _report(), _report(_finding("sc-1", severity="low",
                                    kind="todo_marker")))
    assert result["ok"] is True
    assert result["summary"]["new_high"] == 0
    assert result["summary"]["new_medium"] == 0


def test_invalid_reports_raise():
    with pytest.raises(ValueError):
        compare_reports([], _report())
    with pytest.raises(ValueError):
        compare_reports(_report(), {"findings": [{"kind": "x"}]})
    with pytest.raises(ValueError):
        compare_reports(_report(), {"findings": "nope"})


def test_cli_json_and_exit_code(tmp_path, capsys):
    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    before.write_text(json.dumps(_report(_finding("sc-1"))), encoding="utf-8")
    after.write_text(json.dumps(_report()), encoding="utf-8")
    assert main([str(before), str(after), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["summary"]["resolved"] == 1
    assert payload["auto_apply"] is False

    after.write_text(json.dumps(_report(
        _finding("sc-2", severity="high"))), encoding="utf-8")
    assert main([str(before), str(after)]) == 1
