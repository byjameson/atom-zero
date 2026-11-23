import io
import json
import subprocess
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from atomzero import evaluation


def test_parse_check():
    assert evaluation.parse_check("pytest:tests/test_claims.py") == {
        "name": "tests/test_claims.py", "kind": "pytest",
        "target": "tests/test_claims.py"}


def test_parse_check_rejects_bad_values():
    with pytest.raises(ValueError):
        evaluation.parse_check("pytest")
    with pytest.raises(ValueError):
        evaluation.parse_check("shell:dir")
    with pytest.raises(ValueError):
        evaluation.parse_check("pytest:")


def test_run_check_pytest_command(monkeypatch):
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stdout="12 passed", stderr="")

    monkeypatch.setattr(evaluation.subprocess, "run", fake_run)
    result = evaluation.run_check(
        {"name": "claims", "kind": "pytest", "target": "tests/test_claims.py"})
    assert result["ok"] is True
    assert result["status"] == "pass"
    assert captured["command"][:3] == [
        evaluation.sys.executable, "-m", "pytest"]
    assert "tests/test_claims.py" in captured["command"]


def test_run_check_aisc_uses_cwd(monkeypatch, tmp_path):
    captured = {}

    def fake_run(command, **kwargs):
        captured["cwd"] = kwargs["cwd"]
        captured["command"] = command
        return SimpleNamespace(returncode=1, stdout="", stderr="FAIL")

    monkeypatch.setattr(evaluation.subprocess, "run", fake_run)
    result = evaluation.run_check(
        {"name": "spec", "kind": "aisc",
         "target": "docs/aisc/atomzero-claims.toml"},
        aisc_path=str(tmp_path))
    assert result["ok"] is False
    assert captured["cwd"] == str(tmp_path)
    assert "aisc" in captured["command"]


def test_manual_check_is_not_auto_passed():
    result = evaluation.run_check(
        {"name": "review", "kind": "manual", "target": ""})
    assert result["ok"] is False
    assert result["status"] == "manual_required"


def test_evaluate_gate_all_pass(monkeypatch):
    monkeypatch.setattr(
        evaluation.subprocess, "run",
        lambda *a, **k: SimpleNamespace(returncode=0, stdout="ok", stderr=""))
    report = evaluation.evaluate_gate([
        {"name": "a", "kind": "pytest", "target": "tests/test_a.py"},
        {"name": "b", "kind": "pytest", "target": "tests/test_b.py"},
    ])
    assert report["ok"] is True
    assert report["summary"]["passed"] == 2
    assert report["auto_apply"] is False
    assert report["weights_changed"] is False


def test_evaluate_gate_one_failure(monkeypatch):
    responses = iter([
        SimpleNamespace(returncode=0, stdout="ok", stderr=""),
        SimpleNamespace(returncode=1, stdout="1 failed", stderr=""),
    ])
    monkeypatch.setattr(
        evaluation.subprocess, "run", lambda *a, **k: next(responses))
    report = evaluation.evaluate_gate([
        {"name": "a", "kind": "pytest", "target": "tests/test_a.py"},
        {"name": "b", "kind": "pytest", "target": "tests/test_b.py"},
    ])
    assert report["ok"] is False
    assert report["summary"]["failed"] == 1
    assert "1 failed" in report["checks"][1]["output"]


def test_evaluate_gate_timeout(monkeypatch):
    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired("pytest", 1)

    monkeypatch.setattr(evaluation.subprocess, "run", fake_run)
    report = evaluation.evaluate_gate(
        [{"name": "slow", "kind": "pytest", "target": "tests/test_slow.py"}],
        timeout=1)
    assert report["ok"] is False
    assert report["checks"][0]["status"] == "timeout"


def test_evaluate_gate_rejects_empty():
    with pytest.raises(ValueError):
        evaluation.evaluate_gate([])


def test_cli_json(monkeypatch):
    monkeypatch.setattr(
        evaluation.subprocess, "run",
        lambda *a, **k: SimpleNamespace(returncode=0, stdout="ok", stderr=""))
    output = io.StringIO()
    with redirect_stdout(output):
        code = evaluation.main([
            "--check", "pytest:tests/test_claims.py", "--json"])
    assert code == 0
    data = json.loads(output.getvalue())
    assert data["ok"] is True


def test_cli_invalid_exit_two():
    assert evaluation.main(["--check", "bogus:x"]) == 2
