import json
import subprocess
import sys
from pathlib import Path

import pytest

from atomzero.self_improve import run_loop


def _repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    (repo / "a.txt").write_text("base\n", encoding="utf-8")
    (repo / "tests").mkdir()
    (repo / "tests" / "test_dummy.py").write_text(
        "def test_ok():\n    assert True\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=repo, check=True)
    return repo


def _args(tmp_path):
    return dict(
        signals={"runner_audit": {"alerts": ["committed_out_of_scope"]}},
        allowed_files=["a.txt"],
        checks=["pytest:tests/test_dummy.py"],
        opt_in=True,
        approve=True,
        repo_root=_repo(tmp_path),
        runner_command=[sys.executable, "-c",
                        "open('a.txt','w').write('improved\\n')"],
        base="master",
        worktree_root=tmp_path / "agent-wt",
        ledger_path=tmp_path / "ledger.jsonl",
        scoreboard_path=tmp_path / "scoreboard.jsonl",
        after={"commit": "aaaaaaa", "tests_passed": 10,
               "benchmark_passed": 1, "benchmark_total": 1,
               "aisc_passed": 1, "aisc_total": 1},
    )


def test_loop_requires_approval_and_veto(tmp_path):
    args = _args(tmp_path)
    awaiting = run_loop(**{**args, "approve": False})
    assert awaiting["status"] == "awaiting_human"
    vetoed = run_loop(**{**args, "veto": True})
    assert vetoed["status"] == "vetoed"
    assert not (tmp_path / "agent-wt").exists()


def test_loop_completes_and_records(tmp_path):
    result = run_loop(**_args(tmp_path))
    assert result["status"] == "completed", result
    assert result["runner"]["ok"] is True
    assert result["decision"]["decision"] == "approve"
    assert result["decision"]["gate_ok"] is True
    assert (tmp_path / "ledger.jsonl").is_file()
    assert (tmp_path / "scoreboard.jsonl").is_file()
    assert result["trend"]["ok"] is True
    assert result["runner"]["changed_files"] == ["a.txt"]


def test_loop_veto_does_nothing(tmp_path):
    args = _args(tmp_path)
    result = run_loop(**{**args, "veto": True})
    assert result["status"] == "vetoed"
    assert not (tmp_path / "ledger.jsonl").exists()


def test_loop_gate_failure_does_not_record(tmp_path):
    args = _args(tmp_path)
    args["checks"] = ["pytest:tests/test_missing.py"]
    result = run_loop(**args)
    assert result["status"] == "gate_failed"
    assert not (tmp_path / "ledger.jsonl").exists()
