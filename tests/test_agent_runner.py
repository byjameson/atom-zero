import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from atomzero.agent_runner import RunnerError, prepare, run_packet, verify

ROOT = Path(__file__).resolve().parents[1]
PACKET = {
    "packet_version": 1,
    "role": "Implementer",
    "action_id": "imp-01-test",
    "kind": "test",
    "title": "Test action",
    "goal": "Apply an allowed test change.",
    "acceptance": "run the gate",
    "allowed_files": ["a.txt"],
    "checks": ["pytest:tests/test_dummy.py"],
    "constraints": ["stay in scope"],
    "requires_human": True,
    "auto_start": False,
    "handoff": "external_agent_harness_only",
    "weights_changed": False,
    "method": "deterministic_agent_task_packet_v1",
}


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


def test_prepare_creates_isolated_worktree(tmp_path):
    repo = _repo(tmp_path)
    wt = tmp_path / "wt"
    result = prepare(PACKET, repo_root=repo, base="master",
                     worktree_root=wt, branch="agent/test")
    assert result["status"] == "prepared"
    assert Path(result["worktree"]) == wt
    assert (wt / ".agent-task.json").is_file()
    assert (wt / "a.txt").read_text(encoding="utf-8") == "base\n"


def test_prepare_requires_opt_in_packet_fields(tmp_path):
    with pytest.raises(RunnerError):
        prepare({**PACKET, "auto_start": True}, repo_root=tmp_path)
    with pytest.raises(RunnerError):
        prepare({**PACKET, "allowed_files": ["../escape"]},
                repo_root=tmp_path)


def _commit(wt, path, message):
    subprocess.run(["git", "add", path], cwd=wt, check=True)
    subprocess.run(["git", "commit", "-qm", message], cwd=wt, check=True)


def test_verify_scope_and_gate(tmp_path):
    repo = _repo(tmp_path)
    wt = tmp_path / "wt"
    prepare(PACKET, repo_root=repo, base="master",
            worktree_root=wt, branch="agent/test")
    (wt / "a.txt").write_text("changed\n", encoding="utf-8")
    result = verify(PACKET, worktree=wt, base="master")
    assert result["ok"] is True
    assert result["changed_files"] == ["a.txt"]
    assert result["out_of_scope"] == []
    assert result["gate"]["ok"] is True


def test_verify_rejects_out_of_scope(tmp_path):
    repo = _repo(tmp_path)
    wt = tmp_path / "wt"
    prepare(PACKET, repo_root=repo, base="master",
            worktree_root=wt, branch="agent/test")
    (wt / "a.txt").write_text("changed\n", encoding="utf-8")
    (wt / "extra.txt").write_text("leak\n", encoding="utf-8")
    result = verify(PACKET, worktree=wt, base="master")
    assert result["ok"] is False
    assert result["out_of_scope"] == ["extra.txt"]


def test_verify_includes_committed_allowed_change(tmp_path):
    repo = _repo(tmp_path)
    wt = tmp_path / "wt"
    prepare(PACKET, repo_root=repo, base="master",
            worktree_root=wt, branch="agent/test")
    (wt / "a.txt").write_text("committed\n", encoding="utf-8")
    _commit(wt, "a.txt", "allowed change")
    result = verify(PACKET, worktree=wt, base="master")
    assert result["ok"] is True
    assert result["changed_files"] == ["a.txt"]
    assert result["out_of_scope"] == []


def test_verify_rejects_committed_out_of_scope(tmp_path):
    repo = _repo(tmp_path)
    wt = tmp_path / "wt"
    prepare(PACKET, repo_root=repo, base="master",
            worktree_root=wt, branch="agent/test")
    (wt / "extra.txt").write_text("leak\n", encoding="utf-8")
    _commit(wt, "extra.txt", "out of scope")
    result = verify(PACKET, worktree=wt, base="master")
    assert result["ok"] is False
    assert result["out_of_scope"] == ["extra.txt"]


def test_run_packet_requires_opt_in(tmp_path):
    with pytest.raises(RunnerError):
        run_packet(PACKET, opt_in=False, repo_root=tmp_path)


def test_run_packet_with_stub_runner(tmp_path):
    repo = _repo(tmp_path)
    wt = tmp_path / "wt"
    command = [sys.executable, "-c", "open('a.txt','w').write('stub\\n')"]
    result = run_packet(
        PACKET, opt_in=True, repo_root=repo, runner_command=command,
        base="master", worktree_root=wt, branch="agent/test")
    assert result["ok"] is True
    assert result["runner_returncode"] == 0
    assert (wt / "a.txt").read_text(encoding="utf-8") == "stub\n"


def test_cli_prepare_requires_opt_in(tmp_path):
    repo = _repo(tmp_path)
    packet = tmp_path / "packet.json"
    packet.write_text(json.dumps(PACKET), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "atomzero.agent_runner", "prepare",
         "--packet", str(packet), "--repo", str(repo), "--base", "master"],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
        env={"PYTHONPATH": str(ROOT / "src"), "PYTHONIOENCODING": "utf-8"},
        check=False)
    assert result.returncode == 2
    assert "opt-in" in result.stderr
