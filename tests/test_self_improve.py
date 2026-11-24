import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from atomzero.agent_runner import RunnerError
from atomzero.self_improve import main, run_loop


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


def _self_code_repo(tmp_path):
    repo = _repo(tmp_path)
    (repo / "src").mkdir()
    for name in ("alpha", "beta"):
        (repo / "src" / f"{name}.py").write_text(
            "def f():\n    try:\n        return 1\n"
            "    except Exception:\n        return None\n",
            encoding="utf-8")
    (repo / "tests" / "test_alpha.py").write_text(
        "def test_ok():\n    assert True\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "self-code findings"], cwd=repo,
                   check=True)
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
    action_id = result["packet"]["action_id"]
    gate_id = result["runner"]["gate"].get("gate_id", "no-gate")
    note = (f"self-improve loop: {action_id} gate {gate_id} "
            "passed; human")
    expected = hashlib.sha256(note.encode("utf-8")).hexdigest()[:16]
    assert result["decision"]["note_id"] == expected


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


def test_loop_derives_self_code_proposal_scope(tmp_path):
    args = _args(tmp_path)
    args["signals"] = {"self_code": {
        "findings": [{"finding_id": "sc-1", "severity": "medium",
                      "path": "a.txt", "line": 1}],
        "proposals": [{"severity": "medium", "allowed_files": ["a.txt"],
                       "checks": ["pytest:tests/test_dummy.py"]}],
    }}
    args["allowed_files"] = []
    args["checks"] = []
    result = run_loop(**args)
    assert result["status"] == "completed", result
    assert result["packet"]["allowed_files"] == ["a.txt"]
    assert result["runner"]["changed_files"] == ["a.txt"]


def test_loop_self_code_root_builds_packet_without_approval(tmp_path):
    repo = _repo(tmp_path)
    (repo / "src").mkdir()
    (repo / "src" / "mod.py").write_text(
        "def f():\n    try:\n        return 1\n"
        "    except Exception:\n        return None\n", encoding="utf-8")
    result = run_loop(
        signals={}, allowed_files=[], checks=[], opt_in=True,
        approve=False, repo_root=repo, self_code_root=repo)
    assert result["status"] == "awaiting_human"
    assert result["plan"]["actions"][0]["kind"] == "self_code_medium"
    assert result["packet"]["allowed_files"] == [
        "src/mod.py", "tests/test_mod.py"]
    assert result["packet"]["checks"] == ["pytest:tests/test_mod.py"]
    assert not (tmp_path / "agent-wt").exists()


def test_loop_no_action_when_self_code_is_clean(tmp_path):
    result = run_loop(
        signals={"self_code": {"findings": []}}, allowed_files=[],
        checks=[], opt_in=True, approve=True, repo_root=tmp_path)
    assert result["status"] == "no_action"
    assert not (tmp_path / "ledger.jsonl").exists()


def test_loop_self_code_diff_records_resolved_finding(tmp_path):
    repo = _self_code_repo(tmp_path)
    result = run_loop(
        signals={}, allowed_files=[], checks=[], opt_in=True, approve=True,
        repo_root=repo, base="master",
        runner_command=[sys.executable, "-c",
                        "open('src/alpha.py','w').write('def f():\\n"
                        "    return 1\\n')"],
        worktree_root=tmp_path / "agent-wt",
        ledger_path=tmp_path / "ledger.jsonl",
        self_code_root=repo, target_path="src/alpha.py")
    assert result["status"] == "completed", result
    diff = result["self_code_diff"]
    assert diff["ok"] is True
    assert diff["summary"]["resolved"] == 1
    assert diff["summary"]["persisting"] == 1
    assert diff["summary"]["introduced"] == 0
    assert (tmp_path / "ledger.jsonl").is_file()


def test_loop_audits_with_high_per_kind_cap(tmp_path, monkeypatch):
    import atomzero.self_code as self_code
    calls = []
    real = self_code.audit_report

    def spy(*args, **kwargs):
        calls.append(dict(kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(self_code, "audit_report", spy)
    repo = _self_code_repo(tmp_path)
    result = run_loop(
        signals={}, allowed_files=[], checks=[], opt_in=True, approve=False,
        repo_root=repo, self_code_root=repo, target_path="src/alpha.py")
    assert result["status"] == "awaiting_human"
    assert calls and calls[0]["max_per_kind"] == 200


def test_loop_self_code_regression_blocks_decision(tmp_path):
    repo = _self_code_repo(tmp_path)
    added = """def f():
    try:
        return 1
    except Exception:
        return None
def g():
    try:
        return 2
    except Exception:
        return None
"""
    result = run_loop(
        signals={}, allowed_files=[], checks=[], opt_in=True, approve=True,
        repo_root=repo, base="master",
        runner_command=[sys.executable, "-c",
                        f"open('src/alpha.py','w').write({added!r})"],
        worktree_root=tmp_path / "agent-wt",
        ledger_path=tmp_path / "ledger.jsonl",
        self_code_root=repo, target_path="src/alpha.py")
    assert result["status"] == "self_code_regression", result
    assert result["self_code_diff"]["ok"] is False
    assert result["self_code_diff"]["summary"]["new_medium"] == 1
    assert "decision" not in result
    assert not (tmp_path / "ledger.jsonl").exists()


def test_loop_uses_ledger_history_for_priorities(tmp_path):
    report = {"self_code": {
        "findings": [{"finding_id": "sc-1", "severity": "medium",
                      "path": "a.txt", "line": 1}],
        "proposals": [{"severity": "medium", "allowed_files": ["a.txt"],
                       "checks": ["pytest:tests/test_dummy.py"]}],
    }}
    signals = {**report, "claim_audit": {"conflict_count": 1}}
    base = dict(
        signals=signals, allowed_files=["a.txt"],
        checks=["pytest:tests/test_dummy.py"], opt_in=True, approve=False,
        repo_root=tmp_path, ledger_path=tmp_path / "ledger.jsonl")
    before = run_loop(**base)
    assert before["plan"]["actions"][0]["kind"] == "conflicts"

    (tmp_path / "ledger.jsonl").write_text(
        "\n".join(json.dumps({"kind": "self_code_medium",
                              "decision": "approve"})
                  for _ in range(2)) + "\n", encoding="utf-8")
    after = run_loop(**base)
    assert after["plan"]["actions"][0]["kind"] == "self_code_medium"
    assert after["plan"]["actions"][0]["priority"] > 75


def test_loop_explicit_history_is_not_overwritten(tmp_path):
    ledger_path = tmp_path / "ledger.jsonl"
    ledger_path.write_text(
        "\n".join(json.dumps({"kind": "self_code_medium",
                              "decision": "approve"})
                  for _ in range(2)) + "\n", encoding="utf-8")
    signals = {
        "self_code": {
            "findings": [{"finding_id": "sc-1", "severity": "medium",
                          "path": "a.txt", "line": 1}],
            "proposals": [{"severity": "medium", "allowed_files": ["a.txt"],
                           "checks": ["pytest:tests/test_dummy.py"]}],
        },
        "claim_audit": {"conflict_count": 1},
        "history": [],
    }
    result = run_loop(
        signals=signals, allowed_files=["a.txt"],
        checks=["pytest:tests/test_dummy.py"], opt_in=True, approve=False,
        repo_root=tmp_path, ledger_path=ledger_path)
    assert result["plan"]["actions"][0]["kind"] == "conflicts"


def test_loop_runs_adapter_candidate(tmp_path):
    from atomzero.adapter_candidates import build_candidate
    examples = [
        {"user": f"q{index}", "assistant": f"a{index}",
         "verification": {"verified": True, "promotable": True}}
        for index in range(4)
    ]
    # The deterministic split puts one example in the held-out set.
    candidate = build_candidate(examples, eval_ratio=0.25)
    comparison = {
        "examples": candidate["metadata"]["eval_examples"],
        "baseline": {"loss": 1.2, "ppl": 3.3},
        "adapter": {"loss": 1.0, "ppl": 3.0},
    }
    ledger = tmp_path / "adapter.jsonl"
    result = run_loop(
        signals={"adapter": {"promotable": 4, "candidates": 0}},
        allowed_files=[], checks=[], opt_in=True, approve=False,
        autonomous=True, repo_root=tmp_path,
        adapter_examples=examples,
        adapter_out_dir=tmp_path / "candidate",
        adapter_comparison=comparison,
        adapter_ledger_path=ledger,
        adapter_eval_ratio=0.25)
    assert result["status"] == "completed", result
    assert result["adapter"]["status"] == "pass"
    assert result["adapter"]["evaluation"]["promotable"] is True
    assert ledger.is_file()


def test_loop_adapter_gate_failure(tmp_path):
    examples = [
        {"user": f"q{index}", "assistant": f"a{index}",
         "verification": {"verified": True, "promotable": True}}
        for index in range(4)
    ]
    result = run_loop(
        signals={"adapter": {"promotable": 4, "candidates": 0}},
        allowed_files=[], checks=[], opt_in=True, approve=False,
        autonomous=True, repo_root=tmp_path,
        adapter_examples=examples,
        adapter_out_dir=tmp_path / "candidate",
        adapter_comparison={
            "examples": 1,
            "baseline": {"loss": 1.2, "ppl": 3.3},
            "adapter": {"loss": 1.3, "ppl": 3.5}},
        adapter_eval_ratio=0.25)
    assert result["status"] == "adapter_gate_failed"
    assert result["adapter"]["evaluation"]["ok"] is False


def test_loop_runs_adapter_trainer_command(tmp_path):
    script = tmp_path / "trainer.py"
    script.write_text(
        "import json, pathlib, sys\n"
        "comparison = pathlib.Path(sys.argv[1])\n"
        "adapter = pathlib.Path(sys.argv[2])\n"
        "adapter.mkdir(parents=True, exist_ok=True)\n"
        "(adapter / 'adapter.pt').write_bytes(b'fake')\n"
        "comparison.write_text(json.dumps({'examples': 1, "
        "'baseline': {'loss': 1.2, 'ppl': 3.3}, "
        "'adapter': {'loss': 1.0, 'ppl': 3.0}}))\n",
        encoding="utf-8")
    examples = [
        {"user": f"q{index}", "assistant": f"a{index}",
         "verification": {"verified": True, "promotable": True}}
        for index in range(4)
    ]
    result = run_loop(
        signals={"adapter": {"promotable": 4, "candidates": 0}},
        allowed_files=[], checks=[], opt_in=True, approve=False,
        autonomous=True, repo_root=tmp_path,
        adapter_examples=examples,
        adapter_out_dir=tmp_path / "candidate",
        adapter_base="base-fixture",
        adapter_trainer_command=[sys.executable, str(script),
                                 "{comparison_path}", "{adapter_dir}"],
        adapter_eval_ratio=0.25)
    assert result["status"] == "completed", result
    assert result["adapter"]["source"] == "trainer"
    assert result["adapter"]["trainer"]["ok"] is True
    assert result["adapter"]["evaluation"]["promotable"] is True
    assert (tmp_path / "candidate" / "adapter" / "adapter.pt").is_file()


def test_loop_runs_adapter_explicit_splits(tmp_path):
    train = [
        {"user": f"train {index}", "assistant": f"a{index}",
         "verification": {"verified": True, "promotable": True}}
        for index in range(4)
    ]
    eval_rows = [
        {"user": f"cross {index}", "assistant": f"b{index}",
         "verification": {"verified": True, "promotable": True}}
        for index in range(2)
    ]
    result = run_loop(
        signals={"adapter": {"promotable": 2, "candidates": 0}},
        allowed_files=[], checks=[], opt_in=True, approve=False,
        autonomous=True, repo_root=tmp_path,
        adapter_train_examples=train, adapter_eval_examples=eval_rows,
        adapter_out_dir=tmp_path / "candidate",
        adapter_comparison={
            "examples": 2,
            "baseline": {"loss": 1.2, "ppl": 3.3},
            "adapter": {"loss": 1.0, "ppl": 3.0}})
    assert result["status"] == "completed", result
    assert result["adapter"]["candidate"]["split_method"] == (
        "explicit_cross_distribution")
    assert result["adapter"]["candidate"]["eval_examples"] == 2


def test_loop_adapter_requires_inputs(tmp_path):
    result = run_loop(
        signals={"adapter": {"promotable": 1, "candidates": 0}},
        allowed_files=[], checks=[], opt_in=True, approve=False,
        autonomous=True, repo_root=tmp_path)
    assert result["status"] == "adapter_input_missing"


def test_loop_without_self_code_root_has_no_diff(tmp_path):
    result = run_loop(**_args(tmp_path))
    assert result["status"] == "completed"
    assert "self_code_diff" not in result


def _autonomous_runner():
    return [sys.executable, "-c",
            "import subprocess; open('a.txt','w').write('auto'); "
            "subprocess.run(['git','add','a.txt'], check=True); "
            "subprocess.run(['git','commit','-qm','autonomous change'], "
            "check=True)"]


def test_autonomous_loop_verifies_and_merges(tmp_path):
    args = _args(tmp_path)
    args.update(
        approve=False, autonomous=True, auto_merge=True,
        runner_command=_autonomous_runner(),
        test_command=[sys.executable, "-c",
                      "print('1 passed in 0.01s')"])
    result = run_loop(**args)
    assert result["status"] == "completed", result
    assert result["autonomous"] is True
    assert result["verification"]["ok"] is True
    assert result["verification"]["tests_passed"] == 1
    assert result["merge"]["ok"] is True
    assert result["merge"]["after"]
    assert result["decision"]["decision"] == "approve"
    assert result["sandbox"]["os_vm"] is False
    assert (args["repo_root"] / "a.txt").read_text(encoding="utf-8") == "auto"


def test_autonomous_failed_verification_does_not_merge(tmp_path):
    args = _args(tmp_path)
    args.update(
        approve=False, autonomous=True, auto_merge=True,
        runner_command=_autonomous_runner(),
        test_command=[sys.executable, "-c",
                      "import sys; print('1 failed, 0 passed in 0.01s'); "
                      "sys.exit(1)"])
    result = run_loop(**args)
    assert result["status"] == "verification_failed"
    assert "merge" not in result
    assert (args["repo_root"] / "a.txt").read_text(encoding="utf-8") == "base\n"
    assert not (tmp_path / "ledger.jsonl").exists()


def test_autonomous_without_opt_in_does_nothing(tmp_path):
    args = _args(tmp_path)
    args.update(opt_in=False, approve=False, autonomous=True)
    result = run_loop(**args)
    assert result["status"] == "awaiting_opt_in"
    assert not (tmp_path / "ledger.jsonl").exists()


def test_loop_target_path_selects_matching_proposal(tmp_path):
    repo = _self_code_repo(tmp_path)
    result = run_loop(
        signals={}, allowed_files=[], checks=[], opt_in=True,
        approve=False, repo_root=repo, self_code_root=repo,
        target_path="./src\\alpha.py")
    assert result["status"] == "awaiting_human"
    assert result["plan"]["actions"][0]["kind"] == "self_code_medium"
    assert result["packet"]["allowed_files"] == [
        "src/alpha.py", "tests/test_alpha.py"]
    assert result["packet"]["checks"] == ["pytest:tests/test_alpha.py"]
    assert result["auto_apply"] is False
    assert result["weights_changed"] is False


def test_loop_target_path_uses_matching_proposal_from_signals(tmp_path):
    args = _args(tmp_path)
    args.update(
        signals={"self_code": {
            "findings": [
                {"finding_id": "sc-a", "kind": "broad_exception",
                 "severity": "medium", "path": "a.txt", "line": 1},
                {"finding_id": "sc-b", "kind": "broad_exception",
                 "severity": "medium", "path": "b.txt", "line": 1}],
            "proposals": [
                {"finding_id": "sc-a", "severity": "medium",
                 "path": "a.txt", "allowed_files": ["a.txt"],
                 "checks": ["pytest:tests/test_dummy.py"]},
                {"finding_id": "sc-b", "severity": "medium",
                 "path": "b.txt", "allowed_files": ["b.txt"],
                 "checks": ["pytest:tests/test_dummy.py"]}],
        }},
        allowed_files=[], checks=[], approve=False)
    result = run_loop(**args, target_path="b.txt")
    assert result["status"] == "awaiting_human"
    assert result["packet"]["allowed_files"] == ["b.txt"]
    assert result["packet"]["checks"] == ["pytest:tests/test_dummy.py"]


def test_loop_target_path_unknown_raises(tmp_path):
    repo = _self_code_repo(tmp_path)
    with pytest.raises(
            RunnerError, match="no self-code finding or proposal matches"):
        run_loop(
            signals={}, allowed_files=[], checks=[], opt_in=False,
            approve=False, repo_root=repo, self_code_root=repo,
            target_path="src/missing.py")


def test_loop_target_path_high_finding_builds_packet(tmp_path):
    repo = _repo(tmp_path)
    (repo / "src").mkdir()
    (repo / "src" / "danger.py").write_text(
        "def f(expression):\n    return eval(expression)\n",
        encoding="utf-8")
    result = run_loop(
        signals={}, allowed_files=[], checks=[], opt_in=True,
        approve=False, repo_root=repo, self_code_root=repo,
        target_path="src/danger.py")
    assert result["plan"]["actions"][0]["kind"] == "self_code_high"
    assert result["packet"]["allowed_files"] == ["src/danger.py"]
    assert result["packet"]["checks"] == [
        "pytest:tests/test_self_code.py",
        "pytest:tests/test_improvement.py"]


def test_main_sandbox_info(capsys):
    code = main(["--sandbox-info"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["backend"] == "worktree"
    assert payload["os_vm"] is False


def test_main_passes_target_path(tmp_path, capsys):
    repo = _self_code_repo(tmp_path)
    code = main([
        "--self-code-root", str(repo), "--target-path", "src/beta.py",
        "--opt-in", "--require-approval", "--json"])
    assert code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "awaiting_human"
    assert payload["packet"]["allowed_files"] == [
        "src/beta.py", "tests/test_beta.py"]
