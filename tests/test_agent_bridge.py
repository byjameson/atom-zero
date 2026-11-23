import json
import subprocess
import sys
import tempfile
from pathlib import Path

from atomzero.agent_bridge import build_task_packet, render_markdown

ROOT = Path(__file__).resolve().parents[1]
ACTION = {
    "action_id": "imp-01-conflicts",
    "kind": "conflicts",
    "title": "Claim conflicts",
    "detail": "Resolve conflicting claims.",
    "acceptance": "resolve conflicting claims and rerun claim audit",
    "requires_human": True,
}


def run_cli(*args: str):
    return subprocess.run(
        [sys.executable, "-m", "atomzero.agent_bridge", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={"PYTHONPATH": str(ROOT / "src"), "PYTHONIOENCODING": "utf-8"},
        check=False,
    )


def test_packet_is_data_only_and_human_gated():
    packet = build_task_packet(
        ACTION,
        allowed_files=["src/atomzero/claims.py", "tests/test_claims.py"],
        checks=["pytest:tests/test_claims.py"],
        repository="C:/work/atom-zero",
    )
    assert packet["requires_human"] is True
    assert packet["auto_start"] is False
    assert packet["handoff"] == "external_agent_harness_only"
    assert packet["weights_changed"] is False
    assert packet["allowed_files"] == [
        "src/atomzero/claims.py", "tests/test_claims.py"]
    assert packet["checks"] == ["pytest:tests/test_claims.py"]
    assert packet["acceptance"]


def test_packet_rejects_unsafe_inputs():
    try:
        build_task_packet({**ACTION, "requires_human": False},
                          allowed_files=["a.py"], checks=["pytest:a.py"])
    except ValueError:
        pass
    else:
        raise AssertionError("human gate must be required")
    try:
        build_task_packet(ACTION, allowed_files=[], checks=["pytest:a.py"])
    except ValueError:
        pass
    else:
        raise AssertionError("allowed_files must not be empty")
    try:
        build_task_packet(ACTION, allowed_files=["a.py"], checks=["shell:dir"])
    except ValueError:
        pass
    else:
        raise AssertionError("shell check must be rejected")


def test_packet_custom_constraints_and_markdown():
    packet = build_task_packet(
        ACTION, allowed_files=["a.py"], checks=["manual:"],
        constraints=["do not touch data"])
    assert packet["constraints"] == ["do not touch data"]
    text = render_markdown(packet)
    assert "# Agent task packet" in text
    assert "src/atomzero" not in text
    assert "external harness" in text


def test_cli_json_and_action_selection(tmp_path):
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"actions": [ACTION]}), encoding="utf-8")
    result = run_cli(
        "--plan", str(plan), "--action-id", ACTION["action_id"],
        "--allowed-file", "src/atomzero/claims.py",
        "--check", "pytest:tests/test_claims.py", "--json")
    assert result.returncode == 0, result.stderr
    packet = json.loads(result.stdout)
    assert packet["action_id"] == ACTION["action_id"]
    assert packet["checks"] == ["pytest:tests/test_claims.py"]


def test_cli_markdown(tmp_path):
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"actions": [ACTION]}), encoding="utf-8")
    result = run_cli(
        "--plan", str(plan), "--allowed-file", "a.py",
        "--check", "manual:")
    assert result.returncode == 0, result.stderr
    assert "# Agent task packet" in result.stdout


def test_cli_invalid_exit_two(tmp_path):
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"actions": []}), encoding="utf-8")
    result = run_cli("--plan", str(plan), "--allowed-file", "a.py",
                     "--check", "manual:")
    assert result.returncode == 2
    assert "input error" in result.stderr
