import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from atomzero.scoreboard import (
    load_jsonl,
    record_run,
    summarize_runs,
    verify_runs,
    write_jsonl,
)

NOW = datetime(2026, 11, 23, 12, 0, 0, tzinfo=timezone.utc)
ROOT = Path(__file__).resolve().parents[1]


def run_cli(*args: str):
    return subprocess.run(
        [sys.executable, "-m", "atomzero.scoreboard", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={"PYTHONPATH": str(ROOT / "src"), "PYTHONIOENCODING": "utf-8"},
        check=False,
    )


def test_record_run_is_content_free():
    runs = []
    record = record_run(
        runs, commit="514632e", tests_passed=755, tests_failed=0,
        benchmark_passed=10, benchmark_total=10, aisc_passed=14,
        aisc_total=16, note="private release note", now=NOW)
    assert record["at"] == "2026-11-23T12:00:00Z"
    assert record["note_id"] and len(record["note_id"]) == 16
    assert "private release note" not in json.dumps(record)
    assert verify_runs(runs)["ok"] is True


def test_record_rejects_bad_values():
    with pytest.raises(ValueError):
        record_run([], commit="XYZ", tests_passed=1)
    with pytest.raises(ValueError):
        record_run([], commit="514632e", tests_passed=1,
                   benchmark_passed=2, benchmark_total=1)
    with pytest.raises(ValueError):
        record_run([], commit="514632e", tests_passed=1,
                   aisc_passed=2, aisc_total=1)
    with pytest.raises(ValueError):
        record_run([], commit="514632e", tests_passed=-1)


def test_summary_latest_and_delta():
    runs = []
    record_run(runs, commit="aaaaaaa", tests_passed=100, tests_failed=1,
               benchmark_passed=8, benchmark_total=10, aisc_passed=10,
               aisc_total=16, now="2026-11-20T00:00:00Z")
    record_run(runs, commit="bbbbbbb", tests_passed=120, tests_failed=0,
               benchmark_passed=10, benchmark_total=10, aisc_passed=12,
               aisc_total=16, now="2026-11-21T00:00:00Z")
    summary = summarize_runs(runs)
    assert summary["runs"] == 2
    assert summary["latest"]["commit"] == "bbbbbbb"
    assert summary["previous"]["commit"] == "aaaaaaa"
    assert summary["delta"] == {
        "tests_passed": 20, "tests_failed": -1,
        "benchmark_passed": 2, "aisc_passed": 2,
    }
    assert summary["green"] is True
    assert summary["weights_changed"] is False


def test_empty_summary():
    summary = summarize_runs([])
    assert summary["runs"] == 0
    assert summary["latest"] is None
    assert summary["green"] is False


def test_verify_rejects_content_keys():
    runs = []
    record_run(runs, commit="514632e", tests_passed=1, now=NOW)
    runs[0]["output"] = "private log"
    result = verify_runs(runs)
    assert result["ok"] is False
    assert "output" in result["errors"][0]


def test_verify_rejects_bad_counts():
    runs = []
    record_run(runs, commit="514632e", tests_passed=1, now=NOW)
    runs[0]["tests_passed"] = -1
    assert verify_runs(runs)["ok"] is False


def test_jsonl_round_trip(tmp_path):
    runs = []
    record_run(runs, commit="514632e", tests_passed=1, now=NOW)
    path = tmp_path / "scoreboard.jsonl"
    write_jsonl(path, runs)
    assert load_jsonl(path) == runs
    assert path.read_text(encoding="utf-8").endswith("\n")


def test_write_rejects_invalid(tmp_path):
    with pytest.raises(ValueError):
        write_jsonl(tmp_path / "bad.jsonl", [{"version": 1, "text": "leak"}])


def test_load_missing_file(tmp_path):
    assert load_jsonl(tmp_path / "missing.jsonl") == []


def test_cli_record_summary_and_verify(tmp_path):
    path = tmp_path / "scoreboard.jsonl"
    record = run_cli(
        "--file", str(path), "--record", "--commit", "514632e",
        "--tests-passed", "755", "--benchmark-passed", "10",
        "--benchmark-total", "10", "--aisc-passed", "14",
        "--aisc-total", "16", "--json")
    assert record.returncode == 0, record.stderr
    data = json.loads(record.stdout)
    assert data["latest"]["tests_passed"] == 755
    verify = run_cli("--file", str(path), "--verify", "--json")
    assert verify.returncode == 0, verify.stderr
    assert json.loads(verify.stdout)["ok"] is True


def test_cli_bad_record_exit_two(tmp_path):
    path = tmp_path / "scoreboard.jsonl"
    result = run_cli(
        "--file", str(path), "--record", "--commit", "bad")
    assert result.returncode == 2
    assert "input error" in result.stderr
