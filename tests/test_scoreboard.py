import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from atomzero.scoreboard import (
    load_jsonl,
    parse_aisc_summary,
    parse_benchmark_summary,
    parse_pytest_summary,
    record_run,
    summarize_runs,
    verify_runs,
    write_jsonl,
)

NOW = datetime(2026, 11, 23, 12, 0, 0, tzinfo=timezone.utc)
ROOT = Path(__file__).resolve().parents[1]


def run_cli(*args: str, cwd=None, stdin_text=None):
    return subprocess.run(
        [sys.executable, "-m", "atomzero.scoreboard", *args],
        cwd=cwd or ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        input=stdin_text,
        env={
            **os.environ,
            "PYTHONPATH": str(ROOT / "src"),
            "PYTHONIOENCODING": "utf-8",
        },
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


def test_parse_pytest_summary_variants():
    assert parse_pytest_summary("857 passed in 81.49s") == {
        "tests_passed": 857, "tests_failed": 0}
    assert parse_pytest_summary("2 failed, 855 passed in 12.3s") == {
        "tests_passed": 855, "tests_failed": 2}
    assert parse_pytest_summary(
        "1 passed, 2 skipped, 1 warning in 0.5s") == {
        "tests_passed": 1, "tests_failed": 0}
    assert parse_pytest_summary("1 passed, 2 errors in 0.4s") == {
        "tests_passed": 1, "tests_failed": 2}


def test_parse_pytest_summary_uses_last_summary_line():
    text = "2 failed, 1 passed in 1.0s\nnoise\n857 passed in 81.49s\n"
    assert parse_pytest_summary(text) == {
        "tests_passed": 857, "tests_failed": 0}


@pytest.mark.parametrize("text", [
    "", "no tests ran in 0.01s", "collected 0 items\nno tests ran in 0.01s",
])
def test_parse_pytest_summary_malformed(text):
    with pytest.raises(ValueError):
        parse_pytest_summary(text)


def test_parse_benchmark_summary_variants():
    assert parse_benchmark_summary(
        '{"passed": 13, "total": 13, "ok": true}') == {
        "benchmark_passed": 13, "benchmark_total": 13}
    assert parse_benchmark_summary("summary: 13/13 checks passed") == {
        "benchmark_passed": 13, "benchmark_total": 13}


@pytest.mark.parametrize("text", [
    "", "not a report", '{"passed": 1}', '{"passed": 2, "total": 1}',
    '{"passed": true, "total": 1}',
])
def test_parse_benchmark_summary_malformed(text):
    with pytest.raises(ValueError):
        parse_benchmark_summary(text)


def test_parse_aisc_summary_variants():
    assert parse_aisc_summary(
        "summary: 22 specs, 20 pass, 2 fail, 0 error") == {
        "aisc_passed": 20, "aisc_total": 22}
    assert parse_aisc_summary("22 specs, 20 PASS, 2 FAIL") == {
        "aisc_passed": 20, "aisc_total": 22}


@pytest.mark.parametrize("text", [
    "", "summary: nothing", "22 specs, 20 pass",
    "22 specs, 23 pass, 0 fail, 0 error",
])
def test_parse_aisc_summary_malformed(text):
    with pytest.raises(ValueError):
        parse_aisc_summary(text)


def write_source_files(tmp_path):
    pytest_out = tmp_path / "pytest.txt"
    pytest_out.write_text(
        "PRIVATE TEST OUTPUT\n2 failed, 855 passed in 12.3s\n",
        encoding="utf-8")
    benchmark_out = tmp_path / "benchmark.json"
    benchmark_out.write_text(
        json.dumps({"passed": 13, "total": 13, "ok": True}),
        encoding="utf-8")
    aisc_out = tmp_path / "aisc.txt"
    aisc_out.write_text(
        "PRIVATE AISC OUTPUT\n"
        "summary: 22 specs, 20 pass, 2 fail, 0 error\n",
        encoding="utf-8")
    return pytest_out, benchmark_out, aisc_out


def test_cli_record_from_source_files(tmp_path):
    pytest_out, benchmark_out, aisc_out = write_source_files(tmp_path)
    path = tmp_path / "scoreboard.jsonl"
    result = run_cli(
        "--file", str(path), "--record", "--commit", "514632e",
        "--from-pytest", str(pytest_out),
        "--from-benchmark", str(benchmark_out),
        "--from-aisc", str(aisc_out), "--json")
    assert result.returncode == 0, result.stderr
    latest = json.loads(result.stdout)["latest"]
    assert latest["tests_passed"] == 855
    assert latest["tests_failed"] == 2
    assert latest["benchmark_passed"] == 13
    assert latest["benchmark_total"] == 13
    assert latest["aisc_passed"] == 20
    assert latest["aisc_total"] == 22
    assert verify_runs(load_jsonl(path))["ok"] is True
    stored = path.read_text(encoding="utf-8")
    assert "PRIVATE" not in stored
    assert "passed in" not in stored


def test_cli_explicit_counts_override_sources(tmp_path):
    pytest_out, benchmark_out, aisc_out = write_source_files(tmp_path)
    path = tmp_path / "scoreboard.jsonl"
    result = run_cli(
        "--file", str(path), "--record", "--commit", "514632e",
        "--from-pytest", str(pytest_out),
        "--from-benchmark", str(benchmark_out),
        "--from-aisc", str(aisc_out),
        "--tests-passed", "857", "--tests-failed", "0",
        "--benchmark-passed", "12", "--aisc-total", "24", "--json")
    assert result.returncode == 0, result.stderr
    latest = json.loads(result.stdout)["latest"]
    assert latest["tests_passed"] == 857
    assert latest["tests_failed"] == 0
    assert latest["benchmark_passed"] == 12
    assert latest["benchmark_total"] == 13
    assert latest["aisc_passed"] == 20
    assert latest["aisc_total"] == 24


def test_cli_reads_one_source_from_stdin(tmp_path):
    path = tmp_path / "scoreboard.jsonl"
    result = run_cli(
        "--file", str(path), "--record", "--commit", "514632e",
        "--from-pytest", "-", "--json",
        stdin_text="1 passed, 2 skipped in 0.5s\n")
    assert result.returncode == 0, result.stderr
    latest = json.loads(result.stdout)["latest"]
    assert latest["tests_passed"] == 1
    assert latest["tests_failed"] == 0


def test_cli_rejects_multiple_stdin_sources(tmp_path):
    path = tmp_path / "scoreboard.jsonl"
    result = run_cli(
        "--file", str(path), "--record", "--commit", "514632e",
        "--from-pytest", "-", "--from-benchmark", "-",
        stdin_text="1 passed in 0.5s\n")
    assert result.returncode == 2
    assert "stdin" in result.stderr


def test_cli_sources_require_record(tmp_path):
    result = run_cli(
        "--file", str(tmp_path / "scoreboard.jsonl"),
        "--from-pytest", str(tmp_path / "missing.txt"))
    assert result.returncode == 2
    assert "--record" in result.stderr


def test_cli_commit_falls_back_to_git_head(tmp_path):
    path = tmp_path / "scoreboard.jsonl"
    result = run_cli(
        "--file", str(path), "--record", "--tests-passed", "857", "--json")
    assert result.returncode == 0, result.stderr
    expected = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
        capture_output=True, text=True, encoding="utf-8", check=True)
    assert json.loads(result.stdout)["latest"]["commit"] == (
        expected.stdout.strip().lower())


def test_cli_commit_fallback_fails_outside_git(tmp_path):
    elsewhere = tmp_path / "work"
    elsewhere.mkdir()
    path = tmp_path / "scoreboard.jsonl"
    result = run_cli(
        "--file", str(path), "--record", "--tests-passed", "1",
        cwd=elsewhere)
    assert result.returncode == 2
    assert "commit" in result.stderr
    assert not path.exists()
