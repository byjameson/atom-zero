import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from atomzero.improvement import record_decision
from atomzero.scoreboard import record_run
from atomzero.trend import build_trend

ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 11, 23, 12, 0, tzinfo=timezone.utc)
ACTION = {"action_id": "imp-01-unsupported_claims",
          "kind": "unsupported_claims"}


def run_cli(*args: str):
    return subprocess.run(
        [sys.executable, "-m", "atomzero.trend", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={"PYTHONPATH": str(ROOT / "src"), "PYTHONIOENCODING": "utf-8"},
        check=False,
    )


def _runs(tests=10, failed=0, benchmark=5, total=5, aisc=4, aisc_total=6):
    runs = []
    record_run(runs, commit="aaaaaaa", tests_passed=8, tests_failed=1,
               benchmark_passed=4, benchmark_total=5, aisc_passed=3,
               aisc_total=6, now="2026-11-20T00:00:00Z")
    record_run(runs, commit="bbbbbbb", tests_passed=tests, tests_failed=failed,
               benchmark_passed=benchmark, benchmark_total=total,
               aisc_passed=aisc, aisc_total=aisc_total,
               now="2026-11-21T00:00:00Z")
    return runs


def test_empty_trend():
    report = build_trend([], [])
    assert report["ok"] is True
    assert report["alerts"] == []
    assert report["weights_changed"] is False


def test_positive_trend_has_no_alerts():
    report = build_trend([], _runs())
    assert report["alerts"] == []
    assert report["ok"] is True
    assert report["scoreboard"]["delta"]["tests_passed"] == 2


def test_regressions_raise_alerts():
    report = build_trend([], _runs(tests=5, failed=1, benchmark=3, aisc=2))
    assert "tests_failed" in report["alerts"]
    assert "tests_passed_decreased" in report["alerts"]
    assert "benchmark_passed_decreased" in report["alerts"]
    assert "aisc_passed_decreased" in report["alerts"]
    assert report["ok"] is False


def test_decisions_are_counted():
    ledger = []
    record_decision(ledger, ACTION, "approve", now=NOW)
    record_decision(ledger, ACTION, "reject", now=NOW)
    report = build_trend(ledger, [])
    assert report["decisions"]["total"] == 2
    assert report["decisions"]["decisions"] == {"approve": 1, "reject": 1}
    assert report["weights_changed"] is False


def test_cli_json(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    scoreboard = tmp_path / "scoreboard.jsonl"
    ledger.write_text(json.dumps({
        "version": 1, "at": "2026-11-23T12:00:00Z",
        "action_id": ACTION["action_id"], "kind": ACTION["kind"],
        "decision": "approve", "note_id": "",
    }) + "\n", encoding="utf-8")
    runs = []
    record_run(runs, commit="bbbbbbb", tests_passed=10,
               benchmark_passed=5, benchmark_total=5,
               aisc_passed=4, aisc_total=6, now=NOW)
    scoreboard.write_text("\n".join(
        json.dumps(run) for run in runs) + "\n", encoding="utf-8")
    result = run_cli("--ledger", str(ledger),
                     "--scoreboard", str(scoreboard), "--json")
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data["scoreboard"]["runs"] == 1
    assert data["decisions"]["total"] == 1


def test_cli_invalid_ledger_exit_two(tmp_path):
    ledger = tmp_path / "bad.jsonl"
    ledger.write_text('{"bad": true}\n', encoding="utf-8")
    result = run_cli("--ledger", str(ledger),
                     "--scoreboard", str(tmp_path / "none.jsonl"))
    assert result.returncode == 2
    assert "input error" in result.stderr
