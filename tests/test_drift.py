import json
import subprocess
import sys
import tempfile
from pathlib import Path

from atomzero.drift import compare_profiles

ROOT = Path(__file__).resolve().parents[1]


def run_cli(*args: str):
    return subprocess.run(
        [sys.executable, "-m", "atomzero.drift", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={"PYTHONPATH": str(ROOT / "src"), "PYTHONIOENCODING": "utf-8"},
        check=False,
    )


def _profile(brier, ece, n=200, commit="aaaaaaa", threshold=0.5):
    return {"brier": brier, "ece": ece, "n": n,
            "commit": commit, "threshold": threshold}


def test_improved_profile():
    report = compare_profiles(_profile(0.24, 0.13), _profile(0.20, 0.10))
    assert report["status"] == "improved"
    assert report["ok"] is True
    assert report["delta"]["brier"] == -0.04
    assert report["weights_changed"] is False


def test_stable_profile():
    report = compare_profiles(_profile(0.24, 0.13), _profile(0.241, 0.131))
    assert report["status"] == "stable"
    assert report["alerts"] == []


def test_brier_and_ece_drift():
    report = compare_profiles(_profile(0.24, 0.13), _profile(0.30, 0.20))
    assert report["status"] == "drifted"
    assert report["ok"] is False
    assert "brier_increased" in report["alerts"]
    assert "ece_increased" in report["alerts"]


def test_tolerance_can_suppress_small_drift():
    report = compare_profiles(
        _profile(0.24, 0.13), _profile(0.241, 0.131),
        max_brier_increase=0.01, max_ece_increase=0.01)
    assert report["alerts"] == []


def test_sample_size_drop():
    report = compare_profiles(_profile(0.24, 0.13, n=200),
                              _profile(0.24, 0.13, n=100))
    assert "sample_size_dropped" in report["alerts"]


def test_invalid_profile_and_tolerances():
    try:
        compare_profiles({"brier": 1}, _profile(0.2, 0.1))
    except ValueError:
        pass
    else:
        raise AssertionError("missing ece should fail")
    try:
        compare_profiles(_profile(0.2, 0.1), _profile(0.2, 0.1),
                         max_n_drop=-1)
    except ValueError:
        pass
    else:
        raise AssertionError("negative tolerance should fail")


def test_cli_json_and_exit_code(tmp_path):
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"
    baseline.write_text(json.dumps(_profile(0.24, 0.13)), encoding="utf-8")
    current.write_text(json.dumps(_profile(0.30, 0.20)), encoding="utf-8")
    result = run_cli("--baseline", str(baseline),
                     "--profile", str(current), "--json")
    assert result.returncode == 1
    data = json.loads(result.stdout)
    assert data["status"] == "drifted"
    assert data["ok"] is False


def test_cli_bad_input_exit_two(tmp_path):
    result = run_cli("--baseline", str(tmp_path / "none.json"),
                     "--profile", str(tmp_path / "none2.json"))
    assert result.returncode == 2
    assert "input error" in result.stderr
