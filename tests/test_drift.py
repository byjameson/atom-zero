import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from atomzero.drift import compare_profiles

ROOT = Path(__file__).resolve().parents[1]
SIGNAL_PROFILE = (ROOT / "benchmarks" / "results" / "answerability_signal"
                  / "answerability_profile.json")
TRANSFORMER_PROFILE = (ROOT / "benchmarks" / "results"
                       / "answerability_transformer"
                       / "answerability_profile.json")


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


def test_nested_metrics_brier_and_ece():
    baseline = {"metrics": {"brier": 0.24, "ece": 0.13}}
    current = {"metrics": {"brier": 0.30, "ece": 0.20}}
    report = compare_profiles(baseline, current)
    assert report["baseline"]["brier"] == 0.24
    assert report["current"]["ece"] == 0.20
    assert "brier_increased" in report["alerts"]
    assert "ece_increased" in report["alerts"]


def test_top_level_metrics_take_precedence():
    baseline = {"brier": 0.24, "ece": 0.13,
                "metrics": {"brier": 0.99, "ece": 0.99}}
    report = compare_profiles(baseline, baseline)
    assert report["baseline"]["brier"] == 0.24
    assert report["baseline"]["ece"] == 0.13


def test_sample_size_mapping_prefers_heldout():
    baseline = _profile(0.24, 0.13,
                        n={"train": 1000, "calibration": 300, "heldout": 200})
    current = _profile(0.24, 0.13,
                       n={"train": 1000, "calibration": 300, "heldout": 100})
    report = compare_profiles(baseline, current)
    assert report["baseline"]["n"] == 200
    assert report["current"]["n"] == 100
    assert report["delta"]["n"] == -100
    assert "sample_size_dropped" in report["alerts"]


def test_sample_size_mapping_falls_back_to_calibration():
    profile = _profile(0.24, 0.13, n={"train": 1000, "calibration": 289})
    report = compare_profiles(profile, profile)
    assert report["baseline"]["n"] == 289


def test_sample_size_falls_back_to_metrics_n():
    profile = {"brier": 0.24, "ece": 0.13, "metrics": {"n": 200}}
    report = compare_profiles(profile, profile)
    assert report["baseline"]["n"] == 200
    assert report["current"]["n"] == 200


def test_missing_sample_size_does_not_crash():
    profile = {"brier": 0.24, "ece": 0.13}
    report = compare_profiles(profile, profile)
    assert report["baseline"]["n"] is None
    assert report["current"]["n"] is None
    assert report["delta"]["n"] is None
    assert report["alerts"] == []


def test_nested_selection_threshold_is_reported():
    profile = {"brier": 0.24, "ece": 0.13,
               "selection": {"threshold": 0.55}}
    report = compare_profiles(profile, profile)
    assert report["baseline"]["threshold"] == 0.55
    assert report["current"]["threshold"] == 0.55
    assert report["alerts"] == []


def test_top_level_threshold_takes_precedence():
    profile = {"brier": 0.24, "ece": 0.13, "threshold": 0.5,
               "selection": {"threshold": 0.9}}
    report = compare_profiles(profile, profile)
    assert report["baseline"]["threshold"] == 0.5


def test_metrics_threshold_fallback_and_change():
    baseline = {"brier": 0.24, "ece": 0.13, "metrics": {"threshold": 0.5}}
    current = {"brier": 0.24, "ece": 0.13, "metrics": {"threshold": 0.6}}
    report = compare_profiles(baseline, current)
    assert report["baseline"]["threshold"] == 0.5
    assert report["current"]["threshold"] == 0.6
    assert "threshold_changed" in report["alerts"]


def test_threshold_change_alerts():
    report = compare_profiles(_profile(0.24, 0.13, threshold=0.5),
                              _profile(0.24, 0.13, threshold=0.6))
    assert report["status"] == "drifted"
    assert report["ok"] is False
    assert "threshold_changed" in report["alerts"]


def test_tiny_threshold_change_is_ignored():
    report = compare_profiles(_profile(0.24, 0.13, threshold=0.5),
                              _profile(0.24, 0.13, threshold=0.5 + 1e-10))
    assert report["alerts"] == []


def _policy_profile(brier=0.24, ece=0.13, **policy):
    profile = _profile(brier, ece)
    profile.update(policy)
    return profile


def _aisc_point(**policy):
    return {"aisc": {"policy": {"recommendation": {"point": policy}}}}


def test_policy_metrics_are_reported():
    baseline = _policy_profile(
        refusal_recall=0.80, refusal_precision=0.75,
        answerable_coverage=0.70)
    current = _policy_profile(
        refusal_recall=0.79, refusal_precision=0.74,
        answerable_coverage=0.69)
    report = compare_profiles(baseline, current)
    assert report["policy"]["baseline"] == {
        "refusal_recall": 0.80, "refusal_precision": 0.75,
        "answerable_coverage": 0.70}
    assert report["policy"]["current"] == {
        "refusal_recall": 0.79, "refusal_precision": 0.74,
        "answerable_coverage": 0.69}
    assert report["policy"]["delta"] == {
        "refusal_recall": -0.01, "refusal_precision": -0.01,
        "answerable_coverage": -0.01}
    assert report["alerts"] == []


@pytest.mark.parametrize("metric,alert", [
    ("refusal_recall", "refusal_recall_decreased"),
    ("refusal_precision", "refusal_precision_decreased"),
    ("answerable_coverage", "answerable_coverage_decreased"),
])
def test_policy_metric_decrease_alerts(metric, alert):
    report = compare_profiles(
        _policy_profile(**{metric: 0.80}),
        _policy_profile(**{metric: 0.70}))
    assert report["status"] == "drifted"
    assert report["ok"] is False
    assert alert in report["alerts"]
    assert report["policy"]["delta"][metric] == -0.1


def test_nested_policy_metrics_from_aisc_recommendation():
    baseline = _profile(0.24, 0.13)
    baseline.update(_aisc_point(
        refusal_recall=0.80, refusal_precision=0.75,
        answerable_coverage=0.70))
    current = _profile(0.24, 0.13)
    current.update(_aisc_point(
        refusal_recall=0.70, refusal_precision=0.65,
        answerable_coverage=0.60))
    report = compare_profiles(baseline, current)
    assert report["policy"]["baseline"]["refusal_recall"] == 0.80
    assert report["policy"]["current"]["refusal_precision"] == 0.65
    assert report["policy"]["current"]["answerable_coverage"] == 0.60
    assert "refusal_recall_decreased" in report["alerts"]
    assert "refusal_precision_decreased" in report["alerts"]
    assert "answerable_coverage_decreased" in report["alerts"]


def test_nested_refusal_recall_accepts_aisc_no_answer_refusal_name():
    profile = _profile(0.24, 0.13)
    profile.update(_aisc_point(
        no_answer_refusal=0.60, refusal_precision=0.50,
        answerable_coverage=0.40))
    report = compare_profiles(profile, profile)
    assert report["policy"]["baseline"]["refusal_recall"] == 0.60
    assert report["policy"]["baseline"]["refusal_precision"] == 0.50
    assert report["policy"]["baseline"]["answerable_coverage"] == 0.40


def test_top_level_policy_metrics_take_precedence():
    profile = _profile(0.24, 0.13)
    profile.update(
        refusal_recall=0.90,
        **_aisc_point(
            refusal_precision=0.80, answerable_coverage=0.75))
    report = compare_profiles(profile, profile)
    assert report["policy"]["baseline"] == {
        "refusal_recall": 0.90, "refusal_precision": 0.80,
        "answerable_coverage": 0.75}


def test_policy_metrics_fall_back_per_metric():
    profile = _profile(0.24, 0.13)
    profile["refusal_recall"] = 0.90
    profile.update(_aisc_point(
        refusal_precision=0.80, answerable_coverage=0.75))
    report = compare_profiles(profile, profile)
    assert report["policy"]["baseline"]["refusal_recall"] == 0.90
    assert report["policy"]["baseline"]["refusal_precision"] == 0.80
    assert report["policy"]["baseline"]["answerable_coverage"] == 0.75


@pytest.mark.parametrize("metric,param,alert", [
    ("refusal_recall", "max_refusal_recall_drop",
     "refusal_recall_decreased"),
    ("refusal_precision", "max_refusal_precision_drop",
     "refusal_precision_decreased"),
    ("answerable_coverage", "max_coverage_drop",
     "answerable_coverage_decreased"),
])
def test_policy_drop_tolerance_boundary(metric, param, alert):
    baseline = _policy_profile(**{metric: 0.80})
    current = _policy_profile(**{metric: 0.70})
    exact_drop = 0.80 - 0.70
    at_boundary = compare_profiles(
        baseline, current, **{param: exact_drop})
    assert alert not in at_boundary["alerts"]
    assert at_boundary["tolerances"][param] == exact_drop
    just_over = compare_profiles(
        baseline, current, **{param: exact_drop - 1e-9})
    assert alert in just_over["alerts"]
    suppressed = compare_profiles(baseline, current, **{param: 0.2})
    assert alert not in suppressed["alerts"]


@pytest.mark.parametrize("param", [
    "max_refusal_recall_drop",
    "max_refusal_precision_drop",
    "max_coverage_drop",
])
def test_policy_tolerances_must_be_non_negative(param):
    with pytest.raises(ValueError):
        compare_profiles(
            _profile(0.24, 0.13), _profile(0.24, 0.13),
            **{param: -0.01})


def test_missing_policy_metrics_do_not_alert():
    report = compare_profiles(_profile(0.24, 0.13), _profile(0.24, 0.13))
    assert report["policy"]["baseline"] == {
        "refusal_recall": None, "refusal_precision": None,
        "answerable_coverage": None}
    assert report["policy"]["current"] == {
        "refusal_recall": None, "refusal_precision": None,
        "answerable_coverage": None}
    assert report["policy"]["delta"] == {
        "refusal_recall": None, "refusal_precision": None,
        "answerable_coverage": None}
    assert report["alerts"] == []


def test_policy_metric_missing_on_one_side_does_not_alert():
    baseline = _policy_profile(
        refusal_recall=0.80, refusal_precision=0.75,
        answerable_coverage=0.70)
    report = compare_profiles(baseline, _profile(0.24, 0.13))
    assert report["policy"]["current"]["refusal_recall"] is None
    assert report["policy"]["delta"]["refusal_recall"] is None
    assert "refusal_recall_decreased" not in report["alerts"]
    assert "refusal_precision_decreased" not in report["alerts"]
    assert "answerable_coverage_decreased" not in report["alerts"]


def test_malformed_nested_policy_metrics_are_treated_as_missing():
    profile = _profile(0.24, 0.13)
    profile["aisc"] = {"policy": {"recommendation": {
        "point": {"refusal_recall": "high",
                  "refusal_precision": None,
                  "answerable_coverage": True}}}}
    report = compare_profiles(profile, profile)
    assert report["policy"]["baseline"]["refusal_recall"] is None
    assert report["policy"]["baseline"]["answerable_coverage"] is None
    assert report["alerts"] == []


def test_cli_reports_policy_metrics_and_tolerance(tmp_path):
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"
    baseline.write_text(json.dumps(_policy_profile(
        refusal_recall=0.80, refusal_precision=0.75,
        answerable_coverage=0.70)), encoding="utf-8")
    current.write_text(json.dumps(_policy_profile(
        refusal_recall=0.70, refusal_precision=0.65,
        answerable_coverage=0.60)), encoding="utf-8")
    result = run_cli("--baseline", str(baseline),
                     "--profile", str(current), "--json",
                     "--max-refusal-recall-drop", "0.2")
    assert result.returncode == 1
    data = json.loads(result.stdout)
    assert data["policy"]["baseline"]["refusal_recall"] == 0.80
    assert data["policy"]["current"]["answerable_coverage"] == 0.60
    assert data["tolerances"]["max_refusal_recall_drop"] == 0.2
    assert "refusal_recall_decreased" not in data["alerts"]
    assert "refusal_precision_decreased" in data["alerts"]
    assert "answerable_coverage_decreased" in data["alerts"]


def test_real_answerability_profiles():
    if not SIGNAL_PROFILE.is_file() or not TRANSFORMER_PROFILE.is_file():
        pytest.skip("stored answerability profiles are not available")
    baseline = json.loads(SIGNAL_PROFILE.read_text(encoding="utf-8-sig"))
    current = json.loads(TRANSFORMER_PROFILE.read_text(encoding="utf-8-sig"))
    # Tighten the tolerances so the small observed Brier move is surfaced;
    # the defaults allow +0.02 and would hide it.
    report = compare_profiles(
        baseline, current,
        max_brier_increase=0.001, max_ece_increase=0.001)
    assert report["ok"] is False
    assert "brier_increased" in report["alerts"]
    assert "ece_increased" in report["alerts"]
    assert any(alert in report["alerts"] for alert in (
        "refusal_recall_decreased", "refusal_precision_decreased",
        "answerable_coverage_decreased"))
    # The prior-weighted signal profile selects a low internal threshold with
    # zero measured refusals; the transformer keeps high recall at a much
    # lower answerable coverage. The direction below follows the new baseline.
    assert "answerable_coverage_decreased" in report["alerts"]
    assert "refusal_precision_decreased" not in report["alerts"]
    assert report["policy"]["baseline"]["refusal_recall"] == 0.0
    # The transformer stores the same ratio as no_answer_refusal at 0.7.
    assert report["policy"]["current"]["refusal_recall"] == 0.752577
    assert report["baseline"]["n"] == 200
    assert report["current"]["n"] == 200
    assert report["current"]["threshold"] == 0.6
    assert len(json.dumps(report)) < 4096  # raw profile arrays are not copied


def test_cli_accepts_real_transformer_profile():
    if not SIGNAL_PROFILE.is_file() or not TRANSFORMER_PROFILE.is_file():
        pytest.skip("stored answerability profiles are not available")
    result = run_cli("--baseline", str(SIGNAL_PROFILE),
                     "--profile", str(TRANSFORMER_PROFILE), "--json")
    assert "input error" not in result.stderr
    assert result.returncode == 1
    data = json.loads(result.stdout)
    assert data["status"] == "drifted"
    assert "ece_increased" in data["alerts"]
    assert data["current"]["brier"] == 0.276168
    assert data["current"]["n"] == 200
