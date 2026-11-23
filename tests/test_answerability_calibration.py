import csv
import json

import pytest

from atomzero.answerability import (
    FEATURE_NAMES, AnswerabilityGate, load_profile)
from benchmarks.answerability_calibration import (
    CALIBRATION_METHOD, DEFAULT_TARGET_PREVALENCE, brier_score,
    expected_calibration_error, grouped_out_of_fold_calibration,
    prior_weighted_isotonic)


def _shifted_sample(seed=7, size=800):
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(seed)
    calibration_labels = (rng.random(size) < 0.65).astype(int)
    calibration_scores = np.where(
        calibration_labels == 1,
        rng.beta(5, 3, size),
        rng.beta(3, 5, size))
    target_labels = (rng.random(size) < 0.40).astype(int)
    target_scores = np.where(
        target_labels == 1,
        rng.beta(5, 3, size),
        rng.beta(3, 5, size))
    return (calibration_scores, calibration_labels,
            target_scores, target_labels)


def test_prior_weighting_reduces_ece_under_label_shift():
    pytest.importorskip("sklearn")
    from sklearn.isotonic import IsotonicRegression

    (calibration_scores, calibration_labels,
     target_scores, target_labels) = _shifted_sample()
    target_prevalence = float(target_labels.mean())
    unweighted = IsotonicRegression(
        y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(
            calibration_scores, calibration_labels)
    weighted = prior_weighted_isotonic(
        calibration_scores, calibration_labels, target_prevalence)

    unweighted_ece = expected_calibration_error(
        target_labels.tolist(), unweighted.predict(target_scores).tolist())
    weighted_ece = expected_calibration_error(
        target_labels.tolist(), weighted.predict(target_scores).tolist())
    assert weighted_ece < unweighted_ece
    assert weighted.target_prevalence == pytest.approx(target_prevalence)
    calibration_prevalence = float(calibration_labels.mean())
    assert weighted.calibration_prevalence == pytest.approx(
        calibration_prevalence)
    assert weighted.positive_weight == pytest.approx(
        target_prevalence / calibration_prevalence)
    assert weighted.negative_weight == pytest.approx(
        (1.0 - target_prevalence) / (1.0 - calibration_prevalence))


def test_isotonic_calibration_preserves_pairwise_ranking_and_roc():
    np = pytest.importorskip("numpy")
    from sklearn.metrics import roc_auc_score

    labels = np.array([0] * 40 + [1] * 40)
    raw_scores = np.concatenate((
        np.linspace(0.01, 0.45, 40),
        np.linspace(0.55, 0.99, 40)))
    model = prior_weighted_isotonic(raw_scores, labels, 0.5)
    calibrated = model.predict(raw_scores)
    assert np.all(np.diff(calibrated) >= -1e-12)
    assert roc_auc_score(labels, raw_scores) == pytest.approx(
        roc_auc_score(labels, calibrated), abs=1e-12)


@pytest.mark.parametrize("target", [0.0, 1.0, -0.1, 1.1, "high", None])
def test_invalid_target_prevalence_raises(target):
    with pytest.raises(ValueError):
        prior_weighted_isotonic([0.1, 0.9], [0, 1], target)


@pytest.mark.parametrize("labels", [[0, 0, 0], [1, 1, 1], []])
def test_invalid_calibration_prevalence_raises(labels):
    with pytest.raises(ValueError):
        prior_weighted_isotonic([0.1] * len(labels), labels, 0.5)


def test_metric_helpers():
    assert expected_calibration_error([0, 0, 1, 1], [0, 0, 1, 1]) == 0.0
    assert brier_score([0, 1], [0, 1]) == 0.0
    assert brier_score([], []) == 0.0
    with pytest.raises(ValueError):
        expected_calibration_error([0, 1], [0.0])
    with pytest.raises(ValueError):
        brier_score([0, 1], [0.0])


def _grouped_sample():
    scores, labels, groups = [], [], []
    for group in range(5):
        for index in range(10):
            label = index % 2
            scores.append(round(
                0.05 + 0.90 * (group * 10 + index) / 49.0
                + (0.01 if label else 0.0), 6))
            labels.append(label)
            groups.append(f"group-{group}")
    return scores, labels, groups


def test_out_of_fold_is_deterministic_and_one_probability_per_row():
    scores, labels, groups = _grouped_sample()
    first = grouped_out_of_fold_calibration(
        scores, labels, groups, target_prevalence=0.5, n_splits=5)
    second = grouped_out_of_fold_calibration(
        scores, labels, groups, target_prevalence=0.5, n_splits=5)
    assert len(first) == len(scores)
    assert first == second
    assert all(0.0 <= probability <= 1.0 for probability in first)

    without_groups = grouped_out_of_fold_calibration(
        scores, labels, target_prevalence=0.5, seed=3)
    repeated = grouped_out_of_fold_calibration(
        scores, labels, target_prevalence=0.5, seed=3)
    assert without_groups == repeated
    assert len(without_groups) == len(scores)


def test_out_of_fold_does_not_fit_on_the_held_out_group():
    scores, labels, groups = _grouped_sample()
    baseline = grouped_out_of_fold_calibration(
        scores, labels, groups, target_prevalence=0.5, n_splits=5)
    flipped = list(labels)
    for index, group in enumerate(groups):
        if group == "group-0":
            flipped[index] = 1 - flipped[index]
    changed = grouped_out_of_fold_calibration(
        scores, flipped, groups, target_prevalence=0.5, n_splits=5)
    held_out = [index for index, group in enumerate(groups)
                if group == "group-0"]
    assert [baseline[index] for index in held_out] == pytest.approx(
        [changed[index] for index in held_out], abs=1e-12)


def _profile():
    return {
        "version": 1,
        "features": list(FEATURE_NAMES),
        "coefficients": [0.0] * len(FEATURE_NAMES),
        "intercept": -2.0,
        "calibration": {"x": [0.0, 0.5, 1.0], "y": [0.0, 0.25, 1.0]},
        "threshold": 0.5,
        "calibration_method": CALIBRATION_METHOD,
        "prevalence": {"calibration": 0.640139, "target": 0.515},
    }


def _hit():
    return {
        "source": "note-1",
        "title": "Launch",
        "section": "Codes",
        "page": None,
        "chunk": "The launch code is L-7781. The backup room is B-12.",
        "rerank_score": 0.9,
        "embedding_score": 0.8,
        "kind": "document",
        "verified": True,
    }


def test_profile_round_trip_keeps_method_and_prevalence(tmp_path):
    path = tmp_path / "answerability.json"
    path.write_text(json.dumps(_profile()), encoding="utf-8")
    loaded = load_profile(path)
    assert loaded["calibration_method"] == CALIBRATION_METHOD
    assert loaded["prevalence"] == {
        "calibration": 0.640139, "target": 0.515}
    gate = AnswerabilityGate(profile=path, mode="shadow")
    assert gate.enabled is True
    probability = gate.probability("what is the launch code?", [_hit()])
    assert 0.0 <= probability <= 1.0
    assert gate.evaluate("what is the launch code?", [_hit()])["applied"]


def test_profile_rejects_malformed_prevalence(tmp_path):
    profile = _profile()
    profile["prevalence"] = "not-an-object"
    path = tmp_path / "bad-prevalence.json"
    path.write_text(json.dumps(profile), encoding="utf-8")
    with pytest.raises(ValueError):
        load_profile(path)


def test_observations_persist_raw_scores(tmp_path):
    from benchmarks.answerability_signal import write_observations

    path = tmp_path / "observations.csv"
    rows = [
        {"question_id": "q1", "label": 1},
        {"question_id": "q2", "label": 0},
    ]
    write_observations(
        path, rows, [0.70, 0.20], "heldout", [0.90, 0.10])
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == [
            "question_id", "prediction", "outcome", "answerable", "split",
            "raw_score"]
        output = list(reader)
    assert output[0]["prediction"] == "0.700000"
    assert output[0]["raw_score"] == "0.900000"
    assert output[1]["raw_score"] == "0.100000"
    assert DEFAULT_TARGET_PREVALENCE == 0.515
