import json

import pytest

from atomzero.calibrate import (
    calibrate, calibrate_file, fit_isotonic, load_csv, predict,
    recommend_threshold)
from atomzero.citation_gate import CitationGate


def test_isotonic_map_is_monotonic():
    scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    labels = [0, 1, 0, 0, 1, 1]
    mapping = fit_isotonic(scores, labels)
    assert mapping["x_thresholds"] == sorted(mapping["x_thresholds"])
    assert mapping["y_thresholds"] == sorted(mapping["y_thresholds"])
    assert all(0.0 <= value <= 1.0
               for value in mapping["y_thresholds"])


def test_duplicate_scores_are_pooled():
    mapping = fit_isotonic(
        [0.2, 0.2, 0.8, 0.8], [0, 1, 0, 1])
    assert mapping["x_thresholds"] == [0.2, 0.8]
    assert mapping["y_thresholds"] == [0.5, 0.5]


def test_calibration_recommends_a_threshold():
    scores = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
    labels = [0, 0, 0, 1, 1, 1]
    mapping = fit_isotonic(scores, labels)
    report = recommend_threshold(
        scores, labels, mapping, target_precision=0.9)
    assert report["precision"] >= 0.9
    assert report["recall"] > 0
    gate = CitationGate(
        threshold=report["threshold"], calibration=mapping)
    assert gate.probability(0.8) >= report["threshold"]
    assert gate.probability(0.1) < report["threshold"]


def test_csv_round_trip_and_gate_map(tmp_path):
    source = tmp_path / "citations.csv"
    source.write_text(
        "score,label\n0.1,0\n0.2,0\n0.3,0\n0.7,1\n0.8,1\n0.9,1\n",
        encoding="utf-8")
    scores, labels = load_csv(source)
    assert len(scores) == 6 and sum(labels) == 3
    result = calibrate_file(source, target_precision=0.9)
    assert result["isotonic"]["x_thresholds"] == sorted(
        result["isotonic"]["x_thresholds"])
    assert result["threshold"] > 0
    gate = CitationGate(
        threshold=result["threshold"],
        calibration=result["isotonic"])
    assert gate.probability(0.9) >= result["threshold"]
    output = tmp_path / "calibration.json"
    output.write_text(json.dumps(result), encoding="utf-8")
    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded["isotonic"] == result["isotonic"]


def test_calibration_rejects_bad_input(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("score,label\n0.4,maybe\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_csv(bad)
    with pytest.raises(ValueError):
        fit_isotonic([0.2], [1])
