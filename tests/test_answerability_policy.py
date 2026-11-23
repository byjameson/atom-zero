from benchmarks.answerability_signal import (
    TARGETS, expected_calibration_error, policy_point, select_threshold,
    wilson_interval)


def test_policy_point_counts_acceptance_and_refusal():
    labels = [1, 1, 0, 0]
    probabilities = [0.9, 0.4, 0.6, 0.1]
    point = policy_point(labels, probabilities, 0.5)
    assert point["accepted"] == 2
    assert point["refused"] == 2
    assert point["answerable_accepted"] == 1
    assert point["no_answer_refused"] == 1
    assert point["answerable_coverage"] == 0.5
    assert point["refusal_precision"] == 0.5
    assert point["refusal_recall"] == 0.5
    assert point["overall_accuracy"] == 0.5


def test_perfect_separation_meets_targets():
    labels = [1, 1, 0, 0]
    probabilities = [0.9, 0.8, 0.2, 0.1]
    selection = select_threshold(labels, probabilities)
    assert selection["feasible"] is True
    point = selection["point"]
    assert point["answerable_coverage"] >= TARGETS["answerable_coverage"]
    assert point["refusal_precision"] >= TARGETS["refusal_precision"]
    assert point["refusal_recall"] >= TARGETS["refusal_recall"]


def test_policy_reports_misses_when_targets_are_unreachable():
    labels = [1, 0, 1, 0]
    probabilities = [0.6, 0.6, 0.4, 0.4]
    selection = select_threshold(labels, probabilities)
    assert selection["feasible"] is False
    assert selection["misses"]


def test_ece_is_zero_for_perfect_calibration():
    labels = [0, 0, 1, 1]
    probabilities = [0.0, 0.0, 1.0, 1.0]
    assert expected_calibration_error(labels, probabilities) == 0.0


def test_wilson_interval_is_bounded():
    low, high = wilson_interval(8, 10)
    assert 0.0 <= low < 0.8 < high <= 1.0
    assert wilson_interval(0, 0) == [0.0, 0.0]
