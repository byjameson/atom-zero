"""Prevalence-corrected calibration helpers for answerability benchmarks.

The answerability profiles are fitted on a context-grouped calibration split
whose answerable prevalence can differ from the held-out target distribution.
Plain isotonic regression inherits the calibration prevalence, so when the
target split is more imbalanced the calibrated probabilities are systematically
too high. The helpers here fit the same monotone map with prior weights that
rebalance the calibration rows to a target prevalence.

They also provide an out-of-fold report so calibration metrics are not measured
on the rows that fitted the map. The fitted out-of-fold maps are measurement
artifacts only; the production profile keeps the single map fitted on the
calibration split.

All helpers are local and deterministic. scikit-learn and numpy are benchmark
dependencies, so imports are lazy and the module stays importable without them.
"""
from __future__ import annotations

from typing import Iterable, Sequence

CALIBRATION_METHOD = "prior_weighted_isotonic_v1"
DEFAULT_TARGET_PREVALENCE = 0.515


def _probability(value: float, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number in (0, 1)") from exc
    if not 0.0 < number < 1.0:
        raise ValueError(f"{name} must be in (0, 1), got {number!r}")
    return number


def calibration_prevalence(labels: Sequence[int]) -> float:
    values = list(labels)
    if not values:
        raise ValueError("labels must not be empty")
    for label in values:
        if label not in (0, 1):
            raise ValueError(f"labels must be binary, got {label!r}")
    prevalence = sum(values) / len(values)
    return _probability(prevalence, "calibration prevalence")


def prior_weighted_isotonic(raw_scores: Sequence[float],
                            labels: Sequence[int],
                            target_prevalence: float):
    """Fit isotonic calibration reweighted to ``target_prevalence``.

    Positive rows get weight ``target / calibration_prevalence`` and negative
    rows get weight ``(1 - target) / (1 - calibration_prevalence)``. The map
    stays monotone, so raw ranking is preserved apart from the plateaus that
    isotonic regression always introduces.
    """
    target = _probability(target_prevalence, "target prevalence")
    scores = [float(value) for value in raw_scores]
    values = list(labels)
    if not scores or not values:
        raise ValueError("raw_scores and labels must not be empty")
    if len(scores) != len(values):
        raise ValueError("raw_scores and labels must have equal length")
    observed = calibration_prevalence(values)
    positive_weight = target / observed
    negative_weight = (1.0 - target) / (1.0 - observed)
    weights = [positive_weight if label == 1 else negative_weight
               for label in values]
    try:
        from sklearn.isotonic import IsotonicRegression
    except ImportError as exc:
        raise RuntimeError(
            "install the benchmark dependency first: pip install scikit-learn"
        ) from exc
    model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    model.fit(scores, values, sample_weight=weights)
    model.target_prevalence = target
    model.calibration_prevalence = observed
    model.positive_weight = positive_weight
    model.negative_weight = negative_weight
    return model


def isotonic_curve(model, raw_scores: Iterable[float]):
    """Return the sorted ``(x, y)`` grid stored in a profile."""
    import numpy as np

    scores = np.asarray([float(value) for value in raw_scores], dtype=float)
    if scores.size == 0:
        return [], []
    grid = np.unique(np.concatenate(([0.0, 1.0], scores)))
    calibrated = model.predict(grid)
    return ([float(value) for value in grid],
            [float(value) for value in calibrated])


def expected_calibration_error(labels: Sequence[int],
                               probabilities: Sequence[float],
                               bins: int = 10) -> float:
    values = list(labels)
    scores = [float(value) for value in probabilities]
    if len(values) != len(scores):
        raise ValueError("labels and probabilities must have equal length")
    total = len(scores)
    if total == 0:
        return 0.0
    if bins < 1:
        raise ValueError("bins must be positive")
    error = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        members = [i for i, probability in enumerate(scores)
                   if (low <= probability < high
                       or (index == bins - 1 and probability == 1.0))]
        if not members:
            continue
        mean_prediction = sum(scores[i] for i in members) / len(members)
        accuracy = sum(values[i] for i in members) / len(members)
        error += (len(members) / total) * abs(mean_prediction - accuracy)
    return error


def brier_score(labels: Sequence[int],
                probabilities: Sequence[float]) -> float:
    values = list(labels)
    scores = [float(value) for value in probabilities]
    if len(values) != len(scores):
        raise ValueError("labels and probabilities must have equal length")
    if not values:
        return 0.0
    return sum((probability - label) ** 2
               for probability, label in zip(scores, values)) / len(values)


def grouped_out_of_fold_calibration(
        raw_scores: Sequence[float],
        labels: Sequence[int],
        groups: Sequence | None = None,
        target_prevalence: float = DEFAULT_TARGET_PREVALENCE,
        n_splits: int = 5,
        seed: int = 0) -> list[float]:
    """Return one prior-weighted isotonic probability per row, out of fold.

    ``GroupKFold`` is used when ``groups`` is given, otherwise shuffled
    ``KFold`` with a fixed seed. Each fold's map is fitted only on the other
    folds and is discarded; only the held-out predictions are returned.
    """
    scores = [float(value) for value in raw_scores]
    values = list(labels)
    if not scores or not values:
        raise ValueError("raw_scores and labels must not be empty")
    if len(scores) != len(values):
        raise ValueError("raw_scores and labels must have equal length")
    _probability(target_prevalence, "target prevalence")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    try:
        from sklearn.model_selection import GroupKFold, KFold
    except ImportError as exc:
        raise RuntimeError(
            "install the benchmark dependency first: pip install scikit-learn"
        ) from exc
    group_values = None if groups is None else list(groups)
    if group_values is not None and len(group_values) != len(values):
        raise ValueError("groups must have the same length as labels")
    if group_values is not None:
        unique_groups = len(set(group_values))
        folds = min(n_splits, unique_groups)
        if folds < 2:
            raise ValueError("at least two groups are required for GroupKFold")
        splitter = GroupKFold(n_splits=folds)
        splits = splitter.split(scores, values, group_values)
    else:
        folds = min(n_splits, len(values))
        if folds < 2:
            raise ValueError("at least two rows are required for KFold")
        splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
        splits = splitter.split(scores)
    output = [0.0] * len(scores)
    for train_index, test_index in splits:
        train_scores = [scores[index] for index in train_index]
        train_labels = [values[index] for index in train_index]
        model = prior_weighted_isotonic(
            train_scores, train_labels, target_prevalence)
        predictions = model.predict([scores[index] for index in test_index])
        for index, probability in zip(test_index, predictions):
            output[index] = max(0.0, min(1.0, float(probability)))
    return output
