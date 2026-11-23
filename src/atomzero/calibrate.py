"""Local isotonic calibration and threshold recommendation.

Given labelled citation scores (``score,label`` CSV), this module fits a
monotonic isotonic map with pool-adjacent-violators (PAVA), reports Brier/ECE,
and recommends a threshold for a target precision. The resulting JSON can be
used with ``ATOMZERO_CITATION_CALIBRATION``.

Example:

    python -m atomzero.calibrate citations.csv \
      --out citation-calibration.json --target-precision 0.9
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def _as_label(value) -> int:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "valid", "answerable", "positive"}:
        return 1
    if text in {"0", "false", "no", "n", "invalid", "unanswerable", "negative"}:
        return 0
    try:
        number = float(text)
    except (TypeError, ValueError):
        raise ValueError(f"invalid label: {value!r}")
    if number not in (0.0, 1.0):
        raise ValueError(f"label must be 0 or 1: {value!r}")
    return int(number)


def _as_score(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"invalid score: {value!r}")
    if not math.isfinite(number):
        raise ValueError(f"score must be finite: {value!r}")
    return number


def _interpolate(x_values, y_values, score: float) -> float:
    if score <= x_values[0]:
        return float(y_values[0])
    if score >= x_values[-1]:
        return float(y_values[-1])
    for index in range(len(x_values) - 1):
        left, right = x_values[index], x_values[index + 1]
        if left <= score <= right:
            if right <= left:
                return float(y_values[index + 1])
            ratio = (score - left) / (right - left)
            return float(y_values[index] + ratio *
                         (y_values[index + 1] - y_values[index]))
    return float(y_values[-1])


def fit_isotonic(scores, labels) -> dict:
    """Fit a monotonic score -> probability map with PAVA.

    Returns ``{"x_thresholds": [...], "y_thresholds": [...]}``. The map uses
    the sorted unique scores as x thresholds and the pooled means as y values.
    """
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have equal length")
    if not scores:
        raise ValueError("at least one sample is required")
    grouped = {}
    for score, label in zip(scores, labels):
        score = _as_score(score)
        label = _as_label(label)
        total, count = grouped.get(score, (0.0, 0))
        grouped[score] = (total + label, count + 1)
    x_values = sorted(grouped)
    if len(x_values) < 2:
        raise ValueError("at least two distinct scores are required")
    blocks = []
    for score in x_values:
        total, count = grouped[score]
        blocks.append({
            "sum": float(total),
            "weight": int(count),
            "x_start": score,
            "x_end": score,
        })
        while len(blocks) >= 2:
            left, right = blocks[-2], blocks[-1]
            left_mean = left["sum"] / left["weight"]
            right_mean = right["sum"] / right["weight"]
            if left_mean <= right_mean:
                break
            blocks[-2:] = [{
                "sum": left["sum"] + right["sum"],
                "weight": left["weight"] + right["weight"],
                "x_start": left["x_start"],
                "x_end": right["x_end"],
            }]
    y_by_x = {}
    for block in blocks:
        mean = block["sum"] / block["weight"]
        for score in x_values:
            if block["x_start"] <= score <= block["x_end"]:
                y_by_x[score] = mean
    y_values = [max(0.0, min(1.0, y_by_x[score]))
                for score in x_values]
    return {"x_thresholds": [float(value) for value in x_values],
            "y_thresholds": y_values}


def calibrate(scores, labels) -> list[tuple[float, float]]:
    mapping = fit_isotonic(scores, labels)
    return list(zip(mapping["x_thresholds"], mapping["y_thresholds"]))


def predict(mapping: dict, score: float) -> float:
    return _interpolate(mapping["x_thresholds"],
                        mapping["y_thresholds"], score)


def metrics(scores, labels, mapping=None) -> dict:
    scores = [_as_score(value) for value in scores]
    labels = [_as_label(value) for value in labels]
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have equal length")
    if not scores:
        raise ValueError("at least one sample is required")
    mapping = mapping or fit_isotonic(scores, labels)
    predictions = [predict(mapping, score) for score in scores]
    brier = sum((prob - label) ** 2
                for prob, label in zip(predictions, labels)) / len(labels)
    bins = []
    for index in range(10):
        lower, upper = index / 10, (index + 1) / 10
        members = [(prob, label)
                   for prob, label in zip(predictions, labels)
                   if lower <= prob < upper
                   or (index == 9 and prob == 1.0)]
        if members:
            mean_prob = sum(prob for prob, _ in members) / len(members)
            accuracy = sum(label for _, label in members) / len(members)
            bins.append((len(members), mean_prob, accuracy))
    ece = sum(count * abs(mean_prob - accuracy)
              for count, mean_prob, accuracy in bins) / len(labels)
    return {
        "n": len(labels),
        "positives": sum(labels),
        "negatives": len(labels) - sum(labels),
        "brier": round(brier, 6),
        "ece": round(ece, 6),
        "mean_prediction": round(
            sum(predictions) / len(predictions), 6),
    }


def threshold_report(scores, labels, mapping=None,
                     threshold: float = 0.5) -> dict:
    scores = [_as_score(value) for value in scores]
    labels = [_as_label(value) for value in labels]
    mapping = mapping or fit_isotonic(scores, labels)
    accepted = []
    for score, label in zip(scores, labels):
        if predict(mapping, score) >= threshold:
            accepted.append(label)
    true_positive = sum(accepted)
    positives = sum(labels)
    precision = (true_positive / len(accepted)) if accepted else 0.0
    recall = (true_positive / positives) if positives else 0.0
    return {
        "threshold": round(float(threshold), 4),
        "accepted": len(accepted),
        "true_positive": true_positive,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
    }


def recommend_threshold(scores, labels, mapping=None,
                        target_precision: float = 0.9,
                        min_recall: float = 0.0) -> dict:
    """Pick the highest-recall threshold meeting the precision target."""
    scores = [_as_score(value) for value in scores]
    labels = [_as_label(value) for value in labels]
    mapping = mapping or fit_isotonic(scores, labels)
    candidates = {round(index / 100, 2) for index in range(101)}
    candidates.update(round(predict(mapping, score), 4) for score in scores)
    reports = [threshold_report(scores, labels, mapping, threshold)
               for threshold in sorted(candidates)]
    feasible = [report for report in reports
                if report["precision"] >= target_precision
                and report["recall"] >= min_recall
                and report["accepted"] > 0]
    if feasible:
        chosen = max(feasible,
                     key=lambda report: (report["recall"],
                                         report["precision"],
                                         -report["threshold"]))
        chosen["matched_target"] = True
    else:
        chosen = max(reports,
                     key=lambda report: (report["precision"],
                                         report["recall"],
                                         -report["threshold"]))
        chosen["matched_target"] = chosen["precision"] >= target_precision
    return chosen


def load_csv(path) -> tuple[list[float], list[int]]:
    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("CSV needs a header row")
        fields = {name.strip().lower(): name for name in reader.fieldnames}
        score_field = next((fields[name] for name in
                            ("score", "rerank_score", "probability", "p")
                            if name in fields), None)
        label_field = next((fields[name] for name in
                            ("label", "valid", "answerable",
                             "citation_valid")
                            if name in fields), None)
        if score_field is None or label_field is None:
            raise ValueError(
                "CSV needs score/rerank_score and label/valid columns")
        scores = []
        labels = []
        for line, row in enumerate(reader, 2):
            try:
                scores.append(_as_score(row[score_field]))
                labels.append(_as_label(row[label_field]))
            except ValueError as exc:
                raise ValueError(f"line {line}: {exc}")
    return scores, labels


def calibrate_file(path, target_precision: float = 0.9,
                   min_recall: float = 0.0) -> dict:
    scores, labels = load_csv(path)
    mapping = fit_isotonic(scores, labels)
    report = recommend_threshold(
        scores, labels, mapping, target_precision=target_precision,
        min_recall=min_recall)
    result = {
        "profile": "custom",
        "isotonic": mapping,
        "threshold": report["threshold"],
        "metrics": {
            **metrics(scores, labels, mapping),
            **report,
        },
        "target_precision": target_precision,
        "min_recall": min_recall,
    }
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", help="CSV with score and label columns")
    parser.add_argument("--out", default="citation-calibration.json")
    parser.add_argument("--target-precision", type=float, default=0.9)
    parser.add_argument("--min-recall", type=float, default=0.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = calibrate_file(
            args.csv, target_precision=args.target_precision,
            min_recall=args.min_recall)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
        return 2
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"wrote {output}")
        print(f"threshold={result['threshold']} "
              f"precision={result['metrics']['precision']} "
              f"recall={result['metrics']['recall']} "
              f"brier={result['metrics']['brier']} "
              f"ece={result['metrics']['ece']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
