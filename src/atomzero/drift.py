"""Deterministic calibration-drift monitor.

A local model can stay the same file while its measured calibration changes
because the evaluation set, retrieval, or thresholds changed. This module
compares a current profile with a stored baseline and reports the deltas. It is
a reporting layer: no benchmark is run, no weight is touched and no gate is
enabled automatically.

    atomzero-drift --profile current.json --baseline baseline.json --json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

__all__ = ["compare_profiles", "load_profile", "main"]

POLICY_METRICS = (
    "refusal_recall",
    "refusal_precision",
    "answerable_coverage",
)

DEFAULTS = {
    "max_brier_increase": 0.02,
    "max_ece_increase": 0.02,
    "max_n_drop": 0.20,
    "max_refusal_recall_drop": 0.02,
    "max_refusal_precision_drop": 0.02,
    "max_coverage_drop": 0.02,
}


def load_profile(path) -> dict:
    target = Path(path)
    data = json.loads(target.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError("profile must be a JSON object")
    return data


def _is_number(value) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float))


def _number(profile: dict, key: str) -> float:
    """Read a metric from the top level, falling back to ``metrics``."""
    value = profile.get(key)
    if not _is_number(value):
        metrics = profile.get("metrics")
        if isinstance(metrics, dict):
            value = metrics.get(key)
    if not _is_number(value):
        raise ValueError(f"profile.{key} must be a number")
    return float(value)


def _count(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value) if float(value).is_integer() else value


def _sample_size(profile: dict):
    """Resolve ``n`` from a scalar, a split mapping, or ``metrics.n``."""
    value = profile.get("n")
    if isinstance(value, dict):
        for key in ("heldout", "calibration", "n"):
            resolved = _count(value.get(key))
            if resolved is not None:
                return resolved
        return None
    resolved = _count(value)
    if resolved is not None:
        return resolved
    metrics = profile.get("metrics")
    if isinstance(metrics, dict):
        return _count(metrics.get("n"))
    return None


def _threshold(profile: dict):
    """Resolve the decision threshold from the supported locations."""
    value = profile.get("threshold")
    if _is_number(value):
        return value
    for section in ("selection", "metrics"):
        nested = profile.get(section)
        if isinstance(nested, dict) and _is_number(nested.get("threshold")):
            return nested.get("threshold")
    return None


def _policy_number(profile: dict, key: str):
    """Read a policy metric, falling back to the AISC recommendation point.

    Top-level values win. The transformer profile stores these metrics only at
    ``aisc.policy.recommendation.point``, so that path is used as a fallback.
    AISC names the refusal-recall ratio ``no_answer_refusal``; both names are
    accepted there so the stored transformer profile resolves without a
    schema change. Missing or non-numeric values resolve to ``None`` rather
    than raising: an optional metric should not stop the calibration
    comparison.
    """
    value = profile.get(key)
    if _is_number(value):
        return float(value)
    nested_keys = ((key, "no_answer_refusal")
                   if key == "refusal_recall" else (key,))
    aisc = profile.get("aisc")
    if isinstance(aisc, dict):
        policy = aisc.get("policy")
        if isinstance(policy, dict):
            recommendation = policy.get("recommendation")
            if isinstance(recommendation, dict):
                point = recommendation.get("point")
                if isinstance(point, dict):
                    for nested_key in nested_keys:
                        if _is_number(point.get(nested_key)):
                            return float(point[nested_key])
    return None


def compare_profiles(
    baseline: dict,
    current: dict,
    *,
    max_brier_increase: float = 0.02,
    max_ece_increase: float = 0.02,
    max_n_drop: float = 0.20,
    max_refusal_recall_drop: float = 0.02,
    max_refusal_precision_drop: float = 0.02,
    max_coverage_drop: float = 0.02,
) -> dict:
    """Compare two calibration profiles and return a deterministic report."""
    if not isinstance(baseline, dict) or not isinstance(current, dict):
        raise ValueError("baseline and current must be mappings")
    for name, value in (
            ("max_brier_increase", max_brier_increase),
            ("max_ece_increase", max_ece_increase),
            ("max_n_drop", max_n_drop),
            ("max_refusal_recall_drop", max_refusal_recall_drop),
            ("max_refusal_precision_drop", max_refusal_precision_drop),
            ("max_coverage_drop", max_coverage_drop)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a number")
        if value < 0:
            raise ValueError(f"{name} must be non-negative")

    old_brier = _number(baseline, "brier")
    new_brier = _number(current, "brier")
    old_ece = _number(baseline, "ece")
    new_ece = _number(current, "ece")
    old_n = _sample_size(baseline)
    new_n = _sample_size(current)
    old_threshold = _threshold(baseline)
    new_threshold = _threshold(current)
    old_policy = {key: _policy_number(baseline, key)
                  for key in POLICY_METRICS}
    new_policy = {key: _policy_number(current, key)
                  for key in POLICY_METRICS}
    alerts = []
    if new_brier > old_brier + max_brier_increase:
        alerts.append("brier_increased")
    if new_ece > old_ece + max_ece_increase:
        alerts.append("ece_increased")
    if (isinstance(old_n, int) and not isinstance(old_n, bool)
            and isinstance(new_n, int) and not isinstance(new_n, bool)
            and old_n > 0 and (old_n - new_n) / old_n > max_n_drop):
        alerts.append("sample_size_dropped")
    if (old_threshold is not None and new_threshold is not None
            and abs(old_threshold - new_threshold) > 1e-9):
        alerts.append("threshold_changed")
    for key, alert, tolerance in (
            ("refusal_recall", "refusal_recall_decreased",
             max_refusal_recall_drop),
            ("refusal_precision", "refusal_precision_decreased",
             max_refusal_precision_drop),
            ("answerable_coverage", "answerable_coverage_decreased",
             max_coverage_drop)):
        old_value = old_policy[key]
        new_value = new_policy[key]
        if (old_value is not None and new_value is not None
                and old_value - new_value > tolerance):
            alerts.append(alert)
    delta_brier = new_brier - old_brier
    delta_ece = new_ece - old_ece
    delta_policy = {
        key: (round(new_policy[key] - old_policy[key], 6)
              if old_policy[key] is not None
              and new_policy[key] is not None else None)
        for key in POLICY_METRICS
    }
    if alerts:
        status = "drifted"
    elif delta_brier < 0 and delta_ece < 0:
        status = "improved"
    else:
        status = "stable"
    return {
        "status": status,
        "ok": not alerts,
        "alerts": alerts,
        "baseline": {
            "brier": old_brier, "ece": old_ece, "n": old_n,
            "threshold": old_threshold,
            "commit": baseline.get("commit"),
        },
        "current": {
            "brier": new_brier, "ece": new_ece, "n": new_n,
            "threshold": new_threshold,
            "commit": current.get("commit"),
        },
        "delta": {
            "brier": round(delta_brier, 6),
            "ece": round(delta_ece, 6),
            "n": (new_n - old_n
                  if isinstance(old_n, int) and isinstance(new_n, int)
                  else None),
        },
        "policy": {
            "baseline": old_policy,
            "current": new_policy,
            "delta": delta_policy,
        },
        "tolerances": {
            "max_brier_increase": max_brier_increase,
            "max_ece_increase": max_ece_increase,
            "max_n_drop": max_n_drop,
            "max_refusal_recall_drop": max_refusal_recall_drop,
            "max_refusal_precision_drop": max_refusal_precision_drop,
            "max_coverage_drop": max_coverage_drop,
        },
        "weights_changed": False,
        "method": "deterministic_calibration_drift_v1",
    }


def _render(report: dict) -> str:
    delta = report["delta"]
    lines = [
        f"Calibration drift: {report['status']}",
        f"- Brier: {report['baseline']['brier']:.4f} -> "
        f"{report['current']['brier']:.4f} ({delta['brier']:+.4f})",
        f"- ECE: {report['baseline']['ece']:.4f} -> "
        f"{report['current']['ece']:.4f} ({delta['ece']:+.4f})",
    ]
    if delta["n"] is not None:
        lines.append(f"- n: {report['baseline']['n']} -> "
                     f"{report['current']['n']} ({delta['n']:+d})")
    policy = report["policy"]
    for key in POLICY_METRICS:
        old_value = policy["baseline"][key]
        new_value = policy["current"][key]
        if old_value is None and new_value is None:
            continue
        old_text = f"{old_value:.4f}" if old_value is not None else "n/a"
        new_text = f"{new_value:.4f}" if new_value is not None else "n/a"
        policy_delta = policy["delta"][key]
        if policy_delta is None:
            lines.append(f"- {key}: {old_text} -> {new_text}")
        else:
            lines.append(f"- {key}: {old_text} -> {new_text} "
                         f"({policy_delta:+.4f})")
    lines.append("- alerts: " + (", ".join(report["alerts"])
                                 if report["alerts"] else "none"))
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--max-brier-increase", type=float, default=0.02)
    parser.add_argument("--max-ece-increase", type=float, default=0.02)
    parser.add_argument("--max-n-drop", type=float, default=0.20)
    parser.add_argument("--max-refusal-recall-drop", type=float, default=0.02)
    parser.add_argument("--max-refusal-precision-drop", type=float,
                        default=0.02)
    parser.add_argument("--max-coverage-drop", type=float, default=0.02)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = compare_profiles(
            load_profile(args.baseline), load_profile(args.profile),
            max_brier_increase=args.max_brier_increase,
            max_ece_increase=args.max_ece_increase,
            max_n_drop=args.max_n_drop,
            max_refusal_recall_drop=args.max_refusal_recall_drop,
            max_refusal_precision_drop=args.max_refusal_precision_drop,
            max_coverage_drop=args.max_coverage_drop)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"input error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(_render(report))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
