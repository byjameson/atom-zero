"""Calibrated citation validity gate.

The gate maps a rerank score to a citation-validity probability, then rejects
citations below a configurable threshold. The built-in maps come from the
AISC empirical citation calibration; they are starting points, not universal
calibration. Production deployments should recalibrate on their own query
distribution.

Environment:
  ATOMZERO_CITATION_THRESHOLD   float in 0..1 (default 0.5)
  ATOMZERO_CITATION_PROFILE     squad_v2_stress (default) or authored_v1
  ATOMZERO_CITATION_CALIBRATION path to a JSON calibration map
  ATOMZERO_CITATION_GATE        enforce (default), shadow or off

The calibration JSON accepts either ``x_thresholds``/``y_thresholds`` at the
top level or nested under ``isotonic``.
"""
from __future__ import annotations

import json
import math
import os
from bisect import bisect_right
from pathlib import Path

# AISC citation-calibration held-out signal (118 citations, 40 positive).
# x = rerank score thresholds; y = isotonic validity probability.
AUTHORED_V1 = (
    (0.071526, 0.271077, 0.273963, 0.447603, 0.452132,
     0.541905, 0.549168, 0.570138, 0.576199, 0.816038),
    (0.0, 0.0, 0.03225806451612903, 0.03225806451612903,
     0.4166666666666667, 0.4166666666666667, 0.5, 0.5, 1.0, 1.0),
)
# AISC SQuAD v2 stress calibration (100 questions).
SQUAD_V2_STRESS = (
    (0.326, 0.347, 0.348, 0.462, 0.463,
     0.743, 0.749, 0.773, 0.776, 0.808),
    (0.0, 0.0, 0.26666666666666666, 0.26666666666666666,
     0.5428571428571428, 0.5428571428571428, 0.6, 0.6,
     0.6666666666666666, 0.6666666666666666),
)
PROFILES = {
    "authored_v1": AUTHORED_V1,
    "squad_v2_stress": SQUAD_V2_STRESS,
}
DEFAULT_PROFILE = "squad_v2_stress"
DEFAULT_THRESHOLD = 0.5
ENV_THRESHOLD = "ATOMZERO_CITATION_THRESHOLD"
ENV_PROFILE = "ATOMZERO_CITATION_PROFILE"
ENV_CALIBRATION = "ATOMZERO_CITATION_CALIBRATION"
ENV_MODE = "ATOMZERO_CITATION_GATE"


def _parse_threshold(value, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return max(0.0, min(1.0, number))


def _normalize_calibration(data) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if isinstance(data, dict) and "isotonic" in data:
        data = data["isotonic"]
    if not isinstance(data, dict):
        raise ValueError("calibration must be a JSON object")
    x_values = data.get("x_thresholds")
    y_values = data.get("y_thresholds")
    if x_values is None or y_values is None:
        raise ValueError("calibration needs x_thresholds and y_thresholds")
    x_values = tuple(float(value) for value in x_values)
    y_values = tuple(float(value) for value in y_values)
    if len(x_values) != len(y_values) or len(x_values) < 2:
        raise ValueError("calibration thresholds must have equal length >= 2")
    if any(not math.isfinite(value) for value in x_values + y_values):
        raise ValueError("calibration values must be finite")
    if list(x_values) != sorted(x_values):
        raise ValueError("calibration x_thresholds must be sorted")
    return x_values, y_values


def _load_calibration(path: str):
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return _normalize_calibration(data), f"custom:{Path(path).name}"
    except (OSError, ValueError, TypeError):
        return None, ""


class CitationGate:
    """Map rerank scores to probabilities and filter citations."""

    def __init__(self, threshold=None, profile=None, calibration=None,
                 mode=None, env=None):
        env = os.environ if env is None else env
        self.threshold = _parse_threshold(
            threshold if threshold is not None else env.get(ENV_THRESHOLD),
            DEFAULT_THRESHOLD)
        self.mode = str(
            mode if mode is not None else env.get(ENV_MODE, "enforce")
        ).strip().lower()
        if self.mode not in {"enforce", "shadow", "off"}:
            self.mode = "enforce"
        self.enabled = self.mode != "off"

        selected_profile = str(
            profile if profile is not None else env.get(ENV_PROFILE, "")
        ).strip()
        custom_path = env.get(ENV_CALIBRATION, "")
        calibration_data = calibration
        profile_label = ""
        if calibration_data is None and custom_path:
            loaded, profile_label = _load_calibration(custom_path)
            if loaded is not None:
                self.x, self.y = loaded
        if calibration_data is not None:
            try:
                self.x, self.y = _normalize_calibration(calibration_data)
                profile_label = "custom"
            except (ValueError, TypeError):
                calibration_data = None
        if not hasattr(self, "x"):
            name = selected_profile or DEFAULT_PROFILE
            if name not in PROFILES:
                name = DEFAULT_PROFILE
            self.x, self.y = PROFILES[name]
            profile_label = name
        self.profile = profile_label or self.profile_name()
        self.x = list(self.x)
        self.y = list(self.y)

    def profile_name(self) -> str:
        for name, value in PROFILES.items():
            if tuple(self.x) == value[0] and tuple(self.y) == value[1]:
                return name
        return "custom"

    def probability(self, raw_score) -> float:
        """Piecewise-linear isotonic interpolation, clamped to 0..1."""
        try:
            value = float(raw_score)
        except (TypeError, ValueError):
            return 0.0
        if not math.isfinite(value):
            return 0.0
        value = max(0.0, min(1.0, value))
        if value <= self.x[0]:
            return max(0.0, min(1.0, self.y[0]))
        if value >= self.x[-1]:
            return max(0.0, min(1.0, self.y[-1]))
        index = bisect_right(self.x, value) - 1
        left_x, right_x = self.x[index], self.x[index + 1]
        left_y, right_y = self.y[index], self.y[index + 1]
        if right_x <= left_x:
            return max(0.0, min(1.0, right_y))
        ratio = (value - left_x) / (right_x - left_x)
        return max(0.0, min(1.0, left_y + ratio * (right_y - left_y)))

    def filter(self, hits: list[dict]) -> tuple[list[dict], dict]:
        """Return accepted hits and gate metadata.

        If no hit carries a rerank score, the gate is not applied. This keeps
        direct callers that already chose their evidence compatible.
        """
        metadata = {
            "applied": False,
            "mode": self.mode,
            "profile": self.profile,
            "threshold": round(self.threshold, 4),
            "candidates": len(hits),
            "accepted": len(hits),
            "dropped": 0,
            "max_probability": 0.0,
            "reason": "",
        }
        if not hits:
            metadata["reason"] = "no_candidates"
            return list(hits), metadata
        if not self.enabled:
            metadata["reason"] = "disabled"
            return list(hits), metadata
        scores = []
        for hit in hits:
            value = hit.get("rerank_score")
            try:
                score = float(value)
            except (TypeError, ValueError):
                score = None
            if score is None or not math.isfinite(score):
                metadata["reason"] = "no_rerank_scores"
                return list(hits), metadata
            scores.append(score)
        accepted = []
        probabilities = []
        for hit, score in zip(hits, scores):
            probability = self.probability(score)
            probabilities.append(probability)
            item = dict(hit)
            item["citation_probability"] = round(probability, 4)
            item["citation_threshold"] = round(self.threshold, 4)
            if self.mode == "shadow" or probability >= self.threshold:
                accepted.append(item)
        metadata.update({
            "applied": True,
            "accepted": len(accepted),
            "dropped": len(hits) - len(accepted),
            "max_probability": round(max(probabilities), 4),
        })
        if self.mode == "shadow":
            metadata["would_drop"] = sum(
                1 for probability in probabilities
                if probability < self.threshold)
            metadata["shadow"] = True
        if not accepted:
            metadata["reason"] = "no_citation_above_threshold"
        return accepted, metadata


def citation_gate_from_env(env=None) -> CitationGate:
    return CitationGate(env=env)
