"""Independent answerability signal for document answers.

The citation gate answers "is this citation valid?"; this module answers a
different question: "does the retrieved context contain an answer to this
question?" It is deliberately separate from :mod:`atomzero.citation_gate` and
can be enabled independently.

The default product path does not load a profile and does not change any
answer/refuse decision. A profile is produced by
``benchmarks/answerability_signal.py`` and can be enabled in shadow or enforce
mode:

  ATOMZERO_ANSWERABILITY_GATE      off (default), shadow or enforce
  ATOMZERO_ANSWERABILITY_PROFILE   path to the calibrated JSON profile
  ATOMZERO_ANSWERABILITY_THRESHOLD optional threshold override

The signal uses local retrieval features (score distribution, query coverage,
question type, answer-type candidates, entity/number alignment). It does not
call a model or an external service. The profile contains a fitted logistic
model and an isotonic calibration map.
"""
from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path

ENV_MODE = "ATOMZERO_ANSWERABILITY_GATE"
ENV_PROFILE = "ATOMZERO_ANSWERABILITY_PROFILE"
ENV_THRESHOLD = "ATOMZERO_ANSWERABILITY_THRESHOLD"
DEFAULT_MODE = "off"

WH_WORDS = {
    "who", "whom", "whose", "what", "when", "where", "which", "why", "how",
    "is", "are", "was", "were", "do", "does", "did", "can", "could", "will",
    "would", "has", "have", "had", "should", "may", "might", "the", "a", "an",
    "many", "much", "long", "old", "far", "year", "date", "time", "place",
    "name", "number", "percentage", "type", "kind",
}
NAME_STOP = {
    "The", "A", "An", "In", "On", "At", "After", "Before", "During", "When",
    "Who", "What", "Where", "Why", "How", "It", "He", "She", "They", "This",
    "That", "These", "Those", "However", "Although", "Because", "While",
    "There", "Here", "His", "Her", "Their", "Its", "Our", "Your", "My",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday",
    "Sunday", "January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December",
}

DATE_RE = re.compile(
    r"\b(?:1[0-9]{3}|20[0-9]{2})\b|"
    r"\b(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s+\d{1,2}(?:,\s*\d{4})?|"
    r"\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\s+\d{4}\b",
    re.IGNORECASE)
NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?\b")
PLACE_RE = re.compile(
    r"\b(?:in|at|on|near|from|inside|outside)\s+(?:the\s+)?"
    r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*")
PERSON_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b")

FEATURE_NAMES = (
    "top_score", "second_score", "score_margin", "score_std",
    "score_entropy", "above_05", "above_07", "n_hits",
    "top_coverage", "top_focus_coverage", "top_title_coverage",
    "top_embedding", "snippet_len", "context_len", "question_len",
    "terms", "focus_terms", "context_coverage", "context_focus_coverage",
    "snippet_coverage", "snippet_focus_coverage", "sentence_count",
    "avg_sentence_len", "who", "when", "where", "amount", "yesno", "why",
    "how", "novel_person", "novel_date", "novel_number", "novel_place",
    "type_answer", "answer_candidate_count", "answer_candidate_density",
    "question_entity_coverage", "context_entity_novelty",
    "number_alignment", "year_alignment", "retrieval_confidence",
)


def _names(text: str) -> set[str]:
    found = {match.group(0) for match in PERSON_RE.finditer(str(text or ""))}
    return {name for name in found if name.split()[0] not in NAME_STOP}


def _numbers(text: str) -> set[str]:
    return set(NUMBER_RE.findall(str(text or "")))


def _dates(text: str) -> set[str]:
    return {match.group(0).lower() for match in DATE_RE.finditer(str(text or ""))}


def _places(text: str) -> set[str]:
    return {match.group(0).lower() for match in PLACE_RE.finditer(str(text or ""))}


def _coverage(terms: set[str], text: str) -> float:
    if not terms:
        return 0.0
    value = str(text or "").lower()
    return sum(1 for term in terms if term in value) / len(terms)


def _score_entropy(scores: list[float]) -> float:
    values = [max(0.0, float(score)) for score in scores]
    total = sum(values)
    if total <= 0 or len(values) < 2:
        return 0.0
    probabilities = [value / total for value in values]
    entropy = -sum(p * math.log(p + 1e-12) for p in probabilities)
    return max(0.0, min(1.0, entropy / math.log(len(values))))


def extract_features(query: str, hits: list[dict]) -> dict:
    """Return the calibrated-model feature dictionary for one question.

    Only the query and retrieved hits are used. This matches the production
    call site and keeps the signal independent from the citation gate.
    """
    from atomzero.doc_memory import (
        _query_terms, _term_coverage, best_snippet, retrieval_confidence)

    if not hits:
        return {name: 0.0 for name in FEATURE_NAMES}
    top = hits[0]
    chunk = str(top.get("chunk", ""))
    snippet = best_snippet(query, chunk)
    context_text = " ".join(str(hit.get("chunk", "")) for hit in hits[:4])
    terms = _query_terms(query)
    focus = {term for term in terms if term not in WH_WORDS}
    low = str(query or "").lower()
    scores = []
    for hit in hits:
        try:
            scores.append(float(hit.get("rerank_score") or 0.0))
        except (TypeError, ValueError):
            scores.append(0.0)
    top_score = scores[0] if scores else 0.0
    second_score = scores[1] if len(scores) > 1 else 0.0
    margin = max(0.0, top_score - second_score)
    mean = sum(scores) / len(scores) if scores else 0.0
    variance = (sum((score - mean) ** 2 for score in scores)
                / len(scores)) if scores else 0.0
    std = math.sqrt(max(0.0, variance))
    sentences = [part.strip() for part in
                 re.split(r"(?<=[.!?])\s+|\n+", snippet) if part.strip()]
    sentence_coverages = [_coverage(terms, sentence)
                          for sentence in sentences] or [0.0]
    sentence_focus_coverages = [_coverage(focus, sentence)
                                for sentence in sentences] or [0.0]
    question_names = _names(query)
    snippet_names = _names(snippet)
    context_names = _names(context_text)
    question_years = {year for year in _dates(query)}
    snippet_years = _dates(snippet)
    context_years = _dates(context_text)
    question_numbers = _numbers(query)
    snippet_numbers = _numbers(snippet)
    context_numbers = _numbers(context_text)
    question_places = _places(query)
    snippet_places = _places(snippet)
    context_places = _places(context_text)
    who = bool(re.search(r"\b(who|whom|whose)\b", low))
    when = bool(re.search(
        r"\b(when|what year|what date|what time|which year)\b", low))
    where = bool(re.search(r"\b(where|which place|what place)\b", low))
    amount = bool(re.search(
        r"\b(how many|how much|how long|how old|how far|percentage)\b", low))
    yesno = bool(re.match(
        r"^(is|are|was|were|do|does|did|can|could|will|would|has|have|had|"
        r"should|may|might)\b", low.strip()))
    why = bool(re.search(r"\bwhy\b", low))
    how = bool(re.search(r"\bhow\b", low))
    novel_person = len(snippet_names - question_names)
    novel_date = len(snippet_years - question_years)
    novel_number = len(snippet_numbers - question_numbers)
    novel_place = len(snippet_places - question_places)
    candidate_count = novel_person + novel_date + novel_number + novel_place
    type_answer = bool(
        (who and novel_person)
        or (when and novel_date)
        or (amount and novel_number)
        or (where and novel_place)
    )
    confidence = retrieval_confidence(query, hits)
    context_name_total = max(1, len(context_names))
    features = {
        "top_score": top_score,
        "second_score": second_score,
        "score_margin": margin,
        "score_std": std,
        "score_entropy": _score_entropy(scores),
        "above_05": sum(1 for score in scores if score >= 0.5),
        "above_07": sum(1 for score in scores if score >= 0.7),
        "n_hits": len(hits),
        "top_coverage": _term_coverage(terms, chunk),
        "top_focus_coverage": _term_coverage(focus, chunk),
        "top_title_coverage": _term_coverage(
            terms, f"{top.get('title', '')} {top.get('section', '')}"),
        "top_embedding": float(top.get("embedding_score") or 0.0),
        "snippet_len": len(snippet),
        "context_len": len(context_text),
        "question_len": len(query),
        "terms": len(terms),
        "focus_terms": len(focus),
        "context_coverage": _coverage(terms, context_text),
        "context_focus_coverage": _coverage(focus, context_text),
        "snippet_coverage": _coverage(terms, snippet),
        "snippet_focus_coverage": _coverage(focus, snippet),
        "sentence_count": len(sentences),
        "avg_sentence_len": (sum(len(sentence) for sentence in sentences)
                             / max(1, len(sentences))),
        "who": who,
        "when": when,
        "where": where,
        "amount": amount,
        "yesno": yesno,
        "why": why,
        "how": how,
        "novel_person": novel_person,
        "novel_date": novel_date,
        "novel_number": novel_number,
        "novel_place": novel_place,
        "type_answer": type_answer,
        "answer_candidate_count": candidate_count,
        "answer_candidate_density": candidate_count / max(
            1.0, len(snippet) / 100.0),
        "question_entity_coverage": (
            len(question_names & context_names) / len(question_names)
            if question_names else 1.0),
        "context_entity_novelty": (
            len(context_names - question_names) / context_name_total),
        "number_alignment": (
            len(question_numbers & context_numbers) / len(question_numbers)
            if question_numbers else 1.0),
        "year_alignment": (
            len(question_years & context_years) / len(question_years)
            if question_years else 1.0),
        "retrieval_confidence": float(confidence.get("score", 0.0)),
    }
    return {name: float(features[name]) for name in FEATURE_NAMES}


def vectorize(features: dict) -> list[float]:
    return [float(features.get(name, 0.0)) for name in FEATURE_NAMES]


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp = math.exp(value)
    return exp / (1.0 + exp)


def _interpolate(x_values: list[float], y_values: list[float],
                 value: float) -> float:
    if not x_values:
        return max(0.0, min(1.0, value))
    value = max(0.0, min(1.0, value))
    if value <= x_values[0]:
        return max(0.0, min(1.0, y_values[0]))
    if value >= x_values[-1]:
        return max(0.0, min(1.0, y_values[-1]))
    low, high = 0, len(x_values) - 1
    while high - low > 1:
        middle = (low + high) // 2
        if x_values[middle] <= value:
            low = middle
        else:
            high = middle
    left_x, right_x = x_values[low], x_values[high]
    if right_x <= left_x:
        return max(0.0, min(1.0, y_values[high]))
    ratio = (value - left_x) / (right_x - left_x)
    return max(0.0, min(1.0,
                        y_values[low] + ratio * (y_values[high] - y_values[low])))


def load_profile(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("answerability profile must be an object")
    if data.get("features") != list(FEATURE_NAMES):
        raise ValueError("answerability profile feature list mismatch")
    coefficients = data.get("coefficients")
    if not isinstance(coefficients, list) or len(coefficients) != len(FEATURE_NAMES):
        raise ValueError("answerability profile coefficients mismatch")
    data["coefficients"] = [float(value) for value in coefficients]
    data["intercept"] = float(data.get("intercept", 0.0))
    calibration = data.get("calibration") or {}
    x_values = calibration.get("x") or []
    y_values = calibration.get("y") or []
    if len(x_values) != len(y_values):
        raise ValueError("answerability calibration length mismatch")
    if x_values:
        x_values = [float(value) for value in x_values]
        y_values = [float(value) for value in y_values]
        if x_values != sorted(x_values):
            raise ValueError("answerability calibration x must be sorted")
    data["calibration"] = {"x": x_values, "y": y_values}
    return data


class AnswerabilityGate:
    """Apply an optional calibrated answerability profile."""

    def __init__(self, profile=None, threshold=None, mode=None, env=None):
        env = os.environ if env is None else env
        self.mode = str(
            mode if mode is not None else env.get(ENV_MODE, DEFAULT_MODE)
        ).strip().lower()
        if self.mode not in {"off", "shadow", "enforce"}:
            self.mode = DEFAULT_MODE
        self.profile = None
        self.reason = ""
        self.enabled = False
        self.threshold = 0.5
        if self.mode == "off":
            self.reason = "disabled"
            return
        source = profile if profile is not None else env.get(ENV_PROFILE, "")
        if not source:
            self.reason = "no_profile"
            return
        try:
            self.profile = load_profile(source) if isinstance(
                source, (str, Path)) else source
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            self.reason = f"invalid_profile: {exc}"
            return
        self.enabled = True
        value = threshold if threshold is not None else env.get(ENV_THRESHOLD)
        try:
            self.threshold = float(value)
        except (TypeError, ValueError):
            self.threshold = float(self.profile.get("threshold", 0.5))
        self.threshold = max(0.0, min(1.0, self.threshold))
        self.reason = "ready"

    def probability(self, query: str, hits: list[dict]) -> float:
        if not self.enabled or self.profile is None:
            return 0.0
        features = extract_features(query, hits)
        raw = self.profile["intercept"] + sum(
            coefficient * features[name]
            for coefficient, name in zip(
                self.profile["coefficients"], FEATURE_NAMES))
        probability = _sigmoid(raw)
        calibration = self.profile.get("calibration") or {}
        if calibration.get("x"):
            probability = _interpolate(
                calibration["x"], calibration["y"], probability)
        return max(0.0, min(1.0, probability))

    def evaluate(self, query: str, hits: list[dict]) -> dict:
        probability = self.probability(query, hits)
        return {
            "applied": self.enabled,
            "mode": self.mode,
            "probability": round(probability, 4),
            "threshold": round(self.threshold, 4),
            "refused": (self.mode == "enforce"
                        and probability < self.threshold),
            "reason": ("below_threshold"
                       if probability < self.threshold else "above_threshold"),
            "profile": self.reason,
        }


def answerability_gate_from_env(env=None) -> AnswerabilityGate | None:
    gate = AnswerabilityGate(env=env)
    if not gate.enabled:
        return None
    return gate
