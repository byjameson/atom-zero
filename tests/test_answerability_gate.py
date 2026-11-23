import json

import pytest

from atomzero.answerability import (
    FEATURE_NAMES, AnswerabilityGate, extract_features, load_profile)
from atomzero.citation_gate import CitationGate
from atomzero.doc_memory import grounded_answer


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


def _profile(tmp_path, intercept=-3.0, threshold=0.5):
    path = tmp_path / "answerability.json"
    path.write_text(json.dumps({
        "version": 1,
        "features": list(FEATURE_NAMES),
        "coefficients": [0.0] * len(FEATURE_NAMES),
        "intercept": intercept,
        "calibration": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
        "threshold": threshold,
    }), encoding="utf-8")
    return path


def test_feature_extraction_is_complete_and_finite():
    features = extract_features(
        "what is the launch code?",
        [_hit()])
    assert set(features) == set(FEATURE_NAMES)
    assert all(isinstance(value, float) and value == value
               for value in features.values())


def test_gate_probability_applies_isotonic_calibration():
    profile = {
        "features": list(FEATURE_NAMES),
        "coefficients": [0.0] * len(FEATURE_NAMES),
        "intercept": 0.0,
        "calibration": {"x": [0.0, 0.5, 1.0], "y": [0.0, 0.25, 1.0]},
        "threshold": 0.5,
    }
    gate = AnswerabilityGate(profile=profile, mode="shadow")
    assert gate.probability("what is the launch code?", [_hit()]) == (
        pytest.approx(0.25, abs=1e-6))


def test_gate_enforce_refuses_below_threshold(tmp_path, monkeypatch):
    monkeypatch.setenv("ATOMZERO_ANSWERABILITY_GATE", "enforce")
    monkeypatch.setenv("ATOMZERO_ANSWERABILITY_PROFILE",
                       str(_profile(tmp_path)))
    result = grounded_answer(
        "what is the launch code?", [_hit()], gate=CitationGate(mode="off"))
    assert result["refused"] is True
    assert result["reason"] == "no_answerability"
    assert result["answerability"]["probability"] < 0.5


def test_gate_shadow_keeps_answer_and_adds_metadata(tmp_path, monkeypatch):
    monkeypatch.setenv("ATOMZERO_ANSWERABILITY_GATE", "shadow")
    monkeypatch.setenv("ATOMZERO_ANSWERABILITY_PROFILE",
                       str(_profile(tmp_path)))
    result = grounded_answer(
        "what is the launch code?", [_hit()], gate=CitationGate(mode="off"))
    assert result["refused"] is False
    assert result["answerability"]["mode"] == "shadow"
    assert result["answerability"]["refused"] is False
    assert "L-7781" in result["text"]


def test_gate_off_keeps_legacy_response(tmp_path, monkeypatch):
    monkeypatch.setenv("ATOMZERO_ANSWERABILITY_GATE", "off")
    monkeypatch.setenv("ATOMZERO_ANSWERABILITY_PROFILE",
                       str(_profile(tmp_path)))
    result = grounded_answer(
        "what is the launch code?", [_hit()], gate=CitationGate(mode="off"))
    assert "answerability" not in result


def test_profile_feature_mismatch_is_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({
        "features": ["wrong"],
        "coefficients": [1.0],
        "intercept": 0.0,
    }), encoding="utf-8")
    with pytest.raises(ValueError):
        load_profile(path)
