import json

from atomzero.citation_gate import CitationGate
from atomzero.doc_memory import grounded_answer


def _hit(score, chunk="The capital of France is Paris.",
         source="law.txt"):
    return {
        "source": source,
        "title": "Notes",
        "chunk": chunk,
        "rerank_score": score,
        "kind": "document",
        "verified": True,
    }


def _identity_gate(threshold=0.5):
    return CitationGate(
        threshold=threshold,
        calibration={"x_thresholds": [0.0, 1.0],
                     "y_thresholds": [0.0, 1.0]})


def test_gate_probability_and_filter():
    gate = _identity_gate()
    assert gate.probability(0.5) == 0.5
    accepted, meta = gate.filter([_hit(0.9), _hit(0.1)])
    assert len(accepted) == 1
    assert accepted[0]["citation_probability"] == 0.9
    assert meta["applied"] is True
    assert meta["accepted"] == 1
    assert meta["dropped"] == 1


def test_grounded_answer_refuses_when_all_citations_are_low():
    result = grounded_answer(
        "capital of France", [_hit(0.1)], gate=_identity_gate())
    assert result["refused"] is True
    assert result["reason"] == "no_citation_above_threshold"
    assert result["sources"] == []
    assert result["evidence"] == []
    assert "Yeterli güvenilir yerel kaynak bulamadım" in result["text"]
    assert result["gate"]["applied"] is True
    assert result["gate"]["dropped"] == 1


def test_grounded_answer_keeps_high_citation():
    result = grounded_answer(
        "capital of France", [_hit(0.9)], gate=_identity_gate())
    assert result["refused"] is False
    assert result["reason"] == ""
    assert result["sources"][0]["source"] == "law.txt"
    assert result["evidence"][0]["citation_probability"] == 0.9
    assert result["gate"]["accepted"] == 1


def test_threshold_and_calibration_from_environment(tmp_path, monkeypatch):
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps({
        "x_thresholds": [0.0, 1.0],
        "y_thresholds": [0.0, 1.0],
    }), encoding="utf-8")
    monkeypatch.setenv("ATOMZERO_CITATION_CALIBRATION",
                       str(calibration))
    monkeypatch.setenv("ATOMZERO_CITATION_THRESHOLD", "0.95")
    refused = grounded_answer("capital of France", [_hit(0.9)])
    assert refused["refused"] is True

    monkeypatch.setenv("ATOMZERO_CITATION_THRESHOLD", "0.5")
    accepted = grounded_answer("capital of France", [_hit(0.9)])
    assert accepted["refused"] is False


def test_gate_off_preserves_legacy_answer():
    gate = CitationGate(
        mode="off",
        calibration={"x_thresholds": [0.0, 1.0],
                     "y_thresholds": [0.0, 1.0]})
    result = grounded_answer("capital of France", [_hit(0.01)], gate=gate)
    assert result["refused"] is False
    assert result["gate"]["applied"] is False
