"""Unit tests for the content-free seed summary."""
import pytest

from research.seed_summary import summarize


def _report(name, *, single, multi, copy, retention):
    return {
        "method": "product_suite_v1",
        "status": "pass",
        "product_ok": True,
        "checkpoints": {"candidate": name},
        "single_turn_useful_rate": single,
        "multi_turn_useful_rate": multi,
        "multi_turn_both_turns_useful_rate": multi / 2,
        "copy_exact": copy,
        "content_loss_ratio": retention,
        "safety_pass_rate": 1.0,
        "rag_pass_rate": 1.0,
        "benign_useful_rate": single - 0.05,
    }


def test_seed_summary_reports_mean_min_max():
    summary = summarize([
        _report("s1", single=0.8, multi=0.4, copy=0.48, retention=0.70),
        _report("s2", single=0.9, multi=0.6, copy=0.50, retention=0.90),
    ])
    single = summary["metrics"]["single_turn_useful_rate"]
    assert single == {"mean": 0.85, "min": 0.8, "max": 0.9, "n": 2}
    retention = summary["metrics"]["content_loss_ratio"]
    assert retention["mean"] == pytest.approx(0.8)
    assert retention["min"] == 0.70
    assert retention["max"] == 0.90
    assert summary["runs"][0]["run"] == "s1"
    assert summary["runs"][1]["run"] == "s2"


def test_seed_summary_handles_missing_metrics_and_gate_fallback():
    report = _report("s1", single=0.9, multi=0.5, copy=0.48, retention=0.70)
    report.pop("rag_pass_rate")
    report.pop("benign_gate_rate", None)
    summary = summarize([report])
    assert summary["metrics"]["rag_pass_rate"] is None
    assert summary["metrics"]["benign_gate_rate"]["mean"] == 0.9


def test_seed_summary_rejects_unknown_method():
    with pytest.raises(ValueError):
        summarize([{"method": "something_else"}])
