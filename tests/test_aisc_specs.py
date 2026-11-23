from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    tomllib = None


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    if tomllib is None:
        pytest.skip("AISC spec test requires Python 3.11+ tomllib")
    return tomllib.loads(
        (ROOT / "docs" / "aisc" / name).read_text(encoding="utf-8"))


def test_aisc_architecture_specs_are_safe_and_verified():
    read = _load("atomzero-read.toml")
    task = _load("atomzero-task-actions.toml")
    plan = _load("atomzero-tool-plan.toml")
    proactive = _load("atomzero-proactive.toml")
    memory = _load("atomzero-memory-lifecycle.toml")
    answerability = _load("atomzero-answerability.toml")
    claims = _load("atomzero-claims.toml")
    consensus = _load("atomzero-consensus.toml")
    doc_diff = _load("atomzero-document-diff.toml")
    freshness = _load("atomzero-freshness.toml")
    reliability = _load("atomzero-reliability.toml")
    temporal = _load("atomzero-temporal.toml")
    improvement = _load("atomzero-improvement.toml")
    scoreboard = _load("atomzero-scoreboard.toml")
    trend = _load("atomzero-trend.toml")

    read_components = read["components"]
    task_components = task["components"]
    plan_components = plan["components"]
    assert [item["name"] for item in read_components] == [
        "LocalRetriever", "GroundedAnswer", "CitationGate", "LocalRenderer"]
    assert [item["name"] for item in task_components] == [
        "TaskParser", "TaskIntentEvaluator", "HumanConfirmation",
        "TaskStore"]
    assert [item["name"] for item in plan_components] == [
        "PlanValidator", "HumanConfirmation", "ToolExecutor"]
    assert all(item["verified"] is True
               for item in read_components + task_components + plan_components)
    assert read["min_assurance"] >= 0.90
    assert task["min_assurance"] >= 0.90
    assert plan["min_assurance"] >= 0.90
    assert answerability["answerability_n"] == 200
    assert answerability["answerability_brier"] >= 0.0
    assert answerability["answerability_ece"] >= 0.0
    assert answerability["min_refusal_precision"] == 0.90
    assert answerability["min_refusal_recall"] == 0.90
    assert answerability["min_answerable_coverage"] == 0.80
    assert answerability["refusal_threshold"] == 0.60
    assert [item["name"] for item in proactive["components"]] == [
        "SignalCollector", "ClaimGapReader", "ImprovementPlanReader",
        "ConsentGate", "ProactivePolicy", "MessageComposer", "UserChannel"]
    assert [item["name"] for item in memory["components"]] == [
        "MemoryRequestParser", "ScopeResolver", "HumanConfirmation",
        "MemoryStore", "LedgerWriter"]
    assert memory["components"][-2]["reversible"] is False
    assert memory["components"][-3]["kind"] == "human"
    assert memory["components"][-1]["reversible"] is True

    assert [item["name"] for item in improvement["components"]] == [
        "SignalCollector", "ImprovementPlanner", "HumanApprovalGate",
        "PlanReport"]
    assert improvement["components"][2]["kind"] == "human"
    assert all(item["verified"] is True
               for item in improvement["components"])

    for spec in (claims, consensus, doc_diff, freshness, reliability, temporal,
                 improvement, scoreboard, trend):
        assert spec["min_assurance"] >= 0.90
        assert all(item["verified"] is True for item in spec["components"])
        assert all(item.get("reversible", True) is True
                   for item in spec["components"])

    for components in (task_components, plan_components):
        irreversible = [index for index, item in enumerate(components)
                        if item.get("reversible") is False]
        assert irreversible
        for index in irreversible:
            assert any(item.get("kind") == "human"
                       for item in components[:index])
