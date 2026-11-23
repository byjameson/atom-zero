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
    answerability = _load("atomzero-answerability.toml")

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

    for components in (task_components, plan_components):
        irreversible = [index for index, item in enumerate(components)
                        if item.get("reversible") is False]
        assert irreversible
        for index in irreversible:
            assert any(item.get("kind") == "human"
                       for item in components[:index])
