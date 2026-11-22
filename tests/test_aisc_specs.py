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

    read_components = read["components"]
    task_components = task["components"]
    assert [item["name"] for item in read_components] == [
        "LocalRetriever", "GroundedAnswer", "CitationGate", "LocalRenderer"]
    assert [item["name"] for item in task_components] == [
        "TaskParser", "TaskIntentEvaluator", "HumanConfirmation",
        "TaskStore"]
    assert all(item["verified"] is True
               for item in read_components + task_components)
    assert read["min_assurance"] >= 0.90
    assert task["min_assurance"] >= 0.90

    irreversible = [index for index, item in enumerate(task_components)
                    if item.get("reversible") is False]
    assert irreversible
    for index in irreversible:
        assert any(item.get("kind") == "human"
                   for item in task_components[:index])
