from pathlib import Path
from types import SimpleNamespace

from atomzero.aisc_coverage import (
    component_coverage,
    evidence_files,
    spec_coverage,
)


def _component(name="Reader", evidence="tests/test_demo.py", verified=True,
               calibration_n=0, reversible=True):
    contract = SimpleNamespace(
        evidence=evidence, verified=verified, calibration_n=calibration_n)
    return SimpleNamespace(
        name=name, kind="model", contract=contract, reversible=reversible)


def test_evidence_files_extracts_and_dedupes():
    evidence = (
        "tests/test_a.py numeric case; tests\\test_b.py; "
        "tests/test_a.py again; docs/README.md")
    assert evidence_files(evidence) == ["tests/test_a.py", "tests/test_b.py"]


def test_component_coverage_missing_and_present_files(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_demo.py").write_text("x", encoding="utf-8")
    present = component_coverage(_component(), tmp_path)
    assert present["verified"] is True
    assert present["missing_evidence_files"] == []
    missing = component_coverage(
        _component(name="Other", evidence="tests/test_missing.py"), tmp_path)
    assert missing["missing_evidence_files"] == ["tests/test_missing.py"]


def test_spec_coverage_summarizes_components(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_demo.py").write_text("x", encoding="utf-8")
    components = (
        _component(),
        _component(name="Store", evidence="", verified=False,
                   reversible=False),
    )
    composition = SimpleNamespace(components=components)
    definition = SimpleNamespace(
        source="spec.toml", name="Demo", version="0.1",
        composition=composition,
        to_dict=lambda: {"topology": "sequential", "context_tokens": 42},
    )
    report = spec_coverage(definition, tmp_path)
    assert report["ok"] is False
    assert report["summary"]["components"] == 2
    assert report["summary"]["verified"] == 1
    assert report["summary"]["with_evidence"] == 1
    assert report["summary"]["unverified_components"] == ["Store"]
    assert report["summary"]["irreversible_components"] == ["Store"]


def test_component_coverage_empty_evidence():
    entry = component_coverage(_component(evidence=""), Path("."))
    assert entry["has_evidence"] is False
    assert entry["evidence_files"] == []
