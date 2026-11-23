import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from atomzero import aisc_coverage
from atomzero.aisc_coverage import (
    component_coverage,
    coverage_report,
    evidence_files,
    spec_coverage,
)


def _install_fake_aisc(monkeypatch, load_spec):
    package = types.ModuleType("aisc")
    module = types.ModuleType("aisc.spec")
    module.load_spec = load_spec
    package.spec = module
    monkeypatch.setitem(sys.modules, "aisc", package)
    monkeypatch.setitem(sys.modules, "aisc.spec", module)


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


def test_coverage_report_records_expected_bad_spec(monkeypatch, tmp_path):
    def bad_load_spec(path):
        raise ValueError("bad spec")

    _install_fake_aisc(monkeypatch, bad_load_spec)
    report = coverage_report(["broken.toml"], project_root=tmp_path)
    assert report["ok"] is False
    assert report["summary"]["errors"] == 1
    assert report["specs"][0]["error"].startswith("ValueError:")


def test_coverage_report_propagates_unexpected_errors(monkeypatch, tmp_path):
    _install_fake_aisc(monkeypatch, lambda path: object())

    def boom(definition, project_root):
        raise RuntimeError("boom")

    monkeypatch.setattr(aisc_coverage, "spec_coverage", boom)
    with pytest.raises(RuntimeError, match="boom"):
        coverage_report(["spec.toml"], project_root=tmp_path)
