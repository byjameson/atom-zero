import sys
import types

import pytest

from atomzero.audit import (
    discover_specs,
    resolve_spec_path,
    run_audit,
    summarize_results,
)


def _install_fake_aisc(monkeypatch, load_spec):
    package = types.ModuleType("aisc")
    module = types.ModuleType("aisc.spec")
    module.load_spec = load_spec
    package.spec = module
    monkeypatch.setitem(sys.modules, "aisc", package)
    monkeypatch.setitem(sys.modules, "aisc.spec", module)


def test_discover_specs_includes_atomzero_specs():
    names = {path.name for path in discover_specs()}
    assert {
        "atomzero-read.toml",
        "atomzero-task-actions.toml",
        "atomzero-tool-plan.toml",
        "atomzero-context-budget.toml",
        "atomzero-proactive.toml",
        "atomzero-memory-lifecycle.toml",
        "atomzero-answerability.toml",
        "atomzero-answerability-generated.toml",
        "atomzero-claims.toml",
        "atomzero-consensus.toml",
    } <= names


def test_summarize_results_counts_errors_and_passes():
    summary = summarize_results([
        {"ok": True},
        {"ok": False, "error": "bad spec"},
        {"ok": False},
    ])
    assert summary == {
        "total": 3, "passed": 1, "failed": 2, "errors": 1, "ok": False,
    }
    assert summarize_results([])["ok"] is False


def test_resolve_spec_path_keeps_absolute_paths(tmp_path):
    path = tmp_path / "spec.toml"
    assert resolve_spec_path(path) == path


def test_resolve_spec_path_uses_repo_root_for_relative_paths():
    path = resolve_spec_path("docs/aisc/atomzero-read.toml")
    assert path.is_absolute()
    assert path.name == "atomzero-read.toml"
    assert path.parent.name == "aisc"


def test_run_audit_reports_expected_bad_spec(monkeypatch):
    def bad_load_spec(path):
        raise ValueError("bad spec")

    _install_fake_aisc(monkeypatch, bad_load_spec)
    results = run_audit(["broken.toml"])
    assert results == [{
        "source": str(resolve_spec_path("broken.toml")),
        "ok": False,
        "error": "bad spec",
    }]


def test_run_audit_propagates_unexpected_errors(monkeypatch):
    _install_fake_aisc(monkeypatch, lambda path: object())

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("atomzero.audit.audit_definition", boom)

    with pytest.raises(RuntimeError, match="boom"):
        run_audit(["spec.toml"])
