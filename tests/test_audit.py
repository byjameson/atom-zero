from atomzero.audit import discover_specs, resolve_spec_path


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
    } <= names


def test_resolve_spec_path_keeps_absolute_paths(tmp_path):
    path = tmp_path / "spec.toml"
    assert resolve_spec_path(path) == path


def test_resolve_spec_path_uses_repo_root_for_relative_paths():
    path = resolve_spec_path("docs/aisc/atomzero-read.toml")
    assert path.is_absolute()
    assert path.name == "atomzero-read.toml"
    assert path.parent.name == "aisc"
