import json

from atomzero.self_code import audit_report, audit_tree, main, propose_patch


def _write(root, relative, text):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_audit_finds_code_boundaries_without_string_false_positives(tmp_path):
    _write(tmp_path, "src/pkg/mod.py", '''\
import subprocess


def risky(expression):
    """Never use eval() here."""
    subprocess.run(["echo", expression], shell=True)
    return eval(expression)


def broad():
    try:
        return 1 / 0
    except Exception:
        return None
# TODO: narrow this
''')
    _write(tmp_path, "tests/test_mod.py", '''\
from datetime import datetime


def test_now():
    assert datetime.now().year > 2000
''')
    findings = audit_tree(tmp_path)
    kinds = [item["kind"] for item in findings]
    assert "shell_true" in kinds
    assert "dangerous_builtin" in kinds
    assert "broad_exception" in kinds
    assert "todo_marker" in kinds
    assert "wall_clock_test" in kinds
    # The parser sees real calls, so model.eval() and strings are ignored.
    _write(tmp_path, "src/pkg/torch_like.py", '''\
class Model:
    def eval(self):
        return self

Model().eval()
TEXT = "shell=True and eval( are just words"
''')
    assert "shell_true" not in [
        item["kind"] for item in audit_tree(tmp_path)
        if item["path"] == "src/pkg/torch_like.py"]


def test_audit_is_deterministic_and_bounded(tmp_path):
    body = "\n".join(
        f"def f{index}():\n    try:\n        return {index}\n"
        "    except Exception:\n        return None\n"
        for index in range(5))
    _write(tmp_path, "src/pkg/mod.py", body)
    first = audit_tree(tmp_path, max_per_kind=2)
    second = audit_tree(tmp_path, max_per_kind=2)
    assert first == second
    assert len(first) == 2
    assert len({item["finding_id"] for item in first}) == 2


def test_proposals_are_bounded_and_human_gated(tmp_path):
    _write(tmp_path, "src/pkg/mod.py", '''\
def broad():
    try:
        return 1
    except Exception:
        return None
''')
    findings = audit_tree(tmp_path)
    medium = [item for item in findings if item["kind"] == "broad_exception"][0]
    proposal = propose_patch(medium)
    assert proposal["allowed_files"] == [
        "src/pkg/mod.py", "tests/test_mod.py"]
    assert proposal["checks"] == ["pytest:tests/test_mod.py"]
    assert proposal["requires_human"] is True
    assert proposal["auto_apply"] is False
    assert proposal["weights_changed"] is False
    assert proposal["risk"] == "medium"

    high = dict(medium, kind="dangerous_builtin", severity="high")
    high_proposal = propose_patch(high)
    assert high_proposal["risk"] == "high"
    assert high_proposal["checks"] == []
    assert high_proposal["requires_human"] is True


def test_audit_report_summary_and_json(tmp_path, capsys):
    _write(tmp_path, "src/pkg/mod.py", "def f():\n    return 1\n# FIXME later\n")
    report = audit_report(tmp_path, propose=True)
    assert report["summary"]["findings"] == 1
    assert report["summary"]["proposals"] == 1
    assert report["counts"] == {"todo_marker": 1}
    assert report["auto_apply"] is False
    assert report["weights_changed"] is False

    assert main(["--root", str(tmp_path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["counts"] == {"todo_marker": 1}
