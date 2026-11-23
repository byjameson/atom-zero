import socket
from pathlib import Path

from atomzero import doctor
from atomzero.doctor import (
    check_aisc, check_cache, check_docs_index, check_port, render)
from atomzero.doc_memory import DocMemory


def test_cache_exact_sizes(tmp_path: Path):
    (tmp_path / "model.pt").write_bytes(b"x" * 10)
    (tmp_path / "config.json").write_bytes(b"{}")
    checks = check_cache(tmp_path, {"model.pt": 10, "config.json": 2})
    assert [check.status for check in checks] == ["ok", "ok"]


def test_cache_missing_and_wrong_size(tmp_path: Path):
    (tmp_path / "model.pt").write_bytes(b"x" * 3)
    checks = check_cache(tmp_path, {"model.pt": 10, "config.json": 2})
    assert checks[0].status == "fail"
    assert "expected 10" in checks[0].detail
    assert checks[1].status == "fail"
    assert "missing" in checks[1].detail


def test_free_port_is_ok():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    assert check_port("127.0.0.1", port).status == "ok"


def test_docs_index_absent_is_ok(tmp_path):
    check = check_docs_index(tmp_path / "missing.db")
    assert check.status == "ok"
    assert "no index yet" in check.detail


def test_docs_index_clean_and_orphan(tmp_path):
    db = tmp_path / "docs.db"
    memory = DocMemory(db)
    memory.add_text("a.txt", "The launch code is L-7781.")
    memory.close()
    clean = check_docs_index(db)
    assert clean.status == "ok"
    assert "1 chunks, 1 sources" in clean.detail

    memory = DocMemory(db)
    memory.con.execute(
        "INSERT INTO doc_vectors(doc_rowid, model, dim, vec) "
        "VALUES (9999, 'x', 1, ?)", (b"\x00",))
    memory.con.commit()
    memory.close()
    broken = check_docs_index(db)
    assert broken.status == "fail"
    assert "orphan vector" in broken.detail


def test_render_summarizes():
    checks = [
        check_cache(Path("does-not-exist"), {"model.pt": 10})[0],
        check_port("127.0.0.1", 0),
    ]
    text = render(checks)
    assert "FAIL" in text
    assert "summary: 1 fail" in text


def test_aisc_not_configured_is_a_warning():
    check = check_aisc("")
    assert check.status == "warn"
    assert "not configured" in check.detail


def test_aisc_strict_warning_fails(monkeypatch):
    monkeypatch.setattr(
        "atomzero.audit.run_audit",
        lambda *args, **kwargs: [{
            "ok": False, "critical_count": 0, "warning_count": 2,
        }])
    strict = check_aisc("fake", strict=True)
    assert strict.status == "fail"
    assert "warnings=2" in strict.detail
    relaxed = check_aisc("fake", strict=False)
    assert relaxed.status == "warn"


def test_doctor_run_checks_includes_aisc_only_when_requested(
        tmp_path, monkeypatch):
    monkeypatch.setattr(
        "atomzero.audit.run_audit",
        lambda *args, **kwargs: [{
            "ok": True, "critical_count": 0, "warning_count": 0,
        }])
    args = doctor.argparse.Namespace(
        cache=str(tmp_path), docs=str(tmp_path / "docs.db"),
        sessions=str(tmp_path / "sessions"), host="127.0.0.1", port=0,
        provider=False, provider_url="", aisc=True, aisc_path="fake",
        aisc_strict=True)
    checks = doctor.run_checks(args)
    assert any(check.name == "aisc" and check.status == "ok"
               for check in checks)
