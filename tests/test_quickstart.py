import pytest

from atomzero import __version__
from atomzero import quickstart
from atomzero.quickstart import main as quickstart_main
from atomzero.quickstart import reset_demo


def test_reset_demo_removes_db(tmp_path):
    db = tmp_path / "demo.db"
    db.write_text("old", encoding="utf-8")
    reset_demo(db)
    assert not db.exists()
    reset_demo(db)  # missing is fine


def test_reset_demo_missing_is_noop(tmp_path):
    reset_demo(tmp_path / "missing.db")
    assert not (tmp_path / "missing.db").exists()


def test_version_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        quickstart_main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_quickstart_passes_teacher_flags(monkeypatch, tmp_path):
    monkeypatch.setattr(
        quickstart, "ensure_assets",
        lambda cache, download=True: {
            "chat_en_bpe8k.json": str(tmp_path / "tokenizer.json")})
    monkeypatch.setattr(quickstart, "seed_demo", lambda path: 0)
    captured = {}

    def fake_serve(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return 0

    monkeypatch.setattr(quickstart, "serve", fake_serve)
    code = quickstart_main([
        "--no-download", "--no-browser", "--teacher",
        "--docs", str(tmp_path / "docs.db"),
        "--cache", str(tmp_path / "cache"),
        "--reports-index", str(tmp_path / "reports.json"),
        "--teacher-url", "http://127.0.0.1:9999/v1",
        "--teacher-model", "fake"])
    assert code == 0
    assert captured["kwargs"]["teacher_url"] == "http://127.0.0.1:9999/v1"
    assert captured["kwargs"]["teacher_model"] == "fake"
    assert captured["kwargs"]["teacher_label"] == "ollama"
    assert captured["kwargs"]["weekly_reports_path"] == str(
        tmp_path / "reports.json")
