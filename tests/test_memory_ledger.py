import json

import pytest

from atomzero.memory_ledger import (
    MemoryLedger, MemoryLedgerError, _contains_content)


def test_append_read_summary_and_verify(tmp_path):
    ledger = MemoryLedger(tmp_path / "ledger.jsonl")
    first = ledger.append(
        "document_forget", "document", "note.txt",
        before={"chunks": 2}, after={"chunks": 0}, request_id="r1")
    second = ledger.append(
        "session_forget", "session", "web-1",
        before={"facts": 1}, after={"facts": 0}, request_id="r2")
    assert first["id"].startswith("m_")
    assert ledger.verify()["ok"] is True
    assert [event["id"] for event in ledger.read()] == [
        second["id"], first["id"]]
    assert ledger.summary()["by_op"] == {
        "document_forget": 1, "session_forget": 1}
    assert ledger.read(op="session_forget")[0]["target"] == "web-1"


def test_ledger_rejects_content_fields(tmp_path):
    ledger = MemoryLedger(tmp_path / "ledger.jsonl")
    with pytest.raises(MemoryLedgerError):
        ledger.append(
            "document_forget", "document", "note.txt",
            before={"text": "secret"})
    assert not ledger.path.exists()


def test_content_detection_is_recursive():
    assert _contains_content({"meta": {"answer": "x"}}) == "meta.answer"
    assert _contains_content({"ok": {"chunks": 2}}) == ""
    assert _contains_content({"ok": {"count": 2}}) == ""


def test_verify_reports_bad_lines(tmp_path):
    path = tmp_path / "ledger.jsonl"
    path.write_text(
        '{"id":"m_1","ts":"2026-01-01T00:00:00","op":"x",'
        '"scope":"s","target":"t","status":"committed"}\n'
        'not-json\n', encoding="utf-8")
    result = MemoryLedger(path).verify()
    assert result["ok"] is False
    assert "invalid JSON" in result["errors"][0]
