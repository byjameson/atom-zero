import json

from atomzero.briefing import main
from atomzero.chat_human import ChatHuman
from atomzero.doc_memory import DocMemory


BOM = b"\xef\xbb\xbf"


def _write_bom_json(path, payload) -> None:
    path.write_bytes(
        BOM + json.dumps(payload, ensure_ascii=False).encode("utf-8"))


def test_cli_weekly_review_reads_bom_session(tmp_path, capsys):
    session_path = tmp_path / "session.json"
    now = "2026-11-22T09:00:00"
    _write_bom_json(session_path, {"tasks": [
        {"id": 1, "text": "raporu yaz", "status": "open",
         "created": now, "completed": ""},
    ]})
    docs = DocMemory(tmp_path / "docs.db")
    docs.add_text("notes/launch.txt", "The launch code is L-7781.")
    docs.close()
    assert main([
        "--session", str(session_path),
        "--docs", str(tmp_path / "docs.db"),
        "--weekly", "--markdown"]) == 0
    markdown = capsys.readouterr().out
    assert markdown.startswith("# Weekly review - ")
    assert "- [ ] **#1** raporu yaz" in markdown


def test_chat_human_reads_bom_session(tmp_path):
    memory = tmp_path / "memory.json"
    _write_bom_json(memory, {
        "persona": {"name": "Tiny", "style": "warm"},
        "facts": [{"key": "name", "value": "Alex"}],
        "episodes": [],
        "tasks": [{"id": 4, "text": "raporu yaz", "status": "open"}],
        "next_task": 5,
    })
    human = ChatHuman(memory)
    assert human.data["facts"][0]["value"] == "Alex"
    assert human.data["tasks"][0]["text"] == "raporu yaz"
    assert "raporu yaz" in human.respond("list tasks")


def test_research_chat_human_reads_bom_session(tmp_path):
    from research.chat_human import ChatHuman as ResearchChatHuman

    memory = tmp_path / "memory.json"
    _write_bom_json(memory, {
        "persona": {"name": "Tiny", "style": "warm"},
        "facts": [],
        "episodes": [],
        "tasks": [{"id": 6, "text": "raporu yaz", "status": "open"}],
        "next_task": 7,
    })
    human = ResearchChatHuman(memory)
    assert human.data["tasks"][0]["text"] == "raporu yaz"
