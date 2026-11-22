import json

from atomzero.briefing import build_briefing, load_session, main, plan_goal
from atomzero.doc_memory import DocMemory
from atomzero.knowledge_gaps import KnowledgeGapStore


def test_build_briefing_with_tasks_gaps_and_docs(tmp_path):
    docs = DocMemory(tmp_path / "docs.db")
    docs.add_text("notes/launch.txt", "The launch code is L-7781.",
                  title="Launch")
    docs.add_text("notes/warranty.txt",
                  "The warranty period is 24 months.", title="Warranty")
    gaps = KnowledgeGapStore(tmp_path / "gaps.json")
    gaps.add("What is the capital of France?")
    session = {"tasks": [
        {"id": 1, "text": "buy milk", "status": "open"},
        {"id": 2, "text": "old task", "status": "done"},
    ]}
    brief = build_briefing(session, gaps, docs, now="2026-11-22 09:00")
    assert brief["open_task_count"] == 1
    assert brief["pending_gap_count"] == 1
    assert "buy milk" in brief["focus"]
    assert "Daily brief - 2026-11-22 09:00" in brief["text"]
    assert "Index: 2 chunk(s) / 2 source(s)" in brief["text"]
    assert "L-7781" in {node["name"]
                         for node in brief["graph"]["entities"]}
    assert brief["weights_changed"] is False
    docs.close()


def test_plan_goal_adds_cited_context_and_suggestions(tmp_path):
    docs = DocMemory(tmp_path / "docs.db")
    docs.add_text("notes/warranty.txt",
                  "The warranty period is 24 months. Claims need a receipt.",
                  title="Warranty")
    gaps = KnowledgeGapStore(tmp_path / "gaps.json")
    gaps.add("What is the warranty claim deadline?")
    session = {"tasks": [{"id": 3, "text": "publish page", "status": "open"}]}
    plan = plan_goal("warranty page", docs, gaps, session)
    assert plan["goal"] == "warranty page"
    assert plan["context"][0]["source"] == "notes/warranty.txt"
    assert "24 months" in plan["context"][0]["snippet"]
    assert len(plan["suggested_tasks"]) == 3
    assert "warranty" in plan["text"].lower()
    assert plan["weights_changed"] is False
    docs.close()


def test_cli_brief_and_goal(tmp_path, capsys):
    session_path = tmp_path / "session.json"
    session_path.write_text(json.dumps({
        "tasks": [{"id": 1, "text": "buy milk", "status": "open"}]
    }), encoding="utf-8")
    docs = DocMemory(tmp_path / "docs.db")
    docs.add_text("notes/warranty.txt",
                  "The warranty period is 24 months.", title="Warranty")
    docs.close()
    gaps_path = tmp_path / "gaps.json"
    KnowledgeGapStore(gaps_path).add("What is the claim deadline?")
    assert main([
        "--session", str(session_path), "--docs", str(tmp_path / "docs.db"),
        "--gaps", str(gaps_path), "--json"]) == 0
    brief = json.loads(capsys.readouterr().out)
    assert brief["open_task_count"] == 1
    assert main([
        "--session", str(session_path), "--docs", str(tmp_path / "docs.db"),
        "--gaps", str(gaps_path), "--goal", "warranty page",
        "--json"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["goal"] == "warranty page"
    assert plan["context"]


def test_load_session_missing_is_empty(tmp_path):
    assert load_session(tmp_path / "missing.json") == {}
