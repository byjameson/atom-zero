import json
import time
from datetime import datetime, timedelta

from atomzero.briefing import (
    build_briefing, build_weekly_review, load_session, main, plan_goal)
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


def test_weekly_review_aggregates_tasks_sources_gaps_and_graph(tmp_path):
    docs = DocMemory(tmp_path / "docs.db")
    docs.add_text("notes/launch.txt", "The launch code is L-7781.",
                  title="Launch")
    docs.add_text("notes/warranty.txt", "The warranty period is 24 months.",
                  title="Warranty")
    docs.add_text("old.txt", "An old note.", title="Old")
    docs.con.execute(
        "UPDATE doc_meta SET created = ? WHERE source = 'old.txt'",
        (str(int(time.time()) - 10 * 86400),))
    docs.con.commit()
    gaps = KnowledgeGapStore(tmp_path / "gaps.json")
    gaps.add("What is the warranty claim deadline?")
    now = datetime.now()
    session = {"tasks": [
        {"id": 1, "text": "open task", "status": "open",
         "created": now.strftime("%Y-%m-%dT%H:%M:%S"), "completed": ""},
        {"id": 2, "text": "recent task", "status": "done",
         "created": now.strftime("%Y-%m-%dT%H:%M:%S"),
         "completed": now.strftime("%Y-%m-%dT%H:%M:%S")},
        {"id": 3, "text": "old task", "status": "done",
         "created": (now - timedelta(days=10)).strftime("%Y-%m-%dT%H:%M:%S"),
         "completed": (now - timedelta(days=10)).strftime("%Y-%m-%dT%H:%M:%S")},
    ]}
    review = build_weekly_review(
        session, gaps, docs, days=7, now="2026-11-22 09:00")
    assert review["task_counts"] == {
        "open": 1, "completed_recent": 1, "completed_total": 2}
    assert review["pending_gap_count"] == 1
    assert review["recent_source_count"] == 2
    assert "L-7781" in {node["name"]
                         for node in review["graph"]["entities"]}
    assert "Weekly review - 2026-11-22 09:00 (last 7 days)" in review["text"]
    assert "Completed this week" in review["text"]
    assert "Recent sources: 2" in review["text"]
    assert review["next_week"]
    assert review["weights_changed"] is False
    docs.close()


def test_cli_weekly_review(tmp_path, capsys):
    session_path = tmp_path / "session.json"
    now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    session_path.write_text(json.dumps({"tasks": [
        {"id": 1, "text": "open task", "status": "open",
         "created": now, "completed": ""},
        {"id": 2, "text": "done task", "status": "done",
         "created": now, "completed": now},
    ]}), encoding="utf-8")
    docs = DocMemory(tmp_path / "docs.db")
    docs.add_text("notes/launch.txt", "The launch code is L-7781.")
    docs.close()
    gaps_path = tmp_path / "gaps.json"
    KnowledgeGapStore(gaps_path).add("What is the warranty deadline?")
    assert main([
        "--session", str(session_path), "--docs", str(tmp_path / "docs.db"),
        "--gaps", str(gaps_path), "--weekly", "--days", "7",
        "--json"]) == 0
    review = json.loads(capsys.readouterr().out)
    assert review["task_counts"]["open"] == 1
    assert review["task_counts"]["completed_recent"] == 1
    assert review["recent_source_count"] == 1


def test_cli_weekly_markdown_export(tmp_path, capsys):
    session_path = tmp_path / "session.json"
    now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    session_path.write_text(json.dumps({"tasks": [
        {"id": 1, "text": "buy milk", "status": "done",
         "created": now, "completed": now},
        {"id": 2, "text": "write report", "status": "open",
         "created": now, "completed": ""},
    ]}), encoding="utf-8")
    docs = DocMemory(tmp_path / "docs.db")
    docs.add_text("notes/launch.txt", "The launch code is L-7781.",
                  title="Launch")
    docs.close()
    gaps_path = tmp_path / "gaps.json"
    KnowledgeGapStore(gaps_path).add("What is the warranty deadline?")
    assert main([
        "--session", str(session_path), "--docs", str(tmp_path / "docs.db"),
        "--gaps", str(gaps_path), "--weekly", "--days", "7",
        "--markdown"]) == 0
    markdown = capsys.readouterr().out
    assert markdown.startswith("# Weekly review - ")
    assert "Period: last 7 days" in markdown
    assert "- Open: 1" in markdown
    assert "- Completed in the last 7 days: 1" in markdown
    assert "### Open" in markdown
    assert "- [ ] **#2** write report" in markdown
    assert "### Completed in the last 7 days" in markdown
    assert "- [x] **#1** buy milk" in markdown
    assert "## Pending gaps (1)" in markdown
    assert "What is the warranty deadline?" in markdown
    assert "## Recent sources (1)" in markdown
    assert "**Launch** - `notes/launch.txt` (1 chunk(s))" in markdown
    assert "## Graph" in markdown
    assert "L-7781" in markdown
    assert "## Next week" in markdown
    assert "No model weights were changed." in markdown


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
