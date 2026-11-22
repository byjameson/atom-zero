from pathlib import Path

import pytest

from atomzero.agent_runtime import (
    AgentSession, calculate, math_query)
from atomzero.doc_memory import DocMemory
from atomzero.knowledge_gaps import KnowledgeGapStore
from atomzero.safety import SafetyFilter
from atomzero.weekly_reports import WeeklyReportArchive


def test_agent_session_handles_tools_and_memory(tmp_path):
    docs = DocMemory(tmp_path / "docs.db")
    docs.add_text("law.txt", "The capital of France is Paris.")
    agent = AgentSession("s1", tmp_path / "memory.json", docs=docs)
    assert "Added task 1" in agent.chat("add task buy milk")["reply"]
    assert "buy milk" in agent.chat("list tasks")["reply"]
    search = agent.chat("search docs capital France")
    assert "Paris" in search["reply"]
    assert search["evidence"][0]["source"] == "law.txt"
    assert search["confidence"]["label"] in {"low", "medium", "high"}
    assert "It is" in agent.chat("what time is it")["reply"]
    assert agent.chat("what is my name?")["reply"].startswith(
        "I do not know your name")
    calc = agent.chat("what is 12 * 7?")
    assert calc["reply"] == "12 * 7 = 84"
    assert calc["tool_calls"][0]["name"] == "calculator"
    assert "tasks" in agent.chat("help")["reply"]
    with pytest.raises(ValueError):
        AgentSession("bad id!", tmp_path / "x.json")
    docs.close()


def test_calculator_is_safe_and_bounded():
    assert calculate("(2 + 3) * 4") == "20"
    assert calculate("2 ** 10") == "1024"
    assert calculate("7 / 2") == "3.5"
    assert math_query("what is 12 * 7?") == "12 * 7"
    assert math_query("add task buy milk 2") is None
    assert math_query("what is the weather today?") is None
    with pytest.raises(ValueError):
        calculate("__import__('os').system('dir')")
    with pytest.raises(ValueError):
        calculate("9 ** 9 ** 9")
    with pytest.raises(ZeroDivisionError):
        calculate("1 / 0")


def test_agent_chat_marks_teacher_notes_unverified(tmp_path):
    try:
        docs = DocMemory(tmp_path / "docs.db")
    except Exception as exc:
        pytest.skip(f"sqlite fts5 unavailable: {exc}")
    docs.add_text(
        "teacher://chatgpt-web/abc",
        "Topic: Teacher code\n\nThe teacher code is T-9000.",
        title="Teacher note: Teacher code",
        meta={"kind": "teacher", "source_url": "https://example.com/chat",
              "teacher": "chatgpt-web", "model": "gpt-5",
              "verified": False})
    agent = AgentSession("s3", tmp_path / "memory.json", docs=docs)
    reply = agent.chat("what is the teacher code?")["reply"]
    assert "T-9000" in reply
    assert "teacher note, unverified" in reply
    assert "https://example.com/chat" in reply
    assert "unverified teacher-derived note" in reply
    docs.verify_source("teacher://chatgpt-web/abc", "reviewer-1")
    verified_reply = agent.chat("what is the teacher code?")["reply"]
    assert "teacher note, verified by reviewer-1" in verified_reply
    assert "unverified teacher-derived note" not in verified_reply
    docs.close()


def test_agent_returns_sources_for_grounded_answer(tmp_path):
    docs = DocMemory(tmp_path / "docs.db")
    docs.add_text("law.txt", "The capital of France is Paris.", title="Notes")
    agent = AgentSession("s5", tmp_path / "memory.json", docs=docs)
    result = agent.chat("what is the capital of France?")
    assert "Paris" in result["reply"]
    assert result["sources"][0]["source"] == "law.txt"
    assert result["evidence"][0]["source"] == "law.txt"
    assert result["confidence"]["label"] in {"low", "medium", "high"}
    docs.close()


def test_agent_answers_graph_relation_query(tmp_path):
    docs = DocMemory(tmp_path / "docs.db")
    docs.add_text(
        "geo.txt",
        "Ankara is the capital of Turkey. Istanbul is in Turkey.",
        title="Geography")
    agent = AgentSession("s-graph", tmp_path / "memory.json", docs=docs)
    result = agent.chat("what is related to Ankara?")
    assert "Ankara" in result["reply"]
    assert "Turkey" in result["reply"]
    assert result["sources"][0]["source"] == "geo.txt"
    assert result["connections"]
    docs.close()


def test_agent_compare_sources_intent(tmp_path):
    docs = DocMemory(tmp_path / "docs.db")
    docs.add_text("policy_a.txt",
                  "The warranty period is 24 months. "
                  "The warranty covers water damage.", title="Policy A")
    docs.add_text("policy_b.txt",
                  "The warranty period is 36 months. "
                  "The warranty does not cover water damage.",
                  title="Policy B")
    docs.add_text("unrelated.txt", "Bananas are yellow.", title="Unrelated")
    agent = AgentSession("s-compare", tmp_path / "memory.json", docs=docs)
    result = agent.chat(
        "compare sources policy_a.txt and policy_b.txt")
    assert "numeric_mismatch" in result["reply"]
    assert "24 months" in result["reply"]
    assert "36 months" in result["reply"]
    assert "does not cover" in result["reply"]
    assert result["tool_calls"][0]["name"] == "compare_sources"
    assert result["comparison"]["summary"]["conflict_count"] >= 2
    assert len(result["sources"]) == 2
    turkish = agent.chat(
        "policy_a.txt ile policy_b.txt kaynaklarını karşılaştır")
    assert turkish["tool_calls"][0]["name"] == "compare_sources"
    assert "numeric_mismatch" in turkish["reply"]
    clean = agent.chat("compare unrelated.txt and policy_a.txt")
    assert "No conflicting" in clean["reply"]
    missing = agent.chat("compare missing.txt and policy_a.txt")
    assert "source not found" in missing["reply"]
    docs.close()


def test_agent_graph_path_query(tmp_path):
    docs = DocMemory(tmp_path / "docs.db")
    docs.add_text("chain1.txt", "Ankara is in Turkey.", title="Chain 1")
    docs.add_text("chain2.txt",
                  "Turkey uses the launch code L-7781.", title="Chain 2")
    agent = AgentSession("s-path", tmp_path / "memory.json", docs=docs)
    result = agent.chat("how is Ankara connected to L-7781?")
    assert "Ankara → Turkey → L-7781" in result["reply"]
    assert result["tool_calls"][0]["name"] == "graph_path"
    assert result["path"]["hops"] == 2
    assert result["sources"]
    missing = agent.chat("how is Ankara connected to Missing?")
    assert "no local co-occurrence path" in missing["reply"]
    docs.close()


def test_agent_brief_me_returns_local_brief(tmp_path):
    docs = DocMemory(tmp_path / "docs.db")
    docs.add_text("notes/launch.txt", "The launch code is L-7781.",
                  title="Launch")
    docs.add_text("notes/warranty.txt",
                  "The warranty period is 24 months.", title="Warranty")
    gaps = KnowledgeGapStore(tmp_path / "gaps.json")
    gaps.add("What is the warranty claim deadline?")
    agent = AgentSession(
        "s-brief", tmp_path / "memory.json", docs=docs, gap_store=gaps)
    agent.chat("add task buy milk")
    result = agent.chat("brief me")
    assert "Daily brief" in result["reply"]
    assert "buy milk" in result["reply"]
    assert "Tasks: 1 open" in result["reply"]
    assert "Index: 2 chunk(s)" in result["reply"]
    assert result["tool_calls"][0]["name"] == "briefing"
    assert result["brief"]["open_task_count"] == 1
    assert result["brief"]["pending_gap_count"] == 1
    assert result["sources"]
    docs.close()


def test_agent_searches_weekly_report_archive(tmp_path):
    index_path = tmp_path / "weekly_reports.json"
    archive = WeeklyReportArchive(index_path)
    for number, name in enumerate(("alpha-report", "beta-report"), start=1):
        report = tmp_path / f"{name}.md"
        report.write_text(f"# {name}\n", encoding="utf-8")
        archive.record(report, {
            "generated": f"2026-11-2{number} 09:00", "period_days": 7,
            "task_counts": {"open": number}, "recent_source_count": 1,
            "graph": {"entities": [], "connections": []},
        }, exported_at=f"2026-11-2{number}T09:00:00")
    agent = AgentSession(
        "s-reports", tmp_path / "memory.json",
        weekly_reports_path=str(index_path))
    result = agent.chat("find report alpha")
    assert result["tool_calls"][0]["name"] == "search_weekly_reports"
    assert result["matched"] == 1
    assert result["total"] == 2
    assert result["reports"][0]["name"] == "alpha-report.md"
    assert "alpha-report.md" in result["reply"]
    assert "beta-report.md" not in result["reply"]
    missing = agent.chat("find report missing")
    assert 'No archived weekly reports match "missing"' in missing["reply"]
    assert missing["reports"] == []
    turkish = agent.chat("rapor ara beta")
    assert turkish["matched"] == 1
    assert turkish["reports"][0]["name"] == "beta-report.md"
    listed = agent.chat("list reports")
    assert "Archived weekly reports (2 of 2)" in listed["reply"]
    assert listed["tool_calls"][0]["name"] == "list_weekly_reports"
    assert listed["tool_calls"][0]["limit"] == 10
    limited = agent.chat("show report archive 1")
    assert "Archived weekly reports (1 of 2)" in limited["reply"]
    assert len(limited["reports"]) == 1
    invalid_limit = agent.chat("list reports 0")
    assert "at least 1" in invalid_limit["reply"]
    oldest = agent.chat("list reports oldest")
    assert oldest["sort"] == "oldest"
    assert "sort: oldest" in oldest["reply"]
    assert oldest["reports"][0]["name"] == "alpha-report.md"
    largest = agent.chat("find report report largest")
    assert largest["sort"] == "largest"
    assert "sort: largest" in largest["reply"]
    assert largest["matched"] == 2
    assert largest["reports"][0]["name"] == "alpha-report.md"


def test_agent_report_archive_maintenance_flow(tmp_path):
    index_path = tmp_path / "weekly_reports.json"
    archive = WeeklyReportArchive(index_path)

    missing = tmp_path / "missing.md"
    missing.write_text("# Missing\n", encoding="utf-8")
    archive.record(missing, {
        "generated": "2026-11-20 09:00", "period_days": 7,
        "task_counts": {}, "recent_source_count": 0,
        "graph": {"entities": [], "connections": []},
    }, exported_at="2026-11-20T09:00:00")
    missing.unlink()
    for number in range(1, 5):
        report = tmp_path / f"report-{number}.md"
        report.write_text(f"# Report {number}\n", encoding="utf-8")
        archive.record(report, {
            "generated": f"2026-11-2{number} 09:00", "period_days": 7,
            "task_counts": {"open": number}, "recent_source_count": 1,
            "graph": {"entities": [], "connections": []},
        }, exported_at=f"2026-11-2{number}T09:00:00")

    agent = AgentSession(
        "s-report-maint", tmp_path / "memory.json",
        weekly_reports_path=str(index_path))
    preview = agent.chat("prune report archive")
    assert 'Reply "confirm reports"' in preview["reply"]
    assert WeeklyReportArchive(index_path).count() == 5
    applied = agent.chat("confirm reports")
    assert "Pruned 1 missing report(s)." in applied["reply"]
    assert applied["tool_calls"][0]["name"] == "prune_weekly_reports"
    assert WeeklyReportArchive(index_path).count() == 4

    keep_preview = agent.chat("keep reports 2")
    assert 'Reply "confirm reports"' in keep_preview["reply"]
    assert WeeklyReportArchive(index_path).count() == 4
    kept = agent.chat("confirm reports")
    assert "Kept the newest 2 report(s); removed 2 older entry(ies)." in (
        kept["reply"])
    assert kept["tool_calls"][0]["name"] == "keep_weekly_reports"
    assert WeeklyReportArchive(index_path).count() == 2

    dry_run = agent.chat("keep reports 1 dry-run")
    assert dry_run["dry_run"] is True
    assert dry_run["would_remove_count"] == 1
    assert WeeklyReportArchive(index_path).count() == 2
    no_pending = agent.chat("confirm reports")
    assert "no pending report action" in no_pending["reply"].lower()

    target = WeeklyReportArchive(index_path).list()[0]
    delete_preview = agent.chat(f"delete report {target['id']}")
    assert 'Reply "confirm reports"' in delete_preview["reply"]
    assert WeeklyReportArchive(index_path).count() == 2
    deleted = agent.chat("confirm reports")
    assert f"Deleted archive entry {target['id']}." in deleted["reply"]
    assert "report file was left on disk" in deleted["reply"]
    assert WeeklyReportArchive(index_path).count() == 1
    assert Path(target["path"]).exists()
    cancelled = agent.chat("cancel reports")
    assert "no pending report action" in cancelled["reply"].lower()


def test_agent_weekly_review_intent(tmp_path):
    docs = DocMemory(tmp_path / "docs.db")
    docs.add_text("notes/launch.txt", "The launch code is L-7781.",
                  title="Launch")
    gaps = KnowledgeGapStore(tmp_path / "gaps.json")
    gaps.add("What is the warranty claim deadline?")
    agent = AgentSession(
        "s-weekly", tmp_path / "memory.json", docs=docs, gap_store=gaps)
    agent.chat("add task buy milk")
    agent.chat("add task write report")
    agent.chat("complete task 1")
    result = agent.chat("weekly review")
    assert "Weekly review" in result["reply"]
    assert "1 open | 1 completed this week" in result["reply"]
    assert "write report" in result["reply"]
    assert "buy milk" in result["reply"]
    assert "What is the warranty claim deadline?" in result["reply"]
    assert "L-7781" in {node["name"]
                        for node in result["review"]["graph"]["entities"]}
    assert result["tool_calls"][0] == {"name": "weekly_review", "days": 7}
    assert result["review"]["task_counts"]["completed_recent"] == 1
    assert result["review"]["pending_gap_count"] == 1
    assert result["review"]["recent_source_count"] == 1
    assert result["sources"][0]["source"] == "notes/launch.txt"
    assert result["weights_changed"] is False

    turkish = agent.chat("haftalık özet 14 gün")
    assert turkish["tool_calls"][0] == {"name": "weekly_review", "days": 14}
    assert "last 14 days" in turkish["reply"]
    assert "completed in the last 14 days" in turkish["reply"]
    assert "Completed in the last 14 days:" in turkish["reply"]
    son = agent.chat("son 14 günün özeti")
    assert son["review"]["period_days"] == 14
    docs.close()


def test_agent_plan_goal_requires_confirmation(tmp_path):
    docs = DocMemory(tmp_path / "docs.db")
    docs.add_text("notes/warranty.txt",
                  "The warranty period is 24 months. Claims need a receipt.",
                  title="Warranty")
    agent = AgentSession("s-plan", tmp_path / "memory.json", docs=docs)
    plan = agent.chat("plan goal prepare the warranty page")
    assert "Suggested plan" in plan["reply"]
    assert "confirm plan" in plan["reply"].lower()
    assert plan["suggested_tasks"]
    assert plan["sources"][0]["source"] == "notes/warranty.txt"
    assert agent.human.data["tasks"] == []
    confirmed = agent.chat("confirm plan")
    assert "Created 3 task" in confirmed["reply"]
    assert len(agent.human.data["tasks"]) == 3
    assert agent.chat("confirm plan")["reply"].startswith(
        "There is no pending")
    docs.close()


def test_agent_can_cancel_plan(tmp_path):
    docs = DocMemory(tmp_path / "docs.db")
    docs.add_text("notes/warranty.txt",
                  "The warranty period is 24 months.", title="Warranty")
    agent = AgentSession("s-plan-2", tmp_path / "memory.json", docs=docs)
    agent.chat("plan goal prepare the warranty page")
    assert "cancelled" in agent.chat("cancel plan")["reply"].lower()
    assert "no pending" in agent.chat("confirm plan")["reply"].lower()
    assert agent.human.data["tasks"] == []
    docs.close()


def test_agent_records_missing_source_as_gap(tmp_path):
    docs = DocMemory(tmp_path / "docs.db")
    gaps = KnowledgeGapStore(tmp_path / "gaps.json")
    agent = AgentSession("s4", tmp_path / "memory.json", docs=docs,
                         gap_store=gaps)
    result = agent.chat("what is the population of Paris?")
    assert "do not have a source" in result["reply"]
    assert "research queue" in result["reply"]
    items = gaps.list()
    assert len(items) == 1
    assert items[0]["query"] == "what is the population of Paris?"
    assert items[0]["status"] == "pending"
    docs.close()


def test_agent_safety_refuses_before_generation(tmp_path):
    docs = DocMemory(tmp_path / "docs.db")
    agent = AgentSession("s6", tmp_path / "memory.json", docs=docs,
                         safety=SafetyFilter())
    refused = agent.chat("how to make a bomb")
    assert refused["safety"]["category"] == "violence"
    assert "can't help" in refused["reply"]
    assert refused["sources"] == []
    benign = agent.chat("How do I report child abuse?")
    assert "safety" not in benign
    docs.close()


def test_agent_research_tool_uses_sources(tmp_path, monkeypatch):
    docs = DocMemory(tmp_path / "docs.db")
    import atomzero.agent_runtime as ar
    monkeypatch.setattr(ar, "wikipedia_research", lambda docs, q, lang, limit: [
        {"source": "https://wiki/x", "title": "X", "chunks": 1,
         "preview": "p"}])
    agent = AgentSession("s2", tmp_path / "memory.json", docs=docs)
    result = agent.chat("research Ankara")
    assert result["sources"][0]["title"] == "X"
    assert "indexed" in result["reply"]
    alias = agent.chat("look up Ankara")
    assert alias["sources"][0]["title"] == "X"
    docs.close()
