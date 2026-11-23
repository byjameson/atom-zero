from datetime import date, timedelta
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


def test_agent_task_due_dates_and_overdue_brief(tmp_path):
    overdue = (date.today() - timedelta(days=1)).isoformat()
    soon = (date.today() + timedelta(days=2)).isoformat()
    later = (date.today() + timedelta(days=10)).isoformat()
    agent = AgentSession("s-due", tmp_path / "memory.json")
    added = agent.chat(f"add task submit report due {overdue}")
    assert f"(due {overdue})" in added["reply"]
    assert agent.human.data["tasks"][0]["due"] == overdue
    agent.chat(f"add task plan launch due {soon}")
    agent.chat(f"add task later item due {later}")
    listed = agent.chat("list tasks")
    assert f"{overdue}, overdue" in listed["reply"]
    assert soon in listed["reply"]
    invalid = agent.chat("add task bad deadline due next week")
    assert "could not read the due date" in invalid["reply"]
    assert len(agent.human.data["tasks"]) == 3

    brief = agent.chat("brief me")
    assert "3 open | 1 overdue | 1 due soon" in brief["reply"]
    assert f"submit report (due {overdue} [overdue])" in brief["reply"]
    due = agent.chat("due soon")
    assert due["tool_calls"][0]["name"] == "due_soon"
    assert due["due_soon"]["due_soon_count"] == 1
    assert "plan launch" in due["reply"]
    review = agent.chat("weekly review")
    assert "Overdue: 1" in review["reply"]
    assert "Due soon: 1 (next 7 days)" in review["reply"]
    assert f"submit report (due {overdue} [overdue])" in review["reply"]
    rescheduled = agent.chat(f"reschedule task 3 due {soon}")
    assert f"Rescheduled task 3 to {soon}." in rescheduled["reply"]
    assert agent.human.data["tasks"][2]["due"] == soon
    missing_task = agent.chat(f"reschedule task 99 due {soon}")
    assert "could not find that task" in missing_task["reply"]
    invalid_due = agent.chat("reschedule task 1 due next week")
    assert "due date must be" in invalid_due["reply"]


def test_agent_deletes_task_with_confirmation(tmp_path):
    agent = AgentSession("s-delete-task", tmp_path / "memory.json")
    agent.chat("add task buy milk")
    agent.chat("add task write report")
    preview = agent.chat("delete task 1")
    assert "Delete task 1: buy milk?" in preview["reply"]
    assert 'Reply "confirm task delete"' in preview["reply"]
    assert len(agent.human.data["tasks"]) == 2
    deleted = agent.chat("confirm task delete")
    assert "Deleted task 1: buy milk." in deleted["reply"]
    assert [task["id"] for task in agent.human.data["tasks"]] == [2]
    assert "buy milk" not in agent.chat("list tasks")["reply"]
    cancelled = agent.chat("delete task 2")
    assert "cancel task delete" in cancelled["reply"]
    assert "Task delete cancelled." in agent.chat(
        "cancel task delete")["reply"]
    assert len(agent.human.data["tasks"]) == 1
    missing = agent.chat("delete task 99")
    assert "could not find that task" in missing["reply"]


def test_agent_searches_task_text(tmp_path):
    today = date.today()
    soon = (today + timedelta(days=2)).isoformat()
    overdue = (today - timedelta(days=1)).isoformat()
    agent = AgentSession("s-search-tasks", tmp_path / "memory.json")
    agent.chat(
        f"add task submit quarterly report priority high due {soon} "
        "tags work")
    agent.chat(
        f"add task submit expenses due {overdue} priority low tags finance")
    agent.chat("add task plan launch tags work")
    found = agent.chat("search tasks report")
    assert found["tool_calls"][0]["name"] == "search_tasks"
    assert found["task_search"]["count"] == 1
    assert found["task_search"]["tasks"][0]["text"] == "submit quarterly report"
    assert "Task search" in found["reply"]
    assert "submit quarterly report" in found["reply"]
    assert "plan launch" not in found["reply"]
    by_find = agent.chat("find task launch")
    assert by_find["task_search"]["count"] == 1
    assert "plan launch" in by_find["reply"]
    open_only = agent.chat("search tasks submit open")
    assert open_only["tool_calls"][0]["status"] == "open"
    assert open_only["tool_calls"][0]["due"] == "all"
    assert open_only["tool_calls"][0]["priority"] == "all"
    assert open_only["task_search"]["count"] == 2
    week = agent.chat("search tasks submit open due week")
    assert week["task_search"]["status"] == "open"
    assert week["task_search"]["due"] == "week"
    assert week["task_search"]["count"] == 1
    assert "status=open" in week["reply"]
    assert "due=week" in week["reply"]
    high = agent.chat("search tasks submit priority high")
    assert high["task_search"]["priority"] == "high"
    assert high["task_search"]["count"] == 1
    assert "priority=high" in high["reply"]
    assert high["task_search"]["tasks"][0]["id"] == 1
    combined = agent.chat("search tasks submit high open due week")
    assert combined["task_search"]["priority"] == "high"
    assert combined["task_search"]["status"] == "open"
    assert combined["task_search"]["due"] == "week"
    assert combined["task_search"]["count"] == 1
    low = agent.chat("search tasks submit low due overdue")
    assert low["task_search"]["priority"] == "low"
    assert low["task_search"]["count"] == 1
    assert low["task_search"]["tasks"][0]["id"] == 2
    tagged = agent.chat("search tasks submit tag work")
    assert tagged["tool_calls"][0]["tag"] == "work"
    assert tagged["task_search"]["tag"] == "work"
    assert tagged["task_search"]["count"] == 1
    assert tagged["task_search"]["tasks"][0]["id"] == 1
    assert "tag=work" in tagged["reply"]
    hash_tag = agent.chat("search tasks launch #work")
    assert hash_tag["task_search"]["tag"] == "work"
    assert hash_tag["task_search"]["count"] == 1
    assert hash_tag["task_search"]["tasks"][0]["id"] == 3
    overdue_only = agent.chat("search tasks submit overdue")
    assert overdue_only["task_search"]["due"] == "overdue"
    assert overdue_only["task_search"]["tasks"][0]["text"] == "submit expenses"
    undated = agent.chat("task search launch none")
    assert undated["task_search"]["due"] == "none"
    assert undated["task_search"]["count"] == 1
    agent.chat("complete task 1")
    done = agent.chat("search tasks submit done")
    assert done["task_search"]["status"] == "done"
    assert done["task_search"]["count"] == 1
    assert done["task_search"]["tasks"][0]["id"] == 1
    done_high = agent.chat("search tasks submit done high")
    assert done_high["task_search"]["status"] == "done"
    assert done_high["task_search"]["priority"] == "high"
    assert done_high["task_search"]["count"] == 1
    explicit = agent.chat("search tasks report status done")
    assert explicit["task_search"]["status"] == "done"
    assert explicit["task_search"]["count"] == 1
    missing = agent.chat("task search missing")
    assert missing["task_search"]["count"] == 0
    assert "None" in missing["reply"]
    no_text = agent.chat("search tasks open")
    assert "Tell me task text to search" in no_text["reply"]


def test_agent_lists_task_tags(tmp_path):
    agent = AgentSession("s-task-tags", tmp_path / "memory.json")
    empty = agent.chat("list tags")
    assert empty["tool_calls"][0]["name"] == "list_task_tags"
    assert empty["task_tags"]["tag_count"] == 0
    assert "There are no tasks." in empty["reply"]
    agent.chat("add task write report tags work")
    agent.chat("add task pay invoice tags work,finance")
    agent.chat("add task read book")
    agent.chat("complete task 1")
    result = agent.chat("list tags")
    assert result["tool_calls"][0]["name"] == "list_task_tags"
    assert result["tool_calls"][0]["tag_count"] == 2
    assert result["tool_calls"][0]["tagged_task_count"] == 2
    assert result["tool_calls"][0]["untagged_task_count"] == 1
    view = result["task_tags"]
    assert view["open_tagged"] == 1
    assert view["done_tagged"] == 1
    assert view["tags"] == [
        {"tag": "work", "open": 1, "done": 1, "total": 2},
        {"tag": "finance", "open": 1, "done": 0, "total": 1},
    ]
    assert ("Task tags: 2 distinct | 2 tagged task(s) "
            "(1 open, 1 done) | 1 untagged") in result["reply"]
    assert "- finance: 1 open, 0 done, 1 total" in result["reply"]
    assert "- work: 1 open, 1 done, 2 total" in result["reply"]
    assert agent.chat("show tags")["task_tags"]["tag_count"] == 2
    assert agent.chat("list task tags")["task_tags"]["tag_count"] == 2
    assert agent.chat("tags?")["task_tags"]["tag_count"] == 2
    assert agent.chat("tag task 2 finance,home")["reply"].startswith(
        "Set tags on task 2")
    refreshed = agent.chat("tags")
    assert refreshed["task_tags"]["tag_count"] == 3
    assert [item["tag"] for item in refreshed["task_tags"]["tags"]] == [
        "finance", "home", "work"]
    assert agent.chat("tag task 2 clear")["reply"] == (
        "Cleared tags on task 2.")


def test_agent_renames_tag_with_confirmation(tmp_path):
    agent = AgentSession("s-rename-tag", tmp_path / "memory.json")
    agent.chat("add task write report tags work,home")
    agent.chat("add task pay invoice tags work,finance")
    agent.chat("add task read book")

    preview = agent.chat("rename tag work job")
    assert preview["tool_calls"][0]["name"] == "preview_tag_rename"
    assert preview["tag_rename"]["old"] == "work"
    assert preview["tag_rename"]["new"] == "job"
    assert preview["tag_rename"]["tasks"] == 2
    assert "on 2 task(s)" in preview["reply"]
    assert "confirm tag rename" in preview["reply"]
    assert [task["tags"] for task in agent.human.data["tasks"]] == [
        ["work", "home"], ["work", "finance"], []]

    unrelated = agent.chat("cancel task delete")
    assert unrelated["reply"] == "There is no pending task delete."
    assert agent.pending_task_action["action"] == "rename_tag"

    applied = agent.chat("confirm tag rename")
    assert applied["tool_calls"][0] == {
        "name": "rename_task_tag", "old": "work", "new": "job",
        "updated": 2}
    assert applied["reply"] == "Renamed tag 'work' to 'job' on 2 task(s)."
    assert [task["tags"] for task in agent.human.data["tasks"]] == [
        ["job", "home"], ["job", "finance"], []]
    assert agent.human.data["tasks"][0]["renamed"]

    tags = agent.chat("tags")
    assert tags["task_tags"]["tag_count"] == 3
    assert [item["tag"] for item in tags["task_tags"]["tags"]] == [
        "job", "finance", "home"]

    assert agent.chat("confirm tag rename")["reply"] == (
        "There is no pending tag rename.")
    missing = agent.chat("rename tag missing job")
    assert "Tag 'missing' is not used." == missing["reply"]
    assert agent.pending_task_action is None
    same = agent.chat("rename tag job job")
    assert "must be different" in same["reply"]
    invalid = agent.chat("rename tag job bad!")
    assert "letters, numbers" in invalid["reply"]

    second = agent.chat("rename tag job office")
    assert second["tag_rename"]["tasks"] == 2
    assert agent.pending_task_action["action"] == "rename_tag"
    assert agent.chat("cancel tag rename")["reply"] == (
        "Tag rename cancelled.")
    assert agent.pending_task_action is None
    assert [task["tags"] for task in agent.human.data["tasks"]] == [
        ["job", "home"], ["job", "finance"], []]

    to_form = agent.chat("rename tag job to office")
    assert to_form["tag_rename"]["new"] == "office"
    assert agent.chat("cancel tag rename")["reply"] == (
        "Tag rename cancelled.")


def test_agent_priority_sort_task_views(tmp_path):
    today = date.today()
    agent = AgentSession("s-priority-task-views", tmp_path / "memory.json")
    for text, offset, priority in (
            ("low overdue task", -1, "low"),
            ("high overdue task", -2, "high"),
            ("normal soon task", 2, "normal"),
            ("high soon task", 1, "high")):
        due = (today + timedelta(days=offset)).isoformat()
        agent.chat(f"add task {text} due {due} priority {priority}")
    default_search = agent.chat("search tasks task")
    assert [task["id"] for task in default_search["task_search"]["tasks"]] == [
        1, 2, 3, 4]
    search = agent.chat("search tasks task sort priority")
    assert search["tool_calls"][0]["sort"] == "priority"
    assert search["task_search"]["sort"] == "priority"
    assert [task["id"] for task in search["task_search"]["tasks"]] == [
        2, 4, 3, 1]
    assert "sort=priority" in search["reply"]
    filtered = agent.chat("search tasks task priority high sort priority")
    assert filtered["tool_calls"][0]["priority"] == "high"
    assert filtered["tool_calls"][0]["sort"] == "priority"
    assert [task["id"] for task in filtered["task_search"]["tasks"]] == [2, 4]
    default_result = agent.chat("list tasks")
    default_list = default_result["reply"]
    assert default_result["tool_calls"][0] == {
        "name": "list_tasks", "status": "open", "due": "all",
        "priority": "all", "tag": "all", "sort": "default", "count": 4}
    assert default_list.startswith("Open tasks: ")
    assert (default_list.index("low overdue task")
            < default_list.index("high overdue task"))
    priority_result = agent.chat("list tasks priority")
    priority_list = priority_result["reply"]
    assert priority_result["tool_calls"][0]["sort"] == "priority"
    assert priority_result["task_list"]["count"] == 4
    assert priority_list.startswith("Open tasks (priority first): ")
    assert (priority_list.index("high overdue task")
            < priority_list.index("high soon task")
            < priority_list.index("normal soon task")
            < priority_list.index("low overdue task"))
    assert agent.chat("list tasks sort priority")["reply"] == priority_list
    assert (agent.chat("list tasks sorted by priority")["reply"]
            == priority_list)
    agent.chat("add task done filtered task priority high")
    agent.chat("complete task 5")
    done_result = agent.chat("list tasks done")
    done_list = done_result["reply"]
    assert done_result["tool_calls"][0] == {
        "name": "list_tasks", "status": "done", "due": "all",
        "priority": "all", "tag": "all", "sort": "default", "count": 1}
    assert done_list.startswith("Done tasks: ")
    assert "done filtered task" in done_list
    assert "high overdue task" not in done_list
    due_high_result = agent.chat("list tasks due week priority high")
    due_high = due_high_result["reply"]
    assert due_high_result["tool_calls"][0] == {
        "name": "list_tasks", "status": "open", "due": "week",
        "priority": "high", "tag": "all", "sort": "default", "count": 1}
    assert [task["id"] for task in
            due_high_result["task_list"]["tasks"]] == [4]
    assert due_high.startswith(
        "Open tasks (due week, priority high): ")
    assert "high soon task" in due_high
    assert "high overdue task" not in due_high
    all_result = agent.chat("list tasks all")
    all_list = all_result["reply"]
    assert all_result["tool_calls"][0] == {
        "name": "list_tasks", "status": "all", "due": "all",
        "priority": "all", "tag": "all", "sort": "default", "count": 5}
    assert all_list.startswith("All tasks: ")
    assert "done filtered task" in all_list
    assert "normal soon task" in all_list
    high_result = agent.chat("list tasks high")
    high_list = high_result["reply"]
    assert high_result["tool_calls"][0]["priority"] == "high"
    assert high_result["task_list"]["count"] == 2
    assert "high overdue task" in high_list
    assert "high soon task" in high_list
    assert "normal soon task" not in high_list
    overdue_high_result = agent.chat("list tasks due overdue priority")
    overdue_high = overdue_high_result["reply"]
    assert overdue_high_result["tool_calls"][0] == {
        "name": "list_tasks", "status": "open", "due": "overdue",
        "priority": "all", "tag": "all", "sort": "priority", "count": 2}
    assert overdue_high.startswith(
        "Open tasks (due overdue, priority first): ")
    assert (overdue_high.index("high overdue task")
            < overdue_high.index("low overdue task"))
    undated_high_result = agent.chat(
        "list tasks all due none priority high")
    undated_high = undated_high_result["reply"]
    assert undated_high_result["tool_calls"][0]["count"] == 1
    assert [task["id"] for task in
            undated_high_result["task_list"]["tasks"]] == [5]
    assert "done filtered task" in undated_high
    assert "high soon task" not in undated_high
    parity_search = agent.chat(
        "search tasks task open due week priority high")
    assert ([task["id"] for task in parity_search["task_search"]["tasks"]]
            == [task["id"] for task in due_high_result["task_list"]["tasks"]])
    default_overdue = agent.chat("overdue tasks")
    assert default_overdue["tool_calls"][0] == {
        "name": "overdue", "sort": "default", "count": 2}
    assert [task["id"] for task in
            default_overdue["overdue"]["overdue_tasks"]] == [1, 2]
    overdue = agent.chat("overdue tasks priority")
    assert overdue["tool_calls"][0] == {
        "name": "overdue", "sort": "priority", "count": 2}
    assert [task["id"] for task in overdue["overdue"]["overdue_tasks"]] == [
        2, 1]
    assert "sort=priority" in overdue["reply"]
    default_due = agent.chat("due soon")
    assert [task["id"] for task in
            default_due["due_soon"]["due_soon_tasks"]] == [3, 4]
    due = agent.chat("due soon priority")
    assert due["tool_calls"][0] == {
        "name": "due_soon", "days": 7, "sort": "priority"}
    assert [task["id"] for task in due["due_soon"]["due_soon_tasks"]] == [
        4, 3]
    assert "sort=priority" in due["reply"]
    longer = agent.chat("due soon in 7 days sort priority")
    assert longer["tool_calls"][0] == {
        "name": "due_soon", "days": 7, "sort": "priority"}


def test_agent_task_tags_filter_and_clear(tmp_path):
    agent = AgentSession("s-task-tags", tmp_path / "memory.json")
    added = agent.chat(
        "add task submit report priority high tags work,finance")
    assert "(tags work, finance)" in added["reply"]
    assert agent.human.data["tasks"][0]["tags"] == ["work", "finance"]
    agent.chat("add task buy milk tags home")
    retagged = agent.chat("tag task 1 work,urgent")
    assert "Set tags on task 1: work, urgent." in retagged["reply"]
    assert agent.human.data["tasks"][0]["tags"] == ["work", "urgent"]
    work = agent.chat("list tasks tag work")
    assert work["tool_calls"][0]["tag"] == "work"
    assert work["task_list"]["count"] == 1
    assert work["reply"].startswith("Open tasks (tag work): ")
    assert "submit report" in work["reply"]
    assert "buy milk" not in work["reply"]
    home = agent.chat("list tasks #home")
    assert home["task_list"]["count"] == 1
    assert "buy milk" in home["reply"]
    cleared = agent.chat("tag task 1 clear")
    assert "Cleared tags on task 1." in cleared["reply"]
    empty = agent.chat("list tasks tag work")
    assert empty["task_list"]["count"] == 0
    assert "No tasks match." in empty["reply"]
    invalid = agent.chat("add task bad tags bad tag!")
    assert "tags must use letters" in invalid["reply"]
    assert len(agent.human.data["tasks"]) == 2


def test_agent_task_priority_metadata(tmp_path):
    agent = AgentSession("s-task-priority", tmp_path / "memory.json")
    added = agent.chat("add task submit report priority high")
    assert "(priority high)" in added["reply"]
    assert agent.human.data["tasks"][0]["priority"] == "high"
    agent.chat("add task plan launch due tomorrow priority low")
    assert agent.human.data["tasks"][1]["priority"] == "low"
    listed = agent.chat("list tasks")["reply"]
    assert "1 [high] submit report" in listed
    assert "2 [low] plan launch" in listed
    changed = agent.chat("set task 1 priority low")
    assert "Set task 1 priority to low." in changed["reply"]
    assert agent.human.data["tasks"][0]["priority"] == "low"
    no_priority_word = agent.chat("set task 1 high")
    assert "Set task 1 priority to high." in no_priority_word["reply"]
    missing = agent.chat("set task 99 priority high")
    assert "could not find that task" in missing["reply"]
    invalid_add = agent.chat("add task bad priority urgent")
    assert "could not read the priority 'urgent'" in invalid_add["reply"]
    assert len(agent.human.data["tasks"]) == 2
    invalid_set = agent.chat("set task 1 priority urgent")
    assert ("Use 'set task ID priority low|normal|high'."
            in invalid_set["reply"])
    assert agent.human.data["tasks"][0]["priority"] == "high"


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
    shown = agent.chat("show report w0001")
    assert shown["tool_calls"][0]["name"] == "show_weekly_report"
    assert shown["report_id"] == "w0001"
    assert shown["report_markdown"].startswith("# alpha-report")
    assert "exported" in shown["reply"]
    missing_show = agent.chat("show report w9999")
    assert "not found" in missing_show["reply"]
    compared = agent.chat("compare reports w0001 w0002")
    assert compared["tool_calls"][0]["name"] == "compare_weekly_reports"
    metrics = {item["key"]: item
               for item in compared["comparison"]["metrics"]}
    assert metrics["open"]["delta"] == 1
    assert compared["comparison"]["hash_equal"] is False
    assert "Report comparison: w0001 -> w0002" in compared["reply"]
    missing_compare = agent.chat("compare reports w0001 w9999")
    assert "not found" in missing_compare["reply"]


def test_agent_verifies_report_archive_integrity(tmp_path):
    index_path = tmp_path / "weekly_reports.json"
    archive = WeeklyReportArchive(index_path)
    good = tmp_path / "good.md"
    good.write_text("# Good\n", encoding="utf-8")
    changed = tmp_path / "changed.md"
    changed.write_text("# Alpha\n", encoding="utf-8")
    missing = tmp_path / "missing.md"
    missing.write_text("# Missing\n", encoding="utf-8")
    for number, path in enumerate((good, changed, missing), start=1):
        archive.record(path, {
            "generated": f"2026-11-2{number} 09:00", "period_days": 7,
            "task_counts": {}, "recent_source_count": 0,
            "graph": {"entities": [], "connections": []},
        }, exported_at=f"2026-11-2{number}T09:00:00")
    changed.write_text("# Bravo\n", encoding="utf-8")
    missing.unlink()
    agent = AgentSession(
        "s-verify", tmp_path / "memory.json",
        weekly_reports_path=str(index_path))

    result = agent.chat("verify report archive")
    assert result["tool_calls"][0]["scope"] == "archive"
    assert result["verification"]["summary"]["ok"] == 1
    assert result["verification"]["summary"]["failed"] == 2
    assert result["verification"]["summary"]["missing"] == 1
    assert "1 ok, 2 failed, 1 missing" in result["reply"]
    good_check = agent.chat("verify report w0001")
    assert good_check["ok"] is True
    assert good_check["reports"][0]["status"] == "ok"
    changed_check = agent.chat("verify report w0002")
    assert changed_check["ok"] is False
    assert "hash_mismatch" in changed_check["reply"]
    missing_check = agent.chat("verify report w9999")
    assert "not found" in missing_check["reply"]


def test_agent_repairs_report_metadata_with_confirmation(tmp_path):
    index_path = tmp_path / "weekly_reports.json"
    archive = WeeklyReportArchive(index_path)
    changed = tmp_path / "changed.md"
    changed.write_text("# Alpha\n", encoding="utf-8")
    missing = tmp_path / "missing.md"
    missing.write_text("# Missing\n", encoding="utf-8")
    for number, path in enumerate((changed, missing), start=1):
        archive.record(path, {
            "generated": f"2026-11-2{number} 09:00", "period_days": 7,
            "task_counts": {}, "recent_source_count": 0,
            "graph": {"entities": [], "connections": []},
        }, exported_at=f"2026-11-2{number}T09:00:00")
    changed.write_text("# Bravo changed\n", encoding="utf-8")
    missing.unlink()
    agent = AgentSession(
        "s-repair", tmp_path / "memory.json",
        weekly_reports_path=str(index_path))

    preview = agent.chat("repair report archive")
    assert 'Reply "confirm reports"' in preview["reply"]
    assert preview["tool_calls"][0]["action"] == "repair"
    before = archive.verify()
    assert before["summary"]["failed"] == 2
    assert before["summary"]["missing"] == 1

    applied = agent.chat("confirm reports")
    assert "Repaired 1 report metadata; 0 unchanged; 1 missing." in (
        applied["reply"])
    assert applied["tool_calls"][0]["name"] == "repair_weekly_reports"
    after = archive.verify()
    assert after["summary"]["ok"] == 1
    assert after["summary"]["missing"] == 1


def test_agent_repairs_single_report_with_confirmation(tmp_path):
    index_path = tmp_path / "weekly_reports.json"
    archive = WeeklyReportArchive(index_path)
    first = tmp_path / "first.md"
    first.write_text("# First\n", encoding="utf-8")
    second = tmp_path / "second.md"
    second.write_text("# Second\n", encoding="utf-8")
    missing = tmp_path / "missing.md"
    missing.write_text("# Missing\n", encoding="utf-8")
    for number, path in enumerate((first, second, missing), start=1):
        archive.record(path, {
            "generated": f"2026-11-2{number} 09:00", "period_days": 7,
            "task_counts": {}, "recent_source_count": 0,
            "graph": {"entities": [], "connections": []},
        }, exported_at=f"2026-11-2{number}T09:00:00")
    first.write_text("# First changed\n", encoding="utf-8")
    second.write_text("# Second changed\n", encoding="utf-8")
    missing.unlink()
    agent = AgentSession(
        "s-single-repair", tmp_path / "memory.json",
        weekly_reports_path=str(index_path))

    preview = agent.chat("repair report w0001")
    assert "Report w0001 metadata repair preview" in preview["reply"]
    assert 'Reply "confirm reports"' in preview["reply"]
    assert archive.verify()["summary"]["size_mismatch"] == 2
    applied = agent.chat("confirm reports")
    assert "Report w0001 metadata repair: 1 repaired" in applied["reply"]
    after_first = archive.verify()
    assert after_first["summary"]["size_mismatch"] == 1
    assert after_first["summary"]["ok"] == 1

    preview_second = agent.chat("repair report w0002")
    assert "Report w0002 metadata repair preview" in preview_second["reply"]
    assert "Report w0002 metadata repair: 1 repaired" in (
        agent.chat("confirm reports")["reply"])
    assert archive.verify()["summary"]["ok"] == 2

    missing_check = agent.chat("repair report w0003")
    assert "file is missing and cannot be repaired" in missing_check["reply"]
    unknown = agent.chat("repair report w9999")
    assert "not found" in unknown["reply"]


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


def test_agent_priority_first_brief_and_review(tmp_path):
    agent = AgentSession("s-priority-brief", tmp_path / "memory.json")
    agent.chat("add task low task priority low")
    agent.chat("add task normal task")
    agent.chat("add task high task priority high")
    default_brief = agent.chat("brief me")
    assert default_brief["tool_calls"] == [
        {"name": "briefing", "sort": "default"}]
    assert default_brief["brief"]["sort"] == "default"
    assert [task["id"] for task in default_brief["brief"]["tasks"]] == [1, 2, 3]
    brief = agent.chat("brief me priority")
    assert brief["tool_calls"] == [{"name": "briefing", "sort": "priority"}]
    assert brief["brief"]["sort"] == "priority"
    assert [task["id"] for task in brief["brief"]["tasks"]] == [3, 2, 1]
    assert "sort=priority" in brief["reply"]
    assert brief["reply"].index("high task") < brief["reply"].index("low task")
    daily = agent.chat("daily brief priority")
    assert daily["brief"]["sort"] == "priority"
    default_review = agent.chat("weekly review")
    assert default_review["tool_calls"] == [
        {"name": "weekly_review", "days": 7, "sort": "default"}]
    assert [task["id"] for task in default_review["review"]["open_tasks"]] == [
        1, 2, 3]
    review = agent.chat("weekly review priority")
    assert review["tool_calls"] == [
        {"name": "weekly_review", "days": 7, "sort": "priority"}]
    assert review["review"]["sort"] == "priority"
    assert [task["id"] for task in review["review"]["open_tasks"]] == [3, 2, 1]
    assert "sort=priority" in review["reply"]
    longer = agent.chat("weekly review 14 days priority")
    assert longer["tool_calls"] == [
        {"name": "weekly_review", "days": 14, "sort": "priority"}]
    assert longer["review"]["sort"] == "priority"


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
    assert result["tool_calls"][0] == {
        "name": "weekly_review", "days": 7, "sort": "default"}
    assert result["review"]["task_counts"]["completed_recent"] == 1
    assert result["review"]["pending_gap_count"] == 1
    assert result["review"]["recent_source_count"] == 1
    assert result["sources"][0]["source"] == "notes/launch.txt"
    assert result["weights_changed"] is False

    turkish = agent.chat("haftalık özet 14 gün")
    assert turkish["tool_calls"][0] == {
        "name": "weekly_review", "days": 14, "sort": "default"}
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
