import csv
import io
import json
import time
from datetime import date, datetime, timedelta

import pytest

from atomzero.briefing import (
    build_briefing, build_weekly_review, delete_task_tag, due_soon_view,
    load_session, main, overdue_view, plan_goal, rename_task_tag,
    render_due_soon_view, render_overdue_view, render_weekly_markdown,
    render_weekly_report_index, task_tag_audit_view, undo_delete_task_tag)
from atomzero.doc_memory import DocMemory
from atomzero.knowledge_gaps import KnowledgeGapStore
from atomzero.weekly_reports import WeeklyReportArchive


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


def test_brief_tag_filter_scopes_tasks_and_tags(tmp_path, capsys):
    # The brief is evaluated as-of 2026-11-23; derive the fixtures from the
    # same fixed date so the test does not change with the wall clock.
    today = date(2026, 11, 23)
    soon = (today + timedelta(days=2)).isoformat()
    overdue = (today - timedelta(days=1)).isoformat()
    session = {"tasks": [
        {"id": 1, "text": "write report", "status": "open",
         "due": soon, "tags": ["work"]},
        {"id": 2, "text": "buy milk", "status": "open",
         "due": overdue, "tags": ["home"]},
        {"id": 3, "text": "pay invoice", "status": "open",
         "due": "", "tags": ["work", "home"]},
        {"id": 4, "text": "old task", "status": "done",
         "tags": ["archive"]},
    ]}
    brief = build_briefing(
        session, None, None, now="2026-11-23 09:00", tag="work")
    assert brief["tag"] == "work"
    assert brief["open_task_count"] == 2
    assert [task["id"] for task in brief["tasks"]] == [1, 3]
    assert brief["overdue_task_count"] == 0
    assert brief["due_soon_count"] == 1
    assert [task["id"] for task in brief["due_soon_tasks"]] == [1]
    assert brief["tags"] == [
        {"tag": "work", "open": 2, "done": 0, "total": 2},
        {"tag": "home", "open": 1, "done": 0, "total": 1},
    ]
    assert brief["tag_count"] == 2
    assert brief["tagged_task_count"] == 2
    assert brief["untagged_task_count"] == 0
    assert "| tag=work" in brief["text"]
    assert "Focus: due soon: task 1: write report" in brief["text"]
    assert ("Tags for #work: 2 distinct | work (2 open) | "
            "home (1 open)") in brief["text"]
    assert "write report" in brief["text"]
    assert "pay invoice" in brief["text"]
    assert "buy milk" not in brief["text"]
    assert "archive" not in brief["text"]

    unfiltered = build_briefing(
        session, None, None, now="2026-11-23 09:00")
    assert unfiltered["tag"] == "all"
    assert unfiltered["open_task_count"] == 3
    assert unfiltered["overdue_task_count"] == 1
    assert "Focus: overdue task 2: buy milk" in unfiltered["text"]
    assert ("Tags: 2 distinct | home (2 open) | work (2 open)"
            in unfiltered["text"])
    assert "archive" not in unfiltered["text"]

    session_path = tmp_path / "session.json"
    session_path.write_text(json.dumps(session), encoding="utf-8")
    common = [
        "--session", str(session_path),
        "--docs", str(tmp_path / "missing.db"),
        "--gaps", str(tmp_path / "missing-gaps.json")]
    assert main(common + ["--brief-tag", "work", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["tag"] == "work"
    assert payload["open_task_count"] == 2
    assert payload["tag_count"] == 2
    assert [task["id"] for task in payload["tasks"]] == [1, 3]
    assert main(common + ["--brief-tag", "work"]) == 0
    output = capsys.readouterr().out
    assert "Daily brief - " in output
    assert "| tag=work" in output
    assert "Tags for #work" in output
    assert "buy milk" not in output
    assert main(common + ["--brief-tag", "bad tag!"]) == 1
    assert "letters, numbers" in capsys.readouterr().out
    with pytest.raises(SystemExit) as exc:
        main(common + ["--brief-tag", "work", "--weekly"])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        main(common + ["--brief-tag", "work", "--task-list"])
    assert exc.value.code == 2


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
         "created": now.strftime("%Y-%m-%dT%H:%M:%S"), "completed": "",
         "tags": ["work"]},
        {"id": 2, "text": "recent task", "status": "done",
         "created": now.strftime("%Y-%m-%dT%H:%M:%S"),
         "completed": now.strftime("%Y-%m-%dT%H:%M:%S"),
         "tags": ["work", "finance"]},
        {"id": 3, "text": "old task", "status": "done",
         "created": (now - timedelta(days=10)).strftime("%Y-%m-%dT%H:%M:%S"),
         "completed": (now - timedelta(days=10)).strftime(
             "%Y-%m-%dT%H:%M:%S"),
         "tags": ["archive"]},
    ]}
    review = build_weekly_review(
        session, gaps, docs, days=7, now="2026-11-22 09:00")
    assert review["task_counts"] == {
        "open": 1, "completed_recent": 1, "completed_total": 2}
    assert review["tags"] == [
        {"tag": "work", "open": 1, "done": 1, "total": 2},
        {"tag": "finance", "open": 0, "done": 1, "total": 1},
    ]
    assert review["tag_count"] == 2
    assert review["tagged_task_count"] == 2
    assert review["untagged_task_count"] == 0
    assert review["pending_gap_count"] == 1
    assert review["recent_source_count"] == 2
    assert "L-7781" in {node["name"]
                         for node in review["graph"]["entities"]}
    assert "Weekly review - 2026-11-22 09:00 (last 7 days)" in review["text"]
    assert "Completed this week" in review["text"]
    assert ("Tags: 2 distinct | work (1 open, 1 done) | "
            "finance (0 open, 1 done)") in review["text"]
    assert "archive" not in review["text"]
    assert "Recent sources: 2" in review["text"]
    assert review["next_week"]
    assert review["weights_changed"] is False
    docs.close()


def test_weekly_review_tag_filter_and_archive(tmp_path):
    now = datetime.now()
    stamp = now.strftime("%Y-%m-%dT%H:%M:%S")
    old = (now - timedelta(days=10)).strftime("%Y-%m-%dT%H:%M:%S")
    session = {"tasks": [
        {"id": 1, "text": "write report", "status": "open",
         "created": stamp, "completed": "",
         "tags": ["work", "home"]},
        {"id": 2, "text": "pay invoice", "status": "done",
         "created": stamp, "completed": stamp,
         "tags": ["work", "finance"]},
        {"id": 3, "text": "archive task", "status": "done",
         "created": old, "completed": old, "tags": ["archive"]},
        {"id": 4, "text": "buy milk", "status": "open",
         "created": stamp, "completed": "", "tags": ["home"]},
    ]}
    review = build_weekly_review(
        session, None, None, days=7, tag="work",
        now="2026-11-22 09:00")
    assert review["tag"] == "work"
    assert review["task_counts"] == {
        "open": 1, "completed_recent": 1, "completed_total": 1}
    assert review["tags"] == [
        {"tag": "work", "open": 1, "done": 1, "total": 2},
        {"tag": "finance", "open": 0, "done": 1, "total": 1},
        {"tag": "home", "open": 1, "done": 0, "total": 1},
    ]
    assert review["tag_count"] == 3
    assert review["tagged_task_count"] == 2
    assert review["untagged_task_count"] == 0
    assert [task["id"] for task in review["open_tasks"]] == [1]
    assert [task["id"] for task in review["completed_tasks"]] == [2]
    assert "| tag=work" in review["text"]
    assert "Tasks: 1 open | 1 completed this week" in review["text"]
    assert ("Tags for #work: 3 distinct | work (1 open, 1 done) | "
            "finance (0 open, 1 done) | home (1 open, 0 done)") in review["text"]
    assert "write report" in review["text"]
    assert "pay invoice" in review["text"]
    assert "buy milk" not in review["text"]
    assert "archive task" not in review["text"]

    markdown = render_weekly_markdown(review)
    assert markdown.startswith("# Weekly review - 2026-11-22 09:00 | tag=work")
    assert "- Open: 1" in markdown
    assert "- Completed in the last 7 days: 1" in markdown
    assert "- Completed total: 1" in markdown
    assert "## Tags for #work (3 tags)" in markdown
    assert "| work | 1 | 1 | 2 |" in markdown
    assert "| finance | 0 | 1 | 1 |" in markdown
    assert "| home | 1 | 0 | 1 |" in markdown
    assert "- Tagged tasks: 2" in markdown
    assert "0 untagged" not in markdown
    assert "buy milk" not in markdown
    assert "archive task" not in markdown

    report = tmp_path / "weekly.md"
    report.write_text(markdown, encoding="utf-8")
    archive = WeeklyReportArchive(tmp_path / "weekly_reports.json")
    entry = archive.record(report, review)
    assert entry["tag"] == "work"
    assert "| tag=work" in entry["title"]
    listing = render_weekly_report_index(
        archive.list(), archive.path)
    assert "tag=work" in listing
    assert archive.count(query="work") == 1


def test_cli_weekly_review(tmp_path, capsys):
    session_path = tmp_path / "session.json"
    now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    session_path.write_text(json.dumps({"tasks": [
        {"id": 1, "text": "open task", "status": "open",
         "created": now, "completed": "", "tags": ["work"]},
        {"id": 2, "text": "done task", "status": "done",
         "created": now, "completed": now, "tags": ["home"]},
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
    assert review["tag"] == "all"

    assert main([
        "--session", str(session_path), "--docs", str(tmp_path / "docs.db"),
        "--gaps", str(gaps_path), "--weekly", "--days", "7",
        "--weekly-tag", "work", "--json"]) == 0
    filtered = json.loads(capsys.readouterr().out)
    assert filtered["tag"] == "work"
    assert filtered["task_counts"] == {
        "open": 1, "completed_recent": 0, "completed_total": 0}
    assert [task["id"] for task in filtered["open_tasks"]] == [1]
    assert filtered["tags"] == [
        {"tag": "work", "open": 1, "done": 0, "total": 1}]
    assert "| tag=work" in filtered["text"]
    assert "Tags for #work" in filtered["text"]

    assert main([
        "--session", str(session_path), "--docs", str(tmp_path / "docs.db"),
        "--gaps", str(gaps_path), "--weekly", "--weekly-tag",
        "bad tag!"]) == 1
    assert "letters, numbers" in capsys.readouterr().out
    with pytest.raises(SystemExit) as exc:
        main(["--session", str(session_path), "--weekly-tag", "work"])
    assert exc.value.code == 2


def test_cli_overdue_tasks(tmp_path, capsys):
    session_path = tmp_path / "session.json"
    overdue = (date.today() - timedelta(days=1)).isoformat()
    future = (date.today() + timedelta(days=1)).isoformat()
    session_path.write_text(json.dumps({"tasks": [
        {"id": 1, "text": "submit report", "status": "open",
         "created": "2026-11-01T09:00:00", "completed": "",
         "due": overdue},
        {"id": 2, "text": "plan launch", "status": "open",
         "created": "2026-11-01T09:00:00", "completed": "",
         "due": future},
        {"id": 3, "text": "done task", "status": "done",
         "created": "2026-11-01T09:00:00", "completed": overdue,
         "due": overdue},
    ]}), encoding="utf-8")
    common = [
        "--session", str(session_path),
        "--docs", str(tmp_path / "missing.db"),
        "--gaps", str(tmp_path / "missing-gaps.json")]
    assert main(common + ["--overdue", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["open_task_count"] == 2
    assert payload["overdue_task_count"] == 1
    assert payload["overdue_tasks"][0]["text"] == "submit report"
    assert main(common + ["--overdue"]) == 0
    output = capsys.readouterr().out
    assert "Overdue tasks (1 of 2 open)" in output
    assert "submit report | due" in output
    assert "plan launch" not in output
    with pytest.raises(SystemExit) as exc:
        main(common + ["--overdue", "--weekly"])
    assert exc.value.code == 2


def test_cli_due_soon_tasks(tmp_path, capsys):
    session_path = tmp_path / "session.json"
    today = date.today()
    due_soon = (today + timedelta(days=2)).isoformat()
    later = (today + timedelta(days=10)).isoformat()
    overdue = (today - timedelta(days=1)).isoformat()
    session_path.write_text(json.dumps({"tasks": [
        {"id": 1, "text": "submit report", "status": "open",
         "created": "2026-11-01T09:00:00", "completed": "",
         "due": due_soon},
        {"id": 2, "text": "plan launch", "status": "open",
         "created": "2026-11-01T09:00:00", "completed": "",
         "due": later},
        {"id": 3, "text": "late task", "status": "open",
         "created": "2026-11-01T09:00:00", "completed": "",
         "due": overdue},
    ]}), encoding="utf-8")
    common = [
        "--session", str(session_path),
        "--docs", str(tmp_path / "missing.db"),
        "--gaps", str(tmp_path / "missing-gaps.json"),
        "--due-soon", "--due-days", "7"]
    assert main(common + ["--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["days"] == 7
    assert payload["open_task_count"] == 3
    assert payload["due_soon_count"] == 1
    assert payload["due_soon_tasks"][0]["text"] == "submit report"
    assert main(common) == 0
    output = capsys.readouterr().out
    assert "Due soon (1 of 3 open, next 7 days)" in output
    assert "submit report | due" in output
    assert "plan launch" not in output
    assert "late task" not in output
    with pytest.raises(SystemExit) as exc:
        main(["--due-soon", "--due-days", "0"])
    assert exc.value.code == 2


def test_overdue_and_due_soon_tag_filters(tmp_path, capsys):
    today = date.today()
    overdue_work = (today - timedelta(days=1)).isoformat()
    overdue_home = (today - timedelta(days=2)).isoformat()
    soon_work = (today + timedelta(days=2)).isoformat()
    soon_home = (today + timedelta(days=1)).isoformat()
    session = {"tasks": [
        {"id": 1, "text": "work overdue", "status": "open",
         "due": overdue_work, "tags": ["work"]},
        {"id": 2, "text": "home overdue", "status": "open",
         "due": overdue_home, "tags": ["home"]},
        {"id": 3, "text": "work soon", "status": "open",
         "due": soon_work, "tags": ["work"]},
        {"id": 4, "text": "home soon", "status": "open",
         "due": soon_home, "tags": ["home"]},
        {"id": 5, "text": "work undated", "status": "open",
         "due": "", "tags": ["work"]},
    ]}
    now = today.isoformat()
    overdue = overdue_view(session, now=now, tag="work")
    assert overdue["tag"] == "work"
    assert overdue["open_task_count"] == 3
    assert overdue["overdue_task_count"] == 1
    assert [task["id"] for task in overdue["overdue_tasks"]] == [1]
    overdue_text = render_overdue_view(overdue)
    assert "| tag=work" in overdue_text
    assert "work overdue" in overdue_text
    assert "home overdue" not in overdue_text
    assert overdue_view(session, now=now)["overdue_task_count"] == 2
    assert overdue_view(session, now=now)["open_task_count"] == 5

    due = due_soon_view(session, days=7, now=now, tag="work")
    assert due["tag"] == "work"
    assert due["open_task_count"] == 3
    assert due["due_soon_count"] == 1
    assert [task["id"] for task in due["due_soon_tasks"]] == [3]
    due_text = render_due_soon_view(due)
    assert "| tag=work" in due_text
    assert "work soon" in due_text
    assert "home soon" not in due_text
    unfiltered_due = due_soon_view(session, days=7, now=now)
    assert unfiltered_due["open_task_count"] == 5
    assert unfiltered_due["due_soon_count"] == 2

    session_path = tmp_path / "session.json"
    session_path.write_text(json.dumps(session), encoding="utf-8")
    common = [
        "--session", str(session_path),
        "--docs", str(tmp_path / "missing.db"),
        "--gaps", str(tmp_path / "missing-gaps.json")]
    assert main(common + [
        "--overdue", "--overdue-tag", "work",
        "--task-sort", "priority", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["tag"] == "work"
    assert payload["sort"] == "priority"
    assert payload["open_task_count"] == 3
    assert payload["overdue_task_count"] == 1
    assert [task["id"] for task in payload["overdue_tasks"]] == [1]
    assert main(common + ["--overdue", "--overdue-tag", "work"]) == 0
    text = capsys.readouterr().out
    assert "| tag=work" in text
    assert "work overdue" in text
    assert "home overdue" not in text

    assert main(common + [
        "--due-soon", "--due-days", "7", "--due-soon-tag", "work",
        "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["tag"] == "work"
    assert payload["days"] == 7
    assert payload["open_task_count"] == 3
    assert payload["due_soon_count"] == 1
    assert [task["id"] for task in payload["due_soon_tasks"]] == [3]
    assert main(common + [
        "--due-soon", "--due-days", "7", "--due-soon-tag", "work"]) == 0
    text = capsys.readouterr().out
    assert "| tag=work" in text
    assert "work soon" in text
    assert "home soon" not in text

    assert main(common + [
        "--overdue", "--overdue-tag", "bad tag!"]) == 1
    assert "letters, numbers" in capsys.readouterr().out
    assert main(common + [
        "--due-soon", "--due-soon-tag", "bad tag!"]) == 1
    assert "letters, numbers" in capsys.readouterr().out
    with pytest.raises(SystemExit) as exc:
        main(common + ["--overdue-tag", "work"])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        main(common + ["--due-soon-tag", "work"])
    assert exc.value.code == 2


def test_cli_search_tasks(tmp_path, capsys):
    session_path = tmp_path / "session.json"
    today = date.today()
    soon = (today + timedelta(days=2)).isoformat()
    overdue = (today - timedelta(days=1)).isoformat()
    session_path.write_text(json.dumps({"tasks": [
        {"id": 1, "text": "Submit quarterly report", "status": "open",
         "created": "2026-11-01T09:00:00", "completed": "",
         "due": soon, "priority": "high", "tags": ["work"]},
        {"id": 2, "text": "submit expenses", "status": "done",
         "created": "2026-11-01T09:00:00", "completed": "2026-11-02",
         "due": "2000-01-01", "priority": "normal",
         "tags": ["finance"]},
        {"id": 3, "text": "plan launch", "status": "open",
         "created": "2026-11-01T09:00:00", "completed": "",
         "priority": "low", "tags": ["work"]},
        {"id": 4, "text": "submit late report", "status": "open",
         "created": "2026-11-01T09:00:00", "completed": "",
         "due": overdue, "priority": "high", "tags": ["work"]},
    ]}), encoding="utf-8")
    common = [
        "--session", str(session_path),
        "--docs", str(tmp_path / "missing.db"),
        "--gaps", str(tmp_path / "missing-gaps.json")]
    assert main(common + ["--search-tasks", "SUBMIT", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["query"] == "SUBMIT"
    assert payload["status"] == "all"
    assert payload["due"] == "all"
    assert payload["priority"] == "all"
    assert payload["tag"] == "all"
    assert payload["count"] == 3
    assert [task["id"] for task in payload["tasks"]] == [1, 2, 4]
    assert payload["open_task_count"] == 3

    assert main(common + ["--search-tasks", "submit"]) == 0
    output = capsys.readouterr().out
    assert 'Task search "submit" (3 match(es))' in output
    assert (f"- 1 [open] Submit quarterly report [high] [tags: work] "
            f"(due {soon})") in output
    assert ("- 2 [done] submit expenses [tags: finance] "
            "(due 2000-01-01)") in output
    assert (f"- 4 [open] submit late report [high] [tags: work] "
            f"(due {overdue} [overdue])") in output
    assert "plan launch" not in output

    assert main(common + [
        "--search-tasks", "submit", "--search-status", "open", "--json"]) == 0
    open_only = json.loads(capsys.readouterr().out)
    assert open_only["status"] == "open"
    assert [task["id"] for task in open_only["tasks"]] == [1, 4]
    assert main(common + [
        "--search-tasks", "submit", "--search-status", "done", "--json"]) == 0
    done_only = json.loads(capsys.readouterr().out)
    assert [task["id"] for task in done_only["tasks"]] == [2]

    assert main(common + [
        "--search-tasks", "submit", "--search-due", "week", "--json"]) == 0
    week = json.loads(capsys.readouterr().out)
    assert week["due"] == "week"
    assert [task["id"] for task in week["tasks"]] == [1]
    assert main(common + [
        "--search-tasks", "submit", "--search-due", "overdue"]) == 0
    overdue_output = capsys.readouterr().out
    assert ('Task search "submit" (1 match(es)) | due=overdue'
            in overdue_output)
    assert "submit late report" in overdue_output
    assert main(common + [
        "--search-tasks", "submit", "--search-status", "open",
        "--search-due", "week", "--json"]) == 0
    combined = json.loads(capsys.readouterr().out)
    assert combined["status"] == "open"
    assert combined["due"] == "week"
    assert [task["id"] for task in combined["tasks"]] == [1]
    assert main(common + [
        "--search-tasks", "submit", "--search-status", "open",
        "--search-due", "week", "--search-priority", "high",
        "--json"]) == 0
    combined = json.loads(capsys.readouterr().out)
    assert combined["priority"] == "high"
    assert [task["id"] for task in combined["tasks"]] == [1]
    assert main(common + [
        "--search-tasks", "submit", "--search-tag", "work",
        "--json"]) == 0
    tagged = json.loads(capsys.readouterr().out)
    assert tagged["tag"] == "work"
    assert [task["id"] for task in tagged["tasks"]] == [1, 4]
    assert main(common + [
        "--search-tasks", "submit", "--search-status", "open",
        "--search-due", "week", "--search-priority", "high",
        "--search-tag", "work", "--json"]) == 0
    fully_filtered = json.loads(capsys.readouterr().out)
    assert [task["id"] for task in fully_filtered["tasks"]] == [1]
    assert main(common + [
        "--search-tasks", "submit", "--search-priority", "high",
        "--json"]) == 0
    high = json.loads(capsys.readouterr().out)
    assert high["priority"] == "high"
    assert [task["id"] for task in high["tasks"]] == [1, 4]
    assert main(common + [
        "--search-tasks", "launch", "--search-priority", "low",
        "--json"]) == 0
    low = json.loads(capsys.readouterr().out)
    assert [task["id"] for task in low["tasks"]] == [3]
    assert main(common + [
        "--search-tasks", "submit", "--task-sort", "priority",
        "--json"]) == 0
    priority_sorted = json.loads(capsys.readouterr().out)
    assert priority_sorted["sort"] == "priority"
    assert [task["id"] for task in priority_sorted["tasks"]] == [4, 1, 2]
    assert main(common + [
        "--search-tasks", "submit", "--task-sort", "priority"]) == 0
    sorted_output = capsys.readouterr().out
    assert '| sort=priority' in sorted_output
    assert (sorted_output.index("submit late report")
            < sorted_output.index("Submit quarterly report"))
    assert main(common + [
        "--search-tasks", "launch", "--search-due", "none", "--json"]) == 0
    undated = json.loads(capsys.readouterr().out)
    assert [task["id"] for task in undated["tasks"]] == [3]

    assert main(common + [
        "--set-task-priority", "3", "high", "--json"]) == 0
    updated = json.loads(capsys.readouterr().out)
    assert updated["task"]["id"] == 3
    assert updated["task"]["priority"] == "high"
    assert main(common + [
        "--search-tasks", "launch", "--search-priority", "high",
        "--json"]) == 0
    persisted = json.loads(capsys.readouterr().out)
    assert [task["id"] for task in persisted["tasks"]] == [3]
    assert main(common + ["--set-task-priority", "3", "normal"]) == 0
    assert "Task 3 priority set to normal." in capsys.readouterr().out
    assert main(common + ["--set-task-priority", "99", "high"]) == 1
    assert "task 99 not found" in capsys.readouterr().out

    assert main(common + ["--search-tasks", "missing", "--json"]) == 0
    empty = json.loads(capsys.readouterr().out)
    assert empty["count"] == 0
    assert empty["tasks"] == []
    with pytest.raises(SystemExit) as exc:
        main(common + ["--search-tasks", " "])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        main(common + ["--search-tasks", "submit", "--weekly"])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        main(common + ["--search-tasks", "submit", "--search-status", "bad"])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        main(common + ["--search-tasks", "submit", "--search-due", "month"])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        main(common + [
            "--search-tasks", "submit", "--search-priority", "urgent"])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        main(common + ["--search-status", "open"])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        main(common + ["--search-priority", "high"])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        main(common + ["--search-tag", "work"])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        main(common + ["--set-task-priority", "1", "urgent"])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        main(common + ["--set-task-priority", "1", "high", "--weekly"])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        main(common + ["--task-sort", "priority", "--reports"])
    assert exc.value.code == 2


def test_cli_task_tags(tmp_path, capsys):
    session_path = tmp_path / "session.json"
    session_path.write_text(json.dumps({"tasks": [
        {"id": 1, "text": "write report", "status": "open",
         "tags": ["work"]},
        {"id": 2, "text": "send invoice", "status": "done",
         "tags": ["work", "finance"]},
        {"id": 3, "text": "read book", "status": "open", "tags": []},
        {"id": 4, "text": "plan budget", "status": "open",
         "tags": ["work", "finance"]},
    ]}), encoding="utf-8")
    common = [
        "--session", str(session_path),
        "--docs", str(tmp_path / "missing.db"),
        "--gaps", str(tmp_path / "missing-gaps.json")]
    assert main(common + ["--list-tags", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["tag_count"] == 2
    assert payload["task_count"] == 4
    assert payload["tagged_task_count"] == 3
    assert payload["untagged_task_count"] == 1
    assert payload["open_tagged"] == 2
    assert payload["done_tagged"] == 1
    assert payload["tags"] == [
        {"tag": "work", "open": 2, "done": 1, "total": 3},
        {"tag": "finance", "open": 1, "done": 1, "total": 2},
    ]
    assert payload["weights_changed"] is False

    assert main(common + ["--list-tags"]) == 0
    output = capsys.readouterr().out
    assert ("Task tags: 2 distinct | 3 tagged task(s) "
            "(2 open, 1 done) | 1 untagged") in output
    assert "- work: 2 open, 1 done, 3 total" in output
    assert "- finance: 1 open, 1 done, 2 total" in output

    empty_path = tmp_path / "empty.json"
    empty_path.write_text(json.dumps({"tasks": [
        {"id": 1, "text": "no tags", "status": "open", "tags": []},
    ]}), encoding="utf-8")
    assert main(["--session", str(empty_path), "--list-tags"]) == 0
    assert "Task tags: none" in capsys.readouterr().out

    with pytest.raises(SystemExit) as exc:
        main(common + ["--list-tags", "--weekly"])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        main(common + ["--list-tags", "--task-list"])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        main(common + ["--list-tags", "--task-sort", "priority"])
    assert exc.value.code == 2


def test_cli_rename_tag(tmp_path, capsys):
    session_path = tmp_path / "session.json"
    session_path.write_text(json.dumps({"tasks": [
        {"id": 1, "text": "write report", "status": "open",
         "tags": ["work", "home"]},
        {"id": 2, "text": "pay invoice", "status": "done",
         "tags": ["work", "finance"]},
        {"id": 3, "text": "read book", "status": "open", "tags": []},
    ]}), encoding="utf-8")
    common = [
        "--session", str(session_path),
        "--docs", str(tmp_path / "missing.db"),
        "--gaps", str(tmp_path / "missing-gaps.json")]

    assert main(common + ["--rename-tag", "work", "job"]) == 0
    preview = capsys.readouterr().out
    assert ("Would rename tag 'work' to 'job' on 2 task(s) "
            "(1 open, 1 done)") in preview
    assert "rerun with --confirm-rename-tag to apply" in preview
    assert json.loads(session_path.read_text(encoding="utf-8"))[
        "tasks"][0]["tags"] == ["work", "home"]

    assert main(common + [
        "--rename-tag", "work", "job", "--json"]) == 0
    preview_json = json.loads(capsys.readouterr().out)
    assert preview_json["applied"] is False
    assert preview_json["old"] == "work"
    assert preview_json["new"] == "job"
    assert preview_json["would_update"] == 2
    assert preview_json["open"] == 1
    assert preview_json["done"] == 1

    assert main(common + [
        "--rename-tag", "work", "job", "--confirm-rename-tag",
        "--json"]) == 0
    applied = json.loads(capsys.readouterr().out)
    assert applied["applied"] is True
    assert applied["old"] == "work"
    assert applied["new"] == "job"
    assert applied["updated"] == 2
    assert applied["task_ids"] == [1, 2]
    saved = json.loads(session_path.read_text(encoding="utf-8"))
    assert saved["tasks"][0]["tags"] == ["job", "home"]
    assert saved["tasks"][1]["tags"] == ["job", "finance"]
    assert saved["tasks"][2]["tags"] == []
    assert saved["tasks"][0]["renamed"]

    assert main(common + ["--list-tags", "--json"]) == 0
    tags = json.loads(capsys.readouterr().out)
    assert [item["tag"] for item in tags["tags"]] == [
        "job", "finance", "home"]

    assert main(common + ["--rename-tag", "missing", "job"]) == 1
    assert "tag 'missing' is not used" in capsys.readouterr().out
    assert main(common + ["--rename-tag", "job", "job"]) == 1
    assert "must be different" in capsys.readouterr().out
    assert main(common + ["--rename-tag", "job", "bad tag!"]) == 1
    assert "letters, numbers" in capsys.readouterr().out
    with pytest.raises(SystemExit) as exc:
        main(common + ["--confirm-rename-tag"])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        main(common + ["--rename-tag", "job", "office", "--weekly"])
    assert exc.value.code == 2

    deduped = {"tasks": [
        {"id": 7, "text": "mixed tags", "status": "open",
         "tags": ["work", "job", "work"]},
    ]}
    result = rename_task_tag(deduped, "work", "job")
    assert result["updated"] == 1
    assert result["task_ids"] == [7]
    assert deduped["tasks"][0]["tags"] == ["job"]


def test_cli_delete_tag(tmp_path, capsys):
    session_path = tmp_path / "session.json"
    session_path.write_text(json.dumps({"tasks": [
        {"id": 1, "text": "write report", "status": "open",
         "tags": ["work", "home"]},
        {"id": 2, "text": "pay invoice", "status": "done",
         "tags": ["work", "finance"]},
        {"id": 3, "text": "read book", "status": "open",
         "tags": ["home"]},
    ]}), encoding="utf-8")
    common = [
        "--session", str(session_path),
        "--docs", str(tmp_path / "missing.db"),
        "--gaps", str(tmp_path / "missing-gaps.json")]

    assert main(common + ["--delete-tag", "work"]) == 0
    preview = capsys.readouterr().out
    assert ("Would delete tag 'work' from 2 task(s) "
            "(1 open, 1 done)") in preview
    assert "rerun with --confirm-delete-tag" in preview
    assert json.loads(session_path.read_text(encoding="utf-8"))[
        "tasks"][0]["tags"] == ["work", "home"]

    assert main(common + ["--delete-tag", "work", "--json"]) == 0
    preview_json = json.loads(capsys.readouterr().out)
    assert preview_json["applied"] is False
    assert preview_json["tag"] == "work"
    assert preview_json["would_update"] == 2
    assert preview_json["open"] == 1
    assert preview_json["done"] == 1

    assert main(common + [
        "--delete-tag", "work", "--confirm-delete-tag", "--json"]) == 0
    applied = json.loads(capsys.readouterr().out)
    assert applied["applied"] is True
    assert applied["tag"] == "work"
    assert applied["updated"] == 2
    assert applied["task_ids"] == [1, 2]
    saved = json.loads(session_path.read_text(encoding="utf-8"))
    assert saved["tasks"][0]["tags"] == ["home"]
    assert saved["tasks"][1]["tags"] == ["finance"]
    assert saved["tasks"][2]["tags"] == ["home"]
    assert saved["tasks"][0]["tag_removed"]

    assert main(common + ["--list-tags", "--json"]) == 0
    tags = json.loads(capsys.readouterr().out)
    assert tags["tag_count"] == 2
    assert [item["tag"] for item in tags["tags"]] == ["home", "finance"]

    assert main(common + ["--undo-delete-tag", "--json"]) == 0
    undone = json.loads(capsys.readouterr().out)
    assert undone["tag"] == "work"
    assert undone["restored"] == 2
    assert undone["task_ids"] == [1, 2]
    assert undone["already"] == []
    assert undone["skipped"] == []
    saved = json.loads(session_path.read_text(encoding="utf-8"))
    assert saved["tasks"][0]["tags"] == ["work", "home"]
    assert saved["tasks"][1]["tags"] == ["work", "finance"]
    assert saved["tasks"][2]["tags"] == ["home"]
    assert "tag_delete_undo" not in saved
    assert main(common + ["--undo-delete-tag"]) == 1
    assert "no tag delete to undo" in capsys.readouterr().out
    with pytest.raises(SystemExit) as exc:
        main(common + ["--undo-delete-tag", "--weekly"])
    assert exc.value.code == 2

    assert main(common + ["--delete-tag", "missing"]) == 1
    assert "tag 'missing' is not used" in capsys.readouterr().out
    assert main(common + ["--delete-tag", "bad tag!"]) == 1
    assert "letters, numbers" in capsys.readouterr().out
    with pytest.raises(SystemExit) as exc:
        main(common + ["--confirm-delete-tag"])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        main(common + ["--delete-tag", "home", "--weekly"])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        main(common + ["--delete-tag", "home", "--rename-tag", "a", "b"])
    assert exc.value.code == 2


def test_tag_audit_and_weekly_archive(tmp_path, capsys):
    session = {"tasks": [
        {"id": 1, "text": "write report", "status": "open",
         "tags": ["work", "home"]},
        {"id": 2, "text": "pay invoice", "status": "done",
         "tags": ["work"]},
        {"id": 3, "text": "read book", "status": "open", "tags": []},
    ]}
    delete_task_tag(session, "work")
    undo_delete_task_tag(session)
    view = task_tag_audit_view(session, limit=5)
    assert view["count"] == 2
    assert view["total"] == 2
    assert [event["action"] for event in view["events"]] == [
        "undo_tag_delete", "delete_tag"]
    assert view["events"][0]["tag"] == "work"
    assert view["events"][0]["restored"] == 2
    assert view["events"][1]["updated"] == 2
    assert "Tag audit (2 of 2 event(s)):" in view["text"]
    assert "undo delete tag 'work' | restored 2 task(s)" in view["text"]
    assert ("delete tag 'work' | 2 task(s) "
            "(1 open, 1 done)") in view["text"]
    limited = task_tag_audit_view(session, limit=1)
    assert limited["count"] == 1
    assert limited["total"] == 2
    assert limited["events"][0]["action"] == "undo_tag_delete"
    tagged_view = task_tag_audit_view(session, tag="work")
    assert tagged_view["tag"] == "work"
    assert tagged_view["count"] == 2
    assert tagged_view["total"] == 2
    assert "| tag=work:" in tagged_view["text"]
    delete_view = task_tag_audit_view(session, action="delete_tag")
    assert delete_view["action"] == "delete_tag"
    assert delete_view["count"] == 1
    assert delete_view["total"] == 1
    assert delete_view["events"][0]["action"] == "delete_tag"
    assert "| action=delete_tag:" in delete_view["text"]
    undo_view = task_tag_audit_view(session, action="undo")
    assert undo_view["action"] == "undo_tag_delete"
    assert undo_view["count"] == 1
    assert task_tag_audit_view(
        session, tag="home")["total"] == 0
    assert task_tag_audit_view(
        session, tag="work", action="undo_tag_delete")["count"] == 1

    review = build_weekly_review(session, None, None, days=7)
    assert review["tag_audit_total"] == 2
    assert [event["action"] for event in review["tag_audit"]] == [
        "undo_tag_delete", "delete_tag"]
    markdown = render_weekly_markdown(review)
    assert "## Tag audit (2)" in markdown
    assert "undo delete tag 'work' | restored 2 task(s)" in markdown
    assert ("delete tag 'work' | 2 task(s) "
            "(1 open, 1 done)") in markdown
    filtered_review = build_weekly_review(
        session, None, None, days=7, tag="work",
        audit_action="delete_tag")
    assert filtered_review["tag_audit_total"] == 1
    assert filtered_review["tag_audit_action"] == "delete_tag"
    assert [event["action"]
            for event in filtered_review["tag_audit"]] == ["delete_tag"]
    assert ("Tag audit: 1 event(s) | tag=work | action=delete_tag"
            in filtered_review["text"])
    filtered_markdown = render_weekly_markdown(filtered_review)
    assert ("## Tag audit (1) | tag=work | action=delete_tag"
            in filtered_markdown)
    assert ("delete tag 'work' | 2 task(s) "
            "(1 open, 1 done)") in filtered_markdown
    assert "undo delete tag 'work'" not in filtered_markdown

    brief = build_briefing(
        session, None, None, now="2026-11-23 10:00")
    assert brief["tag_audit_total"] == 2
    assert [event["action"] for event in brief["tag_audit"]] == [
        "undo_tag_delete", "delete_tag"]
    assert "Tag audit: 2 event(s)" in brief["text"]
    assert ("undo delete tag 'work' | restored 2 task(s)"
            in brief["text"])
    assert ("delete tag 'work' | 2 task(s) "
            "(1 open, 1 done)") in brief["text"]
    home_brief = build_briefing(
        session, None, None, now="2026-11-23 10:00", tag="home")
    assert home_brief["tag_audit_total"] == 0
    assert "Tag audit" not in home_brief["text"]

    session_path = tmp_path / "session.json"
    session_path.write_text(json.dumps(session), encoding="utf-8")
    common = [
        "--session", str(session_path),
        "--docs", str(tmp_path / "missing.db"),
        "--gaps", str(tmp_path / "missing-gaps.json")]
    assert main(common + ["--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["tag_audit_total"] == 2
    assert payload["tag_audit"][0]["action"] == "undo_tag_delete"
    assert main(common) == 0
    assert "Tag audit: 2 event(s)" in capsys.readouterr().out
    assert main(common + ["--brief-tag", "home", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["tag_audit_total"] == 0
    assert payload["tag_audit"] == []

    assert main(common + ["--tag-audit", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["count"] == 2
    assert payload["total"] == 2
    assert payload["events"][0]["action"] == "undo_tag_delete"
    assert main(common + [
        "--tag-audit", "--tag-audit-limit", "1", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["count"] == 1
    assert payload["total"] == 2
    assert main(common + ["--tag-audit"]) == 0
    assert "Tag audit (2 of 2 event(s))" in capsys.readouterr().out
    assert main(common + [
        "--tag-audit", "--tag-audit-tag", "work",
        "--tag-audit-action", "delete_tag", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["tag"] == "work"
    assert payload["action"] == "delete_tag"
    assert payload["count"] == 1
    assert payload["total"] == 1
    assert main(common + [
        "--tag-audit", "--tag-audit-tag", "home", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["count"] == 0
    assert payload["total"] == 0
    assert main(common + [
        "--tag-audit", "--tag-audit-tag", "work",
        "--tag-audit-action", "delete_tag",
        "--tag-audit-format", "csv"]) == 0
    csv_output = capsys.readouterr().out
    assert csv_output.startswith("ts,action,tag,task_ids")
    assert "delete_tag,work" in csv_output
    assert "undo_tag_delete" not in csv_output
    assert main(common + [
        "--tag-audit", "--tag-audit-tag", "work",
        "--tag-audit-action", "delete_tag",
        "--tag-audit-format", "markdown"]) == 0
    audit_markdown = capsys.readouterr().out
    assert audit_markdown.startswith("# Tag audit - ")
    assert "- Tag filter: work" in audit_markdown
    assert "- Action filter: delete_tag" in audit_markdown
    assert "| delete_tag | work |" in audit_markdown
    assert "| undo_tag_delete | work |" not in audit_markdown
    with pytest.raises(SystemExit) as exc:
        main(common + ["--tag-audit-format", "csv"])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        main(common + [
            "--tag-audit", "--tag-audit-format", "csv", "--json"])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        main(common + ["--tag-audit-tag", "work"])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        main(common + ["--tag-audit-action", "delete_tag"])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        main(common + ["--tag-audit", "--tag-audit-tag", "bad tag!"])
    assert exc.value.code == 2
    assert main(common + ["--weekly", "--markdown"]) == 0
    assert "## Tag audit (2)" in capsys.readouterr().out
    assert main(common + [
        "--weekly", "--weekly-tag", "work",
        "--weekly-audit-action", "delete_tag", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["tag_audit_total"] == 1
    assert payload["tag_audit_action"] == "delete_tag"
    assert [event["action"] for event in payload["tag_audit"]] == [
        "delete_tag"]
    assert ("Tag audit: 1 event(s) | tag=work | action=delete_tag"
            in payload["text"])
    assert main(common + [
        "--weekly", "--weekly-tag", "work",
        "--weekly-audit-action", "delete_tag", "--markdown"]) == 0
    weekly_markdown = capsys.readouterr().out
    assert ("## Tag audit (1) | tag=work | action=delete_tag"
            in weekly_markdown)
    assert "undo delete tag 'work'" not in weekly_markdown
    with pytest.raises(SystemExit) as exc:
        main(common + ["--weekly-audit-action", "delete_tag"])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        main(common + ["--weekly", "--weekly-audit-action", "bad"])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        main(common + ["--tag-audit-limit", "2"])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        main(common + ["--tag-audit", "--list-tags"])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        main(common + ["--tag-audit", "--tag-audit-limit", "0"])
    assert exc.value.code == 2


def test_cli_priority_sort_views(tmp_path, capsys):
    session_path = tmp_path / "session.json"
    today = date.today()
    session_path.write_text(json.dumps({"tasks": [
        {"id": 1, "text": "low overdue task", "status": "open",
         "created": "2026-11-01T09:00:00", "completed": "",
         "due": (today - timedelta(days=1)).isoformat(),
         "priority": "low"},
        {"id": 2, "text": "high overdue task", "status": "open",
         "created": "2026-11-01T09:00:00", "completed": "",
         "due": (today - timedelta(days=2)).isoformat(),
         "priority": "high"},
        {"id": 3, "text": "normal soon task", "status": "open",
         "created": "2026-11-01T09:00:00", "completed": "",
         "due": (today + timedelta(days=2)).isoformat(),
         "priority": "normal"},
        {"id": 4, "text": "high soon task", "status": "open",
         "created": "2026-11-01T09:00:00", "completed": "",
         "due": (today + timedelta(days=1)).isoformat(),
         "priority": "high"},
    ]}), encoding="utf-8")
    common = [
        "--session", str(session_path),
        "--docs", str(tmp_path / "missing.db"),
        "--gaps", str(tmp_path / "missing-gaps.json")]
    assert main(common + [
        "--overdue", "--task-sort", "priority", "--json"]) == 0
    overdue = json.loads(capsys.readouterr().out)
    assert overdue["sort"] == "priority"
    assert [task["id"] for task in overdue["overdue_tasks"]] == [2, 1]
    assert main(common + [
        "--due-soon", "--due-days", "7", "--task-sort", "priority",
        "--json"]) == 0
    due_soon = json.loads(capsys.readouterr().out)
    assert due_soon["sort"] == "priority"
    assert [task["id"] for task in due_soon["due_soon_tasks"]] == [4, 3]
    assert main(common + ["--overdue", "--task-sort", "priority"]) == 0
    output = capsys.readouterr().out
    assert "sort=priority" in output
    assert output.index("high overdue task") < output.index("low overdue task")
    assert main(common + ["--task-sort", "priority", "--json"]) == 0
    brief = json.loads(capsys.readouterr().out)
    assert brief["sort"] == "priority"
    assert [task["id"] for task in brief["tasks"]] == [2, 4, 3, 1]
    assert brief["focus"].startswith("overdue task 2")
    assert "sort=priority" in brief["text"]
    assert main(common + [
        "--weekly", "--task-sort", "priority", "--json"]) == 0
    review = json.loads(capsys.readouterr().out)
    assert review["sort"] == "priority"
    assert [task["id"] for task in review["open_tasks"]] == [2, 4, 3, 1]
    assert "sort=priority" in review["text"]
    assert main(common + [
        "--weekly", "--markdown", "--task-sort", "priority"]) == 0
    markdown = capsys.readouterr().out
    assert markdown.startswith("# Weekly review - ")
    assert "sort=priority" in markdown
    assert markdown.index("high overdue task") < \
        markdown.index("low overdue task")
    with pytest.raises(SystemExit) as exc:
        main(common + ["--task-sort", "priority", "--reports"])
    assert exc.value.code == 2


def test_cli_task_list_export(tmp_path, capsys):
    session_path = tmp_path / "session.json"
    today = date.today()
    soon = (today + timedelta(days=1)).isoformat()
    overdue = (today - timedelta(days=1)).isoformat()
    later = (today + timedelta(days=2)).isoformat()
    now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    session_path.write_text(json.dumps({"tasks": [
        {"id": 1, "text": "high soon task", "status": "open",
         "created": now, "completed": "", "due": soon,
         "priority": "high", "tags": ["work", "finance"]},
        {"id": 2, "text": "low overdue task", "status": "open",
         "created": now, "completed": "", "due": overdue,
         "priority": "low", "tags": ["home"]},
        {"id": 3, "text": "normal soon task", "status": "open",
         "created": now, "completed": "", "due": later,
         "tags": ["work"]},
        {"id": 4, "text": "done task", "status": "done",
         "created": now, "completed": now, "priority": "high",
         "tags": ["done"]},
    ]}), encoding="utf-8")
    common = [
        "--session", str(session_path),
        "--docs", str(tmp_path / "missing.db"),
        "--gaps", str(tmp_path / "missing-gaps.json")]
    assert main(common + [
        "--task-list", "--task-status", "open", "--task-due", "week",
        "--task-priority", "high", "--task-tag", "work",
        "--task-sort", "priority", "--task-format", "csv"]) == 0
    output = capsys.readouterr().out
    rows = list(csv.reader(io.StringIO(output)))
    assert rows[0] == [
        "id", "status", "priority", "tags", "text", "due", "overdue",
        "created", "completed"]
    assert len(rows) == 2
    assert rows[1][:5] == [
        "1", "open", "high", "work,finance", "high soon task"]
    assert rows[1][5] == soon
    assert rows[1][6] == "false"

    assert main(common + [
        "--task-list", "--task-status", "all",
        "--task-sort", "priority", "--task-format", "markdown"]) == 0
    markdown = capsys.readouterr().out
    assert markdown.startswith("# Task list - ")
    assert "- Status: all" in markdown
    assert "- Count: 4" in markdown
    assert "| ID | Status | Priority | Tags | Task | Due | Overdue |" in markdown
    assert ("| 1 | open | high | work,finance | high soon task |"
            in markdown)
    assert (markdown.index("high soon task")
            < markdown.index("low overdue task"))

    assert main(common + [
        "--task-list", "--task-priority", "high", "--json"]) == 0
    priority = json.loads(capsys.readouterr().out)
    assert priority["status"] == "open"
    assert priority["priority"] == "high"
    assert [task["id"] for task in priority["tasks"]] == [1]

    assert main(common + [
        "--task-list", "--task-status", "done", "--json"]) == 0
    done = json.loads(capsys.readouterr().out)
    assert [task["id"] for task in done["tasks"]] == [4]

    assert main(common + [
        "--task-list", "--task-tag", "work", "--json"]) == 0
    tagged = json.loads(capsys.readouterr().out)
    assert tagged["tag"] == "work"
    assert [task["id"] for task in tagged["tasks"]] == [1, 3]

    with pytest.raises(SystemExit) as exc:
        main(common + ["--task-format", "csv"])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        main(common + ["--task-tag", "work"])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        main(common + ["--task-list", "--weekly"])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        main(common + [
            "--task-list", "--json", "--task-format", "csv"])
    assert exc.value.code == 2


def test_cli_weekly_markdown_export(tmp_path, capsys):
    session_path = tmp_path / "session.json"
    now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    session_path.write_text(json.dumps({"tasks": [
        {"id": 1, "text": "buy milk", "status": "done",
         "created": now, "completed": now, "tags": ["home"]},
        {"id": 2, "text": "write report", "status": "open",
         "created": now, "completed": "", "tags": ["work", "home"]},
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
    assert "## Tags (2 tags, 0 untagged)" in markdown
    assert "| home | 1 | 1 | 2 |" in markdown
    assert "| work | 1 | 0 | 1 |" in markdown
    assert "- Tagged tasks: 2" in markdown
    assert "## Pending gaps (1)" in markdown
    assert "What is the warranty deadline?" in markdown
    assert "## Recent sources (1)" in markdown
    assert "**Launch** - `notes/launch.txt` (1 chunk(s))" in markdown
    assert "## Graph" in markdown
    assert "L-7781" in markdown
    assert "## Next week" in markdown
    assert "No model weights were changed." in markdown

    assert main([
        "--session", str(session_path), "--docs", str(tmp_path / "docs.db"),
        "--gaps", str(gaps_path), "--weekly", "--days", "7",
        "--weekly-tag", "home", "--markdown"]) == 0
    filtered = capsys.readouterr().out
    assert filtered.splitlines()[0].endswith("| tag=home")
    assert "## Tags for #home (2 tags)" in filtered
    assert "| home | 1 | 1 | 2 |" in filtered
    assert "| work | 1 | 0 | 1 |" in filtered
    assert "- Tagged tasks: 2" in filtered
    assert "0 untagged" not in filtered
    assert "- [ ] **#2** write report" in filtered
    assert "- [x] **#1** buy milk" in filtered


def test_cli_weekly_markdown_out_file(tmp_path, capsys):
    session_path = tmp_path / "session.json"
    now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    session_path.write_text(json.dumps({"tasks": [
        {"id": 1, "text": "raporu yaz", "status": "open",
         "created": now, "completed": "", "tags": ["work"]},
    ]}), encoding="utf-8")
    docs = DocMemory(tmp_path / "docs.db")
    docs.add_text("notes/launch.txt", "The launch code is L-7781.",
                  title="Launch")
    docs.close()
    out = tmp_path / "reports" / "weekly.md"
    index_path = tmp_path / "reports" / "index.json"
    assert main([
        "--session", str(session_path), "--docs", str(tmp_path / "docs.db"),
        "--weekly", "--weekly-tag", "work", "--markdown", "--out", str(out),
        "--reports-index", str(index_path)]) == 0
    captured = capsys.readouterr()
    assert "weekly review written to" in captured.out
    assert "(archived w0001)" in captured.out
    assert out.exists()
    payload = out.read_bytes()
    assert not payload.startswith(b"\xef\xbb\xbf")
    markdown = payload.decode("utf-8")
    assert markdown.startswith("# Weekly review - ")
    assert "| tag=work" in markdown.splitlines()[0]
    assert "## Tags for #work (1 tag)" in markdown
    assert "| work | 1 | 0 | 1 |" in markdown
    assert "- [ ] **#1** raporu yaz" in markdown
    assert not list(out.parent.glob(f".{out.name}.*.tmp"))
    index = json.loads(index_path.read_text(encoding="utf-8"))
    assert index["reports"][0]["id"] == "w0001"
    assert index["reports"][0]["path"] == str(out.resolve())
    assert index["reports"][0]["period_days"] == 7
    assert index["reports"][0]["tag"] == "work"
    assert "| tag=work" in index["reports"][0]["title"]
    assert main(["--reports", "--reports-index", str(index_path)]) == 0
    listing = capsys.readouterr().out
    assert "Weekly report archive:" in listing
    assert "w0001" in listing
    assert "last 7 days" in listing
    assert "tag=work" in listing
    assert "1 open" in listing
    assert main(["--reports", "--json", "--reports-index",
                 str(index_path)]) == 0
    listing = json.loads(capsys.readouterr().out)
    assert listing["total"] == 1
    assert listing["data"][0]["exists"] is True
    assert main(["--reports", "--tag", "work", "--reports-index",
                 str(index_path)]) == 0
    listing = capsys.readouterr().out
    assert "tag: work" in listing
    assert "w0001" in listing
    assert "tag=work" in listing
    assert main(["--reports", "--tag", "work", "--json",
                 "--reports-index", str(index_path)]) == 0
    listing = json.loads(capsys.readouterr().out)
    assert listing["tag"] == "work"
    assert listing["matched"] == 1
    assert listing["data"][0]["id"] == "w0001"
    assert main(["--reports", "--tag", "home", "--reports-index",
                 str(index_path)]) == 0
    assert "No archived reports tagged 'home'." in capsys.readouterr().out
    with pytest.raises(SystemExit) as exc:
        main(["--tag", "work"])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        main(["--reports", "--tag", "work", "--prune",
              "--reports-index", str(index_path)])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        main(["--out", str(tmp_path / "invalid.md")])
    assert exc.value.code == 2


def test_cli_reports_prune_missing_files(tmp_path, capsys):
    index_path = tmp_path / "weekly_reports.json"
    present = tmp_path / "present.md"
    present.write_text("# Present\n", encoding="utf-8")
    missing = tmp_path / "missing.md"
    missing.write_text("# Missing\n", encoding="utf-8")
    archive = WeeklyReportArchive(index_path)
    kept = archive.record(present, {
        "generated": "2026-11-22 09:00", "period_days": 7,
        "task_counts": {"open": 0}, "recent_source_count": 1,
        "graph": {"entities": [], "connections": []},
    })
    archive.record(missing, {
        "generated": "2026-11-22 10:00", "period_days": 14,
        "task_counts": {"open": 1}, "recent_source_count": 0,
        "graph": {"entities": [], "connections": []},
    })
    missing.unlink()

    assert main(["--reports", "--prune", "--json", "--reports-index",
                 str(index_path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["removed"] == 1
    assert payload["total"] == 1
    assert payload["data"][0]["id"] == kept["id"]

    assert main(["--reports", "--prune", "--reports-index",
                 str(index_path)]) == 0
    assert "Removed 0 missing report(s)." in capsys.readouterr().out
    with pytest.raises(SystemExit) as exc:
        main(["--prune"])
    assert exc.value.code == 2


def test_cli_reports_verify_integrity(tmp_path, capsys):
    index_path = tmp_path / "weekly_reports.json"
    archive = WeeklyReportArchive(index_path)
    good = tmp_path / "good.md"
    good.write_text("# Good\n", encoding="utf-8")
    changed = tmp_path / "changed.md"
    changed.write_text("# Alpha\n", encoding="utf-8")
    archive.record(good, {
        "generated": "2026-11-21 09:00", "period_days": 7,
        "task_counts": {}, "recent_source_count": 0,
        "graph": {"entities": [], "connections": []},
    }, exported_at="2026-11-21T09:00:00")
    archive.record(changed, {
        "generated": "2026-11-22 09:00", "period_days": 7,
        "task_counts": {}, "recent_source_count": 0,
        "graph": {"entities": [], "connections": []},
    }, exported_at="2026-11-22T09:00:00")
    changed.write_text("# Bravo\n", encoding="utf-8")

    assert main(["--reports", "--verify", "--json",
                 "--reports-index", str(index_path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["verified"] is False
    assert payload["verification"]["total"] == 2
    assert payload["verification"]["ok"] == 1
    assert payload["verification"]["hash_mismatch"] == 1
    assert main(["--reports", "--verify", "--reports-index",
                 str(index_path)]) == 0
    output = capsys.readouterr().out
    assert "Report integrity:" in output
    assert "hash_mismatch" in output
    with pytest.raises(SystemExit) as exc:
        main(["--reports", "--verify", "--keep", "1"])
    assert exc.value.code == 2


def test_cli_reports_repair_metadata(tmp_path, capsys):
    index_path = tmp_path / "weekly_reports.json"
    archive = WeeklyReportArchive(index_path)
    report = tmp_path / "changed.md"
    report.write_text("# Alpha\n", encoding="utf-8")
    archive.record(report, {
        "generated": "2026-11-22 09:00", "period_days": 7,
        "task_counts": {}, "recent_source_count": 0,
        "graph": {"entities": [], "connections": []},
    }, exported_at="2026-11-22T09:00:00")
    report.write_text("# Bravo changed\n", encoding="utf-8")

    assert main(["--reports", "--repair", "--json",
                 "--reports-index", str(index_path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["repair"]["repaired"] == 1
    assert payload["repair"]["missing"] == 0
    assert payload["data"][0]["status"] == "repaired"
    assert payload["data"][0]["actual_size_bytes"] == report.stat().st_size
    assert main(["--reports", "--repair", "--reports-index",
                 str(index_path)]) == 0
    output = capsys.readouterr().out
    assert "Report metadata repair:" in output
    assert "0 repaired" in output
    with pytest.raises(SystemExit) as exc:
        main(["--reports", "--repair", "--verify"])
    assert exc.value.code == 2


def test_cli_reports_compare(tmp_path, capsys):
    index_path = tmp_path / "weekly_reports.json"
    archive = WeeklyReportArchive(index_path)
    for number, name in enumerate(("left", "right"), start=1):
        report = tmp_path / f"{name}.md"
        report.write_text(name * (10 * number), encoding="utf-8")
        archive.record(report, {
            "generated": f"2026-11-2{number} 09:00",
            "period_days": 7 * number,
            "task_counts": {"open": number,
                            "completed_recent": number - 1,
                            "completed_total": number},
            "pending_gap_count": number - 1,
            "recent_source_count": number,
            "graph": {"entities": [], "connections": []},
        }, exported_at=f"2026-11-2{number}T09:00:00")

    assert main(["--reports", "--compare", "w0001", "w0002", "--json",
                 "--reports-index", str(index_path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    metrics = {item["key"]: item for item in payload["metrics"]}
    assert metrics["open"]["delta"] == 1
    assert metrics["recent_source_count"]["delta"] == 1
    assert metrics["size_bytes"]["delta"] == 60
    assert payload["hash_equal"] is False
    assert main(["--reports", "--compare", "w0001", "w0002",
                 "--reports-index", str(index_path)]) == 0
    output = capsys.readouterr().out
    assert "Report comparison: w0001 -> w0002" in output
    assert "- open tasks: 1 -> 2 (+1)" in output
    assert "SHA-256: different" in output
    assert main(["--reports", "--compare", "w0001", "w0002", "--markdown",
                 "--reports-index", str(index_path)]) == 0
    markdown = capsys.readouterr().out
    assert markdown.startswith("# Report comparison: w0001 -> w0002")
    assert "| open tasks | 1 | 2 | +1 |" in markdown
    assert "## Content hash" in markdown
    assert "- Equal: no" in markdown
    with pytest.raises(SystemExit) as exc:
        main(["--compare", "w0001", "w0002"])
    assert exc.value.code == 2


def test_cli_reports_sort_orders(tmp_path, capsys):
    index_path = tmp_path / "weekly_reports.json"
    archive = WeeklyReportArchive(index_path)
    for number, name in enumerate(("a", "b", "c"), start=1):
        report = tmp_path / f"report-{name}.md"
        report.write_text(name * (5 * (number + 1)), encoding="utf-8")
        archive.record(report, {
            "generated": f"2026-11-2{number} 09:00", "period_days": 7,
            "task_counts": {"open": number}, "recent_source_count": 1,
            "graph": {"entities": [], "connections": []},
        }, exported_at=f"2026-11-2{number}T09:00:00")

    assert main(["--reports", "--sort", "oldest", "--json",
                 "--reports-index", str(index_path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["sort"] == "oldest"
    assert [item["name"] for item in payload["data"]] == [
        "report-a.md", "report-b.md", "report-c.md"]
    assert main(["--reports", "--sort", "largest",
                 "--reports-index", str(index_path)]) == 0
    output = capsys.readouterr().out
    assert "sort: largest" in output
    assert output.index("report-c.md") < output.index("report-b.md")
    with pytest.raises(SystemExit) as exc:
        main(["--reports", "--sort", "random"])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        main(["--sort", "oldest"])
    assert exc.value.code == 2


def test_cli_reports_delete_entry(tmp_path, capsys):
    index_path = tmp_path / "weekly_reports.json"
    report = tmp_path / "kept.md"
    report.write_text("# Kept\n", encoding="utf-8")
    second = tmp_path / "deleted.md"
    second.write_text("# Deleted\n", encoding="utf-8")
    archive = WeeklyReportArchive(index_path)
    first_entry = archive.record(report, {
        "generated": "2026-11-22 09:00", "period_days": 7,
        "task_counts": {"open": 1}, "recent_source_count": 1,
        "graph": {"entities": [], "connections": []},
    })
    deleted_entry = archive.record(second, {
        "generated": "2026-11-23 09:00", "period_days": 7,
        "task_counts": {"open": 0}, "recent_source_count": 0,
        "graph": {"entities": [], "connections": []},
    })

    assert main(["--reports", "--delete", deleted_entry["id"], "--json",
                 "--reports-index", str(index_path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["deleted"]["id"] == deleted_entry["id"]
    assert payload["total"] == 1
    assert payload["data"][0]["id"] == first_entry["id"]
    assert second.exists()
    with pytest.raises(SystemExit) as exc:
        main(["--reports", "--delete", first_entry["id"], "--keep", "1"])
    assert exc.value.code == 2

    assert main(["--reports", "--delete", "w9999", "--reports-index",
                 str(index_path)]) == 1
    assert "weekly report not found" in capsys.readouterr().out


def test_cli_reports_query_filter(tmp_path, capsys):
    index_path = tmp_path / "weekly_reports.json"
    archive = WeeklyReportArchive(index_path)
    for number, name in enumerate(("alpha", "beta"), start=1):
        report = tmp_path / f"{name}.md"
        report.write_text(f"# {name}\n", encoding="utf-8")
        archive.record(report, {
            "generated": f"2026-11-2{number} 09:00", "period_days": 7,
            "task_counts": {"open": number}, "recent_source_count": 1,
            "graph": {"entities": [], "connections": []},
        }, exported_at=f"2026-11-2{number}T09:00:00")

    assert main(["--reports", "--query", "alpha", "--json",
                 "--reports-index", str(index_path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["query"] == "alpha"
    assert payload["matched"] == 1
    assert payload["total"] == 2
    assert payload["data"][0]["name"] == "alpha.md"
    assert main(["--reports", "--query", "alpha",
                 "--reports-index", str(index_path)]) == 0
    output = capsys.readouterr().out
    assert "query: alpha | 1 match(es) of 2" in output
    assert "alpha.md" in output
    assert "beta.md" not in output
    assert main(["--reports", "--query", "missing", "--json",
                 "--reports-index", str(index_path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["matched"] == 0
    assert payload["data"] == []
    with pytest.raises(SystemExit) as exc:
        main(["--query", "alpha"])
    assert exc.value.code == 2


def test_cli_reports_keep_retention(tmp_path, capsys):
    index_path = tmp_path / "weekly_reports.json"
    archive = WeeklyReportArchive(index_path)
    for number in range(1, 4):
        report = tmp_path / f"report-{number}.md"
        report.write_text(f"# Report {number}\n", encoding="utf-8")
        archive.record(report, {
            "generated": f"2026-11-2{number} 09:00", "period_days": 7,
            "task_counts": {"open": number}, "recent_source_count": 1,
            "graph": {"entities": [], "connections": []},
        }, exported_at=f"2026-11-2{number}T09:00:00")

    assert main(["--reports", "--keep", "2", "--dry-run", "--json",
                 "--reports-index", str(index_path)]) == 0
    preview = json.loads(capsys.readouterr().out)
    assert preview["dry_run"] is True
    assert preview["would_remove_count"] == 1
    assert preview["would_remove"][0]["id"] == "w0001"
    assert preview["total"] == 3
    assert WeeklyReportArchive(index_path).count() == 3
    assert main(["--reports", "--keep", "2", "--dry-run",
                 "--reports-index", str(index_path)]) == 0
    dry_text = capsys.readouterr().out
    assert "Dry run: would remove 1 older report(s) to keep 2." in dry_text
    assert "Archive unchanged: 3 report(s)." in dry_text

    assert main(["--reports", "--keep", "2", "--json", "--reports-index",
                 str(index_path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["trimmed"] == 1
    assert payload["keep"] == 2
    assert payload["total"] == 2
    assert [item["id"] for item in payload["data"]] == ["w0003", "w0002"]

    assert main(["--reports", "--keep", "2", "--reports-index",
                 str(index_path)]) == 0
    assert "Removed 0 older report(s) to keep 2." in capsys.readouterr().out
    with pytest.raises(SystemExit) as exc:
        main(["--reports", "--keep", "0"])
    assert exc.value.code == 2


def test_cli_weekly_markdown_out_rejects_file_parent(tmp_path, capsys):
    blocker = tmp_path / "blocking-file"
    blocker.write_text("not a directory", encoding="utf-8")
    target = blocker / "weekly.md"
    assert main([
        "--session", str(tmp_path / "missing.json"),
        "--docs", str(tmp_path / "missing.db"),
        "--gaps", str(tmp_path / "missing-gaps.json"),
        "--weekly", "--markdown", "--out", str(target),
        "--reports-index", str(tmp_path / "index.json")]) == 1
    captured = capsys.readouterr()
    assert "briefing error:" in captured.out
    assert not target.exists()


def test_cli_weekly_markdown_out_atomic_replace_failure(
        tmp_path, capsys, monkeypatch):
    target = tmp_path / "weekly.md"
    target.write_text("original report", encoding="utf-8")

    def fail_replace(source, destination):
        raise OSError("simulated replace failure")

    monkeypatch.setattr("atomzero.briefing.os.replace", fail_replace)
    assert main([
        "--session", str(tmp_path / "missing.json"),
        "--docs", str(tmp_path / "missing.db"),
        "--gaps", str(tmp_path / "missing-gaps.json"),
        "--weekly", "--markdown", "--out", str(target)]) == 1
    captured = capsys.readouterr()
    assert "simulated replace failure" in captured.out
    assert target.read_text(encoding="utf-8") == "original report"
    assert not list(tmp_path.glob(f".{target.name}.*.tmp"))


def test_brief_and_weekly_review_show_due_and_overdue_tasks(tmp_path):
    session = {"tasks": [
        {"id": 1, "text": "submit report", "status": "open",
         "created": "2026-11-01T09:00:00", "completed": "",
         "due": "2026-11-20", "priority": "high"},
        {"id": 2, "text": "plan launch", "status": "open",
         "created": "2026-11-01T09:00:00", "completed": "",
         "due": "2026-11-25", "priority": "low"},
        {"id": 3, "text": "no deadline", "status": "open",
         "created": "2026-11-01T09:00:00", "completed": ""},
    ]}
    brief = build_briefing(
        session, None, None, now="2026-11-22 09:00")
    assert brief["open_task_count"] == 3
    assert brief["overdue_task_count"] == 1
    assert brief["due_soon_count"] == 1
    assert brief["focus"].startswith("overdue task 1")
    assert "3 open | 1 overdue | 1 due soon" in brief["text"]
    assert ("submit report [high] (due 2026-11-20 [overdue])"
            in brief["text"])
    assert "plan launch [low] (due 2026-11-25)" in brief["text"]

    review = build_weekly_review(
        session, None, None, days=7, now="2026-11-22 09:00")
    assert review["overdue_task_count"] == 1
    assert review["due_soon_count"] == 1
    assert "Overdue: 1" in review["text"]
    assert "Due soon: 1 (next 7 days)" in review["text"]
    assert ("submit report [high] (due 2026-11-20 [overdue])"
            in review["text"])
    markdown = render_weekly_markdown(review)
    assert "### Overdue (1)" in markdown
    assert "### Due soon (1)" in markdown
    assert ("submit report [high] (due 2026-11-20 [overdue])"
            in markdown)


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
