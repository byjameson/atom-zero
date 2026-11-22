import csv
import io
import json
import time
from datetime import date, datetime, timedelta

import pytest

from atomzero.briefing import (
    build_briefing, build_weekly_review, load_session, main, plan_goal,
    render_weekly_markdown)
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


def test_cli_search_tasks(tmp_path, capsys):
    session_path = tmp_path / "session.json"
    today = date.today()
    soon = (today + timedelta(days=2)).isoformat()
    overdue = (today - timedelta(days=1)).isoformat()
    session_path.write_text(json.dumps({"tasks": [
        {"id": 1, "text": "Submit quarterly report", "status": "open",
         "created": "2026-11-01T09:00:00", "completed": "",
         "due": soon, "priority": "high"},
        {"id": 2, "text": "submit expenses", "status": "done",
         "created": "2026-11-01T09:00:00", "completed": "2026-11-02",
         "due": "2000-01-01", "priority": "normal"},
        {"id": 3, "text": "plan launch", "status": "open",
         "created": "2026-11-01T09:00:00", "completed": "",
         "priority": "low"},
        {"id": 4, "text": "submit late report", "status": "open",
         "created": "2026-11-01T09:00:00", "completed": "",
         "due": overdue, "priority": "high"},
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
    assert payload["count"] == 3
    assert [task["id"] for task in payload["tasks"]] == [1, 2, 4]
    assert payload["open_task_count"] == 3

    assert main(common + ["--search-tasks", "submit"]) == 0
    output = capsys.readouterr().out
    assert 'Task search "submit" (3 match(es))' in output
    assert (f"- 1 [open] Submit quarterly report [high] "
            f"(due {soon})") in output
    assert "- 2 [done] submit expenses (due 2000-01-01)" in output
    assert (f"- 4 [open] submit late report [high] "
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
        main(common + ["--set-task-priority", "1", "urgent"])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        main(common + ["--set-task-priority", "1", "high", "--weekly"])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        main(common + ["--task-sort", "priority", "--reports"])
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


def test_cli_weekly_markdown_out_file(tmp_path, capsys):
    session_path = tmp_path / "session.json"
    now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    session_path.write_text(json.dumps({"tasks": [
        {"id": 1, "text": "raporu yaz", "status": "open",
         "created": now, "completed": ""},
    ]}), encoding="utf-8")
    docs = DocMemory(tmp_path / "docs.db")
    docs.add_text("notes/launch.txt", "The launch code is L-7781.",
                  title="Launch")
    docs.close()
    out = tmp_path / "reports" / "weekly.md"
    index_path = tmp_path / "reports" / "index.json"
    assert main([
        "--session", str(session_path), "--docs", str(tmp_path / "docs.db"),
        "--weekly", "--markdown", "--out", str(out),
        "--reports-index", str(index_path)]) == 0
    captured = capsys.readouterr()
    assert "weekly review written to" in captured.out
    assert "(archived w0001)" in captured.out
    assert out.exists()
    payload = out.read_bytes()
    assert not payload.startswith(b"\xef\xbb\xbf")
    markdown = payload.decode("utf-8")
    assert markdown.startswith("# Weekly review - ")
    assert "- [ ] **#1** raporu yaz" in markdown
    assert not list(out.parent.glob(f".{out.name}.*.tmp"))
    index = json.loads(index_path.read_text(encoding="utf-8"))
    assert index["reports"][0]["id"] == "w0001"
    assert index["reports"][0]["path"] == str(out.resolve())
    assert index["reports"][0]["period_days"] == 7
    assert main(["--reports", "--reports-index", str(index_path)]) == 0
    listing = capsys.readouterr().out
    assert "Weekly report archive:" in listing
    assert "w0001" in listing
    assert "last 7 days" in listing
    assert "1 open" in listing
    assert main(["--reports", "--json", "--reports-index",
                 str(index_path)]) == 0
    listing = json.loads(capsys.readouterr().out)
    assert listing["total"] == 1
    assert listing["data"][0]["exists"] is True
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
