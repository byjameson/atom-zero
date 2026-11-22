import json
import time
from datetime import datetime, timedelta

import pytest

from atomzero.briefing import (
    build_briefing, build_weekly_review, load_session, main, plan_goal)
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
