import json

import pytest

from atomzero.weekly_reports import WeeklyReportArchive, WeeklyReportError


BOM = b"\xef\xbb\xbf"


def _review(period_days=7, open_tasks=1):
    return {
        "generated": "2026-11-22 09:00",
        "period_days": period_days,
        "task_counts": {
            "open": open_tasks,
            "completed_recent": 1,
            "completed_total": 2,
        },
        "pending_gap_count": 1,
        "recent_source_count": 2,
        "graph": {
            "entities": [{"name": "L-7781"}],
            "connections": [{"source": "L-7781", "target": "Launch"}],
        },
    }


def test_archive_records_dedupes_and_reads_bom_index(tmp_path):
    index_path = tmp_path / "weekly_reports.json"
    archive = WeeklyReportArchive(index_path)
    first_report = tmp_path / "first.md"
    first_report.write_text("# First report\n", encoding="utf-8")
    first = archive.record(
        first_report, _review(), exported_at="2026-11-22T09:00:00")
    assert first["id"] == "w0001"
    assert first["period_days"] == 7
    assert first["task_counts"]["open"] == 1
    assert first["entities"] == 1
    assert first["connections"] == 1
    assert first["size_bytes"] == first_report.stat().st_size
    assert first["sha256"]

    second_report = tmp_path / "second.md"
    second_report.write_text("# Second report\n", encoding="utf-8")
    second = archive.record(
        second_report, _review(period_days=14, open_tasks=3))
    assert second["id"] == "w0002"
    assert [item["id"] for item in archive.list()] == ["w0002", "w0001"]

    updated = archive.record(
        first_report, _review(period_days=30, open_tasks=4),
        exported_at="2026-11-23T09:00:00")
    assert updated["id"] == "w0001"
    entries = archive.list()
    assert len(entries) == 2
    assert entries[0]["id"] == "w0001"
    assert entries[0]["period_days"] == 30
    assert entries[0]["exists"] is True

    index_path.write_bytes(BOM + index_path.read_bytes())
    reloaded = WeeklyReportArchive(index_path)
    assert len(reloaded.list()) == 2
    assert reloaded.list()[0]["task_counts"]["open"] == 4


def test_archive_sort_orders(tmp_path):
    index_path = tmp_path / "weekly_reports.json"
    archive = WeeklyReportArchive(index_path)
    for number, name in enumerate(("a", "b", "c"), start=1):
        report = tmp_path / f"report-{name}.md"
        report.write_text(name * (5 * (number + 1)), encoding="utf-8")
        archive.record(
            report, _review(),
            exported_at=f"2026-11-2{number}T09:00:00")

    assert [item["name"] for item in archive.list(order="newest")] == [
        "report-c.md", "report-b.md", "report-a.md"]
    assert [item["name"] for item in archive.list(order="oldest")] == [
        "report-a.md", "report-b.md", "report-c.md"]
    assert [item["name"] for item in archive.list(order="largest")] == [
        "report-c.md", "report-b.md", "report-a.md"]
    assert [item["name"] for item in archive.list(
        order="oldest", query="report-b")] == ["report-b.md"]
    with pytest.raises(WeeklyReportError):
        archive.list(order="random")


def test_archive_query_filters_entries(tmp_path):
    index_path = tmp_path / "weekly_reports.json"
    archive = WeeklyReportArchive(index_path)
    for number, name in enumerate(("alpha", "beta"), start=1):
        report = tmp_path / f"{name}.md"
        report.write_text(f"# {name}\n", encoding="utf-8")
        archive.record(
            report, _review(),
            exported_at=f"2026-11-22T0{number}:00:00")
    assert archive.count() == 2
    assert archive.count(query="alpha") == 1
    assert [item["name"] for item in archive.list(query="alpha")] == [
        "alpha.md"]
    assert archive.list(query="missing") == []
    assert archive.count(query="missing") == 0
    assert len(archive.list(query="WEEKLY")) == 2


def test_archive_verify_detects_changed_and_missing_files(tmp_path):
    index_path = tmp_path / "weekly_reports.json"
    archive = WeeklyReportArchive(index_path)
    unchanged = tmp_path / "unchanged.md"
    unchanged.write_text("# Unchanged\n", encoding="utf-8")
    changed = tmp_path / "changed.md"
    changed.write_text("# Alpha\n", encoding="utf-8")
    missing = tmp_path / "missing.md"
    missing.write_text("# Missing\n", encoding="utf-8")
    for number, path in enumerate((unchanged, changed, missing), start=1):
        archive.record(
            path, _review(),
            exported_at=f"2026-11-2{number}T09:00:00")
    changed.write_text("# Bravo\n", encoding="utf-8")
    missing.unlink()

    result = archive.verify()
    assert result["ok"] is False
    assert result["summary"] == {
        "total": 3, "ok": 1, "failed": 2, "missing": 1,
        "size_mismatch": 0, "hash_mismatch": 1,
    }
    by_name = {item["name"]: item for item in result["entries"]}
    assert by_name["unchanged.md"]["status"] == "ok"
    assert by_name["changed.md"]["status"] == "failed"
    assert "hash_mismatch" in by_name["changed.md"]["problems"]
    assert by_name["missing.md"]["status"] == "missing"


def test_archive_repair_updates_changed_metadata_atomically(tmp_path):
    index_path = tmp_path / "weekly_reports.json"
    archive = WeeklyReportArchive(index_path)
    good = tmp_path / "good.md"
    good.write_text("# Good\n", encoding="utf-8")
    changed = tmp_path / "changed.md"
    changed.write_text("# Alpha\n", encoding="utf-8")
    missing = tmp_path / "missing.md"
    missing.write_text("# Missing\n", encoding="utf-8")
    for number, path in enumerate((good, changed, missing), start=1):
        archive.record(
            path, _review(),
            exported_at=f"2026-11-2{number}T09:00:00")
    changed.write_text("# Bravo changed\n", encoding="utf-8")
    missing.unlink()

    result = archive.repair()
    assert result["ok"] is False
    assert result["summary"] == {
        "total": 3, "repaired": 1, "ok": 1, "missing": 1, "failed": 1,
    }
    by_name = {item["name"]: item for item in result["entries"]}
    assert by_name["good.md"]["status"] == "ok"
    assert by_name["changed.md"]["status"] == "repaired"
    assert by_name["changed.md"]["actual_size_bytes"] == (
        changed.stat().st_size)
    assert by_name["changed.md"]["actual_sha256"] != (
        by_name["changed.md"]["old_sha256"])
    assert by_name["missing.md"]["status"] == "missing"

    verification = archive.verify()
    statuses = {item["name"]: item["status"]
                for item in verification["entries"]}
    assert statuses["changed.md"] == "ok"
    assert statuses["missing.md"] == "missing"
    assert archive.repair()["summary"]["repaired"] == 0


def test_archive_repair_single_report(tmp_path):
    index_path = tmp_path / "weekly_reports.json"
    archive = WeeklyReportArchive(index_path)
    first = tmp_path / "first.md"
    first.write_text("# First\n", encoding="utf-8")
    second = tmp_path / "second.md"
    second.write_text("# Second\n", encoding="utf-8")
    archive.record(first, _review(), exported_at="2026-11-21T09:00:00")
    archive.record(second, _review(), exported_at="2026-11-22T09:00:00")
    second.write_text("# Second changed\n", encoding="utf-8")

    result = archive.repair("w0002")
    assert result["summary"] == {
        "total": 1, "repaired": 1, "ok": 0, "missing": 0, "failed": 0,
    }
    assert result["entries"][0]["id"] == "w0002"
    assert result["entries"][0]["status"] == "repaired"
    verification = archive.verify()
    assert verification["summary"]["ok"] == 2
    with pytest.raises(WeeklyReportError):
        archive.repair("w9999")


def test_archive_compare_reports_metrics(tmp_path):
    index_path = tmp_path / "weekly_reports.json"
    archive = WeeklyReportArchive(index_path)
    left = tmp_path / "left.md"
    left.write_text("a" * 10, encoding="utf-8")
    right = tmp_path / "right.md"
    right.write_text("b" * 20, encoding="utf-8")
    archive.record(left, {
        "generated": "2026-11-21 09:00", "period_days": 7,
        "task_counts": {"open": 1, "completed_recent": 0,
                        "completed_total": 1},
        "pending_gap_count": 0, "recent_source_count": 1,
        "graph": {"entities": [], "connections": []},
    }, exported_at="2026-11-21T09:00:00")
    archive.record(right, {
        "generated": "2026-11-22 09:00", "period_days": 14,
        "task_counts": {"open": 3, "completed_recent": 2,
                        "completed_total": 4},
        "pending_gap_count": 2, "recent_source_count": 3,
        "graph": {"entities": [], "connections": []},
    }, exported_at="2026-11-22T09:00:00")

    comparison = archive.compare("w0001", "w0002")
    metrics = {item["key"]: item for item in comparison["metrics"]}
    assert metrics["open"]["delta"] == 2
    assert metrics["completed_recent"]["delta"] == 2
    assert metrics["completed_total"]["delta"] == 3
    assert metrics["pending_gap_count"]["delta"] == 2
    assert metrics["recent_source_count"]["delta"] == 2
    assert metrics["size_bytes"]["delta"] == 10
    assert metrics["period_days"]["delta"] == 7
    assert comparison["hash_equal"] is False
    same = archive.compare("w0001", "w0001")
    assert same["same_report"] is True
    assert same["hash_equal"] is True
    assert all(item["delta"] == 0 for item in same["metrics"])
    with pytest.raises(WeeklyReportError):
        archive.compare("w0001", "w9999")


def test_archive_delete_removes_one_entry_only(tmp_path):
    index_path = tmp_path / "weekly_reports.json"
    archive = WeeklyReportArchive(index_path)
    report = tmp_path / "weekly.md"
    report.write_text("# Weekly\n", encoding="utf-8")
    first = archive.record(report, _review())
    second_report = tmp_path / "second.md"
    second_report.write_text("# Second\n", encoding="utf-8")
    second = archive.record(second_report, _review())

    deleted = archive.delete(first["id"])
    assert deleted["id"] == first["id"]
    assert report.exists()
    assert archive.count() == 1
    assert archive.list()[0]["id"] == second["id"]
    with pytest.raises(WeeklyReportError):
        archive.delete(first["id"])


def test_archive_keep_trims_oldest_reports(tmp_path):
    index_path = tmp_path / "weekly_reports.json"
    archive = WeeklyReportArchive(index_path)
    for number in range(1, 4):
        report = tmp_path / f"report-{number}.md"
        report.write_text(f"# Report {number}\n", encoding="utf-8")
        archive.record(
            report, _review(), exported_at=f"2026-11-2{number}T09:00:00")
    assert archive.count() == 3
    preview = archive.keep_preview(2)
    assert [item["id"] for item in preview] == ["w0001"]
    assert archive.count() == 3

    assert archive.keep(2) == 1
    entries = archive.list()
    assert [item["id"] for item in entries] == ["w0003", "w0002"]
    assert archive.keep(2) == 0
    assert archive.keep(10) == 0
    with pytest.raises(WeeklyReportError):
        archive.keep(0)


def test_archive_prune_removes_missing_reports(tmp_path):
    index_path = tmp_path / "weekly_reports.json"
    archive = WeeklyReportArchive(index_path)
    present = tmp_path / "present.md"
    present.write_text("# Present\n", encoding="utf-8")
    missing = tmp_path / "missing.md"
    missing.write_text("# Missing\n", encoding="utf-8")
    archive.record(present, _review())
    archive.record(missing, _review(period_days=14))
    missing.unlink()
    assert archive.count() == 2

    assert archive.prune() == 1
    entries = archive.list()
    assert len(entries) == 1
    assert entries[0]["path"] == str(present.resolve())
    assert archive.prune() == 0


def test_archive_markdown_download_and_missing_files(tmp_path):
    index_path = tmp_path / "weekly_reports.json"
    report = tmp_path / "weekly.md"
    report.write_text("# Weekly review\n\n- [ ] task\n", encoding="utf-8")
    archive = WeeklyReportArchive(index_path)
    entry = archive.record(report, _review())

    loaded, markdown = archive.markdown(entry["id"])
    assert loaded["path"] == str(report.resolve())
    assert markdown.startswith("# Weekly review")

    report.unlink()
    assert archive.list()[0]["exists"] is False
    with pytest.raises(WeeklyReportError):
        archive.markdown(entry["id"])
    with pytest.raises(WeeklyReportError):
        archive.get("w9999")


def test_archive_rejects_bad_index(tmp_path):
    index_path = tmp_path / "weekly_reports.json"
    index_path.write_text("not-json", encoding="utf-8")
    with pytest.raises(WeeklyReportError):
        WeeklyReportArchive(index_path)
