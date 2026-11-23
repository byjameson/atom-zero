import io
import json
import sys
from datetime import datetime

import pytest

from atomzero.freshness import assess_hit, audit_freshness, main

NOW = "2026-01-01"
METHOD = "deterministic_source_freshness_v1"


def _write_hits(tmp_path, payload, name="hits.json"):
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_undated_hit_has_no_date_or_age():
    result = assess_hit(
        {"source": "note.md", "title": "Note",
         "chunk": "No calendar date appears here."},
        now=NOW)
    assert result == {
        "source": "note.md",
        "title": "Note",
        "date": None,
        "age_days": None,
        "status": "undated",
        "span": None,
        "precision": None,
    }


@pytest.mark.parametrize(
    "dated,status,age_days",
    [
        ("2025-10-03", "fresh", 90),
        ("2025-10-02", "aging", 91),
        ("2025-01-01", "aging", 365),
        ("2024-12-31", "stale", 366),
    ],
)
def test_fresh_aging_stale_boundaries(dated, status, age_days):
    result = assess_hit(
        {"source": "doc.md", "chunk": f"Published {dated}."},
        now=NOW)
    assert result["date"] == dated
    assert result["age_days"] == age_days
    assert result["status"] == status
    assert result["precision"] == "day"
    assert result["span"] == [10, 20]


def test_custom_thresholds_change_status():
    hit = {"source": "doc.md", "chunk": "Published 2025-12-01."}
    assert assess_hit(hit, now=NOW)["status"] == "fresh"
    assert assess_hit(
        hit, now=NOW, max_age_days=365, fresh_days=30)["status"] == "aging"
    assert assess_hit(
        hit, now=NOW, max_age_days=20, fresh_days=10)["status"] == "stale"


def test_future_date_is_reported_separately():
    result = assess_hit(
        {"source": "plan.md", "chunk": "Scheduled for 2026-02-01."},
        now=NOW)
    assert result["status"] == "future"
    assert result["date"] == "2026-02-01"
    assert result["age_days"] == -31
    assert result["precision"] == "day"


def test_latest_past_date_wins_and_future_is_ignored():
    result = assess_hit(
        {"source": "doc.md",
         "chunk": ("Published 2025-01-01; revised 2025-12-01; "
                   "planned 2026-06-01.")},
        now=NOW)
    assert result["date"] == "2025-12-01"
    assert result["age_days"] == 31
    assert result["status"] == "fresh"


def test_only_future_dates_choose_the_nearest_future_date():
    result = assess_hit(
        {"source": "doc.md",
         "chunk": "Planned 2027-01-01, review 2026-02-01."},
        now=NOW)
    assert result["status"] == "future"
    assert result["date"] == "2026-02-01"
    assert result["age_days"] == -31


def test_invalid_calendar_date_is_skipped():
    result = assess_hit(
        {"source": "doc.md", "chunk": "Deadline 2026-02-31."},
        now=NOW)
    assert result["status"] == "undated"
    assert result["date"] is None


def test_source_and_title_fall_back_to_each_other():
    title_only = assess_hit(
        {"title": "Release notes", "chunk": "Updated 2025-12-01."},
        now=NOW)
    assert title_only["source"] == "Release notes"
    assert title_only["title"] == "Release notes"

    source_only = assess_hit(
        {"source": "doc.md", "chunk": "Updated 2025-12-01."},
        now=NOW)
    assert source_only["source"] == "doc.md"
    assert source_only["title"] == "doc.md"

    anonymous = assess_hit({"chunk": "Updated 2025-12-01."}, now=NOW)
    assert anonymous["source"] == "hit-0"
    assert anonymous["title"] == ""


def test_text_falls_back_across_chunk_snippet_and_title():
    snippet = assess_hit(
        {"source": "doc.md", "snippet": "Reviewed 2025-11-02."},
        now=NOW)
    assert snippet["date"] == "2025-11-02"
    assert snippet["span"] == [9, 19]

    title = assess_hit(
        {"source": "doc.md", "title": "Report 2025-10-05"},
        now=NOW)
    assert title["date"] == "2025-10-05"
    assert title["span"] == [7, 17]


def test_datetime_now_is_accepted():
    result = assess_hit(
        {"source": "doc.md", "chunk": "Published 2025-12-31."},
        now=datetime(2026, 1, 1, 23, 59, 59))
    assert result["status"] == "fresh"
    assert result["age_days"] == 1


def test_audit_summary_collects_counts_and_sources():
    hits = [
        {"source": "fresh.md", "chunk": "Published 2025-12-01."},
        {"source": "aging.md", "chunk": "Published 2025-06-01."},
        {"source": "stale.md", "chunk": "Published 2024-01-01."},
        {"title": "Future plan", "chunk": "Planned 2026-03-01."},
        {"source": "undated.md", "title": "Undated",
         "chunk": "No date is written here."},
    ]
    report = audit_freshness(hits, now=NOW)
    summary = report["summary"]
    assert [item["status"] for item in report["hits"]] == [
        "fresh", "aging", "stale", "future", "undated"]
    assert summary["hit_count"] == 5
    assert summary["status_counts"] == {
        "fresh": 1, "aging": 1, "stale": 1, "future": 1, "undated": 1}
    assert summary["source_count"] == 5
    assert summary["oldest_date"] == "2024-01-01"
    assert summary["newest_date"] == "2026-03-01"
    assert summary["stale_sources"] == ["stale.md"]
    assert summary["undated_sources"] == ["undated.md"]
    assert report["weights_changed"] is False
    assert report["method"] == METHOD


def test_empty_hits_produce_a_zero_report():
    report = audit_freshness([])
    assert report == {
        "hits": [],
        "summary": {
            "hit_count": 0,
            "status_counts": {
                "fresh": 0, "aging": 0, "stale": 0,
                "future": 0, "undated": 0,
            },
            "source_count": 0,
            "oldest_date": None,
            "newest_date": None,
            "stale_sources": [],
            "undated_sources": [],
        },
        "weights_changed": False,
        "method": METHOD,
    }


def test_non_list_hits_and_non_dict_entries_are_rejected():
    with pytest.raises(ValueError):
        audit_freshness({"hits": []})
    with pytest.raises(ValueError):
        audit_freshness(({"chunk": "x"},))
    with pytest.raises(ValueError):
        audit_freshness([{"chunk": "x"}, "not a hit"])
    with pytest.raises(ValueError):
        audit_freshness([None])
    with pytest.raises(ValueError):
        assess_hit("not a hit")


@pytest.mark.parametrize(
    "max_age_days,fresh_days",
    [
        (10, 20),
        (-1, 0),
        (365, -1),
        (float("nan"), 0),
        (float("inf"), 0),
        ("365", 90),
        (True, 0),
    ],
)
def test_invalid_thresholds_are_rejected(max_age_days, fresh_days):
    with pytest.raises(ValueError):
        audit_freshness([], max_age_days=max_age_days, fresh_days=fresh_days)
    with pytest.raises(ValueError):
        assess_hit(
            {}, max_age_days=max_age_days, fresh_days=fresh_days)


def test_invalid_now_is_rejected():
    with pytest.raises(ValueError):
        audit_freshness([], now="not-a-date")
    with pytest.raises(ValueError):
        assess_hit({}, now="not-a-date")


def test_output_is_deterministic():
    hits = [
        {"source": "b.md", "chunk": "Updated 2025-12-01."},
        {"source": "a.md", "chunk": "Updated 2024-01-01."},
        {"source": "c.md", "chunk": "Undated."},
    ]
    first = audit_freshness(hits, now=NOW)
    second = audit_freshness(hits, now=NOW)
    assert first == second
    assert json.dumps(first, sort_keys=True) == json.dumps(
        second, sort_keys=True)


def test_cli_json_file_and_wrapped_object(tmp_path, capsys):
    hits = [
        {"source": "fresh.md", "chunk": "Published 2025-12-01."},
        {"source": "undated.md", "chunk": "No date."},
    ]
    path = _write_hits(tmp_path, hits)
    assert main(["--file", str(path), "--now", NOW, "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["method"] == METHOD
    assert payload["summary"]["status_counts"]["fresh"] == 1
    assert payload["summary"]["undated_sources"] == ["undated.md"]

    wrapped = _write_hits(tmp_path, {"hits": hits}, name="wrapped.json")
    assert main(["--file", str(wrapped), "--now", NOW, "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["summary"]["hit_count"] == 2


def test_cli_reads_stdin_json(monkeypatch, capsys):
    hits = [{"source": "doc.md", "chunk": "Published 2025-12-01."}]
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(hits)))
    assert main(["--now", NOW, "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["summary"]["status_counts"]["fresh"] == 1


def test_cli_human_output_has_one_line_per_hit_and_summary(tmp_path, capsys):
    path = _write_hits(tmp_path, [
        {"source": "fresh.md", "title": "Fresh",
         "chunk": "Published 2025-12-01."},
        {"source": "undated.md", "title": "Undated", "chunk": "No date."},
    ])
    assert main(["--file", str(path), "--now", NOW]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 3
    assert lines[0].startswith("- [fresh] fresh.md")
    assert "2025-12-01" in lines[0]
    assert lines[1].startswith("- [undated] undated.md")
    assert lines[2].startswith("Freshness:")


def test_cli_empty_inputs_exit_two(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    assert main(["--json"]) == 2
    assert "no hits" in capsys.readouterr().err

    empty = _write_hits(tmp_path, [], name="empty.json")
    assert main(["--file", str(empty)]) == 2


def test_cli_unreadable_and_invalid_json_exit_two(tmp_path):
    missing = tmp_path / "missing.json"
    assert main(["--file", str(missing)]) == 2

    invalid = tmp_path / "invalid.json"
    invalid.write_text("{not json", encoding="utf-8")
    assert main(["--file", str(invalid)]) == 2

    non_list = _write_hits(tmp_path, {"hits": "not a list"},
                           name="non-list.json")
    assert main(["--file", str(non_list)]) == 2


def test_cli_invalid_values_exit_two(tmp_path):
    path = _write_hits(tmp_path, [
        {"source": "doc.md", "chunk": "Published 2025-12-01."},
    ])
    assert main([
        "--file", str(path), "--now", NOW,
        "--max-age-days", "10", "--fresh-days", "20",
    ]) == 2
    assert main(["--file", str(path), "--now", "not-a-date"]) == 2
