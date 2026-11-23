import copy
import json
import re
from datetime import datetime, timezone

import pytest

from atomzero.reliability import (
    add_observation,
    load_jsonl,
    main,
    source_id,
    summarize,
    verify,
    write_jsonl,
)

FIXED = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
FORBIDDEN = {
    "text", "query", "answer", "chunk", "content", "session", "session_id",
    "source", "source_text",
}


def _add(ledger, source, *, supported=1, weak=0, unsupported=0,
         conflicts=0, now=FIXED):
    return add_observation(
        ledger, source, supported=supported, weak=weak,
        unsupported=unsupported, conflicts=conflicts, now=now)


def test_source_id_is_stable_trimmed_and_rejects_empty():
    identifier = source_id("  example.org  ")
    assert identifier == source_id("example.org")
    assert identifier != source_id("other.example")
    assert len(identifier) == 16
    assert re.fullmatch(r"[0-9a-f]{16}", identifier)
    for empty in ("", "   ", "\n\t"):
        with pytest.raises(ValueError):
            source_id(empty)
    with pytest.raises(ValueError):
        source_id(None)


def test_add_observation_appends_content_free_entry():
    ledger = []
    entry = _add(
        ledger, "https://example.org/page", supported=2, weak=1,
        unsupported=1, conflicts=1)
    assert ledger == [entry]
    assert entry == {
        "version": 1,
        "at": "2026-01-01T12:00:00Z",
        "source_id": source_id("https://example.org/page"),
        "supported": 2,
        "weak": 1,
        "unsupported": 1,
        "claims": 4,
        "conflicts": 1,
    }
    serialized = json.dumps(entry)
    assert "example.org" not in serialized
    assert set(entry) == {
        "version", "at", "source_id", "supported", "weak", "unsupported",
        "claims", "conflicts",
    }
    assert not (set(entry) & FORBIDDEN)


def test_add_observation_normalizes_now_to_utc():
    ledger = []
    entry = _add(ledger, "a", now="2026-01-01T03:00:00+03:00")
    assert entry["at"] == "2026-01-01T00:00:00Z"
    naive = _add(ledger, "a", now="2026-01-01T00:00:00")
    assert naive["at"] == "2026-01-01T00:00:00Z"
    with pytest.raises(ValueError):
        _add(ledger, "a", now="not-a-date")
    assert len(ledger) == 2


@pytest.mark.parametrize("counts", [
    {"supported": -1},
    {"weak": -1},
    {"unsupported": 1.5},
    {"conflicts": -1},
    {"supported": True},
    {"supported": 0, "weak": 0, "unsupported": 0},
    {"supported": 1, "unsupported": 1, "conflicts": 3},
])
def test_add_observation_rejects_bad_counts(counts):
    ledger = []
    with pytest.raises(ValueError):
        _add(ledger, "example.org", **counts)
    assert ledger == []


def test_summary_thresholds_and_totals():
    ledger = []
    _add(ledger, "reliable.example", supported=3, unsupported=1)
    _add(ledger, "mixed.example", supported=2, unsupported=3)
    _add(ledger, "weak.example", supported=1, unsupported=2)

    summary = summarize(ledger)
    by_source = {row["source_id"]: row for row in summary["sources"]}
    reliable = by_source[source_id("reliable.example")]
    mixed = by_source[source_id("mixed.example")]
    weak = by_source[source_id("weak.example")]
    assert (reliable["support_ratio"], reliable["status"]) == (0.75, "reliable")
    assert (mixed["support_ratio"], mixed["status"]) == (0.4, "mixed")
    assert (weak["support_ratio"], weak["status"]) == (0.3333, "weak")

    assert [row["source_id"] for row in summary["sources"]] == sorted(
        by_source)
    assert summary["totals"] == {
        "sources": 3,
        "observations": 3,
        "supported": 6,
        "weak": 0,
        "unsupported": 6,
        "claims": 12,
        "conflicts": 0,
        "support_ratio": 0.5,
        "status": "mixed",
    }
    assert summary["observations"] == 3
    assert summary["claims"] == 12
    assert summary["conflicts"] == 0
    assert summary["support_ratio"] == 0.5
    assert summary["status"] == "mixed"


def test_weak_claims_are_weighted_half():
    ledger = []
    _add(ledger, "half", supported=0, weak=1, unsupported=1)
    _add(ledger, "boundary", supported=1, weak=1)
    _add(ledger, "mixed", supported=0, weak=2)

    rows = {row["source_id"]: row for row in summarize(ledger)["sources"]}
    assert rows[source_id("half")]["support_ratio"] == 0.25
    assert rows[source_id("half")]["status"] == "weak"
    assert rows[source_id("boundary")]["support_ratio"] == 0.75
    assert rows[source_id("boundary")]["status"] == "reliable"
    assert rows[source_id("mixed")]["support_ratio"] == 0.5
    assert rows[source_id("mixed")]["status"] == "mixed"


def test_min_observations_filters_sources_and_totals():
    ledger = []
    _add(ledger, "twice", supported=1, unsupported=1)
    _add(ledger, "twice", supported=2)
    _add(ledger, "once", supported=1)

    filtered = summarize(ledger, min_observations=2)
    assert filtered["min_observations"] == 2
    assert [row["source_id"] for row in filtered["sources"]] == [
        source_id("twice")]
    assert filtered["sources"][0]["observations"] == 2
    assert filtered["totals"]["observations"] == 2
    assert filtered["totals"]["sources"] == 1
    assert filtered["totals"]["claims"] == 4
    assert filtered["totals"]["support_ratio"] == 0.75

    complete = summarize(ledger)
    assert len(complete["sources"]) == 2
    assert complete["totals"]["observations"] == 3


@pytest.mark.parametrize("bad", [-1, 1.5, True, "2"])
def test_min_observations_is_validated(bad):
    with pytest.raises(ValueError):
        summarize([], min_observations=bad)


def test_verify_accepts_valid_and_empty_ledgers():
    ledger = []
    _add(ledger, "a")
    _add(ledger, "b", supported=0, weak=1)
    result = verify(ledger)
    assert result == {"ok": True, "entries": 2, "errors": []}
    assert verify([]) == {"ok": True, "entries": 0, "errors": []}
    assert verify("not-a-list") == {
        "ok": False, "entries": 0, "errors": ["ledger must be a list"]}


def test_verify_rejects_invalid_entries():
    base = _add([], "a", supported=1, unsupported=1)
    invalid = []

    missing = copy.deepcopy(base)
    del missing["weak"]
    invalid.append(missing)

    wrong_version = copy.deepcopy(base)
    wrong_version["version"] = 2
    invalid.append(wrong_version)

    bad_time = copy.deepcopy(base)
    bad_time["at"] = "yesterday"
    invalid.append(bad_time)

    naive_time = copy.deepcopy(base)
    naive_time["at"] = "2026-01-01T00:00:00"
    invalid.append(naive_time)

    bad_id = copy.deepcopy(base)
    bad_id["source_id"] = "NOT-A-HASH"
    invalid.append(bad_id)

    negative = copy.deepcopy(base)
    negative["supported"] = -1
    invalid.append(negative)

    boolean = copy.deepcopy(base)
    boolean["supported"] = True
    invalid.append(boolean)

    mismatch = copy.deepcopy(base)
    mismatch["claims"] = 99
    invalid.append(mismatch)

    too_many_conflicts = copy.deepcopy(base)
    too_many_conflicts["conflicts"] = 3
    invalid.append(too_many_conflicts)

    zero_claims = copy.deepcopy(base)
    zero_claims.update(
        supported=0, weak=0, unsupported=0, claims=0, conflicts=0)
    invalid.append(zero_claims)

    content = copy.deepcopy(base)
    content["text"] = "secret"
    invalid.append(content)

    nested = copy.deepcopy(base)
    nested["meta"] = {"query": "secret"}
    invalid.append(nested)

    result = verify(invalid)
    assert result["ok"] is False
    assert result["entries"] == len(invalid)
    joined = "\n".join(result["errors"])
    for expected in (
            "missing key", "version must be 1", "not valid ISO-8601",
            "zero UTC offset", "source_id", "non-negative integer",
            "does not equal", "conflicts", "at least 1",
            "forbidden content key", "unexpected key"):
        assert expected in joined


@pytest.mark.parametrize("key", sorted(FORBIDDEN))
def test_verify_rejects_each_forbidden_content_key(key):
    entry = _add([], "a")
    entry[key] = "secret"
    result = verify([entry])
    assert result["ok"] is False
    assert f"forbidden content key: {key}" in "\n".join(result["errors"])


def test_verify_rejects_non_object_entries():
    result = verify([{"ok": True}, 3])
    assert result["ok"] is False
    assert result["entries"] == 2
    assert "must be a JSON object" in "\n".join(result["errors"])


def test_summary_rejects_invalid_ledger():
    ledger = []
    _add(ledger, "a")
    ledger[0]["claims"] = 5
    with pytest.raises(ValueError):
        summarize(ledger)


def test_jsonl_round_trip_and_atomic_write(tmp_path):
    path = tmp_path / "ledger.jsonl"
    ledger = []
    _add(ledger, "a", supported=2, weak=1, unsupported=1, conflicts=1)
    _add(ledger, "b", supported=0, unsupported=1)

    write_jsonl(path, ledger)
    assert load_jsonl(path) == ledger
    text = path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert "\r" not in text
    assert len(text.splitlines()) == 2
    assert not list(tmp_path.glob(".ledger.jsonl.*.tmp"))

    write_jsonl(path, [])
    assert path.read_text(encoding="utf-8") == ""
    assert load_jsonl(path) == []
    assert load_jsonl(tmp_path / "missing.jsonl") == []


def test_load_jsonl_rejects_bad_lines_and_duplicate_keys(tmp_path):
    path = tmp_path / "ledger.jsonl"
    path.write_text('{"a": 1}\nnot-json\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line|:2:"):
        load_jsonl(path)

    path.write_text('{"a": 1, "a": 2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate key"):
        load_jsonl(path)

    path.write_text("[1, 2]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        load_jsonl(path)


def test_summary_is_deterministic_for_different_input_order():
    first = []
    _add(first, "b", supported=1, unsupported=1)
    _add(first, "a", supported=2, weak=1)
    _add(first, "b", supported=0, weak=2)

    second = [first[2], first[0], first[1]]
    assert summarize(first) == summarize(second)
    assert json.dumps(summarize(first), sort_keys=True) == json.dumps(
        summarize(second), sort_keys=True)


def test_cli_add_summary_verify(tmp_path, capsys):
    path = tmp_path / "ledger.jsonl"
    args = ["--file", str(path), "--add-source", "docs.example"]

    assert main(args + [
        "--supported", "2", "--weak", "1", "--unsupported", "1",
        "--json"]) == 0
    added = json.loads(capsys.readouterr().out)
    assert added["added"]["source_id"] == source_id("docs.example")
    assert "docs.example" not in path.read_text(encoding="utf-8")

    assert main(["--file", str(path), "--summary", "--json"]) == 0
    summary = json.loads(capsys.readouterr().out)["summary"]
    assert summary["totals"]["observations"] == 1
    assert summary["totals"]["claims"] == 4
    assert summary["totals"]["support_ratio"] == 0.625
    assert summary["totals"]["status"] == "mixed"

    assert main(["--file", str(path), "--verify", "--json"]) == 0
    verified = json.loads(capsys.readouterr().out)["verify"]
    assert verified == {"ok": True, "entries": 1, "errors": []}


def test_cli_plain_summary_and_default_action(tmp_path, capsys):
    path = tmp_path / "ledger.jsonl"
    assert main([
        "--file", str(path), "--add-source", "a", "--supported", "1"]) == 0
    capsys.readouterr()
    assert main(["--file", str(path)]) == 0
    output = capsys.readouterr().out
    assert "Source reliability" in output
    assert "reliable" in output
    assert source_id("a") in output


def test_cli_invalid_add_exits_2(tmp_path, capsys):
    path = tmp_path / "ledger.jsonl"
    assert main([
        "--file", str(path), "--add-source", "a", "--supported", "0"]) == 2
    assert "error:" in capsys.readouterr().err
    assert not path.exists()

    assert main([
        "--file", str(path), "--add-source", "  ", "--supported", "1"]) == 2
    capsys.readouterr()
    assert not path.exists()

    assert main([
        "--file", str(path), "--add-source", "a", "--supported", "1",
        "--conflicts", "2"]) == 2
    capsys.readouterr()
    assert not path.exists()


def test_cli_verify_failure_exits_2(tmp_path, capsys):
    path = tmp_path / "ledger.jsonl"
    entry = _add([], "a")
    entry["claims"] = 5
    path.write_text(json.dumps(entry) + "\n", encoding="utf-8")
    assert main(["--file", str(path), "--verify", "--json"]) == 2
    result = json.loads(capsys.readouterr().out)["verify"]
    assert result["ok"] is False
    assert result["entries"] == 1
    assert "does not equal" in result["errors"][0]


def test_cli_invalid_json_and_summary_exit_2(tmp_path, capsys):
    path = tmp_path / "ledger.jsonl"
    path.write_text("not-json\n", encoding="utf-8")
    assert main(["--file", str(path), "--verify"]) == 2
    assert "error:" in capsys.readouterr().err

    path.write_text('{"version": 1}\n', encoding="utf-8")
    assert main(["--file", str(path), "--summary"]) == 2
    assert "error:" in capsys.readouterr().err
