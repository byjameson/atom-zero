"""Tests for deterministic content replay row construction."""
import hashlib
import json

import pytest

from research.build_content_rows import (
    ContentRowsError,
    build_rows,
    clean_extract,
    main,
    read_facts,
    run_build,
)


def _fact(title, *, text=None, fact_id=None, lang="en"):
    if text is None:
        text = (f"{title} is a factual subject with enough text to pass the "
                f"minimum length filter. " * 4)
    return {
        "id": fact_id or f"{lang}:{title}",
        "title": title,
        "lang": lang,
        "text": text,
    }


def _write_facts(path, rows, raw_lines=()):
    lines = [json.dumps(row, ensure_ascii=False) for row in rows]
    lines.extend(raw_lines)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _read_jsonl(path):
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_build_is_deterministic_for_the_same_seed(tmp_path):
    facts = _write_facts(
        tmp_path / "world_facts.jsonl",
        [_fact(f"Subject{index}") for index in range(10)])

    first = run_build(
        facts, tmp_path / "first.jsonl", seed=7, max_rows=4,
        manifest_path=tmp_path / "first_manifest.json")
    second = run_build(
        facts, tmp_path / "second.jsonl", seed=7, max_rows=4,
        manifest_path=tmp_path / "second_manifest.json")

    first_bytes = (tmp_path / "first.jsonl").read_bytes()
    assert first_bytes == (tmp_path / "second.jsonl").read_bytes()
    assert first["sha256"] == second["sha256"]
    assert first["selected"] == second["selected"] == 4
    assert len(_read_jsonl(tmp_path / "first.jsonl")) == 4

    third = run_build(
        facts, tmp_path / "third.jsonl", seed=8, max_rows=4)
    assert third["sha256"] != first["sha256"]


@pytest.mark.parametrize("style", ["qa", "summary", "citation"])
def test_style_shapes_are_source_grounded(tmp_path, style):
    title = "Alpha"
    text = ("Alpha is the first letter of the Greek alphabet. "
            "It is used as a symbol in mathematics and science. "
            "The uppercase form is written as A.")
    facts = _write_facts(
        tmp_path / "world_facts.jsonl",
        [_fact(title, text=text)])

    manifest = run_build(
        facts, tmp_path / "train.jsonl", style=style, min_chars=0,
        manifest_path=tmp_path / "manifest.json")
    row = _read_jsonl(tmp_path / "train.jsonl")[0]
    prefix = {
        "qa": "",
        "summary": "In summary: ",
        "citation": f'According to the source "{title}", ',
    }[style]

    assert list(row) == [
        "user", "assistant", "family", "source_id", "source_title", "style"]
    assert row["style"] == style
    assert row["family"] == f"content_{style}"
    assert row["source_id"] == f"en:{title}"
    assert row["source_title"] == title
    assert row["assistant"].startswith(prefix)
    extract = row["assistant"][len(prefix):]
    assert extract
    assert extract in " ".join(text.split())
    assert manifest["selected"] == 1
    if style == "qa":
        assert row["user"] == f"What does the source say about {title}?"
    elif style == "summary":
        assert row["user"] == f"Summarize the source about {title}."
        assert row["assistant"].startswith("In summary: Alpha")
    else:
        assert row["user"] == f"Give a source-cited answer about {title}."
        assert title in row["assistant"]
        assert row["assistant"].startswith("According to the source")


@pytest.mark.parametrize("style", ["qa", "summary", "citation"])
def test_answers_respect_the_character_cap(tmp_path, style):
    text = ("This subject has a first sentence. " * 40).strip()
    facts = _write_facts(
        tmp_path / "world_facts.jsonl", [_fact("Capped", text=text)])

    run_build(
        facts, tmp_path / "train.jsonl", style=style, min_chars=0,
        max_answer_chars=60)
    row = _read_jsonl(tmp_path / "train.jsonl")[0]

    assert len(row["assistant"]) <= 60
    assert row["assistant"]


def test_long_answers_cut_on_a_sentence_boundary():
    text = ("First sentence is short. Second sentence is also short. "
            "Third sentence is much longer and should not be used.")
    rows, counts = build_rows(
        [_fact("Boundary", text=text)], style="qa", min_chars=0,
        max_answer_chars=60, max_rows=10)

    assert counts["selected"] == 1
    assert rows[0]["assistant"] == (
        "First sentence is short. Second sentence is also short.")
    assert rows[0]["assistant"].endswith(".")
    assert len(rows[0]["assistant"]) <= 60


def test_sentence_boundary_trim_accounts_for_the_lead_in():
    text = ("First sentence is short. Second sentence is also short. "
            "Third sentence is much longer and should not be used.")
    rows, _ = build_rows(
        [_fact("Boundary", text=text)], style="summary", min_chars=0,
        max_answer_chars=50, max_rows=10)

    assert rows[0]["assistant"] == "In summary: First sentence is short."
    assert len(rows[0]["assistant"]) <= 50


def test_short_duplicate_malformed_and_non_english_rows_are_skipped(tmp_path):
    good = _fact("Good")
    facts = _write_facts(
        tmp_path / "world_facts.jsonl",
        [
            good,
            {"id": "fr:Bonjour", "title": "Bonjour", "lang": "fr",
             "text": "Bonjour " * 80},
            _fact("Short", text="too short"),
            _fact(" good "),
            {"id": "en:NoText", "title": "NoText", "lang": "en"},
            {"title": "NoId", "lang": "en", "text": "text " * 80},
            "not an object",
        ],
        raw_lines=["{not json", ""])

    values, read_counts = read_facts(tmp_path / "world_facts.jsonl")
    rows, counts = build_rows(values, min_chars=200, max_rows=10)

    assert read_counts == {"rows_read": 8, "blank_lines": 1}
    assert [row["source_title"] for row in rows] == ["Good"]
    assert counts["rows_read"] == 8
    assert counts["eligible"] == 1
    assert counts["selected"] == 1
    assert counts["skip_reasons"] == {
        "malformed": 4,
        "non_english": 1,
        "short": 1,
        "duplicate_title": 1,
        "answer_budget_too_small": 0,
    }
    assert counts["skipped"] == 7


def test_lang_override_keeps_the_requested_language(tmp_path):
    facts = _write_facts(
        tmp_path / "world_facts.jsonl",
        [
            _fact("English"),
            _fact("Francais", lang="fr"),
        ])

    run_build(facts, tmp_path / "train.jsonl", lang="fr", min_chars=0)
    rows = _read_jsonl(tmp_path / "train.jsonl")

    assert len(rows) == 1
    assert rows[0]["source_title"] == "Francais"


def test_manifest_is_content_free_and_records_the_hash(tmp_path):
    title = "TITLE_SENTINEL_ALPHA"
    fact_id = "SOURCE_ID_SENTINEL_BETA"
    answer_text = ("ANSWER_SENTINEL_GAMMA is a sentence in the source. "
                   "It has enough repeated text to pass the length filter. "
                   * 4)
    facts = _write_facts(
        tmp_path / "world_facts.jsonl",
        [_fact(title, text=answer_text, fact_id=fact_id)])
    out = tmp_path / "train.jsonl"
    manifest_path = tmp_path / "manifest.json"

    manifest = run_build(
        facts, out, seed=9, min_chars=0, manifest_path=manifest_path)

    text = manifest_path.read_text(encoding="utf-8")
    for sentinel in (title, fact_id, "ANSWER_SENTINEL_GAMMA"):
        assert sentinel not in text
    payload = json.loads(text)
    assert payload["style"] == "qa"
    assert payload["seed"] == 9
    assert payload["source"] == "world_facts.jsonl"
    assert payload["selected"] == 1
    assert payload["rows_read"] == 1
    assert payload["skip_reasons"]["malformed"] == 0
    assert payload["sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    assert manifest["sha256"] == payload["sha256"]
    assert str(tmp_path) not in text


def test_existing_output_requires_force(tmp_path):
    facts = _write_facts(
        tmp_path / "world_facts.jsonl", [_fact("Force")])
    out = tmp_path / "train.jsonl"

    assert main(["--facts", str(facts), "--out", str(out)]) == 0
    first = out.read_bytes()

    assert main(["--facts", str(facts), "--out", str(out)]) == 2
    assert out.read_bytes() == first

    assert main([
        "--facts", str(facts), "--out", str(out), "--force",
    ]) == 0
    assert out.read_bytes() == first
    assert len(_read_jsonl(out)) == 1


def test_missing_facts_file_fails_without_writing_output(tmp_path):
    out = tmp_path / "train.jsonl"
    assert main([
        "--facts", str(tmp_path / "missing.jsonl"), "--out", str(out),
    ]) == 2
    assert not out.exists()


def test_invalid_limits_and_style_raise(tmp_path):
    facts = _write_facts(
        tmp_path / "world_facts.jsonl", [_fact("Limits")])

    with pytest.raises(ContentRowsError):
        build_rows([], style="unknown")
    with pytest.raises(ContentRowsError):
        build_rows([], max_rows=-1)
    with pytest.raises(ContentRowsError):
        build_rows([], min_chars=-1)
    with pytest.raises(ContentRowsError):
        build_rows([], max_answer_chars=0)
    with pytest.raises(ContentRowsError):
        run_build(facts, tmp_path / "train.jsonl", lang="")


def test_clean_extract_prefers_sentences_and_falls_back_to_words():
    text = "One complete sentence. Another complete sentence here."
    assert clean_extract(text, 22) == "One complete sentence."
    assert clean_extract(text, 10) == "One"
    assert clean_extract(text, 0) == ""
