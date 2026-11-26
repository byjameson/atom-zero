"""Tests for deterministic benign chat replay construction."""
import hashlib
import json
from collections import Counter

import pytest

from research import build_benign_chat as chat

ROW_FIELDS = {"user", "assistant", "family", "source", "style"}
CONTENT_KEYS = {
    "user", "prompt", "assistant", "reply", "messages", "text", "rows",
}


def _read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _build(root, *, per_category=4, seed=41):
    rows = chat.build_rows(per_category=per_category, seed=seed)
    out = root / "train.jsonl"
    manifest_path = root / "manifest.json"
    manifest = chat.write_dataset(
        rows, out, manifest=manifest_path, seed=seed,
        per_category=per_category)
    return out, manifest_path, rows, manifest


def _manifest_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _manifest_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _manifest_keys(item)


def test_default_counts_and_row_shape():
    rows = chat.build_rows()

    assert chat.DEFAULT_PER_CATEGORY == 60
    assert len(rows) == (
        chat.DEFAULT_PER_CATEGORY * len(chat.CATEGORY_ORDER))
    assert len(rows) == 480
    counts = Counter(row["family"] for row in rows)
    assert counts == {
        category: chat.DEFAULT_PER_CATEGORY
        for category in chat.CATEGORY_ORDER
    }
    assert counts == {category: 60 for category in chat.CATEGORY_ORDER}
    for row in rows:
        assert set(row) == ROW_FIELDS
        assert row["family"] in chat.CATEGORY_ORDER
        assert row["source"] == "synthetic_template"
        assert row["style"] == "benign_chat"
        assert chat.validate_row(row)


def test_cli_default_builds_480_rows(tmp_path):
    out = tmp_path / "train.jsonl"
    manifest = tmp_path / "manifest.json"

    assert chat.main([
        "--out", str(out),
        "--manifest", str(manifest),
    ]) == 0

    rows = _read_jsonl(out)
    assert len(rows) == 480
    assert Counter(row["family"] for row in rows) == {
        category: 60 for category in chat.CATEGORY_ORDER
    }
    summary = json.loads(manifest.read_text(encoding="utf-8"))
    assert summary["per_category"] == 60
    assert summary["total"] == 480


def test_build_is_deterministic_and_seed_changes_selection(tmp_path):
    first = chat.build_rows(seed=41)
    again = chat.build_rows(seed=41)
    assert first == again

    first_out, first_manifest, _, _ = _build(
        tmp_path / "first", seed=41)
    second_out, second_manifest, _, _ = _build(
        tmp_path / "second", seed=41)
    assert first_out.read_bytes() == second_out.read_bytes()
    assert first_manifest.read_bytes() == second_manifest.read_bytes()

    other = chat.build_rows(seed=42)
    assert len(other) == len(first)
    assert other != first
    assert {
        (row["user"], row["assistant"]) for row in other
    } != {
        (row["user"], row["assistant"]) for row in first
    }


def test_per_category_controls_size_and_rejects_invalid_values():
    rows = chat.build_rows(per_category=5, seed=13)
    assert len(rows) == 5 * len(chat.CATEGORY_ORDER)
    assert Counter(row["family"] for row in rows) == {
        category: 5 for category in chat.CATEGORY_ORDER
    }

    small = chat.build_rows(per_category=1, seed=13)
    assert len(small) == len(chat.CATEGORY_ORDER)

    for bad in (0, -1, 999):
        with pytest.raises(ValueError):
            chat.build_rows(per_category=bad)
    with pytest.raises(ValueError):
        chat.build_rows(per_category=True)
    with pytest.raises(ValueError):
        chat.build_rows(seed=1.5)
    with pytest.raises(ValueError):
        chat.build_rows(seed=True)


def test_pairs_are_unique_and_required_fields_are_exact():
    rows = chat.build_rows()

    pairs = {
        (chat.normalize_text(row["user"]),
         chat.normalize_text(row["assistant"]))
        for row in rows
    }
    assert len(pairs) == len(rows)
    assert {row["family"] for row in rows} == set(chat.CATEGORY_ORDER)
    assert all(set(row) == ROW_FIELDS for row in rows)
    assert all(row["user"].strip() for row in rows)
    assert all(row["assistant"].strip() for row in rows)
    assert all("\n" not in row["user"] for row in rows)
    assert all("\n" not in row["assistant"] for row in rows)


def test_answers_are_short_helpful_and_have_no_refusal_markers():
    rows = chat.build_rows()

    for row in rows:
        answer = row["assistant"]
        reply = chat.normalize_text(answer)
        assert not any(
            marker in reply for marker in chat.REFUSAL_MARKERS), answer
        assert chat.MIN_ANSWER_CHARS <= len(answer) <= chat.MAX_ANSWER_CHARS
        assert 1 <= chat.sentence_count(answer) <= 3
        assert chat.validate_answer(answer)
        assert not any(
            term in reply for term in chat.UNSAFE_TERMS), answer


def test_answers_pass_the_local_safety_filter():
    from atomzero.safety import SafetyFilter

    safety = SafetyFilter()
    for row in chat.build_rows():
        _filtered, verdict = safety.filter_reply(row["assistant"])
        assert verdict is None, row


def test_manifest_is_content_free_and_matches_the_output(tmp_path):
    out, manifest_path, rows, manifest = _build(
        tmp_path, per_category=3, seed=9)
    text = manifest_path.read_text(encoding="utf-8")

    assert manifest["schema"] == "benign_chat_v1"
    assert manifest["source"] == "synthetic_template"
    assert manifest["style"] == "benign_chat"
    assert manifest["seed"] == 9
    assert manifest["per_category"] == 3
    assert manifest["total"] == 3 * len(chat.CATEGORY_ORDER)
    assert manifest["counts"]["by_category"] == {
        category: 3 for category in chat.CATEGORY_ORDER
    }
    assert sum(manifest["counts"]["by_category"].values()) == len(rows)
    assert manifest["sha256"] == hashlib.sha256(
        out.read_bytes()).hexdigest()
    assert manifest["bytes"] == len(out.read_bytes())
    assert manifest["content_free"] is True

    assert not (set(_manifest_keys(manifest)) & CONTENT_KEYS)
    for row in rows:
        assert row["user"] not in text
        assert row["assistant"] not in text
    assert str(tmp_path) not in text
    assert len(_read_jsonl(out)) == len(rows)


def test_existing_output_requires_force(tmp_path):
    out = tmp_path / "train.jsonl"
    manifest = tmp_path / "manifest.json"
    args = [
        "--out", str(out),
        "--manifest", str(manifest),
        "--per-category", "2",
        "--seed", "11",
    ]

    assert chat.main(args) == 0
    first_out = out.read_bytes()
    first_manifest = manifest.read_bytes()
    assert len(_read_jsonl(out)) == 2 * len(chat.CATEGORY_ORDER)

    with pytest.raises(SystemExit) as excinfo:
        chat.main(args)
    assert excinfo.value.code == 2
    assert out.read_bytes() == first_out
    assert manifest.read_bytes() == first_manifest

    # A remaining sibling blocks a write that would create the missing file.
    out.unlink()
    with pytest.raises(SystemExit) as excinfo:
        chat.main(args)
    assert excinfo.value.code == 2
    assert not out.exists()
    assert manifest.read_bytes() == first_manifest

    assert chat.main(args + ["--force"]) == 0
    assert out.read_bytes() == first_out
    assert manifest.read_bytes() == first_manifest

    larger = [
        "--out", str(out),
        "--manifest", str(manifest),
        "--per-category", "3",
        "--seed", "11",
        "--force",
    ]
    assert chat.main(larger) == 0
    assert len(_read_jsonl(out)) == 3 * len(chat.CATEGORY_ORDER)


def test_every_category_has_multiple_prompts_and_answers():
    for category in chat.CATEGORY_ORDER:
        groups = chat.TEMPLATES[category]
        assert len(groups) >= 8
        for group in groups:
            assert len(group["prompts"]) >= 2
            assert len(group["answers"]) >= 2
            assert len(set(group["prompts"])) == len(group["prompts"])
            assert len(set(group["answers"])) == len(group["answers"])
        combinations = chat._combinations(category)
        assert len(combinations) == 72
        assert len(combinations) >= chat.DEFAULT_PER_CATEGORY
        assert len(combinations) >= 60
        assert len(set(combinations)) == len(combinations)


def test_full_template_pool_is_unique_and_valid():
    pairs = set()
    for category in chat.CATEGORY_ORDER:
        for user, assistant in chat._combinations(category):
            assert chat.validate_user(user)
            assert chat.validate_answer(assistant)
            key = (
                chat.normalize_text(user),
                chat.normalize_text(assistant),
            )
            assert key not in pairs
            pairs.add(key)
    assert len(pairs) == 72 * len(chat.CATEGORY_ORDER)
    assert len(pairs) == 576


def test_answer_styles_vary_in_length_and_sentence_count():
    answers = [
        answer
        for category in chat.CATEGORY_ORDER
        for group in chat.TEMPLATES[category]
        for answer in group["answers"]
    ]
    sentence_counts = {chat.sentence_count(answer) for answer in answers}
    lengths = {len(answer) for answer in answers}

    assert sentence_counts == {1, 2, 3}
    assert len(lengths) > 20
    assert max(lengths) - min(lengths) > 100


def test_invalid_output_and_manifest_paths_are_rejected(tmp_path):
    rows = chat.build_rows(per_category=1)
    path = tmp_path / "same.jsonl"
    with pytest.raises(ValueError):
        chat.write_dataset(rows, path, manifest=path)
