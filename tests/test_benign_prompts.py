"""Tests for the deterministic benign prompt suite builder."""
import hashlib
import json
from collections import Counter

import pytest

from research import build_benign_prompts as builder

SINGLE_TURN_FIELDS = {
    "id", "user", "category", "expect", "forbidden_refusal_markers",
}
MULTI_TURN_FIELDS = {"id", "category", "expect", "messages"}
CONTENT_KEYS = {
    "user", "prompt", "messages", "content", "text", "rows", "assistant",
}
CATEGORIES = set(builder.CATEGORY_ORDER)
DEFAULT_SINGLE = builder.DEFAULT_PER_CATEGORY * len(builder.CATEGORY_ORDER)
DEFAULT_MULTI = (
    builder.DEFAULT_MULTI_TURN_PER_CATEGORY * len(builder.CATEGORY_ORDER))


def _read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _single_turn(rows):
    return [row for row in rows if "user" in row]


def _multi_turn(rows):
    return [row for row in rows if "messages" in row]


def _manifest_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _manifest_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _manifest_keys(item)


def _write(root, *, rows=None, seed=17):
    rows = rows if rows is not None else builder.build_rows(seed=seed)
    out = root / "prompts.jsonl"
    manifest_path = root / "manifest.json"
    manifest = builder.write_dataset(
        rows, out, manifest=manifest_path, seed=seed)
    return out, manifest_path, manifest


def test_default_counts_and_row_shape():
    rows = builder.build_rows()
    single = _single_turn(rows)
    multi = _multi_turn(rows)

    assert len(single) == DEFAULT_SINGLE == 120
    assert len(single) == builder.TOTAL
    assert len(multi) == DEFAULT_MULTI == 16
    assert len(rows) == 136
    assert Counter(row["category"] for row in single) == {
        category: builder.DEFAULT_PER_CATEGORY
        for category in builder.CATEGORY_ORDER
    }
    assert Counter(row["category"] for row in multi) == {
        category: builder.DEFAULT_MULTI_TURN_PER_CATEGORY
        for category in builder.CATEGORY_ORDER
    }
    assert {row["category"] for row in rows} == CATEGORIES

    for row in single:
        assert set(row) == SINGLE_TURN_FIELDS
        assert row["expect"] == "answer"
        assert row["forbidden_refusal_markers"] == []
        assert builder.validate_row(row)
    for row in multi:
        assert set(row) == MULTI_TURN_FIELDS
        assert row["expect"] == "answer"
        assert builder.validate_row(row)


def test_per_category_controls_size_and_rejects_invalid_values():
    rows = builder.build_rows(
        per_category=5, multi_turn_per_category=0, seed=13)
    assert len(rows) == 5 * len(builder.CATEGORY_ORDER)
    assert Counter(row["category"] for row in rows) == {
        category: 5 for category in builder.CATEGORY_ORDER
    }
    assert all(builder.validate_row(row) for row in rows)
    assert not _multi_turn(rows)

    both = builder.build_rows(
        per_category=1, multi_turn_per_category=1, seed=13)
    assert len(both) == 2 * len(builder.CATEGORY_ORDER)
    assert len(_single_turn(both)) == len(builder.CATEGORY_ORDER)
    assert len(_multi_turn(both)) == len(builder.CATEGORY_ORDER)

    full = builder.build_rows(
        per_category=18, multi_turn_per_category=3, seed=13)
    assert len(_single_turn(full)) == 18 * len(builder.CATEGORY_ORDER)
    assert len(_multi_turn(full)) == 3 * len(builder.CATEGORY_ORDER)

    for bad in (0, -1, 19):
        with pytest.raises(ValueError):
            builder.build_rows(per_category=bad)
    for bad in (-1, 4):
        with pytest.raises(ValueError):
            builder.build_rows(multi_turn_per_category=bad)
    with pytest.raises(ValueError):
        builder.build_rows(per_category=True)
    with pytest.raises(ValueError):
        builder.build_rows(multi_turn_per_category=True)
    with pytest.raises(ValueError):
        builder.build_rows(seed=True)
    with pytest.raises(ValueError):
        builder.build_rows(seed=1.5)


def test_template_pools_are_large_and_unique():
    all_prompts = []
    for category in builder.CATEGORY_ORDER:
        pool = builder.PROMPTS[category]
        assert len(pool) >= builder.DEFAULT_PER_CATEGORY
        assert len(pool) >= 15
        assert len({template_id for template_id, _ in pool}) == len(pool)
        all_prompts.extend(text for _, text in pool)

        multi_pool = builder.MULTI_TURN_PROMPTS[category]
        assert len(multi_pool) >= builder.DEFAULT_MULTI_TURN_PER_CATEGORY
        assert len(multi_pool) >= 2
        assert len({item[0] for item in multi_pool}) == len(multi_pool)
        for _template_id, first, reply, follow_up in multi_pool:
            assert builder._is_single_line(first)
            assert builder._is_single_line(reply)
            assert builder._is_single_line(follow_up)

    normalized = [builder.normalize_text(text) for text in all_prompts]
    assert len(normalized) == len(set(normalized))


def test_multi_turn_rows_have_the_documented_shape():
    rows = _multi_turn(builder.build_rows())

    for row in rows:
        assert row["id"].startswith("benign-multi-")
        assert row["category"] in CATEGORIES
        assert row["expect"] == "answer"
        assert "user" not in row
        messages = row["messages"]
        assert isinstance(messages, list)
        assert len(messages) == 3
        assert [message["role"] for message in messages] == [
            "user", "assistant", "user"]
        assert all(set(message) == {"role", "content"}
                   for message in messages)
        assert all(builder._is_single_line(message["content"])
                   for message in messages)
        first_reply = messages[1]["content"]
        assert 3 <= len(first_reply.split()) <= 40
        assert first_reply.endswith((".", "?"))
        assert messages[2]["content"].strip().endswith(("?", "."))
        assert builder.validate_messages(messages)


def test_ids_and_prompts_are_unique_across_the_file():
    rows = builder.build_rows()
    ids = [row["id"] for row in rows]
    assert len(ids) == len(set(ids))

    single_prompts = {
        builder.normalize_text(row["user"])
        for row in _single_turn(rows)
    }
    assert len(single_prompts) == len(_single_turn(rows))

    signatures = [builder._conversation_signature(row) for row in rows]
    assert len(signatures) == len(set(signatures))

    user_texts = [
        builder.normalize_text(text)
        for row in rows
        for text in builder._user_texts(row)
    ]
    assert len(user_texts) == len(set(user_texts))


def test_build_is_deterministic_and_seed_changes_selection(tmp_path):
    first = builder.build_rows(seed=17)
    again = builder.build_rows(seed=17)
    assert first == again

    first_out, first_manifest, _ = _write(tmp_path / "first", seed=17)
    second_out, second_manifest, _ = _write(tmp_path / "second", seed=17)
    assert first_out.read_bytes() == second_out.read_bytes()
    assert first_manifest.read_bytes() == second_manifest.read_bytes()

    other = builder.build_rows(seed=18)
    assert len(other) == len(first)
    assert other != first
    assert {
        builder._conversation_signature(row) for row in other
    } != {
        builder._conversation_signature(row) for row in first
    }


def test_manifest_is_content_free_and_matches_the_output(tmp_path):
    rows = builder.build_rows(seed=9)
    manifest = builder.build_manifest(rows, seed=9)
    out, manifest_path, written = _write(
        tmp_path, rows=builder.build_rows(seed=9), seed=9)
    text = manifest_path.read_text(encoding="utf-8")

    assert manifest == written
    assert manifest["schema"] == "benign_prompts_v2"
    assert manifest["source"] == "authored_benign"
    assert manifest["seed"] == 9
    assert manifest["per_category"] == 15
    assert manifest["multi_turn_per_category"] == 2
    assert manifest["total"] == 136
    assert manifest["single_turn"] == 120
    assert manifest["multi_turn"] == 16
    assert manifest["counts"]["by_category"] == {
        category: 17 for category in builder.CATEGORY_ORDER
    }
    assert manifest["counts"]["single_turn_by_category"] == {
        category: 15 for category in builder.CATEGORY_ORDER
    }
    assert manifest["counts"]["multi_turn_by_category"] == {
        category: 2 for category in builder.CATEGORY_ORDER
    }
    assert manifest["counts"]["by_kind"] == {
        "single_turn": 120, "multi_turn": 16,
    }
    assert manifest["sha256"] == hashlib.sha256(
        out.read_bytes()).hexdigest()
    assert manifest["bytes"] == len(out.read_bytes())
    assert manifest["content_free"] is True

    assert not (set(_manifest_keys(manifest)) & CONTENT_KEYS)
    for row in rows:
        if "user" in row:
            assert row["user"] not in text
            assert builder.normalize_text(row["user"]) not in (
                builder.normalize_text(text))
        for message in row.get("messages", []):
            assert message["content"] not in text
        assert row["id"] not in text
    assert str(tmp_path) not in text
    assert len(_read_jsonl(out)) == len(rows)


def test_validate_row_rejects_malformed_multi_turn_rows():
    valid = _multi_turn(builder.build_rows())[0]
    assert builder.validate_row(valid)

    wrong_roles = json.loads(json.dumps(valid))
    wrong_roles["messages"][1]["role"] = "user"
    assert not builder.validate_row(wrong_roles)

    too_short = json.loads(json.dumps(valid))
    too_short["messages"] = too_short["messages"][:2]
    assert not builder.validate_row(too_short)

    empty_content = json.loads(json.dumps(valid))
    empty_content["messages"][0]["content"] = "  "
    assert not builder.validate_row(empty_content)

    both_forms = json.loads(json.dumps(valid))
    both_forms["user"] = "duplicate form"
    assert not builder.validate_row(both_forms)

    not_a_list = json.loads(json.dumps(valid))
    not_a_list["messages"] = "user: hello"
    assert not builder.validate_row(not_a_list)


def test_existing_output_requires_force(tmp_path):
    out = tmp_path / "prompts.jsonl"
    manifest = tmp_path / "manifest.json"
    args = [
        "--out", str(out),
        "--manifest", str(manifest),
        "--per-category", "2",
        "--multi-turn-per-category", "1",
        "--seed", "11",
    ]

    assert builder.main(args) == 0
    first_out = out.read_bytes()
    first_manifest = manifest.read_bytes()
    expected = (
        2 * len(builder.CATEGORY_ORDER)
        + 1 * len(builder.CATEGORY_ORDER))
    assert len(_read_jsonl(out)) == expected

    with pytest.raises(SystemExit) as excinfo:
        builder.main(args)
    assert excinfo.value.code == 2
    assert out.read_bytes() == first_out
    assert manifest.read_bytes() == first_manifest

    # A remaining sibling blocks a write that would create the missing file.
    out.unlink()
    with pytest.raises(SystemExit) as excinfo:
        builder.main(args)
    assert excinfo.value.code == 2
    assert not out.exists()
    assert manifest.read_bytes() == first_manifest

    assert builder.main(args + ["--force"]) == 0
    assert out.read_bytes() == first_out
    assert manifest.read_bytes() == first_manifest

    larger = [
        "--out", str(out),
        "--manifest", str(manifest),
        "--per-category", "3",
        "--multi-turn-per-category", "2",
        "--seed", "11",
        "--force",
    ]
    assert builder.main(larger) == 0
    assert len(_read_jsonl(out)) == 5 * len(builder.CATEGORY_ORDER)


def test_cli_default_writes_120_single_and_16_multi(tmp_path):
    out = tmp_path / "prompts.jsonl"
    manifest = tmp_path / "manifest.json"
    args = ["--out", str(out), "--manifest", str(manifest)]

    assert builder.main(args) == 0
    rows = _read_jsonl(out)
    assert len(_single_turn(rows)) == 120
    assert len(_multi_turn(rows)) == 16
    summary = json.loads(manifest.read_text(encoding="utf-8"))
    assert summary["total"] == 136
    assert summary["single_turn"] == 120
    assert summary["multi_turn"] == 16
