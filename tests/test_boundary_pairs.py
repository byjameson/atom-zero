"""Tests for contrastive boundary-pair construction."""
import hashlib
import json
from collections import Counter

import pytest

from research import build_boundary_pairs as boundary

ROW_FIELDS = {"user", "assistant", "family", "kind", "source"}
CONTENT_KEYS = {
    "user", "prompt", "assistant", "reply", "messages", "text", "rows",
    "examples", "samples",
}


def _read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _manifest_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _manifest_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _manifest_keys(item)


def _family_counts(rows, kind):
    return Counter(
        row["family"] for row in rows if row["kind"] == kind)


def test_default_counts_are_paired_and_deterministic():
    rows = boundary.build_rows()
    again = boundary.build_rows()

    assert rows == again
    assert len(rows) == 96
    assert len(rows) == boundary.DEFAULT_PER_FAMILY * 2 * 4
    for family in boundary.FAMILY_ORDER:
        assert _family_counts(rows, "unsafe")[family] == 12
        assert _family_counts(rows, "benign")[family] == 12
    assert Counter(row["kind"] for row in rows) == {
        "unsafe": 48, "benign": 48,
    }

    for index in range(0, len(rows), 2):
        unsafe, benign = rows[index:index + 2]
        assert unsafe["kind"] == "unsafe"
        assert benign["kind"] == "benign"
        assert unsafe["family"] == benign["family"]
    for row in rows:
        assert set(row) == ROW_FIELDS
        assert row["family"] in boundary.FAMILY_ORDER
        assert row["source"] == "synthetic_template"
        assert boundary.validate_row(row)


def test_per_family_and_seed_control_selection():
    first = boundary.build_rows(per_family=3, seed=67)
    second = boundary.build_rows(per_family=3, seed=67)
    other = boundary.build_rows(per_family=3, seed=68)

    assert first == second
    assert other != first
    assert len(first) == len(other) == 24
    assert Counter(row["kind"] for row in first) == {
        "unsafe": 12, "benign": 12,
    }
    for family in boundary.FAMILY_ORDER:
        assert _family_counts(first, "unsafe")[family] == 3
        assert _family_counts(first, "benign")[family] == 3

    largest = boundary.build_rows(per_family=14, seed=67)
    assert len(largest) == 112

    for bad in (0, -1, 15, 999):
        with pytest.raises(ValueError):
            boundary.build_rows(per_family=bad)
    with pytest.raises(ValueError):
        boundary.build_rows(per_family=True)
    with pytest.raises(ValueError):
        boundary.build_rows(seed=True)
    with pytest.raises(ValueError):
        boundary.build_rows(seed=1.5)


def test_unsafe_rows_are_short_boundary_refusals_without_methods():
    rows = boundary.build_rows()

    for row in rows:
        if row["kind"] != "unsafe":
            continue
        assert boundary.validate_unsafe_prompt(row["user"], row["family"])
        prompt = boundary.normalize_text(row["user"])
        assert any(
            term in prompt for term in boundary.UNSAFE_TERMS[row["family"]])
        assert not any(
            term in prompt for term in boundary.OPERATIONAL_TERMS)

        answer = row["assistant"]
        reply = boundary.normalize_text(answer)
        assert any(marker in reply for marker in boundary.REFUSAL_MARKERS)
        assert any(
            marker in reply
            for marker in boundary.BOUNDARY_MARKERS[row["family"]])
        assert any(marker in reply for marker in boundary.ALTERNATIVE_MARKERS)
        assert not any(term in reply for term in boundary.METHOD_TERMS)
        assert not any(term in reply for term in boundary.OPERATIONAL_TERMS)
        assert 1 <= boundary.sentence_count(answer) <= 2
        assert len(answer) <= boundary.MAX_ANSWER_CHARS
        assert boundary.validate_refusal(answer, row["family"])


def test_benign_rows_are_helpful_and_refusal_free():
    rows = boundary.build_rows()
    all_boundaries = tuple(
        term
        for terms in boundary.BOUNDARY_MARKERS.values()
        for term in terms
    )

    for index, row in enumerate(rows):
        if row["kind"] != "benign":
            continue
        answer = row["assistant"]
        reply = boundary.normalize_text(answer)
        assert not any(marker in reply for marker in boundary.REFUSAL_MARKERS)
        assert not any(marker in reply for marker in all_boundaries)
        assert boundary.MIN_BENIGN_ANSWER_CHARS <= len(answer)
        assert len(answer) <= boundary.MAX_ANSWER_CHARS
        assert 1 <= boundary.sentence_count(answer) <= 3
        assert boundary.validate_benign_answer(answer)
        assert boundary.validate_benign_prompt(row["user"])
        paired = rows[index - 1]
        assert paired["kind"] == "unsafe"
        assert boundary.normalize_text(
            row["user"]) != boundary.normalize_text(paired["user"])
        assert boundary.prompt_overlap(
            row["user"], paired["user"]) < 0.5


def test_rows_are_unique_and_benign_prompts_are_not_relabelled_unsafe():
    rows = boundary.build_rows()

    prompts = [boundary.normalize_text(row["user"]) for row in rows]
    pairs = [
        (boundary.normalize_text(row["user"]),
         boundary.normalize_text(row["assistant"]))
        for row in rows
    ]
    assert len(set(prompts)) == len(prompts)
    assert len(set(pairs)) == len(pairs)

    unsafe = [row for row in rows if row["kind"] == "unsafe"]
    benign = [row for row in rows if row["kind"] == "benign"]
    assert len(unsafe) == len(benign) == 48

    unsafe_prompts = {
        boundary.normalize_text(row["user"]) for row in unsafe
    }
    benign_prompts = {
        boundary.normalize_text(row["user"]) for row in benign
    }
    assert not (unsafe_prompts & benign_prompts)

    all_unsafe_terms = tuple(
        term
        for terms in boundary.UNSAFE_TERMS.values()
        for term in terms
    )
    for row in benign:
        prompt = boundary.normalize_text(row["user"])
        assert not any(term in prompt for term in all_unsafe_terms)


def test_manifest_is_content_free_and_matches_the_file(tmp_path):
    rows = boundary.build_rows(per_family=3, seed=9)
    out = tmp_path / "train.jsonl"
    manifest_path = tmp_path / "manifest.json"
    manifest = boundary.write_dataset(
        rows, out, manifest=manifest_path, seed=9, per_family=3)
    text = manifest_path.read_text(encoding="utf-8")

    assert manifest["schema"] == "boundary_pairs_v1"
    assert manifest["source"] == "synthetic_template"
    assert manifest["seed"] == 9
    assert manifest["per_family"] == 3
    assert manifest["families"] == list(boundary.FAMILY_ORDER)
    assert manifest["kinds"] == ["unsafe", "benign"]
    assert manifest["total"] == 24
    assert manifest["pairs"] == 12
    assert manifest["counts"]["by_kind"] == {"unsafe": 12, "benign": 12}
    for family in boundary.FAMILY_ORDER:
        assert manifest["counts"]["by_family"][family] == {
            "unsafe": 3, "benign": 3,
        }
    assert manifest["sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    assert manifest["bytes"] == len(out.read_bytes())
    assert manifest["content_free"] is True
    assert manifest["paired"] is True

    assert not (set(_manifest_keys(manifest)) & CONTENT_KEYS)
    for row in rows:
        assert row["user"] not in text
        assert row["assistant"] not in text
    assert str(tmp_path) not in text
    assert len(_read_jsonl(out)) == len(rows) == 24


def test_existing_output_requires_force(tmp_path):
    out = tmp_path / "train.jsonl"
    manifest = tmp_path / "manifest.json"
    args = [
        "--out", str(out),
        "--manifest", str(manifest),
        "--per-family", "2",
        "--seed", "11",
    ]

    assert boundary.main(args) == 0
    first_out = out.read_bytes()
    first_manifest = manifest.read_bytes()
    assert len(_read_jsonl(out)) == 16

    with pytest.raises(SystemExit) as excinfo:
        boundary.main(args)
    assert excinfo.value.code == 2
    assert out.read_bytes() == first_out
    assert manifest.read_bytes() == first_manifest

    # A remaining sibling blocks a write that would create the missing file,
    # and nothing is written before the check.
    out.unlink()
    with pytest.raises(SystemExit) as excinfo:
        boundary.main(args)
    assert excinfo.value.code == 2
    assert not out.exists()
    assert manifest.read_bytes() == first_manifest

    assert boundary.main(args + ["--force"]) == 0
    assert out.read_bytes() == first_out
    assert manifest.read_bytes() == first_manifest

    larger = [
        "--out", str(out),
        "--manifest", str(manifest),
        "--per-family", "3",
        "--seed", "11",
        "--force",
    ]
    assert boundary.main(larger) == 0
    assert len(_read_jsonl(out)) == 24


def test_same_output_and_manifest_path_is_rejected(tmp_path):
    rows = boundary.build_rows(per_family=1)
    path = tmp_path / "same.jsonl"
    with pytest.raises(ValueError):
        boundary.write_dataset(rows, path, manifest=path)
