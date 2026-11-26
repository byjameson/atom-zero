"""Tests for contrastive self-harm replay construction."""
import hashlib
import json
import re
from collections import Counter

import pytest

from research import build_selfharm_replay as replay

ROW_FIELDS = {"user", "assistant", "family", "kind", "source"}
CONTENT_KEYS = {
    "user", "prompt", "assistant", "reply", "messages", "text", "rows",
}


def _read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _build(root, *, per_family=3, seed=53, with_eval=False):
    rows = replay.build_rows(per_family=per_family, seed=seed)
    train = root / "train.jsonl"
    manifest_path = root / "manifest.json"
    eval_rows = None
    evals = None
    if with_eval:
        eval_rows = replay.build_eval_rows(
            per_family=per_family, seed=seed)
        evals = root / "eval.jsonl"
    summary = replay.write_dataset(
        rows, train, eval_rows=eval_rows, eval_out=evals,
        manifest=manifest_path, seed=seed, per_family=per_family)
    return train, evals, manifest_path, rows, eval_rows, summary


def _manifest_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _manifest_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _manifest_keys(item)


def _tokens(value):
    return set(re.findall(r"[a-z0-9]+", replay.normalize_text(value)))


def _jaccard(left, right):
    return len(left & right) / len(left | right)


def test_default_counts_are_paired_and_deterministic():
    rows = replay.build_rows()
    again = replay.build_rows()

    assert rows == again
    assert len(rows) == replay.DEFAULT_PER_FAMILY * 2
    assert len(rows) == 16
    counts = Counter(row["kind"] for row in rows)
    assert counts == {"crisis": 8, "benign": 8}
    for index in range(0, len(rows), 2):
        crisis, benign = rows[index:index + 2]
        assert crisis["kind"] == "crisis"
        assert benign["kind"] == "benign"
        assert crisis["family"] == "self_harm_crisis"
        assert benign["family"] == "benign_support"
    for row in rows:
        assert set(row) == ROW_FIELDS
        assert row["source"] == "synthetic_contrastive_selfharm"
        assert replay.validate_row(row)


def test_seed_changes_selection_and_per_family_controls_size():
    first = replay.build_rows(per_family=8, seed=53)
    second = replay.build_rows(per_family=8, seed=53)
    other = replay.build_rows(per_family=8, seed=54)

    assert first == second
    assert other != first
    assert len(other) == len(first)

    small = replay.build_rows(per_family=3, seed=7)
    assert len(small) == 6
    assert Counter(row["kind"] for row in small) == {
        "crisis": 3, "benign": 3,
    }

    for bad in (0, -1, 999):
        with pytest.raises(ValueError):
            replay.build_rows(per_family=bad)
    with pytest.raises(ValueError):
        replay.build_rows(per_family=True)
    with pytest.raises(ValueError):
        replay.build_rows(seed=True)
    with pytest.raises(ValueError):
        replay.build_rows(seed=1.5)


def test_crisis_and_benign_prompts_are_distinct_and_unique():
    rows = replay.build_rows(per_family=8, seed=53)
    prompts = [replay.normalize_text(row["user"]) for row in rows]
    crisis = [
        row for row in rows if row["kind"] == "crisis"
    ]
    benign = [
        row for row in rows if row["kind"] == "benign"
    ]

    assert len(set(prompts)) == len(prompts)
    assert len(crisis) == len(benign) == 8
    crisis_prompts = {replay.normalize_text(row["user"]) for row in crisis}
    benign_prompts = {replay.normalize_text(row["user"]) for row in benign}
    assert not (crisis_prompts & benign_prompts)
    for row in crisis:
        prompt = replay.normalize_text(row["user"])
        assert any(term in prompt for term in replay.CRISIS_TERMS)
    for row in benign:
        prompt = replay.normalize_text(row["user"])
        assert not any(term in prompt for term in replay.CRISIS_TERMS)

    # A benign prompt must not be a lightly edited crisis prompt. A low token
    # overlap is a simple guard against relabelling or cosmetic rewrites.
    crisis_tokens = [_tokens(row["user"]) for row in crisis]
    benign_tokens = [_tokens(row["user"]) for row in benign]
    assert max(
        _jaccard(left, right)
        for left in crisis_tokens for right in benign_tokens
    ) < 0.5


def test_crisis_answers_are_supportive_refusals_without_method_words():
    rows = replay.build_rows(per_family=8, seed=53)

    for row in rows:
        if row["kind"] != "crisis":
            continue
        answer = row["assistant"]
        reply = replay.normalize_text(answer)
        assert any(marker in reply for marker in replay.REFUSAL_MARKERS)
        assert any(marker in reply for marker in replay.BOUNDARY_MARKERS)
        assert any(term in reply for term in replay.HELP_SEEKING_TERMS)
        assert not any(term in reply for term in replay.METHOD_TERMS)
        assert reply.startswith(replay.EMPATHY_OPENERS)
        assert len(answer) < 400
        assert 2 <= replay.sentence_count(answer) <= 3
        assert replay.validate_crisis_answer(answer)


def test_benign_answers_have_no_refusal_markers():
    rows = replay.build_rows(per_family=8, seed=53)

    for row in rows:
        if row["kind"] != "benign":
            continue
        answer = row["assistant"]
        reply = replay.normalize_text(answer)
        assert not any(marker in reply for marker in replay.REFUSAL_MARKERS)
        assert not any(marker in reply for marker in replay.BOUNDARY_MARKERS)
        assert replay.MIN_BENIGN_ANSWER_CHARS <= len(answer) < 400
        assert 1 <= replay.sentence_count(answer) <= 3
        assert replay.validate_benign_answer(answer)


def test_all_answers_pass_the_local_safety_filter():
    from atomzero.safety import SafetyFilter

    safety = SafetyFilter()
    for row in replay.build_rows(per_family=8, seed=53):
        _filtered, verdict = safety.filter_reply(row["assistant"])
        assert verdict is None, row


def test_manifest_is_content_free_and_matches_both_files(tmp_path):
    train, evals, manifest_path, rows, eval_rows, manifest = _build(
        tmp_path, per_family=3, seed=9, with_eval=True)
    text = manifest_path.read_text(encoding="utf-8")

    assert manifest["schema"] == "selfharm_replay_v1"
    assert manifest["source"] == "synthetic_contrastive_selfharm"
    assert manifest["seed"] == 9
    assert manifest["per_family"] == 3
    assert manifest["total"] == 6
    assert manifest["counts"]["by_kind"] == {"crisis": 3, "benign": 3}
    assert manifest["sha256"] == hashlib.sha256(
        train.read_bytes()).hexdigest()
    assert manifest["bytes"] == len(train.read_bytes())
    assert manifest["content_free"] is True
    assert manifest["paired"] is True

    assert manifest["splits"]["train"]["count"] == len(rows) == 6
    assert manifest["splits"]["eval"]["count"] == len(eval_rows) == 6
    assert manifest["splits"]["eval"]["sha256"] == hashlib.sha256(
        evals.read_bytes()).hexdigest()
    assert manifest["splits"]["eval"]["counts"]["by_kind"] == {
        "crisis": 3, "benign": 3,
    }
    assert manifest["total_all"] == 12
    assert manifest["overlap"] == 0
    assert manifest["disjoint"] is True

    assert not (set(_manifest_keys(manifest)) & CONTENT_KEYS)
    for row in rows + eval_rows:
        assert row["user"] not in text
        assert row["assistant"] not in text
    assert str(tmp_path) not in text
    assert len(_read_jsonl(train)) == len(rows)
    assert len(_read_jsonl(evals)) == len(eval_rows)


def test_eval_split_is_disjoint_from_training_prompts():
    train_rows = replay.build_rows(per_family=8, seed=53)
    eval_rows = replay.build_eval_rows(per_family=8, seed=53)

    assert len(eval_rows) == 16
    assert Counter(row["kind"] for row in eval_rows) == {
        "crisis": 8, "benign": 8,
    }
    train_prompts = {
        replay.normalize_text(row["user"]) for row in train_rows
    }
    eval_prompts = {
        replay.normalize_text(row["user"]) for row in eval_rows
    }
    assert train_prompts
    assert eval_prompts
    assert not (train_prompts & eval_prompts)
    for row in eval_rows:
        assert replay.validate_row(row)


def test_existing_output_requires_force(tmp_path):
    train = tmp_path / "train.jsonl"
    manifest = tmp_path / "manifest.json"
    args = [
        "--train-out", str(train),
        "--manifest", str(manifest),
        "--per-family", "2",
        "--seed", "11",
    ]

    assert replay.main(args) == 0
    first_train = train.read_bytes()
    first_manifest = manifest.read_bytes()
    assert len(_read_jsonl(train)) == 4

    with pytest.raises(SystemExit) as excinfo:
        replay.main(args)
    assert excinfo.value.code == 2
    assert train.read_bytes() == first_train
    assert manifest.read_bytes() == first_manifest

    # A remaining sibling blocks a write that would create the missing file,
    # and nothing is written before the check.
    train.unlink()
    with pytest.raises(SystemExit) as excinfo:
        replay.main(args)
    assert excinfo.value.code == 2
    assert not train.exists()
    assert manifest.read_bytes() == first_manifest

    assert replay.main(args + ["--force"]) == 0
    assert train.read_bytes() == first_train
    assert manifest.read_bytes() == first_manifest

    larger = [
        "--train-out", str(train),
        "--manifest", str(manifest),
        "--per-family", "3",
        "--seed", "11",
        "--force",
    ]
    assert replay.main(larger) == 0
    assert len(_read_jsonl(train)) == 6


def test_optional_eval_output_and_force(tmp_path):
    train = tmp_path / "train.jsonl"
    evals = tmp_path / "eval.jsonl"
    manifest = tmp_path / "manifest.json"
    args = [
        "--train-out", str(train),
        "--eval-out", str(evals),
        "--manifest", str(manifest),
        "--per-family", "2",
        "--seed", "13",
    ]

    assert replay.main(args) == 0
    first_train = train.read_bytes()
    first_eval = evals.read_bytes()
    first_manifest = manifest.read_bytes()
    assert len(_read_jsonl(train)) == 4
    assert len(_read_jsonl(evals)) == 4
    summary = json.loads(first_manifest)
    assert summary["splits"]["eval"]["count"] == 4
    assert summary["total_all"] == 8

    with pytest.raises(SystemExit) as excinfo:
        replay.main(args)
    assert excinfo.value.code == 2
    assert train.read_bytes() == first_train
    assert evals.read_bytes() == first_eval
    assert manifest.read_bytes() == first_manifest

    assert replay.main(args + ["--force"]) == 0
    assert train.read_bytes() == first_train
    assert evals.read_bytes() == first_eval
    assert manifest.read_bytes() == first_manifest


def test_eval_rows_and_path_must_be_used_together(tmp_path):
    rows = replay.build_rows(per_family=1)
    with pytest.raises(ValueError):
        replay.write_dataset(
            rows, tmp_path / "train.jsonl",
            manifest=tmp_path / "manifest.json",
            eval_rows=replay.build_eval_rows(per_family=1))
    with pytest.raises(ValueError):
        replay.write_dataset(
            rows, tmp_path / "train.jsonl",
            manifest=tmp_path / "manifest.json",
            eval_out=tmp_path / "eval.jsonl")


def test_same_output_and_manifest_path_is_rejected(tmp_path):
    rows = replay.build_rows(per_family=1)
    path = tmp_path / "same.jsonl"
    with pytest.raises(ValueError):
        replay.write_dataset(rows, path, manifest=path)
