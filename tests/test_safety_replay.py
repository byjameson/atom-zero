import hashlib
import json
from collections import Counter

import pytest

from research import build_safety_replay as replay
from research.build_safety_probes import FAMILY_ORDER, SEVERITIES
from research.checkpoint_gate import read_safety_rows

TRAIN_FIELDS = {"user", "assistant", "family", "severity", "source"}
EVAL_FIELDS = {
    "user", "forbidden", "expect", "family", "severity", "source",
}
CONTENT_KEYS = {
    "user", "prompt", "assistant", "forbidden", "messages", "text", "rows",
    "reply",
}


def _read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _build(root, *, per_family=4, seed=17):
    rows = replay.build_rows(per_family=per_family, seed=seed)
    train = root / "train.jsonl"
    evals = root / "eval.jsonl"
    manifest_path = root / "manifest.json"
    manifest = replay.write_dataset(
        rows["train"], rows["eval"], train, evals,
        manifest=manifest_path, seed=seed, per_family=per_family)
    return train, evals, manifest_path, rows, manifest


def _manifest_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _manifest_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _manifest_keys(item)


def test_build_is_deterministic_with_expected_counts():
    rows = replay.build_rows(
        per_family=replay.DEFAULT_PER_FAMILY, seed=replay.DEFAULT_SEED)
    again = replay.build_rows(
        per_family=replay.DEFAULT_PER_FAMILY, seed=replay.DEFAULT_SEED)

    assert rows == again
    assert replay.FAMILY_ORDER == FAMILY_ORDER
    assert replay.FAMILY_ORDER == (
        "prompt_injection",
        "secret_exfiltration",
        "harmful_or_illegal",
        "privacy_personal_data",
        "self_harm_crisis",
    )
    assert len(rows["train"]) == (
        replay.DEFAULT_PER_FAMILY * len(replay.FAMILY_ORDER))
    assert len(rows["eval"]) == (
        replay.DEFAULT_PER_FAMILY * len(replay.FAMILY_ORDER))
    assert len(rows["train"]) == 40
    assert len(rows["eval"]) == 40
    for split in ("train", "eval"):
        counts = Counter(row["family"] for row in rows[split])
        assert counts == {
            family: replay.DEFAULT_PER_FAMILY
            for family in replay.FAMILY_ORDER
        }
    assert all(
        row["source"] == "synthetic_safety_replay"
        for split in ("train", "eval") for row in rows[split]
    )


def test_per_family_counts_and_seed_changes_order():
    small = replay.build_rows(per_family=3, seed=5)
    assert len(small["train"]) == 15
    assert len(small["eval"]) == 15
    counts = Counter(row["family"] for row in small["train"])
    assert counts == {family: 3 for family in replay.FAMILY_ORDER}

    first = replay.build_rows(per_family=8, seed=1)
    second = replay.build_rows(per_family=8, seed=2)
    assert json.dumps(first, sort_keys=True) != json.dumps(
        second, sort_keys=True)


def test_required_fields_and_row_shape():
    rows = replay.build_rows(per_family=4, seed=13)
    for row in rows["train"]:
        assert set(row) == TRAIN_FIELDS
        assert replay.validate_train_row(row)
    for row in rows["eval"]:
        assert set(row) == EVAL_FIELDS
        assert replay.validate_eval_row(row)
        assert row["expect"] == "refusal"
        assert isinstance(row["forbidden"], list) and row["forbidden"]
        assert len(row["forbidden"]) == len(set(row["forbidden"]))
        prompt = replay.normalize_text(row["user"])
        for term in row["forbidden"]:
            assert replay.normalize_text(term) not in prompt

    for family in replay.FAMILY_ORDER:
        assert any(
            row["family"] == family for row in rows["train"])
        assert any(row["family"] == family for row in rows["eval"])


def test_prompts_are_disjoint_between_splits():
    rows = replay.build_rows(per_family=8, seed=replay.DEFAULT_SEED)
    train_prompts = {row["user"] for row in rows["train"]}
    eval_prompts = {row["user"] for row in rows["eval"]}

    assert train_prompts
    assert eval_prompts
    assert not (train_prompts & eval_prompts)
    assert not ({
        replay.normalize_text(prompt) for prompt in train_prompts
    } & {
        replay.normalize_text(prompt) for prompt in eval_prompts
    })
    assert len(train_prompts) == len(rows["train"])
    assert len(eval_prompts) == len(rows["eval"])


def test_refusals_are_safe_and_name_a_boundary():
    rows = replay.build_rows(per_family=8, seed=replay.DEFAULT_SEED)
    forbidden = {
        replay.normalize_text(term)
        for term in replay.all_forbidden_terms()
    }
    operational = {term.casefold() for term in replay.OPERATIONAL_TERMS}

    for row in rows["train"]:
        reply = replay.normalize_text(row["assistant"])
        assert any(marker in reply for marker in replay.REFUSAL_MARKERS)
        assert not any(term in reply for term in forbidden)
        assert not any(term in reply for term in operational)
        assert row["assistant"] in replay.REFUSAL_TEMPLATES[row["family"]]
        assert len(row["assistant"]) < 240
        assert "i can" in reply or "please" in reply or "encourage" in reply


def test_manifest_is_content_free_and_matches_files(tmp_path):
    train, evals, manifest_path, rows, manifest = _build(
        tmp_path, per_family=4, seed=9)
    text = manifest_path.read_text(encoding="utf-8")

    assert manifest["schema"] == "safety_replay_v1"
    assert manifest["seed"] == 9
    assert manifest["per_family"] == 4
    assert manifest["total"] == 40
    assert manifest["overlap"] == 0
    assert manifest["disjoint"] is True
    assert manifest["content_free"] is True
    assert manifest["splits"]["train"]["count"] == len(rows["train"])
    assert manifest["splits"]["eval"]["count"] == len(rows["eval"])
    assert sum(
        manifest["splits"]["train"]["counts"]["by_family"].values()) == 20
    assert sum(
        manifest["splits"]["eval"]["counts"]["by_family"].values()) == 20
    for split in ("train", "eval"):
        assert manifest["splits"][split]["counts"]["by_family"] == {
            family: 4 for family in replay.FAMILY_ORDER
        }
        assert sum(
            manifest["splits"][split]["counts"]["by_severity"].values()
        ) == 20

    train_raw = train.read_bytes()
    eval_raw = evals.read_bytes()
    assert manifest["splits"]["train"]["sha256"] == hashlib.sha256(
        train_raw).hexdigest()
    assert manifest["splits"]["eval"]["sha256"] == hashlib.sha256(
        eval_raw).hexdigest()
    assert manifest["splits"]["train"]["bytes"] == len(train_raw)
    assert manifest["splits"]["eval"]["bytes"] == len(eval_raw)

    assert not (set(_manifest_keys(manifest)) & CONTENT_KEYS)
    for row in rows["train"]:
        assert row["user"] not in text
        assert row["assistant"] not in text
    for row in rows["eval"]:
        assert row["user"] not in text
        for term in row["forbidden"]:
            assert term not in text


def test_existing_output_requires_force(tmp_path):
    train = tmp_path / "train.jsonl"
    evals = tmp_path / "eval.jsonl"
    manifest = tmp_path / "manifest.json"
    args = [
        "--train-out", str(train),
        "--eval-out", str(evals),
        "--manifest", str(manifest),
        "--per-family", "2",
        "--seed", "11",
    ]

    assert replay.main(args) == 0
    first_train = train.read_text(encoding="utf-8")
    first_eval = evals.read_text(encoding="utf-8")
    first_manifest = manifest.read_text(encoding="utf-8")
    assert len(_read_jsonl(train)) == 2 * len(replay.FAMILY_ORDER)
    assert len(_read_jsonl(evals)) == 2 * len(replay.FAMILY_ORDER)

    with pytest.raises(SystemExit) as excinfo:
        replay.main(args)
    assert excinfo.value.code == 2
    assert train.read_text(encoding="utf-8") == first_train
    assert evals.read_text(encoding="utf-8") == first_eval
    assert manifest.read_text(encoding="utf-8") == first_manifest

    # A remaining sibling also blocks a write that would create the missing
    # file, and nothing is written before the check.
    train.unlink()
    with pytest.raises(SystemExit) as excinfo:
        replay.main(args)
    assert excinfo.value.code == 2
    assert not train.exists()
    assert evals.read_text(encoding="utf-8") == first_eval
    assert manifest.read_text(encoding="utf-8") == first_manifest

    assert replay.main(args + ["--force"]) == 0
    assert train.read_text(encoding="utf-8") == first_train

    larger = [
        "--train-out", str(train),
        "--eval-out", str(evals),
        "--manifest", str(manifest),
        "--per-family", "3",
        "--seed", "11",
        "--force",
    ]
    assert replay.main(larger) == 0
    assert len(_read_jsonl(train)) == 3 * len(replay.FAMILY_ORDER)
    assert len(_read_jsonl(evals)) == 3 * len(replay.FAMILY_ORDER)


def test_refusals_pass_the_local_filter_and_gate_marker():
    from atomzero.safety import SafetyFilter
    from research.checkpoint_gate import _is_refusal

    rows = replay.build_rows(per_family=8, seed=replay.DEFAULT_SEED)
    safety = SafetyFilter()
    for row in rows["train"]:
        _filtered, verdict = safety.filter_reply(row["assistant"])
        assert verdict is None
        assert _is_refusal(row["assistant"])


def test_eval_rows_are_consumable_by_checkpoint_gate(tmp_path):
    evals = tmp_path / "eval.jsonl"
    train = tmp_path / "train.jsonl"
    manifest = tmp_path / "manifest.json"
    rows = replay.build_rows(
        per_family=replay.DEFAULT_PER_FAMILY, seed=replay.DEFAULT_SEED)
    replay.write_dataset(
        rows["train"], rows["eval"], train, evals,
        manifest=manifest, seed=replay.DEFAULT_SEED,
        per_family=replay.DEFAULT_PER_FAMILY)

    parsed, invalid = read_safety_rows(evals)
    assert invalid == 0
    assert len(parsed) == len(rows["eval"]) == 40
    for raw, row in zip(rows["eval"], parsed):
        assert "assistant" not in raw
        assert set(row) == {"user", "expect", "forbidden"}
        assert row["user"] == raw["user"]
        assert row["expect"] == "refusal"
        assert row["forbidden"] == raw["forbidden"]


def test_invalid_per_family_bounds():
    with pytest.raises(ValueError):
        replay.build_rows(per_family=0)
    with pytest.raises(ValueError):
        replay.build_rows(per_family=999)


def test_every_family_uses_known_and_valid_severities():
    rows = replay.build_rows(per_family=8, seed=23)
    for split in ("train", "eval"):
        for row in rows[split]:
            assert row["family"] in FAMILY_ORDER
            assert row["severity"] in SEVERITIES
    assert all(
        row["expect"] == "refusal" for row in rows["eval"])
