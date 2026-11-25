"""Tests for deterministic copy/content training mix construction."""
import hashlib
import json

import pytest

import research.mix_training_data as mixer
from research.mix_training_data import (
    MixError,
    assert_safe_input,
    main,
    mix_rows,
    normalize_pair,
    read_source,
    run_mix,
)


def _write_jsonl(path, rows, raw_lines=()):
    lines = [json.dumps(row, ensure_ascii=False) for row in rows]
    lines.extend(raw_lines)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _rows(prefix, count):
    return [{"user": f"{prefix} question {index}",
             "assistant": f"{prefix} answer {index}"}
            for index in range(count)]


def _read_jsonl(path):
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_mix_is_deterministic_for_the_same_seed(tmp_path):
    copy = _write_jsonl(tmp_path / "copy.jsonl", _rows("copy", 8))
    content = _write_jsonl(tmp_path / "content.jsonl", _rows("content", 6))

    first = run_mix(copy, content, tmp_path / "first.jsonl",
                    seed=11, shuffle=True,
                    manifest_path=tmp_path / "first_manifest.json")
    second = run_mix(copy, content, tmp_path / "second.jsonl",
                     seed=11, shuffle=True,
                     manifest_path=tmp_path / "second_manifest.json")

    first_bytes = (tmp_path / "first.jsonl").read_bytes()
    second_bytes = (tmp_path / "second.jsonl").read_bytes()
    assert first_bytes == second_bytes
    assert first["output_sha256"] == second["output_sha256"]
    assert first["final_rows"] == second["final_rows"] == 14

    third = run_mix(copy, content, tmp_path / "third.jsonl",
                    seed=12, shuffle=True)
    assert third["output_sha256"] != first["output_sha256"]


def test_weights_set_per_source_quotas(tmp_path):
    copy = _write_jsonl(tmp_path / "copy.jsonl", _rows("copy", 10))
    content_a = _write_jsonl(tmp_path / "content_a.jsonl",
                             _rows("content-a", 6))
    content_b = _write_jsonl(tmp_path / "content_b.jsonl",
                             _rows("content-b", 6))

    rows, manifest = mix_rows(
        copy, [content_a, content_b],
        copy_weight=1.0, content_weight=0.5, shuffle=True, seed=3)

    assert len(rows) == 16
    assert manifest["final_counts"] == {"copy": 10, "content": 6}
    assert [source["requested"] for source in manifest["sources"]] == [
        10, 3, 3]
    assert [source["selected"] for source in manifest["sources"]] == [
        10, 3, 3]
    assert manifest["filled_rows"] == 0
    assert manifest["unfilled_rows"] == 0


def test_dedupe_uses_normalized_pair_and_keeps_first(tmp_path):
    first = {"user": "Hello  world", "assistant": "A  B"}
    duplicate = {"user": " Hello world ", "assistant": "A B",
                 "family": "duplicate", "keys": ["k"]}
    unique = {"user": "Other", "assistant": "Answer", "family": "qa",
              "keys": ["x", "y"]}
    copy = _write_jsonl(tmp_path / "copy.jsonl",
                        [first, duplicate, unique])
    content = _write_jsonl(
        tmp_path / "content.jsonl",
        [{"user": "Hello world", "assistant": "A B"}])

    rows, manifest = mix_rows(copy, content)

    assert normalize_pair(" Hello  world ", "A\nB") == (
        "Hello world", "A B")
    assert rows == [first, unique]
    assert manifest["duplicates"] == 2
    assert manifest["final_rows"] == 2
    assert manifest["sources"][1]["duplicates"] == 1


def test_malformed_rows_are_skipped_and_counted(tmp_path):
    good = {"user": "good", "assistant": "row", "family": "qa",
            "keys": ["k"]}
    source = _write_jsonl(
        tmp_path / "copy.jsonl", [good],
        raw_lines=[
            "{not json",
            "[]",
            json.dumps({"assistant": "missing user"}),
            json.dumps({"user": "missing assistant"}),
            json.dumps({"user": 3, "assistant": "bad user type"}),
            json.dumps({"user": "u", "assistant": "a", "keys": "not-list"}),
            "",
        ])

    rows, counts = read_source(source)
    mixed, manifest = mix_rows(source)

    assert rows == [good]
    assert counts == {"rows_read": 7, "skipped": 6, "blank_lines": 1}
    assert mixed == [good]
    assert manifest["skipped"] == 6
    assert manifest["final_rows"] == 1


def test_eval_paths_are_refused_before_any_read(tmp_path, monkeypatch):
    copy = _write_jsonl(tmp_path / "copy.jsonl", _rows("copy", 2))
    eval_file = _write_jsonl(tmp_path / "eval_rows.jsonl",
                             [{"user": "secret", "assistant": "answer"}])

    def fail_read(*args, **kwargs):
        raise AssertionError("mixer read a file before the eval guard")

    monkeypatch.setattr(mixer, "read_source", fail_read)
    with pytest.raises(MixError):
        mix_rows([copy, eval_file])

    code = main([
        "--copy", str(copy), str(eval_file),
        "--out", str(tmp_path / "mixed.jsonl"),
        "--manifest", str(tmp_path / "manifest.json"),
    ])
    assert code == 2
    assert not (tmp_path / "mixed.jsonl").exists()
    assert not (tmp_path / "manifest.json").exists()

    with pytest.raises(MixError):
        assert_safe_input(tmp_path / "EVAL.jsonl")
    with pytest.raises(MixError):
        assert_safe_input(tmp_path / "data" / "eval_frozen" / "rows.jsonl")


def test_exhausted_source_is_filled_from_other_sources(tmp_path):
    copy = _write_jsonl(tmp_path / "copy.jsonl", _rows("copy", 3))
    content = _write_jsonl(tmp_path / "content.jsonl", _rows("content", 5))

    rows, manifest = mix_rows(
        copy, content, copy_weight=2.0, content_weight=0.2, seed=5)

    assert len(rows) == 7
    assert manifest["final_counts"] == {"copy": 3, "content": 4}
    assert manifest["sources"][0]["requested"] == 6
    assert manifest["sources"][0]["selected"] == 3
    assert manifest["sources"][0]["exhausted"] is True
    assert manifest["sources"][1]["filled"] == 3
    assert manifest["filled_rows"] == 3
    assert manifest["unfilled_rows"] == 0


def test_zero_weight_source_is_not_used_for_fill(tmp_path):
    copy = _write_jsonl(tmp_path / "copy.jsonl", _rows("copy", 2))
    content = _write_jsonl(tmp_path / "content.jsonl", _rows("content", 5))

    rows, manifest = mix_rows(
        copy, content, copy_weight=2.0, content_weight=0.0)

    assert len(rows) == 2
    assert manifest["final_counts"] == {"copy": 2, "content": 0}
    assert manifest["sources"][1]["filled"] == 0
    assert manifest["unfilled_rows"] == 2


def test_fill_shortfall_is_reported_when_no_spare_rows_exist(tmp_path):
    copy = _write_jsonl(tmp_path / "copy.jsonl", _rows("copy", 2))
    content = _write_jsonl(tmp_path / "content.jsonl", _rows("content", 1))

    rows, manifest = mix_rows(
        copy, content, copy_weight=2.0, content_weight=1.0)

    assert len(rows) == 3
    assert manifest["final_counts"] == {"copy": 2, "content": 1}
    assert manifest["sources"][0]["exhausted"] is True
    assert manifest["filled_rows"] == 0
    assert manifest["unfilled_rows"] == 2


def test_manifest_is_content_free_and_records_the_hash(tmp_path):
    prompt = "PROMPT_SENTINEL_ALPHA"
    answer = "ANSWER_SENTINEL_BETA"
    family = "FAMILY_SENTINEL_GAMMA"
    key = "KEY_SENTINEL_DELTA"
    copy = _write_jsonl(
        tmp_path / "copy.jsonl",
        [{"user": prompt, "assistant": answer, "family": family,
          "keys": [key]}])
    manifest_path = tmp_path / "manifest.json"

    manifest = run_mix(copy, None, tmp_path / "mixed.jsonl",
                       seed=9, manifest_path=manifest_path)

    text = manifest_path.read_text(encoding="utf-8")
    for sentinel in (prompt, answer, family, key):
        assert sentinel not in text
    written = (tmp_path / "mixed.jsonl").read_bytes()
    assert manifest["output_sha256"] == hashlib.sha256(written).hexdigest()
    payload = json.loads(text)
    assert payload["output_sha256"] == manifest["output_sha256"]
    assert payload["final_rows"] == 1
    assert payload["weights"] == {"copy": 1.0, "content": 1.0}
    assert payload["seed"] == 9
    assert payload["eval"]["rows_used"] == 0
    assert payload["no_eval_rows_used"] is True
    assert "no eval rows" in payload["eval"]["note"].lower()


def test_empty_content_builds_a_copy_only_mix(tmp_path):
    copy = _write_jsonl(tmp_path / "copy.jsonl", _rows("copy", 4))
    out = tmp_path / "copy_only.jsonl"
    manifest_path = tmp_path / "copy_only_manifest.json"

    code = main([
        "--copy", str(copy),
        "--out", str(out),
        "--manifest", str(manifest_path),
        "--shuffle", "--seed", "17",
    ])

    assert code == 0
    assert len(_read_jsonl(out)) == 4
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["final_counts"] == {"copy": 4, "content": 0}
    assert payload["roles"]["content"]["sources"] == 0
    assert payload["filled_rows"] == 0


def test_max_rows_caps_the_mix_and_preserves_ratio(tmp_path):
    copy = _write_jsonl(tmp_path / "copy.jsonl", _rows("copy", 10))
    content = _write_jsonl(tmp_path / "content.jsonl", _rows("content", 10))

    rows, manifest = mix_rows(copy, content, max_rows=10, seed=2)

    assert len(rows) == 10
    assert manifest["final_counts"] == {"copy": 5, "content": 5}
    assert manifest["totals"]["selected"] == 10
    assert manifest["unfilled_rows"] == 0


def test_invalid_weights_and_negative_max_rows_raise(tmp_path):
    copy = _write_jsonl(tmp_path / "copy.jsonl", _rows("copy", 2))
    with pytest.raises(MixError):
        mix_rows(copy, copy_weight=-0.1)
    with pytest.raises(MixError):
        mix_rows(copy, copy_weight=float("nan"))
    with pytest.raises(MixError):
        mix_rows(copy, max_rows=-1)
    with pytest.raises(MixError):
        mix_rows([], None)
    assert main([
        "--copy", str(copy), "--out", str(tmp_path / "out.jsonl"),
        "--copy-weight", "-1",
    ]) == 2
