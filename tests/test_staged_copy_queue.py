import json
import re
from pathlib import Path

import pytest

from research import build_staged_copy_queue as queue


def _read_jsonl(path):
    return [json.loads(line) for line in
            Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _dataset(rows=24, eval_rows=8, **overrides):
    options = {
        "rows": rows,
        "eval_rows": eval_rows,
        "stages": queue.STAGE_ORDER,
        "key_style": "short",
        "seed": 23,
        "shuffle": False,
    }
    options.update(overrides)
    return queue.build_dataset(**options)


def _stage_rows(dataset, stage):
    entry = dataset["stages"][stage]
    return entry["train"] + entry["eval"]


def _all_rows(dataset):
    rows = []
    for stage in dataset["stage_order"]:
        rows.extend(_stage_rows(dataset, stage))
    return rows


def _row_signature(rows):
    return sorted(json.dumps(row, sort_keys=True, ensure_ascii=False)
                  for row in rows)


def test_stages_have_schema_pair_counts_and_formats():
    dataset = _dataset()
    assert dataset["stage_order"] == list(queue.STAGE_ORDER)
    assert dataset["summary"]["stage_order"] == list(queue.STAGE_ORDER)

    for stage in queue.STAGE_ORDER:
        entry = dataset["stages"][stage]
        spec = queue.STAGE_SPECS[stage]
        assert entry["name"] == stage
        assert entry["index"] == spec["index"]
        assert entry["summary"]["formats"] == list(spec["formats"])
        assert entry["summary"]["train_rows"] == 24
        assert entry["summary"]["eval_rows"] == 8
        train = entry["train"]
        eval_rows = entry["eval"]
        assert all(row["partition"] == "train" for row in train)
        assert all(row["partition"] == "eval" for row in eval_rows)
        for row in train + eval_rows:
            assert {
                "user", "assistant", "family", "stage", "stage_index",
                "format", "partition", "key_style", "keys", "values",
                "pairs", "key_styles", "value_pools", "verification",
            } <= set(row)
            assert row["stage"] == stage
            assert row["stage_index"] == spec["index"]
            assert row["family"] == f"staged_copy_{stage}"
            assert row["format"] in spec["formats"]
            assert spec["min_pairs"] <= len(row["keys"]) <= spec["max_pairs"]
            assert len(row["keys"]) == len(row["values"]) == len(row["pairs"])
            assert len(row["keys"]) == len(set(row["keys"]))
            assert [pair[0] for pair in row["pairs"]] == row["keys"]
            assert [pair[1] for pair in row["pairs"]] == row["values"]
            assert row["key_style"] == "short"
            assert row["key_styles"] == ["short"] * len(row["keys"])
            assert row["verification"]["verified"] is True
            assert row["verification"]["promotable"] is True
            assert row["verification"]["status"] == "verified"
            assert row["verification"]["method"] == queue.VERIFICATION_METHOD
            assert queue.validate_row(row)

    assert {len(row["keys"]) for row in _stage_rows(dataset, "echo_kv")} == {1}
    assert {len(row["keys"]) for row in _stage_rows(dataset, "echo_json")} == {1}
    assert {len(row["keys"]) for row in _stage_rows(dataset, "two_kv")} == {2}
    assert {len(row["keys"]) for row in _stage_rows(dataset, "two_json")} == {2}
    assert {row["format"] for row in _stage_rows(dataset, "multi")} == {
        "json", "kv"}
    assert {len(row["keys"]) for row in _stage_rows(dataset, "multi")} <= {
        2, 3, 4}


def test_each_stage_parses_and_matches_prompt_pairs():
    dataset = _dataset(rows=40, eval_rows=12)
    for stage in queue.STAGE_ORDER:
        for row in _stage_rows(dataset, stage):
            parsed = queue.parse_answer(row["format"], row["assistant"])
            assert set(parsed) == set(row["keys"])
            assert [str(parsed[key]) for key in row["keys"]] == [
                str(value) for value in row["values"]]
            assert queue.validate_answer(
                row["format"], row["assistant"], row["pairs"])
            prompt_pairs = queue.extract_prompt_pairs(row["user"])
            assert prompt_pairs == [
                [key, str(value)] for key, value in row["pairs"]]


def test_echo_stage_answer_equals_exact_prompt_pair():
    dataset = _dataset(rows=30, eval_rows=10, stages=("echo_kv",))
    rows = _stage_rows(dataset, "echo_kv")
    assert rows
    for row in rows:
        prompt_pairs = queue.extract_prompt_pairs(row["user"])
        assert len(prompt_pairs) == 1
        key, value = prompt_pairs[0]
        assert row["keys"] == [key]
        assert str(row["values"][0]) == value
        assert row["assistant"] == f"{key}={value}"
        assert "\n" not in row["assistant"]


def test_json_wrapper_shape():
    dataset = _dataset(rows=30, eval_rows=10)
    for stage in ("echo_json", "two_json"):
        for row in _stage_rows(dataset, stage):
            answer = row["assistant"]
            assert answer.startswith("{") and answer.endswith("}")
            parsed = json.loads(answer)
            assert isinstance(parsed, dict)
            assert list(parsed) == row["keys"]
            assert len(parsed) == len(row["keys"])
            assert answer == queue.render_answer(
                "json", row["pairs"])
    single = _stage_rows(dataset, "echo_json")
    assert all(len(json.loads(row["assistant"])) == 1 for row in single)
    multi_json = [row for row in _stage_rows(dataset, "multi")
                  if row["format"] == "json"]
    assert multi_json
    for row in multi_json:
        assert row["assistant"].startswith("{")
        assert json.loads(row["assistant"]) == dict(row["pairs"])


def test_train_and_eval_share_no_keys_or_values():
    dataset = _dataset(rows=120, eval_rows=40)
    all_train_keys = set()
    all_eval_keys = set()
    all_train_values = set()
    all_eval_values = set()
    for stage in queue.STAGE_ORDER:
        entry = dataset["stages"][stage]
        train_keys = {key for row in entry["train"] for key in row["keys"]}
        eval_keys = {key for row in entry["eval"] for key in row["keys"]}
        train_values = {str(value) for row in entry["train"]
                        for value in row["values"]}
        eval_values = {str(value) for row in entry["eval"]
                       for value in row["values"]}
        assert train_keys & eval_keys == set()
        assert train_values & eval_values == set()
        assert train_keys & queue.RESERVED_KEYS == set()
        assert entry["summary"]["key_overlap"] == 0
        assert entry["summary"]["key_overlap_examples"] == []
        assert entry["summary"]["value_overlap"] == 0
        assert entry["summary"]["value_overlap_examples"] == []
        assert entry["summary"]["unique_keys"] == {
            "train": len(train_keys), "eval": len(eval_keys),
            "total": len(train_keys | eval_keys),
        }
        assert entry["summary"]["unique_values"] == {
            "train": len(train_values), "eval": len(eval_values),
            "total": len(train_values | eval_values),
        }
        all_train_keys |= train_keys
        all_eval_keys |= eval_keys
        all_train_values |= train_values
        all_eval_values |= eval_values

    assert all_train_keys & all_eval_keys == set()
    assert all_train_values & all_eval_values == set()
    assert len(all_train_keys | all_eval_keys) == sum(
        len(row["keys"]) for row in _all_rows(dataset))
    assert len(all_train_values | all_eval_values) == sum(
        len(row["values"]) for row in _all_rows(dataset))
    summary = dataset["summary"]
    assert summary["key_overlap"] == 0
    assert summary["key_overlap_examples"] == []
    assert summary["value_overlap"] == 0
    assert summary["value_overlap_examples"] == []
    assert summary["rows"] == {"train": 600, "eval": 200, "total": 800}


def test_seed_is_deterministic_and_shuffle_preserves_rows():
    first = _dataset(rows=30, eval_rows=10, seed=23)
    second = _dataset(rows=30, eval_rows=10, seed=23)
    other = _dataset(rows=30, eval_rows=10, seed=24)

    assert first["stages"] == second["stages"]
    assert first["summary"] == second["summary"]
    assert _row_signature(_all_rows(first)) != _row_signature(
        _all_rows(other))

    plain = _dataset(rows=50, eval_rows=10, shuffle=False)
    shuffled_a = _dataset(rows=50, eval_rows=10, shuffle=True)
    shuffled_b = _dataset(rows=50, eval_rows=10, shuffle=True)
    for stage in queue.STAGE_ORDER:
        assert shuffled_a["stages"][stage]["train"] == (
            shuffled_b["stages"][stage]["train"])
        assert shuffled_a["stages"][stage]["eval"] == (
            shuffled_b["stages"][stage]["eval"])
        assert _row_signature(shuffled_a["stages"][stage]["train"]) == (
            _row_signature(plain["stages"][stage]["train"]))
        assert _row_signature(shuffled_a["stages"][stage]["eval"]) == (
            _row_signature(plain["stages"][stage]["eval"]))
    assert [row["user"] for row in shuffled_a["stages"]["multi"]["train"]] != [
        row["user"] for row in plain["stages"]["multi"]["train"]]


@pytest.mark.parametrize("key_style", queue.KEY_STYLES)
def test_key_style_constraints(key_style):
    dataset = _dataset(rows=40, eval_rows=10, key_style=key_style)
    keys = [key for row in _all_rows(dataset) for key in row["keys"]]
    assert keys
    for key in keys:
        assert key not in queue.RESERVED_KEYS
        assert queue.KEY_PATTERNS[key_style].fullmatch(key)
        if key_style == "short":
            assert 3 <= len(key) <= 6
            assert re.fullmatch(r"[a-z][a-z0-9]{2,5}", key)
        elif key_style == "cv":
            assert len(key) in {6, 8}
            assert re.fullmatch(
                r"(?:[bcdfghjklmnprstvwz][aeiou]){3,4}", key)
        else:
            assert 4 <= len(key) <= 8
            assert key.isalpha() and key.islower()
    for stage in queue.STAGE_ORDER:
        summary = dataset["stages"][stage]["summary"]
        assert summary["key_style"] == key_style
        assert summary["key_styles"][key_style] > 0


def test_mixed_stage_value_pools():
    dataset = _dataset(rows=60, eval_rows=20, stages=("multi",))
    rows = _stage_rows(dataset, "multi")
    assert {pool for row in rows for pool in row["value_pools"]} == set(
        queue.VALUE_POOLS)
    for row in rows:
        assert "multiword" in row["value_pools"]
        if len(row["keys"]) >= 3:
            assert "number" in row["value_pools"]
            assert set(row["value_pools"]) & {"token", "word"}


def test_early_stages_use_single_word_values_only():
    dataset = _dataset(rows=40, eval_rows=10,
                       stages=("echo_kv", "echo_json", "two_kv", "two_json"))
    for stage in dataset["stage_order"]:
        for row in _stage_rows(dataset, stage):
            assert set(row["value_pools"]) <= {"token", "word"}
            for value in row["values"]:
                assert isinstance(value, str)
                assert " " not in value


def test_tampered_rows_fail_row_validation():
    row = _dataset(rows=1, eval_rows=0, stages=("echo_json",))[
        "stages"]["echo_json"]["train"][0]
    assert queue.validate_row(row)
    broken = json.loads(json.dumps(row))
    broken["assistant"] = queue.render_answer(
        "json", [[broken["keys"][0], "definitely-not-the-value"]])
    assert not queue.validate_row(broken)
    broken = json.loads(json.dumps(row))
    broken["keys"] = broken["keys"] + ["zzzz"]
    assert not queue.validate_row(broken)
    broken = json.loads(json.dumps(row))
    broken["assistant"] = '{"wrong":"shape"}'
    assert not queue.validate_row(broken)


def test_invalid_oracle_answers_are_regenerated(monkeypatch):
    original = queue.render_answer
    calls = {"count": 0}

    def flaky(fmt, pairs):
        calls["count"] += 1
        if calls["count"] == 1:
            return "this is not the requested format"
        return original(fmt, pairs)

    monkeypatch.setattr(queue, "render_answer", flaky)
    dataset = queue.build_dataset(
        rows=3, eval_rows=1, stages=("echo_json",), seed=11)
    assert calls["count"] >= 2
    assert dataset["summary"]["rejected_attempts"] == 1
    assert dataset["stages"]["echo_json"]["summary"][
        "rejected_attempts"] == 1
    assert all(queue.validate_row(row)
               for row in _all_rows(dataset))


def test_unknown_stage_or_key_style_is_rejected():
    with pytest.raises(ValueError):
        queue.build_dataset(rows=1, eval_rows=0, stages=("bogus",))
    with pytest.raises(ValueError):
        queue.build_dataset(rows=1, eval_rows=0, key_style="bogus")
    with pytest.raises(ValueError):
        queue.build_dataset(rows=-1, eval_rows=0)
    assert queue.normalize_stages("multi,echo_kv") == (
        "echo_kv", "multi")
    assert queue.normalize_stages("all") == queue.STAGE_ORDER


def test_cli_writes_per_stage_directories_and_manifests(tmp_path, capsys):
    out_dir = tmp_path / "staged"
    args = [
        "--out-dir", str(out_dir),
        "--stages", "echo_kv,two_json,multi",
        "--rows", "12", "--eval-rows", "4",
        "--seed", "7", "--key-style", "short", "--shuffle",
    ]
    assert queue.main(args) == 0
    root = json.loads(capsys.readouterr().out)
    assert root["stage_order"] == ["echo_kv", "two_json", "multi"]
    assert root["train_rows"] == 36
    assert root["eval_rows"] == 12
    assert root["key_overlap"] == 0
    assert root["value_overlap"] == 0
    assert (out_dir / "manifest.json").exists()

    stage_dirs = sorted(path.name for path in out_dir.iterdir()
                        if path.is_dir())
    assert stage_dirs == [
        "stage_echo_kv", "stage_multi", "stage_two_json"]

    for stage in root["stage_order"]:
        stage_dir = out_dir / f"stage_{stage}"
        manifest = json.loads((stage_dir / "manifest.json").read_text(
            encoding="utf-8"))
        assert manifest["stage"] == stage
        assert manifest["stage_order"] == root["stage_order"]
        assert manifest["train_rows"] == 12
        assert manifest["eval_rows"] == 4
        assert manifest["key_overlap"] == 0
        assert manifest["value_overlap"] == 0
        assert Path(manifest["paths"]["train"]) == stage_dir / "train.jsonl"
        assert Path(manifest["paths"]["eval"]) == stage_dir / "eval.jsonl"
        train = _read_jsonl(stage_dir / "train.jsonl")
        eval_rows = _read_jsonl(stage_dir / "eval.jsonl")
        assert len(train) == 12 and len(eval_rows) == 4
        assert {row["partition"] for row in train} == {"train"}
        assert {row["partition"] for row in eval_rows} == {"eval"}
        assert all(queue.validate_row(row) for row in train + eval_rows)

    assert not (out_dir / "stage_echo_json").exists()
    assert not (out_dir / "stage_two_kv").exists()

    second_out = tmp_path / "staged_again"
    second_args = list(args)
    second_args[second_args.index(str(out_dir))] = str(second_out)
    capsys.readouterr()
    assert queue.main(second_args) == 0
    capsys.readouterr()
    for stage in root["stage_order"]:
        for filename in ("train.jsonl", "eval.jsonl"):
            assert (out_dir / f"stage_{stage}" / filename).read_bytes() == (
                second_out / f"stage_{stage}" / filename).read_bytes()


def test_manifest_reports_stage_order_and_global_counts(tmp_path):
    dataset = _dataset(rows=6, eval_rows=2)
    summary = queue.write_dataset(dataset, tmp_path / "queue")
    manifest = json.loads((tmp_path / "queue" / "manifest.json").read_text(
        encoding="utf-8"))
    assert manifest["stage_order"] == list(queue.STAGE_ORDER)
    assert manifest["rows"] == {"train": 30, "eval": 10, "total": 40}
    assert manifest["key_overlap"] == 0
    assert manifest["value_overlap"] == 0
    assert len(manifest["stages"]) == len(queue.STAGE_ORDER)
    assert [entry["stage"] for entry in manifest["stages"]] == list(
        queue.STAGE_ORDER)
    assert manifest["rows"] == summary["rows"]
    assert manifest["formats"] == ["json", "kv"]
    assert sum(manifest["format_counts"]["total"].values()) == 40


def test_generator_has_no_model_dependency():
    source = Path(queue.__file__).read_text(encoding="utf-8")
    assert "import torch" not in source
    assert "from transformers" not in source
    assert "import transformers" not in source
