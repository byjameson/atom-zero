import json
import re
from pathlib import Path

import pytest

from research import build_multi_short_queue as queue


def _read_jsonl(path):
    return [json.loads(line) for line in
            Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _dataset(rows=24, eval_rows=8, **overrides):
    options = {
        "rows": rows,
        "eval_rows": eval_rows,
        "difficulty": queue.STAGE_ORDER,
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


def test_stage_shapes_pair_counts_and_value_plans():
    dataset = _dataset()
    assert dataset["stage_order"] == list(queue.STAGE_ORDER)
    assert dataset["stage_order"] == [
        "2short", "2mixed", "3short", "4short"]
    assert dataset["summary"]["stage_order"] == list(queue.STAGE_ORDER)
    assert dataset["summary"]["difficulty_order"] == list(queue.STAGE_ORDER)

    for stage in queue.STAGE_ORDER:
        entry = dataset["stages"][stage]
        spec = queue.STAGE_SPECS[stage]
        assert entry["name"] == stage
        assert entry["index"] == spec["index"]
        assert entry["summary"]["formats"] == list(spec["formats"])
        assert entry["summary"]["pair_count"] == spec["pairs"]
        assert entry["summary"]["train_rows"] == 24
        assert entry["summary"]["eval_rows"] == 8
        assert entry["summary"]["value_plan"] == spec["value_plan"]
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
            assert row["family"] == f"multi_short_{stage}"
            assert row["format"] in spec["formats"]
            assert len(row["keys"]) == spec["pairs"]
            assert len(row["keys"]) == len(row["values"]) == len(row["pairs"])
            assert len(row["keys"]) == len(set(row["keys"]))
            assert len(row["values"]) == len(set(map(str, row["values"])))
            assert [pair[0] for pair in row["pairs"]] == row["keys"]
            assert [pair[1] for pair in row["pairs"]] == row["values"]
            assert row["key_style"] == "short"
            assert row["key_styles"] == ["short"] * len(row["keys"])
            assert len(row["value_pools"]) == len(row["values"])
            for key in row["keys"]:
                assert 3 <= len(key) <= 5
                assert key not in queue.RESERVED_KEYS
            if spec["value_plan"] == "single":
                assert set(row["value_pools"]) <= {"token", "word"}
                assert all(" " not in str(value) for value in row["values"])
            else:
                assert row["value_pools"].count("multiword") == 1
                assert set(row["value_pools"]) <= {"token", "word",
                                                   "multiword"}
                multiword = row["values"][
                    row["value_pools"].index("multiword")]
                assert " " in str(multiword)
            assert row["verification"]["verified"] is True
            assert row["verification"]["promotable"] is True
            assert row["verification"]["status"] == "verified"
            assert row["verification"]["method"] == queue.VERIFICATION_METHOD
            assert queue.validate_row(row)

    assert {len(row["keys"]) for row in _stage_rows(dataset, "2short")} == {2}
    assert {len(row["keys"]) for row in _stage_rows(dataset, "2mixed")} == {2}
    assert {len(row["keys"]) for row in _stage_rows(dataset, "3short")} == {3}
    assert {len(row["keys"]) for row in _stage_rows(dataset, "4short")} == {4}
    for stage in queue.STAGE_ORDER:
        assert {row["format"] for row in _stage_rows(dataset, stage)} == {
            "json", "kv"}


def test_answers_parse_and_round_trip_prompt_pairs():
    dataset = _dataset(rows=40, eval_rows=12)
    for stage in queue.STAGE_ORDER:
        for row in _stage_rows(dataset, stage):
            parsed = queue.parse_answer(row["format"], row["assistant"])
            assert set(parsed) == set(row["keys"])
            assert [str(parsed[key]) for key in row["keys"]] == [
                str(value) for value in row["values"]]
            assert queue.validate_answer(
                row["format"], row["assistant"], row["pairs"])
            assert row["assistant"] == queue.render_answer(
                row["format"], row["pairs"])
            prompt_pairs = queue.extract_prompt_pairs(row["user"])
            assert prompt_pairs == [
                [key, str(value)] for key, value in row["pairs"]]
            for key, value in row["pairs"]:
                assert f"{key}={value}" in row["user"]
            if row["format"] == "json":
                assert row["assistant"].startswith("{")
                assert json.loads(row["assistant"]) == dict(row["pairs"])
                assert queue.JSON_PROMPT_PREFIX in row["user"]
            else:
                assert "\n" not in row["assistant"]
                assert ";" in row["assistant"]
                assert queue.KV_PROMPT_PREFIX in row["user"]


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
    assert summary["rows"] == {"train": 480, "eval": 160, "total": 640}


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
    assert [row["user"] for row in shuffled_a["stages"]["4short"]["train"]] != [
        row["user"] for row in plain["stages"]["4short"]["train"]]


def test_difficulty_filter_selects_stages_in_ladder_order():
    assert queue.normalize_difficulties("all") == queue.STAGE_ORDER
    assert queue.normalize_difficulties("2short") == ("2short",)
    assert queue.normalize_difficulties("3short,2short") == (
        "2short", "3short")
    assert queue.normalize_difficulties(["4short", "2mixed", "4short"]) == (
        "2mixed", "4short")
    assert queue.normalize_difficulties("ALL") == queue.STAGE_ORDER
    with pytest.raises(ValueError):
        queue.normalize_difficulties("bogus")
    with pytest.raises(ValueError):
        queue.normalize_difficulties("")
    with pytest.raises(ValueError):
        queue.build_dataset(rows=1, eval_rows=0, difficulty=("bogus",))

    dataset = _dataset(rows=6, eval_rows=2, difficulty="2mixed,4short")
    assert dataset["stage_order"] == ["2mixed", "4short"]
    assert list(dataset["stages"]) == ["2mixed", "4short"]
    assert dataset["summary"]["stage_order"] == ["2mixed", "4short"]
    assert dataset["summary"]["rows"] == {"train": 12, "eval": 4, "total": 16}
    assert dataset["summary"]["key_overlap"] == 0
    assert dataset["summary"]["value_overlap"] == 0
    assert "2short" not in dataset["stages"]
    assert "3short" not in dataset["stages"]
    for stage in dataset["stage_order"]:
        assert dataset["stages"][stage]["summary"]["train_rows"] == 6
        assert dataset["stages"][stage]["summary"]["eval_rows"] == 2


@pytest.mark.parametrize("key_style", queue.KEY_STYLES)
def test_key_style_constraints(key_style):
    dataset = _dataset(rows=40, eval_rows=10, key_style=key_style)
    keys = [key for row in _all_rows(dataset) for key in row["keys"]]
    assert keys
    for key in keys:
        assert key not in queue.RESERVED_KEYS
        assert queue.KEY_PATTERNS[key_style].fullmatch(key)
        assert 3 <= len(key) <= 5
        if key_style == "short":
            assert re.fullmatch(r"[a-z][a-z0-9]{2,4}", key)
        elif key_style == "cv":
            assert len(key) == 5
            assert re.fullmatch(
                r"(?:[bcdfghjklmnprstvwz][aeiou]){2}"
                r"[bcdfghjklmnprstvwz]", key)
        else:
            assert key.isalpha() and key.islower()
    for stage in queue.STAGE_ORDER:
        summary = dataset["stages"][stage]["summary"]
        assert summary["key_style"] == key_style
        assert summary["key_styles"][key_style] > 0
        assert summary["key_length"] == {"min": 3, "max": 5}


def test_mixed_stage_has_one_multiword_value():
    dataset = _dataset(rows=30, eval_rows=10, difficulty=("2mixed",))
    for row in _stage_rows(dataset, "2mixed"):
        assert row["value_pools"].count("multiword") == 1
        assert len(row["value_pools"]) == 2
        other = [pool for pool in row["value_pools"] if pool != "multiword"]
        assert other in (["token"], ["word"])
        for pool, value in zip(row["value_pools"], row["values"]):
            if pool == "multiword":
                assert " " in value
            else:
                assert " " not in value
    for stage in ("2short", "3short", "4short"):
        dataset = _dataset(rows=10, eval_rows=2, difficulty=(stage,))
        assert all("multiword" not in row["value_pools"]
                   for row in _stage_rows(dataset, stage))


def test_tampered_rows_fail_row_validation():
    row = _dataset(rows=1, eval_rows=0, difficulty=("2short",))[
        "stages"]["2short"]["train"][0]
    assert queue.validate_row(row)
    broken = json.loads(json.dumps(row))
    broken["assistant"] = queue.render_answer(
        "json" if row["format"] == "kv" else "kv", row["pairs"])
    assert not queue.validate_row(broken)
    broken = json.loads(json.dumps(row))
    broken["keys"] = broken["keys"] + ["zzzz"]
    assert not queue.validate_row(broken)
    broken = json.loads(json.dumps(row))
    broken["user"] = broken["user"].replace("Copy these pairs exactly", "Do")
    assert not queue.validate_row(broken)
    broken = json.loads(json.dumps(row))
    broken["verification"]["answer_sha256"] = "0" * 64
    assert not queue.validate_row(broken)

    mixed = _dataset(rows=1, eval_rows=0, difficulty=("2mixed",))[
        "stages"]["2mixed"]["train"][0]
    assert queue.validate_row(mixed)
    broken = json.loads(json.dumps(mixed))
    index = broken["value_pools"].index("multiword")
    broken["values"][index] = "single"
    broken["pairs"] = [[key, value] for key, value in zip(
        broken["keys"], broken["values"])]
    broken["assistant"] = queue.render_answer(
        broken["format"], broken["pairs"])
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
        rows=3, eval_rows=1, difficulty=("2short",), seed=11)
    assert calls["count"] >= 2
    assert dataset["summary"]["rejected_attempts"] == 1
    assert dataset["stages"]["2short"]["summary"][
        "rejected_attempts"] == 1
    assert all(queue.validate_row(row) for row in _all_rows(dataset))


def test_manifests_report_ladder_and_global_counts(tmp_path):
    dataset = _dataset(rows=6, eval_rows=2)
    summary = queue.write_dataset(dataset, tmp_path / "queue")
    manifest = json.loads((tmp_path / "queue" / "manifest.json").read_text(
        encoding="utf-8"))
    assert manifest["stage_order"] == list(queue.STAGE_ORDER)
    assert manifest["difficulty_order"] == list(queue.STAGE_ORDER)
    assert manifest["rows"] == {"train": 24, "eval": 8, "total": 32}
    assert manifest["key_overlap"] == 0
    assert manifest["value_overlap"] == 0
    assert manifest["verified_rows"] == 32
    assert manifest["rejected_attempts"] == 0
    assert len(manifest["stages"]) == len(queue.STAGE_ORDER)
    assert [entry["stage"] for entry in manifest["stages"]] == list(
        queue.STAGE_ORDER)
    assert [entry["pair_count"] for entry in manifest["stages"]] == [
        2, 2, 3, 4]
    assert manifest["rows"] == summary["rows"]
    assert manifest["formats"] == ["json", "kv"]
    assert sum(manifest["format_counts"]["total"].values()) == 32

    for stage in queue.STAGE_ORDER:
        stage_dir = tmp_path / "queue" / f"stage_{stage}"
        stage_manifest = json.loads((stage_dir / "manifest.json").read_text(
            encoding="utf-8"))
        assert stage_manifest["stage"] == stage
        assert stage_manifest["difficulty"] == stage
        assert stage_manifest["stage_index"] == queue.STAGE_SPECS[
            stage]["index"]
        assert stage_manifest["pair_count"] == queue.STAGE_SPECS[
            stage]["pairs"]
        assert stage_manifest["train_rows"] == 6
        assert stage_manifest["eval_rows"] == 2
        assert stage_manifest["key_overlap"] == 0
        assert stage_manifest["value_overlap"] == 0
        assert Path(stage_manifest["paths"]["train"]) == (
            stage_dir / "train.jsonl")
        assert Path(stage_manifest["paths"]["eval"]) == (
            stage_dir / "eval.jsonl")
        train = _read_jsonl(stage_dir / "train.jsonl")
        eval_rows = _read_jsonl(stage_dir / "eval.jsonl")
        assert len(train) == 6 and len(eval_rows) == 2
        assert {row["partition"] for row in train} == {"train"}
        assert {row["partition"] for row in eval_rows} == {"eval"}
        assert all(queue.validate_row(row) for row in train + eval_rows)


def test_cli_writes_stage_dirs_and_is_byte_identical(tmp_path, capsys):
    out_dir = tmp_path / "multi_short"
    args = [
        "--out-dir", str(out_dir),
        "--difficulty", "2short,3short",
        "--rows", "12", "--eval-rows", "4",
        "--seed", "7", "--key-style", "short", "--shuffle",
    ]
    assert queue.main(args) == 0
    root = json.loads(capsys.readouterr().out)
    assert root["stage_order"] == ["2short", "3short"]
    assert root["train_rows"] == 24
    assert root["eval_rows"] == 8
    assert root["key_overlap"] == 0
    assert root["value_overlap"] == 0
    assert (out_dir / "manifest.json").exists()

    stage_dirs = sorted(path.name for path in out_dir.iterdir()
                        if path.is_dir())
    assert stage_dirs == ["stage_2short", "stage_3short"]
    for stage in root["stage_order"]:
        stage_dir = out_dir / f"stage_{stage}"
        manifest = json.loads((stage_dir / "manifest.json").read_text(
            encoding="utf-8"))
        assert manifest["stage"] == stage
        assert manifest["train_rows"] == 12
        assert manifest["eval_rows"] == 4
        assert manifest["pair_count"] == queue.STAGE_SPECS[stage]["pairs"]
        train = _read_jsonl(stage_dir / "train.jsonl")
        eval_rows = _read_jsonl(stage_dir / "eval.jsonl")
        assert len(train) == 12 and len(eval_rows) == 4
        assert all(queue.validate_row(row) for row in train + eval_rows)

    second_out = tmp_path / "multi_short_again"
    second_args = list(args)
    second_args[second_args.index(str(out_dir))] = str(second_out)
    capsys.readouterr()
    assert queue.main(second_args) == 0
    capsys.readouterr()
    for stage in root["stage_order"]:
        for filename in ("train.jsonl", "eval.jsonl"):
            assert (out_dir / f"stage_{stage}" / filename).read_bytes() == (
                second_out / f"stage_{stage}" / filename).read_bytes()
        first_manifest = json.loads(
            (out_dir / f"stage_{stage}" / "manifest.json").read_text(
                encoding="utf-8"))
        second_manifest = json.loads(
            (second_out / f"stage_{stage}" / "manifest.json").read_text(
                encoding="utf-8"))
        first_manifest.pop("paths")
        second_manifest.pop("paths")
        assert first_manifest == second_manifest


def test_cli_single_difficulty_and_key_style(tmp_path, capsys):
    out_dir = tmp_path / "single"
    assert queue.main([
        "--out-dir", str(out_dir), "--difficulty", "4short",
        "--rows", "5", "--eval-rows", "2", "--seed", "3",
        "--key-style", "cv",
    ]) == 0
    root = json.loads(capsys.readouterr().out)
    assert root["stage_order"] == ["4short"]
    assert root["key_style"] == "cv"
    assert root["train_rows"] == 5
    assert root["eval_rows"] == 2
    rows = (_read_jsonl(out_dir / "stage_4short" / "train.jsonl")
            + _read_jsonl(out_dir / "stage_4short" / "eval.jsonl"))
    assert all(len(row["keys"]) == 4 for row in rows)
    assert all(len(key) == 5 for row in rows for key in row["keys"])


def test_generator_has_no_model_dependency():
    source = Path(queue.__file__).read_text(encoding="utf-8")
    assert "import torch" not in source
    assert "from transformers" not in source
    assert "import transformers" not in source
