import json
import re
from pathlib import Path

import pytest

from research import build_diverse_teacher_queue as queue


REQUIRED_KEY_PAIRS = {
    ("name", "score"), ("product", "price"), ("label", "points"),
    ("code", "rank"), ("team", "wins"), ("author", "year"),
    ("song", "length"), ("model", "params"), ("river", "km"),
    ("planet", "mass"), ("recipe", "minutes"), ("city", "population"),
}


def _read_jsonl(path):
    return [json.loads(line) for line in
            Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _entities(rows):
    return {(row["entity_pool"], row["entity"]) for row in rows
            if row.get("entity_pool")}


def _row_key(row):
    return (row["family"], row["user"], row["assistant"])


def _cli_summary(capsys):
    output = capsys.readouterr().out
    marker = '{\n  "requested"'
    return json.loads(output[output.rindex(marker):])


def test_family_specs_are_reusable_without_a_model():
    city = queue.FAMILIES["city_yaml"]
    assert (city.entity_name, city.key1, city.key2) == (
        "city", "city", "population")
    assert city.format == "yaml"
    assert callable(city.prompt_builder)
    assert queue.FAMILIES["arith_plain"].format == "plain"
    assert queue.FAMILIES["arith_json"].format == "json"
    assert "arith_plain" in queue.TRAIN_FAMILIES
    assert "arith_json" in queue.TRAIN_FAMILIES
    assert "city_yaml" in queue.CROSS_FAMILIES


def test_json_validator_accepts_and_rejects_attribute_answers():
    assert queue.validate_answer(
        "person_json", '{"name": "Ada", "score": 7}', "Ada", 7)
    assert queue.validate_answer(
        "person_json", '```json\n{"name": "Ada", "score": 7}\n```', "Ada", 7)
    assert not queue.validate_answer(
        "person_json", '{"name": "Bora", "score": 7}', "Ada", 7)
    assert not queue.validate_answer(
        "person_json", '{"name": "Ada", "score": 8}', "Ada", 7)
    assert not queue.validate_answer(
        "person_json", '["Ada", 7]', "Ada", 7)
    assert not queue.validate_answer(
        "person_json", 'name=Ada;score=7', "Ada", 7)
    assert not queue.validate_answer(
        "person_json", 'here is the JSON: {"name": "Ada", "score": 7}',
        "Ada", 7)


def test_json_validator_accepts_and_rejects_arithmetic_answers():
    assert queue.validate_answer("arith_json", '{"sum": 42}', "12", 30)
    assert queue.validate_answer(
        "arith_json", '```json\n{"sum": 42}\n```', "12", 30)
    assert not queue.validate_answer("arith_json", '{"sum": 41}', "12", 30)
    assert not queue.validate_answer("arith_json", '{"sum": 42}', "12", 31)
    assert not queue.validate_answer(
        "arith_json", "The sum of 12 and 30 is 42.", "12", 30)
    assert not queue.validate_answer("arith_json", '{"total": 42}', "12", 30)


def test_kv_validator_accepts_and_rejects_answers():
    assert queue.validate_answer("person_kv", "name=Ada;score=7", "Ada", 7)
    assert queue.validate_answer(
        "person_kv", "name = Ada ; score = 7", "Ada", 7)
    assert not queue.validate_answer(
        "person_kv", "name=Bora;score=7", "Ada", 7)
    assert not queue.validate_answer("person_kv", "name=Ada;score=8", "Ada", 7)
    assert not queue.validate_answer(
        "person_kv", '{"name": "Ada", "score": 7}', "Ada", 7)
    assert not queue.validate_answer(
        "person_kv", "name=Ada", "Ada", 7)


def test_yaml_validator_accepts_and_rejects_answers():
    answer = "city: Ankara\npopulation: 500000"
    assert queue.validate_answer("city_yaml", answer, "Ankara", 500000)
    assert queue.validate_answer(
        "city_yaml", "population: 500000\ncity: Ankara", "Ankara", 500000)
    assert queue.validate_answer(
        "city_yaml", 'city: "Ankara"\npopulation: "500000"',
        "Ankara", 500000)
    assert not queue.validate_answer(
        "city_yaml", "city: Bursa\npopulation: 500000", "Ankara", 500000)
    assert not queue.validate_answer(
        "city_yaml", "city: Ankara\npopulation: 500001", "Ankara", 500000)
    assert not queue.validate_answer(
        "city_yaml", '{"city": "Ankara", "population": 500000}',
        "Ankara", 500000)
    assert not queue.validate_answer(
        "city_yaml", "city: Ankara", "Ankara", 500000)
    assert not queue.validate_answer(
        "city_yaml", "city: Ankara\npopulation: 500000\nsource: census",
        "Ankara", 500000)
    assert not queue.validate_answer(
        "city_yaml", "city: {Ankara}\npopulation: 500000",
        "Ankara", 500000)


def test_plain_validator_accepts_and_rejects_answers():
    assert queue.validate_answer(
        "arith_plain", "The sum of 12 and 30 is 42.", "12", 30)
    assert queue.validate_answer(
        "arith_plain", "The sum of 12 and 30 is 42", "12", 30)
    assert not queue.validate_answer(
        "arith_plain", "The sum of 12 and 30 is 41.", "12", 30)
    assert not queue.validate_answer(
        "arith_plain", "The sum of 12 and 31 is 42.", "12", 30)
    assert not queue.validate_answer(
        "arith_plain", "12 + 30 = 42", "12", 30)
    assert not queue.validate_answer(
        "arith_plain", "The sum of 12 and 30 is 42. Thanks!", "12", 30)


def test_clean_answer_strips_a_code_fence():
    assert queue.clean_answer("```yaml\ncity: Ankara\n```") == "city: Ankara"


def test_stub_teacher_produces_verified_and_rejected_counts():
    calls = []

    def stub_teacher(prompt):
        calls.append(prompt)
        match = re.search(r"name=(\w+) and score=(\d+)", prompt)
        if match:
            return json.dumps(
                {"name": match.group(1), "score": int(match.group(2))})
        return "not the requested format"

    counts = {"person_json": 3, "city_yaml": 2, "arith_json": 1}
    rows = queue.build_rows(
        ["person_json", "city_yaml", "arith_json"], stub_teacher,
        counts=counts, seed=5)
    assert len(calls) == 6
    assert len(rows) == 6
    by_family = {}
    for row in rows:
        by_family.setdefault(row["family"], []).append(row)
    assert all(row["verification"]["verified"]
               for row in by_family["person_json"])
    assert all(row["verification"]["promotable"]
               for row in by_family["person_json"])
    assert not any(row["verification"]["verified"]
                   for row in by_family["city_yaml"])
    assert not any(row["verification"]["verified"]
                   for row in by_family["arith_json"])
    verified = sum(1 for row in rows if row["verification"]["verified"])
    assert (verified, len(rows) - verified) == (3, 3)


def test_oracle_teacher_verifies_every_rollout():
    rows = queue.build_rows(
        ["city_yaml", "arith_plain", "arith_json"], None,
        count=2, seed=1)
    assert len(rows) == 6
    assert all(row["verification"]["verified"] for row in rows)
    assert all(row["verification"]["status"] == "verified" for row in rows)


def test_build_rows_is_deterministic_for_a_seed():
    families = ["person_json", "city_yaml", "arith_plain", "arith_json"]
    counts = {family: 4 for family in families}
    first = queue.build_rows(families, None, counts=counts, seed=23)
    second = queue.build_rows(families, None, counts=counts, seed=23)
    assert first == second
    dumped = "".join(json.dumps(row, sort_keys=True) for row in first)
    assert dumped == "".join(
        json.dumps(row, sort_keys=True) for row in second)


def test_cross_split_never_puts_a_held_out_family_in_train():
    rows = queue.build_rows(
        ["person_json", "arith_plain", "city_yaml"], None,
        counts={"person_json": 10, "arith_plain": 5, "city_yaml": 4},
        seed=3)
    files = queue.split_rows(rows)
    train_families = {row["family"] for row in files["train"]}
    assert train_families <= set(queue.TRAIN_FAMILIES)
    assert "city_yaml" not in train_families
    assert {row["family"] for row in files["cross"]} == {"city_yaml"}
    assert len(files["train"]) == 12
    assert len(files["indist"]) == 3
    assert len(files["cross"]) == 4
    assert len(files["all"]) == 19


def test_cross_split_rejects_overlapping_train_and_cross_families():
    rows = queue.build_rows(["person_json", "city_yaml"], None, count=1)
    with pytest.raises(ValueError, match="overlap"):
        queue.split_rows(
            rows, train_families=("person_json", "city_yaml"),
            cross_families=("city_yaml",))


def test_resolve_counts_handles_new_family_flags():
    counts = queue.resolve_counts(
        ("person_json", "city_yaml", "arith_plain", "arith_json"),
        per_family=2, city_per_family=3, city_yaml_count=4,
        arith_count=5,
        overrides={"arith_json": 6})
    assert counts == {
        "person_json": 2,
        "city_yaml": 4,
        "arith_plain": 5,
        "arith_json": 6,
    }


def test_key_pair_pool_and_expansion():
    assert len(queue.KEY_PAIRS) >= 12
    assert REQUIRED_KEY_PAIRS <= set(queue.KEY_PAIRS)
    assert queue.parse_key_pairs("all") == queue.KEY_PAIRS
    assert queue.parse_key_pairs("label:points,code") == (
        ("label", "points"), ("code", "rank"))

    plan = queue.resolve_family_plan("label:points,code")
    assert "label_json" in plan["train_families"]
    assert "label_kv" in plan["train_families"]
    assert "code_json" in plan["train_families"]
    assert "code_kv" in plan["train_families"]
    assert "person_json" not in plan["train_families"]

    label = queue.FAMILIES["label_json"]
    assert (label.key1, label.key2) == ("label", "points")
    assert label.pool == "label"
    assert label.entities
    assert queue.FAMILIES["label_kv"].key2 == "points"
    assert queue.FAMILIES["person_yaml"].format == "yaml"
    assert queue.FAMILIES["product_yaml"].format == "yaml"
    assert "person_yaml" in queue.TRAIN_FAMILIES
    assert "product_yaml" in queue.TRAIN_FAMILIES


def test_holdout_keys_make_matching_families_eval_only():
    plan = queue.resolve_family_plan("all", "points,name")
    for family in plan["train_families"]:
        pair = queue.PAIR_BY_FAMILY.get(family)
        if pair:
            assert not (set(pair) & {"points", "name"})
    for family in ("person_json", "person_kv", "person_yaml",
                   "label_json", "label_kv"):
        assert family in plan["cross_families"]
        assert family not in plan["train_families"]

    rows = queue.build_rows(
        plan["train_families"] + plan["cross_families"], None, count=3,
        seed=5, split_by="entity",
        train_families=plan["train_families"],
        cross_families=plan["cross_families"])
    files = queue.split_rows(
        rows, train_families=plan["train_families"],
        cross_families=plan["cross_families"], split_by="entity")
    train_families = {row["family"] for row in files["train"]}
    assert not train_families & set(plan["cross_families"])
    assert {row["family"] for row in files["cross"]} == set(
        plan["cross_families"])


def test_entity_split_has_zero_overlap_and_shared_pool_sides():
    plan = queue.resolve_family_plan("all", "population")
    rows = queue.build_rows(
        plan["train_families"] + plan["cross_families"], None, count=5,
        seed=7, split_by="entity",
        train_families=plan["train_families"],
        cross_families=plan["cross_families"])
    files = queue.split_rows(
        rows, train_families=plan["train_families"],
        cross_families=plan["cross_families"], split_by="entity")
    report = queue.split_summary(files, mode="entity")

    assert files["train"]
    assert files["indist"]
    assert report["entities_train"] > 0
    assert report["entities_eval"] > 0
    assert report["entity_overlap"] == 0
    assert not (_entities(files["train"])
                & _entities(files["indist"] + files["cross"]))

    sides = {}
    for row in files["train"] + files["indist"]:
        if row["family"] in {"person_json", "person_kv", "person_yaml"}:
            sides.setdefault(row["entity"], set()).add(row["partition"])
    assert sides
    assert all(len(values) == 1 for values in sides.values())


def test_entity_split_keeps_both_sides_for_small_counts():
    plan = queue.resolve_family_plan("label")
    rows = queue.build_rows(
        plan["train_families"] + plan["cross_families"], None, count=2,
        seed=2, split_by="entity",
        train_families=plan["train_families"],
        cross_families=plan["cross_families"])
    files = queue.split_rows(
        rows, train_families=plan["train_families"],
        cross_families=plan["cross_families"], split_by="entity")
    assert files["train"]
    assert files["indist"]
    assert queue.split_summary(
        files, mode="entity")["entity_overlap"] == 0


def test_canonical_json_and_kv_after_validation():
    assert queue.canonical_answer(
        "person_json", '{"name": "Ada", "score": 7}', "Ada", 7
    ) == '{"name":"Ada","score":7}'
    assert queue.canonical_answer(
        "person_kv", "name = Ada ; score = 7", "Ada", 7
    ) == "name=Ada;score=7"
    assert queue.canonical_answer(
        "person_json", '{"name": "Bora", "score": 7}', "Ada", 7
    ) == '{"name": "Bora", "score": 7}'

    def stub_teacher(prompt):
        match = re.search(r"name=(\w+) and score=(\d+)", prompt)
        return json.dumps(
            {"name": match.group(1), "score": int(match.group(2))})

    rows = queue.build_rows(
        ["person_json"], stub_teacher, count=2, seed=11, canonical=True)
    assert all(row["verification"]["verified"] for row in rows)
    assert all(
        row["assistant"] == json.dumps(
            json.loads(row["assistant"]), separators=(",", ":"))
        for row in rows)
    assert all(", " not in row["assistant"] for row in rows)

    def kv_teacher(prompt):
        match = re.search(r"name=(\w+);score=(\d+)", prompt)
        return f"name = {match.group(1)} ; score = {match.group(2)}"

    kv_rows = queue.build_rows(
        ["person_kv"], kv_teacher, count=1, seed=11, canonical=True)
    match = re.search(r"name=(\w+);score=(\d+)", kv_rows[0]["user"])
    assert kv_rows[0]["verification"]["verified"]
    assert kv_rows[0]["assistant"] == (
        f"name={match.group(1)};score={match.group(2)}")

    oracle = queue.build_rows(
        ["person_json"], None, count=1, seed=11, canonical=True)
    assert oracle[0]["assistant"].startswith('{"name":"')
    assert ", " not in oracle[0]["assistant"]

    legacy = queue.build_rows(["person_json"], None, count=1, seed=11)
    assert ", " in legacy[0]["assistant"]


def test_replay_out_excludes_primary_train_rows(tmp_path):
    out_dir = tmp_path / "out"
    replay_path = tmp_path / "replay.jsonl"
    code = queue.main([
        "--dry-run", "--key-pairs", "all", "--split-by", "entity",
        "--per-family", "4", "--replay-out", str(replay_path),
        "--out-dir", str(out_dir), "--seed", "9",
    ])
    assert code == 0
    train = _read_jsonl(out_dir / "train.jsonl")
    indist = _read_jsonl(out_dir / "indist.jsonl")
    cross = _read_jsonl(out_dir / "cross.jsonl")
    replay = _read_jsonl(replay_path)

    assert replay
    assert all(row.get("replay") is True for row in replay)
    assert all(row.get("partition") == "replay" for row in replay)
    assert not ({_row_key(row) for row in train}
                & {_row_key(row) for row in replay})
    assert not ({row["family"] for row in train}
                & set(queue.LEGACY_TRAIN_FAMILIES))
    assert not (_entities(train + replay)
                & _entities(indist + cross))


def test_shuffle_is_deterministic_for_a_seed():
    rows = queue.build_rows(
        ["person_json", "label_json"], None, count=6, seed=3)
    first = queue.shuffle_rows(rows, seed=3)
    second = queue.shuffle_rows(rows, seed=3)
    assert first == second
    assert sorted(row["assistant"] for row in first) == sorted(
        row["assistant"] for row in rows)


def test_cli_curriculum_defaults_to_twenty_rows(capsys, tmp_path):
    code = queue.main([
        "--dry-run", "--key-pairs", "label:points,code",
        "--out-dir", str(tmp_path / "out"),
    ])
    assert code == 0
    summary = _cli_summary(capsys)
    assert summary["counts"]["label_json"] == 20
    assert summary["counts"]["label_kv"] == 20
    assert summary["counts"]["code_json"] == 20
    assert summary["counts"]["arith_plain"] == 20


def test_cli_legacy_counts_are_unchanged(capsys, tmp_path):
    code = queue.main([
        "--dry-run", "--out-dir", str(tmp_path / "out"),
    ])
    assert code == 0
    summary = _cli_summary(capsys)
    assert summary["counts"]["person_json"] == 40
    assert summary["counts"]["city_json"] == 25
    assert summary["counts"]["arith_plain"] == 20
    assert summary["key_pairs"] == [
        "name:score", "product:price", "city:population"]
    assert summary["split"]["mode"] == "row"
