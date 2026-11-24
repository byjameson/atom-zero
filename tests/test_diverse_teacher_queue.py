import json
import re

import pytest

from research import build_diverse_teacher_queue as queue


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
