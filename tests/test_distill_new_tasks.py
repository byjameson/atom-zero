import json
import re
from pathlib import Path

import pytest

from research import build_distill_new_tasks as newtasks


class StubTeacher:
    """Deterministic teacher that returns well-formed tasks per prompt."""

    def __init__(self):
        self.calls = 0
        self.prompts = []

    def __call__(self, prompt):
        self.calls += 1
        self.prompts.append(prompt)
        key_count = int(re.search(
            r"exactly (\d+) key/value pairs", prompt).group(1))
        count = int(re.search(
            r"Generate (\d+) new copy-practice records", prompt).group(1))
        lines = []
        for index in range(count):
            pairs = {}
            for offset in range(key_count):
                key = f"k{self.calls:02d}{index:02d}{offset}"
                pairs[key] = f"v{self.calls}_{index}_{offset}"
            lines.append(json.dumps(pairs, separators=(",", ":")))
        return "\n".join(lines)


def _train_row(user="old prompt", assistant="old answer", pairs=None,
               partition="train"):
    row = {
        "user": user,
        "assistant": assistant,
        "family": "multi_short_2short",
        "format": "json",
        "partition": partition,
    }
    if pairs is not None:
        row["pairs"] = pairs
        row["keys"] = [key for key, _ in pairs]
        row["values"] = [value for _, value in pairs]
    return row


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_tolerant_parse_flat_fenced_pretty_and_prefixed_json():
    expected = {"a1b": "one", "c2d": "two"}
    assert newtasks.parse_teacher_tasks(
        '{"a1b": "one", "c2d": "two"}') == [expected]
    assert newtasks.parse_teacher_tasks(
        '```json\n{"a1b": "one", "c2d": "two"}\n```') == [expected]
    assert newtasks.parse_teacher_tasks(
        '1. {"a1b": "one", "c2d": "two"}') == [expected]
    assert newtasks.parse_teacher_tasks(
        '- `{"a1b": "one", "c2d": "two"}`') == [expected]
    assert newtasks.parse_teacher_tasks(
        'Here are the records:\n'
        '{\n  "a1b": "one",\n  "c2d": "two"\n}\n') == [expected]
    assert newtasks.parse_teacher_tasks(
        '{"a1b": "one", "c2d": "two"}\n'
        '{"e3f": "three", "g4h": "four"}') == [
            expected, {"e3f": "three", "g4h": "four"}]


def test_tolerant_parse_kv_yaml_and_single_object_array():
    assert newtasks.parse_teacher_tasks("a1b=one; c2d=two") == [
        {"a1b": "one", "c2d": "two"}]
    assert newtasks.parse_teacher_tasks("a1b: one, c2d: two") == [
        {"a1b": "one", "c2d": "two"}]
    assert newtasks.parse_teacher_tasks(
        "```yaml\na1b: one\nc2d: two\n```") == [
            {"a1b": "one", "c2d": "two"}]
    assert newtasks.parse_teacher_tasks(
        'Here are the records:\na1b: one\nc2d: two') == [
            {"a1b": "one", "c2d": "two"}]
    assert newtasks.parse_teacher_tasks(
        '[{"a1b": "one", "c2d": "two"}]') == [
            {"a1b": "one", "c2d": "two"}]
    assert newtasks.parse_task_line(
        'Record 2: {"a1b": "one", "c2d": "two"}') == {
            "a1b": "one", "c2d": "two"}


def test_one_line_kv_batch_falls_back_when_block_count_is_wrong():
    answer = "a1b=one; c2d=two\ne3f=three; g4h=four"
    assert newtasks.parse_teacher_tasks(answer, key_count=2) == [
        {"a1b": "one", "c2d": "two"},
        {"e3f": "three", "g4h": "four"},
    ]


def test_validation_rejects_bad_keys_and_values_and_accepts_numbers():
    assert newtasks.validate_task(
        {"a1b": "one", "c2d": 42}, 2) == (
            ("a1b", "one"), ("c2d", "42"))
    with pytest.raises(ValueError):
        newtasks.validate_task({"a1b": "one"}, 2)
    with pytest.raises(ValueError):
        newtasks.validate_task(
            {"a1b": "one", "c2d": "two", "e3f": "three"}, 2)
    for bad_key in ("ab", "abcdefg", "Abc", "a-b", "key", "value",
                    "1ab", "123"):
        with pytest.raises(ValueError):
            newtasks.validate_task({bad_key: "one", "c2d": "two"}, 2)
    for bad_value in ("", "x" * 41, "hello,world", "a=b", "x;y",
                      "line\nbreak", {"nested": 1}, [1], True, None):
        with pytest.raises(ValueError):
            newtasks.validate_task({"a1b": bad_value, "c2d": "two"}, 2)
    assert newtasks.validate_task(
        {"a1b": 1.0, "c2d": -3}, 2) == (
            ("a1b", "1.0"), ("c2d", "-3"))


def test_exact_canonical_form_and_prompt_list_pairs():
    pairs = (("a1b", "one"), ("c2d", "2"))
    assert newtasks.canonicalize("json", pairs) == '{"a1b":"one","c2d":"2"}'
    assert newtasks.canonicalize("kv", pairs) == "a1b=one;c2d=2"
    assert newtasks.canonicalize("yaml", pairs) == "a1b: one\nc2d: 2"

    json_prompt = newtasks.build_prompt("json", pairs)
    assert "a1b=one; c2d=2" in json_prompt
    assert "JSON" in json_prompt
    assert "exactly" in json_prompt
    kv_prompt = newtasks.build_prompt("kv", pairs)
    assert "a1b=one; c2d=2" in kv_prompt
    assert "key=value" in kv_prompt


def test_rows_have_requested_format_key_count_and_metadata():
    teacher = StubTeacher()
    rows, counts = newtasks.build_new_tasks(
        teacher, rows=6, batch_size=2, keys_per_row="2,3",
        formats="json,kv", seed=7, model="stub-teacher")

    assert len(rows) == 6
    assert counts["requested"] == 6
    assert counts["accepted"] == 6
    assert counts["rejected"] == 0
    assert counts["by_keys"] == {"2": 3, "3": 3}
    assert counts["by_format"]["json"] + counts["by_format"]["kv"] == 6
    assert counts["by_format"]["yaml"] == 0
    for row in rows:
        assert row["partition"] == "train"
        assert row["teacher_verified"] is True
        assert row["canonicalized"] is True
        assert row["teacher_model"] == "stub-teacher"
        assert row["format"] in {"json", "kv"}
        assert row["family"] == (
            f"newtask_{row['key_count']}key_{row['format']}")
        assert len(row["pairs"]) == row["key_count"] == len(row["keys"])
        assert row["user"].count("=") >= len(row["pairs"])
        if row["format"] == "json":
            parsed = json.loads(row["assistant"])
            assert [[key, value] for key, value in parsed.items()] == (
                row["pairs"])
        else:
            assert row["assistant"] == ";".join(
                f"{key}={value}" for key, value in row["pairs"])


def test_duplicate_teacher_rows_are_dropped_and_counted():
    duplicate = '{"a1b":"one","c2d":"two"}'

    def teacher(prompt):
        return "\n".join([duplicate] * 5)

    rows, counts = newtasks.build_new_tasks(
        teacher, rows=4, batch_size=4, keys_per_row="2",
        formats="json", seed=3)
    assert len(rows) == 1
    assert counts["accepted"] == 1
    assert counts["rejected"] == 3
    assert counts["duplicate_count"] == 4


def test_invalid_and_truncated_teacher_output_is_rejected():
    def teacher(prompt):
        return "\n".join([
            '{"a1b": "one"}',
            '{"AB": "one", "c2d": "two"}',
            '{"key": "one", "c2d": "two"}',
            '{"a1b": "' + "x" * 41 + '", "c2d": "two"}',
            "I cannot generate that.",
            '{"a1b": "one", "c2d":',
            '{"e3f": "three", "g4h": "four"}',
        ])

    rows, counts = newtasks.build_new_tasks(
        teacher, rows=5, batch_size=5, keys_per_row="2",
        formats="json", seed=3)
    assert len(rows) == 1
    assert counts["accepted"] == 1
    assert counts["rejected"] == 4
    assert counts["invalid_tasks"] == 4
    assert rows[0]["pairs"] == [["e3f", "three"], ["g4h", "four"]]


def test_truncated_pretty_json_is_not_salvaged_as_yaml():
    truncated = '{\n  "a1b": "one",\n  "c2d": "two"\n'
    assert newtasks.parse_teacher_tasks(truncated, key_count=2) == []

    def teacher(prompt):
        return truncated

    rows, counts = newtasks.build_new_tasks(
        teacher, rows=1, batch_size=1, keys_per_row="2",
        formats="json", seed=3)
    assert rows == []
    assert counts["accepted"] == 0
    assert counts["rejected"] == 1


def test_deterministic_seed_and_shuffle_schedule():
    first = newtasks.build_task_specs(
        12, keys_per_row="2,3,4", formats="json,kv", yaml_count=2,
        seed=11, shuffle=True)
    second = newtasks.build_task_specs(
        12, keys_per_row="2,3,4", formats="json,kv", yaml_count=2,
        seed=11, shuffle=True)
    other = newtasks.build_task_specs(
        12, keys_per_row="2,3,4", formats="json,kv", yaml_count=2,
        seed=12, shuffle=True)
    assert first == second
    assert first != other
    assert sum(spec.format == "yaml" for spec in first) == 2
    assert {spec.key_count for spec in first} == {2, 3, 4}

    ordered = newtasks.build_task_specs(
        6, keys_per_row="2,3,4", formats="json,kv", seed=11,
        shuffle=False)
    assert [spec.key_count for spec in ordered] == [2, 3, 4, 2, 3, 4]
    assert [spec.format for spec in ordered] == [
        "json", "kv", "json", "kv", "json", "kv"]

    rows_a, counts_a = newtasks.build_new_tasks(
        StubTeacher(), rows=6, batch_size=3, keys_per_row="2,3",
        formats="json,kv", seed=5, shuffle=True)
    rows_b, counts_b = newtasks.build_new_tasks(
        StubTeacher(), rows=6, batch_size=3, keys_per_row="2,3",
        formats="json,kv", seed=5, shuffle=True)
    assert rows_a == rows_b
    assert counts_a == counts_b


def test_manifest_is_content_free_and_counts_add_up(tmp_path):
    def teacher(prompt):
        key_count = int(re.search(
            r"exactly (\d+) key/value pairs", prompt).group(1))
        count = int(re.search(
            r"Generate (\d+) new copy-practice records", prompt).group(1))
        rows = []
        for index in range(count):
            pairs = {}
            for offset in range(key_count):
                pairs[f"s{index}{offset}"] = (
                    f"SENTINELVALUE{index}{offset}")
            rows.append(json.dumps(pairs, separators=(",", ":")))
        return "\n".join(rows)

    rows, counts = newtasks.build_new_tasks(
        teacher, rows=2, batch_size=2, keys_per_row="2",
        formats="json", seed=23, model="stub-teacher")
    manifest = newtasks.build_manifest(
        counts, model="stub-teacher", seed=23, keys_per_row="2",
        formats="json", batch_size=2)
    text = json.dumps(manifest)
    for sentinel in ("s00", "SENTINELVALUE00", "Copy every key",
                     "compact JSON object"):
        assert sentinel not in text
    assert manifest["requested"] == 2
    assert manifest["accepted"] == 2
    assert manifest["rejected"] == 0
    assert manifest["duplicate_count"] == 0
    assert manifest["by_keys"] == {"2": 2}
    assert manifest["by_format"]["json"] == 2
    assert manifest["teacher_model"] == "stub-teacher"
    assert manifest["note"] == newtasks.MANIFEST_NOTE

    paths = newtasks.write_dataset(rows, manifest, tmp_path / "out")
    train_path = Path(paths["train"])
    manifest_path = Path(paths["manifest"])
    written = [json.loads(line) for line in
               train_path.read_text(encoding="utf-8").splitlines()]
    assert len(written) == 2
    assert "SENTINELVALUE00" not in manifest_path.read_text(
        encoding="utf-8")
    assert written[0]["family"] == "newtask_2key_json"
    assert written[0]["teacher_verified"] is True


def test_avoid_source_rejects_source_text_and_eval_paths(tmp_path):
    avoid_path = _write_jsonl(
        tmp_path / "stage_2short" / "train.jsonl",
        [_train_row(pairs=[["a1b", "one"], ["c2d", "two"]])])
    avoid_rows = newtasks.load_avoid_rows([avoid_path])
    assert newtasks.build_avoid_index(avoid_rows).signatures

    def teacher(prompt):
        return '{"a1b":"one","c2d":"two"}'

    rows, counts = newtasks.build_new_tasks(
        teacher, rows=1, batch_size=1, keys_per_row="2",
        formats="json", avoid_rows=avoid_rows)
    assert rows == []
    assert counts["accepted"] == 0
    assert counts["source_overlap"] == 1
    assert counts["rejected"] == 1

    eval_path = _write_jsonl(
        tmp_path / "stage_2short" / "eval.jsonl",
        [_train_row(pairs=[["a1b", "one"], ["c2d", "two"]])])
    with pytest.raises(ValueError, match="refusing eval source"):
        newtasks.load_avoid_rows([eval_path])


def test_cli_writes_dataset_with_injected_teacher(tmp_path):
    out_dir = tmp_path / "out"

    def factory(model, *, device, max_new_tokens, local_files_only):
        assert model == "stub-teacher"
        assert device == "cpu"
        assert max_new_tokens == 128
        assert local_files_only is True
        return StubTeacher()

    code = newtasks.main([
        "--out-dir", str(out_dir),
        "--rows", "4",
        "--batch-size", "2",
        "--keys-per-row", "2,3",
        "--formats", "json,kv",
        "--max-new-tokens", "128",
        "--seed", "9",
        "--model", "stub-teacher",
        "--device", "cpu",
        "--shuffle",
    ], teacher_factory=factory)
    assert code == 0
    manifest = json.loads(
        (out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["requested"] == 4
    assert manifest["accepted"] == 4
    assert manifest["rejected"] == 0
    assert manifest["teacher_model"] == "stub-teacher"
    rows = [json.loads(line) for line in
            (out_dir / "train_new_tasks.jsonl")
            .read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 4
    assert all(row["teacher_verified"] for row in rows)


def test_cli_zero_rows_writes_empty_dataset_without_loading_teacher(tmp_path):
    def factory(*args, **kwargs):
        raise AssertionError("teacher must not be loaded when rows is zero")

    out_dir = tmp_path / "empty"
    code = newtasks.main([
        "--out-dir", str(out_dir),
        "--rows", "0",
        "--model", "stub-teacher",
        "--device", "cpu",
    ], teacher_factory=factory)
    assert code == 0
    manifest = json.loads(
        (out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["requested"] == 0
    assert manifest["accepted"] == 0
    assert manifest["rejected"] == 0
    assert (out_dir / "train_new_tasks.jsonl").read_text(
        encoding="utf-8") == ""


def test_cli_refuses_eval_avoid_source_before_loading_teacher(tmp_path):
    eval_path = _write_jsonl(
        tmp_path / "stage_2short" / "eval.jsonl",
        [_train_row(pairs=[["a1b", "one"], ["c2d", "two"]])])

    def factory(*args, **kwargs):
        raise AssertionError("teacher must not be loaded for an invalid path")

    with pytest.raises(SystemExit):
        newtasks.main([
            "--out-dir", str(tmp_path / "out"),
            "--rows", "1",
            "--avoid-source", str(eval_path),
        ], teacher_factory=factory)


def test_yaml_count_reserves_yaml_rows_and_uses_one_line_yaml():
    specs = newtasks.build_task_specs(
        4, keys_per_row="2", formats="json,kv", yaml_count=1, seed=1)
    assert [spec.format for spec in specs] == ["yaml", "json", "kv", "json"]

    rows, counts = newtasks.build_new_tasks(
        StubTeacher(), rows=3, batch_size=3, keys_per_row="2",
        formats="json,kv", yaml_count=1, seed=1)
    assert len(rows) == 3
    assert counts["by_format"]["yaml"] == 1
    yaml_rows = [row for row in rows if row["format"] == "yaml"]
    assert len(yaml_rows) == 1
    lines = yaml_rows[0]["assistant"].splitlines()
    assert len(lines) == yaml_rows[0]["key_count"]
    assert all(": " in line for line in lines)
    assert yaml_rows[0]["family"] == "newtask_2key_yaml"
