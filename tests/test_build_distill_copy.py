import json
import re
from pathlib import Path

import pytest

from research import build_distill_copy as distill


PAIR_BLOCK = re.compile(r"\[([^\]]*)\]")


def _pairs(answer):
    return [(str(key), value) for key, value in
            distill.parse_teacher_answer(answer)]


def _source_row(keys, values, fmt="json", stage="2short", family=None):
    pairs = [[key, value] for key, value in zip(keys, values)]
    rendered = "; ".join(f"{key}={value}" for key, value in pairs)
    label = {"json": "JSON", "kv": "KV", "yaml": "YAML"}[fmt]
    user = (f"Copy these pairs exactly into {label}: [{rendered}]. "
            "Return only the requested copy.")
    assistant = distill.canonicalize(fmt, pairs)
    return {
        "user": user,
        "assistant": assistant,
        "family": family or f"multi_short_{stage}",
        "stage": stage,
        "format": fmt,
        "partition": "train",
        "keys": list(keys),
        "values": list(values),
        "pairs": pairs,
    }


def _oracle_teacher(prompt):
    match = PAIR_BLOCK.search(prompt)
    assert match, prompt
    pairs = []
    for chunk in match.group(1).split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        key, value = chunk.split("=", 1)
        pairs.append((key.strip(), value.strip()))
    if "JSON" in prompt:
        fmt = "json"
    elif "KV" in prompt:
        fmt = "kv"
    else:
        fmt = "yaml"
    return distill.canonicalize(fmt, pairs)


def _write_source(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_tolerant_parse_flat_json_pairs_and_fences():
    assert _pairs('{"a": "1", "b": "two"}') == [
        ("a", "1"), ("b", "two")]
    assert _pairs('{"pairs": [["a", "1"], ["b", "two"]]}') == [
        ("a", "1"), ("b", "two")]
    assert _pairs(
        '{"pairs": [{"source": "a", "target": "1"}, '
        '{"source": "b", "target": "two"}]}') == [
        ("a", "1"), ("b", "two")]
    assert _pairs('```json\n{"a": "1", "b": "two"}\n```') == [
        ("a", "1"), ("b", "two")]
    assert _pairs(
        'Here is the answer:\n```json\n{"b": "two", "a": "1"}\n```\nDone.'
    ) == [("b", "two"), ("a", "1")]
    assert _pairs('[["a", "1"], ["b", "two"]]') == [
        ("a", "1"), ("b", "two")]


def test_tolerant_parse_kv_and_yaml():
    assert _pairs("a=1; b=two") == [("a", "1"), ("b", "two")]
    assert _pairs("a = 1\nb = two") == [("a", "1"), ("b", "two")]
    assert _pairs("```\na=1;b=two\n```") == [("a", "1"), ("b", "two")]
    assert _pairs("a: 1; b: two") == [("a", "1"), ("b", "two")]
    assert _pairs("a: 1\nb: two") == [("a", "1"), ("b", "two")]
    assert _pairs("```yaml\nb: 'two'\na: 1\n```") == [
        ("b", "two"), ("a", "1")]


def test_tolerant_parse_pair_objects_with_numbered_keys():
    assert _pairs(
        '{"pairs": [{"pair1": "a", "value1": "1"}, '
        '{"pair2": "b", "value2": "two"}]}') == [
        ("a", "1"), ("b", "two")]
    assert _pairs(
        '{"pairs": [{"key1": "a", "value1": "1"}, '
        '{"key2": "b", "value2": "two"}]}') == [
        ("a", "1"), ("b", "two")]


def test_canonicalization_to_flat_json_kv_and_yaml():
    pairs = [("k1", "v1"), ("k2", "v two")]
    assert distill.canonicalize("json", pairs) == (
        '{"k1":"v1","k2":"v two"}')
    assert distill.canonicalize("kv", pairs) == "k1=v1;k2=v two"
    assert distill.canonicalize("yaml", pairs) == "k1: v1\nk2: v two"


def test_exact_key_value_verification_is_order_insensitive():
    expected = [("a", "1"), ("b", "two")]
    assert distill.pairs_match([("b", "two"), ("a", "1")], expected)
    assert not distill.pairs_match([("a", "1"), ("c", "two")], expected)
    assert not distill.pairs_match([("a", "1")], expected)
    assert not distill.pairs_match(
        [("a", "1"), ("b", "two"), ("c", "3")], expected)
    assert not distill.pairs_match([("a", "2"), ("b", "two")], expected)
    assert not distill.pairs_match([("a", "1"), ("a", "two")], expected)


def test_invalid_teacher_output_is_rejected_or_falls_back():
    first = _source_row(["a1", "b1"], ["one", "two"])
    second = _source_row(["c2", "d2"], ["three", "four"], stage="3short")

    def teacher(prompt):
        if "a1=one" in prompt:
            return '```json\n{"b1": "two", "a1": "one"}\n```'
        return "I cannot copy that."

    rows, counts = distill.build_distill(
        [first, second], teacher, model="stub-teacher",
        fallback_oracle=False)
    assert len(rows) == 1
    assert counts["requested"] == 2
    assert counts["generated"] == 1
    assert counts["teacher_valid"] == 1
    assert counts["fallback_oracle"] == 0
    assert counts["rejected"] == 1
    assert rows[0]["teacher_verified"] is True
    assert rows[0]["canonicalized"] is True
    assert rows[0]["assistant"] == '{"a1":"one","b1":"two"}'

    rows, counts = distill.build_distill(
        [first, second], teacher, model="stub-teacher",
        fallback_oracle=True)
    assert len(rows) == 2
    assert counts["teacher_valid"] == 1
    assert counts["fallback_oracle"] == 1
    assert counts["rejected"] == 0
    fallback = [row for row in rows if not row["teacher_verified"]]
    assert len(fallback) == 1
    assert fallback[0]["assistant"] == second["assistant"]
    assert fallback[0]["canonicalized"] is False
    assert fallback[0]["stage"] == "3short"


def test_renamed_or_missing_keys_are_rejected_even_with_fallback():
    row = _source_row(["key1", "key2"], ["alpha", "beta"])

    def renamed(prompt):
        return '{"key1": "alpha", "other": "beta"}'

    rows, counts = distill.build_distill(
        [row], renamed, fallback_oracle=True)
    # The source assistant is a valid oracle fallback, so it is retained, but
    # the teacher answer is not marked verified.
    assert len(rows) == 1
    assert counts["teacher_valid"] == 0
    assert counts["fallback_oracle"] == 1
    assert rows[0]["teacher_verified"] is False

    rows, counts = distill.build_distill(
        [row], renamed, fallback_oracle=False)
    assert rows == []
    assert counts["rejected"] == 1


def test_truncated_teacher_json_is_rejected_not_salvaged():
    row = _source_row(["a1", "b1"], ["one", "two"])
    truncated = ('{"pairs": [{"source": "a1", "target": "one"}, '
                 '{"source": "b1", "target":')

    def teacher(prompt):
        return "```json\n" + truncated

    rows, counts = distill.build_distill(
        [row], teacher, fallback_oracle=False)
    assert rows == []
    assert counts["teacher_valid"] == 0
    assert counts["rejected"] == 1


def test_yaml_source_is_canonicalized_from_a_reordered_teacher_answer():
    row = _source_row(["q1", "q2"], ["alpha", "beta"], fmt="yaml")

    def teacher(prompt):
        return "```yaml\nq2: beta\nq1: alpha\n```"

    rows, counts = distill.build_distill([row], teacher)
    assert counts["teacher_valid"] == 1
    assert rows[0]["format"] == "yaml"
    assert rows[0]["assistant"] == "q1: alpha\nq2: beta"
    assert rows[0]["keys"] == ["q1", "q2"]
    assert rows[0]["values"] == ["alpha", "beta"]


def test_row_limit_seed_and_round_robin_are_deterministic(tmp_path):
    rows_a = [
        _source_row([f"a{index}", f"b{index}"],
                    [f"va{index}", f"vb{index}"], stage="2short")
        for index in range(10)
    ]
    rows_b = [
        _source_row([f"c{index}", f"d{index}"],
                    [f"vc{index}", f"vd{index}"], stage="3short")
        for index in range(10)
    ]
    source_a = _write_source(tmp_path / "stage_2short" / "train.jsonl",
                             rows_a)
    source_b = _write_source(tmp_path / "stage_3short" / "train.jsonl",
                             rows_b)

    first = distill.load_source_rows(
        [source_a, source_b], rows=8, seed=7, shuffle=True)
    second = distill.load_source_rows(
        [source_a, source_b], rows=8, seed=7, shuffle=True)
    other = distill.load_source_rows(
        [source_a, source_b], rows=8, seed=8, shuffle=True)
    assert len(first) == len(second) == len(other) == 8
    assert [row["user"] for row in first] == [row["user"] for row in second]
    assert [row["user"] for row in first] != [row["user"] for row in other]

    ordered = distill.load_source_rows(
        [source_a, source_b], rows=4, seed=7, shuffle=False)
    assert len(ordered) == 4
    assert [row["stage"] for row in ordered] == [
        "2short", "3short", "2short", "3short"]

    limited = distill.load_source_rows(
        [source_a, source_b], rows=3, seed=7, shuffle=False)
    assert len(limited) == 3


def test_manifest_is_content_free_and_counts_add_up(tmp_path):
    rows = [
        _source_row(["SENTINELKEY1", "SENTINELKEY2"],
                    ["SENTINELVALUE1", "SENTINELVALUE2"], stage="2short"),
        _source_row(["SENTINELKEY3", "SENTINELKEY4"],
                    ["SENTINELVALUE3", "SENTINELVALUE4"], stage="3short"),
    ]
    built, counts = distill.build_distill(
        rows, _oracle_teacher, model="stub-teacher")
    manifest = distill.build_manifest(
        counts, model="stub-teacher", seed=23,
        sources=["data/multi_short_v1/stage_2short/train.jsonl"],
        formats="json,kv,yaml")
    text = json.dumps(manifest)
    for sentinel in ("SENTINELKEY1", "SENTINELVALUE1", "SENTINELKEY4",
                     "SENTINELVALUE4", "Copy these pairs"):
        assert sentinel not in text
    assert manifest["requested"] == 2
    assert manifest["generated"] == 2
    assert manifest["teacher_valid"] == 2
    assert manifest["fallback_oracle"] == 0
    assert manifest["rejected"] == 0
    assert manifest["by_format"]["json"] == 2
    assert manifest["by_stage"] == {"2short": 1, "3short": 1}
    assert "note" in manifest
    assert manifest["note"] == distill.MANIFEST_NOTE

    paths = distill.write_dataset(built, manifest, tmp_path / "out")
    train_path = Path(paths["train"])
    manifest_path = Path(paths["manifest"])
    written = [json.loads(line) for line in
               train_path.read_text(encoding="utf-8").splitlines()]
    assert len(written) == 2
    assert "SENTINELKEY1" not in manifest_path.read_text(encoding="utf-8")
    assert written[0]["family"] == "distill_copy_2short"
    assert written[0]["source_family"] == "multi_short_2short"
    assert written[0]["teacher_model"] == "stub-teacher"
    assert written[0]["teacher_verified"] is True
    assert written[0]["canonicalized"] is True


def test_cli_writes_manifest_and_train_file(tmp_path):
    source = _write_source(
        tmp_path / "stage_2short" / "train.jsonl",
        [_source_row(["e1", "f1"], ["one", "two"], stage="2short")])
    out_dir = tmp_path / "out"
    code = distill.main([
        "--source", str(source),
        "--out-dir", str(out_dir),
        "--rows", "1",
        "--model", "stub-teacher",
        "--device", "cpu",
        "--formats", "json,kv,yaml",
        "--shuffle",
    ], teacher_factory=lambda *args, **kwargs: _oracle_teacher)
    assert code == 0
    manifest = json.loads(
        (out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["requested"] == 1
    assert manifest["generated"] == 1
    assert manifest["teacher_valid"] == 1
    rows = [json.loads(line) for line in
            (out_dir / "train_distill.jsonl")
            .read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["assistant"] == '{"e1":"one","f1":"two"}'


def test_eval_sources_and_eval_partitions_are_refused(tmp_path):
    eval_source = _write_source(
        tmp_path / "stage_2short" / "eval.jsonl",
        [_source_row(["g1", "h1"], ["one", "two"])])
    with pytest.raises(ValueError, match="refusing eval source"):
        distill.load_source_rows([eval_source])

    leaked = _source_row(["g1", "h1"], ["one", "two"])
    leaked["partition"] = "eval"
    train_source = _write_source(
        tmp_path / "stage_2short" / "train.jsonl", [leaked])
    with pytest.raises(ValueError, match="refusing non-train row"):
        distill.load_source_rows([train_source])
