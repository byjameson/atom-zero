import json
from pathlib import Path

import pytest

from research import build_random_copy_queue as queue


def _read_jsonl(path):
    return [json.loads(line) for line in
            Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _dataset(rows=24, eval_rows=8, **overrides):
    options = {
        "rows": rows, "eval_rows": eval_rows,
        "formats": ("json", "kv"), "keys_per_row": 2, "seed": 23,
        "shuffle": False,
    }
    options.update(overrides)
    return queue.build_dataset(**options)


def _row_signature(rows):
    return sorted(json.dumps(row, sort_keys=True, ensure_ascii=False)
                  for row in rows)


def test_rows_have_the_teacher_queue_schema():
    dataset = _dataset()
    train = dataset["train"]
    eval_rows = dataset["eval"]
    assert len(train) == 24
    assert len(eval_rows) == 8
    assert dataset["summary"]["train_rows"] == 24
    assert dataset["summary"]["eval_rows"] == 8

    for row in train + eval_rows:
        assert {
            "user", "assistant", "family", "format", "partition", "keys",
            "values", "pairs", "key_styles", "value_pools", "verification",
        } <= set(row)
        assert row["format"] in queue.FORMATS
        assert row["family"] == f"random_copy_{row['format']}"
        assert row["partition"] in {"train", "eval"}
        assert row["verification"]["verified"] is True
        assert row["verification"]["promotable"] is True
        assert row["verification"]["method"] == queue.VERIFICATION_METHOD
        assert 2 <= len(row["keys"]) <= 4
        assert len(row["keys"]) == len(row["values"]) == len(row["pairs"])
        assert len(row["keys"]) == len(set(row["keys"]))
        assert len(row["keys"]) == len(row["key_styles"])
        assert len(row["values"]) == len(row["value_pools"])
        assert "multiword" in row["value_pools"]
        assert [pair[0] for pair in row["pairs"]] == row["keys"]
        assert [pair[1] for pair in row["pairs"]] == row["values"]
        assert queue.validate_row(row)
        for key, value in row["pairs"]:
            assert f"{key}={value}" in row["user"]

    assert all(row["partition"] == "train" for row in train)
    assert all(row["partition"] == "eval" for row in eval_rows)


def test_train_and_eval_share_no_keys_or_values():
    dataset = _dataset(rows=120, eval_rows=40,
                       formats=("json", "kv", "yaml"))
    train = dataset["train"]
    eval_rows = dataset["eval"]
    train_keys = {key for row in train for key in row["keys"]}
    eval_keys = {key for row in eval_rows for key in row["keys"]}
    train_values = {str(value) for row in train for value in row["values"]}
    eval_values = {str(value) for row in eval_rows for value in row["values"]}

    assert train_keys & eval_keys == set()
    assert train_values & eval_values == set()
    assert train_keys & queue.RESERVED_KEYS == set()
    assert dataset["summary"]["key_overlap"] == 0
    assert dataset["summary"]["key_overlap_examples"] == []
    assert dataset["summary"]["value_overlap"] == 0
    assert dataset["summary"]["value_overlap_examples"] == []
    assert dataset["summary"]["unique_keys"] == {
        "train": len(train_keys), "eval": len(eval_keys),
        "total": len(train_keys | eval_keys),
    }
    assert dataset["summary"]["unique_values"] == {
        "train": len(train_values), "eval": len(eval_values),
        "total": len(train_values | eval_values),
    }


def test_seed_is_deterministic():
    first = _dataset(seed=23, formats=("json", "kv", "yaml"))
    second = _dataset(seed=23, formats=("json", "kv", "yaml"))
    other = _dataset(seed=24, formats=("json", "kv", "yaml"))

    assert first["train"] == second["train"]
    assert first["eval"] == second["eval"]
    assert first["summary"] == second["summary"]
    assert _row_signature(first["train"]) != _row_signature(other["train"])


def test_shuffle_is_deterministic_and_preserves_rows():
    plain = _dataset(rows=50, eval_rows=10, shuffle=False)
    shuffled_a = _dataset(rows=50, eval_rows=10, shuffle=True)
    shuffled_b = _dataset(rows=50, eval_rows=10, shuffle=True)

    assert shuffled_a["train"] == shuffled_b["train"]
    assert shuffled_a["eval"] == shuffled_b["eval"]
    assert _row_signature(shuffled_a["train"]) == _row_signature(plain["train"])
    assert _row_signature(shuffled_a["eval"]) == _row_signature(plain["eval"])
    assert [row["user"] for row in shuffled_a["train"]] != [
        row["user"] for row in plain["train"]]


@pytest.mark.parametrize("fmt", queue.FORMATS)
def test_every_format_parses_and_round_trips(fmt):
    dataset = _dataset(rows=12, eval_rows=4, formats=(fmt,))
    rows = dataset["train"] + dataset["eval"]
    assert rows
    for row in rows:
        parsed = queue.parse_answer(fmt, row["assistant"])
        assert set(parsed) == set(row["keys"])
        assert [str(parsed[key]) for key in row["keys"]] == [
            str(value) for value in row["values"]]
        assert queue.validate_answer(fmt, row["assistant"], row["pairs"])
        assert row["format"] == fmt

    if fmt == "json":
        assert all(row["assistant"].startswith("{")
                   and row["assistant"].endswith("}") for row in rows)
        numeric = [value for row in rows for value in row["values"]
                   if isinstance(value, int)]
        assert numeric
        for row in rows:
            parsed = queue.parse_answer(fmt, row["assistant"])
            for key, value in row["pairs"]:
                if isinstance(value, int):
                    assert isinstance(parsed[key], int)
                    assert parsed[key] == value
    elif fmt == "kv":
        assert all("\n" not in row["assistant"] for row in rows)
    else:
        for row in rows:
            assert row["assistant"].count("\n") == len(row["keys"]) - 1
            assert "\n".join(f"{key}: {value}"
                             for key, value in row["pairs"]) == (
                                 row["assistant"])


def test_parsers_accept_valid_answers_and_reject_mismatches():
    pairs = [["k_ab12cd34", "Ada"], ["field_9x2k1q", 7]]
    answers = {
        "json": '{"k_ab12cd34":"Ada","field_9x2k1q":7}',
        "kv": "k_ab12cd34=Ada;field_9x2k1q=7",
        "yaml": "k_ab12cd34: Ada\nfield_9x2k1q: 7",
    }
    for fmt, answer in answers.items():
        assert queue.validate_answer(fmt, answer, pairs)
        assert queue.parse_answer(fmt, answer) == {
            "k_ab12cd34": "Ada",
            "field_9x2k1q": "7" if fmt != "json" else 7,
        }
        assert not queue.validate_answer(
            fmt, answer.replace("Ada", "Bora"), pairs)
        assert not queue.validate_answer(
            fmt, answer.replace("field_9x2k1q", "field_wrong"), pairs)

    assert queue.validate_answer(
        "json", '```json\n{"k_ab12cd34":"Ada","field_9x2k1q":7}\n```',
        pairs)
    assert not queue.validate_answer(
        "json", '{"k_ab12cd34":"Ada","field_9x2k1q":7,"extra":1}', pairs)
    assert not queue.validate_answer(
        "json", 'note: {"k_ab12cd34":"Ada","field_9x2k1q":7}', pairs)
    assert not queue.validate_answer(
        "json", '{"k_ab12cd34":"Ada","k_ab12cd34":"Ada","field_9x2k1q":7}',
        pairs)
    assert not queue.validate_answer(
        "kv", "k_ab12cd34=Ada", pairs)
    assert not queue.validate_answer(
        "kv", "k_ab12cd34=Ada;field_9x2k1q=7;extra=1", pairs)
    assert not queue.validate_answer(
        "kv", "k_ab12cd34=Ada\nfield_9x2k1q=7", pairs)
    assert not queue.validate_answer(
        "yaml", "k_ab12cd34: Ada", pairs)
    assert not queue.validate_answer(
        "yaml", "k_ab12cd34: Ada\nfield_9x2k1q: 7\nextra: 1", pairs)
    assert not queue.validate_answer(
        "yaml", '{"k_ab12cd34": "Ada", "field_9x2k1q": 7}', pairs)
    with pytest.raises(ValueError):
        queue.parse_answer("csv", "k,v")
    with pytest.raises(ValueError):
        queue.parse_answer("json", "")


def test_tampered_rows_fail_row_validation():
    row = _dataset(rows=1, eval_rows=0)["train"][0]
    assert queue.validate_row(row)
    broken = json.loads(json.dumps(row))
    broken["assistant"] = queue.render_answer(
        broken["format"],
        [[broken["keys"][0], "definitely-not-the-value"]]
        + broken["pairs"][1:])
    assert not queue.validate_row(broken)
    broken = json.loads(json.dumps(row))
    broken["keys"] = broken["keys"] + [broken["keys"][0]]
    assert not queue.validate_row(broken)


def test_pair_counts_and_keys_per_row_bounds():
    dataset = _dataset(rows=40, eval_rows=10)
    counts = [len(row["keys"]) for row in dataset["train"] + dataset["eval"]]
    assert min(counts) == 2
    assert max(counts) == 4
    assert dataset["summary"]["pairs_per_row"]["min"] == 2
    assert dataset["summary"]["pairs_per_row"]["max"] == 4

    four = _dataset(rows=8, eval_rows=2, keys_per_row=4)
    assert {len(row["keys"]) for row in four["train"] + four["eval"]} == {4}
    three = _dataset(rows=12, eval_rows=2, keys_per_row=3)
    assert {len(row["keys"])
            for row in three["train"] + three["eval"]} <= {3, 4}
    with pytest.raises(ValueError):
        queue.build_dataset(rows=1, eval_rows=0, keys_per_row=1)
    with pytest.raises(ValueError):
        queue.build_dataset(rows=1, eval_rows=0,
                            formats=("json", "csv"))
    with pytest.raises(ValueError):
        queue.build_dataset(rows=-1, eval_rows=0)


def test_format_counts_cover_every_requested_format():
    dataset = _dataset(rows=10, eval_rows=4,
                       formats=("json", "kv", "yaml"))
    summary = dataset["summary"]
    assert sum(summary["format_counts"]["train"].values()) == 10
    assert sum(summary["format_counts"]["eval"].values()) == 4
    assert summary["format_counts"]["total"] == {
        "json": 6, "kv": 4, "yaml": 4}
    assert set(summary["format_counts"]["train"]) == set(queue.FORMATS)


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
        rows=3, eval_rows=1, formats=("json",), seed=11)
    assert calls["count"] >= 2
    assert dataset["summary"]["rejected_attempts"] == 1
    assert all(queue.validate_row(row)
               for row in dataset["train"] + dataset["eval"])


def test_cli_writes_train_eval_and_manifest(tmp_path, capsys):
    out_dir = tmp_path / "queue"
    holdout = tmp_path / "holdout" / "eval_rows.jsonl"
    args = [
        "--out-dir", str(out_dir),
        "--rows", "12", "--eval-rows", "4",
        "--formats", "json,kv,yaml", "--keys-per-row", "2",
        "--seed", "7", "--holdout", str(holdout), "--shuffle",
    ]
    assert queue.main(args) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["train_rows"] == 12
    assert summary["eval_rows"] == 4
    assert summary["key_overlap"] == 0
    assert summary["value_overlap"] == 0
    assert summary["paths"]["eval"] == str(holdout)
    assert Path(summary["paths"]["train"]) == out_dir / "train.jsonl"
    assert Path(summary["paths"]["manifest"]) == out_dir / "manifest.json"
    assert holdout.exists()
    assert (out_dir / "manifest.json").exists()

    train = _read_jsonl(out_dir / "train.jsonl")
    eval_rows = _read_jsonl(holdout)
    assert len(train) == 12 and len(eval_rows) == 4
    assert {row["partition"] for row in train} == {"train"}
    assert {row["partition"] for row in eval_rows} == {"eval"}
    assert sum(1 for row in train if row["format"] == "json") == 4
    assert sum(1 for row in train if row["format"] == "kv") == 4
    assert sum(1 for row in train if row["format"] == "yaml") == 4
    manifest = json.loads((out_dir / "manifest.json").read_text(
        encoding="utf-8"))
    assert manifest["train_rows"] == 12
    assert manifest["format_counts"] == summary["format_counts"]

    second_out = tmp_path / "queue_again"
    second_holdout = tmp_path / "holdout_again" / "eval_rows.jsonl"
    second_args = list(args)
    second_args[second_args.index(str(out_dir))] = str(second_out)
    second_args[second_args.index(str(holdout))] = str(second_holdout)
    capsys.readouterr()
    assert queue.main(second_args) == 0
    capsys.readouterr()
    assert (out_dir / "train.jsonl").read_bytes() == (
        second_out / "train.jsonl").read_bytes()
    assert holdout.read_bytes() == second_holdout.read_bytes()


def test_write_dataset_defaults_eval_next_to_train(tmp_path):
    dataset = _dataset(rows=6, eval_rows=2)
    summary = queue.write_dataset(dataset, tmp_path / "default")
    assert summary["paths"]["eval"] == str(
        (tmp_path / "default" / "eval.jsonl"))
    assert (tmp_path / "default" / "eval.jsonl").exists()
    assert len(_read_jsonl(tmp_path / "default" / "eval.jsonl")) == 2


def test_generator_has_no_model_dependency():
    source = Path(queue.__file__).read_text(encoding="utf-8")
    assert "import torch" not in source
    assert "from transformers" not in source
    assert "import transformers" not in source
