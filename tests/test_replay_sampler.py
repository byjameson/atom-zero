"""Tests for deterministic replay sampling and its trainer integration."""
import copy
import json
import math

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit

from atomzero.transformer import TransformerLM
from research.replay_sampler import sample_steps, summarize_mix
from research.train_adapter_candidate import main, train

VOCAB = ["<|user|>", "<|assistant|>", "<|end|>", "<|unk|>",
         "hello", "hi", "bye", "alpha", "beta", "gamma", "delta"]


def _rows(prefix, count):
    return [{"user": f"{prefix} {index}", "assistant": "hi"}
            for index in range(count)]


def _tiny_base(tmp_path):
    tokenizer = Tokenizer(WordLevel(
        vocab={word: index for index, word in enumerate(VOCAB)},
        unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    base = tmp_path / "base"
    base.mkdir()
    tokenizer.save(str(base / "chat_en_bpe8k.json"))
    config = {"vocab": len(VOCAB), "dim": 16, "layers": 1, "heads": 2,
              "ff": 32, "seq": 32, "arch": "transformer"}
    torch.manual_seed(0)
    model = TransformerLM(config["vocab"], config["dim"], config["layers"],
                          config["heads"], config["ff"])
    torch.save({"config": config, "state": model.state_dict()},
               base / "model.pt")
    return base


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _is_from(row, rows):
    return any(row is candidate for candidate in rows)


def test_sample_steps_is_deterministic_per_seed():
    primary = _rows("alpha", 5)
    replay = _rows("beta", 4)
    first = sample_steps(primary, replay, 40, replay_weight=0.25, seed=13)
    second = sample_steps(primary, replay, 40, replay_weight=0.25, seed=13)
    other = sample_steps(primary, replay, 40, replay_weight=0.25, seed=14)
    assert first == second
    assert [id(row) for row in first] == [id(row) for row in second]
    assert first != other
    assert len(first) == 40


def test_zero_weight_and_empty_replay_use_primary_order():
    primary = _rows("alpha", 5)
    replay = _rows("beta", 3)
    expected = [primary[index % len(primary)] for index in range(7)]
    assert sample_steps(primary, replay, 7, replay_weight=0.0) == expected
    assert sample_steps(primary, [], 4, replay_weight=0.5) == primary[:4]
    assert sample_steps(primary, None, 5) == primary


def test_replay_weight_bounds_and_exact_mix():
    primary = _rows("alpha", 4)
    replay = _rows("beta", 6)
    all_replay = sample_steps(primary, replay, 20, replay_weight=1.0)
    assert all(_is_from(row, replay) for row in all_replay)
    assert summarize_mix(all_replay, primary, replay) == {
        "primary": 0, "replay": 20}
    mixed = sample_steps(primary, replay, 40, replay_weight=0.25)
    assert summarize_mix(mixed, primary, replay) == {
        "primary": 30, "replay": 10}
    assert summarize_mix([], primary, replay) == {"primary": 0, "replay": 0}
    assert sample_steps(primary, replay, 0) == []


@pytest.mark.parametrize("weight", [-0.01, 1.01, float("nan"), float("inf")])
def test_invalid_weights_raise(weight):
    with pytest.raises(ValueError):
        sample_steps(_rows("alpha", 2), _rows("beta", 2), 4,
                     replay_weight=weight)


@pytest.mark.parametrize("weight", ["0.5", None, True, [0.5]])
def test_invalid_weight_types_raise(weight):
    with pytest.raises(TypeError):
        sample_steps(_rows("alpha", 2), _rows("beta", 2), 4,
                     replay_weight=weight)


@pytest.mark.parametrize("steps", [-1])
def test_negative_steps_raise(steps):
    with pytest.raises(ValueError):
        sample_steps(_rows("alpha", 2), [], steps)


@pytest.mark.parametrize("steps", [1.5, True, "3"])
def test_invalid_step_types_raise(steps):
    with pytest.raises(TypeError):
        sample_steps(_rows("alpha", 2), [], steps)


def test_primary_and_replay_types_are_validated():
    with pytest.raises(ValueError):
        sample_steps([], [], 3)
    with pytest.raises(TypeError):
        sample_steps("not-rows", [], 3)
    with pytest.raises(TypeError):
        sample_steps([{"user": "q", "assistant": "a"}], "not-rows", 3)
    with pytest.raises(TypeError):
        sample_steps([1], [], 3)
    with pytest.raises(TypeError):
        sample_steps(_rows("alpha", 2), [1], 3)
    with pytest.raises(TypeError):
        sample_steps(_rows("alpha", 2), [], 3, seed="13")


def test_sample_steps_does_not_mutate_inputs():
    primary = _rows("alpha", 5)
    replay = _rows("beta", 3)
    primary_before = copy.deepcopy(primary)
    replay_before = copy.deepcopy(replay)
    primary_id = id(primary)
    sampled = sample_steps(primary, replay, 25, replay_weight=0.4, seed=5)
    assert primary == primary_before
    assert replay == replay_before
    assert id(primary) == primary_id
    sampled.append({"user": "extra", "assistant": "hi"})
    assert len(primary) == 5
    assert len(replay) == 3


def test_summarize_mix_rejects_unknown_rows_and_prefers_primary():
    shared = {"user": "hello", "assistant": "hi"}
    assert summarize_mix([shared], [shared], [shared]) == {
        "primary": 1, "replay": 0}
    with pytest.raises(ValueError):
        summarize_mix([{"user": "new", "assistant": "hi"}],
                      [shared], [shared])


def test_trainer_uses_replay_and_records_the_mix(tmp_path):
    base = _tiny_base(tmp_path)
    primary = _rows("alpha", 4)
    replay = _rows("beta", 3)
    out = tmp_path / "adapter"
    result = train(
        str(base), primary, out, steps=5, lr=1e-2, rank=2, alpha=4,
        max_seq=32, device="cpu", seed=7, replay_rows=replay,
        replay_weight=0.4)
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert config["train_examples"] == 4
    assert config["replay_examples"] == 3
    assert config["replay_weight"] == pytest.approx(0.4)
    assert result["replay_examples"] == 3
    assert result["replay_weight"] == pytest.approx(0.4)
    assert result["train_mix"] == {"primary": 3, "replay": 2}
    assert math.isfinite(result["train_loss"])
    assert (out / "adapter.pt").is_file()


def test_trainer_without_replay_keeps_primary_only(tmp_path):
    base = _tiny_base(tmp_path)
    primary = _rows("alpha", 4)
    out = tmp_path / "adapter"
    result = train(
        str(base), primary, out, steps=3, lr=1e-2, rank=2, alpha=4,
        max_seq=32, device="cpu", seed=7)
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert config["replay_examples"] == 0
    assert config["replay_weight"] == pytest.approx(0.25)
    assert result["train_mix"] == {"primary": 3, "replay": 0}
    assert math.isfinite(result["train_loss"])


def test_cli_loads_replay_and_writes_result(tmp_path):
    base = _tiny_base(tmp_path)
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    replay_path = tmp_path / "replay.jsonl"
    _write_jsonl(train_path, _rows("alpha", 4))
    _write_jsonl(eval_path, _rows("gamma", 2))
    _write_jsonl(replay_path, _rows("beta", 3))
    result_path = tmp_path / "result.json"
    code = main([
        "--base", str(base),
        "--train", str(train_path),
        "--eval", str(eval_path),
        "--out", str(tmp_path / "adapter"),
        "--steps", "4", "--lr", "0.01", "--rank", "2", "--alpha", "4",
        "--max-seq", "32", "--seed", "3",
        "--replay-jsonl", str(replay_path), "--replay-weight", "0.5",
        "--result", str(result_path),
        "--comparison", str(tmp_path / "comparison.json"),
    ])
    assert code == 0
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["replay_examples"] == 3
    assert result["replay_weight"] == pytest.approx(0.5)
    assert result["train_mix"] == {"primary": 2, "replay": 2}
    assert result["weights_changed"] is False
