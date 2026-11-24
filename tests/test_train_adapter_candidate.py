import json

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit

from atomzero.lora import apply_lora, save_adapter
from atomzero.transformer import TransformerLM
from research import train_adapter_candidate as trainer
from research.train_adapter_candidate import (
    _token_weights,
    group_weighted_loss,
    load_init_adapter,
)

VOCAB = ["<|user|>", "<|assistant|>", "<|end|>", "<|unk|>",
         "hello", "hi", "bye", "alpha", "beta", "gamma", "delta"]


class _Tokenizer:
    def decode(self, ids):
        return {1: "hi", 2: "{", 3: "Ada", 4: "}"}[int(ids[0])]


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


def test_token_weights_only_scale_scored_content_tokens():
    y = torch.tensor([[1, 2, 3, 4, -100]])
    weights = _token_weights(_Tokenizer(), y, 5.0)
    assert weights.tolist() == [[5.0, 1.0, 5.0, 1.0, 0.0]]


def test_token_weights_default_is_plain_cross_entropy():
    y = torch.tensor([[1, 2, 3, -100]])
    weights = _token_weights(_Tokenizer(), y, 1.0)
    assert weights.tolist() == [[1.0, 1.0, 1.0, 0.0]]


def test_group_weighted_loss_normalizes_by_total_token_weight():
    terms = [(torch.tensor(2.0), torch.tensor(1.0)),
             (torch.tensor(9.0), torch.tensor(3.0))]
    loss = group_weighted_loss(terms)
    # 11 weighted loss over 4 tokens, not the mean of row means (2.5).
    assert loss.item() == pytest.approx(11.0 / 4.0)
    assert loss.item() != pytest.approx((2.0 + 3.0) / 2.0)


def test_group_weighted_loss_matches_legacy_single_row():
    token_loss = torch.tensor([1.0, 2.0, 3.0])
    weights = torch.tensor([1.0, 0.0, 2.0])
    expected = (token_loss * weights).sum() / weights.sum().clamp_min(1e-8)
    actual = group_weighted_loss([
        ((token_loss * weights).sum(), weights.sum())])
    assert torch.equal(actual, expected)


def test_group_weighted_loss_gradient_uses_combined_denominator():
    first = torch.tensor(2.0, requires_grad=True)
    second = torch.tensor(6.0, requires_grad=True)
    loss = group_weighted_loss([
        (first, torch.tensor(1.0)),
        (second, torch.tensor(3.0)),
    ])
    loss.backward()
    assert first.grad.item() == pytest.approx(0.25)
    assert second.grad.item() == pytest.approx(0.25)


def test_group_weighted_loss_rejects_empty_group():
    with pytest.raises(ValueError):
        group_weighted_loss([])


def test_batch_size_one_steps_once_per_seen_row(tmp_path, monkeypatch):
    base = _tiny_base(tmp_path)
    original = trainer._encode
    calls = {"count": 0}

    def skip_second_row(tok, row, max_seq, device):
        calls["count"] += 1
        if calls["count"] == 2:
            return None
        return original(tok, row, max_seq, device)

    monkeypatch.setattr(trainer, "_encode", skip_second_row)
    out = tmp_path / "adapter"
    result = trainer.train(
        str(base), _rows("alpha", 5), out, steps=5, lr=1e-2, rank=2,
        alpha=4, max_seq=32, device="cpu", seed=7)
    assert result["rows_seen"] == 4
    assert result["optimizer_steps"] == 4
    assert result["batch_size"] == 1
    assert result["lr_scale"] == "none"
    assert result["effective_lr"] == pytest.approx(1e-2)
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert config["rows_seen"] == 4
    assert config["optimizer_steps"] == 4
    assert config["batch_size"] == 1


def test_batch_size_four_steps_once_per_group(tmp_path):
    base = _tiny_base(tmp_path)
    out = tmp_path / "adapter"
    result = trainer.train(
        str(base), _rows("alpha", 4), out, steps=10, lr=1e-2, rank=2,
        alpha=4, max_seq=32, device="cpu", seed=7, batch_size=4)
    assert result["rows_seen"] == 10
    assert result["optimizer_steps"] == 3  # ceil(10 / 4), partial group flushed
    assert result["batch_size"] == 4
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert config["batch_size"] == 4
    assert config["optimizer_steps"] == 3
    assert config["rows_seen"] == 10


def test_skipped_rows_do_not_consume_batch_slots(tmp_path, monkeypatch):
    base = _tiny_base(tmp_path)
    original = trainer._encode
    calls = {"count": 0}

    def skip_third_row(tok, row, max_seq, device):
        calls["count"] += 1
        if calls["count"] == 3:
            return None
        return original(tok, row, max_seq, device)

    monkeypatch.setattr(trainer, "_encode", skip_third_row)
    result = trainer.train(
        str(base), _rows("alpha", 3), tmp_path / "adapter", steps=9,
        lr=1e-2, rank=2, alpha=4, max_seq=32, device="cpu", seed=7,
        batch_size=4)
    assert result["rows_seen"] == 8
    assert result["optimizer_steps"] == 2


def test_lr_scale_linear_is_explicit(tmp_path):
    base = _tiny_base(tmp_path)
    out = tmp_path / "adapter"
    result = trainer.train(
        str(base), _rows("alpha", 2), out, steps=2, lr=1e-2, rank=2,
        alpha=4, max_seq=32, device="cpu", seed=7,
        batch_size=2, lr_scale="linear")
    assert result["effective_lr"] == pytest.approx(2e-2)
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert config["lr"] == pytest.approx(1e-2)
    assert config["lr_scale"] == "linear"
    assert config["effective_lr"] == pytest.approx(2e-2)


@pytest.mark.parametrize("batch_size", [0, -1, 1.5, True])
def test_train_rejects_invalid_batch_size(tmp_path, batch_size):
    with pytest.raises(ValueError):
        trainer.train(
            str(tmp_path / "base"), [], tmp_path / "adapter", steps=1,
            lr=1e-2, rank=2, alpha=4, max_seq=8, device="cpu",
            batch_size=batch_size)


@pytest.mark.parametrize("value", ["0", "-2", "1.5", "abc"])
def test_cli_rejects_invalid_batch_size(value):
    with pytest.raises(SystemExit):
        trainer.main([
            "--base", "base", "--train", "train.jsonl",
            "--eval", "eval.jsonl", "--out", "adapter",
            "--batch-size", value,
        ])


class _LinearStack(torch.nn.Module):
    """Tiny non-transformer module with the names ``apply_lora`` targets."""

    def __init__(self):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(8, 6), torch.nn.ReLU(),
            torch.nn.Linear(6, 8))
        self.attn = torch.nn.Module()
        self.attn.out_proj = torch.nn.Linear(8, 8)


def _randomize_lora(model):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith((".A", ".B")):
                values = torch.arange(
                    param.numel(), dtype=torch.float32).reshape(param.shape)
                param.copy_(values / 100.0 + 0.01)


def _saved_linear_adapter(tmp_path, *, r_attn=2, alpha_attn=4, r_mlp=3,
                          alpha_mlp=6):
    model = _LinearStack()
    apply_lora(model, r_attn=r_attn, r_mlp=r_mlp,
               alpha_attn=alpha_attn, alpha_mlp=alpha_mlp)
    _randomize_lora(model)
    adapter = tmp_path / "init_adapter"
    save_adapter(model, adapter, r_attn=r_attn, alpha_attn=alpha_attn,
                 r_mlp=r_mlp, alpha_mlp=alpha_mlp)
    return model, adapter


def test_load_init_adapter_roundtrips_ab_tensors(tmp_path):
    source, adapter = _saved_linear_adapter(tmp_path)
    fresh = _LinearStack()
    config = load_init_adapter(fresh, adapter)
    assert config["r_attn"] == 2
    assert config["alpha_attn"] == 4
    assert config["r_mlp"] == 3
    assert config["alpha_mlp"] == 6

    original = {name: param for name, param in source.named_parameters()
                if name.endswith((".A", ".B"))}
    reloaded = {name: param for name, param in fresh.named_parameters()
                if name.endswith((".A", ".B"))}
    assert set(original) == set(reloaded)
    assert set(original) == {
        "attn.out_proj.A", "attn.out_proj.B", "mlp.0.A", "mlp.0.B",
        "mlp.2.A", "mlp.2.B"}
    for name, param in original.items():
        assert torch.equal(param, reloaded[name])


@pytest.mark.parametrize("override", [
    {"rank": 4},
    {"alpha": 12},
    {"rank_attn": 1},
    {"alpha_attn": 8},
])
def test_load_init_adapter_rejects_geometry_mismatch(tmp_path, override):
    _, adapter = _saved_linear_adapter(tmp_path)
    with pytest.raises(ValueError, match="geometry mismatch"):
        load_init_adapter(_LinearStack(), adapter, **override)


def test_load_init_adapter_accepts_matching_geometry(tmp_path):
    _, adapter = _saved_linear_adapter(tmp_path)
    fresh = _LinearStack()
    load_init_adapter(fresh, adapter, rank=3, alpha=6,
                      rank_attn=2, alpha_attn=4)
    names = {name for name, _ in fresh.named_parameters()}
    assert {"attn.out_proj.A", "mlp.0.A", "mlp.2.B"} <= names


def test_load_init_adapter_rejects_missing_adapter(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_init_adapter(_LinearStack(), tmp_path / "missing_adapter")


def test_load_init_adapter_rejects_missing_tensors(tmp_path):
    _, adapter = _saved_linear_adapter(tmp_path)
    checkpoint = torch.load(adapter / "adapter.pt", map_location="cpu",
                            weights_only=False)
    missing = next(iter(checkpoint["lora"]))
    del checkpoint["lora"][missing]
    torch.save(checkpoint, adapter / "adapter.pt")
    with pytest.raises(ValueError, match="missing="):
        load_init_adapter(_LinearStack(), adapter)


def test_load_init_adapter_reads_config_from_checkpoint(tmp_path):
    _, adapter = _saved_linear_adapter(tmp_path)
    (adapter / "config.json").unlink()
    fresh = _LinearStack()
    config = load_init_adapter(fresh, adapter)
    assert config["r_mlp"] == 3
    assert config["alpha_mlp"] == 6
    assert "mlp.0.A" in dict(fresh.named_parameters())


def _saved_transformer_adapter(tmp_path, base):
    checkpoint = torch.load(base / "model.pt", map_location="cpu",
                            weights_only=False)
    config = checkpoint["config"]
    model = TransformerLM(config["vocab"], config["dim"], config["layers"],
                          config["heads"], config["ff"])
    model.load_state_dict(checkpoint["state"])
    apply_lora(model, r_attn=2, r_mlp=2, alpha_attn=4, alpha_mlp=4)
    _randomize_lora(model)
    adapter = tmp_path / "init_adapter"
    save_adapter(model, adapter, r_attn=2, alpha_attn=4, r_mlp=2,
                 alpha_mlp=4)
    return adapter


def test_cli_resumes_from_init_adapter_and_records_it(tmp_path):
    base = _tiny_base(tmp_path)
    init_adapter = _saved_transformer_adapter(tmp_path, base)
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    _write_jsonl(train_path, _rows("alpha", 3))
    _write_jsonl(eval_path, _rows("gamma", 2))
    result_path = tmp_path / "result.json"
    out = tmp_path / "adapter"
    code = trainer.main([
        "--base", str(base),
        "--train", str(train_path),
        "--eval", str(eval_path),
        "--out", str(out),
        "--init-adapter", str(init_adapter),
        "--steps", "2", "--lr", "0.01", "--max-seq", "32", "--seed", "3",
        "--result", str(result_path),
        "--comparison", str(tmp_path / "comparison.json"),
    ])
    assert code == 0
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["init_adapter"] == str(init_adapter)
    assert result["rank_mlp"] == 2
    assert result["alpha_mlp"] == 4
    assert result["rank_attn"] == 2
    assert result["alpha_attn"] == 4
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert config["init_adapter"] == str(init_adapter)
    assert config["r_mlp"] == 2
    assert config["alpha_mlp"] == 4
    assert config["r_attn"] == 2
    assert config["alpha_attn"] == 4


def test_cli_rejects_mismatched_init_adapter_geometry(tmp_path):
    base = _tiny_base(tmp_path)
    init_adapter = _saved_transformer_adapter(tmp_path, base)
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    _write_jsonl(train_path, _rows("alpha", 2))
    _write_jsonl(eval_path, _rows("gamma", 1))
    with pytest.raises(SystemExit):
        trainer.main([
            "--base", str(base),
            "--train", str(train_path),
            "--eval", str(eval_path),
            "--out", str(tmp_path / "adapter"),
            "--init-adapter", str(init_adapter),
            "--rank", "4",
        ])


def test_cli_records_gradient_accumulation_flags(tmp_path):
    base = _tiny_base(tmp_path)
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    _write_jsonl(train_path, _rows("alpha", 4))
    _write_jsonl(eval_path, _rows("gamma", 2))
    result_path = tmp_path / "result.json"
    out = tmp_path / "adapter"
    code = trainer.main([
        "--base", str(base),
        "--train", str(train_path),
        "--eval", str(eval_path),
        "--out", str(out),
        "--steps", "8", "--lr", "0.01", "--rank", "2", "--alpha", "4",
        "--max-seq", "32", "--seed", "3",
        "--batch-size", "4", "--lr-scale", "linear",
        "--result", str(result_path),
        "--comparison", str(tmp_path / "comparison.json"),
    ])
    assert code == 0
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["batch_size"] == 4
    assert result["rows_seen"] == 8
    assert result["optimizer_steps"] == 2
    assert result["lr_scale"] == "linear"
    assert result["effective_lr"] == pytest.approx(0.04)
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert config["batch_size"] == 4
    assert config["optimizer_steps"] == 2
    assert config["rows_seen"] == 8
