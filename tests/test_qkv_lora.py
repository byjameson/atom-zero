"""Fused QKV attention LoRA: forward, gradients, roundtrip and validation."""
from __future__ import annotations

import json

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit

from atomzero.lora import (
    LoRALinear,
    LoRAMultiheadAttention,
    apply_adapter,
    apply_lora,
    apply_qkv_lora,
    count_lora_params,
    qkv_lora_state_dict,
    save_adapter,
    set_qkv_lora_state_dict,
)
from atomzero.transformer import TransformerLM
from research.train_adapter_candidate import main

VOCAB = ["<|user|>", "<|assistant|>", "<|end|>", "hello", "hi", "[UNK]"]
CONFIG = {"vocab": len(VOCAB), "dim": 16, "layers": 2, "heads": 2,
          "ff": 32, "seq": 32, "arch": "transformer"}
RANK_QKV = 2
ALPHA_QKV = 4


def _build_model(seed: int = 0) -> TransformerLM:
    torch.manual_seed(seed)
    return TransformerLM(CONFIG["vocab"], CONFIG["dim"], CONFIG["layers"],
                         CONFIG["heads"], CONFIG["ff"])


def _fixed_input(model: TransformerLM, length: int = 6) -> torch.Tensor:
    generator = torch.Generator().manual_seed(7)
    return torch.randint(0, model.vocab, (1, length), generator=generator)


def _randomize_lora(model: torch.nn.Module, seed: int = 11) -> None:
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora" in name or name.endswith((".A", ".B")):
                param.copy_(torch.randn(param.shape, generator=generator)
                            * 0.05)


def _tiny_base(tmp_path, seed: int = 0):
    model = _build_model(seed)
    base = tmp_path / "base"
    base.mkdir()
    tokenizer = Tokenizer(WordLevel(
        vocab={word: index for index, word in enumerate(VOCAB)},
        unk_token="[UNK]"))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    tokenizer.save(str(base / "chat_en_bpe8k.json"))
    torch.save({"config": CONFIG, "state": model.state_dict()},
               base / "model.pt")
    return base, model


def test_qkv_lora_forward_starts_as_identity_and_shapes():
    model = _build_model(seed=1)
    x = _fixed_input(model)
    with torch.no_grad():
        expected = model(x).clone()

    apply_lora(model, r_attn=0, r_mlp=0,
               r_qkv=RANK_QKV, alpha_qkv=ALPHA_QKV)

    attn = model.blocks[0].attn
    assert isinstance(attn, LoRAMultiheadAttention)
    assert attn.lora_A.shape == (RANK_QKV, CONFIG["dim"])
    assert attn.lora_B.shape == (3 * CONFIG["dim"], RANK_QKV)
    assert attn.in_proj_weight.requires_grad is False
    assert count_lora_params(model) == (
        CONFIG["layers"] * 4 * RANK_QKV * CONFIG["dim"])

    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (1, x.size(1), model.vocab)
    assert torch.isfinite(logits).all()
    # B is zero-initialized, so a fresh adapter is bitwise the base model.
    assert torch.equal(expected, logits)


def test_qkv_lora_a_b_receive_gradients_and_base_stays_frozen():
    model = _build_model(seed=2)
    apply_lora(model, r_attn=0, r_mlp=0,
               r_qkv=RANK_QKV, alpha_qkv=ALPHA_QKV)
    _randomize_lora(model)

    base_weight = {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if name.endswith("in_proj_weight")
    }
    logits = model(_fixed_input(model))
    logits.sum().backward()

    for block in model.blocks:
        attn = block.attn
        assert isinstance(attn, LoRAMultiheadAttention)
        assert attn.lora_A.grad is not None
        assert attn.lora_B.grad is not None
        assert attn.lora_A.grad.abs().sum() > 0
        assert attn.lora_B.grad.abs().sum() > 0
        assert attn.base.in_proj_weight.grad is None
        assert attn.base.in_proj_weight.requires_grad is False

    optimizer = torch.optim.SGD(
        [param for param in model.parameters() if param.requires_grad], lr=0.1)
    optimizer.step()
    for name, param in model.named_parameters():
        if name in base_weight:
            assert torch.equal(param, base_weight[name])


def test_qkv_lora_save_apply_adapter_roundtrip(tmp_path):
    model = _build_model(seed=3)
    base_state = {key: value.clone()
                  for key, value in model.state_dict().items()}
    apply_lora(model, r_attn=0, r_mlp=0,
               r_qkv=RANK_QKV, alpha_qkv=ALPHA_QKV)
    _randomize_lora(model, seed=5)

    adapter = tmp_path / "adapter"
    save_adapter(model, adapter)
    config = json.loads((adapter / "config.json").read_text(encoding="utf-8"))
    assert config["r_qkv"] == RANK_QKV
    assert config["alpha_qkv"] == ALPHA_QKV
    assert config["qkv_bias"] is False

    fresh = _build_model(seed=99)
    fresh.load_state_dict(base_state)
    apply_adapter(fresh, adapter)
    assert isinstance(fresh.blocks[0].attn, LoRAMultiheadAttention)
    assert not isinstance(fresh.blocks[0].attn.out_proj, LoRALinear)
    assert count_lora_params(fresh) == count_lora_params(model)

    saved = qkv_lora_state_dict(model)
    reloaded = qkv_lora_state_dict(fresh)
    assert set(saved) == set(reloaded)
    for name in saved:
        assert torch.equal(saved[name], reloaded[name])
    set_qkv_lora_state_dict(fresh, saved)

    x = _fixed_input(model)
    model.eval()
    fresh.eval()
    with torch.no_grad():
        expected = model(x)
        actual = fresh(x)
    assert torch.equal(expected, actual)
    assert torch.isfinite(actual).all()


def test_optional_qkv_bias_adapter_roundtrip(tmp_path):
    model = _build_model(seed=4)
    base_state = {key: value.clone()
                  for key, value in model.state_dict().items()}
    apply_qkv_lora(model, RANK_QKV, ALPHA_QKV, adapt_bias=True)
    _randomize_lora(model, seed=9)
    assert count_lora_params(model) == (
        CONFIG["layers"] * (4 * RANK_QKV * CONFIG["dim"]
                            + RANK_QKV * (3 * CONFIG["dim"] + 1)))

    adapter = tmp_path / "adapter_bias"
    save_adapter(model, adapter)
    config = json.loads((adapter / "config.json").read_text(encoding="utf-8"))
    assert config["qkv_bias"] is True

    fresh = _build_model(seed=98)
    fresh.load_state_dict(base_state)
    apply_adapter(fresh, adapter)
    assert count_lora_params(fresh) == count_lora_params(model)
    x = _fixed_input(model)
    model.eval()
    fresh.eval()
    with torch.no_grad():
        assert torch.equal(model(x), fresh(x))


def test_qkv_rank_and_alpha_validation():
    model = _build_model()
    with pytest.raises(ValueError, match="ranks"):
        apply_lora(model, r_attn=0, r_mlp=0, r_qkv=-1)
    with pytest.raises(ValueError, match="qkv LoRA alpha"):
        apply_lora(model, r_attn=0, r_mlp=0, r_qkv=2, alpha_qkv=0)

    with pytest.raises(ValueError, match="QKV LoRA rank"):
        apply_qkv_lora(model, -1, 4)
    with pytest.raises(ValueError, match="QKV LoRA alpha"):
        apply_qkv_lora(model, 2, 0)

    with pytest.raises(ValueError, match="no nn.MultiheadAttention"):
        apply_qkv_lora(torch.nn.Linear(4, 4), 2, 4)


def test_cli_trains_qkv_adapter_and_records_flags(tmp_path):
    base, model = _tiny_base(tmp_path, seed=6)
    base_state = {key: value.clone()
                  for key, value in model.state_dict().items()}
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    train_path.write_text(
        json.dumps({"user": "hello", "assistant": "hi"}) + "\n",
        encoding="utf-8")
    eval_path.write_text(
        json.dumps({"user": "hi", "assistant": "hello"}) + "\n",
        encoding="utf-8")
    out = tmp_path / "adapter"
    result_path = tmp_path / "result.json"

    code = main([
        "--base", str(base),
        "--train", str(train_path),
        "--eval", str(eval_path),
        "--out", str(out),
        "--steps", "2", "--lr", "0.01",
        "--rank", "0", "--alpha", "0",
        "--qkv-rank", str(RANK_QKV), "--qkv-alpha", str(ALPHA_QKV),
        "--max-seq", "16", "--seed", "3",
        "--result", str(result_path),
        "--comparison", str(tmp_path / "comparison.json"),
    ])
    assert code == 0
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["rank_qkv"] == RANK_QKV
    assert result["alpha_qkv"] == ALPHA_QKV
    assert result["rank_mlp"] == 0
    assert result["rank_attn"] == 0
    assert result["lora_parameters"] == (
        CONFIG["layers"] * 4 * RANK_QKV * CONFIG["dim"])

    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert config["r_qkv"] == RANK_QKV
    assert config["alpha_qkv"] == ALPHA_QKV

    fresh = _build_model(seed=99)
    fresh.load_state_dict(base_state)
    apply_adapter(fresh, out)
    assert isinstance(fresh.blocks[0].attn, LoRAMultiheadAttention)
    assert count_lora_params(fresh) == result["lora_parameters"]


def test_cli_rejects_zero_qkv_alpha_for_positive_rank():
    with pytest.raises(SystemExit):
        main([
            "--base", "base", "--train", "train.jsonl",
            "--eval", "eval.jsonl", "--out", "adapter",
            "--qkv-rank", "2", "--qkv-alpha", "0",
        ])
