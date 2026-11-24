"""Attention-LoRA coverage for the tiny LM and the candidate trainer."""
from __future__ import annotations

import json

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit

from atomzero.lora import (
    LoRALinear,
    apply_adapter,
    apply_lora,
    count_lora_params,
    save_adapter,
)
from atomzero.transformer import TransformerLM
from research.adapter_runtime import check_adapter
from research.train_adapter_candidate import main

VOCAB = ["<|user|>", "<|assistant|>", "<|end|>", "hello", "hi", "[UNK]"]
CONFIG = {"vocab": len(VOCAB), "dim": 16, "layers": 2, "heads": 2,
          "ff": 32, "seq": 32, "arch": "transformer"}


def _build_model(seed: int = 0) -> TransformerLM:
    torch.manual_seed(seed)
    return TransformerLM(CONFIG["vocab"], CONFIG["dim"], CONFIG["layers"],
                         CONFIG["heads"], CONFIG["ff"])


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


def _fixed_input(model: TransformerLM, length: int = 6) -> torch.Tensor:
    generator = torch.Generator().manual_seed(7)
    return torch.randint(0, model.vocab, (1, length), generator=generator)


def _randomize_lora(model: torch.nn.Module) -> None:
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith((".A", ".B")):
                param.copy_(torch.randn_like(param) * 0.05)


def test_attention_lora_wraps_out_proj_and_forward_runs():
    model = _build_model()
    apply_lora(model, r_attn=2, r_mlp=0, alpha_attn=4)
    _randomize_lora(model)

    out_proj = model.blocks[0].attn.out_proj
    assert isinstance(out_proj, LoRALinear)
    assert not isinstance(model.blocks[0].mlp[0], LoRALinear)
    names = {name for name, _ in model.named_parameters()}
    assert "blocks.0.attn.out_proj.A" in names
    assert "blocks.0.attn.out_proj.B" in names
    assert "blocks.0.attn.in_proj_weight" in names

    # QKV is a fused Parameter, not a Linear module: it must stay frozen.
    qkv = model.blocks[0].attn.in_proj_weight
    assert isinstance(qkv, torch.nn.Parameter)
    assert qkv.requires_grad is False

    per_layer = 2 * 2 * CONFIG["dim"]  # A plus B for one out_proj
    assert count_lora_params(model) == CONFIG["layers"] * per_layer

    x = _fixed_input(model)
    logits = model(x)
    assert logits.shape == (1, x.size(1), model.vocab)
    assert torch.isfinite(logits).all()

    logits.sum().backward()
    assert out_proj.A.grad is not None and out_proj.A.grad.abs().sum() > 0
    assert out_proj.B.grad is not None and out_proj.B.grad.abs().sum() > 0
    assert out_proj.base.weight.grad is None
    assert qkv.grad is None


def test_attention_adapter_roundtrip_matches_parameters_and_outputs(tmp_path):
    model = _build_model(seed=3)
    base_state = {key: value.clone()
                  for key, value in model.state_dict().items()}
    apply_lora(model, r_attn=2, r_mlp=2, alpha_attn=4, alpha_mlp=8)
    _randomize_lora(model)

    adapter = tmp_path / "adapter"
    save_adapter(model, adapter, r_attn=2, r_mlp=2,
                 alpha_attn=4, alpha_mlp=8)
    config = json.loads((adapter / "config.json").read_text(encoding="utf-8"))
    assert config["r_attn"] == 2
    assert config["alpha_attn"] == 4
    assert config["r_mlp"] == 2
    assert config["alpha_mlp"] == 8

    fresh = _build_model(seed=99)
    fresh.load_state_dict(base_state)
    apply_adapter(fresh, adapter)
    assert isinstance(fresh.blocks[0].attn.out_proj, LoRALinear)

    original = {name: param for name, param in model.named_parameters()
                if name.endswith((".A", ".B"))}
    reloaded = {name: param for name, param in fresh.named_parameters()
                if name.endswith((".A", ".B"))}
    assert set(original) == set(reloaded)
    for name in original:
        assert original[name].shape == reloaded[name].shape
        assert torch.equal(original[name], reloaded[name])
    assert count_lora_params(model) == count_lora_params(fresh)

    x = _fixed_input(model)
    model.eval()
    fresh.eval()
    with torch.no_grad():
        expected = model(x)
        actual = fresh(x)
    assert torch.equal(expected, actual)
    assert torch.isfinite(actual).all()


def test_save_adapter_infers_attention_config(tmp_path):
    model = _build_model()
    apply_lora(model, r_attn=2, r_mlp=3, alpha_attn=4, alpha_mlp=6)
    adapter = tmp_path / "inferred"
    save_adapter(model, adapter)
    config = json.loads((adapter / "config.json").read_text(encoding="utf-8"))
    assert config["r_attn"] == 2
    assert config["alpha_attn"] == 4
    assert config["r_mlp"] == 3
    assert config["alpha_mlp"] == 6


def test_adapter_check_does_not_silently_drop_attention_tensors(tmp_path):
    base, model = _tiny_base(tmp_path, seed=5)
    apply_lora(model, r_attn=2, r_mlp=0, alpha_attn=4)
    _randomize_lora(model)
    adapter = tmp_path / "adapter"
    save_adapter(model, adapter, r_attn=2, r_mlp=0,
                 alpha_attn=4, alpha_mlp=0)

    result = check_adapter(base, adapter)
    assert result["ok"] is True
    assert result["config"]["r_attn"] == 2
    assert result["lora_parameters"] == count_lora_params(model)

    # A config that forgets attention must fail, not load a base model.
    broken = tmp_path / "broken_config"
    save_adapter(model, broken, r_attn=0, r_mlp=0,
                 alpha_attn=0, alpha_mlp=0)
    result = check_adapter(base, broken)
    assert result["ok"] is False
    assert "unexpected=" in result["error"]

    # A missing A tensor must fail instead of leaving a random adapter value.
    checkpoint = torch.load(adapter / "adapter.pt", map_location="cpu",
                            weights_only=False)
    missing = next(key for key in checkpoint["lora"]
                   if key.endswith("out_proj.A"))
    del checkpoint["lora"][missing]
    torch.save(checkpoint, adapter / "adapter.pt")
    result = check_adapter(base, adapter)
    assert result["ok"] is False
    assert "missing=" in result["error"]


def test_cli_trains_attention_adapter_and_records_flags(tmp_path):
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
        "--rank", "2", "--alpha", "4",
        "--rank-attn", "2", "--alpha-attn", "4",
        "--max-seq", "16", "--seed", "3",
        "--result", str(result_path),
        "--comparison", str(tmp_path / "comparison.json"),
    ])
    assert code == 0
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["rank_attn"] == 2
    assert result["alpha_attn"] == 4
    assert result["rank_mlp"] == 2
    assert result["alpha_mlp"] == 4

    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert config["r_attn"] == 2
    assert config["alpha_attn"] == 4

    expected = (CONFIG["layers"] * 2 * 2 * CONFIG["dim"]
                + CONFIG["layers"] * 2 * 2
                * (CONFIG["dim"] + CONFIG["ff"]))
    assert result["lora_parameters"] == expected

    fresh = _build_model(seed=99)
    fresh.load_state_dict(base_state)
    apply_adapter(fresh, out)
    assert isinstance(fresh.blocks[0].attn.out_proj, LoRALinear)
    assert count_lora_params(fresh) == expected


def test_cli_rejects_zero_alpha_for_attention_rank():
    with pytest.raises(SystemExit):
        main([
            "--base", "base", "--train", "train.jsonl",
            "--eval", "eval.jsonl", "--out", "adapter",
            "--rank-attn", "2", "--alpha-attn", "0",
        ])
