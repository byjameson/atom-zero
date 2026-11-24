import json
import math

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit

from atomzero.transformer import TransformerLM
from research.adapter_runtime import load_model
from research.finetune_copy_control import (
    build_result, checkpoint_config, sanitize_comparison, save_checkpoint,
    set_trainable_parameters, train_control, write_outputs)

VOCAB = ["<|user|>", "<|assistant|>", "<|end|>", "hello", "hi", "bye",
         "alpha", "beta", "[UNK]"]
ROWS = [
    {"user": "hello", "assistant": "hi"},
    {"user": "bye", "assistant": "hello"},
    {"user": "alpha", "assistant": "beta"},
]


def tiny_assets(tmp_path, layers=4):
    tokenizer = Tokenizer(WordLevel(
        vocab={word: index for index, word in enumerate(VOCAB)},
        unk_token="[UNK]"))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    base = tmp_path / "base"
    base.mkdir()
    tokenizer.save(str(base / "chat_en_bpe8k.json"))
    config = {"vocab": len(VOCAB), "dim": 16, "layers": layers, "heads": 2,
              "ff": 32, "seq": 32, "arch": "transformer"}
    torch.manual_seed(0)
    model = TransformerLM(config["vocab"], config["dim"], config["layers"],
                          config["heads"], config["ff"])
    torch.save({"config": config, "state": model.state_dict()},
               base / "model.pt")
    return base, config


def test_one_training_step_changes_weights(tmp_path):
    base, _ = tiny_assets(tmp_path)
    model, tok = load_model(base)
    before = {key: value.detach().clone()
              for key, value in model.state_dict().items()}
    summary = train_control(
        model, tok, ROWS, steps=1, lr=1e-2, batch_size=2, max_seq=32,
        content_weight=2.0, seed=0, freeze="none", device="cpu",
        log_every=0)
    assert summary["train_steps"] == 1
    assert summary["train_examples"] == len(ROWS)
    assert summary["trainable_parameters"] == summary["total_parameters"]
    assert math.isfinite(summary["train_loss"])
    changed = [key for key, value in before.items()
               if not torch.equal(value, model.state_dict()[key])]
    assert changed


def test_checkpoint_roundtrip_restores_weights_and_tokenizer(tmp_path):
    base, _ = tiny_assets(tmp_path)
    model, tok = load_model(base)
    train_control(
        model, tok, ROWS, steps=1, lr=1e-2, batch_size=2, max_seq=32,
        seed=0, freeze="none", device="cpu", log_every=0)
    out = tmp_path / "checkpoints" / "fullft_control" / "v0001"
    saved = save_checkpoint(
        model, out, config=checkpoint_config(model), tokenizer=tok,
        metadata={"base": str(base)}, train_summary={"steps": 1})
    assert saved["tokenizer"] == "chat_en_bpe8k.json"
    assert (out / "model.pt").is_file()
    assert (out / "config.json").is_file()
    assert (out / "chat_en_bpe8k.json").is_file()
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert config["research_control"] is True
    assert config["product_weights"] is False
    assert config["weights_changed"] is True
    loaded, loaded_tok = load_model(base, out)
    assert loaded_tok.token_to_id("<|assistant|>") == 1
    original = model.state_dict()
    restored = loaded.state_dict()
    assert set(original) == set(restored)
    for key, value in original.items():
        assert torch.equal(restored[key].cpu(), value.cpu())


def test_result_json_has_research_flags_and_no_prompt_text(tmp_path):
    exact_match = {
        "examples": 3, "matches": 1, "rate": 1 / 3, "max_new": 8,
        "families": {"copy": {"examples": 3, "matches": 1,
                              "rate": 1 / 3}},
    }
    comparison = {
        "base": "base",
        "data": "eval.jsonl",
        "examples": 3,
        "created": "2026-01-01T00:00:00",
        "models": [
            {
                "label": "base",
                "path": "base",
                "metrics": {
                    "examples": 3, "tokens": 6, "loss": 4.0,
                    "ppl": math.exp(4.0), "content_loss": 3.5,
                    "syntax_loss": 4.5, "content_tokens": 4,
                    "exact_match": {**exact_match, "matches": 0,
                                    "rate": 0.0},
                    "rows": [{"user": "SECRET PROMPT",
                              "assistant": "SECRET ANSWER", "loss": 4.0,
                              "tokens": 2}],
                },
            },
            {
                "label": "fullft_control/v0001",
                "path": "checkpoints/fullft_control/v0001",
                "metrics": {
                    "examples": 3, "tokens": 6, "loss": 2.0,
                    "ppl": math.exp(2.0), "content_loss": 2.5,
                    "syntax_loss": 1.5, "content_tokens": 4,
                    "exact_match": exact_match,
                    "rows": [{"user": "SECRET PROMPT",
                              "assistant": "SECRET ANSWER", "loss": 2.0,
                              "tokens": 2}],
                },
                "vs_base": {"loss_delta": -2.0, "loss_relative": -0.5,
                            "ppl_ratio": -0.5},
            },
        ],
        "families": {
            "copy": {
                "baseline": {"examples": 3, "loss": 4.0,
                             "content_loss": 3.5},
                "adapter": {"examples": 3, "loss": 2.0,
                            "content_loss": 2.5},
            },
        },
    }
    training = {"freeze": "none", "device": "cpu", "train_steps": 1}
    result = build_result(
        comparison, training, base="base",
        checkpoint="checkpoints/fullft_control/v0001",
        train_path="train.jsonl", eval_path="eval.jsonl", device="cpu")
    assert result["research_control"] is True
    assert result["product_weights"] is False
    assert result["weights_changed"] is True
    assert result["base_metrics"]["loss"] == 4.0
    assert result["control_metrics"]["loss"] == 2.0
    assert result["control_metrics"]["content_loss"] == 2.5
    assert result["control_metrics"]["syntax_loss"] == 1.5
    assert result["control_metrics"]["exact_match_rate"] == pytest.approx(
        1 / 3)
    assert result["base_loss"] == 4.0
    assert result["control_loss"] == 2.0
    assert result["loss_delta"] == pytest.approx(-2.0)
    assert result["control_content_loss"] == 2.5
    assert result["control_syntax_loss"] == 1.5
    assert result["control_exact_match"] == pytest.approx(1 / 3)
    assert set(result["families"]) == {"copy"}
    result_path = tmp_path / "result.json"
    comparison_path = tmp_path / "comparison.json"
    clean = write_outputs(result, comparison, result_path, comparison_path)
    assert "rows" not in clean["models"][0]["metrics"]
    result_text = result_path.read_text(encoding="utf-8")
    comparison_text = comparison_path.read_text(encoding="utf-8")
    assert "SECRET PROMPT" not in result_text
    assert "SECRET PROMPT" not in comparison_text
    assert "SECRET ANSWER" not in comparison_text
    assert '"rows"' not in comparison_text
    saved = json.loads(result_text)
    assert saved["research_control"] is True
    assert saved["product_weights"] is False
    assert saved["weights_changed"] is True
    assert saved["control_metrics"]["exact_match"]["rate"] == pytest.approx(
        1 / 3)


def test_freeze_modes_select_the_expected_parameters(tmp_path):
    base, _ = tiny_assets(tmp_path, layers=4)
    model, _ = load_model(base)
    assert model.head.weight is model.tok.weight

    summary = set_trainable_parameters(model, "none")
    assert all(parameter.requires_grad for parameter in model.parameters())
    assert summary["frozen_parameters"] == 0
    assert summary["trainable_parameters"] == summary["total_parameters"]

    summary = set_trainable_parameters(model, "embeddings")
    assert not model.tok.weight.requires_grad
    assert not model.pos.weight.requires_grad
    assert not model.head.weight.requires_grad
    assert model.blocks[0].mlp[0].weight.requires_grad
    assert summary["frozen_groups"] == ["tok", "pos"]
    assert summary["frozen_parameters"] > 0

    summary = set_trainable_parameters(model, "first-half")
    assert not model.blocks[0].mlp[0].weight.requires_grad
    assert not model.blocks[1].mlp[0].weight.requires_grad
    assert model.blocks[2].mlp[0].weight.requires_grad
    assert model.blocks[3].mlp[0].weight.requires_grad
    assert model.ln.weight.requires_grad
    assert summary["frozen_groups"] == [
        "tok", "pos", "blocks.0", "blocks.1"]

    set_trainable_parameters(model, "none")
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_unknown_freeze_mode_is_rejected(tmp_path):
    base, _ = tiny_assets(tmp_path)
    model, _ = load_model(base)
    with pytest.raises(ValueError, match="unknown freeze mode"):
        set_trainable_parameters(model, "everything")


def test_sanitize_comparison_keeps_aggregates_only():
    payload = {
        "models": [
            {"label": "base", "metrics": {"loss": 1.0,
                                          "rows": [{"user": "hidden"}]}},
            {"label": "control", "metrics": {"loss": 0.5,
                                             "rows": [{"user": "hidden"}]}},
        ],
        "families": {"copy": {"baseline": {"loss": 1.0}}},
    }
    clean = sanitize_comparison(payload)
    assert "rows" not in clean["models"][0]["metrics"]
    assert "rows" not in clean["models"][1]["metrics"]
    assert clean["families"] == payload["families"]
