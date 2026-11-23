import json
from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit

from atomzero.lora import count_lora_params, inject_lora, save_adapter
from atomzero.transformer import TransformerLM
from research.adapter_runtime import (
    adapter_metadata, assistant_label_mask, check_adapter, evaluate_bpb,
    fixed_windows, load_model, mask_user_labels, read_rows, score_rows)

VOCAB = ["<|user|>", "<|assistant|>", "<|end|>", "hello", "hi", "bye",
         "[UNK]"]


def tiny_assets(tmp_path):
    tokenizer = Tokenizer(WordLevel(
        vocab={word: index for index, word in enumerate(VOCAB)},
        unk_token="[UNK]"))
    tokenizer.pre_tokenizer = WhitespaceSplit()

    base = tmp_path / "base"
    base.mkdir()
    tokenizer_path = base / "chat_en_bpe8k.json"
    tokenizer.save(str(tokenizer_path))
    config = {"vocab": len(VOCAB), "dim": 16, "layers": 1, "heads": 2,
              "ff": 32, "seq": 32, "arch": "transformer"}
    torch.manual_seed(0)
    model = TransformerLM(config["vocab"], config["dim"], config["layers"],
                          config["heads"], config["ff"])
    torch.save({"config": config, "state": model.state_dict()},
               base / "model.pt")
    (base / "config.json").write_text(json.dumps(config), encoding="utf-8")

    adapter = tmp_path / "v0001"
    inject_lora(model, rank=4, alpha=8)
    save_adapter(model, adapter, r_attn=0, r_mlp=4, alpha_mlp=8)
    return base, adapter, tokenizer_path


def test_mask_user_labels_only_drops_user_turns():
    y = torch.tensor([[0, 3, 4, 1, 5, 6, 2, 0, 7]])
    masked = mask_user_labels(y, user_id=0, assistant_id=1)
    assert masked.tolist() == [[-100, -100, -100, 1, 5, 6, 2, -100, -100]]


def test_assistant_mask_covers_tag_answer_and_end():
    y = torch.tensor([[3, 4, 1, 5, 6, 2, 7]])
    mask = assistant_label_mask(y, assistant_id=1, end_id=2)
    assert mask.tolist() == [[False, False, True, True, True, True, False]]


def test_load_model_applies_external_adapter(tmp_path):
    base, adapter, _ = tiny_assets(tmp_path)
    model, tokenizer = load_model(base, adapter)
    assert count_lora_params(model) > 0
    assert tokenizer.token_to_id("<|assistant|>") == 1
    ids = tokenizer.encode("hello").ids
    with torch.no_grad():
        logits = model(torch.tensor([ids]))
    assert logits.shape[0] == 1 and logits.shape[1] == len(ids)


def test_adapter_metadata_and_check(tmp_path):
    base, adapter, _ = tiny_assets(tmp_path)
    metadata = adapter_metadata(adapter)
    assert metadata["parameters"] > 0
    assert metadata["config"]["r_mlp"] == 4
    result = check_adapter(base, adapter)
    assert result["ok"] is True
    assert result["lora_parameters"] > 0
    assert result["logits_shape"][0] == 1


def test_check_adapter_reports_missing_files(tmp_path):
    base, _, _ = tiny_assets(tmp_path)
    result = check_adapter(base, tmp_path / "missing")
    assert result["ok"] is False
    assert "adapter.pt missing" in result["error"]


def test_check_adapter_rejects_empty_lora(tmp_path):
    base, adapter, _ = tiny_assets(tmp_path)
    torch.save({"lora": {}, "config": {"r_mlp": 4}}, adapter / "adapter.pt")
    result = check_adapter(base, adapter)
    assert result["ok"] is False
    assert "no LoRA tensors" in result["error"]


def test_fixed_windows_are_deterministic():
    tokens = torch.arange(40)
    first = fixed_windows(tokens, seq=8, count=2, batch=2, seed=3,
                          device="cpu")
    second = fixed_windows(tokens, seq=8, count=2, batch=2, seed=3,
                           device="cpu")
    assert len(first) == 2
    for (x1, y1), (x2, y2) in zip(first, second):
        assert torch.equal(x1, x2)
        assert torch.equal(y1, y2)
        assert x1.shape == (2, 8)
        assert y1.shape == (2, 8)


def test_evaluate_bpb_is_finite():
    torch.manual_seed(0)
    model = TransformerLM(64, 16, 1, 2, 32)
    tokenizer = SimpleNamespace(decode=lambda ids: "abcd")
    tokens = torch.arange(40) % 64
    batches = fixed_windows(tokens, seq=8, count=2, batch=2, seed=0,
                            device="cpu")
    result = evaluate_bpb(model, tokenizer, batches)
    assert result["bpb"] > 0
    assert result["tokens"] == 32


def test_compare_reports_base_and_adapter(tmp_path):
    from research.compare_adapters import compare

    base, adapter, _ = tiny_assets(tmp_path)
    data = tmp_path / "rows.jsonl"
    data.write_text(
        '{"user": "hello", "assistant": "hi"}\n'
        '{"user": "bye", "assistant": "hello"}\n', encoding="utf-8")
    payload = compare(base, data, [adapter])
    assert payload["examples"] == 2
    assert payload["models"][0]["label"] == "base"
    assert payload["models"][1]["label"].endswith("/v0001")
    assert payload["models"][1]["vs_base"]["loss_relative"] == pytest.approx(
        payload["models"][1]["metrics"]["loss"]
        / payload["models"][0]["metrics"]["loss"] - 1)


def test_read_rows_skips_incomplete_examples(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text(
        '{"user": "hello", "assistant": "hi"}\n'
        '{"user": "empty", "assistant": ""}\n'
        '{"assistant": "orphan"}\n'
        '{"user": "bye", "assistant": "hello"}\n', encoding="utf-8")
    assert read_rows(path) == [
        {"user": "hello", "assistant": "hi"},
        {"user": "bye", "assistant": "hello"},
    ]
