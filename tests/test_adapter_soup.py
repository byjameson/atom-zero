"""Adapter soup: weighted A/B merging, validation and CLI schema."""
from __future__ import annotations

import json
import shutil

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit

from atomzero.lora import (
    AdapterMergeError,
    LoRALinear,
    apply_adapter,
    apply_lora,
    count_lora_params,
    merge_adapters,
    merge_adapter_state_dicts,
    normalize_adapter_weights,
    save_adapter,
)
from atomzero.transformer import TransformerLM
from research.merge_adapters import build_parser, main

VOCAB = ["<|user|>", "<|assistant|>", "<|end|>", "[UNK]",
         "hello", "hi", "bye"]
CONFIG = {"vocab": len(VOCAB), "dim": 16, "layers": 2, "heads": 2,
          "ff": 32, "seq": 32, "arch": "transformer"}


def _build_model(seed: int = 0) -> TransformerLM:
    torch.manual_seed(seed)
    return TransformerLM(CONFIG["vocab"], CONFIG["dim"], CONFIG["layers"],
                         CONFIG["heads"], CONFIG["ff"])


def _write_base(tmp_path, seed: int = 0):
    base = tmp_path / "base"
    base.mkdir()
    model = _build_model(seed)
    tokenizer = Tokenizer(WordLevel(
        vocab={word: index for index, word in enumerate(VOCAB)},
        unk_token="[UNK]"))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    tokenizer.save(str(base / "chat_en_bpe8k.json"))
    torch.save({"config": CONFIG, "state": model.state_dict()},
               base / "model.pt")
    return base, model


def _randomize_lora(model, seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith((".A", ".B")):
                param.copy_(torch.randn(param.shape, generator=generator)
                            * 0.05)


def _make_adapter(tmp_path, name: str, seed: int, *, base_state=None,
                  r_attn: int = 2, r_mlp: int = 2,
                  alpha_attn: int = 4, alpha_mlp: int = 4):
    model = _build_model(seed=seed + 100)
    if base_state is not None:
        model.load_state_dict(base_state)
    apply_lora(model, r_attn=r_attn, r_mlp=r_mlp,
               alpha_attn=alpha_attn, alpha_mlp=alpha_mlp)
    _randomize_lora(model, seed)
    path = tmp_path / name
    save_adapter(model, path, r_attn=r_attn, r_mlp=r_mlp,
                 alpha_attn=alpha_attn, alpha_mlp=alpha_mlp)
    return path


def _state(model):
    return {key: value.clone() for key, value in model.state_dict().items()}


def _load_lora(path):
    return torch.load(path / "adapter.pt", map_location="cpu",
                      weights_only=False)["lora"]


def _all_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from _all_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _all_keys(child)


def test_equal_weight_ab_merge_averages_each_tensor(tmp_path):
    _, base_model = _write_base(tmp_path)
    state = _state(base_model)
    first = _make_adapter(tmp_path, "first", seed=1, base_state=state)
    second = _make_adapter(tmp_path, "second", seed=2, base_state=state)

    merged = tmp_path / "merged"
    summary = merge_adapters([first, second], out=merged)

    assert summary["mode"] == "ab"
    assert summary["weights"] == [0.5, 0.5]
    assert summary["config"]["r_attn"] == 2
    assert summary["config"]["alpha_attn"] == 4
    assert summary["config"]["r_mlp"] == 2
    assert summary["config"]["alpha_mlp"] == 4
    assert summary["config"]["merge"]["mode"] == "ab"

    left = _load_lora(first)
    right = _load_lora(second)
    actual = _load_lora(merged)
    assert set(actual) == set(left) == set(right)
    for key in left:
        expected = (left[key].float() + right[key].float()) / 2.0
        torch.testing.assert_close(actual[key].float(), expected,
                                   rtol=0.0, atol=1e-7)

    config = json.loads((merged / "config.json").read_text(encoding="utf-8"))
    assert config["r_attn"] == 2
    assert config["r_mlp"] == 2


def test_merged_adapter_roundtrip_loads_and_forwards(tmp_path):
    base, base_model = _write_base(tmp_path)
    state = _state(base_model)
    first = _make_adapter(tmp_path, "first", seed=3, base_state=state)
    second = _make_adapter(tmp_path, "second", seed=4, base_state=state)
    merged = tmp_path / "merged"
    merge_adapters([first, second], out=merged)

    fresh = _build_model(seed=99)
    fresh.load_state_dict(state)
    apply_adapter(fresh, merged)
    assert isinstance(fresh.blocks[0].attn.out_proj, LoRALinear)
    expected_params = count_lora_params(fresh)

    generator = torch.Generator().manual_seed(5)
    x = torch.randint(0, CONFIG["vocab"], (1, 6), generator=generator)
    fresh.eval()
    with torch.no_grad():
        logits = fresh(x)
    assert logits.shape == (1, 6, CONFIG["vocab"])
    assert torch.isfinite(logits).all()

    # The saved merged tensors are the ones that were loaded.
    reloaded = _load_lora(merged)
    with torch.no_grad():
        for name, param in fresh.named_parameters():
            if name.endswith((".A", ".B")):
                torch.testing.assert_close(param, reloaded[name])
    assert expected_params == sum(
        int(tensor.numel()) for tensor in reloaded.values())


def test_geometry_mismatch_raises_with_clear_error(tmp_path):
    _, base_model = _write_base(tmp_path)
    state = _state(base_model)
    first = _make_adapter(tmp_path, "first", seed=6, base_state=state,
                          r_mlp=2)
    second = _make_adapter(tmp_path, "second", seed=7, base_state=state,
                           r_mlp=3)

    with pytest.raises(AdapterMergeError, match="geometry mismatch"):
        merge_adapters([first, second], out=tmp_path / "merged")
    with pytest.raises(AdapterMergeError, match="r_mlp"):
        merge_adapters([first, second], out=tmp_path / "merged")


def test_missing_or_differently_shaped_tensors_raise(tmp_path):
    _, base_model = _write_base(tmp_path)
    state = _state(base_model)
    first = _make_adapter(tmp_path, "first", seed=8, base_state=state)
    second = _make_adapter(tmp_path, "second", seed=9, base_state=state)

    missing = tmp_path / "missing"
    shutil.copytree(second, missing)
    checkpoint = torch.load(missing / "adapter.pt", map_location="cpu",
                            weights_only=False)
    removed = next(key for key in checkpoint["lora"]
                   if key.endswith("out_proj.A"))
    del checkpoint["lora"][removed]
    torch.save(checkpoint, missing / "adapter.pt")
    with pytest.raises(AdapterMergeError, match="missing"):
        merge_adapters([first, missing], out=tmp_path / "merged")

    changed = tmp_path / "changed"
    shutil.copytree(second, changed)
    checkpoint = torch.load(changed / "adapter.pt", map_location="cpu",
                            weights_only=False)
    key = next(key for key in checkpoint["lora"] if key.endswith("out_proj.A"))
    checkpoint["lora"][key] = torch.zeros(
        checkpoint["lora"][key].shape[0],
        checkpoint["lora"][key].shape[1] + 1)
    torch.save(checkpoint, changed / "adapter.pt")
    with pytest.raises(AdapterMergeError, match="shape mismatch"):
        merge_adapters([first, changed], out=tmp_path / "merged")


def test_weights_are_normalized_and_validated(tmp_path):
    equal = normalize_adapter_weights(None, 3)
    assert equal == pytest.approx([1 / 3, 1 / 3, 1 / 3])
    assert normalize_adapter_weights([1, 3], 2) == [0.25, 0.75]
    assert normalize_adapter_weights([0, 2], 2) == [0.0, 1.0]

    with pytest.raises(AdapterMergeError, match="one value per adapter"):
        normalize_adapter_weights([1], 2)
    with pytest.raises(AdapterMergeError, match="finite"):
        normalize_adapter_weights([1, float("nan")], 2)
    with pytest.raises(AdapterMergeError, match=">= 0"):
        normalize_adapter_weights([1, -1], 2)
    with pytest.raises(AdapterMergeError, match="sum to > 0"):
        normalize_adapter_weights([0, 0], 2)
    with pytest.raises(AdapterMergeError, match="numbers"):
        normalize_adapter_weights([True, 1], 2)


def test_weighted_merge_uses_normalized_weights(tmp_path):
    _, base_model = _write_base(tmp_path)
    state = _state(base_model)
    first = _make_adapter(tmp_path, "first", seed=10, base_state=state)
    second = _make_adapter(tmp_path, "second", seed=11, base_state=state)
    merged = tmp_path / "merged"
    summary = merge_adapters([first, second], out=merged, weights=[1, 3])
    assert summary["weights"] == [0.25, 0.75]

    left = _load_lora(first)
    right = _load_lora(second)
    actual = _load_lora(merged)
    for key in left:
        expected = left[key].float() * 0.25 + right[key].float() * 0.75
        torch.testing.assert_close(actual[key].float(), expected,
                                   rtol=0.0, atol=1e-7)


def test_delta_svd_merge_roundtrips(tmp_path):
    _, base_model = _write_base(tmp_path)
    state = _state(base_model)
    first = _make_adapter(tmp_path, "first", seed=12, base_state=state)
    second = _make_adapter(tmp_path, "second", seed=13, base_state=state)
    state_dicts = merge_adapter_state_dicts(
        [first, second], mode="delta-svd", rank=2)
    assert state_dicts["config"]["r_attn"] == 2
    assert state_dicts["config"]["r_mlp"] == 2
    assert state_dicts["tensors"] == len(_load_lora(first))

    merged = tmp_path / "merged"
    merge_adapters([first, second], out=merged, mode="delta-svd", rank=2)
    fresh = _build_model(seed=98)
    fresh.load_state_dict(state)
    apply_adapter(fresh, merged)
    generator = torch.Generator().manual_seed(14)
    x = torch.randint(0, CONFIG["vocab"], (1, 5), generator=generator)
    with torch.no_grad():
        logits = fresh(x)
    assert torch.isfinite(logits).all()


def test_delta_svd_preserves_weighted_delta_when_rank_allows(tmp_path):
    _, base_model = _write_base(tmp_path)
    state = _state(base_model)
    first = _make_adapter(tmp_path, "first", seed=12, base_state=state)
    second = _make_adapter(tmp_path, "second", seed=13, base_state=state)
    merged = merge_adapter_state_dicts(
        [first, second], mode="delta-svd", rank=4)
    assert merged["config"]["r_attn"] == 4
    assert merged["config"]["r_mlp"] == 4

    left = _load_lora(first)
    right = _load_lora(second)
    for prefix in sorted({key[:-2] for key in left}):
        expected = (left[prefix + ".B"].float()
                    @ left[prefix + ".A"].float()
                    + right[prefix + ".B"].float()
                    @ right[prefix + ".A"].float()) / 2.0
        actual = (merged["lora"][prefix + ".B"].float()
                  @ merged["lora"][prefix + ".A"].float())
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_cli_parser_defaults_and_validation(tmp_path):
    parser = build_parser()
    args = parser.parse_args(["--adapters", "a", "b", "--out", "soup"])
    assert args.merge_mode == "ab"
    assert args.weights is None
    assert args.rank is None
    assert args.result.endswith("adapter_soup.json")

    _, base_model = _write_base(tmp_path)
    state = _state(base_model)
    first = _make_adapter(tmp_path, "first", seed=15, base_state=state)
    second = _make_adapter(tmp_path, "second", seed=16, base_state=state)
    with pytest.raises(SystemExit):
        main(["--adapters", str(first), str(second),
              "--out", str(tmp_path / "soup"), "--weights", "1"])
    with pytest.raises(SystemExit):
        main(["--adapters", str(first), str(second),
              "--out", str(tmp_path / "soup"), "--rank", "4"])


def test_cli_writes_result_and_content_free_comparison(tmp_path):
    base, base_model = _write_base(tmp_path)
    state = _state(base_model)
    first = _make_adapter(tmp_path, "first", seed=17, base_state=state)
    second = _make_adapter(tmp_path, "second", seed=18, base_state=state)
    eval_path = tmp_path / "eval.jsonl"
    eval_path.write_text(
        json.dumps({"user": "hello", "assistant": "hi"}) + "\n"
        + json.dumps({"user": "hi", "assistant": "bye"}) + "\n",
        encoding="utf-8")
    out = tmp_path / "soup"
    result_path = tmp_path / "result.json"
    comparison_path = tmp_path / "comparison.json"

    code = main([
        "--adapters", str(first), str(second),
        "--weights", "1", "1",
        "--out", str(out),
        "--base", str(base),
        "--eval", str(eval_path),
        "--result", str(result_path),
        "--comparison", str(comparison_path),
    ])
    assert code == 0

    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["mode"] == "ab"
    assert result["weights"] == [0.5, 0.5]
    assert result["config"]["r_attn"] == 2
    assert result["config"]["alpha_attn"] == 4
    assert result["config"]["r_mlp"] == 2
    assert result["config"]["alpha_mlp"] == 4
    assert result["tensors"] == len(_load_lora(first))
    assert result["parameters"] > 0
    assert result["validation"]["ok"] is True
    assert result["validation"]["lora_parameters"] > 0
    assert result["evaluation"]["examples"] == 2
    models = result["evaluation"]["models"]
    assert [model["label"] for model in models] == ["base", "soup"]
    assert all(isinstance(model["metrics"]["loss"], float) for model in models)
    assert all("rows" not in model["metrics"] for model in models)

    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    assert comparison["examples"] == 2
    assert comparison["weights"] == [0.5, 0.5]
    assert len(comparison["models"]) == 2

    forbidden = {"rows", "user", "assistant"}
    assert forbidden.isdisjoint(set(_all_keys(result)))
    assert forbidden.isdisjoint(set(_all_keys(comparison)))
    assert "hello" not in comparison_path.read_text(encoding="utf-8")
    assert "bye" not in comparison_path.read_text(encoding="utf-8")
