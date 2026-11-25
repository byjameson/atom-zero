"""Tiny-model coverage for depth upcycling and its research metadata."""
from __future__ import annotations

import json

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit

from atomzero.transformer import TransformerLM
from research.adapter_runtime import load_model
from research.upcycle_model import (
    UpcycleError,
    expand_state,
    layer_mapping,
    main,
    upcycle,
)
from research.upcycle_report import parameter_counts

VOCAB = ["<|user|>", "<|assistant|>", "<|end|>", "hello", "hi", "bye",
         "alpha", "beta", "[UNK]"]
CONFIG = {"vocab": len(VOCAB), "dim": 16, "layers": 4, "heads": 2,
          "ff": 32, "seq": 16, "arch": "transformer"}
NON_BLOCK_KEYS = ("tok.weight", "pos.weight", "ln.weight", "ln.bias",
                  "head.weight")


def tiny_base(tmp_path, layers=4, tokenizer=True):
    base = tmp_path / "base"
    base.mkdir()
    if tokenizer:
        word_tokenizer = Tokenizer(WordLevel(
            vocab={word: index for index, word in enumerate(VOCAB)},
            unk_token="[UNK]"))
        word_tokenizer.pre_tokenizer = WhitespaceSplit()
        word_tokenizer.save(str(base / "chat_en_bpe8k.json"))
    config = {**CONFIG, "layers": layers}
    torch.manual_seed(0)
    model = TransformerLM(config["vocab"], config["dim"], config["layers"],
                          config["heads"], config["ff"])
    torch.save({"config": config, "state": model.state_dict(), "step": 100},
               base / "model.pt")
    (base / "config.json").write_text(
        json.dumps(config), encoding="utf-8")
    return base, config, model


def _block_suffixes(model):
    prefix = "blocks.0."
    return sorted(
        key[len(prefix):] for key in model.state_dict()
        if key.startswith(prefix))


def test_parameter_counts_match_known_architectures():
    product = {"vocab": 8192, "dim": 768, "layers": 6, "heads": 12,
               "ff": 3072, "seq": 128, "arch": "transformer"}
    source = parameter_counts(product)
    eight = parameter_counts(product, layers=8)
    twelve = parameter_counts(product, layers=12)
    assert source["parameters"] == 49_213_440
    assert eight["parameters"] == 63_389_184
    assert twelve["parameters"] == 91_740_672
    assert eight["parameters"] - source["parameters"] == 14_175_744
    assert twelve["parameters"] - source["parameters"] == 42_527_232

    torch.manual_seed(0)
    tiny = TransformerLM(CONFIG["vocab"], CONFIG["dim"], CONFIG["layers"],
                         CONFIG["heads"], CONFIG["ff"])
    assert parameter_counts(CONFIG)["parameters"] == sum(
        parameter.numel() for parameter in tiny.parameters())


def test_layer_mapping_is_ordered_and_covers_the_stack():
    assert layer_mapping(4, 6) == [0, 0, 1, 2, 2, 3]
    assert layer_mapping(6, 8) == [0, 0, 1, 2, 3, 3, 4, 5]
    assert layer_mapping(6, 12) == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    for source, target in ((4, 6), (6, 8), (6, 12)):
        mapping = layer_mapping(source, target)
        assert mapping[0] == 0
        assert mapping[-1] == source - 1
        assert mapping == sorted(mapping)
    with pytest.raises(UpcycleError, match="greater than"):
        layer_mapping(6, 6)


def test_duplicate_expands_layers_and_copies_selected_blocks(tmp_path):
    base, config, source_model = tiny_base(tmp_path, layers=4)
    out = tmp_path / "checkpoints" / "upcycle_6l"
    result = upcycle(
        base=base, out=out, layers=6, mode="duplicate", seed=7)
    assert result["status"] == "completed"
    assert result["target_layers"] == 6
    assert result["layer_mapping"] == [0, 0, 1, 2, 2, 3]
    assert result["parameter_counts"]["target"]["layers"] == 6
    assert result["parameter_counts"]["target"]["parameters"] == sum(
        parameter.numel() for parameter in load_model(out)[0].parameters())

    saved = torch.load(out / "model.pt", map_location="cpu",
                       weights_only=False)
    assert saved["config"]["layers"] == 6
    assert saved["step"] == 0
    source_state = source_model.state_dict()
    for target, source in enumerate(result["layer_mapping"]):
        for suffix in _block_suffixes(source_model):
            assert torch.equal(
                saved["state"][f"blocks.{target}.{suffix}"],
                source_state[f"blocks.{source}.{suffix}"])
    for key in NON_BLOCK_KEYS:
        assert torch.equal(saved["state"][key], source_state[key])
    assert (out / "chat_en_bpe8k.json").is_file()


def test_interpolate_blends_only_new_duplicate_blocks(tmp_path):
    base, config, source_model = tiny_base(tmp_path, layers=4)
    out = tmp_path / "checkpoints" / "upcycle_6l_interp"
    result = upcycle(
        base=base, out=out, layers=6, mode="interpolate", seed=2)
    assert result["layer_mapping"] == [0, 0, 1, 2, 2, 3]
    assert result["interpolated_blocks"] == [
        {"target": 1, "source": 0, "neighbor": 1, "alpha": 0.5},
        {"target": 4, "source": 2, "neighbor": 3, "alpha": 0.5},
    ]
    saved = torch.load(out / "model.pt", map_location="cpu",
                       weights_only=False)
    source_state = source_model.state_dict()
    for suffix in _block_suffixes(source_model):
        expected = 0.5 * (
            source_state[f"blocks.0.{suffix}"]
            + source_state[f"blocks.1.{suffix}"])
        assert torch.allclose(
            saved["state"][f"blocks.1.{suffix}"], expected)
        expected = 0.5 * (
            source_state[f"blocks.2.{suffix}"]
            + source_state[f"blocks.3.{suffix}"])
        assert torch.allclose(
            saved["state"][f"blocks.4.{suffix}"], expected)
    # Non-duplicated targets stay exact copies.
    for target, source in ((0, 0), (2, 1), (3, 2), (5, 3)):
        for suffix in _block_suffixes(source_model):
            assert torch.equal(
                saved["state"][f"blocks.{target}.{suffix}"],
                source_state[f"blocks.{source}.{suffix}"])


def test_forward_shape_and_research_metadata(tmp_path):
    base, _, _ = tiny_base(tmp_path)
    out = tmp_path / "checkpoints" / "upcycle_6l"
    upcycle(base=base, out=out, layers=6, mode="duplicate", seed=1)
    checkpoint = torch.load(out / "model.pt", map_location="cpu",
                            weights_only=False)
    assert checkpoint["config"]["layers"] == 6
    assert checkpoint["research"]["research_control"] is True
    assert checkpoint["research"]["product_weights"] is False
    assert checkpoint["research"]["weights_changed"] is True
    assert checkpoint["research"]["trained"] is False

    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert config["layers"] == 6
    assert config["research_control"] is True
    assert config["product_weights"] is False
    assert config["weights_changed"] is True
    assert config["trained"] is False
    assert config["tokenizer"] == "chat_en_bpe8k.json"

    metadata = json.loads((out / "upcycle.json").read_text(encoding="utf-8"))
    assert metadata["schema"] == "atomzero.upcycle.v1"
    assert metadata["research_control"] is True
    assert metadata["product_weights"] is False
    assert metadata["weights_changed"] is True
    assert metadata["trained"] is False
    assert metadata["layer_mapping"] == [0, 0, 1, 2, 2, 3]
    assert metadata["forward_check"]["ok"] is True
    assert metadata["forward_check"]["logits_shape"] == [1, 8, len(VOCAB)]
    assert metadata["parameter_counts"]["target"]["parameters"] == sum(
        parameter.numel() for parameter in load_model(out)[0].parameters())


def test_output_is_a_loadable_new_base(tmp_path):
    base, _, _ = tiny_base(tmp_path)
    out = tmp_path / "checkpoints" / "upcycle_6l"
    upcycle(base=base, out=out, layers=6, mode="duplicate", seed=0)
    model, tokenizer = load_model(out)
    assert len(model.blocks) == 6
    assert model.vocab == len(VOCAB)
    assert tokenizer.token_to_id("<|assistant|>") == 1
    tokens = torch.randint(0, model.vocab, (1, 7), dtype=torch.long)
    with torch.no_grad():
        logits = model(tokens)
    assert tuple(logits.shape) == (1, 7, len(VOCAB))
    assert torch.isfinite(logits).all()


def test_dry_run_writes_nothing(tmp_path, capsys):
    base, _, _ = tiny_base(tmp_path)
    out = tmp_path / "checkpoints" / "upcycle_dry"
    report = tmp_path / "reports" / "upcycle_dry.json"
    code = main([
        "--base", str(base), "--out", str(out), "--layers", "6",
        "--mode", "interpolate", "--seed", "3", "--dry-run",
        "--report", str(report),
    ])
    assert code == 0
    assert not out.exists()
    assert not report.exists()
    assert not report.with_suffix(".md").exists()
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    assert payload["status"] == "dry_run"
    assert payload["schema"] == "atomzero.upcycle.v1"
    assert payload["layer_mapping"] == [0, 0, 1, 2, 2, 3]
    assert payload["interpolated_blocks"][0]["target"] == 1
    assert payload["parameter_counts"]["target"]["layers"] == 6
    assert payload["research_control"] is True
    assert payload["product_weights"] is False
    assert payload["weights_changed"] is True
    assert payload["tokenizer_ready"] is True


def test_refuses_to_overwrite_base_and_nested_paths(tmp_path):
    base, _, _ = tiny_base(tmp_path)
    before = (base / "model.pt").read_bytes()
    with pytest.raises(SystemExit):
        main(["--base", str(base), "--out", str(base), "--layers", "6"])
    assert (base / "model.pt").read_bytes() == before
    with pytest.raises(UpcycleError, match="inside the base"):
        upcycle(base=base, out=base / "nested", layers=6)
    assert not (base / "nested").exists()
    assert (base / "model.pt").read_bytes() == before


def test_existing_checkpoint_requires_force(tmp_path):
    base, _, _ = tiny_base(tmp_path)
    out = tmp_path / "checkpoints" / "upcycle_6l"
    upcycle(base=base, out=out, layers=6, mode="duplicate", seed=0)
    with pytest.raises(UpcycleError, match="already contains a checkpoint"):
        upcycle(base=base, out=out, layers=6, mode="duplicate", seed=0)
    result = upcycle(
        base=base, out=out, layers=6, mode="duplicate", seed=0, force=True)
    assert result["status"] == "completed"


def test_report_contains_counts_mapping_and_caveats(tmp_path):
    base, _, _ = tiny_base(tmp_path)
    out = tmp_path / "checkpoints" / "upcycle_6l"
    report_path = tmp_path / "reports" / "upcycle_6l.json"
    upcycle(
        base=base, out=out, layers=6, mode="duplicate", seed=0,
        report=report_path)
    assert report_path.is_file()
    markdown_path = report_path.with_suffix(".md")
    assert markdown_path.is_file()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["schema"] == "atomzero.upcycle_report.v1"
    assert report["status"] == "completed"
    assert report["parameters"]["source"]["layers"] == 4
    assert report["parameters"]["target"]["layers"] == 6
    assert report["layer_mapping"] == [0, 0, 1, 2, 2, 3]
    assert report["estimates"]["chinchilla_tokens"] > 0
    assert report["estimates"]["fits_6gb"] in {"likely", "tight",
                                               "unlikely"}
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "research only" in markdown.lower()
    assert "not the product 0.8.0 checkpoint" in markdown
    assert "RTX 3050 6 GB" in markdown
    assert "layer mapping" in markdown.lower()
    assert "No prompt or answer text is stored" in markdown


def test_requires_a_tokenizer_for_a_real_run(tmp_path, monkeypatch):
    import research.upcycle_model as upcycle_model

    monkeypatch.setattr(
        upcycle_model, "DEFAULT_TOKENIZER", "data/__missing_tokenizer__.json")
    base, _, _ = tiny_base(tmp_path, tokenizer=False)
    out = tmp_path / "checkpoints" / "upcycle_6l"
    with pytest.raises(UpcycleError, match="tokenizer not found"):
        upcycle(base=base, out=out, layers=6)
    assert not out.exists()


def test_expand_state_returns_expected_keys(tmp_path):
    _, config, source_model = tiny_base(tmp_path)
    state, mapping, plan = expand_state(
        source_model.state_dict(), config, 6, "duplicate")
    assert mapping == [0, 0, 1, 2, 2, 3]
    assert plan == []
    assert len([key for key in state if key.startswith("blocks.")]) == (
        6 * len(_block_suffixes(source_model)))
    assert "blocks.5.mlp.2.weight" in state
