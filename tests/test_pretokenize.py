import json
import math

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit

from atomzero.transformer import TransformerLM
from research.adapter_runtime import load_model
from research.finetune_copy_control import train_control
from research.pretokenize import (
    PAYLOAD_FILES,
    SCHEMA,
    build_cache,
    file_sha256,
    load_cache,
    load_training_cache,
    main,
)
from research.train_adapter_candidate import _encode, _token_weights

VOCAB = ["<|user|>", "<|assistant|>", "<|end|>", "hello", "hi", "bye",
         "alpha", "beta", "SECRETPROMPT", "SECRETANSWER", "[UNK]"]
ROWS = [
    {"user": "hello", "assistant": "hi"},
    {"user": "bye", "assistant": "hello"},
    {"user": "alpha", "assistant": "beta"},
]


def _tiny_tokenizer(path):
    tokenizer = Tokenizer(WordLevel(
        vocab={word: index for index, word in enumerate(VOCAB)},
        unk_token="[UNK]"))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    tokenizer.save(str(path))
    return tokenizer


def _tiny_base(tmp_path):
    tokenizer = _tiny_tokenizer(tmp_path / "tokenizer.json")
    base = tmp_path / "base"
    base.mkdir()
    tokenizer.save(str(base / "chat_en_bpe8k.json"))
    config = {"vocab": len(VOCAB), "dim": 16, "layers": 2, "heads": 2,
              "ff": 32, "seq": 32, "arch": "transformer"}
    torch.manual_seed(0)
    model = TransformerLM(config["vocab"], config["dim"], config["layers"],
                          config["heads"], config["ff"])
    torch.save({"config": config, "state": model.state_dict()},
               base / "model.pt")
    return base, tokenizer


def _build(tmp_path, rows=ROWS, *, name="cache", max_seq=32,
           content_weight=5.0, tokenizer=None):
    if tokenizer is None:
        tokenizer = _tiny_tokenizer(tmp_path / "tokenizer.json")
    out = tmp_path / name
    meta = build_cache(
        rows, tokenizer, out, max_seq=max_seq,
        content_weight=content_weight,
        tokenizer_sha256=file_sha256(tmp_path / "tokenizer.json"),
        tokenizer_name="tokenizer.json")
    return out, meta, tokenizer


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_cache_is_deterministic(tmp_path):
    first, _, tok = _build(tmp_path, name="first")
    second, _, _ = _build(tmp_path, name="second", tokenizer=tok)
    assert (first / "meta.json").read_bytes() == (
        second / "meta.json").read_bytes()
    for name in PAYLOAD_FILES:
        assert (first / name).read_bytes() == (second / name).read_bytes()
    meta = json.loads((first / "meta.json").read_text(encoding="utf-8"))
    assert meta["schema"] == SCHEMA
    assert meta["purpose"] == "training_only"
    assert meta["sha256"] == json.loads(
        (second / "meta.json").read_text(encoding="utf-8"))["sha256"]


def test_metadata_is_content_free(tmp_path):
    rows = [
        {"user": "SECRETPROMPT one", "assistant": "SECRETANSWER one"},
        {"user": "SECRETPROMPT two", "assistant": "SECRETANSWER two"},
    ]
    out, _, _ = _build(tmp_path, rows=rows)
    text = (out / "meta.json").read_text(encoding="utf-8")
    assert "SECRETPROMPT" not in text
    assert "SECRETANSWER" not in text
    meta = json.loads(text)
    assert meta["cached_rows"] == 2
    assert meta["skipped_rows"] == 0
    assert meta["rows_seen"] == 2
    assert meta["max_seq"] == 32
    assert meta["content_weight"] == 5.0
    assert len(meta["tokenizer"]["sha256"]) == 64
    assert len(meta["sha256"]) == 64
    assert set(meta["dtypes"]) == set(PAYLOAD_FILES)


def test_skip_count_and_manifest_are_reported(tmp_path):
    rows = ROWS + [
        {"user": "<|user|> <|user|>", "assistant": "hi"},
    ]
    out, meta, _ = _build(tmp_path, rows=rows, max_seq=3,
                          content_weight=1.0)
    assert meta["cached_rows"] == len(ROWS)
    assert meta["skipped_rows"] == 1
    assert meta["rows_seen"] == len(rows)
    assert meta["skip_reasons"]["no_scorable_token"] == 1
    assert meta["skip_reasons"]["too_short"] == 0
    assert len(load_cache(out)) == len(ROWS)


def test_cache_roundtrip_matches_encoder_and_weights(tmp_path):
    out, _, tok = _build(tmp_path)
    cache = load_cache(out)
    assert len(cache) == len(ROWS)
    assert cache.lengths() == [
        len(tok.encode(f"<|user|> {row['user']} <|assistant|> "
                       f"{row['assistant']} <|end|>").ids) - 1
        for row in ROWS]
    for index, row in enumerate(ROWS):
        x, y, weights = cache[index]
        expected_x, expected_y = _encode(tok, row, 32, torch.device("cpu"))
        expected_weights = _token_weights(tok, expected_y, 5.0)
        assert x.shape == expected_x.shape
        assert y.shape == expected_y.shape
        assert weights.shape == expected_weights.shape
        assert torch.equal(x, expected_x)
        assert torch.equal(y, expected_y)
        assert torch.allclose(weights, expected_weights)
        assert x.dtype == torch.long
        assert y.dtype == torch.long
        assert weights.dtype == torch.float32


def test_trainer_accepts_cache_and_matches_on_the_fly_loss(tmp_path):
    base, tokenizer = _tiny_base(tmp_path)
    out, _, _ = _build(tmp_path, tokenizer=tokenizer)
    cache = load_cache(out)
    on_the_fly_model, tok = load_model(base)
    cached_model, cached_tok = load_model(base)
    assert all(torch.equal(a, b) for a, b in zip(
        on_the_fly_model.parameters(), cached_model.parameters()))
    common = dict(
        steps=1, lr=1e-2, batch_size=2, max_seq=32, content_weight=5.0,
        seed=0, freeze="none", device="cpu", log_every=0)
    on_the_fly = train_control(on_the_fly_model, tok, ROWS, **common)
    cached = train_control(
        cached_model, cached_tok, [], cache=cache, **common)
    assert cached["cache_rows"] == len(ROWS)
    assert cached["cache_sha256"] == cache.meta["sha256"]
    assert cached["train_examples"] == len(ROWS)
    assert math.isfinite(on_the_fly["train_loss"])
    assert cached["train_loss"] == pytest.approx(
        on_the_fly["train_loss"], rel=1e-6, abs=1e-7)
    for first, second in zip(on_the_fly_model.parameters(),
                             cached_model.parameters()):
        assert torch.allclose(first, second, atol=1e-7, rtol=1e-6)


def test_training_cache_checks_settings(tmp_path):
    out, _, _ = _build(tmp_path, max_seq=32, content_weight=5.0)
    cache = load_training_cache(out, max_seq=32, content_weight=5.0)
    assert len(cache) == len(ROWS)
    with pytest.raises(ValueError, match="max_seq"):
        load_training_cache(out, max_seq=16, content_weight=5.0)
    with pytest.raises(ValueError, match="content_weight"):
        load_training_cache(out, max_seq=32, content_weight=2.0)


def test_bad_cache_paths_error_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="cache directory not found"):
        load_cache(tmp_path / "missing")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="meta.json is missing"):
        load_cache(empty)

    out, _, _ = _build(tmp_path, name="corrupt")
    meta_path = out / "meta.json"
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    payload["schema"] = "wrong.schema"
    meta_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        load_cache(out)

    out, _, _ = _build(tmp_path, name="modified")
    with (out / "weights.npy").open("ab") as handle:
        handle.write(b"\0")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_cache(out)


def test_cli_builds_cache_and_manifest(tmp_path, capsys):
    base, _ = _tiny_base(tmp_path)
    data = tmp_path / "train.jsonl"
    rows = ROWS + [
        {"user": "<|user|> <|user|>", "assistant": "hi"},
    ]
    _write_jsonl(data, rows)
    out = tmp_path / "cache"
    code = main([
        "--data", str(data),
        "--tokenizer", str(base / "chat_en_bpe8k.json"),
        "--out", str(out),
        "--max-seq", "3",
        "--content-weight", "1.0",
        "--manifest",
    ])
    assert code == 0
    summary = json.loads(capsys.readouterr().out.strip())
    assert summary["cached_rows"] == len(ROWS)
    assert summary["skipped_rows"] == 1
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["data"]["sha256"] == file_sha256(data)
    assert meta["tokenizer"]["sha256"] == file_sha256(
        base / "chat_en_bpe8k.json")
    manifest = json.loads(
        (out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["cached_rows"] == len(ROWS)
    assert manifest["skipped_rows"] == 1
    assert [row["kept"] for row in manifest["rows"]] == [
        True, True, True, False]
    assert "hello" not in (out / "manifest.json").read_text(encoding="utf-8")


def test_payload_arrays_are_compact_dtypes(tmp_path):
    out, _, _ = _build(tmp_path)
    expected = {
        "input_ids.npy": np.dtype("int32"),
        "labels.npy": np.dtype("int32"),
        "weights.npy": np.dtype("float16"),
        "lengths.npy": np.dtype("int32"),
    }
    for name, dtype in expected.items():
        array = np.load(out / name, allow_pickle=False)
        assert array.dtype == dtype
        assert array.ndim == 1
