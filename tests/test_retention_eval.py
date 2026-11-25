import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import research.retention_eval as module
from research.retention_eval import (
    DEFAULT_MAX_ROWS,
    DEFAULT_THRESHOLD,
    _parser,
    main,
    resolve_device,
    retention_ok,
    retention_ratio,
    run_retention_eval,
)

_SPECIAL_IDS = {
    "<|user|>": 0,
    "<|assistant|>": 1,
    "<|end|>": 2,
}
_WORDS = (
    "name", "city", "Alice", "Paris", "q1", "q2", "q3", "q4",
    "plain", "answer", "alpha", "one", "beta", "two", "other", "word",
    "SECRET_PROMPT", "SECRET_KEY", "SECRET_VALUE",
)
_WORD_IDS = {word: index + 3 for index, word in enumerate(_WORDS)}
_ALL_IDS = {**_SPECIAL_IDS, **_WORD_IDS}
_UNKNOWN_ID = len(_ALL_IDS) + 2
_VOCAB = _UNKNOWN_ID + 1


class FakeTokenizer:
    def encode(self, text):
        ids = []
        offsets = []
        cursor = 0
        for part in str(text).split():
            start = str(text).find(part, cursor)
            end = start + len(part)
            ids.append(_ALL_IDS.get(part, _UNKNOWN_ID))
            offsets.append((start, end))
            cursor = end
        return SimpleNamespace(ids=ids, offsets=offsets)

    def token_to_id(self, token):
        return _ALL_IDS.get(token)

    def decode(self, ids):
        reverse = {value: key for key, value in _ALL_IDS.items()}
        reverse[_UNKNOWN_ID] = "word"
        return " ".join(reverse.get(int(item), "word") for item in ids)


TOKENIZER = FakeTokenizer()


class BiasLM(nn.Module):
    def __init__(self, bias=None, tag="baseline"):
        super().__init__()
        self.vocab = _VOCAB
        self.max_seq = 64
        self.tag = tag
        vector = torch.zeros(_VOCAB) if bias is None else torch.tensor(
            bias, dtype=torch.float)
        self.bias = nn.Parameter(vector)

    def forward(self, x):
        return self.bias.view(1, 1, -1).expand(x.size(0), x.size(1), -1)


def _bad_bias():
    vector = [0.0] * _VOCAB
    vector[_ALL_IDS["other"]] = 6.0
    return vector


def _rows():
    return [
        {"user": "q1", "assistant": "name: Alice\ncity: Paris",
         "family": "city"},
        {"user": "q2", "assistant": "alpha: one\nbeta: two",
         "family": "copy"},
        {"user": "q3", "assistant": "plain answer", "family": "plain"},
    ]


def _answers(rows):
    return {row["user"]: row["assistant"] for row in rows}


def _write_rows(tmp_path, rows, name="eval.jsonl"):
    path = tmp_path / name
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def _loader(models, calls=None):
    def load(path, *, device, tokenizer=""):
        if calls is not None:
            calls.append((str(path), device, tokenizer))
        return models[str(path)], TOKENIZER

    return load


def _generator(answers, calls=None):
    def generate(model, tok, prompt, max_new):
        if calls is not None:
            calls.append((model.tag, prompt, max_new))
        user = prompt[len("<|user|> "):-len(" <|assistant|>")]
        if model.tag == "candidate":
            return "other"
        return answers[user]

    return generate


def test_identical_model_passes_retention(tmp_path):
    rows = _rows()
    data = _write_rows(tmp_path, rows)
    model = BiasLM(tag="baseline")
    calls = []
    payload = run_retention_eval(
        "candidate", "baseline", data, device="cpu", max_rows=10,
        per_family=True,
        loader=_loader({"candidate": model, "baseline": model}),
        generator=_generator(_answers(rows), calls))

    assert payload["retention_ratio"] == pytest.approx(1.0)
    assert payload["retention_ok"] is True
    assert payload["baseline"]["content_loss"] == pytest.approx(
        payload["candidate"]["content_loss"])
    assert payload["baseline"]["exact_match"]["rate"] == pytest.approx(1.0)
    assert payload["candidate"]["exact_match"]["rate"] == pytest.approx(1.0)
    assert payload["baseline"]["key_copy"]["examples"] == 2
    assert payload["baseline"]["key_copy"]["skipped"] == 1
    assert payload["baseline"]["key_copy"]["rate"] == pytest.approx(1.0)
    assert payload["candidate"]["key_copy"]["rate"] == pytest.approx(1.0)
    # exact-match and key-copy share one generation per row per model
    assert len(calls) == len(rows) * 2


def test_worse_candidate_fails_retention_ratio(tmp_path):
    rows = _rows()
    data = _write_rows(tmp_path, rows)
    loader = _loader({
        "baseline": BiasLM(tag="baseline"),
        "candidate": BiasLM(_bad_bias(), tag="candidate"),
    })
    payload = run_retention_eval(
        "candidate", "baseline", data, device="cpu",
        loader=loader, generator=_generator(_answers(rows)))

    assert payload["candidate"]["content_loss"] > (
        payload["baseline"]["content_loss"])
    assert payload["retention_ratio"] > DEFAULT_THRESHOLD
    assert payload["retention_ok"] is False
    assert payload["candidate"]["exact_match"]["rate"] == pytest.approx(0.0)
    assert payload["candidate"]["key_copy"]["rate"] == pytest.approx(0.0)
    assert payload["candidate"]["key_copy"]["key_recall"] == pytest.approx(0.0)

    lenient = run_retention_eval(
        "candidate", "baseline", data, device="cpu", threshold=100.0,
        loader=loader, generator=_generator(_answers(rows)))
    assert lenient["retention_ok"] is True


def test_per_family_aggregation_is_token_weighted(tmp_path):
    rows = _rows()
    data = _write_rows(tmp_path, rows)
    model = BiasLM(tag="baseline")
    payload = run_retention_eval(
        "candidate", "baseline", data, device="cpu", per_family=True,
        loader=_loader({"candidate": model, "baseline": model}),
        generator=_generator(_answers(rows)))

    families = payload["families"]
    assert list(families) == ["city", "copy", "plain"]
    for family in families.values():
        assert set(family) == {
            "baseline", "candidate", "retention_ratio", "retention_ok"}
        assert family["retention_ratio"] == pytest.approx(1.0)
        assert family["retention_ok"] is True
        assert family["baseline"]["content_tokens"] > 0
        assert family["candidate"]["content_tokens"] == (
            family["baseline"]["content_tokens"])
    assert families["city"]["baseline"]["exact_match"]["rate"] == 1.0
    assert families["city"]["baseline"]["key_copy"]["rate"] == 1.0
    assert families["plain"]["baseline"]["key_copy"] is None

    content_tokens = sum(
        family["baseline"]["content_tokens"] for family in families.values())
    weighted = sum(
        family["baseline"]["content_loss"]
        * family["baseline"]["content_tokens"]
        for family in families.values()) / content_tokens
    assert payload["baseline"]["content_loss"] == pytest.approx(weighted)


def test_output_is_content_free(tmp_path):
    rows = [{
        "user": "SECRET_PROMPT",
        "assistant": "SECRET_KEY: SECRET_VALUE",
        "family": "secret_family",
    }]
    data = _write_rows(tmp_path, rows)
    model = BiasLM(tag="baseline")
    payload = run_retention_eval(
        "candidate", "baseline", data, device="cpu", per_family=True,
        loader=_loader({"candidate": model, "baseline": model}),
        generator=_generator(_answers(rows)))

    blob = json.dumps(payload)
    assert payload["content_free"] is True
    assert payload["rows"] == 1
    assert "SECRET_PROMPT" not in blob
    assert "SECRET_KEY" not in blob
    assert "SECRET_VALUE" not in blob
    assert "rows" not in payload["baseline"]
    assert payload["baseline"]["exact_match"]["rate"] == 1.0


def test_family_order_is_deterministic_and_json_stable(tmp_path):
    rows = [
        {"user": "q1", "assistant": "name: Alice", "family": "zeta"},
        {"user": "q2", "assistant": "alpha: one", "family": "alpha"},
    ]
    data = _write_rows(tmp_path, rows)
    model = BiasLM(tag="baseline")
    loader = _loader({"candidate": model, "baseline": model})
    first = run_retention_eval(
        "candidate", "baseline", data, device="cpu", per_family=True,
        loader=loader, generator=_generator(_answers(rows)))
    second = run_retention_eval(
        "candidate", "baseline", data, device="cpu", per_family=True,
        loader=loader, generator=_generator(_answers(rows)))

    assert list(first["families"]) == ["alpha", "zeta"]
    assert json.dumps(first, sort_keys=True) == json.dumps(
        second, sort_keys=True)


def test_missing_keys_are_skipped_without_crashing(tmp_path):
    rows = [
        {"user": "q1", "assistant": "plain answer", "family": "plain"},
        {"user": "q2", "assistant": "other word", "family": "plain"},
    ]
    data = _write_rows(tmp_path, rows)
    model = BiasLM(tag="baseline")
    payload = run_retention_eval(
        "candidate", "baseline", data, device="cpu", per_family=True,
        loader=_loader({"candidate": model, "baseline": model}),
        generator=_generator(_answers(rows)))

    key_copy = payload["baseline"]["key_copy"]
    assert key_copy["examples"] == 0
    assert key_copy["skipped"] == 2
    assert key_copy["matches"] == 0
    assert key_copy["rate"] is None
    assert key_copy["key_recall"] is None
    assert payload["families"]["plain"]["baseline"]["key_copy"] is None
    assert payload["retention_ratio"] == pytest.approx(1.0)


def test_max_rows_truncates_combined_file_stream(tmp_path):
    first = _write_rows(tmp_path, _rows()[:2], name="first.jsonl")
    second = _write_rows(tmp_path, _rows()[2:], name="second.jsonl")
    model = BiasLM(tag="baseline")
    payload = run_retention_eval(
        "candidate", "baseline", [first, second], device="cpu", max_rows=2,
        per_family=True,
        loader=_loader({"candidate": model, "baseline": model}),
        generator=_generator(_answers(_rows())))

    assert payload["rows"] == 2
    assert list(payload["families"]) == ["city", "copy"]


def test_retention_ratio_and_threshold_helpers():
    assert retention_ratio(2.0, 1.0) == pytest.approx(2.0)
    assert retention_ratio(1.05, 1.0) == pytest.approx(1.05)
    assert retention_ratio(None, 1.0) is None
    assert retention_ratio(1.0, 0.0) is None
    assert retention_ratio(1.0, None) is None
    assert retention_ok(1.10, 1.10) is True
    assert retention_ok(1.11, 1.10) is False
    assert retention_ok(None, 1.10) is None


def test_resolve_device_keeps_explicit_choice():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda") == "cuda"


def test_cli_defaults_and_flags():
    parser = _parser()
    args = parser.parse_args([
        "--candidate", "cand", "--baseline", "base",
        "--data", "a.jsonl", "b.jsonl",
    ])
    assert args.candidate == "cand"
    assert args.baseline == "base"
    assert args.data == ["a.jsonl", "b.jsonl"]
    assert args.out is None
    assert args.max_rows == DEFAULT_MAX_ROWS
    assert args.device == "auto"
    assert args.per_family is False
    assert args.threshold == pytest.approx(DEFAULT_THRESHOLD)
    assert args.exact_match_max_new == 64

    args = parser.parse_args([
        "--candidate", "cand", "--baseline", "base", "--data", "a.jsonl",
        "--out", "out.json", "--max-rows", "5", "--device", "cpu",
        "--per-family", "--threshold", "1.05",
        "--exact-match-max-new", "16",
    ])
    assert args.out == "out.json"
    assert args.max_rows == 5
    assert args.device == "cpu"
    assert args.per_family is True
    assert args.threshold == pytest.approx(1.05)
    assert args.exact_match_max_new == 16


def test_main_writes_content_free_json(tmp_path, monkeypatch, capsys):
    payload = {
        "schema": "atomzero.retention_eval.v1",
        "content_free": True,
        "rows": 3,
        "retention_ratio": 1.0,
        "retention_ok": True,
        "families": {"city": {"retention_ratio": 1.0}},
    }
    calls = {}

    def fake_run(candidate, baseline, data, **kwargs):
        calls.update({
            "candidate": candidate,
            "baseline": baseline,
            "data": data,
            **kwargs,
        })
        return payload

    monkeypatch.setattr(module, "run_retention_eval", fake_run)
    out = tmp_path / "nested" / "report.json"
    code = main([
        "--candidate", "cand", "--baseline", "base",
        "--data", "a.jsonl", "b.jsonl", "--out", str(out),
        "--max-rows", "3", "--device", "cpu", "--per-family",
        "--threshold", "1.05",
    ])

    assert code == 0
    assert json.loads(out.read_text(encoding="utf-8")) == payload
    assert json.loads(capsys.readouterr().out) == payload
    assert calls["candidate"] == "cand"
    assert calls["baseline"] == "base"
    assert calls["data"] == ["a.jsonl", "b.jsonl"]
    assert calls["max_rows"] == 3
    assert calls["device"] == "cpu"
    assert calls["per_family"] is True
    assert calls["threshold"] == pytest.approx(1.05)
