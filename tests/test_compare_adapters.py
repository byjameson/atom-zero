import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from research.compare_adapters import (answers_match, assistant_label_mask,
                                       exact_match_metrics, label_for,
                                       normalize_answer, per_family_metrics,
                                       per_family_summary, score_rows)


class DummyTokenizer:
    ids = {
        "<|user|>": 0,
        "<|assistant|>": 1,
        "<|end|>": 2,
        "hello": 3,
        "hi": 4,
        "bye": 5,
    }

    def token_to_id(self, token):
        return self.ids.get(token)

    def encode(self, text):
        return SimpleNamespace(
            ids=[self.ids.get(word, 6) for word in text.split()])

    def decode(self, ids):
        reverse = {value: key for key, value in self.ids.items()}
        return " ".join(reverse.get(int(item), "?") for item in ids)


class DummyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.vocab = 7
        self.max_seq = 32
        self.bias = nn.Parameter(torch.zeros(self.vocab))

    def forward(self, x):
        return self.bias.view(1, 1, -1).expand(x.size(0), x.size(1), -1)


def test_assistant_mask_covers_tag_answer_and_end():
    # y starts after <|user|>: hello <|assistant|> hi <|end|>
    y = torch.tensor([[3, 4, 1, 5, 6, 2, 7]])
    mask = assistant_label_mask(y, assistant_id=1, end_id=2)
    assert mask.tolist() == [[False, False, True, True, True, True, False]]


def test_score_rows_reports_assistant_only_loss():
    rows = [
        {"user": "hello", "assistant": "hi"},
        {"user": "bye", "assistant": "hello"},
    ]
    result = score_rows(DummyLM(), DummyTokenizer(), rows)
    assert result["examples"] == 2
    assert result["tokens"] == 6
    assert result["loss"] == pytest.approx(math.log(7.0), abs=1e-6)
    assert len(result["rows"]) == 2
    assert all(row["tokens"] == 3 for row in result["rows"])
    assert all(not row["truncated"] for row in result["rows"])


def test_label_for_adapter_path():
    assert label_for("checkpoints/adapters_feedback/v0002") == (
        "adapters_feedback/v0002")
    assert label_for("checkpoints/my_lora") == "my_lora"


def test_score_rows_keeps_family_labels_per_row():
    rows = [{"user": "hello", "assistant": "hi", "family": "greeting",
             "key_family": "kv"}]
    result = score_rows(DummyLM(), DummyTokenizer(), rows)
    assert result["rows"][0]["family"] == "greeting"
    assert result["rows"][0]["key_family"] == "kv"


def _metrics(rows_by_family):
    rows = []
    for family, values in rows_by_family.items():
        for loss, tokens, content_loss, content_tokens in values:
            rows.append({
                "family": family,
                "loss": loss,
                "tokens": tokens,
                "content_loss": content_loss,
                "content_tokens": content_tokens,
                "user": "hidden",
                "assistant": "hidden",
            })
    return {"rows": rows}


def test_per_family_summary_aggregates_weighted_rows():
    baseline = _metrics({
        "json": [(2.0, 10, 4.0, 4), (4.0, 30, 6.0, 6)],
        "kv": [(1.0, 5, None, 0)],
    })
    adapter = _metrics({
        "json": [(1.0, 20, 2.0, 5)],
        "kv": [(3.0, 5, 4.0, 5)],
    })
    families = per_family_summary(baseline, adapter)
    assert set(families) == {"json", "kv"}
    json_base = families["json"]["baseline"]
    assert json_base["loss"] == pytest.approx((2.0 * 10 + 4.0 * 30) / 40)
    assert json_base["ppl"] == pytest.approx(math.exp(3.5))
    assert json_base["content_loss"] == pytest.approx(
        (4.0 * 4 + 6.0 * 6) / 10)
    assert families["json"]["adapter"]["tokens"] == 20
    assert families["json"]["adapter"]["content_loss"] == pytest.approx(2.0)
    assert families["kv"]["baseline"]["content_loss"] is None
    assert families["kv"]["adapter"]["loss"] == pytest.approx(3.0)


def test_per_family_metrics_ignores_rows_without_family():
    metrics = _metrics({"json": [(1.0, 2, None, 0)]})
    metrics["rows"].append({"loss": 1.0, "tokens": 2})
    assert set(per_family_metrics(metrics)) == {"json"}


def test_answers_match_json_dict_equality():
    assert answers_match(
        '{"name": "Alice", "age": 30}',
        '{"age": 30, "name": "Alice"}')
    assert not answers_match(
        '{"name": "Alice", "age": 30}',
        '{"name": "Alice", "age": 31}')
    assert not answers_match('{"name": "Alice"}', "name: Alice")
    assert not answers_match('{"name": "Alice"}', "not json")


def test_answers_match_normalizes_kv_yaml_and_plain_whitespace():
    assert answers_match(
        "name: Alice\nage: 30", "name:Alice\n\nage : 30  ")
    assert answers_match("- name: Alice\n- age: 30",
                         "- name:Alice\n- age:30")
    assert answers_match("hello   world", "hello world")
    assert answers_match("hello\nworld", "hello world")
    assert not answers_match("name: Alice", "name: Bob")
    assert normalize_answer("name : Alice") == "name:Alice"


def test_exact_match_metrics_with_stub_generator():
    rows = [
        {"user": "q1", "assistant": '{"a": 1}', "family": "json"},
        {"user": "q2", "assistant": "name: Alice", "family": "kv"},
        {"user": "q3", "assistant": "plain answer"},
    ]
    answers = {
        "<|user|> q1 <|assistant|>": '{"a": 1}',
        "<|user|> q2 <|assistant|>": "name:Alice",
        "<|user|> q3 <|assistant|>": "wrong",
    }
    calls = []

    def fake_generator(model, tok, prompt, max_new):
        calls.append((prompt, max_new))
        return answers[prompt]

    result = exact_match_metrics(
        DummyLM(), DummyTokenizer(), rows, generator=fake_generator,
        max_new=16)
    assert result["examples"] == 3
    assert result["matches"] == 2
    assert result["rate"] == pytest.approx(2 / 3)
    assert result["families"]["json"]["rate"] == 1.0
    assert result["families"]["kv"]["rate"] == 1.0
    assert all(4 <= max_new <= 16 for _, max_new in calls)
    assert all(prompt.startswith("<|user|> ") for prompt, _ in calls)


def test_exact_match_metrics_returns_no_rate_for_empty_rows():
    result = exact_match_metrics(
        DummyLM(), DummyTokenizer(), [],
        generator=lambda *args: "")
    assert result["examples"] == 0
    assert result["rate"] is None


def test_compare_emits_families_and_exact_match(monkeypatch, tmp_path):
    import research.compare_adapters as module

    data = tmp_path / "rows.jsonl"
    data.write_text(
        '{"user": "hello", "assistant": "hi", "family": "greeting"}\n'
        '{"user": "bye", "assistant": "hello", "family": "greeting"}\n',
        encoding="utf-8")
    monkeypatch.setattr(
        module, "load_model",
        lambda *args, **kwargs: (DummyLM(), DummyTokenizer()))
    payload = module.compare(
        "base", data, ["adapter"], per_family=True, exact_match=True,
        generator=lambda model, tok, prompt, max_new: "hi")
    adapter = payload["models"][1]
    assert set(adapter["families"]) == {"greeting"}
    assert adapter["families"]["greeting"]["baseline"]["examples"] == 2
    assert adapter["families"]["greeting"]["adapter"]["loss"] == (
        pytest.approx(adapter["metrics"]["loss"]))
    assert adapter["metrics"]["exact_match"]["rate"] == pytest.approx(0.5)
    assert adapter["metrics"]["exact_match"]["families"]["greeting"][
        "rate"] == pytest.approx(0.5)
    assert payload["families"] == adapter["families"]
    assert "families" not in payload["models"][0]
    assert payload["models"][0]["metrics"]["exact_match"]["rate"] == (
        pytest.approx(0.5))
