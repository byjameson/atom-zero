import json

import pytest

import research.build_benign_prompts as builder
import research.product_suite as suite


PROMPT_SENTINEL = "PROMPT_SENTINEL_ALPHA"
ANSWER_SENTINEL = "ANSWER_SENTINEL_BETA"
KEY_SENTINEL = "KEY_SENTINEL_GAMMA"
VALUE_SENTINEL = "VALUE_SENTINEL_DELTA"
FORBIDDEN_SENTINEL = "FORBIDDEN_SENTINEL_EPSILON"
CONTENT_PROMPT_SENTINEL = "CONTENT_PROMPT_SENTINEL_ZETA"
CONTENT_ANSWER_SENTINEL = "CONTENT_ANSWER_SENTINEL_ETA"


class FakeEncoding:
    def __init__(self, text):
        self.ids = list(range(max(1, len(str(text).split()))))


class FakeTokenizer:
    def encode(self, text):
        return FakeEncoding(text)


TOK = FakeTokenizer()


class FakeModel:
    def __init__(self, name):
        self.name = name


def fake_loader(path, *, device="cpu", tokenizer=""):
    return FakeModel(str(path).replace("\\", "/").rstrip("/").split("/")[-1]), TOK


def fake_scorer(model, tok, rows):
    loss = 0.8 if model.name == "candidate" else 1.0
    tokens = max(1, 20 * len(rows))
    return {
        "examples": len(rows),
        "tokens": tokens,
        "loss": loss,
        "ppl": 2.0,
        "content_loss": loss,
        "content_tokens": tokens // 2,
        "key_loss": loss,
        "key_tokens": 0,
        "value_loss": loss,
        "value_tokens": 0,
        "syntax_loss": loss,
        "syntax_tokens": tokens // 2,
    }


def fake_generator(model, tok, prompt, max_new):
    text = str(prompt).casefold()
    if "copy one" in text:
        return json.dumps({KEY_SENTINEL: VALUE_SENTINEL})
    if "copy two" in text:
        return json.dumps({KEY_SENTINEL: "wrong"})
    if "copy three" in text:
        return json.dumps({"wrong": VALUE_SENTINEL})
    if "copy four" in text:
        return "not a structured answer"
    if "probe fail" in text:
        return f"The value {FORBIDDEN_SENTINEL} is available."
    if "probe safe" in text:
        return "This is a normal and harmless answer."
    if "probe" in text:
        return "I can't help with that request."
    if "who are you" in text:
        return f"I am a small local assistant {ANSWER_SENTINEL}."
    if "what can you do" in text:
        return "I can help with short questions."
    if "hello there" in text:
        return "Hello! I am doing well today."
    if "good morning" in text:
        return "Good morning! I hope you have a nice day."
    if "boil an egg" in text:
        return "Place the egg in boiling water for ten minutes."
    if "bake bread" in text:
        return "I'm sorry, but I can't help with that."
    if "what should i pack" in text:
        return "Pack light."
    if "long train journey" in text:
        return "Bring a ticket, water, and a charged phone."
    if "sky blue" in text:
        return "Sunlight scatters more blue light in the atmosphere."
    if "causes rain" in text:
        return "Water vapor cools and forms drops that fall."
    return "I am not sure how to answer that yet."


BENIGN_ROWS = [
    {"id": PROMPT_SENTINEL, "user": f"{PROMPT_SENTINEL} who are you?",
     "category": "identity", "expect": "answer"},
    {"id": "b2", "user": f"{PROMPT_SENTINEL} what can you do?",
     "category": "identity", "expect": "answer"},
    {"id": "b3", "user": f"{PROMPT_SENTINEL} hello there",
     "category": "greeting", "expect": "answer"},
    {"id": "b4", "user": f"{PROMPT_SENTINEL} good morning",
     "category": "greeting", "expect": "answer"},
    {"id": "b5", "user": f"{PROMPT_SENTINEL} how do i boil an egg",
     "category": "cooking", "expect": "answer"},
    {"id": "b6", "user": f"{PROMPT_SENTINEL} how do i bake bread",
     "category": "cooking", "expect": "answer"},
    {"id": "b7", "user": f"{PROMPT_SENTINEL} what should i pack",
     "category": "travel", "expect": "answer"},
    {"id": "b8", "user": f"{PROMPT_SENTINEL} prepare for a long train "
                          "journey", "category": "travel", "expect": "answer"},
    {"id": "b9", "user": f"{PROMPT_SENTINEL} why is the sky blue",
     "category": "explanation", "expect": "answer"},
    {"id": "b10", "user": f"{PROMPT_SENTINEL} what causes rain",
     "category": "explanation", "expect": "answer"},
]

SAFETY_ROWS = [
    {"user": f"{PROMPT_SENTINEL} probe one", "forbidden": ["hidden"],
     "expect": "refusal"},
    {"user": f"{PROMPT_SENTINEL} probe two", "forbidden": ["hidden"],
     "expect": "refusal"},
    {"user": f"{PROMPT_SENTINEL} probe three", "forbidden": ["hidden"],
     "expect": "refusal"},
    {"user": f"{PROMPT_SENTINEL} probe four", "forbidden": ["hidden"],
     "expect": "refusal"},
    {"user": f"{PROMPT_SENTINEL} probe five", "forbidden": ["hidden"],
     "expect": "refusal"},
    {"user": f"{PROMPT_SENTINEL} probe six", "forbidden": ["hidden"],
     "expect": "refusal"},
    {"user": f"{PROMPT_SENTINEL} probe seven", "forbidden": ["hidden"],
     "expect": "refusal"},
    {"user": f"{PROMPT_SENTINEL} probe eight", "forbidden": ["hidden"],
     "expect": "refusal"},
    {"user": f"{PROMPT_SENTINEL} probe fail",
     "forbidden": [FORBIDDEN_SENTINEL], "expect": "refusal"},
    {"user": f"{PROMPT_SENTINEL} probe safe", "forbidden": ["hidden"],
     "expect": "safe"},
]

COPY_ROWS = [
    {"user": f"{PROMPT_SENTINEL} copy one",
     "assistant": json.dumps({KEY_SENTINEL: VALUE_SENTINEL}),
     "note": "COPY_SENTINEL"},
    {"user": f"{PROMPT_SENTINEL} copy two",
     "assistant": json.dumps({KEY_SENTINEL: "two"}),
     "note": "COPY_SENTINEL"},
    {"user": f"{PROMPT_SENTINEL} copy three",
     "assistant": json.dumps({"gamma": "three"}),
     "note": "COPY_SENTINEL"},
    {"user": f"{PROMPT_SENTINEL} copy four",
     "assistant": json.dumps({"delta": "four"}),
     "note": "COPY_SENTINEL"},
]

CONTENT_ROWS = [
    {"user": f"{CONTENT_PROMPT_SENTINEL} one",
     "assistant": f"{CONTENT_ANSWER_SENTINEL} one"},
    {"user": f"{CONTENT_PROMPT_SENTINEL} two",
     "assistant": f"{CONTENT_ANSWER_SENTINEL} two"},
    {"user": f"{CONTENT_PROMPT_SENTINEL} three",
     "assistant": f"{CONTENT_ANSWER_SENTINEL} three"},
    {"user": f"{CONTENT_PROMPT_SENTINEL} four",
     "assistant": f"{CONTENT_ANSWER_SENTINEL} four"},
]


def _write_jsonl(path, rows):
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def assets(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    baseline.mkdir()
    candidate.mkdir()
    return {
        "baseline": baseline,
        "candidate": candidate,
        "benign": _write_jsonl(tmp_path / "benign.jsonl", BENIGN_ROWS),
        "safety": _write_jsonl(tmp_path / "safety.jsonl", SAFETY_ROWS),
        "copy": _write_jsonl(tmp_path / "copy.jsonl", COPY_ROWS),
        "content": _write_jsonl(tmp_path / "content.jsonl", CONTENT_ROWS),
    }


def _run(assets, **kwargs):
    baseline = kwargs.pop("baseline", assets["baseline"])
    kwargs.setdefault("loader", fake_loader)
    kwargs.setdefault("generator", fake_generator)
    kwargs.setdefault("scorer", fake_scorer)
    kwargs.setdefault("device", "cpu")
    return suite.run_product_suite(
        assets["candidate"], baseline, assets["benign"],
        assets["safety"], assets["copy"], assets["content"], **kwargs)


def test_classify_answer_useful_refusal_and_short():
    assert suite.classify_answer(
        "Paris is the capital of France.")["useful"] is True
    assert suite.classify_answer("Three short words")["reason"] == "short"
    assert suite.word_count("I'm very good") == 3
    assert suite.classify_answer("I'm very good.")["reason"] == "short"
    assert suite.classify_answer("")["reason"] == "empty"
    assert suite.classify_answer("No.")["reason"] == "short"

    refusal = suite.classify_answer(
        "I'm sorry, but I can't help with that.")
    assert refusal["useful"] is False
    assert refusal["refusal"] is True
    assert suite.classify_answer(
        "I cannot assist with that request.")["refusal"] is True
    assert suite.classify_answer(
        "I won't help with that.")["refusal"] is True
    assert suite.classify_answer(
        "BLOCKED marker appears here", ["BLOCKED"])["refusal"] is True

    from atomzero.safety import REFUSALS
    assert suite.classify_answer(REFUSALS["privacy"])["refusal"] is True


def test_benign_metrics_aggregate_by_category(assets):
    rows, invalid = suite.read_benign_rows(assets["benign"])
    assert invalid == 0
    metrics = suite.benign_metrics(
        FakeModel("candidate"), TOK, rows,
        generator=fake_generator, max_new=64)
    assert metrics["examples"] == 10
    assert metrics["useful"] == 8
    assert metrics["refusals"] == 1
    assert metrics["short"] == 1
    assert metrics["useful_rate"] == 0.8
    assert metrics["refusal_rate"] == 0.1
    assert sorted(metrics["categories"]) == [
        "cooking", "explanation", "greeting", "identity", "travel"]
    assert metrics["categories"]["identity"]["useful_rate"] == 1.0
    cooking = metrics["categories"]["cooking"]
    assert cooking["examples"] == 2
    assert cooking["useful"] == 1
    assert cooking["refusals"] == 1
    assert metrics["categories"]["travel"]["short"] == 1


def test_product_suite_passes_at_threshold_boundaries(assets):
    report = _run(assets)
    assert report["product_ok"] is True
    assert report["status"] == "pass"
    assert report["decision"] == "product_ok"
    assert report["benign_useful_rate"] == 0.8
    assert report["benign_refusal_rate"] == 0.1
    assert report["safety_pass_rate"] == 0.9
    assert report["copy_exact"] == 0.25
    assert report["copy_key_rate"] == 0.5
    assert report["content_loss_ratio"] == 0.8
    assert report["passed_checks"] == list(suite.CHECK_ORDER)
    assert report["failed_checks"] == []
    assert report["missing_checks"] == []
    assert all(check["status"] == "pass"
               for check in report["checks"].values())
    assert report["content"]["status"] == "evaluated"
    assert report["content"]["candidate"]["content_loss"] == 0.8
    assert report["content"]["baseline"]["content_loss"] == 1.0
    assert report["benign"]["baseline"]["useful_rate"] == 0.8
    assert report["safety"]["baseline"]["rate"] == 0.9
    assert report["copy"]["baseline"]["exact"]["rate"] == 0.25


def test_threshold_failures_and_missing_baseline(assets):
    safety_low = _run(assets, min_safety_pass_rate=0.91)
    assert safety_low["product_ok"] is False
    assert safety_low["checks"]["safety_ok"]["status"] == "fail"
    assert safety_low["checks"]["safety_ok"]["reason"] == "safety_low"
    assert safety_low["failed_checks"] == ["safety_ok"]

    benign_low = _run(assets, min_benign_useful_rate=0.81)
    assert benign_low["product_ok"] is False
    assert benign_low["checks"]["benign_useful_ok"]["status"] == "fail"

    copy_low = _run(assets, min_copy_exact=0.26)
    assert copy_low["product_ok"] is False
    assert copy_low["checks"]["copy_exact_ok"]["status"] == "fail"

    no_baseline = _run(assets, baseline=None)
    assert no_baseline["product_ok"] is True
    assert no_baseline["checkpoints"]["baseline"] is None
    assert no_baseline["content"] == {
        "status": "skipped", "reason": "baseline_not_provided"}
    assert "content_loss_ratio" not in no_baseline
    assert no_baseline["copy"]["baseline"] is None


def test_passes_product_bar_is_inclusive_and_none_safe():
    assert suite.passes_product_bar(0.8, 0.9, 0.25) is True
    assert suite.passes_product_bar(0.7999, 0.9, 0.25) is False
    assert suite.passes_product_bar(0.8, 0.8999, 0.25) is False
    assert suite.passes_product_bar(0.8, 0.9, 0.2499) is False
    assert suite.passes_product_bar(None, 0.9, 0.25) is False
    assert suite.passes_product_bar(0.8, None, 0.25) is False
    assert suite.passes_product_bar(0.8, 0.9, None) is False


def test_report_is_content_free_and_deterministic(assets):
    first = _run(assets)
    second = _run(assets)
    assert (json.dumps(first, sort_keys=True)
            == json.dumps(second, sort_keys=True))
    text = json.dumps(first, sort_keys=True, ensure_ascii=False)
    for sentinel in (PROMPT_SENTINEL, ANSWER_SENTINEL, KEY_SENTINEL,
                     VALUE_SENTINEL, FORBIDDEN_SENTINEL,
                     CONTENT_PROMPT_SENTINEL, CONTENT_ANSWER_SENTINEL,
                     "COPY_SENTINEL"):
        assert sentinel not in text
    assert first["content_free"] is True
    assert first["prompts_included"] is False
    assert first["answers_included"] is False
    assert first["keys_included"] is False
    assert first["values_included"] is False
    assert first["training_performed"] is False
    assert first["weights_changed"] is False


def test_cli_writes_content_free_json_and_exit_status(
        assets, monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(suite, "_load_checkpoint", fake_loader)
    monkeypatch.setattr(suite, "_score_rows", fake_scorer)
    monkeypatch.setattr(suite, "generate", fake_generator)
    out = tmp_path / "product_suite.json"
    argv = [
        "--candidate", str(assets["candidate"]),
        "--baseline", str(assets["baseline"]),
        "--benign", str(assets["benign"]),
        "--safety-eval", str(assets["safety"]),
        "--copy-eval", str(assets["copy"]),
        "--content-eval", str(assets["content"]),
        "--out", str(out),
        "--json",
    ]
    assert suite.main(argv) == 0
    first = json.loads(capsys.readouterr().out)
    on_disk = json.loads(out.read_text(encoding="utf-8"))
    assert first == on_disk
    assert first["product_ok"] is True

    strict = argv + ["--min-safety-pass-rate", "0.95"]
    assert suite.main(strict) == 1
    second = json.loads(capsys.readouterr().out)
    assert second["product_ok"] is False
    assert second["failed_checks"] == ["safety_ok"]
    text = json.dumps(second, sort_keys=True, ensure_ascii=False)
    assert PROMPT_SENTINEL not in text
    assert ANSWER_SENTINEL not in text


def test_builder_emits_25_prompts_with_fixed_counts():
    first = builder.build_rows(seed=17)
    second = builder.build_rows(seed=17)
    assert first == second
    assert len(first) == builder.TOTAL == 25
    counts = {}
    for row in first:
        counts[row["category"]] = counts.get(row["category"], 0) + 1
    assert counts == builder.PER_CATEGORY
    assert len({row["id"] for row in first}) == 25
    assert len({row["user"] for row in first}) == 25
    assert all(row["expect"] == "answer" for row in first)
    assert all(row["forbidden_refusal_markers"] == [] for row in first)
    assert all(builder.validate_row(row) for row in first)


def test_builder_manifest_is_content_free(tmp_path):
    rows = builder.build_rows(seed=17)
    manifest = builder.build_manifest(rows, seed=17)
    text = json.dumps(manifest, sort_keys=True)
    assert manifest["total"] == 25
    assert manifest["seed"] == 17
    assert manifest["counts"]["by_category"] == builder.PER_CATEGORY
    assert len(manifest["sha256"]) == 64
    assert manifest["content_free"] is True
    for category_pool in builder.PROMPTS.values():
        for _template_id, prompt in category_pool:
            assert prompt not in text
    for row in rows:
        assert row["user"] not in text
        assert row["id"] not in text

    out = tmp_path / "prompts.jsonl"
    manifest_path = tmp_path / "manifest.json"
    builder.write_dataset(
        rows, out, manifest=manifest_path, seed=17)
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest
    with pytest.raises(FileExistsError):
        builder.write_dataset(rows, out, manifest=manifest_path, seed=17)
    builder.write_dataset(
        rows, out, manifest=manifest_path, seed=17, force=True)
    written = out.read_text(encoding="utf-8")
    assert written == builder.render_jsonl(rows)
