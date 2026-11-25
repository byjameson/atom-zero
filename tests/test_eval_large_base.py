from __future__ import annotations

import json

import pytest

from research.eval_large_base import (
    DEFAULT_DATA,
    DEFAULT_MODEL,
    FLAT_INSTRUCTION,
    ParsedPairs,
    _parser,
    build_chat_prompt,
    evaluate_rows,
    is_flat_json_output,
    main,
    parse_model_output,
    run_evaluation,
)


def _row(pairs: list[tuple[str, str]], prompt: str = "copy these") -> dict:
    return {
        "user": prompt,
        "pairs": [[key, value] for key, value in pairs],
    }


@pytest.mark.parametrize(
    ("text", "expected", "kind", "fenced"),
    [
        (
            '{"alpha": "one", "beta": "two"}',
            {"alpha": "one", "beta": "two"},
            "flat_json",
            False,
        ),
        (
            '{"pairs": [["alpha", "one"], ["beta", "two"]]}',
            {"alpha": "one", "beta": "two"},
            "pairs_list_json",
            False,
        ),
        (
            '{"pairs": [{"source": "alpha", "target": "one"}, '
            '{"source": "beta", "target": "two"}]}',
            {"alpha": "one", "beta": "two"},
            "pairs_object_json",
            False,
        ),
        (
            '```json\n{"alpha": "one", "beta": "two"}\n```',
            {"alpha": "one", "beta": "two"},
            "flat_json",
            True,
        ),
        (
            'Here you go:\n```json\n{"pairs": [["alpha", "one"]]}\n```\n',
            {"alpha": "one"},
            "pairs_list_json",
            True,
        ),
        (
            "alpha=one; beta=two",
            {"alpha": "one", "beta": "two"},
            "kv_yaml",
            False,
        ),
        (
            'alpha: "one"\nbeta: "two"',
            {"alpha": "one", "beta": "two"},
            "kv_yaml",
            False,
        ),
        (
            "[alpha=one; beta=two]",
            {"alpha": "one", "beta": "two"},
            "kv_yaml",
            False,
        ),
        (
            '[["alpha", "one"], ["beta", "two"]]',
            {"alpha": "one", "beta": "two"},
            "pairs_list_json",
            False,
        ),
        (
            '```\nalpha=one; beta=two\n```',
            {"alpha": "one", "beta": "two"},
            "kv_yaml",
            True,
        ),
    ],
)
def test_tolerant_parser_accepts_supported_shapes(
    text: str, expected: dict[str, str], kind: str, fenced: bool,
):
    parsed = parse_model_output(text)

    assert isinstance(parsed, ParsedPairs)
    assert parsed.pairs == expected
    assert parsed.kind == kind
    assert parsed.fenced is fenced


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "I cannot help with that.",
        '{"pairs": "not-a-list"}',
        '{"pairs": [["only-a-key"]]}',
        '{"pairs": []}',
        '{"pairs": [{"source": "alpha"}]}',
        '{"pairs": [{"target": "one"}]}',
        "```json\nnot json\n```",
    ],
)
def test_tolerant_parser_reports_parse_failures(text: str):
    with pytest.raises(ValueError):
        parse_model_output(text)


def test_parser_rejects_non_string_output():
    with pytest.raises(ValueError):
        parse_model_output(None)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"alpha": "one"}', True),
        ('  {"alpha": 1, "beta": true}  ', True),
        ('```json\n{"alpha": "one"}\n```', False),
        ('{"pairs": [["alpha", "one"]]}', False),
        ('Here: {"alpha": "one"}', False),
        ('{"alpha": {"nested": "one"}}', False),
        ("", False),
    ],
)
def test_flat_format_compliance(text: str, expected: bool):
    assert is_flat_json_output(text) is expected


def test_metrics_cover_exact_recall_parse_failures_and_flat_format():
    rows = [
        _row([("a", "1"), ("b", "2")], prompt="PROMPT_ONE"),
        _row([("c", "3"), ("d", "4")], prompt="PROMPT_TWO"),
        _row([("e", "5"), ("f", "6")], prompt="PROMPT_THREE"),
        _row([("g", "7"), ("h", "8")], prompt="PROMPT_FOUR"),
        {"pairs": [["i", "9"]]},
    ]
    outputs = iter([
        '{"a": "1", "b": "2"}',
        '```json\n{"pairs": [["c", "3"], ["d", "4"]]}\n```',
        '{"e": "5"}',
        "no idea",
    ])

    summary = evaluate_rows(rows, lambda prompt: next(outputs))

    assert summary["rows_total"] == 5
    assert summary["rows_evaluated"] == 4
    assert summary["rows_invalid"] == 1
    assert summary["rows_exact"] == 2
    assert summary["exact_match"] == 0.5
    assert summary["key_recall"] == 0.625
    assert summary["value_recall"] == 0.625
    assert summary["parse_failures"] == 1
    assert summary["parse_failure_rate"] == 0.25
    assert summary["rows_flat_json"] == 2
    assert summary["flat_format_rate"] == 0.5
    assert summary["rows_with_code_fence"] == 1
    assert summary["parse_kinds"] == {
        "flat_json": 2,
        "pairs_list_json": 1,
    }
    assert summary["invalid_reasons"] == {"missing_prompt": 1}
    assert summary["content_free"] is True
    assert summary["prompt_text_included"] is False
    assert summary["weights_changed"] is False


def test_output_summary_never_contains_prompt_or_pair_text():
    prompt = "SECRET_PROMPT_TEXT_XYZ"
    key = "SECRET_KEY_XYZ"
    value = "SECRET_VALUE_XYZ"
    summary = evaluate_rows(
        [_row([(key, value)], prompt=prompt)],
        lambda text: json.dumps({key: value}),
    )

    blob = json.dumps(summary)

    assert prompt not in blob
    assert key not in blob
    assert value not in blob
    assert summary["exact_match"] == 1.0


def test_build_chat_prompt_uses_template_and_optional_flat_instruction():
    class StubTokenizer:
        def __init__(self) -> None:
            self.calls = []

        def apply_chat_template(
            self, messages, *, tokenize, add_generation_prompt,
        ):
            self.calls.append((messages, tokenize, add_generation_prompt))
            return f"CHAT::{messages[0]['content']}"

    tokenizer = StubTokenizer()

    plain = build_chat_prompt(tokenizer, "copy these")
    flat = build_chat_prompt(tokenizer, "copy these", flat_prompt=True)

    assert plain == "CHAT::copy these"
    assert FLAT_INSTRUCTION not in plain
    assert flat.startswith("CHAT::copy these")
    assert FLAT_INSTRUCTION in flat
    assert tokenizer.calls[0][1:] == (False, True)
    assert tokenizer.calls[0][0] == [
        {"role": "user", "content": "copy these"}]


def test_run_evaluation_accepts_injected_loader_and_generator(tmp_path):
    data_path = tmp_path / "eval.jsonl"
    data_path.write_text(
        json.dumps(_row([("k1", "v1")], prompt="PROMPT_TEXT")) + "\n",
        encoding="utf-8",
    )

    class StubTokenizer:
        def __init__(self) -> None:
            self.messages = []
            self.pad_token_id = None
            self.eos_token_id = None

        def apply_chat_template(
            self, messages, *, tokenize, add_generation_prompt,
        ):
            self.messages.append(messages)
            return "CHAT::" + messages[0]["content"]

    class StubModel:
        marker = "STUB_MODEL"

    tokenizer = StubTokenizer()
    model = StubModel()
    loader_calls = []
    generator_calls = []

    def loader(model_name, *, device, local_files_only):
        loader_calls.append((model_name, device, local_files_only))
        return model, tokenizer

    def generator(
        loaded_model, loaded_tokenizer, prompt, *,
        max_new_tokens, device,
    ):
        assert loaded_model is model
        assert loaded_tokenizer is tokenizer
        generator_calls.append((prompt, max_new_tokens, device))
        return '{"k1": "v1"}'

    summary = run_evaluation(
        data_path,
        model_name="stub/model",
        rows=1,
        max_new_tokens=16,
        device="cpu",
        flat_prompt=True,
        local_files_only=True,
        loader=loader,
        generator=generator,
    )

    assert loader_calls == [("stub/model", "cpu", True)]
    assert len(generator_calls) == 1
    prompt, max_new_tokens, device = generator_calls[0]
    assert prompt.startswith("CHAT::PROMPT_TEXT")
    assert FLAT_INSTRUCTION in prompt
    assert (max_new_tokens, device) == (16, "cpu")
    assert summary["model"] == "stub/model"
    assert summary["device"] == "cpu"
    assert summary["flat_prompt"] is True
    assert summary["rows_total"] == 1
    assert summary["exact_match"] == 1.0
    assert summary["flat_format_rate"] == 1.0


def test_run_evaluation_rows_limit(tmp_path):
    data_path = tmp_path / "eval.jsonl"
    lines = [
        json.dumps(_row([("a", "1")], prompt="FIRST")),
        json.dumps(_row([("b", "2")], prompt="SECOND")),
    ]
    data_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    class StubTokenizer:
        def apply_chat_template(
            self, messages, *, tokenize, add_generation_prompt,
        ):
            return messages[0]["content"]

    def loader(model_name, *, device, local_files_only):
        return object(), StubTokenizer()

    def generator(model, tokenizer, prompt, *, max_new_tokens, device):
        return '{"a": "1"}'

    summary = run_evaluation(
        data_path,
        rows=1,
        device="cpu",
        loader=loader,
        generator=generator,
    )

    assert summary["rows_total"] == 1
    assert summary["rows_evaluated"] == 1


def test_main_writes_full_and_compact_results(
    tmp_path, monkeypatch, capsys,
):
    import research.eval_large_base as module

    summary = evaluate_rows(
        [_row([("a", "1")], prompt="PROMPT_TEXT")],
        lambda prompt: '{"a": "1"}',
    )
    monkeypatch.setattr(
        module, "run_evaluation", lambda *args, **kwargs: summary)
    full_path = tmp_path / "full.json"
    compact_path = tmp_path / "compact.json"

    code = main([
        "--out", str(full_path),
        "--result", str(compact_path),
        "--rows", "1",
    ])

    assert code == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed == summary
    assert json.loads(full_path.read_text(encoding="utf-8")) == summary
    compact = json.loads(compact_path.read_text(encoding="utf-8"))
    assert compact["exact_match"] == 1.0
    assert compact["rows_total"] == 1
    assert "parse_kinds" not in compact
    assert "PROMPT_TEXT" not in json.dumps(compact)


def test_cli_defaults_and_flags():
    parser = _parser()
    args = parser.parse_args([])
    assert args.model == DEFAULT_MODEL
    assert args.data == str(DEFAULT_DATA)
    assert args.rows is None
    assert args.max_new_tokens == 64
    assert args.device == "auto"
    assert args.flat_prompt is False
    assert args.local_files_only is True
    assert args.out is None
    assert args.result is None

    args = parser.parse_args([
        "--model", "other/model",
        "--data", "other.jsonl",
        "--rows", "20",
        "--max-new-tokens", "32",
        "--device", "cpu",
        "--flat-prompt",
        "--no-local-files-only",
        "--out", "full.json",
        "--result", "compact.json",
    ])
    assert args.model == "other/model"
    assert args.data == "other.jsonl"
    assert args.rows == 20
    assert args.max_new_tokens == 32
    assert args.device == "cpu"
    assert args.flat_prompt is True
    assert args.local_files_only is False
    assert args.out == "full.json"
    assert args.result == "compact.json"
