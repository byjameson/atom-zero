from __future__ import annotations

import pytest

from atomzero.chat_server import copy_answer
from atomzero.copy_router import detect_copy_request, route_copy
from atomzero.copy_tool import parse_rendered


def _body(pairs: list[tuple[str, str]]) -> str:
    return "; ".join(f"{key}={value}" for key, value in pairs)


def _prompt(fmt: str, pairs: list[tuple[str, str]]) -> str:
    body = _body(pairs)
    if fmt == "json":
        return (
            "Copy these pairs exactly into compact JSON: "
            f"[{body}]. Return only the JSON object with no prose or "
            "code fences.")
    if fmt == "kv":
        return (
            "Return only one line of semicolon-separated key=value pairs "
            f"with no prose: [{body}]. Copy every key and value exactly.")
    if fmt == "yaml":
        return (
            "Return only YAML with exactly these keys and values: "
            f"[{body}]. Do not add prose or code fences.")
    raise AssertionError(fmt)


JSON_PROMPT = _prompt("json", [
    ("alpha", "one"),
    ("beta", "two words"),
])
KV_PROMPT = _prompt("kv", [
    ("alpha", "one"),
    ("beta", "two words"),
])
YAML_PROMPT = _prompt("yaml", [
    ("alpha", "one"),
    ("beta", "two words"),
])


@pytest.mark.parametrize(
    ("prompt", "fmt", "expected"),
    [
        (JSON_PROMPT, "json", '{"alpha":"one","beta":"two words"}'),
        (KV_PROMPT, "kv", "alpha=one;beta=two words"),
        (YAML_PROMPT, "yaml", "alpha: one\nbeta: two words"),
    ],
)
def test_explicit_copy_requests_are_handled(prompt, fmt, expected):
    assert detect_copy_request(prompt) == fmt
    result = route_copy(prompt)
    assert result["handled"] is True
    assert result["format"] == fmt
    assert result["answer"] == expected
    assert result["pairs_count"] == 2
    assert result["reason"]


@pytest.mark.parametrize(
    ("prompt", "fmt"),
    [
        (
            "Copy these pairs exactly: [a=1; b=2]",
            "kv",
        ),
        (
            "Copy these pairs exactly: a=1; b=2",
            "kv",
        ),
        (
            "Return only compact JSON object with exactly these keys and "
            "values: [a=1; b=2]",
            "json",
        ),
        (
            "Copy this key=value pair exactly. Return only one line in the "
            "form key=value with no prose: [a=1; b=2]",
            "kv",
        ),
        (
            "Copy these pairs exactly: [format=json; mode=yaml]. Return "
            "only the result.",
            "kv",
        ),
    ],
)
def test_copy_format_detection(prompt, fmt):
    assert detect_copy_request(prompt) == fmt
    assert route_copy(prompt)["format"] == fmt


@pytest.mark.parametrize(
    "prompt",
    [
        "What is the population of Paris?",
        "Summarize the document about Apollo.",
        "How do I copy a file in PowerShell?",
        "Copy the launch code from the document.",
        "Explain this config: host=localhost; port=8080.",
        "Return only JSON.",
        "Compare the two policy documents.",
        "Write a short story about a robot.",
        "Can you copy these pairs exactly: [a=1; b=2]?",
    ],
)
def test_non_copy_prompts_are_not_handled(prompt):
    assert detect_copy_request(prompt) is None
    result = route_copy(prompt)
    assert result["handled"] is False
    assert result["format"] is None
    assert result["answer"] == ""
    assert result["pairs_count"] == 0


@pytest.mark.parametrize(
    "prompt",
    [
        (
            "Copy these pairs exactly into compact JSON: "
            "[alpha=one; broken]. Return only the JSON object."
        ),
        (
            "Return only YAML with exactly these keys and values: "
            "[alpha='unterminated; beta=two]."
        ),
        (
            "Copy these pairs exactly: "
            "[alpha=one; beta=two; gamma=three; delta=four; epsilon=five]."
        ),
        "Copy these pairs exactly: [only=one].",
    ],
)
def test_malformed_or_out_of_bound_pairs_fall_back(prompt):
    result = route_copy(prompt)
    assert result["handled"] is False
    assert result["answer"] == ""
    assert result["reason"]
    assert result["pairs_count"] in (0, 5)


def test_malformed_pairs_reason_is_specific():
    result = route_copy(
        "Copy these pairs exactly into compact JSON: "
        "[alpha=one; broken]. Return only the JSON object.")
    assert result["handled"] is False
    assert result["format"] == "json"
    assert "parse" in result["reason"].lower()


@pytest.mark.parametrize("pair_count", [2, 3, 4])
@pytest.mark.parametrize("fmt", ["json", "kv", "yaml"])
def test_pair_counts_round_trip(pair_count, fmt):
    pairs = []
    for index in range(pair_count):
        value = (f"multi word value {index}" if index % 2
                 else f"value{index}")
        pairs.append((f"key{index}", value))
    prompt = _prompt(fmt, pairs)
    result = route_copy(prompt)
    assert result["handled"] is True
    assert result["pairs_count"] == pair_count
    parsed = parse_rendered(result["answer"], fmt=fmt)
    assert list(parsed.items()) == pairs


def test_router_never_raises_on_bad_input():
    for prompt in (None, 123, [], {"prompt": "copy"}):
        assert detect_copy_request(prompt) is None
        result = route_copy(prompt)
        assert result["handled"] is False
        assert result["answer"] == ""


def test_copy_answer_helper_marks_provenance():
    result = copy_answer(JSON_PROMPT)
    assert result is not None
    assert result["source"] == "copy_tool"
    assert result["sources"] == []
    assert result["format"] == "json"
    assert result["pairs_count"] == 2
    assert result["text"] == '{"alpha":"one","beta":"two words"}'


def test_copy_answer_helper_falls_back_for_normal_prompts():
    assert copy_answer("What is the population of Paris?") is None
    assert copy_answer("Summarize the document about Apollo.") is None
    assert copy_answer(
        "Copy these pairs exactly: [broken].") is None
