from __future__ import annotations

from atomzero.chat_server import numeric_answer
from atomzero.numeric_router import route_numeric


def _ask(*turns: str) -> dict:
    return route_numeric(list(turns))


def test_budget_yes_multiturn() -> None:
    result = _ask("My budget is $500.", "Can I spend $300?")
    assert result["handled"] is True
    assert result["kind"] == "budget"
    assert result["answer"] == "Yes, $300 is within your $500 budget."


def test_budget_no_multiturn() -> None:
    result = _ask("My budget is $500.", "Can I spend $300?")
    assert result["handled"] is True
    assert result["answer"].startswith("Yes,")
    result = _ask("My budget is $500.", "Can I spend $650?")
    assert result["answer"] == "No, $650 is over your $500 budget by $150."


def test_budget_exact() -> None:
    result = _ask("My budget is $500.", "Can I spend $500?")
    assert result["answer"] == "Yes, $500 is exactly your $500 budget."


def test_budget_commas_and_decimals() -> None:
    result = _ask("I can spend up to $1,250.50.", "Is $900.25 okay?")
    assert result["answer"] == "Yes, $900.25 is within your $1250.50 budget."


def test_budget_single_turn() -> None:
    result = route_numeric(["My budget is $500, can I spend $300?"])
    assert result["handled"] is True
    assert result["answer"] == "Yes, $300 is within your $500 budget."


def test_budget_extra_history() -> None:
    result = _ask("Hello!", "I have $800 set aside for this.",
                  "What about $950, is that okay?")
    assert result["answer"] == "No, $950 is over your $800 budget by $150."


def test_budget_inside_phrasing() -> None:
    result = _ask("I set aside $590 for this.", "Is $458 still inside it?")
    assert result["answer"] == "Yes, $458 is within your $590 budget."


def test_time_yes() -> None:
    result = _ask("I have 45 minutes.", "Can I do a 30-minute task?")
    assert result["kind"] == "time"
    assert result["answer"] == "Yes, 30 minutes fits in your 45-minute window."


def test_time_no() -> None:
    result = _ask("Only 20 minutes are left today.",
                  "Will 60 minutes fit?")
    assert result["answer"] == ("No, a 60-minute task is 40 minutes over your "
                                "20-minute window.")


def test_time_single_turn() -> None:
    result = route_numeric(["I have 90 minutes, can I do a 120-minute task?"])
    assert result["answer"] == ("No, a 120-minute task is 30 minutes over your "
                                "90-minute window.")


def test_plain_chat_is_not_handled() -> None:
    assert route_numeric(["Hello! How are you today?"])["handled"] is False


def test_budget_statement_without_question() -> None:
    assert route_numeric(["My budget is $500."])["handled"] is False


def test_question_without_limit() -> None:
    assert route_numeric(["Can I spend $300?"])["handled"] is False


def test_empty_messages() -> None:
    assert route_numeric([])["handled"] is False
    assert route_numeric(None)["handled"] is False


def test_chat_server_wrapper() -> None:
    result = numeric_answer([
        {"role": "user", "content": "My budget is $500."},
        {"role": "assistant", "content": "Okay, $500."},
        {"role": "user", "content": "Can I spend $300?"},
    ])
    assert result["source"] == "numeric_tool"
    assert result["kind"] == "budget"
    assert result["text"] == "Yes, $300 is within your $500 budget."


def test_chat_server_wrapper_ignores_plain_chat() -> None:
    assert numeric_answer([{"role": "user", "content": "Hello!"}]) is None
