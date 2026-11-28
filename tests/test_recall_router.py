from __future__ import annotations

from atomzero.chat_server import recall_answer
from atomzero.recall_router import route_recall


def _ask(*turns: str) -> dict:
    return route_recall(list(turns))


def test_name_multiturn() -> None:
    result = _ask("My name is Ada.", "What is my name?")
    assert result["handled"] is True
    assert result["kind"] == "name"
    assert result["answer"] == "Ada."


def test_name_greeting() -> None:
    result = _ask("Hi, I'm Leo.", "Do you remember my name?")
    assert result["answer"] == "Leo."


def test_name_is_not_an_adjective() -> None:
    assert _ask("I'm tired.", "What is my name?")["handled"] is False


def test_city() -> None:
    result = _ask("I live in Izmir.", "Which city do I live in?")
    assert result["answer"] == "Izmir."


def test_language_correction() -> None:
    result = _ask("Use Java for this project.",
                  "Actually, use Go. Which language are we using?")
    assert result["answer"] == "Go."


def test_language_first_statement() -> None:
    result = _ask("Use Python for this project.",
                  "What language are we using?")
    assert result["answer"] == "Python."


def test_preference() -> None:
    result = _ask("I prefer coffee over water.", "What do I prefer?")
    assert result["answer"] == "coffee."


def test_preference_favorite() -> None:
    result = _ask("Tea is my favorite, not coffee.", "Which one do I like?")
    assert result["answer"] == "Tea."


def test_day_sunday() -> None:
    result = _ask("Today is Sunday.", "What day is tomorrow?")
    assert result["answer"] == "Tomorrow is Monday."


def test_day_tuesday() -> None:
    result = _ask("Today is Tuesday.", "Which day comes next?")
    assert result["answer"] == "Tomorrow is Wednesday."


def test_day_invalid_word() -> None:
    assert _ask("Today is great.", "What day is tomorrow?")["handled"] is False


def test_pet() -> None:
    result = _ask("My dog's name is Lucy.", "What is my dog's name?")
    assert result["answer"] == "Lucy."


def test_question_without_fact() -> None:
    assert _ask("What is my name?")["handled"] is False


def test_fact_without_question() -> None:
    assert _ask("My name is Ada.")["handled"] is False


def test_plain_chat_not_handled() -> None:
    assert _ask("Hello! How are you today?")["handled"] is False


def test_string_input() -> None:
    result = route_recall("My name is Ada. What is my name?")
    assert result["handled"] is True
    assert result["answer"] == "Ada."


def test_assistant_turns_ignored() -> None:
    messages = [
        {"role": "user", "content": "My name is Ada."},
        {"role": "assistant", "content": "Nice to meet you, Ada."},
        {"role": "user", "content": "What is my name?"},
    ]
    assert route_recall(messages)["answer"] == "Ada."


def test_chat_server_wrapper() -> None:
    result = recall_answer([
        {"role": "user", "content": "Today is Sunday."},
        {"role": "assistant", "content": "Okay, Sunday."},
        {"role": "user", "content": "What day is tomorrow?"},
    ])
    assert result["source"] == "recall_tool"
    assert result["kind"] == "day"
    assert result["text"] == "Tomorrow is Monday."


def test_chat_server_wrapper_ignores_plain_chat() -> None:
    assert recall_answer([{"role": "user", "content": "Hello!"}]) is None


def test_name_embedded_question() -> None:
    result = _ask("My name is Ada, by the way.",
                  "Do you remember what my name is?")
    assert result["answer"] == "Ada."


def test_city_am_based() -> None:
    result = _ask("These days I am based in Denizli.",
                  "Which city am I in?")
    assert result["answer"] == "Denizli."


def test_day_what_comes_next() -> None:
    result = _ask("Today is Monday, right?", "So what day comes next?")
    assert result["answer"] == "Tomorrow is Tuesday."


def test_preference_remind() -> None:
    result = _ask("Given a choice I take tea, not coffee.",
                  "Remind me which one I like.")
    assert result["answer"] == "tea."
