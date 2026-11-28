from __future__ import annotations

from atomzero.chat_server import identity_answer
from atomzero.identity_router import route_identity


def test_name() -> None:
    result = route_identity(["What is your name?"])
    assert result["handled"] is True
    assert "Atom Zero" in result["answer"]
    assert "Gökhan Çava" in result["answer"]


def test_who_are_you() -> None:
    result = route_identity(["Who are you?"])
    assert result["answer"].startswith("I am Atom Zero")


def test_creator() -> None:
    result = route_identity(["Who created you?"])
    assert result["answer"] == "Atom Zero was created by Gökhan Çava."


def test_creator_pronoun() -> None:
    result = route_identity(["Who is your creator?"])
    assert result["kind"] == "creator"


def test_person() -> None:
    result = route_identity(["Who is Gökhan Çava?"])
    assert result["kind"] == "creator"
    assert "creator of Atom Zero" in result["answer"]


def test_project() -> None:
    result = route_identity(["What is Atom Zero?"])
    assert result["kind"] == "project"
    assert "63M" in result["answer"]


def test_plain_chat_not_handled() -> None:
    assert route_identity(["Hello! How are you today?"])["handled"] is False


def test_third_party_name_not_handled() -> None:
    assert route_identity(["What is your friend's name?"])["handled"] is False


def test_assistant_turn_ignored() -> None:
    messages = [
        {"role": "user", "content": "Tell me about local AI."},
        {"role": "assistant", "content": "Atom Zero is a local assistant."},
        {"role": "user", "content": "Who created you?"},
    ]
    assert route_identity(messages)["answer"] == "Atom Zero was created by Gökhan Çava."


def test_chat_server_wrapper() -> None:
    result = identity_answer(["What is your name?"])
    assert result["source"] == "identity_tool"
    assert "Atom Zero" in result["text"]


def test_chat_server_wrapper_plain() -> None:
    assert identity_answer(["Hello!"]) is None
