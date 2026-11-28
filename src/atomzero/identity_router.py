"""Deterministic identity answers for explicit self/creator questions.

The 63M model partially knows the string "Atom Zero" but invents a name for
itself and does not know its creator. These questions have a fixed public
answer, so this module handles only the explicit forms and leaves every other
prompt to the model/RAG path.

Public API::

    from atomzero.identity_router import route_identity

    result = route_identity(["Who created you?"])
    if result["handled"]:
        print(result["answer"])
"""
from __future__ import annotations

import re

__all__ = ["route_identity"]

NAME = "Atom Zero"
CREATOR = "Gökhan Çava"

_NAME = re.compile(
    r"\b(?:what(?:'s| is) your name|who are you|what are you|"
    r"what should i call you)\b", re.IGNORECASE)
_CREATOR = re.compile(
    r"\b(?:who (?:created|made|built|designed|trained|owns) you|"
    r"who is your (?:creator|author|maker|owner)|your creator|"
    r"who created atom ?zero|who made atom ?zero)\b", re.IGNORECASE)
_PERSON = re.compile(
    r"\bwho is (?:gokhan|gökhan) (?:cava|çava)\b", re.IGNORECASE)
_PROJECT = re.compile(
    r"\bwhat is atom ?zero\b|\bwhat(?:'s| is) atom ?zero\b", re.IGNORECASE)


def _turns(messages) -> list[str]:
    if isinstance(messages, str):
        return [messages]
    if not messages:
        return []
    turns = []
    for item in messages:
        if isinstance(item, str):
            turns.append(item)
        elif isinstance(item, dict) and item.get("role") == "user":
            turns.append(str(item.get("content", "")))
    return [turn for turn in turns if turn.strip()]


def route_identity(messages) -> dict:
    """Return a deterministic identity reply, or ``handled: False``."""
    turns = _turns(messages)
    if not turns:
        return {"handled": False}
    last = turns[-1]
    if _PERSON.search(last):
        return {"handled": True, "kind": "creator",
                "answer": f"{CREATOR} is the creator of {NAME}, a local, "
                          f"source-cited 63M assistant."}
    if _CREATOR.search(last):
        return {"handled": True, "kind": "creator",
                "answer": f"{NAME} was created by {CREATOR}."}
    if _NAME.search(last):
        return {"handled": True, "kind": "name",
                "answer": f"I am {NAME}, a local assistant created by "
                          f"{CREATOR}."}
    if _PROJECT.search(last):
        return {"handled": True, "kind": "project",
                "answer": f"{NAME} is a local, source-cited 63M assistant "
                          f"created by {CREATOR}."}
    return {"handled": False}
