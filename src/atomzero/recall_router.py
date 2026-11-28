"""Deterministic recall for explicit "what is my ..." questions.

The 63M model recalls facts from its trained phrasings but fails on new ones
and can overwrite a day answer with a frequent weekday. Those questions have
an exact answer in the conversation history, so this module parses the facts a
user stated and answers only explicit recall questions. Everything else returns
``handled: False`` and keeps the model/RAG path.

Public API::

    from atomzero.recall_router import route_recall

    result = route_recall(["My name is Ada.", "What is my name?"])
    if result["handled"]:
        print(result["answer"])       # "Ada."
"""
from __future__ import annotations

import re

__all__ = ["route_recall"]

DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")

_NAME = r"([A-Z][A-Za-z'’-]{1,30})"
_NAME_PATTERNS = (
    re.compile(r"\bmy name is " + _NAME, re.IGNORECASE),
    re.compile(r"\bcall me " + _NAME, re.IGNORECASE),
    re.compile(r"\byou can call me " + _NAME, re.IGNORECASE),
    re.compile(r"\bi go by " + _NAME, re.IGNORECASE),
    re.compile(r"\b(?:[Hh]i|[Hh]ello),?\s+[Ii]'?m " + _NAME),
)

_CITY = r"([A-Z][A-Za-z .'’-]{1,40})"
_CITY_PATTERNS = (
    re.compile(r"\bi live in " + _CITY, re.IGNORECASE),
    re.compile(r"\bi(?:'m| am) based in " + _CITY, re.IGNORECASE),
    re.compile(r"\bi moved to " + _CITY, re.IGNORECASE),
    re.compile(r"\bmy city is " + _CITY, re.IGNORECASE),
    re.compile(r"\bhome is " + _CITY, re.IGNORECASE),
)

_LANG = r"([A-Za-z][A-Za-z0-9+#.]{0,20})"
_LANG_FIRST = (
    re.compile(r"\buse " + _LANG + r" for this project", re.IGNORECASE),
    re.compile(r"\bwe started in " + _LANG, re.IGNORECASE),
    re.compile(r"\bthe project is written in " + _LANG, re.IGNORECASE),
    re.compile(r"\b(?:let'?s )?begin with " + _LANG, re.IGNORECASE),
)
_LANG_SWITCH = (
    re.compile(r"\b(?:actually,?\s+)?use " + _LANG + r" (?:instead|now)\b",
               re.IGNORECASE),
    re.compile(r"\bactually,?\s+use " + _LANG, re.IGNORECASE),
    re.compile(r"\bswitch to " + _LANG, re.IGNORECASE),
    re.compile(r"\bchange it to " + _LANG, re.IGNORECASE),
    re.compile(r"\bmove it to " + _LANG, re.IGNORECASE),
    re.compile(r"\bswitch to " + _LANG + r"\b", re.IGNORECASE),
)

_PREF = r"([^,.!?\n]{1,40}?)"
_PREF_PATTERNS = (
    re.compile(r"\bi prefer " + _PREF + r" over ", re.IGNORECASE),
    re.compile(r"\bi like " + _PREF + r", not ", re.IGNORECASE),
    re.compile(r"\bgiven a choice i take " + _PREF + r"(?:\.|,|$)",
               re.IGNORECASE),
    re.compile(r"\b" + _PREF + r" is my favorite, not ", re.IGNORECASE),
    re.compile(r"\bi would rather have " + _PREF + r"(?:\.|,|$)",
               re.IGNORECASE),
    re.compile(r"\bi'?ll take " + _PREF + r"(?: any day|\.|,|$)", re.IGNORECASE),
    re.compile(r"\bfor me it'?s " + _PREF + r", not ", re.IGNORECASE),
    re.compile(r"\bi usually pick " + _PREF + r"(?:\.|,|$)", re.IGNORECASE),
)

_DAY = r"([A-Za-z]+)"
_DAY_PATTERNS = (
    re.compile(r"\btoday is " + _DAY, re.IGNORECASE),
    re.compile(r"\bit is " + _DAY + r" today", re.IGNORECASE),
    re.compile(r"\bwe are on " + _DAY, re.IGNORECASE),
    re.compile(r"\bit'?s " + _DAY + r" already", re.IGNORECASE),
    re.compile(r"\bwe'?re at " + _DAY, re.IGNORECASE),
)

_PET = r"([A-Z][A-Za-z'’-]{1,30})"
_PET_PATTERNS = (
    re.compile(r"\bmy (?:dog|cat|pet)'?s name is " + _PET, re.IGNORECASE),
    re.compile(r"\bmy (?:dog|cat|pet) is called " + _PET, re.IGNORECASE),
    re.compile(r"\bi have a (?:dog|cat) named " + _PET, re.IGNORECASE),
    re.compile(r"\bwe adopted a (?:dog|cat), " + _PET, re.IGNORECASE),
)

_QUESTION = {
    "name": re.compile(
        r"\b(?:what(?:'s| is) my name|do you remember (?:what )?my name|"
        r"what should you call me|who am i|say my name)\b", re.IGNORECASE),
    "city": re.compile(
        r"\b(?:which|what) city|where do i live|where am i based|"
        r"what is my city\b", re.IGNORECASE),
    "language": re.compile(
        r"\b(?:which|what) language|what are we using|which one wins|"
        r"what is current|which one is active\b", re.IGNORECASE),
    "preference": re.compile(
        r"\bwhat do i prefer|which one do i like|which one i like|"
        r"remind me which one i like|my preference|"
        r"which should i get|what is my pick|which one did i choose\b",
        re.IGNORECASE),
    "day": re.compile(
        r"\bwhat day is tomorrow|(?:which|what) day comes next|"
        r"what is the next day|and tomorrow|what follows today|"
        r"what day follows\b", re.IGNORECASE),
    "pet": re.compile(
        r"\bwhat is my (?:dog|cat|pet)'?s name|what is my pet called|"
        r"my pet'?s name\b", re.IGNORECASE),
}


def _clean(value: str) -> str:
    return str(value or "").strip(" \t.,!?:;")


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


def _facts(turns: list[str]) -> dict[str, str]:
    facts: dict[str, str] = {}
    language = ""
    language_switched = ""
    for turn in turns:
        for pattern in _NAME_PATTERNS:
            match = pattern.search(turn)
            if match:
                facts["name"] = _clean(match.group(1))
                break
        for pattern in _CITY_PATTERNS:
            match = pattern.search(turn)
            if match:
                facts["city"] = _clean(match.group(1))
                break
        for pattern in _LANG_SWITCH:
            match = pattern.search(turn)
            if match:
                language_switched = _clean(match.group(1))
                break
        if not language:
            for pattern in _LANG_FIRST:
                match = pattern.search(turn)
                if match:
                    language = _clean(match.group(1))
                    break
        for pattern in _PREF_PATTERNS:
            match = pattern.search(turn)
            if match:
                facts["preference"] = _clean(match.group(1))
                break
        for pattern in _DAY_PATTERNS:
            match = pattern.search(turn)
            if match:
                word = _clean(match.group(1)).capitalize()
                if word in DAYS:
                    facts["day"] = word
                    break
        for pattern in _PET_PATTERNS:
            match = pattern.search(turn)
            if match:
                facts["pet"] = _clean(match.group(1))
                break
    if language_switched or language:
        facts["language"] = language_switched or language
    return facts


def route_recall(messages) -> dict:
    """Return a deterministic recall reply, or ``handled: False``."""
    turns = _turns(messages)
    if not turns:
        return {"handled": False}
    last = turns[-1]
    facts = _facts(turns)
    for kind, pattern in _QUESTION.items():
        if not pattern.search(last):
            continue
        value = facts.get(kind, "")
        if not value:
            continue
        if kind == "day":
            nxt = DAYS[(DAYS.index(value) + 1) % len(DAYS)]
            answer = f"Tomorrow is {nxt}."
        else:
            answer = f"{value}."
        return {"handled": True, "kind": kind, "answer": answer,
                "value": value}
    return {"handled": False}
