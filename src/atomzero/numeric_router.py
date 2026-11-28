"""Deterministic budget and time-fit checker for explicit numeric questions.

The 63M chat model reliably recalls names, cities, languages, days and
preferences, but it cannot compare two amounts: its tokenizer splits numbers
inconsistently and the model answers "No" regardless of the values. Those
questions have an exact answer, so this module recognises only the narrow
"limit + spend" and "window + task" shapes and computes the reply.

Everything else returns ``handled: False`` and keeps the model/RAG path. This
is a runtime tool, not model self-improvement or training.

Public API::

    from atomzero.numeric_router import route_numeric

    result = route_numeric(["My budget is $500.", "Can I spend $300?"])
    if result["handled"]:
        print(result["answer"])       # "Yes, $300 is within your $500 budget."
"""
from __future__ import annotations

import re

__all__ = ["route_numeric"]

_MONEY_RE = re.compile(r"\$\s*(\d[\d,]*(?:\.\d{1,2})?)")
_MINUTES_RE = re.compile(r"\b(\d{1,4})\s*(?:-|\s)?\s*(?:minutes?|mins?)\b",
                         re.IGNORECASE)

_MONEY_LIMIT_RE = re.compile(
    r"(?:\bbudget\s*(?:is|of|=)|"
    r"\bcan\s+spend\s+up\s+to|"
    r"\bset\s+aside|"
    r"\bcap\s*(?:is|of)|"
    r"\blimit\s*(?:is|of)|"
    r"\bdo\s+not\s+go\s+over|"
    r"\bunder\s+\$)",
    re.IGNORECASE)

_MONEY_ASK_RE = re.compile(
    r"\b(?:can|could|would|should|is|will|do)\b[^?]{0,80}"
    r"\b(?:spend|afford|within|inside|over|okay|ok|enough|possible|"
    r"too\s+much|fit|left)\b"
    r"|\bwhat\s+about\s+\$",
    re.IGNORECASE)

_TIME_LIMIT_RE = re.compile(
    r"(?:\bi\s+have|\bonly|\bwindow\s+is|\bcan\s+spare|"
    r"\bleaves?\s+me|\bleft\b)[^.]{0,24}?\d{1,4}\s*"
    r"(?:minutes?|mins?)\b",
    re.IGNORECASE)

_TIME_ASK_RE = re.compile(
    r"\b(?:can|could|will|is|do)\b[^?]{0,80}"
    r"\b(?:fit|possible|time|too\s+long|do\s+a|task|job|minutes?)\b",
    re.IGNORECASE)


def _amount(text: str) -> float:
    return float(text.replace(",", ""))


def _money(value: float) -> str:
    if value == int(value):
        return f"${int(value)}"
    return "${:.2f}".format(value)


def _minutes(value: float) -> str:
    if value == int(value):
        return f"{int(value)}"
    return f"{value:g}"


def _budget_answer(limit: float, spend: float) -> dict:
    if spend <= limit:
        if spend == limit:
            answer = (f"Yes, {_money(spend)} is exactly your "
                      f"{_money(limit)} budget.")
        else:
            answer = (f"Yes, {_money(spend)} is within your "
                      f"{_money(limit)} budget.")
    else:
        answer = (f"No, {_money(spend)} is over your {_money(limit)} "
                  f"budget by {_money(spend - limit)}.")
    return {"handled": True, "kind": "budget", "answer": answer,
            "limit": limit, "value": spend}


def _time_answer(limit: float, task: float) -> dict:
    if task <= limit:
        answer = (f"Yes, {_minutes(task)} minutes fits in your "
                  f"{_minutes(limit)}-minute window.")
    else:
        answer = (f"No, a {_minutes(task)}-minute task is "
                  f"{_minutes(task - limit)} minutes over your "
                  f"{_minutes(limit)}-minute window.")
    return {"handled": True, "kind": "time", "answer": answer,
            "limit": limit, "value": task}


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


def route_numeric(messages) -> dict:
    """Return a deterministic budget/time reply, or ``handled: False``."""
    turns = _turns(messages)
    if not turns:
        return {"handled": False}
    last = turns[-1]
    earlier = " ".join(turns[:-1])

    # Budget: limit from an earlier turn, or two amounts in the last turn.
    if _MONEY_ASK_RE.search(last):
        limit = None
        spend = None
        earlier_money = _MONEY_RE.findall(earlier)
        last_money = _MONEY_RE.findall(last)
        if earlier_money and _MONEY_LIMIT_RE.search(earlier) and last_money:
            limit = _amount(earlier_money[-1])
            spend = _amount(last_money[-1])
        elif len(last_money) >= 2 and _MONEY_LIMIT_RE.search(last):
            limit = _amount(last_money[0])
            spend = _amount(last_money[-1])
        if limit is not None and spend is not None:
            return _budget_answer(limit, spend)

    # Time: window from an earlier turn, or two durations in the last turn.
    if _TIME_ASK_RE.search(last):
        limit = None
        task = None
        earlier_minutes = _MINUTES_RE.findall(earlier)
        last_minutes = _MINUTES_RE.findall(last)
        if earlier_minutes and _TIME_LIMIT_RE.search(earlier) and last_minutes:
            limit = float(earlier_minutes[-1])
            task = float(last_minutes[-1])
        elif len(last_minutes) >= 2 and _TIME_LIMIT_RE.search(last):
            limit = float(last_minutes[0])
            task = float(last_minutes[-1])
        if limit is not None and task is not None:
            return _time_answer(limit, task)

    return {"handled": False}
