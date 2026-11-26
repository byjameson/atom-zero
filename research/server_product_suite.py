"""Score the real Atom Zero server path with a content-free product suite.

    python research/server_product_suite.py \
        --model-dir checkpoints/mixed8_benign_b \
        --port 8031 \
        --out research/results/server_product_suite_mixed8_benign_b.json \
        --json

The model-level :mod:`research.product_suite` loads a checkpoint directly. It
cannot see the quickstart demo corpus, the document-answer endpoint, the copy
router or a multi-turn messages array. This runner starts the real quickstart
server, sends authored requests over HTTP and scores four areas:

- **rag**: five questions answerable from the quickstart demo corpus. A pass
  needs the expected answer substring and a citation of the expected source.
- **copy**: three explicit transcription prompts rendered as compact JSON or
  one-line KV. A pass needs the exact deterministic copy answer.
- **multi-turn**: three two-turn conversations sent through
  ``/v1/chat/completions``. The second answer must be non-empty, non-refusal and
  at least :data:`MIN_USEFUL_WORDS` words.
- **benign**: ten ordinary single-turn prompts from
  ``research/build_benign_prompts.py`` when available, otherwise a small local
  list. The same useful/refusal rule as ``research/product_suite.py`` applies.

``server_ok`` requires ``rag >= 0.8``, ``copy >= 0.8``, ``multi-turn useful >=
0.66`` and ``benign useful >= 0.8``. Exit code is ``0`` only when ``server_ok``.

The JSON report is content-free: per-area totals, pass counts and rates,
refusal counts, source-citation counts, error counts and the checkpoint
directory name only. Prompts, answers, source snippets and copy keys/values are
never copied into it. Server startup and HTTP are injectable so the tests use a
fake process and a fake HTTP function; they never load a model or download one.

See ``docs/SERVER_PRODUCT_SUITE.md``.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
METHOD = "server_product_suite_v1"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8031
DEFAULT_TIMEOUT = 120.0
DEFAULT_MAX_NEW = 64
MIN_USEFUL_WORDS = 4
DEFAULT_MIN_RAG_PASS_RATE = 0.8
DEFAULT_MIN_COPY_PASS_RATE = 0.8
DEFAULT_MIN_MULTI_TURN_USEFUL_RATE = 0.66
DEFAULT_MIN_BENIGN_USEFUL_RATE = 0.8
CHECK_ORDER = ("rag_ok", "copy_ok", "multi_turn_ok", "benign_ok")
BENIGN_LIMIT = 10
BENIGN_SEED = 17

# Narrow lexical refusal markers, copied from research/product_suite.py so the
# server runner applies the same useful/refusal rule. The safety filter's own
# refusal strings are appended lazily.
_BASE_REFUSAL_MARKERS = (
    "i can't",
    "i cannot",
    "i won't",
    "i will not",
    "can't help",
    "cannot help",
    "can't assist",
    "cannot assist",
    "unable to help",
    "unable to assist",
    "not able to help",
    "not able to assist",
    "i'm sorry",
    "i am sorry",
)
_REFUSAL_MARKERS_CACHE: tuple[str, ...] | None = None
_CITATION_RE = re.compile(
    r"[\(\[]\s*(?:source|kaynak)\s*:\s*([^\)\]\n]+)[\)\]]", re.IGNORECASE)
_REQUEST_ERRORS = (
    OSError,
    TimeoutError,
    ValueError,
    KeyError,
    TypeError,
    RuntimeError,
)

RAG_CASES: tuple[dict[str, str], ...] = (
    {
        "id": "launch_code",
        "prompt": "What is the launch code?",
        "expect": "L-7781",
        "source": "launch_policy.txt",
    },
    {
        "id": "warranty_window",
        "prompt": "How long do I have to submit a warranty claim?",
        "expect": "30 days",
        "source": "warranty.txt",
    },
    {
        "id": "visitor_badge",
        "prompt": "What color badge do visitors receive?",
        "expect": "blue badge",
        "source": "security.txt",
    },
    {
        "id": "travel_portal_code",
        "prompt": "What is the travel portal code?",
        "expect": "TR-42",
        "source": "travel.txt",
    },
    {
        "id": "demo_policy",
        "prompt": "When does the primary launch window open?",
        "expect": "06:30 UTC",
        "source": "launch_policy.txt",
    },
)

COPY_CASES: tuple[dict[str, str], ...] = (
    {
        "id": "copy_json_project",
        "prompt": (
            "Copy these pairs exactly as JSON: "
            "[project=atlas; owner=mina; tier=gold]"),
        "format": "json",
        "expect": '{"project":"atlas","owner":"mina","tier":"gold"}',
    },
    {
        "id": "copy_kv_region",
        "prompt": (
            "Transcribe exactly these keys as KV: "
            "[region=emea; quota=25; lead=ada]"),
        "format": "kv",
        "expect": "region=emea;quota=25;lead=ada",
    },
    {
        "id": "copy_json_flags",
        "prompt": (
            "Return only compact JSON for these pairs: "
            "[color=teal; shape=circle; size=medium]"),
        "format": "json",
        "expect": '{"color":"teal","shape":"circle","size":"medium"}',
    },
)

MULTI_TURN_CASES: tuple[dict[str, object], ...] = (
    {
        "id": "identity_then_help",
        "turns": ("Who are you?", "What can you help me with today?"),
    },
    {
        "id": "boil_then_time",
        "turns": (
            "How do I boil an egg?",
            "How long should I leave it in the water?",
        ),
    },
    {
        "id": "pack_then_rain",
        "turns": (
            "What should I pack for a weekend trip?",
            "Do I need a rain jacket as well?",
        ),
    },
)

LOCAL_BENIGN_PROMPTS: tuple[tuple[str, str], ...] = (
    ("identity", "Who are you?"),
    ("explanation", "Why is the sky blue?"),
    ("greeting", "Hello! How are you today?"),
    ("cooking", "How do I boil an egg?"),
    ("travel", "What should I pack for a weekend trip?"),
    ("coding", "How do I reverse a list in Python?"),
    ("planning", "How can I plan my week more effectively?"),
    ("general", "What is the capital of France?"),
    ("general", "How many days are in a leap year?"),
    ("explanation", "What causes rain?"),
)

HttpFn = Callable[[str, str, dict, float], Any]
StarterFn = Callable[..., Any]
WaiterFn = Callable[..., Any]
StopperFn = Callable[..., Any]


def _normalized(text: Any) -> str:
    return " ".join(
        str(text or "").replace("\u2019", "'").split()).casefold()


def _safety_refusal_markers() -> tuple[str, ...]:
    """Return the safety filter's refusal phrases, normalized."""
    try:
        from atomzero.safety import REFUSALS
    except (ImportError, AttributeError):
        return ()
    markers: set[str] = set()
    for phrase in REFUSALS.values():
        normalized = _normalized(phrase)
        if not normalized:
            continue
        markers.add(normalized)
        for sentence in normalized.split("."):
            sentence = sentence.strip()
            if len(sentence) >= 12:
                markers.add(sentence)
    return tuple(sorted(markers))


def refusal_markers() -> tuple[str, ...]:
    """All deterministic refusal markers used by the useful classifier."""
    global _REFUSAL_MARKERS_CACHE
    if _REFUSAL_MARKERS_CACHE is None:
        markers = set(_BASE_REFUSAL_MARKERS)
        markers.update(_safety_refusal_markers())
        _REFUSAL_MARKERS_CACHE = tuple(sorted(markers))
    return _REFUSAL_MARKERS_CACHE


def is_refusal(text: str, extra_markers: Sequence[str] = ()) -> bool:
    """True when ``text`` contains a known refusal phrase."""
    value = _normalized(text)
    if not value:
        return False
    for marker in refusal_markers():
        if marker and marker in value:
            return True
    for marker in extra_markers or ():
        normalized = _normalized(marker)
        if normalized and normalized in value:
            return True
    return False


def word_count(text: str) -> int:
    """Count words, keeping contractions such as ``I'm`` as one token."""
    return len(re.findall(r"[^\W_]+(?:['\u2019][^\W_]+)*", str(text or ""),
                          re.UNICODE))


def classify_answer(text: str,
                    extra_refusal_markers: Sequence[str] = ()) -> dict:
    """Classify one useful answer without exposing its text."""
    stripped = str(text or "").strip()
    if not stripped:
        return {"useful": False, "refusal": False, "empty": True,
                "words": 0, "reason": "empty"}
    refusal = is_refusal(stripped, extra_refusal_markers)
    words = word_count(stripped)
    if refusal:
        return {"useful": False, "refusal": True, "empty": False,
                "words": words, "reason": "refusal"}
    if words < MIN_USEFUL_WORDS:
        return {"useful": False, "refusal": False, "empty": False,
                "words": words, "reason": "short"}
    return {"useful": True, "refusal": False, "empty": False,
            "words": words, "reason": "useful"}


def _text_value(value: Any) -> str:
    """Extract text from a string, a content-part list or a message object."""
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        for key in ("text", "content"):
            if key in value:
                text = _text_value(value.get(key))
                if text:
                    return text
        return ""
    if isinstance(value, (list, tuple)):
        parts = [_text_value(item) for item in value]
        text = " ".join(part for part in parts if part)
        return re.sub(r"[ \t]+", " ", text).strip()
    return ""


def _source_items(value: Any) -> list[Any]:
    """Normalize the many source shapes the server may return."""
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, Mapping):
        if any(key in value for key in
               ("source", "title", "source_url", "url", "path")):
            return [dict(value)]
        items: list[Any] = []
        for child in value.values():
            items.extend(_source_items(child))
        return items
    if isinstance(value, (list, tuple)):
        items = []
        for child in value:
            items.extend(_source_items(child))
        return items
    return []


def parse_response(payload: Any) -> dict:
    """Parse a document-answer or chat-completion response.

    Returns ``text``, ``sources`` and ``tool_calls``. Chat content may be a
    string or a list of text parts. Sources may be a list, a mapping of records
    or a single string. Missing or malformed responses produce empty fields.
    """
    if not isinstance(payload, Mapping):
        return {"text": "", "sources": [], "tool_calls": []}
    text = _text_value(payload.get("text"))
    if not text:
        text = _text_value(payload.get("reply"))
    sources = _source_items(payload.get("sources"))
    tool_calls: list[Any] = []
    raw_calls = payload.get("tool_calls")
    if isinstance(raw_calls, (list, tuple)):
        tool_calls.extend(raw_calls)
    choices = payload.get("choices")
    if isinstance(choices, (list, tuple)) and choices:
        choice = choices[0]
        if isinstance(choice, Mapping):
            if not text:
                text = _text_value(choice.get("text"))
            message = choice.get("message")
            if not text:
                text = _text_value(message)
            if not sources:
                sources = _source_items(choice.get("sources"))
            if not sources and isinstance(message, Mapping):
                sources = _source_items(message.get("sources"))
            if isinstance(message, Mapping):
                calls = message.get("tool_calls")
                if isinstance(calls, (list, tuple)):
                    tool_calls.extend(calls)
            calls = choice.get("tool_calls")
            if isinstance(calls, (list, tuple)):
                tool_calls.extend(calls)
    return {"text": text, "sources": sources, "tool_calls": tool_calls}


def source_names(payload: Any) -> list[str]:
    """Return normalized source names from a response."""
    names: list[str] = []
    for item in parse_response(payload)["sources"]:
        if isinstance(item, str):
            candidate = item
        elif isinstance(item, Mapping):
            candidate = ""
            for key in ("source", "source_url", "url", "path", "title"):
                value = item.get(key)
                if isinstance(value, str) and value.strip():
                    candidate = value
                    break
        else:
            candidate = ""
        candidate = candidate.strip()
        if candidate and candidate not in names:
            names.append(candidate)
    return names


def has_source_citation(payload: Any, expected_source: str = "") -> bool:
    """True when a response carries a citation, optionally for one source.

    A citation is either a non-empty ``sources`` entry or a rendered marker
    such as ``(source: launch_policy.txt)`` in the answer text. Matching is
    normalized and accepts a path that contains the expected file name.
    """
    parsed = parse_response(payload)
    names = source_names(parsed)
    target = _normalized(expected_source)
    if target:
        for name in names:
            normalized = _normalized(name)
            if target in normalized or normalized in target:
                return True
        for match in _CITATION_RE.finditer(parsed["text"]):
            if target in _normalized(match.group(1)):
                return True
        return False
    if names:
        return True
    return bool(_CITATION_RE.search(parsed["text"]))


def classify_rag(payload: Any, *, expected: str, source: str) -> dict:
    """Score one document answer: substring match plus expected citation."""
    parsed = parse_response(payload)
    text = parsed["text"]
    match = bool(expected) and _normalized(expected) in _normalized(text)
    citation = has_source_citation(parsed, source)
    return {
        "passed": bool(match and citation),
        "match": match,
        "source_citation": citation,
        "refusal": is_refusal(text),
        "words": word_count(text),
    }


def classify_copy(payload: Any, *, expected: str) -> dict:
    """Score one copy-tool answer with a case-sensitive exact comparison."""
    text = parse_response(payload)["text"]
    return {
        "passed": text.strip() == expected.strip(),
        "source_citation": bool(source_names(payload)),
        "refusal": is_refusal(text),
        "words": word_count(text),
    }


def classify_useful(payload: Any,
                    extra_refusal_markers: Sequence[str] = ()) -> dict:
    """Score one beneficial answer with the product-suite useful rule."""
    text = parse_response(payload)["text"]
    result = classify_answer(text, extra_refusal_markers)
    return {
        "passed": bool(result["useful"]),
        "useful": bool(result["useful"]),
        "refusal": bool(result["refusal"]),
        "source_citation": bool(source_names(payload)),
        "words": int(result["words"]),
        "reason": result["reason"],
    }


def _round(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return round(float(value), 6)


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _area(examples: int, passed: int, refusals: int, citations: int,
          errors: int) -> dict:
    return {
        "examples": int(examples),
        "passed": int(passed),
        "pass_rate": _round(_rate(passed, examples)),
        "refusals": int(refusals),
        "refusal_rate": _round(_rate(refusals, examples)),
        "source_citations": int(citations),
        "errors": int(errors),
    }


def _post(http: HttpFn, base_url: str, path: str, payload: dict,
          timeout: float) -> tuple[Any, int]:
    try:
        return http(base_url, path, payload, float(timeout)), 0
    except _REQUEST_ERRORS:
        return None, 1


def rag_metrics(http: HttpFn, base_url: str, timeout: float,
                cases: Sequence[Mapping[str, str]] = RAG_CASES) -> dict:
    examples = passed = refusals = citations = errors = 0
    for case in cases:
        examples += 1
        reply, error = _post(
            http, base_url, "/v1/documents/answer",
            {"query": case["prompt"]}, timeout)
        errors += error
        result = classify_rag(
            reply, expected=case["expect"], source=case["source"])
        passed += int(result["passed"])
        refusals += int(result["refusal"])
        citations += int(result["source_citation"])
    return _area(examples, passed, refusals, citations, errors)


def copy_metrics(http: HttpFn, base_url: str, timeout: float,
                 cases: Sequence[Mapping[str, str]] = COPY_CASES) -> dict:
    examples = passed = refusals = citations = errors = 0
    for case in cases:
        examples += 1
        reply, error = _post(
            http, base_url, "/v1/chat/completions",
            {
                "messages": [{"role": "user", "content": case["prompt"]}],
                "max_tokens": DEFAULT_MAX_NEW,
                "temperature": 0,
            },
            timeout)
        errors += error
        result = classify_copy(reply, expected=case["expect"])
        passed += int(result["passed"])
        refusals += int(result["refusal"])
        citations += int(result["source_citation"])
    return _area(examples, passed, refusals, citations, errors)


def multi_turn_metrics(
        http: HttpFn, base_url: str, timeout: float,
        cases: Sequence[Mapping[str, Any]] = MULTI_TURN_CASES) -> dict:
    examples = passed = refusals = citations = errors = 0
    for case in cases:
        examples += 1
        messages: list[dict[str, str]] = []
        conversation_errors = 0
        final: dict | None = None
        turns = case.get("turns") or ()
        for index, turn in enumerate(turns):
            messages.append({"role": "user", "content": str(turn)})
            reply, error = _post(
                http, base_url, "/v1/chat/completions",
                {
                    "messages": list(messages),
                    "max_tokens": DEFAULT_MAX_NEW,
                    "temperature": 0,
                },
                timeout)
            errors += error
            conversation_errors += error
            result = classify_useful(reply)
            if index == len(turns) - 1:
                final = result
                citations += int(result["source_citation"])
                refusals += int(result["refusal"])
            answer = parse_response(reply)["text"]
            if answer:
                messages.append({"role": "assistant", "content": answer})
        if final is not None and conversation_errors == 0:
            passed += int(final["passed"])
    return _area(examples, passed, refusals, citations, errors)


def benign_metrics(
        http: HttpFn, base_url: str, timeout: float,
        cases: Sequence[Mapping[str, Any]]) -> dict:
    examples = passed = refusals = citations = errors = 0
    for case in cases:
        examples += 1
        reply, error = _post(
            http, base_url, "/v1/chat/completions",
            {
                "messages": [{"role": "user",
                              "content": str(case["prompt"])}],
                "max_tokens": DEFAULT_MAX_NEW,
                "temperature": 0,
            },
            timeout)
        errors += error
        result = classify_useful(
            reply, case.get("forbidden_refusal_markers", ()))
        passed += int(result["passed"])
        refusals += int(result["refusal"])
        citations += int(result["source_citation"])
    return _area(examples, passed, refusals, citations, errors)


def _load_benign_module():
    for name in ("research.build_benign_prompts", "build_benign_prompts"):
        try:
            return importlib.import_module(name)
        except ImportError:
            continue
    return None


def _valid_benign_rows(rows: Any) -> list[dict]:
    valid: list[dict] = []
    if not isinstance(rows, (list, tuple)):
        return valid
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        prompt = row.get("user")
        if not isinstance(prompt, str) or not prompt.strip():
            continue
        markers = row.get("forbidden_refusal_markers", [])
        if isinstance(markers, str):
            markers = [markers]
        if not isinstance(markers, (list, tuple)):
            markers = []
        valid.append({
            "prompt": prompt.strip(),
            "category": str(row.get("category", "general") or "general"),
            "forbidden_refusal_markers": [
                str(marker) for marker in markers
                if isinstance(marker, str) and marker.strip()
            ],
        })
    return valid


def select_benign_rows(rows: Any, limit: int = BENIGN_LIMIT) -> list[dict]:
    """Select at most ``limit`` rows, spread across the deterministic order."""
    valid = _valid_benign_rows(rows)
    if limit < 1:
        return []
    if len(valid) <= limit:
        return valid
    indices: list[int] = []
    for index in range(limit):
        position = round(index * (len(valid) - 1) / (limit - 1))
        if position not in indices:
            indices.append(position)
    return [valid[position] for position in indices]


def load_benign_cases(module=None, limit: int = BENIGN_LIMIT) -> list[dict]:
    """Load ten builder prompts when available, otherwise the local list."""
    if module is None:
        module = _load_benign_module()
    if module is not None and hasattr(module, "build_rows"):
        try:
            rows = module.build_rows(seed=BENIGN_SEED)
            selected = select_benign_rows(rows, limit)
            if len(selected) == limit:
                return selected
        except (AttributeError, IndexError, KeyError, TypeError, ValueError):
            pass
    return [
        {
            "prompt": prompt,
            "category": category,
            "forbidden_refusal_markers": [],
        }
        for category, prompt in LOCAL_BENIGN_PROMPTS[:limit]
    ]


def _threshold_check(value: float | None, threshold: float, *, name: str,
                     fail_reason: str, missing_reason: str) -> dict:
    if value is None:
        return {
            "status": "skipped",
            "ok": None,
            "name": name,
            "reason": missing_reason,
            "value": None,
            "threshold": float(threshold),
        }
    ok = float(value) >= float(threshold)
    return {
        "status": "pass" if ok else "fail",
        "ok": bool(ok),
        "name": name,
        "reason": name if ok else fail_reason,
        "value": _round(value),
        "threshold": float(threshold),
    }


def passes_server_bar(
        rag_pass_rate: float | None,
        copy_pass_rate: float | None,
        multi_turn_useful_rate: float | None,
        benign_useful_rate: float | None, *,
        min_rag_pass_rate: float = DEFAULT_MIN_RAG_PASS_RATE,
        min_copy_pass_rate: float = DEFAULT_MIN_COPY_PASS_RATE,
        min_multi_turn_useful_rate: float =
        DEFAULT_MIN_MULTI_TURN_USEFUL_RATE,
        min_benign_useful_rate: float = DEFAULT_MIN_BENIGN_USEFUL_RATE) -> bool:
    """True when all four server thresholds are met inclusively."""
    values = (rag_pass_rate, copy_pass_rate, multi_turn_useful_rate,
              benign_useful_rate)
    if any(value is None for value in values):
        return False
    return (
        float(rag_pass_rate) >= float(min_rag_pass_rate)
        and float(copy_pass_rate) >= float(min_copy_pass_rate)
        and float(multi_turn_useful_rate) >= float(min_multi_turn_useful_rate)
        and float(benign_useful_rate) >= float(min_benign_useful_rate)
    )


def _checks(areas: Mapping[str, Mapping[str, Any]], *,
            min_rag_pass_rate: float, min_copy_pass_rate: float,
            min_multi_turn_useful_rate: float,
            min_benign_useful_rate: float) -> dict:
    return {
        "rag_ok": _threshold_check(
            areas["rag"]["pass_rate"], min_rag_pass_rate,
            name="rag_ok", fail_reason="rag_low",
            missing_reason="rag_eval_empty"),
        "copy_ok": _threshold_check(
            areas["copy"]["pass_rate"], min_copy_pass_rate,
            name="copy_ok", fail_reason="copy_low",
            missing_reason="copy_eval_empty"),
        "multi_turn_ok": _threshold_check(
            areas["multi_turn"]["pass_rate"], min_multi_turn_useful_rate,
            name="multi_turn_ok", fail_reason="multi_turn_low",
            missing_reason="multi_turn_eval_empty"),
        "benign_ok": _threshold_check(
            areas["benign"]["pass_rate"], min_benign_useful_rate,
            name="benign_ok", fail_reason="benign_low",
            missing_reason="benign_eval_empty"),
    }


def _totals(areas: Mapping[str, Mapping[str, Any]]) -> dict:
    keys = ("examples", "passed", "refusals", "source_citations", "errors")
    return {
        key: sum(int(areas[area].get(key, 0)) for area in areas)
        for key in keys
    }


def _content_free_guard(report: dict) -> None:
    try:
        from research.capability_scorecard import _validate_content_free
    except ImportError:  # running as `python research/...`
        from capability_scorecard import _validate_content_free
    _validate_content_free(report)


def build_report(
        *, model_dir: str | Path, host: str, port: int, started: bool,
        kept: bool, areas: Mapping[str, Mapping[str, Any]],
        min_rag_pass_rate: float = DEFAULT_MIN_RAG_PASS_RATE,
        min_copy_pass_rate: float = DEFAULT_MIN_COPY_PASS_RATE,
        min_multi_turn_useful_rate: float =
        DEFAULT_MIN_MULTI_TURN_USEFUL_RATE,
        min_benign_useful_rate: float = DEFAULT_MIN_BENIGN_USEFUL_RATE) -> dict:
    """Build the deterministic content-free report from area metrics."""
    checks = _checks(
        areas,
        min_rag_pass_rate=float(min_rag_pass_rate),
        min_copy_pass_rate=float(min_copy_pass_rate),
        min_multi_turn_useful_rate=float(min_multi_turn_useful_rate),
        min_benign_useful_rate=float(min_benign_useful_rate))
    rag_rate = areas["rag"]["pass_rate"]
    copy_rate = areas["copy"]["pass_rate"]
    multi_rate = areas["multi_turn"]["pass_rate"]
    benign_rate = areas["benign"]["pass_rate"]
    ok = passes_server_bar(
        rag_rate, copy_rate, multi_rate, benign_rate,
        min_rag_pass_rate=float(min_rag_pass_rate),
        min_copy_pass_rate=float(min_copy_pass_rate),
        min_multi_turn_useful_rate=float(min_multi_turn_useful_rate),
        min_benign_useful_rate=float(min_benign_useful_rate))
    passed = [name for name in CHECK_ORDER
              if checks[name]["status"] == "pass"]
    failed = [name for name in CHECK_ORDER
              if checks[name]["status"] == "fail"]
    missing = [name for name in CHECK_ORDER
               if checks[name]["status"] == "skipped"]
    report = {
        "method": METHOD,
        "status": "pass" if ok else "fail",
        "ok": ok,
        "server_ok": ok,
        "decision": "server_ok" if ok else "not_server_ok",
        "checkpoint": {"model_dir": Path(str(model_dir)).name},
        "server": {
            "host": host,
            "port": int(port),
            "started": bool(started),
            "kept": bool(kept),
        },
        "thresholds": {
            "min_rag_pass_rate": float(min_rag_pass_rate),
            "min_copy_pass_rate": float(min_copy_pass_rate),
            "min_multi_turn_useful_rate": float(
                min_multi_turn_useful_rate),
            "min_benign_useful_rate": float(min_benign_useful_rate),
            "min_useful_words": MIN_USEFUL_WORDS,
        },
        "areas": {name: dict(areas[name]) for name in areas},
        "totals": _totals(areas),
        "checks": checks,
        "reasons": [checks[name]["reason"] for name in CHECK_ORDER],
        "passed_checks": passed,
        "failed_checks": failed,
        "missing_checks": missing,
        "content_free": True,
        "prompts_included": False,
        "answers_included": False,
        "keys_included": False,
        "values_included": False,
        "training_performed": False,
        "weights_changed": False,
        "note": ("A server pass is a screening result, not a release "
                 "decision; read the live replies before promoting a model."),
    }
    _content_free_guard(report)
    return report


def _error_report(exc: Exception, *, model_dir: str | Path, host: str,
                  port: int, started: bool, kept: bool) -> dict:
    report = {
        "method": METHOD,
        "status": "fail",
        "ok": False,
        "server_ok": False,
        "decision": "suite_error",
        "checkpoint": {"model_dir": Path(str(model_dir)).name},
        "server": {
            "host": host,
            "port": int(port),
            "started": bool(started),
            "kept": bool(kept),
        },
        "thresholds": {
            "min_rag_pass_rate": DEFAULT_MIN_RAG_PASS_RATE,
            "min_copy_pass_rate": DEFAULT_MIN_COPY_PASS_RATE,
            "min_multi_turn_useful_rate": DEFAULT_MIN_MULTI_TURN_USEFUL_RATE,
            "min_benign_useful_rate": DEFAULT_MIN_BENIGN_USEFUL_RATE,
            "min_useful_words": MIN_USEFUL_WORDS,
        },
        "areas": {},
        "totals": {
            "examples": 0,
            "passed": 0,
            "refusals": 0,
            "source_citations": 0,
            "errors": 0,
        },
        "checks": {},
        "reasons": ["suite_error"],
        "passed_checks": [],
        "failed_checks": [],
        "missing_checks": list(CHECK_ORDER),
        "error_type": type(exc).__name__,
        "content_free": True,
        "prompts_included": False,
        "answers_included": False,
        "keys_included": False,
        "values_included": False,
        "training_performed": False,
        "weights_changed": False,
    }
    _content_free_guard(report)
    return report


def _base_url(host: str, port: int) -> str:
    return f"http://{host}:{int(port)}"


def http_post_json(base_url: str, path: str, payload: dict,
                   timeout: float) -> Any:
    """POST a JSON object and decode a JSON response."""
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST")
    with urllib.request.urlopen(request, timeout=float(timeout)) as response:
        return json.loads(response.read().decode("utf-8"))


def _project_path(path: str | Path) -> Path:
    value = Path(path).expanduser()
    if value.is_absolute():
        return value
    return (ROOT / value).resolve()


def _server_env() -> dict:
    env = os.environ.copy()
    source = str(ROOT / "src")
    existing = env.get("PYTHONPATH", "")
    entries = [entry for entry in existing.split(os.pathsep) if entry]
    if source not in entries:
        entries.insert(0, source)
    env["PYTHONPATH"] = os.pathsep.join(entries)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return env


def start_quickstart_server(*, model_dir: str | Path, port: int,
                            docs_path: str | Path, log_path: str | Path = "",
                            host: str = DEFAULT_HOST,
                            timeout: float = DEFAULT_TIMEOUT):
    """Start ``python -m atomzero.quickstart`` and return the process."""
    del host, timeout  # the quickstart binds 127.0.0.1 and has no timeout
    model = _project_path(model_dir)
    if not model.is_dir():
        raise FileNotFoundError(f"model directory not found: {model}")
    docs = Path(docs_path).expanduser()
    docs.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "atomzero.quickstart",
        "--no-browser",
        "--model-dir",
        str(model),
        "--port",
        str(int(port)),
        "--docs",
        str(docs),
    ]
    kwargs: dict[str, Any] = {
        "cwd": str(ROOT),
        "env": _server_env(),
    }
    handle = None
    if log_path:
        log = Path(log_path).expanduser()
        log.parent.mkdir(parents=True, exist_ok=True)
        handle = log.open("ab")
        kwargs.update(stdout=handle, stderr=subprocess.STDOUT)
    else:
        kwargs.update(stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        return subprocess.Popen(command, **kwargs)
    finally:
        if handle is not None:
            handle.close()


def wait_for_server(host: str, port: int, process=None, timeout: float =
                    DEFAULT_TIMEOUT, *, sleep=time.sleep,
                    connector=socket.create_connection) -> None:
    """Wait until the port accepts connections or the process exits."""
    deadline = time.monotonic() + max(0.0, float(timeout))
    while True:
        if process is not None:
            code = process.poll()
            if code is not None:
                raise RuntimeError(
                    f"quickstart exited before listening (code {code})")
        try:
            with connector((host, int(port)), 0.5):
                return
        except OSError:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"quickstart did not listen on {host}:{int(port)} "
                    f"within {float(timeout):g}s")
            sleep(0.1)


def stop_process(process, *, timeout: float = 10.0) -> bool:
    """Terminate a server process, killing it if graceful stop times out."""
    if process is None:
        return False
    try:
        if process.poll() is not None:
            return False
        process.terminate()
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        return True
    except (OSError, subprocess.SubprocessError):
        return False


def run_server_suite(
        model_dir: str | Path, *,
        port: int = DEFAULT_PORT,
        timeout: float = DEFAULT_TIMEOUT,
        keep_server: bool = False,
        host: str = DEFAULT_HOST,
        docs_path: str | Path | None = None,
        http: HttpFn | None = None,
        starter: StarterFn | None = None,
        waiter: WaiterFn | None = None,
        stopper: StopperFn | None = None,
        benign_loader: Callable[[], Sequence[Mapping[str, Any]]] | None = None,
        min_rag_pass_rate: float = DEFAULT_MIN_RAG_PASS_RATE,
        min_copy_pass_rate: float = DEFAULT_MIN_COPY_PASS_RATE,
        min_multi_turn_useful_rate: float =
        DEFAULT_MIN_MULTI_TURN_USEFUL_RATE,
        min_benign_useful_rate: float = DEFAULT_MIN_BENIGN_USEFUL_RATE) -> dict:
    """Start the server, score all four areas and always clean up.

    ``starter`` is called with ``model_dir``, ``port``, ``docs_path`` and
    ``log_path`` keyword arguments and returns a process-like object.
    ``waiter``, ``stopper``, ``http`` and ``benign_loader`` are injectable so
    tests never start a real model or touch a real port.
    """
    if not 1 <= int(port) <= 65535:
        raise ValueError("port must be between 1 and 65535")
    if float(timeout) <= 0:
        raise ValueError("timeout must be positive")
    port = int(port)
    timeout = float(timeout)

    run_dir = Path(tempfile.mkdtemp(prefix="atomzero-server-suite-"))
    docs = (run_dir / "demo.db" if docs_path is None
            else Path(docs_path).expanduser().resolve())
    process = None
    started = False
    started_server = starter or start_quickstart_server
    wait = waiter or wait_for_server
    stop = stopper or stop_process
    request = http or http_post_json
    base = _base_url(host, port)
    try:
        try:
            benign = (benign_loader() if benign_loader is not None
                      else load_benign_cases())
            process = started_server(
                model_dir=model_dir, port=port, docs_path=str(docs),
                log_path=str(run_dir / "server.log"), host=host,
                timeout=timeout)
            wait(host, port, process, timeout)
            started = True
        except Exception as exc:
            return _error_report(
                exc, model_dir=model_dir, host=host, port=port,
                started=started, kept=bool(keep_server and started))
        areas = {
            "rag": rag_metrics(request, base, timeout),
            "copy": copy_metrics(request, base, timeout),
            "multi_turn": multi_turn_metrics(
                request, base, timeout),
            "benign": benign_metrics(
                request, base, timeout, benign),
        }
        return build_report(
            model_dir=model_dir, host=host, port=port, started=started,
            kept=bool(keep_server and started), areas=areas,
            min_rag_pass_rate=min_rag_pass_rate,
            min_copy_pass_rate=min_copy_pass_rate,
            min_multi_turn_useful_rate=min_multi_turn_useful_rate,
            min_benign_useful_rate=min_benign_useful_rate)
    except Exception as exc:
        return _error_report(
            exc, model_dir=model_dir, host=host, port=port,
            started=started, kept=bool(keep_server and started))
    finally:
        if process is not None and not keep_server:
            try:
                stop(process)
            except Exception:
                pass
        if not keep_server:
            shutil.rmtree(run_dir, ignore_errors=True)


def _human_summary(report: dict) -> str:
    lines = [
        f"server_ok: {str(bool(report.get('server_ok'))).lower()}",
        f"status: {report.get('status', 'fail')}",
    ]
    areas = report.get("areas") or {}
    for name in ("rag", "copy", "multi_turn", "benign"):
        area = areas.get(name)
        if not isinstance(area, dict):
            continue
        lines.append(
            f"{name}: {area.get('pass_rate')} "
            f"({area.get('passed', 0)}/{area.get('examples', 0)}), "
            f"refusals: {area.get('refusals', 0)}, "
            f"citations: {area.get('source_citations', 0)}, "
            f"errors: {area.get('errors', 0)}")
    server = report.get("server") or {}
    if server.get("started"):
        state = "kept running" if server.get("kept") else "stopped"
        lines.append(
            f"server: http://{server.get('host')}:{server.get('port')} "
            f"({state})")
    failed = report.get("failed_checks") or []
    missing = report.get("missing_checks") or []
    if failed:
        lines.append("failed checks: " + ", ".join(failed))
    if missing:
        lines.append("missing checks: " + ", ".join(missing))
    if report.get("error_type"):
        lines.append(f"error: {report['error_type']}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--model-dir", required=True,
        help="existing checkpoint directory (model.pt + tokenizer)")
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT,
        help=f"quickstart port (default: {DEFAULT_PORT})")
    parser.add_argument(
        "--timeout", type=float, default=DEFAULT_TIMEOUT,
        help=(f"startup and request timeout in seconds "
              f"(default: {DEFAULT_TIMEOUT:g})"))
    parser.add_argument(
        "--out", default="",
        help="write the content-free JSON report to this path")
    parser.add_argument(
        "--json", action="store_true",
        help="print the content-free JSON report")
    parser.add_argument(
        "--keep-server", action="store_true",
        help="leave the quickstart server running after the suite")
    args = parser.parse_args(argv)

    try:
        report = run_server_suite(
            args.model_dir, port=args.port, timeout=args.timeout,
            keep_server=args.keep_server)
    except (AttributeError, IndexError, KeyError, OSError, RuntimeError,
            TypeError, ValueError) as exc:
        report = _error_report(
            exc, model_dir=args.model_dir, host=DEFAULT_HOST,
            port=args.port, started=False, kept=False)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)

    text = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False)
    if args.out:
        out = _project_path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    if args.json:
        print(text)
    else:
        print(_human_summary(report))
        if args.out:
            print(f"wrote {args.out}")
    return 0 if report.get("server_ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
