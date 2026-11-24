"""Conservative router for explicit copy requests.

The chat path normally sends every prompt through document retrieval and the
model. Some prompts, however, already contain the exact key/value pairs they
want back. Those are transcription requests, not knowledge questions, and the
small model cannot transcribe them reliably.

This module recognises only that narrow case and hands it to the deterministic
:mod:`atomzero.copy_tool`. It is deliberately cautious: questions, summaries,
paraphrases and ordinary chat return ``None`` so the existing RAG/model path is
unchanged. Tool use here is not model self-improvement, training or ASI.

Public API::

    from atomzero.copy_router import detect_copy_request, route_copy

    fmt = detect_copy_request(prompt)       # "json", "kv", "yaml" or None
    result = route_copy(prompt)
    if result["handled"]:
        print(result["answer"])
"""
from __future__ import annotations

import re

from atomzero.copy_tool import extract_pairs, render_pairs

__all__ = ["detect_copy_request", "route_copy"]

MIN_PAIRS = 2
MAX_PAIRS = 4
MAX_PROMPT_CHARS = 4000

_BRACKET_RE = re.compile(r"\[[^\[\]]*\]", re.DOTALL)

# A pair sequence outside brackets normally starts after an instruction colon
# ("copy these pairs: a=1; b=2") or at the beginning of a line.
_PAIR_TAIL_RE = re.compile(
    r"(?:^|[:;\n])\s*(?P<body>[A-Za-z0-9_][A-Za-z0-9_.-]*"
    r"\s*[=:]\s*\S.*)$",
    re.DOTALL,
)

_COPY_INTENT_RE = re.compile(
    r"\b(?:copy|transcribe|echo)\b[^.;\n]{0,60}\b(?:pairs?|keys?|values?)\b"
    r"|\b(?:copy|transcribe|echo)\b[^.;\n]{0,20}\bexactly\b"
    r"|\breturn\s+only\b[^.;\n]{0,60}\b(?:json|yaml|yml|kv|"
    r"key\s*=\s*value|one[\s-]+line|compact|semicolon[\s-]+separated)\b"
    r"|\bexactly\s+these\s+(?:keys|pairs|values)\b"
    r"|\b(?:render|format|convert)\s+(?:these|the|following)\s+"
    r"(?:pairs|keys|values)\b",
    re.IGNORECASE,
)

_QUESTION_START_RE = re.compile(
    r"^(?:what|why|how|when|where|who|whom|whose|which|can|could|would|"
    r"should|do|does|did|is|are|was|were|explain|summari[sz]e|describe|"
    r"analy[sz]e|compare|discuss|interpret|paraphrase|rewrite|translate|"
    r"tell\s+me\s+about)\b",
    re.IGNORECASE,
)

_ANALYSIS_RE = re.compile(
    r"\b(?:summari[sz]e|summary|explain|describe|analy[sz]e|analysis|"
    r"compare|comparison|discuss|interpret|paraphrase|rewrite|"
    r"tell\s+me\s+about)\b",
    re.IGNORECASE,
)

_JSON_RE = re.compile(r"\bjson\b", re.IGNORECASE)
_YAML_RE = re.compile(r"\b(?:yaml|yml)\b", re.IGNORECASE)
_KV_RE = re.compile(
    r"\bkv\b|\bkey\s*=\s*value\b|\bone[\s-]+line\b|"
    r"\bsemicolon[\s-]+separated\b",
    re.IGNORECASE,
)
_NEGATED_FORMAT_RE = re.compile(
    r"\bnot\s+(?:json|yaml|yml|kv)\b", re.IGNORECASE)

# Used only when normal parsing fails, to distinguish a malformed explicit
# copy prompt (worth reporting) from ordinary prose.
_BRACKET_PAIR_RE = re.compile(
    r"\[[^\[\]]*[A-Za-z0-9_][A-Za-z0-9_.-]*\s*[=:]\s*[^\[\]\s][^\[\]]*\]",
    re.DOTALL,
)
_EQ_PAIR_RE = re.compile(
    r"(?:^|[;\[\n])\s*[A-Za-z0-9_][A-Za-z0-9_.-]*\s*=\s*\S",
    re.MULTILINE,
)


def _instruction_text(prompt: str) -> str:
    """Return the request text with pair blocks and pair tails removed."""
    text = _BRACKET_RE.sub(" ", prompt)
    match = _PAIR_TAIL_RE.search(text)
    if match:
        text = text[:match.start()]
    return text.strip()


def _is_question(instruction: str) -> bool:
    if "?" in instruction:
        return True
    return bool(_QUESTION_START_RE.match(instruction.strip()))


def _detect_format(instruction: str) -> str:
    clean = _NEGATED_FORMAT_RE.sub(" ", instruction)
    if _JSON_RE.search(clean):
        return "json"
    if _YAML_RE.search(clean):
        return "yaml"
    # Bare "copy these pairs" requests default to the pair syntax they use.
    return "kv"


def _pairs_from_prompt(prompt: str) -> list[tuple[str, str]]:
    """Extract pairs, retrying the pair tail after an instruction prefix."""
    try:
        return extract_pairs(prompt)
    except ValueError as first_error:
        match = _PAIR_TAIL_RE.search(prompt)
        if match:
            try:
                return extract_pairs(match.group("body"))
            except ValueError:
                pass
        raise first_error


def _has_pair_evidence(prompt: str) -> bool:
    if _BRACKET_PAIR_RE.search(prompt):
        return True
    return len(_EQ_PAIR_RE.findall(prompt)) >= 2


def detect_copy_request(prompt: str) -> str | None:
    """Return the requested copy format, or ``None`` for every other prompt.

    Detection is intentionally conservative. The prompt must contain an
    explicit transcription instruction (for example ``copy these pairs`` or
    ``return only compact JSON``) and pair-shaped content with 2-4 pairs.
    Questions, summaries, comparisons, paraphrases and open-ended chat return
    ``None`` even when they mention keys or values.
    """
    if not isinstance(prompt, str):
        return None
    text = prompt.strip()
    if not text or len(text) > MAX_PROMPT_CHARS:
        return None
    instruction = _instruction_text(text)
    if not instruction:
        return None
    if _is_question(instruction) or _ANALYSIS_RE.search(instruction):
        return None
    if not _COPY_INTENT_RE.search(instruction):
        return None
    try:
        pairs = _pairs_from_prompt(text)
    except ValueError:
        if not _has_pair_evidence(text):
            return None
        return _detect_format(instruction)
    if not MIN_PAIRS <= len(pairs) <= MAX_PAIRS:
        return None
    return _detect_format(instruction)


def route_copy(prompt: str) -> dict:
    """Route an explicit copy request to the deterministic copy tool.

    Returns ``{handled, format, answer, pairs_count, reason}``. This function
    never raises: an unsupported prompt, malformed pairs, an unsupported pair
    count or a render failure all return ``handled=False`` with a reason so the
    caller can fall back to the normal chat path.
    """
    result: dict = {
        "handled": False,
        "format": None,
        "answer": "",
        "pairs_count": 0,
        "reason": "not an explicit copy request",
    }
    try:
        fmt = detect_copy_request(prompt)
        if fmt is None:
            return result
        result["format"] = fmt
        try:
            pairs = _pairs_from_prompt(prompt)
        except (TypeError, ValueError) as exc:
            result["reason"] = f"could not parse pairs: {exc}"
            return result
        result["pairs_count"] = len(pairs)
        if not MIN_PAIRS <= len(pairs) <= MAX_PAIRS:
            result["reason"] = (
                f"expected {MIN_PAIRS}-{MAX_PAIRS} pairs, found {len(pairs)}")
            return result
        try:
            answer = render_pairs(pairs, fmt=fmt)
        except (TypeError, ValueError) as exc:
            result["reason"] = f"could not render {fmt}: {exc}"
            return result
        result.update({
            "handled": True,
            "answer": answer,
            "reason": "explicit copy request matched",
        })
        return result
    except Exception as exc:  # defensive: routing must never break chat
        result["handled"] = False
        result["answer"] = ""
        result["reason"] = f"copy routing failed: {exc}"
        return result
