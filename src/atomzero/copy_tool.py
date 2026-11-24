"""Deterministic runtime copy tool.

The 49.2M base model cannot reliably transcribe several arbitrary key/value
pairs. This module does the transcription in ordinary Python: it extracts the
pairs that an explicit request already contains and renders them in the
requested syntax. It is a tool, not a model capability and not a training
method. It never imports ``torch`` and never loads a checkpoint.

Supported pair syntax::

    key=value; key2=two words
    key: value
    key="quoted; value"

Pairs may sit directly in the text or inside one or more ``[...]`` blocks, as
in the staged copy prompts. Values may be quoted and may contain spaces,
semicolons, colons or newlines when quoted.

Public API::

    from atomzero.copy_tool import copy_pairs, extract_pairs
    pairs = extract_pairs("[a=1; b=hello world]")
    text = copy_pairs("[a=1; b=hello world]", fmt="json")
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from typing import Any

__all__ = [
    "FORMATS",
    "CopyTool",
    "copy_pairs",
    "extract_pairs",
    "normalize_format",
    "parse_rendered",
    "render_pairs",
]

FORMATS = ("json", "kv", "yaml")

_FORMAT_ALIASES = {
    "json": "json",
    "kv": "kv",
    "yaml": "yaml",
    "yml": "yaml",
}
_KEY_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$")
_YAML_PLAIN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_. -]*$")
_YAML_RESERVED = frozenset({
    "true", "false", "null", "none", "yes", "no", "on", "off", "~",
})
_UNESCAPE = {
    "n": "\n",
    "r": "\r",
    "t": "\t",
    "\\": "\\",
    '"': '"',
    "'": "'",
}


def normalize_format(fmt: str) -> str:
    """Return the canonical format name, accepting ``yml`` as ``yaml``."""
    if not isinstance(fmt, str):
        raise ValueError(
            f"format must be a string, got {type(fmt).__name__}")
    name = fmt.strip().lower()
    if name not in _FORMAT_ALIASES:
        raise ValueError(
            f"unknown format {fmt!r}; expected one of {', '.join(FORMATS)}")
    return _FORMAT_ALIASES[name]


def _preview(text: str, limit: int = 80) -> str:
    text = text.replace("\n", "\\n").replace("\r", "\\r")
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _unescape(text: str, quote: str) -> str:
    out: list[str] = []
    index = 0
    while index < len(text):
        char = text[index]
        if char == "\\" and index + 1 < len(text):
            nxt = text[index + 1]
            if nxt == quote:
                out.append(quote)
                index += 2
                continue
            if nxt in _UNESCAPE:
                out.append(_UNESCAPE[nxt])
                index += 2
                continue
        out.append(char)
        index += 1
    return "".join(out)


def _opens_quote(text: str, index: int, start: int = 0) -> bool:
    """Treat a quote as syntax only at a token start, not inside a word."""
    if index <= start:
        return True
    previous = text[index - 1]
    return not (previous.isalnum() or previous in "_-")


def _split_segments(text: str) -> list[str]:
    """Split on ``;`` or newlines that are outside quotes."""
    segments: list[str] = []
    start = 0
    quote: str | None = None
    escaped = False
    index = 0
    while index < len(text):
        char = text[index]
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            index += 1
            continue
        if char in "\"'" and _opens_quote(text, index, start):
            quote = char
        elif char in ";\n\r":
            segments.append(text[start:index])
            start = index + 1
        index += 1
    if quote is not None:
        raise ValueError("unterminated quoted value")
    segments.append(text[start:])
    return [segment.strip() for segment in segments if segment.strip()]


def _find_separator(segment: str) -> int:
    """Find the first top-level ``=`` or ``:`` in a pair segment."""
    quote: str | None = None
    escaped = False
    for index, char in enumerate(segment):
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in "\"'" and _opens_quote(segment, index):
            quote = char
        elif char in "=:":
            return index
    if quote is not None:
        raise ValueError("unterminated quoted value")
    return -1


def _decode_key(raw: str, index: int) -> str:
    key = raw.strip()
    if len(key) >= 2 and key[0] == key[-1] and key[0] in "\"'":
        key = _unescape(key[1:-1], key[0])
    if not key:
        raise ValueError(f"pair {index}: missing key before the separator")
    if not _KEY_RE.fullmatch(key):
        raise ValueError(
            f"pair {index}: invalid key {key!r}; keys must start with a "
            "letter, digit or underscore and contain only letters, digits, "
            "underscore, dot or hyphen")
    return key


def _decode_value(raw: str, index: int) -> str:
    value = raw.strip()
    if not value:
        raise ValueError(f"pair {index}: missing value after the separator")
    if value[0] in "\"'":
        if len(value) < 2 or value[-1] != value[0]:
            raise ValueError(
                f"pair {index}: unterminated quoted value {_preview(value)!r}")
        return _unescape(value[1:-1], value[0])
    return value


def _parse_body(text: str) -> list[tuple[str, str]]:
    segments = _split_segments(text)
    if not segments:
        raise ValueError("no key=value or key: value pairs found")
    pairs: list[tuple[str, str]] = []
    seen: set[str] = set()
    for index, segment in enumerate(segments, start=1):
        separator = _find_separator(segment)
        if separator < 0:
            raise ValueError(
                f"malformed pair {index}: expected key=value or key: value, "
                f"got {_preview(segment)!r}")
        key = _decode_key(segment[:separator], index)
        value = _decode_value(segment[separator + 1:], index)
        if key in seen:
            raise ValueError(f"duplicate key {key!r} (pair {index})")
        seen.add(key)
        pairs.append((key, value))
    return pairs


def _bracket_blocks(text: str) -> list[str]:
    """Return the contents of top-level ``[...]`` blocks."""
    blocks: list[str] = []
    index = 0
    length = len(text)
    while index < length:
        if text[index] != "[":
            index += 1
            continue
        start = index + 1
        cursor = start
        quote: str | None = None
        escaped = False
        while cursor < length:
            char = text[cursor]
            if quote is not None:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == quote:
                    quote = None
                cursor += 1
                continue
            if char in "\"'" and _opens_quote(text, cursor, start):
                quote = char
            elif char == "[":
                raise ValueError("nested '[' is not supported")
            elif char == "]":
                break
            cursor += 1
        if cursor >= length:
            raise ValueError("unclosed '[' in input")
        blocks.append(text[start:cursor])
        index = cursor + 1
    return blocks


def _looks_like_pairs(block: str) -> bool:
    try:
        segments = _split_segments(block)
    except ValueError:
        return False
    for segment in segments:
        try:
            if _find_separator(segment) >= 0:
                return True
        except ValueError:
            continue
    return False


def _check_duplicates(pairs: Sequence[tuple[str, str]]) -> None:
    seen: set[str] = set()
    for key, _ in pairs:
        if key in seen:
            raise ValueError(f"duplicate key {key!r}")
        seen.add(key)


def extract_pairs(text: str) -> list[tuple[str, str]]:
    """Extract ordered ``(key, value)`` pairs from ``text``.

    The parser accepts ``key=value`` and ``key: value`` forms, semicolon or
    newline separators, quoted values and pairs inside ``[...]`` blocks. It
    raises ``ValueError`` for an empty input, a malformed pair, a missing key
    or value, an unterminated quote or a duplicate key.
    """
    if not isinstance(text, str):
        raise ValueError(f"text must be a string, got {type(text).__name__}")
    if not text.strip():
        raise ValueError("no pairs found: input is empty")

    blocks = _bracket_blocks(text)
    candidates = [block for block in blocks if _looks_like_pairs(block)]
    if candidates:
        pairs: list[tuple[str, str]] = []
        for block in candidates:
            pairs.extend(_parse_body(block))
        _check_duplicates(pairs)
        return pairs
    if blocks:
        raise ValueError(
            "no key=value or key: value pairs found inside brackets")
    return _parse_body(text)


def _normalize_pairs(
    pairs: Sequence[Sequence[Any]],
) -> list[tuple[str, str]]:
    normalized: list[tuple[str, str]] = []
    seen: set[str] = set()
    for index, pair in enumerate(pairs, start=1):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(
                f"pair {index}: expected a (key, value) pair")
        raw_key, raw_value = pair
        key = str(raw_key).strip()
        value = raw_value if isinstance(raw_value, str) else str(raw_value)
        if not key:
            raise ValueError(f"pair {index}: missing key")
        if not _KEY_RE.fullmatch(key):
            raise ValueError(f"pair {index}: invalid key {key!r}")
        if not value:
            raise ValueError(f"pair {index}: missing value for key {key!r}")
        if key in seen:
            raise ValueError(f"duplicate key {key!r} (pair {index})")
        seen.add(key)
        normalized.append((key, value))
    if not normalized:
        raise ValueError("no pairs to render")
    return normalized


def _kv_value(value: str) -> str:
    needs_quotes = (
        value != value.strip()
        or any(char in value for char in ';"\\\n\r')
        or value[0] in "\"'"
    )
    if needs_quotes:
        return json.dumps(value, ensure_ascii=False)
    return value


def _yaml_key(key: str) -> str:
    if (_YAML_PLAIN_RE.fullmatch(key)
            and key.lower() not in _YAML_RESERVED
            and not key[0].isdigit()):
        return key
    return json.dumps(key, ensure_ascii=False)


def _yaml_value(value: str) -> str:
    if (_YAML_PLAIN_RE.fullmatch(value)
            and value.lower() not in _YAML_RESERVED):
        return value
    return json.dumps(value, ensure_ascii=False)


def render_pairs(
    pairs: Sequence[Sequence[Any]], fmt: str = "json",
) -> str:
    """Render ordered pairs as compact JSON, one-line KV or YAML lines."""
    canonical = normalize_format(fmt)
    normalized = _normalize_pairs(pairs)
    if canonical == "json":
        return json.dumps(
            dict(normalized), ensure_ascii=False, separators=(",", ":"))
    if canonical == "kv":
        return ";".join(
            f"{key}={_kv_value(value)}" for key, value in normalized)
    return "\n".join(
        f"{_yaml_key(key)}: {_yaml_value(value)}"
        for key, value in normalized)


def copy_pairs(text: str, fmt: str = "json") -> str:
    """Extract pairs from ``text`` and render them in ``fmt``."""
    return render_pairs(extract_pairs(text), fmt=fmt)


def _json_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def parse_rendered(text: str, fmt: str = "json") -> dict[str, Any]:
    """Parse rendered JSON/KV/YAML back into a mapping.

    This is the inverse used by the benchmark. Duplicate JSON keys are
    rejected. KV and YAML input use the same pair parser as ``extract_pairs``.
    """
    canonical = normalize_format(fmt)
    if not isinstance(text, str):
        raise ValueError(f"text must be a string, got {type(text).__name__}")
    if canonical == "json":
        try:
            parsed = json.loads(text, object_pairs_hook=_json_no_duplicates)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("JSON output must be an object")
        return parsed
    return dict(extract_pairs(text))


class CopyTool:
    """Small stateless wrapper around the copy functions."""

    def __init__(self, fmt: str = "json") -> None:
        self.format = normalize_format(fmt)

    def extract(self, text: str) -> list[tuple[str, str]]:
        return extract_pairs(text)

    def render(self, pairs: Sequence[Sequence[Any]]) -> str:
        return render_pairs(pairs, fmt=self.format)

    def run(self, text: str) -> str:
        return copy_pairs(text, fmt=self.format)

    def __call__(self, text: str) -> str:
        return self.run(text)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="atomzero-copy",
        description=(
            "Copy explicit key/value pairs from text without a model. "
            "This is a deterministic runtime tool."),
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--text", help="text containing the pairs")
    source.add_argument(
        "--text-file", help="UTF-8 file containing the pairs")
    parser.add_argument(
        "--format", choices=FORMATS, default="json",
        help="output format (default: json)")
    args = parser.parse_args(argv)

    if args.text is not None:
        text = args.text
    elif args.text_file is not None:
        try:
            from pathlib import Path
            text = Path(args.text_file).read_text(encoding="utf-8-sig")
        except OSError as exc:
            print(f"error: could not read input file: {exc}", file=sys.stderr)
            return 2
    else:
        if sys.stdin is None or sys.stdin.isatty():
            parser.error("provide --text, --text-file, or pipe text on stdin")
        text = sys.stdin.read()

    try:
        output = copy_pairs(text, fmt=args.format)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
