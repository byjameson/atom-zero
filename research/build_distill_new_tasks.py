"""Generate and verify new copy tasks with a teacher model.

Hard-label distillation reuses the prompts and exact pairs of an existing
staged corpus, so the student mostly sees new surface forms of known content.
This builder is a different experiment: the teacher invents **new** copy tasks
with new keys, values and formats. The generated task is then checked by
ordinary Python, not trusted blindly.

The pipeline is:

1. ask the teacher for a batch of compact JSON objects, one per line, each
   with exactly the requested number of key/value pairs;
2. parse each answer tolerantly (flat JSON, pretty-printed JSON, code fences,
   bullet/number prefixes, KV or YAML);
3. require the parsed object to satisfy deterministic constraints:
   exact pair count, unique 3-6 character lowercase alphanumeric keys that
   start with a letter, a fixed key blocklist, and non-empty values of at most
   40 characters;
4. reject duplicate rows and, when an avoid source is supplied, rows whose
   pairs or text already occur in a train corpus;
5. canonicalize the accepted pairs to the requested JSON, KV or YAML form
   with a prompt that lists the exact pairs.

The builder never reads an eval file. ``--avoid-source`` accepts train JSONL
files only; an eval-looking path is refused before it is opened.

``build_new_tasks`` accepts an injected teacher callable, so tests never load
torch or Qwen:

    rows, counts = build_new_tasks(lambda prompt: answer, rows=10)

Command (host GPU, 1000 requested tasks)::

    python research/build_distill_new_tasks.py `
      --out-dir data/distill_new_v1 --rows 1000 --batch-size 5 `
      --keys-per-row 2,3,4 --formats json,kv --max-new-tokens 512 `
      --seed 23 --local-files-only --shuffle

The output is training data, not a model result or a product claim.
``manifest.json`` contains aggregate counts only: no prompts, answers, keys,
values or other row text. Generated data lives under the gitignored ``data/``
directory by default. See ``docs/DISTILL_NEW_TASKS.md``.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FORMATS = ("json", "kv", "yaml")
DEFAULT_FORMATS = ("json", "kv")
DEFAULT_KEYS_PER_ROW = (2, 3, 4)
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_OUT_DIR = "data/distill_new_v1"
DEFAULT_BATCH_SIZE = 20
DEFAULT_MAX_NEW_TOKENS = 256
DATASET_NAME = "distill_new_v1"
VERIFICATION_METHOD = "deterministic_new_task_constraints_v1"
TRAIN_FILE = "train_new_tasks.jsonl"
MANIFEST_FILE = "manifest.json"
MANIFEST_NOTE = (
    "Aggregate counts only: this manifest contains no prompts, answers, keys, "
    "values or row text. Accepted rows passed deterministic constraint checks "
    "and are teacher-generated training data, not a model capability claim."
)

KEY_RE = re.compile(r"^[a-z][a-z0-9]{2,5}$")
VALUE_MAX_CHARS = 40
VALUE_FORBIDDEN = frozenset(";=[]{},\n\r\t")
YAML_UNSAFE = frozenset(":#{}[],&*!|>'\"%@`\n\r\t")
FENCE_MARKER_RE = re.compile(r"```[A-Za-z0-9_.+-]*")
BULLET_RE = re.compile(r"^(?:[-*\u2022]|\d+[.)])\s+")
LABEL_RE = re.compile(
    r"^(?:task|record|item|example|pair)\s+\d+\s*[:.)-]\s*", re.IGNORECASE)
LOWER = "abcdefghijklmnopqrstuvwxyz"
LOWER_DIGITS = LOWER + "0123456789"

# Fixed blocklist. A teacher key is rejected if it is a common structured
# output word or a generic English stop word, even when it satisfies the
# 3-6 lowercase alphanumeric shape. This keeps generated keys from being
# memorized schema names such as "key", "value" or "answer".
KEY_BLOCKLIST = frozenset({
    "all", "and", "answer", "any", "api", "app", "array", "assistant",
    "avg", "body", "bool", "boolean", "bottom", "city", "close", "code",
    "copy", "count", "cpu", "css", "data", "del", "dict", "end", "entity",
    "example", "false", "family", "file", "first", "float", "for", "format",
    "from", "get", "git", "gpu", "header", "host", "html", "http", "id",
    "input", "int", "into", "item", "json", "key", "key0", "key1", "key2",
    "key3", "key4", "key5", "key6", "key7", "key8", "key9", "keys", "label",
    "last", "left", "len", "line", "link", "list", "load", "long", "main",
    "max", "min", "model", "name", "new", "next", "none", "not", "null",
    "num", "number", "object", "old", "one", "open", "os", "output", "page",
    "pair", "pairs", "params", "path", "port", "prev", "prompt", "ram",
    "random", "read", "reply", "response", "result", "right", "run",
    "sample", "save", "score", "set", "short", "size", "sql", "stage",
    "start", "step", "str", "string", "sum", "task", "ten", "test", "text",
    "that", "the", "this", "three", "title", "top", "true", "tuple", "two",
    "type", "url", "use", "user", "val0", "val1", "val2", "val3", "val4",
    "val5", "val6", "val7", "val8", "val9", "value", "value0", "value1",
    "value2", "value3", "value4", "value5", "value6", "value7", "value8",
    "value9", "values", "web", "with", "word", "write", "yaml", "yes",
})

# A short preview of the blocklist is included in the teacher prompt. The
# full set is enforced by validation; the preview only improves acceptance.
PROMPT_BLOCKED_KEYS = (
    "task", "key", "value", "json", "yaml", "pair", "answer", "prompt",
    "data", "text", "format", "name", "type", "item",
)

EXAMPLE_WORDS = (
    "amber", "harbor", "delta", "north", "silver", "island", "seven",
    "42", "river", "quiet", "13", "forest", "copper", "valley", "twenty",
    "8", "cobalt", "meadow", "stone", "31", "orange", "winter", "64",
)

TEACHER_SYSTEM_PROMPT = (
    "You generate synthetic copy-practice data. You reply with compact flat "
    "JSON objects only. Never explain and never add a code fence."
)


class ParseError(ValueError):
    """Raised when a teacher answer cannot be parsed into a task object."""


@dataclass(frozen=True)
class TaskSpec:
    """One requested output slot: how many pairs and which answer format."""

    key_count: int
    format: str


@dataclass(frozen=True)
class AvoidIndex:
    """Train-corpus text and pair signatures that generated rows must avoid."""

    texts: frozenset[str]
    signatures: frozenset[tuple[tuple[str, str], ...]]

    def overlaps(self, prompt: str, answer: str,
                 pairs: Sequence[Sequence[Any]]) -> bool:
        if prompt in self.texts or answer in self.texts:
            return True
        return pairs_signature(pairs) in self.signatures


def normalize_keys_per_row(
        values: str | Iterable[int | str]) -> tuple[int, ...]:
    """Return a validated, sorted, duplicate-free tuple of key counts."""

    if isinstance(values, str):
        items = [item.strip() for item in values.split(",")]
    else:
        items = [str(item).strip() for item in values]
    items = [item for item in items if item]
    if not items:
        raise ValueError("at least one key count is required")
    result: list[int] = []
    for item in items:
        try:
            count = int(item)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"key count must be an integer, got {item!r}") from exc
        if count not in (2, 3, 4):
            raise ValueError(
                f"key count must be 2, 3 or 4, got {count}")
        if count not in result:
            result.append(count)
    return tuple(sorted(result))


def normalize_formats(formats: str | Iterable[str]) -> tuple[str, ...]:
    """Return a validated format tuple in canonical order."""

    if isinstance(formats, str):
        items = [item.strip().lower() for item in formats.split(",")]
    else:
        items = [str(item).strip().lower() for item in formats]
    items = [item for item in items if item]
    if not items:
        raise ValueError("at least one format is required")
    result: list[str] = []
    for item in items:
        if item not in FORMATS:
            raise ValueError(
                f"unknown format {item!r}; expected one of {FORMATS}")
        if item not in result:
            result.append(item)
    return tuple(fmt for fmt in FORMATS if fmt in result)


def _assert_train_source(path: Path) -> None:
    """Refuse an eval source so held-out rows cannot leak into generation."""

    parts = [part.lower() for part in path.parts]
    stem = path.stem.lower()
    if stem.startswith("eval") or "eval" in stem or "eval" in parts:
        raise ValueError(
            f"refusing eval source {path}: new-task avoid sources must be "
            "train.jsonl files")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"{path}:{line_number}: row must be an object")
            rows.append(row)
    return rows


def load_avoid_rows(sources: str | Iterable[str]) -> list[dict[str, Any]]:
    """Load train-only rows used to reject source text or pair reuse."""

    if isinstance(sources, (str, Path)):
        items: list[str | Path] = [sources]
    else:
        items = list(sources)
    if not items:
        return []
    rows: list[dict[str, Any]] = []
    for item in items:
        path = Path(item).expanduser()
        _assert_train_source(path)
        if not path.exists():
            raise FileNotFoundError(f"avoid source not found: {path}")
        for row in _read_jsonl(path):
            partition = row.get("partition")
            if (partition is not None
                    and str(partition).strip().lower() != "train"):
                raise ValueError(
                    f"refusing non-train row in {path}: "
                    f"partition={partition!r}")
            rows.append(row)
    return rows


def _value_text(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    return str(value)


def _row_pairs(row: Mapping[str, Any]) -> tuple[tuple[str, str], ...] | None:
    raw_pairs = row.get("pairs")
    pairs: list[tuple[str, str]] = []
    if isinstance(raw_pairs, (list, tuple)) and raw_pairs:
        for item in raw_pairs:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                pairs.append((str(item[0]), _value_text(item[1])))
            elif isinstance(item, Mapping) and len(item) == 2:
                key, value = list(item.items())[0]
                pairs.append((str(key), _value_text(value)))
            else:
                return None
        return tuple(pairs)
    keys = row.get("keys")
    values = row.get("values")
    if (isinstance(keys, (list, tuple))
            and isinstance(values, (list, tuple))
            and len(keys) == len(values) and keys):
        return tuple(
            (str(key), _value_text(value))
            for key, value in zip(keys, values))
    return None


def build_avoid_index(rows: Sequence[Mapping[str, Any]]) -> AvoidIndex:
    """Build the text and pair-signature index for a train corpus."""

    texts: set[str] = set()
    signatures: set[tuple[tuple[str, str], ...]] = set()
    for row in rows:
        for field in ("user", "assistant"):
            text = str(row.get(field, "")).strip()
            if text:
                texts.add(text)
        pairs = _row_pairs(row)
        if pairs:
            signatures.add(pairs_signature(pairs))
    return AvoidIndex(frozenset(texts), frozenset(signatures))


def pairs_signature(
        pairs: Sequence[Sequence[Any]]) -> tuple[tuple[str, str], ...]:
    """Return an order-insensitive signature for a row's key/value pairs."""

    return tuple(sorted(
        (str(key), _value_text(value)) for key, value in pairs))


def validate_key(key: Any) -> bool:
    """Return True iff ``key`` is a permitted generated key."""

    if not isinstance(key, str):
        return False
    if not KEY_RE.fullmatch(key):
        return False
    return key not in KEY_BLOCKLIST


def _normalize_value(value: Any) -> str:
    if isinstance(value, bool):
        raise ValueError("boolean values are not accepted")
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("numeric value must be finite")
        text = str(value)
    elif isinstance(value, str):
        text = value.strip()
    else:
        raise ValueError("value must be a string or a number")
    if not text:
        raise ValueError("value must be non-empty")
    if len(text) > VALUE_MAX_CHARS:
        raise ValueError(
            f"value is longer than {VALUE_MAX_CHARS} characters")
    if any(char in VALUE_FORBIDDEN for char in text):
        raise ValueError("value contains a reserved separator")
    if not re.search(r"[A-Za-z0-9]", text):
        raise ValueError("value must contain a letter or digit")
    return text


def validate_task(data: Any,
                  key_count: int) -> tuple[tuple[str, str], ...]:
    """Validate one parsed teacher object and return normalized pairs.

    The rules are deliberately deterministic and literal. They do not check
    whether the task is useful or semantically sensible, only whether it is a
    well-formed, new-looking copy task.
    """

    if not isinstance(data, Mapping):
        raise ValueError("task must parse to a JSON object")
    if len(data) != key_count:
        raise ValueError(
            f"task must have exactly {key_count} pairs, got {len(data)}")
    pairs: list[tuple[str, str]] = []
    seen: set[str] = set()
    for key, value in data.items():
        if not validate_key(key):
            raise ValueError(f"invalid key: {key!r}")
        if key in seen:
            raise ValueError(f"duplicate key: {key}")
        seen.add(key)
        pairs.append((key, _normalize_value(value)))
    if len(pairs) != key_count:
        raise ValueError(f"task must have exactly {key_count} pairs")
    return tuple(pairs)


def canonicalize(fmt: str,
                 pairs: Sequence[Sequence[Any]]) -> str:
    """Render verified pairs in the canonical answer form for ``fmt``."""

    if fmt not in FORMATS:
        raise ValueError(f"unknown format: {fmt}")
    rendered = [(str(key), _value_text(value)) for key, value in pairs]
    if len({key for key, _ in rendered}) != len(rendered):
        raise ValueError("pairs contain duplicate keys")
    if fmt == "json":
        return json.dumps(
            dict(rendered), separators=(",", ":"), ensure_ascii=False)
    if fmt == "kv":
        return ";".join(f"{key}={value}" for key, value in rendered)
    return "\n".join(
        f"{key}: {_yaml_scalar(value)}" for key, value in rendered)


def _yaml_scalar(value: Any) -> str:
    text = _value_text(value)
    if any(char in YAML_UNSAFE for char in text):
        return json.dumps(text, ensure_ascii=False)
    return text


def build_prompt(fmt: str,
                 pairs: Sequence[Sequence[Any]]) -> str:
    """Render an instruction that explicitly lists every pair to copy."""

    if fmt not in FORMATS:
        raise ValueError(f"unknown format: {fmt}")
    rendered = "; ".join(
        f"{key}={_value_text(value)}" for key, value in pairs)
    if fmt == "json":
        return (
            "Return only a compact JSON object with exactly these keys and "
            f"values: [{rendered}]. Copy every key and value exactly; do not "
            "rename or reorder them. Do not add prose or code fences.")
    if fmt == "kv":
        return (
            "Return only one line of semicolon-separated key=value pairs with "
            "no prose and no code fence. Use exactly these pairs in this "
            f"order: [{rendered}]. Copy every key and value exactly; do not "
            "rename them.")
    return (
        "Return only YAML mapping lines with no prose and no code fence. Use "
        "exactly these pairs in this order, one key: value per line: "
        f"[{rendered}]. Copy every key and value exactly; do not rename them.")


def _strip_fence_markers(text: str) -> str:
    return FENCE_MARKER_RE.sub("", str(text or "")).replace("```", "")


def _strip_wrapping(line: str) -> str:
    line = line.strip()
    line = BULLET_RE.sub("", line)
    line = LABEL_RE.sub("", line)
    return line.strip().strip("`*").strip()


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        if value[0] == '"':
            try:
                parsed = json.loads(value)
                if isinstance(parsed, str):
                    return parsed
            except json.JSONDecodeError:
                pass
        return value[1:-1]
    return value


def _balanced_span(text: str, start: int) -> str | None:
    stack: list[str] = []
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        current = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif current == "\\":
                escaped = True
            elif current == '"':
                in_string = False
            continue
        if current == '"':
            in_string = True
        elif current in "[{":
            stack.append(current)
        elif current in "]}":
            if not stack:
                return None
            opening = stack.pop()
            if ((opening == "{" and current != "}")
                    or (opening == "[" and current != "]")):
                return None
            if not stack:
                return text[start:index + 1]
    return None


def _balanced_spans(text: str, limit: int = 512) -> Iterable[str]:
    """Yield complete top-level JSON-ish spans from ``text``.

    A truncated top-level object is skipped, so a valid object that follows a
    malformed one is still parsed. Nested fragments inside an incomplete object
    are not yielded as if they were complete tasks.
    """

    index = 0
    yielded = 0
    length = len(text)
    while index < length:
        current = text[index]
        if current == '"':
            index += 1
            while index < length:
                if text[index] == "\\":
                    index += 2
                elif text[index] == '"':
                    index += 1
                    break
                else:
                    index += 1
            continue
        if current in "[{":
            span = _balanced_span(text, index)
            if span is not None:
                yield span
                yielded += 1
                if yielded >= limit:
                    return
                index += len(span)
                continue
        index += 1


def _json_object_no_duplicates(
        pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        key = str(key)
        if key in result:
            raise ParseError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse_kv_line(text: str) -> dict[str, Any]:
    chunks = re.split(r"[;,|]+", text)
    pairs: dict[str, Any] = {}
    for chunk in chunks:
        chunk = _strip_wrapping(chunk)
        if not chunk or chunk.startswith("#"):
            continue
        if "=" in chunk:
            key, value = chunk.split("=", 1)
        elif ":" in chunk:
            key, value = chunk.split(":", 1)
        else:
            continue
        key = _unquote(key.strip())
        value = _unquote(value.strip())
        if not key or not value:
            continue
        if key in pairs:
            raise ParseError(f"duplicate key: {key}")
        pairs[key] = value
    if not pairs:
        raise ParseError("answer has no KV or YAML pairs")
    return pairs


def _yaml_blocks(text: str) -> list[list[str]]:
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped in {"---", "..."}:
            if current:
                blocks.append(current)
                current = []
            continue
        current.append(stripped)
    if current:
        blocks.append(current)
    return blocks


def _parse_yaml_block(lines: Sequence[str]) -> dict[str, Any]:
    pairs: dict[str, Any] = {}
    for line in lines:
        chunk = _strip_wrapping(line)
        if not chunk or chunk.startswith("#"):
            continue
        try:
            line_pairs = _parse_kv_line(chunk)
        except ParseError:
            continue
        for key, value in line_pairs.items():
            if key in pairs:
                raise ParseError(f"duplicate key: {key}")
            pairs[key] = value
    if not pairs:
        raise ParseError("answer has no KV or YAML pairs")
    return pairs


def parse_task_line(line: Any) -> dict[str, Any]:
    """Parse one teacher line into a flat object.

    Accepts a flat JSON object (with surrounding prose, backticks, bullets or
    numbering), a single-object JSON array, or a ``key=value`` /
    ``key: value`` line. Raises ``ParseError`` when no object is found.
    """

    text = _strip_wrapping(str(line or ""))
    if not text:
        raise ParseError("empty teacher line")
    for span in _balanced_spans(text):
        try:
            data = json.loads(
                span, object_pairs_hook=_json_object_no_duplicates)
        except (json.JSONDecodeError, ParseError, ValueError):
            continue
        if isinstance(data, dict):
            return data
        if (isinstance(data, list) and len(data) == 1
                and isinstance(data[0], dict)):
            return data[0]
    return _parse_kv_line(text)


def _json_like(text: str) -> bool:
    """Return True for quoted-key JSON, including truncated pretty JSON."""

    return re.search(r'"[^"]+"\s*:', text) is not None


def parse_teacher_tasks(answer: Any,
                        key_count: int | None = None
                        ) -> list[dict[str, Any]]:
    """Parse a teacher answer into every complete task object it contains.

    The parser prefers JSON objects and can therefore read pretty-printed
    objects even though the prompt asks for one object per line. If the answer
    contains no JSON object, it groups KV/YAML lines into blocks. When
    ``key_count`` is supplied, a block is accepted as one task only if it has
    exactly that many pairs; otherwise the parser falls back to one task per
    line. Prose and truncated objects are skipped.
    """

    text = _strip_fence_markers(answer)
    if not text.strip():
        return []

    tasks: list[dict[str, Any]] = []
    for span in _balanced_spans(text):
        try:
            data = json.loads(
                span, object_pairs_hook=_json_object_no_duplicates)
        except (json.JSONDecodeError, ParseError, ValueError):
            continue
        if isinstance(data, dict):
            tasks.append(data)
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    tasks.append(item)
    if tasks:
        return tasks

    json_like = _json_like(text)
    block_tasks: list[dict[str, Any]] = []
    if not json_like:
        blocks = _yaml_blocks(text)
        try:
            for block in blocks:
                block_tasks.append(_parse_yaml_block(block))
        except ParseError:
            block_tasks = []
        if block_tasks and (
                key_count is None
                or all(len(task) == key_count for task in block_tasks)):
            return block_tasks

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if json_like and (
                line.startswith(("{", "["))
                or re.match(r'^"[^"]+"\s*:', line)):
            continue
        try:
            tasks.append(parse_task_line(line))
        except ParseError:
            continue
    return tasks


def _example_task(rng: random.Random, key_count: int,
                  used_keys: set[str]) -> dict[str, str]:
    keys: list[str] = []
    while len(keys) < key_count:
        length = rng.randint(3, 6)
        key = rng.choice(LOWER) + "".join(
            rng.choice(LOWER_DIGITS) for _ in range(length - 1))
        if not KEY_RE.fullmatch(key) or key in KEY_BLOCKLIST:
            continue
        if key in used_keys or key in keys:
            continue
        keys.append(key)
    used_keys.update(keys)
    return {key: rng.choice(EXAMPLE_WORDS) for key in keys}


def build_teacher_prompt(key_count: int, count: int, batch_index: int,
                         seed: int = 23) -> str:
    """Render one deterministic teacher prompt for a batch of new tasks."""

    if key_count not in (2, 3, 4):
        raise ValueError(f"key_count must be 2, 3 or 4, got {key_count}")
    if count < 1:
        raise ValueError("count must be positive")
    rng = random.Random(
        (int(seed) * 1_000_003) ^ (int(batch_index) * 7_919)
        ^ (int(key_count) << 16))
    used_keys: set[str] = set()
    examples = [
        _example_task(rng, key_count, used_keys)
        for _ in range(2)
    ]
    shape = "\n".join(
        json.dumps(example, separators=(",", ":"), ensure_ascii=False)
        for example in examples)
    blocked = ", ".join(PROMPT_BLOCKED_KEYS)
    return (
        f"Generate {count} new copy-practice records for batch "
        f"{batch_index}. Every record has exactly {key_count} key/value "
        "pairs.\n"
        f"Count carefully: EXACTLY {key_count} pairs means {key_count} keys "
        f"and {key_count} values, no more and no fewer.\n"
        "Keys are unique inside a record, start with a lowercase letter, and "
        "are 3-6 lowercase letters or digits in total; they are not reserved "
        f"words such as: {blocked}.\n"
        "Values are one short word or a number (1-40 characters) with no "
        "semicolon, equals sign, comma, bracket or brace.\n"
        "Each record must be a compact flat JSON object. Do not repeat a "
        "record. Vary keys and values across records when you can.\n"
        f"The following 2 shape examples are NOT part of the {count} records; "
        "do not copy them:\n"
        f"{shape}\n"
        f"Return the {count} new records now, one compact JSON object per "
        f"line. Every line must contain exactly {key_count} pairs.")


def build_task_specs(
        rows: int,
        *,
        keys_per_row: str | Iterable[int | str] = DEFAULT_KEYS_PER_ROW,
        formats: str | Iterable[str] = DEFAULT_FORMATS,
        yaml_count: int = 0,
        seed: int = 23,
        shuffle: bool = False) -> list[TaskSpec]:
    """Build the deterministic key-count and format schedule.

    ``yaml_count`` reserves that many output slots for YAML; the remaining
    slots rotate through the non-YAML formats. When it is zero and ``yaml`` is
    in ``formats``, YAML participates in the normal rotation.
    """

    if rows < 0:
        raise ValueError("rows must be non-negative")
    if yaml_count < 0:
        raise ValueError("yaml_count must be non-negative")
    if yaml_count > rows:
        raise ValueError("yaml_count cannot exceed rows")
    keys = normalize_keys_per_row(keys_per_row)
    base_formats = normalize_formats(formats)
    if yaml_count:
        rotation = tuple(fmt for fmt in base_formats if fmt != "yaml")
        if rows > yaml_count and not rotation:
            raise ValueError(
                "yaml_count reserves every slot but no non-YAML format "
                "remains")
    else:
        rotation = base_formats

    specs: list[TaskSpec] = []
    for index in range(rows):
        key_count = keys[index % len(keys)]
        if index < yaml_count:
            fmt = "yaml"
        else:
            fmt = rotation[(index - yaml_count) % len(rotation)]
        specs.append(TaskSpec(key_count=key_count, format=fmt))
    if shuffle:
        random.Random(seed).shuffle(specs)
    return specs


def _make_row(pairs: Sequence[Sequence[Any]], spec: TaskSpec,
              model: str) -> dict[str, Any]:
    keys = [str(key) for key, _ in pairs]
    values = [_value_text(value) for _, value in pairs]
    return {
        "user": build_prompt(spec.format, pairs),
        "assistant": canonicalize(spec.format, pairs),
        "family": f"newtask_{spec.key_count}key_{spec.format}",
        "format": spec.format,
        "partition": "train",
        "keys": keys,
        "values": values,
        "pairs": [[key, value] for key, value in zip(keys, values)],
        "key_count": spec.key_count,
        "teacher_model": str(model),
        "teacher_verified": True,
        "canonicalized": True,
        "verification": {
            "verified": True,
            "method": VERIFICATION_METHOD,
            "checks": [
                "parses_to_dict",
                "exact_pair_count",
                "keys_unique",
                "keys_lower_alnum",
                "keys_not_blocklisted",
                "values_nonempty",
                "values_short",
                "row_not_duplicate",
                "row_not_source_reuse",
            ],
        },
    }


def build_new_tasks(
        teacher: Callable[[str], str],
        *,
        rows: int = 2000,
        batch_size: int = DEFAULT_BATCH_SIZE,
        keys_per_row: str | Iterable[int | str] = DEFAULT_KEYS_PER_ROW,
        formats: str | Iterable[str] = DEFAULT_FORMATS,
        yaml_count: int = 0,
        seed: int = 23,
        shuffle: bool = False,
        model: str = DEFAULT_MODEL,
        avoid_rows: Sequence[Mapping[str, Any]] = (),
        progress: Callable[[Mapping[str, Any]], None] | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate verified new tasks with an injected teacher callable.

    ``teacher`` receives one rendered prompt and returns raw text. No model is
    loaded here. Returns ``(rows, counts)`` where ``counts`` is content-free.
    """

    if rows < 0:
        raise ValueError("rows must be non-negative")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    keys = normalize_keys_per_row(keys_per_row)
    specs = build_task_specs(
        rows, keys_per_row=keys, formats=formats, yaml_count=yaml_count,
        seed=seed, shuffle=shuffle)
    avoid = build_avoid_index(avoid_rows)

    counts: dict[str, Any] = {
        "requested": len(specs),
        "accepted": 0,
        "rejected": 0,
        "by_format": {fmt: 0 for fmt in FORMATS},
        "by_keys": {str(key): 0 for key in keys},
        "duplicate_count": 0,
        "source_overlap": 0,
        "invalid_tasks": 0,
        "overflow_tasks": 0,
        "teacher_calls": 0,
        "parsed_candidates": 0,
    }
    groups: dict[int, list[int]] = {key: [] for key in keys}
    for index, spec in enumerate(specs):
        groups[spec.key_count].append(index)

    filled: dict[int, dict[str, Any]] = {}
    seen: set[tuple[tuple[str, str], ...]] = set()
    batch_index = 0
    for key_count in keys:
        pending = groups[key_count]
        for start in range(0, len(pending), batch_size):
            batch = pending[start:start + batch_size]
            prompt = build_teacher_prompt(
                key_count, len(batch), batch_index, seed=seed)
            batch_index += 1
            counts["teacher_calls"] += 1
            raw = teacher(prompt)
            parsed = parse_teacher_tasks(raw, key_count=key_count)
            counts["parsed_candidates"] += len(parsed)
            accepted_in_batch = 0
            for data in parsed:
                try:
                    pairs = validate_task(data, key_count)
                except ValueError:
                    counts["invalid_tasks"] += 1
                    continue
                signature = pairs_signature(pairs)
                if signature in seen:
                    counts["duplicate_count"] += 1
                    continue
                if accepted_in_batch >= len(batch):
                    counts["overflow_tasks"] += 1
                    continue
                spec_index = batch[accepted_in_batch]
                spec = specs[spec_index]
                row = _make_row(pairs, spec, model)
                if avoid.overlaps(
                        row["user"], row["assistant"], pairs):
                    counts["source_overlap"] += 1
                    continue
                seen.add(signature)
                filled[spec_index] = row
                accepted_in_batch += 1
                counts["accepted"] += 1
                counts["by_format"][spec.format] += 1
                counts["by_keys"][str(key_count)] += 1
            if progress is not None:
                progress(dict(counts))

    rows_out = [
        filled[index] for index in range(len(specs)) if index in filled
    ]
    for output_index, row in enumerate(rows_out):
        row["task_id"] = f"newtask-{int(seed)}-{output_index:06d}"
    counts["rejected"] = counts["requested"] - counts["accepted"]
    return rows_out, counts


def build_manifest(
        counts: Mapping[str, Any],
        *,
        model: str = DEFAULT_MODEL,
        seed: int = 23,
        keys_per_row: str | Iterable[int | str] = DEFAULT_KEYS_PER_ROW,
        formats: str | Iterable[str] = DEFAULT_FORMATS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        yaml_count: int = 0,
        local_files_only: bool = True,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        avoid_sources: Sequence[str | Path] = ()) -> dict[str, Any]:
    """Build the content-free aggregate manifest."""

    keys = normalize_keys_per_row(keys_per_row)
    normalized_formats = normalize_formats(formats)
    if isinstance(avoid_sources, (str, Path)):
        avoid_sources = [avoid_sources]
    requested = int(counts.get("requested", 0))
    accepted = int(counts.get("accepted", 0))
    rejected = int(counts.get("rejected", 0))
    if accepted + rejected != requested:
        raise ValueError(
            "manifest counts must satisfy accepted + rejected == requested")
    manifest = {
        "dataset": DATASET_NAME,
        "teacher_model": str(model),
        "verification_method": VERIFICATION_METHOD,
        "seed": int(seed),
        "requested": requested,
        "accepted": accepted,
        "rejected": rejected,
        "duplicate_count": int(counts.get("duplicate_count", 0)),
        "source_overlap_count": int(counts.get("source_overlap", 0)),
        "invalid_task_count": int(counts.get("invalid_tasks", 0)),
        "overflow_task_count": int(counts.get("overflow_tasks", 0)),
        "teacher_calls": int(counts.get("teacher_calls", 0)),
        "parsed_candidates": int(counts.get("parsed_candidates", 0)),
        "by_format": {
            fmt: int(counts.get("by_format", {}).get(fmt, 0))
            for fmt in FORMATS
        },
        "by_keys": {
            str(key): int(counts.get("by_keys", {}).get(str(key), 0))
            for key in keys
        },
        "batch_size": int(batch_size),
        "keys_per_row": list(keys),
        "formats": list(normalized_formats),
        "yaml_count": int(yaml_count),
        "local_files_only": bool(local_files_only),
        "max_new_tokens": int(max_new_tokens),
        "avoid_source_files": [str(Path(path)) for path in avoid_sources],
        "note": MANIFEST_NOTE,
    }
    return manifest


def write_dataset(rows: Sequence[Mapping[str, Any]],
                  manifest: Mapping[str, Any],
                  out_dir: str | Path) -> dict[str, str]:
    """Write ``train_new_tasks.jsonl`` and ``manifest.json``."""

    root = Path(out_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    train_path = root / TRAIN_FILE
    manifest_path = root / MANIFEST_FILE
    body = "".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    train_path.write_text(body, encoding="utf-8")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    return {"train": str(train_path), "manifest": str(manifest_path)}


def load_teacher(
        model_name: str = DEFAULT_MODEL,
        *,
        device: str = "cuda",
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        local_files_only: bool = True) -> Callable[[str], str]:
    """Load a greedy chat teacher. Kept separate so tests never need torch."""

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA device requested but torch.cuda.is_available() is false; "
            "pass --device cpu for a slow CPU smoke test")
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map=device,
        local_files_only=local_files_only)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, local_files_only=local_files_only)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.eval()

    def teacher(prompt: str) -> str:
        messages = [
            {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
            {"role": "user", "content": str(prompt)},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.eos_token_id)
        generated = output[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(generated, skip_special_tokens=True)

    return teacher


def _progress_printer(
        interval: int = 10) -> Callable[[Mapping[str, Any]], None]:
    seen = 0

    def report(counts: Mapping[str, Any]) -> None:
        nonlocal seen
        seen += 1
        if seen % interval:
            return
        requested = int(counts.get("requested", 0))
        accepted = int(counts.get("accepted", 0))
        print(json.dumps({
            "accepted": accepted,
            "unfilled_so_far": requested - accepted,
            "teacher_calls": int(counts.get("teacher_calls", 0)),
            "duplicate_count": int(counts.get("duplicate_count", 0)),
        }), file=sys.stderr, flush=True)

    return report


def main(argv: Sequence[str] | None = None, *,
         teacher_factory: Callable[..., Callable[[str], str]] | None = None
         ) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--out-dir", default=DEFAULT_OUT_DIR,
        help=f"output directory (default: {DEFAULT_OUT_DIR})")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"teacher model (default: {DEFAULT_MODEL})")
    parser.add_argument(
        "--device", default="cuda",
        help="torch device for the teacher (default: cuda)")
    parser.add_argument(
        "--rows", type=int, default=2000,
        help="number of new tasks to request (default: 2000)")
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help="tasks requested per teacher prompt (default: 20)")
    parser.add_argument(
        "--keys-per-row", default="2,3,4",
        help="key counts per row, comma separated (default: 2,3,4)")
    parser.add_argument(
        "--formats", default=",".join(DEFAULT_FORMATS),
        help="answer formats to rotate, comma separated "
             f"(default: {','.join(DEFAULT_FORMATS)})")
    parser.add_argument(
        "--yaml-count", type=int, default=0,
        help="reserve this many rows for YAML output (default: 0)")
    parser.add_argument(
        "--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS,
        help="teacher generation budget (default: 256)")
    parser.add_argument(
        "--seed", type=int, default=23,
        help="deterministic sampling and prompt seed (default: 23)")
    parser.add_argument(
        "--local-files-only", action=argparse.BooleanOptionalAction,
        default=True,
        help="do not download the teacher (default: true)")
    parser.add_argument(
        "--avoid-source", nargs="+", default=None,
        help="optional train JSONL files whose text/pairs must not be reused")
    parser.add_argument(
        "--shuffle", action="store_true",
        help="shuffle the key-count/format schedule with --seed")
    args = parser.parse_args(argv)

    try:
        keys = normalize_keys_per_row(args.keys_per_row)
        formats = normalize_formats(args.formats)
        if args.rows < 0:
            raise ValueError("--rows must be non-negative")
        if args.batch_size < 1:
            raise ValueError("--batch-size must be positive")
        if args.max_new_tokens < 1:
            raise ValueError("--max-new-tokens must be positive")
        if args.yaml_count < 0:
            raise ValueError("--yaml-count must be non-negative")
        if args.yaml_count > args.rows:
            raise ValueError("--yaml-count cannot exceed --rows")
        if (args.yaml_count and args.rows > args.yaml_count
                and not [fmt for fmt in formats if fmt != "yaml"]):
            raise ValueError(
                "--yaml-count reserves every slot but no non-YAML format "
                "remains")
        avoid_rows = (load_avoid_rows(args.avoid_source)
                      if args.avoid_source else [])
    except (ValueError, OSError) as exc:
        parser.error(str(exc))

    if args.rows == 0:
        teacher: Callable[[str], str] = lambda prompt: ""
    elif teacher_factory is None:
        teacher = load_teacher(
            args.model, device=args.device,
            max_new_tokens=args.max_new_tokens,
            local_files_only=args.local_files_only)
    else:
        teacher = teacher_factory(
            args.model, device=args.device,
            max_new_tokens=args.max_new_tokens,
            local_files_only=args.local_files_only)

    rows, counts = build_new_tasks(
        teacher, rows=args.rows, batch_size=args.batch_size,
        keys_per_row=keys, formats=formats, yaml_count=args.yaml_count,
        seed=args.seed, shuffle=args.shuffle, model=args.model,
        avoid_rows=avoid_rows, progress=_progress_printer())
    manifest = build_manifest(
        counts, model=args.model, seed=args.seed, keys_per_row=keys,
        formats=formats, batch_size=args.batch_size,
        yaml_count=args.yaml_count,
        local_files_only=args.local_files_only,
        max_new_tokens=args.max_new_tokens,
        avoid_sources=args.avoid_source or ())
    paths = write_dataset(rows, manifest, args.out_dir)
    print(json.dumps(
        {**manifest, "paths": paths}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
