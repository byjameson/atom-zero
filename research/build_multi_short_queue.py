"""Build a short-value multi-key copy curriculum.

The ``multi`` stage of the staged copy curriculum mixes several difficulties at
once: two to four arbitrary keys, single-token and multiword values, JSON and
KV output. A small model can fail for any one of those reasons, so an 8-12%
exact-match result does not identify the bottleneck.

This generator isolates multi-key copying with a short-value ladder:

======  =========  ======  ==========================================
#       stage      pairs   values
======  =========  ======  ==========================================
1       ``2short`` 2       single tokens (short names or synthetic words)
2       ``2mixed`` 2       one single token + one multiword value
3       ``3short`` 3       single tokens
4       ``4short`` 4       single tokens
======  =========  ======  ==========================================

Every stage asks for JSON or KV. Keys are 3-5 characters in one of three
styles, fresh per row, and unique across the whole build. Values are unique
across the whole build. Train and eval splits therefore share no key names and
no values, per stage and across stages.

Each canonical answer is rendered deterministically and then parsed back and
compared key by key before the row is kept. A failing row is regenerated. The
generator uses only the standard library; it never imports ``torch`` or a
model, so the queue can be built and tested on CPU in a fraction of a second.

Output layout (default ``data/multi_short_v1``, gitignored)::

    data/multi_short_v1/
      manifest.json
      stage_2short/
        train.jsonl
        eval.jsonl
        manifest.json
      stage_2mixed/...
      stage_3short/...
      stage_4short/...

Rows follow the teacher-queue schema used elsewhere in ``research/``:
``user``, ``assistant``, ``family``, ``format`` plus copy-curriculum metadata
(``stage``, ``keys``, ``values``, ``pairs``) and verification hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

DIFFICULTY_ORDER = ("2short", "2mixed", "3short", "4short")
STAGE_ORDER = DIFFICULTY_ORDER
DIFFICULTIES = DIFFICULTY_ORDER

STAGE_SPECS: dict[str, dict[str, Any]] = {
    "2short": {
        "index": 1,
        "pairs": 2,
        "value_plan": "single",
        "formats": ("json", "kv"),
        "description": "two keys with single-token values",
    },
    "2mixed": {
        "index": 2,
        "pairs": 2,
        "value_plan": "mixed",
        "formats": ("json", "kv"),
        "description": "two keys, one single token and one multiword value",
    },
    "3short": {
        "index": 3,
        "pairs": 3,
        "value_plan": "single",
        "formats": ("json", "kv"),
        "description": "three keys with single-token values",
    },
    "4short": {
        "index": 4,
        "pairs": 4,
        "value_plan": "single",
        "formats": ("json", "kv"),
        "description": "four keys with single-token values",
    },
}

FORMATS = ("json", "kv")
KEY_STYLES = ("short", "cv", "word")
SINGLE_VALUE_POOLS = ("token", "word")
VALUE_POOLS = SINGLE_VALUE_POOLS + ("multiword",)
VERIFICATION_METHOD = "deterministic_multi_short_roundtrip_v1"
DATASET_NAME = "multi_short_v1"

# Key names from the fixed-key curricula and common structured-output
# conventions. Random keys must never collide with these, so the model cannot
# fall back on a memorized key name from an earlier corpus.
RESERVED_KEYS = frozenset({
    "name", "score", "product", "price", "city", "population", "label",
    "points", "code", "rank", "team", "wins", "author", "year", "song",
    "length", "model", "params", "river", "km", "planet", "mass", "recipe",
    "minutes", "value", "key", "id", "data", "text", "answer", "output",
    "result", "input", "prompt", "response", "format", "type", "family",
    "stage", "keys", "values", "pairs", "json", "kv", "yaml", "reply",
    "assistant", "user", "question", "entity", "number", "sum", "count",
    "multi", "short", "mixed",
})

LOWER = "abcdefghijklmnopqrstuvwxyz"
ALNUM = LOWER + "0123456789"
CONSONANTS = "bcdfghjklmnprstvwz"
VOWELS = "aeiou"

# Every style stays inside the 3-5 character key budget. ``cv`` uses five
# characters (three consonant-vowel syllables) because a shorter CV space
# cannot supply tens of thousands of unique keys.
KEY_PATTERNS = {
    "short": re.compile(r"[a-z][a-z0-9]{2,4}"),
    "cv": re.compile(r"(?:[bcdfghjklmnprstvwz][aeiou]){2}"
                     r"[bcdfghjklmnprstvwz]"),
    "word": re.compile(r"[a-z]{3,5}"),
}

WORD_ONSETS = (
    "b", "br", "c", "cl", "d", "dr", "f", "fl", "g", "gr", "h", "j", "k",
    "l", "m", "n", "p", "pl", "pr", "qu", "r", "s", "sh", "sk", "sl", "sm",
    "sn", "sp", "st", "t", "th", "tr", "v", "w", "z",
)
WORD_CODAS = (
    "", "b", "ck", "d", "f", "g", "l", "ld", "m", "n", "nd", "ng", "nt",
    "p", "r", "rd", "rk", "rn", "s", "sh", "sk", "st", "t", "th", "x",
)

# Synthetic single-word values. All generated values are registered globally
# so train and eval never share a value.
ADJECTIVES = (
    "amber", "quiet", "silver", "crimson", "hidden", "ancient", "gentle",
    "rapid", "frozen", "golden", "ivory", "jade", "lunar", "misty", "narrow",
    "opal", "pale", "rustic", "scarlet", "tidal", "velvet", "wandering",
    "young", "brisk", "clouded", "distant", "emerald", "faint", "glowing",
    "hollow", "keen", "lonely", "mellow", "northern", "open", "patient",
    "restless", "southern", "still", "tender", "warm", "wild", "bright",
    "calm", "dusky", "eastern", "shallow", "western",
)
NOUNS = (
    "lantern", "harbor", "meadow", "compass", "bridge", "forest", "river",
    "stone", "window", "orchard", "vessel", "canyon", "garden", "island",
    "market", "pillar", "prairie", "quarry", "saddle", "temple", "valley",
    "willow", "anchor", "beacon", "cabin", "delta", "ferry", "glacier",
    "junction", "kettle", "ledge", "mosaic", "nest", "oar", "path", "quill",
    "reef", "shelf", "trail", "urn", "vine", "wharf", "yield", "zephyr",
    "arch", "bramble", "cairn", "drift", "ember", "fjord", "grove", "hearth",
    "inlet", "jetty", "knoll", "lagoon",
)
SECOND_NOUNS = (
    "ridge", "hollow", "reach", "moor", "glade", "shore", "plain", "bank",
    "bend", "run", "walk", "gate", "rest", "watch", "field", "cross",
    "marsh", "cove", "island", "landing", "trail", "pass", "vale", "point",
)

MAX_SAMPLE_ATTEMPTS = 50000
MAX_ROW_ATTEMPTS = 100

PROMPT_PAIRS_PATTERN = re.compile(r"\[([^\[\]]+)\]")
JSON_PROMPT_PREFIX = "Copy these pairs exactly into compact JSON"
KV_PROMPT_PREFIX = "Copy these pairs exactly into one-line KV"


def normalize_difficulties(
        difficulty: str | Iterable[str]) -> tuple[str, ...]:
    """Return a validated stage tuple in canonical ladder order.

    ``all`` selects every stage. A comma-separated string or an iterable may
    also select a subset. Unknown names raise ``ValueError``.
    """

    if isinstance(difficulty, str):
        items = [item.strip().lower() for item in difficulty.split(",")]
    else:
        items = [str(item).strip().lower() for item in difficulty]
    items = [item for item in items if item]
    if not items:
        raise ValueError("at least one difficulty is required")
    if "all" in items:
        return DIFFICULTY_ORDER
    result: list[str] = []
    for item in items:
        if item not in STAGE_SPECS:
            raise ValueError(
                f"unknown difficulty {item!r}; expected one of "
                f"{DIFFICULTY_ORDER} or 'all'")
        if item not in result:
            result.append(item)
    return tuple(stage for stage in DIFFICULTY_ORDER if stage in result)


def normalize_key_style(key_style: str) -> str:
    """Validate and return a key style name."""

    style = str(key_style).strip().lower()
    if style not in KEY_STYLES:
        raise ValueError(
            f"unknown key style {key_style!r}; expected one of {KEY_STYLES}")
    return style


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _pronounceable(rng: random.Random, syllables: int) -> str:
    return "".join(
        rng.choice(WORD_ONSETS) + rng.choice(VOWELS) + rng.choice(WORD_CODAS)
        for _ in range(syllables)
    )


def random_key(rng: random.Random, used: set[str],
               key_style: str, local: set[str] | None = None) -> str:
    """Draw a fresh 3-5 character key name in ``key_style``.

    ``used`` is the global registry; ``local`` optionally adds names already
    drawn for the current row. Keeping the two sets separate avoids copying a
    large registry on every draw.
    """

    if key_style not in KEY_PATTERNS:
        raise ValueError(f"unknown key style: {key_style}")
    for _ in range(MAX_SAMPLE_ATTEMPTS):
        if key_style == "short":
            length = rng.randint(3, 5)
            key = rng.choice(LOWER) + "".join(
                rng.choice(ALNUM) for _ in range(length - 1))
        elif key_style == "cv":
            key = "".join(
                rng.choice(CONSONANTS) + rng.choice(VOWELS)
                for _ in range(2)) + rng.choice(CONSONANTS)
        else:
            length = rng.randint(3, 5)
            key = "".join(rng.choice(LOWER) for _ in range(length))
        if (key not in used
                and (local is None or key not in local)
                and key not in RESERVED_KEYS
                and KEY_PATTERNS[key_style].fullmatch(key)):
            return key
    raise RuntimeError("could not draw a fresh random key")


def _token(rng: random.Random) -> str:
    for _ in range(MAX_SAMPLE_ATTEMPTS):
        token = _pronounceable(rng, rng.randint(1, 2))
        if 3 <= len(token) <= 6:
            return token
    raise RuntimeError("could not draw a short token")


def _single_word(rng: random.Random) -> str:
    for _ in range(MAX_SAMPLE_ATTEMPTS):
        word = _pronounceable(rng, rng.randint(2, 3))
        if 6 <= len(word) <= 10:
            return word
    raise RuntimeError("could not draw a single-word value")


def _multiword(rng: random.Random) -> str:
    words = [rng.choice(ADJECTIVES), rng.choice(NOUNS)]
    if rng.random() < 0.4:
        words.append(rng.choice(SECOND_NOUNS))
    return " ".join(words)


def _sample_value(pool: str, rng: random.Random, used: set[str],
                  local: set[str] | None = None) -> Any:
    """Draw a value from ``pool`` not present in ``used`` or ``local``."""

    for _ in range(MAX_SAMPLE_ATTEMPTS):
        if pool == "token":
            value: Any = _token(rng)
        elif pool == "word":
            value = _single_word(rng)
        elif pool == "multiword":
            value = _multiword(rng)
        else:
            raise ValueError(f"unknown value pool: {pool}")
        text = str(value)
        if text not in used and (local is None or text not in local):
            return value
    raise RuntimeError(f"value pool {pool!r} is exhausted")


def _value_pool_plan(rng: random.Random, stage: str) -> list[str]:
    spec = STAGE_SPECS[stage]
    if spec["value_plan"] == "mixed":
        pools = ["multiword", rng.choice(SINGLE_VALUE_POOLS)]
        rng.shuffle(pools)
        return pools
    return [rng.choice(SINGLE_VALUE_POOLS) for _ in range(spec["pairs"])]


def build_prompt(fmt: str, pairs: Sequence[Sequence[Any]]) -> str:
    """Render the explicit copy instruction with the exact pairs listed."""

    if fmt not in FORMATS:
        raise ValueError(f"unknown format: {fmt}")
    rendered = "; ".join(f"{key}={value}" for key, value in pairs)
    if fmt == "json":
        return (
            f"{JSON_PROMPT_PREFIX}: [{rendered}]. Return only the JSON object "
            "and keep exactly these keys and values. Do not rename keys, "
            "change values, reorder them, or add prose or code fences."
        )
    return (
        f"{KV_PROMPT_PREFIX}: [{rendered}]. Return only one line of "
        "semicolon-separated key=value pairs and keep exactly these keys and "
        "values. Do not rename them, change values, or add prose or code "
        "fences."
    )


def render_answer(fmt: str, pairs: Sequence[Sequence[Any]]) -> str:
    """Render the canonical answer for ``pairs`` in ``fmt``."""

    if fmt == "json":
        return json.dumps(dict(pairs), separators=(",", ":"),
                          ensure_ascii=False)
    if fmt == "kv":
        return ";".join(f"{key}={value}" for key, value in pairs)
    raise ValueError(f"unknown format: {fmt}")


def _clean(text: Any) -> str:
    text = str(text or "").strip()
    if text.startswith("```") and text.endswith("```") and "\n" in text:
        lines = text.splitlines()
        if len(lines) >= 2:
            text = "\n".join(lines[1:-1]).strip()
    return text


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def _json_object_no_duplicates(pairs: list[tuple[str, Any]]) -> dict:
    keys = [key for key, _ in pairs]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate JSON keys")
    return dict(pairs)


def parse_answer(fmt: str, answer: Any) -> dict[str, Any]:
    """Parse an answer into a mapping. Raise ``ValueError`` if malformed."""

    if fmt not in FORMATS:
        raise ValueError(f"unknown format: {fmt}")
    text = _clean(answer)
    if not text:
        raise ValueError("empty answer")
    if fmt == "json":
        try:
            parsed = json.loads(
                text, object_pairs_hook=_json_object_no_duplicates)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"invalid JSON answer: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("JSON answer must be an object")
        return parsed
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError("KV answer must be exactly one line")
    pairs: dict[str, str] = {}
    for chunk in lines[0].split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError("KV pair is missing '='")
        key, value = chunk.split("=", 1)
        key = key.strip()
        value = _unquote(value)
        if not key:
            raise ValueError("KV pair has an empty key")
        if key in pairs:
            raise ValueError(f"duplicate KV key: {key}")
        pairs[key] = value
    if not pairs:
        raise ValueError("KV answer has no pairs")
    return pairs


def _values_equal(parsed: Any, expected: Any) -> bool:
    if isinstance(expected, bool):
        return isinstance(parsed, bool) and parsed is expected
    if isinstance(expected, int):
        if isinstance(parsed, bool):
            return False
        if isinstance(parsed, int):
            return parsed == expected
        return str(parsed).strip() == str(expected)
    return str(parsed).strip() == str(expected)


def validate_answer(fmt: str, answer: Any,
                    pairs: Sequence[Sequence[Any]]) -> bool:
    """Return True iff ``answer`` parses to exactly the requested pairs."""

    expected = {str(key): value for key, value in pairs}
    try:
        parsed = parse_answer(fmt, answer)
    except ValueError:
        return False
    if set(parsed) != set(expected):
        return False
    return all(_values_equal(parsed[key], expected[key]) for key in expected)


def extract_prompt_pairs(prompt: Any) -> list[list[str]]:
    """Extract the bracketed ``key=value`` pairs from a prompt."""

    matches = PROMPT_PAIRS_PATTERN.findall(str(prompt))
    if len(matches) != 1:
        raise ValueError("prompt must contain exactly one pair block")
    extracted: list[list[str]] = []
    for chunk in matches[0].split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError("prompt pair is missing '='")
        key, value = chunk.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise ValueError("prompt pair has an empty key or value")
        extracted.append([key, value])
    if not extracted:
        raise ValueError("prompt has no pairs")
    return extracted


def validate_row(row: Mapping[str, Any]) -> bool:
    """Validate one generated row end to end."""

    stage = row.get("stage")
    fmt = row.get("format")
    if stage not in STAGE_SPECS or fmt not in STAGE_SPECS[stage]["formats"]:
        return False
    if row.get("family") != f"multi_short_{stage}":
        return False
    keys = list(row.get("keys", ()))
    values = list(row.get("values", ()))
    raw_pairs = list(row.get("pairs", ()))
    if any(not isinstance(pair, (list, tuple)) or len(pair) != 2
           for pair in raw_pairs):
        return False
    pairs = [[pair[0], pair[1]] for pair in raw_pairs]
    if len(keys) != len(values) or len(keys) != len(pairs):
        return False
    spec = STAGE_SPECS[stage]
    if len(keys) != spec["pairs"]:
        return False
    if len(keys) != len(set(keys)):
        return False
    if [(pair[0], pair[1]) for pair in pairs] != list(zip(keys, values)):
        return False
    key_style = row.get("key_style")
    if key_style not in KEY_PATTERNS:
        return False
    if any(not KEY_PATTERNS[key_style].fullmatch(str(key))
           or str(key) in RESERVED_KEYS for key in keys):
        return False
    if list(row.get("key_styles", ())) != [key_style] * len(keys):
        return False
    pools = list(row.get("value_pools", ()))
    if len(pools) != len(values):
        return False
    if spec["value_plan"] == "single":
        if set(pools) - set(SINGLE_VALUE_POOLS):
            return False
        if any(" " in str(value) for value in values):
            return False
    else:
        if pools.count("multiword") != 1:
            return False
        if set(pools) - set(VALUE_POOLS):
            return False
        multiword = values[pools.index("multiword")]
        if " " not in str(multiword):
            return False
        for pool, value in zip(pools, values):
            if pool != "multiword" and " " in str(value):
                return False
    prompt = str(row.get("user", ""))
    expected_prefix = (JSON_PROMPT_PREFIX if fmt == "json"
                       else KV_PROMPT_PREFIX)
    if expected_prefix not in prompt:
        return False
    try:
        extracted = extract_prompt_pairs(prompt)
    except ValueError:
        return False
    if extracted != [[str(key), str(value)] for key, value in pairs]:
        return False
    answer = str(row.get("assistant", ""))
    if answer != render_answer(fmt, pairs):
        return False
    if not validate_answer(fmt, answer, pairs):
        return False
    verification = row.get("verification")
    if not isinstance(verification, Mapping):
        return False
    if (verification.get("verified") is not True
            or verification.get("status") != "verified"
            or verification.get("method") != VERIFICATION_METHOD):
        return False
    if verification.get("prompt_sha256") != _sha256(prompt):
        return False
    if verification.get("answer_sha256") != _sha256(answer):
        return False
    return True


def _build_row(rng: random.Random, stage: str, fmt: str, split: str,
               key_style: str, used_keys: set[str],
               used_values: set[str]) -> tuple[dict[str, Any], int]:
    """Build one row, regenerating until its canonical answer validates."""

    spec = STAGE_SPECS[stage]
    rejected = 0
    for _ in range(MAX_ROW_ATTEMPTS):
        keys: list[str] = []
        local_keys: set[str] = set()
        for _ in range(spec["pairs"]):
            key = random_key(rng, used_keys, key_style, local_keys)
            keys.append(key)
            local_keys.add(key)
        pools = _value_pool_plan(rng, stage)
        values: list[Any] = []
        local_values: set[str] = set()
        for pool in pools:
            value = _sample_value(pool, rng, used_values, local_values)
            values.append(value)
            local_values.add(str(value))
        pairs = [[key, value] for key, value in zip(keys, values)]
        prompt = build_prompt(fmt, pairs)
        answer = render_answer(fmt, pairs)
        row: dict[str, Any] = {
            "user": prompt,
            "assistant": answer,
            "family": f"multi_short_{stage}",
            "stage": stage,
            "stage_index": spec["index"],
            "format": fmt,
            "partition": split,
            "key_style": key_style,
            "keys": keys,
            "values": values,
            "pairs": pairs,
            "key_styles": [key_style] * len(keys),
            "value_pools": pools,
            "verification": {
                "verified": True,
                "promotable": True,
                "status": "verified",
                "method": VERIFICATION_METHOD,
                "checks": [
                    "prompt_contains_exact_pairs",
                    "answer_parses",
                    "keys_match",
                    "values_match",
                    "keys_unique",
                    "key_length_3_to_5",
                    "verification_hashes_match",
                ],
                "prompt_sha256": _sha256(prompt),
                "answer_sha256": _sha256(answer),
            },
        }
        if validate_row(row):
            used_keys.update(keys)
            used_values.update(str(value) for value in values)
            return row, rejected
        rejected += 1
    raise RuntimeError(
        f"could not build a valid {stage} row after {MAX_ROW_ATTEMPTS} "
        "attempts")


def _generate_split(count: int, split: str, stage: str,
                    key_style: str, rng: random.Random,
                    used_keys: set[str],
                    used_values: set[str]) -> tuple[list[dict[str, Any]], int]:
    formats = STAGE_SPECS[stage]["formats"]
    rows: list[dict[str, Any]] = []
    rejected = 0
    for index in range(count):
        fmt = formats[index % len(formats)]
        row, row_rejected = _build_row(
            rng, stage, fmt, split, key_style, used_keys, used_values)
        rows.append(row)
        rejected += row_rejected
    return rows, rejected


def _row_keys(rows: Iterable[Mapping[str, Any]]) -> set[str]:
    return {str(key) for row in rows for key in row.get("keys", ())}


def _row_values(rows: Iterable[Mapping[str, Any]]) -> set[str]:
    return {str(value) for row in rows for value in row.get("values", ())}


def _examples(values: set[str], limit: int = 5) -> list[str]:
    return sorted(values)[:limit]


def _pair_stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = [len(row.get("keys", ())) for row in rows]
    if not counts:
        return {"min": 0, "max": 0, "mean": 0.0}
    return {
        "min": min(counts),
        "max": max(counts),
        "mean": round(sum(counts) / len(counts), 3),
    }


def summarize_stage(stage: str, train: Sequence[dict],
                    eval_rows: Sequence[dict], *, seed: int,
                    key_style: str, shuffle: bool,
                    stage_order: Sequence[str],
                    requested_rows: int, requested_eval_rows: int,
                    rejected_attempts: int) -> dict[str, Any]:
    """Build the per-stage machine-readable summary."""

    spec = STAGE_SPECS[stage]
    train_keys = _row_keys(train)
    eval_keys = _row_keys(eval_rows)
    train_values = _row_values(train)
    eval_values = _row_values(eval_rows)
    key_overlap = train_keys & eval_keys
    value_overlap = train_values & eval_values
    all_rows = list(train) + list(eval_rows)
    formats = list(spec["formats"])
    train_formats = Counter(row["format"] for row in train)
    eval_formats = Counter(row["format"] for row in eval_rows)
    total_formats = Counter(row["format"] for row in all_rows)
    key_styles: Counter[str] = Counter(
        key_style for row in all_rows for _ in row.get("keys", ()))
    value_pools: Counter[str] = Counter()
    for row in all_rows:
        value_pools.update(row.get("value_pools", ()))
    return {
        "dataset": DATASET_NAME,
        "stage": stage,
        "difficulty": stage,
        "stage_index": spec["index"],
        "stage_order": list(stage_order),
        "description": spec["description"],
        "formats": formats,
        "pair_count": spec["pairs"],
        "value_plan": spec["value_plan"],
        "seed": seed,
        "key_style": key_style,
        "key_length": {"min": 3, "max": 5},
        "shuffle": shuffle,
        "requested_rows": requested_rows,
        "requested_eval_rows": requested_eval_rows,
        "train_rows": len(train),
        "eval_rows": len(eval_rows),
        "total_rows": len(all_rows),
        "rows": {
            "train": len(train),
            "eval": len(eval_rows),
            "total": len(all_rows),
        },
        "format_counts": {
            "train": {fmt: train_formats.get(fmt, 0) for fmt in formats},
            "eval": {fmt: eval_formats.get(fmt, 0) for fmt in formats},
            "total": {fmt: total_formats.get(fmt, 0) for fmt in formats},
        },
        "key_overlap": len(key_overlap),
        "key_overlap_examples": _examples(key_overlap),
        "value_overlap": len(value_overlap),
        "value_overlap_examples": _examples(value_overlap),
        "unique_keys": {
            "train": len(train_keys),
            "eval": len(eval_keys),
            "total": len(train_keys | eval_keys),
        },
        "unique_values": {
            "train": len(train_values),
            "eval": len(eval_values),
            "total": len(train_values | eval_values),
        },
        "pairs_per_row": _pair_stats(all_rows),
        "key_styles": {style: key_styles.get(style, 0)
                       for style in KEY_STYLES},
        "value_pools": {pool: value_pools.get(pool, 0)
                        for pool in VALUE_POOLS},
        "verified_rows": sum(
            1 for row in all_rows
            if row.get("verification", {}).get("verified") is True),
        "rejected_attempts": rejected_attempts,
    }


def summarize_dataset(stage_data: Mapping[str, Mapping[str, Any]], *,
                      stage_order: Sequence[str], seed: int,
                      key_style: str, shuffle: bool,
                      requested_rows: int,
                      requested_eval_rows: int,
                      rejected_attempts: int) -> dict[str, Any]:
    """Build the root summary across all generated stages."""

    all_train: list[dict] = []
    all_eval: list[dict] = []
    stage_entries: list[dict[str, Any]] = []
    all_formats: set[str] = set()
    for stage in stage_order:
        entry = stage_data[stage]
        all_train.extend(entry["train"])
        all_eval.extend(entry["eval"])
        summary = entry["summary"]
        all_formats.update(summary["formats"])
        stage_entries.append({
            "stage": stage,
            "difficulty": stage,
            "stage_index": STAGE_SPECS[stage]["index"],
            "description": STAGE_SPECS[stage]["description"],
            "formats": summary["formats"],
            "pair_count": summary["pair_count"],
            "train_rows": summary["train_rows"],
            "eval_rows": summary["eval_rows"],
            "key_overlap": summary["key_overlap"],
            "value_overlap": summary["value_overlap"],
        })
    formats = sorted(all_formats)
    train_keys = _row_keys(all_train)
    eval_keys = _row_keys(all_eval)
    train_values = _row_values(all_train)
    eval_values = _row_values(all_eval)
    key_overlap = train_keys & eval_keys
    value_overlap = train_values & eval_values
    train_formats = Counter(row["format"] for row in all_train)
    eval_formats = Counter(row["format"] for row in all_eval)
    total_formats = Counter(
        row["format"] for row in all_train + all_eval)
    return {
        "dataset": DATASET_NAME,
        "seed": seed,
        "key_style": key_style,
        "key_length": {"min": 3, "max": 5},
        "shuffle": shuffle,
        "requested_rows": requested_rows,
        "requested_eval_rows": requested_eval_rows,
        "difficulty_order": list(stage_order),
        "stage_order": list(stage_order),
        "stages": stage_entries,
        "formats": formats,
        "train_rows": len(all_train),
        "eval_rows": len(all_eval),
        "total_rows": len(all_train) + len(all_eval),
        "rows": {
            "train": len(all_train),
            "eval": len(all_eval),
            "total": len(all_train) + len(all_eval),
        },
        "format_counts": {
            "train": {fmt: train_formats.get(fmt, 0) for fmt in formats},
            "eval": {fmt: eval_formats.get(fmt, 0) for fmt in formats},
            "total": {fmt: total_formats.get(fmt, 0) for fmt in formats},
        },
        "key_overlap": len(key_overlap),
        "key_overlap_examples": _examples(key_overlap),
        "value_overlap": len(value_overlap),
        "value_overlap_examples": _examples(value_overlap),
        "unique_keys": {
            "train": len(train_keys),
            "eval": len(eval_keys),
            "total": len(train_keys | eval_keys),
        },
        "unique_values": {
            "train": len(train_values),
            "eval": len(eval_values),
            "total": len(train_values | eval_values),
        },
        "pairs_per_row": _pair_stats(all_train + all_eval),
        "verified_rows": sum(
            1 for row in all_train + all_eval
            if row.get("verification", {}).get("verified") is True),
        "rejected_attempts": rejected_attempts,
    }


def build_dataset(*, rows: int = 2000, eval_rows: int = 200,
                  difficulty: str | Iterable[str] = "all",
                  key_style: str = "short", seed: int = 23,
                  shuffle: bool = False) -> dict[str, Any]:
    """Generate the requested stages. No files are written."""

    if rows < 0 or eval_rows < 0:
        raise ValueError("rows and eval_rows must be non-negative")
    stage_order = normalize_difficulties(difficulty)
    style = normalize_key_style(key_style)
    used_keys: set[str] = set()
    used_values: set[str] = set()
    stage_data: dict[str, dict[str, Any]] = {}
    total_rejected = 0
    for stage in stage_order:
        rng = random.Random(f"{seed}:{stage}")
        train, train_rejected = _generate_split(
            rows, "train", stage, style, rng, used_keys, used_values)
        eval_data, eval_rejected = _generate_split(
            eval_rows, "eval", stage, style, rng, used_keys, used_values)
        if shuffle:
            random.Random(f"{seed}:{stage}:train").shuffle(train)
            random.Random(f"{seed}:{stage}:eval").shuffle(eval_data)
        overlap = _row_keys(train) & _row_keys(eval_data)
        if overlap:
            raise RuntimeError(
                f"stage {stage} leaked {len(overlap)} keys into eval")
        value_overlap = _row_values(train) & _row_values(eval_data)
        if value_overlap:
            raise RuntimeError(
                f"stage {stage} leaked {len(value_overlap)} values into eval")
        rejected = train_rejected + eval_rejected
        total_rejected += rejected
        summary = summarize_stage(
            stage, train, eval_data, seed=seed, key_style=style,
            shuffle=shuffle, stage_order=stage_order,
            requested_rows=rows, requested_eval_rows=eval_rows,
            rejected_attempts=rejected)
        stage_data[stage] = {
            "name": stage,
            "index": STAGE_SPECS[stage]["index"],
            "train": train,
            "eval": eval_data,
            "summary": summary,
        }
    root_summary = summarize_dataset(
        stage_data, stage_order=stage_order, seed=seed, key_style=style,
        shuffle=shuffle, requested_rows=rows,
        requested_eval_rows=eval_rows, rejected_attempts=total_rejected)
    if root_summary["key_overlap"] or root_summary["value_overlap"]:
        raise RuntimeError("train and eval overlap across stages")
    return {
        "stage_order": list(stage_order),
        "stages": stage_data,
        "summary": root_summary,
    }


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(row, ensure_ascii=False) + "\n"
                   for row in rows)
    path.write_text(body, encoding="utf-8")


def _write_manifest(path: Path, summary: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def write_dataset(dataset: Mapping[str, Any],
                  out_dir: str | Path) -> dict[str, Any]:
    """Write one directory per stage plus a root manifest."""

    root = Path(out_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    stage_paths: dict[str, dict[str, str]] = {}
    for stage in dataset["stage_order"]:
        entry = dataset["stages"][stage]
        stage_dir = root / f"stage_{stage}"
        train_path = stage_dir / "train.jsonl"
        eval_path = stage_dir / "eval.jsonl"
        manifest_path = stage_dir / "manifest.json"
        _write_jsonl(train_path, entry["train"])
        _write_jsonl(eval_path, entry["eval"])
        summary = dict(entry["summary"])
        summary["paths"] = {
            "train": str(train_path),
            "eval": str(eval_path),
            "manifest": str(manifest_path),
        }
        _write_manifest(manifest_path, summary)
        stage_paths[stage] = summary["paths"]
    root_summary = dict(dataset["summary"])
    root_summary["paths"] = {
        "root": str(root),
        "manifest": str(root / "manifest.json"),
        "stages": stage_paths,
    }
    _write_manifest(root / "manifest.json", root_summary)
    return root_summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--out-dir", default="data/multi_short_v1",
        help="output directory (default: data/multi_short_v1)")
    parser.add_argument(
        "--rows", type=int, default=2000,
        help="number of training rows per stage (default: 2000)")
    parser.add_argument(
        "--eval-rows", type=int, default=200,
        help="number of held-out eval rows per stage (default: 200)")
    parser.add_argument(
        "--seed", type=int, default=23,
        help="deterministic RNG seed (default: 23)")
    parser.add_argument(
        "--difficulty", default="all",
        metavar="{2short,2mixed,3short,4short,all}",
        help="difficulty stage, comma list, or all (default: all)")
    parser.add_argument(
        "--key-style", choices=KEY_STYLES, default="short",
        help="random key style (default: short)")
    parser.add_argument(
        "--shuffle", action="store_true",
        help="deterministically shuffle train and eval rows per stage")
    args = parser.parse_args(argv)
    try:
        dataset = build_dataset(
            rows=args.rows, eval_rows=args.eval_rows,
            difficulty=args.difficulty, key_style=args.key_style,
            seed=args.seed, shuffle=args.shuffle)
        summary = write_dataset(dataset, args.out_dir)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
