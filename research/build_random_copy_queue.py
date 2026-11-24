"""Build a deterministic randomized-key copy curriculum.

The fixed key-pair curriculum reuses a small set of key names (``name``,
``score``, ``city``, ``population`` ...). A small model can memorize those key
names and still fail to copy arbitrary keys from the prompt. This generator
removes that shortcut:

- every row draws fresh key names from a huge space (``k_<hex>``,
  ``field_<base36>`` or short consonant-vowel strings);
- keys are unique within a row and across the whole build, so ``train.jsonl``
  and ``eval.jsonl`` share no key names;
- entity values are synthetic and globally unique, so the two splits share no
  values either (``value_overlap == 0`` for the default build);
- the prompt lists the exact ``key=value`` pairs, and the answer must copy them
  into JSON, key-value or YAML form.

The generator uses only the standard library. It never imports ``torch`` or a
model, so the queue can be built and tested on CPU in a fraction of a second.

Rows follow the teacher-queue schema used elsewhere in ``research/``:
``user``, ``assistant``, ``family``, ``verification`` plus copy-curriculum
metadata (``keys``, ``values``, ``pairs``, ``key_styles``, ``value_pools``).

Default output is ``data/random_copy_v1`` (gitignored). ``--rows`` is the
number of training rows; ``--eval-rows`` is the number of held-out rows, written
to ``--holdout`` (default: ``<out-dir>/eval.jsonl``).
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

FORMATS = ("json", "kv", "yaml")
DEFAULT_FORMATS = ("json", "kv")
KEY_STYLES = ("hex", "base36", "cv")
BASE_VALUE_POOLS = ("token", "name", "city", "number")
VALUE_POOLS = BASE_VALUE_POOLS + ("multiword",)
VERIFICATION_METHOD = "deterministic_random_copy_roundtrip_v1"

# Key names from the fixed-key curriculum and common structured-output
# conventions. Random keys must never collide with these, so the model cannot
# fall back on a memorized key name from the earlier curriculum.
RESERVED_KEYS = frozenset({
    "name", "score", "product", "price", "city", "population", "label",
    "points", "code", "rank", "team", "wins", "author", "year", "song",
    "length", "model", "params", "river", "km", "planet", "mass", "recipe",
    "minutes", "value", "key", "id", "data", "text", "answer", "output",
    "result", "input", "prompt", "response", "format", "type",
})

KEY_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_]{2,31}")
BASE36 = "0123456789abcdefghijklmnopqrstuvwxyz"
HEX = "0123456789abcdef"

CONSONANTS = "bcdfghjklmnprstvwz"
VOWELS = "aeiou"

# Synthetic name pieces. The space is thousands of names; the used-value
# registry guarantees no repetition across train and eval.
NAME_PREFIXES = (
    "ka", "lo", "mi", "ra", "te", "vo", "zu", "bel", "dar", "fen",
    "gil", "hal", "jor", "kel", "mar", "nor", "pel", "quin", "ros", "sar",
    "tor", "val", "wen", "xan", "yar", "zel", "bri", "cal", "do", "e",
)
NAME_MIDDLES = (
    "va", "lo", "mi", "ra", "ne", "di", "so", "tu", "ke", "ba",
    "ri", "no", "lu", "ma", "de", "fi", "go", "ha", "je", "ki",
    "po", "se", "ta", "vo", "we", "ya", "ze", "bo", "cu", "ga",
)
NAME_TAILS = ("n", "ra", "l", "th", "s", "nd", "ric", "na", "el", "ia")

CITY_PREFIXES = (
    "nor", "bel", "mar", "kel", "tor", "val", "bri", "cal", "dun", "east",
    "west", "north", "south", "port", "new", "upper", "lower", "ash",
    "black", "clear", "fair", "glen", "high", "iron", "kings", "long",
)
CITY_ROOTS = (
    "vale", "brook", "field", "haven", "ford", "gate", "holm", "stead",
    "wick", "dale", "moor", "crest", "ridge", "bay", "port", "marsh",
    "hill", "glen", "water", "wood",
)
CITY_SUFFIXES = ("", "ton", "burg", "shire", "by", "mouth", "stead", "ford")

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

MAX_SAMPLE_ATTEMPTS = 20000
MAX_ROW_ATTEMPTS = 100


def normalize_formats(formats: str | Iterable[str]) -> tuple[str, ...]:
    """Return a validated, de-duplicated format tuple in request order."""

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
    return tuple(result)


def pair_count_bounds(keys_per_row: int) -> tuple[int, int]:
    """Return the inclusive pair-count range for a row.

    ``keys_per_row`` is the minimum number of key/value pairs. With the default
    value of 2, rows contain 2-4 pairs. Values above 4 are used exactly.
    """

    if keys_per_row < 2:
        raise ValueError("keys_per_row must be at least 2")
    low = keys_per_row
    high = 4 if keys_per_row <= 4 else keys_per_row
    return low, high


def random_key(rng: random.Random, used: set[str]) -> tuple[str, str]:
    """Draw a fresh arbitrary key name and return ``(key, style)``."""

    for _ in range(MAX_SAMPLE_ATTEMPTS):
        style = rng.choice(KEY_STYLES)
        if style == "hex":
            key = "k_" + "".join(rng.choice(HEX) for _ in range(8))
        elif style == "base36":
            key = "field_" + "".join(rng.choice(BASE36) for _ in range(6))
        else:
            syllables = rng.randint(2, 3)
            key = "".join(rng.choice(CONSONANTS) + rng.choice(VOWELS)
                          for _ in range(syllables))
        if (key not in used and key not in RESERVED_KEYS
                and KEY_PATTERN.fullmatch(key)):
            return key, style
    raise RuntimeError("could not draw a fresh random key")


def _token(rng: random.Random) -> str:
    syllables = rng.randint(2, 3)
    return "".join(rng.choice(CONSONANTS) + rng.choice(VOWELS)
                   for _ in range(syllables))


def _name(rng: random.Random) -> str:
    parts = [rng.choice(NAME_PREFIXES), rng.choice(NAME_MIDDLES)]
    if rng.random() < 0.55:
        parts.append(rng.choice(NAME_TAILS))
    return "".join(parts).capitalize()


def _city(rng: random.Random) -> str:
    name = rng.choice(CITY_PREFIXES) + rng.choice(CITY_ROOTS)
    if rng.random() < 0.65:
        name += rng.choice(CITY_SUFFIXES)
    return name.capitalize()


def _multiword(rng: random.Random) -> str:
    words = [rng.choice(ADJECTIVES), rng.choice(NOUNS)]
    if rng.random() < 0.4:
        words.append(rng.choice(SECOND_NOUNS))
    return " ".join(words)


def _sample_value(pool: str, rng: random.Random,
                  used: set[str]) -> Any:
    """Draw a value from ``pool`` whose text form is not in ``used``."""

    for _ in range(MAX_SAMPLE_ATTEMPTS):
        if pool == "token":
            value: Any = _token(rng)
        elif pool == "name":
            value = _name(rng)
        elif pool == "city":
            value = _city(rng)
        elif pool == "number":
            value = rng.randint(1, 999999)
        elif pool == "multiword":
            value = _multiword(rng)
        else:
            raise ValueError(f"unknown value pool: {pool}")
        if str(value) not in used:
            return value
    raise RuntimeError(f"value pool {pool!r} is exhausted")


def build_prompt(fmt: str, pairs: Sequence[Sequence[Any]]) -> str:
    """Render the instruction with the exact key=value pairs to copy."""

    if fmt not in FORMATS:
        raise ValueError(f"unknown format: {fmt}")
    rendered = "; ".join(f"{key}={value}" for key, value in pairs)
    if fmt == "json":
        return (
            "Return only a compact JSON object with exactly these keys and "
            f"values: [{rendered}]. Copy every key and value exactly; do not "
            "rename or reorder them. Do not add prose or code fences."
        )
    if fmt == "kv":
        return (
            "Return only one line of semicolon-separated key=value pairs with "
            "no prose and no code fence. Use exactly these pairs in this "
            f"order: [{rendered}]. Copy every key and value exactly; do not "
            "rename them."
        )
    return (
        "Return only YAML lines in the form key: value, one pair per line, "
        "with no prose and no code fence. Use exactly these keys and values: "
        f"[{rendered}]. Copy every key and value exactly; do not rename or "
        "reorder them."
    )


def render_answer(fmt: str, pairs: Sequence[Sequence[Any]]) -> str:
    """Render the canonical answer for ``pairs`` in ``fmt``."""

    if fmt == "json":
        return json.dumps(dict(pairs), separators=(",", ":"),
                          ensure_ascii=False)
    if fmt == "kv":
        return ";".join(f"{key}={value}" for key, value in pairs)
    if fmt == "yaml":
        return "\n".join(f"{key}: {value}" for key, value in pairs)
    raise ValueError(f"unknown format: {fmt}")


def _clean(text: Any) -> str:
    text = str(text or "").strip()
    if text.startswith("```") and text.endswith("```") and "\n" in text:
        lines = text.splitlines()
        if len(lines) >= 2:
            text = "\n".join(lines[1:-1]).strip()
    return text


def _expected_map(pairs: Sequence[Sequence[Any]]) -> dict[str, Any]:
    return {str(key): value for key, value in pairs}


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


def _json_object_no_duplicates(pairs: list[tuple[str, Any]]) -> dict:
    keys = [key for key, _ in pairs]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate JSON keys")
    return dict(pairs)


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


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
    if fmt == "kv":
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
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("empty YAML answer")
    pairs = {}
    for line in lines:
        if ":" not in line:
            raise ValueError("YAML line is missing ':'")
        key, value = line.split(":", 1)
        key = key.strip()
        value = _unquote(value)
        if not key:
            raise ValueError("YAML line has an empty key")
        if key in pairs:
            raise ValueError(f"duplicate YAML key: {key}")
        pairs[key] = value
    return pairs


def validate_answer(fmt: str, answer: Any,
                    pairs: Sequence[Sequence[Any]]) -> bool:
    """Return True iff ``answer`` parses to exactly the requested pairs."""

    expected = _expected_map(pairs)
    try:
        parsed = parse_answer(fmt, answer)
    except ValueError:
        return False
    if set(parsed) != set(expected):
        return False
    return all(_values_equal(parsed[key], expected[key]) for key in expected)


def validate_row(row: Mapping[str, Any]) -> bool:
    """Validate one generated row end to end."""

    fmt = row.get("format")
    keys = list(row.get("keys", ()))
    values = list(row.get("values", ()))
    raw_pairs = list(row.get("pairs", ()))
    if any(not isinstance(pair, (list, tuple)) or len(pair) != 2
           for pair in raw_pairs):
        return False
    pairs = [[pair[0], pair[1]] for pair in raw_pairs]
    if fmt not in FORMATS or len(keys) != len(values) or len(keys) != len(pairs):
        return False
    if len(keys) != len(set(keys)):
        return False
    if [(pair[0], pair[1]) for pair in pairs] != list(zip(keys, values)):
        return False
    prompt = str(row.get("user", ""))
    if any(f"{key}={value}" not in prompt for key, value in pairs):
        return False
    return validate_answer(fmt, row.get("assistant"), pairs)


def _build_row(rng: random.Random, fmt: str, keys_per_row: int,
               used_keys: set[str], used_values: set[str],
               split: str) -> tuple[dict, int]:
    """Build one row, regenerating until its canonical answer validates."""

    low, high = pair_count_bounds(keys_per_row)
    rejected = 0
    for _ in range(MAX_ROW_ATTEMPTS):
        pair_count = rng.randint(low, high)
        local_keys: set[str] = set()
        keys: list[str] = []
        key_styles: list[str] = []
        for _ in range(pair_count):
            key, style = random_key(rng, used_keys | local_keys)
            keys.append(key)
            key_styles.append(style)
            local_keys.add(key)
        pool_plan = ["multiword"] + [
            rng.choice(BASE_VALUE_POOLS) for _ in range(pair_count - 1)]
        rng.shuffle(pool_plan)
        local_values: set[str] = set()
        values: list[Any] = []
        for pool in pool_plan:
            value = _sample_value(
                pool, rng, used_values | local_values)
            values.append(value)
            local_values.add(str(value))
        pairs = [[key, value] for key, value in zip(keys, values)]
        row = {
            "user": build_prompt(fmt, pairs),
            "assistant": render_answer(fmt, pairs),
            "family": f"random_copy_{fmt}",
            "format": fmt,
            "partition": split,
            "keys": keys,
            "values": values,
            "pairs": pairs,
            "key_styles": key_styles,
            "value_pools": pool_plan,
            "verification": {
                "verified": True,
                "promotable": True,
                "status": "verified",
                "method": VERIFICATION_METHOD,
            },
        }
        if validate_row(row):
            used_keys.update(keys)
            used_values.update(str(value) for value in values)
            return row, rejected
        rejected += 1
    raise RuntimeError(
        f"could not build a valid {fmt} row after {MAX_ROW_ATTEMPTS} attempts")


def _generate_split(count: int, split: str, formats: Sequence[str],
                    keys_per_row: int, rng: random.Random,
                    used_keys: set[str], used_values: set[str]) -> tuple[
                        list[dict], int]:
    rows: list[dict] = []
    rejected = 0
    for index in range(count):
        fmt = formats[index % len(formats)]
        row, row_rejected = _build_row(
            rng, fmt, keys_per_row, used_keys, used_values, split)
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


def summarize_dataset(train: Sequence[dict], eval_rows: Sequence[dict], *,
                      seed: int,
                      requested_rows: int,
                      requested_eval_rows: int,
                      formats: Sequence[str],
                      keys_per_row: int,
                      shuffle: bool,
                      rejected_attempts: int) -> dict[str, Any]:
    """Build the machine-readable summary, including overlap checks."""

    train_keys = _row_keys(train)
    eval_keys = _row_keys(eval_rows)
    train_values = _row_values(train)
    eval_values = _row_values(eval_rows)
    key_overlap = train_keys & eval_keys
    value_overlap = train_values & eval_values
    all_rows = list(train) + list(eval_rows)
    train_formats = Counter(row["format"] for row in train)
    eval_formats = Counter(row["format"] for row in eval_rows)
    total_formats = Counter(row["format"] for row in all_rows)
    key_styles: Counter[str] = Counter()
    value_pools: Counter[str] = Counter()
    for row in all_rows:
        key_styles.update(row.get("key_styles", ()))
        value_pools.update(row.get("value_pools", ()))
    return {
        "seed": seed,
        "shuffle": shuffle,
        "keys_per_row": keys_per_row,
        "requested_rows": requested_rows,
        "requested_eval_rows": requested_eval_rows,
        "formats": list(formats),
        "train_rows": len(train),
        "eval_rows": len(eval_rows),
        "total_rows": len(all_rows),
        "rows": {"train": len(train), "eval": len(eval_rows),
                 "total": len(all_rows)},
        "format_counts": {
            "train": {fmt: train_formats.get(fmt, 0) for fmt in formats},
            "eval": {fmt: eval_formats.get(fmt, 0) for fmt in formats},
            "total": {fmt: total_formats.get(fmt, 0) for fmt in formats},
        },
        "key_overlap": len(key_overlap),
        "key_overlap_examples": _examples(key_overlap),
        "value_overlap": len(value_overlap),
        "value_overlap_examples": _examples(value_overlap),
        "unique_keys": {"train": len(train_keys), "eval": len(eval_keys),
                        "total": len(train_keys | eval_keys)},
        "unique_values": {"train": len(train_values),
                          "eval": len(eval_values),
                          "total": len(train_values | eval_values)},
        "pairs_per_row": _pair_stats(all_rows),
        "key_styles": {style: key_styles.get(style, 0)
                       for style in KEY_STYLES},
        "value_pools": {pool: value_pools.get(pool, 0)
                        for pool in VALUE_POOLS},
        "rejected_attempts": rejected_attempts,
    }


def build_dataset(*, rows: int = 2000, eval_rows: int = 200,
                  formats: str | Sequence[str] = DEFAULT_FORMATS,
                  keys_per_row: int = 2, seed: int = 23,
                  shuffle: bool = False) -> dict[str, Any]:
    """Generate train and eval rows plus a summary. No files are written."""

    if rows < 0 or eval_rows < 0:
        raise ValueError("rows and eval_rows must be non-negative")
    format_list = normalize_formats(formats)
    pair_count_bounds(keys_per_row)
    rng = random.Random(seed)
    used_keys: set[str] = set()
    used_values: set[str] = set()
    train, train_rejected = _generate_split(
        rows, "train", format_list, keys_per_row, rng, used_keys, used_values)
    eval_data, eval_rejected = _generate_split(
        eval_rows, "eval", format_list, keys_per_row, rng,
        used_keys, used_values)
    if shuffle:
        random.Random(f"{seed}:train").shuffle(train)
        random.Random(f"{seed}:eval").shuffle(eval_data)
    summary = summarize_dataset(
        train, eval_data, seed=seed, requested_rows=rows,
        requested_eval_rows=eval_rows, formats=format_list,
        keys_per_row=keys_per_row, shuffle=shuffle,
        rejected_attempts=train_rejected + eval_rejected)
    return {"train": train, "eval": eval_data, "summary": summary}


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(row, ensure_ascii=False) + "\n"
                   for row in rows)
    path.write_text(body, encoding="utf-8")


def write_dataset(dataset: Mapping[str, Any], out_dir: str | Path,
                  holdout: str | Path | None = None) -> dict[str, Any]:
    """Write ``train.jsonl``, the eval file and ``manifest.json``."""

    directory = Path(out_dir).expanduser()
    train_path = directory / "train.jsonl"
    eval_path = (Path(holdout).expanduser() if holdout is not None
                 else directory / "eval.jsonl")
    if train_path.resolve() == eval_path.resolve():
        raise ValueError("train and eval paths must be different")
    _write_jsonl(train_path, dataset["train"])
    _write_jsonl(eval_path, dataset["eval"])
    summary = dict(dataset["summary"])
    summary["paths"] = {
        "train": str(train_path),
        "eval": str(eval_path),
        "manifest": str(directory / "manifest.json"),
    }
    manifest_path = directory / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(summary, indent=2) + "\n",
                             encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--out-dir", default="data/random_copy_v1",
        help="output directory (default: data/random_copy_v1)")
    parser.add_argument(
        "--rows", type=int, default=2000,
        help="number of training rows (default: 2000)")
    parser.add_argument(
        "--formats", default="json,kv",
        help="comma list of json,kv,yaml (default: json,kv)")
    parser.add_argument(
        "--keys-per-row", type=int, default=2,
        help="minimum key/value pairs per row; rows vary from this up to 4 "
             "(default: 2)")
    parser.add_argument("--seed", type=int, default=23,
                        help="deterministic RNG seed (default: 23)")
    parser.add_argument(
        "--holdout", default=None,
        help="path for eval rows (default: <out-dir>/eval.jsonl)")
    parser.add_argument(
        "--eval-rows", type=int, default=200,
        help="number of held-out eval rows (default: 200)")
    parser.add_argument("--shuffle", action="store_true",
                        help="deterministically shuffle train and eval rows")
    args = parser.parse_args(argv)
    try:
        dataset = build_dataset(
            rows=args.rows, eval_rows=args.eval_rows,
            formats=args.formats, keys_per_row=args.keys_per_row,
            seed=args.seed, shuffle=args.shuffle)
        summary = write_dataset(
            dataset, args.out_dir, holdout=args.holdout)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
