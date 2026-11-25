"""Build verified, canonical teacher-distillation data for copy tasks.

The 49M student cannot reliably copy three or four arbitrary key/value pairs.
A larger local teacher (Qwen2.5-0.5B-Instruct by default) can copy them much
more often, but its output is not automatically training data: it may wrap the
answer in a code fence, use a ``{"pairs": ...}`` shape, rename a key, drop a
pair, or otherwise differ from the requested copy.

This builder turns staged copy prompts into a teacher-distillation corpus:

1. read only ``train.jsonl`` source rows (never eval rows);
2. ask the teacher with the row's own prompt using greedy decoding;
3. parse the answer tolerantly (flat JSON, pairs lists, source/target dicts,
   code fences, KV or YAML);
4. require the parsed key/value pairs to equal the source pairs exactly,
   ignoring order;
5. canonicalize accepted answers to the row's requested format;
6. reject invalid teacher output, or optionally fall back to the verified
   source assistant with ``--fallback-oracle``.

The output is training data, not a model result. ``manifest.json`` contains
aggregate counts only: no prompts, answers, keys, values or other row text.
Generated data lives under the gitignored ``data/`` directory by default.

``build_distill`` accepts an injected teacher callable, so tests never load
torch or Qwen:

    rows, counts = build_distill(source_rows, lambda prompt: answer)

Command (host GPU, 2000 requested rows)::

    python research/build_distill_copy.py `
      --source data/multi_short_v1/stage_2short/train.jsonl `
               data/multi_short_v1/stage_3short/train.jsonl `
               data/multi_short_v1/stage_4short/train.jsonl `
      --out-dir data/distill_copy_v1 --rows 2000 --seed 23 --shuffle

See ``docs/DISTILL_COPY.md`` for the verification rule and limitations.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

FORMATS = ("json", "kv", "yaml")
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_SOURCES = (
    "data/multi_short_v1/stage_2short/train.jsonl",
    "data/multi_short_v1/stage_3short/train.jsonl",
    "data/multi_short_v1/stage_4short/train.jsonl",
)
DEFAULT_OUT_DIR = "data/distill_copy_v1"
DATASET_NAME = "distill_copy_v1"
VERIFICATION_METHOD = "teacher_exact_pair_match_v1"
MANIFEST_NOTE = (
    "Aggregate counts only: this manifest contains no prompts, answers, keys, "
    "values or source row text. Teacher-verified rows are distillation "
    "training data, not a claim about model capability."
)

FENCE_RE = re.compile(r"```[A-Za-z0-9_.+-]*\s*\n?(.*?)```", re.DOTALL)
PAIR_ITEM_KEY_ALIASES = ("source", "key", "k", "from")
PAIR_ITEM_VALUE_ALIASES = ("target", "value", "v", "to")
YAML_UNSAFE = set(":#{}[],&*!|>'\"%@`")


class ParseError(ValueError):
    """Raised when a teacher answer cannot be parsed into key/value pairs."""


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


def normalize_source_paths(sources: str | Iterable[str]) -> tuple[Path, ...]:
    """Normalize one or more source paths and reject eval files."""

    if isinstance(sources, (str, Path)):
        items = [sources]
    else:
        items = list(sources)
    if not items:
        raise ValueError("at least one source file is required")
    paths: list[Path] = []
    for item in items:
        path = Path(item).expanduser()
        _assert_train_source(path)
        paths.append(path)
    return tuple(paths)


def _assert_train_source(path: Path) -> None:
    """Refuse an eval source so held-out rows cannot leak into training."""

    parts = [part.lower() for part in path.parts]
    stem = path.stem.lower()
    if stem.startswith("eval") or "eval" in stem or "eval" in parts:
        raise ValueError(
            f"refusing eval source {path}: distillation sources must be "
            "train.jsonl files")


def _value_text(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    return str(value)


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


def _yaml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if (text and text == text.strip()
            and not any(char in YAML_UNSAFE for char in text)
            and text.lower() not in {
                "true", "false", "null", "~", "yes", "no", "on", "off"}
            and not re.fullmatch(r"[-+]?(?:\d+\.?\d*|\.\d+)", text)):
        return text
    return json.dumps(text, ensure_ascii=False)


def canonicalize(fmt: str, pairs: Sequence[Sequence[Any]]) -> str:
    """Render verified ``pairs`` in the canonical form for ``fmt``."""

    if fmt not in FORMATS:
        raise ValueError(f"unknown format: {fmt}")
    if fmt == "json":
        return json.dumps(
            {str(key): value for key, value in pairs},
            separators=(",", ":"), ensure_ascii=False)
    if fmt == "kv":
        return ";".join(
            f"{key}={_value_text(value)}" for key, value in pairs)
    return "\n".join(
        f"{key}: {_yaml_scalar(value)}" for key, value in pairs)


def _strip_code_fences(text: str) -> str:
    text = str(text or "").strip()
    if not text:
        return ""
    matches = [match.strip() for match in FENCE_RE.findall(text)]
    matches = [match for match in matches if match]
    if matches:
        return matches[0]
    lines = text.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return text


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


def _balanced_spans(text: str) -> Iterable[str]:
    """Yield complete top-level JSON-ish substrings found in ``text``.

    Nested fragments inside an incomplete object are not yielded. A teacher
    answer truncated mid-object therefore fails to parse instead of being
    mistaken for a complete smaller object.
    """

    depth = 0
    in_string = False
    escaped = False
    yielded = 0
    for index, current in enumerate(text):
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
            if depth == 0:
                span = _balanced_span(text, index)
                if span is not None:
                    yield span
                    yielded += 1
                    if yielded >= 8:
                        return
            depth += 1
        elif current in "]}":
            if depth:
                depth -= 1


def _pairs_from_pairs_value(value: Any) -> list[tuple[str, Any]]:
    if isinstance(value, Mapping):
        items = list(value.items())
    elif isinstance(value, (list, tuple)):
        items = []
        for item in value:
            if isinstance(item, Mapping):
                key = next(
                    (item[alias] for alias in PAIR_ITEM_KEY_ALIASES
                     if alias in item), None)
                val = next(
                    (item[alias] for alias in PAIR_ITEM_VALUE_ALIASES
                     if alias in item), None)
                if key is None or val is None:
                    # Some teachers emit {"pair1": key, "value1": value} or
                    # {"key1": key, "value1": value}. Accept any two-scalar
                    # object in insertion order; verification still decides.
                    if len(item) != 2:
                        raise ParseError(
                            "pair object must have source/target or key/value")
                    key, val = list(item.values())
                    if isinstance(key, (Mapping, list, tuple)) or isinstance(
                            val, (Mapping, list, tuple)):
                        raise ParseError("pair values must be scalar")
                items.append((key, val))
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                items.append((item[0], item[1]))
            else:
                raise ParseError("pairs must be two-item lists or dicts")
    else:
        raise ParseError("pairs must be a list or object")

    pairs: list[tuple[str, Any]] = []
    seen: set[str] = set()
    for key, value in items:
        key = str(key)
        if not key:
            raise ParseError("pair has an empty key")
        if key in seen:
            raise ParseError(f"duplicate key: {key}")
        if isinstance(value, (Mapping, list, tuple)):
            raise ParseError("pair values must be scalar")
        seen.add(key)
        pairs.append((key, value))
    if not pairs:
        raise ParseError("answer has no pairs")
    return pairs


def _pairs_from_json(data: Any) -> list[tuple[str, Any]]:
    if isinstance(data, Mapping):
        if "pairs" in data:
            return _pairs_from_pairs_value(data["pairs"])
        pairs: list[tuple[str, Any]] = []
        for key, value in data.items():
            if isinstance(value, (Mapping, list, tuple)):
                raise ParseError("flat JSON values must be scalar")
            pairs.append((str(key), value))
        if not pairs:
            raise ParseError("JSON object has no pairs")
        return pairs
    if isinstance(data, (list, tuple)):
        return _pairs_from_pairs_value(data)
    raise ParseError("JSON answer must be an object or a pairs list")


def _json_object_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        key = str(key)
        if key in result:
            raise ParseError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _try_json(text: str) -> list[tuple[str, Any]] | None:
    for candidate in _balanced_spans(text):
        try:
            data = json.loads(
                candidate, object_pairs_hook=_json_object_no_duplicates)
        except (json.JSONDecodeError, ParseError, ValueError):
            continue
        try:
            return _pairs_from_json(data)
        except ParseError:
            continue
    return None


def _parse_kv_or_yaml(text: str) -> list[tuple[str, Any]]:
    """Parse one-line/multiline KV or YAML lines.

    A chunk may use ``key=value`` (canonical KV) or ``key: value`` (YAML or a
    colon-separated teacher answer). Empty keys, empty values and prose chunks
    without a separator are skipped; duplicates fail the parse.
    """

    chunks = re.split(r"[;\n]+", text)
    pairs: list[tuple[str, Any]] = []
    seen: set[str] = set()
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk or chunk.startswith("#") or chunk in {"---", "..."}:
            continue
        if chunk.startswith("- "):
            chunk = chunk[2:].strip()
        if "=" in chunk:
            key, value = chunk.split("=", 1)
        elif ":" in chunk:
            key, value = chunk.split(":", 1)
        else:
            continue
        key = _unquote(key.strip())
        value = _unquote(value)
        if not key or not value:
            continue
        if key in seen:
            raise ParseError(f"duplicate key: {key}")
        seen.add(key)
        pairs.append((key, value))
    if not pairs:
        raise ParseError("answer has no KV or YAML pairs")
    return pairs


def parse_teacher_answer(answer: Any) -> list[tuple[str, Any]]:
    """Parse a teacher answer into ordered key/value pairs.

    Accepts flat JSON objects, ``{"pairs": [[key, value], ...]}``,
    ``{"pairs": [{"source": ..., "target": ...}, ...]}``, code fences,
    one-line or multiline KV, and YAML lines. Raises ``ParseError`` when no
    pair structure is found. Parsing is deliberately tolerant; the exact
    pair check happens separately in :func:`pairs_match`.
    """

    text = _strip_code_fences(answer)
    if not text:
        raise ParseError("empty teacher answer")
    pairs = _try_json(text)
    if pairs is not None:
        return pairs
    return _parse_kv_or_yaml(text)


def extract_expected_pairs(row: Mapping[str, Any]) -> list[tuple[str, Any]]:
    """Return the source row's expected pairs, validating its metadata."""

    raw_pairs = row.get("pairs")
    if raw_pairs is not None:
        if not isinstance(raw_pairs, (list, tuple)) or not raw_pairs:
            raise ValueError("source row pairs must be a non-empty list")
        pairs: list[tuple[str, Any]] = []
        for pair in raw_pairs:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError("source row pair must have two items")
            pairs.append((str(pair[0]), pair[1]))
    else:
        keys = row.get("keys")
        values = row.get("values")
        if not isinstance(keys, (list, tuple)) or not isinstance(
                values, (list, tuple)):
            raise ValueError("source row needs pairs or keys/values")
        if len(keys) != len(values) or not keys:
            raise ValueError("source row keys/values must be non-empty and "
                             "the same length")
        pairs = [(str(key), value) for key, value in zip(keys, values)]

    keys = [key for key, _ in pairs]
    if len(keys) != len(set(keys)):
        raise ValueError("source row has duplicate keys")
    if any(not key for key in keys):
        raise ValueError("source row has an empty key")

    row_keys = row.get("keys")
    row_values = row.get("values")
    if row_keys is not None and row_values is not None:
        if ([str(key) for key in row_keys] != keys
                or list(row_values) != [value for _, value in pairs]):
            raise ValueError("source row pairs disagree with keys/values")
    return pairs


def pairs_match(parsed: Sequence[Sequence[Any]],
                expected: Sequence[Sequence[Any]]) -> bool:
    """Return True iff both sides have the same keys and values, no order."""

    if len(parsed) != len(expected):
        return False
    parsed_map: dict[str, str] = {}
    for key, value in parsed:
        key = str(key)
        if key in parsed_map:
            return False
        parsed_map[key] = _value_text(value)
    expected_map = {str(key): _value_text(value) for key, value in expected}
    if set(parsed_map) != set(expected_map):
        return False
    return all(parsed_map[key] == expected_map[key] for key in expected_map)


def _stage_of(row: Mapping[str, Any]) -> str:
    stage = row.get("stage")
    if stage:
        return str(stage)
    family = str(row.get("family", ""))
    if family.startswith("multi_short_"):
        return family[len("multi_short_"):]
    return family or "unknown"


def _row_is_train(row: Mapping[str, Any]) -> bool:
    partition = row.get("partition")
    if partition is None:
        return True
    return str(partition).strip().lower() == "train"


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
                raise ValueError(f"{path}:{line_number}: row must be an object")
            rows.append(row)
    return rows


def load_source_rows(sources: str | Iterable[str], *,
                     formats: str | Iterable[str] = FORMATS,
                     rows: int | None = None,
                     seed: int = 23,
                     shuffle: bool = False) -> list[dict[str, Any]]:
    """Load train rows, filter formats, and take a deterministic sample.

    Without ``shuffle`` the rows are drawn round-robin from each source, so a
    small row budget covers every stage instead of exhausting the first file.
    With ``shuffle`` all eligible rows are shuffled with ``seed`` before the
    limit is applied.
    """

    paths = normalize_source_paths(sources)
    allowed = set(normalize_formats(formats))
    if rows is not None and rows < 0:
        raise ValueError("rows must be non-negative")

    groups: list[list[dict[str, Any]]] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"source file not found: {path}")
        group: list[dict[str, Any]] = []
        for row in _read_jsonl(path):
            if not _row_is_train(row):
                raise ValueError(
                    f"refusing non-train row in {path}: "
                    f"partition={row.get('partition')!r}")
            fmt = str(row.get("format", "")).lower()
            if fmt not in allowed:
                continue
            group.append(row)
        groups.append(group)

    if shuffle:
        combined = [row for group in groups for row in group]
        random.Random(seed).shuffle(combined)
        selected = combined
    else:
        selected = []
        indexes = [0] * len(groups)
        while True:
            added = False
            for index, group in enumerate(groups):
                if indexes[index] < len(group):
                    selected.append(group[indexes[index]])
                    indexes[index] += 1
                    added = True
            if not added:
                break
    if rows is not None:
        selected = selected[:rows]
    return selected


def build_distill(source_rows: Sequence[Mapping[str, Any]],
                  teacher: Callable[[str], str], *,
                  model: str = DEFAULT_MODEL,
                  fallback_oracle: bool = False,
                  progress: Callable[[Mapping[str, Any]], None] | None = None
                  ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate verified distillation rows with an injected teacher.

    ``teacher`` receives the source row's ``user`` prompt and returns raw
    text. Invalid teacher output is rejected, or replaced by the source
    assistant when ``fallback_oracle`` is true. Returns ``(rows, counts)``.
    """

    rows: list[dict[str, Any]] = []
    counts: dict[str, Any] = {
        "requested": len(source_rows),
        "generated": 0,
        "teacher_valid": 0,
        "fallback_oracle": 0,
        "rejected": 0,
        "by_format": {fmt: 0 for fmt in FORMATS},
        "by_stage": {},
    }
    by_stage: Counter[str] = Counter()

    for source_row in source_rows:
        fmt = str(source_row.get("format", "")).lower()
        if fmt not in FORMATS:
            raise ValueError(f"source row has unknown format: {fmt!r}")
        prompt = str(source_row.get("user", ""))
        if not prompt:
            raise ValueError("source row has no user prompt")
        expected = extract_expected_pairs(source_row)

        try:
            parsed = parse_teacher_answer(teacher(prompt))
        except ParseError:
            parsed = None
        verified = parsed is not None and pairs_match(parsed, expected)

        if verified:
            answer = canonicalize(fmt, expected)
            teacher_verified = True
            canonicalized = True
            counts["teacher_valid"] += 1
        elif fallback_oracle:
            answer = str(source_row.get("assistant", ""))
            try:
                fallback_pairs = parse_teacher_answer(answer)
            except ParseError:
                fallback_pairs = None
            if fallback_pairs is None or not pairs_match(
                    fallback_pairs, expected):
                counts["rejected"] += 1
                if progress is not None:
                    progress(dict(counts))
                continue
            teacher_verified = False
            canonicalized = False
            counts["fallback_oracle"] += 1
        else:
            counts["rejected"] += 1
            if progress is not None:
                progress(dict(counts))
            continue

        stage = _stage_of(source_row)
        checks = [
            "answer_parses",
            "keys_exactly_match",
            "values_exactly_match",
        ]
        if teacher_verified:
            checks.append("answer_canonicalized")
        else:
            checks.append("oracle_fallback_verified")
        row = {
            "user": prompt,
            "assistant": answer,
            "family": f"distill_copy_{stage}",
            "stage": stage,
            "format": fmt,
            "partition": "train",
            "keys": [str(key) for key, _ in expected],
            "values": [value for _, value in expected],
            "source_family": str(source_row.get("family", "")),
            "teacher_model": str(model),
            "teacher_verified": teacher_verified,
            "canonicalized": canonicalized,
            "verification": {
                "verified": True,
                "status": "teacher_verified" if teacher_verified
                          else "oracle_fallback",
                "method": VERIFICATION_METHOD,
                "checks": checks,
            },
        }
        rows.append(row)
        counts["generated"] += 1
        counts["by_format"][fmt] = counts["by_format"].get(fmt, 0) + 1
        by_stage[stage] += 1
        if progress is not None:
            progress(dict(counts))

    counts["by_stage"] = dict(sorted(by_stage.items()))
    return rows, counts


def build_manifest(counts: Mapping[str, Any], *,
                   model: str = DEFAULT_MODEL,
                   seed: int = 23,
                   sources: Sequence[str | Path] = (),
                   formats: str | Iterable[str] = FORMATS,
                   fallback_oracle: bool = False,
                   local_files_only: bool = True,
                   max_new_tokens: int = 96) -> dict[str, Any]:
    """Build the content-free aggregate manifest."""

    manifest = {
        "dataset": DATASET_NAME,
        "teacher_model": str(model),
        "verification_method": VERIFICATION_METHOD,
        "seed": int(seed),
        "source_files": [str(Path(path)) for path in sources],
        "formats": list(normalize_formats(formats)),
        "fallback_oracle_enabled": bool(fallback_oracle),
        "local_files_only": bool(local_files_only),
        "max_new_tokens": int(max_new_tokens),
        "requested": int(counts.get("requested", 0)),
        "generated": int(counts.get("generated", 0)),
        "teacher_valid": int(counts.get("teacher_valid", 0)),
        "fallback_oracle": int(counts.get("fallback_oracle", 0)),
        "rejected": int(counts.get("rejected", 0)),
        "by_format": {
            fmt: int(counts.get("by_format", {}).get(fmt, 0))
            for fmt in FORMATS
        },
        "by_stage": {
            str(stage): int(count)
            for stage, count in sorted(
                counts.get("by_stage", {}).items())
        },
        "note": MANIFEST_NOTE,
    }
    return manifest


def write_dataset(rows: Sequence[Mapping[str, Any]],
                  manifest: Mapping[str, Any],
                  out_dir: str | Path) -> dict[str, str]:
    """Write ``train_distill.jsonl`` and ``manifest.json``."""

    root = Path(out_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    train_path = root / "train_distill.jsonl"
    manifest_path = root / "manifest.json"
    body = "".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    train_path.write_text(body, encoding="utf-8")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    return {"train": str(train_path), "manifest": str(manifest_path)}


def load_teacher(model_name: str = DEFAULT_MODEL, *,
                 device: str = "cuda",
                 max_new_tokens: int = 96,
                 local_files_only: bool = True
                 ) -> Callable[[str], str]:
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

    def teacher(prompt: str) -> str:
        messages = [{"role": "user", "content": str(prompt)}]
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


def _progress_printer(interval: int = 50) -> Callable[[Mapping[str, Any]], None]:
    seen = 0

    def report(counts: Mapping[str, Any]) -> None:
        nonlocal seen
        seen += 1
        if seen % interval:
            return
        print(json.dumps({
            "generated": counts.get("generated", 0),
            "teacher_valid": counts.get("teacher_valid", 0),
            "fallback_oracle": counts.get("fallback_oracle", 0),
            "rejected": counts.get("rejected", 0),
        }), file=sys.stderr, flush=True)

    return report


def main(argv: Sequence[str] | None = None, *,
         teacher_factory: Callable[..., Callable[[str], str]] | None = None
         ) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--source", nargs="+", default=list(DEFAULT_SOURCES),
        help="one or more staged train JSONL files "
             "(default: stage_2short/3short/4short train.jsonl)")
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
        help="maximum source rows to attempt (default: 2000)")
    parser.add_argument(
        "--seed", type=int, default=23,
        help="deterministic sampling seed (default: 23)")
    parser.add_argument(
        "--formats", default=",".join(FORMATS),
        help="source formats to keep, comma separated "
             f"(default: {','.join(FORMATS)})")
    parser.add_argument(
        "--max-new-tokens", type=int, default=96,
        help="teacher generation budget (default: 96)")
    parser.add_argument(
        "--local-files-only", action=argparse.BooleanOptionalAction,
        default=True,
        help="do not download the teacher (default: true)")
    parser.add_argument(
        "--fallback-oracle", action="store_true",
        help="write the verified source assistant when the teacher fails")
    parser.add_argument(
        "--shuffle", action="store_true",
        help="shuffle eligible source rows with --seed before limiting")
    args = parser.parse_args(argv)

    try:
        formats = normalize_formats(args.formats)
    except ValueError as exc:
        parser.error(str(exc))
    if args.rows < 0:
        parser.error("--rows must be non-negative")

    try:
        source_rows = load_source_rows(
            args.source, formats=formats, rows=args.rows, seed=args.seed,
            shuffle=args.shuffle)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))

    if teacher_factory is None:
        teacher = load_teacher(
            args.model, device=args.device,
            max_new_tokens=args.max_new_tokens,
            local_files_only=args.local_files_only)
    else:
        teacher = teacher_factory(
            args.model, device=args.device,
            max_new_tokens=args.max_new_tokens,
            local_files_only=args.local_files_only)

    rows, counts = build_distill(
        source_rows, teacher, model=args.model,
        fallback_oracle=args.fallback_oracle,
        progress=_progress_printer())
    manifest = build_manifest(
        counts, model=args.model, seed=args.seed, sources=args.source,
        formats=formats, fallback_oracle=args.fallback_oracle,
        local_files_only=args.local_files_only,
        max_new_tokens=args.max_new_tokens)
    paths = write_dataset(rows, manifest, args.out_dir)
    print(json.dumps(
        {**manifest, "paths": paths}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
