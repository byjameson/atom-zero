"""Reproducible large-base capacity probe for the multi-short copy task.

This harness measures an external, pretrained Hugging Face causal LM on the
same staged eval JSONL used for the 49.2M Atom Zero model. It is a research
probe, not a product model and not a training result. Model loading is
injectable so the parser and metrics can be tested without downloading or
loading weights.

Default model (local cache; ``--local-files-only`` defaults to true)::

    Qwen/Qwen2.5-0.5B-Instruct

Default dataset (gitignored)::

    data/multi_short_v1/stage_4short/eval.jsonl

The model is prompted through its chat template and generated greedily. Output
parsing is intentionally tolerant because the probe model often wraps the
answer in markdown fences or emits ``{"pairs": [...]}`` instead of a flat JSON
object. The summary is content-free: it contains counts, rates and parser
categories only, never prompts, keys, values or generations.

Usage::

    python -m research.eval_large_base --rows 20
    python -m research.eval_large_base \
        --data data/multi_short_v1/stage_2short/eval.jsonl \
        --rows 20 --flat-prompt \
        --out research/results/large_base_2short_flat.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from atomzero.copy_tool import extract_pairs

METHOD = "large_base_capacity_eval_v1"
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_DATA = Path("data/multi_short_v1/stage_4short/eval.jsonl")
FLAT_INSTRUCTION = (
    "Return only one flat compact JSON object that maps each key directly to "
    'its value. Do not use a "pairs" list, do not use markdown code fences, '
    "and do not add prose."
)

COMPACT_KEYS = (
    "method",
    "model",
    "dataset",
    "flat_prompt",
    "rows_total",
    "rows_evaluated",
    "rows_exact",
    "exact_match",
    "key_recall",
    "value_recall",
    "parse_failures",
    "parse_failure_rate",
    "rows_flat_json",
    "flat_format_rate",
    "content_free",
)


class RowError(ValueError):
    """An eval row cannot be evaluated. The reason is content-free."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class ParsedPairs:
    """A tolerant parse of one generation."""

    pairs: dict[str, str]
    kind: str
    fenced: bool


def _scalar_text(value: Any, where: str) -> str:
    if isinstance(value, str):
        if not value:
            raise ValueError(f"{where}: empty value")
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    raise ValueError(f"{where}: value must be a JSON scalar")


def _pairs_from_sequence(
    items: Sequence[Any],
) -> tuple[dict[str, str], str]:
    if not isinstance(items, Sequence) or not items:
        raise ValueError("pairs must be a non-empty list")
    result: dict[str, str] = {}
    kinds: set[str] = set()
    for index, item in enumerate(items, start=1):
        if isinstance(item, Mapping):
            source = item.get("source", item.get("key", item.get("k")))
            target = item.get("target", item.get("value", item.get("v")))
            if source is None or target is None:
                raise ValueError(
                    f"pair {index}: expected source and target")
            kinds.add("objects")
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            source, target = item
            kinds.add("lists")
        else:
            raise ValueError(f"pair {index}: expected a two-item pair")
        key = str(source).strip()
        if not key:
            raise ValueError(f"pair {index}: empty key")
        if key in result:
            raise ValueError(f"pair {index}: duplicate key")
        result[key] = _scalar_text(target, f"pair {index}")
    if kinds == {"objects"}:
        kind = "pairs_object_json"
    elif kinds == {"lists"}:
        kind = "pairs_list_json"
    else:
        kind = "pairs_mixed_json"
    return result, kind


def _pairs_from_payload(payload: Any) -> tuple[dict[str, str], str]:
    if isinstance(payload, Mapping):
        raw_pairs = payload.get("pairs")
        if isinstance(raw_pairs, list):
            return _pairs_from_sequence(raw_pairs)
        if "pairs" in payload:
            raise ValueError("pairs must be a list")
        result: dict[str, str] = {}
        for raw_key, raw_value in payload.items():
            key = str(raw_key).strip()
            if not key:
                raise ValueError("empty key")
            if key in result:
                raise ValueError("duplicate key")
            result[key] = _scalar_text(raw_value, "flat object")
        if not result:
            raise ValueError("empty JSON object")
        return result, "flat_json"
    if isinstance(payload, list):
        return _pairs_from_sequence(payload)
    raise ValueError("JSON output must be an object")


def _fenced_blocks(text: str) -> list[str]:
    """Return the contents of triple-backtick blocks, without language tags."""
    blocks: list[str] = []
    index = 0
    while True:
        start = text.find("```", index)
        if start < 0:
            break
        end = text.find("```", start + 3)
        if end < 0:
            break
        body = text[start + 3:end]
        newline = body.find("\n")
        first_line = body[:newline].strip().lower() if newline >= 0 else ""
        if newline >= 0 and first_line in {"json", "kv", "yaml", "yml", ""}:
            body = body[newline + 1:]
        if body.strip():
            blocks.append(body.strip())
        index = end + 3
    return blocks


def _json_objects(text: str) -> list[str]:
    """Return balanced top-level ``{...}`` substrings, in order."""
    objects: list[str] = []
    search_from = 0
    while True:
        start = text.find("{", search_from)
        if start < 0:
            break
        depth = 0
        in_string = False
        escaped = False
        end = -1
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
            elif current == "{":
                depth += 1
            elif current == "}":
                depth -= 1
                if depth == 0:
                    end = index
                    break
        if end < 0:
            search_from = start + 1
            continue
        objects.append(text[start:end + 1])
        search_from = end + 1
    return objects


def _try_json(candidate: str) -> tuple[dict[str, str], str] | None:
    for raw in _json_objects(candidate):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        try:
            return _pairs_from_payload(payload)
        except ValueError:
            continue
    stripped = candidate.strip()
    if stripped.startswith("["):
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            payload = None
        if payload is not None:
            try:
                return _pairs_from_payload(payload)
            except ValueError:
                pass
    return None


def parse_model_output(text: str) -> ParsedPairs:
    """Parse a generation into ``(key, value)`` pairs, tolerantly.

    Accepted forms, optionally inside markdown fences and surrounded by prose:

    - a flat JSON object: ``{"a": "1", "b": "2"}``;
    - ``{"pairs": [["a", "1"], ["b", "2"]]}``;
    - ``{"pairs": [{"source": "a", "target": "1"}, ...]}`` (also
      ``key``/``value`` and ``k``/``v``);
    - a bare pair list;
    - one-line KV or YAML, parsed with
      :func:`atomzero.copy_tool.extract_pairs`.

    Raises ``ValueError`` when no non-empty mapping can be recovered.
    """
    if not isinstance(text, str):
        raise ValueError("model output must be a string")
    stripped = text.strip()
    if not stripped:
        raise ValueError("empty model output")
    blocks = _fenced_blocks(stripped)
    candidates = list(blocks)
    if stripped not in candidates:
        candidates.append(stripped)
    fenced = bool(blocks)

    for candidate in candidates:
        parsed = _try_json(candidate)
        if parsed is not None:
            pairs, kind = parsed
            return ParsedPairs(pairs=pairs, kind=kind, fenced=fenced)

    for candidate in candidates:
        if "{" in candidate or "}" in candidate:
            continue
        try:
            pairs = dict(extract_pairs(candidate))
        except ValueError:
            continue
        if pairs:
            return ParsedPairs(
                pairs=pairs, kind="kv_yaml", fenced=fenced)
    raise ValueError("could not parse model output as JSON, KV or YAML")


def is_flat_json_output(text: str) -> bool:
    """Return true when the raw generation is exactly one flat JSON object."""
    if not isinstance(text, str):
        return False
    stripped = text.strip()
    if not stripped or "```" in stripped:
        return False
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return False
    if not isinstance(payload, dict) or not payload:
        return False
    if isinstance(payload.get("pairs"), list):
        return False
    for value in payload.values():
        if isinstance(value, (dict, list)) or value is None:
            return False
    return True


def _row_prompt_and_pairs(row: Any) -> tuple[str, dict[str, str]]:
    if not isinstance(row, Mapping):
        raise RowError("invalid_row")
    prompt = row.get("user")
    if not isinstance(prompt, str) or not prompt.strip():
        raise RowError("missing_prompt")
    raw_pairs = row.get("pairs")
    if raw_pairs is None:
        keys = row.get("keys")
        values = row.get("values")
        if (not isinstance(keys, list) or not isinstance(values, list)
                or len(keys) != len(values) or not keys):
            raise RowError("missing_pairs")
        raw_pairs = list(zip(keys, values))
    try:
        pairs, _ = _pairs_from_sequence(raw_pairs)
    except ValueError:
        raise RowError("invalid_pairs") from None
    return prompt, pairs


def _rate(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round(count / total, 6)


def evaluate_rows(
    rows: Sequence[Any],
    generate: Callable[[str], str],
    *,
    model: str = DEFAULT_MODEL,
    dataset: str = "<memory>",
    flat_prompt: bool = False,
    local_files_only: bool = True,
    device: str = "cpu",
    max_new_tokens: int = 64,
) -> dict[str, Any]:
    """Evaluate rows with ``generate(user_text) -> completion``.

    ``exact_match`` is dict equality after tolerant parsing. ``key_recall`` is
    the mean fraction of expected keys present; ``value_recall`` is the mean
    fraction of expected key/value pairs copied exactly. ``flat_format_rate``
    counts raw generations that are a single flat JSON object with no fences.
    Invalid rows are excluded from rates and reported separately.
    """
    exact = 0
    parse_failures = 0
    flat_json = 0
    fenced = 0
    evaluated = 0
    key_recall_total = 0.0
    value_recall_total = 0.0
    parse_kinds: Counter[str] = Counter()
    invalid_reasons: Counter[str] = Counter()

    for row in rows:
        try:
            prompt, expected = _row_prompt_and_pairs(row)
        except RowError as exc:
            invalid_reasons[exc.reason] += 1
            continue
        output = generate(prompt)
        evaluated += 1
        if is_flat_json_output(output):
            flat_json += 1
        if isinstance(output, str) and "```" in output:
            fenced += 1

        parsed: ParsedPairs | None
        try:
            parsed = parse_model_output(output)
        except ValueError:
            parsed = None
            parse_failures += 1

        if parsed is None:
            continue
        parse_kinds[parsed.kind] += 1
        if parsed.pairs == expected:
            exact += 1
        key_hits = sum(
            1 for key in expected if key in parsed.pairs)
        value_hits = sum(
            1 for key, value in expected.items()
            if parsed.pairs.get(key) == value)
        total = len(expected)
        key_recall_total += key_hits / total
        value_recall_total += value_hits / total

    return {
        "method": METHOD,
        "model": model,
        "dataset": str(dataset),
        "flat_prompt": bool(flat_prompt),
        "local_files_only": bool(local_files_only),
        "device": device,
        "max_new_tokens": max_new_tokens,
        "rows_total": len(rows),
        "rows_evaluated": evaluated,
        "rows_invalid": len(rows) - evaluated,
        "rows_exact": exact,
        "exact_match": _rate(exact, evaluated),
        "key_recall": round(key_recall_total / evaluated, 6)
        if evaluated else 0.0,
        "value_recall": round(value_recall_total / evaluated, 6)
        if evaluated else 0.0,
        "parse_failures": parse_failures,
        "parse_failure_rate": _rate(parse_failures, evaluated),
        "rows_flat_json": flat_json,
        "flat_format_rate": _rate(flat_json, evaluated),
        "rows_with_code_fence": fenced,
        "parse_kinds": {
            name: parse_kinds.get(name, 0)
            for name in sorted(parse_kinds)
        },
        "invalid_reasons": {
            name: invalid_reasons.get(name, 0)
            for name in sorted(invalid_reasons)
        },
        "content_free": True,
        "prompt_text_included": False,
        "weights_changed": False,
    }


def load_rows(path: str | Path) -> list[Any]:
    """Read JSONL rows. Malformed lines become invalid rows, not crashes."""
    rows: list[Any] = []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                rows.append(None)
    return rows


def build_chat_prompt(
    tokenizer: Any, user_text: str, *, flat_prompt: bool = False,
) -> str:
    """Apply the tokenizer's chat template to one user message."""
    content = user_text
    if flat_prompt:
        content = f"{user_text}\n\n{FLAT_INSTRUCTION}"
    messages = [{"role": "user", "content": content}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    if not isinstance(prompt, str):
        raise ValueError("tokenizer chat template did not return text")
    return prompt


def resolve_device(device: str) -> str:
    """Resolve ``auto`` to CUDA when available, otherwise CPU."""
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model_and_tokenizer(
    model_name: str = DEFAULT_MODEL,
    *,
    device: str = "auto",
    local_files_only: bool = True,
) -> tuple[Any, Any]:
    """Load a causal LM and tokenizer from the local Hugging Face cache."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved = resolve_device(device)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, local_files_only=local_files_only)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, local_files_only=local_files_only)
    model.to(resolved)
    model.eval()
    return model, tokenizer


def generate_completion(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    max_new_tokens: int = 64,
    device: str = "cpu",
) -> str:
    """Generate greedily and decode only the new tokens."""
    import torch

    encoded = tokenizer(prompt, return_tensors="pt")
    moved = {
        name: tensor.to(device) for name, tensor in encoded.items()
    }
    kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
    }
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if pad_token_id is not None:
        kwargs["pad_token_id"] = pad_token_id
    if eos_token_id is not None:
        kwargs["eos_token_id"] = eos_token_id
    with torch.inference_mode():
        output = model.generate(**moved, **kwargs)
    prompt_length = moved["input_ids"].shape[-1]
    new_tokens = output[0][prompt_length:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def run_evaluation(
    data: str | Path = DEFAULT_DATA,
    *,
    model_name: str = DEFAULT_MODEL,
    rows: int | None = None,
    max_new_tokens: int = 64,
    device: str = "auto",
    flat_prompt: bool = False,
    local_files_only: bool = True,
    loader: Callable[..., tuple[Any, Any]] | None = None,
    generator: Callable[..., str] | None = None,
) -> dict[str, Any]:
    """Load the model, evaluate the staged JSONL and return a summary.

    ``loader`` and ``generator`` are injectable for tests. The loader is called
    as ``loader(model_name, device=..., local_files_only=...)`` and must return
    ``(model, tokenizer)``. The generator is called as
    ``generator(model, tokenizer, prompt, max_new_tokens=..., device=...)``.
    """
    if rows is not None and rows < 0:
        raise ValueError("rows must be non-negative")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be at least 1")

    path = Path(data)
    loaded_rows = load_rows(path)
    if rows is not None:
        loaded_rows = loaded_rows[:rows]
    resolved_device = resolve_device(device)
    load = loader or load_model_and_tokenizer
    model, tokenizer = load(
        model_name,
        device=resolved_device,
        local_files_only=local_files_only,
    )
    generate_text = generator or generate_completion

    def generate(user_text: str) -> str:
        prompt = build_chat_prompt(
            tokenizer, user_text, flat_prompt=flat_prompt)
        return generate_text(
            model,
            tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            device=resolved_device,
        )

    return evaluate_rows(
        loaded_rows,
        generate,
        model=model_name,
        dataset=str(path),
        flat_prompt=flat_prompt,
        local_files_only=local_files_only,
        device=resolved_device,
        max_new_tokens=max_new_tokens,
    )


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    if target.parent and not target.parent.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument(
        "--rows", type=int, default=None,
        help="evaluate only the first N rows (default: all)")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--flat-prompt", action="store_true",
        help="append a flat-JSON-only instruction to every prompt")
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="never download; read the model from the local cache "
             "(default: true)",
    )
    parser.add_argument(
        "--out", default=None,
        help="write the full content-free summary JSON to this path")
    parser.add_argument(
        "--result", default=None,
        help="write a compact content-free result JSON to this path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.rows is not None and args.rows < 0:
        parser.error("--rows must be non-negative")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be at least 1")
    if args.out and args.result and Path(args.out) == Path(args.result):
        parser.error("--out and --result must be different files")

    try:
        summary = run_evaluation(
            args.data,
            model_name=args.model,
            rows=args.rows,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
            flat_prompt=args.flat_prompt,
            local_files_only=args.local_files_only,
        )
    except (OSError, ValueError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.out:
        _write_json(args.out, summary)
    if args.result:
        compact = {key: summary[key] for key in COMPACT_KEYS}
        _write_json(args.result, compact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
