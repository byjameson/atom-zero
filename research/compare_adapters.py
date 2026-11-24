"""Compare the base model and LoRA adapters on reviewed examples.

    python research/compare_adapters.py \\
        --base checkpoints/transformer_behavior_50m_long \\
        --data data/feedback_training.jsonl \\
        --adapters checkpoints/adapters_feedback/v0001 \\
                   checkpoints/adapters_feedback/v0002

The score is assistant-turn cross-entropy: user turns are masked, the
``<|assistant|>`` tag, the answer and ``<|end|>`` are scored. Lower is better.
This is a comparison tool, not a general benchmark.

The adapter can be produced by an external trainer. This script only needs the
local ``atomzero`` runtime; it does not import ``streamft``.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path

try:
    from research.adapter_runtime import (
        assistant_label_mask, fmt, load_model, read_rows, score_rows)
except ImportError:  # running as `python research/compare_adapters.py`
    from adapter_runtime import (
        assistant_label_mask, fmt, load_model, read_rows, score_rows)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXACT_MATCH_MAX_NEW = 64
_KV_LINE = re.compile(r"^\s*-?\s*[\w.]+\s*:")
_KV_SPACING = re.compile(r"\s*([:=,\[\]{}])\s*")


def label_for(path: str) -> str:
    version = Path(path).name
    parent = Path(path).parent.name
    if parent and version.startswith("v") and version[1:].isdigit():
        return f"{parent}/{version}"
    return version or path


def _family_of(row) -> str | None:
    if not isinstance(row, dict):
        return None
    family = row.get("family")
    if not isinstance(family, str):
        return None
    family = family.strip()
    return family or None


def _aggregate_rows(rows: list[dict]) -> dict | None:
    tokens = 0
    nll = 0.0
    content_tokens = 0
    content_nll = 0.0
    for row in rows:
        count = int(row.get("tokens") or 0)
        loss = row.get("loss")
        if isinstance(loss, bool) or not isinstance(loss, (int, float)):
            continue
        tokens += count
        nll += float(loss) * count
        content_count = int(row.get("content_tokens") or 0)
        content_loss = row.get("content_loss")
        if content_count and isinstance(content_loss, (int, float)) and not (
                isinstance(content_loss, bool)):
            content_tokens += content_count
            content_nll += float(content_loss) * content_count
    if tokens == 0:
        return None
    loss = nll / tokens
    content_loss = content_nll / content_tokens if content_tokens else None
    return {
        "examples": len(rows),
        "tokens": tokens,
        "loss": loss,
        "ppl": math.exp(min(loss, 20.0)),
        "content_loss": content_loss,
        "content_tokens": content_tokens,
    }


def per_family_metrics(metrics: dict) -> dict:
    """Aggregate scored rows by their optional ``family`` label."""
    rows = metrics.get("rows") if isinstance(metrics, dict) else None
    if not isinstance(rows, list):
        return {}
    groups: dict[str, list[dict]] = {}
    for row in rows:
        family = _family_of(row)
        if family is None:
            continue
        groups.setdefault(family, []).append(row)
    summaries = {}
    for family in sorted(groups):
        summary = _aggregate_rows(groups[family])
        if summary is not None:
            summaries[family] = summary
    return summaries


def per_family_summary(baseline: dict, adapter: dict) -> dict:
    """Base-vs-adapter metrics for every labelled family."""
    base_families = per_family_metrics(baseline)
    adapter_families = per_family_metrics(adapter)
    families = {}
    for family in sorted(set(base_families) | set(adapter_families)):
        families[family] = {
            "baseline": base_families.get(family),
            "adapter": adapter_families.get(family),
        }
    return families


def generation_prompt(row: dict) -> str:
    """Prompt shape used by evaluation, without the assistant answer."""
    return f"<|user|> {row['user']} <|assistant|>"


def _json_dict(text: str) -> dict | None:
    try:
        value = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _looks_like_json(text: str) -> bool:
    value = str(text or "").strip()
    return value.startswith("{") and value.endswith("}")


def is_key_value_answer(text: str) -> bool:
    """True when every non-empty line looks like ``key: value`` / YAML."""
    lines = [line for line in str(text or "").splitlines() if line.strip()]
    return bool(lines) and all(_KV_LINE.match(line) for line in lines)


def normalize_answer(text: str) -> str:
    """Whitespace-insensitive answer form for exact-match comparison."""
    if not isinstance(text, str):
        return ""
    stripped = text.strip()
    if _looks_like_json(stripped):
        parsed = _json_dict(stripped)
        if parsed is not None:
            return json.dumps(parsed, sort_keys=True, separators=(",", ":"),
                              ensure_ascii=False)
    if is_key_value_answer(stripped):
        lines = []
        for line in stripped.splitlines():
            line = re.sub(r"\s+", " ", line).strip()
            if not line:
                continue
            lines.append(_KV_SPACING.sub(r"\1", line))
        return "\n".join(lines)
    return re.sub(r"\s+", " ", stripped)


def answers_match(expected, generated) -> bool:
    """Compare a generated answer with the reviewed assistant answer."""
    expected_text = str(expected or "").strip()
    generated_text = str(generated or "").strip()
    if not expected_text or not generated_text:
        return False
    if _looks_like_json(expected_text):
        expected_json = _json_dict(expected_text)
        if expected_json is not None:
            return _json_dict(generated_text) == expected_json
    return normalize_answer(expected_text) == normalize_answer(generated_text)


def _default_generator(model, tok, prompt: str, max_new: int) -> str:
    from atomzero.transformer import generate

    text, _stop = generate(
        model, tok, prompt, max_new=max_new, temperature=0.0, top_k=1,
        repetition_penalty=1.0)
    return text


def _generation_limit(tok, row: dict, cap: int) -> int:
    cap = max(1, int(cap))
    try:
        expected = len(tok.encode(str(row.get("assistant", ""))).ids)
    except (AttributeError, TypeError, ValueError):
        expected = cap
    return max(4, min(cap, expected + 4))


def exact_match_metrics(model, tok, rows: list[dict], *, generator=None,
                        max_new: int = DEFAULT_EXACT_MATCH_MAX_NEW) -> dict:
    """Generate each answer and report normalized exact-match accuracy.

    ``generator`` is injectable (``generator(model, tok, prompt, max_new)``)
    so tests never need a real checkpoint.
    """
    generate_fn = generator or _default_generator
    cap = max(1, int(max_new))
    examples = 0
    matches = 0
    family_counts: dict[str, dict] = {}
    for row in rows:
        if not isinstance(row, dict) or not row.get("user"):
            continue
        examples += 1
        limit = _generation_limit(tok, row, cap)
        text = generate_fn(model, tok, generation_prompt(row), limit)
        matched = answers_match(row.get("assistant", ""), text)
        matches += int(matched)
        family = _family_of(row)
        if family is not None:
            bucket = family_counts.setdefault(
                family, {"examples": 0, "matches": 0})
            bucket["examples"] += 1
            bucket["matches"] += int(matched)
    families = {}
    for family in sorted(family_counts):
        bucket = family_counts[family]
        families[family] = {
            "examples": bucket["examples"],
            "matches": bucket["matches"],
            "rate": (bucket["matches"] / bucket["examples"]
                     if bucket["examples"] else None),
        }
    return {
        "examples": examples,
        "matches": matches,
        "rate": matches / examples if examples else None,
        "max_new": cap,
        "families": families,
    }


def compare(base: str, data: str | Path, adapters: list[str],
            tokenizer: str = "", per_family: bool = False,
            exact_match: bool = False, generator=None,
            exact_match_max_new: int = DEFAULT_EXACT_MATCH_MAX_NEW) -> dict:
    rows = read_rows(data)
    base_model, tok = load_model(base, tokenizer=tokenizer)
    models = [{"label": "base", "path": base,
               "metrics": score_rows(base_model, tok, rows)}]
    loaded = [(models[0], base_model)]
    for adapter in adapters:
        model, _ = load_model(base, adapter, tokenizer=tokenizer)
        item = {"label": label_for(adapter), "path": adapter,
                "metrics": score_rows(model, tok, rows)}
        models.append(item)
        loaded.append((item, model))
    reference = models[0]["metrics"]
    for item in models[1:]:
        metrics = item["metrics"]
        item["vs_base"] = {
            "loss_delta": metrics["loss"] - reference["loss"],
            "loss_relative": metrics["loss"] / reference["loss"] - 1,
            "ppl_ratio": metrics["ppl"] / reference["ppl"] - 1,
        }
    if exact_match:
        for item, model in loaded:
            try:
                item["metrics"]["exact_match"] = exact_match_metrics(
                    model, tok, rows, generator=generator,
                    max_new=exact_match_max_new)
            except (AttributeError, IndexError, KeyError, RuntimeError,
                    TypeError, ValueError) as exc:
                item["metrics"]["exact_match"] = {
                    "available": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
    payload = {
        "base": base,
        "data": str(data),
        "examples": len(rows),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "models": models,
    }
    if per_family and len(models) > 1:
        for item in models[1:]:
            item["families"] = per_family_summary(reference, item["metrics"])
        if len(models) == 2:
            payload["families"] = models[1]["families"]
    return payload


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base",
                    default="checkpoints/transformer_behavior_50m_long")
    ap.add_argument("--data", default="data/feedback_training.jsonl")
    ap.add_argument("--adapters", nargs="*", default=[])
    ap.add_argument("--tokenizer", default="",
                    help="tokenizer path; defaults to the base directory or "
                         "data/chat_en_bpe8k.json")
    ap.add_argument("--out", default="research/results/adapter_comparison.json")
    ap.add_argument("--per-family", action="store_true",
                    help="report loss/ppl/content_loss per family label")
    ap.add_argument("--exact-match", action="store_true",
                    help="generate answers and report normalized exact match")
    ap.add_argument("--exact-match-max-new", type=int,
                    default=DEFAULT_EXACT_MATCH_MAX_NEW,
                    help="cap on generated answer tokens (default: 64)")
    args = ap.parse_args(argv)
    payload = compare(
        args.base, args.data, args.adapters, args.tokenizer,
        per_family=args.per_family, exact_match=args.exact_match,
        exact_match_max_new=args.exact_match_max_new)
    if args.out:
        out = ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n",
                       encoding="utf-8")
    summary = []
    for item in payload["models"]:
        entry = {
            "label": item["label"],
            "path": item["path"],
            "loss": round(item["metrics"]["loss"], 6),
            "ppl": round(item["metrics"]["ppl"], 4),
            "tokens": item["metrics"]["tokens"],
            **{k: round(v, 6) for k, v in item.get("vs_base", {}).items()},
        }
        exact = item["metrics"].get("exact_match")
        if isinstance(exact, dict) and exact.get("rate") is not None:
            entry["exact_match"] = round(exact["rate"], 6)
        summary.append(entry)
    print(json.dumps({"examples": payload["examples"], "models": summary},
                     indent=2, ensure_ascii=False))
    if args.out:
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
