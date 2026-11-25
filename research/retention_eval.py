"""Measure whether a candidate checkpoint retains content behaviour.

The 0.80.0 63M copy checkpoint was fully fine-tuned on copy data. It can beat
the product 49M ``0.8.0`` base on copy exact match while losing the assistant's
content/QA behaviour. This harness compares a candidate with a baseline on the
same JSONL rows and reports four things:

- assistant-turn token loss;
- content loss on non-syntax (alphanumeric) tokens;
- normalized exact match from greedy generation;
- key-name copy rate and key recall on rows whose answer has keys.

The headline number is::

    retention_ratio = candidate_content_loss / baseline_content_loss

``retention_ok`` is true when the ratio is at or below ``--threshold`` (default
1.10, i.e. no worse than 10%). A ratio above the threshold means the candidate
models answer content less well than the product base even if it copies better.

The JSON report is content-free: counts, losses, rates and family names only.
Prompts, answers, keys and values are never written to the report. Model
loading and generation are injectable so tests run on CPU with fake models and
never load a checkpoint.

Usage::

    python research/retention_eval.py \
        --baseline ~/.cache/atomzero/atom-zero-49m \
        --candidate checkpoints/fullft_upcycle8l_v1 \
        --data data/eval_frozen/heldout_city_yaml.jsonl \
               data/eval_frozen/heldout_city_river.jsonl \
               data/eval_frozen/heldout_arithmetic.jsonl \
        --max-rows 200 --device auto --per-family \
        --out research/results/retention_49m_vs_63m.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

try:
    from research.adapter_runtime import (
        DEFAULT_TOKENIZER, load_model, read_rows, resolve_path, score_rows)
    from research.compare_adapters import (
        exact_match_metrics, key_copy_metrics, per_family_metrics)
except ImportError:  # running as `python research/retention_eval.py`
    from adapter_runtime import (
        DEFAULT_TOKENIZER, load_model, read_rows, resolve_path, score_rows)
    from compare_adapters import (
        exact_match_metrics, key_copy_metrics, per_family_metrics)

SCHEMA = "atomzero.retention_eval.v1"
DEFAULT_MAX_ROWS = 200
DEFAULT_THRESHOLD = 1.10
DEFAULT_EXACT_MATCH_MAX_NEW = 64
LOSS_FIELDS = (
    "examples",
    "tokens",
    "loss",
    "ppl",
    "content_loss",
    "content_tokens",
    "key_loss",
    "key_tokens",
    "value_loss",
    "value_tokens",
    "syntax_loss",
    "syntax_tokens",
)
_GENERATION_ERRORS = (
    AttributeError, IndexError, KeyError, OSError, RuntimeError, TypeError,
    ValueError)


def resolve_device(device: str) -> str:
    """Resolve ``auto`` to CUDA when available, otherwise CPU."""
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _tokenizer_path(base: str | Path) -> str:
    """Find the tokenizer beside the baseline before loading the candidate.

    The 63M research checkpoint has no tokenizer of its own. Both models must
    use the same tokenizer for token losses to be comparable.
    """
    base_dir = resolve_path(base)
    candidates = [base_dir / "chat_en_bpe8k.json",
                  resolve_path(DEFAULT_TOKENIZER)]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return ""


def _default_loader(path: str | Path, *, device: str,
                    tokenizer: str = ""):
    return load_model(path, device=device, tokenizer=tokenizer)


def _default_generator(model, tok, prompt: str, max_new: int) -> str:
    from atomzero.transformer import generate

    text, _stop = generate(
        model, tok, prompt, max_new=max_new, temperature=0.0, top_k=1,
        repetition_penalty=1.0)
    return text


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def retention_ratio(candidate_content_loss: Any,
                    baseline_content_loss: Any) -> float | None:
    """Candidate / baseline content loss, or ``None`` when undefined."""
    candidate = _as_float(candidate_content_loss)
    baseline = _as_float(baseline_content_loss)
    if candidate is None or baseline is None or baseline <= 0:
        return None
    return candidate / baseline


def retention_ok(ratio: Any,
                 threshold: Any = DEFAULT_THRESHOLD) -> bool | None:
    """True when the ratio is at or below the threshold."""
    value = _as_float(ratio)
    limit = _as_float(threshold)
    if value is None or limit is None:
        return None
    return value <= limit


def _loss_summary(metrics: Mapping[str, Any] | None) -> dict | None:
    if not isinstance(metrics, Mapping):
        return None
    return {field: metrics.get(field) for field in LOSS_FIELDS}


def _exact_summary(metrics: Mapping[str, Any] | None) -> dict | None:
    if not isinstance(metrics, Mapping):
        return None
    if metrics.get("available") is False:
        return {"available": False,
                "error_type": metrics.get("error_type")}
    return {
        "examples": metrics.get("examples"),
        "matches": metrics.get("matches"),
        "rate": metrics.get("rate"),
    }


def _key_copy_summary(metrics: Mapping[str, Any] | None) -> dict | None:
    if not isinstance(metrics, Mapping):
        return None
    if metrics.get("available") is False:
        return {"available": False,
                "error_type": metrics.get("error_type")}
    return {
        "examples": metrics.get("examples"),
        "skipped": metrics.get("skipped"),
        "matches": metrics.get("matches"),
        "rate": metrics.get("rate"),
        "key_recall": metrics.get("key_recall"),
    }


def _family_entry(metrics: Mapping[str, Any] | None,
                  family: str) -> dict | None:
    if not isinstance(metrics, Mapping):
        return None
    families = metrics.get("families")
    if not isinstance(families, Mapping):
        return None
    entry = families.get(family)
    if not isinstance(entry, Mapping):
        return None
    result = {
        "examples": entry.get("examples"),
        "matches": entry.get("matches"),
        "rate": entry.get("rate"),
    }
    if "key_recall" in entry:
        result["key_recall"] = entry.get("key_recall")
    return result


def _generation_metrics(model, tok, rows: list[dict], *,
                        generator: Callable[..., str] | None,
                        max_new: int) -> tuple[dict, dict]:
    """Run exact-match and key-copy generation once per row per model."""
    generate_fn = generator or _default_generator
    cache: dict[tuple[str, int], str] = {}

    def cached(loaded_model, loaded_tok, prompt: str, limit: int) -> str:
        key = (prompt, limit)
        if key not in cache:
            cache[key] = generate_fn(loaded_model, loaded_tok, prompt, limit)
        return cache[key]

    try:
        exact = exact_match_metrics(
            model, tok, rows, generator=cached, max_new=max_new)
    except _GENERATION_ERRORS as exc:
        exact = {"available": False, "error_type": type(exc).__name__}
    try:
        key_copy = key_copy_metrics(
            model, tok, rows, generator=cached, max_new=max_new)
    except _GENERATION_ERRORS as exc:
        key_copy = {"available": False, "error_type": type(exc).__name__}
    return exact, key_copy


def _read_eval_rows(data: str | Path | Sequence[str | Path],
                    max_rows: int) -> list[dict]:
    if isinstance(data, (str, Path)):
        paths = [data]
    else:
        paths = list(data)
    if not paths:
        raise ValueError("at least one --data file is required")
    rows: list[dict] = []
    for path in paths:
        try:
            part = read_rows(path)
        except SystemExit:
            raise ValueError(f"no user/assistant rows in {path}") from None
        rows.extend(part)
        if len(rows) >= max_rows:
            break
    if not rows:
        raise ValueError("no user/assistant rows found")
    return rows[:max_rows]


def _families_report(baseline: Mapping[str, Any],
                     candidate: Mapping[str, Any],
                     baseline_exact: Mapping[str, Any],
                     candidate_exact: Mapping[str, Any],
                     baseline_key: Mapping[str, Any],
                     candidate_key: Mapping[str, Any],
                     threshold: float) -> dict:
    baseline_families = per_family_metrics(baseline)
    candidate_families = per_family_metrics(candidate)
    names = sorted(set(baseline_families) | set(candidate_families))
    families: dict[str, dict] = {}
    for name in names:
        base_loss = _loss_summary(baseline_families.get(name))
        cand_loss = _loss_summary(candidate_families.get(name))
        base_entry: dict[str, Any] = dict(base_loss or {})
        cand_entry: dict[str, Any] = dict(cand_loss or {})
        base_entry["exact_match"] = _family_entry(baseline_exact, name)
        cand_entry["exact_match"] = _family_entry(candidate_exact, name)
        # ``None`` means the family has no rows with keys; key-copy does not
        # apply there. Keeping the key makes that explicit and testable.
        base_entry["key_copy"] = _family_entry(baseline_key, name)
        cand_entry["key_copy"] = _family_entry(candidate_key, name)
        ratio = retention_ratio(
            (cand_loss or {}).get("content_loss"),
            (base_loss or {}).get("content_loss"))
        families[name] = {
            "baseline": base_entry,
            "candidate": cand_entry,
            "retention_ratio": ratio,
            "retention_ok": retention_ok(ratio, threshold),
        }
    return families


def run_retention_eval(
    candidate: str | Path,
    baseline: str | Path,
    data: str | Path | Sequence[str | Path],
    *,
    max_rows: int = DEFAULT_MAX_ROWS,
    device: str = "auto",
    per_family: bool = False,
    threshold: float = DEFAULT_THRESHOLD,
    exact_match_max_new: int = DEFAULT_EXACT_MATCH_MAX_NEW,
    loader: Callable[..., tuple[Any, Any]] | None = None,
    generator: Callable[..., str] | None = None,
) -> dict:
    """Compare ``candidate`` with ``baseline`` on shared eval rows.

    ``loader`` is called as ``loader(path, device=..., tokenizer=...)`` and
    returns ``(model, tokenizer)``. ``generator`` is called as
    ``generator(model, tokenizer, prompt, max_new)``. Both are injectable so
    tests never load a real checkpoint.
    """
    if max_rows < 1:
        raise ValueError("max_rows must be at least 1")
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    if exact_match_max_new < 1:
        raise ValueError("exact_match_max_new must be at least 1")

    rows = _read_eval_rows(data, max_rows)
    resolved_device = resolve_device(device)
    load = loader or _default_loader
    tokenizer_path = _tokenizer_path(baseline)
    baseline_model, tok = load(
        baseline, device=resolved_device, tokenizer=tokenizer_path)
    candidate_model, _candidate_tok = load(
        candidate, device=resolved_device, tokenizer=tokenizer_path)

    baseline_metrics = score_rows(baseline_model, tok, rows)
    candidate_metrics = score_rows(candidate_model, tok, rows)
    baseline_exact, baseline_key = _generation_metrics(
        baseline_model, tok, rows, generator=generator,
        max_new=exact_match_max_new)
    candidate_exact, candidate_key = _generation_metrics(
        candidate_model, tok, rows, generator=generator,
        max_new=exact_match_max_new)

    ratio = retention_ratio(
        candidate_metrics.get("content_loss"),
        baseline_metrics.get("content_loss"))
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "content_free": True,
        "threshold": float(threshold),
        "exact_match_max_new": int(exact_match_max_new),
        "rows": baseline_metrics.get("examples"),
        "tokens": baseline_metrics.get("tokens"),
        "baseline": {
            **_loss_summary(baseline_metrics),
            "exact_match": _exact_summary(baseline_exact),
            "key_copy": _key_copy_summary(baseline_key),
        },
        "candidate": {
            **_loss_summary(candidate_metrics),
            "exact_match": _exact_summary(candidate_exact),
            "key_copy": _key_copy_summary(candidate_key),
        },
        "retention_ratio": ratio,
        "retention_ok": retention_ok(ratio, threshold),
    }
    if per_family:
        payload["families"] = _families_report(
            baseline_metrics, candidate_metrics,
            baseline_exact, candidate_exact,
            baseline_key, candidate_key, float(threshold))
    return payload


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    if target.parent != Path(""):
        target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
        + "\n",
        encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--candidate", required=True,
        help="candidate checkpoint directory (with model.pt)")
    parser.add_argument(
        "--baseline", required=True,
        help="product baseline checkpoint directory (with model.pt)")
    parser.add_argument(
        "--data", nargs="+", required=True,
        help="one or more JSONL files with user/assistant rows")
    parser.add_argument("--out", default=None,
                        help="write the content-free JSON report here")
    parser.add_argument(
        "--max-rows", type=int, default=DEFAULT_MAX_ROWS,
        help="maximum rows across all files, in file order (default: 200)")
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--per-family", action="store_true",
        help="also report per-family loss and retention metrics")
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help="maximum acceptable content-loss ratio (default: 1.10)")
    parser.add_argument(
        "--exact-match-max-new", type=int,
        default=DEFAULT_EXACT_MATCH_MAX_NEW,
        help="cap on generated answer tokens (default: 64)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.max_rows < 1:
        parser.error("--max-rows must be at least 1")
    if args.threshold <= 0:
        parser.error("--threshold must be positive")
    if args.exact_match_max_new < 1:
        parser.error("--exact-match-max-new must be at least 1")

    try:
        payload = run_retention_eval(
            args.candidate, args.baseline, args.data,
            max_rows=args.max_rows, device=args.device,
            per_family=args.per_family, threshold=args.threshold,
            exact_match_max_new=args.exact_match_max_new)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
    if args.out:
        _write_json(args.out, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
