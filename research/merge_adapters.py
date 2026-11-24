"""Merge saved LoRA adapters with weighted A/B averaging or delta SVD.

Adapter soup combines adapters that were trained from the same base with the
same LoRA geometry. It does **not** retrain, so it is cheaper than replay, but
it also does not guarantee that both skills survive. Measure the merged
adapter on each held-out skill before drawing conclusions.

    python research/merge_adapters.py ^
      --adapters checkpoints/staged_copy/echo_json ^
                 checkpoints/staged_copy/two_json ^
      --weights 1 1 ^
      --out checkpoints/adapter_soup/echo_two_json ^
      --base checkpoints/transformer_behavior_50m_long ^
      --eval data/staged_copy_v1/stage_multi/eval.jsonl ^
      --result research/results/adapter_soup.json ^
      --comparison research/results/adapter_soup_comparison.json

Two merge modes are available:

``ab``
    ``A = sum_i w_i A_i`` and ``B = sum_i w_i B_i``. Cheap, keeps the original
    rank, but the average of products is not the product of averages.

``delta-svd``
    ``D = sum_i w_i B_i A_i``, then factor ``D`` back into a new low-rank pair
    with SVD. ``--rank`` caps the new rank; it defaults to the original rank.
    Alpha is rescaled so the effective LoRA scaling is preserved.

Only the merged adapter is written. Input adapter directories are never
modified. The result and comparison files contain aggregate metrics only; no
prompt or answer rows are copied from the evaluation JSONL.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

try:
    from research.adapter_runtime import check_adapter
    from research.compare_adapters import compare
except ImportError:  # running as `python research/merge_adapters.py`
    from adapter_runtime import check_adapter
    from compare_adapters import compare

from atomzero.lora import (
    AdapterMergeError,
    merge_adapters,
)

ROOT = Path(__file__).resolve().parents[1]

METRIC_FIELDS = (
    "examples",
    "tokens",
    "loss",
    "ppl",
    "content_loss",
    "content_ppl",
    "content_tokens",
    "key_loss",
    "key_tokens",
    "value_loss",
    "value_tokens",
    "syntax_loss",
    "syntax_tokens",
    "categories",
    "key_copy_rate",
)
GENERATION_FIELDS = (
    "examples",
    "skipped",
    "matches",
    "rate",
    "key_recall",
    "max_new",
    "available",
    "error",
)
VS_BASE_FIELDS = ("loss_delta", "loss_relative", "ppl_ratio")


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def _display_path(path: str | Path) -> str:
    value = Path(path)
    try:
        return value.relative_to(ROOT).as_posix()
    except ValueError:
        return str(value)


def _sanitize_metrics(metrics) -> dict:
    """Keep aggregate numbers and drop any row-level prompts/answers."""
    if not isinstance(metrics, dict):
        return {}
    cleaned = {}
    for key in METRIC_FIELDS:
        if key in metrics:
            cleaned[key] = metrics[key]
    for key in ("exact_match", "key_copy"):
        block = metrics.get(key)
        if not isinstance(block, dict):
            continue
        summary = {field: block[field] for field in GENERATION_FIELDS
                   if field in block}
        families = block.get("families")
        if isinstance(families, dict):
            summary["families"] = {
                name: {
                    field: values[field]
                    for field in ("examples", "matches", "rate",
                                  "key_recall")
                    if isinstance(values, dict) and field in values
                }
                for name, values in families.items()
                if isinstance(values, dict)
            }
        cleaned[key] = summary
    return cleaned


def _sanitize_families(families) -> dict:
    if not isinstance(families, dict):
        return {}
    cleaned = {}
    for family, pair in families.items():
        if not isinstance(pair, dict):
            continue
        cleaned[str(family)] = {
            "baseline": (None if pair.get("baseline") is None
                         else _sanitize_metrics(pair.get("baseline"))),
            "adapter": (None if pair.get("adapter") is None
                        else _sanitize_metrics(pair.get("adapter"))),
        }
    return cleaned


def _sanitize_models(payload: dict) -> list[dict]:
    models = []
    for item in payload.get("models", []):
        if not isinstance(item, dict):
            continue
        model = {
            "label": item.get("label"),
            "path": item.get("path"),
            "metrics": _sanitize_metrics(item.get("metrics")),
        }
        comparison = item.get("vs_base")
        if isinstance(comparison, dict):
            model["vs_base"] = {
                key: comparison.get(key) for key in VS_BASE_FIELDS
                if key in comparison
            }
        families = item.get("families")
        if isinstance(families, dict):
            model["per_family"] = _sanitize_families(families)
        models.append(model)
    return models


def _comparison_payload(payload: dict, base: str, data: str,
                        adapters: list[str], weights: list[float]) -> dict:
    """Build a content-free comparison document from compare_adapters data."""
    sanitized = {
        "created": payload.get("created") or time.strftime(
            "%Y-%m-%dT%H:%M:%S"),
        "base": _display_path(base),
        "data": _display_path(data),
        "examples": payload.get("examples"),
        "adapters": [_display_path(path) for path in adapters],
        "weights": weights,
        "models": _sanitize_models(payload),
    }
    families = payload.get("families")
    if isinstance(families, dict):
        sanitized["per_family"] = _sanitize_families(families)
    return sanitized


def _validate_merged(base: str, out: str) -> dict:
    result = check_adapter(base, out)
    validation = {
        key: result.get(key)
        for key in ("ok", "base", "adapter", "lora_parameters",
                    "logits_shape", "error")
        if key in result
    }
    if not result.get("ok"):
        raise AdapterMergeError(
            "merged adapter failed validation against --base: "
            + str(result.get("error") or "unknown error"))
    return validation


def _write_json(path: str | Path, payload: dict) -> Path:
    target = _resolve(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n",
                      encoding="utf-8")
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--adapters", nargs="+", required=True,
        help="adapter directories saved by save_adapter")
    parser.add_argument(
        "--weights", nargs="+", type=float, default=None,
        help="one non-negative weight per adapter (default: equal)")
    parser.add_argument(
        "--out", required=True,
        help="directory for the merged adapter")
    parser.add_argument(
        "--base", default="",
        help="optional base checkpoint for validation and comparison")
    parser.add_argument(
        "--eval", default="",
        help="optional JSONL holdout scored with compare_adapters.py")
    parser.add_argument(
        "--result", default="research/results/adapter_soup.json",
        help="content-free merge/evaluation summary")
    parser.add_argument(
        "--comparison",
        default="research/results/adapter_soup_comparison.json",
        help="content-free base-vs-merged comparison (only written with "
             "--eval)")
    parser.add_argument(
        "--merge-mode", choices=("ab", "delta-svd"), default="ab",
        help="ab averages A/B tensors; delta-svd factors the merged delta")
    parser.add_argument(
        "--rank", type=int, default=None,
        help="rank cap for --merge-mode delta-svd (default: source rank)")
    parser.add_argument(
        "--tokenizer", default="",
        help="tokenizer path; defaults to the base directory or "
             "data/chat_en_bpe8k.json")
    parser.add_argument(
        "--per-family", action="store_true",
        help="include per-family aggregates when --eval rows carry a family")
    parser.add_argument(
        "--exact-match", action="store_true",
        help="also report generated exact-match accuracy")
    parser.add_argument(
        "--key-copy", action="store_true",
        help="also report generated key-name copy accuracy")
    parser.add_argument(
        "--exact-match-max-new", type=int, default=64,
        help="generation cap for --exact-match/--key-copy (default: 64)")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.eval and not args.base:
        parser.error("--eval requires --base so compare_adapters.py can load "
                     "the reference model")
    if args.rank is not None and args.merge_mode != "delta-svd":
        parser.error("--rank is only valid with --merge-mode delta-svd")
    if args.rank is not None and args.rank < 1:
        parser.error("--rank must be >= 1")
    try:
        summary = merge_adapters(
            [_resolve(path) for path in args.adapters],
            out=_resolve(args.out),
            weights=args.weights,
            mode=args.merge_mode,
            rank=args.rank,
        )
    except (AdapterMergeError, OSError, RuntimeError, KeyError) as exc:
        parser.error(str(exc))
    out = _resolve(args.out)

    validation = None
    if args.base:
        try:
            validation = _validate_merged(args.base, str(out))
        except AdapterMergeError as exc:
            parser.error(str(exc))

    evaluation = None
    comparison = None
    if args.eval:
        try:
            payload = compare(
                args.base, str(_resolve(args.eval)), [str(out)],
                tokenizer=args.tokenizer,
                per_family=args.per_family,
                exact_match=args.exact_match,
                key_copy=args.key_copy,
                exact_match_max_new=args.exact_match_max_new)
        except (OSError, RuntimeError, ValueError) as exc:
            parser.error(str(exc))
        comparison = _comparison_payload(
            payload, args.base, str(_resolve(args.eval)),
            [str(path) for path in summary["adapters"]], summary["weights"])
        evaluation = {
            "data": comparison["data"],
            "examples": comparison["examples"],
            "models": comparison["models"],
        }
        if "per_family" in comparison:
            evaluation["per_family"] = comparison["per_family"]

    result = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "mode": summary["mode"],
        "rank": summary["rank"],
        "out": _display_path(out),
        "adapters": [_display_path(path) for path in summary["adapters"]],
        "weights": summary["weights"],
        "config": summary["config"],
        "tensors": summary["tensors"],
        "parameters": summary["parameters"],
        "validation": validation,
        "evaluation": evaluation,
    }
    if args.result:
        _write_json(args.result, result)
    if args.comparison and comparison is not None:
        _write_json(args.comparison, comparison)

    printable = {key: value for key, value in result.items()
                 if key != "config"}
    printable["config"] = {
        key: summary["config"][key]
        for key in ("r_attn", "alpha_attn", "r_mlp", "alpha_mlp")
    }
    print(json.dumps(printable, indent=2, ensure_ascii=False))
    if args.result:
        print(f"wrote {_display_path(_resolve(args.result))}")
    if args.comparison and comparison is not None:
        print(f"wrote {_display_path(_resolve(args.comparison))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
