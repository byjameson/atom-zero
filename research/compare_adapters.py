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
import time
from pathlib import Path

try:
    from research.adapter_runtime import (
        assistant_label_mask, fmt, load_model, read_rows, score_rows)
except ImportError:  # running as `python research/compare_adapters.py`
    from adapter_runtime import (
        assistant_label_mask, fmt, load_model, read_rows, score_rows)

ROOT = Path(__file__).resolve().parents[1]


def label_for(path: str) -> str:
    version = Path(path).name
    parent = Path(path).parent.name
    if parent and version.startswith("v") and version[1:].isdigit():
        return f"{parent}/{version}"
    return version or path


def compare(base: str, data: str | Path, adapters: list[str],
            tokenizer: str = "") -> dict:
    rows = read_rows(data)
    base_model, tok = load_model(base, tokenizer=tokenizer)
    models = [{"label": "base", "path": base,
               "metrics": score_rows(base_model, tok, rows)}]
    for adapter in adapters:
        model, _ = load_model(base, adapter, tokenizer=tokenizer)
        models.append({"label": label_for(adapter), "path": adapter,
                       "metrics": score_rows(model, tok, rows)})
    reference = models[0]["metrics"]
    for item in models[1:]:
        metrics = item["metrics"]
        item["vs_base"] = {
            "loss_delta": metrics["loss"] - reference["loss"],
            "loss_relative": metrics["loss"] / reference["loss"] - 1,
            "ppl_ratio": metrics["ppl"] / reference["ppl"] - 1,
        }
    return {
        "base": base,
        "data": str(data),
        "examples": len(rows),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "models": models,
    }


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
    args = ap.parse_args(argv)
    payload = compare(args.base, args.data, args.adapters, args.tokenizer)
    if args.out:
        out = ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n",
                       encoding="utf-8")
    summary = [{
        "label": item["label"],
        "path": item["path"],
        "loss": round(item["metrics"]["loss"], 6),
        "ppl": round(item["metrics"]["ppl"], 4),
        "tokens": item["metrics"]["tokens"],
        **{k: round(v, 6) for k, v in item.get("vs_base", {}).items()},
    } for item in payload["models"]]
    print(json.dumps({"examples": payload["examples"], "models": summary},
                     indent=2, ensure_ascii=False))
    if args.out:
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
