"""Validate an external LoRA adapter against a local base checkpoint.

    python research/adapter_check.py \\
        --base checkpoints/transformer_behavior_50m_long \\
        --adapter checkpoints/adapters_50m/v0001

The adapter can be trained by ``streamft``, ``continual-lora`` or another
external trainer. This command only checks compatibility: base config, adapter
metadata, LoRA tensor loading and a forward pass. Add ``--data`` to score an
assistant-turn holdout with :mod:`research.compare_adapters`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from research.adapter_runtime import check_adapter, resolve_path
    from research.compare_adapters import compare
except ImportError:  # running as `python research/adapter_check.py`
    from adapter_runtime import check_adapter, resolve_path
    from compare_adapters import compare

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--tokenizer", default="")
    ap.add_argument("--data", default="", help="optional JSONL holdout")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    result = check_adapter(args.base, args.adapter, tokenizer=args.tokenizer)
    if args.data:
        if not result.get("ok"):
            result["comparison"] = {"error": "adapter check failed"}
        else:
            result["comparison"] = compare(
                args.base, args.data, [args.adapter], args.tokenizer)
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.out:
        out = resolve_path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
