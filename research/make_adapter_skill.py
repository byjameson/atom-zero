"""Generate a deterministic verified-teacher fixture for adapter training.

This is a research fixture, not product data. It creates a narrow JSON-format
skill so a real LoRA run and held-out gate can be reproduced on a small local
base model.

    python research/make_adapter_skill.py \
      --out data/adapter_skill_verified.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

NAMES = [
    "Ada", "Bora", "Cem", "Derya", "Ege", "Funda", "Gizem", "Hakan",
    "Ipek", "Jale", "Kaan", "Lale", "Mert", "Naz", "Ozan", "Pelin",
    "Rana", "Selin", "Tolga", "Umut", "Vera", "Yasin", "Zeynep", "Aylin",
]


def make_rows(count: int, seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for index in range(max(1, int(count))):
        name = NAMES[index % len(NAMES)]
        score = rng.randint(1, 99)
        rows.append({
            "user": f"Return only JSON for name={name} and score={score}.",
            "assistant": json.dumps({"name": name, "score": score}),
            "verification": {
                "verified": True,
                "promotable": True,
                "status": "verified",
            },
        })
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", default="data/adapter_skill_verified.jsonl")
    parser.add_argument("--count", type=int, default=320)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    rows = make_rows(args.count, args.seed)
    target = Path(args.out).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8")
    print(json.dumps({
        "path": str(target),
        "examples": len(rows),
        "verified": sum(1 for row in rows
                        if row["verification"].get("verified") is True),
        "seed": args.seed,
        "method": "deterministic_json_format_fixture_v1",
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
