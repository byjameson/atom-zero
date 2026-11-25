"""Vast.ai preflight: verify credentials, balance and suitable offers.

This script is read-only. It never creates or destroys an instance.

Usage
-----
    python scripts/vastai/preflight.py
    python scripts/vastai/preflight.py --max-price 0.45 --min-vram 24 --json

The API key is read from ``--key-file`` (default: Desktop/vastai.txt) and is
passed to the ``vastai`` CLI through ``VAST_API_KEY``. The key is never printed.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import Offer, estimate_cost, format_offer_table, select_offers  # noqa: E402

DEFAULT_KEY_FILE = Path.home() / "Desktop" / "vastai.txt"
DEFAULT_IMAGE = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime"


def read_key(path: Path) -> str:
    if not path.exists():
        raise SystemExit(f"key file not found: {path}")
    key = path.read_text(encoding="utf-8").strip().splitlines()[0].strip()
    if len(key) < 20 or any(ch.isspace() for ch in key):
        raise SystemExit("key file does not look like a single API key")
    return key


def run_vastai(args: list[str], *, key: str) -> str:
    exe = shutil.which("vastai")
    if not exe:
        raise SystemExit(
            "vastai CLI not found. Install it with: python -m pip install vastai"
        )
    env = os.environ.copy()
    env["VAST_API_KEY"] = key
    proc = subprocess.run(
        [exe, *args], capture_output=True, text=True, env=env, timeout=120
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()[:400]
        raise SystemExit(f"vastai {' '.join(args)} failed: {detail}")
    return proc.stdout


def parse_offers(payload: str) -> list[Offer]:
    data = json.loads(payload)
    if isinstance(data, dict):
        data = data.get("offers") or data.get("instances") or []
    offers: list[Offer] = []
    for row in data:
        try:
            offers.append(
                Offer(
                    offer_id=int(row["id"]),
                    gpu_name=str(row.get("gpu_name", "?")),
                    num_gpus=int(row.get("num_gpus", 1)),
                    gpu_ram_gb=float(row.get("gpu_ram", 0)) / 1024.0,
                    dph=float(row.get("dph_total", row.get("dph", 0))),
                    reliability=float(row.get("reliability2", row.get("reliability", 0))),
                    inet_down_mbps=float(row.get("inet_down", 0)),
                    inet_up_mbps=float(row.get("inet_up", 0)),
                    cuda_max=float(row.get("cuda_max_good", row.get("cuda_max", 0)) or 0),
                    location=str(row.get("geolocation", "") or ""),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return offers


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Vast.ai read-only preflight")
    parser.add_argument("--key-file", default=str(DEFAULT_KEY_FILE))
    parser.add_argument("--max-price", type=float, default=0.50)
    parser.add_argument("--min-vram", type=float, default=24.0)
    parser.add_argument("--min-reliability", type=float, default=0.95)
    parser.add_argument("--hours", type=float, default=3.0,
                        help="planned run length used for the cost estimate")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    key = read_key(Path(args.key_file))
    user_raw = run_vastai(["show", "user", "--raw"], key=key)
    try:
        user = json.loads(user_raw)
    except json.JSONDecodeError:
        user = {}

    balance = float(user.get("balance", 0) or 0)
    credit = float(user.get("credit", 0) or 0)
    available = round(balance + credit, 2)

    # Vast.ai's gpu_ram search unit is GiB, not MiB.
    query = (
        "num_gpus=1 rentable=true "
        f"dph<{args.max_price:.3f} "
        f"gpu_ram>={args.min_vram:.0f}"
    )
    offers_raw = run_vastai(
        ["search", "offers", query, "-o", "dph", "--limit", "80", "--raw"],
        key=key,
    )
    offers = select_offers(
        parse_offers(offers_raw),
        max_price=args.max_price,
        min_vram_gb=args.min_vram,
        min_reliability=args.min_reliability,
    )

    best = offers[0] if offers else None
    report = {
        "balance_usd": balance,
        "credit_usd": credit,
        "available_usd": available,
        "max_price": args.max_price,
        "min_vram_gb": args.min_vram,
        "hours": args.hours,
        "recommended": None if best is None else {
            "offer_id": best.offer_id,
            "gpu_name": best.gpu_name,
            "vram_gb": best.vram_gb,
            "dph": best.dph,
            "reliability": best.reliability,
            "estimated_cost_usd": estimate_cost(best.dph, args.hours),
        },
        "offers": [
            {
                "offer_id": o.offer_id,
                "gpu_name": o.gpu_name,
                "vram_gb": o.vram_gb,
                "dph": o.dph,
                "reliability": o.reliability,
                "inet_down_mbps": o.inet_down_mbps,
            }
            for o in offers[:10]
        ],
        "image": DEFAULT_IMAGE,
    }

    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    print(f"balance: {balance:.2f} USD + credit {credit:.2f} USD = {available:.2f} USD available")
    print(f"image:   {DEFAULT_IMAGE}")
    print()
    print(format_offer_table(offers))
    if best:
        cost = estimate_cost(best.dph, args.hours)
        print()
        print(
            f"recommended: offer {best.offer_id} {best.gpu_name} "
            f"{best.vram_gb:.0f} GB at ${best.dph:.2f}/hr"
        )
        print(f"planned {args.hours:.1f} h -> about ${cost:.2f} (compute only)")
        if cost > available:
            print("WARNING: estimated compute cost exceeds the available credit")
    else:
        print()
        print("no matching offers; raise --max-price or lower --min-reliability")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
