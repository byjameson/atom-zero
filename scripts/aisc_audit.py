"""Run optional AISC architecture checks for Atom Zero.

AISC is a separate local prototype. It is not a runtime dependency of
atomzero. Install it or point this script at its checkout:

    python scripts/aisc_audit.py --aisc-path "C:\path\to\ai-systems-algebra"
    python scripts/aisc_audit.py --aisc-path "$env:AISC_PATH" --json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPECS = (
    "docs/aisc/atomzero-read.toml",
    "docs/aisc/atomzero-task-actions.toml",
)


def _resolve_aisc_path(value: str | None) -> Path | None:
    if not value:
        return None
    base = Path(value).expanduser().resolve()
    candidates = (base, base / "ai-systems-algebra")
    for candidate in candidates:
        if (candidate / "aisc").is_dir():
            return candidate
    raise SystemExit(f"AISC package not found under: {base}")


def _spec_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "specs", nargs="*", default=list(DEFAULT_SPECS),
        help="AISC TOML specs (default: Atom Zero architecture specs)")
    parser.add_argument(
        "--aisc-path", default=os.environ.get("AISC_PATH", ""),
        help="path to the ai-systems-algebra checkout")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    aisc_path = _resolve_aisc_path(args.aisc_path)
    if aisc_path is not None:
        sys.path.insert(0, str(aisc_path))
    try:
        from aisc.spec import load_spec
    except ImportError as exc:
        print(
            "AISC is not installed. Pass --aisc-path or set AISC_PATH.",
            file=sys.stderr)
        return 2

    results = []
    for raw in args.specs or DEFAULT_SPECS:
        path = _spec_path(raw)
        try:
            definition = load_spec(path)
            ok = definition.system.ok(definition.spec)
            report = definition.system.report(definition.spec)
        except Exception as exc:
            results.append({
                "source": str(path),
                "ok": False,
                "error": str(exc),
            })
            continue
        results.append({
            "source": str(path),
            "ok": ok,
            "report": report,
            "data": definition.to_dict(),
        })

    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        for index, result in enumerate(results):
            if index:
                print("\n" + "=" * 72 + "\n")
            if "error" in result:
                print(f"[FAIL] {result['source']}: {result['error']}")
            else:
                print(result["report"])
    return 0 if all(result["ok"] for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
