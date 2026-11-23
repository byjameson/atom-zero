"""Allowlisted evaluation gates for proposed changes.

A plan is not accepted because it sounds good. This module runs an explicit,
allowlisted check set and reports the evidence: a focused pytest target, an
AISC strict spec check, or a manual check that a human must confirm. It never
runs arbitrary shell commands, never edits files and never applies a plan.

    atomzero-gate --check pytest:tests/test_claims.py
    atomzero-gate --check aisc:docs/aisc/atomzero-claims.toml --aisc-path PATH
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KINDS = ("pytest", "aisc", "manual")
_MAX_OUTPUT = 4000


def parse_check(value: str) -> dict:
    """`kind:target` biçimindeki bir kontrolü ayrıştırır."""
    if not isinstance(value, str) or ":" not in value:
        raise ValueError("check must be kind:target")
    kind, target = value.split(":", 1)
    kind = kind.strip().lower()
    target = target.strip()
    if kind not in KINDS:
        raise ValueError(f"unknown check kind: {kind}")
    if kind != "manual" and not target:
        raise ValueError(f"{kind} check requires a target")
    return {"name": target or kind, "kind": kind, "target": target}


def _command(check: dict, aisc_path: str) -> tuple[list[str], Path]:
    if check["kind"] == "pytest":
        return ([sys.executable, "-m", "pytest", check["target"], "-q"],
                ROOT)
    if check["kind"] == "aisc":
        base = Path(aisc_path).expanduser().resolve() if aisc_path else ROOT
        return ([sys.executable, "-m", "aisc", "check",
                 str((ROOT / check["target"]).resolve()), "--strict"], base)
    raise ValueError("manual checks have no command")


def run_check(
    check: dict,
    *,
    project_root: str | Path = ROOT,
    aisc_path: str = "",
    timeout: int = 300,
) -> dict:
    """Tek bir kontrolü çalıştırır (veya manual olarak işaretler)."""
    if not isinstance(check, dict):
        raise ValueError("check must be an object")
    kind = str(check.get("kind", "")).lower()
    if kind not in KINDS:
        raise ValueError(f"unknown check kind: {kind}")
    if kind == "manual":
        return {
            "name": str(check.get("name") or "manual"),
            "kind": "manual",
            "target": str(check.get("target") or ""),
            "ok": False,
            "status": "manual_required",
            "exit_code": None,
            "duration_ms": 0,
            "output": "manual confirmation is required",
        }
    if not check.get("target"):
        raise ValueError(f"{kind} check requires a target")
    command, cwd = _command(check, aisc_path)
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command, cwd=str(cwd), capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            check=False)
        duration = int((time.monotonic() - started) * 1000)
        output = ((completed.stdout or "") + (completed.stderr or "")).strip()
        return {
            "name": str(check.get("name") or check["target"]),
            "kind": kind,
            "target": str(check["target"]),
            "ok": completed.returncode == 0,
            "status": "pass" if completed.returncode == 0 else "fail",
            "exit_code": completed.returncode,
            "duration_ms": duration,
            "output": output[-_MAX_OUTPUT:],
        }
    except subprocess.TimeoutExpired:
        return {
            "name": str(check.get("name") or check["target"]),
            "kind": kind,
            "target": str(check["target"]),
            "ok": False,
            "status": "timeout",
            "exit_code": None,
            "duration_ms": int((time.monotonic() - started) * 1000),
            "output": f"check timed out after {timeout}s",
        }


def evaluate_gate(
    checks,
    *,
    project_root: str | Path = ROOT,
    aisc_path: str = "",
    timeout: int = 300,
) -> dict:
    """Bir kontrol kümesini çalıştırır ve kabul kapısı raporu üretir."""
    if not isinstance(checks, list) or not checks:
        raise ValueError("checks must be a non-empty list")
    if timeout < 1:
        raise ValueError("timeout must be positive")
    results = [
        run_check(check, project_root=project_root, aisc_path=aisc_path,
                  timeout=timeout)
        for check in checks
    ]
    passed = sum(1 for item in results if item["ok"])
    return {
        "checks": results,
        "summary": {
            "total": len(results),
            "passed": passed,
            "failed": len(results) - passed,
            "manual_required": sum(
                1 for item in results if item["status"] == "manual_required"),
        },
        "ok": passed == len(results),
        "auto_apply": False,
        "weights_changed": False,
        "method": "allowlisted_evaluation_gate_v1",
    }


def _render(report: dict) -> str:
    summary = report["summary"]
    lines = [
        f"Evaluation gate — {summary['passed']}/{summary['total']} passed, "
        f"{summary['failed']} failed, {summary['manual_required']} manual",
    ]
    for item in report["checks"]:
        lines.append(
            f"- {item['status'].upper()} {item['kind']} {item['target']} "
            f"({item['duration_ms']} ms)")
        if not item["ok"] and item["output"]:
            lines.append("  " + item["output"].splitlines()[-1][:220])
    lines.append(f"Gate: {'PASS' if report['ok'] else 'FAIL'}")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="append", default=[],
        help="kind:target, repeatable (pytest:..., aisc:..., manual:...)")
    parser.add_argument("--aisc-path", default=os.environ.get("AISC_PATH", ""))
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        checks = [parse_check(value) for value in args.check]
        report = evaluate_gate(
            checks, aisc_path=args.aisc_path, timeout=args.timeout)
    except ValueError as exc:
        print(f"input error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(_render(report))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
