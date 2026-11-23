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
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
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


def _command(check: dict, aisc_path: str,
             project_root: str | Path = ROOT) -> tuple[list[str], Path]:
    root = Path(project_root).expanduser().resolve()
    if check["kind"] == "pytest":
        target = Path(check["target"])
        if not target.is_absolute():
            target = root / target
        return ([sys.executable, "-m", "pytest", str(target), "-q"], root)
    if check["kind"] == "aisc":
        base = Path(aisc_path).expanduser().resolve() if aisc_path else root
        target = Path(check["target"])
        if not target.is_absolute():
            target = root / target
        return ([sys.executable, "-m", "aisc", "check",
                 str(target.resolve()), "--strict"], base)
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
    command, cwd = _command(check, aisc_path, project_root)
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


FORBIDDEN_ARCHIVE_KEYS = frozenset({
    "output", "text", "query", "answer", "chunk", "content", "note",
    "session", "source",
})


def archive_report(report: dict, now=None) -> dict:
    """Build a content-free archive entry from an evaluation report."""
    if not isinstance(report, dict) or not isinstance(report.get("checks"), list):
        raise ValueError("report.checks must be a list")
    checks = []
    for index, item in enumerate(report["checks"]):
        if not isinstance(item, dict):
            raise ValueError(f"report.checks[{index}] must be a mapping")
        checks.append({
            "name": str(item.get("name") or ""),
            "kind": str(item.get("kind") or ""),
            "status": str(item.get("status") or ""),
            "ok": bool(item.get("ok")),
            "duration_ms": int(item.get("duration_ms") or 0),
        })
    summary = report.get("summary") if isinstance(
        report.get("summary"), dict) else {}
    material = json.dumps({
        "method": report.get("method", ""),
        "checks": checks,
        "summary": summary,
    }, sort_keys=True, separators=(",", ":"))
    gate_id = hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]
    moment = now or datetime.now(timezone.utc)
    if isinstance(moment, datetime):
        at = moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    else:
        at = str(moment)
    return {
        "version": 1,
        "at": at,
        "gate_id": gate_id,
        "ok": bool(report.get("ok")),
        "total": int(summary.get("total", len(checks))),
        "passed": int(summary.get("passed", 0)),
        "failed": int(summary.get("failed", 0)),
        "manual_required": int(summary.get("manual_required", 0)),
        "method": str(report.get("method", ""))[:80],
        "checks": checks,
    }


def verify_archive(entries) -> dict:
    """Validate content-free gate archive entries."""
    if not isinstance(entries, list):
        return {"ok": False, "entries": 0, "errors": ["archive must be a list"]}
    errors = []
    for index, entry in enumerate(entries):
        where = f"entry {index}"
        if not isinstance(entry, dict):
            errors.append(f"{where}: must be a mapping")
            continue
        forbidden = sorted(set(entry) & FORBIDDEN_ARCHIVE_KEYS)
        if forbidden:
            errors.append(
                f"{where}: content key(s) not allowed: " + ", ".join(forbidden))
        if entry.get("version") != 1:
            errors.append(f"{where}: version must be 1")
        if (not isinstance(entry.get("gate_id"), str)
                or len(entry["gate_id"]) != 16):
            errors.append(f"{where}: invalid gate_id")
        if not isinstance(entry.get("checks"), list):
            errors.append(f"{where}: checks must be a list")
    return {"ok": not errors, "entries": len(entries), "errors": errors}


def load_archive(path) -> list[dict]:
    target = Path(path)
    if not target.exists():
        return []
    entries = []
    for number, line in enumerate(
            target.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{target}:{number}: invalid JSON") from exc
        if not isinstance(entry, dict):
            raise ValueError(f"{target}:{number}: must be a JSON object")
        entries.append(entry)
    return entries


def write_archive(path, entries) -> None:
    report = verify_archive(entries)
    if not report["ok"]:
        raise ValueError(
            "invalid gate archive: " + "; ".join(report["errors"]))
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(
        json.dumps(item, ensure_ascii=False, sort_keys=True,
                   separators=(",", ":")) + "\n"
        for item in entries)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        os.replace(tmp_name, target)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


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
    parser.add_argument("--archive", type=Path, default=None,
                        help="append a content-free gate entry to this JSONL")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        checks = [parse_check(value) for value in args.check]
        report = evaluate_gate(
            checks, aisc_path=args.aisc_path, timeout=args.timeout)
        if args.archive is not None:
            entries = load_archive(args.archive)
            entry = archive_report(report)
            entries.append(entry)
            write_archive(args.archive, entries)
            report["gate_id"] = entry["gate_id"]
    except (OSError, ValueError) as exc:
        print(f"input error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(_render(report))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
