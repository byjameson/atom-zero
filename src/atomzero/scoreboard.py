"""Content-free scoreboard for measuring self-improvement over time.

A plan or a version is only an improvement if the measured checks move in the
right direction. This module records a small, local run summary (commit, test
counts, benchmark counts, AISC spec counts) and reports the latest result and
the delta from the previous run. It never stores test output, prompts, queries
or source text.

    atomzero-scoreboard --file data/scoreboard.jsonl \
      --record --commit 514632e --tests-passed 755 \
      --benchmark-passed 10 --benchmark-total 10 \
      --aisc-passed 14 --aisc-total 16 --json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

__all__ = [
    "record_run",
    "summarize_runs",
    "verify_runs",
]

VERSION = 1
FORBIDDEN_KEYS = frozenset({
    "text", "query", "answer", "chunk", "content", "note", "session",
    "source", "output", "log",
})
RECORD_KEYS = frozenset({
    "version", "at", "commit", "tests_passed", "tests_failed",
    "benchmark_passed", "benchmark_total", "aisc_passed", "aisc_total",
    "note_id",
})
_COMMIT_RE = re.compile(r"\A[0-9a-f]{7,40}\Z")
_NOTE_RE = re.compile(r"\A(?:[0-9a-f]{1,64})?\Z")


def _int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _stamp(now=None) -> str:
    moment = now or datetime.now(timezone.utc)
    if isinstance(moment, datetime):
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return str(moment)


def record_run(runs, *, commit, tests_passed, tests_failed=0,
               benchmark_passed=0, benchmark_total=0, aisc_passed=0,
               aisc_total=0, note="", now=None) -> dict:
    """Append one content-free measured run to ``runs``."""
    if not isinstance(runs, list):
        raise ValueError("runs must be a list")
    commit = str(commit or "").strip().lower()
    if not _COMMIT_RE.fullmatch(commit):
        raise ValueError("commit must be a 7-40 char lowercase hex prefix")
    values = {
        "tests_passed": _int(tests_passed, "tests_passed"),
        "tests_failed": _int(tests_failed, "tests_failed"),
        "benchmark_passed": _int(benchmark_passed, "benchmark_passed"),
        "benchmark_total": _int(benchmark_total, "benchmark_total"),
        "aisc_passed": _int(aisc_passed, "aisc_passed"),
        "aisc_total": _int(aisc_total, "aisc_total"),
    }
    if values["benchmark_passed"] > values["benchmark_total"]:
        raise ValueError("benchmark_passed cannot exceed benchmark_total")
    if values["aisc_passed"] > values["aisc_total"]:
        raise ValueError("aisc_passed cannot exceed aisc_total")
    if not isinstance(note, str):
        raise ValueError("note must be a string")
    trimmed = note.strip()
    record = {
        "version": VERSION,
        "at": _stamp(now),
        "commit": commit,
        **values,
        "note_id": (
            hashlib.sha256(trimmed.encode("utf-8")).hexdigest()[:16]
            if trimmed else ""),
    }
    runs.append(record)
    return record


def _validate(runs) -> tuple[list[str], list[dict]]:
    errors = []
    valid = []
    if not isinstance(runs, list):
        return ["runs must be a list"], []
    for index, entry in enumerate(runs):
        where = f"entry {index}"
        if not isinstance(entry, dict):
            errors.append(f"{where}: must be a mapping")
            continue
        forbidden = sorted(set(entry) & FORBIDDEN_KEYS)
        if forbidden:
            errors.append(
                f"{where}: content key(s) not allowed: " + ", ".join(forbidden))
        unknown = sorted(set(entry) - RECORD_KEYS)
        if unknown:
            errors.append(f"{where}: unknown key(s): " + ", ".join(unknown))
        if entry.get("version") != VERSION:
            errors.append(f"{where}: version must be {VERSION}")
        if not isinstance(entry.get("at"), str) or not entry["at"].strip():
            errors.append(f"{where}: at must be a non-empty string")
        if not isinstance(entry.get("commit"), str) or not _COMMIT_RE.fullmatch(
                entry["commit"]):
            errors.append(f"{where}: invalid commit")
        for key in ("tests_passed", "tests_failed", "benchmark_passed",
                    "benchmark_total", "aisc_passed", "aisc_total"):
            value = entry.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                errors.append(f"{where}: {key} must be a non-negative integer")
        if (isinstance(entry.get("benchmark_passed"), int)
                and isinstance(entry.get("benchmark_total"), int)
                and entry["benchmark_passed"] > entry["benchmark_total"]):
            errors.append(f"{where}: benchmark_passed exceeds total")
        if (isinstance(entry.get("aisc_passed"), int)
                and isinstance(entry.get("aisc_total"), int)
                and entry["aisc_passed"] > entry["aisc_total"]):
            errors.append(f"{where}: aisc_passed exceeds total")
        note_id = entry.get("note_id")
        if not isinstance(note_id, str) or not _NOTE_RE.fullmatch(note_id):
            errors.append(f"{where}: note_id must be empty or lowercase hex")
        if not errors:
            valid.append(entry)
    return errors, runs


def verify_runs(runs) -> dict:
    """Validate a scoreboard without mutating it."""
    errors, items = _validate(runs)
    return {"ok": not errors, "entries": len(items), "errors": errors}


def summarize_runs(runs) -> dict:
    """Report the latest measured run and the delta from the previous run."""
    errors, _ = _validate(runs)
    if errors:
        raise ValueError("invalid scoreboard: " + "; ".join(errors))
    ordered = sorted(runs, key=lambda item: (item["at"], item["commit"]))
    if not ordered:
        return {
            "runs": 0,
            "latest": None,
            "previous": None,
            "delta": {},
            "green": False,
            "weights_changed": False,
            "method": "content_free_scoreboard_v1",
        }
    latest = ordered[-1]
    previous = ordered[-2] if len(ordered) > 1 else None
    delta = {}
    if previous is not None:
        for key in ("tests_passed", "tests_failed", "benchmark_passed",
                    "aisc_passed"):
            delta[key] = latest[key] - previous[key]
    return {
        "runs": len(ordered),
        "latest": latest,
        "previous": previous,
        "delta": delta,
        "green": latest["tests_failed"] == 0,
        "weights_changed": False,
        "method": "content_free_scoreboard_v1",
    }


def load_jsonl(path) -> list[dict]:
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


def write_jsonl(path, runs) -> None:
    report = verify_runs(runs)
    if not report["ok"]:
        raise ValueError("invalid scoreboard: " + "; ".join(report["errors"]))
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(
        json.dumps(item, ensure_ascii=False, sort_keys=True,
                   separators=(",", ":")) + "\n"
        for item in runs)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        os.replace(tmp_name, target)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def _render(summary: dict) -> str:
    if not summary["runs"]:
        return "Scoreboard: no runs"
    latest = summary["latest"]
    lines = [
        f"Scoreboard: {summary['runs']} run(s); latest {latest['commit']} "
        f"at {latest['at']}",
        f"- tests: {latest['tests_passed']} passed / "
        f"{latest['tests_failed']} failed",
        f"- benchmark: {latest['benchmark_passed']}/"
        f"{latest['benchmark_total']}",
        f"- AISC: {latest['aisc_passed']}/{latest['aisc_total']}",
        f"- green: {summary['green']}",
    ]
    if summary["delta"]:
        lines.append("- delta: " + ", ".join(
            f"{key}={value:+d}" for key, value in summary["delta"].items()))
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, default=Path("data/scoreboard.jsonl"))
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--commit", default="")
    parser.add_argument("--tests-passed", type=int, default=0)
    parser.add_argument("--tests-failed", type=int, default=0)
    parser.add_argument("--benchmark-passed", type=int, default=0)
    parser.add_argument("--benchmark-total", type=int, default=0)
    parser.add_argument("--aisc-passed", type=int, default=0)
    parser.add_argument("--aisc-total", type=int, default=0)
    parser.add_argument("--note", default="")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        runs = load_jsonl(args.file)
        if args.verify:
            result = verify_runs(runs)
            if args.json:
                print(json.dumps(result, indent=2, ensure_ascii=False))
            else:
                print("scoreboard verify: " + (
                    "OK" if result["ok"] else "; ".join(result["errors"])))
            return 0 if result["ok"] else 2
        if args.record:
            record_run(
                runs, commit=args.commit, tests_passed=args.tests_passed,
                tests_failed=args.tests_failed,
                benchmark_passed=args.benchmark_passed,
                benchmark_total=args.benchmark_total,
                aisc_passed=args.aisc_passed, aisc_total=args.aisc_total,
                note=args.note)
            write_jsonl(args.file, runs)
        summary = summarize_runs(runs)
    except (OSError, ValueError) as exc:
        print(f"input error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(_render(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
