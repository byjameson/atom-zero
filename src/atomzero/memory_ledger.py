"""Append-only, content-free ledger for local memory mutations.

The ledger records *that* a document or session memory was changed, with
counts and scope, but never the document text, query, answer or session
content. It is the first runtime slice of the Memory Lifecycle Algebra and
complements the AISC ``AtomZeroMemoryLifecycle`` spec.

    python -m atomzero.memory_ledger --list
    python -m atomzero.memory_ledger --verify
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import uuid
from pathlib import Path

CONTENT_KEYS = {
    "text", "query", "answer", "chunk", "snippet", "content",
    "document", "prompt", "message", "fact", "episode",
}
REQUIRED_KEYS = {"id", "ts", "op", "scope", "target", "status"}


def _contains_content(value, path="") -> str:
    if isinstance(value, dict):
        for key, item in value.items():
            lowered = str(key).lower()
            if lowered in CONTENT_KEYS:
                return f"{path}.{key}" if path else str(key)
            found = _contains_content(item, f"{path}.{key}" if path else str(key))
            if found:
                return found
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found = _contains_content(item, f"{path}[{index}]")
            if found:
                return found
    return ""


class MemoryLedgerError(ValueError):
    pass


class MemoryLedger:
    """A small JSONL ledger with content-free event validation."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser()
        self._lock = threading.Lock()

    def append(self, op: str, scope: str, target: str = "", *,
               before=None, after=None, actor: str = "local",
               request_id: str = "", status: str = "committed",
               meta=None) -> dict:
        event = {
            "id": "m_" + uuid.uuid4().hex[:16],
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "op": str(op)[:80],
            "scope": str(scope)[:80],
            "target": str(target)[:512],
            "before": before or {},
            "after": after or {},
            "actor": str(actor)[:80],
            "request_id": str(request_id)[:120],
            "status": str(status)[:40],
            "meta": meta or {},
        }
        content = _contains_content(event)
        if content:
            raise MemoryLedgerError(
                f"ledger event contains content field: {content}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n"
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()
                try:
                    os.fsync(handle.fileno())
                except OSError:
                    pass
        return event

    def read(self, limit: int = 50, op: str = "",
             scope: str = "") -> list[dict]:
        if not self.path.exists():
            return []
        entries = []
        for line_no, line in enumerate(
                self.path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise MemoryLedgerError(
                    f"{self.path}:{line_no}: invalid JSON: {exc}") from exc
            if op and event.get("op") != op:
                continue
            if scope and event.get("scope") != scope:
                continue
            entries.append(event)
        return entries[-max(1, int(limit)):][::-1]

    def verify(self) -> dict:
        errors = []
        entries = []
        if self.path.exists():
            for line_no, line in enumerate(
                    self.path.read_text(encoding="utf-8").splitlines(),
                    start=1):
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as exc:
                    errors.append(f"line {line_no}: invalid JSON: {exc}")
                    continue
                missing = REQUIRED_KEYS - set(event)
                if missing:
                    errors.append(
                        f"line {line_no}: missing {', '.join(sorted(missing))}")
                content = _contains_content(event)
                if content:
                    errors.append(
                        f"line {line_no}: content field {content}")
                entries.append(event)
        return {
            "ok": not errors,
            "path": str(self.path),
            "events": len(entries),
            "errors": errors,
        }

    def summary(self) -> dict:
        entries = self.read(limit=1_000_000)
        by_op = {}
        by_scope = {}
        for event in entries:
            op = str(event.get("op", ""))
            scope = str(event.get("scope", ""))
            by_op[op] = by_op.get(op, 0) + 1
            by_scope[scope] = by_scope.get(scope, 0) + 1
        return {
            "events": len(entries),
            "by_op": by_op,
            "by_scope": by_scope,
            "path": str(self.path),
        }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", default="data/memory_ledger.jsonl")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    ledger = MemoryLedger(args.path)
    if args.verify:
        result = ledger.verify()
    elif args.list:
        result = {"data": ledger.read(args.limit),
                  "summary": ledger.summary()}
    else:
        result = ledger.summary()
    print(json.dumps(result, indent=2, ensure_ascii=False) if args.json
          else result)
    return 0 if (not args.verify or result["ok"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
