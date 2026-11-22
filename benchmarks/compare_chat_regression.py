"""End-to-end HTTP/chat regression for source comparison.

Checks:
- chat "compare sources A and B" intent;
- numeric and negation reasons;
- both snippets and sources;
- tool-call/comparison payload;
- no-conflict and missing-source honesty.

Run:

    python benchmarks/compare_chat_regression.py
"""
from __future__ import annotations

import argparse
import json
import tempfile
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from atomzero import chat_server
from atomzero.doc_memory import DocMemory


def _post(base: str, route: str, payload: dict) -> dict:
    request = urllib.request.Request(
        base + route,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        return json.loads(response.read().decode("utf-8"))


def run(output: str | Path | None = None) -> dict:
    checks = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    with tempfile.TemporaryDirectory(prefix="atomzero-compare-chat-") as folder:
        docs = DocMemory(Path(folder) / "docs.db")
        docs.add_text("policy_a.txt",
                      "The warranty period is 24 months. "
                      "The warranty covers water damage.", title="Policy A")
        docs.add_text("policy_b.txt",
                      "The warranty period is 36 months. "
                      "The warranty does not cover water damage.",
                      title="Policy B")
        docs.add_text("unrelated.txt", "Bananas are yellow.",
                      title="Unrelated")
        old = dict(chat_server._state)
        chat_server._state.update({
            "model": None, "tok": None, "docs": docs, "adapter": "",
            "name": "compare-chat-test", "params": 1, "config": {}, "step": 0,
            "api_key": "", "metrics": {"requests": 0, "generated_tokens": 0},
            "safety": None, "wiki_lang": "en",
            "session_dir": str(Path(folder) / "sessions"), "sessions": {},
            "generator": None, "allow_general_chat": False, "skills": None,
        })
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), chat_server.Handler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            result = _post(base, "/v1/agent/chat", {
                "session_id": "compare-1",
                "message": "compare sources policy_a.txt and policy_b.txt"})
            reply = result.get("reply", "")
            check(
                "chat_compare_reasons",
                "numeric_mismatch" in reply
                and "negation_mismatch" in reply,
                reply[:200])
            check(
                "chat_compare_snippets",
                "24 months" in reply and "36 months" in reply
                and "does not cover" in reply,
                reply[:300])
            check(
                "chat_compare_payload",
                (result.get("tool_calls") or [{}])[0].get("name")
                == "compare_sources"
                and result.get("comparison", {}
                               ).get("summary", {}
                                     ).get("conflict_count", 0) >= 2,
                str(result.get("comparison", {}).get("summary")))
            check(
                "chat_compare_sources",
                len(result.get("sources", [])) == 2,
                str(result.get("sources")))
            clean = _post(base, "/v1/agent/chat", {
                "session_id": "compare-1",
                "message": "compare unrelated.txt and policy_a.txt"})
            check(
                "chat_compare_no_false_positive",
                "No conflicting" in clean.get("reply", ""),
                clean.get("reply", "")[:160])
            missing = _post(base, "/v1/agent/chat", {
                "session_id": "compare-1",
                "message": "compare missing.txt and policy_a.txt"})
            check(
                "chat_compare_missing_source",
                "source not found" in missing.get("reply", ""),
                missing.get("reply", "")[:160])
        finally:
            httpd.shutdown()
            httpd.server_close()
            chat_server._state.clear()
            chat_server._state.update(old)
            docs.close()

    failed = sum(1 for item in checks if not item["ok"])
    payload = {
        "benchmark": "atomzero-compare-chat-regression",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "checks": checks,
        "passed": len(checks) - failed,
        "failed": failed,
        "ok": failed == 0,
    }
    if output:
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", default="benchmarks/results/compare_chat_regression.json")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    payload = run(args.out)
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        for item in payload["checks"]:
            status = "OK" if item["ok"] else "FAIL"
            print(f"[{status}] {item['name']}: {item['detail']}")
        print(f"summary: {payload['passed']} passed, "
              f"{payload['failed']} failed")
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
