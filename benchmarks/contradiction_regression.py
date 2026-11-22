"""End-to-end HTTP/UI regression for source comparison.

Checks the served compare controls, numeric/negation conflicts, identifier
conflicts and a no-conflict comparison.

Run:

    python benchmarks/contradiction_regression.py
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


def _get(base: str, route: str) -> dict:
    with urllib.request.urlopen(base + route, timeout=15) as response:
        return json.loads(response.read().decode("utf-8"))


def run(output: str | Path | None = None) -> dict:
    checks = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    with tempfile.TemporaryDirectory(prefix="atomzero-compare-") as folder:
        docs = DocMemory(Path(folder) / "docs.db")
        docs.add_text("policy_a.txt",
                      "The warranty period is 24 months. "
                      "The warranty covers water damage.", title="Policy A")
        docs.add_text("policy_b.txt",
                      "The warranty period is 36 months. "
                      "The warranty does not cover water damage.",
                      title="Policy B")
        docs.add_text("launch_a.txt", "The launch code is L-7781.",
                      title="Launch A")
        docs.add_text("launch_b.txt", "The launch code is W-7781.",
                      title="Launch B")
        docs.add_text("unrelated.txt", "Bananas are yellow.",
                      title="Unrelated")
        old = dict(chat_server._state)
        chat_server._state.update({
            "model": None, "tok": None, "docs": docs, "adapter": "",
            "name": "compare-test", "params": 1, "config": {}, "step": 0,
            "api_key": "", "metrics": {"requests": 0, "generated_tokens": 0},
            "safety": None, "wiki_lang": "en",
            "session_dir": str(Path(folder) / "sessions"), "sessions": {},
        })
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), chat_server.Handler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            with urllib.request.urlopen(base + "/", timeout=15) as response:
                html = response.read().decode("utf-8", "replace")
            controls_ok = (
                "/v1/documents/compare" in html
                and 'id="cmpbtn"' in html
                and "compareSources" in html)
            check("ui_compare_controls", controls_ok,
                  "present" if controls_ok else "missing")
            policy = _get(
                base,
                "/v1/documents/compare?source=policy_a.txt"
                "&against=policy_b.txt")
            reasons = {reason for conflict in policy["conflicts"]
                       for reason in conflict["reasons"]}
            check(
                "policy_conflict_reasons",
                "numeric_mismatch" in reasons
                and "negation_mismatch" in reasons,
                str(sorted(reasons)))
            snippets = " ".join(
                side["text"]
                for conflict in policy["conflicts"]
                for side in (conflict["left"], conflict["right"]))
            check(
                "policy_conflict_snippets",
                "24 months" in snippets and "36 months" in snippets
                and "does not cover" in snippets,
                snippets[:200])
            code = _get(
                base,
                "/v1/documents/compare?source=launch_a.txt"
                "&against=launch_b.txt")
            check(
                "identifier_conflict",
                any("identifier_mismatch" in conflict["reasons"]
                    for conflict in code["conflicts"]),
                str(code["summary"]))
            clean = _get(
                base,
                "/v1/documents/compare?source=unrelated.txt"
                "&against=policy_a.txt")
            check(
                "no_false_positive",
                clean["conflicts"] == []
                and clean["summary"]["conflict_count"] == 0,
                str(clean["summary"]))
        finally:
            httpd.shutdown()
            httpd.server_close()
            chat_server._state.clear()
            chat_server._state.update(old)
            docs.close()

    failed = sum(1 for item in checks if not item["ok"])
    payload = {
        "benchmark": "atomzero-contradiction-regression",
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
        "--out", default="benchmarks/results/contradiction_regression.json")
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
