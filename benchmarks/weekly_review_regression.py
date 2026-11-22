"""End-to-end HTTP/UI regression for the weekly review.

Checks the served UI controls, then creates open/completed tasks and requests
the weekly review over the real handler. Verifies tasks, gaps, recent sources,
graph highlights and honest no-weight behavior.

Run:

    python benchmarks/weekly_review_regression.py
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
from atomzero.knowledge_gaps import KnowledgeGapStore


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

    with tempfile.TemporaryDirectory(prefix="atomzero-weekly-") as folder:
        docs = DocMemory(Path(folder) / "docs.db")
        docs.add_text("notes/launch.txt", "The launch code is L-7781.",
                      title="Launch")
        gaps = KnowledgeGapStore(Path(folder) / "gaps.json")
        gaps.add("What is the warranty claim deadline?")
        old = dict(chat_server._state)
        chat_server._state.update({
            "model": None, "tok": None, "docs": docs, "adapter": "",
            "name": "weekly-test", "params": 1, "config": {}, "step": 0,
            "api_key": "", "metrics": {"requests": 0, "generated_tokens": 0},
            "safety": None, "wiki_lang": "en", "gaps": gaps,
            "session_dir": str(Path(folder) / "sessions"), "sessions": {},
            "generator": None, "allow_general_chat": False, "skills": None,
        })
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), chat_server.Handler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            with urllib.request.urlopen(base + "/", timeout=15) as response:
                html = response.read().decode("utf-8", "replace")
            controls_ok = (
                "/v1/weekly-review" in html
                and 'id="briefweekly"' in html
                and "getWeeklyReview" in html)
            check("ui_weekly_controls", controls_ok,
                  "present" if controls_ok else "missing")
            _post(base, "/v1/agent/chat", {
                "session_id": "weekly-1",
                "message": "add task buy milk"})
            _post(base, "/v1/agent/chat", {
                "session_id": "weekly-1",
                "message": "add task write report"})
            _post(base, "/v1/agent/chat", {
                "session_id": "weekly-1",
                "message": "complete task 1"})
            review = _post(base, "/v1/weekly-review", {
                "session_id": "weekly-1", "days": 7})
            counts = review.get("task_counts", {})
            check(
                "weekly_task_counts",
                counts.get("open") == 1
                and counts.get("completed_recent") == 1
                and counts.get("completed_total") == 1,
                str(counts))
            check(
                "weekly_gaps",
                review.get("pending_gap_count") == 1,
                str(review.get("pending_gap_count")))
            check(
                "weekly_sources_and_graph",
                review.get("recent_source_count") == 1
                and "L-7781" in {
                    node["name"]
                    for node in review.get("graph", {}).get("entities", [])},
                str(review.get("recent_sources")))
            text = review.get("text", "")
            check(
                "weekly_text",
                "Weekly review" in text
                and "Completed this week" in text
                and "buy milk" in text
                and "write report" in text,
                text[:200])
            check(
                "weekly_no_weights",
                review.get("weights_changed") is False,
                str(review.get("weights_changed")))
        finally:
            httpd.shutdown()
            httpd.server_close()
            chat_server._state.clear()
            chat_server._state.update(old)
            docs.close()

    failed = sum(1 for item in checks if not item["ok"])
    payload = {
        "benchmark": "atomzero-weekly-review-regression",
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
        "--out", default="benchmarks/results/weekly_review_regression.json")
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
