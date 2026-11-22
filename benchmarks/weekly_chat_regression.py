"""End-to-end HTTP/chat regression for the weekly review.

Creates tasks through chat, completes one, then asks for the weekly review.
Checks the tool call, counts, text sections, sources, graph, priority-first
chat commands and the 14-day Turkish variant.

Run:

    python benchmarks/weekly_chat_regression.py
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

    with tempfile.TemporaryDirectory(prefix="atomzero-weekly-chat-") as folder:
        docs = DocMemory(Path(folder) / "docs.db")
        docs.add_text("notes/launch.txt", "The launch code is L-7781.",
                      title="Launch")
        gaps = KnowledgeGapStore(Path(folder) / "gaps.json")
        gaps.add("What is the warranty claim deadline?")
        old = dict(chat_server._state)
        chat_server._state.update({
            "model": None, "tok": None, "docs": docs, "adapter": "",
            "name": "weekly-chat-test", "params": 1, "config": {}, "step": 0,
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
            for message in ("add task buy milk",
                            "add task write report",
                            "complete task 1"):
                _post(base, "/v1/agent/chat", {
                    "session_id": "weekly-chat", "message": message})
            result = _post(base, "/v1/agent/chat", {
                "session_id": "weekly-chat", "message": "weekly review"})
            review = result.get("review", {})
            counts = review.get("task_counts", {})
            check(
                "chat_weekly_tool",
                result.get("tool_calls") == [
                    {"name": "weekly_review", "days": 7,
                     "sort": "default"}],
                str(result.get("tool_calls")))
            check(
                "chat_weekly_tasks",
                counts.get("open") == 1
                and counts.get("completed_recent") == 1
                and counts.get("completed_total") == 1,
                str(counts))
            text = result.get("reply", "")
            check(
                "chat_weekly_sections",
                "Weekly review" in text
                and "Completed this week" in text
                and "write report" in text
                and "buy milk" in text
                and "What is the warranty claim deadline?" in text
                and "Recent sources: 1" in text
                and "Graph:" in text,
                text[:220])
            check(
                "chat_weekly_sources",
                review.get("recent_source_count") == 1
                and (result.get("sources") or [{}])[0].get("source")
                == "notes/launch.txt",
                str(result.get("sources")))
            check(
                "chat_weekly_honest",
                review.get("pending_gap_count") == 1
                and "L-7781" in {
                    node["name"]
                    for node in review.get("graph", {}).get("entities", [])}
                and result.get("weights_changed") is False,
                str(review.get("graph")))
            priority_review = _post(base, "/v1/agent/chat", {
                "session_id": "weekly-chat",
                "message": "weekly review priority"})
            priority_brief = _post(base, "/v1/agent/chat", {
                "session_id": "weekly-chat",
                "message": "brief me priority"})
            check(
                "chat_priority_sort",
                priority_review.get("review", {}).get("sort") == "priority"
                and priority_brief.get("brief", {}).get("sort") == "priority"
                and (priority_review.get("tool_calls") or [{}])[0].get("sort")
                == "priority"
                and "sort=priority" in priority_review.get("reply", "")
                and "sort=priority" in priority_brief.get("reply", ""),
                priority_review.get("reply", "")[:160])
            turkish = _post(base, "/v1/agent/chat", {
                "session_id": "weekly-chat",
                "message": "haftalık özet 14 gün"})
            check(
                "chat_weekly_days",
                (turkish.get("tool_calls") or [{}])[0].get("days") == 14
                and turkish.get("review", {}).get("period_days") == 14
                and "completed in the last 14 days"
                in turkish.get("reply", ""),
                turkish.get("reply", "")[:160])
        finally:
            httpd.shutdown()
            httpd.server_close()
            chat_server._state.clear()
            chat_server._state.update(old)
            docs.close()

    failed = sum(1 for item in checks if not item["ok"])
    payload = {
        "benchmark": "atomzero-weekly-chat-regression",
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
        "--out", default="benchmarks/results/weekly_chat_regression.json")
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
