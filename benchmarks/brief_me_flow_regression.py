"""End-to-end HTTP/chat regression for the `brief me` intent.

Starts the real chat handler, adds a task, then checks that the brief includes:
- the open task;
- a pending knowledge gap;
- document chunk/source counts;
- graph entities;
- the briefing tool call and source items.

Run:

    python benchmarks/brief_me_flow_regression.py
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

    with tempfile.TemporaryDirectory(prefix="atomzero-brief-me-") as folder:
        docs = DocMemory(Path(folder) / "docs.db")
        docs.add_text("notes/launch.txt", "The launch code is L-7781.",
                      title="Launch")
        docs.add_text("notes/warranty.txt",
                      "The warranty period is 24 months.", title="Warranty")
        gaps = KnowledgeGapStore(Path(folder) / "gaps.json")
        gaps.add("What is the warranty claim deadline?")
        old = dict(chat_server._state)
        chat_server._state.update({
            "model": None, "tok": None, "docs": docs, "adapter": "",
            "name": "brief-test", "params": 1, "config": {}, "step": 0,
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
            added = _post(base, "/v1/agent/chat", {
                "session_id": "brief-1",
                "message": "add task buy milk"})
            check(
                "task_added",
                "Added task" in added.get("reply", ""),
                added.get("reply", "")[:120])
            result = _post(base, "/v1/agent/chat", {
                "session_id": "brief-1", "message": "brief me"})
            brief = result.get("brief", {})
            reply = result.get("reply", "")
            check(
                "brief_me_reply",
                "Daily brief" in reply and "buy milk" in reply,
                reply[:160])
            check(
                "brief_task_count",
                brief.get("open_task_count") == 1,
                str(brief.get("open_task_count")))
            check(
                "brief_pending_gap",
                brief.get("pending_gap_count") == 1
                and "warranty claim deadline" in reply,
                str(brief.get("pending_gap_count")))
            check(
                "brief_index_status",
                brief.get("documents", {}).get("chunks") == 2
                and brief.get("documents", {}).get("sources") == 2
                and "Index: 2 chunk(s) / 2 source(s)" in reply,
                str(brief.get("documents")))
            names = {node.get("name")
                     for node in brief.get("graph", {}).get("entities", [])}
            check(
                "brief_graph_entities",
                "L-7781" in names and "Graph:" in reply,
                str(sorted(names)))
            check(
                "brief_tool_and_sources",
                (result.get("tool_calls") or [{}])[0].get("name")
                == "briefing"
                and bool(result.get("sources")),
                str(result.get("sources")))
        finally:
            httpd.shutdown()
            httpd.server_close()
            chat_server._state.clear()
            chat_server._state.update(old)
            docs.close()

    failed = sum(1 for item in checks if not item["ok"])
    payload = {
        "benchmark": "atomzero-brief-me-flow-regression",
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
        "--out", default="benchmarks/results/brief_me_flow_regression.json")
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
