"""End-to-end goal-planning regression over the real HTTP chat handler.

Checks:
- a plan request creates no tasks;
- the plan cites local evidence and asks for confirmation;
- one confirmation creates exactly three tasks;
- a second confirmation is a no-op;
- cancel leaves task memory unchanged;
- confirm after cancel is a no-op.

Run:

    python benchmarks/goal_flow_regression.py
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


def _brief(base: str, session_id: str) -> dict:
    return _post(base, "/v1/briefing", {"session_id": session_id})


def run(output: str | Path | None = None) -> dict:
    checks = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    with tempfile.TemporaryDirectory(prefix="atomzero-goal-") as folder:
        docs = DocMemory(Path(folder) / "docs.db")
        docs.add_text(
            "notes/warranty.txt",
            "The warranty period is 24 months. Claims need a receipt.",
            title="Warranty")
        gaps = KnowledgeGapStore(Path(folder) / "gaps.json")
        old = dict(chat_server._state)
        chat_server._state.update({
            "model": None, "tok": None, "docs": docs, "adapter": "",
            "name": "goal-test", "params": 1, "config": {}, "step": 0,
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
            plan = _post(base, "/v1/agent/chat", {
                "session_id": "goal-1",
                "message": "plan goal prepare the warranty page"})
            brief = _brief(base, "goal-1")
            check(
                "plan_creates_no_tasks",
                brief["open_task_count"] == 0,
                f"{brief['open_task_count']} task(s)")
            check(
                "plan_requests_confirmation",
                "Suggested plan" in plan.get("reply", "")
                and "confirm plan" in plan.get("reply", "").lower(),
                plan.get("reply", "")[:160])
            check(
                "plan_has_cited_context",
                (plan.get("sources") or [{}])[0].get("source")
                == "notes/warranty.txt",
                str(plan.get("sources")))

            confirmed = _post(base, "/v1/agent/chat", {
                "session_id": "goal-1", "message": "confirm plan"})
            brief = _brief(base, "goal-1")
            check(
                "confirm_creates_exactly_three",
                brief["open_task_count"] == 3
                and len(confirmed.get("created_tasks", [])) == 3,
                f"{brief['open_task_count']} task(s)")

            second = _post(base, "/v1/agent/chat", {
                "session_id": "goal-1", "message": "confirm plan"})
            brief = _brief(base, "goal-1")
            check(
                "second_confirm_is_noop",
                brief["open_task_count"] == 3
                and "no pending" in second.get("reply", "").lower(),
                second.get("reply", "")[:160])

            _post(base, "/v1/agent/chat", {
                "session_id": "cancel-1",
                "message": "plan goal prepare the warranty page"})
            cancelled = _post(base, "/v1/agent/chat", {
                "session_id": "cancel-1", "message": "cancel plan"})
            after_cancel = _post(base, "/v1/agent/chat", {
                "session_id": "cancel-1", "message": "confirm plan"})
            brief = _brief(base, "cancel-1")
            check(
                "cancel_leaves_memory_unchanged",
                brief["open_task_count"] == 0
                and "cancelled" in cancelled.get("reply", "").lower()
                and "no pending" in after_cancel.get("reply", "").lower(),
                f"{brief['open_task_count']} task(s)")
        finally:
            httpd.shutdown()
            httpd.server_close()
            chat_server._state.clear()
            chat_server._state.update(old)
            docs.close()

    failed = sum(1 for item in checks if not item["ok"])
    payload = {
        "benchmark": "atomzero-goal-flow-regression",
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
        "--out", default="benchmarks/results/goal_flow_regression.json")
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
