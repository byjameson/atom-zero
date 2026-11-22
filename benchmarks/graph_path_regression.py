"""End-to-end HTTP/UI regression for multi-hop graph paths.

Checks:
- UI path controls/contract;
- /v1/graph/path multi-hop result;
- chat "how is X connected to Y?" intent;
- missing-entity path honesty.

Run:

    python benchmarks/graph_path_regression.py
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


def _get(base: str, route: str) -> dict:
    with urllib.request.urlopen(base + route, timeout=15) as response:
        return json.loads(response.read().decode("utf-8"))


def run(output: str | Path | None = None) -> dict:
    checks = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    with tempfile.TemporaryDirectory(prefix="atomzero-path-") as folder:
        docs = DocMemory(Path(folder) / "docs.db")
        docs.add_text("chain1.txt", "Ankara is in Turkey.", title="Chain 1")
        docs.add_text("chain2.txt",
                      "Turkey uses the launch code L-7781.", title="Chain 2")
        old = dict(chat_server._state)
        chat_server._state.update({
            "model": None, "tok": None, "docs": docs, "adapter": "",
            "name": "path-test", "params": 1, "config": {}, "step": 0,
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
            with urllib.request.urlopen(base + "/", timeout=15) as response:
                html = response.read().decode("utf-8", "replace")
            controls_ok = (
                "/v1/graph/path" in html
                and 'id="graphpathbtn"' in html
                and "showGraphPath" in html)
            check("ui_path_controls", controls_ok,
                  "present" if controls_ok else "missing")
            path = _get(
                base,
                "/v1/graph/path?source=Ankara&target=L-7781&max_depth=3")
            check(
                "api_multihop_path",
                path.get("found") is True
                and path.get("path") == ["Ankara", "Turkey", "L-7781"]
                and path.get("hops") == 2,
                str(path.get("path")))
            check(
                "api_path_edges",
                len(path.get("edges", [])) == 2
                and all(edge.get("sources") for edge in path["edges"]),
                str(path.get("edges")))
            chat = _post(base, "/v1/agent/chat", {
                "session_id": "path-1",
                "message": "how is Ankara connected to L-7781?"})
            check(
                "chat_path_intent",
                "Ankara → Turkey → L-7781" in chat.get("reply", "")
                and (chat.get("tool_calls") or [{}])[0].get("name")
                == "graph_path",
                chat.get("reply", "")[:200])
            check(
                "chat_path_sources",
                bool(chat.get("sources"))
                and chat.get("path", {}).get("hops") == 2,
                str(chat.get("sources")))
            missing = _get(
                base, "/v1/graph/path?source=Ankara&target=Missing")
            check(
                "missing_path_is_honest",
                missing.get("found") is False
                and missing.get("path") == []
                and "reason" in missing,
                missing.get("reason", ""))
        finally:
            httpd.shutdown()
            httpd.server_close()
            chat_server._state.clear()
            chat_server._state.update(old)
            docs.close()

    failed = sum(1 for item in checks if not item["ok"])
    payload = {
        "benchmark": "atomzero-graph-path-regression",
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
        "--out", default="benchmarks/results/graph_path_regression.json")
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
