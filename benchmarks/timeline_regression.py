"""End-to-end regression for the document timeline.

Checks the served timeline route, date sorting, source filtering, year-only
precision, limits and missing-source behavior.

Run:

    python benchmarks/timeline_regression.py
"""
from __future__ import annotations

import argparse
import json
import tempfile
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from atomzero import chat_server
from atomzero.doc_memory import DocMemory


def _get(base: str, route: str) -> dict:
    with urllib.request.urlopen(base + route, timeout=15) as response:
        return json.loads(response.read().decode("utf-8"))


def _post(url: str, payload: dict) -> dict:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=15) as response:
        return json.loads(response.read().decode("utf-8"))


def run(output: str | Path | None = None) -> dict:
    checks = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    with tempfile.TemporaryDirectory(prefix="atomzero-timeline-") as folder:
        docs = DocMemory(Path(folder) / "docs.db")
        docs.add_text("history.txt",
                      "The launch was on 5 January 2020. "
                      "A follow-up happened in 2021.", title="History")
        docs.add_text("toplanti.txt",
                      "Toplantı 6 Ocak 2020 tarihinde yapıldı.",
                      title="Toplantı")
        docs.add_text("notes.txt", "Bananas are yellow.", title="Notes")
        old = dict(chat_server._state)
        chat_server._state.update({
            "model": None, "tok": None, "docs": docs, "adapter": "",
            "name": "timeline-test", "params": 1, "config": {}, "step": 0,
            "api_key": "", "metrics": {"requests": 0, "generated_tokens": 0},
            "safety": None, "wiki_lang": "en",
            "session_dir": str(Path(folder) / "sessions"), "sessions": {},
        })
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), chat_server.Handler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            timeline = _get(base, "/v1/documents/timeline")
            dates = [event["date"] for event in timeline["events"]]
            check("timeline_sorted",
                  dates == ["2020-01-05", "2020-01-06", "2021"],
                  str(dates))
            check("timeline_provenance",
                  timeline["events"][0]["source"] == "history.txt"
                  and timeline["events"][0]["title"] == "History"
                  and "launch" in timeline["events"][0]["text"].lower(),
                  str(timeline["events"][0]))
            check("timeline_year_precision",
                  timeline["events"][-1]["precision"] == "year",
                  str(timeline["events"][-1]))
            filtered = _get(
                base, "/v1/documents/timeline?source=toplanti.txt")
            check("timeline_source_filter",
                  filtered["count"] == 1
                  and filtered["events"][0]["date"] == "2020-01-06",
                  str(filtered["summary"] if "summary" in filtered
                      else filtered["count"]))
            limited = _get(base, "/v1/documents/timeline?limit=1")
            check("timeline_limit", len(limited["events"]) == 1,
                  str(limited["limit"]))
            try:
                _get(base, "/v1/documents/timeline?source=missing.txt")
                missing_ok = False
            except urllib.error.HTTPError as exc:
                missing_ok = exc.code == 404
            check("timeline_missing_source", missing_ok, "404")
            with urllib.request.urlopen(base + "/", timeout=15) as response:
                html = response.read().decode("utf-8", "replace").lower()
            check("ui_timeline_control",
                  "/v1/documents/timeline" in html
                  and 'id="timelinebtn"' in html
                  and "showtimeline" in html,
                  "present" if "timelinebtn" in html else "missing")
            chat = _post(base + "/v1/agent/chat", {
                "session_id": "timeline", "message": "timeline"})
            calls = chat.get("tool_calls") or [{}]
            check("chat_timeline",
                  calls[0].get("name") == "document_timeline"
                  and "2020-01-05" in chat.get("reply", ""),
                  chat.get("reply", "")[:160])
            turkish = _post(base + "/v1/agent/chat", {
                "session_id": "timeline", "message": "zaman çizelgesi"})
            turkish_calls = turkish.get("tool_calls") or [{}]
            check("chat_timeline_turkish",
                  turkish_calls[0].get("name") == "document_timeline",
                  turkish.get("reply", "")[:160])
        finally:
            httpd.shutdown()
            httpd.server_close()
            chat_server._state.clear()
            chat_server._state.update(old)
            docs.close()

    failed = sum(1 for item in checks if not item["ok"])
    payload = {
        "benchmark": "atomzero-timeline-regression",
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
        "--out", default="benchmarks/results/timeline_regression.json")
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
