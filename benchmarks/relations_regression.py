"""End-to-end regression for typed relation extraction.

Checks the served relations route, relation types, entity/relation filters,
provenance, a control sentence without a relation, and missing-source behavior.

Run:

    python benchmarks/relations_regression.py
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

    with tempfile.TemporaryDirectory(prefix="atomzero-relations-") as folder:
        docs = DocMemory(Path(folder) / "docs.db")
        docs.add_text("geo.txt",
                      "Ankara is the capital of Turkey. "
                      "Ankara is in Turkey.", title="Geography")
        docs.add_text("company.txt",
                      "Acme was founded by Jane Doe. "
                      "Jane Doe is the CEO of Acme.", title="Company")
        docs.add_text("life.txt", "Jane Doe was born in Istanbul.",
                      title="Life")
        docs.add_text("tr.txt",
                      "Ankara, Türkiye'nin başkentidir. "
                      "Acme, Jane Doe tarafından kuruldu.",
                      title="Türkçe")
        docs.add_text("control.txt",
                      "Ankara and Turkey are mentioned together.",
                      title="Control")
        old = dict(chat_server._state)
        chat_server._state.update({
            "model": None, "tok": None, "docs": docs, "adapter": "",
            "name": "relations-test", "params": 1, "config": {}, "step": 0,
            "api_key": "", "metrics": {"requests": 0, "generated_tokens": 0},
            "safety": None, "wiki_lang": "en",
            "session_dir": str(Path(folder) / "sessions"), "sessions": {},
        })
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), chat_server.Handler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            all_relations = _get(base, "/v1/graph/relations")
            names = {item["relation"]
                     for item in all_relations["relations"]}
            check("relation_types",
                  {"capital_of", "located_in", "founded_by", "ceo_of",
                   "born_in"} <= names,
                  str(sorted(names)))
            entity = _get(
                base, "/v1/graph/relations?entity=Ankara")
            check("entity_filter",
                  entity["count"] >= 2
                  and all("ankara" in (
                      item["subject"] + " " + item["object"]).casefold()
                      for item in entity["relations"]),
                  str(entity["count"]))
            capitals = _get(
                base, "/v1/graph/relations?relation=capital_of")
            check("relation_filter",
                  capitals["count"] >= 1
                  and all(item["relation"] == "capital_of"
                          for item in capitals["relations"]),
                  str(capitals["count"]))
            first = all_relations["relations"][0]
            check("provenance",
                  bool(first["source"]) and bool(first["text"])
                  and bool(first["entities"]),
                  str(first))
            control = [item for item in all_relations["relations"]
                       if item["source"] == "control.txt"]
            check("control_no_false_positive", control == [],
                  str(control))
            try:
                _get(base,
                     "/v1/graph/relations?source=missing.txt")
                missing_ok = False
            except urllib.error.HTTPError as exc:
                missing_ok = exc.code == 404
            check("missing_source", missing_ok, "404")
            with urllib.request.urlopen(base + "/", timeout=15) as response:
                html = response.read().decode("utf-8", "replace").lower()
            check("ui_relations_control",
                  "/v1/graph/relations" in html
                  and 'id="relationsbtn"' in html
                  and "showrelations" in html,
                  "present" if "relationsbtn" in html else "missing")
            chat = _post(base + "/v1/agent/chat", {
                "session_id": "relations",
                "message": "relations for Ankara"})
            calls = chat.get("tool_calls") or [{}]
            check("chat_relations",
                  calls[0].get("name") == "typed_relations"
                  and "capital_of" in chat.get("reply", ""),
                  chat.get("reply", "")[:160])
            turkish = _post(base + "/v1/agent/chat", {
                "session_id": "relations",
                "message": "Ankara ilişkileri"})
            turkish_calls = turkish.get("tool_calls") or [{}]
            check("chat_relations_turkish",
                  turkish_calls[0].get("name") == "typed_relations",
                  turkish.get("reply", "")[:160])
        finally:
            httpd.shutdown()
            httpd.server_close()
            chat_server._state.clear()
            chat_server._state.update(old)
            docs.close()

    failed = sum(1 for item in checks if not item["ok"])
    payload = {
        "benchmark": "atomzero-relations-regression",
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
        "--out", default="benchmarks/results/relations_regression.json")
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
