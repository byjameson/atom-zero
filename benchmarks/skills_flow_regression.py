"""End-to-end HTTP/chat regression for custom skills.

Covers:
- provider skill: local context is rendered and the fake provider is called;
- template skill: no provider call, rendered prompt is returned;
- no-provider fallback: an explicit message plus local sources;
- GET /v1/skills lists the loaded skills.

Run:

    python benchmarks/skills_flow_regression.py
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
from atomzero.skills import SkillRegistry


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


def _write_skill(folder: Path, name: str, triggers: list[str],
                 prompt: str, provider: bool) -> None:
    (folder / f"{name}.json").write_text(json.dumps({
        "name": name,
        "description": f"{name} regression skill",
        "triggers": triggers,
        "prompt": prompt,
        "use_context": True,
        "max_chunks": 3,
        "provider": provider,
    }), encoding="utf-8")


def run(output: str | Path | None = None) -> dict:
    checks = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    with tempfile.TemporaryDirectory(prefix="atomzero-skills-") as folder:
        skills_dir = Path(folder) / "skills"
        skills_dir.mkdir()
        _write_skill(
            skills_dir, "provider-skill", ["provider skill"],
            "Provider prompt for {query}.\nContext:\n{context}", True)
        _write_skill(
            skills_dir, "template-skill", ["template skill"],
            "Template result for {query}.\nContext:\n{context}", False)
        _write_skill(
            skills_dir, "fallback-skill", ["fallback skill"],
            "Fallback prompt for {query}.\nContext:\n{context}", True)
        registry = SkillRegistry.from_dir(skills_dir)

        docs = DocMemory(Path(folder) / "docs.db")
        docs.add_text("notes/launch.txt", "The launch code is L-7781.",
                      title="Launch")
        prompts = []

        def fake_generator(prompt: str) -> str:
            prompts.append(prompt)
            return "Provider answer."

        old = dict(chat_server._state)
        chat_server._state.update({
            "model": None, "tok": None, "docs": docs, "adapter": "",
            "name": "skills-test", "params": 1, "config": {}, "step": 0,
            "api_key": "", "metrics": {"requests": 0, "generated_tokens": 0},
            "safety": None, "wiki_lang": "en",
            "session_dir": str(Path(folder) / "sessions"), "sessions": {},
            "generator": fake_generator, "allow_general_chat": True,
            "skills": registry, "skills_dir": str(skills_dir),
        })
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), chat_server.Handler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            listed = _get(base, "/v1/skills")
            check(
                "skills_listed",
                {item["name"] for item in listed.get("data", [])} == {
                    "provider-skill", "template-skill", "fallback-skill"},
                str([item["name"] for item in listed.get("data", [])]))

            provider = _post(base, "/v1/agent/chat", {
                "session_id": "provider-1",
                "message": "provider skill launch code"})
            check(
                "provider_reply",
                provider.get("reply") == "Provider answer.",
                provider.get("reply", "")[:120])
            check(
                "provider_context",
                prompts and "L-7781" in prompts[0]
                and "Provider prompt" in prompts[0],
                prompts[0][:160] if prompts else "no provider call")
            check(
                "provider_tool_and_source",
                (provider.get("tool_calls") or [{}])[0].get("name")
                == "skill"
                and (provider.get("sources") or [{}])[0].get("source")
                == "notes/launch.txt",
                str(provider.get("sources")))

            calls_before = len(prompts)
            template = _post(base, "/v1/agent/chat", {
                "session_id": "template-1",
                "message": "template skill launch code"})
            check(
                "template_reply",
                template.get("reply", "").startswith("Template result")
                and "L-7781" in template.get("reply", ""),
                template.get("reply", "")[:160])
            check(
                "template_does_not_call_provider",
                len(prompts) == calls_before,
                f"{len(prompts)} provider call(s)")

            chat_server._state["generator"] = None
            fallback = _post(base, "/v1/agent/chat", {
                "session_id": "fallback-1",
                "message": "fallback skill launch code"})
            check(
                "fallback_reply",
                "needs a local provider" in fallback.get("reply", ""),
                fallback.get("reply", "")[:160])
            check(
                "fallback_source",
                (fallback.get("sources") or [{}])[0].get("source")
                == "notes/launch.txt",
                str(fallback.get("sources")))
            check(
                "fallback_does_not_call_provider",
                len(prompts) == calls_before,
                f"{len(prompts)} provider call(s)")
        finally:
            httpd.shutdown()
            httpd.server_close()
            chat_server._state.clear()
            chat_server._state.update(old)
            docs.close()

    failed = sum(1 for item in checks if not item["ok"])
    payload = {
        "benchmark": "atomzero-skills-flow-regression",
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
        "--out", default="benchmarks/results/skills_flow_regression.json")
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
