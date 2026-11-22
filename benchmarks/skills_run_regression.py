"""End-to-end HTTP regression for explicit skill execution by name.

Covers `POST /v1/skills/run` for:
- provider mode: local context + provider call;
- template mode: no provider call;
- no-provider fallback: explicit message + local sources.

Run:

    python benchmarks/skills_run_regression.py
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


def _post_status(base: str, route: str, payload: dict) -> tuple[dict, int]:
    try:
        return _post(base, route, payload), 200
    except urllib.error.HTTPError as exc:
        try:
            data = json.loads(exc.read().decode("utf-8"))
        except Exception:
            data = {"error": str(exc)}
        return data, exc.code


def _write_skill(folder: Path, name: str, prompt: str,
                 provider: bool) -> None:
    (folder / f"{name}.json").write_text(json.dumps({
        "name": name,
        "triggers": [name.replace("-", " ")],
        "prompt": prompt,
        "use_context": True,
        "max_chunks": 2,
        "provider": provider,
    }), encoding="utf-8")


def run(output: str | Path | None = None) -> dict:
    checks = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    with tempfile.TemporaryDirectory(prefix="atomzero-skill-run-") as folder:
        skills_dir = Path(folder) / "skills"
        skills_dir.mkdir()
        _write_skill(skills_dir, "provider-skill",
                     "Provider: {context} {query}", True)
        _write_skill(skills_dir, "template-skill",
                     "Template: {context}", False)
        registry = SkillRegistry.from_dir(skills_dir)
        docs = DocMemory(Path(folder) / "docs.db")
        docs.add_text("notes/launch.txt", "The launch code is L-7781.",
                      title="Launch")
        calls = []

        def generator(prompt):
            calls.append(prompt)
            return "Provider answer."

        old = dict(chat_server._state)
        chat_server._state.update({
            "model": None, "tok": None, "docs": docs, "adapter": "",
            "name": "skill-run-test", "params": 1, "config": {}, "step": 0,
            "api_key": "", "metrics": {"requests": 0, "generated_tokens": 0},
            "safety": None, "wiki_lang": "en",
            "session_dir": str(Path(folder) / "sessions"), "sessions": {},
            "generator": generator, "allow_general_chat": True,
            "skills": registry, "skills_dir": str(skills_dir),
        })
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), chat_server.Handler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            provider = _post(base, "/v1/skills/run", {
                "name": "provider-skill", "query": "launch code"})
            check(
                "provider_reply",
                provider.get("reply") == "Provider answer."
                and provider.get("provider_called") is True,
                provider.get("reply", "")[:120])
            check(
                "provider_context",
                calls and "L-7781" in calls[0],
                calls[0][:160] if calls else "no provider call")
            check(
                "provider_sources",
                (provider.get("sources") or [{}])[0].get("source")
                == "notes/launch.txt",
                str(provider.get("sources")))

            before = len(calls)
            template = _post(base, "/v1/skills/run", {
                "name": "template-skill", "query": "launch code"})
            check(
                "template_reply",
                template.get("reply", "").startswith("Template:")
                and "L-7781" in template.get("reply", "")
                and template.get("provider_called") is False,
                template.get("reply", "")[:160])
            check(
                "template_no_provider",
                len(calls) == before,
                f"{len(calls)} provider call(s)")

            chat_server._state["generator"] = None
            fallback = _post(base, "/v1/skills/run", {
                "name": "provider-skill", "query": "launch code"})
            check(
                "fallback_reply",
                "needs a local provider" in fallback.get("reply", "")
                and fallback.get("provider_called") is False,
                fallback.get("reply", "")[:160])
            check(
                "fallback_sources",
                (fallback.get("sources") or [{}])[0].get("source")
                == "notes/launch.txt",
                str(fallback.get("sources")))
            check(
                "fallback_no_provider",
                len(calls) == before,
                f"{len(calls)} provider call(s)")
            _, status = _post_status(base, "/v1/skills/run", {
                "name": "missing-skill", "query": "x"})
            check(
                "unknown_name_is_404",
                status == 404,
                f"HTTP {status}")
        finally:
            httpd.shutdown()
            httpd.server_close()
            chat_server._state.clear()
            chat_server._state.update(old)
            docs.close()

    failed = sum(1 for item in checks if not item["ok"])
    payload = {
        "benchmark": "atomzero-skills-run-regression",
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
        "--out", default="benchmarks/results/skills_run_regression.json")
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
