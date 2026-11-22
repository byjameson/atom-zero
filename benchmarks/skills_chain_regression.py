"""End-to-end HTTP regression for multi-skill chains.

Checks explicit previous-output passing:
- provider -> template chain;
- one provider call;
- source union;
- no-carry mode;
- unknown skill returns 400.

Run:

    python benchmarks/skills_chain_regression.py
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

    with tempfile.TemporaryDirectory(prefix="atomzero-chain-") as folder:
        skills_dir = Path(folder) / "skills"
        skills_dir.mkdir()
        _write_skill(skills_dir, "provider-skill",
                     "Provider: {query} {context}", True)
        _write_skill(skills_dir, "chain-skill",
                     "Chain previous: {previous}", False)
        registry = SkillRegistry.from_dir(skills_dir)
        docs = DocMemory(Path(folder) / "docs.db")
        docs.add_text("notes/launch.txt", "The launch code is L-7781.",
                      title="Launch")
        calls = []

        def generator(prompt):
            calls.append(prompt)
            return "Provider answer L-7781."

        old = dict(chat_server._state)
        chat_server._state.update({
            "model": None, "tok": None, "docs": docs, "adapter": "",
            "name": "chain-test", "params": 1, "config": {}, "step": 0,
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
            chain = _post(base, "/v1/skills/chain", {
                "skills": ["provider-skill", "chain-skill"],
                "query": "launch code"})
            check(
                "chain_context_passed",
                chain.get("reply", "").startswith("Chain previous:")
                and "Provider answer L-7781." in chain.get("reply", ""),
                chain.get("reply", "")[:160])
            check(
                "chain_steps_and_provider_calls",
                [step["skill"] for step in chain.get("steps", [])]
                == ["provider-skill", "chain-skill"]
                and chain.get("provider_calls") == 1,
                str(chain.get("provider_calls")))
            check(
                "chain_source_union",
                (chain.get("sources") or [{}])[0].get("source")
                == "notes/launch.txt",
                str(chain.get("sources")))
            check(
                "chain_provider_step_flags",
                chain["steps"][0]["provider_called"] is True
                and chain["steps"][1]["provider_called"] is False,
                str([step["provider_called"]
                     for step in chain.get("steps", [])]),
            )
            no_carry = _post(base, "/v1/skills/chain", {
                "skills": ["provider-skill", "chain-skill"],
                "query": "launch code", "carry_context": False})
            check(
                "no_carry_does_not_leak",
                "Provider answer" not in no_carry.get("reply", "")
                and no_carry.get("reply", "").startswith("Chain previous:"),
                no_carry.get("reply", "")[:160])
            _, status = _post_status(base, "/v1/skills/chain", {
                "skills": ["missing-skill"], "query": "launch code"})
            check("unknown_chain_skill_400", status == 400,
                  f"HTTP {status}")
        finally:
            httpd.shutdown()
            httpd.server_close()
            chat_server._state.clear()
            chat_server._state.update(old)
            docs.close()

    failed = sum(1 for item in checks if not item["ok"])
    payload = {
        "benchmark": "atomzero-skills-chain-regression",
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
        "--out", default="benchmarks/results/skills_chain_regression.json")
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
