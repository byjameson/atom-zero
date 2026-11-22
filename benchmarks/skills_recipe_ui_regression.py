"""UI-contract + HTTP regression for named recipe selection/execution.

Checks the served page contains the recipe select, Run recipe button and the
`/v1/skills/chains/run` request contract. Then executes the exact UI request
through the real handler: context hand-off, recipe metadata, carry override and
unknown recipe.

Run:

    python benchmarks/skills_recipe_ui_regression.py
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
from atomzero.skills import ChainRecipeRegistry, SkillRegistry


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

    with tempfile.TemporaryDirectory(prefix="atomzero-recipe-ui-") as folder:
        skills_dir = Path(folder) / "skills"
        skills_dir.mkdir()
        _write_skill(skills_dir, "provider-skill",
                     "Provider: {query} {context}", True)
        _write_skill(skills_dir, "chain-skill",
                     "Chain previous: {previous}", False)
        chains_dir = skills_dir / "chains"
        chains_dir.mkdir()
        (chains_dir / "demo-recipe.json").write_text(json.dumps({
            "name": "demo-recipe",
            "description": "provider then template",
            "steps": ["provider-skill", "chain-skill"],
            "carry_context": True,
        }), encoding="utf-8")
        skills = SkillRegistry.from_dir(skills_dir)
        chains = ChainRecipeRegistry.from_skills_dir(skills_dir)
        docs = DocMemory(Path(folder) / "docs.db")
        docs.add_text("notes/launch.txt", "The launch code is L-7781.",
                      title="Launch")

        def generator(prompt):
            return "Provider answer L-7781."

        old = dict(chat_server._state)
        chat_server._state.update({
            "model": None, "tok": None, "docs": docs, "adapter": "",
            "name": "recipe-ui-test", "params": 1, "config": {}, "step": 0,
            "api_key": "", "metrics": {"requests": 0, "generated_tokens": 0},
            "safety": None, "wiki_lang": "en",
            "session_dir": str(Path(folder) / "sessions"), "sessions": {},
            "generator": generator, "allow_general_chat": True,
            "skills": skills, "skills_dir": str(skills_dir),
            "chains": chains, "chains_dir": chains.directory,
        })
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), chat_server.Handler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            with urllib.request.urlopen(base + "/", timeout=15) as response:
                html = response.read().decode("utf-8", "replace")
            controls_ok = (
                'id="chainrecipe"' in html
                and 'id="chainreciperun"' in html
                and "Run recipe" in html
                and "runRecipe" in html)
            check("ui_recipe_controls", controls_ok,
                  "present" if controls_ok else "missing")
            contract_ok = (
                "/v1/skills/chains" in html
                and "/v1/skills/chains/run" in html
                and "name, query" in html)
            check("ui_recipe_contract", contract_ok,
                  "present" if contract_ok else "missing")

            listed = _get(base, "/v1/skills/chains")
            check(
                "ui_recipe_listed",
                listed.get("data", [{}])[0].get("name") == "demo-recipe",
                str(listed.get("data")))
            result = _post(base, "/v1/skills/chains/run", {
                "name": "demo-recipe", "query": "launch code"})
            check(
                "ui_recipe_context",
                result.get("reply", "").startswith("Chain previous:")
                and "Provider answer L-7781." in result.get("reply", ""),
                result.get("reply", "")[:160])
            check(
                "ui_recipe_metadata",
                result.get("recipe") == "demo-recipe"
                and [step["skill"] for step in result.get("steps", [])]
                == ["provider-skill", "chain-skill"]
                and result.get("provider_calls") == 1
                and (result.get("sources") or [{}])[0].get("source")
                == "notes/launch.txt",
                str(result.get("provider_calls")))
            no_carry = _post(base, "/v1/skills/chains/run", {
                "name": "demo-recipe", "query": "launch code",
                "carry_context": False})
            check(
                "ui_recipe_carry_override",
                "Provider answer" not in no_carry.get("reply", "")
                and no_carry.get("reply", "").startswith("Chain previous:"),
                no_carry.get("reply", "")[:160])
            _, status = _post_status(base, "/v1/skills/chains/run", {
                "name": "missing-recipe", "query": "launch code"})
            check("ui_recipe_unknown_404", status == 404, f"HTTP {status}")
        finally:
            httpd.shutdown()
            httpd.server_close()
            chat_server._state.clear()
            chat_server._state.update(old)
            docs.close()

    failed = sum(1 for item in checks if not item["ok"])
    payload = {
        "benchmark": "atomzero-skills-recipe-ui-regression",
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
        "--out",
        default="benchmarks/results/skills_recipe_ui_regression.json")
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
