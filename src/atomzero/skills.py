"""Declarative local skills for shaping Atom Zero without editing core code.

A skill is a JSON file in ``data/skills`` (or another directory):

    {
      "name": "daily-brief",
      "description": "Turn local context into a short daily brief.",
      "triggers": ["daily brief", "brief me"],
      "system": "You are a concise local planning assistant.",
      "prompt": "Use the local context below...\\n\\nContext:\\n{context}\\n\\nRequest: {query}",
      "use_context": true,
      "max_chunks": 4,
      "provider": true
    }

Skills are declarative: they cannot execute arbitrary code. They can read local
retrieval context and call the configured provider. Users remain responsible
for the skills they write and for the provider they point at.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from atomzero import __version__
from atomzero.doc_memory import best_snippet, format_citation


class SkillError(ValueError):
    pass


def _validate_skill(data, path: Path) -> dict:
    if not isinstance(data, dict):
        raise SkillError(f"{path.name}: skill must be a JSON object")
    name = str(data.get("name", "")).strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name):
        raise SkillError(f"{path.name}: name must match [A-Za-z0-9_-]{{1,64}}")
    triggers = data.get("triggers")
    if not isinstance(triggers, list) or not triggers:
        raise SkillError(f"{path.name}: triggers must be a non-empty list")
    compiled = []
    clean_triggers = []
    for trigger in triggers:
        trigger = str(trigger or "").strip()
        if len(trigger) < 2 or len(trigger) > 120:
            raise SkillError(
                f"{path.name}: each trigger must be 2-120 chars")
        if trigger.lower().startswith("re:"):
            pattern = trigger[3:].strip()
            try:
                compiled.append(re.compile(pattern, re.IGNORECASE))
            except re.error as exc:
                raise SkillError(
                    f"{path.name}: invalid regex trigger: {exc}") from exc
        else:
            compiled.append(re.compile(re.escape(trigger), re.IGNORECASE))
        clean_triggers.append(trigger)
    prompt = str(data.get("prompt") or "").strip()
    if not prompt:
        raise SkillError(f"{path.name}: prompt is required")
    try:
        max_chunks = max(1, min(int(data.get("max_chunks", 4)), 12))
    except (TypeError, ValueError) as exc:
        raise SkillError(
            f"{path.name}: max_chunks must be an integer") from exc
    return {
        "name": name,
        "description": str(data.get("description") or "")[:300],
        "triggers": clean_triggers,
        "prompt": prompt,
        "system": str(data.get("system") or "")[:1000],
        "use_context": bool(data.get("use_context", True)),
        "max_chunks": max_chunks,
        "provider": bool(data.get("provider", True)),
        "enabled": bool(data.get("enabled", True)),
        "path": str(path),
        "_compiled": compiled,
    }


class SkillRegistry:
    def __init__(self, skills: list[dict] | None = None,
                 directory: str = ""):
        self.skills = skills or []
        self.directory = str(directory)

    @classmethod
    def from_dir(cls, directory: str | Path) -> "SkillRegistry":
        folder = Path(directory).expanduser()
        if not folder.exists():
            return cls([], str(folder))
        if not folder.is_dir():
            raise SkillError(f"skills path is not a directory: {folder}")
        skills = []
        names = set()
        for path in sorted(folder.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise SkillError(f"{path.name}: {exc}") from exc
            skill = _validate_skill(data, path)
            if skill["name"] in names:
                raise SkillError(f"duplicate skill name: {skill['name']}")
            names.add(skill["name"])
            if skill["enabled"]:
                skills.append(skill)
        return cls(skills, str(folder))

    def match(self, text: str) -> dict | None:
        value = str(text or "")
        best = None
        best_length = 0
        for skill in self.skills:
            for pattern in skill["_compiled"]:
                match = pattern.search(value)
                if match and len(match.group(0)) > best_length:
                    best = skill
                    best_length = len(match.group(0))
        return best

    def list(self) -> list[dict]:
        return [{key: value for key, value in skill.items()
                 if not key.startswith("_")}
                for skill in self.skills]


def render_skill(skill: dict, query: str, hits: list[dict],
                 max_chars: int = 6000) -> str:
    context = []
    for hit in hits[:int(skill.get("max_chunks", 4))]:
        context.append(
            f"- {best_snippet(query, hit.get('chunk', ''))} "
            f"(source: {format_citation(hit)})")
    block = "\n".join(context) if context else "(no local context found)"
    prompt = str(skill.get("prompt", ""))
    prompt = prompt.replace("{query}", str(query))
    prompt = prompt.replace("{context}", block)
    return prompt[:max_chars]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="atomzero-skills",
        description="List and validate declarative Atom Zero skills.")
    parser.add_argument("--version", action="version",
                        version=f"atomzero-skills {__version__}")
    parser.add_argument("--dir", default="data/skills")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="list enabled skills")
    sub.add_parser("validate", help="validate every skill file")
    show = sub.add_parser("show", help="show one skill")
    show.add_argument("name")
    args = parser.parse_args(argv)
    try:
        registry = SkillRegistry.from_dir(args.dir)
    except SkillError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2))
        return 1
    if args.cmd == "validate":
        print(json.dumps({
            "ok": True,
            "directory": registry.directory,
            "skills": len(registry.skills),
        }, indent=2))
        return 0
    if args.cmd == "show":
        for skill in registry.list():
            if skill["name"] == args.name:
                print(json.dumps(skill, indent=2, ensure_ascii=False))
                return 0
        print(json.dumps({"ok": False, "error": "skill not found"}, indent=2))
        return 1
    print(json.dumps(registry.list(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
