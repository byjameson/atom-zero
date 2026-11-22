import json

import pytest

from atomzero.agent_runtime import AgentSession
from atomzero.doc_memory import DocMemory
from atomzero.skills import SkillError, SkillRegistry, main, render_skill


def write_skill(folder, name, **overrides):
    data = {
        "name": name,
        "description": "test skill",
        "triggers": [name.replace("-", " ")],
        "prompt": "Request: {query}\nContext:\n{context}",
        "use_context": True,
        "max_chunks": 3,
        "provider": True,
    }
    data.update(overrides)
    path = folder / f"{name}.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_skill_registry_loads_matches_and_renders(tmp_path):
    write_skill(tmp_path, "daily-brief")
    write_skill(tmp_path, "daily", triggers=["daily"])
    registry = SkillRegistry.from_dir(tmp_path)
    assert len(registry.list()) == 2
    skill = registry.match("please give me a daily brief")
    assert skill["name"] == "daily-brief"
    prompt = render_skill(
        skill, "daily brief",
        [{"chunk": "The launch code is L-7781.", "source": "launch.txt",
          "title": "Launch", "page": 1, "section": ""}])
    assert "daily brief" in prompt
    assert "The launch code is L-7781." in prompt
    assert "launch.txt" in prompt


def test_skill_registry_rejects_invalid_files(tmp_path):
    (tmp_path / "bad.json").write_text("{", encoding="utf-8")
    with pytest.raises(SkillError):
        SkillRegistry.from_dir(tmp_path)
    (tmp_path / "bad.json").write_text(json.dumps({
        "name": "bad", "triggers": ["bad"], "prompt": ""}), encoding="utf-8")
    with pytest.raises(SkillError):
        SkillRegistry.from_dir(tmp_path)
    (tmp_path / "bad.json").write_text(json.dumps({
        "name": "bad", "triggers": ["re:["], "prompt": "x"}), encoding="utf-8")
    with pytest.raises(SkillError):
        SkillRegistry.from_dir(tmp_path)


def test_agent_skill_uses_provider_and_local_context(tmp_path):
    write_skill(tmp_path, "daily-brief")
    registry = SkillRegistry.from_dir(tmp_path)
    docs = DocMemory(tmp_path / "docs.db")
    docs.add_text("notes/launch.txt", "The launch code is L-7781.",
                  title="Launch")
    prompts = []

    def generator(prompt):
        prompts.append(prompt)
        return "Provider answer."

    agent = AgentSession(
        "skill-1", tmp_path / "memory.json", docs=docs,
        generator=generator, skills=registry)
    result = agent.chat("daily brief launch code")
    assert result["reply"] == "Provider answer."
    assert result["skill"] == "daily-brief"
    assert result["tool_calls"][0]["name"] == "skill"
    assert result["sources"][0]["source"] == "notes/launch.txt"
    assert "L-7781" in result["evidence"][0]["snippet"]
    assert "L-7781" in prompts[0]
    docs.close()


def test_agent_skill_without_provider_and_template_mode(tmp_path):
    write_skill(tmp_path, "daily-brief")
    registry = SkillRegistry.from_dir(tmp_path)
    docs = DocMemory(tmp_path / "docs.db")
    docs.add_text("notes/launch.txt", "The launch code is L-7781.")
    agent = AgentSession(
        "skill-2", tmp_path / "memory.json", docs=docs, skills=registry)
    missing = agent.chat("daily brief launch code")
    assert "needs a local provider" in missing["reply"]
    docs.close()

    write_skill(tmp_path, "template", provider=False,
                triggers=["template"], prompt="Template: {context}")
    registry = SkillRegistry.from_dir(tmp_path)
    docs = DocMemory(tmp_path / "docs2.db")
    docs.add_text("notes/launch.txt", "The launch code is L-7781.")
    agent = AgentSession(
        "skill-3", tmp_path / "memory2.json", docs=docs, skills=registry)
    template = agent.chat("template launch code")
    assert template["reply"].startswith("Template:")
    assert "L-7781" in template["reply"]
    docs.close()


def test_skills_cli_list_and_validate(tmp_path, capsys):
    write_skill(tmp_path, "daily-brief")
    assert main(["--dir", str(tmp_path), "validate"]) == 0
    assert json.loads(capsys.readouterr().out)["skills"] == 1
    assert main(["--dir", str(tmp_path), "list"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert listed[0]["name"] == "daily-brief"
