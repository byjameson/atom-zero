import json

import pytest

from atomzero.agent_runtime import AgentSession
from atomzero.doc_memory import DocMemory
from atomzero.skills import (
    ChainRecipeRegistry, SkillError, SkillRegistry, execute_chain_recipe,
    execute_skill, main, render_skill)


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


def write_recipe(folder, name, steps, carry_context=True):
    chain_dir = folder / "chains"
    chain_dir.mkdir(exist_ok=True)
    path = chain_dir / f"{name}.json"
    path.write_text(json.dumps({
        "name": name,
        "description": "test recipe",
        "steps": steps,
        "carry_context": carry_context,
    }), encoding="utf-8")
    return path


def test_chain_recipe_registry_and_execution(tmp_path):
    write_skill(tmp_path, "first", provider=False, triggers=["first"],
                prompt="A: {context}")
    write_skill(tmp_path, "second", provider=False, triggers=["second"],
                prompt="B previous: {previous}")
    write_recipe(tmp_path, "recipe-1", ["first", "second"])
    skills = SkillRegistry.from_dir(tmp_path)
    chains = ChainRecipeRegistry.from_skills_dir(tmp_path)
    assert chains.list()[0]["name"] == "recipe-1"
    recipe = chains.get("recipe-1")
    docs = DocMemory(tmp_path / "docs.db")
    docs.add_text("notes/launch.txt", "The launch code is L-7781.")
    result = execute_chain_recipe(
        recipe, "launch code", skills, docs)
    assert result["recipe"] == "recipe-1"
    assert "B previous: A:" in result["reply"]
    no_carry = execute_chain_recipe(
        recipe, "launch code", skills, docs, carry_context=False)
    assert "L-7781" not in no_carry["reply"]
    with pytest.raises(SkillError):
        chains.get("missing")
    docs.close()


def test_chain_recipe_validation(tmp_path):
    chain_dir = tmp_path / "chains"
    chain_dir.mkdir()
    (chain_dir / "bad.json").write_text(
        json.dumps({"name": "bad", "steps": []}), encoding="utf-8")
    with pytest.raises(SkillError):
        ChainRecipeRegistry.from_skills_dir(tmp_path)


def test_cli_recipes_and_recipe_run(tmp_path, capsys):
    write_skill(tmp_path, "first", provider=False, triggers=["first"],
                prompt="A: {context}")
    write_skill(tmp_path, "second", provider=False, triggers=["second"],
                prompt="B previous: {previous}")
    write_recipe(tmp_path, "recipe-1", ["first", "second"])
    docs = DocMemory(tmp_path / "docs.db")
    docs.add_text("notes/launch.txt", "The launch code is L-7781.")
    docs.close()
    assert main(["--dir", str(tmp_path), "recipes"]) == 0
    recipes = json.loads(capsys.readouterr().out)
    assert recipes[0]["name"] == "recipe-1"
    assert main([
        "--dir", str(tmp_path), "recipe", "recipe-1", "launch code",
        "--docs", str(tmp_path / "docs.db")]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["recipe"] == "recipe-1"
    assert "B previous: A:" in result["reply"]


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


def test_execute_skill_provider_template_and_fallback(tmp_path):
    docs = DocMemory(tmp_path / "docs.db")
    docs.add_text("notes/launch.txt", "The launch code is L-7781.",
                  title="Launch")
    provider_skill = {
        "name": "provider", "prompt": "Provider: {context} {query}",
        "use_context": True, "max_chunks": 2, "provider": True,
    }
    calls = []

    def generator(prompt):
        calls.append(prompt)
        return "Provider answer."

    result = execute_skill(provider_skill, "launch code", docs, generator)
    assert result["reply"] == "Provider answer."
    assert result["provider_called"] is True
    assert result["sources"][0]["source"] == "notes/launch.txt"
    assert "L-7781" in calls[0]

    def must_not_call(prompt):
        raise AssertionError("template mode called the provider")

    template_skill = {
        "name": "template", "prompt": "Template: {context}",
        "use_context": True, "max_chunks": 2, "provider": False,
    }
    template = execute_skill(
        template_skill, "launch code", docs, must_not_call)
    assert template["reply"].startswith("Template:")
    assert "L-7781" in template["reply"]
    assert template["provider_called"] is False

    fallback = execute_skill(
        provider_skill, "launch code", docs, None)
    assert "needs a local provider" in fallback["reply"]
    assert fallback["provider_called"] is False
    assert fallback["sources"][0]["source"] == "notes/launch.txt"
    docs.close()


def test_execute_chain_passes_previous_output(tmp_path):
    from atomzero.skills import execute_chain

    docs = DocMemory(tmp_path / "docs.db")
    docs.add_text("notes/launch.txt", "The launch code is L-7781.")
    write_skill(tmp_path, "first", provider=False, triggers=["first"],
                prompt="A: {context}")
    write_skill(tmp_path, "second", provider=False, triggers=["second"],
                prompt="B previous: {previous}")
    registry = SkillRegistry.from_dir(tmp_path)
    result = execute_chain(
        registry, ["first", "second"], "launch code", docs)
    assert result["chain"] == ["first", "second"]
    assert "L-7781" in result["steps"][0]["reply"]
    assert "B previous: A:" in result["steps"][1]["reply"]
    assert "L-7781" in result["steps"][1]["reply"]
    assert result["provider_calls"] == 0
    assert result["weights_changed"] is False
    docs.close()


def test_execute_chain_no_carry_and_validates(tmp_path):
    from atomzero.skills import execute_chain

    docs = DocMemory(tmp_path / "docs.db")
    docs.add_text("notes/launch.txt", "The launch code is L-7781.")
    write_skill(tmp_path, "first", provider=False, triggers=["first"],
                prompt="A: {context}")
    write_skill(tmp_path, "second", provider=False, triggers=["second"],
                prompt="B previous: {previous}")
    registry = SkillRegistry.from_dir(tmp_path)
    result = execute_chain(
        registry, ["first", "second"], "launch code", docs,
        carry_context=False)
    assert "L-7781" not in result["steps"][1]["reply"]
    with pytest.raises(SkillError):
        execute_chain(registry, ["missing"], "launch code", docs)
    with pytest.raises(SkillError):
        execute_chain(registry, [], "launch code", docs)
    docs.close()


def test_skills_cli_chain(tmp_path, capsys):
    write_skill(tmp_path, "first", provider=False, triggers=["first"],
                prompt="A: {context}")
    write_skill(tmp_path, "second", provider=False, triggers=["second"],
                prompt="B previous: {previous}")
    docs = DocMemory(tmp_path / "docs.db")
    docs.add_text("notes/launch.txt", "The launch code is L-7781.")
    docs.close()
    code = main([
        "--dir", str(tmp_path), "chain", "launch code",
        "--skills", "first,second", "--docs", str(tmp_path / "docs.db")])
    assert code == 0
    result = json.loads(capsys.readouterr().out)
    assert result["chain"] == ["first", "second"]
    assert "B previous: A:" in result["reply"]
    assert "L-7781" in result["reply"]


def test_skills_cli_run_template(tmp_path, capsys):
    write_skill(tmp_path, "template", provider=False,
                triggers=["template"], prompt="CLI template: {context}")
    docs = DocMemory(tmp_path / "docs.db")
    docs.add_text("notes/launch.txt", "The launch code is L-7781.")
    docs.close()
    code = main([
        "--dir", str(tmp_path), "run", "template", "launch code",
        "--docs", str(tmp_path / "docs.db")])
    assert code == 0
    result = json.loads(capsys.readouterr().out)
    assert result["ok"] is True
    assert result["provider_called"] is False
    assert "L-7781" in result["reply"]


def test_skills_cli_list_and_validate(tmp_path, capsys):
    write_skill(tmp_path, "daily-brief")
    assert main(["--dir", str(tmp_path), "validate"]) == 0
    assert json.loads(capsys.readouterr().out)["skills"] == 1
    assert main(["--dir", str(tmp_path), "list"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert listed[0]["name"] == "daily-brief"
