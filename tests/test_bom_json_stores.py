import json

from atomzero.feedback import FeedbackStore
from atomzero.knowledge_gaps import KnowledgeGapStore
from atomzero.research_queue import ResearchQueue
from atomzero.skills import ChainRecipeRegistry, SkillRegistry


BOM = b"\xef\xbb\xbf"


def _add_bom(path) -> None:
    path.write_bytes(BOM + path.read_bytes())


def test_feedback_store_reads_bom_json(tmp_path):
    path = tmp_path / "feedback.json"
    store = FeedbackStore(path)
    item = store.add("hello", "hi", rating=1)
    _add_bom(path)
    reloaded = FeedbackStore(path)
    assert reloaded.list()[0]["id"] == item["id"]


def test_knowledge_gap_store_reads_bom_json(tmp_path):
    path = tmp_path / "gaps.json"
    store = KnowledgeGapStore(path)
    gap = store.add("What is the launch code?")
    _add_bom(path)
    reloaded = KnowledgeGapStore(path)
    assert reloaded.get(gap["id"])["query"] == "What is the launch code?"


def test_research_queue_reads_bom_json(tmp_path):
    path = tmp_path / "queue.json"
    queue = ResearchQueue(path)
    item = queue.add("Ankara")
    _add_bom(path)
    reloaded = ResearchQueue(path)
    assert reloaded.list()[0]["id"] == item["id"]


def test_skill_and_recipe_registries_read_bom_json(tmp_path):
    skill_path = tmp_path / "bom-skill.json"
    skill_path.write_text(json.dumps({
        "name": "bom-skill",
        "description": "BOM test skill",
        "triggers": ["bom skill"],
        "prompt": "{query}",
        "use_context": False,
        "max_chunks": 1,
        "provider": False,
    }), encoding="utf-8")
    _add_bom(skill_path)
    chain_dir = tmp_path / "chains"
    chain_dir.mkdir()
    recipe_path = chain_dir / "bom-recipe.json"
    recipe_path.write_text(json.dumps({
        "name": "bom-recipe",
        "description": "BOM test recipe",
        "steps": ["bom-skill"],
        "carry_context": False,
    }), encoding="utf-8")
    _add_bom(recipe_path)
    skills = SkillRegistry.from_dir(tmp_path)
    chains = ChainRecipeRegistry.from_skills_dir(tmp_path)
    assert skills.list()[0]["name"] == "bom-skill"
    assert chains.get("bom-recipe")["name"] == "bom-recipe"
