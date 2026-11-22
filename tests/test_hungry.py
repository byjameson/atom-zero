import json

import pytest

from atomzero.doc_memory import DocMemory
from atomzero.hungry import (
    HungryError, main, run_hungry, teacher_questions)
from atomzero.knowledge_gaps import KnowledgeGapStore


class FakeTeacher:
    label = "fake"
    model = "fake-1"

    def __init__(self, answer="Paris is the capital of France.",
                 questions=None):
        self.answer = answer
        self.questions = questions or []
        self.calls = []

    def ask(self, prompt, system="", max_tokens=512):
        self.calls.append(prompt)
        if prompt.startswith("Propose"):
            return "\n".join(self.questions)
        return self.answer


def test_hungry_learns_unknown_and_marks_known(tmp_path):
    docs = DocMemory(tmp_path / "docs.db")
    docs.add_text("launch.txt", "The launch code is L-7781.",
                  title="Launch")
    gaps = KnowledgeGapStore(tmp_path / "gaps.json")
    known_gap = gaps.add("What is the launch code?")
    unknown_gap = gaps.add("What is the capital of France?")
    teacher = FakeTeacher()
    ledger = tmp_path / "ledger.jsonl"
    summary = run_hungry(
        docs, gaps, teacher, max_questions=2, max_provider_calls=2,
        ledger_path=ledger)
    assert summary["known"] == 1
    assert summary["learned"] == 1
    assert summary["skipped"] == 0
    assert summary["failed"] == 0
    assert summary["provider_calls"] == 1
    assert summary["weights_changed"] is False
    hits = docs.search("capital of France")
    assert hits[0]["kind"] == "teacher"
    assert hits[0]["verified"] is False
    by_id = {item["id"]: item for item in gaps.list()}
    assert by_id[known_gap["id"]]["status"] == "known"
    assert by_id[unknown_gap["id"]]["status"] == "learned"
    assert len(ledger.read_text(encoding="utf-8").splitlines()) == 2
    docs.close()


def test_hungry_dry_run_never_calls_teacher(tmp_path):
    docs = DocMemory(tmp_path / "docs.db")
    gaps = KnowledgeGapStore(tmp_path / "gaps.json")
    gap = gaps.add("What is the capital of France?")
    teacher = FakeTeacher()
    ledger = tmp_path / "ledger.jsonl"
    summary = run_hungry(
        docs, gaps, teacher, dry_run=True, max_questions=1,
        max_provider_calls=3, ledger_path=ledger)
    assert summary["dry_run"] is True
    assert summary["skipped"] == 1
    assert summary["provider_calls"] == 0
    assert teacher.calls == []
    assert gaps.get(gap["id"])["status"] == "pending"
    assert json.loads(
        ledger.read_text(encoding="utf-8").splitlines()[0])["status"] == (
            "unknown")
    docs.close()


def test_hungry_generates_questions_from_a_topic(tmp_path):
    class TopicTeacher(FakeTeacher):
        def ask(self, prompt, system="", max_tokens=512):
            self.calls.append(prompt)
            if prompt.startswith("Propose"):
                return "\n".join(self.questions)
            if "Paris" in prompt:
                return "Paris has about 2.1 million residents."
            return "William Shakespeare wrote Hamlet."

    docs = DocMemory(tmp_path / "docs.db")
    gaps = KnowledgeGapStore(tmp_path / "gaps.json")
    teacher = TopicTeacher(questions=[
        "What is the population of Paris?", "Who wrote Hamlet?"])
    summary = run_hungry(
        docs, gaps, teacher, topics=["Ankara"], max_questions=2,
        max_provider_calls=5, question_limit=2,
        ledger_path=tmp_path / "ledger.jsonl")
    assert summary["learned"] == 2
    assert summary["provider_calls"] == 3
    assert teacher.calls[0].startswith("Propose")
    assert docs.stats()["sources"] == 2
    docs.close()


def test_hungry_skips_teacher_unknown_and_requires_teacher(tmp_path):
    docs = DocMemory(tmp_path / "docs.db")
    gaps = KnowledgeGapStore(tmp_path / "gaps.json")
    gaps.add("What is the capital of France?")
    summary = run_hungry(
        docs, gaps, FakeTeacher(answer="I do not know."),
        max_questions=1, max_provider_calls=2,
        ledger_path=tmp_path / "ledger.jsonl")
    assert summary["learned"] == 0
    assert summary["skipped"] == 1
    assert docs.stats()["sources"] == 0
    with pytest.raises(HungryError):
        run_hungry(docs, gaps, None, max_questions=1)
    docs.close()


def test_teacher_questions_parses_and_deduplicates():
    teacher = FakeTeacher(questions=[
        "- What is X?", "2. Where is Y?", "What is X?"])
    assert teacher_questions(teacher, "X", 3) == [
        "What is X?", "Where is Y?"]


def test_hungry_cli_dry_run(tmp_path, capsys):
    docs = tmp_path / "docs.db"
    gaps_path = tmp_path / "gaps.json"
    gaps = KnowledgeGapStore(gaps_path)
    gaps.add("What is the capital of France?")
    code = main([
        "--dry-run", "--json", "--docs", str(docs),
        "--gaps", str(gaps_path), "--ledger", str(tmp_path / "ledger.jsonl"),
        "--max-questions", "1"])
    assert code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["dry_run"] is True
    assert summary["skipped"] == 1
    assert summary["provider_calls"] == 0
