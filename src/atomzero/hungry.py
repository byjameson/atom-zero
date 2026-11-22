"""Bounded, opt-in "hungry mode" for growing the local RAG index.

Hungry mode is intentionally not an autonomous web agent:

- it asks only the configured local/OpenAI-compatible teacher (Ollama by
  default), never a search engine;
- every run has hard question and provider-call budgets;
- answers are stored as **unverified teacher notes**, with provenance;
- model weights are never changed;
- each result is appended to an audit ledger.

Typical use:

    atomzero-hungry --topic Ankara --max-questions 5
    atomzero-hungry --dry-run --json
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from atomzero import __version__
from atomzero.doc_memory import DocMemory, retrieval_confidence
from atomzero.knowledge_gaps import KnowledgeGapError, KnowledgeGapStore
from atomzero.teacher_notes import add_teacher_note
from atomzero.teacher_provider import TeacherProvider, TeacherProviderError

MAX_QUESTIONS = 50
MAX_PROVIDER_CALLS = 60

HUNGRY_SYSTEM = (
    "You are a local teacher for Atom Zero. Answer factual questions "
    "concisely. If you are not sure, say 'I do not know'. Do not invent "
    "sources or URLs. Do not provide instructions for harmful actions."
)


class HungryError(ValueError):
    pass


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def clean_question(text: str) -> str:
    text = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", str(text or "")).strip()
    text = re.sub(r"\s+", " ", text)
    if not text:
        return ""
    if not text.endswith("?"):
        text += "?"
    if len(text) < 8:
        return ""
    return text[:300]


def teacher_questions(teacher, topic: str, count: int = 3) -> list[str]:
    """Ask the teacher for a few specific questions about a topic."""
    if teacher is None:
        raise HungryError("teacher provider is required")
    topic = str(topic or "").strip()
    if not topic:
        return []
    try:
        count = max(1, min(int(count), 5))
    except (TypeError, ValueError):
        count = 3
    prompt = (
        f"Propose {count} specific factual questions about the topic below. "
        "Return one question per line, no numbering, no answers, at most 20 "
        f"words each.\nTopic: {topic}")
    raw = teacher.ask(prompt, system=HUNGRY_SYSTEM, max_tokens=300)
    out = []
    seen = set()
    for line in str(raw or "").splitlines():
        question = clean_question(line)
        key = question.lower()
        if question and key not in seen:
            seen.add(key)
            out.append(question)
        if len(out) >= count:
            break
    return out


def answer_question(teacher, question: str) -> str:
    prompt = (
        "Answer the question below in at most 120 words. If you are not "
        "sure, say 'I do not know'.\n\nQuestion: " + str(question))
    return teacher.ask(prompt, system=HUNGRY_SYSTEM, max_tokens=300)


def local_lookup(docs, question: str,
                 min_confidence: float = 0.55) -> tuple[dict | None, float]:
    """Return a hit only when the local index has a strong match."""
    if docs is None:
        return None, 0.0
    hits = docs.search(question, k=6)
    if not hits:
        return None, 0.0
    confidence = retrieval_confidence(question, hits)
    score = float(confidence["score"])
    if score >= float(min_confidence):
        return hits[0], score
    return None, score


def _append_ledger(path: str | Path | None, record: dict) -> None:
    if not path:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _teacher_unknown(answer: str) -> bool:
    value = re.sub(r"\s+", " ", str(answer or "")).strip().lower()
    value = value.rstrip(".!")
    return value in ("i do not know", "i don't know", "i am not sure",
                     "i'm not sure", "not sure", "cannot answer")


def run_hungry(docs, gaps, teacher=None, *, topics=None,
               max_questions: int = 5, max_provider_calls: int = 8,
               question_limit: int = 3, min_confidence: float = 0.55,
               expand: bool = False, dry_run: bool = False,
               ledger_path: str | Path | None = None) -> dict:
    """Run one bounded hungry-mode session.

    ``docs`` is a DocMemory, ``gaps`` a KnowledgeGapStore, and ``teacher`` an
    object with ``ask()`` plus optional ``label``/``model`` attributes.
    """
    if not dry_run and teacher is None:
        raise HungryError(
            "teacher provider is required unless --dry-run is used")
    try:
        max_questions = max(1, min(int(max_questions), MAX_QUESTIONS))
        max_provider_calls = max(
            0, min(int(max_provider_calls), MAX_PROVIDER_CALLS))
        question_limit = max(1, min(int(question_limit), 5))
        min_confidence = max(0.0, min(float(min_confidence), 1.0))
    except (TypeError, ValueError) as exc:
        raise HungryError("budgets and confidence must be numeric") from exc

    started = time.time()
    candidates: list[dict] = []
    seen: set[str] = set()
    errors: list[dict] = []

    def add_candidate(question: str, origin: str, gap_id: str = "") -> None:
        value = clean_question(question)
        if not value:
            return
        key = value.lower()
        if key in seen:
            return
        seen.add(key)
        candidates.append({"question": value, "origin": origin,
                           "gap_id": gap_id})

    if gaps is not None:
        for gap in gaps.list():
            if gap.get("status") == "pending":
                add_candidate(gap.get("query", ""),
                              f"gap:{gap.get('id', '')}",
                              str(gap.get("id", "")))

    seed_topics = []
    for topic in topics or []:
        value = str(topic or "").strip()
        if value:
            seed_topics.append(value)
    if docs is not None:
        for item in docs.sources():
            if item.get("kind") != "document":
                continue
            if str(item.get("source", "")).startswith("teacher://"):
                continue
            value = str(item.get("title") or item.get("source") or "").strip()
            if value:
                seed_topics.append(value)
            if len(seed_topics) >= 8:
                break
    seed_topics = list(dict.fromkeys(seed_topics))

    provider_calls = 0
    if dry_run:
        for topic in seed_topics:
            add_candidate(f"What is {topic}?", f"topic:{topic}")
    else:
        # Reserve one provider call per attempted question for answers.
        generation_budget = max(0, max_provider_calls - max_questions)
        for topic in seed_topics:
            if provider_calls >= generation_budget:
                break
            provider_calls += 1
            try:
                for question in teacher_questions(
                        teacher, topic, question_limit):
                    add_candidate(question, f"topic:{topic}")
            except Exception as exc:
                errors.append({"stage": "question_generation",
                               "topic": topic,
                               "error": str(exc)[:300]})

    results = []
    known = learned = skipped = failed = 0
    attempted = 0
    index = 0
    while index < len(candidates) and attempted < max_questions:
        candidate = candidates[index]
        index += 1
        attempted += 1
        question = candidate["question"]
        gap_id = candidate["gap_id"]
        hit, score = local_lookup(docs, question, min_confidence)
        if hit is not None:
            known += 1
            if gap_id and gaps is not None:
                try:
                    gaps.mark_known(gap_id, hit.get("source", ""))
                except (KnowledgeGapError, ValueError):
                    pass
            record = {
                "ts": _now(), "question": question,
                "origin": candidate["origin"], "status": "known",
                "source": hit.get("source", ""), "score": round(score, 3),
                "gap_id": gap_id,
            }
            results.append(record)
            _append_ledger(ledger_path, record)
            continue

        if dry_run:
            skipped += 1
            if not gap_id and gaps is not None:
                try:
                    gap_id = gaps.add(question, reason="hungry_mode")["id"]
                except (KnowledgeGapError, ValueError):
                    pass
            record = {
                "ts": _now(), "question": question,
                "origin": candidate["origin"], "status": "unknown",
                "score": round(score, 3), "gap_id": gap_id,
            }
            results.append(record)
            _append_ledger(ledger_path, record)
            continue

        if provider_calls >= max_provider_calls:
            skipped += 1
            record = {
                "ts": _now(), "question": question,
                "origin": candidate["origin"], "status": "budget_exhausted",
                "score": round(score, 3), "gap_id": gap_id,
            }
            results.append(record)
            _append_ledger(ledger_path, record)
            continue

        provider_calls += 1
        try:
            answer = answer_question(teacher, question)
        except Exception as exc:
            failed += 1
            errors.append({"stage": "answer", "question": question,
                           "error": str(exc)[:300]})
            record = {
                "ts": _now(), "question": question,
                "origin": candidate["origin"], "status": "error",
                "score": round(score, 3), "gap_id": gap_id,
                "error": str(exc)[:300],
            }
            results.append(record)
            _append_ledger(ledger_path, record)
            continue
        if _teacher_unknown(answer):
            skipped += 1
            record = {
                "ts": _now(), "question": question,
                "origin": candidate["origin"], "status": "teacher_unknown",
                "score": round(score, 3), "gap_id": gap_id,
            }
            results.append(record)
            _append_ledger(ledger_path, record)
            continue
        try:
            note = add_teacher_note(
                docs, getattr(teacher, "label", "teacher"),
                getattr(teacher, "model", ""), question, answer)
            if gaps is not None:
                if not gap_id:
                    gap_id = gaps.add(
                        question, reason="hungry_mode")["id"]
                gaps.mark_learned(
                    gap_id, note["source"],
                    getattr(teacher, "label", "teacher"),
                    getattr(teacher, "model", ""))
            learned += 1
            record = {
                "ts": _now(), "question": question,
                "origin": candidate["origin"], "status": "learned",
                "source": note["source"], "score": round(score, 3),
                "gap_id": gap_id,
                "teacher": getattr(teacher, "label", "teacher"),
                "model": getattr(teacher, "model", ""),
                "answer_preview": answer[:200],
            }
            results.append(record)
            _append_ledger(ledger_path, record)
        except Exception as exc:
            failed += 1
            errors.append({"stage": "store", "question": question,
                           "error": str(exc)[:300]})
            record = {
                "ts": _now(), "question": question,
                "origin": candidate["origin"], "status": "error",
                "score": round(score, 3), "gap_id": gap_id,
                "error": str(exc)[:300],
            }
            results.append(record)
            _append_ledger(ledger_path, record)
            continue

        if (expand and teacher is not None
                and provider_calls < max_provider_calls):
            provider_calls += 1
            try:
                for question in teacher_questions(
                        teacher, f"{candidate['question']} {answer}", 2):
                    add_candidate(question, "followup")
            except Exception as exc:
                errors.append({"stage": "expansion",
                               "error": str(exc)[:300]})

    summary = {
        "ok": failed == 0,
        "dry_run": bool(dry_run),
        "questions_attempted": attempted,
        "known": known,
        "learned": learned,
        "skipped": skipped,
        "failed": failed,
        "provider_calls": provider_calls,
        "weights_changed": False,
        "elapsed_seconds": round(time.time() - started, 2),
        "ledger": str(ledger_path or ""),
        "gaps": gaps.stats() if gaps is not None else None,
        "errors": errors,
        "results": results,
    }
    if not candidates:
        summary["message"] = (
            "no candidate questions; add a pending gap or pass --topic")
    return summary


def render(summary: dict) -> str:
    lines = [
        "Hungry mode: {learned} learned, {known} known, {skipped} skipped, "
        "{failed} failed".format(**summary),
        "provider calls: {provider_calls} | weights changed: {weights_changed}"
        " | elapsed: {elapsed_seconds}s".format(**summary),
    ]
    for item in summary.get("results", []):
        status = item.get("status")
        question = item.get("question", "")
        if status == "learned":
            lines.append(f"- learned: {question} -> {item.get('source', '')}")
        elif status == "known":
            lines.append(f"- known: {question} -> {item.get('source', '')}")
        elif status == "error":
            lines.append(f"- error: {question} ({item.get('error', '')})")
        else:
            lines.append(f"- {status}: {question}")
    if summary.get("ledger"):
        lines.append(f"ledger: {summary['ledger']}")
    if summary.get("message"):
        lines.append(summary["message"])
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="atomzero-hungry",
        description=("Bounded curiosity loop: ask the local teacher for "
                     "unknown facts and store unverified notes."))
    parser.add_argument("--version", action="version",
                        version=f"atomzero-hungry {__version__}")
    parser.add_argument("--docs", default="data/atomzero_demo.db")
    parser.add_argument("--gaps", default="data/knowledge_gaps.json")
    parser.add_argument("--ledger", default="data/hungry_ledger.jsonl")
    parser.add_argument("--topic", action="append", default=[],
                        help="seed topic; may be repeated")
    parser.add_argument("--max-questions", type=int, default=5)
    parser.add_argument("--max-provider-calls", type=int, default=8)
    parser.add_argument("--question-limit", type=int, default=3)
    parser.add_argument("--min-confidence", type=float, default=0.55)
    parser.add_argument("--expand", action="store_true",
                        help="ask for follow-up questions after a learned answer")
    parser.add_argument("--dry-run", action="store_true",
                        help="only show unknown questions; never call the teacher")
    parser.add_argument("--provider-url", default="http://127.0.0.1:11434/v1")
    parser.add_argument("--provider-model", default="llama3.2:3b")
    parser.add_argument("--provider-label", default="ollama")
    parser.add_argument("--provider-timeout", type=int, default=60)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    docs = None
    try:
        docs = DocMemory(args.docs)
        gaps = KnowledgeGapStore(args.gaps)
        teacher = None
        if not args.dry_run:
            teacher = TeacherProvider(
                args.provider_url, args.provider_model,
                timeout=args.provider_timeout, label=args.provider_label)
        summary = run_hungry(
            docs, gaps, teacher, topics=args.topic,
            max_questions=args.max_questions,
            max_provider_calls=args.max_provider_calls,
            question_limit=args.question_limit,
            min_confidence=args.min_confidence,
            expand=args.expand, dry_run=args.dry_run,
            ledger_path=args.ledger or None)
    except (HungryError, TeacherProviderError, KnowledgeGapError,
            ValueError, OSError) as exc:
        if args.json:
            print(json.dumps({"ok": False, "error": str(exc)}, indent=2))
        else:
            print(f"hungry mode error: {exc}")
        return 1
    finally:
        if docs is not None:
            docs.close()
    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        print(render(summary))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
