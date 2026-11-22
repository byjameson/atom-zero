"""Public SQuAD v2 retrieval benchmark for Atom Zero's DocMemory.

This measures retrieval, not generation:

- top-1 / top-5 context accuracy
- top-1 / top-5 answer-in-retrieved-chunk
- MRR of the gold context

The benchmark uses `rajpurkar/squad_v2` validation contexts. Each context is
ingested as one source; questions are asked against the local index. No model
weights or GPU are required.
"""
from __future__ import annotations

import argparse
import json
import re
import tempfile
from pathlib import Path

from atomzero.doc_memory import DocMemory


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def answer_in_chunks(answer: str, hits: list[dict]) -> bool:
    needle = normalize(answer)
    if not needle:
        return False
    return any(needle in normalize(hit.get("chunk", "")) for hit in hits)


def reciprocal_rank(source: str, hits: list[dict]) -> float:
    for index, hit in enumerate(hits):
        if hit.get("source") == source:
            return 1.0 / (index + 1)
    return 0.0


def load_rows(max_contexts: int, max_questions: int, per_context: int, seed: int):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "install the benchmark dependency first: pip install datasets") from exc

    rows = load_dataset("rajpurkar/squad_v2", split="validation").shuffle(seed=seed)
    contexts = {}
    selected = []
    for row in rows:
        key = row["context"]
        if key not in contexts:
            if len(contexts) >= max_contexts:
                continue
            contexts[key] = f"squad-{len(contexts):03d}"
        if len(selected) >= max_questions:
            break
        if sum(1 for r in selected if r["context"] == key) >= per_context:
            continue
        selected.append(row)
    return selected, contexts


def run(max_contexts: int, max_questions: int, per_context: int, seed: int,
        output: str) -> dict:
    rows, contexts = load_rows(max_contexts, max_questions, per_context, seed)
    with tempfile.TemporaryDirectory(prefix="atomzero-squad-") as folder:
        docs = DocMemory(Path(folder) / "docs.db")
        try:
            ingested = {}
            for context, source in contexts.items():
                title = next(r["title"] for r in rows if r["context"] == context)
                chunks = docs.add_text(source, context, title=title)
                ingested[source] = chunks
            results = {}
            for mode in ("fts", "hybrid"):
                top1 = top5 = answer1 = answer5 = 0
                rr = 0.0
                answerable = 0
                no_answer = 0
                for row in rows:
                    gold = contexts[row["context"]]
                    hits = docs.search(row["question"], k=5, rerank=(mode == "hybrid"))
                    if hits and hits[0].get("source") == gold:
                        top1 += 1
                    if any(hit.get("source") == gold for hit in hits):
                        top5 += 1
                    rr += reciprocal_rank(gold, hits)
                    answers = row["answers"]["text"]
                    if answers:
                        answerable += 1
                        if hits and answer_in_chunks(answers[0], hits[:1]):
                            answer1 += 1
                        if answer_in_chunks(answers[0], hits):
                            answer5 += 1
                    else:
                        no_answer += 1
                n = len(rows)
                results[mode] = {
                    "questions": n,
                    "answerable": answerable,
                    "no_answer": no_answer,
                    "top1_context": top1 / max(1, n),
                    "top5_context": top5 / max(1, n),
                    "top1_answer": answer1 / max(1, answerable),
                    "top5_answer": answer5 / max(1, answerable),
                    "mrr": rr / max(1, n),
                }
        finally:
            docs.close()

    payload = {
        "benchmark": "atomzero-squad-v2-retrieval",
        "dataset": "rajpurkar/squad_v2",
        "split": "validation",
        "seed": seed,
        "max_contexts": max_contexts,
        "max_questions": max_questions,
        "per_context": per_context,
        "contexts_ingested": len(contexts),
        "chunks_ingested": sum(ingested.values()),
        "results": results,
    }
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-contexts", type=int, default=20)
    parser.add_argument("--max-questions", type=int, default=100)
    parser.add_argument("--per-context", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="benchmarks/results/squad_retrieval.json")
    args = parser.parse_args()
    payload = run(args.max_contexts, args.max_questions, args.per_context,
                  args.seed, args.output)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
