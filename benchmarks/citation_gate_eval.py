"""Evaluate the calibrated CitationGate on SQuAD v2.

This is a policy simulation, not a production-distribution benchmark. It shows
how a threshold transfers (or fails to transfer) across distributions.

Run:

    python benchmarks/citation_gate_eval.py
"""
from __future__ import annotations

import argparse
import json
import re
import tempfile
import time
from pathlib import Path

from datasets import load_dataset

from atomzero.citation_gate import CitationGate, PROFILES
from atomzero.doc_memory import DocMemory, grounded_answer

SEED = 0
MAX_CONTEXTS = 50
MAX_QUESTIONS = 200
PER_CONTEXT = 5


def _norm(text):
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def _rows():
    rows = load_dataset("rajpurkar/squad_v2", split="validation").shuffle(
        seed=SEED)
    contexts = {}
    selected = []
    for row in rows:
        key = row["context"]
        if key not in contexts:
            if len(contexts) >= MAX_CONTEXTS:
                continue
            contexts[key] = f"squad-{len(contexts):03d}"
        if len(selected) >= MAX_QUESTIONS:
            break
        if sum(1 for item in selected if item["context"] == key) >= PER_CONTEXT:
            continue
        selected.append(row)
    return selected, contexts


def evaluate(rows, contexts, profile: str, threshold: float) -> dict:
    gate = CitationGate(profile=profile, threshold=threshold)
    answerable = answered = correct = 0
    no_answer = no_answer_refused = 0
    with tempfile.TemporaryDirectory(prefix="az-gate-eval-") as folder:
        docs = DocMemory(Path(folder) / "docs.db")
        try:
            for context, source in contexts.items():
                title = next(row["title"] for row in rows
                             if row["context"] == context)
                docs.add_text(source, context, title=title)
            for row in rows:
                hits = docs.search(row["question"], k=6, rerank=True)
                result = grounded_answer(row["question"], hits, gate=gate)
                refused = bool(result.get("refused")) or not hits
                answers = row["answers"]["text"]
                if answers:
                    answerable += 1
                    if not refused:
                        answered += 1
                        needle = _norm(answers[0])
                        if needle and any(
                                needle in _norm(item.get("snippet", ""))
                                for item in result.get("evidence", [])):
                            correct += 1
                else:
                    no_answer += 1
                    if refused:
                        no_answer_refused += 1
        finally:
            docs.close()
    released = answered + (no_answer - no_answer_refused)
    precision = correct / released if released else 0.0
    overall = (correct + no_answer_refused) / max(1, len(rows))
    return {
        "profile": profile,
        "threshold": threshold,
        "answerable": answerable,
        "answerable_answered": answered,
        "answerable_correct": correct,
        "answerable_refused": answerable - answered,
        "no_answer": no_answer,
        "no_answer_refused": no_answer_refused,
        "released_answers": released,
        "citation_precision": round(precision, 4),
        "overall_accuracy": round(overall, 4),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", default="benchmarks/results/citation_gate_eval.json")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    rows, contexts = _rows()
    results = []
    for profile in ("authored_v1", "squad_v2_stress"):
        for threshold in (0.3, 0.4, 0.5, 0.6):
            results.append(evaluate(rows, contexts, profile, threshold))
    payload = {
        "benchmark": "atomzero-citation-gate-eval",
        "dataset": "rajpurkar/squad_v2",
        "split": "validation",
        "seed": SEED,
        "contexts": len(contexts),
        "questions": len(rows),
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "profiles": list(PROFILES),
        "results": results,
        "notes": [
            "Citation precision counts released answers whose visible snippet "
            "contains the gold answer.",
            "This is a distribution-shift simulation, not a production "
            "calibration proof.",
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        for item in results:
            print(
                f"{item['profile']:16s} t={item['threshold']:.1f} "
                f"answered={item['answerable_answered']:2d}/"
                f"{item['answerable']:2d} "
                f"correct={item['answerable_correct']:2d} "
                f"no-answer-refused={item['no_answer_refused']:2d}/"
                f"{item['no_answer']:2d} "
                f"precision={item['citation_precision']:.3f} "
                f"overall={item['overall_accuracy']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
