"""Measure whether local features can separate answerable questions.

This is a negative-result benchmark, not a product gate. It uses SQuAD v2
validation questions against their own paragraphs: 103 answerable and 97
no-answer questions in the default sample. SQuAD v2 no-answer questions are
constructed to share entities and vocabulary with the paragraph, so they are a
stress case for lexical answerability.

The measured cross-validated AUC is around 0.56. That is too weak to ship as a
classifier or to use as a refusal gate. The citation gate, the low-confidence
warning and the knowledge-gap queue remain the product mechanisms.

Run (benchmark dependencies: ``datasets`` and ``scikit-learn``):

    python benchmarks/answerability_eval.py
"""
from __future__ import annotations

import argparse
import json
import re
import tempfile
import time
from pathlib import Path

from atomzero.doc_memory import (
    DocMemory, _query_terms, _term_coverage, best_snippet)

SEED = 0
MAX_CONTEXTS = 50
MAX_QUESTIONS = 200
PER_CONTEXT = 5
WH = {
    "who", "whom", "whose", "what", "when", "where", "which", "why", "how",
    "is", "are", "was", "were", "do", "does", "did", "can", "could", "will",
    "would", "has", "have", "had", "should", "may", "might", "the", "a", "an",
    "many", "much", "long", "old", "far", "year", "date", "time", "place",
    "name", "number", "percentage", "type", "kind",
}


def _names(text: str) -> list[str]:
    found = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*", str(text))
    first = str(text).split()[0] if str(text).split() else ""
    if found and found[0] == first:
        found = found[1:]
    return found


def features(question: str, hit: dict) -> dict:
    """Transparent lexical/type features for one retrieved chunk."""
    chunk = str(hit.get("chunk", ""))
    snippet = best_snippet(question, chunk)
    low = question.lower()
    terms = _query_terms(question)
    focus = {term for term in terms if term not in WH}
    who = bool(re.search(r"\b(who|whom|whose)\b", low))
    when = bool(re.search(
        r"\b(when|what year|what date|what time|which year)\b", low))
    where = bool(re.search(r"\b(where|which place|what place)\b", low))
    amount = bool(re.search(
        r"\b(how many|how much|how long|how old|how far|percentage)\b", low))
    question_names = {name.lower() for name in _names(question)}
    snippet_names = {name.lower() for name in _names(snippet)}
    question_years = set(re.findall(r"\b(?:1[0-9]{3}|20[0-9]{2})\b",
                                    question))
    snippet_years = set(re.findall(r"\b(?:1[0-9]{3}|20[0-9]{2})\b", snippet))
    question_numbers = set(re.findall(r"\b\d+(?:[.,]\d+)?\b", question))
    snippet_numbers = set(re.findall(r"\b\d+(?:[.,]\d+)?\b", snippet))
    type_answer = (
        (who and bool(snippet_names - question_names))
        or (when and bool(snippet_years - question_years))
        or (amount and bool(snippet_numbers - question_numbers))
        or (where and bool(re.search(
            r"\b(?:in|at|on|near|from|inside|outside)\b\s+(?:the\s+)?"
            r"[A-Z][a-z]+", snippet)))
    )
    return {
        "who": who,
        "when": when,
        "where": where,
        "amount": amount,
        "coverage_snippet": _term_coverage(terms, snippet),
        "focus_coverage": _term_coverage(focus, snippet),
        "coverage_chunk": _term_coverage(terms, chunk),
        "embedding": float(hit.get("embedding_score") or 0.0),
        "top_score": float(hit.get("rerank_score") or 0.0),
        "snippet_len": len(snippet),
        "terms": len(terms),
        "novel_names": len(snippet_names - question_names),
        "novel_years": len(snippet_years - question_years),
        "novel_numbers": len(snippet_numbers - question_numbers),
        "type_answer": type_answer,
    }


def load_rows(max_contexts: int = MAX_CONTEXTS,
              max_questions: int = MAX_QUESTIONS,
              per_context: int = PER_CONTEXT,
              seed: int = SEED):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "install the benchmark dependency first: pip install datasets"
        ) from exc
    rows = load_dataset("rajpurkar/squad_v2", split="validation").shuffle(
        seed=seed)
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
        if sum(1 for item in selected
               if item["context"] == key) >= per_context:
            continue
        selected.append(row)
    return selected, contexts


def collect(rows, contexts) -> list[dict]:
    data = []
    with tempfile.TemporaryDirectory(prefix="az-answerability-") as folder:
        docs = DocMemory(Path(folder) / "docs.db")
        try:
            for context, source in contexts.items():
                title = next(row["title"] for row in rows
                             if row["context"] == context)
                docs.add_text(source, context, title=title)
            for row in rows:
                hits = docs.search(row["question"], k=1, rerank=True)
                if not hits:
                    continue
                item = features(row["question"], hits[0])
                item["label"] = 1 if row["answers"]["text"] else 0
                item["group"] = contexts[row["context"]]
                data.append(item)
        finally:
            docs.close()
    return data


def _auc(labels, values) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(labels, values))


def evaluate(data: list[dict], folds: int = 5) -> dict:
    try:
        import numpy as np
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import GroupKFold
    except ImportError as exc:
        raise SystemExit(
            "install the benchmark dependencies first: "
            "pip install datasets scikit-learn") from exc

    labels = np.array([item["label"] for item in data])
    groups = np.array([item["group"] for item in data])
    names = [key for key in data[0]
             if key not in ("label", "group")]
    univariate = {}
    for name in names:
        values = np.array([float(item[name]) for item in data])
        univariate[name] = round(_auc(labels, values), 4)
    matrix = np.array([[float(item[name]) for name in names]
                       for item in data])
    matrix = ((matrix - matrix.mean(axis=0))
              / (matrix.std(axis=0) + 1e-9))
    cross_validation = {}
    for label, model in (
            ("logistic", LogisticRegression(max_iter=2000)),
            ("boost", HistGradientBoostingClassifier(
                max_iter=100, max_depth=3, random_state=SEED))):
        fold_scores = []
        for train, test in GroupKFold(n_splits=folds).split(
                matrix, labels, groups):
            model.fit(matrix[train], labels[train])
            probability = model.predict_proba(matrix[test])[:, 1]
            fold_scores.append(round(_auc(labels[test], probability), 4))
        cross_validation[label] = {
            "mean_auc": round(float(np.mean(fold_scores)), 4),
            "folds": fold_scores,
        }
    return {
        "features": names,
        "univariate_auc": univariate,
        "cross_validation": cross_validation,
        "best_univariate": max(univariate, key=univariate.get),
        "best_univariate_auc": max(univariate.values()),
        "max_cv_auc": max(item["mean_auc"]
                          for item in cross_validation.values()),
    }


def run(max_contexts: int = MAX_CONTEXTS,
        max_questions: int = MAX_QUESTIONS,
        per_context: int = PER_CONTEXT,
        folds: int = 5,
        seed: int = SEED,
        output: str = "benchmarks/results/answerability_eval.json") -> dict:
    rows, contexts = load_rows(max_contexts, max_questions, per_context, seed)
    data = collect(rows, contexts)
    metrics = evaluate(data, folds)
    answerable = sum(item["label"] for item in data)
    payload = {
        "benchmark": "atomzero-answerability-features",
        "dataset": "rajpurkar/squad_v2",
        "split": "validation",
        "seed": seed,
        "contexts": len(contexts),
        "questions": len(data),
        "answerable": answerable,
        "no_answer": len(data) - answerable,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        **metrics,
        "decision": (
            "not shipped as a classifier or refusal gate; context-grouped "
            "cross-validated AUC is too weak on the SQuAD v2 stress set"
        ),
        "notes": [
            "SQuAD v2 no-answer questions share entities and vocabulary with "
            "the paragraph.",
            "The citation gate, low-confidence warning and knowledge-gap "
            "queue remain the product mechanisms.",
        ],
    }
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-contexts", type=int, default=MAX_CONTEXTS)
    parser.add_argument("--max-questions", type=int, default=MAX_QUESTIONS)
    parser.add_argument("--per-context", type=int, default=PER_CONTEXT)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--out", default="benchmarks/results/answerability_eval.json")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    payload = run(args.max_contexts, args.max_questions, args.per_context,
                  args.folds, args.seed, args.out)
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(
            f"questions={payload['questions']} "
            f"answerable={payload['answerable']} "
            f"no_answer={payload['no_answer']}")
        for name, value in payload["univariate_auc"].items():
            print(f"{name:18s} auc={value:.4f}")
        for name, item in payload["cross_validation"].items():
            print(f"{name:10s} cv_auc={item['mean_auc']:.4f} "
                  f"folds={item['folds']}")
        print(f"decision: {payload['decision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
