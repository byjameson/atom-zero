"""Deterministic regression benchmark for scoped retrieval and summaries.

This is a product regression gate, not a public leaderboard. It builds a small
temporary corpus and checks:

- source/prefix/kind scoping does not leak other documents;
- grounded answers cite the expected evidence;
- single-source summaries keep page/section locators;
- collection summaries stay inside the selected prefix;
- no-answer questions return nothing;
- exact code identifiers rank the right source.

Run:

    python benchmarks/scoped_regression.py
    python benchmarks/scoped_regression.py --out benchmarks/results/scoped_regression.json
"""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

from atomzero.doc_memory import (
    DocMemory, SourceNotFound, grounded_answer)


def _answer(docs: DocMemory, query: str, **filters):
    hits = docs.search(query, k=6, **filters)
    return grounded_answer(query, hits) if hits else None


def _sources(answer: dict | None) -> list[str]:
    if not answer:
        return []
    return [item.get("source", "") for item in answer.get("evidence", [])]


def run(output: str | Path | None = None) -> dict:
    checks = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    with tempfile.TemporaryDirectory(prefix="atomzero-scoped-") as folder:
        docs = DocMemory(Path(folder) / "docs.db")
        try:
            docs.add_text(
                "notes/launch.txt",
                "The launch code is L-7781. The window opens at 06:30 UTC.",
                title="Launch")
            docs.add_text(
                "notes/warranty.md",
                "The warranty period is 24 months. Claims must be submitted "
                "within 30 days.",
                title="Warranty")
            docs.add_text(
                "notes/travel.txt",
                "Book business class for flights longer than four hours. "
                "The travel portal code is TR-42.",
                title="Travel")
            docs.add_text(
                "notes/legacy.txt",
                "The old warranty code is W-7781.",
                title="Legacy")
            docs.add_blocks(
                "manual.pdf",
                [{"text": "The emergency shutdown code is SD-9000. "
                          "Pull the red handle first.",
                  "page": 4, "section": "Launch"}],
                title="Manual")
            docs.add_text(
                "teacher://ollama/warranty",
                "Topic: Warranty\n\nThe warranty period is 36 months.",
                title="Teacher note: Warranty",
                meta={"kind": "teacher", "verified": False,
                      "source_url": "https://example.com/warranty"})

            answer = _answer(
                docs, "warranty period", source="notes/warranty.md")
            check(
                "source_scope_answer",
                answer is not None and "24 months" in answer["text"]
                and _sources(answer) == ["notes/warranty.md"],
                str(_sources(answer)))

            answer = _answer(docs, "launch code", source_prefix="notes/")
            check(
                "prefix_scope_answer",
                answer is not None and "L-7781" in answer["text"]
                and all(source.startswith("notes/")
                        for source in _sources(answer))
                and "manual.pdf" not in _sources(answer),
                str(_sources(answer)))

            answer = _answer(docs, "warranty period", kind="teacher")
            check(
                "kind_scope_answer",
                answer is not None and "36 months" in answer["text"]
                and all(item.get("kind") == "teacher"
                        for item in answer["evidence"])
                and all(item.get("verified") is False
                        for item in answer["evidence"]),
                str(_sources(answer)))

            leaked = docs.search("launch code", source="notes/warranty.md")
            check("scope_search_no_leak", leaked == [], f"{len(leaked)} hit(s)")

            answer = _answer(docs, "not a real query xyzzy")
            check("no_answer", answer is None, "unexpected answer"
                  if answer else "")

            summary = docs.summarize_source("manual.pdf", max_sentences=3)
            check(
                "source_summary_locator",
                "SD-9000" in summary["text"]
                and {"page": 4, "section": "Launch"}
                in summary["citations"],
                str(summary["citations"]))

            collection = docs.summarize_collection(
                source_prefix="notes/", max_sentences=10)
            check(
                "collection_summary",
                "L-7781" in collection["text"]
                and "24 months" in collection["text"]
                and collection["sources"] >= 4
                and all(item["source"].startswith("notes/")
                        for item in collection["citations"]),
                f"{collection['sources']} source(s), "
                f"{len(collection['citations'])} citation(s)")

            launch = docs.search("L-7781", k=2)
            legacy = docs.search("W-7781", k=2)
            check(
                "exact_code_ranking",
                launch and launch[0]["source"] == "notes/launch.txt"
                and legacy and legacy[0]["source"] == "notes/legacy.txt",
                f"L={launch[0]['source'] if launch else '-'}; "
                f"W={legacy[0]['source'] if legacy else '-'}")

            try:
                docs.summarize_source("missing.pdf")
                missing_source = False
            except SourceNotFound:
                missing_source = True
            check("missing_source_summary", missing_source,
                  "SourceNotFound not raised")

            try:
                docs.summarize_collection(source_prefix="missing/")
                missing_collection = False
            except SourceNotFound:
                missing_collection = True
            check("missing_collection_summary", missing_collection,
                  "SourceNotFound not raised")
        finally:
            docs.close()

    failed = sum(1 for item in checks if not item["ok"])
    payload = {
        "benchmark": "atomzero-scoped-regression",
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
        "--out", default="benchmarks/results/scoped_regression.json")
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
