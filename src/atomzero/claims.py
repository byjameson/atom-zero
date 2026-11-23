"""Deterministic claim extraction, evidence linking and conflict audit.

This module is a small, standalone branch for Atom Zero: it turns a text into
sentence-level claims, attaches the best retrieved evidence to each claim and
compares claims with transparent number/unit/date/code/negation
normalisation. It never calls a model and never changes retrievals or weights.

    from atomzero.claims import audit_answer
    report = audit_answer("The battery lasts 8 hours.", hits)

The score is lexical term coverage plus exact token overlap. It is a useful
audit signal, not a natural-language-inference verdict. See ``docs/CLAIMS.md``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

from atomzero.contradictions import (
    NEGATION_RE,
    _conflict_reasons,
    _dates,
    _numbers_and_measures,
)
from atomzero.doc_memory import (
    _identifier_terms,
    _query_terms,
    _term_coverage,
    best_snippet,
)
from atomzero.knowledge_graph import extract_entities

METHOD = "deterministic_claim_evidence_v1"
DEFAULT_MIN_COVERAGE = 0.45
MIN_CLAIM_CHARS = 12
MIN_CLAIM_TERMS = 2
MAX_EVIDENCE_CHARS = 400
MAX_CONFLICT_PAIRS = 100
SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+|\n+")
WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)


def _normalized(text: str) -> str:
    return " ".join(str(text or "").lower().split())


def _claim_id(text: str, index: int) -> str:
    """Return a stable ID from the filtered position and normalised text."""
    digest = hashlib.sha1(
        _normalized(text).encode("utf-8")).hexdigest()[:8]
    return f"c{index + 1:03d}-{digest}"


def _trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    body = text[start:end]
    left = len(body) - len(body.lstrip())
    right = len(body) - len(body.rstrip())
    return start + left, end - right


def extract_claims(text: str) -> list[dict]:
    """Split text into sentence-level claims with exact source offsets.

    Sentences are separated by ``.``, ``!`` or ``?`` followed by whitespace,
    or by one or more newlines. Empty fragments and fragments shorter than
    ``MIN_CLAIM_CHARS`` or ``MIN_CLAIM_TERMS`` words are skipped. IDs combine
    the filtered sentence position with a short hash of the normalised text,
    so the same input always produces the same IDs.
    """
    value = str(text or "")
    claims = []
    segment_start = 0
    spans = [(match.start(), match.end())
             for match in SENTENCE_BOUNDARY_RE.finditer(value)]
    spans.append((len(value), len(value)))
    for segment_end, next_start in spans:
        start, end = _trim_span(value, segment_start, segment_end)
        segment_start = next_start
        if start >= end:
            continue
        body = value[start:end]
        if (len(body) < MIN_CLAIM_CHARS
                or len(WORD_RE.findall(body)) < MIN_CLAIM_TERMS):
            continue
        claims.append({
            "claim_id": _claim_id(body, len(claims)),
            "text": body,
            "start": start,
            "end": end,
        })
    return claims


def _claim_text(claim) -> str:
    if isinstance(claim, dict):
        return str(claim.get("text", ""))
    return str(claim or "")


def _claim_identifier(claim, index: int) -> str:
    if isinstance(claim, dict) and claim.get("claim_id"):
        return str(claim["claim_id"])
    return _claim_id(_claim_text(claim), index)


def _hit_context(hit: dict) -> str:
    title = str(hit.get("title", ""))
    section = str(hit.get("section", ""))
    chunk = str(hit.get("chunk", ""))
    return " ".join(part for part in (title, section, chunk) if part).strip()


def _support_score(terms: set[str], codes: set[str], context: str) -> float:
    """Blend term coverage, exact token overlap and identifier coverage."""
    coverage = _term_coverage(terms, context)
    overlap = (
        len(terms & _query_terms(context)) / len(terms) if terms else 0.0)
    if codes:
        code_coverage = (
            len(codes & _identifier_terms(context)) / len(codes))
        return max(0.0, min(
            1.0, 0.55 * coverage + 0.25 * overlap + 0.20 * code_coverage))
    return max(0.0, min(1.0, 0.75 * coverage + 0.25 * overlap))


def link_evidence(claims: list[dict], hits: list[dict],
                  min_coverage: float = DEFAULT_MIN_COVERAGE) -> list[dict]:
    """Attach the best supporting hit to every claim.

    Each hit is scored by the fraction of claim terms it covers (with the
    deterministic morphology allowance from
    :func:`atomzero.doc_memory._term_coverage`), exact token overlap and, when
    present, identifier coverage. The highest score wins; ties keep the
    earlier hit, so the result is stable for a given hit order. ``support`` is
    that score in ``0..1``. ``status`` is ``supported`` at or above
    ``min_coverage``, ``weak`` for a non-zero score below it, and
    ``unsupported`` when no hit shares any claim term.
    """
    try:
        threshold = float(min_coverage)
    except (TypeError, ValueError):
        threshold = DEFAULT_MIN_COVERAGE
    threshold = max(0.0, min(1.0, threshold))
    hit_list = [hit for hit in (hits or []) if isinstance(hit, dict)]
    linked = []
    for index, claim in enumerate(claims or []):
        text = _claim_text(claim)
        terms = _query_terms(text)
        codes = _identifier_terms(text)
        best_score = 0.0
        best_hit = None
        for hit in hit_list:
            chunk = str(hit.get("chunk", "")).strip()
            if not chunk:
                continue
            score = _support_score(terms, codes, _hit_context(hit))
            if score > best_score:
                best_score = score
                best_hit = hit
        evidence = None
        if best_hit is not None and best_score > 0.0:
            chunk = str(best_hit.get("chunk", ""))
            evidence = {
                "source": best_hit.get("source", ""),
                "title": best_hit.get("title", ""),
                "section": best_hit.get("section", ""),
                "page": best_hit.get("page"),
                "snippet": best_snippet(
                    text, chunk, max_chars=MAX_EVIDENCE_CHARS),
            }
        if evidence is None:
            status = "unsupported"
        elif best_score >= threshold:
            status = "supported"
        else:
            status = "weak"
        item = dict(claim) if isinstance(claim, dict) else {"text": text}
        item["claim_id"] = _claim_identifier(claim, index)
        item["text"] = text
        item["support"] = round(best_score, 4)
        item["status"] = status
        item["evidence"] = evidence
        linked.append(item)
    return linked


def _comparison_features(text: str) -> dict:
    numbers, measures = _numbers_and_measures(text)
    return {
        "text": text,
        "terms": sorted(_query_terms(text)),
        "numbers": sorted(numbers),
        "measures": sorted(measures),
        "dates": sorted(_dates(text)),
        "codes": sorted(_identifier_terms(text)),
        "entities": sorted({entity.lower()
                            for entity in extract_entities(text)}),
        "negated": bool(NEGATION_RE.search(text)),
    }


def compare_claims(claims: list[dict]) -> dict:
    """Flag phrase-level numeric, date, identifier and negation conflicts.

    Two claims are compared only when they share at least two terms, or an
    entity plus a term. This lexical gate keeps unrelated claims with
    different numbers from being reported as contradictions. Number, unit,
    date, code and negation normalisation is imported from
    :mod:`atomzero.contradictions`; no model is involved and the result is not
    a truth verdict.
    """
    features = [_comparison_features(_claim_text(claim))
                for claim in (claims or [])]
    identifiers = [_claim_identifier(claim, index)
                   for index, claim in enumerate(claims or [])]
    conflicts = []
    counts = {"numeric_mismatch": 0, "date_mismatch": 0,
              "identifier_mismatch": 0, "negation_mismatch": 0}
    for left_index in range(len(features)):
        for right_index in range(left_index + 1, len(features)):
            left = features[left_index]
            right = features[right_index]
            shared_terms = set(left["terms"]) & set(right["terms"])
            shared_entities = set(left["entities"]) & set(right["entities"])
            lexical = (len(shared_terms) >= 2
                       or (shared_entities and shared_terms))
            if not lexical:
                continue
            reasons = _conflict_reasons(
                left, right, shared_terms, shared_entities)
            if not reasons:
                continue
            conflicts.append({
                "claim_ids": [identifiers[left_index],
                              identifiers[right_index]],
                "reasons": reasons,
                "shared_terms": sorted(shared_terms),
                "shared_entities": sorted(shared_entities),
                "left": left["text"],
                "right": right["text"],
            })
            for reason in reasons:
                counts[reason] = counts.get(reason, 0) + 1
            if len(conflicts) >= MAX_CONFLICT_PAIRS:
                break
        if len(conflicts) >= MAX_CONFLICT_PAIRS:
            break
    return {
        "claim_count": len(features),
        "conflict_count": len(conflicts),
        "conflicts": conflicts,
        "counts": counts,
        "method": ("phrase overlap + number/unit/date/code/negation "
                   "normalisation; not a truth verdict"),
        "weights_changed": False,
    }


def audit_answer(query: str, hits: list[dict]) -> dict:
    """Build a complete deterministic claim/evidence audit for one query."""
    text = str(query or "")
    claims = extract_claims(text)
    linked = link_evidence(claims, hits or [])
    comparison = compare_claims(linked)
    status_counts = {"supported": 0, "weak": 0, "unsupported": 0}
    sources = set()
    for item in linked:
        status_counts[item["status"]] += 1
        evidence = item.get("evidence")
        if evidence and evidence.get("source"):
            sources.add(str(evidence["source"]))
    summary = {
        "claim_count": len(linked),
        "evidence_count": sum(1 for item in linked if item["evidence"]),
        "status_counts": status_counts,
        "sources": sorted(sources),
        "conflict_count": comparison["conflict_count"],
        "conflict_types": {
            reason: count for reason, count in comparison["counts"].items()
            if count
        },
        "min_coverage": DEFAULT_MIN_COVERAGE,
    }
    return {
        "query": text,
        "claims": linked,
        "summary": summary,
        "contradictions": comparison,
        "weights_changed": False,
        "method": METHOD,
    }


def _format_report(result: dict) -> str:
    summary = result["summary"]
    counts = summary["status_counts"]
    lines = [
        f"Query: {result['query']}",
        (f"Claims: {summary['claim_count']} "
         f"(supported={counts['supported']}, weak={counts['weak']}, "
         f"unsupported={counts['unsupported']})"),
    ]
    for item in result["claims"]:
        evidence = item.get("evidence") or {}
        source = evidence.get("source", "")
        suffix = f" source={source}" if source else ""
        lines.append(
            f"- {item['claim_id']} {item['status']} "
            f"{item['support']:.3f}{suffix}: {item['text']}")
    lines.append(f"Contradictions: {summary['conflict_count']}")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="atomzero-claims",
        description="Deterministic claim and evidence audit.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--text", help="text to audit")
    group.add_argument("--file", help="read text from a UTF-8 file")
    parser.add_argument("--json", action="store_true",
                        help="print the full JSON report")
    args = parser.parse_args(argv)

    if args.text is not None:
        text = args.text
    elif args.file:
        try:
            text = Path(args.file).read_text(encoding="utf-8-sig")
        except OSError as exc:
            print(f"cannot read file: {exc}", file=sys.stderr)
            return 2
    elif not sys.stdin.isatty():
        text = sys.stdin.read()
    else:
        text = ""
    if not str(text or "").strip():
        print("empty input", file=sys.stderr)
        return 2

    result = audit_answer(text, [])
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(_format_report(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
