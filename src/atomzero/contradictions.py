"""Deterministic source comparison and contradiction hints.

This is not a truth arbiter. It compares claims from different indexed sources
and flags overlapping sentences whose numbers, code identifiers or negation
differ. Every flag keeps both snippets and page/section locators so a human can
inspect the evidence.
"""
from __future__ import annotations

import re

from atomzero.doc_memory import _identifier_terms, _query_terms
from atomzero.knowledge_graph import extract_entities

NEGATION_RE = re.compile(
    r"\b(?:not|no|never|cannot|can't|doesn't|don't|isn't|aren't|won't|"
    r"without|neither|nor)\b", re.IGNORECASE)
NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?\b")


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in
            re.split(r"(?<=[.!?])\s+|\n+", str(text or ""))
            if part.strip()]


def _numbers(text: str) -> set[str]:
    return {match.replace(",", ".") for match in NUMBER_RE.findall(text)}


def extract_claims(docs, source: str) -> list[dict]:
    """Extract comparable sentence-level claims for one source."""
    source = str(source or "").strip()
    if not source:
        raise ValueError("source is required")
    chunks = [item for item in docs.chunks(5000)
              if item.get("source") == source]
    if not chunks:
        raise ValueError(f"source not found: {source}")
    claims = []
    for item in chunks:
        for sentence in _sentences(item.get("chunk", "")):
            terms = _query_terms(sentence)
            codes = _identifier_terms(sentence)
            if len(terms) < 2 and not codes:
                continue
            claims.append({
                "text": sentence,
                "terms": sorted(terms),
                "numbers": sorted(_numbers(sentence)),
                "codes": sorted(codes),
                "entities": sorted({value.lower()
                                    for value in extract_entities(sentence)}),
                "negated": bool(NEGATION_RE.search(sentence)),
                "source": item.get("source", ""),
                "title": item.get("title", ""),
                "page": item.get("page"),
                "section": item.get("section", ""),
            })
    return claims


def _overlap(left: dict, right: dict) -> tuple[set[str], set[str]]:
    shared_terms = set(left["terms"]) & set(right["terms"])
    shared_entities = set(left["entities"]) & set(right["entities"])
    return shared_terms, shared_entities


def _conflict_reasons(left: dict, right: dict,
                      shared_terms: set[str],
                      shared_entities: set[str]) -> list[str]:
    reasons = []
    numbers_left = set(left["numbers"])
    numbers_right = set(right["numbers"])
    if numbers_left and numbers_right:
        only_left = numbers_left - numbers_right
        only_right = numbers_right - numbers_left
        if only_left and only_right:
            reasons.append("numeric_mismatch")
    codes_left = set(left["codes"])
    codes_right = set(right["codes"])
    if codes_left and codes_right:
        only_left = codes_left - codes_right
        only_right = codes_right - codes_left
        if only_left and only_right:
            reasons.append("identifier_mismatch")
    if left["negated"] != right["negated"] and (
            len(shared_terms) >= 2 or shared_entities):
        reasons.append("negation_mismatch")
    return reasons


def compare_sources(docs, source: str, against: str = "",
                    max_claims: int = 200) -> dict:
    """Compare one source with another source or all other sources."""
    source = str(source or "").strip()
    against = str(against or "").strip()
    if not source:
        raise ValueError("source is required")
    try:
        max_claims = max(20, min(int(max_claims), 1000))
    except (TypeError, ValueError):
        max_claims = 200
    left_claims = extract_claims(docs, source)
    if against:
        right_claims = extract_claims(docs, against)
        candidate_sources = [against]
    else:
        candidate_sources = sorted({
            item.get("source", "") for item in docs.chunks(5000)
            if item.get("source") and item.get("source") != source})
        right_claims = []
        for candidate in candidate_sources:
            try:
                right_claims.extend(extract_claims(docs, candidate))
            except ValueError:
                continue
    left_claims = left_claims[:max_claims]
    right_claims = right_claims[:max_claims]
    conflicts = []
    for left in left_claims:
        for right in right_claims:
            if left["source"] == right["source"]:
                continue
            shared_terms, shared_entities = _overlap(left, right)
            if not (len(shared_terms) >= 2
                    or (shared_entities and shared_terms)):
                continue
            reasons = _conflict_reasons(
                left, right, shared_terms, shared_entities)
            if not reasons:
                continue
            conflicts.append({
                "reasons": reasons,
                "shared_terms": sorted(shared_terms),
                "shared_entities": sorted(shared_entities),
                "left": left,
                "right": right,
            })
            if len(conflicts) >= 100:
                break
        if len(conflicts) >= 100:
            break
    counts = {"numeric_mismatch": 0, "identifier_mismatch": 0,
              "negation_mismatch": 0}
    for conflict in conflicts:
        for reason in conflict["reasons"]:
            counts[reason] = counts.get(reason, 0) + 1
    return {
        "source": source,
        "against": against or "*",
        "candidate_sources": candidate_sources[:50],
        "claims": {"source": len(left_claims),
                   "against": len(right_claims)},
        "conflicts": conflicts[:100],
        "summary": {
            "conflict_count": len(conflicts[:100]),
            "sources_compared": len(candidate_sources),
            **counts,
        },
        "method": ("sentence overlap + number/code/negation difference; "
                   "not a truth verdict"),
        "weights_changed": False,
    }
