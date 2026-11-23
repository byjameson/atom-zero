"""Deterministic source comparison and contradiction hints.

This is not a truth arbiter. It compares claims from different indexed sources
and flags overlapping sentences whose numbers, code identifiers or negation
differ. Every flag keeps both snippets and page/section locators so a human can
inspect the evidence.
"""
from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

from atomzero.doc_memory import _identifier_terms, _query_terms
from atomzero.knowledge_graph import extract_entities

NEGATION_RE = re.compile(
    r"\b(?:not|no|never|cannot|can't|doesn't|don't|isn't|aren't|won't|"
    r"without|neither|nor)\b", re.IGNORECASE)
NUMBER_RE = re.compile(
    r"(?<![\w.])(?P<number>"
    r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|"
    r"\d{1,3}(?:\.\d{3})+(?:,\d+)?|"
    r"\d+(?:[.,]\d+)?)"
    r"(?:\s*(?P<scale>k|thousand|million|billion|trillion))?(?![\w])",
    re.IGNORECASE)
SCALES = {
    "k": 10 ** 3,
    "thousand": 10 ** 3,
    "million": 10 ** 6,
    "billion": 10 ** 9,
    "trillion": 10 ** 12,
}
MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3,
    "mar": 3, "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6,
    "july": 7, "jul": 7, "august": 8, "aug": 8, "september": 9,
    "sep": 9, "sept": 9, "october": 10, "oct": 10, "november": 11,
    "nov": 11, "december": 12, "dec": 12,
    "ocak": 1, "şubat": 2, "subat": 2, "mart": 3, "nisan": 4,
    "mayıs": 5, "mayis": 5, "haziran": 6, "temmuz": 7,
    "ağustos": 8, "agustos": 8, "eylül": 9, "eylul": 9,
    "ekim": 10, "kasım": 11, "kasim": 11, "aralık": 12, "aralik": 12,
}
MONTH_PATTERN = "|".join(
    re.escape(name) for name in sorted(MONTHS, key=len, reverse=True))
DATE_PATTERNS = (
    re.compile(
        r"(?<![\w-])(?P<year>\d{4})-(?P<month>\d{1,2})-(?P<day>\d{1,2})"
        r"(?![\w-])"),
    re.compile(
        rf"\b(?P<month>{MONTH_PATTERN})\.?\s+"
        r"(?P<day>\d{1,2})(?:st|nd|rd|th)?,?\s+(?P<year>\d{4})\b",
        re.IGNORECASE),
    re.compile(
        rf"\b(?P<day>\d{{1,2}})(?:st|nd|rd|th)?(?:\s+of)?[\s-]+"
        rf"(?P<month>{MONTH_PATTERN})\.?,?[\s-]+(?P<year>\d{{4}})\b",
        re.IGNORECASE),
)


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in
            re.split(r"(?<=[.!?])\s+|\n+", str(text or ""))
            if part.strip()]


def _plain_number(value: Decimal) -> str:
    if value == value.to_integral_value():
        return str(value.quantize(Decimal(1)))
    return format(value.normalize(), "f")


def _canonical_number(raw: str, scale: str) -> str:
    """Normalize grouping, decimal comma and magnitude suffixes.

    ``1,500``, ``1500`` and ``1.5k`` all become ``1500``; ``1.5 million``
    becomes ``1500000``. A comma followed by exactly three digits is treated
    as a thousands separator, otherwise it is a decimal comma.
    """
    value = raw
    if re.fullmatch(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?", value):
        value = value.replace(",", "")
    elif re.fullmatch(r"\d{1,3}(?:\.\d{3})+(?:,\d+)?", value):
        value = value.replace(".", "").replace(",", ".")
    else:
        value = value.replace(",", ".")
    try:
        number = Decimal(value)
    except InvalidOperation:
        return raw
    if scale:
        number *= SCALES[scale.lower()]
    return _plain_number(number)


def _numbers(text: str) -> set[str]:
    return {_canonical_number(match.group("number"),
                              match.group("scale") or "")
            for match in NUMBER_RE.finditer(text)}


def _dates(text: str) -> set[str]:
    """Return canonical ISO-like dates found in text.

    Supports ISO dates, ``Month D, YYYY``, ``D Month YYYY`` and Turkish month
    names. Numeric slash/dot dates are deliberately ignored because their
    day/month order is locale-dependent and would create false conflicts.
    """
    found = set()
    for pattern in DATE_PATTERNS:
        for match in pattern.finditer(text):
            try:
                year = int(match.group("year"))
                day = int(match.group("day"))
            except (TypeError, ValueError):
                continue
            raw_month = match.group("month")
            if raw_month.isdigit():
                month = int(raw_month)
            else:
                month = MONTHS.get(raw_month.lower().rstrip("."), 0)
            if 1 <= month <= 12 and 1 <= day <= 31 and 1000 <= year <= 2999:
                found.add(f"{year:04d}-{month:02d}-{day:02d}")
    return found


def _without_dates(text: str) -> str:
    value = str(text or "")
    for pattern in DATE_PATTERNS:
        value = pattern.sub(" ", value)
    return value


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
                "numbers": sorted(_numbers(_without_dates(sentence))),
                "dates": sorted(_dates(sentence)),
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
    dates_left = set(left.get("dates", []))
    dates_right = set(right.get("dates", []))
    if dates_left and dates_right:
        only_left = dates_left - dates_right
        only_right = dates_right - dates_left
        if only_left and only_right:
            reasons.append("date_mismatch")
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
    counts = {"numeric_mismatch": 0, "date_mismatch": 0,
              "identifier_mismatch": 0, "negation_mismatch": 0}
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
        "method": ("sentence overlap + number/date/code/negation difference; "
                   "not a truth verdict"),
        "weights_changed": False,
    }
