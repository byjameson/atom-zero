"""Deterministic source comparison and contradiction hints.

This is not a truth arbiter. It compares claims from different indexed sources
and flags overlapping sentences whose numbers, units, dates, code identifiers
or negation differ. Every flag keeps both snippets and page/section locators so
a human can inspect the evidence.
"""
from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

from atomzero.doc_memory import _identifier_terms, _query_terms
from atomzero.knowledge_graph import extract_entities

NEGATION_RE = re.compile(
    r"\b(?:not|no|never|cannot|can't|doesn't|don't|isn't|aren't|won't|"
    r"without|neither|nor)\b", re.IGNORECASE)
NUMBER_BODY = (
    r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|"
    r"\d{1,3}(?:\.\d{3})+(?:,\d+)?|"
    r"\d+(?:[.,]\d+)?"
)
NUMBER_RE = re.compile(
    rf"(?<![\w.])(?P<number>{NUMBER_BODY})"
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
UNITS = {
    # time -> seconds
    "second": ("time", Decimal(1)), "seconds": ("time", Decimal(1)),
    "sec": ("time", Decimal(1)), "secs": ("time", Decimal(1)),
    "minute": ("time", Decimal(60)), "minutes": ("time", Decimal(60)),
    "min": ("time", Decimal(60)), "mins": ("time", Decimal(60)),
    "hour": ("time", Decimal(3600)), "hours": ("time", Decimal(3600)),
    "hr": ("time", Decimal(3600)), "hrs": ("time", Decimal(3600)),
    "day": ("time", Decimal(86400)), "days": ("time", Decimal(86400)),
    "week": ("time", Decimal(604800)),
    "weeks": ("time", Decimal(604800)),
    "saniye": ("time", Decimal(1)), "saniyeler": ("time", Decimal(1)),
    "dakika": ("time", Decimal(60)),
    "dakikalar": ("time", Decimal(60)),
    "saat": ("time", Decimal(3600)), "saatler": ("time", Decimal(3600)),
    "gün": ("time", Decimal(86400)), "gun": ("time", Decimal(86400)),
    "günler": ("time", Decimal(86400)),
    "gunler": ("time", Decimal(86400)),
    "hafta": ("time", Decimal(604800)),
    "haftalar": ("time", Decimal(604800)),
    # months -> months (no month/day conversion because it is not exact)
    "month": ("months", Decimal(1)), "months": ("months", Decimal(1)),
    "year": ("months", Decimal(12)), "years": ("months", Decimal(12)),
    "ay": ("months", Decimal(1)), "aylar": ("months", Decimal(1)),
    "yıl": ("months", Decimal(12)), "yil": ("months", Decimal(12)),
    "yıllar": ("months", Decimal(12)),
    "yillar": ("months", Decimal(12)),
    # length -> meters
    "millimeter": ("length", Decimal("0.001")),
    "millimeters": ("length", Decimal("0.001")),
    "millimetre": ("length", Decimal("0.001")),
    "millimetres": ("length", Decimal("0.001")),
    "mm": ("length", Decimal("0.001")),
    "centimeter": ("length", Decimal("0.01")),
    "centimeters": ("length", Decimal("0.01")),
    "centimetre": ("length", Decimal("0.01")),
    "centimetres": ("length", Decimal("0.01")),
    "cm": ("length", Decimal("0.01")),
    "meter": ("length", Decimal(1)), "meters": ("length", Decimal(1)),
    "metre": ("length", Decimal(1)), "metres": ("length", Decimal(1)),
    "kilometer": ("length", Decimal(1000)),
    "kilometers": ("length", Decimal(1000)),
    "kilometre": ("length", Decimal(1000)),
    "kilometres": ("length", Decimal(1000)),
    "km": ("length", Decimal(1000)),
    "inch": ("length", Decimal("0.0254")),
    "inches": ("length", Decimal("0.0254")),
    "foot": ("length", Decimal("0.3048")),
    "feet": ("length", Decimal("0.3048")),
    "ft": ("length", Decimal("0.3048")),
    "yard": ("length", Decimal("0.9144")),
    "yards": ("length", Decimal("0.9144")),
    "mile": ("length", Decimal("1609.344")),
    "miles": ("length", Decimal("1609.344")),
    # mass -> grams
    "milligram": ("mass", Decimal("0.001")),
    "milligrams": ("mass", Decimal("0.001")),
    "mg": ("mass", Decimal("0.001")),
    "gram": ("mass", Decimal(1)), "grams": ("mass", Decimal(1)),
    "kilogram": ("mass", Decimal(1000)),
    "kilograms": ("mass", Decimal(1000)),
    "kg": ("mass", Decimal(1000)),
    "pound": ("mass", Decimal("453.59237")),
    "pounds": ("mass", Decimal("453.59237")),
    "lb": ("mass", Decimal("453.59237")),
    "lbs": ("mass", Decimal("453.59237")),
    "ounce": ("mass", Decimal("28.349523125")),
    "ounces": ("mass", Decimal("28.349523125")),
    "oz": ("mass", Decimal("28.349523125")),
    "ton": ("mass", Decimal(1000000)),
    "tons": ("mass", Decimal(1000000)),
    "tonne": ("mass", Decimal(1000000)),
    "tonnes": ("mass", Decimal(1000000)),
    # volume -> liters
    "milliliter": ("volume", Decimal("0.001")),
    "milliliters": ("volume", Decimal("0.001")),
    "millilitre": ("volume", Decimal("0.001")),
    "millilitres": ("volume", Decimal("0.001")),
    "ml": ("volume", Decimal("0.001")),
    "liter": ("volume", Decimal(1)), "liters": ("volume", Decimal(1)),
    "litre": ("volume", Decimal(1)), "litres": ("volume", Decimal(1)),
    "gallon": ("volume", Decimal("3.785411784")),
    "gallons": ("volume", Decimal("3.785411784")),
    "percent": ("percent", Decimal(1)),
}
UNIT_PATTERN = "|".join(
    re.escape(unit) for unit in sorted(UNITS, key=len, reverse=True))
MEASURE_RE = re.compile(
    rf"(?<![\w.])(?P<number>{NUMBER_BODY})"
    r"(?:\s*(?P<scale>k|thousand|million|billion|trillion))?"
    rf"(?:\s*(?P<unit>%|{UNIT_PATTERN}))?(?![\w])",
    re.IGNORECASE)


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in
            re.split(r"(?<=[.!?])\s+|\n+", str(text or ""))
            if part.strip()]


def _plain_number(value: Decimal) -> str:
    if value == value.to_integral_value():
        return str(value.quantize(Decimal(1)))
    return format(value.normalize(), "f")


def _number_value(raw: str, scale: str = "") -> Decimal | None:
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
        return None
    if scale:
        number *= SCALES[scale.lower()]
    return number


def _canonical_number(raw: str, scale: str) -> str:
    """Normalize grouping, decimal comma and magnitude suffixes.

    ``1,500``, ``1500`` and ``1.5k`` all become ``1500``; ``1.5 million``
    becomes ``1500000``. A comma followed by exactly three digits is treated
    as a thousands separator, otherwise it is a decimal comma.
    """
    number = _number_value(raw, scale)
    return _plain_number(number) if number is not None else raw


def _canonical_measure(raw: str, scale: str, unit: str) -> str | None:
    """Normalize a number plus unit into a base unit for its dimension.

    ``24 months`` and ``2 years`` both become ``months:24``; ``72 hours`` and
    ``3 days`` both become ``time:259200``; ``5 km`` and ``5000 meters`` both
    become ``length:5000``. Incompatible dimensions stay separate on purpose.
    """
    key = unit.lower().rstrip(".")
    if key == "%":
        dimension, factor = "percent", Decimal(1)
    else:
        entry = UNITS.get(key)
        if entry is None:
            return None
        dimension, factor = entry
    number = _number_value(raw, scale)
    if number is None:
        return None
    return f"{dimension}:{_plain_number(number * factor)}"


def _numbers_and_measures(text: str) -> tuple[set[str], set[str]]:
    numbers: set[str] = set()
    measures: set[str] = set()
    for match in MEASURE_RE.finditer(text):
        unit = match.group("unit") or ""
        scale = match.group("scale") or ""
        if unit:
            measure = _canonical_measure(match.group("number"), scale, unit)
            if measure is not None:
                measures.add(measure)
                continue
        numbers.add(_canonical_number(match.group("number"), scale))
    return numbers, measures


def _numbers(text: str) -> set[str]:
    return _numbers_and_measures(text)[0]


def _measures(text: str) -> set[str]:
    return _numbers_and_measures(text)[1]


def _measures_by_dimension(measures) -> dict[str, set[str]]:
    grouped: dict[str, set[str]] = {}
    for item in measures:
        dimension, _, value = str(item).partition(":")
        if dimension and value:
            grouped.setdefault(dimension, set()).add(value)
    return grouped


def date_matches(text: str) -> list[dict]:
    """Return full-date matches with an ISO date, source span and precision."""
    found = []
    seen = set()
    for pattern in DATE_PATTERNS:
        for match in pattern.finditer(text):
            span = (match.start(), match.end())
            if span in seen:
                continue
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
            if not (1 <= month <= 12 and 1 <= day <= 31
                    and 1000 <= year <= 2999):
                continue
            seen.add(span)
            found.append({
                "date": f"{year:04d}-{month:02d}-{day:02d}",
                "precision": "day",
                "start": span[0],
                "end": span[1],
                "text": match.group(0),
            })
    return sorted(found, key=lambda item: item["start"])


def _dates(text: str) -> set[str]:
    """Return canonical ISO-like dates found in text.

    Supports ISO dates, ``Month D, YYYY``, ``D Month YYYY`` and Turkish month
    names. Numeric slash/dot dates are deliberately ignored because their
    day/month order is locale-dependent and would create false conflicts.
    """
    return {match["date"] for match in date_matches(text)}


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
            body = _without_dates(sentence)
            numbers, measures = _numbers_and_measures(body)
            title_value = str(item.get("title", ""))
            section_value = str(item.get("section", ""))
            claims.append({
                "text": sentence,
                "terms": sorted(terms),
                "numbers": sorted(numbers),
                "measures": sorted(measures),
                "dates": sorted(_dates(sentence)),
                "codes": sorted(codes),
                "entities": sorted({value.lower()
                                    for value in extract_entities(sentence)}),
                "topic_terms": sorted(_query_terms(
                    f"{title_value} {section_value}")),
                "negated": bool(NEGATION_RE.search(sentence)),
                "source": item.get("source", ""),
                "title": title_value,
                "page": item.get("page"),
                "section": section_value,
            })
    return claims


def _overlap(left: dict, right: dict) -> tuple[set[str], set[str]]:
    shared_terms = set(left["terms"]) & set(right["terms"])
    shared_entities = set(left["entities"]) & set(right["entities"])
    return shared_terms, shared_entities


def _topic_overlap(left: dict, right: dict) -> set[str]:
    return set(left.get("topic_terms", [])) & set(
        right.get("topic_terms", []))


def _topic_comparable(left: dict, right: dict) -> bool:
    """Return True when two claims have a comparable evidence dimension."""
    left_measures = _measures_by_dimension(left.get("measures", []))
    right_measures = _measures_by_dimension(right.get("measures", []))
    if set(left_measures) & set(right_measures):
        return True
    if left.get("dates") and right.get("dates"):
        return True
    if left.get("codes") and right.get("codes"):
        return True
    return False


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
    measures_left = _measures_by_dimension(left.get("measures", []))
    measures_right = _measures_by_dimension(right.get("measures", []))
    for dimension in sorted(set(measures_left) & set(measures_right)):
        only_left = measures_left[dimension] - measures_right[dimension]
        only_right = measures_right[dimension] - measures_left[dimension]
        if only_left and only_right:
            if "numeric_mismatch" not in reasons:
                reasons.append("numeric_mismatch")
            break
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
            lexical = (len(shared_terms) >= 2
                       or (shared_entities and shared_terms))
            topic_terms = _topic_overlap(left, right)
            topic = bool(topic_terms) and _topic_comparable(left, right)
            if not (lexical or topic):
                continue
            reasons = _conflict_reasons(
                left, right, shared_terms, shared_entities)
            if not reasons:
                continue
            conflicts.append({
                "reasons": reasons,
                "alignment": "lexical" if lexical else "topic",
                "shared_terms": sorted(shared_terms),
                "shared_entities": sorted(shared_entities),
                "topic_terms": sorted(topic_terms),
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
        "method": ("sentence overlap/topic + number/unit/date/code/negation "
                   "difference; not a truth verdict"),
        "weights_changed": False,
    }
