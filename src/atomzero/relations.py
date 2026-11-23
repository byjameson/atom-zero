"""Deterministic, local relation extraction.

This is a small pattern-based layer on top of the co-occurrence graph. It only
returns relations that are explicitly written in a sentence and keeps the
sentence, source, title, page/section and both entities. It is not a trained
relation model and does not infer missing facts.
"""
from __future__ import annotations

import re

ENTITY = (
    r"[A-ZÇĞİÖŞÜ][\wÇĞİÖŞÜçğıöşü.-]*"
    r"(?:\s+[A-ZÇĞİÖŞÜ][\wÇĞİÖŞÜçğıöşü.-]*)*"
)
RELATION_PATTERNS = (
    ("capital_of", re.compile(
        rf"(?P<subject>{ENTITY})\s+(?i:is)\s+(?:the\s+)?"
        rf"capital\s+of\s+(?P<object>{ENTITY})")),
    ("capital_of", re.compile(
        rf"(?P<subject>{ENTITY})\s*,?\s*(?P<object>{ENTITY})"
        rf"['’]?(?:n[ıi]n|in|un|ün)?\s+başkentidir")),
    ("located_in", re.compile(
        rf"(?P<subject>{ENTITY})\s+(?i:is)\s+(?:located\s+)?"
        rf"in\s+(?P<object>{ENTITY})")),
    ("located_in", re.compile(
        rf"(?P<subject>{ENTITY})\s*,?\s*(?P<object>{ENTITY})"
        rf"['’]?(?:d[ae]|t[ae])\s+bulunur")),
    ("founded_by", re.compile(
        rf"(?P<subject>{ENTITY})\s+(?i:was)\s+(?i:founded)\s+"
        rf"(?i:by)\s+(?P<object>{ENTITY})")),
    ("founded_by", re.compile(
        rf"(?P<subject>{ENTITY})\s*,?\s*(?P<object>{ENTITY})\s+"
        rf"tarafından\s+kuruldu")),
    ("founded", re.compile(
        rf"(?P<subject>{ENTITY})\s+(?i:founded)\s+(?P<object>{ENTITY})")),
    ("founded", re.compile(
        rf"(?P<subject>{ENTITY})\s*,?\s*(?P<object>{ENTITY})"
        rf"['’]?(?:y[ıiuü])?\s+kurdu")),
    ("ceo_of", re.compile(
        rf"(?P<subject>{ENTITY})\s+(?i:is)\s+(?:the\s+)?"
        rf"(?i:CEO|chief\s+executive)\s+of\s+(?P<object>{ENTITY})")),
    ("ceo_of", re.compile(
        rf"(?P<subject>{ENTITY})\s*,?\s*(?P<object>{ENTITY})"
        rf"['’]?(?:n[ıi]n|in|un|ün)?\s+CEO['’]?(?:s[üu]|s[ıi])?")),
    ("born_in", re.compile(
        rf"(?P<subject>{ENTITY})\s+(?i:was)\s+(?i:born)\s+"
        rf"(?i:in)\s+(?P<object>{ENTITY})")),
    ("born_in", re.compile(
        rf"(?P<subject>{ENTITY})\s*,?\s*(?P<object>{ENTITY})"
        rf"['’]?(?:d[ae]|t[ae])\s+doğdu")),
)
RELATION_NAMES = frozenset(name for name, _ in RELATION_PATTERNS)
_GENERIC = frozenset({
    "the", "a", "an", "this", "that", "it", "he", "she", "they", "we", "you",
    "his", "her", "their", "our",
})


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in
            re.split(r"(?<=[.!?])\s+|\n+", str(text or ""))
            if part.strip()]


def _clean(value: str) -> str:
    value = re.sub(r"\s+", " ", str(value or "")).strip(
        " .,;:!?()[]{}\"'’")
    if len(value) < 2 or value.casefold() in _GENERIC:
        return ""
    return value


def extract_relations(docs, entity: str = "", relation: str = "",
                      source: str = "", source_prefix: str = "",
                      limit: int = 100) -> dict:
    """Return explicit relation triples with full provenance."""
    entity = str(entity or "").strip()
    relation = str(relation or "").strip().casefold()
    source = str(source or "").strip()
    source_prefix = str(source_prefix or "").strip()
    for label, value in (("source", source),
                         ("source_prefix", source_prefix)):
        if len(value) > 512:
            raise ValueError(f"{label} is too long (max 512 chars)")
        if any(ord(ch) < 32 for ch in value):
            raise ValueError(f"{label} contains control characters")
    if relation and relation not in RELATION_NAMES:
        raise ValueError(
            "unknown relation; expected one of: "
            + ", ".join(sorted(RELATION_NAMES)))
    try:
        limit = max(1, min(int(limit), 500))
    except (TypeError, ValueError):
        limit = 100

    entity_key = entity.casefold()
    found = []
    matched_source = False
    for item in docs.chunks(5000):
        value = str(item.get("source", ""))
        if source and value != source:
            continue
        if source_prefix and not value.startswith(source_prefix):
            continue
        matched_source = True
        for sentence in _sentences(item.get("chunk", "")):
            for name, pattern in RELATION_PATTERNS:
                if relation and name != relation:
                    continue
                for match in pattern.finditer(sentence):
                    subject = _clean(match.group("subject"))
                    object_value = _clean(match.group("object"))
                    if not subject or not object_value:
                        continue
                    if subject.casefold() == object_value.casefold():
                        continue
                    if entity_key and entity_key not in subject.casefold() \
                            and entity_key not in object_value.casefold():
                        continue
                    found.append({
                        "relation": name,
                        "subject": subject,
                        "object": object_value,
                        "text": sentence[:500],
                        "source": value,
                        "title": item.get("title", ""),
                        "page": item.get("page"),
                        "section": item.get("section", ""),
                        "kind": item.get("kind", "document"),
                        "verified": bool(item.get("verified", True)),
                        "entities": sorted({subject, object_value}),
                    })

    if source and not matched_source:
        from atomzero.doc_memory import SourceNotFound

        raise SourceNotFound(f"source not found: {source}")

    deduped = {}
    for item in found:
        key = (item["relation"], item["subject"].casefold(),
               item["object"].casefold(), item["source"], item["page"],
               item["section"])
        deduped.setdefault(key, item)
    ordered = sorted(
        deduped.values(),
        key=lambda item: (item["relation"], item["subject"].casefold(),
                          item["object"].casefold(), item["source"],
                          item["page"] or 0, item["section"]))
    return {
        "entity": entity,
        "relation": relation,
        "source": source,
        "source_prefix": source_prefix,
        "limit": limit,
        "count": len(ordered),
        "relations": ordered[:limit],
        "weights_changed": False,
    }
