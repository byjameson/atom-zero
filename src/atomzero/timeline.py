"""Deterministic document timeline extraction.

The timeline is a local, explainable view over dated sentences. It keeps the
source, title, page/section and entities for every event; it does not infer
events that are not written in the text.
"""
from __future__ import annotations

import re

from atomzero.contradictions import date_matches
from atomzero.knowledge_graph import extract_entities

YEAR_RE = re.compile(r"\b(?:1[0-9]{3}|20[0-9]{2})\b")


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in
            re.split(r"(?<=[.!?])\s+|\n+", str(text or ""))
            if part.strip()]


def _event(item: dict, date: str, precision: str,
           sentence: str) -> dict:
    return {
        "date": date,
        "precision": precision,
        "text": sentence[:500],
        "source": item.get("source", ""),
        "title": item.get("title", ""),
        "page": item.get("page"),
        "section": item.get("section", ""),
        "kind": item.get("kind", "document"),
        "verified": bool(item.get("verified", True)),
        "entities": sorted({str(value) for value in
                            extract_entities(sentence)}),
    }


def extract_timeline(docs, source: str = "", source_prefix: str = "",
                     limit: int = 100) -> dict:
    """Return dated events sorted by date with full provenance."""
    source = str(source or "").strip()
    source_prefix = str(source_prefix or "").strip()
    for label, value in (("source", source),
                         ("source_prefix", source_prefix)):
        if len(value) > 512:
            raise ValueError(f"{label} is too long (max 512 chars)")
        if any(ord(ch) < 32 for ch in value):
            raise ValueError(f"{label} contains control characters")
    try:
        limit = max(1, min(int(limit), 500))
    except (TypeError, ValueError):
        limit = 100

    events = []
    matched_source = False
    for item in docs.chunks(5000):
        value = str(item.get("source", ""))
        if source and value != source:
            continue
        if source_prefix and not value.startswith(source_prefix):
            continue
        matched_source = True
        for sentence in _sentences(item.get("chunk", "")):
            matches = date_matches(sentence)
            if matches:
                for match in matches:
                    events.append(_event(item, match["date"], "day",
                                         sentence))
                continue
            for year in sorted(set(YEAR_RE.findall(sentence))):
                events.append(_event(item, year, "year", sentence))

    if source and not matched_source:
        from atomzero.doc_memory import SourceNotFound

        raise SourceNotFound(f"source not found: {source}")

    deduped = {}
    for event in events:
        key = (event["source"], event["date"], event["text"])
        deduped.setdefault(key, event)
    ordered = sorted(
        deduped.values(),
        key=lambda event: (event["date"], event["source"],
                           event["page"] or 0, event["section"],
                           event["text"]))
    return {
        "source": source,
        "source_prefix": source_prefix,
        "limit": limit,
        "count": len(ordered),
        "events": ordered[:limit],
        "weights_changed": False,
    }
