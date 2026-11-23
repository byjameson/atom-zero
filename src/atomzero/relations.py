"""Deterministic, local relation extraction.

This is a small pattern-based layer on top of the co-occurrence graph. It only
returns relations that are explicitly written in a sentence and keeps the
sentence, source, title, page/section and both entities. It is not a trained
relation model and does not infer missing facts.
"""
from __future__ import annotations

import re
from collections import deque

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


def _node_key(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def relation_path(docs, source: str, target: str,
                  max_depth: int = 3) -> dict:
    """Find a shortest path through explicit typed-relation edges."""
    source = str(source or "").strip()
    target = str(target or "").strip()
    if not source or not target:
        raise ValueError("source and target are required")
    try:
        max_depth = max(1, min(int(max_depth), 6))
    except (TypeError, ValueError):
        max_depth = 3
    data = extract_relations(docs, limit=5000)
    relations = data.get("relations", [])
    if not relations:
        return {
            "source": source, "target": target, "found": False,
            "reason": "no typed relations indexed", "hops": 0,
            "path": [], "edges": [], "sources": [],
            "weights_changed": False,
        }
    display = {}
    adjacency: dict[str, list] = {}
    for item in relations:
        subject = _node_key(item["subject"])
        object_value = _node_key(item["object"])
        display.setdefault(subject, item["subject"])
        display.setdefault(object_value, item["object"])
        edge = {
            "relation": item["relation"],
            "subject": item["subject"],
            "object": item["object"],
            "source": item["source"],
            "title": item.get("title", ""),
            "page": item.get("page"),
            "section": item.get("section", ""),
            "text": item.get("text", ""),
            "kind": item.get("kind", "document"),
            "verified": bool(item.get("verified", True)),
        }
        adjacency.setdefault(subject, []).append(
            (object_value, "forward", edge))
        adjacency.setdefault(object_value, []).append(
            (subject, "reverse", edge))
    for node in adjacency:
        adjacency[node].sort(
            key=lambda entry: (entry[2]["relation"],
                               entry[0], entry[1]))
    start = _node_key(source)
    goal = _node_key(target)
    if start not in adjacency:
        return {
            "source": source, "target": target, "found": False,
            "reason": "source entity not found", "hops": 0,
            "path": [], "edges": [], "sources": [],
            "weights_changed": False,
        }
    if goal not in adjacency:
        return {
            "source": source, "target": target, "found": False,
            "reason": "target entity not found", "hops": 0,
            "path": [], "edges": [], "sources": [],
            "weights_changed": False,
        }
    if start == goal:
        return {
            "source": source, "target": target, "found": True,
            "reason": "", "hops": 0, "path": [display.get(start, source)],
            "edges": [], "sources": [], "weights_changed": False,
        }
    queue = deque([(start, [start], [])])
    visited = {start}
    while queue:
        node, path, edges = queue.popleft()
        if node == goal:
            sources = []
            for edge in edges:
                item = {
                    "source": edge["source"],
                    "title": edge.get("title", ""),
                    "kind": edge.get("kind", "document"),
                    "verified": bool(edge.get("verified", True)),
                }
                if item not in sources:
                    sources.append(item)
            return {
                "source": source, "target": target, "found": True,
                "reason": "", "hops": len(edges),
                "path": [display.get(name, name) for name in path],
                "edges": edges, "sources": sources,
                "weights_changed": False,
            }
        if len(edges) >= max_depth:
            continue
        for neighbor, direction, edge in adjacency.get(node, []):
            if neighbor in visited:
                continue
            visited.add(neighbor)
            crossed = {**edge, "direction": direction}
            queue.append((neighbor, path + [neighbor], edges + [crossed]))
    return {
        "source": source, "target": target, "found": False,
        "reason": f"no typed relation path within {max_depth} hop(s)",
        "hops": 0, "path": [], "edges": [], "sources": [],
        "weights_changed": False,
    }
