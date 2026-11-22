"""Local, explainable knowledge graph for Atom Zero.

The graph is statistical, not a trained NER or reasoning model:

- entities are capitalized phrases, code-like identifiers and years;
- an edge means two entities appeared in the same retrieved chunk;
- every edge/node keeps source counts and the evidence endpoint can reopen the
  exact chunks.

It is a "second brain" navigation aid over the local index, not a claim of
semantic understanding.
"""
from __future__ import annotations

import re
from collections import defaultdict, deque
from itertools import combinations

from atomzero.doc_memory import _identifier_terms, best_snippet

ENTITY_RE = re.compile(
    r"\b[A-Z][A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)*"
    r"(?:\s+[A-Z][A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)*){0,2}\b")
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")

STOP_ENTITIES = {
    "The", "This", "These", "Those", "A", "An", "And", "But", "Or", "So",
    "For", "From", "In", "Into", "On", "At", "To", "Of", "With", "When",
    "Where", "What", "Who", "Why", "How", "If", "Then", "There", "Their",
    "They", "It", "Its", "Our", "We", "You", "Your", "My", "He", "She",
    "His", "Her", "As", "By", "Is", "Are", "Was", "Were", "Be", "Been",
    "Being", "Can", "Could", "Should", "Would", "Will", "May", "Might",
    "Must", "Not", "No", "Yes", "All", "Some", "Any", "Each", "Every",
    "Other", "Another", "Such", "Than", "That", "Also", "However",
    "Therefore", "Because", "While", "After", "Before", "During",
    "Between", "Under", "Over", "About", "More", "Most", "Very", "Just",
    "Only", "First", "Last", "Next",
}


def _clean_entity(value: str) -> str:
    value = str(value or "").strip(" \t\r\n.,:;!?()[]{}'\"")
    words = value.split()
    while words and words[0] in STOP_ENTITIES:
        words.pop(0)
    value = " ".join(words).strip()
    if len(value) < 2:
        return ""
    if value in STOP_ENTITIES or value.upper() in {
            "THE", "AND", "FOR", "WITH", "THIS", "THAT"}:
        return ""
    if not value.isupper() and len(value) < 3:
        return ""
    return value[:80]


def extract_entities(text: str) -> list[str]:
    """Extract capitalized phrases, code-like tokens and years."""
    value = str(text or "")
    found = []
    for match in ENTITY_RE.finditer(value):
        entity = _clean_entity(match.group(0))
        if entity:
            found.append(entity)
    for year in YEAR_RE.findall(value):
        found.append(year)
    for code in _identifier_terms(value):
        cleaned = _clean_entity(code)
        if cleaned:
            found.append(cleaned)
    # Stable, case-insensitive de-duplication.
    seen = set()
    result = []
    for entity in found:
        key = entity.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(entity)
    return result


def _clamp(value, low: int, high: int, default: int) -> int:
    try:
        return max(low, min(int(value), high))
    except (TypeError, ValueError):
        return default


def build_graph(docs, limit_chunks: int = 500, min_edge: int = 1,
                max_entities: int = 80) -> dict:
    """Build a co-occurrence graph from the local document chunks."""
    if docs is None:
        raise ValueError("document memory is required")
    limit_chunks = _clamp(limit_chunks, 1, 5000, 500)
    min_edge = _clamp(min_edge, 1, 20, 1)
    max_entities = _clamp(max_entities, 10, 300, 80)
    chunks = docs.chunks(limit_chunks)
    nodes: dict[str, dict] = {}
    edges: dict[tuple[str, str], dict] = {}

    for item in chunks:
        entities = extract_entities(item.get("chunk", ""))[:12]
        if not entities:
            continue
        for name in entities:
            key = name.lower()
            node = nodes.setdefault(
                key, {"name": name, "mentions": 0, "sources": set()})
            node["mentions"] += 1
            node["sources"].add(item.get("source", ""))
        ordered = sorted(entities, key=lambda value: value.lower())
        for left, right in combinations(ordered, 2):
            key = (left.lower(), right.lower())
            edge = edges.setdefault(
                key, {"source": left, "target": right, "weight": 0,
                      "sources": set()})
            edge["weight"] += 1
            edge["sources"].add(item.get("source", ""))

    top_nodes = sorted(
        nodes.values(), key=lambda node: (-node["mentions"],
                                          node["name"].lower()))[:max_entities]
    allowed = {node["name"].lower() for node in top_nodes}
    top_edges = [
        edge for edge in edges.values()
        if edge["weight"] >= min_edge
        and edge["source"].lower() in allowed
        and edge["target"].lower() in allowed]
    top_edges.sort(key=lambda edge: (-edge["weight"],
                                     edge["source"].lower(),
                                     edge["target"].lower()))
    return {
        "nodes": [{
            "id": node["name"].lower(),
            "name": node["name"],
            "mentions": node["mentions"],
            "sources": len(node["sources"]),
        } for node in top_nodes],
        "edges": [{
            "source": edge["source"],
            "target": edge["target"],
            "weight": edge["weight"],
            "sources": len(edge["sources"]),
        } for edge in top_edges],
        "stats": {
            "chunks_scanned": len(chunks),
            "nodes": len(top_nodes),
            "edges": len(top_edges),
            "limit_chunks": limit_chunks,
            "min_edge": min_edge,
        },
        "method": "co-occurrence within a chunk; no trained NER",
    }


def graph_path(docs, source: str, target: str,
               max_depth: int = 4, limit_chunks: int = 1000,
               max_entities: int = 300) -> dict:
    """Find a deterministic co-occurrence path between two entities."""
    if docs is None:
        raise ValueError("document memory is required")
    source = str(source or "").strip()
    target = str(target or "").strip()
    if not source or not target:
        raise ValueError("source and target are required")
    max_depth = _clamp(max_depth, 1, 6, 4)
    if source.lower() == target.lower():
        return {
            "found": True,
            "source": source,
            "target": target,
            "path": [source],
            "edges": [],
            "hops": 0,
            "method": "co-occurrence BFS",
        }
    graph = build_graph(
        docs, limit_chunks=limit_chunks, min_edge=1,
        max_entities=max_entities)
    nodes = {node["name"].lower(): node for node in graph["nodes"]}
    if source.lower() not in nodes or target.lower() not in nodes:
        return {
            "found": False,
            "source": source,
            "target": target,
            "path": [],
            "edges": [],
            "reason": "entity not present in the local graph",
            "available_entities": [node["name"]
                                   for node in graph["nodes"][:12]],
            "method": "co-occurrence BFS",
        }
    adjacency = defaultdict(list)
    edge_map = {}
    for edge in graph["edges"]:
        left = edge["source"].lower()
        right = edge["target"].lower()
        adjacency[left].append(right)
        adjacency[right].append(left)
        edge_map[(left, right)] = edge
        edge_map[(right, left)] = edge
    start = source.lower()
    goal = target.lower()
    best_path = None
    queue = deque([[start]])
    seen = {start}
    while queue and best_path is None:
        path = queue.popleft()
        if len(path) - 1 >= max_depth:
            continue
        for neighbour in sorted(adjacency[path[-1]]):
            if neighbour in path:
                continue
            candidate = path + [neighbour]
            if neighbour == goal:
                best_path = candidate
                break
            if neighbour not in seen:
                seen.add(neighbour)
                queue.append(candidate)
    if best_path is None:
        return {
            "found": False,
            "source": nodes[start]["name"],
            "target": nodes[goal]["name"],
            "path": [],
            "edges": [],
            "reason": f"no path within {max_depth} hop(s)",
            "method": "co-occurrence BFS",
        }
    edges = []
    for left, right in zip(best_path, best_path[1:]):
        edge = edge_map[(left, right)]
        edges.append({
            "source": edge["source"],
            "target": edge["target"],
            "weight": edge["weight"],
            "sources": edge["sources"],
        })
    return {
        "found": True,
        "source": nodes[start]["name"],
        "target": nodes[goal]["name"],
        "path": [nodes[key]["name"] for key in best_path],
        "edges": edges,
        "hops": len(edges),
        "weight": sum(edge["weight"] for edge in edges),
        "method": "co-occurrence BFS",
    }


def entity_sources(docs, entity: str, limit: int = 10) -> dict:
    """Return chunks and graph connections for one entity."""
    if docs is None:
        raise ValueError("document memory is required")
    entity = str(entity or "").strip()
    if not entity:
        raise ValueError("entity is required")
    if len(entity) > 80:
        raise ValueError("entity is too long (max 80 chars)")
    limit = _clamp(limit, 1, 50, 10)
    needle = entity.lower()
    sources = []
    for item in docs.chunks(5000):
        chunk = str(item.get("chunk", ""))
        if needle not in chunk.lower():
            continue
        sources.append({
            "source": item.get("source", ""),
            "title": item.get("title", ""),
            "page": item.get("page"),
            "section": item.get("section", ""),
            "kind": item.get("kind", "document"),
            "verified": bool(item.get("verified", True)),
            "snippet": best_snippet(entity, chunk),
        })
        if len(sources) >= limit:
            break
    graph = build_graph(docs, limit_chunks=1000, max_entities=200)
    connections = [
        edge for edge in graph["edges"]
        if entity.lower() in (edge["source"].lower(),
                              edge["target"].lower())]
    return {
        "entity": entity,
        "sources": sources,
        "connections": connections[:20],
        "matched_chunks": len(sources),
        "method": "literal entity match + chunk co-occurrence",
    }
