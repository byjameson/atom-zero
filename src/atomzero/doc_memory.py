"""Local document memory with SQLite FTS5 and provenance.

Documents are chunked, indexed and retrieved with their source. Facts stay in
this index, not in the model weights.

    python -m atomzero.doc_memory add notes.txt --db data/docs.db
    python -m atomzero.doc_memory --db data/docs.db search "kira"
"""
from __future__ import annotations

import argparse
import array
import hashlib
import json
import math
import re
import sqlite3
import threading
import time
import zlib
from collections import Counter
from pathlib import Path

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is a declared dependency
    np = None

MAX_CHARS = 900


class SourceNotFound(ValueError):
    pass


STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but",
    "by", "can", "could", "did", "do", "does", "for", "from", "had", "has",
    "have", "he", "her", "here", "him", "his", "how", "i", "if", "in",
    "into", "is", "it", "its", "me", "my", "no", "not", "of", "on", "or",
    "our", "please", "she", "should", "so", "tell", "than", "that", "the",
    "their", "them", "then", "there", "these", "they", "this", "those",
    "to", "was", "we", "were", "what", "when", "where", "which", "who",
    "why", "will", "with", "would", "you", "your",
})


def _split_long(paragraph: str) -> list[str]:
    """Split a long paragraph near a word boundary, never mid-word if possible."""
    out = []
    while len(paragraph) > MAX_CHARS:
        cut = paragraph.rfind(" ", 0, MAX_CHARS)
        if cut < MAX_CHARS // 2:
            cut = MAX_CHARS
        piece = paragraph[:cut].strip()
        if piece:
            out.append(piece)
        paragraph = paragraph[cut:].lstrip()
    if paragraph.strip():
        out.append(paragraph.strip())
    return out


EMBED_DIM = 512
HASHED_CACHE_KEY = "hashed-ngram-v1"
# Minimum cosine for a semantic-only candidate. Lexical FTS hits are always
# kept; this only stops unrelated chunks from being dragged into the pool.
HYBRID_MIN_HASHED = 0.28
HYBRID_MIN_NEURAL = 0.30
ANN_MIN_CHUNKS = 256
ANN_BUILD_ITERATIONS = 8


class _AnnFallback(RuntimeError):
    """Internal signal: use the exact matrix for this query."""


def _char_ngrams(text: str) -> list[str]:
    words = re.findall(r"[^\W_]+", str(text).lower(), re.UNICODE)
    grams = []
    for word in words:
        padded = f" {word} "
        for size in (3, 4, 5):
            grams.extend(padded[index:index + size]
                         for index in range(len(padded) - size + 1))
    return grams


def embed_text(text: str, dim: int = EMBED_DIM) -> list[float]:
    """Deterministic hashed character n-gram embedding (no neural model)."""
    vector = [0.0] * dim
    for gram in _char_ngrams(text):
        vector[zlib.crc32(gram.encode("utf-8")) % dim] += 1.0
    norm = math.sqrt(sum(value * value for value in vector))
    if norm:
        vector = [value / norm for value in vector]
    return vector


def embedding_similarity(left: str, right: str) -> float:
    a, b = embed_text(left), embed_text(right)
    return sum(x * y for x, y in zip(a, b))


CODE_RE = re.compile(
    r"\b[A-Za-z]{1,8}[-_]?\d{2,}\b|\b\d{2,}[-_][A-Za-z]{1,8}\b")


def _identifier_terms(text: str) -> set[str]:
    """Return code-like tokens such as L-7781, VB_42 or 42-A."""
    return {match.group(0).lower()
            for match in CODE_RE.finditer(str(text or ""))}


def _query_terms(text: str) -> set[str]:
    value = str(text or "")
    words = re.findall(r"[^\W_]+", value.lower(), re.UNICODE)
    terms = {word for word in words
             if len(word) > 2 and word not in STOPWORDS}
    terms.update(_identifier_terms(value))
    return terms


def _term_coverage(terms: set[str], text: str) -> float:
    """Fraction of query terms present, with a small morphology allowance.

    Exact token overlap misses ``claim``/``claims`` and
    ``submit``/``submitted``. Requiring both tokens to be at least four
    characters keeps this from turning into an aggressive fuzzy match.
    """
    if not terms:
        return 0.0
    words = _query_terms(text)
    matched = 0
    for term in terms:
        if term in words:
            matched += 1
            continue
        if len(term) < 4:
            continue
        if any(len(word) >= 4 and
               (word.startswith(term) or term.startswith(word))
               for word in words):
            matched += 1
    return matched / len(terms)


def _pack_vector(values) -> bytes:
    return array.array("f", (float(value) for value in values)).tobytes()


def _unpack_vector(blob: bytes):
    values = array.array("f")
    values.frombytes(blob)
    return values


def _normalized_rows(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def build_ivf(matrix, nlist: int | None = None,
              iterations: int = ANN_BUILD_ITERATIONS) -> dict | None:
    """Deterministic k-means IVF index over normalized rows.

    This is approximate retrieval: a query scores cluster centroids, probes
    the best `nprobe` buckets, then reranks only their members exactly.
    """
    if np is None or matrix is None or len(matrix) < 2:
        return None
    n = len(matrix)
    if nlist is None:
        nlist = min(64, max(2, int(round(math.sqrt(n)))))
    nlist = min(max(2, int(nlist)), n)
    starts = np.linspace(0, n - 1, nlist).astype(int)
    centroids = matrix[starts].copy()
    for _ in range(max(1, iterations)):
        assignments = (matrix @ centroids.T).argmax(axis=1)
        updated = centroids.copy()
        for cluster in range(nlist):
            members = matrix[assignments == cluster]
            if len(members):
                updated[cluster] = members.mean(axis=0)
        centroids = _normalized_rows(updated)
    assignments = (matrix @ centroids.T).argmax(axis=1)
    buckets = [np.where(assignments == cluster)[0].astype(np.int32)
               for cluster in range(nlist)]
    return {"centroids": centroids, "buckets": buckets, "nlist": nlist}


def ann_topk(matrix, ann: dict, query_vector, limit: int,
             probe: int | None = None):
    """Return exact-scored top-k within the best `probe` IVF buckets."""
    if ann is None or matrix is None:
        raise _AnnFallback("no ANN index")
    nlist = int(ann["nlist"])
    if probe is None:
        probe = max(1, int(math.ceil(math.sqrt(nlist))))
    probe = min(nlist, max(1, int(probe)))
    cluster_scores = ann["centroids"] @ query_vector
    clusters = np.argsort(-cluster_scores, kind="stable")[:probe]
    parts = [ann["buckets"][int(cluster)] for cluster in clusters
             if len(ann["buckets"][int(cluster)])]
    if not parts:
        raise _AnnFallback("empty ANN buckets")
    positions = np.concatenate(parts)
    if len(positions) < min(limit, len(matrix)):
        raise _AnnFallback("too few ANN candidates")
    scores = matrix[positions] @ query_vector
    order = np.argsort(-scores, kind="stable")[:limit]
    return positions[order], scores[order]


def rerank_hits(query: str, hits: list[dict], k: int = 5) -> list[dict]:
    """Lexically rerank FTS candidates by coverage, title, phrase and bm25.

    This is a deterministic reranker, not an embedding model. It is intentionally
    cheap: local search should stay fast and work without a second model.
    """
    terms = _query_terms(query)
    if not terms or not hits:
        return hits[:k]
    codes = _identifier_terms(query)
    phrase = " ".join(str(query or "").lower().split())
    query_vector = embed_text(query)
    raw = []
    for hit in hits:
        value = hit.get("score")
        try:
            number = float(value) if value is not None else None
        except (TypeError, ValueError):
            number = None
        raw.append(number if number is not None and math.isfinite(number)
                   else None)
    finite = [value for value in raw if value is not None]
    # bm25 returns lower-is-better; normalize it into 0..1. Embedding-only
    # candidates have no bm25 and get 0 for this component.
    high, low = (max(finite), min(finite)) if finite else (0.0, 0.0)
    span = high - low
    scored = []
    for index, hit in enumerate(hits):
        chunk = str(hit.get("chunk", "")).lower()
        title = str(hit.get("title", "")).lower()
        section = str(hit.get("section", "")).strip().lower()
        searchable = f"{title} {section} {chunk}"
        coverage = _term_coverage(terms, searchable)
        title_hits = _term_coverage(terms, f"{title} {section}")
        section_hits = (_term_coverage(terms, section)
                        if section else 0.0)
        exact = 1.0 if len(terms) >= 2 and phrase in searchable else 0.0
        code_match = (sum(1 for code in codes if code in searchable)
                      / len(codes)) if codes else 0.0
        if raw[index] is not None and span:
            bm25 = 1.0 - ((raw[index] - low) / span)
        elif raw[index] is not None:
            bm25 = 0.5
        else:
            bm25 = 0.0
        embedding = hit.get("embedding_score")
        if embedding is None:
            body_vector = embed_text(searchable)
            embedding = sum(x * y for x, y in zip(query_vector, body_vector))
        score = (0.40 * coverage + 0.10 * title_hits
                 + 0.05 * section_hits + 0.10 * exact
                 + 0.15 * bm25 + 0.25 * embedding)
        if code_match:
            score = min(1.0, score + 0.18 * code_match)
        item = dict(hit)
        item["rerank_score"] = score
        item["embedding_score"] = embedding
        scored.append((score, index, item))
    scored.sort(key=lambda row: (-row[0], row[1]))
    return [item for _, _, item in scored[:k]]


def best_snippet(query: str, chunk: str, max_chars: int = 500) -> str:
    """Select the sentence that best answers the query.

    Number/date questions get a small bonus for sentences containing digits,
    and a sentence that opens with a pronoun pulls in the previous sentence for
    context. Otherwise the result stays short and quote-like instead of
    padding the evidence with unrelated sentences.
    """
    text = str(chunk or "").strip()
    if not text:
        return ""
    parts = [part.strip() for part in
             re.split(r"(?<=[.!?])\s+|\n+", text) if part.strip()]
    if len(parts) <= 1:
        return text[:max_chars]
    terms = _query_terms(query)
    codes = _identifier_terms(query)
    low_query = str(query or "").lower()
    wants_number = bool(re.search(
        r"\b(how many|how much|how long|how old|how far|when|what year|"
        r"what date|what time|price|cost|population|number|percentage)\b",
        low_query))
    scores = []
    for part in parts:
        score = _term_coverage(terms, part)
        if re.search(r"\d", part):
            score += 0.30 if wants_number else 0.10
            if re.search(
                    r"\d+\.\d+|\d+\s*(?:days?|months?|years?|hours?|"
                    r"minutes?|seconds?|million|billion|kg|km|percent|%)",
                    part, re.IGNORECASE):
                score += 0.08
        if codes:
            matches = sum(1 for code in codes if code in part.lower())
            if matches:
                score += 0.25 * (matches / len(codes))
        scores.append(score)
    best = max(range(len(parts)), key=lambda i: (scores[i], -i))
    start = best
    if best > 0 and re.match(
            r"^(it|this|these|they|he|she|its|their|such|that|those)\b",
            parts[best].lower()):
        start = best - 1
    snippet = " ".join(parts[start:best + 1])
    if best + 1 < len(parts) and len(snippet) < max_chars:
        following = parts[best + 1]
        follow_coverage = _term_coverage(terms, following)
        short_context = len(snippet) < 60 and not re.search(r"\d", snippet)
        if follow_coverage > 0.0 or short_context:
            snippet += " " + following
    if len(snippet) > max_chars:
        snippet = parts[best]
    return snippet[:max_chars]


def diverse_hits(hits: list[dict], limit: int = 3,
                 per_source: int = 2) -> list[dict]:
    """Keep the best hits while avoiding one source or duplicate chunks.

    A repeated source is a poor citation list: it hides other evidence and
    makes the answer look more supported than it is. This is deliberately
    simple and deterministic.
    """
    selected = []
    counts: dict[str, int] = {}
    seen: set[str] = set()
    for hit in hits:
        source = str(hit.get("source", ""))
        if counts.get(source, 0) >= per_source:
            continue
        key = re.sub(r"\s+", " ", str(hit.get("chunk", ""))).strip().lower()
        if key and key in seen:
            continue
        selected.append(hit)
        counts[source] = counts.get(source, 0) + 1
        if key:
            seen.add(key)
        if len(selected) >= limit:
            break
    return selected


def dedupe_hits(hits: list[dict], limit: int | None = None,
                threshold: float = 0.82) -> list[dict]:
    """Suppress near-identical chunks while keeping the best-ranked one.

    Long PDFs often repeat the same sentence in a header/footer or on adjacent
    pages. Showing three copies is noise, not stronger evidence.
    """
    selected = []
    seen_sets: list[set[str]] = []
    for hit in hits:
        words = set(_query_terms(hit.get("chunk", "")))
        duplicate = False
        if words:
            for previous in seen_sets:
                union = words | previous
                if union and len(words & previous) / len(union) >= threshold:
                    duplicate = True
                    break
        if duplicate:
            continue
        selected.append(hit)
        seen_sets.append(words)
        if limit is not None and len(selected) >= limit:
            break
    return selected


def retrieval_confidence(query: str, hits: list[dict]) -> dict:
    """Return a transparent 0..1 retrieval match score and a label.

    This is a heuristic, not a calibrated probability. It combines the top
    rerank score, query-term coverage in the winning chunk and the margin over
    the runner-up, so a weak or ambiguous match is visible in the product.
    """
    if not hits:
        return {"score": 0.0, "label": "none", "coverage": 0.0,
                "margin": 0.0}
    top = hits[0]
    try:
        top_score = float(top.get("rerank_score") or 0.0)
    except (TypeError, ValueError):
        top_score = 0.0
    terms = _query_terms(query)
    coverage = _term_coverage(
        terms, f"{top.get('title', '')} {top.get('section', '')} "
                f"{top.get('chunk', '')}")
    second = 0.0
    if len(hits) > 1:
        try:
            second = float(hits[1].get("rerank_score") or 0.0)
        except (TypeError, ValueError):
            second = 0.0
    margin = max(0.0, min(1.0, top_score - second))
    score = max(0.0, min(
        1.0, 0.65 * top_score + 0.25 * coverage + 0.10 * margin))
    if score >= 0.55:
        label = "high"
    elif score >= 0.32:
        label = "medium"
    else:
        label = "low"
    return {"score": round(score, 3), "label": label,
            "coverage": round(coverage, 3), "margin": round(margin, 3)}


def extractive_summary(chunks: list[str], max_sentences: int = 5,
                       max_chars: int = 1200) -> dict:
    """Return an extractive summary without generating new facts.

    Sentences are scored by content-word frequency, with a mild early-position
    bonus and a length penalty. The selected sentences are returned in their
    original order, so the summary reads like the source rather than a bag of
    quotes.
    """
    try:
        max_sentences = int(max_sentences)
    except (TypeError, ValueError):
        max_sentences = 5
    max_sentences = max(1, min(max_sentences, 20))
    text = "\n".join(str(chunk) for chunk in chunks if str(chunk).strip())
    if not text:
        return {"text": "", "sentences": [], "method": "extractive",
                "candidates": 0}
    sentences = [part.strip() for part in
                 re.split(r"(?<=[.!?])\s+|\n+", text) if part.strip()]
    seen = set()
    unique = []
    for sentence in sentences:
        key = re.sub(r"\s+", " ", sentence).strip().lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(sentence)
    sentences = unique
    if len(sentences) <= max_sentences:
        summary = " ".join(sentences)
        return {"text": summary[:max_chars], "sentences": sentences,
                "method": "extractive", "candidates": len(sentences)}
    frequency = Counter(
        word for word in re.findall(r"[^\W_]+", text.lower())
        if len(word) > 2 and word not in STOPWORDS)
    scores = []
    for index, sentence in enumerate(sentences):
        terms = _query_terms(sentence)
        if not terms:
            score = 0.0
        else:
            score = sum(frequency[term] for term in terms) / math.sqrt(
                len(terms))
        score += 0.5 / (index + 1)
        score /= 1.0 + len(sentence) / 400.0
        scores.append(score)
    chosen = sorted(
        sorted(range(len(sentences)),
               key=lambda i: (-scores[i], i))[:max_sentences])
    selected = [sentences[index] for index in chosen]
    summary = " ".join(selected)
    return {"text": summary[:max_chars], "sentences": selected,
            "method": "extractive", "candidates": len(sentences)}


def extractive_summary_units(units: list[dict], max_sentences: int = 8,
                             max_chars: int = 2000) -> dict:
    """Summarize multiple sources while keeping each sentence's locator."""
    try:
        max_sentences = int(max_sentences)
    except (TypeError, ValueError):
        max_sentences = 8
    max_sentences = max(1, min(max_sentences, 30))
    expanded = []
    seen = set()
    for unit in units:
        source = str(unit.get("source", ""))
        page = unit.get("page")
        section = str(unit.get("section", "") or "")
        text = str(unit.get("text", "")).strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
            sentence = sentence.strip()
            if not sentence:
                continue
            key = re.sub(r"\s+", " ", sentence).strip().lower()
            if key in seen:
                continue
            seen.add(key)
            expanded.append({"text": sentence, "source": source,
                             "page": page, "section": section})
    if not expanded:
        return {"text": "", "sentences": [], "citations": [],
                "method": "extractive", "candidates": 0}
    if len(expanded) <= max_sentences:
        selected = expanded
    else:
        frequency = Counter(
            word for unit in expanded
            for word in re.findall(r"[^\W_]+", unit["text"].lower())
            if len(word) > 2 and word not in STOPWORDS)
        scores = []
        for index, unit in enumerate(expanded):
            terms = _query_terms(unit["text"])
            score = (sum(frequency[term] for term in terms)
                     / math.sqrt(len(terms))) if terms else 0.0
            score += 0.5 / (index + 1)
            score /= 1.0 + len(unit["text"]) / 400.0
            scores.append(score)
        chosen = sorted(
            sorted(range(len(expanded)),
                   key=lambda i: (-scores[i], i))[:max_sentences])
        selected = [expanded[index] for index in chosen]
    citations = []
    for unit in selected:
        citation = {"source": unit["source"], "page": unit["page"],
                    "section": unit["section"]}
        if citation not in citations:
            citations.append(citation)
    text = " ".join(unit["text"] for unit in selected)
    return {"text": text[:max_chars], "sentences": selected,
            "citations": citations, "method": "extractive",
            "candidates": len(expanded)}


def format_citation(hit: dict) -> str:
    """Format a hit as ``source (p. N) [section]`` for chat/UI display."""
    location = str(hit.get("source_url") or hit.get("source") or "")
    page = hit.get("page")
    if page not in (None, ""):
        location += f" (p. {page})"
    section = str(hit.get("section") or "").strip()
    if section:
        location += f" [{section}]"
    return location


def add_citation_source(sources: dict, hit: dict) -> dict:
    """Merge a page/section hit into a source-keyed citation map."""
    key = str(hit.get("source", ""))
    source = sources.get(key)
    if source is None:
        source = {"source": hit.get("source", ""),
                  "title": hit.get("title", "")}
        for field in ("kind", "source_url", "teacher", "model", "verified",
                      "reviewer", "verified_at"):
            if field in hit:
                source[field] = hit[field]
        sources[key] = source
    page = hit.get("page")
    if page not in (None, ""):
        pages = source.setdefault("pages", [])
        if page not in pages:
            pages.append(page)
    section = str(hit.get("section") or "").strip()
    if section:
        sections = source.setdefault("sections", [])
        if section not in sections:
            sections.append(section)
    return source


def grounded_answer(query: str, hits: list[dict],
                    limit: int = 3) -> dict:
    """Build a cited answer, evidence list and confidence from search hits.

    This is the shared core used by both the HTTP endpoint and the
    ``atomzero-docs answer`` CLI. It does not generate new facts; it quotes the
    retrieved local sentences and labels teacher notes.
    """
    scored = []
    for hit in hits:
        try:
            value = float(hit.get("rerank_score") or 0.0)
        except (TypeError, ValueError):
            value = 0.0
        scored.append((value, hit))
    top_score = max((value for value, _ in scored), default=0.0)
    # Keep only notes close to the winning match. Without this, a strong hit
    # can drag unrelated documents into the answer just to fill three slots.
    cutoff = max(0.18, top_score * 0.5)
    relevant = [hit for value, hit in scored if value >= cutoff]
    if not relevant and scored:
        relevant = [scored[0][1]]
    selected = diverse_hits(relevant, limit=limit)
    confidence = retrieval_confidence(query, selected)
    notes = []
    evidence = []
    source_map: dict = {}
    unverified = False
    for hit in selected:
        teacher = hit.get("kind") == "teacher"
        verified = bool(hit.get("verified", True))
        if teacher and not verified:
            label = "teacher note, unverified"
        elif teacher:
            reviewer = str(hit.get("reviewer", "")).strip()
            label = (f"teacher note, verified by {reviewer}"
                     if reviewer else "teacher note, verified")
        else:
            label = "source"
        snippet = best_snippet(query, hit.get("chunk", ""))
        location = format_citation(hit)
        notes.append(f"- {snippet} ({label}: {location})")
        try:
            score = round(float(hit.get("rerank_score") or 0.0), 3)
        except (TypeError, ValueError):
            score = 0.0
        evidence.append({
            "source": hit.get("source", ""),
            "title": hit.get("title", ""),
            "page": hit.get("page"),
            "section": hit.get("section", ""),
            "snippet": snippet,
            "chunk": hit.get("chunk", ""),
            "score": score,
            "kind": hit.get("kind", "document"),
            "verified": verified,
            "source_url": hit.get("source_url", ""),
        })
        add_citation_source(source_map, hit)
        if not verified:
            unverified = True
    text = (f"I found these notes:\n" + "\n".join(notes) + "\n\n"
            "This is a local note, not professional advice.")
    if confidence["label"] == "low":
        text += ("\n\nMatch confidence is low; check the source before "
                 "relying on this answer.")
    if unverified:
        text += "\n\nThis includes an unverified teacher-derived note."
    return {"text": text, "sources": list(source_map.values()),
            "evidence": evidence, "confidence": confidence}


def _chunks(text: str) -> list[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    out = []
    current = ""
    for para in paragraphs:
        if len(para) > MAX_CHARS:
            if current:
                out.append(current)
                current = ""
            out.extend(_split_long(para))
            continue
        if len(current) + len(para) + 2 <= MAX_CHARS:
            current = (current + "\n\n" + para).strip()
        else:
            if current:
                out.append(current)
            current = para
    if current:
        out.append(current)
    return out


def _clean_collection_prefix(prefix: str) -> str:
    value = str(prefix or "").strip().strip("/\\")
    if not value:
        raise ValueError("prefix cannot be empty")
    if len(value) > 200:
        raise ValueError("prefix is too long (max 200 chars)")
    if any(ord(ch) < 32 for ch in value):
        raise ValueError("prefix contains control characters")
    return value


class DocMemory:
    def __init__(self, path: str | Path = "data/docs.db"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.reranker = None
        self._vector_index = None
        self.vector_index_mode = "auto"  # auto | ann | exact
        self.ann_min_chunks = ANN_MIN_CHUNKS
        self.ann_probe = None
        self._vector_fallback_count = 0
        self.con = sqlite3.connect(str(self.path), check_same_thread=False)
        self.con.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS docs USING fts5("
            "chunk, source UNINDEXED, title UNINDEXED, sha UNINDEXED, "
            "ts UNINDEXED)")
        self.con.execute(
            "CREATE TABLE IF NOT EXISTS doc_meta("
            "source TEXT PRIMARY KEY, kind TEXT NOT NULL, "
            "source_url TEXT NOT NULL, teacher TEXT NOT NULL, "
            "model TEXT NOT NULL, verified INTEGER NOT NULL, "
            "created TEXT NOT NULL, reviewer TEXT NOT NULL DEFAULT '', "
            "verified_at TEXT NOT NULL DEFAULT '')")
        self.con.execute(
            "CREATE TABLE IF NOT EXISTS doc_vectors("
            "doc_rowid INTEGER NOT NULL, model TEXT NOT NULL, "
            "dim INTEGER NOT NULL, vec BLOB NOT NULL, "
            "PRIMARY KEY(doc_rowid, model))")
        self.con.execute(
            "CREATE TABLE IF NOT EXISTS doc_chunk_meta("
            "doc_rowid INTEGER PRIMARY KEY, page INTEGER, "
            "section TEXT NOT NULL DEFAULT '', "
            "char_start INTEGER, char_end INTEGER)")
        columns = {row[1] for row in self.con.execute(
            "PRAGMA table_info(doc_meta)")}
        for name, ddl in (
                ("reviewer", "TEXT NOT NULL DEFAULT ''"),
                ("verified_at", "TEXT NOT NULL DEFAULT ''")):
            if name not in columns:
                self.con.execute(
                    f"ALTER TABLE doc_meta ADD COLUMN {name} {ddl}")
        self.con.commit()

    def add_text(self, source: str, text: str, title: str = "",
                 replace: bool = True, meta: dict | None = None) -> int:
        return self.add_blocks(
            source, [{"text": text, "page": None, "section": ""}],
            title=title, replace=replace, meta=meta)

    def add_blocks(self, source: str, blocks: list[dict], title: str = "",
                   replace: bool = True, meta: dict | None = None) -> int:
        """Index text blocks while keeping page/section provenance.

        A block is a piece of text with an optional 1-based ``page`` and a
        ``section`` label. Each block is split with the normal chunker; the
        resulting chunks inherit the block's provenance and a character range
        inside that block.
        """
        return self.add_blocks_result(
            source, blocks, title=title, replace=replace, meta=meta)["chunks"]

    def add_blocks_result(self, source: str, blocks: list[dict],
                          title: str = "", replace: bool = True,
                          meta: dict | None = None,
                          skip_if_unchanged: bool = False) -> dict:
        """Index blocks and return a small ingestion report.

        ``skip_if_unchanged`` makes repeated directory syncs cheap: when the
        extracted content hash matches the stored hash for the same source,
        nothing is deleted or re-embedded.
        """
        source = str(source)
        title = str(title or Path(source).stem)
        meta = meta or {}
        kind = str(meta.get("kind", "document"))
        source_url = str(meta.get("source_url", ""))
        teacher = str(meta.get("teacher", ""))
        model = str(meta.get("model", ""))
        verified = 1 if meta.get("verified", True) else 0
        created = str(meta.get("created", int(time.time())))
        reviewer = str(meta.get("reviewer", ""))
        verified_at = str(meta.get("verified_at", ""))
        block_texts = [str(block.get("text", "")) for block in blocks]
        sha = hashlib.sha256(
            "\n\n".join(block_texts).encode("utf-8")).hexdigest()
        with self._lock:
            if skip_if_unchanged:
                existing = self.con.execute(
                    "SELECT MAX(sha) FROM docs WHERE source = ?",
                    (source,)).fetchone()
                if existing and existing[0] == sha:
                    return {"source": source, "title": title, "chunks": 0,
                            "skipped": True, "content_sha256": sha}
            if replace:
                self.con.execute("DELETE FROM docs WHERE source = ?",
                                 (source,))
            added = 0
            for block in blocks:
                text = str(block.get("text", "")).replace(
                    "\r\n", "\n").replace("\r", "\n").strip()
                if not text:
                    continue
                page = block.get("page")
                try:
                    page = int(page) if page not in (None, "") else None
                except (TypeError, ValueError):
                    page = None
                section = str(block.get("section", "") or "").strip()
                cursor = 0
                for chunk in _chunks(text):
                    position = text.find(chunk, cursor)
                    if position < 0:
                        position = cursor
                    char_start = position
                    char_end = position + len(chunk)
                    inserted = self.con.execute(
                        "INSERT INTO docs(chunk, source, title, sha, ts) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (chunk, source, title, sha, int(time.time())))
                    rowid = inserted.lastrowid
                    if not rowid:
                        rowid = self.con.execute(
                            "SELECT last_insert_rowid()").fetchone()[0]
                    self.con.execute(
                        "INSERT OR REPLACE INTO doc_chunk_meta("
                        "doc_rowid, page, section, char_start, char_end) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (rowid, page, section, char_start, char_end))
                    cursor = char_end
                    added += 1
            self.con.execute(
                "INSERT OR REPLACE INTO doc_meta("
                "source, kind, source_url, teacher, model, verified, created, "
                "reviewer, verified_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (source, kind, source_url, teacher, model, verified, created,
                 reviewer, verified_at))
            if replace:
                self._drop_stale_vectors()
                self._drop_stale_chunk_meta()
            self._vector_index = None
            self.con.commit()
        return {"source": source, "title": title, "chunks": added,
                "skipped": False, "content_sha256": sha}

    def add_file(self, path: str | Path, title: str = "",
                 meta: dict | None = None) -> int:
        return self.add_file_result(path, title=title, meta=meta)["chunks"]

    def add_file_result(self, path: str | Path, title: str = "",
                        meta: dict | None = None,
                        skip_if_unchanged: bool = False,
                        dedupe_across_sources: bool = False,
                        source: str = "") -> dict:
        """Extract and index one file, returning hashes and a skip flag.

        ``source`` overrides the stored locator; this keeps a synced directory
        portable (``notes/policy.md`` instead of an absolute machine path).
        """
        from atomzero.doc_ingest import extract_file

        p = Path(path)
        indexed_source = str(source or p)
        extracted = extract_file(p, title=title)
        content_sha = hashlib.sha256(
            "\n\n".join(str(block.get("text", ""))
                        for block in extracted["blocks"]).encode("utf-8")
        ).hexdigest()
        file_sha = hashlib.sha256(p.read_bytes()).hexdigest()
        if dedupe_across_sources:
            duplicates = self.find_by_content_sha(
                content_sha, exclude_source=indexed_source)
            if duplicates:
                return {
                    "source": indexed_source,
                    "path": str(p),
                    "title": extracted["title"],
                    "chunks": 0,
                    "skipped": True,
                    "duplicate_of": duplicates,
                    "content_sha256": content_sha,
                    "file_sha256": file_sha,
                    "format": extracted.get("format", ""),
                }
        result = self.add_blocks_result(
            indexed_source, extracted["blocks"], title=extracted["title"],
            meta=meta, skip_if_unchanged=skip_if_unchanged)
        result["path"] = str(p)
        result["format"] = extracted.get("format", "")
        result["file_sha256"] = file_sha
        return result

    def add_directory(self, path: str | Path, recursive: bool = True,
                      skip_if_unchanged: bool = False,
                      dedupe_across_sources: bool = False) -> dict:
        """Index every supported file in a directory in deterministic order."""
        from atomzero.doc_ingest import SUPPORTED_SUFFIXES

        root = Path(path)
        if not root.is_dir():
            raise ValueError(f"not a directory: {root}")
        pattern = "**/*" if recursive else "*"
        files = sorted(
            item for item in root.glob(pattern)
            if item.is_file() and item.suffix.lower() in SUPPORTED_SUFFIXES)
        results = []
        for item in files:
            try:
                results.append(self.add_file_result(
                    item, skip_if_unchanged=skip_if_unchanged,
                    dedupe_across_sources=dedupe_across_sources))
            except Exception as exc:  # keep one bad file from aborting a sync
                results.append({"source": str(item), "chunks": 0,
                                "skipped": False, "error": str(exc)})
        return {
            "root": str(root),
            "files": len(files),
            "chunks": sum(int(item.get("chunks", 0)) for item in results),
            "skipped": sum(1 for item in results if item.get("skipped")),
            "duplicates": sum(1 for item in results
                              if item.get("duplicate_of")),
            "failed": sum(1 for item in results if "error" in item),
            "results": results,
        }

    def sync_directory(self, path: str | Path, prefix: str = "",
                       recursive: bool = True, prune: bool = True,
                       dedupe_across_sources: bool = True) -> dict:
        """Keep a directory collection in sync with the local index.

        Sources are stored as ``<prefix>/<relative-path>`` so the index is
        portable across machines. Unchanged files are skipped, duplicate
        content can be skipped, and indexed sources whose files disappeared
        are removed when ``prune`` is true.
        """
        from atomzero.doc_ingest import SUPPORTED_SUFFIXES

        root = Path(path)
        if not root.is_dir():
            raise ValueError(f"not a directory: {root}")
        prefix = _clean_collection_prefix(prefix or root.name)
        pattern = "**/*" if recursive else "*"
        files = sorted(
            item for item in root.glob(pattern)
            if item.is_file() and item.suffix.lower() in SUPPORTED_SUFFIXES)
        current = set()
        results = []
        for item in files:
            relative = item.relative_to(root).as_posix()
            source = f"{prefix}/{relative}"
            current.add(source)
            try:
                results.append(self.add_file_result(
                    item, skip_if_unchanged=True,
                    dedupe_across_sources=dedupe_across_sources,
                    source=source))
            except Exception as exc:
                results.append({"source": source, "path": str(item),
                                "chunks": 0, "skipped": False,
                                "error": str(exc)})
        pruned = []
        if prune:
            existing = [item["source"] for item in self.sources()
                        if item["source"].startswith(f"{prefix}/")]
            for source in sorted(set(existing) - current):
                try:
                    pruned.append(self.forget(source))
                except SourceNotFound:
                    continue
        return {
            "root": str(root),
            "prefix": prefix,
            "files": len(files),
            "chunks": sum(int(item.get("chunks", 0)) for item in results),
            "skipped": sum(1 for item in results if item.get("skipped")),
            "duplicates": sum(1 for item in results
                              if item.get("duplicate_of")),
            "pruned": pruned,
            "failed": sum(1 for item in results if "error" in item),
            "results": results,
        }

    def directory_status(self, path: str | Path, prefix: str = "",
                         recursive: bool = True) -> dict:
        """Compare a directory with the index without changing anything."""
        from atomzero.doc_ingest import SUPPORTED_SUFFIXES, extract_file

        root = Path(path)
        if not root.is_dir():
            raise ValueError(f"not a directory: {root}")
        prefix = _clean_collection_prefix(prefix or root.name)
        pattern = "**/*" if recursive else "*"
        files = sorted(
            item for item in root.glob(pattern)
            if item.is_file() and item.suffix.lower() in SUPPORTED_SUFFIXES)
        existing = {
            item["source"]: item.get("content_sha256", "")
            for item in self.sources()
            if item["source"].startswith(f"{prefix}/")}
        current = set()
        details = []
        added = changed = unchanged = failed = 0
        for item in files:
            relative = item.relative_to(root).as_posix()
            source = f"{prefix}/{relative}"
            current.add(source)
            if source not in existing:
                added += 1
                details.append({"source": source, "status": "added"})
                continue
            try:
                extracted = extract_file(item)
                content_sha = hashlib.sha256(
                    "\n\n".join(str(block.get("text", ""))
                                for block in extracted["blocks"])
                    .encode("utf-8")).hexdigest()
            except Exception as exc:
                failed += 1
                details.append({"source": source, "status": "error",
                                "error": str(exc)})
                continue
            if content_sha != existing[source]:
                changed += 1
                details.append({"source": source, "status": "changed"})
            else:
                unchanged += 1
        missing = sorted(set(existing) - current)
        return {
            "root": str(root),
            "prefix": prefix,
            "files": len(files),
            "added": added,
            "changed": changed,
            "unchanged": unchanged,
            "failed": failed,
            "missing": missing,
            "details": details,
        }

    def _source_chunk_rows(self, source: str) -> list[tuple]:
        with self._lock:
            return self.con.execute(
                "SELECT d.chunk, d.title, m.page, m.section "
                "FROM docs d LEFT JOIN doc_chunk_meta m "
                "ON m.doc_rowid = d.rowid "
                "WHERE d.source = ? ORDER BY d.rowid",
                (str(source),)).fetchall()

    def export_jsonl(self, path: str | Path, source_prefix: str = "",
                     kind: str = "") -> dict:
        """Export sources as one JSONL record each, preserving provenance."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        items = [
            item for item in self.sources()
            if (not source_prefix
                or item["source"].startswith(str(source_prefix)))
            and (not kind or item.get("kind") == kind)]
        chunks = 0
        with p.open("w", encoding="utf-8", newline="\n") as handle:
            for item in items:
                rows = self._source_chunk_rows(item["source"])
                if not rows:
                    continue
                record = {
                    "source": item["source"],
                    "title": item.get("title", ""),
                    "meta": {
                        key: item.get(key)
                        for key in ("kind", "source_url", "teacher", "model",
                                    "verified", "created", "reviewer",
                                    "verified_at")
                    },
                    "blocks": [
                        {"text": row[0], "page": row[2],
                         "section": row[3] or ""}
                        for row in rows
                    ],
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                chunks += len(rows)
        return {"out": str(p), "sources": len(items), "chunks": chunks,
                "bytes": p.stat().st_size}

    def import_jsonl(self, path: str | Path,
                     replace: bool = True) -> dict:
        """Import JSONL records produced by :meth:`export_jsonl`."""
        p = Path(path)
        sources = chunks = failed = 0
        errors = []
        with p.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        raise ValueError("record must be an object")
                    source = str(record.get("source", "")).strip()
                    if not source:
                        raise ValueError("source is required")
                    blocks = record.get("blocks")
                    if not isinstance(blocks, list) or not blocks:
                        raise ValueError("blocks must be a non-empty list")
                    result = self.add_blocks_result(
                        source, blocks,
                        title=str(record.get("title", "")),
                        replace=replace,
                        meta=record.get("meta") or {})
                    sources += 1
                    chunks += int(result["chunks"])
                except Exception as exc:
                    failed += 1
                    errors.append({"line": line_number, "error": str(exc)})
        return {"in": str(p), "sources": sources, "chunks": chunks,
                "failed": failed, "errors": errors}

    def find_by_content_sha(self, sha: str, exclude_source: str = "") -> list[dict]:
        """Find indexed sources with the same extracted-content hash."""
        sha = str(sha or "").strip()
        if len(sha) != 64:
            raise ValueError("sha must be a 64-character hex digest")
        with self._lock:
            rows = self.con.execute(
                "SELECT source, title, COUNT(*) FROM docs "
                "WHERE sha = ? AND source != ? "
                "GROUP BY source, title ORDER BY source",
                (sha, str(exclude_source))).fetchall()
        return [{"source": row[0], "title": row[1], "chunks": row[2]}
                for row in rows]

    def _drop_stale_vectors(self) -> None:
        self.con.execute(
            "DELETE FROM doc_vectors WHERE doc_rowid NOT IN "
            "(SELECT rowid FROM docs)")

    def _drop_stale_chunk_meta(self) -> None:
        self.con.execute(
            "DELETE FROM doc_chunk_meta WHERE doc_rowid NOT IN "
            "(SELECT rowid FROM docs)")

    @staticmethod
    def _vector_model_key(active) -> str:
        if active is None or not hasattr(active, "embed"):
            return HASHED_CACHE_KEY
        key = getattr(active, "cache_key", "")
        if key:
            return str(key)
        path = getattr(active, "model_path", "")
        if path:
            return f"neural:{Path(path).resolve()}"
        return "neural:custom"

    @staticmethod
    def _embed_texts(active, texts: list[str]) -> list[list[float]]:
        if active is not None and hasattr(active, "embed"):
            vectors = active.embed(texts)
            return [[float(value) for value in row] for row in vectors.tolist()]
        return [embed_text(text) for text in texts]

    def _vectors_for_rows(self, active, rows) -> dict[int, list[float]]:
        """Load cached chunk vectors, computing and storing missing ones."""
        if not rows:
            return {}
        key = self._vector_model_key(active)
        wanted = {rowid: text for rowid, text in rows}
        vectors: dict[int, list[float]] = {}
        ids = list(wanted)
        for start in range(0, len(ids), 400):
            batch = ids[start:start + 400]
            placeholders = ",".join("?" for _ in batch)
            found = self.con.execute(
                "SELECT doc_rowid, vec FROM doc_vectors "
                f"WHERE model = ? AND doc_rowid IN ({placeholders})",
                (key, *batch)).fetchall()
            for rowid, blob in found:
                vectors[rowid] = _unpack_vector(blob)
        missing = [rowid for rowid in ids if rowid not in vectors]
        if missing:
            computed = self._embed_texts(
                active, [wanted[rowid] for rowid in missing])
            self.con.executemany(
                "INSERT OR REPLACE INTO doc_vectors"
                "(doc_rowid, model, dim, vec) VALUES (?, ?, ?, ?)",
                [(rowid, key, len(vector), _pack_vector(vector))
                 for rowid, vector in zip(missing, computed)])
            self.con.commit()
            for rowid, vector in zip(missing, computed):
                vectors[rowid] = vector
        return vectors

    def prewarm_vectors(self, active=None) -> dict:
        """Build the in-memory matrix over all current chunk vectors.

        With no explicit reranker, use the attached one (neural when
        configured, hashed otherwise).
        """
        if active is None:
            active = self.reranker
        started = time.perf_counter()
        with self._lock:
            rows = self.con.execute(
                "SELECT rowid, chunk, source, title FROM docs "
                "ORDER BY rowid").fetchall()
            key = self._vector_model_key(active)
            if not rows:
                self._vector_index = {
                    "key": key, "rows": [], "rowids": [], "matrix": None,
                    "positions": {}, "build_s": 0.0}
                return self.vector_index_stats()
            texts = [(row[0], f"{row[3]} {row[1]}".strip()) for row in rows]
            vectors = self._vectors_for_rows(active, texts)
            kept_rows = []
            kept_vectors = []
            for row in rows:
                vector = vectors.get(row[0])
                if vector is None:
                    continue
                if kept_vectors and len(vector) != len(kept_vectors[0]):
                    continue
                kept_rows.append(row)
                kept_vectors.append(vector)
            matrix = None
            if np is not None and kept_vectors:
                matrix = np.asarray(kept_vectors, dtype=np.float32)
                matrix = _normalized_rows(matrix)
            mode = self._resolved_vector_mode(len(kept_rows))
            ann = None
            ann_build_s = 0.0
            if mode == "ann" and matrix is not None:
                ann_started = time.perf_counter()
                try:
                    ann = build_ivf(matrix)
                except (MemoryError, RuntimeError, ValueError):
                    ann = None
                ann_build_s = time.perf_counter() - ann_started
                if ann is None:
                    mode = "exact"
            rowids = [row[0] for row in kept_rows]
            self._vector_index = {
                "key": key,
                "rows": kept_rows,
                "rowids": rowids,
                "matrix": matrix,
                "positions": {rowid: index
                              for index, rowid in enumerate(rowids)},
                "mode": mode,
                "ann": ann,
                "ann_build_s": ann_build_s,
                "build_s": time.perf_counter() - started,
            }
            self._vector_fallback_count = 0
        return self.vector_index_stats()

    def _resolved_vector_mode(self, chunks: int) -> str:
        if self.vector_index_mode == "exact" or np is None:
            return "exact"
        if self.vector_index_mode == "ann":
            return "ann" if chunks >= 2 else "exact"
        return ("ann" if chunks >= max(2, int(self.ann_min_chunks))
                else "exact")

    def vector_index_stats(self) -> dict:
        index = self._vector_index
        backend = "numpy" if np is not None else "scan"
        if not index:
            return {"built": False, "chunks": 0, "dim": 0,
                    "build_s": 0.0, "backend": backend}
        matrix = index.get("matrix")
        ann = index.get("ann")
        nlist = int(ann["nlist"]) if ann else 0
        return {
            "built": matrix is not None,
            "chunks": len(index.get("rows", [])),
            "dim": int(matrix.shape[1]) if matrix is not None else 0,
            "mode": index.get("mode", "exact"),
            "ann": bool(ann),
            "nlist": nlist,
            "nprobe": (int(self.ann_probe) if self.ann_probe
                       else (max(1, int(math.ceil(math.sqrt(nlist))))
                             if nlist else 0)),
            "exact_fallback": True,
            "fallbacks": int(self._vector_fallback_count),
            "build_s": round(float(index.get("build_s", 0.0)), 3),
            "ann_build_s": round(float(index.get("ann_build_s", 0.0)), 3),
            "backend": backend,
        }

    @staticmethod
    def _candidate_filters(query: str, neural: bool):
        threshold = HYBRID_MIN_NEURAL if neural else HYBRID_MIN_HASHED
        grams = set(_char_ngrams(query)) if not neural else None
        return threshold, grams

    @staticmethod
    def _has_hashed_overlap(query_grams, row) -> bool:
        if query_grams is None:
            return True
        # Hashed vectors are sparse and can collide on short queries. Require
        # real character overlap before a hash-only candidate may bypass FTS.
        text = f"{row[3]} {row[1]}".lower()
        return len(query_grams & set(_char_ngrams(text))) >= 2

    @staticmethod
    def _exact_candidates(index, query_vector, limit: int):
        scores = index["matrix"] @ query_vector
        order = np.argsort(-scores, kind="stable")[:limit]
        return order, scores[order]

    def _search_positions(self, index, query_vector, limit: int,
                          required_rowids, threshold: float):
        matrix = index["matrix"]
        ann = index.get("ann")
        positions = None
        scores = None
        if ann is not None:
            try:
                positions, scores = ann_topk(
                    matrix, ann, query_vector, limit, self.ann_probe)
                if not len(positions) or float(scores[0]) < threshold:
                    raise _AnnFallback("low ANN confidence")
            except Exception:
                positions = None
                scores = None
        if positions is None:
            if ann is not None:
                self._vector_fallback_count += 1
            positions, scores = self._exact_candidates(
                index, query_vector, limit)
        by_rowid = {
            index["rowids"][int(position)]: float(score)
            for position, score in zip(positions, scores)
        }
        required = [index["positions"][rowid]
                    for rowid in required_rowids or ()
                    if rowid in index["positions"]]
        if required:
            extra = matrix[required] @ query_vector
            for position, score in zip(required, extra):
                by_rowid[index["rowids"][position]] = float(score)
        return ([int(position) for position in positions],
                [float(score) for score in scores], by_rowid)

    def _embedding_candidates(self, query: str, active, limit: int,
                              required_rowids=None):
        """Score chunks with the prewarmed matrix; rebuild it when invalid."""
        key = self._vector_model_key(active)
        index = self._vector_index
        if index is None or index.get("key") != key:
            self.prewarm_vectors(active)
            index = self._vector_index
        matrix = index.get("matrix") if index else None
        if matrix is None:
            return self._embedding_candidates_scan(
                query, active, limit, required_rowids)
        if not index["rows"]:
            return [], {}
        query_vector = np.asarray(
            self._embed_texts(active, [query])[0], dtype=np.float32)
        if matrix.shape[1] != query_vector.shape[0]:
            return self._embedding_candidates_scan(
                query, active, limit, required_rowids)
        neural = key.startswith("neural:")
        threshold, query_grams = self._candidate_filters(query, neural)
        try:
            top, top_scores, by_rowid = self._search_positions(
                index, query_vector, limit, required_rowids, threshold)
        except (MemoryError, RuntimeError, ValueError, IndexError):
            return self._embedding_candidates_scan(
                query, active, limit, required_rowids)
        hits = []
        for position, similarity in zip(top, top_scores):
            if similarity < threshold:
                break
            row = index["rows"][position]
            if not self._has_hashed_overlap(query_grams, row):
                continue
            hit = {
                "chunk": row[1], "source": row[2], "title": row[3],
                "score": None, "_rowid": row[0],
                "embedding_score": similarity,
            }
            if neural:
                hit["neural_score"] = similarity
            hits.append(hit)
        return hits, by_rowid

    def _embedding_candidates_scan(self, query: str, active, limit: int,
                                   required_rowids=None):
        """Legacy linear scan; used only when numpy is unavailable."""
        rows = self.con.execute(
            "SELECT rowid, chunk, source, title FROM docs "
            "ORDER BY rowid").fetchall()
        if not rows:
            return [], {}
        texts = [(row[0], f"{row[3]} {row[1]}".strip()) for row in rows]
        vectors = self._vectors_for_rows(active, texts)
        query_vector = self._embed_texts(active, [query])[0]
        neural = self._vector_model_key(active).startswith("neural:")
        scored = []
        for row in rows:
            vector = vectors.get(row[0])
            if vector is None or len(vector) != len(query_vector):
                continue
            similarity = sum(a * b for a, b in
                             zip(query_vector, vector))
            scored.append((similarity, row))
        scored.sort(key=lambda item: (-item[0], item[1][0]))
        by_rowid = {row[0]: similarity for similarity, row in scored}
        threshold, query_grams = self._candidate_filters(query, neural)
        hits = []
        for similarity, row in scored[:limit]:
            if similarity < threshold:
                break
            if not self._has_hashed_overlap(query_grams, row):
                continue
            hit = {
                "chunk": row[1], "source": row[2], "title": row[3],
                "score": None, "_rowid": row[0],
                "embedding_score": similarity,
            }
            if neural:
                hit["neural_score"] = similarity
            hits.append(hit)
        return hits, by_rowid

    def _meta_map(self, sources) -> dict[str, dict]:
        sources = list(sources)
        if not sources:
            return {}
        placeholders = ",".join("?" for _ in sources)
        rows = self.con.execute(
            "SELECT source, kind, source_url, teacher, model, verified, "
            f"created, reviewer, verified_at FROM doc_meta "
            f"WHERE source IN ({placeholders})",
            tuple(sources)).fetchall()
        return {row[0]: {
            "source": row[0],
            "kind": row[1],
            "source_url": row[2],
            "teacher": row[3],
            "model": row[4],
            "verified": bool(row[5]),
            "created": row[6],
            "reviewer": row[7],
            "verified_at": row[8],
        } for row in rows}

    def _chunk_meta_map(self, rowids) -> dict[int, dict]:
        rowids = [int(rowid) for rowid in rowids
                  if rowid is not None]
        if not rowids:
            return {}
        placeholders = ",".join("?" for _ in rowids)
        rows = self.con.execute(
            "SELECT doc_rowid, page, section, char_start, char_end "
            f"FROM doc_chunk_meta WHERE doc_rowid IN ({placeholders})",
            tuple(rowids)).fetchall()
        return {row[0]: {
            "page": row[1],
            "section": row[2] or "",
            "char_start": row[3],
            "char_end": row[4],
        } for row in rows}

    def search(self, query: str, k: int = 5,
               rerank: bool = True,
               reranker=None, source: str = "",
               source_prefix: str = "", kind: str = "",
               diverse: bool = False) -> list[dict]:
        """Search the local index, optionally scoped to a source or kind."""
        source = str(source or "").strip()
        source_prefix = str(source_prefix or "").strip()
        kind = str(kind or "").strip()
        for label, value in (("source", source),
                             ("source_prefix", source_prefix)):
            if len(value) > 512:
                raise ValueError(f"{label} is too long (max 512 chars)")
            if any(ord(ch) < 32 for ch in value):
                raise ValueError(f"{label} contains control characters")
        words = re.findall(r"[\wçğıöşüÇĞİÖŞÜ]+", query)
        words = [w for w in words if len(w) > 1]
        meaningful = [w for w in words if w.lower() not in STOPWORDS]
        if meaningful:
            words = meaningful
        for code in _identifier_terms(query):
            if code not in words:
                words.append(code)
        if not words:
            return []
        match = " OR ".join(f'"{w}"' for w in words[:12])
        candidates = max(k * 4, 12) if rerank else k
        if diverse:
            candidates = max(candidates, k * 6, 24)
        active = reranker if reranker is not None else self.reranker
        neural = self._vector_model_key(active).startswith("neural:")
        clauses = ["docs MATCH ?"]
        params: list = [match]
        if source:
            clauses.append("source = ?")
            params.append(source)
        if source_prefix:
            clauses.append("substr(source, 1, ?) = ?")
            params.extend([len(source_prefix), source_prefix])
        if kind:
            clauses.append(
                "source IN (SELECT source FROM doc_meta WHERE kind = ?)")
            params.append(kind)
        sql = ("SELECT rowid, chunk, source, title, bm25(docs) AS score "
               "FROM docs WHERE " + " AND ".join(clauses) +
               " ORDER BY score LIMIT ?")
        with self._lock:
            rows = self.con.execute(sql, (*params, candidates)).fetchall()
            embedding_hits, embedding_scores = ([], {})
            # Hashed n-grams are a fallback reranker, not a semantic retriever:
            # do not answer from hash-only candidates when FTS found nothing.
            # The neural reranker has a calibrated threshold and may do so.
            if rerank and (rows or neural):
                embedding_hits, embedding_scores = \
                    self._embedding_candidates(
                        query, active, candidates,
                        required_rowids={row[0] for row in rows})
            sources = {row[2] for row in rows} | {
                hit["source"] for hit in embedding_hits}
            metas = self._meta_map(sources)
            if source or source_prefix or kind:
                def allowed(hit: dict) -> bool:
                    value = str(hit.get("source", ""))
                    if source and value != source:
                        return False
                    if source_prefix and not value.startswith(source_prefix):
                        return False
                    if kind and metas.get(value, {}).get(
                            "kind", "document") != kind:
                        return False
                    return True

                embedding_hits = [hit for hit in embedding_hits
                                  if allowed(hit)]
            chunk_metas = self._chunk_meta_map(
                {row[0] for row in rows}
                | {hit.get("_rowid") for hit in embedding_hits})
        default = {"kind": "document", "source_url": "", "teacher": "",
                   "model": "", "verified": True, "created": "",
                   "reviewer": "", "verified_at": "", "page": None,
                   "section": "", "char_start": None, "char_end": None}
        hits = [{
            "chunk": row[1],
            "source": row[2],
            "title": row[3],
            "score": float(row[4]),
            "_rowid": row[0],
            **default,
            **metas.get(row[2], {}),
            **chunk_metas.get(row[0], {}),
        } for row in rows]
        for hit in embedding_hits:
            hit.update(chunk_metas.get(hit.get("_rowid"), {}))
        if not rerank:
            return dedupe_hits(hits, limit=k) if diverse else hits[:k]
        # Hybrid candidate pool: FTS rows plus the best embedding rows. A
        # paraphrase with no lexical overlap with any document can now enter
        # the reranker, which was previously capped by FTS recall.
        merged = []
        seen = set()
        for hit in hits + embedding_hits:
            key = (hit["source"], hit["chunk"])
            if key in seen:
                continue
            seen.add(key)
            merged.append(hit)
        finite = [hit["score"] for hit in hits
                  if hit.get("score") is not None]
        fallback = max(finite) if finite else None
        for hit in merged:
            similarity = embedding_scores.get(hit.get("_rowid"))
            if similarity is not None:
                hit["embedding_score"] = similarity
                if neural:
                    hit["neural_score"] = similarity
            if hit.get("score") is None:
                hit["score"] = fallback
        rerank_limit = max(k * 3, 12) if diverse else k
        if active is not None:
            ranked = active.rerank(query, merged, rerank_limit)
        else:
            ranked = rerank_hits(query, merged, rerank_limit)
        if diverse:
            ranked = dedupe_hits(ranked, limit=k)
        for hit in ranked:
            hit.pop("_rowid", None)
        return ranked

    def chunks(self, limit: int = 500) -> list[dict]:
        """Return indexed chunks with provenance for graph/analysis jobs."""
        try:
            limit = max(1, min(int(limit), 5000))
        except (TypeError, ValueError):
            limit = 500
        with self._lock:
            rows = self.con.execute(
                "SELECT d.chunk, d.source, d.title, m.page, m.section, "
                "COALESCE(meta.kind, 'document'), "
                "COALESCE(meta.verified, 1) "
                "FROM docs d "
                "LEFT JOIN doc_chunk_meta m ON m.doc_rowid = d.rowid "
                "LEFT JOIN doc_meta meta ON meta.source = d.source "
                "ORDER BY d.rowid LIMIT ?", (limit,)).fetchall()
        return [{"chunk": row[0], "source": row[1], "title": row[2],
                 "page": row[3], "section": row[4] or "",
                 "kind": row[5], "verified": bool(row[6])}
                for row in rows]

    def sources(self) -> list[dict]:
        with self._lock:
            rows = self.con.execute(
                "SELECT source, title, COUNT(*), MAX(sha) FROM docs "
                "GROUP BY source, title ORDER BY source").fetchall()
            metas = self._meta_map([row[0] for row in rows])
            page_rows = self.con.execute(
                "SELECT docs.source, MIN(m.page), MAX(m.page) "
                "FROM doc_chunk_meta m JOIN docs ON docs.rowid = m.doc_rowid "
                "GROUP BY docs.source").fetchall()
        pages = {row[0]: (row[1], row[2]) for row in page_rows}
        result = []
        for row in rows:
            item = {
                "source": row[0],
                "title": row[1],
                "chunks": row[2],
                "content_sha256": row[3],
                **metas.get(row[0], {"source": row[0], "kind": "document",
                                     "source_url": "", "teacher": "",
                                     "model": "", "verified": True,
                                     "created": "", "reviewer": "",
                                     "verified_at": ""}),
            }
            page_range = pages.get(row[0])
            if page_range and page_range[0] is not None:
                item["page_start"] = page_range[0]
                item["page_end"] = page_range[1]
            result.append(item)
        return result

    def source_evidence(self, source: str, page: int | None = None,
                        section: str = "", limit: int = 20) -> dict:
        """Return the stored chunks for a source/page/section locator."""
        source = str(source or "").strip()
        if not source:
            raise ValueError("source is required")
        if len(source) > 512:
            raise ValueError("source is too long (max 512 chars)")
        if any(ord(ch) < 32 for ch in source):
            raise ValueError("source contains control characters")
        try:
            limit = int(limit)
        except (TypeError, ValueError) as exc:
            raise ValueError("limit must be an integer") from exc
        limit = max(1, min(limit, 50))
        clauses = ["d.source = ?"]
        params: list = [source]
        if page is not None:
            try:
                page = int(page)
            except (TypeError, ValueError) as exc:
                raise ValueError("page must be an integer") from exc
            if page < 1:
                raise ValueError("page must be >= 1")
            clauses.append("m.page = ?")
            params.append(page)
        section = str(section or "").strip()
        if section:
            if len(section) > 200:
                raise ValueError("section is too long (max 200 chars)")
            clauses.append("lower(m.section) = lower(?)")
            params.append(section)
        with self._lock:
            rows = self.con.execute(
                "SELECT d.chunk, d.source, d.title, m.page, m.section, "
                "m.char_start, m.char_end FROM docs d "
                "LEFT JOIN doc_chunk_meta m ON m.doc_rowid = d.rowid "
                f"WHERE {' AND '.join(clauses)} "
                "ORDER BY COALESCE(m.page, 0), d.rowid LIMIT ?",
                (*params, limit)).fetchall()
        if not rows:
            raise SourceNotFound(
                f"no evidence for source/page/section: {source}")
        data = [{
            "chunk": row[0],
            "source": row[1],
            "title": row[2],
            "page": row[3],
            "section": row[4] or "",
            "char_start": row[5],
            "char_end": row[6],
        } for row in rows]
        return {
            "source": source,
            "page": page,
            "section": section,
            "returned": len(data),
            "data": data,
        }

    def summarize_source(self, source: str,
                         max_sentences: int = 5) -> dict:
        """Extractive summary of one indexed source with its locators."""
        evidence = self.source_evidence(source, limit=50)
        data = evidence["data"]
        result = extractive_summary(
            [item["chunk"] for item in data], max_sentences=max_sentences)
        citations = []
        for item in data:
            citation = {"page": item.get("page"),
                        "section": item.get("section", "")}
            if citation not in citations and (citation["page"] is not None
                                              or citation["section"]):
                citations.append(citation)
        result.update({
            "source": evidence["source"],
            "title": data[0].get("title", "") if data else "",
            "chunks": len(data),
            "citations": citations,
        })
        return result

    def summarize_collection(self, source_prefix: str = "", kind: str = "",
                             max_sentences: int = 8) -> dict:
        """Extractive summary across a source prefix and/or kind."""
        prefix = str(source_prefix or "").strip()
        kind = str(kind or "").strip()
        if not prefix and not kind:
            raise ValueError("source_prefix or kind is required")
        items = [
            item for item in self.sources()
            if (not prefix or item["source"].startswith(prefix))
            and (not kind or item.get("kind") == kind)]
        if not items:
            raise SourceNotFound("no sources matched the collection filter")
        units = []
        chunks = 0
        for item in items:
            for row in self._source_chunk_rows(item["source"]):
                units.append({"text": row[0], "source": item["source"],
                              "page": row[2], "section": row[3] or ""})
                chunks += 1
        result = extractive_summary_units(
            units, max_sentences=max_sentences)
        result.update({
            "source_prefix": prefix,
            "kind": kind,
            "sources": len(items),
            "chunks": chunks,
        })
        return result

    def verify_source(self, source: str, reviewer: str) -> dict:
        """Mark a teacher note verified with reviewer and timestamp."""
        source = str(source or "").strip()
        reviewer = str(reviewer or "").strip()
        if not source:
            raise ValueError("source is required")
        if not reviewer:
            raise ValueError("reviewer is required")
        if len(reviewer) > 100:
            raise ValueError("reviewer is too long (max 100 chars)")
        if any(ord(ch) < 32 for ch in reviewer):
            raise ValueError("reviewer contains control characters")
        with self._lock:
            row = self.con.execute(
                "SELECT kind FROM doc_meta WHERE source = ?",
                (source,)).fetchone()
            if row is None:
                raise SourceNotFound(f"source not found: {source}")
            if row[0] != "teacher":
                raise ValueError("only teacher notes can be verified")
            verified_at = time.strftime("%Y-%m-%dT%H:%M:%S")
            self.con.execute(
                "UPDATE doc_meta SET verified = 1, reviewer = ?, "
                "verified_at = ? WHERE source = ?",
                (reviewer, verified_at, source))
            self.con.commit()
            meta = self._meta_map([source]).get(source, {})
            return {"source": source, **meta}

    def forget(self, source: str) -> dict:
        """Delete one source and its metadata, then compact the database."""
        source = str(source or "").strip()
        if not source:
            raise ValueError("source is required")
        variants = [source]
        if "://" not in source:
            # Windows stores local paths with backslashes; accept either form
            # on every platform (Path() treats a backslash as a normal char on
            # POSIX, so do the replacement explicitly too).
            for variant in (source.replace("/", "\\"),
                            source.replace("\\", "/"),
                            Path(source).as_posix(), str(Path(source))):
                if variant not in variants:
                    variants.append(variant)
        placeholders = ",".join("?" for _ in variants)
        with self._lock:
            chunks = self.con.execute(
                f"SELECT COUNT(*) FROM docs WHERE source IN ({placeholders})",
                tuple(variants)).fetchone()[0]
            meta = self.con.execute(
                f"SELECT COUNT(*) FROM doc_meta WHERE source IN "
                f"({placeholders})", tuple(variants)).fetchone()[0]
            if chunks == 0 and meta == 0:
                raise SourceNotFound(f"source not found: {source}")
            self.con.execute(
                f"DELETE FROM docs WHERE source IN ({placeholders})",
                tuple(variants))
            self.con.execute(
                f"DELETE FROM doc_meta WHERE source IN ({placeholders})",
                tuple(variants))
            self._drop_stale_vectors()
            self._drop_stale_chunk_meta()
            self._vector_index = None
            self.con.commit()
            vacuumed = False
            try:
                self.con.execute("VACUUM")
                vacuumed = True
            except sqlite3.OperationalError:
                pass
            return {"source": source, "chunks_deleted": int(chunks),
                    "meta_deleted": int(meta), "vacuumed": vacuumed}

    def forget_all(self) -> dict:
        """Delete every indexed source and compact the database."""
        with self._lock:
            chunks = self.con.execute(
                "SELECT COUNT(*) FROM docs").fetchone()[0]
            meta = self.con.execute(
                "SELECT COUNT(*) FROM doc_meta").fetchone()[0]
            self.con.execute("DELETE FROM docs")
            self.con.execute("DELETE FROM doc_meta")
            self._drop_stale_vectors()
            self._drop_stale_chunk_meta()
            self._vector_index = None
            self.con.commit()
            vacuumed = False
            try:
                self.con.execute("VACUUM")
                vacuumed = True
            except sqlite3.OperationalError:
                pass
            return {"chunks_deleted": int(chunks),
                    "meta_deleted": int(meta), "vacuumed": vacuumed}

    def stats(self) -> dict:
        with self._lock:
            count = self.con.execute(
                "SELECT COUNT(*) FROM docs").fetchone()[0]
        return {"path": str(self.path), "chunks": count,
                "sources": len(self.sources())}

    def integrity(self) -> dict:
        """Check the index for orphan rows and FTS corruption."""
        with self._lock:
            chunks = self.con.execute(
                "SELECT COUNT(*) FROM docs").fetchone()[0]
            sources = self.con.execute(
                "SELECT COUNT(DISTINCT source) FROM docs").fetchone()[0]
            meta = self.con.execute(
                "SELECT COUNT(*) FROM doc_meta").fetchone()[0]
            vectors = self.con.execute(
                "SELECT COUNT(*) FROM doc_vectors").fetchone()[0]
            chunk_meta = self.con.execute(
                "SELECT COUNT(*) FROM doc_chunk_meta").fetchone()[0]
            orphan_vectors = self.con.execute(
                "SELECT COUNT(*) FROM doc_vectors v "
                "LEFT JOIN docs d ON d.rowid = v.doc_rowid "
                "WHERE d.rowid IS NULL").fetchone()[0]
            orphan_chunk_meta = self.con.execute(
                "SELECT COUNT(*) FROM doc_chunk_meta m "
                "LEFT JOIN docs d ON d.rowid = m.doc_rowid "
                "WHERE d.rowid IS NULL").fetchone()[0]
            orphan_meta = self.con.execute(
                "SELECT COUNT(*) FROM doc_meta "
                "WHERE source NOT IN (SELECT source FROM docs)"
            ).fetchone()[0]
            missing_meta = self.con.execute(
                "SELECT COUNT(*) FROM (SELECT DISTINCT source FROM docs) d "
                "WHERE d.source NOT IN (SELECT source FROM doc_meta)"
            ).fetchone()[0]
            empty_chunks = self.con.execute(
                "SELECT COUNT(*) FROM docs WHERE trim(chunk) = ''"
            ).fetchone()[0]
            teachers = self.con.execute(
                "SELECT COUNT(*) FROM doc_meta WHERE kind = 'teacher'"
            ).fetchone()[0]
            unverified = self.con.execute(
                "SELECT COUNT(*) FROM doc_meta "
                "WHERE kind = 'teacher' AND verified = 0").fetchone()[0]
            fts_error = ""
            try:
                self.con.execute(
                    "INSERT INTO docs(docs) VALUES('integrity-check')")
            except sqlite3.DatabaseError as exc:
                fts_error = str(exc)
        issues = []
        if orphan_vectors:
            issues.append(f"{orphan_vectors} orphan vector row(s)")
        if orphan_chunk_meta:
            issues.append(f"{orphan_chunk_meta} orphan chunk-meta row(s)")
        if orphan_meta:
            issues.append(f"{orphan_meta} orphan source-meta row(s)")
        if missing_meta:
            issues.append(f"{missing_meta} source(s) without metadata")
        if empty_chunks:
            issues.append(f"{empty_chunks} empty chunk(s)")
        if fts_error:
            issues.append(f"FTS integrity check failed: {fts_error}")
        return {
            "ok": not issues,
            "issues": issues,
            "path": str(self.path),
            "chunks": int(chunks),
            "sources": int(sources),
            "meta": int(meta),
            "vectors": int(vectors),
            "chunk_meta": int(chunk_meta),
            "teacher_notes": int(teachers),
            "unverified_teacher_notes": int(unverified),
            "orphan_vectors": int(orphan_vectors),
            "orphan_chunk_meta": int(orphan_chunk_meta),
            "orphan_meta": int(orphan_meta),
            "missing_meta": int(missing_meta),
            "empty_chunks": int(empty_chunks),
            "vector_index_built": self._vector_index is not None,
        }

    def close(self):
        with self._lock:
            self.con.close()


def main(argv=None):
    from atomzero import __version__

    ap = argparse.ArgumentParser(
        prog="atomzero-docs",
        description="Index and search documents for Atom Zero.")
    ap.add_argument("--db", default=None,
                    help="SQLite index path (default: data/docs.db)")
    ap.add_argument("--version", action="version",
                    version=f"atomzero-docs {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    add = sub.add_parser("add", help="index a file or a whole directory")
    add.add_argument("path")
    add.add_argument("--title", default="")
    add.add_argument("--recursive", action=argparse.BooleanOptionalAction,
                     default=True)
    add.add_argument("--skip-unchanged", action="store_true")
    add.add_argument("--dedupe", action="store_true",
                     help="skip files whose extracted content hash already "
                          "exists under another source")
    search = sub.add_parser("search", help="search the local index")
    search.add_argument("query")
    search.add_argument("-k", type=int, default=5)
    search.add_argument("--source", default="")
    search.add_argument("--source-prefix", default="")
    search.add_argument("--kind", default="")
    search.add_argument("--no-rerank", action="store_true")
    search.add_argument("--diverse", action="store_true",
                        help="suppress near-identical chunks")
    answer = sub.add_parser(
        "answer", help="return a cited answer with evidence and confidence")
    answer.add_argument("query")
    answer.add_argument("-k", type=int, default=6)
    answer.add_argument("--source", default="")
    answer.add_argument("--source-prefix", default="")
    answer.add_argument("--kind", default="")
    sync = sub.add_parser(
        "sync", help="sync a directory collection into the index")
    sync.add_argument("path")
    sync.add_argument("--prefix", default="")
    sync.add_argument("--recursive", action=argparse.BooleanOptionalAction,
                      default=True)
    sync.add_argument("--no-prune", action="store_true")
    sync.add_argument("--no-dedupe", action="store_true")
    status = sub.add_parser(
        "status", help="compare a directory with the index without writing")
    status.add_argument("path")
    status.add_argument("--prefix", default="")
    status.add_argument("--recursive", action=argparse.BooleanOptionalAction,
                        default=True)
    doctor = sub.add_parser(
        "doctor", help="check the index for orphan rows and FTS corruption")
    list_db = sub.add_parser("list", help="list indexed sources")
    stats_db = sub.add_parser("stats", help="show index statistics")
    evidence = sub.add_parser("evidence", help="show chunks for a source")
    evidence.add_argument("source")
    evidence.add_argument("--page", type=int, default=None)
    evidence.add_argument("--section", default="")
    summary = sub.add_parser(
        "summary", help="extractive summary of one indexed source")
    summary.add_argument("source")
    summary.add_argument("--sentences", type=int, default=5)
    summary_collection = sub.add_parser(
        "summary-collection",
        help="extractive summary across a prefix and/or kind")
    summary_collection.add_argument("--prefix", default="")
    summary_collection.add_argument("--kind", default="")
    summary_collection.add_argument("--sentences", type=int, default=8)
    graph = sub.add_parser(
        "graph", help="build a local co-occurrence knowledge graph")
    graph.add_argument("--limit-chunks", type=int, default=500)
    graph.add_argument("--min-edge", type=int, default=1)
    graph.add_argument("--max-entities", type=int, default=80)
    entity = sub.add_parser(
        "entity", help="show chunks and connections for one entity")
    entity.add_argument("name")
    entity.add_argument("--limit", type=int, default=10)
    path = sub.add_parser(
        "path", help="find a co-occurrence path between two entities")
    path.add_argument("source")
    path.add_argument("target")
    path.add_argument("--max-depth", type=int, default=4)
    export = sub.add_parser(
        "export", help="export the index to portable JSONL")
    export.add_argument("--out", default="atomzero-index.jsonl")
    export.add_argument("--source-prefix", default="")
    export.add_argument("--kind", default="")
    import_cmd = sub.add_parser(
        "import", help="import sources from an export JSONL file")
    import_cmd.add_argument("path")
    import_cmd.add_argument("--no-replace", action="store_true")
    forget = sub.add_parser("forget", help="delete one source or everything")
    forget.add_argument("source", nargs="?")
    forget.add_argument("--all", action="store_true")
    # Accept --db on either side of the subcommand; the docs use both forms.
    for parser in (add, search, answer, sync, status, doctor, list_db,
                   stats_db, evidence, summary, summary_collection, graph,
                   entity, path, export, import_cmd, forget):
        parser.add_argument("--db", dest="db_after", default=None)
    args = ap.parse_args(argv)
    mem = DocMemory(args.db_after or args.db or "data/docs.db")
    try:
        if args.cmd == "add":
            path = Path(args.path)
            if path.is_dir():
                result = mem.add_directory(
                    path, recursive=args.recursive,
                    skip_if_unchanged=args.skip_unchanged,
                    dedupe_across_sources=args.dedupe)
                result["added"] = result["chunks"]
            else:
                result = mem.add_file_result(
                    path, title=args.title,
                    skip_if_unchanged=args.skip_unchanged,
                    dedupe_across_sources=args.dedupe)
                result["added"] = result["chunks"]
            print(json.dumps(result, indent=2, ensure_ascii=False))
        elif args.cmd == "search":
            hits = mem.search(
                args.query, k=args.k, rerank=not args.no_rerank,
                source=args.source, source_prefix=args.source_prefix,
                kind=args.kind, diverse=args.diverse)
            print(json.dumps(hits, indent=2, ensure_ascii=False))
        elif args.cmd == "answer":
            hits = mem.search(
                args.query, k=args.k, source=args.source,
                source_prefix=args.source_prefix, kind=args.kind)
            if not hits:
                result = {
                    "text": "I do not have a source for that yet.",
                    "sources": [], "evidence": [],
                    "confidence": {"score": 0.0, "label": "none",
                                   "coverage": 0.0, "margin": 0.0},
                }
            else:
                result = grounded_answer(args.query, hits)
            print(json.dumps(result, indent=2, ensure_ascii=False))
        elif args.cmd == "sync":
            result = mem.sync_directory(
                args.path, prefix=args.prefix, recursive=args.recursive,
                prune=not args.no_prune,
                dedupe_across_sources=not args.no_dedupe)
            result["added"] = result["chunks"]
            print(json.dumps(result, indent=2, ensure_ascii=False))
        elif args.cmd == "status":
            result = mem.directory_status(
                args.path, prefix=args.prefix, recursive=args.recursive)
            print(json.dumps(result, indent=2, ensure_ascii=False))
        elif args.cmd == "doctor":
            result = mem.integrity()
            print(json.dumps(result, indent=2, ensure_ascii=False))
            if not result["ok"]:
                return 1
        elif args.cmd == "list":
            print(json.dumps(mem.sources(), indent=2, ensure_ascii=False))
        elif args.cmd == "evidence":
            try:
                result = mem.source_evidence(
                    args.source, page=args.page, section=args.section)
            except (ValueError, SourceNotFound) as exc:
                print(json.dumps({"error": str(exc)}))
                return 1
            print(json.dumps(result, indent=2, ensure_ascii=False))
        elif args.cmd == "summary":
            try:
                result = mem.summarize_source(
                    args.source, max_sentences=args.sentences)
            except (ValueError, SourceNotFound) as exc:
                print(json.dumps({"error": str(exc)}))
                return 1
            print(json.dumps(result, indent=2, ensure_ascii=False))
        elif args.cmd == "summary-collection":
            try:
                result = mem.summarize_collection(
                    source_prefix=args.prefix, kind=args.kind,
                    max_sentences=args.sentences)
            except (ValueError, SourceNotFound) as exc:
                print(json.dumps({"error": str(exc)}))
                return 1
            print(json.dumps(result, indent=2, ensure_ascii=False))
        elif args.cmd == "graph":
            from atomzero.knowledge_graph import build_graph

            result = build_graph(
                mem, limit_chunks=args.limit_chunks,
                min_edge=args.min_edge, max_entities=args.max_entities)
            print(json.dumps(result, indent=2, ensure_ascii=False))
        elif args.cmd == "entity":
            from atomzero.knowledge_graph import entity_sources

            try:
                result = entity_sources(mem, args.name, limit=args.limit)
            except ValueError as exc:
                print(json.dumps({"error": str(exc)}))
                return 1
            print(json.dumps(result, indent=2, ensure_ascii=False))
        elif args.cmd == "path":
            from atomzero.knowledge_graph import graph_path

            try:
                result = graph_path(
                    mem, args.source, args.target,
                    max_depth=args.max_depth)
            except ValueError as exc:
                print(json.dumps({"error": str(exc)}))
                return 1
            print(json.dumps(result, indent=2, ensure_ascii=False))
        elif args.cmd == "export":
            result = mem.export_jsonl(
                args.out, source_prefix=args.source_prefix, kind=args.kind)
            print(json.dumps(result, indent=2, ensure_ascii=False))
        elif args.cmd == "import":
            result = mem.import_jsonl(
                args.path, replace=not args.no_replace)
            print(json.dumps(result, indent=2, ensure_ascii=False))
            if result["failed"]:
                return 1
        elif args.cmd == "forget":
            try:
                result = (mem.forget_all() if args.all
                          else mem.forget(args.source or ""))
            except (ValueError, SourceNotFound) as exc:
                print(json.dumps({"error": str(exc)}))
                return 1
            print(json.dumps(result, ensure_ascii=False))
        else:
            print(json.dumps(mem.stats(), indent=2))
    finally:
        mem.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
