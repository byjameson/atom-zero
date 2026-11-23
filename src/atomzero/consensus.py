"""Deterministic cross-source claim consensus.

A single retrieved chunk can support a claim; several chunks can support the
same claim independently. This module clusters sentence-level claims from
retrieved hits by lexical overlap and reports which clusters are supported by
more than one source. It also reuses the transparent claim conflict audit, so
numeric/date/identifier/negation mismatches stay visible instead of being
averaged away.

It makes no model calls, does not persist input and does not change weights.

    from atomzero.consensus import audit_consensus
    report = audit_consensus(hits, min_sources=2)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from atomzero.claims import compare_claims, extract_claims

__all__ = [
    "audit_consensus",
    "cluster_claims",
    "consensus_from_rows",
]

_TOKEN_RE = re.compile(r"[0-9A-Za-zÇĞİÖŞÜçğıöşü]+")
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")
_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "are", "was",
    "were", "has", "have", "had", "not", "but", "its", "their", "they",
    "bir", "ve", "ile", "için", "bu", "şu", "olan", "olarak", "daha",
    "var", "yok", "ama", "çok", "gibi", "kadar", "sonra", "önce",
}


def _tokens(text: str) -> set[str]:
    return {
        token.lower() for token in _TOKEN_RE.findall(text or "")
        if len(token) >= 3 and token.lower() not in _STOPWORDS
    }


def _numbers(text: str) -> set[str]:
    return set(_NUMBER_RE.findall(text or ""))


def _overlap(left: str, right: str) -> float:
    a = _tokens(left)
    b = _tokens(right)
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def _compatible(left: str, right: str) -> bool:
    """Aynı kümede birleşmeden önce sayısal çelişkiyi engeller."""
    a = _numbers(left)
    b = _numbers(right)
    return not (a and b and a != b)


def _hit_text(hit: dict) -> str:
    for key in ("chunk", "snippet", "text"):
        value = hit.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _hit_source(hit: dict, index: int) -> str:
    for key in ("source", "title"):
        value = hit.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return f"hit-{index}"


def _claim_rows(hits: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for index, hit in enumerate(hits):
        if not isinstance(hit, dict):
            continue
        text = _hit_text(hit)
        if not text:
            continue
        source = _hit_source(hit, index)
        for claim in extract_claims(text):
            rows.append({
                "claim_id": f"{index + 1:02d}-{claim['claim_id']}",
                "text": claim["text"],
                "start": claim["start"],
                "end": claim["end"],
                "source": source,
                "title": str(hit.get("title") or ""),
                "section": str(hit.get("section") or ""),
                "page": hit.get("page"),
            })
    rows.sort(key=lambda row: (row["source"], row["start"], row["text"]))
    return rows


def cluster_claims(rows: list[dict], min_overlap: float = 0.55) -> list[dict]:
    """Claim satırlarını sözcüksel örtüşme ile kümeler.

    Aynı sayısal değere sahip olmayan iki claim birleştirilmez; bu, çelişen
    ölçümlerin tek bir "uzlaşma" altında gizlenmesini engeller.
    """
    if not 0.0 <= min_overlap <= 1.0:
        raise ValueError("min_overlap 0 ile 1 arasında olmalıdır")
    clusters: list[dict] = []
    for row in rows:
        match = None
        best = 0.0
        for cluster in clusters:
            representative = cluster["claims"][0]["text"]
            score = _overlap(row["text"], representative)
            if score >= min_overlap and score > best and _compatible(
                    row["text"], representative):
                match = cluster
                best = score
        if match is None:
            clusters.append({"claims": [row]})
        else:
            match["claims"].append(row)
    for index, cluster in enumerate(clusters, start=1):
        claims = sorted(
            cluster["claims"],
            key=lambda row: (row["source"], row["start"], row["claim_id"]),
        )
        representative = min(
            claims, key=lambda row: (-len(row["text"]), row["claim_id"]))
        sources = sorted({claim["source"] for claim in claims})
        cluster.clear()
        cluster.update({
            "cluster_id": f"g{index:03d}",
            "text": representative["text"],
            "claims": claims,
            "claim_count": len(claims),
            "sources": sources,
            "source_count": len(sources),
        })
    return clusters


def consensus_from_rows(
    rows: list[dict],
    *,
    min_overlap: float = 0.55,
    min_sources: int = 2,
) -> dict:
    """Hazır claim satırlarından uzlaşma raporu üretir."""
    if min_sources < 1:
        raise ValueError("min_sources en az 1 olmalıdır")
    clusters = cluster_claims(rows, min_overlap=min_overlap)
    for cluster in clusters:
        cluster["status"] = (
            "consensus" if cluster["source_count"] >= min_sources
            else "single_source"
        )
    conflict_report = compare_claims(rows) if rows else {
        "claim_count": 0, "conflict_count": 0, "conflicts": [], "counts": {},
        "method": "phrase overlap + number/unit/date/code/negation normalisation; not a truth verdict",
        "weights_changed": False,
    }
    consensus_count = sum(
        1 for cluster in clusters if cluster["status"] == "consensus")
    single_count = len(clusters) - consensus_count
    return {
        "clusters": clusters,
        "conflicts": conflict_report["conflicts"],
        "conflict_count": conflict_report["conflict_count"],
        "summary": {
            "claim_count": len(rows),
            "cluster_count": len(clusters),
            "consensus_clusters": consensus_count,
            "single_source_clusters": single_count,
            "source_count": len({row["source"] for row in rows}),
            "conflict_count": conflict_report["conflict_count"],
        },
        "weights_changed": False,
        "method": "deterministic_cross_source_consensus_v1",
    }


def audit_consensus(
    hits: list[dict],
    *,
    min_overlap: float = 0.55,
    min_sources: int = 2,
) -> dict:
    """Retrieval hit'lerinden çapraz kaynak uzlaşma raporu üretir."""
    if not isinstance(hits, list):
        raise ValueError("hits bir liste olmalıdır")
    return consensus_from_rows(
        _claim_rows(hits), min_overlap=min_overlap, min_sources=min_sources)


def _render(report: dict) -> str:
    summary = report["summary"]
    lines = [
        f"Consensus — {summary['cluster_count']} cluster, "
        f"{summary['consensus_clusters']} consensus, "
        f"{summary['single_source_clusters']} single-source, "
        f"{summary['conflict_count']} conflict",
    ]
    for cluster in report["clusters"]:
        lines.append(
            f"- [{cluster['status']}] {cluster['text']} "
            f"({cluster['source_count']} source(s))")
    for conflict in report["conflicts"]:
        lines.append(
            f"- [conflict] {', '.join(conflict['reasons'])}: "
            f"{conflict['left']} <> {conflict['right']}")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, help="JSON hit listesi")
    parser.add_argument("--min-overlap", type=float, default=0.55)
    parser.add_argument("--min-sources", type=int, default=2)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.file:
            payload = json.loads(args.file.read_text(encoding="utf-8-sig"))
        else:
            raw = sys.stdin.read()
            payload = json.loads(raw) if raw.strip() else []
    except (OSError, json.JSONDecodeError) as exc:
        print(f"input error: {exc}", file=sys.stderr)
        return 2
    if isinstance(payload, dict):
        payload = payload.get("hits", [])
    if not isinstance(payload, list) or not payload:
        print("no hits", file=sys.stderr)
        return 2
    try:
        report = audit_consensus(
            payload, min_overlap=args.min_overlap,
            min_sources=args.min_sources)
    except ValueError as exc:
        print(f"input error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(_render(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
