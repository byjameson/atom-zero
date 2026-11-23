"""Deterministic verification of teacher answers.

A teacher answer is untrusted input. This module turns it into a small,
content-free verification record by reusing the claim/evidence branch:

- :func:`atomzero.claims.extract_claims` splits the answer into claims;
- :func:`atomzero.claims.link_evidence` scores every claim against the local
  retrieval hits at the requested support threshold;
- :func:`atomzero.claims.compare_claims` flags numeric, date, identifier and
  negation conflicts inside the answer.

It never calls a teacher, a network, a store or an agent. A ``verified``
result is **evidence**, not a promotion: storing the note in retrieval memory
or using it as an adapter candidate still requires an explicit human decision.
No weights are changed and nothing is applied automatically.

CLI::

    atomzero-verify-teacher --answer-json answer.json --json
    atomzero-verify-teacher --answer-json - --json

See ``docs/TEACHER_VERIFICATION.md``.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping
from pathlib import Path

from atomzero import __version__
from atomzero.claims import compare_claims, extract_claims, link_evidence

__all__ = [
    "DEFAULT_MIN_SUPPORT",
    "METHOD",
    "STATUSES",
    "STATUS_REASONS",
    "verify_answer",
    "verification_event",
    "main",
]

METHOD = "deterministic_teacher_verification_v1"
DEFAULT_MIN_SUPPORT = 0.60
VERSION = 1

EMPTY = "empty"
CONFLICTED = "conflicted"
UNVERIFIED = "unverified"
VERIFIED = "verified"
STATUSES = frozenset({EMPTY, CONFLICTED, UNVERIFIED, VERIFIED})

NO_CLAIMS = "no_claims"
CONTRADICTION = "contradiction"
INSUFFICIENT_SUPPORT = "insufficient_support"
NO_SOURCES = "no_sources"
ALL_CLAIMS_SUPPORTED = "all_claims_supported"
STATUS_REASONS = {
    EMPTY: NO_CLAIMS,
    CONFLICTED: CONTRADICTION,
    UNVERIFIED: INSUFFICIENT_SUPPORT,
    VERIFIED: ALL_CLAIMS_SUPPORTED,
}

OUTCOMES = frozenset({"success", "failure", "unknown"})
COUNT_KEYS = (
    "claim_count",
    "supported_count",
    "weak_count",
    "unsupported_count",
    "conflict_count",
    "source_count",
)
INPUT_KEYS = frozenset({"answer", "hits"})


def _answer(value) -> str:
    if not isinstance(value, str):
        raise ValueError("answer must be a string")
    if not value.strip():
        raise ValueError("answer must be a non-empty string")
    return value


def _hits(value) -> list[dict]:
    if not isinstance(value, list):
        raise ValueError("hits must be a list of mappings")
    result = []
    for index, hit in enumerate(value):
        if not isinstance(hit, Mapping):
            raise ValueError(f"hits[{index}] must be a mapping")
        data = dict(hit)
        for key in data:
            if not isinstance(key, str):
                raise ValueError(f"hits[{index}] keys must be strings")
        result.append(data)
    return result


def _min_support(value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("min_support must be a number between 0 and 1")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError("min_support must be a number between 0 and 1")
    return number


def _require_sources(value) -> bool:
    if not isinstance(value, bool):
        raise ValueError("require_sources must be a boolean")
    return value


def verify_answer(answer, hits, *, min_support=DEFAULT_MIN_SUPPORT,
                  require_sources=True) -> dict:
    """Verify a teacher answer against local retrieval hits.

    The result is deterministic and content-free. Status rules are evaluated
    in this order, first match wins:

    ===================  =========================  ==========================
    status               reason_code                condition
    ===================  =========================  ==========================
    ``empty``            ``no_claims``              no sentence-level claims
    ``conflicted``       ``contradiction``          at least one claim conflict
    ``unverified``       ``insufficient_support``   any weak/unsupported claim
    ``unverified``       ``no_sources``             sources required but absent
    ``verified``         ``all_claims_supported``   every claim is supported
    ===================  =========================  ==========================

    ``verified`` only means the local evidence supports the claims under the
    lexical scoring rules. It is not a truth verdict and it is not a
    promotion: the caller must still approve storing the note.
    """
    text = _answer(answer)
    hit_list = _hits(hits)
    threshold = _min_support(min_support)
    must_have_sources = _require_sources(require_sources)

    claims = extract_claims(text)
    linked = link_evidence(claims, hit_list, min_coverage=threshold)
    comparison = compare_claims(linked)

    supported = sum(1 for item in linked if item["status"] == "supported")
    weak = sum(1 for item in linked if item["status"] == "weak")
    unsupported = sum(1 for item in linked if item["status"] == "unsupported")
    sources = set()
    for item in linked:
        evidence = item.get("evidence")
        if not isinstance(evidence, Mapping):
            continue
        source = evidence.get("source")
        if isinstance(source, str) and source.strip():
            sources.add(source.strip())
    source_count = len(sources)
    conflict_count = int(comparison.get("conflict_count", 0))

    if not linked:
        status, reason = EMPTY, NO_CLAIMS
    elif conflict_count:
        status, reason = CONFLICTED, CONTRADICTION
    elif weak or unsupported:
        status, reason = UNVERIFIED, INSUFFICIENT_SUPPORT
    elif must_have_sources and not source_count:
        status, reason = UNVERIFIED, NO_SOURCES
    else:
        status, reason = VERIFIED, ALL_CLAIMS_SUPPORTED

    verified = status == VERIFIED
    return {
        "status": status,
        "reason_code": reason,
        "verified": verified,
        "claim_count": len(linked),
        "supported_count": supported,
        "weak_count": weak,
        "unsupported_count": unsupported,
        "conflict_count": conflict_count,
        "source_count": source_count,
        "min_support": round(threshold, 4),
        "requires_human": verified,
        "auto_apply": False,
        "weights_changed": False,
        "method": METHOD,
    }


def _event_counts(result: Mapping) -> dict:
    counts = {}
    for key in COUNT_KEYS:
        value = result.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"result.{key} must be a non-negative integer")
        counts[key] = value
    if counts["claim_count"] != (
            counts["supported_count"] + counts["weak_count"]
            + counts["unsupported_count"]):
        raise ValueError("result counts do not add up to claim_count")
    if counts["source_count"] > counts["claim_count"]:
        raise ValueError("result.source_count cannot exceed claim_count")
    return counts


def _validated_result(result) -> tuple[str, str, bool, dict]:
    if not isinstance(result, Mapping):
        raise ValueError("result must be a mapping")
    status = result.get("status")
    if not isinstance(status, str) or status not in STATUSES:
        raise ValueError(
            "result.status must be one of: " + ", ".join(sorted(STATUSES)))
    reason = result.get("reason_code")
    if reason != STATUS_REASONS[status]:
        raise ValueError("result.reason_code is not valid for status")
    verified = result.get("verified")
    if not isinstance(verified, bool):
        raise ValueError("result.verified must be a boolean")
    if verified != (status == VERIFIED):
        raise ValueError("result.verified does not match status")
    return status, reason, verified, _event_counts(result)


def verification_event(result, *, outcome="success") -> dict:
    """Convert a verification result into a content-free growth event.

    The event is standalone rather than directly appendable to the escalation
    growth ledger: the ledger validator accepts only routing records with a
    timestamp and rejects the ``status``/count fields that make verification
    auditable. The caller can still turn a *promotable* event into a
    ``verified_teacher`` escalation event after the human promotion decision.

    ``promotable`` is true only for a successful, verified result. It does not
    apply anything, train an adapter or change weights.
    """
    status, reason, verified, counts = _validated_result(result)
    if not isinstance(outcome, str) or outcome not in OUTCOMES:
        raise ValueError(
            "outcome must be one of: " + ", ".join(sorted(OUTCOMES)))
    promotable = verified and outcome == "success"
    event = {
        "version": VERSION,
        "status": status,
        "reason_code": reason,
        "outcome": outcome,
        "verified": verified,
        "promotable": promotable,
        "requires_human": promotable,
        "auto_apply": False,
        "weights_changed": False,
        "growth_event": "verified_teacher" if promotable else "",
        "method": METHOD,
    }
    event.update(counts)
    return event


def _read_json(path: str):
    if path == "-":
        raw = sys.stdin.read()
        label = "stdin"
    else:
        raw = Path(path).read_text(encoding="utf-8-sig")
        label = path
    if not raw.strip():
        raise ValueError(f"empty JSON input: {label}")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {label}: {exc.msg}") from exc


def _payload(data) -> tuple[str, list]:
    if not isinstance(data, Mapping):
        raise ValueError("answer input must be a JSON object")
    unknown = sorted(set(data) - INPUT_KEYS)
    if unknown:
        raise ValueError(
            "answer input has unknown key(s): " + ", ".join(unknown))
    if "answer" not in data:
        raise ValueError("answer is required")
    if "hits" not in data:
        raise ValueError("hits are required")
    return data["answer"], data["hits"]


def _render_result(result: dict) -> str:
    return "\n".join([
        f"Teacher verification ({result['method']})",
        f"status: {result['status']}",
        f"reason: {result['reason_code']}",
        f"verified: {str(result['verified']).lower()}",
        f"claims: {result['claim_count']} "
        f"(supported={result['supported_count']}, "
        f"weak={result['weak_count']}, "
        f"unsupported={result['unsupported_count']})",
        f"conflicts: {result['conflict_count']}",
        f"sources: {result['source_count']}",
        f"min_support: {result['min_support']:.2f}",
        f"requires_human: {str(result['requires_human']).lower()}",
        "auto_apply: false",
        "weights_changed: false",
    ])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="atomzero-verify-teacher",
        description=("Deterministic, content-free verification of a teacher "
                     "answer against local hits. The CLI never calls a "
                     "teacher, network, store or agent."))
    parser.add_argument("--version", action="version",
                        version=f"atomzero-verify-teacher {__version__}")
    parser.add_argument("--answer-json", default="",
                        help="JSON object with answer and hits; - for stdin")
    parser.add_argument("--json", action="store_true",
                        help="print the verification result as JSON")
    args = parser.parse_args(argv)

    try:
        if not args.answer_json:
            raise ValueError("--answer-json is required")
        payload = _read_json(args.answer_json)
        answer, hits = _payload(payload)
        result = verify_answer(answer, hits)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(_render_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
