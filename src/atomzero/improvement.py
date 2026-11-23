"""Deterministic, human-gated improvement planning for Atom Zero.

The improvement loop is the planning layer of a self-improving system, not an
automatic weight updater::

    local weakness signals -> prioritized plan -> human approval
    -> evaluation gate -> content-free decision ledger

No model is called, no source text is persisted, no weight is changed and no
action is applied automatically. The module only reads local summaries that the
caller already has and turns them into a reviewable plan.

    python -m atomzero.improvement --claims-json claims.json --json
    python -m atomzero.improvement --gaps-json gaps.json --limit 3
    python -m atomzero.improvement < payload.json

The stdin form accepts one JSON object with any of ``claim_audit`` (alias
``claims``), ``gaps``, ``feedback``, ``freshness`` or ``reliability``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

__all__ = [
    "VALID_DECISIONS",
    "collect_signals",
    "build_plan",
    "plan_from_inputs",
    "record_decision",
    "record_gated_decision",
    "summarize_ledger",
    "verify_ledger",
    "main",
]

METHOD = "deterministic_human_gated_improvement_v1"
VERSION = 1
NOTE_ID_LENGTH = 16
MIN_LIMIT = 1
MAX_LIMIT = 20

VALID_DECISIONS = frozenset({"approve", "reject", "defer"})
CLOSED_GAP_STATUSES = frozenset({"done", "dismissed", "closed"})
APPROVED_FEEDBACK_STATUSES = frozenset({"approved", "exported"})

# Ordering follows the user-facing severity order used by the plan report.
SIGNAL_PRIORITIES = {
    "runner_scope_gap": 88,
    "scoreboard_regression": 85,
    "unsupported_claims": 80,
    "calibration_drift": 78,
    "conflicts": 75,
    "aisc_coverage_gaps": 72,
    "knowledge_gaps": 70,
    "single_source": 65,
    "stale_sources": 60,
    "undated_sources": 55,
    "weak_sources": 50,
    "pending_feedback": 40,
}

ACCEPTANCE_GATES = {
    "runner_scope_gap": (
        "fix committed-change scope detection and rerun the runner audit"),
    "scoreboard_regression": (
        "restore the regressed metric and rerun the scoreboard and gate"),
    "calibration_drift": (
        "recalibrate or replace the signal, rerun the answerability profile "
        "and compare it with the baseline"),
    "unsupported_claims": (
        "add a supporting source and rerun claim audit"),
    "conflicts": (
        "resolve conflicting claims and rerun claim audit"),
    "aisc_coverage_gaps": (
        "fix missing evidence files or unverified components and rerun AISC "
        "coverage"),
    "knowledge_gaps": (
        "add a local source for the top pending gap and rerun gap audit"),
    "single_source": (
        "add an independent second source and rerun claim audit"),
    "stale_sources": (
        "refresh or replace stale sources and rerun freshness audit"),
    "undated_sources": (
        "add publication dates or dated sources and rerun freshness audit"),
    "weak_sources": (
        "add more observations or stronger sources and rerun reliability "
        "summary"),
    "pending_feedback": (
        "review pending feedback and approve or reject it"),
}
REGRESSION_GATE = (
    "run the relevant pytest/AISC regression before accepting")
NEXT_STEPS = {
    "runner_scope_gap": (
        "reproduce a committed out-of-scope file, fix the runner diff check, "
        "then rerun the allowlisted gate"),
    "scoreboard_regression": (
        "run atomzero-scoreboard --summary, identify the regressed metric, "
        "fix it and rerun atomzero-gate"),
    "calibration_drift": (
        "run atomzero-drift, recalibrate the signal, then rerun the "
        "answerability profile and the gate"),
    "weak_sources": (
        "review the source reliability summary, remove or replace weak "
        "sources, then rerun atomzero-reliability and atomzero-gate"),
    "single_source": (
        "add an independent second source, rerun consensus, then rerun "
        "atomzero-gate"),
    "knowledge_gaps": (
        "propose the top gap with POST /v1/learning/gaps/propose, approve it "
        "in the research queue, then rerun the plan"),
    "pending_feedback": (
        "review the feedback queue and approve or reject each item"),
    "aisc_coverage_gaps": (
        "run atomzero-coverage, fix the missing evidence or unverified "
        "component, then rerun atomzero-gate"),
}
DEFAULT_NEXT_STEP = (
    "run atomzero-gate with the focused pytest/AISC checks before accepting")
SUGGESTED_CHECKS = {
    "runner_scope_gap": [
        "pytest:tests/test_agent_runner.py",
        "pytest:tests/test_improvement.py",
    ],
    "scoreboard_regression": [
        "pytest:tests/test_scoreboard.py",
        "pytest:tests/test_improvement.py",
    ],
    "calibration_drift": [
        "pytest:tests/test_drift.py",
        "pytest:tests/test_answerability.py",
    ],
    "unsupported_claims": ["pytest:tests/test_claims.py"],
    "conflicts": [
        "pytest:tests/test_claims.py", "pytest:tests/test_consensus.py"],
    "single_source": ["pytest:tests/test_consensus.py"],
    "knowledge_gaps": ["pytest:tests/test_improvement.py"],
    "stale_sources": ["pytest:tests/test_freshness.py"],
    "undated_sources": ["pytest:tests/test_freshness.py"],
    "weak_sources": ["pytest:tests/test_reliability.py"],
    "pending_feedback": ["pytest:tests/test_improvement.py"],
    "aisc_coverage_gaps": [
        "pytest:tests/test_aisc_coverage.py",
        "aisc:docs/aisc/atomzero-read.toml",
    ],
}
DEFAULT_CHECKS = ["pytest:tests/test_improvement.py"]

FORBIDDEN_KEYS = frozenset({
    "text", "query", "answer", "chunk", "content", "note", "session",
    "source",
})
GATE_KEYS = frozenset({
    "gate_ok", "gate_total", "gate_passed", "gate_method",
})
RECORD_KEYS = frozenset({
    "version", "at", "action_id", "kind", "decision", "note_id",
}) | GATE_KEYS
_INPUT_KEYS = frozenset({
    "claim_audit", "gaps", "feedback", "freshness", "reliability",
    "aisc_coverage", "scoreboard", "drift", "runner_audit",
})
_STDIN_ALIASES = {
    "claim_audit": "claim_audit",
    "claims": "claim_audit",
    "claims_audit": "claim_audit",
    "gaps": "gaps",
    "feedback": "feedback",
    "freshness": "freshness",
    "reliability": "reliability",
    "aisc_coverage": "aisc_coverage",
    "coverage": "aisc_coverage",
    "scoreboard": "scoreboard",
    "drift": "drift",
    "calibration_drift": "drift",
    "runner_audit": "runner_audit",
}
_NOTE_ID_RE = re.compile(r"\A(?:[0-9a-f]{1,64})?\Z")


def _non_negative_int(value, name: str) -> int:
    """Return a validated non-negative integer count."""
    if value is None:
        return 0
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer")
    if isinstance(value, int):
        number = value
    elif isinstance(value, float) and math.isfinite(value) and (
            value.is_integer()):
        number = int(value)
    elif isinstance(value, str) and value.strip().isdigit():
        number = int(value.strip())
    else:
        raise ValueError(f"{name} must be a non-negative integer")
    if number < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return number


def _freshness_summary(freshness) -> dict:
    """Return the flat summary block of a freshness/consensus report."""
    if freshness is None:
        return {}
    if not isinstance(freshness, dict):
        raise ValueError("freshness must be a mapping")
    nested = freshness.get("summary")
    return nested if isinstance(nested, dict) else freshness


def _signal(kind: str, count: int, title: str, detail: str,
            refs: list[str]) -> dict:
    return {
        "kind": kind,
        "priority": SIGNAL_PRIORITIES[kind],
        "count": count,
        "title": title,
        "detail": detail,
        "refs": list(refs),
    }


def collect_signals(*, claim_audit=None, gaps=None, feedback=None,
                    freshness=None, reliability=None,
                    aisc_coverage=None, scoreboard=None,
                    drift=None, runner_audit=None) -> list[dict]:
    """Collect deterministic weakness signals from local summaries.

    ``claim_audit`` is the content-free record written by
    :func:`atomzero.proactive.note_claim_audit`. ``gaps`` is a list of
    knowledge-gap dictionaries, ``feedback`` a list of feedback items,
    ``freshness`` a freshness/consensus summary and ``reliability`` a
    reliability summary from :func:`atomzero.reliability.summarize`.

    Only signals with a positive count are returned. Output is sorted by
    priority (highest first), then by kind, so equal inputs always produce the
    same list. No source text or model call is involved.
    """
    if claim_audit is not None and not isinstance(claim_audit, dict):
        raise ValueError("claim_audit must be a mapping")
    if gaps is not None and not isinstance(gaps, list):
        raise ValueError("gaps must be a list")
    if feedback is not None and not isinstance(feedback, list):
        raise ValueError("feedback must be a list")
    if reliability is not None and not isinstance(reliability, dict):
        raise ValueError("reliability must be a mapping")
    if aisc_coverage is not None and not isinstance(aisc_coverage, dict):
        raise ValueError("aisc_coverage must be a mapping")
    if scoreboard is not None and not isinstance(scoreboard, dict):
        raise ValueError("scoreboard must be a mapping")
    if drift is not None and not isinstance(drift, dict):
        raise ValueError("drift must be a mapping")
    if runner_audit is not None and not isinstance(runner_audit, dict):
        raise ValueError("runner_audit must be a mapping")

    claim = claim_audit if isinstance(claim_audit, dict) else {}
    fresh = _freshness_summary(freshness)
    status_counts = fresh.get("status_counts")
    if status_counts is None:
        status_counts = {}
    if not isinstance(status_counts, dict):
        raise ValueError("freshness.status_counts must be a mapping")

    signals: list[dict] = []

    unsupported = _non_negative_int(
        claim.get("unsupported"), "claim_audit.unsupported")
    if unsupported:
        signals.append(_signal(
            "unsupported_claims", unsupported, "Unsupported claims",
            f"{unsupported} claim(s) lack a supporting local source.",
            ["claim_audit"]))

    # A claim audit and a consensus summary may report the same underlying
    # counts. The larger view is used so the same audit is not double-counted.
    claim_conflicts = _non_negative_int(
        claim.get("conflict_count"), "claim_audit.conflict_count")
    fresh_conflicts = _non_negative_int(
        fresh.get("conflict_count"), "freshness.conflict_count")
    conflicts = max(claim_conflicts, fresh_conflicts)
    if conflicts:
        refs = []
        if claim_audit is not None:
            refs.append("claim_audit")
        if freshness is not None:
            refs.append("freshness")
        signals.append(_signal(
            "conflicts", conflicts, "Claim conflicts",
            f"{conflicts} conflicting claim pair(s) need review.", refs))

    if gaps is not None:
        pending = []
        for index, item in enumerate(gaps):
            if not isinstance(item, dict):
                raise ValueError(f"gaps[{index}] must be a mapping")
            status = str(item.get("status") or "").strip().lower()
            if status in CLOSED_GAP_STATUSES:
                continue
            pending.append((index, item))
        if pending:
            def _rank(pair):
                index, item = pair
                count = _non_negative_int(
                    item.get("count"), f"gaps[{index}].count")
                identifier = str(item.get("id") or f"index-{index}")
                return (-count, identifier)

            pending.sort(key=_rank)
            index, top = pending[0]
            gap_id = str(top.get("id") or f"index-{index}")
            query = str(top.get("query") or "").strip()
            refs = [f"gap:{gap_id}"]
            if query:
                refs.append(f"query:{query}")
            count = len(pending)
            signals.append(_signal(
                "knowledge_gaps", count, "Pending knowledge gaps",
                f"{count} gap(s) are waiting for a local source.",
                refs))

    claim_single = _non_negative_int(
        claim.get("single_source"), "claim_audit.single_source")
    fresh_single = _non_negative_int(
        fresh.get("single_source_clusters"),
        "freshness.single_source_clusters")
    single_source = max(claim_single, fresh_single)
    if single_source:
        refs = []
        if claim_audit is not None:
            refs.append("claim_audit")
        if freshness is not None:
            refs.append("freshness")
        signals.append(_signal(
            "single_source", single_source, "Single-source claims",
            f"{single_source} claim cluster(s) rely on one source.", refs))

    stale = _non_negative_int(
        status_counts.get("stale"), "freshness.status_counts.stale")
    if stale:
        signals.append(_signal(
            "stale_sources", stale, "Stale sources",
            f"{stale} retrieval hit(s) are past the freshness window.",
            ["freshness"]))

    undated = _non_negative_int(
        status_counts.get("undated"), "freshness.status_counts.undated")
    if undated:
        signals.append(_signal(
            "undated_sources", undated, "Undated sources",
            f"{undated} retrieval hit(s) have no publication date.",
            ["freshness"]))

    if reliability is not None:
        sources = reliability.get("sources")
        if sources is None:
            sources = []
        if not isinstance(sources, list):
            raise ValueError("reliability.sources must be a list")
        weak = 0
        weak_ids: list[str] = []
        for index, source in enumerate(sources):
            if not isinstance(source, dict):
                raise ValueError(
                    f"reliability.sources[{index}] must be a mapping")
            status = str(source.get("status") or "").strip().lower()
            observations = _non_negative_int(
                source.get("observations"),
                f"reliability.sources[{index}].observations")
            if status == "weak" or observations < 2:
                weak += 1
                source_id = str(source.get("source_id") or "").strip()
                if source_id:
                    weak_ids.append(f"source:{source_id}")
        if weak:
            refs = sorted(set(weak_ids))[:5] or ["reliability"]
            signals.append(_signal(
                "weak_sources", weak, "Weak or low-confidence sources",
                f"{weak} source(s) are weak or have fewer than two "
                "observations.", refs))

    if aisc_coverage is not None:
        coverage = aisc_coverage.get("summary")
        if not isinstance(coverage, dict):
            raise ValueError("aisc_coverage.summary must be a mapping")
        missing = _non_negative_int(
            coverage.get("specs_with_missing_evidence"),
            "aisc_coverage.summary.specs_with_missing_evidence")
        unverified = _non_negative_int(
            coverage.get("specs_with_unverified_components"),
            "aisc_coverage.summary.specs_with_unverified_components")
        errors = _non_negative_int(
            coverage.get("errors"), "aisc_coverage.summary.errors")
        total = missing + unverified + errors
        if total:
            signals.append(_signal(
                "aisc_coverage_gaps", total, "AISC coverage gaps",
                f"{total} AISC coverage problem(s): "
                f"{missing} missing evidence, {unverified} unverified specs, "
                f"{errors} errors.", ["aisc_coverage"]))

    if feedback is not None:
        pending_feedback = 0
        feedback_ids: list[str] = []
        for index, item in enumerate(feedback):
            if not isinstance(item, dict):
                raise ValueError(f"feedback[{index}] must be a mapping")
            status = str(item.get("status") or "").strip().lower()
            if status in APPROVED_FEEDBACK_STATUSES:
                continue
            pending_feedback += 1
            item_id = str(item.get("id") or "").strip()
            if item_id:
                feedback_ids.append(f"feedback:{item_id}")
        if pending_feedback:
            refs = sorted(set(feedback_ids))[:5] or ["feedback"]
            signals.append(_signal(
                "pending_feedback", pending_feedback, "Pending feedback",
                f"{pending_feedback} feedback item(s) are not approved or "
                "exported yet.", refs))

    if scoreboard is not None:
        latest = scoreboard.get("latest")
        delta = scoreboard.get("delta")
        latest = latest if isinstance(latest, dict) else {}
        delta = delta if isinstance(delta, dict) else {}
        regressions = []
        failed = _non_negative_int(
            latest.get("tests_failed"), "scoreboard.latest.tests_failed")
        if failed:
            regressions.append(f"{failed} test failure(s)")
        for key, label in (
                ("tests_passed", "fewer tests passed"),
                ("benchmark_passed", "fewer benchmark checks passed"),
                ("aisc_passed", "fewer AISC specs passed")):
            value = delta.get(key)
            if isinstance(value, bool) or not isinstance(value, int):
                continue
            if value < 0:
                regressions.append(f"{label} ({value:+d})")
        if regressions:
            signals.append(_signal(
                "scoreboard_regression", len(regressions),
                "Measured regression",
                "The scoreboard shows " + ", ".join(regressions) + ".",
                ["scoreboard"]))

    if runner_audit is not None:
        alerts = runner_audit.get("alerts")
        if alerts is None:
            alerts = []
        if not isinstance(alerts, list):
            raise ValueError("runner_audit.alerts must be a list")
        if alerts:
            signals.append(_signal(
                "runner_scope_gap", len(alerts), "Runner scope gap",
                "The runner audit reported "
                + ", ".join(str(alert) for alert in alerts) + ".",
                ["runner_audit"]))

    if drift is not None:
        alerts = drift.get("alerts")
        if alerts is None:
            alerts = []
        if not isinstance(alerts, list):
            raise ValueError("drift.alerts must be a list")
        if alerts:
            signals.append(_signal(
                "calibration_drift", len(alerts),
                "Calibration drift",
                "The calibration profile has "
                + ", ".join(str(alert) for alert in alerts) + ".",
                ["drift"]))

    return sorted(
        signals, key=lambda item: (-item["priority"], item["kind"]))


def _validate_limit(limit) -> int:
    if (isinstance(limit, bool) or not isinstance(limit, int)
            or not MIN_LIMIT <= limit <= MAX_LIMIT):
        raise ValueError(
            f"limit must be an integer between {MIN_LIMIT} and {MAX_LIMIT}")
    return limit


def _acceptance(kind: str) -> str:
    gate = ACCEPTANCE_GATES.get(kind)
    if gate is None:
        return REGRESSION_GATE
    return f"{gate}; {REGRESSION_GATE}"


def build_plan(signals, limit=5) -> dict:
    """Turn collected signals into a prioritized, human-gated plan.

    Signals are sorted by priority (highest first) and then by kind. At most
    ``limit`` actions are returned. Every action requires human approval and
    carries a concrete acceptance gate; nothing is applied or trained.
    """
    if not isinstance(signals, list):
        raise ValueError("signals must be a list")
    limit = _validate_limit(limit)
    normalized = []
    for index, signal in enumerate(signals):
        if not isinstance(signal, dict):
            raise ValueError(f"signals[{index}] must be a mapping")
        raw_kind = signal.get("kind")
        if not isinstance(raw_kind, str) or not raw_kind.strip():
            raise ValueError(f"signals[{index}].kind is required")
        kind = raw_kind.strip()
        priority = signal.get("priority")
        if isinstance(priority, bool) or not isinstance(
                priority, (int, float)):
            raise ValueError(f"signals[{index}].priority must be a number")
        number = float(priority)
        if not math.isfinite(number):
            raise ValueError(f"signals[{index}].priority must be finite")
        count = _non_negative_int(
            signal.get("count"), f"signals[{index}].count")
        title = str(signal.get("title") or "").strip() or (
            kind.replace("_", " ").capitalize())
        detail = str(signal.get("detail") or "").strip() or (
            f"{count} local item(s) flagged.")
        normalized.append({
            "kind": kind,
            "priority": (
                priority if isinstance(priority, int)
                else int(number) if number.is_integer() else number),
            "title": title,
            "detail": detail,
        })

    normalized.sort(
        key=lambda item: (-float(item["priority"]), item["kind"]))
    actions = []
    for position, signal in enumerate(normalized[:limit], start=1):
        actions.append({
            "action_id": f"imp-{position:02d}-{signal['kind']}",
            "kind": signal["kind"],
            "title": signal["title"],
            "detail": signal["detail"],
            "priority": signal["priority"],
            "requires_human": True,
            "acceptance": _acceptance(signal["kind"]),
            "next_step": NEXT_STEPS.get(
                signal["kind"], DEFAULT_NEXT_STEP),
            "suggested_checks": list(
                SUGGESTED_CHECKS.get(signal["kind"], DEFAULT_CHECKS)),
        })

    summary: dict[str, int] = {}
    for action in actions:
        summary[action["kind"]] = summary.get(action["kind"], 0) + 1
    return {
        "actions": actions,
        "summary": dict(sorted(summary.items())),
        "auto_apply": False,
        "weights_changed": False,
        "method": METHOD,
    }


def plan_from_inputs(*, limit=5, **inputs) -> dict:
    """Compose :func:`collect_signals` and :func:`build_plan`."""
    unknown = set(inputs) - _INPUT_KEYS
    if unknown:
        raise ValueError(
            "unknown input(s): " + ", ".join(sorted(unknown)))
    return build_plan(collect_signals(**inputs), limit=limit)


def _format_timestamp(now) -> str:
    if now is None:
        stamp = datetime.now(timezone.utc)
    elif isinstance(now, datetime):
        stamp = now
    elif isinstance(now, str) and now.strip():
        text = now.strip()
        if text.endswith(("Z", "z")):
            text = text[:-1] + "+00:00"
        try:
            stamp = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ValueError(
                "now must be an ISO-8601 datetime or datetime object") from exc
    else:
        raise ValueError("now must be an ISO-8601 datetime or datetime object")
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    stamp = stamp.astimezone(timezone.utc).replace(microsecond=0)
    return stamp.isoformat().replace("+00:00", "Z")


def record_decision(ledger, action, decision, note="", now=None) -> dict:
    """Append a content-free human decision to ``ledger`` and return it.

    The ledger is mutated in place (as documented) and the appended record is
    returned. ``note_id`` is a 16-hex SHA-256 prefix of the trimmed note; the
    note text itself is never stored.
    """
    if not isinstance(ledger, list):
        raise ValueError("ledger must be a list")
    if not isinstance(action, dict):
        raise ValueError("action must be a mapping")
    action_id = action.get("action_id")
    kind = action.get("kind")
    if not isinstance(action_id, str) or not action_id.strip():
        raise ValueError("action.action_id is required")
    if not isinstance(kind, str) or not kind.strip():
        raise ValueError("action.kind is required")
    if (not isinstance(decision, str)
            or decision not in VALID_DECISIONS):
        raise ValueError(
            "decision must be one of: " + ", ".join(sorted(VALID_DECISIONS)))
    if note is None:
        note = ""
    if not isinstance(note, str):
        raise ValueError("note must be a string")
    trimmed = note.strip()
    note_id = (
        hashlib.sha256(trimmed.encode("utf-8")).hexdigest()[:NOTE_ID_LENGTH]
        if trimmed else ""
    )
    record = {
        "version": VERSION,
        "at": _format_timestamp(now),
        "action_id": action_id.strip(),
        "kind": kind.strip(),
        "decision": decision,
        "note_id": note_id,
    }
    ledger.append(record)
    return record


def _gate_fields(gate) -> dict:
    if not isinstance(gate, dict):
        raise ValueError("gate must be an evaluation-gate report")
    ok = gate.get("ok")
    if not isinstance(ok, bool):
        raise ValueError("gate.ok must be a boolean")
    summary = gate.get("summary")
    if not isinstance(summary, dict):
        raise ValueError("gate.summary must be a mapping")
    total = _non_negative_int(summary.get("total"), "gate.summary.total")
    passed = _non_negative_int(
        summary.get("passed"), "gate.summary.passed")
    failed = _non_negative_int(
        summary.get("failed"), "gate.summary.failed")
    if passed + failed != total:
        raise ValueError("gate.summary passed + failed must equal total")
    if passed != total:
        ok = False
    method = str(gate.get("method") or "").strip()
    if not method:
        raise ValueError("gate.method is required")
    return {
        "gate_ok": ok,
        "gate_total": total,
        "gate_passed": passed,
        "gate_method": method[:80],
    }


def record_gated_decision(ledger, action, decision, gate, note="",
                          now=None) -> dict:
    """Record a decision that carries an evaluation-gate summary.

    An ``approve`` decision is rejected unless the gate passed every check.
    ``reject`` and ``defer`` may carry a failed gate, because the failure is
    the reason for the decision. Only gate counts and a method label are stored;
    command output is not copied into the ledger.
    """
    gate_fields = _gate_fields(gate)
    if decision == "approve" and not gate_fields["gate_ok"]:
        raise ValueError("cannot approve without a passing evaluation gate")
    record = record_decision(ledger, action, decision, note=note, now=now)
    record.update(gate_fields)
    return record


def summarize_ledger(ledger) -> dict:
    """Summarize a valid decision ledger without exposing note text.

    The ledger is content-free by construction; this summary reports counts by
    decision and action kind plus the latest timestamp.
    """
    report = verify_ledger(ledger)
    if not report["ok"]:
        raise ValueError("invalid ledger: " + "; ".join(report["errors"]))
    decisions: dict[str, int] = {}
    kinds: dict[str, int] = {}
    last_at = ""
    for record in ledger:
        decision = record["decision"]
        decisions[decision] = decisions.get(decision, 0) + 1
        kind = record["kind"]
        kinds[kind] = kinds.get(kind, 0) + 1
        if record["at"] > last_at:
            last_at = record["at"]
    return {
        "total": len(ledger),
        "decisions": dict(sorted(decisions.items())),
        "kinds": dict(sorted(kinds.items())),
        "last_at": last_at,
        "weights_changed": False,
        "method": METHOD,
    }


def verify_ledger(ledger) -> dict:
    """Validate a decision ledger without mutating it.

    Content keys, unknown keys, malformed records and invalid decisions are
    reported as errors. The check never needs the note text to pass.
    """
    if not isinstance(ledger, list):
        return {
            "ok": False,
            "entries": 0,
            "errors": ["ledger must be a list"],
        }
    errors: list[str] = []
    for index, entry in enumerate(ledger):
        where = f"entry {index}"
        if not isinstance(entry, dict):
            errors.append(f"{where}: must be a mapping")
            continue
        forbidden = sorted(set(entry) & FORBIDDEN_KEYS)
        if forbidden:
            errors.append(
                f"{where}: content key(s) not allowed: "
                + ", ".join(forbidden))
        unknown = sorted(set(entry) - RECORD_KEYS)
        if unknown:
            errors.append(
                f"{where}: unknown key(s): " + ", ".join(unknown))
        version = entry.get("version")
        if isinstance(version, bool) or version != VERSION:
            errors.append(f"{where}: version must be {VERSION}")
        if not isinstance(entry.get("at"), str) or not entry["at"].strip():
            errors.append(f"{where}: at must be a non-empty string")
        for key in ("action_id", "kind"):
            value = entry.get(key)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"{where}: {key} must be a non-empty string")
        decision = entry.get("decision")
        if (not isinstance(decision, str)
                or decision not in VALID_DECISIONS):
            errors.append(f"{where}: invalid decision")
        note_id = entry.get("note_id")
        if not isinstance(note_id, str) or not _NOTE_ID_RE.fullmatch(note_id):
            errors.append(
                f"{where}: note_id must be empty or a lowercase hex prefix")
        present_gate = set(entry) & GATE_KEYS
        if present_gate:
            if present_gate != GATE_KEYS:
                errors.append(
                    f"{where}: gate fields must be present together")
            else:
                if not isinstance(entry["gate_ok"], bool):
                    errors.append(f"{where}: gate_ok must be a boolean")
                for key in ("gate_total", "gate_passed"):
                    value = entry[key]
                    if (isinstance(value, bool)
                            or not isinstance(value, int) or value < 0):
                        errors.append(
                            f"{where}: {key} must be a non-negative integer")
                if (not isinstance(entry["gate_method"], str)
                        or not entry["gate_method"].strip()):
                    errors.append(
                        f"{where}: gate_method must be a non-empty string")
                if (entry.get("decision") == "approve"
                        and entry.get("gate_ok") is not True):
                    errors.append(
                        f"{where}: approval requires a passing gate")
    return {
        "ok": not errors,
        "entries": len(ledger),
        "errors": errors,
    }


def _read_json(path: str):
    if path == "-":
        raw = sys.stdin.read()
    else:
        raw = Path(path).read_text(encoding="utf-8-sig")
    if not raw.strip():
        raise ValueError(f"empty JSON input: {path}")
    return json.loads(raw)


def _payload_from_args(args) -> dict:
    payload = {}
    any_file = False
    stdin_used = False
    for dest, key in (
            ("claims_json", "claim_audit"),
            ("gaps_json", "gaps"),
            ("feedback_json", "feedback"),
            ("freshness_json", "freshness"),
            ("reliability_json", "reliability"),
            ("aisc_coverage_json", "aisc_coverage"),
            ("scoreboard_json", "scoreboard"),
            ("drift_json", "drift"),
            ("runner_audit_json", "runner_audit")):
        path = getattr(args, dest)
        if not path:
            continue
        any_file = True
        if path == "-":
            if stdin_used:
                raise ValueError("stdin can only be read once")
            stdin_used = True
        payload[key] = _read_json(path)
    if any_file:
        return payload

    raw = sys.stdin.read()
    if not raw.strip():
        raise ValueError("no input payload; pass a JSON file or stdin")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("stdin payload must be a JSON object")
    if not data:
        raise ValueError("empty input payload")
    for raw_key, value in data.items():
        key = _STDIN_ALIASES.get(str(raw_key).strip().lower())
        if key is None:
            raise ValueError(f"unknown input key: {raw_key}")
        payload[key] = value
    return payload


def _render_plan(plan: dict) -> str:
    lines = [
        f"Improvement plan ({plan['method']})",
        f"actions: {len(plan['actions'])}",
        "auto_apply: false",
        "weights_changed: false",
    ]
    if plan["summary"]:
        summary = ", ".join(
            f"{kind}={count}" for kind, count in plan["summary"].items())
        lines.append(f"summary: {summary}")
    if not plan["actions"]:
        lines.append("No improvement actions: no positive weakness signals.")
        return "\n".join(lines)
    for action in plan["actions"]:
        lines.append(
            f"{action['action_id']} [{action['kind']}] priority="
            f"{action['priority']:g}")
        lines.append(f"  {action['title']}")
        lines.append(f"  {action['detail']}")
        lines.append("  requires human approval: true")
        lines.append(f"  acceptance: {action['acceptance']}")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claims-json", default="",
                        help="JSON file with a content-free claim audit")
    parser.add_argument("--gaps-json", default="",
                        help="JSON file with a list of knowledge gaps")
    parser.add_argument("--feedback-json", default="",
                        help="JSON file with a list of feedback items")
    parser.add_argument("--freshness-json", default="",
                        help="JSON file with a freshness/consensus summary")
    parser.add_argument("--reliability-json", default="",
                        help="JSON file with a reliability summary")
    parser.add_argument("--aisc-coverage-json", default="",
                        help="JSON file with an AISC coverage report")
    parser.add_argument("--scoreboard-json", default="",
                        help="JSON file with an atomzero-scoreboard summary")
    parser.add_argument("--drift-json", default="",
                        help="JSON file with an atomzero-drift report")
    parser.add_argument("--runner-audit-json", default="",
                        help="JSON file with a runner scope audit")
    parser.add_argument("--limit", type=int, default=5,
                        help="maximum number of actions (1..20)")
    parser.add_argument("--json", action="store_true",
                        help="print the plan as JSON")
    args = parser.parse_args(argv)
    try:
        payload = _payload_from_args(args)
        plan = plan_from_inputs(limit=args.limit, **payload)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
    else:
        print(_render_plan(plan))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
