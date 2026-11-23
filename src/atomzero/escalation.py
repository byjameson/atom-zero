"""Deterministic escalation ladder for Atom Zero.

The router is the first step of the snowball ladder, not a trainer. It decides
whether a bounded task should stay local, go to a configured teacher or be
handed to an opt-in agent. It never calls a network, never starts an agent,
never changes weights and never applies anything.

Output is content-free and deterministic. A small growth ledger can be built
from those decisions; only a verified, successful teacher answer becomes a
``verified_teacher`` event. That event is an adapter *candidate*, not a
training run.

Routing order is fixed:

1. secret privacy -> refuse;
2. code task with an enabled agent and budget -> human-gated agent handoff;
3. confident local answer without a freshness requirement -> local;
4. public task with an enabled teacher, external access and budget -> teacher;
5. otherwise refuse.

CLI::

    atomzero-escalate --task-json task.json --policy-json policy.json --json
    atomzero-escalate --summary-json growth_ledger.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

from atomzero import __version__

__all__ = [
    "METHOD",
    "GROWTH_METHOD",
    "ROUTES",
    "REASON_CODES",
    "decide",
    "record_escalation",
    "summarize_growth",
    "main",
]

METHOD = "deterministic_escalation_router_v1"
GROWTH_METHOD = "deterministic_escalation_growth_v1"
VERSION = 1

LOCAL = "local"
TEACHER = "teacher"
AGENT = "agent"
REFUSE = "refuse"
ROUTES = frozenset({LOCAL, TEACHER, AGENT, REFUSE})
KINDS = frozenset({"answer", "code", "research"})
PRIVACIES = frozenset({"public", "private", "secret"})
COMPLEXITIES = frozenset({"low", "medium", "high"})
OUTCOMES = frozenset({"success", "failure", "unknown"})
VERIFIED_TEACHER = "verified_teacher"

PRIVACY_SECRET = "privacy_secret"
BOUNDED_CODE_HANDOFF = "bounded_code_handoff"
LOCAL_CONFIDENT = "local_confident"
LOW_CONFIDENCE_OR_FRESHNESS = "low_confidence_or_freshness"
NO_AVAILABLE_ROUTE = "no_available_route"
PRIVACY_PRIVATE = "privacy_private"
REASON_CODES = frozenset({
    PRIVACY_SECRET,
    BOUNDED_CODE_HANDOFF,
    LOCAL_CONFIDENT,
    LOW_CONFIDENCE_OR_FRESHNESS,
    NO_AVAILABLE_ROUTE,
    PRIVACY_PRIVATE,
})
ROUTE_REASONS = {
    LOCAL: frozenset({LOCAL_CONFIDENT}),
    TEACHER: frozenset({LOW_CONFIDENCE_OR_FRESHNESS}),
    AGENT: frozenset({BOUNDED_CODE_HANDOFF}),
    REFUSE: frozenset({PRIVACY_SECRET, NO_AVAILABLE_ROUTE, PRIVACY_PRIVATE}),
}

DEFAULT_POLICY = {
    "teacher_enabled": False,
    "agent_enabled": False,
    "allow_external": False,
    "min_local_confidence": 0.55,
    "max_external_calls": 0,
    "max_agent_calls": 0,
}
POLICY_KEYS = frozenset(DEFAULT_POLICY)
TASK_KEYS = frozenset({
    "kind", "confidence", "privacy", "requires_freshness", "complexity",
})
FORBIDDEN_KEYS = frozenset({
    "text", "query", "answer", "prompt", "content", "source", "note",
    "session",
})
EVENT_KEYS = frozenset({
    "version", "at", "timestamp", "route", "reason_code", "reason",
    "outcome", "verified", "growth_event", "method",
})

BUCKET_MEDIUM = 0.5
BUCKET_HIGH = 0.8


def _as_mapping(value, name: str) -> dict:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    data = dict(value)
    for key in data:
        if not isinstance(key, str):
            raise ValueError(f"{name} keys must be strings")
    forbidden = sorted(set(data) & FORBIDDEN_KEYS)
    if forbidden:
        raise ValueError(
            f"{name} content key(s) not allowed: " + ", ".join(forbidden))
    return data


def _reject_unknown(data: dict, allowed, name: str) -> None:
    unknown = sorted(set(data) - set(allowed))
    if unknown:
        raise ValueError(
            f"{name} has unknown key(s): " + ", ".join(unknown))


def _unit_number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number between 0 and 1")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be a number between 0 and 1")
    return number


def _non_negative_int(value, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer")
    if isinstance(value, int):
        number = value
    elif (isinstance(value, float) and math.isfinite(value)
          and value.is_integer()):
        number = int(value)
    else:
        raise ValueError(f"{name} must be a non-negative integer")
    if number < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return number


def _boolean(value, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _choice(data: dict, key: str, allowed, name: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(
            f"{name} must be one of: " + ", ".join(sorted(allowed)))
    return value


def _task(task) -> dict:
    data = _as_mapping(task, "task")
    _reject_unknown(data, TASK_KEYS, "task")
    kind = _choice(data, "kind", KINDS, "task.kind")
    if "confidence" not in data:
        raise ValueError("task.confidence is required")
    confidence = _unit_number(data["confidence"], "task.confidence")
    privacy = _choice(data, "privacy", PRIVACIES, "task.privacy")
    requires_freshness = _boolean(
        data.get("requires_freshness", False), "task.requires_freshness")
    complexity = data.get("complexity")
    if complexity is not None:
        complexity = _choice(data, "complexity", COMPLEXITIES,
                             "task.complexity")
    return {
        "kind": kind,
        "confidence": confidence,
        "privacy": privacy,
        "requires_freshness": requires_freshness,
        "complexity": complexity,
    }


def _policy(policy) -> dict:
    data = _as_mapping(policy, "policy")
    _reject_unknown(data, POLICY_KEYS, "policy")
    result = dict(DEFAULT_POLICY)
    for key in ("teacher_enabled", "agent_enabled", "allow_external"):
        if key in data:
            result[key] = _boolean(data[key], f"policy.{key}")
    if "min_local_confidence" in data:
        result["min_local_confidence"] = _unit_number(
            data["min_local_confidence"], "policy.min_local_confidence")
    for key in ("max_external_calls", "max_agent_calls"):
        if key in data:
            result[key] = _non_negative_int(data[key], f"policy.{key}")
    return result


def _history_counts(history) -> tuple[int, int]:
    if history is None:
        return 0, 0
    if not isinstance(history, list):
        raise ValueError("history must be a list")
    external = 0
    agent = 0
    for index, entry in enumerate(history):
        data = _as_mapping(entry, f"history[{index}]")
        route = data.get("route")
        if not isinstance(route, str) or route not in ROUTES:
            raise ValueError(
                f"history[{index}].route must be one of: "
                + ", ".join(sorted(ROUTES)))
        if route == TEACHER:
            external += 1
        elif route == AGENT:
            agent += 1
    return external, agent


def _confidence_bucket(confidence: float) -> str:
    if confidence < BUCKET_MEDIUM:
        return "low"
    if confidence < BUCKET_HIGH:
        return "medium"
    return "high"


def decide(task, *, policy, history=None) -> dict:
    """Return a deterministic, content-free escalation decision.

    ``task`` carries only routing metadata, never text. ``policy`` is the
    operator configuration and ``history`` is an optional list of prior
    content-free escalation records used to enforce call budgets. The router
    itself performs no call and changes nothing.
    """
    task_data = _task(task)
    policy_data = _policy(policy)
    external_used, agent_used = _history_counts(history)
    budget = {
        "external_calls": external_used,
        "external_limit": policy_data["max_external_calls"],
        "external_remaining": max(
            0, policy_data["max_external_calls"] - external_used),
        "agent_calls": agent_used,
        "agent_limit": policy_data["max_agent_calls"],
        "agent_remaining": max(
            0, policy_data["max_agent_calls"] - agent_used),
    }

    kind = task_data["kind"]
    privacy = task_data["privacy"]
    confidence = task_data["confidence"]
    requires_human = False

    if privacy == "secret":
        route = REFUSE
        reason = PRIVACY_SECRET
    elif (kind == "code" and policy_data["agent_enabled"]
          and budget["agent_remaining"] > 0):
        route = AGENT
        reason = BOUNDED_CODE_HANDOFF
        requires_human = True
    elif (confidence >= policy_data["min_local_confidence"]
          and not task_data["requires_freshness"]):
        route = LOCAL
        reason = LOCAL_CONFIDENT
    elif (policy_data["teacher_enabled"]
          and policy_data["allow_external"]
          and budget["external_remaining"] > 0
          and privacy == "public"):
        route = TEACHER
        reason = LOW_CONFIDENCE_OR_FRESHNESS
    else:
        route = REFUSE
        if (privacy == "private" and policy_data["teacher_enabled"]
                and policy_data["allow_external"]
                and budget["external_remaining"] > 0):
            reason = PRIVACY_PRIVATE
        else:
            reason = NO_AVAILABLE_ROUTE

    return {
        "route": route,
        "reason_code": reason,
        "confidence_bucket": _confidence_bucket(confidence),
        "requires_human": requires_human,
        "external_call": False,
        "auto_apply": False,
        "weights_changed": False,
        "budget": budget,
        "method": METHOD,
    }


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


def _decision_fields(decision) -> tuple[str, str]:
    data = _as_mapping(decision, "decision")
    route = data.get("route")
    if not isinstance(route, str) or route not in ROUTES:
        raise ValueError(
            "decision.route must be one of: " + ", ".join(sorted(ROUTES)))
    reason = data.get("reason_code")
    if not isinstance(reason, str) or reason not in REASON_CODES:
        raise ValueError(
            "decision.reason_code must be one of: "
            + ", ".join(sorted(REASON_CODES)))
    if reason not in ROUTE_REASONS[route]:
        raise ValueError(
            f"decision.reason_code {reason!r} is not valid for route "
            f"{route!r}")
    for key in ("external_call", "auto_apply", "weights_changed"):
        if key in data and data[key] is not False:
            raise ValueError(
                f"decision.{key} must be false; the router changes nothing")
    return route, reason


def _event_fields(entry, index: int) -> dict:
    where = f"ledger[{index}]"
    data = _as_mapping(entry, where)
    _reject_unknown(data, EVENT_KEYS, where)

    version = data.get("version")
    if isinstance(version, bool) or version != VERSION:
        raise ValueError(f"{where}.version must be {VERSION}")

    at = data.get("at")
    timestamp = data.get("timestamp")
    if at is not None and timestamp is not None and at != timestamp:
        raise ValueError(f"{where}.at and timestamp must match")
    value = at if at is not None else timestamp
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where}.at must be a non-empty string")

    route = data.get("route")
    if not isinstance(route, str) or route not in ROUTES:
        raise ValueError(
            f"{where}.route must be one of: " + ", ".join(sorted(ROUTES)))

    reason_code = data.get("reason_code")
    reason = data.get("reason")
    if (reason_code is not None and reason is not None
            and reason_code != reason):
        raise ValueError(f"{where}.reason_code and reason must match")
    reason_value = reason_code if reason_code is not None else reason
    if not isinstance(reason_value, str) or reason_value not in REASON_CODES:
        raise ValueError(
            f"{where}.reason_code must be one of: "
            + ", ".join(sorted(REASON_CODES)))
    if reason_value not in ROUTE_REASONS[route]:
        raise ValueError(
            f"{where}.reason_code {reason_value!r} is not valid for route "
            f"{route!r}")

    outcome = data.get("outcome")
    if not isinstance(outcome, str) or outcome not in OUTCOMES:
        raise ValueError(
            f"{where}.outcome must be one of: "
            + ", ".join(sorted(OUTCOMES)))
    verified = data.get("verified")
    if not isinstance(verified, bool):
        raise ValueError(f"{where}.verified must be a boolean")

    expected = (
        VERIFIED_TEACHER
        if route == TEACHER and outcome == "success" and verified
        else ""
    )
    growth_event = data.get("growth_event")
    if growth_event is not None and growth_event not in ("", VERIFIED_TEACHER):
        raise ValueError(
            f"{where}.growth_event must be empty or {VERIFIED_TEACHER!r}")
    if growth_event is not None and growth_event != expected:
        raise ValueError(
            f"{where}.growth_event does not match route, outcome and verified")
    method = data.get("method")
    if method is not None and method != GROWTH_METHOD:
        raise ValueError(f"{where}.method must be {GROWTH_METHOD!r}")

    return {
        "at": value,
        "route": route,
        "reason_code": reason_value,
        "outcome": outcome,
        "verified": verified,
        "growth_event": expected,
    }


def _validated_events(ledger) -> list[dict]:
    if not isinstance(ledger, list):
        raise ValueError("ledger must be a list")
    return [_event_fields(entry, index)
            for index, entry in enumerate(ledger)]


def record_escalation(ledger, decision, *, outcome,
                      verified=False, now=None) -> dict:
    """Append a content-free growth event to ``ledger`` and return it.

    Only ``route == "teacher"`` with a successful, verified outcome is marked
    as a ``verified_teacher`` growth event. That is an adapter candidate, not a
    training run: nothing is distilled, downloaded or applied here.
    """
    if not isinstance(ledger, list):
        raise ValueError("ledger must be a list")
    route, reason = _decision_fields(decision)
    if not isinstance(outcome, str) or outcome not in OUTCOMES:
        raise ValueError(
            "outcome must be one of: " + ", ".join(sorted(OUTCOMES)))
    if not isinstance(verified, bool):
        raise ValueError("verified must be a boolean")
    _validated_events(ledger)

    growth_event = (
        VERIFIED_TEACHER
        if route == TEACHER and outcome == "success" and verified
        else ""
    )
    event = {
        "version": VERSION,
        "at": _format_timestamp(now),
        "route": route,
        "reason_code": reason,
        "outcome": outcome,
        "verified": verified,
        "growth_event": growth_event,
        "method": GROWTH_METHOD,
    }
    ledger.append(event)
    return event


def summarize_growth(ledger) -> dict:
    """Summarize a validated growth ledger without exposing task content.

    ``parameter_growth`` is always zero and its state is always ``unchanged``.
    The report counts routes, verified teacher answers, adapter candidates,
    external calls, agent handoffs and refusals.
    """
    events = _validated_events(ledger)
    routes = {route: 0 for route in (LOCAL, TEACHER, AGENT, REFUSE)}
    verified_teacher = 0
    last_at = ""
    for event in events:
        routes[event["route"]] += 1
        if event["growth_event"] == VERIFIED_TEACHER:
            verified_teacher += 1
        if event["at"] > last_at:
            last_at = event["at"]
    return {
        "total": len(events),
        "routes": routes,
        "verified_teacher_answers": verified_teacher,
        "adapter_candidates": verified_teacher,
        "external_calls": routes[TEACHER],
        "agent_handoffs": routes[AGENT],
        "refusals": routes[REFUSE],
        "parameter_growth": 0,
        "parameter_growth_state": "unchanged",
        "weights_changed": False,
        "auto_apply": False,
        "last_at": last_at,
        "method": GROWTH_METHOD,
    }


def _read_json(path: str, stdin_state: dict):
    if path == "-":
        if stdin_state["used"]:
            raise ValueError("stdin can only be read once")
        stdin_state["used"] = True
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


def _render_decision(decision: dict) -> str:
    budget = decision["budget"]
    return "\n".join([
        f"Escalation decision ({decision['method']})",
        f"route: {decision['route']}",
        f"reason: {decision['reason_code']}",
        f"confidence: {decision['confidence_bucket']}",
        f"requires_human: {str(decision['requires_human']).lower()}",
        "external_call: false",
        "auto_apply: false",
        "weights_changed: false",
        "budget: external {external_calls}/{external_limit} used, "
        "{external_remaining} left; agent {agent_calls}/{agent_limit} used, "
        "{agent_remaining} left".format(**budget),
    ])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="atomzero-escalate",
        description=("Deterministic, content-free escalation decision. "
                     "The router never calls a model or agent."))
    parser.add_argument("--version", action="version",
                        version=f"atomzero-escalate {__version__}")
    parser.add_argument("--task-json", default="",
                        help="JSON file with task routing metadata; - for stdin")
    parser.add_argument("--policy-json", default="",
                        help="JSON file with the escalation policy")
    parser.add_argument("--history-json", default="",
                        help="JSON file with prior content-free records")
    parser.add_argument("--summary-json", default="",
                        help="summarize a growth ledger instead of deciding")
    parser.add_argument("--json", action="store_true",
                        help="print the decision as JSON")
    args = parser.parse_args(argv)

    stdin_state = {"used": False}
    try:
        if args.summary_json:
            if args.task_json or args.policy_json or args.history_json:
                raise ValueError(
                    "--summary-json cannot be combined with decision inputs")
            ledger = _read_json(args.summary_json, stdin_state)
            summary = summarize_growth(ledger)
            print(json.dumps(summary, indent=2, ensure_ascii=False))
            return 0

        if not args.task_json:
            raise ValueError("--task-json is required")
        task = _read_json(args.task_json, stdin_state)
        policy = (
            _read_json(args.policy_json, stdin_state)
            if args.policy_json else {})
        history = (
            _read_json(args.history_json, stdin_state)
            if args.history_json else None)
        decision = decide(task, policy=policy, history=history)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(decision, indent=2, ensure_ascii=False))
    else:
        print(_render_decision(decision))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
