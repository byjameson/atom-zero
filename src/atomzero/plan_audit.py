"""Audit constrained tool plans before execution.

The executor already rejects unknown tools and requires confirmation for side
effects. This module adds a formal, machine-readable audit layer: it reports
side effects, irreversible steps, confirmation requirements and a risk level.
It is the Atom Zero side of the ``AtomZeroToolPlan`` AISC architecture spec.

    python -m atomzero.plan_audit --file plan.json
    python -m atomzero.plan_audit --steps '[{"tool": "time"}]' --json
"""
from __future__ import annotations

import argparse
import json
import sys

from atomzero.tool_planner import (
    MAX_STEPS, TOOL_SPECS, ToolPlanError, validate_plan)

REVERSIBLE = {
    "time": True,
    "calculator": True,
    "search_docs": True,
    "evidence": True,
    "task_list": True,
    "task_add": False,
    "task_complete": False,
    "research": False,
}


def _issue(code: str, severity: str, message: str) -> dict:
    return {"code": code, "severity": severity, "message": message}


def audit_plan(steps, confirm: bool = False) -> dict:
    """Return a structured audit for a plan without executing it."""
    confirm = bool(confirm)
    try:
        plan = validate_plan(steps)
    except ToolPlanError as exc:
        return {
            "ok": False,
            "allowed": False,
            "valid": False,
            "confirm": confirm,
            "risk": "blocked",
            "requires_confirmation": False,
            "issues": [_issue("invalid_plan", "high", str(exc))],
            "plan": [],
            "side_effects": [],
            "irreversible": [],
            "guarantees": [],
        }
    preview = [{
        "tool": step["tool"],
        "args": step["args"],
        "side_effect": bool(TOOL_SPECS[step["tool"]]["side_effect"]),
        "reversible": REVERSIBLE.get(step["tool"], False),
    } for step in plan]
    side_effects = [step for step in preview if step["side_effect"]]
    irreversible = [step for step in side_effects if not step["reversible"]]
    issues = []
    if side_effects and not confirm:
        issues.append(_issue(
            "confirmation_required", "high",
            f"{len(side_effects)} side-effect step(s) require explicit "
            "confirmation"))
    if irreversible and not confirm:
        issues.append(_issue(
            "irreversible_unconfirmed", "high",
            f"{len(irreversible)} irreversible step(s) are not confirmed"))
    if len(plan) > MAX_STEPS:
        issues.append(_issue(
            "step_budget_exceeded", "high",
            f"plan has {len(plan)} steps; maximum is {MAX_STEPS}"))
    risk = ("blocked" if not plan
            else "high" if irreversible
            else "medium" if side_effects
            else "low")
    allowed = bool(plan) and (not side_effects or confirm)
    guarantees = [
        "only allowlisted tools and arguments are accepted",
        f"at most {MAX_STEPS} steps",
    ]
    if side_effects:
        guarantees.append("side effects require explicit confirmation")
    if irreversible:
        guarantees.append("irreversible steps are reported before execution")
    return {
        "ok": True,
        "allowed": allowed,
        "valid": True,
        "confirm": confirm,
        "risk": risk,
        "requires_confirmation": bool(side_effects and not confirm),
        "issues": issues,
        "plan": preview,
        "side_effects": side_effects,
        "irreversible": irreversible,
        "guarantees": guarantees,
    }


def format_audit(result: dict) -> str:
    lines = [
        f"Tool plan audit: {'ALLOWED' if result['allowed'] else 'BLOCKED'}",
        f"risk: {result['risk']}",
        f"steps: {len(result['plan'])}",
        f"side effects: {len(result['side_effects'])}",
        f"irreversible: {len(result['irreversible'])}",
    ]
    if result["issues"]:
        lines.append("issues:")
        for issue in result["issues"]:
            lines.append(
                f"- {issue['code']} ({issue['severity']}): "
                f"{issue['message']}")
    if result["plan"]:
        lines.append("steps:")
        for index, step in enumerate(result["plan"], start=1):
            lines.append(
                f"{index}. {step['tool']} side_effect="
                f"{step['side_effect']} reversible={step['reversible']}")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", default="",
                        help="JSON file containing a list of steps")
    parser.add_argument("--steps", default="",
                        help="JSON list of steps")
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.file:
        raw = json.loads(open(args.file, encoding="utf-8").read())
    elif args.steps:
        raw = json.loads(args.steps)
    else:
        raw = json.loads(sys.stdin.read() or "[]")
    result = audit_plan(raw, confirm=args.confirm)
    print(json.dumps(result, indent=2) if args.json else format_audit(result))
    return 0 if result["allowed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
