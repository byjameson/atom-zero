"""Build a safe handoff packet from an improvement action to an agent harness.

Atom Zero itself does not spawn coding agents, edit its own source or publish.
This module turns one human-gated improvement action into a structured task
packet that an external agent harness (for example OpenCode) can execute in an
isolated worktree. The packet is data only: nothing is started, downloaded or
applied here.

    atomzero-agent-bridge --plan plan.json --action-id imp-01-conflicts \
      --allowed-file src/atomzero/claims.py \
      --check pytest:tests/test_claims.py --json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

__all__ = ["build_task_packet", "render_markdown", "main"]

PACKET_VERSION = 1
DEFAULT_CONSTRAINTS = (
    "work in an isolated git worktree and branch",
    "edit only the allowed_files listed in this packet",
    "do not merge, push, tag or publish",
    "run the allowlisted checks and return the diff and test output",
    "stop if a secret, credential or out-of-scope file is required",
)
_KNOWN_CHECK_PREFIXES = ("pytest:", "aisc:", "manual:")


def _strings(value, name: str, *, allow_empty: bool = False) -> list[str]:
    if value is None:
        value = []
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    result = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{name}[{index}] must be a non-empty string")
        result.append(item.strip())
    if not allow_empty and not result:
        raise ValueError(f"{name} must not be empty")
    return result


def build_task_packet(
    action: dict,
    *,
    allowed_files,
    checks,
    repository: str = ".",
    role: str = "Implementer",
    constraints=None,
) -> dict:
    """Build a content-light task packet for an external coding agent."""
    if not isinstance(action, dict):
        raise ValueError("action must be an object")
    action_id = str(action.get("action_id") or "").strip()
    kind = str(action.get("kind") or "").strip()
    if not action_id or not kind:
        raise ValueError("action_id and kind are required")
    if action.get("requires_human") is not True:
        raise ValueError("action must require human approval")
    files = _strings(allowed_files, "allowed_files")
    check_list = _strings(checks, "checks")
    for check in check_list:
        if not check.startswith(_KNOWN_CHECK_PREFIXES):
            raise ValueError(f"unsupported check: {check}")
    constraint_list = (
        list(DEFAULT_CONSTRAINTS) if constraints is None
        else _strings(constraints, "constraints"))
    title = str(action.get("title") or kind.replace("_", " ").capitalize())
    detail = str(action.get("detail") or "").strip()
    acceptance = str(action.get("acceptance") or "").strip()
    if not acceptance:
        raise ValueError("action.acceptance is required")
    return {
        "packet_version": PACKET_VERSION,
        "role": str(role or "Implementer"),
        "repository": str(repository or "."),
        "action_id": action_id,
        "kind": kind,
        "title": title,
        "goal": detail or title,
        "acceptance": acceptance,
        "allowed_files": files,
        "checks": check_list,
        "constraints": constraint_list,
        "requires_human": True,
        "auto_start": False,
        "handoff": "external_agent_harness_only",
        "weights_changed": False,
        "method": "deterministic_agent_task_packet_v1",
    }


def render_markdown(packet: dict) -> str:
    """Render a packet as a human-readable task brief."""
    lines = [
        f"# Agent task packet - {packet['action_id']}",
        "",
        f"- Role: {packet['role']}",
        f"- Repository: {packet['repository']}",
        f"- Action: {packet['kind']} - {packet['title']}",
        f"- Goal: {packet['goal']}",
        "",
        "## Allowed files",
        "",
    ]
    lines.extend(f"- `{path}`" for path in packet["allowed_files"])
    lines.extend(["", "## Checks", ""])
    lines.extend(f"- `{check}`" for check in packet["checks"])
    lines.extend(["", "## Acceptance", "", packet["acceptance"], "",
                  "## Constraints", ""])
    lines.extend(f"- {constraint}" for constraint in packet["constraints"])
    lines.extend([
        "",
        "This packet is data only. The external harness must start the worktree,",
        "run the checks and return the diff. Atom Zero does not spawn agents or",
        "publish from this module.",
        "",
    ])
    return "\n".join(lines)


def _load_plan(path: str) -> dict:
    raw = sys.stdin.read() if path == "-" else Path(path).read_text(
        encoding="utf-8-sig")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("plan must be a JSON object")
    return data


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, help="plan JSON path or -")
    parser.add_argument("--action-id", default="")
    parser.add_argument("--allowed-file", action="append", default=[])
    parser.add_argument("--check", action="append", default=[])
    parser.add_argument("--repository", default=".")
    parser.add_argument("--role", default="Implementer")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        plan = _load_plan(args.plan)
        actions = plan.get("actions")
        if not isinstance(actions, list) or not actions:
            raise ValueError("plan.actions must be a non-empty list")
        action = actions[0]
        if args.action_id:
            action = next(
                (item for item in actions
                 if isinstance(item, dict)
                 and item.get("action_id") == args.action_id), None)
            if action is None:
                raise ValueError(f"action not found: {args.action_id}")
        packet = build_task_packet(
            action, allowed_files=args.allowed_file, checks=args.check,
            repository=args.repository, role=args.role)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"input error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(packet, ensure_ascii=False, indent=2))
    else:
        print(render_markdown(packet))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
