"""Opt-in runner for agent-bridge packets.

The bridge produces a bounded packet; this runner can prepare an isolated git
worktree, optionally invoke a configured external agent command, verify that
only allowed files changed, and run the packet's allowlisted checks. It never
merges, pushes, tags or publishes, and it does not invent an agent: without an
explicit opt-in and a runner command it only prepares the worktree.

    atomzero-agent-run prepare --packet packet.json --opt-in --repo .
    atomzero-agent-run verify --packet packet.json --worktree PATH --json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

from atomzero.evaluation import evaluate_gate

__all__ = ["RunnerError", "prepare", "run_packet", "verify"]

_BRANCH_RE = re.compile(r"\A[A-Za-z0-9._/-]{1,80}\Z")


class RunnerError(ValueError):
    pass


def _packet(packet: dict) -> dict:
    if not isinstance(packet, dict):
        raise RunnerError("packet must be an object")
    if packet.get("requires_human") is not True:
        raise RunnerError("packet is not human-gated")
    if packet.get("auto_start") is not False:
        raise RunnerError("packet must have auto_start false")
    if packet.get("handoff") != "external_agent_harness_only":
        raise RunnerError("packet handoff must be external_agent_harness_only")
    action_id = str(packet.get("action_id") or "").strip()
    if not action_id:
        raise RunnerError("packet.action_id is required")
    files = packet.get("allowed_files")
    checks = packet.get("checks")
    if not isinstance(files, list) or not files:
        raise RunnerError("packet.allowed_files must be a non-empty list")
    if not isinstance(checks, list) or not checks:
        raise RunnerError("packet.checks must be a non-empty list")
    for value in files:
        path = Path(str(value))
        if path.is_absolute() or ".." in path.parts or not str(path).strip():
            raise RunnerError(f"allowed file is not repository-relative: {value}")
    return packet


def _branch(packet: dict, explicit: str | None) -> str:
    if explicit:
        branch = explicit.strip()
    else:
        action = re.sub(r"[^A-Za-z0-9._-]+", "-", packet["action_id"])
        branch = f"agent/{action}"
    if not _BRANCH_RE.fullmatch(branch):
        raise RunnerError("invalid branch name")
    return branch


def _git(args, cwd, timeout=120):
    completed = subprocess.run(
        ["git", *args], cwd=str(cwd), capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=timeout, check=False,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})
    return completed


def prepare(packet: dict, *, repo_root, base: str = "main",
            worktree_root=None, branch: str | None = None) -> dict:
    """Create an isolated worktree and write the packet inside it."""
    packet = _packet(packet)
    repo = Path(repo_root).expanduser().resolve()
    if not (repo / ".git").exists():
        raise RunnerError(f"not a git repository: {repo}")
    branch = _branch(packet, branch)
    if worktree_root is None:
        worktree = repo.parent / f"atom-zero-{branch.replace('/', '-')}"
    else:
        worktree = Path(worktree_root).expanduser().resolve()
    if worktree.exists():
        raise RunnerError(f"worktree already exists: {worktree}")
    completed = _git(["worktree", "add", "-b", branch, str(worktree), base],
                     cwd=repo)
    if completed.returncode != 0:
        raise RunnerError(
            "git worktree add failed: " + (completed.stderr or "").strip())
    packet_path = worktree / ".agent-task.json"
    packet_path.write_text(
        json.dumps(packet, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "status": "prepared",
        "worktree": str(worktree),
        "branch": branch,
        "packet_path": str(packet_path),
        "auto_applied": False,
    }


def _changed_files(worktree: Path, base: str = "main") -> list[str]:
    files = set()
    committed = _git(["diff", "--name-only", f"{base}...HEAD"], cwd=worktree)
    if committed.returncode != 0:
        raise RunnerError(
            "git diff failed: " + (committed.stderr or "").strip())
    for line in committed.stdout.splitlines():
        path = line.strip().replace("\\", "/")
        if path and path != ".agent-task.json":
            files.add(path)
    completed = _git(["status", "--porcelain"], cwd=worktree)
    if completed.returncode != 0:
        raise RunnerError(
            "git status failed: " + (completed.stderr or "").strip())
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        path = line[3:].strip()
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        normalized = path.replace("\\", "/")
        if normalized == ".agent-task.json":
            continue
        files.add(normalized)
    return sorted(files)


def verify(packet: dict, *, worktree, base: str = "main",
           aisc_path: str = "", archive=None, timeout: int = 300) -> dict:
    """Check scope and run the packet's allowlisted gate."""
    packet = _packet(packet)
    path = Path(worktree).expanduser().resolve()
    if not path.is_dir():
        raise RunnerError(f"worktree not found: {path}")
    changed = _changed_files(path, base)
    allowed = {str(value).replace("\\", "/") for value in packet["allowed_files"]}
    out_of_scope = [name for name in changed if name not in allowed]
    checks = [{"name": item.split(":", 1)[-1], "kind": item.split(":", 1)[0],
               "target": item.split(":", 1)[-1], "check": item}
              for item in packet["checks"]]
    # evaluate_gate expects kind/target, not the combined check string.
    normalized = []
    for item in checks:
        normalized.append({"name": item["name"], "kind": item["kind"],
                           "target": item["target"]})
    try:
        gate = evaluate_gate(
            normalized, project_root=path, aisc_path=aisc_path, timeout=timeout)
    except Exception as exc:  # noqa: BLE001 - report a runner failure
        gate = {"ok": False, "checks": [],
                "summary": {"total": 0, "passed": 0, "failed": 0,
                            "manual_required": 0},
                "error": f"{type(exc).__name__}: {exc}",
                "auto_apply": False, "weights_changed": False,
                "method": "allowlisted_evaluation_gate_v1"}
    if archive is not None:
        from atomzero.evaluation import archive_report, load_archive, write_archive
        entries = load_archive(archive)
        entry = archive_report(gate)
        entries.append(entry)
        write_archive(archive, entries)
        gate["gate_id"] = entry["gate_id"]
    ok = not out_of_scope and gate.get("ok") is True
    return {
        "status": "verified" if ok else "failed",
        "worktree": str(path),
        "changed_files": changed,
        "out_of_scope": out_of_scope,
        "gate": gate,
        "ok": ok,
        "auto_merge": False,
        "auto_apply": False,
        "weights_changed": False,
    }


def run_packet(packet: dict, *, opt_in: bool, repo_root, runner_command=None,
               base: str = "main", worktree_root=None, branch=None,
               aisc_path: str = "", archive=None, timeout: int = 1800) -> dict:
    """Prepare, optionally run an external command, then verify the packet."""
    if opt_in is not True:
        raise RunnerError("opt-in is required")
    prepared = prepare(
        packet, repo_root=repo_root, base=base,
        worktree_root=worktree_root, branch=branch)
    worktree = Path(prepared["worktree"])
    if runner_command:
        if isinstance(runner_command, str):
            command = shlex.split(runner_command)
        else:
            command = [str(item) for item in runner_command]
        if not command:
            raise RunnerError("runner command is empty")
        prompt = str(packet.get("goal") or packet.get("title") or "")
        command = [
            arg.format(worktree=str(worktree),
                       packet=str(prepared["packet_path"]),
                       prompt=prompt) for arg in command]
        completed = subprocess.run(
            command, cwd=str(worktree), capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout, check=False)
        prepared["runner_returncode"] = completed.returncode
        prepared["runner_output"] = (
            (completed.stdout or "") + (completed.stderr or ""))[-4000:]
        if completed.returncode != 0:
            prepared["status"] = "runner_failed"
            return prepared
    result = verify(packet, worktree=worktree, base=base, aisc_path=aisc_path,
                    archive=archive)
    result.update({
        "branch": prepared["branch"],
        "packet_path": prepared["packet_path"],
        "runner_returncode": prepared.get("runner_returncode"),
    })
    return result


def _load_packet(path: str) -> dict:
    raw = sys.stdin.read() if path == "-" else Path(path).read_text(
        encoding="utf-8-sig")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise RunnerError("packet must be a JSON object")
    return data


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "verify", "run"])
    parser.add_argument("--packet", required=True)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--worktree", default="")
    parser.add_argument("--worktree-root", default="")
    parser.add_argument("--base", default="main")
    parser.add_argument("--branch", default="")
    parser.add_argument("--runner", default="")
    parser.add_argument("--aisc-path", default=os.environ.get("AISC_PATH", ""))
    parser.add_argument("--archive", default="")
    parser.add_argument("--opt-in", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        packet = _load_packet(args.packet)
        if args.command == "prepare":
            if not args.opt_in:
                raise RunnerError("--opt-in is required for prepare")
            result = prepare(
                packet, repo_root=args.repo, base=args.base,
                worktree_root=args.worktree_root or None,
                branch=args.branch or None)
        elif args.command == "run":
            result = run_packet(
                packet, opt_in=args.opt_in, repo_root=args.repo,
                runner_command=args.runner or None, base=args.base,
                worktree_root=args.worktree_root or None,
                branch=args.branch or None, aisc_path=args.aisc_path,
                archive=args.archive or None)
        else:
            if not args.worktree:
                raise RunnerError("--worktree is required for verify")
            result = verify(
                packet, worktree=args.worktree, base=args.base,
                aisc_path=args.aisc_path, archive=args.archive or None)
    except (OSError, RunnerError, ValueError, json.JSONDecodeError) as exc:
        print(f"input error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("ok", result.get("status") == "prepared") else 1


if __name__ == "__main__":
    raise SystemExit(main())
