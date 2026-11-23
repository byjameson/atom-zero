"""Opt-in bounded self-improvement loop.

Composes the existing pieces in one explicit flow:

    live signals -> planner -> bridge packet -> opt-in runner -> gate
    -> human approval/veto -> gated decision -> scoreboard -> trend

The loop never merges, pushes or edits source itself. It executes an external
runner command only when both `opt_in` and `approve` are true, and it records a
decision only after the gate passed.

    atomzero-self-improve --signal signals.json --opt-in --approve \
      --runner opencode --allowed-file src/... --check pytest:tests/... \
      --ledger data/improvement_ledger.jsonl \
      --scoreboard data/scoreboard.jsonl --after-tests 839 --json

    atomzero-self-improve --self-code-root . \
      --target-path src/atomzero/agent_runner.py --opt-in --approve --json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from atomzero.agent_bridge import build_task_packet
from atomzero.agent_runner import RunnerError, run_packet
from atomzero.improvement import plan_from_inputs, record_gated_decision
from atomzero.scoreboard import (
    load_jsonl as load_scoreboard,
    record_run,
    write_jsonl as write_scoreboard,
)
from atomzero.trend import _load_ledger, build_trend

__all__ = ["run_loop", "main"]

_SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}


def _normalize_repo_path(path) -> str:
    """Return a repo-relative path with POSIX separators."""
    text = str(path or "").strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return text


def _filter_self_code_target(report: dict, target: str) -> dict:
    """Keep only the findings and proposals for one repo-relative path."""
    findings = [
        item for item in (report.get("findings") or [])
        if isinstance(item, dict)
        and _normalize_repo_path(item.get("path")) == target
    ]
    proposals = [
        item for item in (report.get("proposals") or [])
        if isinstance(item, dict)
        and _normalize_repo_path(item.get("path")) == target
    ]
    if not findings or not proposals:
        raise RunnerError(
            "no self-code finding or proposal matches target path: "
            f"{target}")
    findings.sort(key=lambda item: (
        _SEVERITY_ORDER.get(str(item.get("severity") or "").lower(), 3),
        int(item.get("line") or 0),
        str(item.get("kind") or "")))
    proposals.sort(key=lambda item: (
        _SEVERITY_ORDER.get(str(item.get("severity") or "").lower(), 3),
        str(item.get("proposal_id") or "")))
    filtered = dict(report)
    filtered["findings"] = findings
    filtered["proposals"] = proposals
    counts = {}
    severities = {}
    for item in findings:
        kind = str(item.get("kind") or "")
        counts[kind] = counts.get(kind, 0) + 1
        severity = str(item.get("severity") or "").strip().lower()
        severities[severity] = severities.get(severity, 0) + 1
    summary = dict(report.get("summary") or {})
    summary.update({
        "findings": len(findings),
        "high": severities.get("high", 0),
        "medium": severities.get("medium", 0),
        "low": severities.get("low", 0),
        "proposals": len(proposals),
    })
    filtered["counts"] = dict(sorted(counts.items()))
    filtered["summary"] = summary
    return filtered


def run_loop(
    *,
    signals: dict,
    allowed_files,
    checks,
    opt_in: bool,
    approve: bool,
    veto: bool = False,
    repo_root=".",
    runner_command=None,
    base: str = "main",
    worktree_root=None,
    ledger_path=None,
    scoreboard_path=None,
    after: dict | None = None,
    aisc_path: str = "",
    archive=None,
    timeout: int = 1800,
    self_code_root=None,
    target_path: str | None = None,
) -> dict:
    """Run one bounded loop, returning every stage and its evidence."""
    normalized = dict(signals) if isinstance(signals, dict) else {}
    before_self_code = None
    if self_code_root:
        from atomzero.self_code import audit_report
        before_self_code = audit_report(self_code_root, propose=True)
        report = before_self_code
        previous = normalized.get("self_code")
        if isinstance(previous, dict):
            seen = {item.get("finding_id") for item in report["findings"]}
            for item in previous.get("findings") or []:
                if isinstance(item, dict) and item.get("finding_id") not in seen:
                    report["findings"].append(item)
            report["summary"]["findings"] = len(report["findings"])
        normalized["self_code"] = report
    target = ""
    target_severity = ""
    if target_path:
        target = _normalize_repo_path(target_path)
        if not target:
            raise RunnerError(
                "target_path must be a non-empty repo-relative path")
        report = normalized.get("self_code")
        if not isinstance(report, dict):
            raise RunnerError(
                "target_path requires a self-code report; pass "
                "self_code_root or a self_code signal")
        normalized["self_code"] = _filter_self_code_target(report, target)
        target_severity = str(
            normalized["self_code"]["findings"][0].get("severity") or ""
        ).strip().lower()
        if target_severity not in {"high", "medium"}:
            raise RunnerError(
                "target finding severity has no planner action: "
                f"{target_severity or 'unknown'}")
    if not normalized:
        raise RunnerError("signals or self_code_root must provide an input")
    plan = plan_from_inputs(**normalized)
    if not plan["actions"]:
        if target:
            raise RunnerError(
                "planner produced no action for target severity: "
                f"{target_severity}")
        return {
            "status": "no_action",
            "plan": plan,
            "auto_apply": False,
            "weights_changed": False,
        }
    if target:
        action = next(
            (item for item in plan["actions"]
             if item.get("kind") == f"self_code_{target_severity}"), None)
        if action is None:
            raise RunnerError(
                "planner produced no action for target severity: "
                f"{target_severity}")
    else:
        action = plan["actions"][0]
    if target:
        proposal = next(
            (item for item in normalized["self_code"]["proposals"]
             if _normalize_repo_path(item.get("path")) == target
             and str(item.get("severity") or "").strip().lower()
             == target_severity),
            None)
        if proposal is None:
            raise RunnerError(
                "no self-code proposal matches target path and severity: "
                f"{target} ({target_severity})")
        allowed_files = list(proposal.get("allowed_files") or [])
        checks = list(proposal.get("checks") or action["suggested_checks"])
    elif not allowed_files or not checks:
        for proposal in (normalized.get("self_code") or {}).get(
                "proposals", []):
            if not isinstance(proposal, dict):
                continue
            wanted = action["kind"] == f"self_code_{proposal.get('severity')}"
            if not wanted:
                continue
            if not allowed_files:
                allowed_files = proposal.get("allowed_files") or []
            if not checks:
                checks = proposal.get("checks") or action["suggested_checks"]
            break
    packet = build_task_packet(
        action, allowed_files=allowed_files, checks=checks,
        repository=str(repo_root))
    result = {
        "status": "awaiting_human",
        "plan": plan,
        "packet": packet,
        "auto_apply": False,
        "weights_changed": False,
    }
    if veto:
        result["status"] = "vetoed"
        return result
    if not opt_in or not approve:
        return result
    runner = run_packet(
        packet, opt_in=True, repo_root=repo_root,
        runner_command=runner_command, base=base,
        worktree_root=worktree_root, aisc_path=aisc_path,
        archive=archive, timeout=timeout)
    result["runner"] = runner
    if runner.get("ok") is not True:
        result["status"] = "gate_failed"
        return result
    if before_self_code is not None:
        from atomzero.self_code import audit_report
        from atomzero.self_code_diff import compare_reports
        after_self_code = audit_report(runner["worktree"], propose=False)
        diff = compare_reports(before_self_code, after_self_code)
        result["self_code_diff"] = diff
        if diff["ok"] is not True:
            result["status"] = "self_code_regression"
            return result
    gate = runner["gate"]
    ledger = _load_ledger(ledger_path) if ledger_path else []
    record = record_gated_decision(
        ledger, action, "approve", gate,
        note=(f"self-improve loop: {action['action_id']} gate "
              f"{gate.get('gate_id', 'no-gate')} passed and human approved"))
    if ledger_path:
        path = Path(ledger_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(item, ensure_ascii=False, sort_keys=True,
                               separators=(",", ":")) + "\n"
                    for item in ledger), encoding="utf-8")
    result["decision"] = record
    result["ledger"] = ledger
    if after is not None and scoreboard_path:
        runs = load_scoreboard(scoreboard_path)
        record_run(runs, **after)
        write_scoreboard(scoreboard_path, runs)
        result["scoreboard"] = runs
        result["trend"] = build_trend(ledger, runs)
    result["status"] = "completed"
    return result


def _load_signals(path: str) -> dict:
    raw = sys.stdin.read() if path == "-" else Path(path).read_text(
        encoding="utf-8-sig")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise RunnerError("signals must be a JSON object")
    return data


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signal", default="")
    parser.add_argument("--self-code-root", default="")
    parser.add_argument(
        "--target-path", default="",
        help="limit the self-code loop to one repo-relative file")
    parser.add_argument("--allowed-file", action="append", default=[])
    parser.add_argument("--check", action="append", default=[])
    parser.add_argument("--opt-in", action="store_true")
    parser.add_argument("--approve", action="store_true")
    parser.add_argument("--veto", action="store_true")
    parser.add_argument("--repo", default=".")
    parser.add_argument("--runner", default="")
    parser.add_argument("--base", default="main")
    parser.add_argument("--worktree-root", default="")
    parser.add_argument("--ledger", default="")
    parser.add_argument("--scoreboard", default="")
    parser.add_argument("--after-commit", default="")
    parser.add_argument("--after-tests", type=int, default=0)
    parser.add_argument("--after-failed", type=int, default=0)
    parser.add_argument("--after-benchmark", type=int, default=0)
    parser.add_argument("--after-benchmark-total", type=int, default=0)
    parser.add_argument("--after-aisc", type=int, default=0)
    parser.add_argument("--after-aisc-total", type=int, default=0)
    parser.add_argument("--aisc-path", default=os.environ.get("AISC_PATH", ""))
    parser.add_argument("--archive", default="")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        signals = _load_signals(args.signal) if args.signal else {}
        if not signals and not args.self_code_root:
            raise RunnerError("--signal or --self-code-root is required")
        after = None
        if args.after_commit:
            after = {
                "commit": args.after_commit,
                "tests_passed": args.after_tests,
                "tests_failed": args.after_failed,
                "benchmark_passed": args.after_benchmark,
                "benchmark_total": args.after_benchmark_total,
                "aisc_passed": args.after_aisc,
                "aisc_total": args.after_aisc_total,
            }
        result = run_loop(
            signals=signals, allowed_files=args.allowed_file,
            checks=args.check, opt_in=args.opt_in, approve=args.approve,
            veto=args.veto, repo_root=args.repo,
            runner_command=args.runner or None, base=args.base,
            worktree_root=args.worktree_root or None,
            ledger_path=args.ledger or None,
            scoreboard_path=args.scoreboard or None,
            after=after, aisc_path=args.aisc_path,
            archive=args.archive or None,
            self_code_root=args.self_code_root or None,
            target_path=args.target_path or None)
    except (OSError, ValueError, RunnerError, json.JSONDecodeError) as exc:
        print(f"input error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
