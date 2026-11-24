"""Opt-in, autonomous bounded self-improvement loop.

Composes the existing pieces in one explicit flow:

    live signals -> planner -> bridge packet -> opt-in runner -> focused gate
    -> self-code diff -> full sandboxed tests -> local merge
    -> post-merge tests -> gated decision -> scoreboard -> trend

The loop is autonomous by default. `--opt-in` enables execution; the focused
gate, self-code diff and full test suites replace per-change human approval.
`--require-approval` restores the human gate. The loop never pushes, tags or
publishes, and it rolls back a failed post-merge suite.

    atomzero-self-improve --signal signals.json --opt-in \
      --runner opencode --allowed-file src/... --check pytest:tests/... \
      --ledger data/improvement_ledger.jsonl \
      --scoreboard data/scoreboard.jsonl --after-tests 839 --json

    atomzero-self-improve --self-code-root . \
      --target-path src/atomzero/agent_runner.py --opt-in --json

    atomzero-self-improve --signal signals.json --opt-in \
      --adapter-curriculum-command "python research/run_staged_curriculum.py \
        --data-dir {data_dir} --out-dir {out_dir} --base {base} \
        --result {result} --final-eval {final_eval} \
        --promote --active-file {active_file}" \
      --adapter-curriculum-result research/results/staged_curriculum_run.json \
      --adapter-curriculum-out checkpoints/staged_curriculum \
      --adapter-curriculum-active-file data/curriculum_active_adapter.json \
      --adapter-curriculum-final-eval data/staged_copy_v1/mixed_eval.jsonl \
      --json
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

from atomzero.agent_bridge import build_task_packet
from atomzero.agent_runner import RunnerError, run_packet
from atomzero.improvement import plan_from_inputs, record_gated_decision
from atomzero.sandbox import backend_info, run_tests
from atomzero.scoreboard import (
    load_jsonl as load_scoreboard,
    record_run,
    write_jsonl as write_scoreboard,
)
from atomzero.trend import _load_ledger, build_trend

__all__ = ["run_loop", "main"]

_SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}
_AUDIT_MAX_PER_KIND = 200

# Defaults mirror research/run_staged_curriculum.py so a command template can
# use every placeholder even when only --adapter-curriculum-command is set.
_CURRICULUM_DEFAULT_DATA_DIR = "data/staged_copy_v1"
_CURRICULUM_DEFAULT_OUT_DIR = "checkpoints/staged_curriculum"
_CURRICULUM_DEFAULT_RESULT = "research/results/staged_curriculum_run.json"
_CURRICULUM_DEFAULT_ACTIVE_FILE = "data/curriculum_active_adapter.json"
_CURRICULUM_DEFAULT_FINAL_EVAL = "data/staged_copy_v1/mixed_eval.jsonl"
_CURRICULUM_GATE_STATUSES = {"pass", "fail", "skipped", "error"}
_CURRICULUM_PROMOTION_REASONS = {
    "promoted", "gate_failed", "gate_skipped", "dry_run",
    "not_requested", "promote_failed"}


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


def _git(args, cwd):
    return subprocess.run(
        ["git", *args], cwd=str(cwd), env=os.environ.copy(),
        capture_output=True, text=True, encoding="utf-8", errors="replace")


def _git_head(cwd) -> str:
    result = _git(["rev-parse", "--short", "HEAD"], cwd)
    return result.stdout.strip() if result.returncode == 0 else ""


def _git_dirty(cwd) -> bool:
    result = _git(["status", "--porcelain"], cwd)
    return result.returncode != 0 or bool(result.stdout.strip())


def _merge_branch(repo_root, branch: str, message: str) -> dict:
    before = _git_head(repo_root)
    merge = _git(["merge", "--no-ff", branch, "-m", message], repo_root)
    if merge.returncode != 0:
        _git(["merge", "--abort"], repo_root)
        return {
            "ok": False,
            "branch": branch,
            "before": before,
            "error": (merge.stderr or merge.stdout or "").strip()[-1000:],
        }
    return {
        "ok": True,
        "branch": branch,
        "before": before,
        "after": _git_head(repo_root),
    }


def _reset_hard(repo_root, commit: str) -> bool:
    if not commit:
        return False
    return _git(["reset", "--hard", commit], repo_root).returncode == 0


def _cleanup_worktree(repo_root, worktree, branch: str) -> dict:
    result = {"ok": True, "removed": False, "branch_deleted": False}
    if worktree and Path(worktree).exists():
        removed = _git(["worktree", "remove", str(worktree), "--force"],
                       repo_root)
        result["removed"] = removed.returncode == 0
        if removed.returncode != 0:
            result["ok"] = False
            result["error"] = (removed.stderr or "").strip()[-1000:]
    if branch:
        deleted = _git(["branch", "-D", branch], repo_root)
        result["branch_deleted"] = deleted.returncode == 0
    _git(["worktree", "prune"], repo_root)
    return result


def _metric_number(value):
    """Return a finite numeric metric, or None for anything else."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value


def _curriculum_command(template, values: dict) -> list[str]:
    """Expand {placeholders} in one shell-style curriculum command."""
    if isinstance(template, str):
        parts = shlex.split(template, posix=False)
    else:
        parts = [str(item) for item in template]
    if not parts:
        raise RunnerError("adapter curriculum command is empty")
    try:
        return [part.format(**values) for part in parts]
    except (KeyError, IndexError, ValueError) as exc:
        raise RunnerError(
            f"invalid adapter curriculum command template: {exc}") from exc


def _curriculum_env(repo_root) -> dict:
    """Expose the repo and src on PYTHONPATH for the runner subprocess."""
    root = Path(repo_root).resolve()
    env = os.environ.copy()
    parts = [str(root), str(root / "src")]
    existing = env.get("PYTHONPATH", "")
    if existing:
        parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return env


def _run_adapter_curriculum(
    *,
    command,
    repo_root,
    base=None,
    data_dir=None,
    out_dir=None,
    result_path=None,
    active_file=None,
    final_eval=None,
    min_exact_match=0.25,
    min_key_copy_rate=0.25,
    timeout: int = 900,
) -> dict:
    """Run the staged curriculum and return a content-free result block.

    The command template may use ``{base}``, ``{data_dir}``, ``{out_dir}``,
    ``{result}``, ``{active_file}`` and ``{final_eval}``. Only aggregate
    numbers and fixed status labels from the runner ledger are copied into the
    block; command output and ledger text fields are never recorded.
    """
    root = Path(repo_root).resolve()
    values = {
        "base": str(base or ""),
        "data_dir": str(data_dir or _CURRICULUM_DEFAULT_DATA_DIR),
        "out_dir": str(out_dir or _CURRICULUM_DEFAULT_OUT_DIR),
        "result": str(result_path or _CURRICULUM_DEFAULT_RESULT),
        "active_file": str(active_file or _CURRICULUM_DEFAULT_ACTIVE_FILE),
        "final_eval": str(final_eval or _CURRICULUM_DEFAULT_FINAL_EVAL),
    }
    target = Path(values["result"]).expanduser()
    if not target.is_absolute():
        target = root / target
    block = {
        "status": "error",
        "returncode": None,
        "elapsed_seconds": 0.0,
        "gate_status": None,
        "exact_match": None,
        "key_copy": None,
        "promotion": None,
        "result_path": str(target),
        "reasons": [],
    }
    try:
        argv = _curriculum_command(command, values)
    except RunnerError as exc:
        block["error"] = str(exc)
        block["reasons"] = ["command_invalid"]
        return block
    started = time.monotonic()
    try:
        completed = subprocess.run(
            argv, cwd=str(root), env=_curriculum_env(root),
            capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=timeout)
        block["returncode"] = completed.returncode
    except subprocess.TimeoutExpired:
        block["elapsed_seconds"] = round(time.monotonic() - started, 3)
        block["reasons"] = ["timed_out"]
        block["error"] = f"curriculum runner timed out after {timeout}s"
        return block
    except OSError as exc:
        block["elapsed_seconds"] = round(time.monotonic() - started, 3)
        block["reasons"] = ["runner_failed"]
        block["error"] = (
            "curriculum runner could not start "
            f"({type(exc).__name__})")
        return block
    block["elapsed_seconds"] = round(time.monotonic() - started, 3)
    payload = None
    read_reason = ""
    try:
        raw = target.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        read_reason = "result_missing"
        block["error"] = "curriculum runner did not write its result JSON"
    except OSError:
        read_reason = "result_unreadable"
        block["error"] = "curriculum result JSON could not be read"
    else:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            read_reason = "result_invalid"
            block["error"] = "curriculum result JSON is invalid"
        if payload is not None and not isinstance(payload, dict):
            payload = None
            read_reason = "result_invalid"
            block["error"] = "curriculum result JSON is not an object"
    if payload is None:
        block["reasons"] = [reason for reason in (
            "runner_failed" if completed.returncode != 0 else "",
            read_reason,
        ) if reason]
        return block
    gate = payload.get("gate")
    gate = gate if isinstance(gate, dict) else {}
    gate_status = gate.get("status")
    if gate_status is None:
        gate_status = payload.get("status")
    if gate_status in _CURRICULUM_GATE_STATUSES:
        block["gate_status"] = gate_status
    final_eval_payload = payload.get("final_eval")
    final_eval_payload = (
        final_eval_payload if isinstance(final_eval_payload, dict) else {})
    exact_match = _metric_number(payload.get("exact_match_rate"))
    if exact_match is None:
        exact_match = _metric_number(
            final_eval_payload.get("exact_match_rate"))
    key_copy = _metric_number(payload.get("key_copy_rate"))
    if key_copy is None:
        key_copy = _metric_number(final_eval_payload.get("key_copy_rate"))
    block["exact_match"] = exact_match
    block["key_copy"] = key_copy
    raw_promotion = payload.get("promotion")
    if isinstance(raw_promotion, dict):
        reason = raw_promotion.get("reason")
        block["promotion"] = {
            "requested": raw_promotion.get("requested") is True,
            "promoted": raw_promotion.get("promoted") is True,
            "reason": (reason if reason in _CURRICULUM_PROMOTION_REASONS
                       else None),
        }
    reasons = []
    if completed.returncode != 0:
        reasons.append("runner_failed")
    if block["gate_status"] != "pass":
        reasons.append("gate_not_pass")
    if gate.get("ok") is False:
        reasons.append("gate_not_ok")
    min_exact_match = _metric_number(min_exact_match)
    if (exact_match is not None and min_exact_match is not None
            and exact_match < min_exact_match):
        reasons.append("exact_match_low")
    min_key_copy_rate = _metric_number(min_key_copy_rate)
    if (key_copy is not None and min_key_copy_rate is not None
            and key_copy < min_key_copy_rate):
        reasons.append("key_copy_low")
    block["reasons"] = reasons
    block["status"] = "pass" if not reasons else "fail"
    return block


def run_loop(
    *,
    signals: dict,
    allowed_files,
    checks,
    opt_in: bool,
    approve: bool = False,
    veto: bool = False,
    repo_root=".",
    runner_command=None,
    base: str = "main",
    worktree_root=None,
    branch: str | None = None,
    ledger_path=None,
    scoreboard_path=None,
    after: dict | None = None,
    aisc_path: str = "",
    archive=None,
    timeout: int = 1800,
    self_code_root=None,
    target_path: str | None = None,
    autonomous: bool = False,
    auto_merge: bool = True,
    test_command=None,
    test_timeout: int = 900,
    adapter_examples=None,
    adapter_train_examples=None,
    adapter_eval_examples=None,
    adapter_out_dir=None,
    adapter_base=None,
    adapter_trainer_command=None,
    adapter_adapter_dir=None,
    adapter_comparison_out=None,
    adapter_comparison=None,
    adapter_evaluator_command=None,
    adapter_ledger_path=None,
    adapter_eval_ratio=0.2,
    adapter_seed=13,
    adapter_min_loss_delta=None,
    adapter_max_ppl_ratio=None,
    adapter_require_content=False,
    adapter_min_content_delta=0.05,
    adapter_max_content_ppl_ratio=1.0,
    adapter_min_key_copy_rate=None,
    adapter_timeout: int = 900,
    adapter_trainer_timeout: int = 900,
    adapter_curriculum_command=None,
    adapter_curriculum_result=None,
    adapter_curriculum_out=None,
    adapter_curriculum_active_file=None,
    adapter_curriculum_final_eval=None,
    adapter_curriculum_data_dir=None,
    adapter_curriculum_min_exact_match: float = 0.25,
    adapter_curriculum_min_key_copy_rate: float = 0.25,
) -> dict:
    """Run one bounded loop, returning every stage and its evidence."""
    normalized = dict(signals) if isinstance(signals, dict) else {}
    ledger = _load_ledger(ledger_path) if ledger_path else []
    if ledger and "history" not in normalized:
        normalized["history"] = list(ledger)
    before_self_code = None
    if self_code_root:
        from atomzero.self_code import audit_report
        before_self_code = audit_report(
            self_code_root, propose=True, max_per_kind=_AUDIT_MAX_PER_KIND)
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
    if action["kind"] == "adapter_candidate_ready":
        result = {
            "status": "awaiting_human",
            "plan": plan,
            "autonomous": bool(autonomous),
            "auto_apply": False,
            "weights_changed": False,
        }
        if veto:
            result["status"] = "vetoed"
            return result
        if not opt_in:
            result["status"] = "awaiting_opt_in"
            return result
        if adapter_curriculum_command:
            result["curriculum"] = _run_adapter_curriculum(
                command=adapter_curriculum_command,
                repo_root=repo_root,
                base=adapter_base,
                data_dir=adapter_curriculum_data_dir,
                out_dir=adapter_curriculum_out,
                result_path=adapter_curriculum_result,
                active_file=adapter_curriculum_active_file,
                final_eval=adapter_curriculum_final_eval,
                min_exact_match=adapter_curriculum_min_exact_match,
                min_key_copy_rate=adapter_curriculum_min_key_copy_rate,
                timeout=adapter_timeout)
            result["status"] = (
                "completed" if result["curriculum"]["status"] == "pass"
                else "curriculum_failed")
            return result
        if not autonomous and not approve:
            return result
        if not adapter_out_dir or not (
                adapter_examples
                or (adapter_train_examples and adapter_eval_examples)):
            result["status"] = "adapter_input_missing"
            result["error"] = (
                "adapter examples or an explicit train/eval pair, plus "
                "adapter_out_dir, are required for adapter_candidate_ready")
            return result
        from atomzero.adapter_candidates import run_adapter_loop
        adapter = run_adapter_loop(
            adapter_examples, adapter_out_dir,
            base=adapter_base,
            trainer_command=adapter_trainer_command,
            adapter_dir=adapter_adapter_dir,
            comparison_path=adapter_comparison_out,
            comparison=adapter_comparison,
            evaluator_command=adapter_evaluator_command,
            repo_root=repo_root,
            train_examples=adapter_train_examples,
            eval_examples=adapter_eval_examples,
            eval_ratio=adapter_eval_ratio, seed=adapter_seed,
            min_loss_delta=adapter_min_loss_delta,
            max_ppl_ratio=adapter_max_ppl_ratio,
            require_content=adapter_require_content,
            min_content_delta=adapter_min_content_delta,
            max_content_ppl_ratio=adapter_max_content_ppl_ratio,
            min_key_copy_rate=adapter_min_key_copy_rate,
            ledger_path=adapter_ledger_path, timeout=adapter_timeout,
            trainer_timeout=adapter_trainer_timeout)
        result["adapter"] = adapter
        result["status"] = {
            "pass": "completed",
            "fail": "adapter_gate_failed",
            "awaiting_evaluation": "adapter_awaiting_evaluation",
            "invalid": "adapter_gate_failed",
        }.get(adapter.get("status"), "adapter_gate_failed")
        return result
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
        "autonomous": bool(autonomous),
        "auto_apply": False,
        "weights_changed": False,
    }
    if veto:
        result["status"] = "vetoed"
        return result
    if not opt_in:
        result["status"] = "awaiting_opt_in"
        return result
    if adapter_curriculum_command:
        result["curriculum"] = _run_adapter_curriculum(
            command=adapter_curriculum_command,
            repo_root=repo_root,
            base=adapter_base,
            data_dir=adapter_curriculum_data_dir,
            out_dir=adapter_curriculum_out,
            result_path=adapter_curriculum_result,
            active_file=adapter_curriculum_active_file,
            final_eval=adapter_curriculum_final_eval,
            min_exact_match=adapter_curriculum_min_exact_match,
            min_key_copy_rate=adapter_curriculum_min_key_copy_rate,
            timeout=adapter_timeout)
        if result["curriculum"]["status"] != "pass":
            result["status"] = "curriculum_failed"
            return result
    if not autonomous and not approve:
        return result
    run_branch = branch
    if autonomous and not run_branch:
        run_branch = (
            f"agent/{action['action_id']}-{time.strftime('%Y%m%d-%H%M%S')}")
    runner = run_packet(
        packet, opt_in=True, repo_root=repo_root,
        runner_command=runner_command, base=base,
        worktree_root=worktree_root, branch=run_branch,
        aisc_path=aisc_path, archive=archive, timeout=timeout)
    result["runner"] = runner
    if runner.get("ok") is not True:
        result["status"] = "gate_failed"
        return result
    if before_self_code is not None:
        from atomzero.self_code import audit_report
        from atomzero.self_code_diff import compare_reports
        after_self_code = audit_report(
            runner["worktree"], propose=False,
            max_per_kind=_AUDIT_MAX_PER_KIND)
        diff = compare_reports(before_self_code, after_self_code)
        result["self_code_diff"] = diff
        if diff["ok"] is not True:
            result["status"] = "self_code_regression"
            return result
    if autonomous:
        result["sandbox"] = backend_info()
        verification = run_tests(
            runner["worktree"], command=test_command, timeout=test_timeout)
        result["verification"] = verification
        if verification.get("ok") is not True:
            result["status"] = "verification_failed"
            return result
        if auto_merge:
            if _git_dirty(repo_root):
                result["status"] = "repo_dirty"
                return result
            branch = str(runner.get("branch") or "")
            if not branch:
                result["status"] = "merge_failed"
                result["merge"] = {"ok": False, "error": "branch missing"}
                return result
            merge = _merge_branch(
                repo_root, branch,
                f"autonomous: merge {action['action_id']}")
            result["merge"] = merge
            if merge.get("ok") is not True:
                result["status"] = "merge_failed"
                return result
            post = run_tests(
                repo_root, command=test_command, timeout=test_timeout)
            result["post_merge_verification"] = post
            if post.get("ok") is not True:
                result["rolled_back_to"] = merge.get("before", "")
                _reset_hard(repo_root, merge.get("before", ""))
                result["status"] = "post_merge_failed"
                return result
        else:
            result["merge"] = {"ok": True, "skipped": True}
    gate = runner["gate"]
    approved_by = "autonomous gate" if autonomous else "human"
    record = record_gated_decision(
        ledger, action, "approve", gate,
        note=(f"self-improve loop: {action['action_id']} gate "
              f"{gate.get('gate_id', 'no-gate')} passed; {approved_by}"))
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
    if (autonomous and auto_merge
            and result.get("merge", {}).get("ok") is True
            and not result.get("merge", {}).get("skipped")):
        result["cleanup"] = _cleanup_worktree(
            repo_root, runner.get("worktree"), runner.get("branch", ""))
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
    parser.add_argument("--approve", action="store_true",
                        help="human approval (only with --require-approval)")
    parser.add_argument(
        "--require-approval", action="store_true",
        help="restore the per-change human approval gate")
    parser.add_argument("--veto", action="store_true")
    parser.add_argument(
        "--no-auto-merge", action="store_true",
        help="verify the change but do not merge it into the repo")
    parser.add_argument(
        "--test-command", default="",
        help="full verification command (default: python -m pytest tests -q)")
    parser.add_argument("--test-timeout", type=int, default=900)
    parser.add_argument("--adapter-examples-jsonl", default="")
    parser.add_argument("--adapter-train-jsonl", default="")
    parser.add_argument("--adapter-eval-jsonl", default="")
    parser.add_argument("--adapter-out", default="")
    parser.add_argument("--adapter-base", default="")
    parser.add_argument("--adapter-trainer-command", default="")
    parser.add_argument("--adapter-adapter-dir", default="")
    parser.add_argument("--adapter-comparison-out", default="")
    parser.add_argument("--adapter-comparison-json", default="")
    parser.add_argument("--adapter-evaluator-command", default="")
    parser.add_argument("--adapter-ledger", default="")
    parser.add_argument("--adapter-eval-ratio", type=float, default=0.2)
    parser.add_argument("--adapter-seed", type=int, default=13)
    parser.add_argument("--adapter-min-loss-delta", type=float, default=None)
    parser.add_argument("--adapter-max-ppl-ratio", type=float, default=None)
    parser.add_argument("--adapter-require-content", action="store_true")
    parser.add_argument("--adapter-min-content-delta", type=float,
                        default=0.05)
    parser.add_argument("--adapter-max-content-ppl-ratio", type=float,
                        default=1.0)
    parser.add_argument("--adapter-min-key-copy-rate", type=float,
                        default=None)
    parser.add_argument("--adapter-timeout", type=int, default=900)
    parser.add_argument("--adapter-trainer-timeout", type=int, default=900)
    parser.add_argument(
        "--adapter-curriculum-command", default="",
        help=("shell command template for research/run_staged_curriculum.py; "
              "placeholders: {base} {data_dir} {out_dir} {result} "
              "{active_file} {final_eval}"))
    parser.add_argument(
        "--adapter-curriculum-result", default="",
        help="path to the curriculum runner JSON ledger")
    parser.add_argument(
        "--adapter-curriculum-out", default="",
        help="curriculum adapter output directory")
    parser.add_argument(
        "--adapter-curriculum-active-file", default="",
        help="curriculum registry written by a passing promotion")
    parser.add_argument(
        "--adapter-curriculum-final-eval", default="",
        help="final held-out mixed eval JSONL")
    parser.add_argument(
        "--adapter-curriculum-data-dir", default="",
        help="staged curriculum queue root")
    parser.add_argument(
        "--adapter-curriculum-min-exact-match", type=float, default=0.25)
    parser.add_argument(
        "--adapter-curriculum-min-key-copy-rate", type=float, default=0.25)
    parser.add_argument("--adapter-promotable", type=int, default=0)
    parser.add_argument("--adapter-candidates", type=int, default=0)
    parser.add_argument("--sandbox-info", action="store_true")
    parser.add_argument("--repo", default=".")
    parser.add_argument("--runner", default="")
    parser.add_argument("--base", default="main")
    parser.add_argument("--branch", default="")
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
    if args.sandbox_info:
        print(json.dumps(backend_info(), ensure_ascii=False, indent=2))
        return 0
    autonomous = not args.require_approval
    try:
        signals = _load_signals(args.signal) if args.signal else {}
        if (args.adapter_promotable and "adapter" not in signals):
            signals["adapter"] = {
                "promotable": args.adapter_promotable,
                "candidates": args.adapter_candidates,
            }
        if (not signals and not args.self_code_root
                and not args.adapter_examples_jsonl
                and not (args.adapter_train_jsonl and args.adapter_eval_jsonl)):
            raise RunnerError(
                "--signal, --self-code-root, --adapter-examples-jsonl or "
                "an adapter train/eval pair is required")
        adapter_examples = None
        if args.adapter_examples_jsonl:
            from atomzero.adapter_candidates import load_examples
            adapter_examples = load_examples(args.adapter_examples_jsonl)
        adapter_train_examples = None
        adapter_eval_examples = None
        if args.adapter_train_jsonl or args.adapter_eval_jsonl:
            if not (args.adapter_train_jsonl and args.adapter_eval_jsonl):
                raise RunnerError(
                    "--adapter-train-jsonl and --adapter-eval-jsonl must be "
                    "used together")
            from atomzero.adapter_candidates import load_examples
            adapter_train_examples = load_examples(args.adapter_train_jsonl)
            adapter_eval_examples = load_examples(args.adapter_eval_jsonl)
        adapter_comparison = None
        if args.adapter_comparison_json:
            adapter_comparison = _load_signals(
                args.adapter_comparison_json)
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
            branch=args.branch or None,
            ledger_path=args.ledger or None,
            scoreboard_path=args.scoreboard or None,
            after=after, aisc_path=args.aisc_path,
            archive=args.archive or None,
            self_code_root=args.self_code_root or None,
            target_path=args.target_path or None,
            autonomous=autonomous,
            auto_merge=not args.no_auto_merge,
            test_command=args.test_command or None,
            test_timeout=args.test_timeout,
            adapter_examples=adapter_examples,
            adapter_train_examples=adapter_train_examples,
            adapter_eval_examples=adapter_eval_examples,
            adapter_out_dir=args.adapter_out or None,
            adapter_base=args.adapter_base or None,
            adapter_trainer_command=args.adapter_trainer_command or None,
            adapter_adapter_dir=args.adapter_adapter_dir or None,
            adapter_comparison_out=args.adapter_comparison_out or None,
            adapter_comparison=adapter_comparison,
            adapter_evaluator_command=args.adapter_evaluator_command or None,
            adapter_ledger_path=args.adapter_ledger or None,
            adapter_eval_ratio=args.adapter_eval_ratio,
            adapter_seed=args.adapter_seed,
            adapter_min_loss_delta=args.adapter_min_loss_delta,
            adapter_max_ppl_ratio=args.adapter_max_ppl_ratio,
            adapter_require_content=args.adapter_require_content,
            adapter_min_content_delta=args.adapter_min_content_delta,
            adapter_max_content_ppl_ratio=args.adapter_max_content_ppl_ratio,
            adapter_min_key_copy_rate=args.adapter_min_key_copy_rate,
            adapter_timeout=args.adapter_timeout,
            adapter_trainer_timeout=args.adapter_trainer_timeout,
            adapter_curriculum_command=args.adapter_curriculum_command or None,
            adapter_curriculum_result=args.adapter_curriculum_result or None,
            adapter_curriculum_out=args.adapter_curriculum_out or None,
            adapter_curriculum_active_file=(
                args.adapter_curriculum_active_file or None),
            adapter_curriculum_final_eval=(
                args.adapter_curriculum_final_eval or None),
            adapter_curriculum_data_dir=(
                args.adapter_curriculum_data_dir or None),
            adapter_curriculum_min_exact_match=(
                args.adapter_curriculum_min_exact_match),
            adapter_curriculum_min_key_copy_rate=(
                args.adapter_curriculum_min_key_copy_rate))
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
