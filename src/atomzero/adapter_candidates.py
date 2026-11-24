"""Verified-teacher adapter candidate export and held-out gate.

The pipeline is deliberately data-only:

    verified teacher examples
      -> deterministic train/held-out split
      -> candidate metadata + train.jsonl + eval.jsonl
      -> external adapter evaluator (optional)
      -> held-out loss/perplexity gate
      -> content-free candidate event

It never trains an adapter, applies one, or changes weights. The output is a
candidate plus a gate result for an external trainer/evaluator.

    atomzero-adapter-candidate export --examples-jsonl examples.jsonl \
      --out-dir data/adapter_candidates --json
    atomzero-adapter-candidate loop --examples-jsonl examples.jsonl \
      --out-dir data/adapter_candidates --comparison-json comparison.json \
      --ledger data/adapter_candidates.jsonl --json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

from atomzero.sandbox import run_command, sanitized_env

__all__ = [
    "DEFAULT_EVAL_RATIO",
    "DEFAULT_FAMILY",
    "DEFAULT_MAX_PPL_RATIO",
    "DEFAULT_MIN_LOSS_DELTA",
    "DEFAULT_SEED",
    "EVAL_METHOD",
    "EVENT_METHOD",
    "MAX_FAMILY_LENGTH",
    "build_candidate",
    "build_cross_candidate",
    "evaluate_candidate",
    "family_summary",
    "load_examples",
    "main",
    "promote_candidate",
    "run_adapter_loop",
    "summarize_adapter_ledger",
    "write_candidate",
]

DEFAULT_EVAL_RATIO = 0.2
DEFAULT_FAMILY = "unspecified"
MAX_FAMILY_LENGTH = 64
DEFAULT_SEED = 13
DEFAULT_MIN_LOSS_DELTA = 0.01
DEFAULT_MIN_CONTENT_DELTA = 0.05
DEFAULT_MAX_PPL_RATIO = 1.0
EVAL_METHOD = "held_out_adapter_gate_v1"
EVENT_METHOD = "content_free_adapter_candidate_event_v1"
FORBIDDEN_KEYS = frozenset({
    "text", "query", "answer_snippet", "prompt_text", "content", "note",
    "session", "source",
})


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _row_digest(row: dict) -> str:
    return _hash(str(row["user"]) + "\0" + str(row["assistant"]))


def _family_label(value, index: int, where: str = "examples") -> str:
    """Validate a content-free task-family label."""
    if value is None:
        return DEFAULT_FAMILY
    if not isinstance(value, str):
        raise ValueError(f"{where}[{index}].family must be a string")
    family = value.strip()
    if not family:
        raise ValueError(f"{where}[{index}].family must not be blank")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in family):
        raise ValueError(
            f"{where}[{index}].family contains control characters")
    if len(family) > MAX_FAMILY_LENGTH:
        raise ValueError(
            f"{where}[{index}].family exceeds {MAX_FAMILY_LENGTH} characters")
    return family


def family_summary(rows) -> dict:
    """Count rows by validated task family without exposing content."""
    if not isinstance(rows, list):
        raise ValueError("rows must be a list")
    counts = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"rows[{index}] must be an object")
        family = _family_label(row.get("family"), index, where="rows")
        counts[family] = counts.get(family, 0) + 1
    return {name: counts[name] for name in sorted(counts)}


def _clean_example(row, index: int) -> dict:
    if not isinstance(row, dict):
        raise ValueError(f"examples[{index}] must be an object")
    user = row.get("user", row.get("prompt"))
    assistant = row.get("assistant", row.get("response"))
    if not isinstance(user, str) or not user.strip():
        raise ValueError(f"examples[{index}].user must be a non-empty string")
    if not isinstance(assistant, str) or not assistant.strip():
        raise ValueError(
            f"examples[{index}].assistant must be a non-empty string")
    if any(ord(ch) < 32 and ch not in "\n\r\t" for ch in user + assistant):
        raise ValueError(f"examples[{index}] contains control characters")
    verification = row.get("verification")
    if not isinstance(verification, dict):
        raise ValueError(
            f"examples[{index}].verification must be an object")
    if verification.get("verified") is not True:
        raise ValueError(f"examples[{index}] is not verified")
    if verification.get("promotable") is False:
        raise ValueError(f"examples[{index}] is verified but not promotable")
    return {
        "user": user.strip(),
        "assistant": assistant.strip(),
        "family": _family_label(row.get("family"), index),
    }


def build_candidate(examples, *, eval_ratio=DEFAULT_EVAL_RATIO,
                    seed=DEFAULT_SEED) -> dict:
    """Split verified examples into a train set and a held-out eval set."""
    if not isinstance(examples, list):
        raise ValueError("examples must be a list")
    try:
        eval_ratio = float(eval_ratio)
        seed = int(seed)
    except (TypeError, ValueError) as exc:
        raise ValueError("eval_ratio and seed must be numeric") from exc
    if not 0.0 <= eval_ratio < 1.0:
        raise ValueError("eval_ratio must be in [0, 1)")
    rows = []
    seen = set()
    duplicates = 0
    for index, raw in enumerate(examples):
        row = _clean_example(raw, index)
        digest = _row_digest(row)
        if digest in seen:
            duplicates += 1
            continue
        seen.add(digest)
        rows.append(row)
    if not rows:
        raise ValueError("no unique verified examples")
    rows.sort(key=lambda item: _hash(f"{seed}|{_row_digest(item)}"))
    if len(rows) == 1 or eval_ratio == 0.0:
        eval_count = 0
    else:
        eval_count = max(1, round(len(rows) * eval_ratio))
        eval_count = min(eval_count, len(rows) - 1)
    train_rows = rows[:len(rows) - eval_count]
    eval_rows = rows[len(rows) - eval_count:]
    train_family_counts = family_summary(train_rows)
    eval_family_counts = family_summary(eval_rows)
    payload = "".join(
        _row_digest(row) for row in rows) + f"|{seed}|{eval_ratio}"
    candidate_id = "ad-" + _hash(payload)[:16]
    metadata = {
        "candidate_id": candidate_id,
        "version": 1,
        "train_examples": len(train_rows),
        "eval_examples": len(eval_rows),
        "total_examples": len(rows),
        "duplicates_skipped": duplicates,
        "eval_ratio": eval_ratio,
        "seed": seed,
        "family_separation": "not_applicable",
        "train_families": sorted(train_family_counts),
        "eval_families": sorted(eval_family_counts),
        "train_family_counts": train_family_counts,
        "eval_family_counts": eval_family_counts,
        "min_loss_delta": DEFAULT_MIN_LOSS_DELTA,
        "max_ppl_ratio": DEFAULT_MAX_PPL_RATIO,
        "auto_apply": False,
        "weights_changed": False,
        "method": EVENT_METHOD,
    }
    return {
        "candidate_id": candidate_id,
        "metadata": metadata,
        "train": train_rows,
        "eval": eval_rows,
    }


def build_cross_candidate(train_examples, eval_examples, *,
                           seed=DEFAULT_SEED, candidate_id=None,
                           split_method="explicit_cross_distribution") -> dict:
    """Build an explicit train/eval candidate without random mixing.

    This is for cross-distribution checks: train and eval come from different
    distributions, and an overlapping row is rejected rather than silently
    leaked into both sides.
    """
    if not isinstance(train_examples, list) or not isinstance(
            eval_examples, list):
        raise ValueError("train_examples and eval_examples must be lists")
    def clean(rows, label):
        cleaned = []
        seen = set()
        duplicates = 0
        for index, raw in enumerate(rows):
            row = _clean_example(raw, index)
            digest = _row_digest(row)
            if digest in seen:
                duplicates += 1
                continue
            seen.add(digest)
            cleaned.append(row)
        if not cleaned:
            raise ValueError(f"{label} has no unique verified examples")
        return cleaned, seen, duplicates

    train_rows, train_seen, train_dupes = clean(
        train_examples, "train_examples")
    eval_rows, eval_seen, eval_dupes = clean(
        eval_examples, "eval_examples")
    overlap = train_seen & eval_seen
    if overlap:
        raise ValueError(
            f"train/eval overlap: {len(overlap)} identical example(s)")
    train_family_counts = family_summary(train_rows)
    eval_family_counts = family_summary(eval_rows)
    train_families = sorted(train_family_counts)
    eval_families = sorted(eval_family_counts)
    train_specified = set(train_families) - {DEFAULT_FAMILY}
    eval_specified = set(eval_families) - {DEFAULT_FAMILY}
    leaked = sorted(train_specified & eval_specified)
    if leaked:
        raise ValueError(
            "task-family leakage: eval families also appear in training: "
            + ", ".join(leaked))
    family_separation = (
        DEFAULT_FAMILY not in train_families
        and DEFAULT_FAMILY not in eval_families)
    payload = (
        "".join(_row_digest(row) for row in train_rows)
        + "|CROSS|"
        + "".join(_row_digest(row) for row in eval_rows)
        + f"|{seed}")
    identifier = candidate_id or ("ad-" + _hash(payload)[:16])
    metadata = {
        "candidate_id": identifier,
        "version": 1,
        "train_examples": len(train_rows),
        "eval_examples": len(eval_rows),
        "total_examples": len(train_rows) + len(eval_rows),
        "duplicates_skipped": train_dupes + eval_dupes,
        "eval_ratio": (
            round(len(eval_rows) / (len(train_rows) + len(eval_rows)), 6)
            if train_rows or eval_rows else 0.0),
        "seed": int(seed),
        "split_method": split_method,
        "family_separation": family_separation,
        "train_families": train_families,
        "eval_families": eval_families,
        "train_family_counts": train_family_counts,
        "eval_family_counts": eval_family_counts,
        "min_loss_delta": DEFAULT_MIN_LOSS_DELTA,
        "max_ppl_ratio": DEFAULT_MAX_PPL_RATIO,
        "auto_apply": False,
        "weights_changed": False,
        "method": EVENT_METHOD,
    }
    return {
        "candidate_id": identifier,
        "metadata": metadata,
        "train": train_rows,
        "eval": eval_rows,
    }


def write_candidate(candidate: dict, out_dir) -> dict:
    """Write metadata and the text-bearing train/eval files locally."""
    if not isinstance(candidate, dict) or not candidate.get("candidate_id"):
        raise ValueError("candidate must contain candidate_id")
    target = Path(out_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    metadata_path = target / "candidate.json"
    train_path = target / "train.jsonl"
    eval_path = target / "eval.jsonl"
    metadata_path.write_text(
        json.dumps(candidate["metadata"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    for path, rows in ((train_path, candidate["train"]),
                       (eval_path, candidate["eval"])):
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n"
                    for row in rows), encoding="utf-8")
    return {
        "candidate_json": str(metadata_path),
        "train_jsonl": str(train_path),
        "eval_jsonl": str(eval_path),
    }


def _metric_pair(comparison: dict) -> tuple[dict, dict, str, int | None]:
    if not isinstance(comparison, dict):
        raise ValueError("comparison must be an object")
    examples = comparison.get("examples")
    if examples is not None and (
            isinstance(examples, bool) or not isinstance(examples, int)):
        raise ValueError("comparison.examples must be an integer")
    if "baseline" in comparison and "adapter" in comparison:
        baseline = comparison["baseline"]
        adapter = comparison["adapter"]
        label = str(comparison.get("adapter_label") or "adapter")
        return baseline, adapter, label, examples
    models = comparison.get("models")
    if not isinstance(models, list):
        raise ValueError("comparison must contain baseline/adapter or models")
    base = next((item for item in models
                 if isinstance(item, dict)
                 and str(item.get("label")) == "base"), None)
    adapter = next((item for item in models
                    if isinstance(item, dict)
                    and str(item.get("label")) != "base"), None)
    if base is None or adapter is None:
        raise ValueError("comparison.models must contain base and an adapter")
    base_metrics = base.get("metrics")
    adapter_metrics = adapter.get("metrics")
    if not isinstance(base_metrics, dict) or not isinstance(
            adapter_metrics, dict):
        raise ValueError("comparison model metrics must be objects")
    return base_metrics, adapter_metrics, str(adapter.get("label") or "adapter"), examples


def _number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return number


def evaluate_candidate(candidate, comparison, *, min_loss_delta=None,
                       max_ppl_ratio=None, require_content=False,
                       min_content_delta=DEFAULT_MIN_CONTENT_DELTA,
                       max_content_ppl_ratio=1.0) -> dict:
    """Run the held-out adapter gate against a comparison report."""
    if not isinstance(candidate, dict):
        raise ValueError("candidate must be an object")
    metadata = candidate.get("metadata", candidate)
    if not isinstance(metadata, dict):
        raise ValueError("candidate.metadata must be an object")
    eval_examples = metadata.get("eval_examples")
    if isinstance(eval_examples, bool) or not isinstance(eval_examples, int):
        raise ValueError("candidate.eval_examples must be an integer")
    threshold = (DEFAULT_MIN_LOSS_DELTA if min_loss_delta is None
                 else float(min_loss_delta))
    max_ratio = (DEFAULT_MAX_PPL_RATIO if max_ppl_ratio is None
                 else float(max_ppl_ratio))
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("min_loss_delta must be finite and non-negative")
    if not math.isfinite(max_ratio) or max_ratio <= 0:
        raise ValueError("max_ppl_ratio must be finite and positive")
    try:
        content_threshold = float(min_content_delta)
        content_max_ratio = float(max_content_ppl_ratio)
    except (TypeError, ValueError) as exc:
        raise ValueError("content thresholds must be numeric") from exc
    if not math.isfinite(content_threshold) or content_threshold < 0:
        raise ValueError("min_content_delta must be finite and non-negative")
    if not math.isfinite(content_max_ratio) or content_max_ratio <= 0:
        raise ValueError("max_content_ppl_ratio must be finite and positive")
    try:
        base, adapter, label, examples = _metric_pair(comparison)
        base_loss = _number(base.get("loss"), "baseline.loss")
        adapter_loss = _number(adapter.get("loss"), "adapter.loss")
        base_ppl = _number(base.get("ppl"), "baseline.ppl")
        adapter_ppl = _number(adapter.get("ppl"), "adapter.ppl")
    except ValueError as exc:
        return {
            "status": "invalid",
            "ok": False,
            "promotable": False,
            "candidate_id": metadata.get("candidate_id", ""),
            "reasons": [f"invalid_comparison: {exc}"],
            "auto_apply": False,
            "requires_training": True,
            "weights_changed": False,
            "method": EVAL_METHOD,
        }
    reasons = []
    if examples is None:
        reasons.append("eval_count_missing")
    elif examples != eval_examples:
        reasons.append(
            f"eval_count_mismatch: comparison={examples} "
            f"candidate={eval_examples}")
    loss_delta = adapter_loss - base_loss
    ppl_ratio = adapter_ppl / base_ppl if base_ppl else float("inf")
    if loss_delta <= -threshold:
        reasons.append("loss_improved")
    else:
        reasons.append("loss_not_improved")
    if ppl_ratio <= max_ratio:
        reasons.append("ppl_not_worse")
    else:
        reasons.append("ppl_regression")
    content_loss_delta = None
    content_ppl_ratio = None
    content_ok = True
    base_content = base.get("content_loss")
    adapter_content = adapter.get("content_loss")
    base_content_ppl = base.get("content_ppl")
    adapter_content_ppl = adapter.get("content_ppl")
    if require_content:
        if not all(isinstance(value, (int, float))
                   and not isinstance(value, bool)
                   for value in (base_content, adapter_content,
                                 base_content_ppl, adapter_content_ppl)):
            reasons.append("content_metrics_missing")
            content_ok = False
        else:
            base_content = float(base_content)
            adapter_content = float(adapter_content)
            base_content_ppl = float(base_content_ppl)
            adapter_content_ppl = float(adapter_content_ppl)
            content_loss_delta = adapter_content - base_content
            content_ppl_ratio = (adapter_content_ppl / base_content_ppl
                                 if base_content_ppl else float("inf"))
            if content_loss_delta <= -content_threshold:
                reasons.append("content_loss_improved")
            else:
                reasons.append("content_loss_not_improved")
                content_ok = False
            if content_ppl_ratio <= content_max_ratio:
                reasons.append("content_ppl_not_worse")
            else:
                reasons.append("content_ppl_regression")
                content_ok = False
    ok = (
        not any(reason.startswith(("eval_count", "invalid"))
                for reason in reasons)
        and "loss_improved" in reasons
        and "ppl_not_worse" in reasons
        and content_ok
    )
    return {
        "status": "pass" if ok else "fail",
        "ok": ok,
        "promotable": ok,
        "candidate_id": metadata.get("candidate_id", ""),
        "adapter_label": label,
        "eval_examples": eval_examples,
        "baseline_loss": base_loss,
        "adapter_loss": adapter_loss,
        "loss_delta": round(loss_delta, 6),
        "baseline_ppl": base_ppl,
        "adapter_ppl": adapter_ppl,
        "ppl_ratio": round(ppl_ratio, 6),
        "baseline_content_loss": base_content,
        "adapter_content_loss": adapter_content,
        "content_loss_delta": (round(content_loss_delta, 6)
                               if content_loss_delta is not None else None),
        "content_ppl_ratio": (round(content_ppl_ratio, 6)
                              if content_ppl_ratio is not None else None),
        "require_content": bool(require_content),
        "min_loss_delta": threshold,
        "max_ppl_ratio": max_ratio,
        "reasons": reasons,
        "auto_apply": False,
        "requires_training": True,
        "weights_changed": False,
        "method": EVAL_METHOD,
    }


def _load_json(path: str) -> dict:
    raw = sys.stdin.read() if path == "-" else Path(path).read_text(
        encoding="utf-8-sig")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _load_examples(path: str) -> list[dict]:
    if path == "-":
        raw = sys.stdin.read()
    else:
        raw = Path(path).read_text(encoding="utf-8-sig")
    rows = []
    for number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{number}: invalid JSON") from exc
        if not isinstance(item, dict):
            raise ValueError(f"{path}:{number}: must be a JSON object")
        rows.append(item)
    return rows


def load_examples(path: str) -> list[dict]:
    """Load a JSONL verified-example file."""
    return _load_examples(path)


def _event(candidate: dict, evaluation: dict) -> dict:
    metadata = candidate["metadata"]
    return {
        "version": 1,
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "candidate_id": metadata["candidate_id"],
        "train_examples": metadata["train_examples"],
        "eval_examples": metadata["eval_examples"],
        "status": evaluation.get("status", "invalid"),
        "promotable": evaluation.get("promotable") is True,
        "loss_delta": evaluation.get("loss_delta"),
        "ppl_ratio": evaluation.get("ppl_ratio"),
        "auto_apply": False,
        "weights_changed": False,
        "method": EVENT_METHOD,
    }


def _append_ledger(path, event: dict) -> list[dict]:
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    if target.exists():
        for number, line in enumerate(
                target.read_text(encoding="utf-8-sig").splitlines(),
                start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{target}:{number}: invalid JSON") from exc
            if not isinstance(item, dict):
                raise ValueError(f"{target}:{number}: must be an object")
            entries.append(item)
    entries.append(event)
    target.write_text(
        "".join(json.dumps(item, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":")) + "\n"
                for item in entries), encoding="utf-8")
    return entries


def _format_command(command, values: dict) -> list[str]:
    if isinstance(command, str):
        import shlex
        command = shlex.split(command, posix=False)
    if not isinstance(command, (list, tuple)) or not command:
        raise ValueError("command must be a non-empty list or string")
    return [str(part).format(**values) for part in command]


def run_adapter_loop(examples, out_dir, *, base=None, trainer_command=None,
                     comparison=None, evaluator_command=None,
                     adapter_dir=None, comparison_path=None,
                     repo_root=".", train_examples=None, eval_examples=None,
                     eval_ratio=DEFAULT_EVAL_RATIO, seed=DEFAULT_SEED,
                     min_loss_delta=None, max_ppl_ratio=None,
                     require_content=False,
                     min_content_delta=DEFAULT_MIN_CONTENT_DELTA,
                     max_content_ppl_ratio=1.0,
                     ledger_path=None, timeout: int = 900,
                     trainer_timeout: int = 900) -> dict:
    """Export a candidate, optionally train/evaluate, and record a gate event."""
    if train_examples is not None or eval_examples is not None:
        if train_examples is None or eval_examples is None:
            raise ValueError(
                "train_examples and eval_examples must be provided together")
        candidate = build_cross_candidate(
            train_examples, eval_examples, seed=seed)
    else:
        candidate = build_candidate(
            examples, eval_ratio=eval_ratio, seed=seed)
    paths = write_candidate(candidate, out_dir)
    target_dir = Path(out_dir).resolve()
    adapter_path = Path(
        adapter_dir or (target_dir / "adapter")).resolve()
    comparison_file = Path(
        comparison_path or (target_dir / "comparison.json")).resolve()
    env = sanitized_env(target_dir)
    if repo_root:
        env["PYTHONPATH"] = str(Path(repo_root).resolve() / "src")
    source = "provided"
    trainer = None
    if trainer_command:
        values = {
            "out_dir": str(target_dir),
            "candidate_json": paths["candidate_json"],
            "train_jsonl": paths["train_jsonl"],
            "eval_jsonl": paths["eval_jsonl"],
            "candidate_id": candidate["candidate_id"],
            "adapter_dir": str(adapter_path),
            "comparison_path": str(comparison_file),
            "base": str(base or ""),
        }
        command = _format_command(trainer_command, values)
        trainer_result = run_command(
            command, cwd=target_dir, timeout=trainer_timeout, env=env)
        trainer = {
            "ok": trainer_result.get("ok") is True,
            "returncode": trainer_result.get("returncode"),
            "timed_out": trainer_result.get("timed_out") is True,
            "elapsed_seconds": trainer_result.get("elapsed_seconds"),
        }
        source = "trainer"
        if trainer_result.get("ok") is not True:
            evaluation = {
                "status": "invalid",
                "ok": False,
                "promotable": False,
                "candidate_id": candidate["candidate_id"],
                "reasons": ["trainer_failed"],
                "auto_apply": False,
                "requires_training": True,
                "weights_changed": False,
                "method": EVAL_METHOD,
            }
            comparison = None
        elif comparison is None and comparison_file.is_file():
            try:
                comparison = json.loads(
                    comparison_file.read_text(encoding="utf-8-sig"))
            except (OSError, json.JSONDecodeError):
                comparison = None
    if trainer is not None and comparison is None:
        if not trainer["ok"]:
            return _adapter_result(
                candidate, paths, evaluation, trainer,
                {"adapter_dir": str(adapter_path),
                 "comparison_json": str(comparison_file)},
                source, ledger_path)
        if evaluator_command is None and not comparison_file.is_file():
            evaluation = {
                "status": "invalid",
                "ok": False,
                "promotable": False,
                "candidate_id": candidate["candidate_id"],
                "reasons": ["trainer_no_comparison"],
                "auto_apply": False,
                "requires_training": True,
                "weights_changed": False,
                "method": EVAL_METHOD,
            }
            return _adapter_result(
                candidate, paths, evaluation, trainer,
                {"adapter_dir": str(adapter_path),
                 "comparison_json": str(comparison_file)},
                source, ledger_path)
    if comparison is None and evaluator_command:
        command = _format_command(evaluator_command, {
            "out_dir": str(target_dir),
            "candidate_json": paths["candidate_json"],
            "train_jsonl": paths["train_jsonl"],
            "eval_jsonl": paths["eval_jsonl"],
            "candidate_id": candidate["candidate_id"],
            "adapter_dir": str(adapter_path),
            "comparison_path": str(comparison_file),
            "base": str(base or ""),
        })
        result = run_command(
            command, cwd=target_dir, timeout=timeout, env=env)
        source = "trainer+evaluator" if trainer else "evaluator"
        if result.get("ok") is not True:
            evaluation = {
                "status": "invalid",
                "ok": False,
                "promotable": False,
                "candidate_id": candidate["candidate_id"],
                "reasons": ["evaluator_failed"],
                "evaluator_returncode": result.get("returncode"),
                "auto_apply": False,
                "requires_training": True,
                "weights_changed": False,
                "method": EVAL_METHOD,
            }
            return _adapter_result(
                candidate, paths, evaluation, trainer,
                {"adapter_dir": str(adapter_path),
                 "comparison_json": str(comparison_file)},
                source, ledger_path)
        try:
            comparison = json.loads(result.get("stdout") or "{}")
        except json.JSONDecodeError:
            evaluation = {
                "status": "invalid",
                "ok": False,
                "promotable": False,
                "candidate_id": candidate["candidate_id"],
                "reasons": ["evaluator_invalid_json"],
                "auto_apply": False,
                "requires_training": True,
                "weights_changed": False,
                "method": EVAL_METHOD,
            }
            return _adapter_result(
                candidate, paths, evaluation, trainer,
                {"adapter_dir": str(adapter_path),
                 "comparison_json": str(comparison_file)},
                source, ledger_path)
    if comparison is not None:
        evaluation = evaluate_candidate(
            candidate, comparison, min_loss_delta=min_loss_delta,
            max_ppl_ratio=max_ppl_ratio, require_content=require_content,
            min_content_delta=min_content_delta,
            max_content_ppl_ratio=max_content_ppl_ratio)
    else:
        evaluation = {
            "status": "awaiting_evaluation",
            "ok": False,
            "promotable": False,
            "candidate_id": candidate["candidate_id"],
            "reasons": ["no_comparison_or_evaluator"],
            "auto_apply": False,
            "requires_training": True,
            "weights_changed": False,
            "method": EVAL_METHOD,
        }
    return _adapter_result(
        candidate, paths, evaluation, trainer,
        {"adapter_dir": str(adapter_path),
         "comparison_json": str(comparison_file)},
        source, ledger_path)


def _adapter_result(candidate, paths, evaluation, trainer, extra_paths,
                    source, ledger_path) -> dict:
    event = _event(candidate, evaluation)
    entries = _append_ledger(ledger_path, event) if ledger_path else []
    return {
        "status": evaluation["status"],
        "candidate": candidate["metadata"],
        "paths": {**paths, **extra_paths},
        "evaluation": evaluation,
        "event": event,
        "source": source,
        "trainer": trainer,
        "ledger_entries": len(entries),
        "auto_apply": False,
        "weights_changed": False,
    }


def promote_candidate(candidate, gate, adapter_dir, active_file,
                      *, now=None) -> dict:
    """Write the active-adapter registry only after a passing gate.

    This is a product configuration action, not training: it does not modify
    base weights. The record is content-free except for the local adapter path.
    """
    if not isinstance(candidate, dict):
        raise ValueError("candidate must be an object")
    metadata = candidate.get("metadata", candidate)
    if not isinstance(metadata, dict):
        raise ValueError("candidate.metadata must be an object")
    if not isinstance(gate, dict):
        raise ValueError("gate must be an object")
    actual_gate = gate.get("gate", gate)
    if not isinstance(actual_gate, dict):
        raise ValueError("gate.gate must be an object")
    if actual_gate.get("status") != "pass" or actual_gate.get("ok") is not True:
        raise ValueError("gate must be a passing gate result")
    if actual_gate.get("promotable") is not True:
        raise ValueError("gate must mark the candidate promotable")
    reasons = actual_gate.get("reasons")
    if not isinstance(reasons, list) or (
            "content_loss_improved" not in reasons
            or "content_ppl_not_worse" not in reasons):
        raise ValueError(
            "gate must include a passing content audit "
            "(content_loss_improved, content_ppl_not_worse)")
    if metadata.get("family_separation") is not True:
        raise ValueError(
            "candidate must have family_separation: true for promotion")
    candidate_id = str(metadata.get("candidate_id") or "")
    if not candidate_id:
        raise ValueError("candidate_id is required")
    if actual_gate.get("candidate_id") and (
            str(actual_gate["candidate_id"]) != candidate_id):
        raise ValueError("gate candidate_id does not match candidate")
    adapter_path = Path(adapter_dir).expanduser()
    if not (adapter_path / "adapter.pt").is_file():
        raise ValueError(f"adapter.pt not found in {adapter_path}")
    target = Path(active_file).expanduser()
    previous = ""
    if target.exists():
        try:
            previous = str(json.loads(
                target.read_text(encoding="utf-8")).get("path", ""))
        except (OSError, json.JSONDecodeError):
            previous = ""
    stamp = now or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    record = {
        "path": str(adapter_path),
        "candidate_id": candidate_id,
        "gate_status": actual_gate.get("status"),
        "promotable": True,
        "family_separation": metadata.get("family_separation"),
        "adapter_bytes": (adapter_path / "adapter.pt").stat().st_size,
        "promoted_at": stamp,
        "previous": previous,
        "weights_changed": False,
        "auto_apply": False,
        "method": "gated_adapter_promotion_v1",
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def summarize_adapter_ledger(ledger) -> dict:
    """Summarize content-free adapter candidate events."""
    if not isinstance(ledger, list):
        raise ValueError("ledger must be a list")
    counts = {"total": 0, "pass": 0, "fail": 0, "awaiting": 0,
              "invalid": 0, "promotable": 0}
    train = eval_count = 0
    for index, item in enumerate(ledger):
        if not isinstance(item, dict):
            raise ValueError(f"ledger[{index}] must be an object")
        if set(item) & FORBIDDEN_KEYS:
            raise ValueError(f"ledger[{index}] contains content-bearing keys")
        status = str(item.get("status") or "")
        counts["total"] += 1
        if status == "pass":
            counts["pass"] += 1
        elif status == "fail":
            counts["fail"] += 1
        elif status == "awaiting_evaluation":
            counts["awaiting"] += 1
        else:
            counts["invalid"] += 1
        if item.get("promotable") is True:
            counts["promotable"] += 1
        try:
            train += int(item.get("train_examples") or 0)
            eval_count += int(item.get("eval_examples") or 0)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"ledger[{index}] has invalid counts") from exc
    return {
        **counts,
        "train_examples": train,
        "eval_examples": eval_count,
        "parameter_growth": 0,
        "parameter_growth_state": "unchanged",
        "weights_changed": False,
        "auto_apply": False,
        "method": EVENT_METHOD,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export")
    export.add_argument("--examples-jsonl", required=True)
    export.add_argument("--out-dir", required=True)
    export.add_argument("--eval-ratio", type=float, default=DEFAULT_EVAL_RATIO)
    export.add_argument("--seed", type=int, default=DEFAULT_SEED)
    export.add_argument("--json", action="store_true")

    cross = sub.add_parser("export-cross")
    cross.add_argument("--train-jsonl", required=True)
    cross.add_argument("--eval-jsonl", required=True)
    cross.add_argument("--out-dir", required=True)
    cross.add_argument("--seed", type=int, default=DEFAULT_SEED)
    cross.add_argument("--json", action="store_true")

    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--candidate-json", required=True)
    evaluate.add_argument("--comparison-json", required=True)
    evaluate.add_argument("--min-loss-delta", type=float, default=None)
    evaluate.add_argument("--max-ppl-ratio", type=float, default=None)
    evaluate.add_argument("--require-content", action="store_true")
    evaluate.add_argument("--min-content-delta", type=float,
                          default=DEFAULT_MIN_CONTENT_DELTA)
    evaluate.add_argument("--max-content-ppl-ratio", type=float, default=1.0)
    evaluate.add_argument("--json", action="store_true")

    promote = sub.add_parser("promote")
    promote.add_argument("--candidate-json", required=True)
    promote.add_argument("--gate-json", required=True)
    promote.add_argument("--adapter-dir", required=True)
    promote.add_argument("--active-file", default="data/active_adapter.json")
    promote.add_argument("--json", action="store_true")

    loop = sub.add_parser("loop")
    loop.add_argument("--examples-jsonl", default="")
    loop.add_argument("--train-jsonl", default="")
    loop.add_argument("--eval-jsonl", default="")
    loop.add_argument("--out-dir", required=True)
    loop.add_argument("--base", default="")
    loop.add_argument("--repo-root", default=".")
    loop.add_argument("--trainer-command", default="")
    loop.add_argument("--adapter-dir", default="")
    loop.add_argument("--comparison-out", default="")
    loop.add_argument("--trainer-timeout", type=int, default=900)
    loop.add_argument("--comparison-json", default="")
    loop.add_argument("--evaluator-command", default="")
    loop.add_argument("--ledger", default="")
    loop.add_argument("--eval-ratio", type=float, default=DEFAULT_EVAL_RATIO)
    loop.add_argument("--seed", type=int, default=DEFAULT_SEED)
    loop.add_argument("--min-loss-delta", type=float, default=None)
    loop.add_argument("--max-ppl-ratio", type=float, default=None)
    loop.add_argument("--require-content", action="store_true")
    loop.add_argument("--min-content-delta", type=float,
                      default=DEFAULT_MIN_CONTENT_DELTA)
    loop.add_argument("--max-content-ppl-ratio", type=float, default=1.0)
    loop.add_argument("--timeout", type=int, default=900)
    loop.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)
    try:
        if args.command == "export":
            candidate = build_candidate(
                _load_examples(args.examples_jsonl),
                eval_ratio=args.eval_ratio, seed=args.seed)
            paths = write_candidate(candidate, args.out_dir)
            payload = {
                "candidate": candidate["metadata"],
                "paths": paths,
                "auto_apply": False,
                "weights_changed": False,
            }
            code = 0
        elif args.command == "export-cross":
            candidate = build_cross_candidate(
                _load_examples(args.train_jsonl),
                _load_examples(args.eval_jsonl), seed=args.seed)
            paths = write_candidate(candidate, args.out_dir)
            payload = {
                "candidate": candidate["metadata"],
                "family_counts": {
                    "train": candidate["metadata"]["train_family_counts"],
                    "eval": candidate["metadata"]["eval_family_counts"],
                },
                "paths": paths,
                "auto_apply": False,
                "weights_changed": False,
            }
            code = 0
        elif args.command == "promote":
            payload = promote_candidate(
                _load_json(args.candidate_json),
                _load_json(args.gate_json),
                args.adapter_dir,
                args.active_file)
            code = 0
        elif args.command == "evaluate":
            candidate = _load_json(args.candidate_json)
            comparison = _load_json(args.comparison_json)
            payload = evaluate_candidate(
                candidate, comparison, min_loss_delta=args.min_loss_delta,
                max_ppl_ratio=args.max_ppl_ratio,
                require_content=args.require_content,
                min_content_delta=args.min_content_delta,
                max_content_ppl_ratio=args.max_content_ppl_ratio)
            code = 0 if payload["ok"] else 1
        else:
            comparison = (_load_json(args.comparison_json)
                          if args.comparison_json else None)
            if args.train_jsonl or args.eval_jsonl:
                if not (args.train_jsonl and args.eval_jsonl):
                    raise ValueError(
                        "--train-jsonl and --eval-jsonl must be used together")
                examples = []
                train_examples = _load_examples(args.train_jsonl)
                eval_examples = _load_examples(args.eval_jsonl)
            else:
                if not args.examples_jsonl:
                    raise ValueError(
                        "--examples-jsonl or --train-jsonl/--eval-jsonl "
                        "is required")
                examples = _load_examples(args.examples_jsonl)
                train_examples = None
                eval_examples = None
            payload = run_adapter_loop(
                examples, args.out_dir,
                base=args.base or None,
                trainer_command=args.trainer_command or None,
                adapter_dir=args.adapter_dir or None,
                comparison_path=args.comparison_out or None,
                comparison=comparison,
                evaluator_command=args.evaluator_command or None,
                repo_root=args.repo_root,
                train_examples=train_examples,
                eval_examples=eval_examples,
                eval_ratio=args.eval_ratio, seed=args.seed,
                min_loss_delta=args.min_loss_delta,
                max_ppl_ratio=args.max_ppl_ratio,
                require_content=args.require_content,
                min_content_delta=args.min_content_delta,
                max_content_ppl_ratio=args.max_content_ppl_ratio,
                ledger_path=args.ledger or None, timeout=args.timeout,
                trainer_timeout=args.trainer_timeout)
            code = 0 if payload["status"] in ("pass", "awaiting_evaluation") else 1
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"input error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
