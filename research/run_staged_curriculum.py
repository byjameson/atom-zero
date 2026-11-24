"""Run the staged copy curriculum end to end and record a content-free ledger.

The staged copy queue (``research/build_staged_copy_queue.py``) produces one
train/eval split per stage.  This runner chains those stages:

1. train the first stage from a fresh LoRA adapter;
2. for every later stage, initialise from the previous stage's adapter
   (``--init-adapter``) and continue training on the next stage;
3. evaluate every stage on its own held-out eval split and on the final
   ``multi`` eval split;
4. evaluate the last adapter on a final held-out eval and run the
   content/behavior gate;
5. optionally promote a passing adapter into a separate curriculum registry;
6. keep only aggregate metrics in a content-free ledger.

The runner never imports ``torch`` or loads a model.  Trainer, comparator,
gate and promotion work happens in subprocesses, driven by
``research/train_adapter_candidate.py``, ``research/compare_adapters.py`` and
``atomzero.adapter_candidates``.  Tests can inject alternative command
prefixes and ``--dry-run`` writes fake metrics without touching a checkpoint
or a registry.

Example::

    python research/run_staged_curriculum.py \\
      --data-dir data/staged_copy_v1 \\
      --out-dir checkpoints/staged_curriculum \\
      --base checkpoints/transformer_behavior_50m_long \\
      --stages echo_kv,echo_json,two_kv,two_json,multi \\
      --steps 2000 --batch-size 4 --lr 3e-4 --content-weight 2 \\
      --rank 8 --alpha 8 --rank-attn 4 --alpha-attn 8 \\
      --seed 23 \\
      --result research/results/staged_curriculum_run.json \\
      --final-eval data/staged_copy_v1/mixed_eval.jsonl \\
      --require-content --min-exact-match 0.25 --min-key-copy-rate 0.25 \\
      --candidate-json checkpoints/staged_curriculum/candidate.json \\
      --gate-json checkpoints/staged_curriculum/gate.json \\
      --promote --active-file data/curriculum_active_adapter.json

Add ``--stop-on-key-copy 0.8`` to stop after the first stage whose held-out
key-copy rate reaches the gate.  Add ``--dry-run`` to validate the pipeline
and produce a result JSON without a model, data files or registry write.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

try:  # running as `python research/run_staged_curriculum.py`
    from research.build_staged_copy_queue import STAGE_ORDER, normalize_stages
except ImportError:  # pragma: no cover - package import is the normal path
    from build_staged_copy_queue import STAGE_ORDER, normalize_stages

ROOT = Path(__file__).resolve().parents[1]
LEDGER_SCHEMA = "staged_curriculum_ledger_v1"
DEFAULT_DATA_DIR = "data/staged_copy_v1"
DEFAULT_OUT_DIR = "checkpoints/staged_curriculum"
DEFAULT_RESULT = "research/results/staged_curriculum_run.json"
DEFAULT_STAGES = ",".join(STAGE_ORDER)
DEFAULT_FINAL_EVAL = "data/staged_copy_v1/mixed_eval.jsonl"
DEFAULT_MIN_CONTENT_DELTA = 0.05
DEFAULT_MAX_CONTENT_PPL_RATIO = 1.0
DRY_RUN_FINAL_EXAMPLES = 250
MAX_FAMILY_LENGTH = 64
DEFAULT_TRAINER = (sys.executable, str(ROOT / "research" /
                                      "train_adapter_candidate.py"))
DEFAULT_COMPARATOR = (sys.executable, str(ROOT / "research" /
                                         "compare_adapters.py"))
DEFAULT_GATE = (sys.executable, "-m", "atomzero.adapter_candidates",
                "evaluate")
DEFAULT_PROMOTE = (sys.executable, "-m", "atomzero.adapter_candidates",
                   "promote")

# Only these trainer-result and adapter-config fields are copied into the
# ledger.  Both files are known to be text-free, but a whitelist keeps the
# ledger stable if either producer grows new fields.
TRAINER_FIELDS = (
    "train_loss",
    "train_steps",
    "train_examples",
    "replay_examples",
    "replay_weight",
    "content_weight",
    "batch_size",
    "optimizer_steps",
    "rows_seen",
    "lr_scale",
    "effective_lr",
    "train_mix",
    "rank_attn",
    "alpha_attn",
    "rank_mlp",
    "alpha_mlp",
    "lora_parameters",
    "base_loss",
    "adapter_loss",
    "loss_delta",
    "loss_relative",
    "base_ppl",
    "adapter_ppl",
    "ppl_ratio",
    "eval_examples",
    "device",
    "decision",
    "weights_changed",
    "init_adapter",
)
CONFIG_FIELDS = (
    "r_mlp",
    "alpha_mlp",
    "r_attn",
    "alpha_attn",
    "rank",
    "alpha",
    "steps",
    "lr",
    "base",
    "train_examples",
    "replay_examples",
    "replay_weight",
    "content_weight",
    "batch_size",
    "optimizer_steps",
    "rows_seen",
    "lr_scale",
    "effective_lr",
    "method",
    "init_adapter",
)
LOSS_FIELDS = (
    "examples",
    "tokens",
    "loss",
    "ppl",
    "content_loss",
    "content_ppl",
    "content_tokens",
    "key_loss",
    "key_tokens",
    "value_loss",
    "value_tokens",
    "syntax_loss",
    "syntax_tokens",
)
COUNT_FIELDS = ("examples", "matches", "skipped", "rate", "key_recall")


class CurriculumError(RuntimeError):
    """A stage could not be trained or evaluated."""


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def _display_path(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(ROOT))
    except (OSError, ValueError):
        return str(path)


def _command(prefix: Sequence[str] | None, default: Sequence[str]
             ) -> list[str]:
    return [str(item) for item in (prefix if prefix is not None else default)]


def _child_env() -> dict[str, str]:
    """Make repository imports work in child processes regardless of cwd."""
    env = os.environ.copy()
    parts = [str(ROOT), str(ROOT / "src")]
    existing = env.get("PYTHONPATH", "")
    if existing:
        parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return env


def build_trainer_command(
    *,
    base: str,
    train: str | Path,
    eval: str | Path,
    out: str | Path,
    result: str | Path,
    comparison: str | Path,
    steps: int,
    batch_size: int,
    lr: float,
    content_weight: float,
    replay_weight: float,
    replay_jsonl: str | Path | None,
    rank: int,
    alpha: int,
    rank_attn: int,
    alpha_attn: int,
    seed: int,
    init_adapter: str | Path | None = None,
    command_prefix: Sequence[str] | None = None,
) -> list[str]:
    """Build the external trainer command for one stage."""
    command = _command(command_prefix, DEFAULT_TRAINER)
    command.extend([
        "--base", str(base),
        "--train", str(train),
        "--eval", str(eval),
        "--out", str(out),
        "--steps", str(steps),
        "--batch-size", str(batch_size),
        "--lr", str(lr),
        "--content-weight", str(content_weight),
        "--replay-weight", str(replay_weight),
        "--rank", str(rank),
        "--alpha", str(alpha),
        "--rank-attn", str(rank_attn),
        "--alpha-attn", str(alpha_attn),
        "--seed", str(seed),
        "--result", str(result),
        "--comparison", str(comparison),
    ])
    if replay_jsonl:
        command.extend(["--replay-jsonl", str(replay_jsonl)])
    if init_adapter:
        command.extend(["--init-adapter", str(init_adapter)])
    return command


def build_comparator_command(
    *,
    base: str,
    data: str | Path,
    adapter: str | Path,
    out: str | Path,
    command_prefix: Sequence[str] | None = None,
) -> list[str]:
    """Build the comparison command for one eval split and one adapter."""
    command = _command(command_prefix, DEFAULT_COMPARATOR)
    command.extend([
        "--base", str(base),
        "--data", str(data),
        "--adapters", str(adapter),
        "--per-family",
        "--exact-match",
        "--key-copy",
        "--out", str(out),
    ])
    return command


def build_gate_command(
    *,
    candidate_json: str | Path,
    comparison_json: str | Path,
    require_content: bool,
    min_content_delta: float,
    max_content_ppl_ratio: float,
    min_exact_match: float | None = None,
    min_key_copy_rate: float | None = None,
    command_prefix: Sequence[str] | None = None,
) -> list[str]:
    """Build the adapter-candidate gate command for the final eval."""
    command = _command(command_prefix, DEFAULT_GATE)
    command.extend([
        "--candidate-json", str(candidate_json),
        "--comparison-json", str(comparison_json),
        "--min-content-delta", str(min_content_delta),
        "--max-content-ppl-ratio", str(max_content_ppl_ratio),
    ])
    if require_content:
        command.append("--require-content")
    if min_exact_match is not None:
        command.extend(["--min-exact-match", str(min_exact_match)])
    if min_key_copy_rate is not None:
        command.extend(["--min-key-copy-rate", str(min_key_copy_rate)])
    command.append("--json")
    return command


def build_promote_command(
    *,
    candidate_json: str | Path,
    gate_json: str | Path,
    adapter_dir: str | Path,
    active_file: str | Path,
    command_prefix: Sequence[str] | None = None,
) -> list[str]:
    """Build the promotion command for a passing gate."""
    command = _command(command_prefix, DEFAULT_PROMOTE)
    command.extend([
        "--candidate-json", str(candidate_json),
        "--gate-json", str(gate_json),
        "--adapter-dir", str(adapter_dir),
        "--active-file", str(active_file),
        "--json",
    ])
    return command


def _number(value: Any) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value


def _aggregate_metrics(metrics: Any) -> dict[str, Any] | None:
    """Keep numeric aggregates only; never copy per-row text."""
    if not isinstance(metrics, Mapping):
        return None
    summary: dict[str, Any] = {}
    for field in LOSS_FIELDS:
        if field in metrics:
            value = _number(metrics.get(field))
            if value is not None:
                summary[field] = value
    exact = metrics.get("exact_match")
    if isinstance(exact, Mapping):
        rates = {field: _number(exact.get(field)) for field in COUNT_FIELDS}
        summary["exact_match"] = {
            key: value for key, value in rates.items() if value is not None}
        summary["exact_match_rate"] = rates.get("rate")
    copy = metrics.get("key_copy")
    if isinstance(copy, Mapping):
        rates = {field: _number(copy.get(field)) for field in COUNT_FIELDS}
        summary["key_copy"] = {
            key: value for key, value in rates.items() if value is not None}
        summary["key_copy_rate"] = rates.get("rate")
        summary["key_recall"] = rates.get("key_recall")
    return summary


def _extract_families(payload: Mapping[str, Any],
                      adapter: Mapping[str, Any] | None
                      ) -> dict[str, Any]:
    families: dict[str, Any] = {}
    raw = adapter.get("families") if isinstance(adapter, Mapping) else None
    if isinstance(raw, Mapping):
        for family, entry in raw.items():
            if not isinstance(entry, Mapping):
                continue
            families[str(family)] = {
                "baseline": _aggregate_metrics(entry.get("baseline")),
                "adapter": _aggregate_metrics(entry.get("adapter")),
            }
    metrics = adapter.get("metrics") if isinstance(adapter, Mapping) else None
    if not isinstance(metrics, Mapping):
        return families
    for field, source in (("exact_match", "exact_match"),
                          ("key_copy", "key_copy")):
        container = metrics.get(source)
        if not isinstance(container, Mapping):
            continue
        per_family = container.get("families")
        if not isinstance(per_family, Mapping):
            continue
        for family, entry in per_family.items():
            if not isinstance(entry, Mapping):
                continue
            values = {key: _number(entry.get(key)) for key in COUNT_FIELDS}
            bundle = {key: value for key, value in values.items()
                      if value is not None}
            families.setdefault(str(family), {"baseline": None,
                                              "adapter": None})[field] = bundle
    return families


def summarize_comparison(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Strip a compare_adapters payload down to content-free aggregates.

    ``compare_adapters.py`` includes full scored rows (``user`` and
    ``assistant`` text) in every model's ``metrics.rows``.  This function
    never reads those rows: it copies explicit numeric metric fields and
    per-family aggregates only.
    """
    models = payload.get("models") if isinstance(payload, Mapping) else None
    models = models if isinstance(models, list) else []
    base = next((item for item in models
                 if isinstance(item, Mapping)
                 and item.get("label") == "base"), None)
    adapter = next((item for item in models
                    if isinstance(item, Mapping)
                    and item.get("label") != "base"), None)
    return {
        "examples": _number(payload.get("examples")),
        "base": _aggregate_metrics(
            base.get("metrics") if isinstance(base, Mapping) else None),
        "adapter": _aggregate_metrics(
            adapter.get("metrics") if isinstance(adapter, Mapping) else None),
        "per_family": _extract_families(payload, adapter),
    }


def _row_digest(row: Mapping[str, Any]) -> str:
    payload = f"{row.get('user', '')}\0{row.get('assistant', '')}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _row_family(row: Mapping[str, Any]) -> str | None:
    value = row.get("family")
    if value is None:
        return None
    if not isinstance(value, str):
        raise CurriculumError("row family must be a string")
    family = value.strip()
    if not family:
        return None
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in family):
        raise CurriculumError("row family contains control characters")
    if len(family) > MAX_FAMILY_LENGTH:
        raise CurriculumError(
            f"row family exceeds {MAX_FAMILY_LENGTH} characters")
    return family


def _family_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        family = _row_family(row) or "unspecified"
        counts[family] = counts.get(family, 0) + 1
    return {name: counts[name] for name in sorted(counts)}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        raw = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise CurriculumError(f"cannot read {path}: {exc}") from exc
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CurriculumError(
                f"{path}:{number}: invalid JSON: {exc}") from exc
        if not isinstance(item, dict):
            raise CurriculumError(f"{path}:{number}: row is not an object")
        rows.append(item)
    return rows


def _string_set(rows: Sequence[Mapping[str, Any]],
                field: str) -> tuple[set[str], bool]:
    values: set[str] = set()
    present = False
    for row in rows:
        raw = row.get(field)
        if not isinstance(raw, list):
            continue
        present = True
        for value in raw:
            if isinstance(value, (str, int, float)) and not isinstance(
                    value, bool):
                values.add(str(value))
    return values, present


def _separation_evidence(
    train_rows: Sequence[Mapping[str, Any]],
    eval_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Check row/key/value holdout without keeping any text.

    The staged curriculum deliberately trains and evaluates on the same task
    family names, so family-name disjointness cannot be claimed here.  What
    can be checked is that every eval row has an explicit family label and
    that no eval row, key or value appears in any training split.
    """
    train_digests = {_row_digest(row) for row in train_rows}
    eval_digests = {_row_digest(row) for row in eval_rows}
    train_keys, train_has_keys = _string_set(train_rows, "keys")
    eval_keys, eval_has_keys = _string_set(eval_rows, "keys")
    train_values, train_has_values = _string_set(train_rows, "values")
    eval_values, eval_has_values = _string_set(eval_rows, "values")
    key_overlap = (len(train_keys & eval_keys)
                   if train_has_keys and eval_has_keys else None)
    value_overlap = (len(train_values & eval_values)
                     if train_has_values and eval_has_values else None)
    row_overlap = len(train_digests & eval_digests)
    families = [_row_family(row) for row in eval_rows]
    explicit_families = bool(families) and all(
        family and family != "unspecified" for family in families)
    separation = (
        explicit_families
        and row_overlap == 0
        and key_overlap in (0, None)
        and value_overlap in (0, None)
    )
    return {
        "family_separation": separation,
        "family_separation_basis": (
            "explicit_family_labels_and_held_out_row_key_value_separation_v1"),
        "explicit_families": explicit_families,
        "train_rows_checked": len(train_rows),
        "eval_rows_checked": len(eval_rows),
        "row_overlap": row_overlap,
        "key_overlap": key_overlap,
        "value_overlap": value_overlap,
    }


def _candidate_id(eval_rows: Sequence[Mapping[str, Any]],
                  stages: Sequence[str], seed: int) -> str:
    payload = "".join(_row_digest(row) for row in eval_rows)
    payload += "|" + ",".join(stages) + f"|{seed}"
    return "ad-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def build_candidate_metadata(
    *,
    eval_rows: Sequence[Mapping[str, Any]],
    train_rows: Sequence[Mapping[str, Any]],
    stages: Sequence[str],
    seed: int,
) -> dict[str, Any]:
    """Build the flat candidate metadata consumed by the gate and promote."""
    if not eval_rows:
        raise CurriculumError("candidate eval set is empty")
    eval_counts = _family_counts(eval_rows)
    train_counts = _family_counts(train_rows)
    evidence = _separation_evidence(train_rows, eval_rows)
    total = len(train_rows) + len(eval_rows)
    metadata = {
        "candidate_id": _candidate_id(eval_rows, stages, seed),
        "version": 1,
        "train_examples": len(train_rows),
        "eval_examples": len(eval_rows),
        "total_examples": total,
        "duplicates_skipped": 0,
        "eval_ratio": round(len(eval_rows) / total, 6) if total else 0.0,
        "seed": seed,
        "split_method": "staged_curriculum_final_eval",
        "train_families": sorted(train_counts),
        "eval_families": sorted(eval_counts),
        "train_family_counts": train_counts,
        "eval_family_counts": eval_counts,
        "min_loss_delta": 0.01,
        "max_ppl_ratio": 1.0,
        "auto_apply": False,
        "weights_changed": False,
        "method": "content_free_adapter_candidate_event_v1",
        **evidence,
    }
    if not evidence["family_separation"]:
        metadata["family_separation_reason"] = "held_out_separation_failed"
    return metadata


def _filter_fields(source: Any, fields: Sequence[str]) -> dict[str, Any]:
    if not isinstance(source, Mapping):
        return {}
    result: dict[str, Any] = {}
    for field in fields:
        value = source.get(field)
        if isinstance(value, bool) or isinstance(value, (int, float, str)):
            result[field] = value
    return result


def _read_json(path: Path, *, stage: str, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise CurriculumError(
            f"stage {stage}: {label} did not write {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise CurriculumError(
            f"stage {stage}: {label} wrote invalid JSON to {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise CurriculumError(
            f"stage {stage}: {label} output {path} is not a JSON object")
    return payload


def _run_unchecked(command: Sequence[str], *, stage: str, label: str
                   ) -> tuple[subprocess.CompletedProcess[str], float]:
    """Run a subprocess and return its result regardless of exit code."""
    started = time.monotonic()
    try:
        completed = subprocess.run(
            list(command), cwd=ROOT, capture_output=True, text=True,
            encoding="utf-8", errors="replace", env=_child_env())
    except OSError as exc:
        raise CurriculumError(
            f"stage {stage}: could not start {label}: {exc}") from exc
    return completed, time.monotonic() - started


def _failure_tail(completed: subprocess.CompletedProcess[str]) -> str:
    detail = (completed.stderr or completed.stdout or "").strip()
    return "\n".join(detail.splitlines()[-5:])


def _parse_json_stdout(completed: subprocess.CompletedProcess[str], *,
                       stage: str, label: str) -> dict[str, Any]:
    """Parse a content-free JSON object printed by a child command."""
    raw = (completed.stdout or "").strip()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CurriculumError(
            f"stage {stage}: {label} did not print valid JSON: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise CurriculumError(
            f"stage {stage}: {label} output is not a JSON object")
    return payload


def _run(command: Sequence[str], *, stage: str, label: str
         ) -> tuple[subprocess.CompletedProcess[str], float]:
    completed, elapsed = _run_unchecked(command, stage=stage, label=label)
    if completed.returncode != 0:
        message = (f"stage {stage}: {label} failed with returncode "
                   f"{completed.returncode}")
        tail = _failure_tail(completed)
        if tail:
            message = f"{message}: {tail}"
        raise CurriculumError(message)
    return completed, elapsed


def _dry_run_rate(stage_index: int) -> float:
    return min(0.9, round(0.1 + 0.15 * (stage_index - 1), 4))


def _dry_run_metrics(stage: str, stage_index: int, target: str
                     ) -> dict[str, Any]:
    rate = _dry_run_rate(stage_index)
    exact = round(rate * 0.5, 4)
    recall = min(1.0, round(rate + 0.12, 4))
    loss = round(4.2 - 0.45 * stage_index, 4)
    family = f"staged_copy_{'multi' if target == 'multi' else stage}"
    adapter = {
        "examples": 200,
        "tokens": 4000,
        "loss": loss,
        "ppl": round(math.exp(loss), 4),
        "content_loss": round(loss - 0.1, 4),
        "content_tokens": 3000,
        "key_loss": round(loss + 0.2, 4),
        "key_tokens": 500,
        "value_loss": round(loss + 0.1, 4),
        "value_tokens": 400,
        "syntax_loss": round(loss - 0.2, 4),
        "syntax_tokens": 100,
        "exact_match": {"examples": 200, "matches": int(200 * exact),
                        "rate": exact},
        "exact_match_rate": exact,
        "key_copy": {"examples": 200, "matches": int(200 * rate),
                     "rate": rate, "key_recall": recall},
        "key_copy_rate": rate,
        "key_recall": recall,
    }
    family_entry = {
        "baseline": {**adapter, "loss": round(loss + 0.5, 4)},
        "adapter": {**adapter, "loss": loss},
        "exact_match": {"examples": 200, "matches": int(200 * exact),
                        "rate": exact},
        "key_copy": {"examples": 200, "matches": int(200 * rate),
                     "rate": rate, "key_recall": recall},
    }
    base = dict(adapter)
    base.update({
        "loss": round(loss + 0.9, 4),
        "ppl": round(math.exp(loss + 0.9), 4),
        "exact_match": {"examples": 200, "matches": 0, "rate": 0.0},
        "exact_match_rate": 0.0,
        "key_copy": {"examples": 200, "matches": 0, "rate": 0.0,
                     "key_recall": 0.0},
        "key_copy_rate": 0.0,
        "key_recall": 0.0,
    })
    return {
        "examples": 200,
        "base": base,
        "adapter": adapter,
        "per_family": {family: family_entry},
    }


def _dry_run_final_metrics() -> dict[str, Any]:
    """A fake but gate-compatible final eval aggregate.

    The numbers mirror the manual 250-row mixed staged run: exact match 0.40,
    key copy 0.548, content loss 4.04 -> 1.00.
    """
    examples = DRY_RUN_FINAL_EXAMPLES
    loss = 0.9
    content_loss = 1.0
    adapter = {
        "examples": examples,
        "tokens": examples * 20,
        "loss": loss,
        "ppl": round(math.exp(loss), 4),
        "content_loss": content_loss,
        "content_ppl": round(math.exp(content_loss), 4),
        "content_tokens": examples * 16,
        "key_loss": round(loss + 0.2, 4),
        "key_tokens": examples * 2,
        "value_loss": round(loss + 0.1, 4),
        "value_tokens": examples,
        "syntax_loss": round(loss - 0.2, 4),
        "syntax_tokens": examples,
        "exact_match": {"examples": examples, "matches": 100, "rate": 0.4},
        "exact_match_rate": 0.4,
        "key_copy": {"examples": examples, "matches": 137, "rate": 0.548,
                     "key_recall": 0.62},
        "key_copy_rate": 0.548,
        "key_recall": 0.62,
    }
    base = dict(adapter)
    base.update({
        "loss": 4.1,
        "ppl": round(math.exp(4.1), 4),
        "content_loss": 4.04,
        "content_ppl": round(math.exp(4.04), 4),
        "exact_match": {"examples": examples, "matches": 0, "rate": 0.0},
        "exact_match_rate": 0.0,
        "key_copy": {"examples": examples, "matches": 0, "rate": 0.0,
                     "key_recall": 0.0},
        "key_copy_rate": 0.0,
        "key_recall": 0.0,
    })
    return {
        "examples": examples,
        "base": base,
        "adapter": adapter,
        "per_family": {
            "staged_copy_mixed": {
                "baseline": dict(base),
                "adapter": dict(adapter),
                "exact_match": adapter["exact_match"],
                "key_copy": adapter["key_copy"],
            },
        },
    }


def _dry_run_candidate(stages: Sequence[str], seed: int) -> dict[str, Any]:
    examples = DRY_RUN_FINAL_EXAMPLES
    counts = {"staged_copy_mixed": examples}
    return {
        "candidate_id": _candidate_id(
            [{"user": stage, "assistant": str(index)}
             for index, stage in enumerate(stages)], stages, seed),
        "version": 1,
        "train_examples": 2000 * max(1, len(stages)),
        "eval_examples": examples,
        "total_examples": 2000 * max(1, len(stages)) + examples,
        "duplicates_skipped": 0,
        "eval_ratio": 0.0,
        "seed": seed,
        "split_method": "staged_curriculum_final_eval",
        "train_families": ["staged_copy_mixed"],
        "eval_families": ["staged_copy_mixed"],
        "train_family_counts": {
            "staged_copy_mixed": 2000 * max(1, len(stages))},
        "eval_family_counts": counts,
        "min_loss_delta": 0.01,
        "max_ppl_ratio": 1.0,
        "auto_apply": False,
        "weights_changed": False,
        "method": "content_free_adapter_candidate_event_v1",
        "family_separation": True,
        "family_separation_basis": "dry_run",
        "explicit_families": True,
        "train_rows_checked": 0,
        "eval_rows_checked": examples,
        "row_overlap": 0,
        "key_overlap": 0,
        "value_overlap": 0,
    }


def _dry_run_gate(candidate: Mapping[str, Any],
                  comparison: Mapping[str, Any], *,
                  require_content: bool, min_content_delta: float,
                  max_content_ppl_ratio: float,
                  min_exact_match: float | None,
                  min_key_copy_rate: float | None) -> dict[str, Any]:
    """Evaluate the fake aggregate with the real gate implementation."""
    try:
        from atomzero.adapter_candidates import evaluate_candidate
    except ImportError:
        sys.path.insert(0, str(ROOT / "src"))
        from atomzero.adapter_candidates import evaluate_candidate
    return evaluate_candidate(
        dict(candidate), dict(comparison),
        require_content=require_content,
        min_content_delta=min_content_delta,
        max_content_ppl_ratio=max_content_ppl_ratio,
        min_exact_match=min_exact_match,
        min_key_copy_rate=min_key_copy_rate)


def _dry_run_trainer(stage_index: int, *, steps: int,
                     batch_size: int, lr: float, content_weight: float,
                     replay_weight: float, rank: int, alpha: int,
                     rank_attn: int, alpha_attn: int,
                     init_adapter: str | None) -> dict[str, Any]:
    optimizer_steps = max(1, math.ceil(steps / max(1, batch_size)))
    result = {
        "train_loss": round(4.0 - 0.4 * stage_index, 4),
        "train_steps": steps,
        "train_examples": steps,
        "replay_examples": 0,
        "replay_weight": replay_weight,
        "content_weight": content_weight,
        "batch_size": batch_size,
        "optimizer_steps": optimizer_steps,
        "rows_seen": steps,
        "lr_scale": "none",
        "effective_lr": lr,
        "train_mix": {"primary": steps, "replay": 0},
        "rank_attn": rank_attn,
        "alpha_attn": alpha_attn,
        "rank_mlp": rank,
        "alpha_mlp": alpha,
        "lora_parameters": 405504,
        "decision": "dry-run",
        "weights_changed": False,
    }
    if init_adapter:
        result["init_adapter"] = init_adapter
    config = {
        "r_mlp": rank,
        "alpha_mlp": alpha,
        "r_attn": rank_attn,
        "alpha_attn": alpha_attn,
        "rank": rank,
        "alpha": alpha,
        "steps": steps,
        "batch_size": batch_size,
        "method": "dry_run",
    }
    if init_adapter:
        config["init_adapter"] = init_adapter
    return {"result": result, "adapter_config": config}


def _final_eval_and_gate(
    *,
    base: str,
    data_root: Path,
    final_eval_path: Path,
    final_eval_explicit: bool,
    final_record: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    stages: Sequence[str],
    seed: int,
    dry_run: bool,
    tmp_root: Path,
    candidate_target: Path | None,
    gate_target: Path | None,
    active_target: Path | None,
    require_content: bool,
    min_exact_match: float | None,
    min_key_copy_rate: float | None,
    min_content_delta: float,
    max_content_ppl_ratio: float,
    gate_thresholds: Mapping[str, Any],
    promote: bool,
    gate_payload: Mapping[str, Any],
    promotion_record: Mapping[str, Any],
    comparator_command: Sequence[str] | None,
    gate_command: Sequence[str] | None,
    promote_command: Sequence[str] | None,
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any],
           dict[str, Any], str | None]:
    """Evaluate the final adapter on the held-out final eval and gate it."""
    final = dict(final_record)
    gate = dict(gate_payload)
    promotion = dict(promotion_record)
    promotion_error: str | None = None
    adapter_dir = _resolve(str(records[-1]["adapter"]))

    def write_gate() -> None:
        if gate_target is not None:
            _write_json(gate, gate_target)

    def finish() -> tuple[dict[str, Any], dict[str, Any] | None,
                          dict[str, Any], dict[str, Any], str | None]:
        nonlocal promotion_error
        write_gate()
        if not promote:
            promotion["reason"] = "not_requested"
        elif gate.get("status") != "pass" or gate.get("ok") is not True:
            promotion["reason"] = (
                "gate_failed" if gate.get("status") == "fail"
                else "gate_skipped")
        elif dry_run:
            promotion["reason"] = "dry_run"
        elif active_target is None:
            raise CurriculumError(
                "--active-file is required when --promote is set")
        else:
            command = build_promote_command(
                candidate_json=candidate_target,
                gate_json=gate_target,
                adapter_dir=adapter_dir,
                active_file=active_target,
                command_prefix=promote_command,
            )
            completed, elapsed = _run_unchecked(
                command, stage="final", label="promoter")
            promotion["command"] = command
            promotion["returncode"] = completed.returncode
            promotion["elapsed_seconds"] = round(elapsed, 3)
            if completed.returncode != 0:
                promotion_error = (
                    _failure_tail(completed)
                    or f"promotion failed with returncode "
                       f"{completed.returncode}")
                promotion["reason"] = "promote_failed"
                promotion["error"] = promotion_error
            else:
                payload = _parse_json_stdout(
                    completed, stage="final", label="promoter")
                promotion["promoted"] = True
                promotion["reason"] = "promoted"
                promotion["record"] = payload
        return final, candidate, gate, promotion, promotion_error

    candidate: dict[str, Any] | None = None
    if dry_run:
        comparison = _dry_run_final_metrics()
        candidate = _dry_run_candidate(stages, seed)
        if candidate_target is not None:
            _write_json(candidate, candidate_target)
        final.update({
            "examples": comparison["examples"],
            "command": build_comparator_command(
                base=base,
                data=final_eval_path,
                adapter=adapter_dir,
                out=tmp_root / "final_comparison.json",
                command_prefix=comparator_command),
            "returncode": 0,
            "elapsed_seconds": 0.0,
            "metrics": comparison,
        })
        adapter_metrics = comparison.get("adapter") or {}
        final["exact_match_rate"] = _number(
            adapter_metrics.get("exact_match_rate"))
        final["key_copy_rate"] = _number(adapter_metrics.get("key_copy_rate"))
        gate_input = {
            "examples": comparison["examples"],
            "adapter_label": "curriculum",
            "baseline": comparison["base"],
            "adapter": comparison["adapter"],
        }
        gate = {
            **_dry_run_gate(
                candidate,
                gate_input,
                require_content=require_content,
                min_content_delta=min_content_delta,
                max_content_ppl_ratio=max_content_ppl_ratio,
                min_exact_match=min_exact_match,
                min_key_copy_rate=min_key_copy_rate),
            "thresholds": dict(gate_thresholds),
            "command": None,
            "returncode": 0,
            "elapsed_seconds": 0.0,
        }
        return finish()

    if not final_eval_path.is_file():
        if final_eval_explicit:
            raise CurriculumError(
                f"final eval not found: {_display_path(final_eval_path)}")
        gate.update({
            "status": "skipped",
            "reasons": ["final_eval_missing"],
            "thresholds": dict(gate_thresholds),
            "command": None,
            "returncode": None,
            "elapsed_seconds": 0.0,
        })
        return finish()

    eval_rows = _read_jsonl(final_eval_path)
    if not eval_rows:
        gate.update({
            "status": "skipped",
            "reasons": ["final_eval_empty"],
            "thresholds": dict(gate_thresholds),
            "command": None,
            "returncode": None,
            "elapsed_seconds": 0.0,
        })
        return finish()

    comparator_out = tmp_root / "final_comparison.json"
    comparator_cmd = build_comparator_command(
        base=base,
        data=final_eval_path,
        adapter=adapter_dir,
        out=comparator_out,
        command_prefix=comparator_command,
    )
    _, comparator_elapsed = _run(
        comparator_cmd, stage="final", label="comparator (final eval)")
    if not comparator_out.is_file():
        raise CurriculumError(
            f"final comparator did not write {comparator_out}")
    payload = _read_json(
        comparator_out, stage="final", label="comparator (final eval)")
    summary = summarize_comparison(payload)
    final.update({
        "examples": summary.get("examples") or len(eval_rows),
        "command": comparator_cmd,
        "returncode": 0,
        "elapsed_seconds": round(comparator_elapsed, 3),
        "metrics": summary,
    })
    adapter_metrics = summary.get("adapter") or {}
    final["exact_match_rate"] = _number(
        adapter_metrics.get("exact_match_rate"))
    final["key_copy_rate"] = _number(adapter_metrics.get("key_copy_rate"))

    train_rows: list[dict[str, Any]] = []
    for stage in stages:
        train_path = data_root / f"stage_{stage}" / "train.jsonl"
        if train_path.is_file():
            train_rows.extend(_read_jsonl(train_path))
    candidate = build_candidate_metadata(
        eval_rows=eval_rows, train_rows=train_rows, stages=stages, seed=seed)
    if candidate_target is not None:
        _write_json(candidate, candidate_target)

    gate_input = {
        "examples": summary.get("examples"),
        "adapter_label": "curriculum",
        "baseline": summary.get("base"),
        "adapter": summary.get("adapter"),
    }
    comparison_path = tmp_root / "final_gate_comparison.json"
    _write_json(gate_input, comparison_path)
    gate_cmd = build_gate_command(
        candidate_json=candidate_target,
        comparison_json=comparison_path,
        require_content=require_content,
        min_content_delta=min_content_delta,
        max_content_ppl_ratio=max_content_ppl_ratio,
        min_exact_match=min_exact_match,
        min_key_copy_rate=min_key_copy_rate,
        command_prefix=gate_command,
    )
    completed, gate_elapsed = _run_unchecked(
        gate_cmd, stage="final", label="gate")
    if completed.returncode not in (0, 1):
        tail = _failure_tail(completed)
        message = (f"final gate failed with returncode "
                   f"{completed.returncode}")
        if tail:
            message = f"{message}: {tail}"
        raise CurriculumError(message)
    gate_result = _parse_json_stdout(
        completed, stage="final", label="gate")
    gate = {
        **gate_result,
        "thresholds": dict(gate_thresholds),
        "command": gate_cmd,
        "returncode": completed.returncode,
        "elapsed_seconds": round(gate_elapsed, 3),
    }
    return finish()


def run_curriculum(
    *,
    base: str,
    data_dir: str | Path,
    out_dir: str | Path,
    stages: Sequence[str] | str = STAGE_ORDER,
    steps: int = 2000,
    batch_size: int = 4,
    lr: float = 3e-4,
    content_weight: float = 2.0,
    replay_weight: float = 0.0,
    replay_jsonl: str | Path | None = None,
    rank: int = 8,
    alpha: int = 8,
    rank_attn: int = 4,
    alpha_attn: int = 8,
    seed: int = 23,
    stop_on_key_copy: float | None = None,
    final_eval: str | Path | None = None,
    require_content: bool = False,
    min_exact_match: float | None = None,
    min_key_copy_rate: float | None = None,
    min_content_delta: float = DEFAULT_MIN_CONTENT_DELTA,
    max_content_ppl_ratio: float = DEFAULT_MAX_CONTENT_PPL_RATIO,
    candidate_json: str | Path | None = None,
    gate_json: str | Path | None = None,
    active_file: str | Path | None = None,
    promote: bool = False,
    dry_run: bool = False,
    trainer_command: Sequence[str] | None = None,
    comparator_command: Sequence[str] | None = None,
    gate_command: Sequence[str] | None = None,
    promote_command: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run the stages, gate the final adapter and return the ledger."""
    order = normalize_stages(stages)
    data_root = _resolve(data_dir)
    out_root = _resolve(out_dir)
    replay_value = _resolve(replay_jsonl) if replay_jsonl else None
    started = time.monotonic()
    records: list[dict[str, Any]] = []
    stopped_early = False
    stop_stage: str | None = None
    previous_adapter: Path | None = None
    multi_eval = data_root / "stage_multi" / "eval.jsonl"
    if final_eval is not None:
        final_eval_path = _resolve(final_eval)
        final_eval_explicit = True
    else:
        preferred = data_root / "mixed_eval.jsonl"
        final_eval_path = preferred if preferred.is_file() else multi_eval
        final_eval_explicit = False
    final_eval_display = _display_path(final_eval_path)
    candidate_target = (_resolve(candidate_json) if candidate_json else
                        (None if dry_run
                         else out_root / "candidate.json"))
    gate_target = (_resolve(gate_json) if gate_json else
                   (None if dry_run else out_root / "gate.json"))
    active_target = _resolve(active_file) if active_file else None
    gate_thresholds = {
        "require_content": bool(require_content),
        "min_exact_match": min_exact_match,
        "min_key_copy_rate": min_key_copy_rate,
        "min_content_delta": min_content_delta,
        "max_content_ppl_ratio": max_content_ppl_ratio,
    }
    final_record: dict[str, Any] = {
        "data": final_eval_display,
        "examples": 0,
        "command": None,
        "returncode": None,
        "elapsed_seconds": 0.0,
        "metrics": None,
        "exact_match_rate": None,
        "key_copy_rate": None,
    }
    candidate_metadata: dict[str, Any] | None = None
    gate_payload: dict[str, Any] = {
        "status": "skipped",
        "ok": False,
        "promotable": False,
        "reasons": ["not_evaluated"],
        "thresholds": gate_thresholds,
    }
    promotion_record: dict[str, Any] = {
        "requested": bool(promote),
        "promoted": False,
        "active_file": _display_path(active_target),
        "reason": None,
        "command": None,
        "returncode": None,
        "elapsed_seconds": 0.0,
        "record": None,
    }
    promotion_error: str | None = None

    with tempfile.TemporaryDirectory(prefix="staged_curriculum_") as tmp:
        tmp_root = Path(tmp)
        for position, stage in enumerate(order):
            stage_index = STAGE_ORDER.index(stage) + 1
            stage_dir = out_root / stage
            train_path = data_root / f"stage_{stage}" / "train.jsonl"
            eval_path = data_root / f"stage_{stage}" / "eval.jsonl"
            if not dry_run:
                for path in (train_path, eval_path):
                    if not path.is_file():
                        raise CurriculumError(
                            f"stage {stage}: missing {_display_path(path)}; "
                            "build the staged queue first")
            trainer_result_path = stage_dir / "trainer_result.json"
            comparison_path = tmp_root / f"{stage}_trainer_comparison.json"
            trainer_cmd = build_trainer_command(
                base=base,
                train=train_path,
                eval=eval_path,
                out=stage_dir,
                result=trainer_result_path,
                comparison=comparison_path,
                steps=steps,
                batch_size=batch_size,
                lr=lr,
                content_weight=content_weight,
                replay_weight=replay_weight,
                replay_jsonl=replay_value,
                rank=rank,
                alpha=alpha,
                rank_attn=rank_attn,
                alpha_attn=alpha_attn,
                seed=seed,
                init_adapter=previous_adapter,
                command_prefix=trainer_command,
            )
            init_display = _display_path(previous_adapter)
            if dry_run:
                trainer_fields = _dry_run_trainer(
                    stage_index, steps=steps, batch_size=batch_size,
                    lr=lr, content_weight=content_weight,
                    replay_weight=replay_weight, rank=rank, alpha=alpha,
                    rank_attn=rank_attn, alpha_attn=alpha_attn,
                    init_adapter=init_display)
                trainer_elapsed = 0.0
                trainer_returncode = 0
            else:
                _, trainer_elapsed = _run(
                    trainer_cmd, stage=stage, label="trainer")
                trainer_returncode = 0
                if not trainer_result_path.is_file():
                    raise CurriculumError(
                        f"stage {stage}: trainer did not write "
                        f"{_display_path(trainer_result_path)}")
                raw_result = _read_json(
                    trainer_result_path, stage=stage, label="trainer")
                raw_config = None
                config_path = stage_dir / "config.json"
                if config_path.is_file():
                    raw_config = _read_json(
                        config_path, stage=stage, label="adapter config")
                trainer_fields = {
                    "result": _filter_fields(raw_result, TRAINER_FIELDS),
                    "adapter_config": _filter_fields(
                        raw_config, CONFIG_FIELDS),
                }
            trainer_record = {
                "returncode": trainer_returncode,
                "elapsed_seconds": round(trainer_elapsed, 3),
                "command": trainer_cmd,
                **trainer_fields,
            }

            evals: dict[str, Any] = {}
            targets: list[tuple[str, Path]] = [("stage", eval_path)]
            if stage != "multi":
                has_multi = (multi_eval.is_file() if not dry_run
                             else "multi" in order)
                if has_multi:
                    targets.append(("multi", multi_eval))
            for target, target_path in targets:
                comparator_out = tmp_root / f"{stage}_{target}_comparison.json"
                comparator_cmd = build_comparator_command(
                    base=base,
                    data=target_path,
                    adapter=stage_dir,
                    out=comparator_out,
                    command_prefix=comparator_command,
                )
                if dry_run:
                    summary = _dry_run_metrics(stage, stage_index, target)
                    eval_elapsed = 0.0
                    eval_returncode = 0
                else:
                    _, eval_elapsed = _run(
                        comparator_cmd, stage=stage,
                        label=f"comparator ({target} eval)")
                    eval_returncode = 0
                    if not comparator_out.is_file():
                        raise CurriculumError(
                            f"stage {stage}: comparator ({target} eval) did "
                            f"not write {comparator_out}")
                    payload = _read_json(
                        comparator_out, stage=stage,
                        label=f"comparator ({target} eval)")
                    summary = summarize_comparison(payload)
                evals[target] = {
                    "returncode": eval_returncode,
                    "elapsed_seconds": round(eval_elapsed, 3),
                    "data": _display_path(target_path),
                    "command": comparator_cmd,
                    "metrics": summary,
                }

            stage_metrics = evals.get("stage", {}).get("metrics", {})
            adapter_metrics = (stage_metrics.get("adapter")
                               if isinstance(stage_metrics, Mapping) else None)
            adapter_metrics = adapter_metrics or {}
            key_copy_rate = _number(adapter_metrics.get("key_copy_rate"))
            key_recall = _number(adapter_metrics.get("key_recall"))
            exact_match_rate = _number(
                adapter_metrics.get("exact_match_rate"))
            gate_passed = (stop_on_key_copy is not None
                           and key_copy_rate is not None
                           and key_copy_rate >= stop_on_key_copy)
            record = {
                "stage": stage,
                "index": stage_index,
                "adapter": _display_path(stage_dir),
                "init_adapter": init_display,
                "train_jsonl": _display_path(train_path),
                "eval_jsonl": _display_path(eval_path),
                "trainer": trainer_record,
                "evals": evals,
                "key_copy_rate": key_copy_rate,
                "key_recall": key_recall,
                "exact_match_rate": exact_match_rate,
                "key_copy_gate": (
                    {"threshold": stop_on_key_copy,
                     "passed": gate_passed}
                    if stop_on_key_copy is not None else None),
            }
            records.append(record)
            previous_adapter = stage_dir
            if gate_passed:
                stopped_early = True
                stop_stage = stage
                break

        if records:
            final_record, candidate_metadata, gate_payload, promotion_record, \
                promotion_error = _final_eval_and_gate(
                    base=base,
                    data_root=data_root,
                    final_eval_path=final_eval_path,
                    final_eval_explicit=final_eval_explicit,
                    final_record=final_record,
                    records=records,
                    stages=[record["stage"] for record in records],
                    seed=seed,
                    dry_run=dry_run,
                    tmp_root=tmp_root,
                    candidate_target=candidate_target,
                    gate_target=gate_target,
                    active_target=active_target,
                    require_content=require_content,
                    min_exact_match=min_exact_match,
                    min_key_copy_rate=min_key_copy_rate,
                    min_content_delta=min_content_delta,
                    max_content_ppl_ratio=max_content_ppl_ratio,
                    gate_thresholds=gate_thresholds,
                    promote=promote,
                    gate_payload=gate_payload,
                    promotion_record=promotion_record,
                    comparator_command=comparator_command,
                    gate_command=gate_command,
                    promote_command=promote_command,
                )

    return {
        "schema": LEDGER_SCHEMA,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dry_run": dry_run,
        "base": base,
        "data_dir": _display_path(data_root),
        "out_dir": _display_path(out_root),
        "stages_requested": list(order),
        "stage_order": [record["stage"] for record in records],
        "stop_on_key_copy": stop_on_key_copy,
        "stopped_early": stopped_early,
        "stop_stage": stop_stage,
        "final_eval": final_record,
        "candidate_id": (candidate_metadata or {}).get("candidate_id"),
        "candidate_json": (_display_path(candidate_target)
                           if candidate_metadata and candidate_target
                           else None),
        "gate": gate_payload,
        "gate_json": _display_path(gate_target) if gate_target else None,
        "exact_match_rate": final_record.get("exact_match_rate"),
        "key_copy_rate": final_record.get("key_copy_rate"),
        "promotion": promotion_record,
        "promotion_error": promotion_error,
        "active_file": _display_path(active_target),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "options": {
            "steps": steps,
            "batch_size": batch_size,
            "lr": lr,
            "content_weight": content_weight,
            "replay_weight": replay_weight,
            "replay_jsonl": _display_path(replay_value),
            "rank": rank,
            "alpha": alpha,
            "rank_attn": rank_attn,
            "alpha_attn": alpha_attn,
            "seed": seed,
            "final_eval": final_eval_display,
            "require_content": require_content,
            "min_exact_match": min_exact_match,
            "min_key_copy_rate": min_key_copy_rate,
            "min_content_delta": min_content_delta,
            "max_content_ppl_ratio": max_content_ppl_ratio,
            "promote": promote,
        },
        "stages": records,
    }


def _write_json(payload: Mapping[str, Any], path: str | Path) -> Path:
    target = _resolve(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    return target


def write_ledger(result: Mapping[str, Any], result_path: str | Path) -> Path:
    """Write the content-free ledger JSON and return its path."""
    return _write_json(result, result_path)


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid integer: {value!r}") from exc
    if number < 1:
        raise argparse.ArgumentTypeError("value must be >= 1")
    return number


def _finite_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid number: {value!r}") from exc
    if not math.isfinite(number):
        raise argparse.ArgumentTypeError("value must be finite")
    return number


def _bounded_weight(value: str) -> float:
    number = _finite_float(value)
    if not 0.0 <= number <= 1.0:
        raise argparse.ArgumentTypeError("value must be between 0.0 and 1.0")
    return number


def _gate(value: str) -> float:
    return _bounded_weight(value)


def _non_negative_float(value: str) -> float:
    number = _finite_float(value)
    if number < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return number


def _positive_float(value: str) -> float:
    number = _finite_float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return number


def _validate_options(parser: argparse.ArgumentParser,
                      args: argparse.Namespace) -> None:
    if not args.dry_run and not args.base:
        parser.error("--base is required unless --dry-run is set")
    if args.steps < 1:
        parser.error("--steps must be >= 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("--lr must be finite and > 0")
    if not math.isfinite(args.content_weight) or args.content_weight < 1.0:
        parser.error("--content-weight must be finite and >= 1.0")
    if args.replay_weight > 0 and not args.replay_jsonl:
        parser.error("--replay-jsonl is required when --replay-weight > 0")
    if args.rank < 0 or args.alpha < 0:
        parser.error("--rank and --alpha must be >= 0")
    if args.rank_attn < 0 or args.alpha_attn < 0:
        parser.error("--rank-attn and --alpha-attn must be >= 0")
    if args.rank > 0 and args.alpha <= 0:
        parser.error("--alpha must be > 0 when --rank > 0")
    if args.rank_attn > 0 and args.alpha_attn <= 0:
        parser.error("--alpha-attn must be > 0 when --rank-attn > 0")
    if args.promote and not args.active_file:
        parser.error("--active-file is required when --promote is set")
    if args.promote and not args.require_content:
        parser.error(
            "--require-content is required when --promote is set; promotion "
            "needs content_loss_improved and content_ppl_not_worse")
    if args.promote and args.min_exact_match is None:
        parser.error("--min-exact-match is required when --promote is set")
    if args.promote and args.min_key_copy_rate is None:
        parser.error("--min-key-copy-rate is required when --promote is set")


def main(argv: Sequence[str] | None = None, *,
         trainer_command: Sequence[str] | None = None,
         comparator_command: Sequence[str] | None = None,
         gate_command: Sequence[str] | None = None,
         promote_command: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--data-dir", default=DEFAULT_DATA_DIR,
        help=f"staged queue root (default: {DEFAULT_DATA_DIR})")
    parser.add_argument(
        "--out-dir", default=DEFAULT_OUT_DIR,
        help=f"adapter output root (default: {DEFAULT_OUT_DIR})")
    parser.add_argument(
        "--base", default="",
        help="base checkpoint directory (required unless --dry-run)")
    parser.add_argument(
        "--stages", default=DEFAULT_STAGES,
        help=("comma list of echo_kv,echo_json,two_kv,two_json,multi "
              f"(default: {DEFAULT_STAGES})"))
    parser.add_argument(
        "--steps", type=int, default=2000,
        help="sampled rows per stage (default: 2000)")
    parser.add_argument(
        "--batch-size", type=_positive_int, default=4,
        help="rows accumulated per optimizer step (default: 4)")
    parser.add_argument(
        "--lr", type=_finite_float, default=3e-4,
        help="learning rate (default: 3e-4)")
    parser.add_argument(
        "--content-weight", type=_finite_float, default=2.0,
        help="alphanumeric content loss multiplier (default: 2.0)")
    parser.add_argument(
        "--replay-weight", type=_bounded_weight, default=0.0,
        help="replay fraction mixed into every stage (default: 0.0)")
    parser.add_argument(
        "--replay-jsonl", default=None,
        help="optional older-distribution JSONL for replay")
    parser.add_argument(
        "--rank", type=int, default=8,
        help="LoRA MLP rank (default: 8)")
    parser.add_argument(
        "--alpha", type=int, default=8,
        help="LoRA MLP alpha (default: 8)")
    parser.add_argument(
        "--rank-attn", type=int, default=4,
        help="LoRA attention out_proj rank (default: 4)")
    parser.add_argument(
        "--alpha-attn", type=int, default=8,
        help="LoRA attention out_proj alpha (default: 8)")
    parser.add_argument(
        "--seed", type=int, default=23,
        help="training seed (default: 23)")
    parser.add_argument(
        "--result", default=DEFAULT_RESULT,
        help=f"content-free result JSON (default: {DEFAULT_RESULT})")
    parser.add_argument(
        "--stop-on-key-copy", type=_gate, default=None,
        help=("stop after the first stage whose held-out key-copy rate "
              "reaches this 0..1 threshold"))
    parser.add_argument(
        "--final-eval", default=None,
        help=("final held-out JSONL for the behavior gate "
              f"(default: {DEFAULT_FINAL_EVAL} when present, else the "
              "multi eval split)"))
    parser.add_argument(
        "--require-content", action="store_true",
        help="require content loss/perplexity improvements at the gate")
    parser.add_argument(
        "--min-exact-match", type=_gate, default=None,
        help="minimum final-eval exact-match rate (default: unset)")
    parser.add_argument(
        "--min-key-copy-rate", type=_gate, default=None,
        help="minimum final-eval key-copy rate (default: unset)")
    parser.add_argument(
        "--min-content-delta", type=_non_negative_float,
        default=DEFAULT_MIN_CONTENT_DELTA,
        help=("minimum content-loss improvement (default: "
              f"{DEFAULT_MIN_CONTENT_DELTA})"))
    parser.add_argument(
        "--max-content-ppl-ratio", type=_positive_float,
        default=DEFAULT_MAX_CONTENT_PPL_RATIO,
        help=("maximum content perplexity ratio (default: "
              f"{DEFAULT_MAX_CONTENT_PPL_RATIO})"))
    parser.add_argument(
        "--candidate-json", default=None,
        help="candidate metadata path (default: <out-dir>/candidate.json)")
    parser.add_argument(
        "--gate-json", default=None,
        help="gate output path (default: <out-dir>/gate.json)")
    parser.add_argument(
        "--active-file", default=None,
        help=("active-adapter registry to write when --promote passes; keep "
              "curriculum promotions in a separate registry"))
    parser.add_argument(
        "--promote", action="store_true",
        help="promote the final adapter when the gate passes")
    parser.add_argument(
        "--dry-run", action="store_true",
        help=("write fake metrics and skip trainer/comparator/gate "
              "subprocesses"))
    args = parser.parse_args(argv)
    _validate_options(parser, args)
    try:
        result = run_curriculum(
            base=args.base,
            data_dir=args.data_dir,
            out_dir=args.out_dir,
            stages=args.stages,
            steps=args.steps,
            batch_size=args.batch_size,
            lr=args.lr,
            content_weight=args.content_weight,
            replay_weight=args.replay_weight,
            replay_jsonl=args.replay_jsonl,
            rank=args.rank,
            alpha=args.alpha,
            rank_attn=args.rank_attn,
            alpha_attn=args.alpha_attn,
            seed=args.seed,
            stop_on_key_copy=args.stop_on_key_copy,
            final_eval=args.final_eval,
            require_content=args.require_content,
            min_exact_match=args.min_exact_match,
            min_key_copy_rate=args.min_key_copy_rate,
            min_content_delta=args.min_content_delta,
            max_content_ppl_ratio=args.max_content_ppl_ratio,
            candidate_json=args.candidate_json,
            gate_json=args.gate_json,
            active_file=args.active_file,
            promote=args.promote,
            dry_run=args.dry_run,
            trainer_command=trainer_command,
            comparator_command=comparator_command,
            gate_command=gate_command,
            promote_command=promote_command,
        )
    except (CurriculumError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    target = write_ledger(result, args.result)
    if not args.dry_run:
        write_ledger(result, _resolve(args.out_dir) / "ledger.json")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"wrote {_display_path(target)}")
    if result.get("promotion_error"):
        print(f"error: {result['promotion_error']}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
