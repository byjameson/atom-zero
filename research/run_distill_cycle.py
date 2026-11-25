"""Run one teacher-distillation cycle and record a content-free ledger.

The cycle has three stages:

1. **Teacher data build** (optional). ``--teacher-command`` is expanded with
   ``{source}``, ``{out_dir}`` and ``{result}`` and writes the verified teacher
   JSONL plus a manifest. The teacher never sees the held-out eval file.
2. **Student training.** ``--trainer-command`` is expanded with ``{base}``,
   ``{train}``, ``{eval}``, ``{out}``, ``{result}`` and ``{comparison}``. The
   student trains on the teacher data from stage 1.
3. **Held-out evaluation.** The evaluator compares the base checkpoint and the
   new adapter on the untouched eval file and records exact match, key copy,
   loss and content loss.

Everything the runner writes is **content-free**: no prompts, answers, key
names, values or scored rows. Raw comparison files are written only inside a
temporary directory and are removed before the runner returns. The teacher
manifest is read only for numeric counts; the teacher JSONL is never parsed.

The runner is deliberately a harness, not a trainer. It uses only the standard
library, never imports ``torch`` and never loads a model. Tests inject fake
commands and the CLI supports ``--dry-run``.

``--train`` is the raw source split consumed by the teacher build (the
``{source}`` placeholder). ``--teacher-data`` is the verified output the
student trains on (the ``{train}`` placeholder). The two files have different
roles and are deliberately not interchangeable.

Example::

    python research/run_distill_cycle.py \\
      --backend gpu \\
      --train data/distill_copy_v1/source_train.jsonl \\
      --teacher-command "python research/build_distill_copy.py --source {source} --out-dir {out_dir} --rows {steps} --seed {seed} --shuffle --fallback-oracle" \\
      --teacher-data data/distill_copy_v1/train_distill.jsonl \\
      --base checkpoints/transformer_behavior_50m_long \\
      --eval data/multi_short_v1/stage_2short/eval.jsonl \\
      --eval data/multi_short_v1/stage_3short/eval.jsonl \\
      --eval data/multi_short_v1/stage_4short/eval.jsonl \\
      --out checkpoints/distill_cycle \\
      --steps 2000 --batch-size 4 --lr 3e-4 \\
      --content-weight 2 --key-weight 2 --value-weight 2 --seed 23 \\
      --result research/results/distill_cycle_run.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
LEDGER_SCHEMA = "distill_cycle_ledger_v1"

DEFAULT_BASE = "checkpoints/transformer_behavior_50m_long"
DEFAULT_TRAIN = "data/distill_copy_v1/source_train.jsonl"
DEFAULT_TEACHER_DATA = "data/distill_copy_v1/train_distill.jsonl"
DEFAULT_OUT = "checkpoints/distill_cycle"
DEFAULT_RESULT = "research/results/distill_cycle_run.json"
DEFAULT_BACKEND = "local"

STATUS_COMPLETED = "completed"
STATUS_TEACHER_FAILED = "teacher_failed"
STATUS_TRAINER_FAILED = "trainer_failed"
STATUS_EVAL_FAILED = "eval_failed"

DEFAULT_TRAINER_COMMAND = (
    sys.executable,
    str(ROOT / "research" / "train_adapter_candidate.py"),
    "--base", "{base}",
    "--train", "{train}",
    "--eval", "{eval}",
    "--out", "{out}",
    "--steps", "{steps}",
    "--batch-size", "{batch_size}",
    "--lr", "{lr}",
    "--content-weight", "{content_weight}",
    "--key-weight", "{key_weight}",
    "--value-weight", "{value_weight}",
    "--seed", "{seed}",
    "--result", "{result}",
    "--comparison", "{comparison}",
)
DEFAULT_EVALUATOR_COMMAND = (
    sys.executable,
    str(ROOT / "research" / "compare_adapters.py"),
    "--base", "{base}",
    "--data", "{eval}",
    "--adapters", "{out}",
    "--per-family",
    "--exact-match",
    "--key-copy",
    "--out", "{comparison}",
)

# Metric fields copied out of comparison reports. ``compare_adapters.py``
# stores full scored rows under ``metrics.rows``; nothing here reads that key.
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
COPY_FIELDS = ("examples", "matches", "skipped", "rate", "key_recall")

# Trainer outputs are whitelisted as well. A trainer could grow a debug field
# containing a prompt; the ledger must remain content-free even then.
TRAINER_RESULT_FIELDS = (
    "train_loss",
    "train_steps",
    "train_examples",
    "eval_examples",
    "tokens",
    "rows_seen",
    "optimizer_steps",
    "batch_size",
    "content_weight",
    "key_weight",
    "value_weight",
    "replay_examples",
    "replay_weight",
    "rank_mlp",
    "alpha_mlp",
    "rank_attn",
    "alpha_attn",
    "lora_parameters",
    "base_loss",
    "adapter_loss",
    "loss_delta",
    "loss_relative",
    "base_ppl",
    "adapter_ppl",
    "ppl_ratio",
    "effective_lr",
    "lr_scale",
    "device",
    "decision",
    "method",
    "weights_changed",
    "init_adapter",
)
TRAINER_CONFIG_FIELDS = (
    "r_mlp",
    "alpha_mlp",
    "r_attn",
    "alpha_attn",
    "rank",
    "alpha",
    "steps",
    "lr",
    "batch_size",
    "content_weight",
    "key_weight",
    "value_weight",
    "optimizer_steps",
    "rows_seen",
    "lr_scale",
    "effective_lr",
    "method",
    "init_adapter",
)

# The teacher manifest is read only for numeric counts. These containers are
# traversed and these leaf names are kept; every other field is ignored.
COUNT_CONTAINERS = frozenset({
    "rows", "counts", "totals", "summary", "stats", "source", "teacher",
    "verification", "fallback", "oracle", "split", "splits", "data",
    "by_format", "by_stage", "by_split", "by_status",
})
COUNT_LEAF_CONTAINERS = frozenset({
    "rows", "counts", "totals", "by_format", "by_stage", "by_split",
    "by_status",
})
COUNT_KEYS = frozenset({
    "rows", "examples", "accepted", "accepted_rows", "rejected",
    "rejected_rows", "verified", "verified_rows", "invalid", "invalid_rows",
    "skipped", "skipped_rows", "duplicates", "duplicate_rows", "fallback",
    "fallbacks", "fallback_oracle", "oracle_fallback", "repaired",
    "repaired_rows", "parsed", "parse_failures", "format_errors",
    "key_mismatches", "value_mismatches", "total", "total_rows",
    "source_rows", "teacher_rows", "train_rows", "eval_rows",
    "requested", "requested_rows", "generated", "generated_rows",
    "teacher_valid", "teacher_invalid", "kept_rows", "dropped_rows",
    "empty", "failed", "failures", "count", "counts",
})
COUNT_SUFFIXES = ("_rows", "_count", "_counts", "_examples", "_total")


class DistillCycleError(RuntimeError):
    """The cycle could not be configured or the ledger could not be written."""


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def _display_path(path: str | Path | None) -> str | None:
    if path is None:
        return None
    value = Path(path)
    try:
        return str(value.resolve().relative_to(ROOT))
    except (OSError, ValueError):
        return str(value)


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


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        return value[1:-1]
    return value


def _split_template(template: str) -> list[str]:
    """Split a shell-style command without destroying Windows backslashes."""
    lexer = shlex.shlex(template, posix=False)
    lexer.whitespace_split = True
    lexer.commenters = ""
    return [_strip_quotes(part) for part in lexer]


def format_command(
    template: Sequence[str] | str,
    values: Mapping[str, Any],
    *,
    label: str = "command",
) -> list[str]:
    """Expand placeholders in one command template into an argv list."""
    if isinstance(template, str):
        parts = _split_template(template)
    else:
        parts = [str(item) for item in template]
    if not parts:
        raise DistillCycleError(f"{label} template is empty")
    expanded: list[str] = []
    for part in parts:
        try:
            expanded.append(str(part).format(**values))
        except (KeyError, IndexError, ValueError) as exc:
            raise DistillCycleError(
                f"invalid {label} template: {exc}") from exc
    return expanded


def _uses_placeholder(template: Sequence[str] | str, name: str) -> bool:
    if isinstance(template, str):
        parts = _split_template(template)
    else:
        parts = [str(item) for item in template]
    needle = "{" + name + "}"
    return any(needle in part for part in parts)


def _optional_command(value: Sequence[str] | str | None
                      ) -> Sequence[str] | str | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    if not isinstance(value, str):
        items = list(value)
        if not items:
            return None
    return value


def _number(value: Any) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _integer(value: Any) -> int | None:
    number = _number(value)
    if number is None or isinstance(number, float):
        return None
    return int(number)


def _file_size(path: Path) -> int | None:
    try:
        return path.stat().st_size
    except OSError:
        return None


def _manifest_candidate_paths(teacher_data: Path) -> list[Path]:
    if teacher_data.is_dir():
        parent = teacher_data
        stem = "teacher"
    else:
        parent = teacher_data.parent
        stem = teacher_data.stem
    return [
        parent / "manifest.json",
        parent / "teacher_manifest.json",
        parent / f"{stem}.manifest.json",
        parent / "distill_manifest.json",
    ]


def find_teacher_manifest(teacher_data: str | Path) -> Path | None:
    """Return the first existing teacher manifest next to the JSONL."""
    path = _resolve(teacher_data)
    for candidate in _manifest_candidate_paths(path):
        if candidate.is_file():
            return candidate
    return None


def _clear_teacher_outputs(data_path: Path, manifest_path: Path,
                           source: Path | None) -> None:
    """Remove stale teacher output before a rebuild, never the source."""
    targets = [data_path, manifest_path]
    targets.extend(_manifest_candidate_paths(data_path))
    seen: set[Path] = set()
    for target in targets:
        if target in seen or target == source:
            continue
        seen.add(target)
        if not target.is_file():
            continue
        try:
            target.unlink()
        except OSError:
            pass


def _is_count_key(key: str) -> bool:
    lowered = key.strip().lower()
    return lowered in COUNT_KEYS or lowered.endswith(COUNT_SUFFIXES)


def _collect_counts(payload: Any, *, prefix: str = "",
                    accept_all: bool = False) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not isinstance(payload, Mapping):
        return counts
    for raw_key, raw_value in payload.items():
        key = str(raw_key)
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(raw_value, Mapping):
            child_all = (
                key.lower() in COUNT_LEAF_CONTAINERS
                or (accept_all and key.lower() not in {"paths", "files"})
            )
            recurse = (
                key.lower() in COUNT_CONTAINERS
                or accept_all
                or _is_count_key(key)
            )
            if recurse:
                counts.update(_collect_counts(
                    raw_value, prefix=path, accept_all=child_all))
            continue
        if isinstance(raw_value, (list, tuple)):
            continue
        value = _integer(raw_value)
        if value is None:
            continue
        if accept_all or _is_count_key(key):
            counts[path] = value
    return counts


def _empty_manifest(path: str | Path | None,
                    error: str | None = None) -> dict[str, Any]:
    return {
        "path": _display_path(path),
        "found": False,
        "counts": {},
        "error": error,
    }


def read_teacher_manifest(path: str | Path | None) -> dict[str, Any]:
    """Read a manifest for numeric counts only; never keep other fields."""
    if path is None:
        return _empty_manifest(None)
    target = _resolve(path)
    record: dict[str, Any] = {
        "path": _display_path(target),
        "found": target.is_file(),
        "counts": {},
        "error": None,
    }
    if not target.is_file():
        return record
    try:
        payload = json.loads(target.read_text(encoding="utf-8-sig"))
    except OSError as exc:
        record["error"] = f"read_error: {type(exc).__name__}"
        return record
    except json.JSONDecodeError:
        record["error"] = "invalid_json"
        return record
    if not isinstance(payload, Mapping):
        record["error"] = "manifest_not_object"
        return record
    record["counts"] = _collect_counts(payload)
    return record


def _invoke(command: Sequence[str], *, runner=None
            ) -> tuple[Any | None, float, str | None]:
    """Run one child command and return (completed, elapsed, spawn_error)."""
    started = time.monotonic()
    invoke = runner or subprocess.run
    try:
        completed = invoke(
            list(command), cwd=str(ROOT), capture_output=True, text=True,
            encoding="utf-8", errors="replace", env=_child_env())
    except Exception as exc:  # command failures are recorded, not raised
        return None, round(time.monotonic() - started, 3), (
            f"{type(exc).__name__}: {exc}")
    return completed, round(time.monotonic() - started, 3), None


def _returncode(completed: Any) -> int | None:
    value = getattr(completed, "returncode", None)
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _read_json_mapping(path: Path) -> tuple[dict[str, Any] | None,
                                             str | None]:
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        return None, f"read_error: {type(exc).__name__}"
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None, "invalid_json"
    if not isinstance(payload, Mapping):
        return None, "not_an_object"
    return dict(payload), None


def _parse_json_object(text: str | None) -> dict[str, Any] | None:
    """Parse a JSON object from a whole stdout blob or its last JSON line."""
    if not isinstance(text, str):
        return None
    stripped = text.strip()
    if not stripped:
        return None
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, Mapping):
        return dict(payload)
    for line in reversed(stripped.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, Mapping):
            return dict(candidate)
    return None


def _filter_scalar_fields(source: Any, fields: Sequence[str]
                          ) -> dict[str, Any]:
    if not isinstance(source, Mapping):
        return {}
    result: dict[str, Any] = {}
    for field in fields:
        value = source.get(field)
        if isinstance(value, float) and not math.isfinite(value):
            continue
        if isinstance(value, bool) or isinstance(value, (int, float, str)):
            result[field] = value
    return result


def _filter_trainer_result(source: Any) -> dict[str, Any]:
    result = _filter_scalar_fields(source, TRAINER_RESULT_FIELDS)
    for field, value in list(result.items()):
        if isinstance(value, (str, bool)):
            continue
        if _number(value) is None:
            result.pop(field)
    return result


def _aggregate_metrics(metrics: Any) -> dict[str, Any] | None:
    """Keep numeric aggregates only; never copy per-row text."""
    if not isinstance(metrics, Mapping):
        return None
    summary: dict[str, Any] = {}
    for field in LOSS_FIELDS:
        value = _number(metrics.get(field))
        if value is not None:
            summary[field] = value
    exact = metrics.get("exact_match")
    if isinstance(exact, Mapping):
        values = {field: _number(exact.get(field)) for field in COPY_FIELDS}
        kept = {key: value for key, value in values.items()
                if value is not None}
        summary["exact_match"] = kept
        summary["exact_match_rate"] = values.get("rate")
    else:
        rate = _number(metrics.get("exact_match_rate"))
        if rate is not None:
            summary["exact_match_rate"] = rate
    copy = metrics.get("key_copy")
    if isinstance(copy, Mapping):
        values = {field: _number(copy.get(field)) for field in COPY_FIELDS}
        kept = {key: value for key, value in values.items()
                if value is not None}
        summary["key_copy"] = kept
        summary["key_copy_rate"] = values.get("rate")
        summary["key_recall"] = values.get("key_recall")
    else:
        rate = _number(metrics.get("key_copy_rate"))
        if rate is not None:
            summary["key_copy_rate"] = rate
        recall = _number(metrics.get("key_recall"))
        if recall is not None:
            summary["key_recall"] = recall
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
                "base": _aggregate_metrics(entry.get("baseline")),
                "adapter": _aggregate_metrics(entry.get("adapter")),
            }
    metrics = adapter.get("metrics") if isinstance(adapter, Mapping) else None
    if not isinstance(metrics, Mapping):
        return families
    for field in ("exact_match", "key_copy"):
        container = metrics.get(field)
        if not isinstance(container, Mapping):
            continue
        per_family = container.get("families")
        if not isinstance(per_family, Mapping):
            continue
        for family, entry in per_family.items():
            if not isinstance(entry, Mapping):
                continue
            values = {key: _number(entry.get(key)) for key in COPY_FIELDS}
            bundle = {key: value for key, value in values.items()
                      if value is not None}
            families.setdefault(str(family), {"base": None, "adapter": None})
            families[str(family)][field] = bundle
    return families


def _delta(base: Mapping[str, Any] | None,
           adapter: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(base, Mapping) or not isinstance(adapter, Mapping):
        return None
    result: dict[str, Any] = {}
    for field in ("loss", "content_loss", "exact_match_rate",
                  "key_copy_rate"):
        before = _number(base.get(field))
        after = _number(adapter.get(field))
        if before is None or after is None:
            result[field] = None
        else:
            result[field] = round(float(after) - float(before), 6)
    return result


def summarize_evaluation(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Strip a compare_adapters payload down to content-free aggregates."""
    models = payload.get("models") if isinstance(payload, Mapping) else None
    models = models if isinstance(models, list) else []
    base_entry = next(
        (item for item in models
         if isinstance(item, Mapping) and item.get("label") == "base"), None)
    adapter_entry = next(
        (item for item in models
         if isinstance(item, Mapping) and item.get("label") != "base"), None)
    if base_entry is None and "base" in payload:
        base_entry = {"metrics": payload.get("base")}
    if adapter_entry is None and "adapter" in payload:
        adapter_entry = {"metrics": payload.get("adapter")}
    if base_entry is None and "baseline" in payload:
        base_entry = {"metrics": payload.get("baseline")}
    base = _aggregate_metrics(
        base_entry.get("metrics") if isinstance(base_entry, Mapping) else None)
    adapter = _aggregate_metrics(
        adapter_entry.get("metrics")
        if isinstance(adapter_entry, Mapping) else None)
    if base is None and "baseline" in payload:
        base = _aggregate_metrics(payload.get("baseline"))
    if adapter is None and "adapter" in payload:
        adapter = _aggregate_metrics(payload.get("adapter"))
    examples = _number(payload.get("examples"))
    if examples is None and adapter is not None:
        examples = adapter.get("examples")
    return {
        "examples": examples,
        "base": base,
        "adapter": adapter,
        "delta": _delta(base, adapter),
        "per_family": _extract_families(payload, adapter_entry),
    }


def _count_rows(path: Path) -> int | None:
    try:
        raw = path.read_text(encoding="utf-8-sig")
    except OSError:
        return None
    return sum(1 for line in raw.splitlines() if line.strip())


def _combine_eval(paths: Sequence[Path], tmp_root: Path
                  ) -> tuple[Path | None, str | None]:
    """Merge held-out files into one temporary JSONL without keeping text."""
    if len(paths) == 1:
        return paths[0], None
    lines: list[str] = []
    for source in paths:
        try:
            raw = source.read_text(encoding="utf-8-sig")
        except OSError:
            return None, "eval_missing"
        lines.extend(line for line in raw.splitlines() if line.strip())
    if not lines:
        return None, "eval_empty"
    target = tmp_root / "holdout_eval.combined.jsonl"
    try:
        target.write_text("".join(line + "\n" for line in lines),
                          encoding="utf-8")
    except OSError:
        return None, "eval_read_error"
    return target, None


def _limited_eval(eval_path: str | Path, max_rows: int | None,
                  tmp_root: Path) -> tuple[Path | None, int | None,
                                           int | None, str | None]:
    """Return (input path, rows available, rows used, error)."""
    source = _resolve(eval_path)
    available = _count_rows(source)
    if available is None:
        return None, None, None, "eval_missing"
    if available == 0:
        return None, 0, 0, "eval_empty"
    if max_rows is None or available <= max_rows:
        return source, available, available, None
    target = tmp_root / "holdout_eval.limited.jsonl"
    try:
        raw = source.read_text(encoding="utf-8-sig")
        kept = [line for line in raw.splitlines() if line.strip()][:max_rows]
        target.write_text("".join(line + "\n" for line in kept),
                          encoding="utf-8")
    except OSError:
        return None, available, None, "eval_read_error"
    return target, available, len(kept), None


def _planned_teacher_data(teacher_data: str | Path | None,
                          teacher_command: Sequence[str] | str | None
                          ) -> Path:
    if teacher_data:
        return _resolve(teacher_data)
    if teacher_command:
        return _resolve(DEFAULT_TEACHER_DATA)
    raise DistillCycleError(
        "--teacher-data is required unless --teacher-command is set")


def _teacher_record(*, requested: bool, source: Path | None,
                    data: Path, manifest_path: Path) -> dict[str, Any]:
    return {
        "requested": requested,
        "status": "skipped",
        "returncode": None,
        "elapsed_seconds": 0.0,
        "command": None,
        "source": _display_path(source),
        "data": _display_path(data),
        "data_bytes": None,
        "manifest": _empty_manifest(manifest_path),
    }


def _trainer_record(*, command: Sequence[str], out_dir: Path,
                    result_path: Path) -> dict[str, Any]:
    return {
        "status": "skipped",
        "returncode": None,
        "elapsed_seconds": 0.0,
        "command": list(command),
        "adapter": _display_path(out_dir),
        "adapter_exists": None,
        "result_path": _display_path(result_path),
        "result_found": False,
        "result": None,
        "adapter_config": None,
    }


def _evaluation_record(*, command: Sequence[str],
                       sources: Sequence[Path], data: Path | None,
                       max_rows: int | None,
                       rows_available: int | None,
                       rows_used: int | None) -> dict[str, Any]:
    return {
        "status": "skipped",
        "returncode": None,
        "elapsed_seconds": 0.0,
        "command": list(command),
        "source": _display_path(sources[0]) if sources else None,
        "sources": [_display_path(source) for source in sources],
        "data": _display_path(data),
        "max_eval_rows": max_rows,
        "rows_available": rows_available,
        "rows_used": rows_used,
        "examples": None,
        "base": None,
        "adapter": None,
        "delta": None,
        "per_family": {},
        "exact_match_rate": None,
        "key_copy_rate": None,
        "loss": None,
        "content_loss": None,
        "synthetic": False,
    }


def _dry_run_metrics() -> dict[str, Any]:
    """Fake, clearly marked metrics for a dry run."""
    examples = 250
    adapter = {
        "examples": examples,
        "tokens": examples * 20,
        "loss": 0.97,
        "ppl": round(math.exp(0.97), 4),
        "content_loss": 1.0,
        "content_tokens": examples * 16,
        "key_loss": 1.2,
        "key_tokens": examples * 2,
        "value_loss": 1.1,
        "value_tokens": examples,
        "syntax_loss": 0.8,
        "syntax_tokens": examples,
        "exact_match": {"examples": examples, "matches": 100, "rate": 0.4},
        "exact_match_rate": 0.4,
        "key_copy": {"examples": examples, "matches": 137, "rate": 0.548,
                     "key_recall": 0.628},
        "key_copy_rate": 0.548,
        "key_recall": 0.628,
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
        "delta": _delta(base, adapter),
        "per_family": {
            "distill_copy_dry_run": {
                "base": dict(base),
                "adapter": dict(adapter),
                "exact_match": adapter["exact_match"],
                "key_copy": adapter["key_copy"],
            },
        },
    }


def run_cycle(
    *,
    base: str,
    eval_path: str | Sequence[str],
    out_dir: str,
    result_path: str | None = None,
    teacher_data: str | None = None,
    train: str | None = None,
    teacher_command: Sequence[str] | str | None = None,
    trainer_command: Sequence[str] | str | None = None,
    evaluator_command: Sequence[str] | str | None = None,
    steps: int = 2000,
    batch_size: int = 4,
    lr: float = 3e-4,
    content_weight: float = 2.0,
    key_weight: float = 2.0,
    value_weight: float = 2.0,
    seed: int = 23,
    dry_run: bool = False,
    max_eval_rows: int | None = None,
    backend: str = DEFAULT_BACKEND,
    runner=None,
) -> dict[str, Any]:
    """Run the teacher, trainer and evaluator stages; return the ledger."""
    started = time.monotonic()
    teacher_command = _optional_command(teacher_command)
    trainer_command = _optional_command(trainer_command)
    evaluator_command = _optional_command(evaluator_command)

    if backend not in {"local", "gpu"}:
        raise DistillCycleError(f"unknown backend: {backend!r}")
    if steps < 1:
        raise DistillCycleError("--steps must be >= 1")
    if batch_size < 1:
        raise DistillCycleError("--batch-size must be >= 1")
    if not math.isfinite(lr) or lr <= 0:
        raise DistillCycleError("--lr must be finite and > 0")
    if not math.isfinite(content_weight) or content_weight < 1.0:
        raise DistillCycleError("--content-weight must be finite and >= 1.0")
    if not math.isfinite(key_weight) or key_weight < 0:
        raise DistillCycleError("--key-weight must be finite and >= 0")
    if not math.isfinite(value_weight) or value_weight < 0:
        raise DistillCycleError("--value-weight must be finite and >= 0")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise DistillCycleError("--seed must be an integer")
    if max_eval_rows is not None and max_eval_rows < 1:
        raise DistillCycleError("--max-eval-rows must be >= 1")

    base_path = _resolve(base) if base else _resolve(DEFAULT_BASE)
    if isinstance(eval_path, (str, Path)):
        eval_values = [eval_path]
    else:
        eval_values = [str(item) for item in eval_path]
    if not eval_values:
        raise DistillCycleError("--eval requires at least one path")
    eval_sources = [_resolve(value) for value in eval_values]
    out_path = _resolve(out_dir) if out_dir else _resolve(DEFAULT_OUT)
    source_train = _resolve(train) if train else None
    if teacher_data is None and teacher_command is None and dry_run:
        teacher_data = DEFAULT_TEACHER_DATA
    data_path = _planned_teacher_data(teacher_data, teacher_command)
    manifest_path = data_path.parent / "manifest.json"
    result_target = _resolve(result_path) if result_path else None

    settings = {
        "steps": steps,
        "batch_size": batch_size,
        "lr": lr,
        "content_weight": content_weight,
        "key_weight": key_weight,
        "value_weight": value_weight,
        "seed": seed,
        "backend": backend,
        "max_eval_rows": max_eval_rows,
        "dry_run": bool(dry_run),
    }

    with tempfile.TemporaryDirectory(prefix="distill_cycle_") as tmp:
        tmp_root = Path(tmp)
        trainer_result_path = out_path / "trainer_result.json"
        trainer_comparison_path = tmp_root / "trainer_comparison.json"
        eval_comparison_path = tmp_root / "eval_comparison.json"

        if not dry_run:
            eval_merged, combine_error = _combine_eval(
                eval_sources, tmp_root)
            if combine_error is not None:
                eval_input = None
                rows_available = None
                rows_used = None
                eval_error = combine_error
            else:
                eval_input, rows_available, rows_used, eval_error = (
                    _limited_eval(eval_merged, max_eval_rows, tmp_root))
        else:
            eval_merged = (eval_sources[0] if len(eval_sources) == 1
                           else tmp_root / "holdout_eval.combined.jsonl")
            eval_input = eval_merged
            rows_available, rows_used, eval_error = None, None, None

        common_values = {
            "base": str(base_path),
            "source": str(source_train) if source_train else "",
            "train": str(data_path),
            "teacher_data": str(data_path),
            "eval": str(eval_merged or eval_sources[0]),
            "eval_source": str(eval_merged or eval_sources[0]),
            "out": str(out_path),
            "out_dir": str(data_path.parent),
            "comparison": str(trainer_comparison_path),
            "steps": str(steps),
            "batch_size": str(batch_size),
            "lr": str(lr),
            "content_weight": str(content_weight),
            "key_weight": str(key_weight),
            "value_weight": str(value_weight),
            "seed": str(seed),
            "backend": backend,
            "max_eval_rows": "" if max_eval_rows is None else str(
                max_eval_rows),
        }

        teacher_argv: list[str] | None = None
        teacher_uses_source = (
            teacher_command is not None
            and _uses_placeholder(teacher_command, "source"))
        if teacher_command is not None:
            # Deliberately omit {eval}: held-out prompts must never reach the
            # teacher. A template that asks for it fails as invalid.
            teacher_argv = format_command(
                teacher_command,
                {
                    "source": common_values["source"],
                    "out_dir": common_values["out_dir"],
                    "result": str(manifest_path),
                    "teacher_data": str(data_path),
                    "base": str(base_path),
                    "out": str(out_path),
                    "steps": str(steps),
                    "batch_size": str(batch_size),
                    "lr": str(lr),
                    "content_weight": str(content_weight),
                    "key_weight": str(key_weight),
                    "value_weight": str(value_weight),
                    "seed": str(seed),
                    "backend": backend,
                    "max_eval_rows": common_values["max_eval_rows"],
                },
                label="teacher command",
            )
        trainer_argv = format_command(
            trainer_command or DEFAULT_TRAINER_COMMAND,
            {
                **common_values,
                "result": str(trainer_result_path),
                "comparison": str(trainer_comparison_path),
            },
            label="trainer command",
        )
        evaluator_argv = format_command(
            evaluator_command or DEFAULT_EVALUATOR_COMMAND,
            {
                **common_values,
                "eval": str(eval_input or eval_merged or eval_sources[0]),
                "result": str(eval_comparison_path),
                "comparison": str(eval_comparison_path),
            },
            label="evaluator command",
        )

        if not dry_run:
            try:
                trainer_result_path.unlink()
            except OSError:
                pass

        teacher = _teacher_record(
            requested=teacher_argv is not None,
            source=source_train,
            data=data_path,
            manifest_path=manifest_path,
        )
        trainer = _trainer_record(
            command=trainer_argv, out_dir=out_path,
            result_path=trainer_result_path)
        evaluation = _evaluation_record(
            command=evaluator_argv, sources=eval_sources, data=eval_input,
            max_rows=max_eval_rows, rows_available=rows_available,
            rows_used=rows_used)

        error: dict[str, Any] | None = None

        # ---- teacher -----------------------------------------------------
        if teacher_argv is not None:
            teacher["command"] = teacher_argv
            if dry_run:
                teacher["status"] = "dry_run"
            elif (teacher_uses_source
                  and (source_train is None or not source_train.is_file())):
                teacher["status"] = "failed"
                teacher["reason"] = "source_missing"
                teacher["manifest"] = _empty_manifest(
                    manifest_path, "teacher_failed")
                error = {"stage": "teacher", "reason": "source_missing",
                         "returncode": None}
            else:
                _clear_teacher_outputs(data_path, manifest_path, source_train)
                completed, elapsed, spawn_error = _invoke(
                    teacher_argv, runner=runner)
                teacher["elapsed_seconds"] = elapsed
                teacher["returncode"] = _returncode(completed)
                if spawn_error is not None:
                    teacher["status"] = "failed"
                    teacher["reason"] = "spawn_error"
                    teacher["error"] = spawn_error
                    error = {"stage": "teacher", "reason": "spawn_error",
                             "returncode": None}
                elif teacher["returncode"] != 0:
                    teacher["status"] = "failed"
                    teacher["reason"] = "returncode"
                    error = {"stage": "teacher", "reason": "returncode",
                             "returncode": teacher["returncode"]}
                elif not data_path.is_file():
                    teacher["status"] = "failed"
                    teacher["reason"] = "teacher_data_missing"
                    error = {"stage": "teacher",
                             "reason": "teacher_data_missing",
                             "returncode": teacher["returncode"]}
                elif (_file_size(data_path) or 0) == 0:
                    teacher["status"] = "failed"
                    teacher["reason"] = "teacher_data_empty"
                    error = {"stage": "teacher",
                             "reason": "teacher_data_empty",
                             "returncode": teacher["returncode"]}
                else:
                    teacher["status"] = "completed"
                if teacher["status"] == "completed":
                    teacher["data_bytes"] = _file_size(data_path)
                    resolved_manifest = (
                        find_teacher_manifest(data_path) or manifest_path)
                    teacher["manifest"] = read_teacher_manifest(
                        resolved_manifest)
                else:
                    teacher["manifest"] = _empty_manifest(
                        manifest_path, "teacher_failed")
        else:
            if dry_run:
                teacher["status"] = "dry_run"
            elif not data_path.is_file():
                teacher["status"] = "failed"
                teacher["reason"] = "teacher_data_missing"
                teacher["manifest"] = _empty_manifest(
                    manifest_path, "teacher_failed")
                error = {"stage": "teacher",
                         "reason": "teacher_data_missing",
                         "returncode": None}
            elif (_file_size(data_path) or 0) == 0:
                teacher["status"] = "failed"
                teacher["reason"] = "teacher_data_empty"
                teacher["manifest"] = _empty_manifest(
                    manifest_path, "teacher_failed")
                error = {"stage": "teacher",
                         "reason": "teacher_data_empty",
                         "returncode": None}
            else:
                teacher["status"] = "skipped"
                teacher["data_bytes"] = _file_size(data_path)
                resolved_manifest = (
                    find_teacher_manifest(data_path) or manifest_path)
                teacher["manifest"] = read_teacher_manifest(resolved_manifest)

        status = STATUS_COMPLETED
        if teacher["status"] == "failed":
            status = STATUS_TEACHER_FAILED
        elif not dry_run and eval_error is not None:
            # A missing/empty eval set cannot produce a final measurement.
            # Record it before spending trainer time.
            evaluation["status"] = "failed"
            evaluation["reason"] = eval_error
            error = {"stage": "evaluation", "reason": eval_error,
                     "returncode": None}
            status = STATUS_EVAL_FAILED

        # ---- trainer -----------------------------------------------------
        if status == STATUS_COMPLETED:
            if dry_run:
                trainer["status"] = "dry_run"
                trainer["returncode"] = 0
                trainer["adapter_exists"] = False
                trainer["result_found"] = False
            else:
                completed, elapsed, spawn_error = _invoke(
                    trainer_argv, runner=runner)
                trainer["elapsed_seconds"] = elapsed
                trainer["returncode"] = _returncode(completed)
                trainer["adapter_exists"] = out_path.is_dir()
                if spawn_error is not None:
                    trainer["status"] = "failed"
                    trainer["reason"] = "spawn_error"
                    trainer["error"] = spawn_error
                    error = {"stage": "trainer", "reason": "spawn_error",
                             "returncode": None}
                elif trainer["returncode"] != 0:
                    trainer["status"] = "failed"
                    trainer["reason"] = "returncode"
                    error = {"stage": "trainer", "reason": "returncode",
                             "returncode": trainer["returncode"]}
                elif not trainer_result_path.is_file():
                    trainer["status"] = "failed"
                    trainer["reason"] = "trainer_result_missing"
                    error = {"stage": "trainer",
                             "reason": "trainer_result_missing",
                             "returncode": trainer["returncode"]}
                elif not out_path.is_dir():
                    trainer["status"] = "failed"
                    trainer["reason"] = "adapter_missing"
                    error = {"stage": "trainer", "reason": "adapter_missing",
                             "returncode": trainer["returncode"]}
                else:
                    trainer["status"] = "completed"
                if trainer_result_path.is_file():
                    trainer["result_found"] = True
                    payload, parse_error = _read_json_mapping(
                        trainer_result_path)
                    trainer["result"] = _filter_trainer_result(payload)
                    if parse_error is not None:
                        trainer["result_error"] = parse_error
                config_path = out_path / "config.json"
                if config_path.is_file():
                    payload, _ = _read_json_mapping(config_path)
                    trainer["adapter_config"] = _filter_scalar_fields(
                        payload, TRAINER_CONFIG_FIELDS)
            if trainer["status"] == "failed":
                status = STATUS_TRAINER_FAILED

        # ---- evaluator ---------------------------------------------------
        if status == STATUS_COMPLETED:
            if dry_run:
                evaluation["status"] = "dry_run"
                evaluation["returncode"] = 0
                evaluation["synthetic"] = True
                evaluation.update(_dry_run_metrics())
                adapter = evaluation.get("adapter") or {}
                evaluation["exact_match_rate"] = adapter.get(
                    "exact_match_rate")
                evaluation["key_copy_rate"] = adapter.get("key_copy_rate")
                evaluation["loss"] = adapter.get("loss")
                evaluation["content_loss"] = adapter.get("content_loss")
            elif eval_error is not None:
                evaluation["status"] = "failed"
                evaluation["reason"] = eval_error
                error = {"stage": "evaluation", "reason": eval_error,
                         "returncode": None}
            else:
                completed, elapsed, spawn_error = _invoke(
                    evaluator_argv, runner=runner)
                evaluation["elapsed_seconds"] = elapsed
                evaluation["returncode"] = _returncode(completed)
                if spawn_error is not None:
                    evaluation["status"] = "failed"
                    evaluation["reason"] = "spawn_error"
                    evaluation["error"] = spawn_error
                    error = {"stage": "evaluation",
                             "reason": "spawn_error", "returncode": None}
                elif evaluation["returncode"] != 0:
                    evaluation["status"] = "failed"
                    evaluation["reason"] = "returncode"
                    error = {"stage": "evaluation",
                             "reason": "returncode",
                             "returncode": evaluation["returncode"]}
                else:
                    payload = None
                    parse_error = None
                    if eval_comparison_path.is_file():
                        payload, parse_error = _read_json_mapping(
                            eval_comparison_path)
                    if payload is None:
                        payload = _parse_json_object(
                            getattr(completed, "stdout", None))
                        parse_error = None if payload is not None else (
                            parse_error or "comparison_missing")
                    if payload is None:
                        evaluation["status"] = "failed"
                        evaluation["reason"] = parse_error
                        error = {"stage": "evaluation",
                                 "reason": parse_error,
                                 "returncode": evaluation["returncode"]}
                    else:
                        summary = summarize_evaluation(payload)
                        if summary.get("adapter") is None:
                            evaluation["status"] = "failed"
                            evaluation["reason"] = "comparison_incomplete"
                            error = {"stage": "evaluation",
                                     "reason": "comparison_incomplete",
                                     "returncode": evaluation["returncode"]}
                        else:
                            evaluation.update(summary)
                            adapter = evaluation.get("adapter") or {}
                            evaluation["exact_match_rate"] = adapter.get(
                                "exact_match_rate")
                            evaluation["key_copy_rate"] = adapter.get(
                                "key_copy_rate")
                            evaluation["loss"] = adapter.get("loss")
                            evaluation["content_loss"] = adapter.get(
                                "content_loss")
                            evaluation["status"] = "completed"
            if evaluation["status"] == "failed":
                status = STATUS_EVAL_FAILED

        ledger = {
            "schema": LEDGER_SCHEMA,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "content_free": True,
            "status": status,
            "dry_run": bool(dry_run),
            "backend": backend,
            "paths": {
                "base": _display_path(base_path),
                "source_train": _display_path(source_train),
                "teacher_data": _display_path(data_path),
                "teacher_manifest": teacher["manifest"].get("path"),
                "eval": _display_path(eval_sources[0]),
                "evals": [_display_path(source) for source in eval_sources],
                "out": _display_path(out_path),
                "result": _display_path(result_target),
            },
            "settings": settings,
            "teacher": teacher,
            "trainer": trainer,
            "evaluation": evaluation,
            "error": error,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
        return ledger


def write_ledger(ledger: Mapping[str, Any], path: str | Path) -> Path:
    """Write the content-free ledger JSON and return its path."""
    target = _resolve(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(ledger, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    return target


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


def _non_negative_float(value: str) -> float:
    number = _finite_float(value)
    if number < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return number


def _content_weight(value: str) -> float:
    number = _finite_float(value)
    if number < 1.0:
        raise argparse.ArgumentTypeError("value must be >= 1.0")
    return number


def _validate_args(parser: argparse.ArgumentParser,
                   args: argparse.Namespace) -> None:
    if not args.dry_run:
        if not args.base:
            parser.error("--base is required unless --dry-run is set")
        if not args.eval_path:
            parser.error("--eval is required unless --dry-run is set")
        if args.teacher_command is None and not args.teacher_data:
            parser.error(
                "--teacher-data is required unless --teacher-command is set")
        if (args.teacher_command is not None
                and not args.train):
            parser.error("--train is required with --teacher-command")


def main(argv: Sequence[str] | None = None, *, runner=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--teacher-data", default=None,
        help=("verified teacher JSONL written by build_distill_copy.py; the "
              "student trains on this file"))
    parser.add_argument(
        "--train", default=DEFAULT_TRAIN,
        help=("raw source training JSONL consumed by the teacher build "
              f"({_display_path(DEFAULT_TRAIN)}); the {{source}} placeholder"))
    parser.add_argument(
        "--base", default="",
        help="base checkpoint directory (required unless --dry-run)")
    parser.add_argument(
        "--eval", default=None, dest="eval_path", action="append",
        metavar="PATH",
        help=("held-out eval JSONL; repeat to evaluate the 2/3/4-key sets "
              "together. These files are never sent to the teacher"))
    parser.add_argument(
        "--out", default=DEFAULT_OUT,
        help=f"adapter output directory (default: {DEFAULT_OUT})")
    parser.add_argument(
        "--result", default=DEFAULT_RESULT,
        help=f"content-free ledger path (default: {DEFAULT_RESULT})")
    parser.add_argument(
        "--teacher-command", default=None,
        help=("command template that builds teacher data; placeholders: "
              "{source}, {out_dir}, {result}"))
    parser.add_argument(
        "--trainer-command", default=None,
        help=("command template for student training; placeholders: "
              "{base}, {train}, {eval}, {out}, {result}, {comparison} and "
              "the numeric settings"))
    parser.add_argument(
        "--evaluator-command", default=None,
        help=("optional evaluator template; defaults to "
              "research/compare_adapters.py with --per-family, --exact-match "
              "and --key-copy"))
    parser.add_argument(
        "--steps", type=_positive_int, default=2000,
        help="student training rows (default: 2000)")
    parser.add_argument(
        "--batch-size", type=_positive_int, default=4,
        help="rows accumulated per optimizer step (default: 4)")
    parser.add_argument(
        "--lr", type=_finite_float, default=3e-4,
        help="learning rate (default: 3e-4)")
    parser.add_argument(
        "--content-weight", type=_content_weight, default=2.0,
        help="content loss multiplier, >= 1.0 (default: 2.0)")
    parser.add_argument(
        "--key-weight", type=_non_negative_float, default=2.0,
        help="key-token loss multiplier (default: 2.0)")
    parser.add_argument(
        "--value-weight", type=_non_negative_float, default=2.0,
        help="value-token loss multiplier (default: 2.0)")
    parser.add_argument(
        "--seed", type=int, default=23,
        help="training seed (default: 23)")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="write a synthetic ledger and run no subprocess")
    parser.add_argument(
        "--max-eval-rows", type=_positive_int, default=None,
        help="cap the held-out eval rows used by the evaluator")
    parser.add_argument(
        "--backend", choices=("local", "gpu"), default=DEFAULT_BACKEND,
        help="execution backend recorded in the ledger (default: local)")
    args = parser.parse_args(argv)
    _validate_args(parser, args)

    try:
        ledger = run_cycle(
            base=args.base,
            eval_path=args.eval_path,
            out_dir=args.out,
            result_path=args.result,
            teacher_data=args.teacher_data,
            train=args.train,
            teacher_command=args.teacher_command,
            trainer_command=args.trainer_command,
            evaluator_command=args.evaluator_command,
            steps=args.steps,
            batch_size=args.batch_size,
            lr=args.lr,
            content_weight=args.content_weight,
            key_weight=args.key_weight,
            value_weight=args.value_weight,
            seed=args.seed,
            dry_run=args.dry_run,
            max_eval_rows=args.max_eval_rows,
            backend=args.backend,
            runner=runner,
        )
    except DistillCycleError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    try:
        target = write_ledger(ledger, args.result)
    except OSError as exc:
        print(f"error: cannot write {args.result}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(ledger, indent=2, ensure_ascii=False))
    print(f"wrote {_display_path(target)}")
    return 0 if ledger["status"] == STATUS_COMPLETED else 1


if __name__ == "__main__":
    raise SystemExit(main())
