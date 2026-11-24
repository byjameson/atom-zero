"""Run the staged copy curriculum end to end and record a content-free ledger.

The staged copy queue (``research/build_staged_copy_queue.py``) produces one
train/eval split per stage.  This runner chains those stages:

1. train the first stage from a fresh LoRA adapter;
2. for every later stage, initialise from the previous stage's adapter
   (``--init-adapter``) and continue training on the next stage;
3. evaluate every stage on its own held-out eval split and on the final
   ``multi`` eval split;
4. keep only aggregate metrics in a content-free ledger.

The runner never imports ``torch`` or loads a model.  Trainer and comparator
work happens in subprocesses, driven by
``research/train_adapter_candidate.py`` and
``research/compare_adapters.py``.  Tests can inject alternative command
prefixes and ``--dry-run`` writes fake metrics without touching a checkpoint.

Example::

    python research/run_staged_curriculum.py \\
      --data-dir data/staged_copy_v1 \\
      --out-dir checkpoints/staged_curriculum \\
      --base checkpoints/transformer_behavior_50m_long \\
      --stages echo_kv,echo_json,two_kv,two_json,multi \\
      --steps 2000 --batch-size 4 --lr 3e-4 --content-weight 2 \\
      --rank 8 --alpha 8 --rank-attn 4 --alpha-attn 8 \\
      --seed 23 \\
      --result research/results/staged_curriculum_run.json

Add ``--stop-on-key-copy 0.8`` to stop after the first stage whose held-out
key-copy rate reaches the gate.  Add ``--dry-run`` to validate the pipeline
and produce a result JSON without a model or data files.
"""
from __future__ import annotations

import argparse
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
DEFAULT_TRAINER = (sys.executable, str(ROOT / "research" /
                                      "train_adapter_candidate.py"))
DEFAULT_COMPARATOR = (sys.executable, str(ROOT / "research" /
                                         "compare_adapters.py"))

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


def _run(command: Sequence[str], *, stage: str, label: str
         ) -> tuple[subprocess.CompletedProcess[str], float]:
    started = time.monotonic()
    try:
        completed = subprocess.run(
            list(command), cwd=ROOT, capture_output=True, text=True,
            encoding="utf-8", errors="replace", env=_child_env())
    except OSError as exc:
        raise CurriculumError(
            f"stage {stage}: could not start {label}: {exc}") from exc
    elapsed = time.monotonic() - started
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        tail = "\n".join(detail.splitlines()[-5:])
        message = (f"stage {stage}: {label} failed with returncode "
                   f"{completed.returncode}")
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
    dry_run: bool = False,
    trainer_command: Sequence[str] | None = None,
    comparator_command: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run every requested stage in canonical order and return the ledger."""
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
        },
        "stages": records,
    }


def write_ledger(result: Mapping[str, Any], result_path: str | Path) -> Path:
    """Write the content-free ledger JSON and return its path."""
    target = _resolve(result_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
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


def _bounded_weight(value: str) -> float:
    number = _finite_float(value)
    if not 0.0 <= number <= 1.0:
        raise argparse.ArgumentTypeError("value must be between 0.0 and 1.0")
    return number


def _gate(value: str) -> float:
    return _bounded_weight(value)


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


def main(argv: Sequence[str] | None = None, *,
         trainer_command: Sequence[str] | None = None,
         comparator_command: Sequence[str] | None = None) -> int:
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
        "--dry-run", action="store_true",
        help="write fake metrics and skip trainer/comparator subprocesses")
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
            dry_run=args.dry_run,
            trainer_command=trainer_command,
            comparator_command=comparator_command,
        )
    except (CurriculumError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    target = write_ledger(result, args.result)
    if not args.dry_run:
        write_ledger(result, _resolve(args.out_dir) / "ledger.json")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"wrote {_display_path(target)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
