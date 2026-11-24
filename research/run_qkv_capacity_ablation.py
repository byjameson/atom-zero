"""Run a controlled fused-QKV vs matched no-QKV LoRA capacity ablation.

The 49M base plus MLP and attention ``out_proj`` LoRA reaches 48% exact / 60.4%
key copy on the mixed ``multi_recover`` split, while three and four arbitrary
keys stay at 0-6%. Fused QKV LoRA adds an adapter on the attention input
projection. A first from-base run scored 6% exact / 18% key copy, but one run
is not evidence. This runner performs the controlled comparison:

1. for every arm (``qkv4``, ``mlp``) and seed, train with
   ``research/train_adapter_candidate.py``;
2. evaluate the adapter on one fixed eval JSONL with
   ``research/compare_adapters.py --key-copy --exact-match``;
3. aggregate exact match and key-copy rate per arm (mean, sample standard
   deviation, min, max, 95% t-interval) and keep one row per seed;
4. write a content-free JSON and Markdown report.

Both arms share the seed, data order, steps, batch size, learning rate, loss
weights, MLP geometry and ``out_proj`` geometry. ``qkv4`` passes
``--qkv-rank/--qkv-alpha``; ``mlp`` passes ``--qkv-rank 0 --qkv-alpha 0`` and
keeps the shared MLP/out_proj adapters. It is "MLP-only" in the sense that the
fused QKV projection is frozen, not that ``out_proj`` is absent. The QKV arm
therefore adds trainable parameters: a win can be extra capacity rather than
the attention locus. A matched-capacity MLP widening is the follow-up control.

    python research/run_qkv_capacity_ablation.py \\
      --base "$env:USERPROFILE\\.cache\\atomzero\\atom-zero-49m" \\
      --train data/multi_short_v1/stage_3short/train.jsonl \\
      --eval data/multi_short_v1/stage_3short/eval.jsonl \\
      --out checkpoints/qkv_capacity_ablation \\
      --result research/results/qkv_capacity_ablation.json \\
      --arms qkv4,mlp --seeds 23,29,37 --steps 4000 \\
      --batch-size 4 --lr 3e-4 --content-weight 2 \\
      --key-weight 3 --value-weight 2

``--max-eval-rows N`` selects a deterministic, hash-based subset of the eval
rows. The subset is materialized in a temporary directory and is identical for
every arm and seed. ``--dry-run`` prints the planned commands without reading
data, training or writing files.

The report never stores prompt or answer text. Per-row comparison payloads are
written to the temporary directory, parsed for aggregate numbers and deleted.
Only whitelisted metrics and paths are copied into the JSON/Markdown report.

No claim is made when the 95% intervals overlap. With three seeds this design
can detect only large effects. This is a narrow synthetic copy-curriculum
measurement, not a general capability claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "atomzero.qkv_capacity_ablation.v1"
DEFAULT_ARMS = "qkv4,mlp"
DEFAULT_SEEDS = "23,29,37"
DEFAULT_BASE = "checkpoints/transformer_behavior_50m_long"
DEFAULT_TRAIN = "data/multi_short_v1/stage_3short/train.jsonl"
DEFAULT_EVAL = "data/multi_short_v1/stage_3short/eval.jsonl"
DEFAULT_OUT = "checkpoints/qkv_capacity_ablation"
DEFAULT_RESULT = "research/results/qkv_capacity_ablation.json"
DEFAULT_TRAINER = (sys.executable, "research/train_adapter_candidate.py")
DEFAULT_EVALUATOR = (sys.executable, "research/compare_adapters.py")
ARM_NAMES = ("qkv4", "mlp")
SUBSET_SEED = 0
METRIC_FIELDS = ("exact", "key_copy")
AGGREGATE_FIELDS = (
    "exact",
    "key_copy",
    "key_recall",
    "loss",
    "content_loss",
    "train_loss",
    "lora_parameters",
)
RUN_FIELDS = (
    "arm",
    "seed",
    "status",
    "returncode",
    "elapsed_seconds",
    "adapter",
    "exact",
    "key_copy",
    "key_recall",
    "loss",
    "content_loss",
    "eval_examples",
    "train_loss",
    "lora_parameters",
    "trainer",
    "evaluator",
    "error",
)
TRAINER_RESULT_FIELDS = (
    "train_loss",
    "train_steps",
    "train_examples",
    "optimizer_steps",
    "rows_seen",
    "batch_size",
    "effective_lr",
    "content_weight",
    "key_weight",
    "value_weight",
    "lora_parameters",
    "rank_mlp",
    "alpha_mlp",
    "rank_attn",
    "alpha_attn",
    "rank_qkv",
    "alpha_qkv",
)

# Two-sided 95% t critical values by degrees of freedom. The normal
# approximation is used above 30 degrees of freedom.
_T95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}


class AblationError(ValueError):
    """Raised when an ablation option, plan or metric payload is invalid."""


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


def parse_arms(value: str) -> list[str]:
    """Parse ``qkv4,mlp`` into a de-duplicated arm list."""
    if not isinstance(value, str):
        raise AblationError("arms must be a comma-separated string")
    arms: list[str] = []
    for chunk in value.split(","):
        name = chunk.strip()
        if not name:
            continue
        if name not in ARM_NAMES:
            raise AblationError(
                f"unknown arm {name!r}; choose from {', '.join(ARM_NAMES)}")
        if name not in arms:
            arms.append(name)
    if not arms:
        raise AblationError("at least one arm is required")
    return arms


def parse_seeds(value: str) -> list[int]:
    """Parse ``23,29,37`` into a de-duplicated integer seed list."""
    if not isinstance(value, str):
        raise AblationError("seeds must be a comma-separated string")
    seeds: list[int] = []
    for chunk in value.split(","):
        text = chunk.strip()
        if not text:
            continue
        try:
            seed = int(text)
        except ValueError as exc:
            raise AblationError(f"invalid seed {text!r}") from exc
        if seed not in seeds:
            seeds.append(seed)
    if not seeds:
        raise AblationError("at least one seed is required")
    return seeds


def expand_runs(arms: Sequence[str], seeds: Sequence[int]) -> list[dict]:
    """Expand arms x seeds into the execution order used by the runner."""
    runs = []
    for arm in arms:
        if arm not in ARM_NAMES:
            raise AblationError(f"unknown arm {arm!r}")
        for seed in seeds:
            runs.append({"arm": arm, "seed": int(seed)})
    if not runs:
        raise AblationError("no arm/seed runs were planned")
    return runs


def arm_geometry(*, arm: str, rank: int, alpha: int, rank_attn: int,
                 alpha_attn: int, qkv_rank: int, qkv_alpha: int) -> dict:
    """LoRA geometry used by one arm. Only ``qkv4`` enables QKV LoRA."""
    if arm not in ARM_NAMES:
        raise AblationError(f"unknown arm {arm!r}")
    enabled = arm == "qkv4"
    return {
        "rank_mlp": int(rank),
        "alpha_mlp": int(alpha),
        "rank_attn": int(rank_attn),
        "alpha_attn": int(alpha_attn),
        "rank_qkv": int(qkv_rank) if enabled else 0,
        "alpha_qkv": int(qkv_alpha) if enabled else 0,
    }


def build_trainer_command(
    *,
    arm: str,
    seed: int,
    base: str,
    train: str,
    eval_path: str,
    out: str,
    result: str,
    comparison: str,
    steps: int,
    batch_size: int,
    lr: float,
    content_weight: float,
    key_weight: float,
    value_weight: float,
    rank: int,
    alpha: int,
    rank_attn: int,
    alpha_attn: int,
    qkv_rank: int,
    qkv_alpha: int,
    command_prefix: Sequence[str] | None = None,
) -> list[str]:
    """Build one ``train_adapter_candidate`` command without running it."""
    geometry = arm_geometry(
        arm=arm, rank=rank, alpha=alpha, rank_attn=rank_attn,
        alpha_attn=alpha_attn, qkv_rank=qkv_rank, qkv_alpha=qkv_alpha)
    command = _command(command_prefix, DEFAULT_TRAINER)
    command.extend([
        "--base", str(base),
        "--train", str(train),
        "--eval", str(eval_path),
        "--out", str(out),
        "--steps", str(steps),
        "--batch-size", str(batch_size),
        "--lr", str(lr),
        "--content-weight", str(content_weight),
        "--key-weight", str(key_weight),
        "--value-weight", str(value_weight),
        "--rank", str(geometry["rank_mlp"]),
        "--alpha", str(geometry["alpha_mlp"]),
        "--rank-attn", str(geometry["rank_attn"]),
        "--alpha-attn", str(geometry["alpha_attn"]),
        "--qkv-rank", str(geometry["rank_qkv"]),
        "--qkv-alpha", str(geometry["alpha_qkv"]),
        "--seed", str(seed),
        "--result", str(result),
        "--comparison", str(comparison),
    ])
    return command


def build_evaluator_command(
    *,
    base: str,
    data: str,
    adapter: str,
    out: str,
    command_prefix: Sequence[str] | None = None,
) -> list[str]:
    """Build one ``compare_adapters`` command without running it."""
    command = _command(command_prefix, DEFAULT_EVALUATOR)
    command.extend([
        "--base", str(base),
        "--data", str(data),
        "--adapters", str(adapter),
        "--key-copy",
        "--exact-match",
        "--out", str(out),
    ])
    return command


def _t_critical(count: int) -> float:
    if count < 2:
        raise AblationError("a t interval needs at least two values")
    return _T95.get(count - 1, 1.96)


def aggregate_metric(values: Sequence[Any]) -> dict | None:
    """Mean, sample std, min, max and a 95% t interval for one metric.

    Non-numeric values (failed or unavailable runs) are ignored. One value has
    no standard deviation or interval, so it cannot support a claim.
    """
    clean = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        number = float(value)
        if math.isfinite(number):
            clean.append(number)
    if not clean:
        return None
    count = len(clean)
    mean = statistics.fmean(clean)
    std = statistics.stdev(clean) if count >= 2 else None
    ci95 = None
    if std is not None:
        half = _t_critical(count) * std / math.sqrt(count)
        ci95 = [round(mean - half, 6), round(mean + half, 6)]
    return {
        "count": count,
        "mean": round(mean, 6),
        "std": round(std, 6) if std is not None else None,
        "min": round(min(clean), 6),
        "max": round(max(clean), 6),
        "ci95": ci95,
    }


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _rate(value: Any, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AblationError(f"{field} rate must be numeric")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise AblationError(f"{field} rate out of range: {value!r}")
    return number


def _read_json(path: str | Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise AblationError(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise AblationError(f"invalid JSON in {path}: {exc}") from exc


def read_jsonl(path: str | Path) -> list[dict]:
    """Read a JSONL file of objects; used only to build an eval subset."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise AblationError(f"cannot read {path}: {exc}") from exc
    rows: list[dict] = []
    for number, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AblationError(
                f"{path}:{number}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise AblationError(f"{path}:{number}: row must be an object")
        rows.append(row)
    return rows


def write_jsonl(path: str | Path, rows: Sequence[Mapping]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def rows_digest(rows: Sequence[Mapping]) -> str:
    """SHA-256 of the canonical eval rows, without storing their text."""
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(
            row, sort_keys=True, ensure_ascii=False,
            separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def select_eval_rows(rows: Sequence[dict], max_rows: int | None,
                     seed: int = SUBSET_SEED) -> list[dict]:
    """Return the full list or a deterministic hash-based subset.

    The hash covers the canonical row, its original index and the subset seed,
    so the same input file always selects the same rows and the selection is
    stable across Python versions. The selected rows keep their original
    relative order.
    """
    if max_rows is None:
        return list(rows)
    if isinstance(max_rows, bool) or not isinstance(max_rows, int):
        raise AblationError("max_rows must be an int")
    if max_rows < 1:
        raise AblationError("max_rows must be >= 1")
    if max_rows >= len(rows):
        return list(rows)

    def sort_key(index: int) -> str:
        row = rows[index]
        payload = json.dumps(
            row, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        material = f"{seed}\0{index}\0{payload}".encode("utf-8")
        return hashlib.sha256(material).hexdigest()

    chosen = sorted(range(len(rows)), key=sort_key)[:max_rows]
    return [rows[index] for index in sorted(chosen)]


def _read_trainer_result(path: str | Path | None) -> dict | None:
    if not path or not Path(path).is_file():
        return None
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, Mapping):
        return None
    result: dict[str, Any] = {}
    for field in TRAINER_RESULT_FIELDS:
        value = payload.get(field)
        if isinstance(value, bool):
            continue
        number = _finite_number(value)
        if number is not None:
            result[field] = value
    return result


def read_comparison_metrics(path: str | Path) -> dict:
    """Extract only aggregate adapter metrics from a comparison payload."""
    payload = _read_json(path)
    if not isinstance(payload, Mapping):
        raise AblationError("comparison payload must be a JSON object")
    models = payload.get("models")
    if not isinstance(models, list) or len(models) < 2:
        raise AblationError("comparison payload needs base and adapter models")
    adapter = models[-1]
    if not isinstance(adapter, Mapping):
        raise AblationError("comparison adapter entry must be an object")
    metrics = adapter.get("metrics")
    if not isinstance(metrics, Mapping):
        raise AblationError("comparison payload has no adapter metrics")
    exact = metrics.get("exact_match")
    copy = metrics.get("key_copy")
    if not isinstance(exact, Mapping) or not isinstance(copy, Mapping):
        raise AblationError(
            "comparison payload has no exact_match/key_copy metrics")
    exact_rate = _rate(exact.get("rate"), "exact_match")
    key_rate = _rate(copy.get("rate"), "key_copy")
    if exact_rate is None or key_rate is None:
        raise AblationError(
            "comparison payload did not report exact/key-copy rates")
    loss = _finite_number(metrics.get("loss"))
    if loss is None:
        raise AblationError("comparison payload has no numeric loss")
    return {
        "exact": exact_rate,
        "key_copy": key_rate,
        "key_recall": _rate(copy.get("key_recall"), "key_recall"),
        "loss": loss,
        "content_loss": _finite_number(metrics.get("content_loss")),
        "eval_examples": payload.get("examples"),
    }


def _invoke(command: Sequence[str], *, runner=None,
            timeout: float | None = None):
    invoke = runner or subprocess.run
    return invoke(
        list(command), cwd=str(ROOT), capture_output=True, text=True,
        timeout=timeout, env=_child_env())


def _returncode(completed) -> int | None:
    value = getattr(completed, "returncode", None)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def run_trainer(command: Sequence[str], *, adapter: str,
                result_path: str, runner=None,
                timeout: float | None = None) -> dict:
    """Run the trainer and return a content-free status record."""
    started = time.monotonic()
    try:
        completed = _invoke(command, runner=runner, timeout=timeout)
    except Exception as exc:  # subprocess failures are recorded, not raised
        return {
            "status": "failed",
            "returncode": None,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "adapter_exists": Path(adapter).is_dir(),
            "command": list(command),
            "result": None,
            "error": f"{type(exc).__name__}: {exc}",
        }
    elapsed = round(time.monotonic() - started, 3)
    returncode = _returncode(completed)
    adapter_exists = Path(adapter).is_dir()
    result = _read_trainer_result(result_path)
    status = "completed"
    error = None
    if returncode != 0:
        status = "failed"
        error = "trainer returned a non-zero exit code"
    elif not adapter_exists:
        status = "failed"
        error = "trainer did not write the adapter directory"
    record = {
        "status": status,
        "returncode": returncode,
        "elapsed_seconds": elapsed,
        "adapter_exists": adapter_exists,
        "command": list(command),
        "result": result,
    }
    if error:
        record["error"] = error
    return record


def run_evaluator(command: Sequence[str], *, payload_path: str,
                  runner=None, timeout: float | None = None) -> dict:
    """Run the comparator and return aggregate metrics only."""
    started = time.monotonic()
    try:
        completed = _invoke(command, runner=runner, timeout=timeout)
    except Exception as exc:
        return {
            "status": "failed",
            "returncode": None,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "command": list(command),
            "metrics": None,
            "error": f"{type(exc).__name__}: {exc}",
        }
    elapsed = round(time.monotonic() - started, 3)
    returncode = _returncode(completed)
    if returncode != 0:
        return {
            "status": "failed",
            "returncode": returncode,
            "elapsed_seconds": elapsed,
            "command": list(command),
            "metrics": None,
            "error": "evaluator returned a non-zero exit code",
        }
    try:
        metrics = read_comparison_metrics(payload_path)
    except AblationError as exc:
        return {
            "status": "failed",
            "returncode": returncode,
            "elapsed_seconds": elapsed,
            "command": list(command),
            "metrics": None,
            "error": str(exc),
        }
    return {
        "status": "completed",
        "returncode": returncode,
        "elapsed_seconds": elapsed,
        "command": list(command),
        "metrics": metrics,
    }


TRAINER_RECORD_FIELDS = (
    "status",
    "returncode",
    "elapsed_seconds",
    "adapter_exists",
    "command",
    "result",
    "error",
)
EVALUATOR_RECORD_FIELDS = (
    "status",
    "returncode",
    "elapsed_seconds",
    "command",
    "metrics",
    "error",
)
COMPARISON_METRIC_FIELDS = (
    "exact", "key_copy", "key_recall", "loss", "content_loss",
    "eval_examples")


def _public_record(record: Mapping) -> dict:
    """Whitelist a run record and its nested records.

    Producers are external processes; a whitelist keeps a future extra field
    (for example a prompt echoed by a fake backend) out of the report.
    """
    public = {field: record[field] for field in RUN_FIELDS if field in record}
    for name, fields in (("trainer", TRAINER_RECORD_FIELDS),
                         ("evaluator", EVALUATOR_RECORD_FIELDS)):
        nested = public.get(name)
        if not isinstance(nested, Mapping):
            continue
        clean = {field: nested[field] for field in fields if field in nested}
        if name == "trainer" and isinstance(clean.get("result"), Mapping):
            clean["result"] = {
                field: clean["result"][field]
                for field in TRAINER_RESULT_FIELDS
                if field in clean["result"]}
        if name == "evaluator" and isinstance(clean.get("metrics"), Mapping):
            clean["metrics"] = {
                field: clean["metrics"][field]
                for field in COMPARISON_METRIC_FIELDS
                if field in clean["metrics"]}
        public[name] = clean
    return public


def run_single(spec: Mapping, *, trainer_runner=None, evaluator_runner=None,
               timeout: float | None = None) -> dict:
    """Train and evaluate one arm/seed pair."""
    started = time.monotonic()
    trainer = run_trainer(
        spec["trainer_command"], adapter=spec["adapter"],
        result_path=spec["trainer_result"], runner=trainer_runner,
        timeout=timeout)
    record: dict[str, Any] = {
        "arm": spec["arm"],
        "seed": spec["seed"],
        "status": "trainer_failed",
        "returncode": trainer.get("returncode"),
        "elapsed_seconds": 0.0,
        "adapter": _display_path(spec["adapter"]),
        "exact": None,
        "key_copy": None,
        "key_recall": None,
        "loss": None,
        "content_loss": None,
        "eval_examples": None,
        "train_loss": None,
        "lora_parameters": None,
        "trainer": trainer,
        "evaluator": {
            "status": "skipped",
            "returncode": None,
            "elapsed_seconds": 0.0,
            "command": list(spec["evaluator_command"]),
            "metrics": None,
        },
    }
    result = trainer.get("result") or {}
    record["train_loss"] = result.get("train_loss")
    record["lora_parameters"] = result.get("lora_parameters")
    if trainer["status"] != "completed":
        record["error"] = trainer.get("error")
        record["elapsed_seconds"] = round(time.monotonic() - started, 3)
        return _public_record(record)
    evaluator = run_evaluator(
        spec["evaluator_command"], payload_path=spec["eval_output"],
        runner=evaluator_runner, timeout=timeout)
    record["evaluator"] = evaluator
    if evaluator["status"] != "completed":
        record["status"] = "eval_failed"
        returncode = evaluator.get("returncode")
        record["returncode"] = (returncode if returncode is not None
                                else trainer.get("returncode"))
        record["error"] = evaluator.get("error")
    else:
        metrics = evaluator["metrics"]
        for field in ("exact", "key_copy", "key_recall", "loss",
                      "content_loss", "eval_examples"):
            record[field] = metrics.get(field)
        record["status"] = "completed"
        record["returncode"] = evaluator.get("returncode")
    record["elapsed_seconds"] = round(time.monotonic() - started, 3)
    return _public_record(record)


def _settings(*, arms: Sequence[str], seeds: Sequence[int], steps: int,
              batch_size: int, lr: float, content_weight: float,
              key_weight: float, value_weight: float, rank: int, alpha: int,
              rank_attn: int, alpha_attn: int, qkv_rank: int,
              qkv_alpha: int, max_eval_rows: int | None) -> dict:
    return {
        "arms": list(arms),
        "seeds": list(seeds),
        "steps": int(steps),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "content_weight": float(content_weight),
        "key_weight": float(key_weight),
        "value_weight": float(value_weight),
        "rank": int(rank),
        "alpha": int(alpha),
        "rank_attn": int(rank_attn),
        "alpha_attn": int(alpha_attn),
        "qkv_rank": int(qkv_rank),
        "qkv_alpha": int(qkv_alpha),
        "max_eval_rows": max_eval_rows,
    }


def _plan(*, base: str, train: str, eval_path: str, out_dir: str | Path,
          arms: Sequence[str], seeds: Sequence[int], settings: Mapping,
          scratch: Path | None, trainer_command: Sequence[str] | None,
          evaluator_command: Sequence[str] | None) -> list[dict]:
    specs = []
    for arm, seed in [(run["arm"], run["seed"])
                      for run in expand_runs(arms, seeds)]:
        run_dir = Path(out_dir) / f"{arm}_s{seed}"
        adapter = run_dir / "adapter"
        result_path = run_dir / "trainer_result.json"
        if scratch is None:
            comparison = run_dir / "trainer_comparison.json"
            eval_output = run_dir / "eval.json"
        else:
            comparison = scratch / f"{arm}_s{seed}_comparison.json"
            eval_output = scratch / f"{arm}_s{seed}_eval.json"
        trainer = build_trainer_command(
            arm=arm, seed=seed, base=base, train=train,
            eval_path=eval_path, out=str(adapter), result=str(result_path),
            comparison=str(comparison), steps=settings["steps"],
            batch_size=settings["batch_size"], lr=settings["lr"],
            content_weight=settings["content_weight"],
            key_weight=settings["key_weight"],
            value_weight=settings["value_weight"], rank=settings["rank"],
            alpha=settings["alpha"], rank_attn=settings["rank_attn"],
            alpha_attn=settings["alpha_attn"],
            qkv_rank=settings["qkv_rank"],
            qkv_alpha=settings["qkv_alpha"],
            command_prefix=trainer_command)
        evaluator = build_evaluator_command(
            base=base, data=eval_path, adapter=str(adapter),
            out=str(eval_output), command_prefix=evaluator_command)
        specs.append({
            "arm": arm,
            "seed": seed,
            "run_dir": str(run_dir),
            "adapter": str(adapter),
            "trainer_result": str(result_path),
            "trainer_comparison": str(comparison),
            "eval_output": str(eval_output),
            "trainer_command": trainer,
            "evaluator_command": evaluator,
        })
    return specs


def _dry_run_payload(*, base: str, train: str, eval_path: str,
                     out_dir: str | Path, arms: Sequence[str],
                     seeds: Sequence[int], settings: Mapping,
                     trainer_command: Sequence[str] | None,
                     evaluator_command: Sequence[str] | None) -> dict:
    specs = _plan(
        base=base, train=train, eval_path=eval_path, out_dir=out_dir,
        arms=arms, seeds=seeds, settings=settings, scratch=None,
        trainer_command=trainer_command,
        evaluator_command=evaluator_command)
    return {
        "schema": SCHEMA,
        "dry_run": True,
        "content_free": True,
        "status": "dry_run",
        "base": _display_path(base),
        "train": _display_path(train),
        "eval": _display_path(eval_path),
        "settings": dict(settings),
        "runs": specs,
        "notes": [
            "Dry run only: nothing was trained, evaluated or written.",
            "Every arm/seed pair receives the same data, steps, batch size, "
            "learning rate and loss weights; only the fused QKV flags differ.",
        ],
    }


def _run_paths_by_arm(records: Sequence[Mapping],
                      arms: Sequence[str]) -> dict[str, list[dict]]:
    groups = {arm: [] for arm in arms}
    for record in records:
        arm = record.get("arm")
        if arm in groups:
            groups[arm].append(_public_record(record))
    return groups


def build_verdict(arm_summaries: Mapping[str, Mapping]) -> dict:
    """Compare arms with non-overlapping 95% intervals; overlap is no claim."""
    base = {
        "claim": "not_applicable",
        "qkv_improves": None,
        "confidence": "95% t-interval over seeds",
        "min_seeds_for_claim": 2,
        "metrics": {},
        "note": (
            "No claim is made when the 95% intervals overlap. With three "
            "seeds only large effects are detectable; overlap means the "
            "experiment did not establish a difference, not that there is "
            "none."),
    }
    if "qkv4" not in arm_summaries or "mlp" not in arm_summaries:
        base["note"] = (
            "Both the qkv4 and mlp arms are required for a comparison. "
            + base["note"])
        return base
    metrics: dict[str, dict] = {}
    winners: list[str | None] = []
    for field in METRIC_FIELDS:
        qkv = arm_summaries["qkv4"].get("aggregate", {}).get(field)
        mlp = arm_summaries["mlp"].get("aggregate", {}).get(field)
        entry: dict[str, Any] = {
            "qkv4": {
                "count": qkv["count"] if qkv else 0,
                "mean": qkv["mean"] if qkv else None,
                "ci95": qkv["ci95"] if qkv else None,
            },
            "mlp": {
                "count": mlp["count"] if mlp else 0,
                "mean": mlp["mean"] if mlp else None,
                "ci95": mlp["ci95"] if mlp else None,
            },
            "delta": None,
            "interval_overlap": None,
            "winner": None,
            "status": "ok",
        }
        if (not qkv or not mlp or qkv["count"] < 2 or mlp["count"] < 2
                or qkv["ci95"] is None or mlp["ci95"] is None):
            entry["status"] = "insufficient_runs"
            metrics[field] = entry
            winners.append(None)
            continue
        q_low, q_high = qkv["ci95"]
        m_low, m_high = mlp["ci95"]
        overlap = not (q_low > m_high or m_low > q_high)
        entry["delta"] = round(qkv["mean"] - mlp["mean"], 6)
        entry["interval_overlap"] = overlap
        if not overlap:
            entry["winner"] = "qkv4" if q_low > m_high else "mlp"
        winners.append(entry["winner"])
        metrics[field] = entry
    base["metrics"] = metrics
    if any(entry["status"] != "ok" for entry in metrics.values()):
        base["claim"] = "inconclusive"
        base["note"] = (
            "At least one metric has fewer than two successful seeds per arm. "
            + base["note"])
        return base
    if all(winner == "qkv4" for winner in winners):
        base["claim"] = "qkv_improves"
        base["qkv_improves"] = True
    elif all(winner == "mlp" for winner in winners):
        base["claim"] = "mlp_better"
        base["qkv_improves"] = False
    elif "qkv4" in winners and "mlp" in winners:
        base["claim"] = "mixed"
        base["note"] = (
            "Exact match and key copy point in opposite directions. "
            + base["note"])
    else:
        base["claim"] = "inconclusive"
    return base


def _mean_aggregate(arm_summary: Mapping, field: str) -> float | None:
    aggregate = arm_summary.get("aggregate", {}).get(field)
    return aggregate.get("mean") if aggregate else None


def build_fair_comparison(*, settings: Mapping,
                          arm_summaries: Mapping[str, Mapping]) -> dict:
    qkv_params = _mean_aggregate(arm_summaries.get("qkv4", {}),
                                 "lora_parameters")
    mlp_params = _mean_aggregate(arm_summaries.get("mlp", {}),
                                 "lora_parameters")
    qkv_adds = None
    if qkv_params is not None and mlp_params is not None:
        qkv_adds = qkv_params > mlp_params
    elif settings["qkv_rank"] > 0:
        qkv_adds = True
    return {
        "same_seed_per_pair": True,
        "same_train_data": True,
        "same_eval_data": True,
        "same_steps": settings["steps"],
        "same_batch_size": settings["batch_size"],
        "same_lr": settings["lr"],
        "same_loss_weights": {
            "content_weight": settings["content_weight"],
            "key_weight": settings["key_weight"],
            "value_weight": settings["value_weight"],
        },
        "same_mlp_geometry": {
            "rank": settings["rank"], "alpha": settings["alpha"]},
        "same_attn_geometry": {
            "rank": settings["rank_attn"],
            "alpha": settings["alpha_attn"]},
        "only_planned_difference": "fused QKV LoRA rank/alpha",
        "qkv_adds_trainable_parameters": qkv_adds,
        "trainable_parameters": {
            "qkv4": qkv_params,
            "mlp": mlp_params,
        },
        "note": (
            "Both arms use the same seeds, data order, steps, batch size, "
            "learning rate, loss weights, MLP geometry and out_proj geometry. "
            "The QKV arm adds trainable parameters, so a win can reflect "
            "extra capacity rather than the attention locus. A matched-"
            "capacity MLP widening is the follow-up control."),
    }


def build_summary(*, records: Sequence[Mapping], settings: Mapping,
                  arms: Sequence[str], base: str, train: str,
                  eval_path: str, eval_subset: Mapping,
                  dry_run: bool = False) -> dict:
    """Aggregate run records into a content-free summary document."""
    groups = _run_paths_by_arm(records, arms)
    arm_summaries: dict[str, dict] = {}
    for arm in arms:
        runs = groups.get(arm, [])
        completed = [run for run in runs if run.get("status") == "completed"]
        aggregate = {
            field: aggregate_metric([run.get(field) for run in completed])
            for field in AGGREGATE_FIELDS
        }
        failed = len(runs) - len(completed)
        if not runs or failed == len(runs):
            status = "failed"
        elif failed:
            status = "partial"
        else:
            status = "completed"
        arm_summaries[arm] = {
            "geometry": arm_geometry(
                arm=arm, rank=settings["rank"], alpha=settings["alpha"],
                rank_attn=settings["rank_attn"],
                alpha_attn=settings["alpha_attn"],
                qkv_rank=settings["qkv_rank"],
                qkv_alpha=settings["qkv_alpha"]),
            "status": status,
            "completed": len(completed),
            "failed": failed,
            "aggregate": aggregate,
            "runs": runs,
        }
    all_runs = [run for arm in arms for run in groups.get(arm, [])]
    completed_total = sum(
        1 for run in all_runs if run.get("status") == "completed")
    if not all_runs or completed_total == 0:
        status = "failed"
    elif completed_total < len(all_runs):
        status = "partial"
    else:
        status = "completed"
    verdict = build_verdict(arm_summaries)
    fair = build_fair_comparison(
        settings=settings, arm_summaries=arm_summaries)
    total_rows = int(eval_subset.get("total_rows",
                                     eval_subset.get("rows", 0)))
    used_rows = int(eval_subset.get("rows", 0))
    summary = {
        "schema": SCHEMA,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "content_free": True,
        "dry_run": bool(dry_run),
        "status": status,
        "base": _display_path(base),
        "train": _display_path(train),
        "eval": _display_path(eval_path),
        "eval_rows": total_rows,
        "eval_rows_used": used_rows,
        "eval_subset": dict(eval_subset),
        "settings": dict(settings),
        "arms": arm_summaries,
        "verdict": verdict,
        "fair_comparison": fair,
        "notes": [
            "Both arms use the same seeds, training data, evaluation rows, "
            "steps, batch size, learning rate, loss weights and MLP/out_proj "
            "geometry; only the fused QKV rank/alpha differs.",
            "The QKV arm adds trainable parameters, so a win can be extra "
            "capacity rather than the attention locus.",
            "No claim is made when the 95% confidence intervals overlap.",
            "Only numeric metrics and paths are stored; prompt and answer "
            "text are never copied into this report.",
        ],
    }
    return summary


def run_ablation(
    *,
    base: str,
    train: str,
    eval_path: str,
    out_dir: str | Path,
    arms: Sequence[str],
    seeds: Sequence[int],
    steps: int = 4000,
    batch_size: int = 4,
    lr: float = 3e-4,
    content_weight: float = 2.0,
    key_weight: float = 3.0,
    value_weight: float = 2.0,
    rank: int = 8,
    alpha: int = 8,
    rank_attn: int = 4,
    alpha_attn: int = 8,
    qkv_rank: int = 4,
    qkv_alpha: int = 8,
    max_eval_rows: int | None = None,
    dry_run: bool = False,
    trainer_command: Sequence[str] | None = None,
    evaluator_command: Sequence[str] | None = None,
    trainer_runner=None,
    evaluator_runner=None,
    timeout: float | None = None,
) -> dict:
    """Plan or run the full arm x seed ablation."""
    settings = _settings(
        arms=arms, seeds=seeds, steps=steps, batch_size=batch_size, lr=lr,
        content_weight=content_weight, key_weight=key_weight,
        value_weight=value_weight, rank=rank, alpha=alpha,
        rank_attn=rank_attn, alpha_attn=alpha_attn, qkv_rank=qkv_rank,
        qkv_alpha=qkv_alpha, max_eval_rows=max_eval_rows)
    if dry_run:
        return _dry_run_payload(
            base=base, train=train, eval_path=eval_path, out_dir=out_dir,
            arms=arms, seeds=seeds, settings=settings,
            trainer_command=trainer_command,
            evaluator_command=evaluator_command)
    rows = read_jsonl(eval_path)
    if not rows:
        raise AblationError("eval file needs at least one row")
    total_rows = len(rows)
    applied = max_eval_rows is not None and max_eval_rows < total_rows
    if applied:
        rows = select_eval_rows(rows, max_eval_rows)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
            prefix="atomzero_qkv_ablation_") as scratch_name:
        scratch = Path(scratch_name)
        eval_used = str(scratch / "eval_subset.jsonl") if applied else eval_path
        if applied:
            write_jsonl(eval_used, rows)
        specs = _plan(
            base=base, train=train, eval_path=eval_used, out_dir=out_dir,
            arms=arms, seeds=seeds, settings=settings, scratch=scratch,
            trainer_command=trainer_command,
            evaluator_command=evaluator_command)
        records = [
            run_single(
                spec, trainer_runner=trainer_runner,
                evaluator_runner=evaluator_runner, timeout=timeout)
            for spec in specs
        ]
        eval_subset = {
            "applied": applied,
            "selection": "sha256_sample" if applied else "all",
            "seed": SUBSET_SEED if applied else None,
            "rows": len(rows),
            "total_rows": total_rows,
            "sha256": rows_digest(rows),
        }
        return build_summary(
            records=records, settings=settings, arms=arms, base=base,
            train=train, eval_path=eval_path, eval_subset=eval_subset)


def _fmt_number(value: Any, places: int = 3) -> str:
    number = _finite_number(value)
    if number is None:
        return "n/a"
    return f"{number:.{places}f}"


def _fmt_aggregate(aggregate: Mapping | None) -> str:
    if not aggregate:
        return "n/a"
    mean = _fmt_number(aggregate.get("mean"))
    std = aggregate.get("std")
    low = _fmt_number(aggregate.get("min"))
    high = _fmt_number(aggregate.get("max"))
    count = aggregate.get("count", 0)
    if std is None:
        return f"{mean} (n={count}, range {low}-{high})"
    return (f"{mean} +/- {_fmt_number(std)} "
            f"(n={count}, range {low}-{high})")


def _fmt_interval(aggregate: Mapping | None) -> str:
    if not aggregate or not aggregate.get("ci95"):
        return "n/a"
    low, high = aggregate["ci95"]
    return f"[{low:.3f}, {high:.3f}]"


def _geometry_text(geometry: Mapping) -> str:
    return (
        f"MLP r{geometry['rank_mlp']}/a{geometry['alpha_mlp']}, "
        f"out_proj r{geometry['rank_attn']}/a{geometry['alpha_attn']}, "
        f"QKV r{geometry['rank_qkv']}/a{geometry['alpha_qkv']}")


def render_markdown(summary: Mapping) -> str:
    """Render the content-free Markdown report."""
    settings = summary["settings"]
    lines = [
        "# QKV LoRA capacity ablation",
        "",
        f"Schema: `{summary['schema']}`  ",
        f"Status: **{summary['status']}**  ",
        f"Created: {summary['created']}",
        "",
        "## Setup",
        "",
        f"- Base: `{summary['base']}`",
        f"- Train: `{summary['train']}`",
        f"- Eval: `{summary['eval']}` "
        f"({summary['eval_rows_used']}/{summary['eval_rows']} rows)",
        f"- Steps: {settings['steps']}, batch size: "
        f"{settings['batch_size']}, lr: {settings['lr']}",
        f"- Loss weights: content {settings['content_weight']}, key "
        f"{settings['key_weight']}, value {settings['value_weight']}",
        f"- Seeds: {', '.join(str(seed) for seed in settings['seeds'])}",
        f"- QKV rank/alpha: {settings['qkv_rank']}/"
        f"{settings['qkv_alpha']} (qkv4 arm only)",
        "",
        "## Per-arm aggregates",
        "",
        "| arm | geometry | runs | exact match | key copy | loss | "
        "content loss |",
        "|---|---|---:|---|---|---|---|",
    ]
    for arm in settings["arms"]:
        entry = summary["arms"][arm]
        aggregate = entry["aggregate"]
        lines.append(
            f"| {arm} | {_geometry_text(entry['geometry'])} | "
            f"{entry['completed']}/{entry['completed'] + entry['failed']} | "
            f"{_fmt_aggregate(aggregate.get('exact'))} | "
            f"{_fmt_aggregate(aggregate.get('key_copy'))} | "
            f"{_fmt_aggregate(aggregate.get('loss'))} | "
            f"{_fmt_aggregate(aggregate.get('content_loss'))} |")
    lines.extend([
        "",
        "## Per-seed rows",
        "",
        "| arm | seed | exact | key copy | key recall | loss | "
        "content loss | status | seconds |",
        "|---|---:|---:|---:|---:|---:|---:|---|---:|",
    ])
    for arm in settings["arms"]:
        for run in summary["arms"][arm]["runs"]:
            lines.append(
                f"| {run['arm']} | {run['seed']} | "
                f"{_fmt_number(run.get('exact'))} | "
                f"{_fmt_number(run.get('key_copy'))} | "
                f"{_fmt_number(run.get('key_recall'))} | "
                f"{_fmt_number(run.get('loss'))} | "
                f"{_fmt_number(run.get('content_loss'))} | "
                f"{run.get('status')} | "
                f"{_fmt_number(run.get('elapsed_seconds'), 1)} |")
    verdict = summary["verdict"]
    lines.extend([
        "",
        "## Verdict",
        "",
        f"- Claim: `{verdict['claim']}`",
        "- QKV improves: "
        + ("yes" if verdict.get("qkv_improves") is True
           else "no" if verdict.get("qkv_improves") is False
           else "not established"),
        f"- Interval method: {verdict['confidence']}",
        "",
        "| metric | qkv4 mean | qkv4 95% CI | mlp mean | mlp 95% CI | "
        "overlap | winner |",
        "|---|---:|---|---:|---|---|---|",
    ])
    qkv_aggregate = summary["arms"].get("qkv4", {}).get("aggregate", {})
    mlp_aggregate = summary["arms"].get("mlp", {}).get("aggregate", {})
    for field in METRIC_FIELDS:
        entry = verdict["metrics"].get(field, {})
        qkv = entry.get("qkv4", {})
        mlp = entry.get("mlp", {})
        overlap = entry.get("interval_overlap")
        overlap_text = ("n/a" if overlap is None
                        else "yes" if overlap else "no")
        lines.append(
            f"| {field} | {_fmt_number(qkv.get('mean'))} | "
            f"{_fmt_interval(qkv_aggregate.get(field))} | "
            f"{_fmt_number(mlp.get('mean'))} | "
            f"{_fmt_interval(mlp_aggregate.get(field))} | "
            f"{overlap_text} | {entry.get('winner') or 'none'} |")
    lines.extend([
        "",
        verdict["note"],
        "",
        "## Fair comparison",
        "",
        summary["fair_comparison"]["note"],
        "",
        "## Notes",
        "",
    ])
    lines.extend(f"- {note}" for note in summary["notes"])
    lines.extend([
        "",
        "No prompt or answer text is stored in this report.",
        "",
    ])
    return "\n".join(lines)


def _markdown_path(result_path: str | Path) -> Path:
    target = Path(result_path)
    if target.suffix.lower() == ".json":
        return target.with_suffix(".md")
    return target.with_name(target.name + ".md")


def write_summary(summary: Mapping, result_path: str | Path
                  ) -> tuple[Path, Path]:
    """Write the JSON summary and the sibling Markdown report."""
    target = Path(result_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    markdown = _markdown_path(target)
    markdown.write_text(render_markdown(summary), encoding="utf-8")
    return target, markdown


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


def _positive_float(value: str) -> float:
    number = _finite_float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return number


def _geometry_error(rank: int, alpha: int, rank_attn: int,
                    alpha_attn: int) -> str | None:
    if rank < 0 or rank_attn < 0:
        return "LoRA ranks must be >= 0"
    if alpha < 0 or alpha_attn < 0:
        return "LoRA alphas must be >= 0"
    if rank > 0 and alpha <= 0:
        return "--alpha must be > 0 when --rank > 0"
    if rank_attn > 0 and alpha_attn <= 0:
        return "--alpha-attn must be > 0 when --rank-attn > 0"
    return None


def main(argv: Sequence[str] | None = None, *,
         trainer_command: Sequence[str] | None = None,
         evaluator_command: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default=DEFAULT_BASE,
                        help=f"base checkpoint directory "
                             f"(default: {DEFAULT_BASE})")
    parser.add_argument(
        "--train", default=DEFAULT_TRAIN,
        help=f"training JSONL (default: {DEFAULT_TRAIN})")
    parser.add_argument(
        "--eval", dest="eval_path", default=DEFAULT_EVAL,
        help=f"fixed evaluation JSONL (default: {DEFAULT_EVAL})")
    parser.add_argument(
        "--out", default=DEFAULT_OUT,
        help=f"adapter/artifact directory (default: {DEFAULT_OUT})")
    parser.add_argument(
        "--result", default=DEFAULT_RESULT,
        help=f"content-free JSON report (default: {DEFAULT_RESULT}); "
             "a sibling .md report is written too")
    parser.add_argument(
        "--arms", default=DEFAULT_ARMS,
        help=f"comma list of arms (default: {DEFAULT_ARMS})")
    parser.add_argument(
        "--seeds", default=DEFAULT_SEEDS,
        help=f"comma list of seeds (default: {DEFAULT_SEEDS})")
    parser.add_argument(
        "--steps", type=_positive_int, default=4000,
        help="sampled rows per seed (default: 4000)")
    parser.add_argument(
        "--batch-size", type=_positive_int, default=4,
        help="rows per optimizer step (default: 4)")
    parser.add_argument(
        "--lr", type=_positive_float, default=3e-4,
        help="learning rate (default: 3e-4)")
    parser.add_argument(
        "--content-weight", type=_finite_float, default=2.0,
        help="alphanumeric content loss multiplier (default: 2)")
    parser.add_argument(
        "--key-weight", type=_non_negative_float, default=3.0,
        help="answer key-name loss multiplier (default: 3)")
    parser.add_argument(
        "--value-weight", type=_non_negative_float, default=2.0,
        help="answer value loss multiplier (default: 2)")
    parser.add_argument("--rank", type=int, default=8,
                        help="LoRA MLP rank (default: 8)")
    parser.add_argument("--alpha", type=int, default=8,
                        help="LoRA MLP alpha (default: 8)")
    parser.add_argument("--rank-attn", type=int, default=4,
                        help="LoRA out_proj rank (default: 4)")
    parser.add_argument("--alpha-attn", type=int, default=8,
                        help="LoRA out_proj alpha (default: 8)")
    parser.add_argument(
        "--qkv-rank", type=int, default=4,
        help="fused QKV LoRA rank for the qkv4 arm (default: 4)")
    parser.add_argument(
        "--qkv-alpha", type=int, default=8,
        help="fused QKV LoRA alpha for the qkv4 arm (default: 8)")
    parser.add_argument(
        "--max-eval-rows", type=_positive_int, default=None,
        help="deterministic eval subset size (default: all rows)")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print the planned commands without training, evaluating or "
             "writing files")
    args = parser.parse_args(argv)
    try:
        arms = parse_arms(args.arms)
        seeds = parse_seeds(args.seeds)
    except AblationError as exc:
        parser.error(str(exc))
    if not math.isfinite(args.content_weight) or args.content_weight < 1.0:
        parser.error("--content-weight must be finite and >= 1.0")
    geometry_error = _geometry_error(
        args.rank, args.alpha, args.rank_attn, args.alpha_attn)
    if geometry_error:
        parser.error(geometry_error)
    if "qkv4" in arms:
        if args.qkv_rank <= 0:
            parser.error("--qkv-rank must be > 0 for the qkv4 arm")
        if args.qkv_alpha <= 0:
            parser.error("--qkv-alpha must be > 0 for the qkv4 arm")
    base = str(_resolve(args.base))
    train = str(_resolve(args.train))
    eval_path = str(_resolve(args.eval_path))
    out_dir = _resolve(args.out)
    result_path = _resolve(args.result)
    if not args.dry_run:
        missing = [
            (label, path) for label, path in
            (("base", base), ("train", train), ("eval", eval_path))
            if not (Path(path).is_dir() if label == "base"
                    else Path(path).is_file())
        ]
        if missing:
            for label, path in missing:
                print(f"missing {label}: {path}", file=sys.stderr)
            return 2
    try:
        summary = run_ablation(
            base=base, train=train, eval_path=eval_path, out_dir=out_dir,
            arms=arms, seeds=seeds, steps=args.steps,
            batch_size=args.batch_size, lr=args.lr,
            content_weight=args.content_weight, key_weight=args.key_weight,
            value_weight=args.value_weight, rank=args.rank, alpha=args.alpha,
            rank_attn=args.rank_attn, alpha_attn=args.alpha_attn,
            qkv_rank=args.qkv_rank, qkv_alpha=args.qkv_alpha,
            max_eval_rows=args.max_eval_rows, dry_run=args.dry_run,
            trainer_command=trainer_command,
            evaluator_command=evaluator_command)
    except AblationError as exc:
        print(f"qkv ablation failed: {exc}", file=sys.stderr)
        return 2
    if args.dry_run:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0
    json_path, markdown_path = write_summary(summary, result_path)
    preview = {
        "status": summary["status"],
        "arms": {
            arm: {
                "completed": summary["arms"][arm]["completed"],
                "failed": summary["arms"][arm]["failed"],
                "exact": summary["arms"][arm]["aggregate"]["exact"],
                "key_copy": summary["arms"][arm]["aggregate"]["key_copy"],
            }
            for arm in arms
        },
        "verdict": summary["verdict"],
    }
    print(json.dumps(preview, indent=2, ensure_ascii=False))
    print(f"wrote {_display_path(json_path)}")
    print(f"wrote {_display_path(markdown_path)}")
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
