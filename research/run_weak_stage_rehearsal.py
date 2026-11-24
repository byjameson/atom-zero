"""Automatically select weak copy-curriculum stages and rehearse them.

The staged copy curriculum has five stages (``echo_kv``, ``echo_json``,
``two_kv``, ``two_json``, ``multi``). Their strengths differ, so the stage that
needs rehearsal changes from adapter to adapter. This runner closes the loop:

1. read a content-free stage-metrics JSON (exact and key-copy per stage), or
   evaluate an adapter on every ``stage_<name>/eval.jsonl`` with
   :func:`research.compare_adapters.compare`;
2. rank stages by ``exact + key_copy`` and build a weighted rehearsal mix
   (50% weakest two, 30% middle, 20% strongest);
3. continue training from that adapter with
   ``research/train_adapter_candidate.py --init-adapter ...``;
4. re-evaluate the new adapter on the same stage eval files and write a
   content-free before/after ledger.

    python research/run_weak_stage_rehearsal.py \\
      --base checkpoints/transformer_behavior_50m_long \\
      --adapter checkpoints/staged_curriculum/multi_recover \\
      --data-dir data/staged_copy_v1 --rows 1000 --seed 23 \\
      --steps 300 --batch-size 1 --lr 5e-4 --content-weight 2 \\
      --out research/results/weak_stage_rehearsal.json \\
      --out-adapter checkpoints/weak_stage_rehearsal

With ``--metrics-json`` the before metrics come from a previously written
content-free summary (``per_stage`` stage -> ``[exact, key_copy]``,
``stages`` stage -> object, or a direct stage -> object mapping). Without it
the adapter is evaluated on all stage eval files first.

This script does not change the trainer, the gate or the model architecture.
It never writes prompts or answers to the ledger, and it does not promote or
merge an adapter. Promotion stays a separate decision.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE_ORDER = ("echo_kv", "echo_json", "two_kv", "two_json", "multi")
DEFAULT_BASE = "checkpoints/transformer_behavior_50m_long"
DEFAULT_DATA_DIR = "data/staged_copy_v1"
DEFAULT_ROWS = 1000
DEFAULT_SEED = 23
DEFAULT_TRAINER_SCRIPT = "research/train_adapter_candidate.py"
MIX_SHARES = {"weakest": 0.5, "middle": 0.3, "strongest": 0.2}
SCHEMA = "atomzero.weak_stage_rehearsal.v1"
TRAINER_RESULT_FIELDS = (
    "train_loss",
    "train_steps",
    "optimizer_steps",
    "lora_parameters",
    "batch_size",
    "lr",
    "seed",
    "content_weight",
    "init_adapter",
)


class RehearsalError(ValueError):
    """Raised when stage metrics, data files or the mix are invalid."""


def resolve(path: str | Path) -> Path:
    """Resolve ``path`` relative to the repository root."""
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def display_path(path: str | Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except (OSError, ValueError):
        return str(path)


def stage_order(stages: Iterable[str]) -> list[str]:
    """Canonical stage order, with unknown stages appended alphabetically."""
    unique = {str(stage) for stage in stages}
    known = [stage for stage in DEFAULT_STAGE_ORDER if stage in unique]
    extra = sorted(unique - set(DEFAULT_STAGE_ORDER))
    return known + extra


def _rate(value: Any, field: str, stage: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RehearsalError(
            f"stage {stage!r} needs a numeric {field} in [0, 1]")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise RehearsalError(
            f"stage {stage!r} has {field}={value!r}; expected [0, 1]")
    return number


def _first_present(mapping: Mapping, names: Sequence[str]) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return None


def _metric_pair(stage: str, value: Any) -> dict[str, float]:
    if isinstance(value, Mapping):
        container = value.get("metrics")
        metrics = container if isinstance(container, Mapping) else value
        # A previous rehearsal ledger stores before/after records. Use the
        # newest available evaluation so it can feed the next cycle.
        for field in ("after", "before"):
            nested = metrics.get(field)
            if isinstance(nested, Mapping):
                metrics = nested
                break
        exact = _first_present(
            metrics, ("exact", "exact_match", "exact_match_rate",
                      "exact_rate"))
        key_copy = _first_present(
            metrics, ("key_copy", "key_copy_rate", "keycopy"))
    elif isinstance(value, (list, tuple)) and len(value) >= 2:
        exact, key_copy = value[0], value[1]
    else:
        raise RehearsalError(
            f"stage {stage!r} must map to a metrics object or "
            "[exact, key_copy]")
    return {
        "exact": _rate(exact, "exact", stage),
        "key_copy": _rate(key_copy, "key_copy", stage),
    }


def normalize_stage_metrics(payload: Mapping) -> dict[str, dict[str, float]]:
    """Extract ``{stage: {exact, key_copy}}`` from a metrics document.

    Accepted shapes are a direct stage mapping, ``{"stages": ...}``,
    ``{"per_stage": ...}`` or ``{"stage_metrics": ...}``. The value can be a
    metrics object or a two-item ``[exact, key_copy]`` sequence. Rates must be
    fractions in ``[0, 1]``; this keeps percent/fraction mix-ups visible.
    """
    if not isinstance(payload, Mapping):
        raise RehearsalError("stage metrics must be a JSON object")
    source: Any = payload
    for key in ("stage_metrics", "per_stage", "stages"):
        if key in payload:
            source = payload[key]
            break
    entries: list[tuple[str, Any]] = []
    if isinstance(source, Mapping):
        entries = [(str(stage), value) for stage, value in source.items()]
    elif isinstance(source, Sequence) and not isinstance(
            source, (str, bytes, bytearray)):
        for index, item in enumerate(source):
            if not isinstance(item, Mapping):
                raise RehearsalError(
                    f"stage metrics entry {index} must be an object")
            stage = item.get("stage", item.get("name"))
            if not isinstance(stage, str) or not stage.strip():
                raise RehearsalError(
                    f"stage metrics entry {index} needs a stage name")
            entries.append((stage.strip(), item))
    else:
        raise RehearsalError(
            "stage metrics need an object or a list of stage objects")
    metrics: dict[str, dict[str, float]] = {}
    for stage, value in entries:
        if stage in metrics:
            raise RehearsalError(f"duplicate stage {stage!r} in metrics")
        metrics[stage] = _metric_pair(stage, value)
    if not metrics:
        raise RehearsalError("stage metrics need at least one stage")
    return {stage: metrics[stage] for stage in stage_order(metrics)}


def load_stage_metrics(path: str | Path) -> dict[str, dict[str, float]]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise RehearsalError(f"cannot read stage metrics: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RehearsalError(f"invalid stage metrics JSON: {exc}") from exc
    return normalize_stage_metrics(payload)


def rank_stages(metrics: Mapping[str, Mapping[str, float]]
                ) -> list[dict[str, Any]]:
    """Rank stages by ``exact + key_copy``, weakest first."""
    if not metrics:
        raise RehearsalError("cannot rank an empty stage metrics mapping")
    ordered = stage_order(metrics)
    ranked = []
    for stage in ordered:
        item = metrics[stage]
        exact = _rate(item.get("exact"), "exact", stage)
        key_copy = _rate(item.get("key_copy"), "key_copy", stage)
        ranked.append({
            "stage": stage,
            "exact": exact,
            "key_copy": key_copy,
            "score": exact + key_copy,
        })
    ranked.sort(key=lambda item: (
        item["score"], ordered.index(item["stage"])))
    for index, item in enumerate(ranked, start=1):
        item["rank"] = index
        item["score"] = round(item["score"], 6)
    return ranked


def stage_groups(ranked: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    """Split a weakest-first ranking into weak/middle/strong groups.

    The canonical five-stage curriculum gives two weakest stages, one middle
    stage and the remaining strongest stages. Smaller curricula are handled
    by leaving groups empty; :func:`mix_weights` redistributes their share.
    """
    if not ranked:
        raise RehearsalError("cannot group an empty ranking")
    stages = [str(item["stage"]) for item in ranked]
    if len(stages) == 1:
        return {"weakest": stages, "middle": [], "strongest": []}
    if len(stages) == 2:
        return {"weakest": stages, "middle": [], "strongest": []}
    if len(stages) == 3:
        return {"weakest": stages[:2], "middle": [stages[2]],
                "strongest": []}
    return {"weakest": stages[:2], "middle": [stages[2]],
            "strongest": stages[3:]}


def mix_weights(ranked: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Per-stage sampling weight for the rehearsal mix.

    Weakest two share 50%, the middle stage gets 30% and the strongest stages
    share 20%. If a group does not exist for a smaller curriculum its share is
    redistributed across the remaining groups, so the weights always sum to 1.
    """
    groups = stage_groups(ranked)
    active = [group for group, names in groups.items() if names]
    total_share = sum(MIX_SHARES[group] for group in active)
    weights: dict[str, float] = {}
    for group in ("weakest", "middle", "strongest"):
        names = groups[group]
        if not names:
            continue
        share = MIX_SHARES[group] / total_share
        for stage in names:
            weights[stage] = share / len(names)
    total = sum(weights.values())
    return {stage: weight / total for stage, weight in weights.items()}


def allocate_counts(weights: Mapping[str, float], rows: int) -> dict[str, int]:
    """Turn fractional weights into integer row counts using remainders."""
    if isinstance(rows, bool) or not isinstance(rows, int):
        raise RehearsalError("rows must be an int")
    if rows < 0:
        raise RehearsalError("rows must be >= 0")
    if not weights:
        raise RehearsalError("mix weights need at least one stage")
    total = sum(float(weight) for weight in weights.values())
    if not math.isfinite(total) or total <= 0:
        raise RehearsalError("mix weights must sum to a positive number")
    raw = {stage: rows * float(weight) / total
           for stage, weight in weights.items()}
    counts = {stage: int(math.floor(value)) for stage, value in raw.items()}
    remaining = rows - sum(counts.values())
    if remaining:
        order = list(raw)
        order.sort(key=lambda stage: (-(raw[stage] - counts[stage]),
                                      order.index(stage)))
        for index in range(remaining):
            counts[order[index % len(order)]] += 1
    return counts


def sample_rehearsal_rows(stage_rows: Mapping[str, Sequence[Mapping]],
                          counts: Mapping[str, int], seed: int
                          ) -> list[Mapping]:
    """Deterministically sample rows per stage and shuffle the mix."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise RehearsalError("seed must be an int")
    rng = random.Random(seed)
    mix: list[Mapping] = []
    for stage, count in counts.items():
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise RehearsalError(f"count for stage {stage!r} must be >= 0")
        rows = list(stage_rows.get(stage) or [])
        if count and not rows:
            raise RehearsalError(f"stage {stage!r} has no training rows")
        for _ in range(count):
            mix.append(rng.choice(rows))
    rng.shuffle(mix)
    return mix


def build_rehearsal_mix(metrics: Mapping[str, Mapping[str, float]],
                        stage_rows: Mapping[str, Sequence[Mapping]],
                        rows: int = DEFAULT_ROWS, seed: int = DEFAULT_SEED
                        ) -> dict[str, Any]:
    """Rank stages, weight them and sample the deterministic training mix."""
    ranked = rank_stages(metrics)
    groups = stage_groups(ranked)
    weights = mix_weights(ranked)
    counts = allocate_counts(weights, rows)
    sample = sample_rehearsal_rows(stage_rows, counts, seed)
    return {
        "ranking": ranked,
        "groups": groups,
        "weights": weights,
        "counts": counts,
        "rows": rows,
        "seed": seed,
        "sample": sample,
    }


def read_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise RehearsalError(f"cannot read {path}: {exc}") from exc
    for number, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RehearsalError(
                f"{path}:{number}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise RehearsalError(f"{path}:{number}: row must be an object")
        rows.append(row)
    return rows


def write_jsonl(path: str | Path, rows: Sequence[Mapping]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def stage_train_path(data_dir: str | Path, stage: str) -> Path:
    return Path(data_dir) / f"stage_{stage}" / "train.jsonl"


def stage_eval_path(data_dir: str | Path, stage: str) -> Path:
    return Path(data_dir) / f"stage_{stage}" / "eval.jsonl"


def discover_stages(data_dir: str | Path) -> list[str]:
    """Find ``stage_<name>`` directories with train and eval files."""
    root = Path(data_dir)
    if not root.is_dir():
        raise RehearsalError(f"data directory not found: {data_dir}")
    stages = []
    for child in root.iterdir():
        if not child.is_dir() or not child.name.startswith("stage_"):
            continue
        stage = child.name[len("stage_"):]
        if not stage:
            continue
        if (child / "train.jsonl").is_file() and (
                child / "eval.jsonl").is_file():
            stages.append(stage)
    if not stages:
        raise RehearsalError(
            f"no stage_<name>/train.jsonl and eval.jsonl under {data_dir}")
    return stage_order(stages)


def load_stage_rows(data_dir: str | Path, stages: Sequence[str]
                    ) -> dict[str, list[dict]]:
    loaded = {}
    for stage in stages:
        path = stage_train_path(data_dir, stage)
        rows = read_jsonl(path)
        if not rows:
            raise RehearsalError(
                f"stage {stage!r} train file is empty: {path}")
        loaded[stage] = rows
    return loaded


def _extract_rate(metrics: Mapping, field: str) -> float | None:
    value = metrics.get(field)
    if isinstance(value, Mapping):
        if value.get("available") is False:
            return None
        value = value.get("rate")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RehearsalError(f"{field} rate must be numeric")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise RehearsalError(f"{field} rate out of range: {value!r}")
    return number


def comparison_stage_metrics(payload: Mapping) -> dict[str, float | None]:
    """Extract adapter exact/key-copy rates from a comparison payload."""
    models = payload.get("models") if isinstance(payload, Mapping) else None
    if not isinstance(models, list) or not models:
        raise RehearsalError("comparison payload has no models")
    if len(models) < 2:
        raise RehearsalError("comparison payload has no adapter model")
    metrics = models[-1].get("metrics")
    if not isinstance(metrics, Mapping):
        raise RehearsalError("adapter comparison has no metrics")
    return {
        "exact": _extract_rate(metrics, "exact_match"),
        "key_copy": _extract_rate(metrics, "key_copy"),
    }


def _default_stage_evaluator(base: str, adapter: str, eval_path: str,
                             stage: str) -> dict[str, float | None]:
    try:
        from research.compare_adapters import compare
    except ImportError:
        # running as `python research/run_weak_stage_rehearsal.py`
        from compare_adapters import compare
    payload = compare(
        base, eval_path, [adapter], exact_match=True, key_copy=True)
    return comparison_stage_metrics(payload)


def evaluate_stages(base: str, adapter: str, data_dir: str | Path,
                    stages: Sequence[str], *, evaluator=None
                    ) -> dict[str, dict[str, float | None]]:
    """Evaluate ``adapter`` on every stage eval file, content-free."""
    evaluate = evaluator or _default_stage_evaluator
    results: dict[str, dict[str, float | None]] = {}
    for stage in stages:
        path = stage_eval_path(data_dir, stage)
        if not path.is_file():
            raise RehearsalError(f"missing stage eval file: {path}")
        metrics = evaluate(base, adapter, str(path), stage)
        if not isinstance(metrics, Mapping):
            raise RehearsalError(
                f"evaluator returned no metrics for stage {stage!r}")
        exact = metrics.get("exact")
        key_copy = metrics.get("key_copy")
        results[stage] = {
            "exact": (None if exact is None
                      else _rate(exact, "exact", stage)),
            "key_copy": (None if key_copy is None
                         else _rate(key_copy, "key_copy", stage)),
        }
    return results


def build_trainer_command(*, base: str, train: str, eval_path: str, out: str,
                          init_adapter: str, steps: int, batch_size: int,
                          lr: float, content_weight: float, seed: int,
                          result: str, comparison: str) -> list[str]:
    """Build the train_adapter_candidate command without running it."""
    return [
        sys.executable,
        DEFAULT_TRAINER_SCRIPT,
        "--base", str(base),
        "--train", str(train),
        "--eval", str(eval_path),
        "--out", str(out),
        "--init-adapter", str(init_adapter),
        "--steps", str(steps),
        "--batch-size", str(batch_size),
        "--lr", str(lr),
        "--content-weight", str(content_weight),
        "--seed", str(seed),
        "--result", str(result),
        "--comparison", str(comparison),
    ]


def _read_trainer_result(path: str | Path | None) -> dict | None:
    if not path:
        return None
    target = Path(path)
    if not target.is_file():
        return None
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, Mapping):
        return None
    return {field: payload[field] for field in TRAINER_RESULT_FIELDS
            if field in payload}


def run_trainer(command: Sequence[str], *, cwd: str | Path | None = None,
                runner=None, adapter_path: str | Path | None = None,
                result_path: str | Path | None = None,
                timeout: float | None = None) -> dict[str, Any]:
    """Run the trainer and return a content-free status record."""
    start = time.monotonic()
    invoke = runner or subprocess.run
    try:
        completed = invoke(
            list(command), cwd=str(cwd or ROOT), capture_output=True,
            text=True, timeout=timeout)
    except Exception as exc:  # subprocess failures are recorded, not raised
        return {
            "status": "failed",
            "returncode": None,
            "elapsed_seconds": round(time.monotonic() - start, 3),
            "adapter_exists": False,
            "error": f"{type(exc).__name__}: {exc}",
            "result": None,
        }
    elapsed = round(time.monotonic() - start, 3)
    returncode = getattr(completed, "returncode", None)
    try:
        returncode = int(returncode) if returncode is not None else None
    except (TypeError, ValueError):
        returncode = None
    adapter_exists = (Path(adapter_path).is_dir()
                      if adapter_path is not None else None)
    status = "completed"
    error = None
    if returncode != 0:
        status = "failed"
        error = "trainer returned a non-zero exit code"
    elif adapter_exists is False:
        status = "failed"
        error = "trainer did not write the adapter directory"
    record = {
        "status": status,
        "returncode": returncode,
        "elapsed_seconds": elapsed,
        "adapter_exists": adapter_exists,
        "result": _read_trainer_result(result_path),
    }
    if error:
        record["error"] = error
    return record


def build_ledger(*, seed: int, rows: int, data_dir: str, base_adapter: str,
                 new_adapter: str, ranking: Sequence[Mapping],
                 groups: Mapping[str, Sequence[str]],
                 weights: Mapping[str, float],
                 counts: Mapping[str, int],
                 before: Mapping[str, Mapping],
                 after: Mapping[str, Mapping] | None,
                 trainer: Mapping[str, Any] | None,
                 status: str, error: Mapping[str, str] | None = None
                 ) -> dict[str, Any]:
    """Build the content-free before/after ledger."""
    group_of = {stage: group for group, names in groups.items()
                for stage in names}
    records = []
    for item in ranking:
        stage = str(item["stage"])
        before_metrics = before.get(stage)
        after_metrics = after.get(stage) if after else None
        delta = None
        if before_metrics and after_metrics:
            delta = {}
            for field in ("exact", "key_copy"):
                if (before_metrics.get(field) is not None
                        and after_metrics.get(field) is not None):
                    delta[field] = round(
                        float(after_metrics[field])
                        - float(before_metrics[field]), 6)
                else:
                    delta[field] = None
        records.append({
            "stage": stage,
            "rank": item.get("rank"),
            "group": group_of.get(stage),
            "score": item.get("score"),
            "before": dict(before_metrics) if before_metrics else None,
            "after": dict(after_metrics) if after_metrics else None,
            "delta": delta,
            "mix_weight": round(float(weights.get(stage, 0.0)), 6),
            "mix_rows": int(counts.get(stage, 0)),
        })
    ledger: dict[str, Any] = {
        "schema": SCHEMA,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "content_free": True,
        "status": status,
        "seed": seed,
        "rows": rows,
        "data_dir": display_path(data_dir),
        "base_adapter": display_path(base_adapter),
        "new_adapter": display_path(new_adapter),
        "stage_order": [str(item["stage"]) for item in ranking],
        "ranking": [dict(item) for item in ranking],
        "mix": {
            "weights": {stage: round(float(weight), 6)
                        for stage, weight in weights.items()},
            "counts": {stage: int(count)
                       for stage, count in counts.items()},
            "total": sum(int(count) for count in counts.values()),
            "groups": {group: list(names) for group, names in groups.items()},
        },
        "stages": records,
        "trainer": dict(trainer) if trainer else {"status": "skipped"},
    }
    if error:
        ledger["error"] = dict(error)
    return ledger


def run_rehearsal(*, base: str, init_adapter: str, data_dir: str,
                  stages: Sequence[str],
                  before: Mapping[str, Mapping[str, float]],
                  stage_rows: Mapping[str, Sequence[Mapping]],
                  rows: int, seed: int, out_adapter: str,
                  train_jsonl: str, eval_jsonl: str,
                  trainer_result: str, trainer_comparison: str,
                  steps: int, batch_size: int, lr: float,
                  content_weight: float, trainer_runner=None,
                  evaluator=None, timeout: float | None = None
                  ) -> dict[str, Any]:
    """Rank, mix, train and re-evaluate; failures become ledger statuses."""
    missing = [stage for stage in stages if stage not in before]
    extra = [stage for stage in before if stage not in stages]
    if missing or extra:
        raise RehearsalError(
            f"stage metrics do not match data stages "
            f"(missing={missing}, extra={extra})")
    mix = build_rehearsal_mix(before, stage_rows, rows=rows, seed=seed)
    ranking = mix["ranking"]
    write_jsonl(train_jsonl, mix["sample"])
    command = build_trainer_command(
        base=base, train=train_jsonl, eval_path=eval_jsonl,
        out=out_adapter, init_adapter=init_adapter, steps=steps,
        batch_size=batch_size, lr=lr, content_weight=content_weight,
        seed=seed, result=trainer_result, comparison=trainer_comparison)
    trainer = run_trainer(
        command, runner=trainer_runner, adapter_path=out_adapter,
        result_path=trainer_result, timeout=timeout)
    trainer["command"] = command
    if trainer["status"] != "completed":
        return build_ledger(
            seed=seed, rows=rows, data_dir=data_dir,
            base_adapter=init_adapter, new_adapter=out_adapter,
            ranking=ranking, groups=mix["groups"], weights=mix["weights"],
            counts=mix["counts"], before=before, after=None,
            trainer=trainer, status="trainer_failed")
    after = None
    error = None
    status = "completed"
    try:
        after = evaluate_stages(
            base, out_adapter, data_dir, stages, evaluator=evaluator)
    except Exception as exc:
        status = "eval_failed"
        error = {"type": type(exc).__name__, "message": str(exc)}
    return build_ledger(
        seed=seed, rows=rows, data_dir=data_dir,
        base_adapter=init_adapter, new_adapter=out_adapter,
        ranking=ranking, groups=mix["groups"], weights=mix["weights"],
        counts=mix["counts"], before=before, after=after, trainer=trainer,
        status=status, error=error)


def _write_json(path: str | Path, payload: Mapping) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")


def _failure_ledger(*, args, status: str, error: Exception) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "content_free": True,
        "status": status,
        "seed": args.seed,
        "rows": args.rows,
        "data_dir": display_path(args.data_dir),
        "base_adapter": display_path(args.adapter),
        "new_adapter": display_path(args.out_adapter),
        "stage_order": [],
        "ranking": [],
        "mix": {"weights": {}, "counts": {}, "total": 0, "groups": {}},
        "stages": [],
        "trainer": {"status": "skipped"},
        "error": {"type": type(error).__name__, "message": str(error)},
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--adapter", required=True,
                        help="existing adapter to rank, rehearse and compare")
    parser.add_argument(
        "--metrics-json", default="",
        help="content-free before metrics; omit to evaluate --adapter first")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS,
                        help="total rehearsal rows (default: 1000)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--out", default="research/results/weak_stage_rehearsal.json",
        help="content-free ledger path")
    parser.add_argument(
        "--out-adapter", default="checkpoints/weak_stage_rehearsal")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--content-weight", type=float, default=2.0)
    parser.add_argument(
        "--train-out", default="",
        help="rehearsal JSONL path (default: sibling of --out-adapter)")
    parser.add_argument(
        "--eval-path", default="",
        help="trainer eval JSONL (default: weakest stage eval)")
    args = parser.parse_args(argv)
    if args.rows < 1:
        parser.error("--rows must be >= 1")
    if args.steps < 0:
        parser.error("--steps must be >= 0")
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    if not math.isfinite(args.lr) or args.lr < 0:
        parser.error("--lr must be finite and >= 0")
    if not math.isfinite(args.content_weight) or args.content_weight < 1:
        parser.error("--content-weight must be finite and >= 1.0")
    data_dir = resolve(args.data_dir)
    base = str(resolve(args.base))
    adapter = str(resolve(args.adapter))
    out_adapter = str(resolve(args.out_adapter))
    out = resolve(args.out)
    train_jsonl = (
        str(resolve(args.train_out)) if args.train_out else
        str(Path(out_adapter).with_name(
            Path(out_adapter).name + "_train.jsonl")))
    trainer_result = str(out.with_name(out.stem + "_trainer.json"))
    trainer_comparison = str(
        out.with_name(out.stem + "_trainer_comparison.json"))

    def fail(status: str, exc: Exception) -> int:
        ledger = _failure_ledger(args=args, status=status, error=exc)
        _write_json(out, ledger)
        print(f"weak-stage rehearsal failed: {exc}", file=sys.stderr)
        print(f"wrote {display_path(out)}")
        return 1

    try:
        if args.metrics_json:
            before = load_stage_metrics(resolve(args.metrics_json))
            stages = list(before)
        else:
            stages = discover_stages(data_dir)
            before = evaluate_stages(base, adapter, data_dir, stages)
    except Exception as exc:
        return fail(
            "metrics_failed" if args.metrics_json else "before_eval_failed",
            exc)
    try:
        stage_rows = load_stage_rows(data_dir, stages)
        weakest = rank_stages(before)[0]["stage"]
        eval_path = (str(resolve(args.eval_path)) if args.eval_path
                     else str(stage_eval_path(data_dir, weakest)))
        ledger = run_rehearsal(
            base=base, init_adapter=adapter, data_dir=str(data_dir),
            stages=stages, before=before, stage_rows=stage_rows,
            rows=args.rows, seed=args.seed, out_adapter=out_adapter,
            train_jsonl=train_jsonl, eval_jsonl=eval_path,
            trainer_result=trainer_result,
            trainer_comparison=trainer_comparison, steps=args.steps,
            batch_size=args.batch_size, lr=args.lr,
            content_weight=args.content_weight)
    except Exception as exc:
        return fail("mix_failed", exc)
    _write_json(out, ledger)
    summary = {
        "status": ledger["status"],
        "seed": ledger["seed"],
        "rows": ledger["rows"],
        "ranking": [
            {"stage": item["stage"], "exact": item["exact"],
             "key_copy": item["key_copy"], "rank": item["rank"]}
            for item in ledger["ranking"]],
        "mix_counts": ledger["mix"]["counts"],
        "trainer": ledger["trainer"].get("status"),
        "stages": [
            {"stage": item["stage"],
             "before": item["before"], "after": item["after"],
             "delta": item["delta"]}
            for item in ledger["stages"]],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"wrote {display_path(out)}")
    return 0 if ledger["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
