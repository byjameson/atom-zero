"""Depth-upcycle a research transformer checkpoint without training.

The product 49.2M base is capacity-limited: three- and four-key copy stays at
0-6% even after distillation. This utility expands the existing 6-layer,
768-dim checkpoint to a deeper research base by copying or interpolating
transformer blocks, then saves a *research* checkpoint for continued
pretraining or fine-tuning on cached data. It never trains, never overwrites
the product checkpoint and never changes product defaults.

    python research/upcycle_model.py `
      --base "$env:USERPROFILE\\.cache\\atomzero\\atom-zero-49m" `
      --out checkpoints/upcycle/atom-zero-49m-8l `
      --layers 8 --mode interpolate --seed 23 `
      --report research/results/upcycle_8l.json

Layer mapping is deterministic: target block ``t`` copies source block
``floor(t * source_layers / target_layers)``. This preserves block order and
duplicates blocks evenly; for a 2x expansion every block is copied once. In
``interpolate`` mode, duplicate occurrences after the first are blended with
the adjacent source block (the previous block for the last source). The first
occurrence of every source block remains an exact copy.

The saved directory contains ``model.pt`` (``config`` / ``state`` / ``step``),
``config.json``, a tokenizer copy and ``upcycle.json``. The sidecar records
``research_control: true``, ``product_weights: false`` and
``weights_changed: true``. The output is a new base: load it with
``research.adapter_runtime.load_model(out)``, not as an adapter attached to the
original base (the depth differs).

This is a research base, not the product 0.8.0 checkpoint and not a capability
claim. Copied depth needs continued training and a full evaluation gate before
it means anything.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Mapping, Sequence

import torch

from atomzero.transformer import TransformerLM

try:
    from research.upcycle_report import build_report, write_report
except ImportError:  # running as `python research/upcycle_model.py`
    from upcycle_report import build_report, write_report

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "atomzero.upcycle.v1"
METHOD = "depth_upcycle_v1"
METADATA_FILE = "upcycle.json"
TOKENIZER_FILE = "chat_en_bpe8k.json"
DEFAULT_TOKENIZER = "data/chat_en_bpe8k.json"
MODES = ("duplicate", "interpolate")
RESEARCH_FLAGS = {
    "research_control": True,
    "product_weights": False,
    "weights_changed": True,
}


class UpcycleError(ValueError):
    """Raised when a checkpoint, option or output path is invalid."""


def resolve_path(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def display_path(path: str | Path | None) -> str | None:
    if path is None:
        return None
    value = Path(path)
    try:
        return value.resolve().relative_to(ROOT).as_posix()
    except (OSError, ValueError):
        return value.resolve().as_posix()


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _sha256(path: str | Path, chunk: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def load_base_checkpoint(base: str | Path
                         ) -> tuple[Path, dict, dict, dict]:
    """Load a base checkpoint read-only and validate its shape."""
    base_dir = resolve_path(base)
    if not base_dir.is_dir():
        raise UpcycleError(f"base checkpoint directory not found: {base_dir}")
    model_path = base_dir / "model.pt"
    if not model_path.is_file():
        raise UpcycleError(f"base checkpoint has no model.pt: {base_dir}")
    try:
        payload = torch.load(model_path, map_location="cpu",
                             weights_only=False)
    except Exception as exc:  # noqa: BLE001 - surface a clear CLI error
        raise UpcycleError(f"cannot load {model_path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise UpcycleError("base model.pt must contain a mapping")
    config = payload.get("config")
    state = payload.get("state")
    if not isinstance(config, Mapping):
        raise UpcycleError("base model.pt has no config mapping")
    if not isinstance(state, Mapping):
        raise UpcycleError("base model.pt has no state mapping")
    for key in ("vocab", "dim", "layers", "heads", "ff"):
        if key not in config:
            raise UpcycleError(f"base config is missing {key!r}")
    for key in ("tok.weight", "pos.weight", "ln.weight", "ln.bias",
                "head.weight"):
        if key not in state:
            raise UpcycleError(f"base state is missing {key!r}")
    if not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise UpcycleError("base state contains non-tensor values")
    layers = int(config["layers"])
    block_ids: set[int] = set()
    for key in state:
        if not key.startswith("blocks."):
            continue
        parts = key.split(".")
        if len(parts) < 3:
            continue
        try:
            block_ids.add(int(parts[1]))
        except ValueError as exc:
            raise UpcycleError(
                f"base state has an invalid block key: {key!r}") from exc
    if block_ids != set(range(layers)):
        raise UpcycleError(
            "base config layers do not match the saved transformer blocks")
    return base_dir, dict(payload), dict(config), dict(state)


def layer_mapping(source_layers: int, target_layers: int) -> list[int]:
    """Return a non-decreasing target -> source block mapping.

    ``floor(t * source / target)`` duplicates blocks evenly while preserving
    order. For target = 2 * source every source block appears exactly twice.
    """
    if isinstance(source_layers, bool) or not isinstance(source_layers, int):
        raise UpcycleError("source_layers must be an integer")
    if isinstance(target_layers, bool) or not isinstance(target_layers, int):
        raise UpcycleError("target_layers must be an integer")
    if source_layers < 1:
        raise UpcycleError("source checkpoint needs at least one block")
    if target_layers <= source_layers:
        raise UpcycleError(
            f"--layers must be greater than the base layer count "
            f"({source_layers}); got {target_layers}")
    mapping = [
        (index * source_layers) // target_layers
        for index in range(target_layers)
    ]
    if mapping[0] != 0 or mapping[-1] != source_layers - 1:
        raise UpcycleError("internal error: layer mapping does not cover the stack")
    if any(left > right for left, right in zip(mapping, mapping[1:])):
        raise UpcycleError("internal error: layer mapping is not ordered")
    return mapping


def interpolate_plan(mapping: Sequence[int], source_layers: int,
                     mode: str) -> list[dict]:
    """Describe blended blocks for ``interpolate`` mode.

    The first occurrence of a source block is an exact copy. Later
    occurrences in the same run are blended toward the adjacent source block
    with alpha = occurrence / run_length.
    """
    if mode not in MODES:
        raise UpcycleError(
            f"unknown mode {mode!r}; choose from {', '.join(MODES)}")
    if mode == "duplicate":
        return []
    if source_layers < 2:
        raise UpcycleError("interpolate mode needs at least two source blocks")
    runs: list[dict] = []
    for target, source in enumerate(mapping):
        if runs and runs[-1]["source"] == source:
            runs[-1]["targets"].append(target)
        else:
            runs.append({"source": int(source), "targets": [target]})
    plan: list[dict] = []
    for run in runs:
        targets = run["targets"]
        if len(targets) < 2:
            continue
        source = run["source"]
        neighbor = source + 1 if source + 1 < source_layers else source - 1
        for occurrence in range(1, len(targets)):
            plan.append({
                "target": targets[occurrence],
                "source": source,
                "neighbor": neighbor,
                "alpha": round(occurrence / len(targets), 6),
            })
    return plan


def _block_suffixes(state: Mapping[str, torch.Tensor],
                    source_layers: int) -> list[str]:
    suffixes: list[str] | None = None
    for index in range(source_layers):
        prefix = f"blocks.{index}."
        current = sorted(
            key[len(prefix):] for key in state if key.startswith(prefix))
        if not current:
            raise UpcycleError(f"base block {index} has no tensors")
        if suffixes is None:
            suffixes = current
        elif current != suffixes:
            raise UpcycleError(
                f"base block {index} has a different tensor layout")
    return list(suffixes or [])


def expand_state(
    state: Mapping[str, torch.Tensor],
    config: Mapping,
    target_layers: int,
    mode: str,
) -> tuple[dict[str, torch.Tensor], list[int], list[dict]]:
    """Build a target state by copying/interpolating source blocks."""
    source_layers = int(config["layers"])
    mapping = layer_mapping(source_layers, target_layers)
    plan = interpolate_plan(mapping, source_layers, mode)
    by_target = {item["target"]: item for item in plan}
    suffixes = _block_suffixes(state, source_layers)
    new_state: dict[str, torch.Tensor] = {
        key: value.detach().clone()
        for key, value in state.items()
        if not key.startswith("blocks.")
    }
    for target, source in enumerate(mapping):
        prefix = f"blocks.{source}."
        item = by_target.get(target)
        for suffix in suffixes:
            key = prefix + suffix
            value = state[key]
            new_key = f"blocks.{target}.{suffix}"
            if item is None or not value.is_floating_point():
                new_state[new_key] = value.detach().clone()
                continue
            neighbor_key = f"blocks.{item['neighbor']}.{suffix}"
            if neighbor_key not in state:
                raise UpcycleError(
                    f"base state is missing interpolation neighbor "
                    f"{neighbor_key}")
            alpha = float(item["alpha"])
            new_state[new_key] = (
                value.detach() * (1.0 - alpha)
                + state[neighbor_key].detach() * alpha
            ).clone()
    return new_state, mapping, plan


def build_model(
    config: Mapping,
    state: Mapping[str, torch.Tensor],
    target_layers: int,
    mode: str,
) -> tuple[TransformerLM, list[int], list[dict], int]:
    """Create the deeper model and return it with mapping and max_seq."""
    max_seq = int(config.get("max_seq", state["pos.weight"].shape[0]))
    new_state, mapping, plan = expand_state(
        state, config, target_layers, mode)
    model = TransformerLM(
        int(config["vocab"]), int(config["dim"]), int(target_layers),
        int(config["heads"]), int(config["ff"]), max_seq=max_seq)
    try:
        model.load_state_dict(new_state, strict=True)
    except RuntimeError as exc:
        raise UpcycleError(f"cannot load the expanded state: {exc}") from exc
    model.eval()
    return model, mapping, plan, max_seq


def verify_forward(model: TransformerLM, *, seed: int = 0,
                   length: int = 8) -> dict:
    """Run one no-grad forward pass and check the logits shape."""
    max_seq = int(getattr(model, "max_seq", 512))
    length = max(1, min(int(length), max_seq))
    generator = torch.Generator().manual_seed(int(seed))
    tokens = torch.randint(
        0, int(model.vocab), (1, length), generator=generator,
        dtype=torch.long)
    with torch.no_grad():
        logits = model(tokens)
    expected = (1, length, int(model.vocab))
    if tuple(logits.shape) != expected:
        raise UpcycleError(
            f"forward check returned {tuple(logits.shape)}, expected "
            f"{expected}")
    if not bool(torch.isfinite(logits).all().item()):
        raise UpcycleError("forward check produced non-finite logits")
    return {
        "ok": True,
        "input_shape": list(tokens.shape),
        "logits_shape": list(logits.shape),
        "finite": True,
    }


def find_tokenizer(base_dir: Path, config: Mapping,
                   override: str | Path = "") -> Path | None:
    """Find the tokenizer next to the base or in the repository default."""
    if override:
        path = resolve_path(override)
        if not path.is_file():
            raise UpcycleError(f"tokenizer not found: {path}")
        return path.resolve()
    candidates: list[Path] = []
    recorded = config.get("tokenizer")
    if isinstance(recorded, str) and recorded:
        candidates.append(resolve_path(recorded))
        candidates.append(base_dir / recorded)
    candidates.extend([
        base_dir / TOKENIZER_FILE,
        base_dir / "tokenizer.json",
        resolve_path(DEFAULT_TOKENIZER),
    ])
    seen: set[Path] = set()
    for candidate in candidates:
        path = Path(candidate)
        if path in seen:
            continue
        seen.add(path)
        if path.is_file():
            return path.resolve()
    return None


def validate_output_path(
    base_dir: Path,
    out_dir: Path,
    *,
    force: bool = False,
    report_path: Path | None = None,
) -> None:
    """Refuse to write into the base or over an existing checkpoint."""
    base_dir = base_dir.resolve()
    out_dir = out_dir.resolve()
    if out_dir == base_dir or _is_within(out_dir, base_dir):
        raise UpcycleError(
            "refusing to write the upcycled checkpoint inside the base "
            f"checkpoint directory: {out_dir}")
    if report_path is not None:
        report = report_path.resolve()
        if report == base_dir or _is_within(report, base_dir):
            raise UpcycleError(
                "refusing to write the report inside the base checkpoint "
                f"directory: {report}")
    if out_dir.exists():
        if not out_dir.is_dir():
            raise UpcycleError(f"--out exists and is not a directory: {out_dir}")
        existing = [
            name for name in ("model.pt", "config.json", METADATA_FILE)
            if (out_dir / name).exists()
        ]
        if existing and not force:
            raise UpcycleError(
                f"--out already contains a checkpoint ({', '.join(existing)}); "
                "pass --force to overwrite it")


def _research_payload(
    *,
    base_dir: Path,
    mode: str,
    seed: int,
    source_layers: int,
    target_layers: int,
    mapping: Sequence[int],
    plan: Sequence[Mapping],
    base_model_sha256: str,
) -> dict:
    return {
        **RESEARCH_FLAGS,
        "method": METHOD,
        "base": display_path(base_dir),
        "base_model_sha256": base_model_sha256,
        "mode": mode,
        "seed": int(seed),
        "source_layers": int(source_layers),
        "target_layers": int(target_layers),
        "layer_mapping": [int(value) for value in mapping],
        "interpolated_blocks": [dict(item) for item in plan],
        "trained": False,
    }


def save_research_checkpoint(
    model: TransformerLM,
    out_dir: Path,
    *,
    config: Mapping,
    report: Mapping,
    forward_check: Mapping,
    tokenizer_source: Path | None,
    research: Mapping,
) -> dict:
    """Write model.pt, config.json, tokenizer copy and the sidecar JSON."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_name = None
    if tokenizer_source is not None:
        destination = out_dir / TOKENIZER_FILE
        if tokenizer_source.resolve() != destination.resolve():
            shutil.copyfile(tokenizer_source, destination)
        tokenizer_name = TOKENIZER_FILE
    state = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
    }
    torch.save(
        {
            "config": dict(config),
            "state": state,
            "step": 0,
            "research": dict(research),
        },
        out_dir / "model.pt")
    config_payload = {
        **dict(config),
        **RESEARCH_FLAGS,
        "method": METHOD,
        "trained": False,
        "tokenizer": tokenizer_name,
        "research": dict(research),
    }
    (out_dir / "config.json").write_text(
        json.dumps(config_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    metadata = {
        "schema": SCHEMA,
        "method": METHOD,
        "created": report.get("created") or time.strftime(
            "%Y-%m-%dT%H:%M:%S"),
        "status": "completed",
        "base": report.get("base"),
        "out": display_path(out_dir),
        "mode": report.get("mode"),
        "seed": report.get("seed"),
        "source_layers": report.get("layers", {}).get("source"),
        "target_layers": report.get("layers", {}).get("target"),
        "layer_mapping": report.get("layer_mapping"),
        "interpolated_blocks": report.get("interpolated_blocks"),
        "parameter_counts": report.get("parameters"),
        "estimates": report.get("estimates"),
        "base_model_sha256": research.get("base_model_sha256"),
        "tokenizer": tokenizer_name,
        "tokenizer_source": display_path(tokenizer_source),
        "forward_check": dict(forward_check),
        "trained": False,
        "checkpoint_files": [
            "model.pt", "config.json", METADATA_FILE,
            *([TOKENIZER_FILE] if tokenizer_name else []),
        ],
        "notes": report.get("notes"),
        **RESEARCH_FLAGS,
    }
    (out_dir / METADATA_FILE).write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    return {
        "checkpoint": display_path(out_dir),
        "model_file": display_path(out_dir / "model.pt"),
        "config_file": display_path(out_dir / "config.json"),
        "metadata_file": display_path(out_dir / METADATA_FILE),
        "tokenizer": tokenizer_name,
    }


def upcycle(
    *,
    base: str | Path,
    out: str | Path,
    layers: int,
    mode: str = "duplicate",
    seed: int = 0,
    dry_run: bool = False,
    force: bool = False,
    tokenizer: str | Path = "",
    report: str | Path | None = None,
    batch_size: int = 4,
) -> dict:
    """Plan or write a depth-upcycled research checkpoint."""
    if mode not in MODES:
        raise UpcycleError(
            f"unknown mode {mode!r}; choose from {', '.join(MODES)}")
    if isinstance(layers, bool) or not isinstance(layers, int):
        raise UpcycleError("--layers must be an integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise UpcycleError("--seed must be a non-negative integer")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) \
            or batch_size < 1:
        raise UpcycleError("--batch-size must be a positive integer")
    base_dir, base_payload, config, state = load_base_checkpoint(base)
    source_layers = int(config["layers"])
    mapping = layer_mapping(source_layers, layers)
    plan = interpolate_plan(mapping, source_layers, mode)
    out_dir = resolve_path(out)
    report_path = resolve_path(report) if report else None
    validate_output_path(
        base_dir, out_dir, force=force, report_path=report_path)
    tokenizer_source = find_tokenizer(base_dir, config, tokenizer)
    if tokenizer_source is None and not dry_run:
        raise UpcycleError(
            "tokenizer not found next to the base checkpoint; pass "
            "--tokenizer PATH")
    base_hash = _sha256(base_dir / "model.pt")
    report_payload = build_report(
        source_config=config,
        target_layers=layers,
        mode=mode,
        seed=seed,
        layer_mapping=mapping,
        interpolated_blocks=plan,
        base=display_path(base_dir),
        output=display_path(out_dir),
        tokenizer=display_path(tokenizer_source),
        max_seq=int(config.get("max_seq", state["pos.weight"].shape[0])),
        batch_size=batch_size,
    )
    if dry_run:
        plan_payload = dict(report_payload)
        plan_payload.update({
            "schema": SCHEMA,
            "report_schema": report_payload["schema"],
            "dry_run": True,
            "status": "dry_run",
            "would_write": [
                display_path(out_dir / "model.pt"),
                display_path(out_dir / "config.json"),
                display_path(out_dir / TOKENIZER_FILE),
                display_path(out_dir / METADATA_FILE),
            ],
            "report_path": display_path(report_path),
            "tokenizer_ready": tokenizer_source is not None,
            "base_model_sha256": base_hash,
            "source_layers": source_layers,
            "target_layers": int(layers),
            "parameter_counts": report_payload["parameters"],
            "estimates": report_payload["estimates"],
        })
        return plan_payload

    model, mapping, plan, _max_seq = build_model(
        config, state, layers, mode)
    actual_parameters = sum(
        parameter.numel() for parameter in model.parameters())
    expected_parameters = report_payload["parameters"]["target"]["parameters"]
    if actual_parameters != expected_parameters:
        raise UpcycleError(
            "target parameter count mismatch: "
            f"model={actual_parameters} report={expected_parameters}")
    forward_check = verify_forward(model, seed=seed)
    research = _research_payload(
        base_dir=base_dir,
        mode=mode,
        seed=seed,
        source_layers=source_layers,
        target_layers=layers,
        mapping=mapping,
        plan=plan,
        base_model_sha256=base_hash,
    )
    target_config = dict(config)
    target_config["layers"] = int(layers)
    saved = save_research_checkpoint(
        model,
        out_dir,
        config=target_config,
        report=report_payload,
        forward_check=forward_check,
        tokenizer_source=tokenizer_source,
        research=research,
    )
    report_payload.update({
        "status": "completed",
        "dry_run": False,
        "checkpoint": saved,
        "forward_check": forward_check,
        "base_step": base_payload.get("step"),
    })
    report_files = None
    if report_path is not None:
        json_path, markdown_path = write_report(report_payload, report_path)
        report_files = {
            "json": display_path(json_path),
            "markdown": display_path(markdown_path),
        }
    return {
        "schema": SCHEMA,
        "status": "completed",
        "dry_run": False,
        "base": display_path(base_dir),
        "out": display_path(out_dir),
        "mode": mode,
        "seed": int(seed),
        "source_layers": source_layers,
        "target_layers": int(layers),
        "layer_mapping": mapping,
        "interpolated_blocks": plan,
        "parameter_counts": report_payload["parameters"],
        "estimates": report_payload["estimates"],
        "tokenizer": saved["tokenizer"],
        "tokenizer_source": display_path(tokenizer_source),
        "forward_check": forward_check,
        "checkpoint": saved,
        "report": report_files,
        "trained": False,
        **RESEARCH_FLAGS,
    }


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid integer: {value!r}") from exc
    if number < 1:
        raise argparse.ArgumentTypeError("value must be >= 1")
    return number


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--base", required=True,
        help="base checkpoint directory containing model.pt and tokenizer")
    parser.add_argument(
        "--out", required=True,
        help="output directory for the research checkpoint")
    parser.add_argument(
        "--layers", type=_positive_int, required=True,
        help="target total layer count (must exceed the base layers)")
    parser.add_argument(
        "--mode", choices=MODES, default="duplicate",
        help="duplicate blocks exactly or interpolate new blocks "
             "(default: duplicate)")
    parser.add_argument(
        "--seed", type=int, default=0,
        help="non-negative seed recorded for the research run and used for "
             "the forward-check input (default: 0)")
    parser.add_argument(
        "--tokenizer", default="",
        help="optional tokenizer file; defaults to the base copy or "
             f"{DEFAULT_TOKENIZER}")
    parser.add_argument(
        "--report", default="",
        help="optional JSON report path; a sibling .md report is written too")
    parser.add_argument(
        "--batch-size", type=_positive_int, default=4,
        help="batch size used only for memory estimates (default: 4)")
    parser.add_argument(
        "--force", action="store_true",
        help="overwrite an existing checkpoint in --out (never the base)")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print the plan and write nothing")
    args = parser.parse_args(argv)
    try:
        result = upcycle(
            base=args.base,
            out=args.out,
            layers=args.layers,
            mode=args.mode,
            seed=args.seed,
            dry_run=args.dry_run,
            force=args.force,
            tokenizer=args.tokenizer,
            report=args.report or None,
            batch_size=args.batch_size,
        )
    except UpcycleError as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
