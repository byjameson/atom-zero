"""Content-free parameter and resource report for depth upcycling.

This module turns a base transformer architecture config and a layer mapping
into exact parameter counts plus rough RTX 3050 6 GB planning estimates. It
never reads training text, never trains and never evaluates a model. The
output is a research report, not a capability claim.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA = "atomzero.upcycle_report.v1"

# Reference point measured for the 6-layer, 768-dim, 49.2M product base on the
# RTX 3050 6 GB: about 0.14 s per optimizer step at batch 16 x seq 128, i.e.
# 2,048 tokens per step. The estimates below scale that rate linearly with
# parameter count. They are planning numbers, not a benchmark.
REFERENCE_PARAMETERS = 49_213_440
REFERENCE_TOKENS_PER_SECOND = 2048 / 0.14
REFERENCE_NOTE = (
    "Linear scaling from a measured 6-layer, 49.2M-parameter step time of "
    "about 0.14 s at batch 16 x seq 128 on an RTX 3050 6 GB. Actual "
    "throughput depends on optimizer, batch size, sequence length, gradient "
    "checkpointing, data loading and thermal state.")
EXISTING_REPLAY_TOKENS = 214_805_319
CHINCHILLA_TOKENS_PER_PARAMETER = 20


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def parameter_counts(config: Mapping, layers: int | None = None,
                     max_seq: int | None = None) -> dict:
    """Exact parameter counts for this decoder-only transformer shape.

    The LM head is tied to ``tok.weight``. ``parameters`` is the number of
    unique parameters (matching ``sum(p.numel() for p in model.parameters())``),
    not the larger state-dict sum that lists the tied tensor twice.
    """
    vocab = _positive_int(config.get("vocab"), "vocab")
    dim = _positive_int(config.get("dim"), "dim")
    ff = _positive_int(config.get("ff"), "ff")
    heads = _positive_int(config.get("heads"), "heads")
    layer_count = _positive_int(
        layers if layers is not None else config.get("layers"), "layers")
    if max_seq is None:
        max_seq = config.get("max_seq", 512)
    max_seq = _positive_int(max_seq, "max_seq")

    tok = vocab * dim
    pos = max_seq * dim
    final_norm = 2 * dim
    block = (
        2 * dim
        + (3 * dim * dim + 3 * dim)
        + (dim * dim + dim)
        + 2 * dim
        + (dim * ff + ff + ff * dim + dim)
    )
    non_block = tok + pos + final_norm
    total = non_block + layer_count * block
    return {
        "layers": layer_count,
        "vocab": vocab,
        "dim": dim,
        "heads": heads,
        "ff": ff,
        "max_seq": max_seq,
        "parameters": total,
        "non_block_parameters": non_block,
        "block_parameters": block,
        "weights_fp32_gb": round(total * 4 / 1e9, 4),
        "state_dict_parameters": total + tok,
    }


def estimate_resources(config: Mapping, counts: Mapping, *,
                       batch_size: int = 4, seq: int | None = None) -> dict:
    """Rough full-fine-tune memory, compute and token-budget estimates.

    AdamW keeps fp32 weights (4 bytes), gradients (4 bytes) and two moments
    (8 bytes), i.e. 16 bytes per parameter. The activation figure is a lower
    bound from the largest per-layer tensors; the practical range adds
    allocator, attention and CUDA-context overhead.
    """
    batch_size = _positive_int(batch_size, "batch_size")
    if seq is None:
        seq = int(config.get("seq", 128))
    seq = _positive_int(seq, "seq")
    layers = _positive_int(counts.get("layers"), "layers")
    parameters = _positive_int(counts.get("parameters"), "parameters")
    dim = _positive_int(config.get("dim"), "dim")
    ff = _positive_int(config.get("ff"), "ff")
    heads = _positive_int(config.get("heads"), "heads")

    training_state_gb = parameters * 16 / 1e9
    activation_gb = (
        layers * batch_size * seq * (6 * dim + ff) * 4 / 1e9)
    attention_gb = batch_size * heads * seq * seq * 4 / 1e9
    activation_lower = activation_gb + attention_gb
    low = training_state_gb + 1.5 * activation_lower + 0.3
    high = training_state_gb + 6.0 * activation_lower + 0.8
    if high <= 3.0:
        fits = "likely"
    elif high <= 5.5:
        fits = "tight"
    else:
        fits = "unlikely"

    rate = REFERENCE_TOKENS_PER_SECOND * REFERENCE_PARAMETERS / parameters
    chinchilla_tokens = CHINCHILLA_TOKENS_PER_PARAMETER * parameters
    return {
        "device": "RTX 3050 Laptop 6 GB (planning target)",
        "assumptions": {
            "optimizer": "AdamW full fine-tune",
            "bytes_per_parameter": 16,
            "batch_size": batch_size,
            "seq": seq,
            "chinchilla_tokens_per_parameter": CHINCHILLA_TOKENS_PER_PARAMETER,
            "reference_tokens_per_second": round(
                REFERENCE_TOKENS_PER_SECOND, 1),
            "reference_parameters": REFERENCE_PARAMETERS,
        },
        "weights_fp32_gb": round(parameters * 4 / 1e9, 3),
        "optimizer_state_gb": round(parameters * 12 / 1e9, 3),
        "training_state_gb": round(training_state_gb, 3),
        "activation_lower_bound_gb": round(activation_lower, 3),
        "estimated_training_gb_range": [round(low, 1), round(high, 1)],
        "fits_6gb": fits,
        "tokens_per_second_estimate": int(round(rate)),
        "chinchilla_tokens": int(chinchilla_tokens),
        "hours_for_chinchilla": round(chinchilla_tokens / rate / 3600, 1),
        "existing_replay_tokens": EXISTING_REPLAY_TOKENS,
        "hours_for_existing_replay": round(
            EXISTING_REPLAY_TOKENS / rate / 3600, 1),
        "reference_note": REFERENCE_NOTE,
        "note": (
            "Rough planning estimate. The lower activation bound omits "
            "allocator, attention and CUDA-context overhead; measure the "
            "actual fit before starting a long run. On 6 GB, reduce batch "
            "size or use gradient accumulation/checkpointing if CUDA runs "
            "out of memory."),
    }


def build_report(
    *,
    source_config: Mapping,
    target_layers: int,
    mode: str,
    seed: int,
    layer_mapping: Sequence[int],
    interpolated_blocks: Sequence[Mapping],
    base: str = "",
    output: str = "",
    tokenizer: str | None = None,
    max_seq: int | None = None,
    batch_size: int = 4,
) -> dict:
    """Build the content-free report document used by the CLI and sidecar."""
    source_counts = parameter_counts(source_config, max_seq=max_seq)
    target_counts = parameter_counts(
        source_config, layers=target_layers, max_seq=max_seq)
    target_config = dict(source_config)
    target_config["layers"] = int(target_layers)
    estimates = estimate_resources(
        target_config, target_counts, batch_size=batch_size)
    added = target_counts["parameters"] - source_counts["parameters"]
    notes = [
        "This is a research base, not the product 0.8.0 checkpoint and not "
        "a capability claim.",
        "No training was run. New layers are copied or interpolated from the "
        "base; copied depth alone does not add capability.",
        "The checkpoint needs continued research pretraining or fine-tuning "
        "on cached data before it can be evaluated fairly.",
        "Parameter counts exclude the tied LM head, which shares storage "
        "with tok.weight.",
        "Memory, throughput and token figures are planning estimates for an "
        "RTX 3050 6 GB, not benchmarks.",
        "Keep the product 0.8.0 checkpoint unchanged. Do not promote this "
        "checkpoint without the full capability, behavior and safety gate.",
    ]
    return {
        "schema": SCHEMA,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "content_free": True,
        "research_control": True,
        "product_weights": False,
        "weights_changed": True,
        "trained": False,
        "base": base,
        "output": output,
        "mode": mode,
        "seed": int(seed),
        "layers": {
            "source": source_counts["layers"],
            "target": target_counts["layers"],
        },
        "layer_mapping": [int(index) for index in layer_mapping],
        "interpolated_blocks": [dict(item) for item in interpolated_blocks],
        "parameters": {
            "source": source_counts,
            "target": target_counts,
            "added": added,
            "added_percent": round(
                100.0 * added / source_counts["parameters"], 1),
            "target_over_source": round(
                target_counts["parameters"] / source_counts["parameters"], 4),
        },
        "estimates": estimates,
        "tokenizer": tokenizer,
        "notes": notes,
    }


def render_markdown(report: Mapping) -> str:
    """Render the content-free report as Markdown."""
    source = report["parameters"]["source"]
    target = report["parameters"]["target"]
    estimates = report["estimates"]
    lines = [
        "# Depth-upcycle report",
        "",
        f"Schema: `{report['schema']}`  ",
        "Status: research only; no training was run.",
        "",
        "## Source and target",
        "",
        f"- Base: `{report.get('base') or 'n/a'}`",
        f"- Output: `{report.get('output') or 'n/a'}`",
        f"- Mode: `{report['mode']}`",
        f"- Seed: {report['seed']}",
        f"- Layers: {source['layers']} -> {target['layers']}",
        f"- Tokenizer: {report.get('tokenizer') or 'not found'}",
        "",
        "## Parameter counts",
        "",
        "| model | layers | parameters | added | fp32 weights |",
        "|---|---:|---:|---:|---:|",
        f"| source | {source['layers']} | {source['parameters']:,} | - | "
        f"{source['weights_fp32_gb']:.3f} GB |",
        f"| target | {target['layers']} | {target['parameters']:,} | "
        f"{report['parameters']['added']:,} "
        f"({report['parameters']['added_percent']:.1f}%) | "
        f"{target['weights_fp32_gb']:.3f} GB |",
        "",
        "The LM head is tied to `tok.weight`; counts exclude the duplicate "
        "state-dict entry. A checkpoint file is roughly the unique parameter "
        "count times four bytes, plus small container overhead.",
        "",
        "## Layer mapping",
        "",
        "Target block -> source block:",
        "",
        f"`{report['layer_mapping']}`",
        "",
    ]
    interpolated = report.get("interpolated_blocks") or []
    if interpolated:
        lines.extend([
            "New blocks use a blend of the source block and its neighbor:",
            "",
            "| target | source | neighbor | alpha |",
            "|---:|---:|---:|---:|",
        ])
        for item in interpolated:
            lines.append(
                f"| {item['target']} | {item['source']} | "
                f"{item['neighbor']} | {item['alpha']:.3f} |")
        lines.append("")
    else:
        lines.extend([
            "Mode `duplicate` copies each selected block exactly; no block "
            "weights are blended.",
            "",
        ])
    lines.extend([
        "## Compute and data (RTX 3050 6 GB)",
        "",
        "| item | estimate |",
        "|---|---:|",
        f"| Full fine-tune state (weights + AdamW) | "
        f"{estimates['training_state_gb']:.2f} GB |",
        f"| Activation lower bound | "
        f"{estimates['activation_lower_bound_gb']:.2f} GB |",
        f"| Practical training memory range | "
        f"{estimates['estimated_training_gb_range'][0]:.1f}-"
        f"{estimates['estimated_training_gb_range'][1]:.1f} GB |",
        f"| Fit on 6 GB | {estimates['fits_6gb']} |",
        f"| Estimated throughput | "
        f"{estimates['tokens_per_second_estimate']:,} tokens/s |",
        f"| Chinchilla-style token target (20/param) | "
        f"{estimates['chinchilla_tokens']:,} tokens |",
        f"| Time for that token target | "
        f"{estimates['hours_for_chinchilla']:.1f} hours |",
        f"| Time for the existing 214.8M replay mix | "
        f"{estimates['hours_for_existing_replay']:.1f} hours |",
        "",
        estimates["note"],
        "",
        "The data budget is usually the binding constraint: the existing "
        "replay mix is about 215M tokens, while a 20-tokens-per-parameter "
        "budget is larger than 1B tokens for both 8 and 12 layers. More clean "
        "data is needed before copied depth can be judged.",
        "",
        "## Caveats",
        "",
    ])
    lines.extend(f"- {note}" for note in report.get("notes", []))
    lines.extend([
        "",
        "## Next step",
        "",
        "Continue research pretraining or fine-tuning on cached data, then "
        "evaluate held-out loss, behavior, safety and the product benchmark. "
        "Do not write this checkpoint into `data/active_adapter.json`, the "
        "promotion path or product defaults.",
        "",
        "No prompt or answer text is stored in this report.",
        "",
    ])
    return "\n".join(lines)


def report_paths(path: str | Path) -> tuple[Path, Path]:
    """Return the JSON and Markdown paths for a report argument."""
    target = Path(path)
    if target.suffix.lower() == ".json":
        json_path = target
    else:
        json_path = target.with_suffix(".json")
    markdown_path = json_path.with_suffix(".md")
    return json_path, markdown_path


def write_report(report: Mapping, path: str | Path) -> tuple[Path, Path]:
    """Write the JSON report and a sibling Markdown report."""
    json_path, markdown_path = report_paths(path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, markdown_path
