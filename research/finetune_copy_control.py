"""Full fine-tune copy control - research only.

The LoRA copy-curriculum adapters learn values and formats but copy *seen* key
names instead of copying the keys from the prompt. Before collecting more LoRA
data, this control asks a narrower question: can the base architecture learn
the copy task at all when its own weights are allowed to change?

It loads the same base checkpoint and row format as
:mod:`research.train_adapter_candidate`, trains all parameters (or a frozen
subset), saves a full research checkpoint, and compares it with the base on a
held-out JSONL using :func:`research.compare_adapters.compare`.

    python research/finetune_copy_control.py \
      --base "$env:USERPROFILE\\.cache\\atomzero\\atom-zero-49m" \
      --train data/copy_curriculum_v1/train.jsonl \
      --eval data/copy_curriculum_v1/cross.jsonl \
      --out checkpoints/fullft_copy_control/v0001 \
      --steps 100 --lr 1e-4 --batch-size 4 --content-weight 5.0 \
      --result research/results/full_finetune_control.json \
      --comparison research/results/full_finetune_control_comparison.json

The checkpoint is **not** a product adapter. It contains `model.pt`,
`config.json` and a tokenizer copy. The result JSON records
`product_weights: false`, `weights_changed: true` and
`research_control: true`. Never write this directory into
`data/active_adapter.json`, the promotion path or product defaults.

The comparison JSON is stripped of per-row prompt/answer text before it is
written; only aggregate and per-family metrics are stored.

``--cache-dir`` (default empty) loads a cache built by
:mod:`research.pretokenize` and draws encoded input ids, labels and token
weights from it instead of tokenizing rows in the loop. ``--max-seq`` and
``--content-weight`` must match the cache metadata; a mismatch is an error
rather than a silent re-weighting. Without ``--cache-dir`` the trainer is
unchanged.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

try:
    from research.adapter_runtime import load_model, read_rows
    from research.compare_adapters import compare
    from research.pretokenize import file_sha256, load_training_cache
    from research.train_adapter_candidate import _encode, _token_weights
except ImportError:  # running as `python research/finetune_copy_control.py`
    from adapter_runtime import load_model, read_rows
    from compare_adapters import compare
    from pretokenize import file_sha256, load_training_cache
    from train_adapter_candidate import _encode, _token_weights

ROOT = Path(__file__).resolve().parents[1]
METHOD = "full_finetune_copy_control_v1"
FREEZE_CHOICES = ("none", "embeddings", "first-half")
RESEARCH_FLAGS = {
    "product_weights": False,
    "weights_changed": True,
    "research_control": True,
}
DEFAULT_RESULT = "research/results/full_finetune_control.json"
DEFAULT_COMPARISON = (
    "research/results/full_finetune_control_comparison.json")


def _resolve(path: str) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def _display_path(path: str | Path) -> str:
    value = Path(path)
    try:
        return value.resolve().relative_to(ROOT).as_posix()
    except (OSError, ValueError):
        return value.as_posix()


def set_trainable_parameters(model, freeze: str = "none") -> dict:
    """Select trainable parameters for a research control run.

    ``none`` trains every parameter. ``embeddings`` freezes ``tok`` and
    ``pos``. ``first-half`` also freezes the first ``layers // 2`` transformer
    blocks. The LM head is tied to ``tok.weight``, so freezing embeddings also
    freezes the output projection; this is reported rather than hidden.
    """
    if freeze not in FREEZE_CHOICES:
        raise ValueError(
            f"unknown freeze mode {freeze!r}; expected one of "
            f"{', '.join(FREEZE_CHOICES)}")
    for parameter in model.parameters():
        parameter.requires_grad = True
    frozen_groups = []
    if freeze in ("embeddings", "first-half"):
        model.tok.weight.requires_grad = False
        model.pos.weight.requires_grad = False
        frozen_groups.extend(["tok", "pos"])
    if freeze == "first-half":
        half = len(model.blocks) // 2
        for index, block in enumerate(model.blocks[:half]):
            for parameter in block.parameters():
                parameter.requires_grad = False
            frozen_groups.append(f"blocks.{index}")
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise ValueError(f"freeze mode {freeze!r} leaves no trainable tensors")
    total = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in trainable)
    return {
        "freeze": freeze,
        "frozen_groups": frozen_groups,
        "total_parameters": total,
        "trainable_parameters": trainable_count,
        "frozen_parameters": total - trainable_count,
        "trainable_tensors": len(trainable),
        "frozen_tensors": sum(
            1 for p in model.parameters() if not p.requires_grad),
    }


def kl_anchor_loss(candidate_logits: torch.Tensor,
                   reference_logits: torch.Tensor,
                   labels: torch.Tensor, *, temperature: float = 1.0,
                   ignore_index: int = -100) -> torch.Tensor:
    """KL(reference || candidate) on assistant tokens only.

    The reference is treated as a fixed distribution. ``labels`` selects the
    assistant positions (anything other than ``ignore_index``); prompt and
    padding positions are excluded. The result is the mean per-token KL, with
    the usual ``T**2`` scaling when a temperature other than 1.0 is used.
    """
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and > 0")
    if candidate_logits.shape != reference_logits.shape:
        raise ValueError(
            "candidate and reference logits must have the same shape")
    if labels.shape != candidate_logits.shape[:2]:
        raise ValueError(
            "labels must match the batch and sequence dimensions of logits")
    flat_candidate = candidate_logits.reshape(-1, candidate_logits.shape[-1])
    flat_reference = reference_logits.reshape(-1, reference_logits.shape[-1])
    flat_labels = labels.reshape(-1)
    mask = flat_labels != ignore_index
    if not bool(mask.any()):
        # Keep the graph connected even when a batch has no assistant tokens.
        return candidate_logits.sum() * 0.0
    candidate = flat_candidate[mask] / temperature
    reference = flat_reference[mask] / temperature
    log_candidate = F.log_softmax(candidate, dim=-1)
    probs_reference = F.softmax(reference, dim=-1)
    return F.kl_div(
        log_candidate, probs_reference, reduction="batchmean"
    ) * (temperature ** 2)


def _batch_stream(rows: list[dict], batch_size: int, seed: int):
    """Endless deterministic batches, reshuffled every epoch."""
    generator = torch.Generator().manual_seed(seed)
    while True:
        order = torch.randperm(len(rows), generator=generator).tolist()
        for start in range(0, len(order), batch_size):
            yield [rows[index] for index in order[start:start + batch_size]]


def _cache_batch_stream(cache, batch_size: int, seed: int):
    """Endless deterministic batches of cache indices.

    The permutation is drawn exactly like :func:`_batch_stream`, so an
    all-scorable cache visits rows in the same order as the on-the-fly path.
    """
    generator = torch.Generator().manual_seed(seed)
    while True:
        order = torch.randperm(len(cache), generator=generator).tolist()
        for start in range(0, len(order), batch_size):
            yield order[start:start + batch_size]


def train_control(model, tok, rows: list[dict], *, steps: int, lr: float,
                  batch_size: int = 4, max_seq: int = 256,
                  content_weight: float = 1.0, seed: int = 13,
                  freeze: str = "none", device=None,
                  log_every: int = 50, cache=None,
                  anchor_rows: list[dict] | None = None,
                  anchor_model=None, anchor_beta: float = 0.0,
                  anchor_batch_size: int = 4,
                  anchor_temperature: float = 1.0) -> dict:
    """Train the full model (or a frozen subset) for ``steps`` updates.

    When ``cache`` is a :class:`research.pretokenize.PretokenizedCache`, rows
    are drawn from the cache instead of tokenizing ``rows``. ``rows`` may be
    empty in that mode. The per-row weighted loss is the same expression as
    the on-the-fly path; only the source of ``x``, ``y`` and ``weights``
    changes.
    """
    if cache is None:
        if not rows:
            raise ValueError("training needs at least one row")
        train_examples = len(rows)
    else:
        if len(cache) < 1:
            raise ValueError("training cache needs at least one row")
        train_examples = len(rows) if rows else len(cache)
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if lr <= 0:
        raise ValueError("lr must be > 0")
    if max_seq < 3:
        raise ValueError("max_seq must be >= 3")
    if (not math.isfinite(content_weight)) or content_weight < 1.0:
        raise ValueError("content_weight must be finite and >= 1.0")
    if not math.isfinite(anchor_beta) or anchor_beta < 0:
        raise ValueError("anchor_beta must be finite and >= 0")
    if anchor_beta > 0 and (anchor_model is None or not anchor_rows):
        raise ValueError(
            "anchor_beta > 0 needs anchor_model and anchor_rows")
    if anchor_beta > 0 and anchor_model is model:
        raise ValueError("anchor_model must be a separate frozen model")
    if anchor_batch_size < 1:
        raise ValueError("anchor_batch_size must be >= 1")
    if (not math.isfinite(anchor_temperature)) or anchor_temperature <= 0:
        raise ValueError("anchor_temperature must be finite and > 0")
    target = torch.device(
        device or next(model.parameters()).device)
    torch.manual_seed(seed)
    model.to(target)
    selection = set_trainable_parameters(model, freeze)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr)
    stream = (_batch_stream(rows, batch_size, seed) if cache is None
              else _cache_batch_stream(cache, batch_size, seed))
    anchor_stream = None
    if anchor_beta > 0:
        anchor_model.to(target)
        anchor_model.eval()
        for parameter in anchor_model.parameters():
            parameter.requires_grad = False
        anchor_stream = _batch_stream(
            anchor_rows, anchor_batch_size, seed + 1)
    model.train()
    losses = []
    task_losses: list[float] = []
    anchor_losses: list[float] = []
    empty_batches = 0
    started = time.time()
    while len(losses) < steps:
        batch = next(stream)
        optimizer.zero_grad(set_to_none=True)
        batch_loss = 0.0
        valid = 0
        for item in batch:
            if cache is None:
                encoded = _encode(tok, item, max_seq, target)
                if encoded is None:
                    continue
                x, y = encoded
                weights = _token_weights(tok, y, content_weight).reshape(-1)
            else:
                x, y, weights = cache[int(item)]
                x = x.to(target)
                y = y.to(target)
                weights = weights.reshape(-1).to(target)
            denominator = weights.sum()
            if float(denominator) == 0.0:
                continue
            logits = model(x)
            token_loss = F.cross_entropy(
                logits.reshape(-1, model.vocab), y.reshape(-1),
                ignore_index=-100, reduction="none")
            batch_loss = batch_loss + (
                (token_loss * weights).sum() / denominator)
            valid += 1
        if valid == 0:
            empty_batches += 1
            if empty_batches > max(100, 10 * train_examples):
                raise RuntimeError(
                    "no row produced a scorable assistant token; check the "
                    "tokenizer tags and max_seq")
            continue
        task_loss = batch_loss / valid
        total_loss = task_loss
        anchor_value = None
        if anchor_beta > 0:
            anchor_batch = next(anchor_stream)
            anchor_sum = None
            anchor_valid = 0
            for item in anchor_batch:
                encoded = _encode(tok, item, max_seq, target)
                if encoded is None:
                    continue
                anchor_x, anchor_y = encoded
                with torch.no_grad():
                    reference_logits = anchor_model(anchor_x)
                candidate_logits = model(anchor_x)
                anchor_loss = kl_anchor_loss(
                    candidate_logits, reference_logits, anchor_y,
                    temperature=anchor_temperature)
                anchor_sum = (anchor_loss if anchor_sum is None
                              else anchor_sum + anchor_loss)
                anchor_valid += 1
            if anchor_valid:
                anchor_value = anchor_sum / anchor_valid
                total_loss = total_loss + anchor_beta * anchor_value
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        losses.append(float(total_loss.detach().cpu()))
        task_losses.append(float(task_loss.detach().cpu()))
        if anchor_value is not None:
            anchor_losses.append(float(anchor_value.detach().cpu()))
        if log_every and len(losses) % log_every == 0:
            record = {
                "step": len(losses),
                "loss": round(losses[-1], 5),
                "elapsed_seconds": round(time.time() - started, 2),
            }
            if anchor_value is not None:
                record["anchor_loss"] = round(
                    float(anchor_value.detach().cpu()), 5)
            print(json.dumps(record), flush=True)
    tail = losses[-20:]
    task_tail = task_losses[-20:]
    anchor_tail = anchor_losses[-20:]
    summary = {
        **selection,
        "train_loss": sum(tail) / len(tail),
        "task_loss": sum(task_tail) / len(task_tail),
        "anchor_loss": (sum(anchor_tail) / len(anchor_tail)
                        if anchor_tail else None),
        "anchor_beta": anchor_beta,
        "anchor_rows": len(anchor_rows) if anchor_rows else 0,
        "anchor_batch_size": anchor_batch_size,
        "anchor_temperature": anchor_temperature,
        "train_steps": steps,
        "train_examples": train_examples,
        "batch_size": batch_size,
        "max_seq": max_seq,
        "content_weight": content_weight,
        "lr": lr,
        "seed": seed,
        "device": str(target),
    }
    if cache is not None:
        summary["cache_rows"] = len(cache)
        summary["cache_dir"] = str(getattr(cache, "path", ""))
        summary["cache_sha256"] = (getattr(cache, "meta", {}) or {}).get(
            "sha256", "")
    return summary


def checkpoint_config(model) -> dict:
    """Architecture config for a tiny transformer checkpoint."""
    first = model.blocks[0]
    return {
        "vocab": int(model.vocab),
        "dim": int(model.tok.embedding_dim),
        "layers": len(model.blocks),
        "heads": int(first.attn.num_heads),
        "ff": int(first.mlp[0].out_features),
        "max_seq": int(model.max_seq),
        "arch": "transformer",
    }


def save_checkpoint(model, out: str | Path, *, config: dict,
                    tokenizer=None, metadata: dict | None = None,
                    train_summary: dict | None = None) -> dict:
    """Save a full research checkpoint with model, config and tokenizer."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    research = {
        **dict(metadata or {}),
        "method": METHOD,
        **RESEARCH_FLAGS,
    }
    state = {key: value.detach().cpu()
             for key, value in model.state_dict().items()}
    torch.save({"config": config, "state": state, "research": research},
               out / "model.pt")
    tokenizer_file = None
    if tokenizer is not None and callable(getattr(tokenizer, "save", None)):
        tokenizer.save(str(out / "chat_en_bpe8k.json"))
        tokenizer_file = "chat_en_bpe8k.json"
    payload = {
        **config,
        "research_control": True,
        "product_weights": False,
        "weights_changed": True,
        "method": METHOD,
        "research": research,
        "tokenizer": tokenizer_file,
        "train": dict(train_summary or {}),
    }
    (out / "config.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    return {
        "checkpoint": str(out),
        "model_file": str(out / "model.pt"),
        "config_file": str(out / "config.json"),
        "tokenizer": tokenizer_file,
    }


def _metric_snapshot(metrics) -> dict:
    metrics = metrics if isinstance(metrics, dict) else {}
    exact = metrics.get("exact_match")
    return {
        "loss": metrics.get("loss"),
        "ppl": metrics.get("ppl"),
        "content_loss": metrics.get("content_loss"),
        "syntax_loss": metrics.get("syntax_loss"),
        "tokens": metrics.get("tokens"),
        "content_tokens": metrics.get("content_tokens"),
        "exact_match": exact,
        "exact_match_rate": (
            exact.get("rate") if isinstance(exact, dict) else None),
    }


def _numeric_delta(baseline: dict, control: dict, key: str):
    first = baseline.get(key)
    second = control.get(key)
    if (isinstance(first, bool) or isinstance(second, bool)
            or not isinstance(first, (int, float))
            or not isinstance(second, (int, float))):
        return None
    return second - first


def build_result(comparison: dict, training: dict, *, base: str,
                 checkpoint: str, train_path: str = "", eval_path: str = "",
                 device: str = "") -> dict:
    """Build the content-free research result from a comparison payload."""
    models = (
        comparison.get("models") if isinstance(comparison, dict) else None)
    if not isinstance(models, list) or len(models) < 2:
        raise ValueError(
            "comparison needs at least a base and a control model")
    base_metrics = _metric_snapshot(models[0].get("metrics"))
    control_metrics = _metric_snapshot(models[1].get("metrics"))
    base_ppl = base_metrics.get("ppl")
    control_ppl = control_metrics.get("ppl")
    loss_relative = None
    base_loss = base_metrics.get("loss")
    control_loss = control_metrics.get("loss")
    if (isinstance(base_loss, (int, float)) and not isinstance(base_loss, bool)
            and base_loss and isinstance(control_loss, (int, float))
            and not isinstance(control_loss, bool)):
        loss_relative = control_loss / base_loss - 1.0
    delta = {
        "loss": _numeric_delta(base_metrics, control_metrics, "loss"),
        "loss_relative": loss_relative,
        "ppl_ratio": (
            control_ppl / base_ppl
            if isinstance(base_ppl, (int, float)) and base_ppl
            and isinstance(control_ppl, (int, float)) else None),
        "content_loss": _numeric_delta(
            base_metrics, control_metrics, "content_loss"),
        "syntax_loss": _numeric_delta(
            base_metrics, control_metrics, "syntax_loss"),
        "exact_match_rate": _numeric_delta(
            base_metrics, control_metrics, "exact_match_rate"),
    }
    return {
        **RESEARCH_FLAGS,
        "method": METHOD,
        "base": base,
        "checkpoint": checkpoint,
        "train_data": train_path,
        "eval_data": eval_path,
        "eval_examples": comparison.get("examples"),
        "device": device or training.get("device", ""),
        "freeze": training.get("freeze"),
        "train": training,
        "base_metrics": base_metrics,
        "control_metrics": control_metrics,
        "delta": delta,
        "families": comparison.get("families") or {},
        "base_loss": base_metrics.get("loss"),
        "control_loss": control_metrics.get("loss"),
        "loss_delta": delta["loss"],
        "loss_relative": delta["loss_relative"],
        "base_content_loss": base_metrics.get("content_loss"),
        "control_content_loss": control_metrics.get("content_loss"),
        "content_loss_delta": delta["content_loss"],
        "base_syntax_loss": base_metrics.get("syntax_loss"),
        "control_syntax_loss": control_metrics.get("syntax_loss"),
        "syntax_loss_delta": delta["syntax_loss"],
        "base_exact_match": base_metrics.get("exact_match_rate"),
        "control_exact_match": control_metrics.get("exact_match_rate"),
        "exact_match_delta": delta["exact_match_rate"],
    }


def sanitize_comparison(comparison: dict) -> dict:
    """Drop per-row prompt/answer text before the comparison is persisted."""
    clean = dict(comparison)
    clean_models = []
    for item in comparison.get("models", []):
        entry = dict(item)
        metrics = entry.get("metrics")
        if isinstance(metrics, dict):
            entry["metrics"] = {
                key: value for key, value in metrics.items()
                if key != "rows"}
        clean_models.append(entry)
    clean["models"] = clean_models
    return clean


def write_outputs(result: dict, comparison: dict, result_path: str | Path,
                  comparison_path: str | Path) -> dict:
    """Write the result and a prompt-free copy of the comparison."""
    result_path = Path(result_path)
    comparison_path = Path(comparison_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    clean = sanitize_comparison(comparison)
    comparison_path.write_text(
        json.dumps(clean, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    return clean


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Full fine-tune copy control (research only)")
    parser.add_argument("--base", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-seq", type=int, default=256)
    parser.add_argument("--content-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--result", default=DEFAULT_RESULT)
    parser.add_argument("--comparison", default=DEFAULT_COMPARISON)
    parser.add_argument("--freeze", choices=FREEZE_CHOICES, default="none")
    parser.add_argument(
        "--cache-dir", default="",
        help="pretokenized training cache from research/pretokenize.py; "
             "empty keeps the on-the-fly tokenizer path")
    parser.add_argument(
        "--anchor-model", default="",
        help="frozen reference checkpoint for the KL anchor (e.g. the J "
             "default); required when --anchor-beta > 0")
    parser.add_argument(
        "--anchor-data", default="",
        help="JSONL of content/copy/boundary training rows used for the KL "
             "anchor; required when --anchor-beta > 0")
    parser.add_argument(
        "--anchor-beta", type=float, default=0.0,
        help="weight of the KL(reference || candidate) anchor; 0 disables it")
    parser.add_argument(
        "--anchor-batch-size", type=int, default=4,
        help="rows sampled from --anchor-data per training step")
    parser.add_argument(
        "--anchor-temperature", type=float, default=1.0,
        help="softmax temperature for the KL anchor")
    parser.add_argument(
        "--device", default="",
        help="cpu or cuda; empty auto-detects (CUDA when available)")
    args = parser.parse_args(argv)
    if args.steps < 1:
        parser.error("--steps must be >= 1")
    if args.lr <= 0 or not math.isfinite(args.lr):
        parser.error("--lr must be finite and > 0")
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    if args.max_seq < 3:
        parser.error("--max-seq must be >= 3")
    if not math.isfinite(args.content_weight) or args.content_weight < 1.0:
        parser.error("--content-weight must be finite and >= 1.0")
    if not math.isfinite(args.anchor_beta) or args.anchor_beta < 0:
        parser.error("--anchor-beta must be finite and >= 0")
    if args.anchor_beta > 0:
        if not args.anchor_model or not args.anchor_data:
            parser.error(
                "--anchor-model and --anchor-data are required when "
                "--anchor-beta > 0")
        if args.anchor_batch_size < 1:
            parser.error("--anchor-batch-size must be >= 1")
        if (not math.isfinite(args.anchor_temperature)
                or args.anchor_temperature <= 0):
            parser.error("--anchor-temperature must be finite and > 0")
    result_path = _resolve(args.result)
    comparison_path = _resolve(args.comparison)
    if result_path == comparison_path:
        parser.error("--result and --comparison must be different files")
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    train_path = _resolve(args.train)
    eval_path = _resolve(args.eval)
    train_rows = read_rows(train_path)
    eval_rows = read_rows(eval_path)
    if not train_rows or not eval_rows:
        raise SystemExit("train and eval need at least one row")
    cache = None
    cache_dir = _resolve(args.cache_dir) if args.cache_dir else None
    if cache_dir is not None:
        try:
            cache = load_training_cache(
                cache_dir, max_seq=args.max_seq,
                content_weight=args.content_weight)
        except (OSError, ValueError) as exc:
            raise SystemExit(f"--cache-dir: {exc}") from exc
        cached_source = (cache.meta.get("data") or {}).get("sha256", "")
        if cached_source and cached_source != file_sha256(train_path):
            raise SystemExit(
                "--cache-dir: the cache was built from a different --train "
                "file (sha256 mismatch); rebuild it from this training JSONL "
                "or point --train at the cached source")
    model, tok = load_model(args.base, device=device)
    anchor_rows = None
    anchor_model = None
    anchor_path = None
    if args.anchor_beta > 0:
        anchor_path = _resolve(args.anchor_data)
        anchor_rows = read_rows(anchor_path)
        if not anchor_rows:
            raise SystemExit("--anchor-data needs at least one row")
        anchor_model, _ = load_model(args.anchor_model, device=device)
    training = train_control(
        model, tok, train_rows, steps=args.steps, lr=args.lr,
        batch_size=args.batch_size, max_seq=args.max_seq,
        content_weight=args.content_weight, seed=args.seed,
        freeze=args.freeze, device=device, cache=cache,
        anchor_rows=anchor_rows, anchor_model=anchor_model,
        anchor_beta=args.anchor_beta,
        anchor_batch_size=args.anchor_batch_size,
        anchor_temperature=args.anchor_temperature)
    training["train_data"] = _display_path(train_path)
    out = _resolve(args.out)
    metadata = {
        "base": str(args.base),
        "train_data": _display_path(train_path),
        "steps": args.steps,
        "freeze": args.freeze,
    }
    if cache_dir is not None:
        metadata["cache_dir"] = _display_path(cache_dir)
    if anchor_path is not None:
        metadata["anchor_model"] = str(args.anchor_model)
        metadata["anchor_data"] = _display_path(anchor_path)
        metadata["anchor_beta"] = args.anchor_beta
        metadata["anchor_batch_size"] = args.anchor_batch_size
        metadata["anchor_temperature"] = args.anchor_temperature
    saved = save_checkpoint(
        model, out, config=checkpoint_config(model), tokenizer=tok,
        metadata=metadata, train_summary=training)
    # Free the training state before loading fresh models for the comparison.
    del model
    if anchor_model is not None:
        del anchor_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    comparison = compare(
        args.base, str(eval_path), [str(out)],
        per_family=True, exact_match=True, device=str(device))
    result = build_result(
        comparison, training, base=str(args.base),
        checkpoint=_display_path(saved["checkpoint"]),
        train_path=_display_path(train_path),
        eval_path=_display_path(eval_path), device=str(device))
    sanitized = write_outputs(result, comparison, result_path,
                              comparison_path)
    if not sanitized.get("families"):
        print("warning: eval rows had no family labels", flush=True)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
