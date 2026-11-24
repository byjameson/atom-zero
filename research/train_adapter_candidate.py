"""Train a LoRA adapter on a verified-teacher candidate and measure it.

This is an **external trainer harness**: it lives in ``research``, not in the
``atomzero`` package. It loads the local base checkpoint, trains only the LoRA
parameters, saves an adapter directory and compares base vs adapter on a
held-out JSONL with :mod:`research.compare_adapters`.

    python research/train_adapter_candidate.py \
      --base C:\\path\\to\\atom-zero-49m \
      --train data/adapter_candidates/train.jsonl \
      --eval data/adapter_candidates/eval.jsonl \
      --out checkpoints/adapters_candidate/v0001 \
      --steps 300 --lr 1e-3 --rank 8 --alpha 16 \
      --rank-attn 4 --alpha-attn 8 \
      --result research/results/adapter_candidate.json

``--batch-size`` (default ``1``) accumulates ``N`` rows before each
``optimizer.step()``; the accumulated loss is the token-weighted mean over the
whole group (sum of weighted cross-entropy divided by the summed token
weights). ``--steps`` still counts sampled rows, so ``--batch-size 4`` performs
``ceil(rows_seen / 4)`` optimizer steps. ``--lr-scale {none,linear}`` defaults
to ``none``; ``linear`` explicitly multiplies the learning rate by
``batch_size`` and the effective value is recorded in the result and adapter
config. No automatic LR scaling happens.

``--rank-attn`` (default ``0``) additionally adapts the attention output
projection (``out_proj``). The model uses a fused
``nn.MultiheadAttention.in_proj_weight`` Parameter for Q/K/V; that projection
is not a module and remains frozen. Only ``out_proj`` is adapted with
``--rank-attn``, so the trainable attention parameters are small.

An optional ``--replay-jsonl`` mixes a bounded fraction (``--replay-weight``,
default 0.25) of older rows into the training sequence with
:mod:`research.replay_sampler`. Without it the trainer cycles primary rows
exactly as before.

It does not apply the adapter to the product and does not change `0.8.0`
weights. Promotion is a separate decision.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from atomzero.lora import (
    apply_lora,
    count_lora_params,
    lora_parameters,
    save_adapter,
)

try:
    from research.adapter_runtime import (
        fmt, load_model, mask_user_labels, read_rows, score_rows)
    from research.compare_adapters import compare
    from research.replay_sampler import (
        DEFAULT_REPLAY_WEIGHT, sample_steps, summarize_mix)
except ImportError:  # running as `python research/train_adapter_candidate.py`
    from adapter_runtime import (
        fmt, load_model, mask_user_labels, read_rows, score_rows)
    from compare_adapters import compare
    from replay_sampler import (
        DEFAULT_REPLAY_WEIGHT, sample_steps, summarize_mix)

ROOT = Path(__file__).resolve().parents[1]


def _resolve(path: str) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def _encode(tok, row, max_seq: int, device):
    ids = tok.encode(fmt(row)).ids
    ids = ids[:max_seq]
    if len(ids) < 3:
        return None
    x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
    y = torch.tensor([ids[1:]], dtype=torch.long, device=device)
    user_id = tok.token_to_id("<|user|>")
    assistant_id = tok.token_to_id("<|assistant|>")
    if user_id is None or assistant_id is None:
        raise SystemExit("tokenizer needs <|user|> and <|assistant|>")
    y = mask_user_labels(y, user_id, assistant_id)
    return x, y


def _token_weights(tok, y, content_weight: float):
    weights = torch.ones_like(y, dtype=torch.float)
    weights[y == -100] = 0.0
    if content_weight == 1.0:
        return weights
    for position in range(y.size(1)):
        if int(y[0, position]) == -100:
            continue
        text = tok.decode([int(y[0, position])])
        if any(char.isalnum() for char in text):
            weights[0, position] = content_weight
    return weights


def group_weighted_loss(terms) -> torch.Tensor:
    """Combine per-row ``(weighted_ce_sum, weight_total)`` terms.

    Returns ``sum(weighted_ce_sum) / sum(weight_total)``: a token-weighted
    mean over every row in the group, not a mean of per-row means. For a
    single term this is exactly the legacy row loss
    ``(token_loss * weights).sum() / weights.sum().clamp_min(1e-8)``.
    """
    if not terms:
        raise ValueError("terms must contain at least one row")
    numerator, denominator = terms[0]
    for row_numerator, row_denominator in terms[1:]:
        numerator = numerator + row_numerator
        denominator = denominator + row_denominator
    return numerator / denominator.clamp_min(1e-8)


def _batch_size_error(batch_size) -> str | None:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        return "batch_size must be an int"
    if batch_size < 1:
        return "batch_size must be >= 1"
    return None


def train(base: str, train_rows: list[dict], out: Path, *, steps: int,
          lr: float, rank: int, alpha: int, max_seq: int,
          device, seed: int = 13, content_weight: float = 1.0,
          replay_rows: list[dict] | None = None,
          replay_weight: float = DEFAULT_REPLAY_WEIGHT,
          rank_attn: int = 0, alpha_attn: int = 0,
          batch_size: int = 1, lr_scale: str = "none") -> dict:
    error = _lora_arg_error(rank, alpha, rank_attn, alpha_attn)
    if error:
        raise ValueError(error)
    batch_error = _batch_size_error(batch_size)
    if batch_error:
        raise ValueError(batch_error)
    if lr_scale not in ("none", "linear"):
        raise ValueError("lr_scale must be 'none' or 'linear'")
    effective_lr = lr * batch_size if lr_scale == "linear" else lr
    replay_rows = list(replay_rows or [])
    sequence = sample_steps(
        train_rows, replay_rows, max(1, steps),
        replay_weight=replay_weight, seed=seed)
    mix = summarize_mix(sequence, train_rows, replay_rows)
    torch.manual_seed(seed)
    model, tok = load_model(base, device=device)
    model = apply_lora(
        model, r_attn=rank_attn, r_mlp=rank,
        alpha_attn=alpha_attn, alpha_mlp=alpha)
    model.to(device)
    params = lora_parameters(model)
    if not params:
        raise SystemExit("no LoRA parameters were created")
    optimizer = torch.optim.AdamW(params, lr=effective_lr)
    model.train()
    losses = []
    started = time.time()
    rows_seen = 0
    optimizer_steps = 0
    pending = []

    def step_group(group) -> float:
        # No gradients are produced while a group is collected, so zeroing
        # here is the start of the group's update.
        optimizer.zero_grad(set_to_none=True)
        loss = group_weighted_loss(group)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        return float(loss.detach().cpu())

    for row in sequence:
        encoded = _encode(tok, row, max_seq, device)
        if encoded is None:
            continue
        x, y = encoded
        logits = model(x)
        token_loss = F.cross_entropy(
            logits.reshape(-1, model.vocab), y.reshape(-1),
            ignore_index=-100, reduction="none")
        weights = _token_weights(tok, y, content_weight).reshape(-1)
        pending.append(((token_loss * weights).sum(), weights.sum()))
        rows_seen += 1
        if len(pending) < batch_size:
            continue
        losses.append(step_group(pending))
        pending.clear()
        optimizer_steps += 1
        if optimizer_steps % 50 == 0:
            print(json.dumps({
                "step": optimizer_steps, "loss": round(losses[-1], 5),
                "elapsed_seconds": round(time.time() - started, 2),
            }), flush=True)
    if pending:
        losses.append(step_group(pending))
        pending.clear()
        optimizer_steps += 1
        if optimizer_steps % 50 == 0:
            print(json.dumps({
                "step": optimizer_steps, "loss": round(losses[-1], 5),
                "elapsed_seconds": round(time.time() - started, 2),
            }), flush=True)
    out.mkdir(parents=True, exist_ok=True)
    save_adapter(
        model, out, r_attn=rank_attn, r_mlp=rank,
        alpha_attn=alpha_attn, alpha_mlp=alpha,
        rank=rank, alpha=alpha, steps=steps, lr=lr,
        base=str(base), train_examples=len(train_rows),
        replay_examples=len(replay_rows), replay_weight=replay_weight,
        content_weight=content_weight, batch_size=batch_size,
        optimizer_steps=optimizer_steps, rows_seen=rows_seen,
        lr_scale=lr_scale, effective_lr=effective_lr,
        method="external_train_adapter_candidate_v1")
    tail = losses[-20:] if losses else [0.0]
    try:
        adapter_path = str(out.relative_to(ROOT))
    except ValueError:
        adapter_path = str(out)
    return {
        "train_loss": sum(tail) / len(tail),
        "train_steps": steps,
        "train_examples": len(train_rows),
        "replay_examples": len(replay_rows),
        "replay_weight": replay_weight,
        "content_weight": content_weight,
        "batch_size": batch_size,
        "optimizer_steps": optimizer_steps,
        "rows_seen": rows_seen,
        "lr_scale": lr_scale,
        "effective_lr": effective_lr,
        "train_mix": mix,
        "rank_attn": rank_attn,
        "alpha_attn": alpha_attn,
        "rank_mlp": rank,
        "alpha_mlp": alpha,
        "lora_parameters": count_lora_params(model),
        "adapter": adapter_path,
    }


def _bounded_weight(value: str) -> float:
    try:
        weight = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid replay weight: {value!r}") from exc
    if not math.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise argparse.ArgumentTypeError(
            "replay weight must be between 0.0 and 1.0")
    return weight


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid integer: {value!r}") from exc
    if number < 1:
        raise argparse.ArgumentTypeError("value must be >= 1")
    return number


def _lora_arg_error(rank: int, alpha: int, rank_attn: int,
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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument(
        "--rank-attn", type=int, default=0,
        help="LoRA rank for attention out_proj (0 disables)")
    parser.add_argument(
        "--alpha-attn", type=int, default=0,
        help="LoRA alpha for attention out_proj (>0 when rank-attn >0)")
    parser.add_argument("--max-seq", type=int, default=256)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--batch-size", type=_positive_int, default=1,
        help="rows accumulated per optimizer step (default 1)")
    parser.add_argument(
        "--lr-scale", choices=("none", "linear"), default="none",
        help="none keeps --lr; linear explicitly multiplies it by batch-size")
    parser.add_argument(
        "--replay-jsonl", default=None,
        help="optional older-distribution JSONL mixed into training")
    parser.add_argument(
        "--replay-weight", type=_bounded_weight,
        default=DEFAULT_REPLAY_WEIGHT,
        help="fraction of training steps drawn from --replay-jsonl (0..1)")
    parser.add_argument(
        "--content-weight", type=float, default=1.0,
        help="loss multiplier for alphanumeric content tokens (>=1)")
    parser.add_argument("--result", default="research/results/adapter_candidate.json")
    parser.add_argument(
        "--comparison",
        default="research/results/adapter_candidate_comparison.json")
    args = parser.parse_args(argv)
    lora_error = _lora_arg_error(
        args.rank, args.alpha, args.rank_attn, args.alpha_attn)
    if lora_error:
        parser.error(lora_error)
    if not math.isfinite(args.content_weight) or args.content_weight < 1.0:
        parser.error("--content-weight must be finite and >= 1.0")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_rows = read_rows(_resolve(args.train))
    eval_rows = read_rows(_resolve(args.eval))
    replay_rows = (read_rows(_resolve(args.replay_jsonl))
                   if args.replay_jsonl else [])
    if not train_rows or not eval_rows:
        raise SystemExit("train and eval need at least one row")
    out = _resolve(args.out)
    summary = train(
        args.base, train_rows, out, steps=args.steps, lr=args.lr,
        rank=args.rank, alpha=args.alpha, max_seq=args.max_seq,
        device=device, seed=args.seed, content_weight=args.content_weight,
        replay_rows=replay_rows, replay_weight=args.replay_weight,
        rank_attn=args.rank_attn, alpha_attn=args.alpha_attn,
        batch_size=args.batch_size, lr_scale=args.lr_scale)
    comparison = compare(
        args.base, str(_resolve(args.eval)), [str(out)])
    models = comparison.get("models", [])
    base_metrics = next(
        (item.get("metrics", {}) for item in models
         if item.get("label") == "base"), {})
    adapter_metrics = next(
        (item.get("metrics", {}) for item in models
         if item.get("label") != "base"), {})
    base_loss = float(base_metrics.get("loss", 0.0) or 0.0)
    adapter_loss = float(adapter_metrics.get("loss", 0.0) or 0.0)
    base_ppl = float(base_metrics.get("ppl", 0.0) or 0.0)
    adapter_ppl = float(adapter_metrics.get("ppl", 0.0) or 0.0)
    result = {
        **summary,
        "base_loss": base_loss,
        "adapter_loss": adapter_loss,
        "loss_delta": adapter_loss - base_loss,
        "loss_relative": (adapter_loss / base_loss - 1.0
                          if base_loss else None),
        "base_ppl": base_ppl,
        "adapter_ppl": adapter_ppl,
        "ppl_ratio": adapter_ppl / base_ppl if base_ppl else None,
        "eval_examples": len(eval_rows),
        "device": str(device),
        "decision": ("accept" if base_loss and adapter_loss < base_loss
                     and adapter_ppl <= base_ppl else "reject"),
        "weights_changed": False,
    }
    target = _resolve(args.result)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    comparison_path = _resolve(args.comparison)
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_path.write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
