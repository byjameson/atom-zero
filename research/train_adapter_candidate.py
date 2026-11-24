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


def train(base: str, train_rows: list[dict], out: Path, *, steps: int,
          lr: float, rank: int, alpha: int, max_seq: int,
          device, seed: int = 13, content_weight: float = 1.0,
          replay_rows: list[dict] | None = None,
          replay_weight: float = DEFAULT_REPLAY_WEIGHT,
          rank_attn: int = 0, alpha_attn: int = 0) -> dict:
    error = _lora_arg_error(rank, alpha, rank_attn, alpha_attn)
    if error:
        raise ValueError(error)
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
    optimizer = torch.optim.AdamW(params, lr=lr)
    model.train()
    losses = []
    started = time.time()
    for step, row in enumerate(sequence):
        encoded = _encode(tok, row, max_seq, device)
        if encoded is None:
            continue
        x, y = encoded
        logits = model(x)
        token_loss = F.cross_entropy(
            logits.reshape(-1, model.vocab), y.reshape(-1),
            ignore_index=-100, reduction="none")
        weights = _token_weights(tok, y, content_weight).reshape(-1)
        loss = (token_loss * weights).sum() / weights.sum().clamp_min(1e-8)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if (step + 1) % 50 == 0:
            print(json.dumps({
                "step": step + 1, "loss": round(losses[-1], 5),
                "elapsed_seconds": round(time.time() - started, 2),
            }), flush=True)
    out.mkdir(parents=True, exist_ok=True)
    save_adapter(
        model, out, r_attn=rank_attn, r_mlp=rank,
        alpha_attn=alpha_attn, alpha_mlp=alpha,
        rank=rank, alpha=alpha, steps=steps, lr=lr,
        base=str(base), train_examples=len(train_rows),
        replay_examples=len(replay_rows), replay_weight=replay_weight,
        content_weight=content_weight,
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
        rank_attn=args.rank_attn, alpha_attn=args.alpha_attn)
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
