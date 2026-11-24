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

``--key-weight`` and ``--value-weight`` weight the key-name and value tokens of
a JSON/KV/YAML assistant answer separately, using the character spans from
:func:`research.adapter_runtime.answer_spans`. Other alphanumeric tokens keep
``--content-weight`` and syntax tokens keep weight 1.0. ``--value-weight``
defaults to the content weight. With the default content weight of 1.0 both
are 1.0, which reproduces the legacy content-only weighting exactly.
Plain-text answers have no key/value spans and fall back to content weighting.

``--init-adapter PATH`` resumes from a saved adapter directory: the LoRA
geometry (``r_attn``/``alpha_attn``/``r_mlp``/``alpha_mlp``) is taken from the
checkpoint and its A/B tensors are loaded strictly before training continues.
Explicit rank/alpha flags must match that geometry. Without the flag the
trainer starts from a fresh LoRA initialization exactly as before.

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
    set_lora_state_dict,
)

try:
    from research.adapter_runtime import (
        answer_spans, assistant_label_mask, fmt, load_model,
        mask_user_labels, read_rows, score_rows)
    from research.compare_adapters import compare
    from research.replay_sampler import (
        DEFAULT_REPLAY_WEIGHT, sample_steps, summarize_mix)
except ImportError:  # running as `python research/train_adapter_candidate.py`
    from adapter_runtime import (
        answer_spans, assistant_label_mask, fmt, load_model,
        mask_user_labels, read_rows, score_rows)
    from compare_adapters import compare
    from replay_sampler import (
        DEFAULT_REPLAY_WEIGHT, sample_steps, summarize_mix)

ROOT = Path(__file__).resolve().parents[1]


def _resolve(path: str) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


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


def _overlaps(span, spans) -> bool:
    start, end = span
    if end <= start:
        return False
    return any(start < other_end and other_start < end
               for other_start, other_end in spans)


def answer_token_weights(tok, row, y, *, key_weight: float = 1.0,
                         value_weight: float | None = None,
                         content_weight: float = 1.0) -> torch.Tensor:
    """Per-token loss weights for one assistant-masked label row.

    ``y`` is the ``(1, seq)`` label tensor returned by :func:`_encode`.
    Ignored labels (``-100``) get 0. Tokens whose target offsets overlap a
    key name or value span from :func:`answer_spans` get ``key_weight`` or
    ``value_weight``; other alphanumeric tokens keep ``content_weight`` and
    syntax tokens stay at 1.0.

    ``value_weight`` defaults to ``content_weight``. When both key and value
    weights are 1.0 this returns ``_token_weights`` unchanged, so the legacy
    content-only weighting is bitwise identical. Tokenizers without usable
    offsets (or plain-text answers without spans) also fall back to
    ``_token_weights``.
    """
    if value_weight is None:
        value_weight = content_weight
    if key_weight == 1.0 and value_weight == 1.0:
        return _token_weights(tok, y, content_weight)
    weights = _token_weights(tok, y, content_weight)
    mask = y != -100
    token_to_id = getattr(tok, "token_to_id", None)
    assistant_id = (token_to_id("<|assistant|>")
                    if callable(token_to_id) else None)
    end_id = token_to_id("<|end|>") if callable(token_to_id) else None
    if assistant_id is not None and end_id is not None:
        turn = assistant_label_mask(y, assistant_id, end_id)
        if bool(turn.any()):
            mask = mask & turn
    weights = weights.masked_fill(~mask, 0.0)
    encoding = tok.encode(fmt(row))
    ids = list(getattr(encoding, "ids", []))
    offsets = getattr(encoding, "offsets", None)
    if offsets is None:
        return weights
    limit = y.size(1) + 1
    ids = ids[:limit]
    offsets = list(offsets)[:limit]
    if len(ids) < 2 or len(offsets) != len(ids):
        return weights
    prefix = f"<|user|> {row['user']} <|assistant|> "
    text = fmt(row)
    if not text.startswith(prefix):
        return weights
    spans = answer_spans(row)
    answer_start = len(prefix)
    key_spans = [(item["start"] + answer_start,
                  item["end"] + answer_start) for item in spans["keys"]]
    value_spans = [(item["start"] + answer_start,
                    item["end"] + answer_start) for item in spans["values"]]
    if not key_spans and not value_spans:
        return weights
    for position in range(y.size(1)):
        if not bool(mask[0, position]):
            continue
        token_index = position + 1
        if token_index >= len(offsets):
            continue
        token_span = tuple(offsets[token_index])
        if _overlaps(token_span, key_spans):
            weights[0, position] = key_weight
        elif _overlaps(token_span, value_spans):
            weights[0, position] = value_weight
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


_GEOMETRY_FLAGS = {
    "r_attn": "--rank-attn",
    "alpha_attn": "--alpha-attn",
    "r_mlp": "--rank",
    "alpha_mlp": "--alpha",
}


def _read_adapter_config(path: Path) -> dict:
    """Read an adapter config from config.json, falling back to adapter.pt."""
    config_path = path / "config.json"
    checkpoint_path = path / "adapter.pt"
    if config_path.is_file():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"cannot read adapter config {config_path}: {exc}") from exc
        if not isinstance(config, dict):
            raise ValueError(
                f"adapter config must be a JSON object: {config_path}")
        return config
    if checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location="cpu",
                                weights_only=False)
        config = (checkpoint.get("config")
                  if isinstance(checkpoint, dict) else None)
        if not isinstance(config, dict) or not config:
            raise ValueError(
                f"adapter checkpoint has no config: {checkpoint_path}")
        return config
    raise FileNotFoundError(
        f"init adapter not found: {path} "
        "(expected config.json or adapter.pt)")


def _config_int(config: dict, key: str, default) -> int:
    value = config.get(key)
    if value is None:
        value = default
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"adapter config {key} must be an integer, got {value!r}"
        ) from exc


def _geometry_from_config(config: dict) -> dict:
    """Normalize an adapter config to its four LoRA geometry values."""
    r_mlp = _config_int(config, "r_mlp", config.get("rank", 0))
    r_attn = _config_int(config, "r_attn", 0)
    alpha_mlp = _config_int(
        config, "alpha_mlp", config.get("alpha", r_mlp))
    alpha_attn = _config_int(config, "alpha_attn", r_attn)
    geometry = {
        "r_attn": r_attn,
        "alpha_attn": alpha_attn,
        "r_mlp": r_mlp,
        "alpha_mlp": alpha_mlp,
    }
    error = _lora_arg_error(r_mlp, alpha_mlp, r_attn, alpha_attn)
    if error:
        raise ValueError(f"adapter config is invalid: {error}")
    return geometry


def init_adapter_geometry(path: str | Path) -> dict:
    """Read the LoRA geometry recorded by an adapter directory."""
    return _geometry_from_config(_read_adapter_config(_resolve(str(path))))


def _check_geometry_override(geometry: dict, path: Path, *, rank=None,
                             alpha=None, rank_attn=None,
                             alpha_attn=None) -> None:
    """Reject explicit rank/alpha values that differ from the checkpoint."""
    supplied = {
        "r_attn": rank_attn,
        "alpha_attn": alpha_attn,
        "r_mlp": rank,
        "alpha_mlp": alpha,
    }
    mismatches = [
        f"{_GEOMETRY_FLAGS[key]}={value} (adapter {key}={geometry[key]})"
        for key, value in supplied.items()
        if value is not None and int(value) != geometry[key]
    ]
    if mismatches:
        raise ValueError(
            "init adapter LoRA geometry mismatch: "
            + ", ".join(mismatches)
            + f"; {path} records r_attn={geometry['r_attn']}, "
            f"alpha_attn={geometry['alpha_attn']}, "
            f"r_mlp={geometry['r_mlp']}, "
            f"alpha_mlp={geometry['alpha_mlp']}. "
            "Omit the rank/alpha flags to reuse the checkpoint geometry.")


def load_init_adapter(model: torch.nn.Module, path: str | Path, *,
                      rank=None, alpha=None, rank_attn=None,
                      alpha_attn=None) -> dict:
    """Rebuild a saved adapter on ``model`` and load its A/B tensors.

    The LoRA geometry comes from the adapter's ``config.json`` (or the
    ``adapter.pt`` config when that file is absent). Explicit ``rank`` /
    ``alpha`` / ``rank_attn`` / ``alpha_attn`` values must match the
    checkpoint exactly; a mismatch raises ``ValueError``. Missing, unexpected
    or differently shaped A/B tensors also raise ``ValueError`` instead of
    being silently dropped. Returns the adapter config with normalized
    geometry.
    """
    adapter_dir = _resolve(str(path))
    config = _read_adapter_config(adapter_dir)
    geometry = _geometry_from_config(config)
    _check_geometry_override(
        geometry, adapter_dir, rank=rank, alpha=alpha,
        rank_attn=rank_attn, alpha_attn=alpha_attn)
    checkpoint_path = adapter_dir / "adapter.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"init adapter checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu",
                            weights_only=False)
    lora = checkpoint.get("lora", {}) if isinstance(checkpoint, dict) else {}
    if not lora:
        raise ValueError(f"init adapter has no LoRA tensors: {checkpoint_path}")
    apply_lora(
        model, r_attn=geometry["r_attn"], r_mlp=geometry["r_mlp"],
        alpha_attn=geometry["alpha_attn"],
        alpha_mlp=geometry["alpha_mlp"])
    set_lora_state_dict(model, lora)
    return {**config, **geometry}


def train(base: str, train_rows: list[dict], out: Path, *, steps: int,
          lr: float, rank: int | None = None, alpha: int | None = None,
          max_seq: int, device, seed: int = 13, content_weight: float = 1.0,
          replay_rows: list[dict] | None = None,
          replay_weight: float = DEFAULT_REPLAY_WEIGHT,
          rank_attn: int | None = None, alpha_attn: int | None = None,
          batch_size: int = 1, lr_scale: str = "none",
          init_adapter: str | Path | None = None,
          key_weight: float = 1.0,
          value_weight: float | None = None) -> dict:
    if value_weight is None:
        value_weight = content_weight
    for name, weight in (("key_weight", key_weight),
                         ("value_weight", value_weight)):
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"{name} must be finite and >= 0")
    init_dir = _resolve(str(init_adapter)) if init_adapter else None
    if init_dir is None:
        rank = 8 if rank is None else rank
        alpha = 16 if alpha is None else alpha
        rank_attn = 0 if rank_attn is None else rank_attn
        alpha_attn = 0 if alpha_attn is None else alpha_attn
    else:
        geometry = init_adapter_geometry(init_dir)
        _check_geometry_override(
            geometry, init_dir, rank=rank, alpha=alpha,
            rank_attn=rank_attn, alpha_attn=alpha_attn)
        rank = geometry["r_mlp"]
        alpha = geometry["alpha_mlp"]
        rank_attn = geometry["r_attn"]
        alpha_attn = geometry["alpha_attn"]
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
    if init_dir is not None:
        load_init_adapter(
            model, init_dir, rank=rank, alpha=alpha,
            rank_attn=rank_attn, alpha_attn=alpha_attn)
    else:
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
        weights = answer_token_weights(
            tok, row, y, key_weight=key_weight, value_weight=value_weight,
            content_weight=content_weight).reshape(-1)
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
    extra_config = {}
    if init_dir is not None:
        extra_config["init_adapter"] = _display_path(init_dir)
    save_adapter(
        model, out, r_attn=rank_attn, r_mlp=rank,
        alpha_attn=alpha_attn, alpha_mlp=alpha,
        rank=rank, alpha=alpha, steps=steps, lr=lr,
        base=str(base), train_examples=len(train_rows),
        replay_examples=len(replay_rows), replay_weight=replay_weight,
        content_weight=content_weight, key_weight=key_weight,
        value_weight=value_weight, batch_size=batch_size,
        optimizer_steps=optimizer_steps, rows_seen=rows_seen,
        lr_scale=lr_scale, effective_lr=effective_lr,
        method="external_train_adapter_candidate_v1", **extra_config)
    tail = losses[-20:] if losses else [0.0]
    adapter_path = _display_path(out)
    summary = {
        "train_loss": sum(tail) / len(tail),
        "train_steps": steps,
        "train_examples": len(train_rows),
        "replay_examples": len(replay_rows),
        "replay_weight": replay_weight,
        "content_weight": content_weight,
        "key_weight": key_weight,
        "value_weight": value_weight,
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
    if init_dir is not None:
        summary["init_adapter"] = _display_path(init_dir)
    return summary


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
    parser.add_argument(
        "--rank", type=int, default=None,
        help="LoRA rank for MLP (default 8, or the --init-adapter value)")
    parser.add_argument(
        "--alpha", type=int, default=None,
        help="LoRA alpha for MLP (default 16, or the --init-adapter value)")
    parser.add_argument(
        "--rank-attn", type=int, default=None,
        help="LoRA rank for attention out_proj "
             "(default 0, or the --init-adapter value)")
    parser.add_argument(
        "--alpha-attn", type=int, default=None,
        help="LoRA alpha for attention out_proj "
             "(default 0, or the --init-adapter value)")
    parser.add_argument(
        "--init-adapter", default="",
        help="adapter directory to resume from (config.json or adapter.pt)")
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
        help="loss multiplier for other alphanumeric tokens (>=1)")
    parser.add_argument(
        "--key-weight", type=float, default=1.0,
        help="loss multiplier for answer key-name tokens (>=0, default 1.0)")
    parser.add_argument(
        "--value-weight", type=float, default=None,
        help="loss multiplier for answer value tokens "
             "(>=0, default: --content-weight)")
    parser.add_argument("--result", default="research/results/adapter_candidate.json")
    parser.add_argument(
        "--comparison",
        default="research/results/adapter_candidate_comparison.json")
    args = parser.parse_args(argv)
    init_adapter = args.init_adapter.strip()
    if init_adapter:
        init_dir = _resolve(init_adapter)
        try:
            geometry = init_adapter_geometry(init_dir)
            _check_geometry_override(
                geometry, init_dir, rank=args.rank, alpha=args.alpha,
                rank_attn=args.rank_attn, alpha_attn=args.alpha_attn)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        rank = geometry["r_mlp"]
        alpha = geometry["alpha_mlp"]
        rank_attn = geometry["r_attn"]
        alpha_attn = geometry["alpha_attn"]
    else:
        rank = 8 if args.rank is None else args.rank
        alpha = 16 if args.alpha is None else args.alpha
        rank_attn = 0 if args.rank_attn is None else args.rank_attn
        alpha_attn = 0 if args.alpha_attn is None else args.alpha_attn
    lora_error = _lora_arg_error(rank, alpha, rank_attn, alpha_attn)
    if lora_error:
        parser.error(lora_error)
    if not math.isfinite(args.content_weight) or args.content_weight < 1.0:
        parser.error("--content-weight must be finite and >= 1.0")
    if not math.isfinite(args.key_weight) or args.key_weight < 0:
        parser.error("--key-weight must be finite and >= 0")
    if args.value_weight is not None and (
            not math.isfinite(args.value_weight) or args.value_weight < 0):
        parser.error("--value-weight must be finite and >= 0")
    value_weight = (args.content_weight if args.value_weight is None
                    else args.value_weight)
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
        rank=rank, alpha=alpha, max_seq=args.max_seq,
        device=device, seed=args.seed, content_weight=args.content_weight,
        replay_rows=replay_rows, replay_weight=args.replay_weight,
        rank_attn=rank_attn, alpha_attn=alpha_attn,
        batch_size=args.batch_size, lr_scale=args.lr_scale,
        init_adapter=init_adapter or None, key_weight=args.key_weight,
        value_weight=value_weight)
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
