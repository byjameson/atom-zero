"""Local runtime helpers for Atom Zero transformer checkpoints and LoRA adapters.

Atom Zero consumes adapter files; it does not train them. The external
``streamft`` / ``continual-lora`` projects can produce an adapter directory
containing ``adapter.pt`` and ``config.json``. This module validates that the
adapter fits a local base checkpoint, applies it with :mod:`atomzero.lora`, and
scores assistant turns on a JSONL holdout.

The helpers deliberately avoid importing the external trainer. A missing
adapter or a shape mismatch is reported as data, not as an import error.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from atomzero.lora import apply_adapter, count_lora_params
from atomzero.transformer import TransformerLM

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOKENIZER = "data/chat_en_bpe8k.json"


def resolve_path(path: str | Path, root: str | Path = ROOT) -> Path:
    """Resolve a path relative to the repository, keeping absolute paths."""
    value = Path(path).expanduser()
    if value.is_absolute():
        return value
    return Path(root) / value


def _tokenizer_path(base_dir: Path, tokenizer: str | Path = "") -> Path:
    candidates = []
    if tokenizer:
        candidates.append(resolve_path(tokenizer))
    candidates.append(base_dir / "chat_en_bpe8k.json")
    candidates.append(resolve_path(DEFAULT_TOKENIZER))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "tokenizer not found; pass --tokenizer or place chat_en_bpe8k.json "
        f"next to the base checkpoint ({base_dir})")


def load_tokenizer(path: str | Path):
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(resolve_path(path)))


def load_model(base: str | Path, adapter: str | Path = "",
               device=None, tokenizer: str | Path = ""):
    """Load a base transformer checkpoint, optionally applying a LoRA adapter.

    Returns ``(model, tokenizer)``. The base directory must contain
    ``model.pt`` with ``config`` and ``state`` keys.
    """
    base_dir = resolve_path(base)
    checkpoint = torch.load(base_dir / "model.pt", map_location="cpu",
                            weights_only=False)
    config = checkpoint["config"]
    model = TransformerLM(config["vocab"], config["dim"], config["layers"],
                          config["heads"], config["ff"])
    model.load_state_dict(checkpoint["state"])
    if adapter:
        apply_adapter(model, resolve_path(adapter))
    target = torch.device(device or "cpu")
    model.to(target)
    model.eval()
    return model, load_tokenizer(_tokenizer_path(base_dir, tokenizer))


def fmt(row: dict) -> str:
    return (f"<|user|> {row['user']} <|assistant|> {row['assistant']} "
            "<|end|>")


def _optional_label(value) -> str | None:
    if not isinstance(value, str):
        return None
    label = value.strip()
    return label or None


def read_rows(path: str | Path) -> list[dict]:
    """Read user/assistant rows from JSONL, skipping malformed examples.

    Optional ``family`` / ``key_family`` labels are preserved when present so
    callers can audit content per family. Rows without labels keep the
    original two-key shape.
    """
    rows = []
    for line in resolve_path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or not row.get("user"):
            continue
        assistant = str(row.get("assistant", "")).strip()
        if not assistant:
            continue
        cleaned = {"user": str(row["user"]), "assistant": assistant}
        for key in ("family", "key_family"):
            label = _optional_label(row.get(key))
            if label is not None:
                cleaned[key] = label
        rows.append(cleaned)
    if not rows:
        raise SystemExit("no user/assistant rows found")
    return rows


def assistant_label_mask(y: torch.Tensor, assistant_id: int,
                         end_id: int) -> torch.Tensor:
    """True on tokens that belong to an assistant turn (tag, text, end)."""
    mask = torch.zeros(y.shape, dtype=torch.bool, device=y.device)
    active = torch.zeros(y.size(0), dtype=torch.bool, device=y.device)
    for t in range(y.size(1)):
        token = y[:, t]
        active = active | (token == assistant_id)
        mask[:, t] = active
        active = active & (token != end_id)
    return mask


def mask_user_labels(y: torch.Tensor, user_id: int,
                     assistant_id: int) -> torch.Tensor:
    """Set user-turn labels to -100 so SFT scores assistant turns only."""
    out = y.clone()
    in_user = torch.zeros(y.size(0), dtype=torch.bool, device=y.device)
    for t in range(y.size(1)):
        token = y[:, t]
        in_user = (in_user | (token == user_id)) & (token != assistant_id)
        out[:, t] = out[:, t].masked_fill(in_user, -100)
    return out


def score_rows(model, tok, rows: list[dict]) -> dict:
    """Assistant-only loss for each row, plus an aggregate."""
    assistant_id = tok.token_to_id("<|assistant|>")
    end_id = tok.token_to_id("<|end|>")
    if assistant_id is None or end_id is None:
        raise SystemExit("tokenizer needs <|assistant|> and <|end|> tokens")
    device = next(model.parameters()).device
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    content_nll = 0.0
    content_tokens = 0
    syntax_nll = 0.0
    syntax_tokens = 0
    per_row = []
    with torch.no_grad():
        for index, row in enumerate(rows):
            ids = tok.encode(fmt(row)).ids
            limit = int(getattr(model, "max_seq", 512)) + 1
            truncated = len(ids) > limit
            ids = ids[:limit]
            if len(ids) < 3:
                continue
            x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
            y = torch.tensor([ids[1:]], dtype=torch.long, device=device)
            mask = assistant_label_mask(y, assistant_id, end_id)
            count = int(mask.sum())
            if count == 0:
                continue
            logits = model(x)
            token_loss = F.cross_entropy(
                logits.reshape(-1, model.vocab), y.reshape(-1),
                ignore_index=-100, reduction="none")
            nll = float((token_loss * mask.reshape(-1)).sum())
            total_nll += nll
            total_tokens += count
            row_content_nll = 0.0
            row_content_tokens = 0
            for position in range(y.size(1)):
                if not bool(mask[0, position]):
                    continue
                token_text = tok.decode([int(y[0, position])])
                loss_value = float(token_loss[position])
                if any(char.isalnum() for char in token_text):
                    content_nll += loss_value
                    content_tokens += 1
                    row_content_nll += loss_value
                    row_content_tokens += 1
                else:
                    syntax_nll += loss_value
                    syntax_tokens += 1
            scored_row = {
                "index": index,
                "user": row["user"][:120],
                "assistant": row["assistant"][:120],
                "loss": nll / count,
                "tokens": count,
                "content_loss": (row_content_nll / row_content_tokens
                                 if row_content_tokens else None),
                "content_tokens": row_content_tokens,
                "truncated": truncated,
            }
            for key in ("family", "key_family"):
                if row.get(key):
                    scored_row[key] = row[key]
            per_row.append(scored_row)
    if total_tokens == 0:
        raise SystemExit("no assistant tokens scored")
    loss = total_nll / total_tokens
    content_loss = (content_nll / content_tokens if content_tokens else None)
    syntax_loss = (syntax_nll / syntax_tokens if syntax_tokens else None)
    return {
        "examples": len(per_row),
        "tokens": total_tokens,
        "loss": loss,
        "ppl": math.exp(min(loss, 20.0)),
        "content_loss": content_loss,
        "content_ppl": (math.exp(min(content_loss, 20.0))
                        if content_loss is not None else None),
        "content_tokens": content_tokens,
        "syntax_loss": syntax_loss,
        "syntax_tokens": syntax_tokens,
        "rows": per_row,
    }


def pack_documents(tok, docs: list[str]) -> torch.Tensor:
    ids = []
    for encoding in tok.encode_batch(docs):
        ids.extend(encoding.ids)
    return torch.tensor(ids, dtype=torch.int64)


def load_tokens(path: str | Path, limit: int = 0) -> torch.Tensor:
    try:
        tokens = torch.load(resolve_path(path), weights_only=True, mmap=True)
    except TypeError:
        tokens = torch.load(resolve_path(path), weights_only=True)
    tokens = tokens.to(torch.int64)
    return tokens[:limit] if limit else tokens


def fixed_windows(tokens: torch.Tensor, seq: int, count: int,
                  batch: int, seed: int, device) -> list[tuple]:
    """Deterministic evaluation windows over a packed token tensor."""
    seq = max(2, min(int(seq), len(tokens) - 1))
    high = len(tokens) - seq - 1
    if high <= 0:
        raise SystemExit("token stream is too short for an evaluation window")
    generator = torch.Generator().manual_seed(seed)
    windows = []
    for _ in range(count):
        starts = torch.randint(0, high, (batch,), generator=generator).tolist()
        xs = torch.stack([tokens[s:s + seq] for s in starts]).to(device)
        ys = torch.stack([tokens[s + 1:s + seq + 1] for s in starts]).to(device)
        windows.append((xs, ys))
    return windows


def evaluate_bpb(model, tok, batches) -> dict:
    """Bits-per-byte and token loss on fixed evaluation windows."""
    model.eval()
    nll = 0.0
    tokens = 0
    nbytes = 0
    with torch.no_grad():
        for x, y in batches:
            logits = model(x)
            nll += float(F.cross_entropy(
                logits.reshape(-1, model.vocab), y.reshape(-1),
                reduction="sum"))
            tokens += y.numel()
            for row in y:
                nbytes += len(tok.decode(row.tolist()).encode("utf-8"))
    loss = nll / max(tokens, 1)
    return {
        "loss": loss,
        "ppl": math.exp(min(loss, 20.0)),
        "bpb": nll / math.log(2) / max(nbytes, 1),
        "tokens": tokens,
        "bytes": nbytes,
    }


def evaluate_assistant(model, tok, batches) -> dict:
    """Loss on assistant turns only, matching the SFT training objective."""
    user_id = tok.token_to_id("<|user|>")
    assistant_id = tok.token_to_id("<|assistant|>")
    if user_id is None or assistant_id is None:
        return {}
    model.eval()
    nll = 0.0
    count = 0
    with torch.no_grad():
        for x, y in batches:
            logits = model(x)
            masked = mask_user_labels(y, user_id, assistant_id)
            nll += float(F.cross_entropy(
                logits.reshape(-1, model.vocab), masked.reshape(-1),
                ignore_index=-100, reduction="sum"))
            count += int((masked != -100).sum())
    if count == 0:
        return {}
    loss = nll / count
    return {
        "loss": loss,
        "ppl": math.exp(min(loss, 20.0)),
        "tokens": count,
    }


def adapter_metadata(adapter: str | Path) -> dict:
    """Read an adapter directory without applying it to a model."""
    adapter_dir = resolve_path(adapter)
    checkpoint_path = adapter_dir / "adapter.pt"
    config_path = adapter_dir / "config.json"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"adapter.pt missing: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu",
                            weights_only=False)
    lora = checkpoint.get("lora", {})
    config = {}
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
    if not config:
        config = checkpoint.get("config", {}) or {}
    parameters = sum(int(value.numel()) for value in lora.values()
                     if torch.is_tensor(value))
    return {
        "path": adapter_dir.as_posix(),
        "config": config,
        "tensors": len(lora),
        "parameters": parameters,
        "files": sorted(path.name for path in adapter_dir.iterdir()
                        if path.is_file()),
    }


def check_adapter(base: str | Path, adapter: str | Path,
                  tokenizer: str | Path = "", device=None) -> dict:
    """Validate that an external adapter loads and runs with a base model."""
    base_dir = resolve_path(base)
    try:
        metadata = adapter_metadata(adapter)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        return {"ok": False, "base": base_dir.as_posix(),
                "adapter": resolve_path(adapter).as_posix(),
                "error": f"{type(exc).__name__}: {exc}"}
    if metadata["tensors"] == 0 or metadata["parameters"] == 0:
        return {"ok": False, "base": base_dir.as_posix(),
                "adapter": resolve_path(adapter).as_posix(),
                "error": "adapter has no LoRA tensors",
                "metadata": metadata}
    try:
        model, tok = load_model(base, adapter, device=device,
                                tokenizer=tokenizer)
        lora_parameters = count_lora_params(model)
        if lora_parameters != metadata["parameters"]:
            raise ValueError(
                "adapter parameter mismatch: "
                f"metadata={metadata['parameters']} "
                f"model={lora_parameters}")
        assistant_id = tok.token_to_id("<|assistant|>")
        end_id = tok.token_to_id("<|end|>")
        if assistant_id is None or end_id is None:
            raise ValueError("tokenizer lacks assistant/end tokens")
        ids = tok.encode(
            "<|user|> adapter check <|assistant|> ok <|end|>").ids[:16]
        if len(ids) < 2:
            raise ValueError("tokenizer produced too few ids")
        tensor = torch.tensor([ids[:-1]], dtype=torch.long,
                              device=next(model.parameters()).device)
        with torch.no_grad():
            logits = model(tensor)
        result = {
            "ok": True,
            "base": base_dir.as_posix(),
            "adapter": resolve_path(adapter).as_posix(),
            "base_config": torch.load(
                base_dir / "model.pt", map_location="cpu",
                weights_only=False)["config"],
            "lora_parameters": count_lora_params(model),
            "logits_shape": list(logits.shape),
            **metadata,
        }
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        result = {
            "ok": False,
            "base": base_dir.as_posix(),
            "adapter": resolve_path(adapter).as_posix(),
            "error": f"{type(exc).__name__}: {exc}",
            "metadata": metadata,
        }
    return result
