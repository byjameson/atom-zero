"""Low-rank adapters (LoRA) for HF transformers and the tiny LM."""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn as nn

ATTN_SUFFIXES = (".q_proj", ".k_proj", ".v_proj", ".o_proj", ".out_proj",
                 ".query", ".key", ".value", ".dense")
MLP_SUFFIXES = (".mlp", ".fc1", ".fc2", ".gate_proj", ".up_proj",
                ".down_proj", ".c_fc", ".c_proj", ".mlp.0", ".mlp.2",
                "mlp.0", "mlp.2")
_TARGET_SUFFIXES = ("mlp.0", "mlp.2")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16):
        super().__init__()
        self.base = base
        for param in self.base.parameters():
            param.requires_grad = False
        self.r = r
        self.scaling = alpha / r
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self._fast_dtype = None

    def set_fast(self, dtype):
        self._fast_dtype = dtype

    @property
    def weight(self):
        """Merged weight, so callers that read ``Linear.weight`` see LoRA.

        ``nn.MultiheadAttention.forward`` passes ``out_proj.weight`` and
        ``out_proj.bias`` straight to its internal linear projection instead
        of calling the module. Exposing the merged weight keeps attention
        LoRA active there.
        """
        delta = (self.B @ self.A) * self.scaling
        return self.base.weight + delta.to(self.base.weight.dtype)

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x):
        out = self.base(x)
        a, b = self.A, self.B
        if a.device != x.device:
            a, b = a.to(x.device), b.to(x.device)
        dtype = self._fast_dtype or out.dtype
        delta = (x.to(dtype) @ a.to(dtype).T @ b.to(dtype).T) * self.scaling
        return out + delta.to(out.dtype)


def _lora_parameter_names(model: nn.Module) -> set[str]:
    """Fully-qualified names of the A/B parameters of every LoRALinear."""
    names = set()
    for module_name, module in model.named_modules():
        if not isinstance(module, LoRALinear):
            continue
        prefix = f"{module_name}." if module_name else ""
        names.add(prefix + "A")
        names.add(prefix + "B")
    return names


def _lora_tensors(state: dict) -> dict:
    return {key: value for key, value in state.items()
            if key.endswith((".A", ".B", "lora_A", "lora_B"))}


def _load_lora_tensors(model: nn.Module, state: dict) -> None:
    """Load A/B tensors and fail loudly if the adapter does not fit."""
    tensors = _lora_tensors(state)
    if not tensors:
        raise ValueError("adapter has no LoRA A/B tensors")
    expected = _lora_parameter_names(model)
    missing = sorted(expected - tensors.keys())
    unexpected = sorted(tensors.keys() - expected)
    if missing or unexpected:
        parts = []
        if missing:
            parts.append(f"missing={missing}")
        if unexpected:
            parts.append(f"unexpected={unexpected}")
        raise ValueError("adapter LoRA tensors do not match model: "
                         + ", ".join(parts))
    model.load_state_dict(tensors, strict=False)


def _infer_lora_config(model: nn.Module) -> dict:
    """Recover r/alpha per target group from the LoRALinear modules."""
    found = {}
    for name, module in model.named_modules():
        if not isinstance(module, LoRALinear):
            continue
        if name.endswith(ATTN_SUFFIXES):
            group = "attn"
        elif name.endswith(MLP_SUFFIXES):
            group = "mlp"
        else:
            continue
        found.setdefault(group, (module.r, int(round(
            module.scaling * module.r))))
    r_attn, alpha_attn = found.get("attn", (0, 0))
    r_mlp, alpha_mlp = found.get("mlp", (0, 0))
    return {"r_attn": r_attn, "alpha_attn": alpha_attn,
            "r_mlp": r_mlp, "alpha_mlp": alpha_mlp}


def apply_lora(model: nn.Module, r_attn: int = 8, r_mlp: int = 0,
               alpha_attn: int | None = None,
               alpha_mlp: int | None = None) -> nn.Module:
    """Replace target Linear modules with LoRALinear and freeze the base.

    With ``r_attn > 0`` the attention ``out_proj`` Linear is wrapped. The
    fused QKV ``in_proj_weight`` is a Parameter, not a module, so it cannot be
    adapted and stays frozen. MLP linears are wrapped when ``r_mlp > 0``.
    """
    if r_attn < 0 or r_mlp < 0:
        raise ValueError("LoRA ranks must be >= 0")
    for label, r, alpha in (("attn", r_attn, alpha_attn),
                            ("mlp", r_mlp, alpha_mlp)):
        if r > 0 and alpha is not None and alpha <= 0:
            raise ValueError(
                f"{label} LoRA alpha must be > 0 when rank > 0")
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear) or isinstance(module, LoRALinear):
            continue
        if not name.endswith(ATTN_SUFFIXES + MLP_SUFFIXES):
            continue
        is_attn = name.endswith(ATTN_SUFFIXES)
        r = r_attn if is_attn else r_mlp
        if r <= 0:
            continue
        alpha = alpha_attn if is_attn else alpha_mlp
        if alpha is None:
            alpha = r
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child, LoRALinear(module, r, int(alpha)))
    for param in model.parameters():
        param.requires_grad = False
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.A.requires_grad = True
            module.B.requires_grad = True
    return model


def inject_lora(model: nn.Module, rank: int = 8, alpha: int = 16,
                **kwargs) -> nn.Module:
    """Tiny-LM helper: adapt MLP layers only."""
    return apply_lora(model, r_attn=0, r_mlp=rank, alpha_mlp=alpha)


def count_lora_params(model: nn.Module) -> int:
    return sum(p.numel() for p in lora_parameters(model))


def lora_parameters(model: nn.Module):
    return [p for p in model.parameters() if p.requires_grad]


def lora_state_dict(model: nn.Module) -> dict:
    return {k: v.detach().cpu() for k, v in model.state_dict().items()
            if k.endswith(".A") or k.endswith(".B")
            or k.endswith("lora_A") or k.endswith("lora_B")}


def set_lora_state_dict(model: nn.Module, state: dict):
    """Load A/B tensors from a full or adapter-only state dict.

    Raises instead of silently dropping tensors that the model cannot use.
    """
    _load_lora_tensors(model, state)


def merge_lora(model: nn.Module):
    with torch.no_grad():
        for module in model.modules():
            if not isinstance(module, LoRALinear):
                continue
            delta = (module.B @ module.A) * module.scaling
            module.base.weight.add_(delta.to(module.base.weight.dtype))
            module.A.zero_()
            module.B.zero_()


def save_adapter(model: nn.Module, out: str | Path, **config) -> Path:
    """Save A/B tensors plus a config that can rebuild the same modules.

    ``r_attn``/``alpha_attn``/``r_mlp``/``alpha_mlp`` are inferred from the
    model when the caller does not pass them, so an attention adapter can
    always be reloaded with :func:`apply_adapter`.
    """
    out = Path(out)
    for key, value in _infer_lora_config(model).items():
        config.setdefault(key, value)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"lora": lora_state_dict(model), "config": config},
               out / "adapter.pt")
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n",
                                     encoding="utf-8")
    return out


def apply_adapter(model: nn.Module, path: str | Path) -> dict:
    ck = torch.load(Path(path) / "adapter.pt", map_location="cpu",
                    weights_only=False)
    config = ck.get("config", {}) or {}
    apply_lora(model, r_attn=int(config.get("r_attn", 0) or 0),
               r_mlp=int(config.get("r_mlp", config.get("rank", 8)) or 0),
               alpha_attn=config.get("alpha_attn"),
               alpha_mlp=config.get("alpha_mlp", config.get("alpha")))
    _load_lora_tensors(model, ck.get("lora", {}))
    model.eval()
    return config
