"""Low-rank adapters (LoRA) for HF transformers and the tiny LM."""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

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


class LoRAMultiheadAttention(nn.Module):
    """Low-rank adapter for the fused Q/K/V projection of an MHA module.

    ``nn.MultiheadAttention`` stores Q, K and V in one ``in_proj_weight``
    Parameter of shape ``(3 * embed_dim, embed_dim)``, so the usual
    :class:`LoRALinear` cannot wrap it. This wrapper keeps the base module
    frozen and adds ``A`` of shape ``(r, embed_dim)`` and ``B`` of shape
    ``(3 * embed_dim, r)``. The projection passed to
    ``F.multi_head_attention_forward`` is::

        in_proj_weight + (B @ A) * (alpha / r)

    ``B`` is initialized to zero, so applying the adapter starts from the
    exact base behavior. ``adapt_bias`` optionally adds a low-rank delta to
    ``in_proj_bias`` (``A`` of shape ``(r, 1)`` and ``B`` of shape
    ``(3 * embed_dim, r)``); it is off by default and is only needed for
    experiments that want to move the Q/K/V bias as well.

    Unlike ``nn.MultiheadAttention.forward`` this always uses the functional
    attention path: the merged weight carries a gradient, so the native
    inference fast path cannot be used.
    """

    def __init__(self, base: nn.MultiheadAttention, r: int = 8,
                 alpha: int = 16, adapt_bias: bool = False):
        super().__init__()
        if not isinstance(base, nn.MultiheadAttention):
            raise TypeError(
                "LoRAMultiheadAttention expects nn.MultiheadAttention, "
                f"got {type(base).__name__}")
        if not base._qkv_same_embed_dim:
            raise ValueError(
                "QKV LoRA requires a fused in_proj_weight "
                "(q, k and v must share embed_dim)")
        if r <= 0:
            raise ValueError("QKV LoRA rank must be > 0")
        if alpha <= 0:
            raise ValueError("QKV LoRA alpha must be > 0")
        self.base = base
        for param in self.base.parameters():
            param.requires_grad = False
        self.r = int(r)
        self.alpha = int(alpha)
        self.scaling = self.alpha / self.r
        self.embed_dim = base.embed_dim
        self.num_heads = base.num_heads
        self.batch_first = base.batch_first
        self.dropout = base.dropout
        self.adapt_bias = bool(adapt_bias)
        self.lora_A = nn.Parameter(torch.empty(self.r, self.embed_dim))
        self.lora_B = nn.Parameter(torch.zeros(3 * self.embed_dim, self.r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        if self.adapt_bias:
            self.lora_bias_A = nn.Parameter(torch.empty(self.r, 1))
            self.lora_bias_B = nn.Parameter(
                torch.zeros(3 * self.embed_dim, self.r))
            nn.init.kaiming_uniform_(self.lora_bias_A, a=math.sqrt(5))

    @property
    def in_proj_weight(self):
        """Base QKV weight, exposed for callers that inspect the module."""
        return self.base.in_proj_weight

    @property
    def in_proj_bias(self):
        return self.base.in_proj_bias

    @property
    def out_proj(self):
        return self.base.out_proj

    @property
    def bias_k(self):
        return self.base.bias_k

    @property
    def bias_v(self):
        return self.base.bias_v

    @property
    def add_zero_attn(self):
        return self.base.add_zero_attn

    def merged_in_proj_weight(self) -> torch.Tensor:
        delta = (self.lora_B @ self.lora_A) * self.scaling
        return self.base.in_proj_weight + delta.to(
            self.base.in_proj_weight.dtype)

    def merged_in_proj_bias(self):
        base_bias = self.base.in_proj_bias
        if not self.adapt_bias:
            return base_bias
        delta = (self.lora_bias_B @ self.lora_bias_A).squeeze(1) * self.scaling
        if base_bias is None:
            return delta
        return base_bias + delta.to(base_bias.dtype)

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights: bool = True, attn_mask=None,
                average_attn_weights: bool = True,
                is_causal: bool = False):
        is_batched = query.dim() == 3
        if self.batch_first and is_batched:
            # Match nn.MultiheadAttention: preserve tensor identity where
            # possible so self-attention stays a single tensor.
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = (x.transpose(1, 0) for x in (query, key))
                    value = key
            else:
                query, key, value = (
                    x.transpose(1, 0) for x in (query, key, value))
        attn_output, attn_output_weights = F.multi_head_attention_forward(
            query, key, value,
            self.embed_dim, self.num_heads,
            self.merged_in_proj_weight(), self.merged_in_proj_bias(),
            self.base.bias_k, self.base.bias_v,
            self.base.add_zero_attn, self.base.dropout,
            self.base.out_proj.weight, self.base.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal,
        )
        if self.batch_first and is_batched:
            attn_output = attn_output.transpose(1, 0)
        return attn_output, attn_output_weights


def _lora_parameter_names(model: nn.Module) -> set[str]:
    """Fully-qualified names of every LoRA A/B parameter in ``model``."""
    names = set()
    for module_name, module in model.named_modules():
        prefix = f"{module_name}." if module_name else ""
        if isinstance(module, LoRALinear):
            names.add(prefix + "A")
            names.add(prefix + "B")
        elif isinstance(module, LoRAMultiheadAttention):
            names.add(prefix + "lora_A")
            names.add(prefix + "lora_B")
            if module.adapt_bias:
                names.add(prefix + "lora_bias_A")
                names.add(prefix + "lora_bias_B")
    return names


def _qkv_parameter_names(model: nn.Module) -> set[str]:
    """Fully-qualified names of the QKV LoRA parameters in ``model``."""
    names = set()
    for module_name, module in model.named_modules():
        if not isinstance(module, LoRAMultiheadAttention):
            continue
        prefix = f"{module_name}." if module_name else ""
        names.add(prefix + "lora_A")
        names.add(prefix + "lora_B")
        if module.adapt_bias:
            names.add(prefix + "lora_bias_A")
            names.add(prefix + "lora_bias_B")
    return names


def _lora_tensors(state: dict) -> dict:
    return {key: value for key, value in state.items()
            if key.endswith((".A", ".B", "lora_A", "lora_B",
                             "lora_bias_A", "lora_bias_B"))}


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
    """Recover r/alpha per target group from the wrapped modules.

    The QKV keys are only emitted when a :class:`LoRAMultiheadAttention`
    exists, so adapters trained without QKV LoRA keep their old config shape.
    """
    found = {}
    qkv_bias = False
    for name, module in model.named_modules():
        if isinstance(module, LoRAMultiheadAttention):
            geometry = (module.r,
                        int(round(module.scaling * module.r)))
            previous = found.get("qkv")
            if previous is not None and previous != geometry:
                raise ValueError("QKV LoRA geometry differs between modules")
            found["qkv"] = geometry
            qkv_bias = qkv_bias or module.adapt_bias
            continue
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
    config = {"r_attn": r_attn, "alpha_attn": alpha_attn,
              "r_mlp": r_mlp, "alpha_mlp": alpha_mlp}
    if "qkv" in found:
        r_qkv, alpha_qkv = found["qkv"]
        config["r_qkv"] = r_qkv
        config["alpha_qkv"] = alpha_qkv
        config["qkv_bias"] = qkv_bias
    return config


def apply_qkv_lora(model: nn.Module, rank: int = 8,
                   alpha: int | None = None,
                   adapt_bias: bool = False) -> nn.Module:
    """Wrap fused-QKV attention modules with :class:`LoRAMultiheadAttention`.

    Every ``nn.MultiheadAttention`` in ``model`` is wrapped and its base
    parameters are frozen. ``rank=0`` is a no-op, which keeps the default
    adapter geometry untouched. The wrapper stores A/B for the fused
    ``in_proj_weight`` and, when ``adapt_bias`` is true, for
    ``in_proj_bias``.
    """
    if rank < 0:
        raise ValueError("QKV LoRA rank must be >= 0")
    if rank == 0:
        return model
    if alpha is None:
        alpha = rank
    if alpha <= 0:
        raise ValueError("QKV LoRA alpha must be > 0 when rank > 0")
    targets = [
        (name, module) for name, module in model.named_modules()
        if isinstance(module, nn.MultiheadAttention)
        and not isinstance(module, LoRAMultiheadAttention)
    ]
    if not targets:
        raise ValueError(
            "QKV LoRA requested but model has no nn.MultiheadAttention "
            "modules with fused Q/K/V weights")
    for name, module in targets:
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child, LoRAMultiheadAttention(
            module, rank, int(alpha), adapt_bias=adapt_bias))
    # A standalone call should leave only adapter parameters trainable, just
    # like apply_lora. Re-enable any LoRA modules that were already present.
    for param in model.parameters():
        param.requires_grad = False
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.A.requires_grad = True
            module.B.requires_grad = True
        elif isinstance(module, LoRAMultiheadAttention):
            module.lora_A.requires_grad = True
            module.lora_B.requires_grad = True
            if module.adapt_bias:
                module.lora_bias_A.requires_grad = True
                module.lora_bias_B.requires_grad = True
    return model


def apply_lora(model: nn.Module, r_attn: int = 8, r_mlp: int = 0,
               alpha_attn: int | None = None,
               alpha_mlp: int | None = None,
               r_qkv: int = 0, alpha_qkv: int | None = None,
               qkv_bias: bool = False) -> nn.Module:
    """Replace target Linear/MHA modules with LoRA wrappers and freeze base.

    With ``r_attn > 0`` the attention ``out_proj`` Linear is wrapped. With
    ``r_qkv > 0`` each fused ``nn.MultiheadAttention`` is replaced by
    :class:`LoRAMultiheadAttention`, which adapts ``in_proj_weight`` (and
    optionally ``in_proj_bias`` when ``qkv_bias`` is true). MLP linears are
    wrapped when ``r_mlp > 0``. The defaults keep QKV frozen and preserve the
    previous behavior exactly.
    """
    if r_attn < 0 or r_mlp < 0 or r_qkv < 0:
        raise ValueError("LoRA ranks must be >= 0")
    for label, r, alpha in (("attn", r_attn, alpha_attn),
                            ("mlp", r_mlp, alpha_mlp),
                            ("qkv", r_qkv, alpha_qkv)):
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
    if r_qkv > 0:
        apply_qkv_lora(model, r_qkv, alpha_qkv, adapt_bias=qkv_bias)
    for param in model.parameters():
        param.requires_grad = False
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.A.requires_grad = True
            module.B.requires_grad = True
        elif isinstance(module, LoRAMultiheadAttention):
            module.lora_A.requires_grad = True
            module.lora_B.requires_grad = True
            if module.adapt_bias:
                module.lora_bias_A.requires_grad = True
                module.lora_bias_B.requires_grad = True
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
            or k.endswith("lora_A") or k.endswith("lora_B")
            or k.endswith("lora_bias_A") or k.endswith("lora_bias_B")}


def qkv_lora_state_dict(model: nn.Module) -> dict:
    """Return only the QKV LoRA tensors, detached on CPU."""
    names = _qkv_parameter_names(model)
    return {k: v.detach().cpu() for k, v in model.state_dict().items()
            if k in names}


def set_lora_state_dict(model: nn.Module, state: dict):
    """Load A/B tensors from a full or adapter-only state dict.

    Raises instead of silently dropping tensors that the model cannot use.
    """
    _load_lora_tensors(model, state)


def set_qkv_lora_state_dict(model: nn.Module, state: dict):
    """Load QKV LoRA tensors from a QKV-only or full LoRA state dict.

    The state may contain tensors for other LoRA modules; only QKV parameters
    are loaded. Missing QKV tensors raise ``ValueError``.
    """
    expected = _qkv_parameter_names(model)
    if not expected:
        raise ValueError("model has no QKV LoRA parameters")
    tensors = _lora_tensors(state)
    missing = sorted(expected - tensors.keys())
    if missing:
        raise ValueError(f"QKV adapter is missing tensors: {missing}")
    model.load_state_dict({key: tensors[key] for key in expected},
                          strict=False)


def merge_lora(model: nn.Module):
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, LoRALinear):
                delta = (module.B @ module.A) * module.scaling
                module.base.weight.add_(delta.to(module.base.weight.dtype))
                module.A.zero_()
                module.B.zero_()
            elif isinstance(module, LoRAMultiheadAttention):
                delta = (module.lora_B @ module.lora_A) * module.scaling
                module.base.in_proj_weight.add_(
                    delta.to(module.base.in_proj_weight.dtype))
                module.lora_A.zero_()
                module.lora_B.zero_()
                if module.adapt_bias:
                    delta = (module.lora_bias_B @ module.lora_bias_A
                             ).squeeze(1) * module.scaling
                    if module.base.in_proj_bias is not None:
                        module.base.in_proj_bias.add_(
                            delta.to(module.base.in_proj_bias.dtype))
                    module.lora_bias_A.zero_()
                    module.lora_bias_B.zero_()


def save_adapter(model: nn.Module, out: str | Path, **config) -> Path:
    """Save A/B tensors plus a config that can rebuild the same modules.

    ``r_attn``/``alpha_attn``/``r_mlp``/``alpha_mlp`` are inferred from the
    model when the caller does not pass them, so an attention adapter can
    always be reloaded with :func:`apply_adapter`. ``r_qkv``/``alpha_qkv``
    (and ``qkv_bias`` when used) are inferred from any
    :class:`LoRAMultiheadAttention` modules.
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
               alpha_mlp=config.get("alpha_mlp", config.get("alpha")),
               r_qkv=int(config.get("r_qkv", 0) or 0),
               alpha_qkv=config.get("alpha_qkv"),
               qkv_bias=bool(config.get("qkv_bias", False)))
    _load_lora_tensors(model, ck.get("lora", {}))
    model.eval()
    return config


# --- Adapter soup -----------------------------------------------------------
#
# A soup combines the A/B tensors of several adapters that were trained from
# the same base with the same LoRA geometry. Two merge modes are supported:
#
# ``ab``
#     ``A = sum_i w_i A_i`` and ``B = sum_i w_i B_i``. This is cheap and keeps
#     the original rank, but the averaged product is not the average of the
#     individual deltas: cross terms can cancel or amplify skills.
#
# ``delta-svd``
#     Form ``D = sum_i w_i B_i A_i`` and factor ``D`` back into a (possibly
#     larger) low-rank pair with SVD. This preserves the weighted sum of the
#     deltas up to the requested rank cap.
#
# Both modes only merge adapters; they never touch a base checkpoint or the
# active registry.

GEOMETRY_KEYS = ("r_attn", "alpha_attn", "r_mlp", "alpha_mlp")
MERGE_MODES = ("ab", "delta-svd")


class AdapterMergeError(ValueError):
    """Raised when adapters cannot be merged without guessing."""


def _adapter_config_value(config: dict, *keys, default=0):
    for key in keys:
        if key in config and config[key] is not None:
            return config[key]
    return default


def _adapter_int(value, label: str) -> int:
    if isinstance(value, bool):
        raise AdapterMergeError(f"{label} must be an integer, got {value!r}")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise AdapterMergeError(
            f"{label} must be an integer, got {value!r}") from exc
    if number != value:
        raise AdapterMergeError(f"{label} must be an integer, got {value!r}")
    return number


def _normalize_geometry(config: dict, path: str | Path) -> dict:
    """Return the four canonical geometry values recorded by an adapter."""
    # Match apply_adapter's fallbacks: attention alpha only comes from
    # alpha_attn, while MLP alpha may use the legacy top-level ``alpha``.
    r_attn = _adapter_int(_adapter_config_value(config, "r_attn"), "r_attn")
    r_mlp = _adapter_int(
        _adapter_config_value(config, "r_mlp", "rank", default=8), "r_mlp")
    alpha_attn = _adapter_int(
        _adapter_config_value(config, "alpha_attn", default=r_attn),
        "alpha_attn")
    alpha_mlp = _adapter_int(
        _adapter_config_value(config, "alpha_mlp", "alpha", default=r_mlp),
        "alpha_mlp")
    for label, rank, alpha in (("attn", r_attn, alpha_attn),
                               ("mlp", r_mlp, alpha_mlp)):
        if rank < 0:
            raise AdapterMergeError(
                f"adapter config is invalid in {path}: r_{label}={rank} < 0")
        if alpha < 0:
            raise AdapterMergeError(
                f"adapter config is invalid in {path}: "
                f"alpha_{label}={alpha} < 0")
        if rank > 0 and alpha <= 0:
            raise AdapterMergeError(
                f"adapter config is invalid in {path}: "
                f"alpha_{label} must be > 0 when r_{label}={rank}")
    # An alpha without a rank does not affect the forward pass. Canonicalize it
    # so two otherwise identical adapters compare equal.
    return {
        "r_attn": r_attn,
        "alpha_attn": alpha_attn if r_attn else 0,
        "r_mlp": r_mlp,
        "alpha_mlp": alpha_mlp if r_mlp else 0,
    }


def _tensor_group(prefix: str) -> str | None:
    if prefix.endswith(ATTN_SUFFIXES):
        return "attn"
    if prefix.endswith(MLP_SUFFIXES):
        return "mlp"
    return None


def _pair_tensors(lora: dict, path: str | Path) -> dict[str, dict]:
    """Group A/B tensor pairs by module prefix and validate their shapes."""
    if not lora:
        raise AdapterMergeError(f"adapter has no LoRA A/B tensors: {path}")
    unsupported = sorted(
        key for key in lora
        if not (key.endswith(".A") or key.endswith(".B")))
    if unsupported:
        raise AdapterMergeError(
            f"unsupported LoRA tensor names in {path}: {unsupported}; "
            "expected names ending in '.A' or '.B'")
    a_names = {key[:-2] for key in lora if key.endswith(".A")}
    b_names = {key[:-2] for key in lora if key.endswith(".B")}
    if a_names != b_names:
        missing = sorted(a_names - b_names)
        unexpected = sorted(b_names - a_names)
        parts = []
        if missing:
            parts.append(f"missing B for {missing}")
        if unexpected:
            parts.append(f"missing A for {unexpected}")
        raise AdapterMergeError(
            "adapter has incomplete A/B pairs in "
            f"{path}: " + "; ".join(parts))
    pairs = {}
    for prefix in sorted(a_names):
        a = lora[prefix + ".A"]
        b = lora[prefix + ".B"]
        if not torch.is_tensor(a) or not torch.is_tensor(b):
            raise AdapterMergeError(
                f"adapter tensor {prefix} is not a torch tensor: {path}")
        if a.dim() != 2 or b.dim() != 2:
            raise AdapterMergeError(
                f"adapter tensor {prefix} must be 2-D: "
                f"A={tuple(a.shape)} B={tuple(b.shape)} ({path})")
        if a.shape[0] != b.shape[1]:
            raise AdapterMergeError(
                f"adapter tensor {prefix} has mismatched ranks: "
                f"A={tuple(a.shape)} B={tuple(b.shape)} ({path})")
        if not torch.is_floating_point(a) or not torch.is_floating_point(b):
            raise AdapterMergeError(
                f"adapter tensor {prefix} must be floating point: {path}")
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise AdapterMergeError(
                f"adapter tensor {prefix} contains non-finite values: {path}")
        group = _tensor_group(prefix)
        if group is None:
            raise AdapterMergeError(
                f"cannot classify adapter tensor {prefix!r} in {path}; "
                "expected an attention or MLP module name")
        pairs[prefix] = {"A": a, "B": b, "group": group, "rank": a.shape[0]}
    return pairs


def _tensor_ranks(pairs: dict[str, dict], path: str | Path) -> dict:
    ranks = {}
    for prefix, pair in pairs.items():
        group = pair["group"]
        rank = pair["rank"]
        previous = ranks.get(group)
        if previous is not None and previous != rank:
            raise AdapterMergeError(
                f"adapter {path} mixes {group} ranks: "
                f"{previous} and {rank} ({prefix})")
        ranks[group] = rank
    return ranks


def load_adapter(path: str | Path) -> dict:
    """Read an adapter directory without applying it to a model.

    Returns ``{"path", "lora", "config", "geometry", "pairs"}``. The geometry
    is canonicalized to ``r_attn``/``alpha_attn``/``r_mlp``/``alpha_mlp`` and
    is checked against the tensor ranks so a merge cannot silently change the
    effective scaling.
    """
    adapter_dir = Path(path)
    checkpoint_path = adapter_dir / "adapter.pt"
    if not checkpoint_path.is_file():
        raise AdapterMergeError(f"adapter.pt missing: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu",
                            weights_only=False)
    if not isinstance(checkpoint, dict):
        raise AdapterMergeError(
            f"adapter checkpoint is not a dict: {checkpoint_path}")
    config = checkpoint.get("config", {}) or {}
    if not isinstance(config, dict):
        raise AdapterMergeError(
            f"adapter config is not a dict: {checkpoint_path}")
    config_path = adapter_dir / "config.json"
    if config_path.is_file():
        try:
            file_config = json.loads(
                config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AdapterMergeError(
                f"cannot read adapter config {config_path}: {exc}") from exc
        if not isinstance(file_config, dict):
            raise AdapterMergeError(
                f"adapter config is not a dict: {config_path}")
        if file_config:
            config = {**config, **file_config}
    lora = checkpoint.get("lora", {}) or {}
    if not isinstance(lora, dict):
        raise AdapterMergeError(
            f"adapter lora state is not a dict: {checkpoint_path}")
    geometry = _normalize_geometry(config, adapter_dir)
    pairs = _pair_tensors(lora, adapter_dir)
    ranks = _tensor_ranks(pairs, adapter_dir)
    for group, key in (("attn", "r_attn"), ("mlp", "r_mlp")):
        expected = geometry[key]
        actual = ranks.get(group, 0)
        if expected != actual:
            raise AdapterMergeError(
                f"adapter config and tensors disagree for {key} in "
                f"{adapter_dir}: config={expected}, tensors={actual}")
    return {
        "path": adapter_dir,
        "lora": lora,
        "config": config,
        "geometry": geometry,
        "pairs": pairs,
    }


def normalize_adapter_weights(weights, count: int) -> list[float]:
    """Return non-negative, finite weights that sum to one.

    ``weights=None`` gives equal weights. Zero weights skip an adapter; the
    remaining weights are renormalized.
    """
    if count <= 0:
        raise AdapterMergeError("at least one adapter is required")
    if weights is None:
        return [1.0 / count] * count
    if isinstance(weights, (str, bytes)):
        raise AdapterMergeError("weights must be a sequence of numbers")
    try:
        values = list(weights)
    except TypeError as exc:
        raise AdapterMergeError(
            "weights must be a sequence of numbers") from exc
    if len(values) != count:
        raise AdapterMergeError(
            f"weights must have one value per adapter "
            f"({count} adapters, got {len(values)})")
    cleaned = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise AdapterMergeError(
                f"weights must be numbers, got {value!r}")
        number = float(value)
        if not math.isfinite(number) or number < 0:
            raise AdapterMergeError(
                "weights must be finite and >= 0, got "
                f"{number!r}")
        cleaned.append(number)
    total = sum(cleaned)
    if total <= 0:
        raise AdapterMergeError("weights must sum to > 0")
    return [value / total for value in cleaned]


def _check_merge_compatibility(loaded: list[dict]) -> None:
    reference = loaded[0]
    ref_geometry = reference["geometry"]
    ref_pairs = reference["pairs"]
    for item in loaded[1:]:
        differences = [
            f"{key}={item['geometry'][key]} (expected {ref_geometry[key]})"
            for key in GEOMETRY_KEYS
            if item["geometry"][key] != ref_geometry[key]
        ]
        if differences:
            raise AdapterMergeError(
                "adapter geometry mismatch: "
                + ", ".join(differences)
                + f"; reference is {reference['path']}, "
                f"mismatch is {item['path']}. Adapters must share "
                "r_attn/alpha_attn/r_mlp/alpha_mlp.")
        missing = sorted(ref_pairs.keys() - item["pairs"].keys())
        unexpected = sorted(item["pairs"].keys() - ref_pairs.keys())
        if missing or unexpected:
            parts = []
            if missing:
                parts.append(f"missing={missing}")
            if unexpected:
                parts.append(f"unexpected={unexpected}")
            raise AdapterMergeError(
                "adapter tensor names do not match: "
                + ", ".join(parts)
                + f"; reference is {reference['path']}, "
                f"mismatch is {item['path']}")
        for prefix in ref_pairs:
            expected_shape = {
                "A": tuple(ref_pairs[prefix]["A"].shape),
                "B": tuple(ref_pairs[prefix]["B"].shape),
            }
            actual_shape = {
                "A": tuple(item["pairs"][prefix]["A"].shape),
                "B": tuple(item["pairs"][prefix]["B"].shape),
            }
            if actual_shape != expected_shape:
                raise AdapterMergeError(
                    f"adapter tensor shape mismatch for {prefix!r}: "
                    f"{actual_shape} != {expected_shape}; "
                    f"reference is {reference['path']}, "
                    f"mismatch is {item['path']}")


def _merge_ab(loaded: list[dict], weights: list[float]) -> dict:
    merged = {}
    for key in sorted(loaded[0]["lora"]):
        reference = loaded[0]["lora"][key]
        accumulator = torch.zeros(reference.shape, dtype=torch.float64)
        for item, weight in zip(loaded, weights):
            tensor = item["lora"][key]
            accumulator.add_(tensor.to(torch.float64), alpha=weight)
        merged[key] = accumulator.to(reference.dtype).clone()
    return merged


def _merge_delta_svd(loaded: list[dict], weights: list[float],
                     rank: int | None) -> tuple[dict, dict]:
    geometry = dict(loaded[0]["geometry"])
    groups = {}
    for group, geom_key in (("attn", "r_attn"), ("mlp", "r_mlp")):
        old_rank = geometry[geom_key]
        if old_rank == 0:
            continue
        max_rank = min(
            min(pair["A"].shape[1], pair["B"].shape[0])
            for pair in loaded[0]["pairs"].values()
            if pair["group"] == group)
        if rank is None:
            new_rank = min(old_rank, max_rank)
        else:
            new_rank = min(int(rank), max_rank)
        if new_rank < 1:
            raise AdapterMergeError(
                f"delta-svd rank for {group} must be >= 1, got {new_rank}")
        old_alpha = geometry["alpha_" + group]
        new_alpha = max(1, int(round(old_alpha * new_rank / old_rank)))
        factor = 1.0
        if old_alpha > 0:
            factor = ((old_alpha / old_rank)
                      / (new_alpha / new_rank))
        groups[group] = {
            "new_rank": new_rank,
            "factor": factor,
        }
        geometry["r_" + group] = new_rank
        geometry["alpha_" + group] = new_alpha

    merged = {}
    for prefix, pair in loaded[0]["pairs"].items():
        group = pair["group"]
        delta = torch.zeros(
            (pair["B"].shape[0], pair["A"].shape[1]), dtype=torch.float64)
        for item, weight in zip(loaded, weights):
            pair_i = item["pairs"][prefix]
            delta.add_(
                pair_i["B"].to(torch.float64)
                @ pair_i["A"].to(torch.float64), alpha=weight)
        u, s, vh = torch.linalg.svd(delta, full_matrices=False)
        new_rank = groups[group]["new_rank"]
        k = min(new_rank, s.numel())
        if k != new_rank:
            raise AdapterMergeError(
                f"cannot factor {prefix!r} to rank {new_rank}; "
                f"SVD has only {s.numel()} values")
        root = s[:k].sqrt()
        b = (u[:, :k] * root.unsqueeze(0)) * groups[group]["factor"]
        a = root.unsqueeze(1) * vh[:k, :]
        reference = pair["A"]
        merged[prefix + ".A"] = a.to(reference.dtype).clone()
        merged[prefix + ".B"] = b.to(pair["B"].dtype).clone()
    return merged, geometry


def merge_adapter_state_dicts(adapters: list[str | Path], weights=None,
                              mode: str = "ab",
                              rank: int | None = None) -> dict:
    """Merge saved adapters into a new ``{"lora", "config"}`` state.

    ``mode="ab"`` averages A and B separately. ``mode="delta-svd"`` averages
    the low-rank deltas and re-factors them with SVD; ``rank`` caps the new
    rank and defaults to the original rank. Weights are normalized to sum to
    one. Raises :class:`AdapterMergeError` on geometry, name or shape
    mismatches.
    """
    if mode not in MERGE_MODES:
        raise AdapterMergeError(
            f"unknown merge mode {mode!r}; expected one of {MERGE_MODES}")
    if not adapters:
        raise AdapterMergeError("at least one adapter is required")
    if rank is not None:
        if isinstance(rank, bool):
            raise AdapterMergeError(f"rank must be an integer, got {rank!r}")
        try:
            rank = int(rank)
        except (TypeError, ValueError) as exc:
            raise AdapterMergeError(
                f"rank must be an integer, got {rank!r}") from exc
        if rank < 1:
            raise AdapterMergeError(f"rank must be >= 1, got {rank}")
    loaded = [load_adapter(path) for path in adapters]
    _check_merge_compatibility(loaded)
    normalized = normalize_adapter_weights(weights, len(loaded))
    if mode == "ab":
        merged = _merge_ab(loaded, normalized)
        geometry = dict(loaded[0]["geometry"])
    else:
        merged, geometry = _merge_delta_svd(
            loaded, normalized, rank)
    config = {
        **geometry,
        "rank": geometry["r_mlp"],
        "alpha": geometry["alpha_mlp"],
        "method": "adapter_soup_v1",
        "merge": {
            "mode": mode,
            "weights": normalized,
            "adapters": [str(item["path"]) for item in loaded],
            "rank": rank,
        },
    }
    return {
        "lora": merged,
        "config": config,
        "geometry": geometry,
        "weights": normalized,
        "tensors": len(merged),
        "parameters": sum(int(tensor.numel()) for tensor in merged.values()),
    }


def _write_adapter_dir(out: str | Path, lora: dict, config: dict) -> Path:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"lora": lora, "config": config}, out / "adapter.pt")
    (out / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return out


def merge_adapters(adapters: list[str | Path], out: str | Path | None = None,
                   weights=None, mode: str = "ab",
                   rank: int | None = None) -> dict:
    """Merge saved adapters and optionally write a loadable adapter directory.

    The returned summary is JSON-serializable and contains no training rows.
    Use :func:`merge_adapter_state_dicts` when the tensors are needed in
    memory.
    """
    state = merge_adapter_state_dicts(
        adapters, weights=weights, mode=mode, rank=rank)
    if out is not None:
        _write_adapter_dir(out, state["lora"], state["config"])
    return {
        "out": str(out) if out is not None else None,
        "mode": mode,
        "rank": state["config"]["merge"]["rank"],
        "adapters": state["config"]["merge"]["adapters"],
        "weights": state["weights"],
        "config": state["config"],
        "tensors": state["tensors"],
        "parameters": state["parameters"],
    }
