"""Checks on the model configuration that do not depend on the variant (data layout, time
base, flow and optimiser settings), and the `REQUIRED` marker for training parameters
that have no default. Architecture-specific checks live in each variant's `config.py`.
"""

from __future__ import annotations

import math


def _positive(name: str, value: float) -> None:
    if not math.isfinite(float(value)) or float(value) <= 0:
        raise ValueError(f"{name} must be finite and > 0, got {value!r}")


def _positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be an integer > 0, got {value!r}")


def validate_common_config(cfg) -> None:
    """Validate the fields every model variant shares. Called from the config's `validate()`."""

    v, m, e, a = cfg.vlm, cfg.memory, cfg.expert, cfg.action
    f, tr, d, o = cfg.flow, cfg.train, cfg.data, cfg.optim

    for name, value in (
        ("vlm.num_layers", v.num_layers),
        ("vlm.hidden_size", v.hidden_size),
        ("vlm.image_size", v.image_size),
        ("vlm.tokens_per_frame", v.tokens_per_frame),
        ("vlm.num_cameras", v.num_cameras),
        ("vlm.max_language_tokens", v.max_language_tokens),
        ("memory.num_slots", m.num_slots),
        ("memory.dim", m.dim),
        ("memory.num_blocks", m.num_blocks),
        ("memory.num_heads", m.num_heads),
        ("memory.bank_capacity", m.bank_capacity),
        ("memory.ffn_intermediate", m.ffn_intermediate),
        ("expert.num_layers", e.num_layers),
        ("expert.hidden_size", e.hidden_size),
        ("expert.num_heads", e.num_heads),
        ("expert.num_kv_heads", e.num_kv_heads),
        ("expert.head_dim", e.head_dim),
        ("expert.self_attn_every_n_layers", e.self_attn_every_n_layers),
        ("action.chunk_size", a.chunk_size),
        ("action.dim", a.dim),
        ("action.executed_steps", a.executed_steps),
        ("action.state_dim", a.state_dim),
        ("train.max_seq_len", tr.max_seq_len),
        ("data.stride", d.stride),
        ("flow.num_inference_steps", f.num_inference_steps),
        ("flow.draws_per_condition", f.draws_per_condition),
        ("optim.grad_accum_steps", o.grad_accum_steps),
    ):
        _positive_int(name, value)

    for name, value in (
        ("action.control_hz", a.control_hz),
        ("action.action_hz", a.action_hz),
        ("action.source_fps", a.source_fps),
        ("optim.lr", o.lr),
        ("optim.eps", o.eps),
        ("optim.grad_clip", o.grad_clip),
    ):
        _positive(name, value)

    if v.train_top_k_layers < 0 or v.train_top_k_layers > v.num_layers:
        raise ValueError(
            "vlm.train_top_k_layers must be between 0 and vlm.num_layers, got "
            f"{v.train_top_k_layers} and {v.num_layers}."
        )
    if m.dim != v.hidden_size:
        raise ValueError(
            f"memory.dim ({m.dim}) must equal vlm.hidden_size ({v.hidden_size}): "
            "memory tokens are concatenated into the VLM prefix."
        )
    if m.dim % m.num_heads:
        raise ValueError("memory.dim must be divisible by memory.num_heads.")
    if m.age_fusion not in ("add", "concat_mlp"):
        raise ValueError(f"unknown memory.age_fusion: {m.age_fusion}")
    if e.num_heads % e.num_kv_heads:
        raise ValueError("expert.num_heads must be a multiple of expert.num_kv_heads (GQA).")
    if e.head_dim % 2:
        raise ValueError("expert.head_dim must be even for rotary embeddings.")

    if a.executed_steps > a.chunk_size:
        raise ValueError("action.executed_steps cannot exceed action.chunk_size.")
    if (a.dim, a.state_dim) != (192, 212):
        raise ValueError(
            "the current VITRA padding layout is fixed at action.dim=192 and "
            f"action.state_dim=212, got {(a.dim, a.state_dim)}."
        )
    if v.num_cameras != 1:
        raise ValueError(
            f"the current EPIC loader supplies exactly one camera, got {v.num_cameras}."
        )
    if a.action_hz + 1e-9 < a.source_fps:
        raise ValueError(
            "the vendored VITRA interpolator only supports upsampling: "
            f"action_hz={a.action_hz} must be >= source_fps={a.source_fps}."
        )

    stride = a.source_fps / a.control_hz
    rounded_stride = round(stride)
    if not math.isclose(stride, rounded_stride, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            "source_fps / control_hz must be an integer because phase anchors are source "
            f"frame indices, got {a.source_fps} / {a.control_hz} = {stride}."
        )
    if d.stride != rounded_stride:
        raise ValueError(
            f"data.stride={d.stride} disagrees with source_fps/control_hz={rounded_stride}."
        )
    if d.interpolation != "pchip_bone_preserving":
        raise ValueError(
            "the loader currently implements only interpolation='pchip_bone_preserving', "
            f"got {d.interpolation!r}."
        )
    for name, value in (
        ("data.geometric_augmentation", d.geometric_augmentation),
        ("data.random_paraphrase", d.random_paraphrase),
        ("optim.bf16", o.bf16),
    ):
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be a bool, got {value!r}.")

    # the executed part of a chunk fills exactly one control period, so chunks abut
    execution_period = a.executed_steps / a.action_hz
    control_period = 1.0 / a.control_hz
    if not math.isclose(execution_period, control_period, rel_tol=1e-7, abs_tol=1e-9):
        raise ValueError(
            "executed_steps/action_hz must equal one control period; otherwise chunks leave "
            f"a gap or overlap. Got {execution_period} s vs {control_period} s."
        )

    # a sequence of length L has L-1 earlier writes in the bank at its supervised step,
    # so L may go up to bank_capacity + 1
    if tr.max_seq_len > m.bank_capacity + 1:
        raise ValueError(
            f"train.max_seq_len={tr.max_seq_len} exceeds bank_capacity+1="
            f"{m.bank_capacity + 1}; older burn-in steps would already have been evicted."
        )
    if not math.isfinite(float(tr.length_alpha)):
        raise ValueError("train.length_alpha must be finite.")
    for name, rate in (("train.state_dropout", tr.state_dropout),
                       ("train.memory_dropout", tr.memory_dropout)):
        if not 0.0 <= float(rate) <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {rate}.")

    groups = tuple(f.component_groups)
    weights = tuple(f.component_weights)
    if len(groups) != len(weights) or not groups:
        raise ValueError("flow.component_weights must have one entry per non-empty group list.")
    names: set[str] = set()
    occupied: set[int] = set()
    for group in groups:
        if len(group) != 3:
            raise ValueError(f"component group must be (name, lo, hi), got {group!r}")
        name, lo, hi = group
        if not isinstance(name, str) or not name:
            raise ValueError(f"component group name must be a non-empty string, got {name!r}")
        if name in names:
            raise ValueError(f"duplicate component group name: {name!r}")
        names.add(name)
        if (isinstance(lo, bool) or not isinstance(lo, int)
                or isinstance(hi, bool) or not isinstance(hi, int)):
            raise ValueError(f"component group {name!r} bounds must be integers")
        if not 0 <= lo < hi <= a.dim:
            raise ValueError(f"component group {name!r} has invalid bounds [{lo}, {hi}).")
        indices = set(range(lo, hi))
        if occupied & indices:
            raise ValueError(f"component group {name!r} overlaps another group.")
        occupied |= indices
    expected_names = {
        "left_translation", "left_rotation", "left_articulation",
        "right_translation", "right_rotation", "right_articulation",
    }
    if names != expected_names or occupied != set(range(138)):
        raise ValueError(
            "flow.component_groups must name the six hand components and cover exactly "
            "the semantic action slots [0, 138); padding slots [138, 192) stay masked."
        )
    if any(isinstance(w, bool) or not isinstance(w, (int, float))
           or not math.isfinite(float(w)) or float(w) < 0 for w in weights):
        raise ValueError("flow.component_weights must be finite and non-negative.")
    if not any(float(w) > 0 for w in weights):
        raise ValueError("at least one flow.component_weight must be positive.")

    if f.horizon_weights is not None:
        horizon = tuple(f.horizon_weights)
        if len(horizon) != a.chunk_size:
            raise ValueError(
                f"flow.horizon_weights has {len(horizon)} entries; expected {a.chunk_size}."
            )
        if any(isinstance(w, bool) or not isinstance(w, (int, float))
               or not math.isfinite(float(w)) or float(w) < 0 for w in horizon):
            raise ValueError("flow.horizon_weights must be finite and non-negative.")
        if not any(float(w) > 0 for w in horizon):
            raise ValueError("at least one flow.horizon_weight must be positive.")

    _positive("flow.time_beta_alpha", f.time_beta_alpha)
    _positive("flow.time_beta_beta", f.time_beta_beta)
    _positive("flow.time_scale", f.time_scale)
    if (not math.isfinite(float(f.time_offset)) or f.time_offset < 0
            or f.time_offset + f.time_scale > 1.0 + 1e-9):
        raise ValueError(
            "flow time range must stay in [0, 1], got "
            f"offset={f.time_offset}, scale={f.time_scale}."
        )
    if (isinstance(o.warmup_steps, bool) or not isinstance(o.warmup_steps, int)
            or isinstance(o.decay_steps, bool) or not isinstance(o.decay_steps, int)
            or o.warmup_steps < 0 or o.decay_steps < 0):
        raise ValueError("optim warmup/decay steps must be non-negative.")
    if o.decay_steps and o.decay_steps < o.warmup_steps:
        raise ValueError("optim.decay_steps must be zero or >= optim.warmup_steps.")
    _positive("optim.final_lr", o.final_lr)
    if o.final_lr > o.lr:
        raise ValueError("optim.final_lr cannot exceed optim.lr.")
    if not math.isfinite(float(o.weight_decay)) or o.weight_decay < 0:
        raise ValueError("optim.weight_decay must be finite and non-negative.")
    if (isinstance(o.no_decay_ndim_max, bool)
            or not isinstance(o.no_decay_ndim_max, int) or o.no_decay_ndim_max < 0):
        raise ValueError("optim.no_decay_ndim_max must be an integer >= 0.")
    if not all(isinstance(part, str) and part for part in o.no_decay_name_parts):
        raise ValueError("optim.no_decay_name_parts must contain non-empty strings.")
    if len(o.betas) != 2 or not all(0.0 <= float(x) < 1.0 for x in o.betas):
        raise ValueError(f"optim.betas must contain two values in [0, 1), got {o.betas!r}.")


# ------------------------------------------------------- training parameters have no defaults

class _Required:
    """The value of a training parameter the run config has not set yet."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "REQUIRED"

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    def __reduce__(self):
        return "REQUIRED"


REQUIRED = _Required()

_CASTS = {"float": float, "int": int, "bool": bool, "str": str}


def set_parameter(obj, name: str, value) -> None:
    """Write a run-config value into a field that has no default, cast to its declared type."""
    declared = type(obj).__dataclass_fields__[name].type
    declared = declared if isinstance(declared, str) else getattr(declared, "__name__", "")
    cast = _CASTS.get(declared.strip())
    if cast is None or value is None or isinstance(value, (list, dict)):
        setattr(obj, name, value)
    else:
        setattr(obj, name, cast(value))


def unset_parameters(obj, prefix: str = "") -> list:
    """Dotted names of the dataclass fields, at any depth, that still hold `REQUIRED`."""
    import dataclasses
    out = []
    for f in dataclasses.fields(obj):
        value = getattr(obj, f.name)
        if value is REQUIRED:
            out.append(prefix + f.name)
        elif dataclasses.is_dataclass(value):
            out += unset_parameters(value, prefix + f.name + ".")
    return out


def _as_tuple(value):
    return tuple(_as_tuple(x) for x in value) if isinstance(value, (list, tuple)) else value


def require_parameters(obj, prefix: str = "") -> None:
    """Raise if a parameter without a default was left unset; JSON lists become tuples
    where the field is declared as a tuple."""
    import dataclasses
    missing = unset_parameters(obj, prefix)
    if missing:
        raise ValueError("these parameters have no default and must be set in the run "
                         "config: " + ", ".join(missing))
    for f in dataclasses.fields(obj):
        value = getattr(obj, f.name)
        declared = f.type if isinstance(f.type, str) else ""
        if dataclasses.is_dataclass(value):
            require_parameters(value, prefix + f.name + ".")
        elif declared.startswith("tuple") and isinstance(value, list):
            setattr(obj, f.name, _as_tuple(value))
