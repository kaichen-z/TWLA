"""Settings of the SmolVLA baseline.

Model values follow the config of the released `lerobot/smolvla_base` and, where it is
silent, the defaults of LeRobot's `SmolVLAConfig`. Three differ because of the action
space: `max_action_dim` 32 -> 192, `max_state_dim` 32 -> 212, `chunk_size` 50 -> 16.
The sections `vlm`, `image`, `action`, `data` and `train` use the field names the
dataloader reads.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any


# --------------------------------------------------------------------------- VLM

@dataclass
class VLMConfig:
    """SmolVLM2-500M and the action expert built beside it.

    The expert is built from the VLM's text config at `hidden_size * expert_width_multiplier`
    (960 -> 720) and is never pretrained.
    """

    # The VLM. The main model starts from the same checkpoint.
    model_id: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    # The released SmolVLA policy. `model.load_pretrained` can take the VLM from its weights
    # (`init.pretrain_path`).
    pretrained_policy: str = "lerobot/smolvla_base"

    num_vlm_layers: int = 16          # layers of the text model that are kept
    num_expert_layers: int = 0        # <= 0: as many layers as the VLM
    expert_width_multiplier: float = 0.75
    self_attn_every_n_layers: int = 2
    attention_mode: str = "cross_attn"

    freeze_vision_encoder: bool = True
    train_expert_only: bool = True    # freezes the whole VLM, vision tower included
    train_state_proj: bool = True
    add_image_special_tokens: bool = False
    prefix_length: int = 0            # 0: the prefix is not padded to a fixed length

    use_cache: bool = True            # prefix keys and values are computed once per sampling call

    # Read by the dataloader: the long side is resized to 512 and the image is padded on
    # the left and top, the same geometry as LeRobot's `resize_with_pad`.
    image_size: int = 512

    max_language_tokens: int = 48     # `tokenizer_max_length` in LeRobot

    gradient_checkpointing: bool = False


# --------------------------------------------------------------------------- image

@dataclass
class ImageConfig:
    """Image preprocessing. Only `pad` is supported: the dataloader's resize and pad."""

    mode: str = "pad"


# --------------------------------------------------------------------------- action

@dataclass
class ActionConfig:
    """The action and state vectors."""

    chunk_size: int = 16          # 16 steps at 80 Hz = 0.2 s, as in the main model
    dim: int = 192                # `max_action_dim`; 138 slots carry data (2 hands x 69)
    state_dim: int = 212          # `max_state_dim`

    control_hz: float = 10.0
    action_hz: float = 80.0
    source_fps: float = 30.0

    action_type: str = "keypoints"    # fingers as 21 keypoints per hand, the main model's layout

    dual_layout: bool = False     # also load the joint-angle layout; not used by this baseline


# --------------------------------------------------------------------------- flow

@dataclass
class FlowConfig:
    """Flow matching: `x_t = t * noise + (1 - t) * a`, target `u_t = noise - a`, MSE loss."""

    num_steps: int = 10           # Euler steps at sampling time

    # Training timestep: Beta(time_alpha, time_beta) * time_scale + time_offset.
    time_alpha: float = 1.5
    time_beta: float = 1.0
    time_scale: float = 0.999
    time_offset: float = 0.001

    # Period range of the sine-cosine timestep embedding.
    min_period: float = 4e-3
    max_period: float = 4.0


# --------------------------------------------------------------------------- loss

@dataclass
class LossConfig:
    """Groups `(name, lo, hi, weight)` of the masked loss over the action slots `lo:hi`.

    Keypoints layout, per hand: 3 wrist translation, 3 wrist rotation, 63 finger slots.
    """

    groups: tuple[tuple[str, int, int, float], ...] = (
        ("left_translation",  0,   3,  1.0),
        ("left_rotation",     3,   6,  1.0),
        ("left_joints",       6,  69,  1.0),
        ("right_translation", 69, 72,  1.0),
        ("right_rotation",    72, 75,  1.0),
        ("right_joints",      75, 138, 1.0),
    )


# --------------------------------------------------------------------------- data

@dataclass
class DataConfig:
    """Read by the dataloader."""

    stride: int = 3               # source_fps / control_hz
    interpolation: str = "pchip_bone_preserving"
    geometric_augmentation: bool = False
    random_paraphrase: bool = True

    sources: tuple[dict[str, Any], ...] = ()
    stats_path: str = ""


# --------------------------------------------------------------------------- train

@dataclass
class TrainConfig:
    # The window of control steps the dataloader draws, as for the main model: up to 9
    # steps, the length L drawn with probability proportional to L^length_alpha.
    max_seq_len: int = 9
    length_alpha: float = 1.0

    use_only_last_step: bool = True   # no memory: only the last step of a window is read
    seed: int = 0
    parallel: str = "ddp"             # "fsdp" | "ddp"


@dataclass
class OptimConfig:
    """Optimiser and schedule: LeRobot's SmolVLA preset, plus `vlm_lr`."""

    lr: float = 1e-4

    # Learning rate of the VLM when it is trained. None: one group at `lr` for everything.
    vlm_lr: float | None = None

    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 1e-10
    grad_clip: float = 10.0
    grad_accum_steps: int = 1

    # Cosine decay with warmup.
    warmup_steps: int = 1_000
    decay_steps: int = 30_000
    decay_lr: float = 2.5e-6


@dataclass
class InitConfig:
    pretrain_path: str = ""           # weights to start from, see `model.load_pretrained`
    allow_partial_load: bool = False
    # A checkpoint of this model carries its normalisation statistics. True: after loading
    # one, install this run's statistics file instead.
    renormalize: bool = False


# --------------------------------------------------------------------------- top

@dataclass
class Config:
    """The whole configuration of the baseline."""

    vlm: VLMConfig = field(default_factory=VLMConfig)
    image: ImageConfig = field(default_factory=ImageConfig)
    action: ActionConfig = field(default_factory=ActionConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    init: InitConfig = field(default_factory=InitConfig)

    # Loss expected on the first batch, checked before training. None: no check (the
    # output projection is not zero-initialised, so there is no fixed value to expect).
    preflight_loss: float | None = None

    # ----------------------------------------------------------------- validation

    def validate(self) -> None:
        v, a, f = self.vlm, self.action, self.flow

        if self.image.mode != "pad":
            raise ValueError(
                f"image.mode {self.image.mode!r}: SmolVLA's preprocessing IS the shared "
                f"loader's `pad` path (512, aspect preserved, padded left and top). "
                f"'squash' is PaliGemma's and belongs to the VITRA baseline.")

        if a.action_type != "keypoints":
            raise ValueError(
                f"action.action_type {a.action_type!r}: this baseline uses the keypoints "
                f"layout.")

        if a.dim != 192 or a.state_dim != 212:
            raise ValueError(
                f"action.dim={a.dim}, state_dim={a.state_dim}: the normalisation "
                f"statistics and the loss groups are tied to 192/212.")

        hi = max(h for _, _, h, _ in self.loss.groups)
        if hi > a.dim:
            raise ValueError(f"loss group ends at {hi}, past action dim {a.dim}")
        if self.loss.groups[2][2] != 69:
            raise ValueError(
                "loss.groups do not carry the keypoints ranges. keypoints is "
                "(0,3)/(3,6)/(6,69)/(69,72)/(72,75)/(75,138).")

        if v.attention_mode not in ("cross_attn", "self_attn"):
            raise ValueError(f"vlm.attention_mode {v.attention_mode!r}")
        if v.num_expert_layers > 0 and v.num_vlm_layers % v.num_expert_layers != 0:
            raise ValueError(
                f"num_vlm_layers {v.num_vlm_layers} is not a multiple of "
                f"num_expert_layers {v.num_expert_layers}; `get_model_layers` divides them")
        if not v.train_expert_only and not v.freeze_vision_encoder:
            # allowed, as in LeRobot: the whole VLM is trained
            pass

        if f.num_steps < 1:
            raise ValueError("flow.num_steps must be >= 1")

        if self.train.parallel not in ("fsdp", "ddp"):
            raise ValueError(f"train.parallel {self.train.parallel!r}")
        if not self.train.use_only_last_step:
            raise ValueError(
                "use_only_last_step=False: this model has no memory and no way to consume "
                "earlier steps.")

        if self.data.stride != int(round(a.source_fps / a.control_hz)):
            raise ValueError(
                f"data.stride {self.data.stride} != source_fps/control_hz = "
                f"{a.source_fps}/{a.control_hz}")
        if not self.data.stats_path:
            raise ValueError(
                "data.stats_path is empty. Set it to the normalisation statistics file "
                "of the data.")


class FlatConfig:
    """Flat view of `Config` with the attribute names `VLAFlowMatching` reads, which are
    those of LeRobot's `SmolVLAConfig`."""

    def __init__(self, cfg: Config):
        v, a, f = cfg.vlm, cfg.action, cfg.flow
        self.vlm_model_name = v.model_id
        self.freeze_vision_encoder = v.freeze_vision_encoder
        self.train_expert_only = v.train_expert_only
        self.train_state_proj = v.train_state_proj
        self.load_vlm_weights = True          # the model sets it from its `load_weights` argument
        self.attention_mode = v.attention_mode
        self.num_expert_layers = v.num_expert_layers
        self.num_vlm_layers = v.num_vlm_layers
        self.self_attn_every_n_layers = v.self_attn_every_n_layers
        self.expert_width_multiplier = v.expert_width_multiplier
        self.add_image_special_tokens = v.add_image_special_tokens
        self.prefix_length = v.prefix_length
        self.use_cache = v.use_cache
        self.device = None
        self.max_state_dim = a.state_dim
        self.max_action_dim = a.dim
        self.chunk_size = a.chunk_size
        self.num_steps = f.num_steps
        self.min_period = f.min_period
        self.max_period = f.max_period


# --------------------------------------------------------------------------- json

def apply_overrides(cfg: Any, overrides: dict[str, Any], path: str = "") -> None:
    """Recursively overwrite dataclass fields from a parsed json. Unknown keys raise."""
    known = {f.name: f for f in fields(cfg)}
    for key, value in overrides.items():
        if key not in known:
            raise KeyError(f"unknown config key {path}{key!r}; known here: {sorted(known)}")
        current = getattr(cfg, key)
        if is_dataclass(current) and isinstance(value, dict):
            apply_overrides(current, value, f"{path}{key}.")
        elif isinstance(current, tuple) and isinstance(value, list):
            setattr(cfg, key, tuple(tuple(v) if isinstance(v, list) else v for v in value))
        else:
            setattr(cfg, key, value)


def from_json(overrides: dict[str, Any]) -> Config:
    cfg = Config()
    apply_overrides(cfg, overrides)
    cfg.validate()
    return cfg
