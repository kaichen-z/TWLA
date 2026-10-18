"""Settings of the VITRA baseline. The defaults are the values VITRA's own code uses.

The sections `vlm`, `action`, `data` and `train` carry the field names that the
`dataloader` package reads.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any


# --------------------------------------------------------------------------- VLM

@dataclass
class VLMConfig:
    """PaliGemma2-3B backbone. The 224 checkpoint yields 16 x 16 = 256 image tokens."""

    model_id: str = "google/paligemma2-3b-mix-224"
    hidden_size: int = 2304       # text width of PaliGemma2-3B; also the DiT's condition size
    image_size: int = 224
    image_tokens: int = 256       # (224/14)^2
    num_cameras: int = 1

    # padded prompt length
    max_language_tokens: int = 256 + 1 + 48 + 1   # image | <bos> | text | \n

    # "full_finetune" | "only_head_and_token" | "freeze_vision_encoder"
    freeze_option: str = "freeze_vision_encoder"

    # Training: the backbone's learning rate is zero before this step.
    llm_freeze_step: int = 5_000

    cognition_token_id: int = 10          # an unused token id
    untied_cognition_token: bool = True   # own embedding, initialised from a word embedding
    cognition_token_init_id: int | None = None   # None -> use `cognition_token_id`

    use_fov: bool = True
    use_state: str = "DiT"        # the state goes to the DiT; "VLM" is not implemented

    use_bf16: bool = True         # bf16 autocast in the VLM forward; the weights stay fp32
    gradient_checkpointing: bool = True   # training only


# --------------------------------------------------------------------------- image

@dataclass
class ImageConfig:
    """How a frame is resized to `image_size` x `image_size`."""

    # "squash": resize to a square without keeping the aspect ratio, as PaliGemma's
    # processor does. "pad": keep the aspect ratio and pad on the left and top.
    mode: str = "squash"

    interpolation: str = "area"   # cv2.INTER_AREA when downscaling


# --------------------------------------------------------------------------- action

@dataclass
class ActionConfig:
    """The action and state vectors."""

    chunk_size: int = 16          # actions per predicted chunk
    dim: int = 192                # padded action width
    state_dim: int = 212          # padded state width

    control_hz: float = 10.0      # one prediction per 0.1 s
    action_hz: float = 80.0       # rate of the chunk; must match the statistics file
    source_fps: float = 30.0

    # Per-hand layout of the action:
    #   angle      51 = dt(3) + dR(3) + 45 joint angles         right hand at 51:102
    #   keypoints  69 = dt(3) + dR(3) + 63 joint coordinates    right hand at 69:138
    action_type: str = "angle"    # the pretrained action head uses joint angles

    # Makes the loader also emit the keypoints layout (`actions_alt`, `state_alt`), which
    # the metric is scored against. `test.py` turns it on.
    dual_layout: bool = False

    state_mask_prob: float = 0.0  # probability of dropping the state; 0 = never


# --------------------------------------------------------------------------- DiT

@dataclass
class DiTConfig:
    """The DiT action head. VITRA builds DiT-B."""

    model_type: str = "DiT-B"     # depth 12, width 768, 12 heads
    depth: int = 12
    hidden_size: int = 768
    num_heads: int = 12
    mlp_ratio: float = 4.0

    # An action token is [x_t ; action_mask], so the embedder's input is 2 x 192.
    concat_mask_to_input: bool = True

    learn_sigma: bool = False
    class_dropout_prob: float = 0.1   # condition dropout in training, for guidance

    causal_attention: bool = True     # over the whole sequence, as in VITRA


# --------------------------------------------------------------------------- diffusion

@dataclass
class DiffusionConfig:
    """DDPM training, DDIM sampling."""

    num_train_steps: int = 100        # VITRA uses 100, not the more common 1000
    schedule: str = "squaredcos_cap_v2"
    num_inference_steps: int = 10     # DDIM steps
    ddim_eta: float = 0.0             # deterministic

    repeated_diffusion_steps: int = 8 # training: (noise, t) draws per encoded sample

    cfg_scale: float = 5.0            # guidance scale at sampling; <= 1.0 turns it off

    # (name, lo, hi, weight) for the angle layout; the keypoints layout would be
    # (0,6)/(6,69)/(69,75)/(75,138). The same ranges define the action mask at inference.
    loss_groups: tuple[tuple[str, int, int, float], ...] = (
        ("left_6d",      0,   6,  1.0),
        ("left_joints",  6,  51,  1.0),
        ("right_6d",    51,  57,  1.0),
        ("right_joints", 57, 102, 1.0),
    )


# --------------------------------------------------------------------------- data

@dataclass
class DataConfig:
    """Read by the dataloader; the field names are its interface."""

    stride: int = 3               # source_fps / control_hz
    interpolation: str = "pchip_bone_preserving"
    geometric_augmentation: bool = False
    random_paraphrase: bool = True

    sources: tuple[dict[str, Any], ...] = ()   # filled from the run json
    stats_path: str = ""                       # normalisation statistics, from the run json


# --------------------------------------------------------------------------- train

@dataclass
class TrainConfig:
    """Window sampling and the training loop.

    The loader draws a window of up to `max_seq_len` control steps, as for the main
    model, and this model reads only the last one.
    """

    parallel: str = "fsdp"        # "fsdp" | "ddp"

    max_seq_len: int = 9
    length_alpha: float = 1.0     # P(L) proportional to L^alpha
    use_only_last_step: bool = True   # the model has no memory

    batch_size: int = 64
    max_steps: int = 65_000
    seed: int = 42

    num_workers: int = 8
    prefetch_factor: int = 2


# --------------------------------------------------------------------------- optim

@dataclass
class OptimConfig:
    """AdamW with constant learning rates: one for the backbone, one for the rest."""

    backbone_lr: float = 1e-5
    action_model_lr: float = 1e-4     # action head, FoV encoder and cognition token

    weight_decay: float = 0.1
    action_model_weight_decay: float | None = None   # None -> `weight_decay`

    cognition_token_weight_decay: bool = False

    grad_clip: float = 1.0
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    grad_accum_steps: int = 1


# --------------------------------------------------------------------------- weights

@dataclass
class InitConfig:
    """Loading a pretrained checkpoint with `Model.load_pretrained`."""

    pretrain_path: str = ""       # VITRA's pretrained checkpoint
    allow_partial_load: bool = False   # False: missing keys raise
    renormalize: bool = False     # True: this run's statistics replace the checkpoint's

    # Prefix renames from the keys of VITRA's checkpoint to this module tree. The first
    # matching prefix wins.
    key_remap: tuple[tuple[str, str], ...] = (
        # PaliGemma: transformers 4.47 names -> 5.x names
        ("backbone.vision_tower.vision_model.", "vlm.pg.model.vision_tower."),
        ("backbone.multi_modal_projector.", "vlm.pg.model.multi_modal_projector."),
        ("backbone.language_model.model.",  "vlm.pg.model.language_model."),
        ("backbone.language_model.lm_head.", "vlm.pg.lm_head."),
        # VITRA's module names -> the names used here
        ("act_model.net.",                  "dit."),
        ("fov_encoder.",                    "vlm.fov_encoder."),
        ("cognition_token",                 "vlm.cognition_token"),
    )


# --------------------------------------------------------------------------- root

@dataclass
class Config:
    """All settings of the baseline."""

    vlm: VLMConfig = field(default_factory=VLMConfig)
    image: ImageConfig = field(default_factory=ImageConfig)
    action: ActionConfig = field(default_factory=ActionConfig)
    dit: DiTConfig = field(default_factory=DiTConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    init: InitConfig = field(default_factory=InitConfig)

    # Loss of a head that outputs zero: with epsilon prediction, E[eps^2] = 1.
    preflight_loss: float = 1.0

    # ----------------------------------------------------------------- validation

    def validate(self) -> None:
        """Raise on settings that are inconsistent or not implemented."""
        v, a, d, f = self.vlm, self.action, self.dit, self.diffusion

        if v.use_state != "DiT":
            raise ValueError(
                f"use_state={v.use_state!r}: only 'DiT' is implemented. The upstream 'VLM' "
                f"path is broken (vitra_paligemma.py:320 calls self.state_encoder, which "
                f"is never assigned), so there is no reference behaviour to reproduce.")

        if v.image_size % 14 != 0:
            raise ValueError(f"image_size {v.image_size} is not a multiple of SigLIP's "
                             f"patch size 14")
        expect_tokens = (v.image_size // 14) ** 2
        if v.image_tokens != expect_tokens:
            raise ValueError(
                f"image_tokens {v.image_tokens} != (image_size/14)^2 = {expect_tokens}. "
                f"These must agree or masked_scatter fails on the first batch.")

        if self.image.mode not in ("squash", "pad"):
            raise ValueError(f"image.mode {self.image.mode!r} must be 'squash' or 'pad'")

        if a.action_type not in ("angle", "keypoints"):
            raise ValueError(f"action.action_type {a.action_type!r} must be 'angle' or "
                             f"'keypoints'")
        if a.action_type == "angle" and f.loss_groups[1][2] != 51:
            raise ValueError(
                "action_type='angle' but diffusion.loss_groups still carry the keypoints "
                "ranges. angle is (0,6)/(6,51)/(51,57)/(57,102); keypoints is "
                "(0,6)/(6,69)/(69,75)/(75,138). Mixing them silently trains on the wrong "
                "channels.")

        if a.dim != 192 or a.state_dim != 212:
            raise ValueError(
                f"action.dim={a.dim}, state_dim={a.state_dim}: the normalisation "
                f"statistics and the loss groups are tied to 192/212.")

        hi = max(h for _, _, h, _ in f.loss_groups)
        if hi > a.dim:
            raise ValueError(f"loss group ends at {hi}, past action dim {a.dim}")

        if d.hidden_size % d.num_heads != 0:
            raise ValueError(f"DiT hidden {d.hidden_size} not divisible by "
                             f"{d.num_heads} heads")
        if d.model_type == "DiT-B" and (d.depth, d.hidden_size, d.num_heads) != (12, 768, 12):
            raise ValueError(
                f"model_type 'DiT-B' means depth 12 / width 768 / 12 heads upstream, got "
                f"{d.depth}/{d.hidden_size}/{d.num_heads}. Rename the type or fix the "
                f"numbers -- do not ship a 'DiT-B' that is not one.")

        if f.num_inference_steps > f.num_train_steps:
            raise ValueError(f"num_inference_steps {f.num_inference_steps} exceeds "
                             f"num_train_steps {f.num_train_steps}")
        if f.num_train_steps % f.num_inference_steps != 0:
            # range(0, T, T // N) must have exactly N points
            raise ValueError(
                f"num_train_steps {f.num_train_steps} is not a multiple of "
                f"num_inference_steps {f.num_inference_steps}; the DDIM grid would be "
                f"ragged. See diffusion.ddim_sample for why the grid matters.")
        if f.repeated_diffusion_steps < 1:
            raise ValueError("repeated_diffusion_steps must be >= 1")

        if v.llm_freeze_step < 0:
            raise ValueError("llm_freeze_step must be >= 0")

        if self.train.parallel not in ("fsdp", "ddp"):
            raise ValueError(f"train.parallel {self.train.parallel!r} must be 'fsdp' or "
                             f"'ddp'")

        if self.train.max_seq_len < 1:
            raise ValueError("train.max_seq_len must be >= 1")
        if not self.train.use_only_last_step:
            raise ValueError(
                "use_only_last_step=False: this model has no memory and no way to consume "
                "earlier steps. Set it to True, or use a model that has one.")

        if self.data.stride != int(round(a.source_fps / a.control_hz)):
            raise ValueError(
                f"data.stride {self.data.stride} != source_fps/control_hz = "
                f"{a.source_fps}/{a.control_hz}")

        if not self.data.stats_path:
            raise ValueError(
                "data.stats_path is empty. The baseline must use normalisation statistics "
                "computed by the same data pipeline as the main model.")

        if self.init.pretrain_path and self.init.allow_partial_load:
            # allowed; nothing to check
            pass


# --------------------------------------------------------------------------- json

def apply_overrides(cfg: Any, overrides: dict[str, Any], path: str = "") -> None:
    """Overwrite dataclass fields from a parsed json, recursively. Unknown keys raise."""
    known = {f.name: f for f in fields(cfg)}
    for key, value in overrides.items():
        if key not in known:
            where = f"{path}{key}"
            raise KeyError(f"unknown config key {where!r}; known here: "
                           f"{sorted(known)}")
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
