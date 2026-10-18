"""Configuration of the model, the training schedule and the data.

Structural fields have values here. Training hyperparameters are `REQUIRED`: they have
no default and are set by the run config.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Type-only: `world_model` imports this package, so a runtime import would be circular.
    from world_model.config import WorldConfig

try:
    from ..config_validation import validate_common_config, REQUIRED, require_parameters
except ImportError:   # imported with T-WLA_foundation_model itself on sys.path
    from config_validation import validate_common_config, REQUIRED, require_parameters


@dataclass
class VLMConfig:
    """The vision-language backbone."""

    model_id: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"

    num_layers: int = 16          # keep the first N of 32 decoder layers, drop the rest
    hidden_size: int = 960        # SmolLM2 width; also the width of the memory tokens
    image_size: int = 512
    tokens_per_frame: int = 64    # 32x32 patches, pixel-shuffled by 4

    # Fixed at run time: inference uses static shapes.
    num_cameras: int = 1
    max_language_tokens: int = 32  # instructions are padded to this length

    # Which parts of the backbone are trained.
    freeze_vision: bool = REQUIRED
    freeze_connector: bool = REQUIRED       # the pixel-shuffle Linear
    freeze_embeddings: bool = REQUIRED
    train_top_k_layers: int = REQUIRED      # number of top decoder layers left trainable


@dataclass
class MemoryConfig:
    """The memory module and its bank."""

    num_slots: int = 8            # S
    dim: int = 960                # must equal VLMConfig.hidden_size
    num_blocks: int = 2           # L
    num_heads: int = 15           # 15 x 64 = 960, the backbone's head_dim

    # One bank entry per observation: at 10 Hz, 8 entries cover the last 0.8 s and hold
    # 8 x 8 = 64 tokens. The bank is reset at every episode start.
    bank_capacity: int = 8

    age_fusion: str = "add"       # how slot and age join a bank entry: "add" | "concat_mlp"

    qk_norm: bool = True                    # normalise q and k, with a learned logit scale
    zero_init_residual_gate: bool = True    # each block starts as the identity

    ffn_intermediate: int = 2560  # SwiGLU hidden width; same parameter count as a 4x MLP


@dataclass
class ExpertConfig:
    """The flow-matching action expert. The sizes follow SmolVLA."""

    num_layers: int = 16
    hidden_size: int = 720
    num_heads: int = 15           # q_proj: 720 -> 15 x 64 = 960
    num_kv_heads: int = 5         # grouped-query attention
    head_dim: int = 64
    self_attn_every_n_layers: int = 2   # even layers SA, odd layers CA

    cond_proj_in: int = 960       # m_f width
    expert_slot_embedding: bool = False  # add a learned per-slot embedding to the memory tokens
    causal_actions: bool = True   # in SA layers, action h attends to actions <= h


@dataclass
class ActionConfig:
    """Action space and control rate.

    Of the 192 action dims, 138 carry data: per hand 3 wrist translation, 3 wrist rotation
    (Euler) and 21 x 3 keypoints. The padded layout follows VITRA.
    """

    chunk_size: int = 16
    dim: int = 192
    executed_steps: int = 8       # actions returned per control step; the rest is headroom

    control_hz: float = 10.0      # one VLM forward and one memory write per 0.1 s
    action_hz: float = 80.0       # ground truth is interpolated to this rate; 8 per control step
    source_fps: float = 30.0      # native rate of the recorded hand pose

    # 212 state dims, concatenated with their mask -> 424. The state always enters through
    # the VLM prefix; `direct_state_to_expert` also feeds it to the expert.
    state_dim: int = 212
    direct_state_to_expert: bool = False
    # Mask out the 63 keypoint dims of each hand in the state, in training and inference,
    # so that the hand pose is read from the image. The wrist pose stays in the state.
    hide_hand_state: bool = False


@dataclass
class FlowConfig:
    """Rectified flow, with the SmolVLA convention:

        x_t = tau * noise + (1 - tau) * action        tau = 1 is the noise end
        target velocity u = noise - action
    """

    # Training distribution of tau: time_offset + time_scale * Beta(alpha, beta).
    time_beta_alpha: float = REQUIRED
    time_beta_beta: float = REQUIRED
    time_scale: float = REQUIRED
    time_offset: float = REQUIRED

    num_inference_steps: int = 10   # Euler

    # Independent (noise, tau) pairs per observation, sharing one VLM forward.
    draws_per_condition: int = REQUIRED

    # Loss groups as (name, start, stop) over the action dims. The loss is averaged within
    # each group and the groups are combined with `component_weights`.
    component_groups: tuple[tuple[str, int, int], ...] = (
        ("left_translation", 0, 3),
        ("left_rotation", 3, 6),
        ("left_articulation", 6, 69),
        ("right_translation", 69, 72),
        ("right_rotation", 72, 75),
        ("right_articulation", 75, 138),
    )
    component_weights: tuple[float, ...] = REQUIRED

    # Per-step weights over the action chunk. None = uniform.
    horizon_weights: tuple[float, ...] | None = REQUIRED


@dataclass
class TrainConfig:
    """Sequence training.

    One sample is L consecutive control steps. The first L-1 run under `no_grad` and fill
    the memory bank; only the last one is supervised.
    """

    # L is drawn from {1 .. max_seq_len} with P(L) proportional to L to the power `length_alpha`.
    max_seq_len: int = REQUIRED
    length_alpha: float = REQUIRED
    log_length_histogram: bool = REQUIRED    # log the realised distribution of L

    # Probability of dropping the state / the memory from the model input.
    state_dropout: float = REQUIRED
    memory_dropout: float = REQUIRED

    freeze_expert: bool = REQUIRED           # keep the action expert fixed
    freeze_extra: tuple[str, ...] = REQUIRED  # further parameter-name prefixes to freeze


@dataclass
class OptimConfig:
    """Optimiser (AdamW) and learning-rate schedule (linear warmup, then cosine decay)."""

    lr: float = REQUIRED                     # peak learning rate
    betas: tuple[float, float] = REQUIRED
    eps: float = REQUIRED

    warmup_steps: int = REQUIRED
    decay_steps: int = REQUIRED              # cosine decay length
    final_lr: float = REQUIRED
    schedule_offset: int = REQUIRED          # absolute step at which this schedule starts

    lr_expert: float | None = REQUIRED       # peak rate for `expert.*`; None = same as `lr`
    expert_warmup_steps: int = REQUIRED
    lr_vlm: float | None = REQUIRED          # peak rate for the trainable part of `vlm.*`; None = same as `lr`

    constant_after_warmup: bool = REQUIRED   # hold the peak after warmup instead of decaying

    # Reduce-on-plateau on top of the schedule, driven by the validator's results.
    # `plateau_factor` 0 = off.
    plateau_factor: float = REQUIRED
    plateau_patience: int = REQUIRED
    plateau_threshold: float = REQUIRED
    plateau_min_lr: float = REQUIRED

    weight_decay: float = REQUIRED
    grad_clip: float = REQUIRED
    grad_accum_steps: int = REQUIRED
    bf16: bool = REQUIRED

    # Parameters excluded from weight decay: by dimensionality, and by name.
    no_decay_ndim_max: int = REQUIRED
    no_decay_name_parts: tuple[str, ...] = REQUIRED


@dataclass
class DataConfig:
    """What the dataloader does to a sample."""

    stride: int = REQUIRED                   # source frames between control steps
    interpolation: str = REQUIRED            # how ground truth is resampled to `action_hz`
    geometric_augmentation: bool = REQUIRED  # trajectory-aware image / action augmentation
    random_paraphrase: bool = REQUIRED       # draw one of the instruction paraphrases per sample


@dataclass
class TWLAConfig:
    vlm: VLMConfig = field(default_factory=VLMConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    expert: ExpertConfig = field(default_factory=ExpertConfig)
    action: ActionConfig = field(default_factory=ActionConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)

    # Optional world head (`world_model/`), a training-time regulariser that predicts
    # future frames from `m_f`. None: the head is not built and the state_dict has no
    # `world.*` keys.
    world: WorldConfig | None = None

    def __post_init__(self) -> None:
        # Validation is in validate(): the required fields are set after construction.
        pass

    def validate(self) -> None:
        """Check the config. Call after the run config has been applied."""
        require_parameters(self)
        validate_common_config(self)

        if self.expert.cond_proj_in != self.vlm.hidden_size:
            raise ValueError("expert.cond_proj_in must equal vlm.hidden_size (m_f width).")

        if self.world is not None:
            # A JSON run config gives a plain dict: build the dataclasses from it.
            # Imported here to avoid a circular import.
            if isinstance(self.world, dict):
                from world_model.config import (AdapterConfig, DCAEConfig, SanaConfig,
                                                WorldConfig, WorldLossConfig)
                d = dict(self.world)
                for key, kls in (("adapter", AdapterConfig), ("sana", SanaConfig),
                                 ("dcae", DCAEConfig), ("loss", WorldLossConfig)):
                    if isinstance(d.get(key), dict):
                        d[key] = kls(**d[key])
                self.world = WorldConfig(**d)
            self.world.validate_against_v4(self)

    @property
    def prefix_len(self) -> int:
        """Number of prefix tokens. Does not depend on the batch content."""
        v = self.vlm
        return (v.tokens_per_frame * v.num_cameras     # image
                + v.num_cameras                        # FoV, one token per camera
                + v.max_language_tokens                # padded
                + 1                                    # state
                + self.memory.num_slots)               # memory

    @property
    def num_age_values(self) -> int:
        """Number of age embeddings: one per age in 1..bank_capacity."""
        return self.memory.bank_capacity

    @property
    def bank_tokens(self) -> int:
        """Number of bank tokens when the bank is full."""
        return self.memory.bank_capacity * self.memory.num_slots

    @property
    def memory_kv_len(self) -> int:
        """K/V length seen by the memory module when the bank is full."""
        return self.memory.num_slots + self.bank_tokens
