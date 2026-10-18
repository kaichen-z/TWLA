"""Configuration of the MemoryVLA baseline.

The architecture settings (VLM, memory banks, DiT, sampler) follow the upstream MemoryVLA code in
`upstream/`. The data, the 192-dim keypoint action layout, the window sampling and the masked group
loss are set here.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any


# --------------------------------------------------------------------------- VLM

@dataclass
class VLMConfig:
    # CogACT-Large, the checkpoint upstream starts from. It has no memory modules: `per_compr`
    # and both banks are not initialised from it.
    pretrained_checkpoint: str = "${VLA_WEIGHTS}/CogACT-Large/checkpoints/CogACT-Large.pt"

    # The Prismatic VLM of CogACT-Large; same values as its entry in `upstream/prismatic/conf/models.py`.
    model_id: str = "prism-dinosiglip-224px+7b"
    vision_backbone_id: str = "dinosiglip-vit-so-224px"
    llm_backbone_id: str = "llama2-7b-pure"
    arch_specifier: str = "no-align+fused-gelu-mlp"
    image_resize_strategy: str = "resize-naive"
    llm_max_length: int = 2048

    # Local directory with the config and tokenizer files of `meta-llama/Llama-2-7b-hf`; the weights
    # come from CogACT. Use the official files: the prompt must end in the token ids [29871, 2]
    # (" </s>"), and a tokenizer that drops the 29871 shifts the EOS the cognitive token is read at.
    llama_local_dir: str = "${VLA_WEIGHTS}/llama2-7b-hf-official"

    # Vision towers, projector and Llama stay frozen; `per_compr`, the memory banks and the DiT train.
    freeze_vlm: bool = True
    use_flash_attention_2: bool = False
    gradient_checkpointing: bool = False  # training side

    # Side of the square frame the dataloader produces; the VLM's input size.
    image_size: int = 224

    # The collate pads the prompt to this many tokens. A longer prompt raises instead of being
    # truncated, because the cognitive token is read at the EOS that ends the prompt.
    max_language_tokens: int = 64


# --------------------------------------------------------------------------- image

@dataclass
class ImageConfig:
    """`squash`: plain resize to the square, aspect ratio not preserved (see data.py)."""

    mode: str = "squash"


# --------------------------------------------------------------------------- action

@dataclass
class ActionConfig:
    chunk_size: int = 16          # CogACT's future_action_window_size (15) + 1
    dim: int = 192                # 7 upstream
    state_dim: int = 212          # read by the loader; the model takes no state

    control_hz: float = 10.0
    action_hz: float = 80.0
    source_fps: float = 30.0

    action_type: str = "keypoints"
    dual_layout: bool = False


# --------------------------------------------------------------------------- memory

@dataclass
class MemoryConfig:
    """Memory bank settings. The first seven fields are upstream's defaults."""

    per_token_size: int = 256
    mem_length: int = 16
    retrieval_layers: int = 2
    use_timestep_pe: bool = True
    fusion_type: str = "gate"
    consolidate_type: str = "tome"
    update_fused: bool = False

    # Upstream's 'group' and 'stream' make `process_batch` clear the banks from the order of the
    # episode ids in a batch (`memory_vla.py:267-288`). Any other value turns that off; model.py
    # clears the banks itself at the start of every window.
    dataloader_type: str = "window"

    # Timestep stored with each memory entry: the position inside the window, 0..L-1, not the
    # step index in the episode.
    timestep: str = "window_relative"


# --------------------------------------------------------------------------- diffusion

@dataclass
class DiffusionConfig:
    """DiT action head, its noise schedule and the sampler."""

    action_model_type: str = "DiT-L"
    diffusion_steps: int = 100               # `ActionModel` default
    noise_schedule: str = "squaredcos_cap_v2"
    repeated_diffusion_steps: int = 4        # noise draws per action chunk in the loss

    use_ddim: bool = True
    num_ddim_steps: int = 10
    cfg_scale: float = 1.5                   # classifier-free guidance; 1.0 turns it off


# --------------------------------------------------------------------------- loss

@dataclass
class LossConfig:
    """Channel groups of the loss: (name, first channel, last channel + 1, weight).

    Upstream averages the squared noise error over all channels. Here it is reduced with the action
    mask, group by group, so that padded channels and an absent hand do not count.
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
    """Settings read by the dataloader."""

    stride: int = 3
    interpolation: str = "pchip_bone_preserving"
    geometric_augmentation: bool = False
    random_paraphrase: bool = True

    sources: tuple[dict[str, Any], ...] = ()
    stats_path: str = ""


# --------------------------------------------------------------------------- train

@dataclass
class TrainConfig:
    # Window sampling: the length L is drawn from 1..max_seq_len with weight L to the power
    # `length_alpha`. The first L-1 steps fill the memory, the last one is supervised.
    max_seq_len: int = 9
    length_alpha: float = 1.0

    use_only_last_step: bool = False   # must stay False: the earlier steps fill the memory
    seed: int = 0
    parallel: str = "ddp"


@dataclass
class OptimConfig:
    """Optimiser and learning-rate schedule (training side)."""

    schedule: str = "constant"
    lr: float = 2e-5
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    grad_accum_steps: int = 1

    warmup_steps: int = 0
    decay_steps: int = 0
    decay_lr: float = 2e-5


@dataclass
class InitConfig:
    pretrain_path: str = ""           # a checkpoint of this baseline to start from
    allow_partial_load: bool = False


# --------------------------------------------------------------------------- top

@dataclass
class Config:
    vlm: VLMConfig = field(default_factory=VLMConfig)
    image: ImageConfig = field(default_factory=ImageConfig)
    action: ActionConfig = field(default_factory=ActionConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    init: InitConfig = field(default_factory=InitConfig)

    # Expected loss of the first training batch: the DiT's final layer starts at zero, so the
    # squared error against unit-variance noise is 1.
    preflight_loss: float | None = 1.0

    def validate(self) -> None:
        a, m, d = self.action, self.memory, self.diffusion
        if self.image.mode != "squash":
            raise ValueError("image.mode must be 'squash' (CogACT's resize-naive)")
        if a.action_type != "keypoints" or a.dim != 192:
            raise ValueError("this baseline trains the 192-dim keypoints layout")
        if a.chunk_size != 16:
            raise ValueError("chunk_size 16 = CogACT's future_action_window_size 15 + 1; "
                             "DiT-L's positional embedding is sized by it")
        if m.dataloader_type in ("group", "stream"):
            raise ValueError("memory.dataloader_type must NOT be 'group'/'stream': those "
                             "auto-clear the bank by batch order, which our windowed "
                             "burn-in does not provide")
        if m.timestep not in ("window_relative",):
            raise ValueError(f"memory.timestep {m.timestep!r}")
        if self.train.use_only_last_step:
            raise ValueError("use_only_last_step=True would skip the burn-in")
        if d.cfg_scale < 1.0:
            raise ValueError("diffusion.cfg_scale < 1")
        if self.loss.groups[2][2] != 69:
            raise ValueError("loss.groups must carry the keypoints ranges")
        if self.data.stride != int(round(a.source_fps / a.control_hz)):
            raise ValueError("data.stride != source_fps / control_hz")
        if not self.data.stats_path:
            raise ValueError("data.stats_path is empty: set the normalisation statistics file")


# --------------------------------------------------------------------------- json

def apply_overrides(cfg: Any, overrides: dict[str, Any], path: str = "") -> None:
    """Overwrite dataclass fields from a parsed json, recursively. Unknown keys raise."""
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
