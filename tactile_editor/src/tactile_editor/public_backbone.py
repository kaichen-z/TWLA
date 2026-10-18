"""Adapters to the sibling T-WLA foundation model; no copied VLA implementation."""
from pathlib import Path
import json
import os
import sys

import torch
from torch import nn

from .editor_interface import BaseActionContext


def foundation_path():
    default = Path(__file__).resolve().parents[3] / "T-WLA_foundation_model"
    root = Path(os.environ.get("TWLA_FOUNDATION_ROOT", default)).resolve()
    if not (root / "inference/policy.py").is_file():
        raise FileNotFoundError("Set TWLA_FOUNDATION_ROOT to T-WLA_foundation_model")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


@torch.no_grad()
def policy_context(policy, *, rgb, state, state_mask, fov, episode_id,
                   action_mask, action_mean, action_std, normalization_id,
                   model_id="twla-public"):
    """Request all 16 raw actions, then apply the editor's training statistics once.

    The policy owns the recurrent bank. Call this once per VLA observation,
    not once per tactile tick. The caller supplies action validity explicitly.
    """
    mean = torch.as_tensor(action_mean, dtype=torch.float32)
    std = torch.as_tensor(action_std, device=mean.device, dtype=torch.float32)
    if mean.shape != (192,) or std.shape != (192,):
        raise ValueError("Action statistics must have shape [192]")
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
        raise ValueError("Action statistics must be finite, with positive standard deviations")
    valid = torch.as_tensor(action_mask, device=mean.device, dtype=torch.bool)
    if valid.shape != (16, 192) or valid[..., 138:].any():
        raise ValueError("action_mask must be [16,192] with padding dimensions invalid")
    actions, memory = policy.act(rgb, state, state_mask, fov, episode_id,
                                 steps=16, return_mf=True)
    actions = torch.as_tensor(actions, device=mean.device, dtype=torch.float32)
    memory = torch.as_tensor(memory, device=mean.device, dtype=torch.float32)
    if actions.shape != (16, 192):
        raise ValueError("The public policy must return a complete [16,192] chunk")
    if memory.ndim != 2 or memory.shape[0] != 8 or not torch.isfinite(memory).all():
        raise ValueError("The public policy must return finite memory [8,D]")
    return BaseActionContext(
        base_action=((actions - mean) / std)[None], action_mask=valid[None],
        memory_tokens=memory[None], model_id=model_id,
        normalization_id=normalization_id,
    )


class RGBWorldAdapter(nn.Module):
    """Expose the public world head to the tactile RGB regularizer.

    Weights stay frozen. Gradients still pass through the head to tactile
    conditioning tokens. Horizon-major row order matches the VLA world loss.
    """
    def __init__(self, head):
        super().__init__()
        self.head = head.requires_grad_(False).eval()

    @property
    def adapter(self):
        return self.head.adapter

    def build_condition(self, image_tokens, final_memory):
        return self.head.adapter(image_tokens, final_memory)

    def predict_velocity(self, noisy_latent, tau, condition):
        if noisy_latent.ndim != 4 or tau.shape != (len(noisy_latent),):
            raise ValueError("Expected latent [B,C,H,W] and tau [B]")
        if condition.ndim != 3 or len(condition) != len(noisy_latent):
            raise ValueError("Condition must be [B,N,D]")
        return self.head.sana(
            hidden_states=noisy_latent, encoder_hidden_states=condition,
            timestep=tau * self.head.cfg.sana.timestep_scale, return_dict=False,
        )[0]

    def train(self, mode=True):
        super().train(False)
        return self


def load_rgb_world_head(checkpoint, run_config, device):
    """Load the exact world architecture and weights. No fallback initialization."""
    foundation_path()
    from world_model import (AdapterConfig, DCAEConfig, SanaConfig,
                             WorldConfig, WorldLossConfig, WorldHead)

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    metadata = json.loads(Path(run_config).read_text())
    saved = payload.get("effective_config")
    if not isinstance(saved, dict):
        saved = metadata.get("effective_config", metadata.get("config", metadata))
    if not isinstance(saved, dict) or not isinstance(saved.get("world"), dict):
        raise ValueError("The matching run configuration must contain a world section")
    config = dict(saved["world"])
    for name, cls in (("adapter", AdapterConfig), ("sana", SanaConfig),
                      ("dcae", DCAEConfig), ("loss", WorldLossConfig)):
        if isinstance(config.get(name), dict):
            config[name] = cls(**config[name])
    cfg = WorldConfig(**config)
    dtype = torch.bfloat16 if torch.device(device).type == "cuda" else torch.float32
    head = WorldHead(cfg, dtype=dtype, inherit=False)
    weights = payload.get("model")
    if not isinstance(weights, dict):
        raise ValueError("The world checkpoint must contain a model state dictionary")
    if any(key.startswith("world.") for key in weights):
        weights = {k.removeprefix("world."): v for k, v in weights.items()
                   if k.startswith("world.")}
    head.load_state_dict(weights, strict=True)
    return RGBWorldAdapter(head).to(device).eval(), cfg
