"""Native 15-region / 880-cell Tachin tactile editors.

This module deliberately does not reshape the glove into the unrelated
EgoTouch 21x21 grid.  One shared spatial encoder is applied to each native
sensor region, followed by sensor and temporal attention.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from .action_layout import FULL_HAND_GROUPS
from .direct_residual import ActionResidualBasisTrunk
from .rgb_world_editor import FrozenRGBWorldResidual, RGBWorldResidualOutput
from .single_stage_world import (TactileWorldModel, gather_action_window,
                                 tactile_flow_pair)
from .unified_world import DirectMemoryActionEditor


SENSOR_SHAPES = ((16, 10), (4, 8), (13, 8), (9, 8), (3, 8), (4, 8),
                 (9, 8), (3, 8), (4, 8), (9, 8), (3, 8), (4, 8),
                 (9, 8), (3, 8), (13, 8))
SENSOR_OFFSETS = (0, 160, 192, 296, 368, 392, 424, 496, 520, 552, 624,
                  648, 680, 752, 776, 880)


class NativeTachinEncoder(nn.Module):
    """Encode masked native cells into one 64-D token per hand."""

    def __init__(self, pressure_mean: torch.Tensor, pressure_scale: torch.Tensor,
                 hidden_dim: int = 64) -> None:
        super().__init__()
        if tuple(pressure_mean.shape) != (2, 880) or tuple(pressure_scale.shape) != (2, 880):
            raise ValueError("Tachin pressure statistics must be [2,880]")
        self.register_buffer("pressure_mean", pressure_mean.float(), persistent=True)
        self.register_buffer("pressure_scale", pressure_scale.float().clamp_min(1e-6), persistent=True)
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, hidden_dim, 3, padding=1), nn.GELU(),
        )
        self.sensor_embedding = nn.Parameter(torch.randn(1, 15, hidden_dim) * 0.02)
        sensor_layer = nn.TransformerEncoderLayer(hidden_dim, 4, hidden_dim * 4,
                                                   dropout=0.1, activation="gelu",
                                                   batch_first=True, norm_first=True)
        self.sensor_blocks = nn.TransformerEncoder(sensor_layer, 1,
                                                    enable_nested_tensor=False)
        self.temporal = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def _encode(self, normalized: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        batch, history, hands = normalized.shape[:3]
        sensor_tokens, sensor_valid = [], []
        for sensor, ((height, width), start, stop) in enumerate(
                zip(SENSOR_SHAPES, SENSOR_OFFSETS[:-1], SENSOR_OFFSETS[1:])):
            values = normalized[..., start:stop].reshape(batch * history * hands, 1, height, width)
            masks = valid[..., start:stop].reshape(batch * history * hands, 1, height, width)
            features = self.spatial(torch.cat((values, masks.to(values.dtype)), 1))
            denominator = masks.sum((2, 3)).clamp_min(1).to(features.dtype)
            pooled = (features * masks.to(features.dtype)).sum((2, 3)) / denominator
            sensor_tokens.append(pooled)
            sensor_valid.append(masks.any((1, 2, 3)))
        tokens = torch.stack(sensor_tokens, 1) + self.sensor_embedding
        sensor_mask = torch.stack(sensor_valid, 1)
        safe_mask = sensor_mask.clone()
        safe_mask[~safe_mask.any(1), 0] = True
        encoded = self.sensor_blocks(tokens, src_key_padding_mask=~safe_mask)
        pooled = (encoded * safe_mask[..., None]).sum(1) / safe_mask.sum(1, keepdim=True).clamp_min(1)
        pooled = pooled.reshape(batch, history, hands, -1).permute(0, 2, 1, 3)
        sequence = pooled.reshape(batch * hands, history, -1)
        self.temporal.flatten_parameters()
        temporal = self.temporal(sequence)[0].reshape(batch, hands, history, -1)
        return self.norm(temporal[:, :, -1])

    def forward(self, pressure: torch.Tensor, cell_valid: torch.Tensor,
                sample_valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if tuple(pressure.shape[1:]) != (8, 2, 880):
            raise ValueError("pressure must be [B,8,2,880]")
        if cell_valid.shape != pressure.shape or sample_valid.shape != pressure.shape[:3]:
            raise ValueError("Tachin tactile masks have the wrong shape")
        normalized = (torch.nan_to_num(pressure.float()) - self.pressure_mean) / self.pressure_scale
        valid = cell_valid.bool() & sample_valid.bool()[..., None]
        normalized = normalized * valid.to(normalized.dtype)
        # Center against an exact train-mean reference with the same missing-cell
        # pattern.  Zero tactile evidence therefore produces an exact zero token.
        combined = self._encode(torch.cat((normalized, torch.zeros_like(normalized)), 0),
                                torch.cat((valid, valid), 0))
        actual, reference = combined.chunk(2, 0)
        hand_valid = sample_valid[:, -1].bool() & cell_valid[:, -1].any(-1)
        output = (actual - reference) * hand_valid[..., None].to(actual.dtype)
        return output, hand_valid


def allowed_mask(edit_mask: torch.Tensor, hand_valid: torch.Tensor) -> torch.Tensor:
    allowed = edit_mask.bool().clone()
    allowed[..., 138:] = False
    allowed[..., :69] &= hand_valid[:, 0, None, None]
    allowed[..., 69:138] &= hand_valid[:, 1, None, None]
    return allowed


class NativeResidualEditor(nn.Module):
    """The identical memory-free editor used for SmolVLA, VITRA, and T-Rex."""

    def __init__(self, encoder: NativeTachinEncoder, residual_bases: int = 8) -> None:
        super().__init__()
        self.encoder = encoder
        self.action = ActionResidualBasisTrunk(torch.zeros(192), torch.ones(192),
                                               residual_bases=residual_bases)
        self.coefficients = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, residual_bases, bias=False))
        self.residual_bases = residual_bases

    def forward(self, base_action, edit_mask, pressure, cell_valid, sample_valid):
        tactile, hand_valid = self.encoder(pressure, cell_valid, sample_valid)
        coefficients = self.coefficients(tactile.flatten(1))
        # LayerNorm has trainable affine parameters.  Without an explicit
        # centered-feature gate, its learned bias can create a global action
        # correction even when pressure is exactly the train-mean reference.
        # Match the EgoTouch editor contract: no tactile evidence means an
        # exact frozen-base plan, including after training.
        active = hand_valid.any(1, keepdim=True) & (
            tactile.abs().sum((1, 2), keepdim=False)[:, None] > 0
        )
        coefficients = coefficients * active.to(coefficients.dtype)
        bases = self.action(base_action, torch.zeros(len(base_action), device=base_action.device))
        raw_delta = (coefficients[:, :, None, None] * bases).sum(1) / math.sqrt(self.residual_bases)
        allowed = allowed_mask(edit_mask, hand_valid)
        delta = raw_delta * allowed.to(raw_delta.dtype)
        return base_action.detach() + delta, delta, allowed, tactile, hand_valid


@dataclass
class NativeW3Output:
    final_action: torch.Tensor
    delta_action: torch.Tensor
    allowed_mask: torch.Tensor
    tactile_features: torch.Tensor
    hand_valid: torch.Tensor
    tactile_velocity: torch.Tensor | None = None
    tactile_target: torch.Tensor | None = None
    tactile_valid: torch.Tensor | None = None
    rgb: RGBWorldResidualOutput | None = None


class NativeTachinW3Editor(nn.Module):
    """Real W3: memory action editor plus RGB and tactile world objectives."""

    def __init__(self, encoder: NativeTachinEncoder, pressure_mean: torch.Tensor,
                 pressure_scale: torch.Tensor) -> None:
        super().__init__()
        self.encoder = encoder
        self.register_buffer("pressure_mean", pressure_mean.float(), persistent=True)
        self.register_buffer("pressure_scale", pressure_scale.float().clamp_min(1e-6), persistent=True)
        self.action = DirectMemoryActionEditor(torch.zeros(192), torch.ones(192),
                                               tactile_dim=64, articulation_only=False)
        self.tactile_world = TactileWorldModel(tactile_dim=64, cells=880)
        self.rgb_world = FrozenRGBWorldResidual(tactile_dim=64, memory_rows=8,
                                                condition_dim=2304, hidden_dim=256)

    def forward(self, *, base_action, edit_mask, pressure, cell_valid, sample_valid,
                final_memory, future_pressure=None, future_cell_valid=None,
                tactile_tau=None, tactile_noise=None, rgb_world_head=None,
                image_tokens=None, clean_latent=None, rgb_valid=None,
                rgb_tau=None, rgb_noise=None, compute_auxiliary=True,
                compute_rgb=True, compute_tactile=True):
        tactile, hand_valid = self.encoder(pressure, cell_valid, sample_valid)
        final, delta, allowed = self.action(base_action, final_memory, tactile, hand_valid,
                                            torch.zeros(len(base_action), device=base_action.device),
                                            edit_mask)
        tactile_velocity = tactile_target = tactile_valid = None
        rgb = None
        if compute_auxiliary and compute_tactile:
            required = (future_pressure, future_cell_valid, tactile_tau, tactile_noise)
            if any(value is None for value in required):
                raise ValueError("W3 tactile auxiliary inputs are incomplete")
            tactile_valid = future_cell_valid.bool()
            clean = ((future_pressure.float() - self.pressure_mean) / self.pressure_scale)
            clean = clean * tactile_valid.to(clean.dtype)
            noisy, tactile_target = tactile_flow_pair(clean, tactile_valid, tactile_tau, tactile_noise)
            action_window, action_valid = gather_action_window(final.detach(),
                torch.zeros(len(final), device=final.device), steps=8)
            tactile_velocity = self.tactile_world.velocity(noisy, tactile_tau,
                final_memory=final_memory.detach(), tactile_features=tactile,
                hand_valid=hand_valid, action_window=action_window, action_valid=action_valid)
        if compute_auxiliary and compute_rgb:
            required = (rgb_world_head, image_tokens, clean_latent, rgb_valid, rgb_tau, rgb_noise)
            if any(value is None for value in required):
                raise ValueError("W3 RGB auxiliary inputs are incomplete")
            rgb = self.rgb_world(world_head=rgb_world_head, image_tokens=image_tokens,
                final_memory=final_memory.detach(), tactile_features=tactile,
                hand_valid=hand_valid, clean_latent=clean_latent, world_valid=rgb_valid,
                tau=rgb_tau, noise=rgb_noise)
        return NativeW3Output(final, delta, allowed, tactile, hand_valid,
                              tactile_velocity, tactile_target, tactile_valid, rgb)

    def optimizer_groups(self, action_lr=1e-4, encoder_lr=1e-5,
                         world_adapter_lr=1e-4, world_block_lr=1e-4):
        return [
            {"params": self.action.parameters(), "lr": action_lr, "name": "action"},
            {"params": self.encoder.parameters(), "lr": encoder_lr, "name": "encoder"},
            {"params": self.tactile_world.parameters(), "lr": world_block_lr, "name": "tactile_world"},
            {"params": self.rgb_world.parameters(), "lr": world_adapter_lr, "name": "rgb_world"},
        ]
