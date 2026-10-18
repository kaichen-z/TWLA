"""W3 tactile editor with RGB and tactile prediction objectives.

The public trainer constructs W3 and controls ablations through loss weights.
Forward flags skip disabled auxiliary branches without changing checkpoint keys.
Legacy world-mode values remain for loading and testing earlier architectures.
The frozen VLA is external: cached memory and normalized actions are inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch
from torch import nn

from .action_objectives import per_sample_action_loss
from .action_layout import ARTICULATION_GROUPS, FULL_ACTION_GROUPS, FULL_HAND_GROUPS
from .encoder import DualHandTactileEncoder
from .rgb_world_editor import (
    FrozenRGBWorldResidual,
    RGBWorldResidualOutput,
    roi_weighted_world_loss,
)
from .single_stage_world import (
    TactileTargetCodec,
    TactileWorldModel,
    gather_action_window,
    masked_tactile_flow_loss,
    tactile_flow_pair,
)


class WorldMode(str, Enum):
    W0 = "w0"
    W1_RGB = "w1_rgb"
    W2_TACTILE = "w2_tactile"
    W3_BOTH = "w3_both"

    @property
    def uses_rgb(self) -> bool:
        return self in {WorldMode.W1_RGB, WorldMode.W3_BOTH}

    @property
    def uses_tactile(self) -> bool:
        return self in {WorldMode.W2_TACTILE, WorldMode.W3_BOTH}


class CenteredTactileFeatures(nn.Module):
    """Return two reference-centred tokens from one shared P1 encoder."""

    def __init__(
        self,
        encoder: DualHandTactileEncoder,
        reference_pressure: torch.Tensor,
    ) -> None:
        super().__init__()
        if tuple(reference_pressure.shape) == (1, 8, 2, 21, 21):
            reference_pressure = reference_pressure[0]
        if tuple(reference_pressure.shape) != (8, 2, 21, 21):
            raise ValueError("reference_pressure must be [8,2,21,21]")
        self.encoder = encoder
        self.register_buffer(
            "reference_pressure", reference_pressure.float(), persistent=True
        )

    def forward(
        self, pressure: torch.Tensor, temporal_valid: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if pressure.ndim != 5 or tuple(pressure.shape[1:]) != (8, 2, 21, 21):
            raise ValueError("pressure must be [B,8,2,21,21]")
        if temporal_valid.shape != pressure.shape[:3]:
            raise ValueError("temporal_valid must be [B,8,2]")
        batch = len(pressure)
        reference = self.reference_pressure[None].expand(batch, -1, -1, -1, -1)
        reference_valid = torch.ones_like(temporal_valid, dtype=torch.bool)
        # One call keeps the actual/reference paths numerically symmetric.
        features = self.encoder(
            torch.cat((pressure, reference.to(pressure.dtype)), dim=0),
            torch.cat((temporal_valid.bool(), reference_valid), dim=0),
        )
        actual, baseline = features.tokens.split(batch, dim=0)
        hand_valid = features.valid[:batch] & temporal_valid[:, -1].bool()
        centered = (actual - baseline) * hand_valid[..., None].to(actual.dtype)
        return centered, hand_valid


class DirectMemoryActionEditor(nn.Module):
    """Predict a direct per-hand delta from ``m_f + a_base + z_touch``.

    Tactile tokens generate bias-free FiLM terms for action features. A zero
    centered token contributes no tactile FiLM term. Reference behavior remains
    a diagnostic because each sample also has a temporal-validity pattern.
    """

    def __init__(
        self,
        action_mean: torch.Tensor,
        action_std: torch.Tensor,
        *,
        memory_dim: int = 960,
        tactile_dim: int = 64,
        hidden_dim: int = 256,
        num_layers: int = 2,
        articulation_only: bool = False,
    ) -> None:
        super().__init__()
        if tuple(action_mean.shape) != (192,) or tuple(action_std.shape) != (192,):
            raise ValueError("action statistics must be [192]")
        self.register_buffer("action_mean", action_mean.float(), persistent=True)
        self.register_buffer(
            "action_std", action_std.float().clamp_min(1e-3), persistent=True
        )
        self.memory_norm = nn.RMSNorm(memory_dim)
        self.memory_projection = nn.Linear(memory_dim, hidden_dim, bias=False)
        self.action_projection = nn.Linear(192, hidden_dim)
        self.memory_type = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.action_type = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.action_position = nn.Parameter(torch.randn(1, 16, hidden_dim) * 0.02)
        self.time_projection = nn.Linear(1, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.touch_film = nn.ModuleList(
            nn.Linear(tactile_dim, 2 * hidden_dim, bias=False) for _ in range(2)
        )
        self.output = nn.ModuleList(
            nn.Linear(hidden_dim, 192, bias=False) for _ in range(2)
        )
        # Start as the exact frozen Method-D plan while preserving a gradient to
        # the tactile path after the first optimiser update.
        for head in self.output:
            nn.init.zeros_(head.weight)
        component = torch.zeros(1, 2, 1, 192, dtype=torch.bool)
        groups = ARTICULATION_GROUPS if articulation_only else FULL_HAND_GROUPS
        for hand, (start, stop) in enumerate(groups):
            component[:, hand, :, start:stop] = True
        self.register_buffer("hand_component_mask", component, persistent=True)

    def forward(
        self,
        base_action: torch.Tensor,
        final_memory: torch.Tensor,
        tactile_features: torch.Tensor,
        hand_valid: torch.Tensor,
        decision_seconds: torch.Tensor,
        edit_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if tuple(base_action.shape[1:]) != (16, 192):
            raise ValueError("base_action must be [B,16,192]")
        if final_memory.shape[:2] != (len(base_action), 8):
            raise ValueError("final_memory must be [B,8,D]")
        if tactile_features.shape[:2] != (len(base_action), 2):
            raise ValueError("tactile_features must be [B,2,D]")
        if hand_valid.shape != (len(base_action), 2):
            raise ValueError("hand_valid must be [B,2]")
        dtype = self.action_projection.weight.dtype
        action = (base_action.detach().to(dtype) - self.action_mean.to(dtype))
        action = action / self.action_std.to(dtype)
        action = self.action_projection(action) + self.action_type + self.action_position
        action = action + self.time_projection(
            decision_seconds.to(dtype)[:, None, None] * 10.0
        )
        memory = self.memory_projection(
            self.memory_norm(final_memory.detach().to(dtype))
        ) + self.memory_type
        hidden = self.blocks(torch.cat((memory, action), dim=1))[:, memory.shape[1] :]
        per_hand = []
        for hand in range(2):
            scale, shift = self.touch_film[hand](
                tactile_features[:, hand].to(dtype)
            ).chunk(2, dim=-1)
            modulated = hidden * scale[:, None] + shift[:, None]
            per_hand.append(self.output[hand](modulated))
        per_hand_delta = torch.stack(per_hand, dim=1)
        allowed = edit_mask[:, None].bool() & self.hand_component_mask
        allowed &= hand_valid[:, :, None, None]
        delta = (per_hand_delta * allowed.to(per_hand_delta.dtype)).sum(1)
        return base_action.detach() + delta, delta, allowed.any(1)


@dataclass
class UnifiedWorldOutput:
    final_action: torch.Tensor
    delta_action: torch.Tensor
    tactile_features: torch.Tensor
    hand_valid: torch.Tensor
    allowed_mask: torch.Tensor
    tactile_velocity: torch.Tensor | None = None
    tactile_target: torch.Tensor | None = None
    tactile_valid: torch.Tensor | None = None
    rgb: RGBWorldResidualOutput | None = None


class UnifiedWorldTactileEditor(nn.Module):
    """W3 editor. Legacy world modes remain for checkpoint compatibility."""

    def __init__(
        self,
        tactile_encoder: DualHandTactileEncoder,
        *,
        world_mode: WorldMode | str,
        action_mean: torch.Tensor,
        action_std: torch.Tensor,
        pressure_masks: torch.Tensor,
        reference_pressure: torch.Tensor,
        pressure_scale: torch.Tensor,
        memory_dim: int = 960,
        tactile_dim: int = 64,
        hidden_dim: int = 256,
        condition_on_action: bool = True,
        world_condition_dim: int = 2304,
        articulation_only: bool = False,
    ) -> None:
        super().__init__()
        self.world_mode = WorldMode(world_mode)
        self.tactile = CenteredTactileFeatures(
            tactile_encoder, reference_pressure
        )
        self.action = DirectMemoryActionEditor(
            action_mean,
            action_std,
            memory_dim=memory_dim,
            tactile_dim=tactile_dim,
            hidden_dim=hidden_dim,
            articulation_only=articulation_only,
        )
        self.articulation_only = articulation_only
        self.target_codec = TactileTargetCodec(
            pressure_masks, reference_pressure, pressure_scale
        )
        self.tactile_world = (
            TactileWorldModel(
                memory_dim=memory_dim,
                tactile_dim=tactile_dim,
                hidden_dim=hidden_dim,
                action_tokens=8 if condition_on_action else 0,
            )
            if self.world_mode.uses_tactile
            else None
        )
        self.rgb_world = (
            FrozenRGBWorldResidual(
                tactile_dim=tactile_dim,
                memory_rows=8,
                condition_dim=world_condition_dim,
                hidden_dim=hidden_dim,
            )
            if self.world_mode.uses_rgb
            else None
        )
        self.condition_on_action = condition_on_action

    def forward(
        self,
        *,
        base_action: torch.Tensor,
        edit_mask: torch.Tensor,
        pressure: torch.Tensor,
        temporal_valid: torch.Tensor,
        decision_seconds: torch.Tensor,
        final_memory: torch.Tensor,
        future_pressure: torch.Tensor | None = None,
        future_valid: torch.Tensor | None = None,
        tactile_tau: torch.Tensor | None = None,
        tactile_noise: torch.Tensor | None = None,
        rgb_world_head=None,
        image_tokens: torch.Tensor | None = None,
        clean_latent: torch.Tensor | None = None,
        rgb_valid: torch.Tensor | None = None,
        rgb_tau: torch.Tensor | None = None,
        rgb_noise: torch.Tensor | None = None,
        compute_auxiliary: bool = True,
        compute_rgb: bool = True,
        compute_tactile: bool = True,
    ) -> UnifiedWorldOutput:
        tactile_features, hand_valid = self.tactile(pressure, temporal_valid)
        final_action, delta, allowed = self.action(
            base_action,
            final_memory,
            tactile_features,
            hand_valid,
            decision_seconds,
            edit_mask,
        )
        tactile_velocity = tactile_target = tactile_valid = None
        if self.tactile_world is not None and compute_auxiliary and compute_tactile:
            if future_pressure is None or future_valid is None or tactile_tau is None:
                raise ValueError("W2/W3 require future tactile targets and tau")
            clean, tactile_valid = self.target_codec.encode(
                future_pressure, future_valid
            )
            if tactile_noise is None:
                tactile_noise = torch.randn_like(clean)
            noisy, tactile_target = tactile_flow_pair(
                clean, tactile_valid, tactile_tau, tactile_noise
            )
            if self.condition_on_action:
                action_window, action_valid = gather_action_window(
                    final_action, decision_seconds, steps=8
                )
            else:
                action_window = action_valid = None
            tactile_velocity = self.tactile_world.velocity(
                noisy,
                tactile_tau,
                final_memory=final_memory.detach(),
                tactile_features=tactile_features,
                hand_valid=hand_valid,
                action_window=action_window,
                action_valid=action_valid,
            )

        rgb = None
        if self.rgb_world is not None and compute_auxiliary and compute_rgb:
            values = (
                rgb_world_head,
                image_tokens,
                clean_latent,
                rgb_valid,
                rgb_tau,
                rgb_noise,
            )
            if any(value is None for value in values):
                raise ValueError("W1/W3 require all cached RGB-world inputs")
            rgb = self.rgb_world(
                world_head=rgb_world_head,
                image_tokens=image_tokens,
                final_memory=final_memory.detach(),
                tactile_features=tactile_features,
                hand_valid=hand_valid,
                clean_latent=clean_latent,
                world_valid=rgb_valid,
                tau=rgb_tau,
                noise=rgb_noise,
            )
        return UnifiedWorldOutput(
            final_action=final_action,
            delta_action=delta,
            tactile_features=tactile_features,
            hand_valid=hand_valid,
            allowed_mask=allowed,
            tactile_velocity=tactile_velocity,
            tactile_target=tactile_target,
            tactile_valid=tactile_valid,
            rgb=rgb,
        )

    def objective(
        self,
        output: UnifiedWorldOutput,
        target_action: torch.Tensor,
        edit_mask: torch.Tensor,
        *,
        rgb_roi: torch.Tensor | None,
        lambda_rgb: float,
        lambda_tactile: float,
        lambda_delta: float = 1e-4,
        rgb_background_weight: float = 0.1,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if min(lambda_rgb, lambda_tactile, lambda_delta) < 0:
            raise ValueError("loss weights must be non-negative")
        mask = edit_mask.bool() & output.allowed_mask
        present = mask.any((1, 2))
        if not present.any():
            raise ValueError("batch has no tactile-editable action target")
        action_loss = per_sample_action_loss(
            output.final_action[present],
            target_action[present],
            mask[present],
            groups=(ARTICULATION_GROUPS if self.articulation_only else FULL_ACTION_GROUPS),
        ).mean()
        delta_l2 = output.delta_action.float().square().masked_fill(~mask, 0).sum()
        delta_l2 = delta_l2 / mask.sum().clamp_min(1)
        zero = action_loss.new_zeros(())
        tactile_world_loss = zero
        if output.tactile_velocity is not None:
            assert output.tactile_target is not None and output.tactile_valid is not None
            tactile_world_loss = masked_tactile_flow_loss(
                output.tactile_velocity,
                output.tactile_target,
                output.tactile_valid,
            )
        elif lambda_tactile:
            raise ValueError("lambda_tactile is nonzero without a tactile WM")
        rgb_world_loss = zero
        if output.rgb is not None:
            if rgb_roi is None:
                raise ValueError("RGB world loss requires ROI")
            rgb_world_loss = roi_weighted_world_loss(
                output.rgb, rgb_roi, background_weight=rgb_background_weight
            )
        elif lambda_rgb:
            raise ValueError("lambda_rgb is nonzero without an RGB WM")
        total = (
            action_loss
            + lambda_rgb * rgb_world_loss
            + lambda_tactile * tactile_world_loss
            + lambda_delta * delta_l2
        )
        return total, {
            "loss": total.detach(),
            "action_loss": action_loss.detach(),
            "rgb_world_loss": rgb_world_loss.detach(),
            "tactile_world_loss": tactile_world_loss.detach(),
            "delta_l2": delta_l2.detach(),
        }

    def optimizer_groups(
        self,
        *,
        action_lr: float = 1e-4,
        encoder_lr: float = 1e-5,
        world_adapter_lr: float = 1e-4,
        world_block_lr: float = 1e-5,
    ) -> list[dict]:
        groups = [
            {"name": "action", "params": list(self.action.parameters()), "lr": action_lr},
            {
                "name": "tactile.encoder",
                "params": [
                    parameter
                    for parameter in self.tactile.encoder.parameters()
                    if parameter.requires_grad
                ],
                "lr": encoder_lr,
            },
        ]
        if self.tactile_world is not None:
            groups += [
                {
                    "name": "tactile_world.adapter",
                    "params": list(self.tactile_world.adapter.parameters()),
                    "lr": world_adapter_lr,
                },
                {
                    "name": "tactile_world.blocks",
                    "params": [
                        parameter
                        for name, parameter in self.tactile_world.named_parameters()
                        if not name.startswith("adapter.")
                    ],
                    "lr": world_block_lr,
                },
            ]
        if self.rgb_world is not None:
            groups.append(
                {
                    "name": "rgb_world.touch_adapter",
                    "params": list(self.rgb_world.parameters()),
                    "lr": world_adapter_lr,
                }
            )
        groups = [
            {**group, "params": [p for p in group["params"] if p.requires_grad]}
            for group in groups
        ]
        groups = [group for group in groups if group["params"]]
        ids = [id(p) for group in groups for p in group["params"]]
        if len(ids) != len(set(ids)):
            raise RuntimeError("optimizer parameter groups overlap")
        expected = {id(p) for p in self.parameters() if p.requires_grad}
        if set(ids) != expected:
            raise RuntimeError("optimizer groups do not cover all trainable parameters")
        return groups


def ramp_weight(step: int, maximum: float, warmup_steps: int) -> float:
    if step < 0 or maximum < 0 or warmup_steps < 0:
        raise ValueError("weight schedule arguments must be non-negative")
    return float(maximum) if warmup_steps == 0 else float(maximum) * min(
        1.0, step / warmup_steps
    )
