"""Frozen-RGB-world regularization for a Stage-2 tactile action editor.

The RGB world model is a teacher and is never stored inside this module.  It predicts
rectified-flow velocity from the current image tokens and Method-D memory.  A
reference-centred tactile history may perturb only the eight memory-condition rows;
the frozen SANA then turns that perturbation into a latent-velocity residual.

The action path is equally strict: action/memory features create residual bases and
per-hand tactile features create their coefficients.  There is no action-only Stage-2
residual path, so reference tactile is an exact frozen Stage-1 identity.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from .action_objectives import per_sample_action_loss
from .action_layout import ARTICULATION_GROUPS, FULL_ACTION_GROUPS, FULL_HAND_GROUPS
from .direct_residual import DirectTactileTrunk
from .single_stage_world import MemoryConditionedActionBasis
from .training import ACTION_GROUPS


def _check_shape(value: torch.Tensor, expected: tuple[int | None, ...], name: str) -> None:
    if value.ndim != len(expected):
        raise ValueError(f"{name} must have {len(expected)} dimensions")
    for actual, wanted in zip(value.shape, expected):
        if wanted is not None and actual != wanted:
            raise ValueError(f"{name} must have shape {expected}, got {tuple(value.shape)}")


class TactileMemoryConditionModulator(nn.Module):
    """Map two hand-local tactile features to the eight W0 memory rows.

    Both projections are bias-free and the output layers are zero-initialized.  Thus a
    zero/reference tactile feature produces an exact zero condition delta, including at
    initialization and after training.
    """

    def __init__(self, tactile_dim: int = 128, memory_rows: int = 8,
                 condition_dim: int = 2304, hidden_dim: int = 256) -> None:
        super().__init__()
        self.memory_rows = memory_rows
        self.condition_dim = condition_dim
        self.input = nn.ModuleList(
            nn.Linear(tactile_dim, hidden_dim, bias=False) for _ in range(2)
        )
        self.output = nn.ModuleList(
            nn.Linear(hidden_dim, memory_rows * condition_dim, bias=False)
            for _ in range(2)
        )
        for layer in self.output:
            nn.init.zeros_(layer.weight)

    def forward(self, tactile_features: torch.Tensor,
                hand_valid: torch.Tensor) -> torch.Tensor:
        _check_shape(tactile_features, (None, 2, None), "tactile_features")
        if hand_valid.shape != tactile_features.shape[:2]:
            raise ValueError("hand_valid must be [B,2]")
        pieces = []
        for hand in range(2):
            feature = tactile_features[:, hand]
            hidden = torch.nn.functional.gelu(self.input[hand](feature))
            delta = self.output[hand](hidden).reshape(
                len(feature), self.memory_rows, self.condition_dim
            )
            pieces.append(delta * hand_valid[:, hand, None, None].to(delta.dtype))
        return pieces[0] + pieces[1]


@dataclass
class RGBWorldResidualOutput:
    predicted_residual: torch.Tensor
    target_residual: torch.Tensor
    baseline_velocity: torch.Tensor
    tactile_velocity: torch.Tensor
    valid: torch.Tensor


class FrozenRGBWorldResidual(nn.Module):
    """Trainable tactile condition in front of a caller-owned, frozen RGB W0."""

    def __init__(self, tactile_dim: int = 128, memory_rows: int = 8,
                 condition_dim: int = 2304, hidden_dim: int = 256) -> None:
        super().__init__()
        self.modulator = TactileMemoryConditionModulator(
            tactile_dim=tactile_dim,
            memory_rows=memory_rows,
            condition_dim=condition_dim,
            hidden_dim=hidden_dim,
        )

    def forward(
        self,
        *,
        world_head,
        image_tokens: torch.Tensor,
        final_memory: torch.Tensor,
        tactile_features: torch.Tensor,
        hand_valid: torch.Tensor,
        clean_latent: torch.Tensor,
        world_valid: torch.Tensor,
        tau: torch.Tensor,
        noise: torch.Tensor,
    ) -> RGBWorldResidualOutput:
        _check_shape(image_tokens, (None, None, None), "image_tokens")
        _check_shape(final_memory, (len(image_tokens), 8, None), "final_memory")
        _check_shape(clean_latent, (len(image_tokens), None, None, None, None), "clean_latent")
        b, horizons = clean_latent.shape[:2]
        if world_valid.shape != (b, horizons):
            raise ValueError("world_valid must be [B,H]")
        if tau.shape != (b, horizons):
            raise ValueError("tau must be [B,H]")
        if noise.shape != clean_latent.shape:
            raise ValueError("noise must match clean_latent")

        # Replace invalid values before arithmetic: NaN * 0 is still NaN.
        valid5 = world_valid[..., None, None, None]
        clean = torch.where(valid5, clean_latent.float(), torch.zeros_like(clean_latent.float()))
        eps = torch.where(valid5, noise.float(), torch.zeros_like(noise.float()))
        mixed = (1.0 - tau[..., None, None, None].float()) * clean
        mixed = mixed + tau[..., None, None, None].float() * eps
        target_velocity = eps - clean

        adapter_dtype = next(world_head.adapter.parameters()).dtype
        base_condition = world_head.build_condition(
            image_tokens.to(adapter_dtype), final_memory.to(adapter_dtype)
        )
        memory_rows = self.modulator.memory_rows
        if base_condition.shape[1] < memory_rows:
            raise ValueError("world condition has fewer rows than the memory modulator")
        delta = self.modulator(tactile_features, hand_valid).to(base_condition.dtype)
        tactile_condition = base_condition.clone()
        tactile_condition[:, -memory_rows:] = tactile_condition[:, -memory_rows:] + delta

        adapter_horizons = getattr(world_head.adapter.cfg, "num_horizons", 1)
        base_rows, tactile_rows = [], []
        for horizon in range(horizons):
            if adapter_horizons == horizons:
                base_rows.append(world_head.adapter.for_horizon(base_condition, horizon))
                tactile_rows.append(world_head.adapter.for_horizon(tactile_condition, horizon))
            elif adapter_horizons == 1:
                base_rows.append(base_condition)
                tactile_rows.append(tactile_condition)
            else:
                raise ValueError(
                    "world adapter horizon count must be one or match clean_latent"
                )
        base_flat = torch.cat(base_rows, dim=0)
        tactile_flat = torch.cat(tactile_rows, dim=0)

        # Horizon-major flattening is required by W0's modulation horizon embedding.
        flatten = lambda value: value.transpose(0, 1).reshape(  # noqa: E731
            horizons * b, *value.shape[2:]
        )
        mixed_flat = flatten(mixed).to(base_flat.dtype)
        tau_flat = tau.transpose(0, 1).reshape(horizons * b).to(base_flat.dtype)

        with torch.no_grad():
            baseline = world_head.predict_velocity(mixed_flat, tau_flat, base_flat)
        # W0 parameters are frozen, but this call keeps autograd with respect to the
        # tactile-conditioned input.
        tactile_velocity = world_head.predict_velocity(
            mixed_flat, tau_flat, tactile_flat
        )
        target_flat = flatten(target_velocity)
        predicted_residual = tactile_velocity.float() - baseline.float().detach()
        target_residual = target_flat.float() - baseline.float().detach()
        return RGBWorldResidualOutput(
            predicted_residual=predicted_residual,
            target_residual=target_residual.detach(),
            baseline_velocity=baseline.detach(),
            tactile_velocity=tactile_velocity,
            valid=world_valid.transpose(0, 1).reshape(horizons * b),
        )


def roi_weighted_world_loss(output: RGBWorldResidualOutput,
                            roi: torch.Tensor,
                            *, background_weight: float = 0.1) -> torch.Tensor:
    """MSE on latent velocity with hand ROI weight 1 and background weight 0.1."""
    if not 0 <= background_weight <= 1:
        raise ValueError("background_weight must be in [0,1]")
    if roi.ndim != 4:
        raise ValueError("roi must be [B,H,Hlatent,Wlatent]")
    flat_roi = roi.transpose(0, 1).reshape(-1, *roi.shape[2:]).bool()
    if flat_roi.shape[0] != output.predicted_residual.shape[0]:
        raise ValueError("roi rows must match world residual rows")
    if flat_roi.shape[1:] != output.predicted_residual.shape[2:]:
        raise ValueError("roi spatial shape must match world latent")
    valid = output.valid.bool()
    if not valid.any():
        raise ValueError("the batch has no valid RGB world targets")
    weight = torch.full_like(flat_roi, background_weight, dtype=torch.float32)
    weight = torch.where(flat_roi, torch.ones_like(weight), weight)
    error = (output.predicted_residual - output.target_residual).square()
    weighted = error * weight[:, None].to(error.device)
    per_row = weighted.flatten(1).sum(1) / (
        weight.flatten(1).sum(1).to(error.device) * error.shape[1]
    ).clamp_min(1)
    return per_row[valid].mean()


@dataclass
class Stage2Output:
    final_action: torch.Tensor
    delta_action: torch.Tensor
    coefficients: torch.Tensor
    tactile_features: torch.Tensor
    tactile_history_features: torch.Tensor
    hand_valid: torch.Tensor
    allowed_mask: torch.Tensor
    world: RGBWorldResidualOutput | None


class RGBWorldRegularizedTactileEditor(nn.Module):
    """Strict per-hand Stage-2 action editor with an RGB-world auxiliary branch."""

    def __init__(
        self,
        *,
        action_mean: torch.Tensor,
        action_std: torch.Tensor,
        pressure_masks: torch.Tensor,
        reference_pressure: torch.Tensor,
        residual_bases: int = 8,
        articulation_only: bool = False,
        memory_dim: int = 960,
        hidden_dim: int = 256,
        world_condition_dim: int = 2304,
    ) -> None:
        super().__init__()
        self.tactile = DirectTactileTrunk(
            pressure_masks,
            reference_pressure,
            mode="history_delta",
            coefficient_scope="hand_local",
            residual_bases=residual_bases,
        )
        self.action = MemoryConditionedActionBasis(
            action_mean,
            action_std,
            memory_dim=memory_dim,
            hidden_dim=hidden_dim,
            residual_bases=2 * residual_bases,
        )
        self.rgb_world = FrozenRGBWorldResidual(
            tactile_dim=128,
            memory_rows=8,
            condition_dim=world_condition_dim,
            hidden_dim=hidden_dim,
        )
        self.residual_bases = residual_bases
        self.articulation_only = articulation_only
        component = torch.zeros(1, 2, 1, 192, dtype=torch.bool)
        groups = (
            ARTICULATION_GROUPS if articulation_only else FULL_HAND_GROUPS
        )
        for hand, (start, stop) in enumerate(groups):
            component[:, hand, :, start:stop] = True
        self.register_buffer("hand_component_mask", component, persistent=True)

    def forward(
        self,
        *,
        anchor_action: torch.Tensor,
        edit_mask: torch.Tensor,
        pressure: torch.Tensor,
        temporal_valid: torch.Tensor,
        decision_seconds: torch.Tensor,
        final_memory: torch.Tensor,
        world_head=None,
        image_tokens: torch.Tensor | None = None,
        clean_latent: torch.Tensor | None = None,
        world_valid: torch.Tensor | None = None,
        tau: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ) -> Stage2Output:
        coefficients, hand_valid, tactile_features, tactile_history = self.tactile(
            pressure, temporal_valid
        )
        all_bases = self.action(anchor_action, final_memory, decision_seconds)
        bases = all_bases.reshape(
            len(anchor_action), 2, self.residual_bases, 16, 192
        )
        per_hand = (coefficients[..., None, None] * bases).sum(2) / math.sqrt(
            self.residual_bases
        )
        hand_allowed = edit_mask[:, None].bool() & self.hand_component_mask
        hand_allowed = hand_allowed & hand_valid[:, :, None, None]
        delta = (per_hand * hand_allowed.to(per_hand.dtype)).sum(1)
        final_action = anchor_action.detach() + delta

        supplied = (
            world_head is not None,
            image_tokens is not None,
            clean_latent is not None,
            world_valid is not None,
            tau is not None,
            noise is not None,
        )
        if any(supplied) and not all(supplied):
            raise ValueError("all RGB world inputs must be supplied together")
        world = None
        if all(supplied):
            world = self.rgb_world(
                world_head=world_head,
                image_tokens=image_tokens,
                final_memory=final_memory.detach(),
                tactile_features=tactile_features,
                hand_valid=hand_valid,
                clean_latent=clean_latent,
                world_valid=world_valid,
                tau=tau,
                noise=noise,
            )
        return Stage2Output(
            final_action=final_action,
            delta_action=delta,
            coefficients=coefficients,
            tactile_features=tactile_features,
            tactile_history_features=tactile_history,
            hand_valid=hand_valid,
            allowed_mask=hand_allowed.any(1),
            world=world,
        )

    def objective(
        self,
        output: Stage2Output,
        target_action: torch.Tensor,
        edit_mask: torch.Tensor,
        *,
        roi: torch.Tensor | None,
        lambda_world: float,
        lambda_delta: float = 1e-4,
        lambda_coefficient: float = 1e-5,
        background_weight: float = 0.1,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if min(lambda_world, lambda_delta, lambda_coefficient) < 0:
            raise ValueError("loss weights must be non-negative")
        mask = edit_mask.bool() & output.allowed_mask
        present = mask.any((1, 2))
        if not present.any():
            raise ValueError("the batch has no tactile-editable action target")
        groups = ARTICULATION_GROUPS if self.articulation_only else FULL_ACTION_GROUPS
        action = per_sample_action_loss(
            output.final_action[present], target_action[present], mask[present], groups=groups
        ).mean()
        delta_l2 = output.delta_action.float().square().masked_fill(~mask, 0).sum()
        delta_l2 = delta_l2 / mask.sum().clamp_min(1)
        coefficient_l2 = output.coefficients.float().square().mean()
        if lambda_world:
            if output.world is None or roi is None:
                raise ValueError("RGB world output and roi are required")
            world = roi_weighted_world_loss(
                output.world, roi, background_weight=background_weight
            )
        else:
            world = action.new_zeros(())
        total = action + lambda_world * world
        total = total + lambda_delta * delta_l2 + lambda_coefficient * coefficient_l2
        return total, {
            "loss": total.detach(),
            "action_loss": action.detach(),
            "world_loss": world.detach(),
            "delta_l2": delta_l2.detach(),
            "coefficient_l2": coefficient_l2.detach(),
        }

    def optimizer_groups(
        self,
        *,
        action_lr: float = 1e-4,
        tactile_gru_lr: float = 1e-5,
        tactile_cnn_lr: float = 3e-6,
        world_modulator_lr: float = 1e-4,
    ) -> list[dict]:
        groups = [
            {
                "name": "action",
                "params": [
                    *self.action.parameters(),
                    *self.tactile.coefficient_head.parameters(),
                ],
                "lr": action_lr,
            },
            {
                "name": "tactile.cnn",
                "params": list(self.tactile.frame_encoder.parameters()),
                "lr": tactile_cnn_lr,
            },
            {
                "name": "world.modulator",
                "params": list(self.rgb_world.parameters()),
                "lr": world_modulator_lr,
            },
        ]
        if self.tactile.temporal is not None:
            groups.append(
                {
                    "name": "tactile.gru",
                    "params": list(self.tactile.temporal.parameters()),
                    "lr": tactile_gru_lr,
                }
            )
        return groups


def ramp_world_weight(step: int, maximum: float, warmup_steps: int) -> float:
    if maximum < 0 or warmup_steps < 0:
        raise ValueError("maximum and warmup_steps must be non-negative")
    if warmup_steps == 0:
        return float(maximum)
    return float(maximum) * min(1.0, max(0.0, step / warmup_steps))
