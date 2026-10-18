"""V1 training primitives derived from the Kai-final VLA timebase."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .action_layout import FULL_ACTION_GROUPS
from typing import Any

# The offline losses do not import or construct a foundation model.
TactileMethodD = Any


# Backwards-compatible name used by existing objectives and scripts.
ACTION_GROUPS = FULL_ACTION_GROUPS


def balanced_masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    groups: tuple[tuple[int, int], ...] = ACTION_GROUPS,
) -> torch.Tensor:
    """Mean over values per group, present groups per sample, then valid samples.

    Empty groups/samples are omitted; an entirely empty batch is an error.
    Invalid values are selected away before reduction (including NaN padding).
    """
    if values.ndim != 3 or mask.shape != values.shape:
        raise ValueError("values and mask must have matching [B,T,A] shapes")
    occupied: set[int] = set()
    means, present = [], []
    for start, end in groups:
        if not 0 <= start < end <= values.shape[-1]:
            raise ValueError("action group outside the action dimension")
        indices = set(range(start, end))
        if occupied & indices:
            raise ValueError("action groups must not overlap")
        occupied |= indices
        selected = mask[..., start:end].bool()
        count = selected.sum((1, 2))
        means.append(
            values[..., start:end].masked_fill(~selected, 0).sum((1, 2))
            / count.clamp_min(1)
        )
        present.append(count > 0)
    if not groups:
        raise ValueError("at least one action group is required")
    valid = torch.stack(present, dim=1)
    per_sample = torch.stack(means, dim=1).sum(1) / valid.sum(1).clamp_min(1)
    nonempty = valid.any(1)
    if not nonempty.any():
        raise ValueError("the batch has no editable action values")
    return per_sample[nonempty].mean()


@dataclass(frozen=True)
class V1TrainingBatch:
    """One cached Method-D plan and its synchronized tactile supervision."""

    base_action: torch.Tensor  # [B,16,192], normalized frozen-VLA output
    target_action: torch.Tensor  # [B,16,192], normalized ground truth
    action_mask: torch.Tensor  # [B,16,192]
    touch_pressure: torch.Tensor  # [B,8,2,21,21]
    touch_valid: torch.Tensor  # [B,8,2]
    decision_after_start_seconds: torch.Tensor  # [B], offline plan-relative time

    def validate(self) -> None:
        if self.base_action.ndim != 3:
            raise ValueError("base_action must be [B,T,A]")
        if self.target_action.shape != self.base_action.shape:
            raise ValueError("target_action and base_action must have the same shape")
        if self.action_mask.shape != self.base_action.shape:
            raise ValueError("action_mask and base_action must have the same shape")
        batch = len(self.base_action)
        if self.touch_pressure.shape != (batch, 8, 2, 21, 21):
            raise ValueError("touch_pressure must be [B,8,2,21,21]")
        if self.touch_valid.shape != (batch, 8, 2):
            raise ValueError("touch_valid must be [B,8,2]")
        if self.decision_after_start_seconds.shape != (batch,):
            raise ValueError("decision_after_start_seconds must be [B]")


def build_future_action_mask(
    action_mask: torch.Tensor,
    decision_after_start_seconds: torch.Tensor,
    *,
    action_hz: float = 80.0,
) -> torch.Tensor:
    """Offline mask at the tactile decision time; no compute-latency model."""

    if action_mask.ndim != 3:
        raise ValueError("action_mask must be [B,T,A]")
    if decision_after_start_seconds.shape != (len(action_mask),):
        raise ValueError("decision_after_start_seconds must be [B]")
    if not math.isfinite(action_hz) or action_hz <= 0:
        raise ValueError("action_hz must be finite and positive")
    ready = decision_after_start_seconds.to(
        device=action_mask.device, dtype=torch.float64
    )
    if not torch.isfinite(ready).all() or (ready < 0).any():
        raise ValueError("decision times must be finite and non-negative")
    # Float32 seconds can put an exact tick a few ulps above its integer index.
    tolerance = 1e-6 if decision_after_start_seconds.dtype != torch.float64 else 1e-9
    committed = torch.ceil(ready * action_hz - tolerance).long()
    committed = committed.clamp(max=action_mask.shape[1])
    steps = torch.arange(action_mask.shape[1], device=action_mask.device)[None, :, None]
    return action_mask.to(dtype=torch.bool) & (steps >= committed[:, None, None])


def v1_residual_objective(
    model: TactileMethodD,
    batch: V1TrainingBatch,
    *,
    action_hz: float = 80.0,
    delta_l2_weight: float = 1e-4,
    action_groups: tuple[tuple[int, int], ...] = ACTION_GROUPS,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Train V1 on only the action positions that deployment can still change."""

    batch.validate()
    if delta_l2_weight < 0:
        raise ValueError("delta_l2_weight must be non-negative")
    edit_mask = build_future_action_mask(
        batch.action_mask,
        batch.decision_after_start_seconds,
        action_hz=action_hz,
    )
    # Exclude all dimensions outside the specified physical groups, even if the
    # supplied mask accidentally marks padding as valid.
    component_mask = torch.zeros_like(edit_mask)
    for start, end in action_groups:
        if not 0 <= start < end <= edit_mask.shape[-1]:
            raise ValueError("action group outside the action dimension")
        component_mask[..., start:end] = True
    edit_mask &= component_mask
    base = batch.base_action.detach()
    target = batch.target_action.detach()
    if not torch.isfinite(base).all():
        raise ValueError("base_action must be finite, including padding")
    if not torch.isfinite(target[edit_mask]).all():
        raise ValueError("editable target_action must be finite")
    # Invalid GT can contain NaN; remove it before Smooth L1 so backward is safe.
    target = torch.where(edit_mask, target, base)
    revised = model.edit_existing_plan(
        base,
        edit_mask,
        batch.touch_pressure,
        batch.touch_valid,
        canonicalized_touch=False,
    )
    per_value = F.smooth_l1_loss(revised, target, reduction="none")
    action_loss = balanced_masked_mean(per_value, edit_mask, action_groups)
    delta = revised - base
    delta_l2 = balanced_masked_mean(delta.square(), edit_mask, action_groups)
    loss = action_loss + delta_l2_weight * delta_l2
    with torch.no_grad():
        base_error = balanced_masked_mean(
            F.smooth_l1_loss(base, target, reduction="none"), edit_mask, action_groups
        )
        metrics = {
            "loss": loss.detach(),
            "action_loss": action_loss.detach(),
            "delta_l2": delta_l2.detach(),
            "base_action_loss": base_error.detach(),
            "editable_fraction": edit_mask.float().mean().detach(),
            "supervised_samples": edit_mask.any(dim=(1, 2)).sum().detach(),
        }
    return loss, metrics


def nominal_revision_times(
    *, tactile_hz: float = 30.0, control_hz: float = 10.0
) -> tuple[float, ...]:
    """Return tactile decision times inside one VLA control period."""

    if tactile_hz <= 0 or control_hz <= 0:
        raise ValueError("rates must be positive")
    count = max(1, math.ceil(tactile_hz / control_hz))
    return tuple(index / tactile_hz for index in range(count))
