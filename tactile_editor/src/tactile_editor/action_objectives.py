"""Group-balanced action objectives for tactile-editor experiments.

All public loss functions return one value per input sample.  Action objectives
require every sample to contain at least one supervised value.  Motion
objectives also return a validity mask because an editable suffix can contain no
adjacent valid pair.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch.nn import functional as F

from .training import ACTION_GROUPS


ActionGroups = Sequence[tuple[int, int]]


def _validate_action_tensors(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if prediction.ndim != 3:
        raise ValueError("prediction must be [B,T,A]")
    if target.shape != prediction.shape or mask.shape != prediction.shape:
        raise ValueError("prediction, target, and mask must have matching [B,T,A] shapes")
    if not prediction.is_floating_point() or not target.is_floating_point():
        raise TypeError("prediction and target must be floating-point tensors")
    if prediction.device != target.device or prediction.device != mask.device:
        raise ValueError("prediction, target, and mask must be on the same device")
    return mask.bool()


def _validate_groups(groups: ActionGroups, action_dim: int) -> tuple[tuple[int, int], ...]:
    groups = tuple(groups)
    if not groups:
        raise ValueError("at least one action group is required")
    occupied: set[int] = set()
    for start, stop in groups:
        if not 0 <= start < stop <= action_dim:
            raise ValueError("action group outside the action dimension")
        indices = set(range(start, stop))
        if occupied & indices:
            raise ValueError("action groups must not overlap")
        occupied |= indices
    return groups


def _validate_timestep_weights(
    timestep_weights: torch.Tensor,
    prediction: torch.Tensor,
) -> torch.Tensor:
    expected = (*prediction.shape[:2], 1)
    if tuple(timestep_weights.shape) != expected:
        raise ValueError(f"timestep_weights must have shape {expected}")
    weights = timestep_weights.to(device=prediction.device, dtype=prediction.dtype)
    if not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("timestep_weights must be finite and non-negative")
    return weights


def _per_sample_group_balanced_smooth_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    groups: ActionGroups,
    timestep_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return group-balanced losses and a mask of nonempty samples."""

    mask = _validate_action_tensors(prediction, target, mask)
    groups = _validate_groups(groups, prediction.shape[-1])
    weights = (
        None
        if timestep_weights is None
        else _validate_timestep_weights(timestep_weights, prediction)
    )

    # Cached padded targets can contain NaN.  Select a finite, detached value
    # before Smooth L1 rather than masking a NaN loss after arithmetic.
    safe_target = torch.where(mask, target, prediction.detach())
    error = F.smooth_l1_loss(prediction, safe_target, reduction="none")
    group_values, group_present = [], []
    for start, stop in groups:
        selected = mask[..., start:stop]
        if weights is None:
            denominator = selected.sum((1, 2))
            numerator = error[..., start:stop].masked_fill(~selected, 0).sum((1, 2))
        else:
            # weights is [B,T,1] and broadcasts over the selected action group.
            weighted_mask = weights * selected.to(dtype=error.dtype)
            denominator = weighted_mask.sum((1, 2))
            numerator = (error[..., start:stop] * weighted_mask).sum((1, 2))
        # A normalized exponential weight sum can be below one (for example,
        # one far-future scalar).  Clamp only for numerical safety; clamping to
        # one would silently shrink such a group's loss.
        group_values.append(
            numerator / denominator.clamp_min(torch.finfo(error.dtype).tiny)
        )
        group_present.append(denominator > 0)

    present = torch.stack(group_present, dim=1)
    per_sample = torch.stack(group_values, dim=1).sum(1) / present.sum(1).clamp_min(1)
    valid = present.any(1)
    return per_sample.masked_fill(~valid, 0), valid


def per_sample_action_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    groups: ActionGroups = ACTION_GROUPS,
) -> torch.Tensor:
    """Return the existing group-balanced Smooth L1 action loss as ``[B]``."""

    values, valid = _per_sample_group_balanced_smooth_l1(
        prediction, target, mask, groups=groups
    )
    if not valid.all():
        raise ValueError("every action-loss sample must contain supervised actions")
    return values


def near_horizon_weights(
    decision_seconds: torch.Tensor,
    horizon: int,
    *,
    tau_seconds: float,
    action_hz: float = 80.0,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return ``exp(-(t_k-t_decision)/tau)`` weights shaped ``[B,T,1]``.

    Negative delays are clamped to zero.  A future/edit mask must still be
    supplied to the loss; this function does not decide which actions are
    committed.
    """

    if decision_seconds.ndim != 1:
        raise ValueError("decision_seconds must be [B]")
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if not math.isfinite(tau_seconds) or tau_seconds <= 0:
        raise ValueError("tau_seconds must be finite and positive")
    if not math.isfinite(action_hz) or action_hz <= 0:
        raise ValueError("action_hz must be finite and positive")
    decision = decision_seconds.to(dtype=torch.float64)
    if not torch.isfinite(decision).all() or (decision < 0).any():
        raise ValueError("decision_seconds must be finite and non-negative")
    ticks = torch.arange(
        horizon, device=decision.device, dtype=torch.float64
    )[None, :, None] / action_hz
    delay = (ticks - decision[:, None, None]).clamp_min(0)
    return torch.exp(-delay / tau_seconds).to(dtype=dtype)


def per_sample_near_horizon_action_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    decision_seconds: torch.Tensor,
    *,
    tau_seconds: float,
    action_hz: float = 80.0,
    groups: ActionGroups = ACTION_GROUPS,
) -> torch.Tensor:
    """Return near-horizon, group-balanced action losses as ``[B]``.

    Each group is divided by its sum of valid temporal weights, so changing
    ``tau_seconds`` changes temporal emphasis without changing the loss scale.
    """

    weights = near_horizon_weights(
        decision_seconds,
        prediction.shape[1],
        tau_seconds=tau_seconds,
        action_hz=action_hz,
        dtype=prediction.dtype,
    )
    if weights.device != prediction.device:
        weights = weights.to(prediction.device)
    values, valid = _per_sample_group_balanced_smooth_l1(
        prediction,
        target,
        mask,
        groups=groups,
        timestep_weights=weights,
    )
    if not valid.all():
        raise ValueError("every near-horizon sample must contain supervised actions")
    return values


def per_sample_motion_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    groups: ActionGroups = ACTION_GROUPS,
    timestep_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return adjacent-action motion losses and valid samples, both shaped ``[B]``.

    A transition is supervised only when both endpoints are in ``mask``.  With a
    future edit mask this excludes the committed-to-first-editable boundary.
    Optional ``timestep_weights`` has shape ``[B,T,1]``; transition ``k-1 -> k``
    uses the weight at its later endpoint ``k``.
    """

    mask = _validate_action_tensors(prediction, target, mask)
    weights = (
        None
        if timestep_weights is None
        else _validate_timestep_weights(timestep_weights, prediction)[:, 1:]
    )
    safe_target = torch.where(mask, target, prediction.detach())
    predicted_motion = prediction[:, 1:] - prediction[:, :-1]
    target_motion = safe_target[:, 1:] - safe_target[:, :-1]
    pair_mask = mask[:, 1:] & mask[:, :-1]
    return _per_sample_group_balanced_smooth_l1(
        predicted_motion,
        target_motion,
        pair_mask,
        groups=groups,
        timestep_weights=weights,
    )
