from __future__ import annotations

import math

import pytest
import torch

from tactile_editor.action_objectives import (
    near_horizon_weights,
    per_sample_action_loss,
    per_sample_motion_loss,
    per_sample_near_horizon_action_loss,
)
from tactile_editor.action_layout import FULL_ACTION_GROUPS


def existing_action_loss(prediction, target, mask):
    """Independent reference from the original group-balanced study loss."""
    safe_target = torch.where(mask, target, prediction.detach())
    error = torch.nn.functional.smooth_l1_loss(prediction, safe_target, reduction="none")
    values, present = [], []
    for lo, hi in FULL_ACTION_GROUPS:
        count = mask[..., lo:hi].sum((1, 2))
        values.append(error[..., lo:hi].masked_fill(~mask[..., lo:hi], 0).sum((1, 2))
                      / count.clamp_min(1))
        present.append(count > 0)
    return torch.stack(values, 1).sum(1) / torch.stack(present, 1).sum(1)


def test_action_loss_matches_existing_group_balanced_helper_and_ignores_nan_padding():
    generator = torch.Generator().manual_seed(17)
    prediction = torch.randn(3, 4, 12, generator=generator)
    target = torch.randn(3, 4, 12, generator=generator)
    mask = torch.ones_like(prediction, dtype=torch.bool)
    mask[0, -1, 3:] = False
    target[~mask] = torch.nan
    groups = ((0, 3), (3, 6), (6, 12))

    actual = per_sample_action_loss(prediction, target, mask, groups=groups)
    expected = existing_action_loss(prediction, target, mask)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.isfinite(actual).all()


def test_near_horizon_weights_use_physical_delay_from_decision_time():
    decision = torch.tensor([0.0, 1 / 30], dtype=torch.float32)
    weights = near_horizon_weights(
        decision, 4, tau_seconds=0.1, action_hz=80.0, dtype=torch.float64
    )

    assert weights.shape == (2, 4, 1)
    expected_first = torch.exp(-torch.arange(4, dtype=torch.float64) / 8)
    torch.testing.assert_close(weights[0, :, 0], expected_first)
    expected_second = torch.ones(4, dtype=torch.float64)
    expected_second[3] = math.exp(-(3 / 80 - 1 / 30) / 0.1)
    torch.testing.assert_close(weights[1, :, 0], expected_second)


def test_near_loss_normalizes_by_valid_weight_sum():
    prediction = torch.zeros(2, 5, 4)
    target = torch.full_like(prediction, 0.5)
    mask = torch.ones_like(prediction, dtype=torch.bool)
    decision = torch.zeros(2)
    groups = ((0, 1), (1, 4))

    ordinary = per_sample_action_loss(prediction, target, mask, groups=groups)
    near = per_sample_near_horizon_action_loss(
        prediction,
        target,
        mask,
        decision,
        tau_seconds=0.01,
        action_hz=10.0,
        groups=groups,
    )

    torch.testing.assert_close(near, ordinary)


def test_near_loss_normalizes_a_single_far_future_value():
    prediction = torch.zeros(1, 5, 1)
    target = torch.full_like(prediction, 0.5)
    mask = torch.zeros_like(prediction, dtype=torch.bool)
    mask[:, -1] = True

    near = per_sample_near_horizon_action_loss(
        prediction,
        target,
        mask,
        torch.zeros(1),
        tau_seconds=0.01,
        action_hz=10.0,
        groups=((0, 1),),
    )

    # Smooth L1(0, 0.5) = 0.125, independent of the nonzero temporal weight.
    torch.testing.assert_close(near, torch.tensor([0.125]))


def test_near_loss_gives_larger_gradient_to_nearer_action():
    prediction = torch.zeros(1, 3, 1, requires_grad=True)
    target = torch.full_like(prediction, 0.5)
    mask = torch.ones_like(prediction, dtype=torch.bool)
    loss = per_sample_near_horizon_action_loss(
        prediction,
        target,
        mask,
        torch.zeros(1),
        tau_seconds=0.1,
        action_hz=10.0,
        groups=((0, 1),),
    ).sum()

    loss.backward()
    gradient = prediction.grad.abs()[0, :, 0]
    assert gradient[0] / gradient[1] == pytest.approx(math.e, rel=1e-5)
    assert gradient[1] / gradient[2] == pytest.approx(math.e, rel=1e-5)


@pytest.mark.parametrize("tau", [0.0, -0.1, math.inf, math.nan])
def test_near_horizon_weights_reject_invalid_tau(tau):
    with pytest.raises(ValueError, match="tau_seconds"):
        near_horizon_weights(torch.zeros(1), 4, tau_seconds=tau)


def test_motion_loss_intersects_pair_mask_and_skips_edit_boundary():
    prediction = torch.tensor([[[100.0], [0.0], [1.0], [100.0]]])
    target = torch.tensor([[[float("nan")], [0.0], [3.0], [float("nan")]]])
    mask = torch.tensor([[[False], [True], [True], [False]]])

    values, valid = per_sample_motion_loss(
        prediction, target, mask, groups=((0, 1),)
    )

    # Only 1 -> 2 is valid.  Its motion error is |1 - 3| = 2, whose Smooth L1 is 1.5.
    torch.testing.assert_close(values, torch.tensor([1.5]))
    assert valid.tolist() == [True]


def test_motion_loss_is_invariant_to_constant_temporal_offset():
    target = torch.tensor([[[0.0], [1.0], [3.0], [6.0]]])
    prediction = target + 7
    mask = torch.ones_like(target, dtype=torch.bool)

    values, valid = per_sample_motion_loss(
        prediction, target, mask, groups=((0, 1),)
    )

    torch.testing.assert_close(values, torch.zeros(1))
    assert valid.tolist() == [True]


def test_motion_loss_marks_samples_without_adjacent_pairs_invalid():
    prediction = torch.zeros(2, 3, 1)
    target = torch.zeros_like(prediction)
    mask = torch.tensor([[[False], [True], [False]], [[True], [True], [False]]])

    values, valid = per_sample_motion_loss(
        prediction, target, mask, groups=((0, 1),)
    )

    torch.testing.assert_close(values, torch.zeros(2))
    assert valid.tolist() == [False, True]


def test_motion_loss_uses_later_endpoint_timestep_weight():
    prediction = torch.tensor([[[0.0], [0.0], [0.0]]])
    target = torch.tensor([[[0.0], [1.0], [3.0]]])
    mask = torch.ones_like(prediction, dtype=torch.bool)
    timestep_weights = torch.tensor([[[1.0], [1.0], [3.0]]])

    values, valid = per_sample_motion_loss(
        prediction,
        target,
        mask,
        groups=((0, 1),),
        timestep_weights=timestep_weights,
    )

    # Smooth L1 transition losses are 0.5 and 1.5, weighted by 1 and 3.
    torch.testing.assert_close(values, torch.tensor([(0.5 + 3 * 1.5) / 4]))
    assert valid.tolist() == [True]
