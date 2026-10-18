from __future__ import annotations

import torch

from tactile_editor.single_stage_world import (
    SingleStageWorldTactileEditor,
    gather_action_window,
    masked_tactile_flow_loss,
    ramp_world_weight,
)


def make_model() -> SingleStageWorldTactileEditor:
    return SingleStageWorldTactileEditor(
        action_mean=torch.zeros(192),
        action_std=torch.ones(192),
        pressure_masks=torch.ones(2, 21, 21, dtype=torch.bool),
        reference_pressure=torch.zeros(1, 8, 2, 21, 21),
        pressure_scale=torch.ones(2, 21, 21),
        hidden_dim=32,
    )


def sample(batch: int = 2):
    base = torch.randn(batch, 16, 192, requires_grad=True)
    memory = torch.randn(batch, 8, 960, requires_grad=True)
    mask = torch.ones_like(base, dtype=torch.bool)
    pressure = torch.randn(batch, 8, 2, 21, 21)
    valid = torch.ones(batch, 8, 2, dtype=torch.bool)
    decision = torch.tensor([0.0, 1 / 30])[:batch]
    future = torch.randn(batch, 2, 21, 21)
    future_valid = torch.ones(batch, 2, dtype=torch.bool)
    return base, memory, mask, pressure, valid, decision, future, future_valid


def test_action_window_uses_plan_relative_decision_and_padding():
    action = torch.arange(16.0).view(1, 16, 1)
    window, valid = gather_action_window(action, torch.tensor([0.1]), steps=10)
    assert window[0, 0, 0] == 8
    assert valid.sum() == 8
    assert window[0, -1, 0] == 0


def test_joint_forward_preserves_frozen_inputs_and_exact_mask():
    model = make_model().eval()
    base, memory, mask, pressure, valid, decision, future, future_valid = sample()
    mask[:, :4] = False
    output = model(
        base_action=base,
        edit_mask=mask,
        pressure=pressure,
        temporal_valid=valid,
        decision_seconds=decision,
        final_memory=memory,
        future_pressure=future,
        future_valid=future_valid,
        tau=torch.full((2,), 0.5),
        noise=torch.zeros(2, 2, 441),
    )
    assert output.final_action.shape == (2, 16, 192)
    assert output.world_velocity is not None
    assert output.world_velocity.shape == (2, 2, 441)
    assert torch.equal(output.final_action[:, :4], base[:, :4])
    assert not output.allowed_mask[..., 138:].any()
    loss, _ = model.objective(output, torch.randn_like(base), mask, lambda_world=0.1)
    loss.backward()
    assert base.grad is None
    assert memory.grad is None
    assert model.tactile.frame_encoder[0].weight.grad is not None
    assert model.tactile.temporal.weight_ih_l0.grad is not None


def test_invalid_hand_is_not_edited_or_used_as_world_target():
    model = make_model().eval()
    base, memory, mask, pressure, valid, decision, future, future_valid = sample(1)
    valid[:, :, 0] = False
    future_valid[:, 0] = False
    output = model(
        base_action=base,
        edit_mask=mask,
        pressure=pressure,
        temporal_valid=valid,
        decision_seconds=decision,
        final_memory=memory,
        future_pressure=future,
        future_valid=future_valid,
        tau=torch.tensor([0.5]),
    )
    assert torch.equal(output.final_action[..., :69], base[..., :69])
    assert not output.world_valid[:, 0].any()


def test_world_loss_updates_shared_touch_but_not_action_head():
    model = make_model().eval()
    base, memory, mask, pressure, valid, decision, future, future_valid = sample()
    output = model(
        base_action=base,
        edit_mask=mask,
        pressure=pressure,
        temporal_valid=valid,
        decision_seconds=decision,
        final_memory=memory,
        future_pressure=future,
        future_valid=future_valid,
        tau=torch.full((2,), 0.5),
    )
    loss = masked_tactile_flow_loss(
        output.world_velocity, output.world_target, output.world_valid
    )
    loss.backward()
    assert model.tactile.frame_encoder[0].weight.grad is not None
    assert model.action.output.weight.grad is None


def test_optimizer_groups_are_disjoint_and_world_ramp_is_bounded():
    model = make_model()
    groups = model.optimizer_groups()
    parameters = [parameter for group in groups for parameter in group["params"]]
    assert len(parameters) == len({id(parameter) for parameter in parameters})
    assert ramp_world_weight(0, 0.2, 100) == 0
    assert ramp_world_weight(50, 0.2, 100) == 0.1
    assert ramp_world_weight(200, 0.2, 100) == 0.2
