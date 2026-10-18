from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from tactile_editor.rgb_world_editor import (
    RGBWorldRegularizedTactileEditor,
    RGBWorldResidualOutput,
    TactileMemoryConditionModulator,
    roi_weighted_world_loss,
)


class _FakeAdapter(nn.Module):
    def __init__(self, input_dim: int = 4, output_dim: int = 6):
        super().__init__()
        self.cfg = SimpleNamespace(num_horizons=1)
        self.projection = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, image, memory):
        return self.projection(torch.cat((image, memory), dim=1))

    def for_horizon(self, condition, horizon):
        return condition


class _FakeFrozenWorld(nn.Module):
    def __init__(self):
        super().__init__()
        self.adapter = _FakeAdapter()
        self.scale = nn.Parameter(torch.tensor(0.25), requires_grad=False)
        self.requires_grad_(False)

    def build_condition(self, image, memory):
        return self.adapter(image, memory)

    def predict_velocity(self, noisy, tau, condition, *, horizon=None):
        signal = condition.mean((1, 2))[:, None, None, None]
        return self.scale * noisy + signal


def _batch(batch: int = 2):
    reference = torch.full((1, 8, 2, 21, 21), 0.2)
    pressure = reference.expand(batch, -1, -1, -1, -1).clone()
    temporal_valid = torch.ones(batch, 8, 2, dtype=torch.bool)
    return reference, pressure, temporal_valid


def _model(reference, *, articulation_only=True):
    return RGBWorldRegularizedTactileEditor(
        action_mean=torch.zeros(192),
        action_std=torch.ones(192),
        pressure_masks=torch.ones(2, 21, 21, dtype=torch.bool),
        reference_pressure=reference,
        articulation_only=articulation_only,
        memory_dim=4,
        hidden_dim=32,
        world_condition_dim=6,
    )


def _forward(model, pressure, temporal_valid, world):
    batch = len(pressure)
    return model(
        anchor_action=torch.randn(batch, 16, 192),
        edit_mask=torch.ones(batch, 16, 192, dtype=torch.bool),
        pressure=pressure,
        temporal_valid=temporal_valid,
        decision_seconds=torch.zeros(batch),
        final_memory=torch.randn(batch, 8, 4),
        world_head=world,
        image_tokens=torch.randn(batch, 3, 4),
        clean_latent=torch.randn(batch, 2, 2, 4, 4),
        world_valid=torch.ones(batch, 2, dtype=torch.bool),
        tau=torch.full((batch, 2), 0.4),
        noise=torch.randn(batch, 2, 2, 4, 4),
    )


def test_reference_tactile_is_exact_action_and_world_identity():
    reference, pressure, valid = _batch()
    model = _model(reference).eval()
    world = _FakeFrozenWorld().eval()
    anchor = torch.randn(2, 16, 192)
    output = model(
        anchor_action=anchor,
        edit_mask=torch.ones_like(anchor, dtype=torch.bool),
        pressure=pressure,
        temporal_valid=valid,
        decision_seconds=torch.zeros(2),
        final_memory=torch.randn(2, 8, 4),
        world_head=world,
        image_tokens=torch.randn(2, 3, 4),
        clean_latent=torch.randn(2, 2, 2, 4, 4),
        world_valid=torch.ones(2, 2, dtype=torch.bool),
        tau=torch.full((2, 2), 0.5),
        noise=torch.randn(2, 2, 2, 4, 4),
    )
    assert torch.equal(output.final_action, anchor)
    assert torch.count_nonzero(output.delta_action) == 0
    assert output.world is not None
    assert torch.equal(output.world.tactile_velocity, output.world.baseline_velocity)
    assert torch.count_nonzero(output.world.predicted_residual) == 0


def test_rgb_loss_reaches_modulator_then_tactile_trunk_but_not_world():
    torch.manual_seed(4)
    reference, pressure, valid = _batch()
    pressure[:, -2:] += 0.3
    model = _model(reference).train()
    world = _FakeFrozenWorld().eval()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # Zero-init makes the first step update only the final condition projection.
    first = _forward(model, pressure, valid, world)
    assert first.world is not None
    roi = torch.ones(2, 2, 4, 4, dtype=torch.bool)
    loss = roi_weighted_world_loss(first.world, roi)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.rgb_world.parameters())
    assert all(p.grad is None for p in world.parameters())
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    second = _forward(model, pressure, valid, world)
    assert second.world is not None
    roi_weighted_world_loss(second.world, roi).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.tactile.frame_encoder.parameters())
    assert all(p.grad is None for p in world.parameters())


def test_action_scope_masks_only_physical_components():
    reference, pressure, valid = _batch(1)
    world = _FakeFrozenWorld().eval()
    articulation = _forward(_model(reference, articulation_only=True).eval(),
                            pressure, valid, world)
    assert not articulation.allowed_mask[..., :6].any()
    assert articulation.allowed_mask[..., 6:69].all()
    assert not articulation.allowed_mask[..., 69:75].any()
    assert articulation.allowed_mask[..., 75:138].all()
    assert not articulation.allowed_mask[..., 138:].any()

    full = _forward(_model(reference, articulation_only=False).eval(),
                    pressure, valid, world)
    assert full.allowed_mask[..., :138].all()
    assert not full.allowed_mask[..., 138:].any()


def test_modulator_zero_input_is_exact_zero_without_bias():
    module = TactileMemoryConditionModulator(
        tactile_dim=5, memory_rows=3, condition_dim=7, hidden_dim=11
    )
    output = module(torch.zeros(4, 2, 5), torch.ones(4, 2, dtype=torch.bool))
    assert torch.equal(output, torch.zeros_like(output))
    assert all(layer.bias is None for layer in module.input)
    assert all(layer.bias is None for layer in module.output)


def test_roi_weighted_loss_ignores_invalid_rows_and_prefers_roi():
    predicted = torch.zeros(2, 1, 2, 2)
    target = torch.ones_like(predicted)
    target[1] = 100
    output = RGBWorldResidualOutput(
        predicted_residual=predicted,
        target_residual=target,
        baseline_velocity=torch.zeros_like(predicted),
        tactile_velocity=torch.zeros_like(predicted),
        valid=torch.tensor([True, False]),
    )
    roi = torch.zeros(1, 2, 2, 2, dtype=torch.bool)
    roi[:, :, 0, 0] = True
    assert torch.isclose(roi_weighted_world_loss(output, roi), torch.tensor(1.0))
