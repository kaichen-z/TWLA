from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tactile_editor.encoder import TouchFeatures
from tactile_editor.unified_world import UnifiedWorldTactileEditor, WorldMode


class FakeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(64))

    def forward(self, pressure, temporal_valid):
        value = pressure.mean((1, 3, 4))  # [B,2]
        token = value[..., None] * self.scale
        return TouchFeatures(token, temporal_valid.any(1))


class FakeRGBHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.marker = nn.Parameter(torch.ones(()), requires_grad=False)
        self.adapter = nn.Linear(1, 1)
        self.adapter.cfg = SimpleNamespace(num_horizons=1)

    def build_condition(self, image_tokens, final_memory):
        return torch.zeros(
            len(image_tokens), 72, 2304,
            device=image_tokens.device, dtype=image_tokens.dtype,
        )

    def predict_velocity(self, noisy_latent, tau, condition):
        signal = condition.mean((1, 2))[:, None, None, None]
        return noisy_latent + signal


def make_model(mode):
    torch.manual_seed(4)
    return UnifiedWorldTactileEditor(
        FakeEncoder(),
        world_mode=mode,
        action_mean=torch.zeros(192),
        action_std=torch.ones(192),
        pressure_masks=torch.ones(2, 21, 21, dtype=torch.bool),
        reference_pressure=torch.zeros(1, 8, 2, 21, 21),
        pressure_scale=torch.ones(2, 21, 21),
    )


def inputs(batch=2):
    return dict(
        base_action=torch.randn(batch, 16, 192),
        edit_mask=torch.ones(batch, 16, 192, dtype=torch.bool),
        pressure=torch.randn(batch, 8, 2, 21, 21),
        temporal_valid=torch.ones(batch, 8, 2, dtype=torch.bool),
        decision_seconds=torch.zeros(batch),
        final_memory=torch.randn(batch, 8, 960),
    )


@pytest.mark.parametrize("mode", list(WorldMode))
def test_all_modes_share_action_contract(mode):
    model = make_model(mode)
    data = inputs()
    kwargs = {}
    if mode.uses_tactile:
        kwargs.update(
            future_pressure=torch.randn(2, 2, 21, 21),
            future_valid=torch.ones(2, 2, dtype=torch.bool),
            tactile_tau=torch.full((2,), 0.5),
            tactile_noise=torch.zeros(2, 2, 441),
        )
    if mode.uses_rgb:
        kwargs.update(
            rgb_world_head=FakeRGBHead(),
            image_tokens=torch.randn(2, 64, 960),
            clean_latent=torch.randn(2, 2, 32, 16, 16),
            rgb_valid=torch.ones(2, 2, dtype=torch.bool),
            rgb_tau=torch.full((2, 2), 0.5),
            rgb_noise=torch.zeros(2, 2, 32, 16, 16),
        )
    output = model(**data, **kwargs)
    assert output.final_action.shape == (2, 16, 192)
    assert output.delta_action.shape == (2, 16, 192)
    assert output.rgb is not None if mode.uses_rgb else output.rgb is None
    assert output.tactile_velocity is not None if mode.uses_tactile else output.tactile_velocity is None


def test_common_action_initialization_is_identical_across_modes():
    states = []
    for mode in WorldMode:
        model = make_model(mode)
        states.append({name: value for name, value in model.action.state_dict().items()})
    for name in states[0]:
        for state in states[1:]:
            torch.testing.assert_close(states[0][name], state[name])


@pytest.mark.parametrize("mode", list(WorldMode))
def test_runtime_router_needs_no_future_supervision(mode):
    from tactile_editor import BaseActionContext, MemoryW0W3Editor, TactileEditorRouter

    editor = make_model(mode).eval()
    data = inputs()
    mask = data["edit_mask"].clone()
    mask[..., 138:] = False
    mask[:, :4] = False
    context = BaseActionContext(data["base_action"], mask, "t-wla", "stats",
                                memory_tokens=data["final_memory"].bfloat16())
    router = TactileEditorRouter(MemoryW0W3Editor(editor), None)
    output = router(context, data["pressure"], data["temporal_valid"], data["decision_seconds"])
    assert torch.isfinite(output.final_action).all()
    assert torch.equal(output.final_action[~mask], data["base_action"][~mask])
    assert output.auxiliary_outputs["rgb"] is None
    assert output.auxiliary_outputs["tactile_velocity"] is None
    if mode != WorldMode.W0:
        with pytest.raises(ValueError, match="require"):
            router(context, data["pressure"], data["temporal_valid"],
                   data["decision_seconds"], compute_auxiliary=True)


def test_optimizer_groups_cover_only_owned_trainable_parameters():
    for mode in WorldMode:
        model = make_model(mode)
        groups = model.optimizer_groups()
        selected = {id(p) for group in groups for p in group["params"]}
        expected = {id(p) for p in model.parameters() if p.requires_grad}
        assert selected == expected
