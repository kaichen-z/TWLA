from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from tactile_editor.public_backbone import policy_context, RGBWorldAdapter, foundation_path


def test_policy_full_chunk_and_editor_normalization():
    class Policy:
        def act(self, *args, **kwargs):
            assert kwargs == {"steps": 16, "return_mf": True}
            return np.full((16, 192), 5, np.float32), torch.ones(8, 960)
    mask = torch.zeros(16, 192, dtype=torch.bool)
    mask[:, :138] = True
    context = policy_context(
        Policy(), rgb=None, state=None, state_mask=None, fov=None, episode_id=1,
        action_mask=mask, action_mean=torch.ones(192), action_std=torch.full((192,), 2.),
        normalization_id="train-statistics",
    )
    assert context.base_action.shape == (1, 16, 192)
    assert context.memory_tokens.shape == (1, 8, 960)
    torch.testing.assert_close(context.base_action, torch.full((1, 16, 192), 2.))


def test_world_adapter_retains_condition_gradients_with_frozen_weights():
    class Sana(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.ones(()))
        def forward(self, hidden_states, encoder_hidden_states, timestep, return_dict):
            return (hidden_states + self.scale * encoder_hidden_states.mean((1, 2))[:, None, None, None],)
    head = nn.Module()
    head.adapter = nn.Linear(3, 4)
    head.sana = Sana()
    head.cfg = SimpleNamespace(sana=SimpleNamespace(timestep_scale=1000))
    bridge = RGBWorldAdapter(head)
    bridge.train()
    assert not bridge.training and not head.training
    condition = torch.randn(2, 4, 3, requires_grad=True)
    bridge.predict_velocity(torch.zeros(2, 1, 2, 2), torch.ones(2), condition).sum().backward()
    assert condition.grad.abs().sum() > 0
    assert all(not p.requires_grad and p.grad is None for p in head.parameters())


def test_sibling_foundation_is_present():
    assert (foundation_path() / "world_model/head.py").is_file()
