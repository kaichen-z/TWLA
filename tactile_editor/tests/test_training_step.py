"""CPU optimization smoke tests for every world-supervision combination."""
import pytest
import torch

from test_unified_world import make_model, inputs, FakeRGBHead
from tactile_editor.unified_world import WorldMode
from train_paired30 import predict, objective, resolve_world_weights


@pytest.mark.parametrize("rgb,tactile", [(0., 0.), (.1, 0.), (0., 1e-4), (.1, 1e-4)])
def test_w3_optimizer_step_for_each_weight_ablation(rgb, tactile):
    model = make_model(WorldMode.W3_BOTH)
    values = inputs(batch=2)
    batch = dict(
        base_action=values["base_action"], final_memory=values["final_memory"],
        action_mask=values["edit_mask"], touch_pressure=values["pressure"],
        touch_valid=values["temporal_valid"], decision_seconds=torch.tensor([0., 1/30]),
        target_action=torch.randn(2, 16, 192),
        future_touch_pressure=torch.randn(2, 2, 21, 21),
        future_touch_valid=torch.ones(2, 2, dtype=torch.bool),
        image_tokens=torch.randn(2, 64, 960),
        clean_latent=torch.randn(2, 2, 32, 16, 16),
        rgb_valid=torch.ones(2, 2, dtype=torch.bool),
    )
    rgb, tactile = resolve_world_weights('w3', rgb, tactile)
    if not rgb:
        for name in ('image_tokens', 'clean_latent', 'rgb_valid'):
            del batch[name]
    if not tactile:
        del batch['future_touch_pressure'], batch['future_touch_valid']
    optimizer = torch.optim.AdamW(model.optimizer_groups())
    pred, mask, detail, out = predict(model, batch, world_head=FakeRGBHead() if rgb else None,
                                    aux=True, compute_rgb=rgb>0, compute_tactile=tactile>0)
    assert (out.rgb is not None) == bool(rgb)
    assert (out.tactile_velocity is not None) == bool(tactile)
    loss, _ = objective(pred, mask, detail, out, batch["target_action"],
                        {"rgb": rgb, "tactile": tactile, "roi": torch.ones(2, 2, 16, 16)})
    assert torch.isfinite(loss)
    loss.backward()
    if not rgb:
        assert all(p.grad is None for p in model.rgb_world.parameters())
    if not tactile:
        assert all(p.grad is None for p in model.tactile_world.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.action.parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    assert all(torch.isfinite(p).all() for p in model.parameters())
