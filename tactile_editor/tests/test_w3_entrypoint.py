"""The public entrypoint exposes W3; coefficients control auxiliary losses."""
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from train_paired30 import PUBLIC_VARIANTS, resolve_world_weights
from test_unified_world import FakeRGBHead


def test_only_w3_and_baselines_are_public():
    assert PUBLIC_VARIANTS == ('w3', 'smolvla', 'vitra', 'trex', 'memoryvla')
    assert resolve_world_weights('w3') == (.1, 1e-4)
    assert resolve_world_weights('smolvla') == (0., 0.)
    script = Path(__file__).resolve().parents[1] / 'scripts/train_paired30.py'
    result = subprocess.run([sys.executable, str(script), '--help'], capture_output=True, text=True)
    assert result.returncode == 0
    assert 'w3,smolvla,vitra,trex,memoryvla' in result.stdout
    for removed in ('w0', 'w1_rgb', 'w2_tactile', 'w3_both'):
        with pytest.raises(ValueError, match='Unsupported editor'):
            resolve_world_weights(removed)
        result = subprocess.run([sys.executable, str(script), '--variant', removed],
                                capture_output=True, text=True)
        assert result.returncode == 2 and 'invalid choice' in result.stderr


@pytest.mark.parametrize('rgb,tactile', [(-1., 0.), (float('nan'), .1), (0., float('inf'))])
def test_reject_invalid_weights(rgb, tactile):
    with pytest.raises(ValueError, match='finite and nonnegative'):
        resolve_world_weights('w3', rgb, tactile)


def test_baseline_rejects_world_weights():
    with pytest.raises(ValueError, match='corresponding branch'):
        resolve_world_weights('vitra', .1, 0.)


@pytest.mark.parametrize('rgb,tactile', [(0., 0.), (.1, 0.), (0., 1e-4), (.1, 1e-4)])
def test_tachin_w3_weight_ablation_backward(rgb, tactile):
    from train_tachin import make_model, objective
    torch.manual_seed(7)
    model = make_model('twla', {'pressure_mean': torch.zeros(2, 880),
                              'pressure_scale': torch.ones(2, 880)}, 'cpu')
    batch = dict(
        base_action=torch.randn(1, 16, 192), target_action=torch.randn(1, 16, 192),
        action_mask=torch.ones(1, 16, 192, dtype=torch.bool),
        decision_seconds=torch.zeros(1), pressure=torch.randn(1, 8, 2, 880),
        pressure_mask=torch.ones(1, 8, 2, 880, dtype=torch.bool),
        sample_valid=torch.ones(1, 8, 2, dtype=torch.bool), final_memory=torch.randn(1, 8, 960),
    )
    if tactile:
        batch.update(future_pressure=torch.randn(1, 2, 880),
                     future_pressure_mask=torch.ones(1, 2, 880, dtype=torch.bool),
                     future_valid=torch.ones(1, 2, dtype=torch.bool))
    if rgb:
        batch.update(clean_latent=torch.randn(1, 2, 32, 16, 16),
                     image_tokens=torch.randn(1, 64, 960),
                     rgb_valid=torch.ones(1, 2, dtype=torch.bool),
                     future_roi=torch.ones(1, 2, 16, 16, dtype=torch.bool))
    loss, terms = objective(model, batch, rgb_head=FakeRGBHead() if rgb else None,
                            generator=None, lambda_rgb=rgb, lambda_tactile=tactile)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None for p in model.action.parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    if not rgb:
        assert terms['rgb'] == 0.
        assert all(p.grad is None for p in model.rgb_world.parameters())
    if not tactile:
        assert terms['tactile'] == 0.
        assert all(p.grad is None for p in model.tactile_world.parameters())
