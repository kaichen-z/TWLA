from __future__ import annotations

import torch
from torch import nn

from tactile_editor import DualHandTactileEncoder, load_formal_t2p_encoder
from egotouch_formal.model import FormalEncoderModel


class CaptureEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.last_touch: torch.Tensor | None = None
        self.last_side: torch.Tensor | None = None

    def forward(self, touch: torch.Tensor, side: torch.Tensor) -> torch.Tensor:
        self.last_touch = touch.detach().clone()
        self.last_side = side.detach().clone()
        return torch.nn.functional.normalize(torch.ones(len(touch), 64), dim=-1)


class MaskCaptureEncoder(CaptureEncoder):
    def __init__(self) -> None:
        super().__init__()
        masks = torch.zeros(2, 21, 21, dtype=torch.bool)
        masks[:, :7] = True
        self.register_buffer("pressure_masks", masks)


def test_dual_hand_bridge_flattens_and_canonicalizes_once() -> None:
    capture = CaptureEncoder()
    bridge = DualHandTactileEncoder(capture)
    pressure = torch.arange(8 * 2 * 21 * 21, dtype=torch.float32).reshape(
        1, 8, 2, 21, 21
    )
    result = bridge(pressure, canonicalized=False)
    assert result.tokens.shape == (1, 2, 64)
    assert result.valid.tolist() == [[True, True]]
    assert capture.last_touch is not None
    torch.testing.assert_close(
        capture.last_touch[0], torch.flip(pressure[0, :, 0], dims=(-1,))
    )
    torch.testing.assert_close(capture.last_touch[1], pressure[0, :, 1])
    assert capture.last_side.tolist() == [0, 1]


def test_temporal_mask_removes_a_missing_hand() -> None:
    bridge = DualHandTactileEncoder(CaptureEncoder())
    pressure = torch.randn(2, 8, 2, 21, 21)
    valid = torch.ones(2, 8, 2, dtype=torch.bool)
    valid[0, :, 0] = False
    result = bridge(pressure, valid, canonicalized=True)
    assert result.valid.tolist() == [[False, True], [True, True]]
    assert torch.count_nonzero(result.tokens[0, 0]) == 0


def test_bridge_removes_non_pressure_cells() -> None:
    capture = MaskCaptureEncoder()
    bridge = DualHandTactileEncoder(capture)
    bridge(torch.ones(1, 8, 2, 21, 21), canonicalized=True)
    assert capture.last_touch is not None
    assert torch.count_nonzero(capture.last_touch) == 2 * 8 * 7 * 21


def test_formal_p1_checkpoint_exports_runtime_tactile_tower(tmp_path) -> None:
    masks = torch.ones(2, 21, 21, dtype=torch.bool)
    formal = FormalEncoderModel(masks, task="p1", pose_temporal="gru")
    checkpoint = tmp_path / "best.pt"
    torch.save(
        {
            "model": formal.state_dict(),
            "config": {
                "task": "p1",
                "embed_dim": 64,
                "tactile_history_frames": 4,
            },
        },
        checkpoint,
    )
    selected = load_formal_t2p_encoder(checkpoint, history_frames=4)
    selected.eval()
    touch = torch.randn(3, 4, 21, 21)
    side = torch.tensor([0, 1, 0])
    with torch.no_grad():
        expected = torch.nn.functional.normalize(
            formal.tactile(touch, masks[side]), dim=-1
        )
        actual = selected(touch, side)
    torch.testing.assert_close(actual, expected)
