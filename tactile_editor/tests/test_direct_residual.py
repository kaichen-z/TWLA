import pytest
import torch

from tactile_editor.direct_residual import (
    DIRECT_INPUT_MODES,
    DirectTactileResidualEditor,
    canonical_tactile_channels,
)


def fixture(mode: str = "history_delta"):
    masks = torch.ones(2, 21, 21, dtype=torch.bool)
    reference = torch.zeros(1, 8, 2, 21, 21)
    model = DirectTactileResidualEditor(
        torch.zeros(192),
        torch.ones(192),
        masks,
        reference,
        mode=mode,
    ).eval()
    anchor = torch.randn(3, 16, 192)
    edit_mask = torch.ones_like(anchor, dtype=torch.bool)
    pressure = torch.randn(3, 8, 2, 21, 21)
    valid = torch.ones(3, 8, 2, dtype=torch.bool)
    elapsed = torch.tensor([0.0, 1 / 30, 2 / 30])
    return model, anchor, edit_mask, pressure, valid, elapsed


@pytest.mark.parametrize("mode", DIRECT_INPUT_MODES)
def test_direct_editor_starts_as_exact_anchor(mode):
    model, anchor, mask, pressure, valid, elapsed = fixture(mode)
    prediction, details = model(anchor, mask, pressure, valid, elapsed)
    torch.testing.assert_close(prediction, anchor, rtol=0, atol=0)
    assert torch.count_nonzero(details["effective_delta"]) == 0


def test_reference_pressure_forces_exact_zero_residual_after_basis_training():
    model, anchor, mask, _, valid, elapsed = fixture()
    with torch.no_grad():
        model.action.basis_head.weight.normal_()
        model.action.basis_head.bias.normal_()
    pressure = model.tactile.reference_pressure.expand(len(anchor), -1, -1, -1, -1)
    prediction, details = model(anchor, mask, pressure, valid, elapsed)
    torch.testing.assert_close(prediction, anchor, rtol=0, atol=0)
    assert torch.count_nonzero(details["coefficients"]) == 0


def test_invalid_touch_forces_exact_zero_residual():
    model, anchor, mask, pressure, valid, elapsed = fixture()
    with torch.no_grad():
        model.action.basis_head.weight.normal_()
        model.action.basis_head.bias.normal_()
    valid.zero_()
    prediction, details = model(anchor, mask, pressure, valid, elapsed)
    torch.testing.assert_close(prediction, anchor, rtol=0, atol=0)
    assert torch.count_nonzero(details["coefficients"]) == 0


def test_invalid_right_hand_cannot_change_right_action():
    model, anchor, mask, pressure, valid, elapsed = fixture("current_delta")
    with torch.no_grad():
        model.action.basis_head.weight.normal_()
        model.action.basis_head.bias.normal_()
    valid[:, :, 1] = False
    prediction, details = model(anchor, mask, pressure, valid, elapsed)
    assert torch.count_nonzero(details["effective_delta"][..., 69:138]) == 0
    assert torch.count_nonzero(details["effective_delta"][..., :6]) == 0
    assert torch.count_nonzero(details["effective_delta"][..., 138:]) == 0
    assert torch.equal(prediction[..., 69:138], anchor[..., 69:138])


def test_canonical_channels_flip_left_and_mask_invalid_differences():
    pressure = torch.zeros(1, 2, 2, 21, 21)
    pressure[0, 0, 0, 0, 0] = 0.2
    pressure[0, 1, 0, 0, 0] = 0.7
    valid = torch.ones(1, 2, 2, dtype=torch.bool)
    valid[0, 0, 1] = False
    masks = torch.ones(2, 21, 21, dtype=torch.bool)
    channels, latest = canonical_tactile_channels(
        pressure, valid, masks, include_delta=True
    )
    assert channels[0, 1, 0, 0, 0, 20] == pytest.approx(0.7)
    assert channels[0, 1, 0, 1, 0, 20] == pytest.approx(0.5)
    assert torch.count_nonzero(channels[0, 1, 1, 1]) == 0
    assert latest.tolist() == [[True, True]]
