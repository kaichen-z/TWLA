import torch

from tactile_editor.native_tachin import (NativeResidualEditor, NativeTachinEncoder,
                                          NativeTachinW3Editor, SENSOR_OFFSETS,
                                          SENSOR_SHAPES)


def inputs(batch=2):
    mean = torch.linspace(0, 0.2, 1760).reshape(2, 880)
    scale = torch.ones(2, 880) * 0.1
    pressure = mean[None, None].expand(batch, 8, -1, -1).clone()
    cell_valid = torch.ones_like(pressure, dtype=torch.bool)
    sample_valid = torch.ones(batch, 8, 2, dtype=torch.bool)
    return mean, scale, pressure, cell_valid, sample_valid


def test_native_regions_cover_each_cell_once():
    assert SENSOR_OFFSETS[0] == 0 and SENSOR_OFFSETS[-1] == 880
    assert all(stop - start == height * width for (height, width), start, stop
               in zip(SENSOR_SHAPES, SENSOR_OFFSETS[:-1], SENSOR_OFFSETS[1:]))


def test_reference_touch_is_exact_base_for_memory_free_editor():
    mean, scale, pressure, cell_valid, sample_valid = inputs()
    model = NativeResidualEditor(NativeTachinEncoder(mean, scale)).eval()
    base = torch.randn(2, 16, 192)
    mask = torch.ones_like(base, dtype=torch.bool)
    final, delta, allowed, features, hand_valid = model(
        base, mask, pressure, cell_valid, sample_valid)
    assert torch.equal(final, base)
    assert not delta.bool().any()
    assert not features.bool().any()
    assert allowed[..., :138].all() and not allowed[..., 138:].any()
    assert hand_valid.all()


def test_reference_touch_stays_exact_after_coefficient_bias_is_trained():
    mean, scale, pressure, cell_valid, sample_valid = inputs()
    pressure = mean[None, None].expand_as(pressure).clone()
    model = NativeResidualEditor(NativeTachinEncoder(mean, scale)).eval()
    with torch.no_grad():
        model.coefficients[0].bias.fill_(0.5)
        model.action.basis_head.bias.fill_(0.5)
    base = torch.randn(2, 16, 192)
    final, delta, _, features, _ = model(
        base, torch.ones_like(base, dtype=torch.bool), pressure,
        cell_valid, sample_valid,
    )
    assert not features.bool().any()
    assert not delta.bool().any()
    assert torch.equal(final, base)


def test_native_w3_no_auxiliary_shape_and_reference_invariance():
    mean, scale, pressure, cell_valid, sample_valid = inputs(batch=1)
    model = NativeTachinW3Editor(NativeTachinEncoder(mean, scale), mean, scale).eval()
    base = torch.randn(1, 16, 192)
    output = model(base_action=base, edit_mask=torch.ones_like(base, dtype=torch.bool),
                   pressure=pressure, cell_valid=cell_valid,
                   sample_valid=sample_valid, final_memory=torch.randn(1, 8, 960),
                   compute_auxiliary=False)
    assert torch.equal(output.final_action, base)
    assert output.tactile_velocity is None and output.rgb is None
