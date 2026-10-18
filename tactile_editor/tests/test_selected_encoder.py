from __future__ import annotations

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from egotouch_formal.model import TactileEncoder
from selected_encoder import SelectedTactileEncoder, canonicalize_raw_touch


def test_raw_touch_canonicalization_only_flips_left() -> None:
    touch = torch.arange(2 * 8 * 21 * 21, dtype=torch.float32).reshape(2, 8, 21, 21)
    side = torch.tensor([0, 1])
    canonical = canonicalize_raw_touch(touch, side)
    torch.testing.assert_close(canonical[0], torch.flip(touch[0], dims=(-1,)))
    torch.testing.assert_close(canonical[1], touch[1])


def test_load_selected_encoder_and_return_normalized_embedding(tmp_path) -> None:
    torch.manual_seed(7)
    source = TactileEncoder(embed_dim=64).eval()
    seed_dir = tmp_path / "seed-20260831"
    seed_dir.mkdir()
    save_file(
        {name: value.detach().contiguous() for name, value in source.state_dict().items()},
        seed_dir / "tactile_encoder.safetensors",
    )
    masks = np.ones((2, 21, 21), dtype=bool)
    np.save(tmp_path / "pressure_masks.npy", masks)

    selected = SelectedTactileEncoder.from_pretrained_dir(tmp_path)
    touch = torch.randn(3, 8, 21, 21)
    side = torch.tensor([0, 1, 0])
    with torch.inference_mode():
        actual = selected(touch, side)
        expected = torch.nn.functional.normalize(
            source(touch, torch.from_numpy(masks)[side]), dim=-1
        )
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual.norm(dim=-1), torch.ones(3))


def test_selected_encoder_rejects_noncanonical_input_shape(tmp_path) -> None:
    model = SelectedTactileEncoder(TactileEncoder(), torch.ones(2, 21, 21, dtype=torch.bool))
    with pytest.raises(ValueError, match="touch must be"):
        model(torch.zeros(1, 7, 21, 21), torch.tensor([1]))
