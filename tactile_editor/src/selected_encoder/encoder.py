"""Small, stable inference wrapper around the selected tactile encoder.

The architecture itself lives in :mod:`egotouch_formal.model`; this module does
not duplicate it.  It only defines the checkpoint/input contract used by the
published Hugging Face artifact.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import nn

from egotouch_formal.model import TactileEncoder


DEFAULT_MODEL_ID = "MIT-Media-Lab/egotouch-tactile-encoder-p1-v1"
DEFAULT_SEED = 20260831


def _validate_touch_and_side(
    touch: torch.Tensor, side: torch.Tensor, history_frames: int = 8
) -> None:
    if touch.ndim != 4 or touch.shape[1:] != (history_frames, 21, 21):
        raise ValueError(
            f"touch must be [B,{history_frames},21,21], got {tuple(touch.shape)}"
        )
    if side.ndim != 1 or side.shape[0] != touch.shape[0]:
        raise ValueError(f"side must be [B], got {tuple(side.shape)} for batch {touch.shape[0]}")
    if side.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"side must be an integer tensor, got {side.dtype}")
    if torch.any((side < 0) | (side > 1)):
        raise ValueError("side values must be 0 (left) or 1 (right)")


def canonicalize_raw_touch(touch: torch.Tensor, side: torch.Tensor) -> torch.Tensor:
    """Convert raw glove grids to the right-hand canonical orientation.

    EgoTouch stores the two gloves in mirrored grid conventions.  The training
    cache flips the last axis for left-hand samples and leaves right-hand
    samples unchanged.  Call this exactly once when starting from raw glove
    coordinates; cached clean-atomic tensors are already canonicalized.
    """

    if touch.ndim != 4 or touch.shape[-2:] != (21, 21):
        raise ValueError(f"touch must be [B,T,21,21], got {tuple(touch.shape)}")
    _validate_touch_and_side(touch, side, int(touch.shape[1]))
    result = touch.clone()
    left = side.to(touch.device) == 0
    result[left] = torch.flip(result[left], dims=(-1,))
    return result


class SelectedTactileEncoder(nn.Module):
    """Inference-ready full-hand CNN+BiGRU encoder with its validity masks.

    ``forward`` expects canonicalized ``[B,T,21,21]`` pressure histories, where
    ``T`` equals ``history_frames``, and returns a normalized 64D P1 embedding.
    """

    def __init__(
        self,
        encoder: TactileEncoder,
        pressure_masks: torch.Tensor,
        *,
        history_frames: int = 8,
    ) -> None:
        super().__init__()
        if tuple(pressure_masks.shape) != (2, 21, 21):
            raise ValueError(
                f"pressure_masks must be [2,21,21], got {tuple(pressure_masks.shape)}"
            )
        self.encoder = encoder
        if history_frames < 1:
            raise ValueError("history_frames must be positive")
        self.history_frames = int(history_frames)
        self.register_buffer("pressure_masks", pressure_masks.to(torch.bool), persistent=True)

    @classmethod
    def from_pretrained_dir(
        cls,
        model_dir: str | Path,
        *,
        seed: int = DEFAULT_SEED,
        device: str | torch.device = "cpu",
        history_frames: int = 8,
    ) -> "SelectedTactileEncoder":
        """Load a local snapshot of the published Hugging Face repository."""

        root = Path(model_dir)
        weights = root / f"seed-{seed}" / "tactile_encoder.safetensors"
        masks = root / "pressure_masks.npy"
        missing = [str(path) for path in (weights, masks) if not path.is_file()]
        if missing:
            raise FileNotFoundError("missing selected-encoder artifact(s): " + ", ".join(missing))

        encoder = TactileEncoder(embed_dim=64)
        encoder.load_state_dict(load_file(str(weights), device=str(device)), strict=True)
        pressure_masks = torch.from_numpy(np.load(masks).astype(bool, copy=False))
        model = cls(
            encoder, pressure_masks, history_frames=history_frames
        ).to(device)
        model.eval()
        return model

    def forward(self, touch: torch.Tensor, side: torch.Tensor) -> torch.Tensor:
        _validate_touch_and_side(touch, side, self.history_frames)
        validity = self.pressure_masks[side.to(self.pressure_masks.device)]
        features = self.encoder(touch, validity)
        return F.normalize(features, dim=-1)
