"""Dataset wrapper that resizes frames the way PaliGemma's processor does.

The shared loader keeps the aspect ratio and pads the left and top. PaliGemma's processor
resizes to a square without keeping the aspect ratio, which is what VITRA was trained
with. `config.ImageConfig.mode` selects between the two.
"""

from __future__ import annotations

import numpy as np

# `test.py` puts this directory on sys.path, so `v4_bridge` is a top-level module.
from v4_bridge import SequenceDataset


class SquashSequenceDataset(SequenceDataset):
    """`SequenceDataset` with the resize to a square that ignores the aspect ratio."""

    def _to_square(self, imgs: np.ndarray) -> np.ndarray:
        """(L, H, W, 3) uint8 -> (L, S, S, 3), aspect ratio not preserved."""
        import cv2

        s = self.cfg.vlm.image_size
        n, h, w = imgs.shape[:3]
        if (h, w) == (s, s):
            return imgs
        # INTER_AREA is for shrinking; on upscale it degenerates to nearest-neighbour.
        interp = cv2.INTER_AREA if (h >= s and w >= s) else cv2.INTER_LINEAR
        out = np.empty((n, s, s, 3), np.uint8)
        for i in range(n):
            out[i] = cv2.resize(imgs[i], (s, s), interpolation=interp)
        return out


def dataset_class(cfg):
    """The dataset class for `cfg.image.mode`."""
    if cfg.image.mode == "squash":
        return SquashSequenceDataset
    if cfg.image.mode == "pad":
        return SequenceDataset
    raise ValueError(f"image.mode {cfg.image.mode!r}")
