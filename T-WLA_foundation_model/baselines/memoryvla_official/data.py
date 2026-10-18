"""Dataset wrapper for the MemoryVLA baseline: frames are resized to 224x224 without keeping the aspect ratio.

This is the `resize-naive` strategy of the Prismatic VLM that CogACT and MemoryVLA use. The model then passes each
frame through upstream's `image_transform`, which applies the DINOv2 and SigLIP normalisations.
"""

from __future__ import annotations

import numpy as np

# `test.py` puts this directory on `sys.path` and imports `data` and `v4_bridge` as top-level modules.
from v4_bridge import SequenceDataset


class SquashSequenceDataset(SequenceDataset):
    """`SequenceDataset` with a plain resize in place of the padded one."""

    def _to_square(self, imgs: np.ndarray) -> np.ndarray:
        """(L, H, W, 3) uint8 -> (L, S, S, 3), aspect ratio not preserved."""
        import cv2

        s = self.cfg.vlm.image_size
        n, h, w = imgs.shape[:3]
        if (h, w) == (s, s):
            return imgs
        # INTER_AREA only when shrinking in both directions; on upscaling it acts like nearest-neighbour.
        interp = cv2.INTER_AREA if (h >= s and w >= s) else cv2.INTER_LINEAR
        out = np.empty((n, s, s, 3), np.uint8)
        for i in range(n):
            out[i] = cv2.resize(imgs[i], (s, s), interpolation=interp)
        return out


def dataset_class(cfg):
    """The dataset class for `cfg.image.mode`; only 'squash' is supported."""
    if cfg.image.mode == "squash":
        return SquashSequenceDataset
    raise ValueError(f"image.mode {cfg.image.mode!r}: this baseline supports only 'squash'")
