"""Dataset class of the SmolVLA baseline.

The dataloader's default image path (long side resized to 512, padded on the left and top)
has the same geometry as LeRobot's `resize_with_pad`, so the dataset is used as it is.
"""

from __future__ import annotations

from v4_bridge import SequenceDataset


def dataset_class(cfg):
    if cfg.image.mode != "pad":
        raise ValueError(
            f"image.mode {cfg.image.mode!r}: SmolVLA's preprocessing is the loader's "
            f"`pad` path and there is no alternative in this tree.")
    return SequenceDataset
