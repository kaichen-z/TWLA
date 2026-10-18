"""Hand-pose geometry and I/O vendored from VITRA, pruned to what `dataloader.py` uses.

    episodic_dataset.py, interp_utils.py   (episode, frame) -> action and state vectors
    dataset_utils.py, data_utils.py        slot layout, intrinsics, FoV, statistics
    augment_utils.py, video_utils.py       augmentation, video frame reads
"""
from .episodic_dataset import EpisodicDatasetCore, pad_action, pad_state_human
from .dataset_utils import ActionFeature, StateFeature

__all__ = ["EpisodicDatasetCore", "pad_action", "pad_state_human",
           "ActionFeature", "StateFeature"]
