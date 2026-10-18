"""Robot teleoperation episodes as a data source: 7 arm joints + 6 hand channels.

`RobotEpisodicCore` offers the part of the `EpisodicDatasetCore` interface that
`SequenceDataset` uses. Actions are 13-D absolute joint-space targets (the commands), the
state is the 13 measured joints at the anchor. Both sit in slots [6:19] of the 192 / 212
vectors; every other slot is masked.
"""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np

ACTION_SLOTS = 192
STATE_SLOTS = 212
ROBOT_DIM = 13
SLOT0 = 6                      # robot dims live at [SLOT0 : SLOT0 + ROBOT_DIM]


class RobotEpisodicCore:
    """Reads `label_folder/<episode_id>.npz`:

        arm_q_measured (T, 7), hand_pos_measured (T, 6)   measured joints -> state
        arm_q_cmd (T, 7), hand_cmd (T, 6)                 commanded targets -> actions
        intrinsics (3, 3), fov (2,) in radians, text (str)
    """
    def __init__(self, annotation_file: str, label_folder: str, *, chunk_size: int = 16,
                 action_hz: float = 80.0, source_fps: float = 30.0, image_size: int = 512):
        z = np.load(annotation_file, allow_pickle=True)
        self.index_frame_pair = z["index_frame_pair"].copy()          # (N, 2) episode_idx, frame
        self.index_to_episode_id = z["index_to_episode_id"].copy()    # (E,) str
        self.label_folder = label_folder
        self.chunk_size = int(chunk_size)
        self.step = float(source_fps) / float(action_hz)               # source frames per action step
        self.image_size = int(image_size)

    # ------------------------------------------------------------------ episodes
    @staticmethod
    @lru_cache(maxsize=512)
    def _load(path: str) -> dict:
        z = np.load(path, allow_pickle=True)
        d = {k: z[k] for k in z.files}
        d["text"] = str(d["text"]) if "text" in d else ""
        return d

    def episode(self, episode_id: str) -> dict:
        return self._load(os.path.join(self.label_folder, str(episode_id) + ".npz"))

    # ------------------------------------------------------------------ one anchor
    def get_item_frame(self, episode_id, frame_id, action_past_window_size=0,
                       action_future_window_size=15, image_past_window_size=0,
                       image_future_window_size=0, rel_mode="step", load_images=True,
                       phase=None, stride=None, anchor_step=None):
        """One anchor: `chunk_size` commanded targets starting at `frame_id`, linearly
        interpolated at the action rate, and the measured joints at `frame_id` as state."""
        if action_past_window_size != 0:
            raise ValueError("robot core: past action window is not supported")
        epi = self.episode(episode_id)
        cmd = np.concatenate([epi["arm_q_cmd"], epi["hand_cmd"]], axis=1).astype(np.float64)       # (T, 13)
        meas = np.concatenate([epi["arm_q_measured"], epi["hand_pos_measured"]], axis=1)          # (T, 13)
        T = cmd.shape[0]
        if not 0 <= frame_id < T:
            raise IndexError(f"{episode_id}: frame {frame_id} outside 0..{T - 1}")
        n_act = action_future_window_size + 1
        if n_act != self.chunk_size:
            raise ValueError(f"robot core built for chunk {self.chunk_size}, asked for {n_act}")
        # fractional source-frame position of each action; the first is the anchor itself
        pos = frame_id + np.arange(n_act) * self.step
        lo = np.floor(pos).astype(int)
        hi = lo + 1
        valid = hi <= T - 1                       # both interpolation ends inside the episode
        # exact hit on the last row is fine too (weight 0 on `hi`)
        valid |= (lo == T - 1) & (pos == lo)
        w = (pos - lo)[:, None]
        lo_c = np.clip(lo, 0, T - 1); hi_c = np.clip(hi, 0, T - 1)
        actions = (1.0 - w) * cmd[lo_c] + w * cmd[hi_c]
        actions[~valid] = cmd[T - 1]              # overrun: repeat the last command, masked below
        mask = np.repeat(valid[:, None], ROBOT_DIM, axis=1)
        state = meas[frame_id].astype(np.float64)
        return {
            "action_list": actions.astype(np.float32),                  # (chunk_size, 13)
            "action_mask": mask,                                        # (chunk_size, 13) bool
            "current_state": state.astype(np.float32),                  # (13,)
            "current_state_mask": np.ones(ROBOT_DIM, dtype=bool),
            "fov": np.asarray(epi["fov"], np.float32),
            "intrinsics": np.asarray(epi["intrinsics"], np.float32),
            "instruction": epi["text"],
            "image_list": None,                   # images always come from the JPEG store
        }

    # ------------------------------------------------------------------ unified layout
    def transform_trajectory(self, sample_dict: dict, normalization: bool = False) -> dict:
        """Scatter the 13 dims into the 192 / 212 slot vectors. Rewrites `sample_dict` in place."""
        if normalization:
            raise ValueError("normalisation happens in the model, from its buffers")
        a, am = sample_dict["action_list"], sample_dict["action_mask"]
        s, sm = sample_dict["current_state"], sample_dict["current_state_mask"]
        A = np.zeros((a.shape[0], ACTION_SLOTS), np.float32); AM = np.zeros((a.shape[0], ACTION_SLOTS), bool)
        A[:, SLOT0:SLOT0 + ROBOT_DIM] = a; AM[:, SLOT0:SLOT0 + ROBOT_DIM] = am
        S = np.zeros(STATE_SLOTS, np.float32); SM = np.zeros(STATE_SLOTS, bool)
        S[SLOT0:SLOT0 + ROBOT_DIM] = s; SM[SLOT0:SLOT0 + ROBOT_DIM] = sm
        sample_dict["action_list"], sample_dict["action_mask"] = A, AM
        sample_dict["current_state"], sample_dict["current_state_mask"] = S, SM
        return sample_dict
