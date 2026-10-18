"""Vendored from VITRA (`vitra/datasets/human_dataset.py`), with interpolation done once per phase.

`EpisodicDatasetCore` turns one (episode, frame) into action and state vectors: world ->
anchor-camera frame, keypoints -> MANO palm-local frame, upsampling to the action rate,
rotations -> XYZ Euler, padding to the 192 / 212 slots. `interpolate_phase` interpolates a
whole (episode, phase, hand) and `side_window_from_phase` slices each anchor out of it.
"""
import bisect
import copy
import json
import math
import os
import random
import time
from functools import lru_cache
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from .augment_utils import (
    augmentation_func,
    center_crop_short_side,
    project_to_image_space,
)

from .interp_utils import interp_mano_state
from .video_utils import load_video_decord
from .dataset_utils import (
    compute_new_intrinsics_crop, 
    compute_new_intrinsics_resize, 
    calculate_fov,
    ActionFeature,
    StateFeature,
)
from .data_utils import (
    read_dataset_statistics,
    GaussianNormalizer,
)

class EpisodicDatasetCore(object):
    """Episodes of MANO hand annotations: actions, state, instruction and optionally frames
    for one anchor frame at a time."""
    def __init__(
        self, 
        video_root, 
        annotation_file, 
        label_folder, 
        training_path=None, 
        statistics_path=None, 
        augmentation=True, 
        flip_augmentation=True, 
        set_none_ratio=0.0, 
        action_type="angle", 
        use_rel=False, 
        upsample_factor=1.0,
        target_image_height=224,
        clip_len=2000,
        state_mask_prob=0.1,
        action_past_window_size=0,
        action_future_window_size=15,
        image_past_window_size=0,
        image_future_window_size=0,
        rel_mode="step",
        load_images=True,
    ):
        self.video_root = video_root
        annotation_dict = np.load(annotation_file, allow_pickle=True)
        self.label_folder = label_folder
        self.index_frame_pair = annotation_dict['index_frame_pair'].copy()
        self.index_to_episode_id = annotation_dict['index_to_episode_id'].copy()

        if training_path is not None:
            self.training_idx = np.load(training_path, allow_pickle=True)
            self.num_valid_frames = len(self.training_idx)
        else:
            self.training_idx = None
            self.num_valid_frames = len(self.index_frame_pair)

        if statistics_path is not None:
            self.data_statistics = read_dataset_statistics(statistics_path)

        self.global_data_statistics = None
        self.clip_len = clip_len  # frames per video part; None when videos are whole files
        self.augmentation = augmentation
        self.target_image_height = target_image_height
        self.flip_augmentation = flip_augmentation
        self.set_none_ratio = set_none_ratio
        self.action_type = action_type  # "angle" (Euler angles) or "keypoints" (3D joint positions)
        self.use_rel = use_rel  # True: hand pose as relative rotations; False: absolute hand pose
        assert upsample_factor >= 1.0, "only support upsample_factor >= 1.0"
        self.upsample_factor = upsample_factor
        self.state_mask_prob = state_mask_prob  # Probability of masking state input

        self.action_past_window_size=action_past_window_size
        self.action_future_window_size=action_future_window_size
        self.image_past_window_size=image_past_window_size
        self.image_future_window_size=image_future_window_size
        # Interpolated phases, keyed by (episode, phase, hand) and reused by every anchor of
        # the phase. The oldest entry is dropped first.
        self._phase_interp_cache: dict = {}
        self._phase_interp_cache_size = 64
        self.rel_mode=rel_mode
        self.load_images=load_images

    def __len__(self):
        return self.num_valid_frames
    
    @staticmethod
    @lru_cache(maxsize=256)
    def _load_episode_npy(episode_path: str):
        """Load an episode .npy (a pickled dict). Cached."""
        return np.load(episode_path, allow_pickle=True).item()

    def _load_or_cache_episode(self, episode_id):
        """Return the episode dict and its world-to-camera extrinsics, R_w2c (T, 3, 3) and
        t_w2c (T, 3)."""
        root = self.label_folder
        epi_path = os.path.join(root, episode_id + '.npy')
        epi = self._load_episode_npy(epi_path)

        extr = epi['extrinsics']                         # world to cam, (T,4,4)
        R_w2c, t_w2c = extr[:, :3, :3], extr[:, :3, 3]

        return epi, R_w2c, t_w2c

    def _mat2euler(self, R_batch: np.ndarray) -> np.ndarray:
        """Batched XYZ-Euler conversion using SciPy."""
        flat = R_batch.reshape(-1, 3, 3)
        eul = R.from_matrix(flat).as_euler('xyz', degrees=False)
        return eul

    def side_window_from_phase(self, interp: dict, anchor_step: int, num_actions: int,
                               R_w2c, t_w2c, idx_anchor: int, *, valid_lo=None, valid_hi=None):
        """One anchor's window, sliced out of an interpolated phase.

        `anchor_step` is the anchor's index within the phase; its poses start at grid index
        `anchor_step * steps_per_control`. `num_actions + 1` poses are taken, since an action
        is the difference between consecutive poses. Poses past the end of the phase repeat
        the last one and are marked not kept.
        """
        spc = interp["steps_per_control"]
        g0 = anchor_step * spc
        need = num_actions + 1
        M = len(interp["kept"])

        take = np.arange(g0, g0 + need)
        oob = take >= M
        take = np.clip(take, 0, M - 1)

        R_mano = interp["R_mano"][take]
        t_mano = interp["t_mano"][take]
        hand_P = interp["hand_P"][take].copy()
        joints_manospace = interp["joints_manospace"][take]
        kept = interp["kept"][take].copy()
        kept[oob] = False

        # Source-frame bounds of the hand's own text clip, if it has a narrower one.
        if valid_lo is not None or valid_hi is not None:
            src = interp["phase"] + take / self.upsample_factor
            if valid_lo is not None:
                kept &= src >= valid_lo - 1e-9
            if valid_hi is not None:
                kept &= src <= valid_hi + 1e-9

        # Anchor-camera frame, applied after the interpolation in world space.
        R_cam = R_w2c[idx_anchor] @ R_mano
        t_cam = (R_w2c[idx_anchor] @ t_mano[..., None])[..., 0] + t_w2c[idx_anchor]
        pose_euler = R.from_matrix(hand_P.reshape(-1, 3, 3)).as_euler(
            "xyz", degrees=False).reshape(len(hand_P), -1)

        sl = slice(0, -1)
        nx = slice(1, None)
        return dict(
            R_cam=R_cam[sl].astype(np.float32), t_cam=t_cam[sl].astype(np.float32),
            pose_euler=pose_euler[sl].astype(np.float32), hand_P=hand_P[sl].astype(np.float32),
            joints_manospace=joints_manospace[sl].astype(np.float32), kept=kept[sl],
            R_cam_next=R_cam[nx].astype(np.float32), t_cam_next=t_cam[nx].astype(np.float32),
            pose_euler_next=pose_euler[nx].astype(np.float32),
            hand_P_next=hand_P[nx].astype(np.float32),
            joints_manospace_next=joints_manospace[nx].astype(np.float32), kept_next=kept[nx],
        )

    def interpolate_phase(self, episode_id, phase: int, stride: int, hand: str):
        """Interpolate one phase's pose sequence for one hand onto the action grid (cached).

        The grid starts at the phase's first source frame and steps by `1 / upsample_factor`,
        so control step k sits at grid index `k * stride * upsample_factor`. The result is in
        world / MANO space; the anchor-camera transform is applied when a window is sliced.
        """
        key = (str(episode_id), int(phase), str(hand))
        hit = self._phase_interp_cache.get(key)
        if hit is not None:
            return hit

        epi, _, _ = self._load_or_cache_episode(episode_id)
        side = epi.get(hand)
        if not side:
            # The hand is absent from the episode: build an all-invalid track for it, so that
            # its `action_mask` comes out all False.
            T = len(epi['video_decode_frame'])
            side = {
                'global_orient_worldspace': np.broadcast_to(np.eye(3), (T, 3, 3)).copy(),
                'transl_worldspace': np.zeros((T, 3)),
                'hand_pose': np.broadcast_to(np.eye(3), (T, 15, 3, 3)).copy(),
                'joints_worldspace': np.zeros((T, 21, 3)),
                'kept_frames': np.zeros(T, bool),
            }
        T = len(side['global_orient_worldspace'])
        if not 0 <= phase < stride:
            raise ValueError(f"phase must be in [0, {stride}), got {phase}")

        sel = np.arange(phase, T)                       # source knots, this phase onward
        kept = side['kept_frames'][sel].astype(bool)
        R_mano = side['global_orient_worldspace'][sel].copy()
        t_mano = side['transl_worldspace'][sel].copy()
        hand_P = side['hand_pose'][sel].copy()
        joints_w = side['joints_worldspace'][sel]

        if not np.all(kept):
            identity = np.eye(3, dtype=hand_P.dtype)
            hand_P[~kept] = np.broadcast_to(identity, (hand_P.shape[1], 3, 3))
            R_mano[~kept] = identity

        joints_mano = (R_mano.transpose(0, 2, 1)
                       @ (joints_w.transpose(0, 2, 1) - t_mano[..., None])).transpose(0, 2, 1)

        if self.upsample_factor > 1:
            R_mano, t_mano, hand_P, joints_mano, kept = interp_mano_state(
                R_mano, t_mano, hand_P, joints_mano, kept,
                self.upsample_factor, method="pchip")

        out = dict(R_mano=R_mano, t_mano=t_mano, hand_P=hand_P,
                   joints_manospace=joints_mano, kept=kept.astype(bool),
                   phase=int(phase), stride=int(stride),
                   steps_per_control=int(round(stride * self.upsample_factor)))
        if len(self._phase_interp_cache) >= self._phase_interp_cache_size:
            self._phase_interp_cache.pop(next(iter(self._phase_interp_cache)))
        self._phase_interp_cache[key] = out
        return out

    def _make_action_window_vec(self, win, anchor_idx: int, *, rel_mode="step", action_type="angle"):
        """Absolute and relative actions of one hand over a window.

        anchor_idx:  index of t0 inside the window.
        rel_mode:    "step" = change from t to t+1, "anchor" = change from t0 to t+1.
        action_type: "angle" = 45 Euler angles, "keypoints" = 21 x 3 MANO-space keypoints.
        Returns (action_abs, action_rel, valid); rows that are not valid are zeroed.
        """

        R_cur, t_cur  = win['R_cam'],        win['t_cam']
        R_nxt, t_nxt  = win['R_cam_next'],   win['t_cam_next']
        P_cur, P_nxt  = win['hand_P'],       win['hand_P_next']
        pose_next     = win['pose_euler_next']
        kpoints_root_next = win['joints_manospace_next']
        kept, kept_n  = win['kept'],         win['kept_next']
        W = len(t_cur)

        # absolute pose of t+1
        if action_type == "keypoints":
            abs_next = kpoints_root_next.reshape(W, -1)
        elif action_type == "angle":
            abs_next = pose_next
        action_abs = np.concatenate(
            [t_nxt,
            self._mat2euler(R_nxt),
            abs_next],
            axis=-1).astype(np.float32)
        action_abs = action_abs.reshape(W, -1)

        if rel_mode == "step":
            t_rel = t_nxt - t_cur
            R_rel = R_nxt @ R_cur.transpose(0,2,1)
            P_rel = np.matmul(P_nxt, P_cur.transpose(0,1,3,2))
            valid = kept & kept_n

        elif rel_mode == "anchor":
            t_anchor  = t_cur[anchor_idx]
            R_anchor  = R_cur[anchor_idx]
            P_anchor  = P_cur[anchor_idx]

            t_rel = t_nxt - t_anchor
            R_rel = R_nxt @ R_anchor.T
            P_rel = np.matmul(P_nxt, P_anchor.transpose(0,2,1))
            valid = kept_n & kept[anchor_idx]

        else:
            raise ValueError('rel_mode must be "step" or "anchor"')

        pose_rel = R.from_matrix(P_rel.reshape(-1,3,3)) \
                    .as_euler('xyz',False).reshape(W,45)

        action_rel = np.concatenate(
            [t_rel,
            self._mat2euler(R_rel),
            pose_rel],
            axis=-1).astype(np.float32)

        action_abs[~valid] = 0.0
        action_rel[~valid] = 0.0
        return action_abs, action_rel, valid

    def _window_indices(self, frame_id, past, future, start, end):
        """Frame indices frame_id - past .. frame_id + future, clipped to [start, end], and a
        bool array marking the ones that were clipped."""
        win = np.arange(-past, future + 1) + frame_id                  # (W,)
        oob = (win < start) | (win > end)
        return win.clip(start, end), oob

    def _resolve_video_path(self, dataset_name: str = None, video_name: str = None, part_index: int = None) -> str:
        if dataset_name=='Ego4D':
            if self.clip_len is not None:
                video_path = os.path.join(self.video_root, video_name + '_part' + str(part_index+1) +'.mp4')
            else:
                video_path = os.path.join(self.video_root, video_name +'.mp4')
            return video_path
        elif dataset_name=='EgoExo4D':
            if self.clip_len is not None:
                video_path = os.path.join(self.video_root, video_name +'_part' + str(part_index+1) +'.mp4')
            else:
                video_path = os.path.join(self.video_root, video_name +'.mp4')
            return video_path
        elif dataset_name == 'epic':
            video_id = video_name.split('_')[0]
            if self.clip_len is not None:
                video_path = os.path.join(self.video_root, video_name+ '_part' + str(part_index+1) + '.mp4')
            else:
                video_path = os.path.join(self.video_root, video_name+ '.MP4')
            return video_path
        elif dataset_name == 'somethingsomethingv2':
            if self.clip_len is not None:
                video_path = os.path.join(self.video_root, video_name+ '_part' + str(part_index+1) + '.mp4')
            else:
                video_path = os.path.join(self.video_root, video_name+'.webm')
            return video_path
        elif dataset_name == 'GigaHands':
            if self.clip_len is not None:
                base, _ = os.path.splitext(video_name)
                return os.path.join(self.video_root, base + '_part' + str(part_index+1) + '.mp4')
            return os.path.join(self.video_root, video_name)
        elif dataset_name == 'OpenTouch':
            if self.clip_len is not None:
                base, _ = os.path.splitext(video_name)
                return os.path.join(self.video_root, base + '_part' + str(part_index+1) + '.mp4')
            return os.path.join(self.video_root, video_name)
        else:
            raise ValueError(f'Unknown dataset prefix {dataset_name}')

    def _mano_forward(self, betas, pose_m):
        """Runs MANO once and returns (vertices, joints) on CPU NumPy."""
        beta_t  = torch.tensor(betas).unsqueeze(0).float().cuda()
        pose_t  = torch.tensor(pose_m).unsqueeze(0).float().cuda()
        out     = mano(betas=beta_t, hand_pose=pose_t)       # no global_orient
        return out.vertices[0].cpu().numpy(), out.joints[0].cpu().numpy()

    def _pack_state(self, R_cam, t_cam, pose_euler, idx):
        return np.concatenate([t_cam[idx],
                            self._mat2euler(R_cam[idx][None,...])[0],
                            pose_euler[idx]])

    def _grab_window_images(self,
                            episode_id: str,
                            epi: dict,
                            frame_id: int,
                            past: int,
                            future: int
                            ):
        """Decode the frames frame_id - past .. frame_id + future of an episode.

        Returns images (L, H, W, 3) uint8 RGB and mask (L,) bool, False where the window was
        clipped at the episode boundary. With `clip_len` set only the anchor frame is decoded.
        """
        dataset_name = episode_id.split('_')[0]
        decode_table = epi['video_decode_frame']                      # (T,)
        T            = len(decode_table)
        frame_in_video = epi['video_decode_frame'][frame_id]

        idx_win, oob = self._window_indices(frame_id, past, future, 0, T-1)     # (W,)

        if self.clip_len is not None:
            part_idx = frame_in_video // self.clip_len # index of the video part
            frame_in_part = frame_in_video % self.clip_len
            video_path = self._resolve_video_path(dataset_name, epi['video_name'], part_idx)
            decode_ids = [frame_in_part]
        else:
            video_path = self._resolve_video_path(dataset_name, epi['video_name'])
            decode_ids = decode_table[idx_win]

        # up to 3 attempts
        last_error = None
        for attempt in range(3):
            try:
                imgs, _ = load_video_decord(video_path, frame_index=decode_ids, rotation=False)
                break
            except Exception as e:
                last_error = e
                print(f"Warning: failed to load video frames from {video_path} (attempt {attempt+1}/3): {e}")
                time.sleep(0.1)
        else:
            raise RuntimeError(
                f"Failed to load video frames from {video_path} after 3 attempts. "
                f"decode_ids={list(np.asarray(decode_ids).astype(int))}"
            ) from last_error

        images = np.stack(imgs, axis=0)           # (L,H,W,3) uint8
        mask   = ~oob                             # (L,) bool

        return images, mask

    def _find_matching_texts(self, text_list, frame_id):
        """Texts of `text_list` = [(text, (start, end)), ...] whose half-open range
        [start, end) contains `frame_id`, and those ranges."""
        matching_texts = []
        matching_ranges = []
        
        for text, (start_frame, end_frame) in text_list:
            if start_frame <= frame_id < end_frame:
                matching_texts.append(text)
                matching_ranges.append((start_frame, end_frame))
        
        return matching_texts, matching_ranges

    def _random_select_text(
        self,
        text,
        text_rephrase,
        hand_type,
        clip_idx,
    ):

        text_list = [text]
        if text_rephrase and isinstance(text_rephrase[hand_type][clip_idx][0], list):
            text_list.extend(text_rephrase[hand_type][clip_idx][0])

        text_selected = random.choice(text_list).strip()
        if not text_selected.endswith('.'):
            text_selected += '.'
        return text_selected

    def _build_instruction(
        self,
        main_type,
        text_clip,
        text_rephrase,
        idx_win,
        oob,
        epi_len, # T
        frame_id,
        action_past_window_size,
        action_future_window_size,
    ):

        sub_type = 'right' if main_type == 'left' else 'left'
        
        # text_clip[hand] is a list like [('Place the pink cup on the table.', (0, 26)), ...]
        main_text_selected = self._random_select_text(
            text_clip[main_type][0][0],
            text_rephrase,
            main_type,
            clip_idx=0,
        )

        sub_text_list = text_clip[sub_type]
        has_sub_text = len(sub_text_list) > 0
        
        sub_oob, sub_idx_win = oob, idx_win
        sub_text_selected = "None."
        sub_win = (0, epi_len)  # Default to the full range if no text available

        if has_sub_text:
            sub_matching_texts, sub_matching_ranges = self._find_matching_texts(sub_text_list, frame_id)
            if len(sub_matching_texts) > 0:
                selected_idx = random.randrange(len(sub_matching_texts))
                sub_win = sub_matching_ranges[selected_idx]
                sub_idx_win, sub_oob = self._window_indices(
                    frame_id,
                    action_past_window_size,
                    action_future_window_size, sub_win[0], sub_win[1]-1
                )     # (W,)

                sub_text_selected = self._random_select_text(
                    sub_matching_texts[selected_idx].strip(),
                    text_rephrase,
                    sub_type,
                    clip_idx=selected_idx,
                )

        # Assign left/right based on main_type
        is_main_left = (main_type == 'left')

        idx_win_left = idx_win if is_main_left else sub_idx_win
        idx_win_right = sub_idx_win if is_main_left else idx_win
        oob_left = oob if is_main_left else sub_oob
        oob_right = sub_oob if is_main_left else oob
        
        text_left = main_text_selected if is_main_left else sub_text_selected
        text_right = sub_text_selected if is_main_left else main_text_selected
        
        start_left = 0 if is_main_left else (sub_win[0] if has_sub_text else 0)
        start_right = (sub_win[0] if has_sub_text else 0) if is_main_left else 0
        end_left = epi_len - 1 if is_main_left else (sub_win[1] - 1 if has_sub_text else epi_len - 1)
        end_right = (sub_win[1] - 1 if has_sub_text else epi_len - 1) if is_main_left else epi_len - 1

        instruction = f"Left hand: {text_left} Right hand: {text_right}"

        return instruction, idx_win_left, oob_left, idx_win_right, oob_right, start_left, end_left, start_right, end_right

    def _get_2d_traj_cur_to_end(self, idx_frame, epi, intrinsics, hand_type, image_size):
        """2D palm trajectory from `idx_frame` to the end of the episode, projected into the
        camera at `idx_frame`: (N, 2), normalised to [0, 1]."""
        H, W = image_size
        intrinsics = intrinsics.copy()
        # normalize intrinsics
        intrinsics[0] /= intrinsics[0,2]*2
        intrinsics[1] /= intrinsics[1,2]*2

        hand_joints_cur_to_end = epi[hand_type]['joints_worldspace'][idx_frame:] # (N, 21, 3)
        hand_palm_cur_to_end = np.mean(hand_joints_cur_to_end[:, [0,2,5,9,13,17], :], axis=1, keepdims=True) # (N, 1, 3)

        extrinsics = epi['extrinsics'].copy()
        extrinsics_cur = extrinsics[idx_frame] # world to cam
        R_world_to_cam = extrinsics_cur[None, :3, :3].repeat(len(hand_palm_cur_to_end), axis=0)
        t_world_to_cam = extrinsics_cur[None, :3, 3:].repeat(len(hand_palm_cur_to_end), axis=0)

        hand_palm_cur_to_end_cam = (R_world_to_cam @ hand_palm_cur_to_end.transpose(0, 2, 1) + t_world_to_cam).transpose(0, 2, 1)

        uv_palm_cur_to_end = project_to_image_space(hand_palm_cur_to_end_cam, intrinsics, (H, W)) # (N, M, 2)
        uv_palm_cur_to_end[..., 0] = np.clip(uv_palm_cur_to_end[..., 0], 0, W)
        uv_palm_cur_to_end[..., 1] = np.clip(uv_palm_cur_to_end[..., 1], 0, H)

        uv_palm_cur_to_end = uv_palm_cur_to_end.reshape(-1, 2)
        uv_palm_cur_to_end = uv_palm_cur_to_end.astype(np.float32)
        uv_palm_cur_to_end[:,0] /= W
        uv_palm_cur_to_end[:,1] /= H

        return uv_palm_cur_to_end

    def get_item_frame(
            self, episode_id, frame_id,
            action_past_window_size=0, 
            action_future_window_size=0,
            image_past_window_size=0, 
            image_future_window_size=0,
            rel_mode: str = "step",
            load_images: bool = True,
            phase: int = None,
            stride: int = None,
            anchor_step: int = None,
        ):
        """Actions, state, instruction, FoV, intrinsics and optionally frames for one anchor.

        `phase`, `stride` and `anchor_step` locate the anchor inside its phase and are
        required: actions are sliced out of the phase's single interpolation.
        """
        # 1. episode and extrinsics
        epi, R_w2c, t_w2c = self._load_or_cache_episode(episode_id)
        T  = len(epi['extrinsics'])

        # 2. frame-window indices
        idx_win, oob  = self._window_indices(frame_id,
                                        action_past_window_size,
                                        action_future_window_size, 0, T-1)     # (W,)
        W   = len(idx_win)
        main_type = epi['anno_type']
        sub_type = 'right' if main_type == 'left' else 'left'
        # 3. instruction text
        instruction, idx_win_left, oob_left, idx_win_right, oob_right, \
        start_left, end_left, start_right, end_right = self._build_instruction(
            main_type = main_type,
            text_clip = epi['text'],
            text_rephrase = epi.get('text_rephrase'),
            idx_win = idx_win,
            oob = oob,
            epi_len = T,
            frame_id = frame_id,
            action_past_window_size = action_past_window_size,
            action_future_window_size = action_future_window_size,
        )
        

        # 4. actions of both hands
        if phase is None or stride is None or anchor_step is None:
            raise ValueError("phase, stride and anchor_step are required: actions are "
                             "sliced out of a phase that was interpolated once")
        n_act = action_past_window_size + action_future_window_size + 1
        win_left = self.side_window_from_phase(
            self.interpolate_phase(episode_id, phase, stride, 'left'),
            anchor_step, n_act, R_w2c, t_w2c, frame_id,
            valid_lo=start_left, valid_hi=end_left)
        win_right = self.side_window_from_phase(
            self.interpolate_phase(episode_id, phase, stride, 'right'),
            anchor_step, n_act, R_w2c, t_w2c, frame_id,
            valid_lo=start_right, valid_hi=end_right)
        W = n_act
        idx_center = action_past_window_size          # local index of t0 in window
        
        abs_L, rel_L, msk_L = self._make_action_window_vec(
            win_left,  anchor_idx=idx_center, rel_mode=rel_mode, action_type=self.action_type
        ) 

        abs_R, rel_R, msk_R = self._make_action_window_vec(
            win_right, anchor_idx=idx_center, rel_mode=rel_mode, action_type=self.action_type
        ) 

        action_abs = np.concatenate([abs_L, abs_R], axis=1)   # (W, 2 * 51) for angles, (W, 2 * 69) for keypoints
        action_rel = np.concatenate([rel_L, rel_R], axis=1)   # (W, 2 * 51): the relative hand pose is always in angles
        action_mask = np.stack([msk_L, msk_R], axis=1)        # (W,2)

        cur_L = self._pack_state(win_left['R_cam'],
                    win_left['t_cam'],
                    win_left['pose_euler'] if self.action_type=='angle' else win_left['joints_manospace'].reshape(W, -1),
                    idx_center)

        cur_R = self._pack_state(win_right['R_cam'],
                            win_right['t_cam'],
                            win_right['pose_euler'] if self.action_type=='angle' else win_right['joints_manospace'].reshape(W, -1),
                            idx_center)

        # The 10 MANO betas per hand are dropped again by `pad_state_human`. Zeros stand in
        # when an episode has none.
        _z10 = np.zeros(10, np.float64)
        betas_L = (epi.get('left') or {}).get('beta', _z10)
        betas_R = (epi.get('right') or {}).get('beta', _z10)

        current_state       = np.concatenate([cur_L, betas_L, cur_R, betas_R])          # two hands of (6 + pose + 10 betas)
        current_state_mask  = np.array([win_left['kept'][idx_center], win_right['kept'][idx_center]])

        # 5. frames
        if load_images:
            image_list, image_mask = self._grab_window_images(
                episode_id, epi,
                frame_id,
                image_past_window_size,
                image_future_window_size
            )
            H = image_list[0].shape[0]
            W = image_list[0].shape[1]
        else:
            image_list = None
            image_mask = None
            H, W = epi['intrinsics'][1,2]*2, epi['intrinsics'][0,2]*2
        # 6. intrinsics at the image size
        dataset_name = episode_id.split('_')[0]
        intrinsics = epi['intrinsics']

        if dataset_name == 'EgoExo4D':
            # EgoExo4D frames are undistorted fisheye images with black borders, removed by a
            # centre crop: the 1408 frames are resized to 448 and cropped to 256.

            new_intrinsics = compute_new_intrinsics_crop(intrinsics, 1408, 256/448*1408, H)
            
        else:
            new_intrinsics = compute_new_intrinsics_resize(intrinsics, (H, W))

        # 7. augmentation
        if self.augmentation:
            try:
                # aspect ratio, log-uniform in [1, 2]
                aspect_ratio = np.exp(random.uniform(np.log(1.0), np.log(2.0)))
                target_size = (int(self.target_image_height * aspect_ratio), self.target_image_height)  # (W, H)
                augment_params = {
                    'tgt_aspect': aspect_ratio, 
                    'flip_augmentation': self.flip_augmentation, 
                    'set_none_ratio': self.set_none_ratio,
                }

                uv_traj = self._get_2d_traj_cur_to_end(frame_id, epi, new_intrinsics, main_type, (H, W))
                image_list, new_intrinsics, (action_abs, action_rel, action_mask), \
                (current_state, current_state_mask), instruction = \
                    augmentation_func(
                        image = image_list, 
                        intrinsics = new_intrinsics,
                        actions = (action_abs, action_rel, action_mask),
                        states = (current_state, current_state_mask),
                        captions = instruction,
                        uv_traj = uv_traj,
                        target_size = target_size,
                        augment_params = augment_params,
                        sub_type = sub_type,
                    )

            except Exception as e:
                print(f"Warning: Augmentation failed for episode {episode_id}, frame {frame_id}: {e}. Do center crop only")
                import traceback
                print(f"Warning: Augmentation failed for episode {episode_id}, frame {frame_id}")
                print(f"Exception: {type(e).__name__}: {e}")
                print(f"Traceback:\n{traceback.format_exc()}")
                image_list = center_crop_short_side(image_list[0])[None, ...]
                new_intrinsics[0][2] = 0.5 * image_list[0].shape[1]  # update the principal point
                new_intrinsics[1][2] = 0.5 * image_list[0].shape[0]
            
            if random.random() < self.state_mask_prob:
                current_state_mask = np.array([False, False])
                current_state[:] = 0.0

        fov = calculate_fov( 2 * new_intrinsics[1][2], 2 * new_intrinsics[0][2], new_intrinsics)

        if self.use_rel:
            action_list = action_rel
        else:
            # wrist increments, absolute hand pose
            rel_L = action_rel[:, :action_rel.shape[1]//2]
            rel_R = action_rel[:, action_rel.shape[1]//2:]
            abs_L = action_abs[:, :action_abs.shape[1]//2]
            abs_R = action_abs[:, action_abs.shape[1]//2:]

            action_list = np.concatenate([rel_L[:, :6], abs_L[:, 6:], rel_R[:, :6], abs_R[:, 6:]], axis=1)

        result_dict = dict(
            instruction             = instruction,
            action_list             = action_list,          # (W, 2 * 51) for angles, (W, 2 * 69) for keypoints
            action_mask             = action_mask,          # (W,2)   bool
            current_state           = current_state,        # (2 * 61,) for angles, (2 * 79,) for keypoints
            current_state_mask      = current_state_mask,   # (2,) bool
            fov                     = fov,                  # (2,) float32
            intrinsics              = new_intrinsics,       # (3,3) float32
        )
        
        if image_list is not None:
            result_dict['image_list'] = image_list          # (L, H, W, 3) uint8
        if image_mask is not None:
            result_dict['image_mask'] = image_mask          # (L,) bool
            
        return result_dict

    def set_global_data_statistics(self, global_data_statistics):
        self.global_data_statistics = global_data_statistics
        if not hasattr(self, 'gaussian_normalizer'):
            self.gaussian_normalizer = GaussianNormalizer(self.global_data_statistics)

    def transform_trajectory(
        self,
        sample_dict: dict = None,
        normalization: bool = True,
    ):
        """Optionally normalise, then pad the actions to 192 and the state to 212 slots.
        Rewrites `sample_dict` in place."""
        action_np = sample_dict["action_list"]
        state_np = sample_dict["current_state"]
       
        action_dim = action_np.shape[1]
        state_dim = state_np.shape[0]
        if normalization:
            action_np = self.gaussian_normalizer.normalize_action(action_np)
            state_np = self.gaussian_normalizer.normalize_state(state_np)

        unified_action_dim = ActionFeature.ALL_FEATURES[1]   # 192
        unified_state_dim = StateFeature.ALL_FEATURES[1]     # 212
        unified_state, unified_state_mask = pad_state_human(
            state_np,
            sample_dict["current_state_mask"],
            action_dim,
            state_dim,
            unified_state_dim
        )
        unified_action, unified_action_mask = pad_action(
            action_np,
            sample_dict["action_mask"],
            action_dim,
            unified_action_dim
        )

        sample_dict["action_list"] = unified_action
        sample_dict["action_mask"] = unified_action_mask
        sample_dict["current_state"] = unified_state
        sample_dict["current_state_mask"] = unified_state_mask
        return sample_dict


def pad_state_human(
    state: torch.Tensor,
    state_mask: torch.Tensor,
    action_dim: int,
    state_dim: int,
    unified_state_dim: int
):
    """Expand the per-hand state mask to per-dim, zero the invalid dims and pad the state.

    `state` is [state_dim]: two hands of action_dim // 2 pose values followed by 10 MANO
    betas. Only the pose values are copied, into slots [0 : action_dim]. Returns (state,
    mask), both [unified_state_dim].
    """

    current_state = torch.tensor(state, dtype=torch.float32)
    current_state_mask = torch.tensor(state_mask, dtype=torch.bool)
    
    expanded_state_mask = current_state_mask.repeat_interleave(state_dim // 2)

    current_state_masked = current_state * expanded_state_mask.to(current_state.dtype)

    padded_state = torch.zeros(unified_state_dim, dtype=current_state.dtype)
    padded_mask = torch.zeros(unified_state_dim, dtype=torch.bool)

    # left hand, without the betas
    padded_state[:action_dim//2] = current_state_masked[:action_dim//2].clone()
    padded_mask[:action_dim//2] = expanded_state_mask[:action_dim//2].clone()

    # right hand, without the betas
    padded_state[action_dim//2:action_dim] = current_state_masked[state_dim//2:state_dim//2+action_dim//2].clone()
    padded_mask[action_dim//2:action_dim] = expanded_state_mask[state_dim//2:state_dim//2+action_dim//2].clone()

    return padded_state, padded_mask

def pad_action(
    actions: torch.Tensor = None,
    action_mask: torch.Tensor = None,
    action_dim: int = None,
    unified_action_dim: int = None
):
    """Expand the per-hand action mask [T, 2] to per-dim, zero the invalid dims and pad to
    `unified_action_dim`. Returns (actions or None, mask [T, unified_action_dim])."""
    
    action_mask = torch.tensor(action_mask, dtype=torch.bool)
    
    mask_left = action_mask[:, 0].unsqueeze(1).expand(-1, action_dim // 2)
    mask_right = action_mask[:, 1].unsqueeze(1).expand(-1, action_dim // 2)
    expanded_action_mask = torch.cat((mask_left, mask_right), dim=1)

    if actions is None:
        padding_mask = torch.zeros(
            (action_mask.shape[0], unified_action_dim - action_dim),
            dtype=torch.bool
        )
        action_mask_padded = torch.cat((expanded_action_mask, padding_mask), dim=1)
        return None, action_mask_padded

    actions = torch.tensor(actions, dtype=torch.float32)
    actions_masked = actions * expanded_action_mask.to(actions.dtype)

    padding = torch.zeros(
        (actions.shape[0], unified_action_dim - action_dim),
        dtype=actions.dtype
    )

    actions_padded = torch.cat((actions_masked, padding), dim=1)
    action_mask_padded = torch.cat((expanded_action_mask, padding.bool()), dim=1)

    return actions_padded, action_mask_padded
