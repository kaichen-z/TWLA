"""Vendored from VITRA (`vitra/datasets/interp_utils.py`), pruned to `interp_mano_state`.

Local change: the output time grid steps by exactly `1 / upsample_factor` instead of being
stretched to end on the last source frame.
"""
import numpy as np
from scipy.interpolate import PchipInterpolator, interp1d

def upsample_euler_with_mask(
    points: np.ndarray,
    mask: np.ndarray,
    upsample_factor: float = 2.0,
    method: str = "linear",
    t_new: np.ndarray = None,
):
    """Upsample (N, D) `points` by `upsample_factor` within each run of valid frames.

    `mask` is the (N,) validity; `method` is "linear", "quadratic", "cubic" or "pchip";
    `t_new` optionally gives the (M,) output times in source-frame units. Returns the (M, D)
    points and the (M,) mask. Samples outside the valid runs get a constant and mask 0.
    """
    points = np.asarray(points)
    mask = np.asarray(mask).astype(bool)
    N, D = points.shape

    # The grid steps by exactly 1/upsample_factor, so it need not end on the last source frame.
    t = np.arange(N, dtype=np.float64)
    if t_new is None:
        if not np.isfinite(upsample_factor) or upsample_factor <= 0:
            raise ValueError(f"upsample_factor must be finite and > 0, got {upsample_factor}")
        M = int(np.floor((N - 1) * upsample_factor + 1e-9)) + 1
        t_new = np.arange(M, dtype=np.float64) / upsample_factor
    else:
        t_new = np.asarray(t_new, dtype=np.float64)

    new_points = np.zeros((len(t_new), D), dtype=points.dtype)
    new_mask = np.zeros(len(t_new), dtype=mask.dtype)

    # runs of valid frames, as [start, end)
    mask_diff = np.diff(mask.astype(int), prepend=0, append=0)
    starts = np.where(mask_diff == 1)[0]
    ends = np.where(mask_diff == -1)[0]

    # `e` is exclusive, so `t_new <= e` reaches up to one source frame past a run that ends
    # before the sequence does.
    for s, e in zip(starts, ends):
        t_seg = t[s:e]
        pts_seg = points[s:e]
        idx = np.where((t_new >= s) & (t_new <= e))[0]

        if len(t_seg) == 0:
            continue
        elif len(t_seg) == 1:
            # a single frame: constant
            new_points[idx] = pts_seg[0]
        else:
            t_seg_new = t_new[idx]
            
            if method == "pchip":
                interp_func = PchipInterpolator(t_seg, pts_seg, axis=0)
                new_points[idx] = interp_func(t_seg_new)
            else:
                interp_func = interp1d(
                    t_seg, pts_seg, kind=method, axis=0, bounds_error=False, fill_value="extrapolate"
                )
                new_points[idx] = interp_func(t_seg_new)

        new_mask[idx] = 1

    # samples outside the valid runs: a constant, mask 0
    zero_idx = np.where(new_mask == 0)[0]
    if len(zero_idx) > 0:
        orig_zero_idx = np.where(mask == 0)[0]
        fill_value = points[0] if len(orig_zero_idx) == 0 else points[orig_zero_idx[0]]
        new_points[zero_idx] = fill_value
        new_mask[zero_idx] = 0

    return new_points, new_mask

def rotation_6d_to_matrix(d6: np.ndarray) -> np.ndarray:
    """(..., 6) 6D rotations -> (..., 3, 3) rotation matrices, by Gram-Schmidt."""
    d6 = np.asarray(d6)
    a1, a2 = d6[..., :3], d6[..., 3:]

    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)

    dot = np.sum(b1 * a2, axis=-1, keepdims=True)
    b2 = a2 - dot * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)

    b3 = np.cross(b1, b2, axis=-1)

    return np.stack([b1, b2, b3], axis=-2)

def matrix_to_rotation_6d(matrix: np.ndarray) -> np.ndarray:
    """(..., 3, 3) rotation matrices -> (..., 6): the first two rows, flattened."""
    matrix = np.asarray(matrix)
    return matrix[..., :2, :].reshape(matrix.shape[:-2] + (6,))

def interp_mano_state(R, t, mano_R, mano_joints_manospace, mask, upsample_factor=1, method="pchip"):
    """Interpolate a MANO hand trajectory. Joint positions keep their bone lengths.

    Args:
        R: (T, 3, 3) global rotations.            t: (T, 3) global translations.
        mano_R: (T, 15, 3, 3) joint rotations.    mano_joints_manospace: (T, 21, 3).
        mask: (T,) valid frames.

    Returns the same five arrays at the upsampled length M. Rotations are interpolated as 6D
    vectors, joints are rebuilt from interpolated bone directions and lengths.
    """
    # MANO kinematic tree: parent index of each of the 21 joints
    parents = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]

    # global rotation, as 6D
    rot6d = matrix_to_rotation_6d(R)  # (T,3,3) -> (T,6)
    rot6d_interp, mask_interp = upsample_euler_with_mask(rot6d, mask, upsample_factor, method)
    R_interp = rotation_6d_to_matrix(rot6d_interp)  # (M,3,3)

    # global translation
    t_interp, _ = upsample_euler_with_mask(t, mask, upsample_factor, method)  # (M,3)

    # joint rotations, as 6D
    mano_rot6d = matrix_to_rotation_6d(mano_R)  # (T,15,3,3) -> (T,15,6)
    mano_rot6d = mano_rot6d.reshape(len(mano_R), -1)  # (T,90)
    mano_rot6d_interp, _ = upsample_euler_with_mask(mano_rot6d, mask, upsample_factor, method)
    mano_rot6d_interp = mano_rot6d_interp.reshape(-1, 15, 6)  # (M,15,6)
    mano_R_interp = rotation_6d_to_matrix(mano_rot6d_interp)  # (M,15,3,3)

    # joint positions: bone vectors and lengths relative to the parent
    T, N, _ = mano_joints_manospace.shape
    parents_arr = np.array(parents)

    bone_vectors = np.zeros_like(mano_joints_manospace)  # [T, 21, 3]
    bone_lengths = np.zeros([T, N])  # [T, 21]

    child_idx = np.where(parents_arr != -1)[0]
    parent_idx = parents_arr[child_idx]

    bone_vectors[:, child_idx, :] = (
        mano_joints_manospace[:, child_idx, :] - mano_joints_manospace[:, parent_idx, :]
    )

    bone_lengths[:, child_idx] = np.linalg.norm(bone_vectors[:, child_idx, :], axis=-1)

    # directions and lengths are interpolated separately
    bone_vectors_interp, _ = upsample_euler_with_mask(bone_vectors.reshape(T, -1), mask, upsample_factor, method)
    bone_vectors_interp = bone_vectors_interp.reshape(-1, N, 3)  # (M,21,3)
    bone_vectors_interp = bone_vectors_interp / (np.linalg.norm(bone_vectors_interp, axis=-1, keepdims=True) + 1e-8)

    bone_lengths_interp, _ = upsample_euler_with_mask(bone_lengths, mask, upsample_factor, method)

    bone_vectors_interp = bone_vectors_interp * bone_lengths_interp[..., None]

    # forward kinematics from the root, which stays at the origin
    M = bone_vectors_interp.shape[0]
    mano_joints_manospace_interp = np.zeros((M, N, 3), dtype=bone_vectors_interp.dtype)

    for i in range(1, N):
        p = parents[i]
        mano_joints_manospace_interp[:, i, :] = mano_joints_manospace_interp[:, p, :] + bone_vectors_interp[:, i, :]

    return R_interp, t_interp, mano_R_interp, mano_joints_manospace_interp, mask_interp