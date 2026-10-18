"""Joint angles -> 21 keypoints per hand, by MANO forward kinematics.

Per hand the angle layout is 51 = wrist dt(3) + dR(3) + 45 absolute Euler joint angles,
the keypoints layout 69 = wrist dt(3) + dR(3) + 63 absolute joint coordinates. Only the
finger block is converted. Both hands go through the right-hand MANO model: a left
hand's pose is used as it is and the output is mirrored in x.
"""

from __future__ import annotations

import numpy as np

# MANO joint i -> index among the 21 keypoints. MANO orders its 16 joints as [wrist,
# index x3, middle x3, pinky x3, ring x3, thumb x3]; the keypoints are [wrist, thumb x4,
# index x4, middle x4, ring x4, pinky x4], the 4th of each finger being the tip.
MANO2DATA = [0, 5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3]

# MANO has no fingertip joints; the tips are mesh vertices (thumb, index, middle, ring,
# pinky) and go to keypoints 4/8/12/16/20.
TIP_VERTS = [744, 320, 443, 554, 671]
TIP_DATA = [4, 8, 12, 16, 20]

# x-mirror for the left hand, applied to the output only.
MIRROR = np.diag([-1.0, 1.0, 1.0]).astype(np.float64)

N_JOINTS = 15          # MANO articulated joints
N_KP = 21              # keypoints per hand


def euler_to_matrix(e: np.ndarray) -> np.ndarray:
    """(..., 45) xyz-Euler angles in radians -> (..., 15, 3, 3), the loader's convention."""
    from scipy.spatial.transform import Rotation as R

    e = np.asarray(e, np.float64)
    lead = e.shape[:-1]
    flat = e.reshape(-1, N_JOINTS, 3).reshape(-1, 3)
    return R.from_euler("xyz", flat, degrees=False).as_matrix().reshape(
        *lead, N_JOINTS, 3, 3)


class ManoFK:
    """MANO forward kinematics returning the 21 keypoints, wrist-centred.

    `flat_hand_mean=True` and `use_pca=False` match how the annotations were made.
    """

    def __init__(self, weights_dir: str, device: str = "cpu"):
        import smplx
        import torch

        self.torch = torch
        self.device = device
        # `weights_dir` is the parent of `mano/`; smplx appends the model type itself.
        import os as _os
        _ext = "npz" if _os.path.exists(_os.path.join(weights_dir, "mano", "MANO_RIGHT.npz")) else "pkl"
        self.model = smplx.create(weights_dir, "mano", is_rhand=True, use_pca=False, ext=_ext,
                                  flat_hand_mean=True, batch_size=1).to(device)
        self.model.eval()

    def keypoints(self, betas: np.ndarray, pose_rotmat: np.ndarray,
                  left: bool) -> np.ndarray:
        """betas (B, 10), pose_rotmat (B, 15, 3, 3) -> (B, 21, 3) in metres, wrist at 0.

        `left` mirrors the whole output, so a batch holds one hand side only.
        """
        from scipy.spatial.transform import Rotation as R

        torch = self.torch
        betas = np.asarray(betas, np.float64)
        pose_rotmat = np.asarray(pose_rotmat, np.float64)
        B = pose_rotmat.shape[0]
        if pose_rotmat.shape[1:] != (N_JOINTS, 3, 3):
            raise ValueError(f"pose_rotmat must be (B,15,3,3), got {pose_rotmat.shape}")
        if betas.shape != (B, 10):
            raise ValueError(f"betas must be ({B},10), got {betas.shape}")

        # smplx takes axis-angle, 45 per sample. The pose is not mirrored for a left hand.
        aa = R.from_matrix(pose_rotmat.reshape(-1, 3, 3)).as_rotvec().reshape(B, 45)
        # smplx allocates the default `global_orient` and `transl` at the model's
        # batch_size (1) and does not broadcast them, so pass zeros of the right size.
        # Zero orientation gives the keypoints in the palm-local frame.
        z3 = torch.zeros(B, 3, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            out = self.model(
                betas=torch.as_tensor(betas, dtype=torch.float32, device=self.device),
                hand_pose=torch.as_tensor(aa, dtype=torch.float32, device=self.device),
                global_orient=z3, transl=z3)
        j = out.joints.detach().cpu().numpy().astype(np.float64)        # (B, 16, 3)
        v = out.vertices.detach().cpu().numpy().astype(np.float64)      # (B, 778, 3)

        kp = np.empty((B, N_KP, 3), np.float64)
        root = j[:, 0:1, :]                     # wrist-centred, as `joints_manospace` is
        kp[:, MANO2DATA, :] = j - root
        kp[:, TIP_DATA, :] = v[:, TIP_VERTS, :] - root
        return kp @ MIRROR if left else kp


def angle_to_keypoints(action_angle: np.ndarray, betas_l: np.ndarray,
                       betas_r: np.ndarray, fk: ManoFK) -> np.ndarray:
    """(B, T, 192) angle layout -> (B, T, 192) keypoints layout.

    action_angle  (B, T, 192)  denormalised; 0:51 left, 51:102 right, the rest padding
    betas_l/r     (B, 10)      MANO shape, per sample

    In the output the left hand is at 0:69 and the right at 69:138. The six wrist
    channels of each hand are copied unchanged.
    """
    action_angle = np.asarray(action_angle, np.float64)
    B, T = action_angle.shape[:2]
    out = np.zeros((B, T, 192), np.float64)

    for lo_a, lo_k, betas, left in ((0, 0, betas_l, True), (51, 69, betas_r, False)):
        out[:, :, lo_k:lo_k + 6] = action_angle[:, :, lo_a:lo_a + 6]      # wrist, verbatim
        P = euler_to_matrix(action_angle[:, :, lo_a + 6:lo_a + 51])       # (B,T,15,3,3)
        kp = fk.keypoints(np.repeat(betas, T, axis=0), P.reshape(B * T, N_JOINTS, 3, 3),
                          left=left)
        out[:, :, lo_k + 6:lo_k + 69] = kp.reshape(B, T, N_KP * 3)
    return out
