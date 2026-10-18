#!/usr/bin/env python3
"""Closed-form eye-to-hand calibration of the head camera.

Solves the robot-world/hand-eye problem  T_cam_board(i) X = Z T_base_flange(i)  with the
Kronecker-product method of Shah (2013), where
    X = T_board_flange   board pose on the flange (unknown, fixed during sampling)
    Z = T_cam_base       base pose in the camera frame (T_base_cam is its inverse)

Input:  `calib/handeye_<timestamp>/samples.jsonl` recorded with `arm_jog.py`. Only the joint
        angles (`q_motor_rad`) and the board pose in the camera frame (`rvec_cam_board`,
        `tvec_cam_board`) are used; the flange pose comes from the SDK forward kinematics.
Output: `result.json` in the same directory, with T_base_cam, T_flange_board, per-sample
        residuals and the leave-one-out spread. `handeye_joint_solve.py` uses it as its
        initial guess.

Usage:
    python tools/handeye_solve.py                            # latest handeye_* directory
    python tools/handeye_solve.py calib/handeye_<timestamp>
"""

from __future__ import annotations

import glob
import itertools
import json
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation as Rot
from realhand.arm.kinetix import ArmKinetix

CALIB_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "calib")


def latest_dir() -> str:
    """Return the most recent handeye_* directory that contains samples."""
    ds = sorted(
        d
        for d in glob.glob(os.path.join(CALIB_ROOT, "handeye_*"))
        if os.path.isfile(os.path.join(d, "samples.jsonl"))
    )
    if not ds:
        raise SystemExit(f"no handeye_*/samples.jsonl under {CALIB_ROOT}")
    return ds[-1]


def sdk_fk(kx, q_motor):
    """Flange pose from the SDK forward kinematics, as a 4x4 matrix.

    The SDK returns Pose(x, y, z, rx, ry, rz) with intrinsic zyx Euler angles.
    """
    d = kx.forward_kinematics([float(v) for v in q_motor]).model_dump()
    T = np.eye(4)
    T[:3, :3] = Rot.from_euler("zyx", [d["rz"], d["ry"], d["rx"]]).as_matrix()
    T[:3, 3] = [d["x"], d["y"], d["z"]]
    return T


def load(d: str):
    """Load the usable samples of one directory.

    Returns (rows, A, B) with A[i] = T_cam_board and B[i] = T_base_flange.
    """
    rows = [json.loads(l) for l in open(os.path.join(d, "samples.jsonl"), encoding="utf-8")]
    rows = [r for r in rows if r.get("usable")]
    side = rows[0]["arm"]
    kx = ArmKinetix.from_builtin("a7", side)
    # The SDK left-arm model starts at the shoulder base, the right-arm model on the floor.
    # Lift the left arm by the shoulder height so that both use Base_Link (see
    # robot/make_dual_a7_urdf.py).
    z_fix = 1.217 if side == "left" else 0.0
    A, B = [], []
    for r in rows:
        T = np.eye(4)
        T[:3, :3] = Rot.from_rotvec(np.array(r["rvec_cam_board"], float)).as_matrix()
        T[:3, 3] = r["tvec_cam_board"]
        Tb = sdk_fk(kx, r["q_motor_rad"])
        Tb[2, 3] += z_fix
        A.append(T)
        B.append(Tb)
    return rows, A, B


def _proj_so3(M):
    """Project a 3x3 matrix onto the nearest rotation matrix."""
    U, _, Vt = np.linalg.svd(M)
    return U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt


def solve(A, B, idx):
    """Solve A[i] X = Z B[i] over the samples in idx. Returns (X, Z).

    Rotations: R_X and R_Z are the leading singular vectors of sum_i kron(R_B, R_A).
    Translations: linear least squares on R_A t_X - t_Z = R_Z t_B - t_A.
    """
    M = np.zeros((9, 9))
    for i in idx:
        M += np.kron(B[i][:3, :3], A[i][:3, :3])
    u, _, vt = np.linalg.svd(M)
    Rx = vt[0].reshape(3, 3, order="F")
    Rz = u[:, 0].reshape(3, 3, order="F")
    if np.linalg.det(Rx) < 0:
        Rx = -Rx
    if np.linalg.det(Rz) < 0:
        Rz = -Rz
    Rx, Rz = _proj_so3(Rx), _proj_so3(Rz)
    Aeq, b = [], []
    for i in idx:
        Aeq.append(np.hstack([A[i][:3, :3], -np.eye(3)]))
        b.append(Rz @ B[i][:3, 3] - A[i][:3, 3])
    sol = np.linalg.lstsq(np.vstack(Aeq), np.concatenate(b), rcond=None)[0]
    X = np.eye(4)
    X[:3, :3] = Rx
    X[:3, 3] = sol[:3]
    Z = np.eye(4)
    Z[:3, :3] = Rz
    Z[:3, 3] = sol[3:]
    return X, Z


def rot_deg(Ra, Rb):
    """Angle between two rotation matrices, in degrees."""
    return float(np.rad2deg(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1))))


def residuals(A, B, X, Z, idx):
    """Per-sample (rotation error in deg, translation error in mm) of the board pose."""
    out = []
    for i in idx:
        P = Z @ B[i] @ np.linalg.inv(X)  # predicted board pose in the camera
        out.append(
            (rot_deg(P[:3, :3], A[i][:3, :3]), float(np.linalg.norm(P[:3, 3] - A[i][:3, 3]) * 1000))
        )
    return out


def main() -> int:
    d = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else latest_dir()
    rows, A, B = load(d)
    n = len(rows)
    if n < 3:
        raise SystemExit(f"only {n} usable samples; at least 3 are needed, 15 or more recommended")
    idx = list(range(n))
    X, Z = solve(A, B, idx)
    T_base_cam = np.linalg.inv(Z)
    res = residuals(A, B, X, Z, idx)
    er = np.array([r[0] for r in res])
    et = np.array([r[1] for r in res])

    # Leave-one-out spread of the camera pose, as a stability check.
    loo = [np.linalg.inv(solve(A, B, [i for i in idx if i != k])[1]) for k in idx] if n >= 4 else []
    loo_pos = np.array([t[:3, 3] for t in loo]) if loo else None
    loo_rot = max(
        (rot_deg(a[:3, :3], b[:3, :3]) for a, b in itertools.combinations(loo, 2)), default=0.0
    )

    z = T_base_cam[:3, :3] @ [0, 0, 1]
    result = {
        "schema": "handeye_result/v1",
        "source": os.path.basename(d),
        "n_samples": n,
        "method": "Shah 2013, Kronecker product",
        "fk_model": "SDK ArmKinetix a7, Base_Link on the floor",
        "arm": rows[0]["arm"],
        "T_base_cam": T_base_cam.tolist(),
        "T_flange_board": np.linalg.inv(X).tolist(),
        "cam_pos_base_m": T_base_cam[:3, 3].tolist(),
        "cam_optical_axis_base": z.tolist(),
        "cam_pitch_down_deg": float(np.rad2deg(np.arcsin(-z[2]))),
        "cam_yaw_deg": float(np.rad2deg(np.arctan2(z[1], z[0]))),
        "residual_rot_deg": {
            "median": float(np.median(er)),
            "max": float(er.max()),
            "per_sample": er.tolist(),
        },
        "residual_trans_mm": {
            "median": float(np.median(et)),
            "max": float(et.max()),
            "per_sample": et.tolist(),
        },
        "loo_pos_spread_mm": ((loo_pos.max(0) - loo_pos.min(0)) * 1000).tolist() if loo else None,
        "loo_rot_spread_deg": loo_rot,
        "intrinsics": rows[0]["intrinsics"],
        "board": rows[0]["board"],
    }
    out = os.path.join(d, "result.json")
    json.dump(result, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

    p = T_base_cam[:3, 3]
    print(f"{d}: {n} samples")
    print(
        f"  camera @ base  ({p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f}) m   "
        f"pitch down {result['cam_pitch_down_deg']:.1f} deg  yaw {result['cam_yaw_deg']:+.1f} deg"
    )
    print(
        f"  residual  rotation median {np.median(er):.2f} deg max {er.max():.2f} deg   "
        f"translation median {np.median(et):.1f} max {et.max():.1f} mm"
    )
    if loo:
        s = result["loo_pos_spread_mm"]
        print(
            f"  leave-one-out  position spread ({s[0]:.0f}, {s[1]:.0f}, {s[2]:.0f}) mm   "
            f"rotation spread {loo_rot:.2f} deg"
        )
    print(
        "  per sample (rotation deg / translation mm): "
        + "  ".join(f"#{i}:{a:.1f}/{b:.0f}" for i, (a, b) in enumerate(res))
    )
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
