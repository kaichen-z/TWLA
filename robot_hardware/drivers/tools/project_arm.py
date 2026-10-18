#!/usr/bin/env python3
"""Project the arm joints of a recording into the camera image.

    python drivers/tools/project_arm.py recordings/<name>

For every video frame: recorded joint angles (interpolated) + joint zero offsets -> forward
kinematics on dual_a7.urdf -> camera frame through the hand-eye calibration in
`drivers/calib/ACTIVE.json` -> pixels through the camera intrinsics. The 3D and 2D joint
positions are written to `<rec>/skeleton_2d.npz`; `overlay_arm.py` draws them. The two
steps are separate so that pinocchio and OpenCV can live in different environments.

The recording directory holds one stream per subdirectory (`camera`, `left_arm`,
`right_arm`), each a sequence of `chunk_*.npz` files with a `t_ns` timestamp array.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pinocchio as pin

_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
URDF = os.path.join(_ROOT, "drivers", "robot", "dual_a7.urdf")
CALIB = os.path.join(_ROOT, "drivers", "calib")

# Factory intrinsics of the colour stream at 1920x1080; recordings at 1280x720 have the same
# field of view, so they are scaled. Replace with the values of your own camera.
INTR_1080 = dict(
    fx=1370.053955078125,
    fy=1368.7196044921875,
    ppx=983.3646240234375,
    ppy=567.791748046875,
    width=1920,
    height=1080,
)

# Frames of the drawn skeleton, in order (prefixed with Left_/Right_).
CHAIN = [
    "Shoulder_Base_Link",
    "Shoulder_Pitch_Link",
    "Shoulder_Roll_Link",
    "Shoulder_Yaw_Link",
    "Elbow_Pitch_Link",
    "Wrist_Yaw_Link",
    "Wrist_Pitch_Link",
    "Wrist_Roll_Link",
]


def load_stream(d: str, name: str, keys) -> dict:
    """Concatenate the given arrays of all chunks of one recorded stream."""
    fs = sorted(glob.glob(os.path.join(d, name, "chunk_*.npz")))
    if not fs:
        fs = sorted(glob.glob(os.path.join(d, name, "frames_*.npz")))
    if not fs:
        raise SystemExit(f"no chunk/frames npz in {d}/{name}")
    out = {k: [] for k in keys}
    for f in fs:
        z = np.load(f)
        for k in keys:
            out[k].append(z[k])
    return {k: np.concatenate(v) for k, v in out.items()}


def intrinsics(w: int, h: int) -> np.ndarray:
    """Camera matrix scaled to a w x h image."""
    s = w / INTR_1080["width"]
    if abs(h / INTR_1080["height"] - s) > 1e-6:
        raise SystemExit(
            f"{w}x{h} and the intrinsics ({INTR_1080['width']}x{INTR_1080['height']}) "
            "have different aspect ratios"
        )
    return np.array(
        [
            [INTR_1080["fx"] * s, 0, INTR_1080["ppx"] * s],
            [0, INTR_1080["fy"] * s, INTR_1080["ppy"] * s],
            [0, 0, 1.0],
        ]
    )


def main() -> int:
    """Command line entry point."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("rec", help="recording directory, e.g. recordings/<name>")
    ap.add_argument(
        "--handeye",
        default="",
        help="use the closed-form result.json of this directory instead (no zero offsets)",
    )
    ap.add_argument(
        "--active", default="", help="use another ACTIVE.json (default drivers/calib/ACTIVE.json)"
    )
    ap.add_argument("--out", default="", help="output npz, default <rec>/skeleton_2d.npz")
    ap.add_argument("--w", type=int, default=1280)
    ap.add_argument("--h", type=int, default=720)
    a = ap.parse_args()
    rec = os.path.abspath(a.rec)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from active_calib import load as load_active

    if a.handeye:
        hedir = a.handeye if os.path.isdir(a.handeye) else os.path.join(CALIB, a.handeye)
        he = json.load(open(os.path.join(hedir, "result.json")))
        T_base_cam = np.array(he["T_base_cam"], float)
        T_cam_base = {s: np.linalg.inv(T_base_cam) for s in ("left", "right")}
        q_off = {s: np.zeros(7) for s in ("left", "right")}
        hetag = os.path.basename(hedir) + "/result (closed form, no zero offsets)"
        print(f"hand-eye {hetag}")
    else:
        cal = load_active(a.active or None)
        T_base_cam = cal["right"].T_base_cam
        T_cam_base = {s: cal[s].T_cam_base for s in ("left", "right")}
        q_off = {s: cal[s].q_off for s in ("left", "right")}
        hetag = cal["right"].label
        if not cal["left"].ok:
            print("  warning: the left arm is not calibrated in ACTIVE.json; its offsets are 0")

    K = intrinsics(a.w, a.h)
    print(
        f"intrinsics @{a.w}x{a.h}: fx {K[0,0]:.1f} fy {K[1,1]:.1f} cx {K[0,2]:.1f} cy {K[1,2]:.1f}"
    )

    cam = load_stream(rec, "camera", ["t_ns", "frame_no"])
    t_cam = cam["t_ns"].astype(np.int64)
    print(f"video {len(t_cam)} frames  {(t_cam[-1]-t_cam[0])/1e9:.1f} s")

    # Joint angles interpolated to the video frame times.
    model = pin.buildModelFromUrdf(URDF)
    data = model.createData()
    fids = {
        s: [model.getFrameId(f"{s.capitalize()}_{c}") for c in CHAIN] for s in ("left", "right")
    }

    q_at = {}
    for side in ("left", "right"):
        st = load_stream(rec, f"{side}_arm", ["t_ns", "q"])
        t, q = st["t_ns"].astype(np.int64), st["q"].astype(float)
        ok = np.isfinite(q).all(axis=1)
        t, q = t[ok], q[ok]
        order = np.argsort(t)
        t, q = t[order], q[order]
        gap = np.diff(t).max() / 1e6 if len(t) > 1 else 0
        print(f"{side}_arm {len(t)} samples  {(t[-1]-t[0])/1e9:.1f} s  largest gap {gap:.0f} ms")
        qi = np.stack([np.interp(t_cam, t, q[:, j]) for j in range(7)], axis=1)
        valid = (t_cam >= t[0]) & (t_cam <= t[-1])
        q_at[side] = (qi, valid)
        print(f"   {valid.sum()}/{len(t_cam)} frames are within the time range of the arm data")

    # Forward kinematics and projection. The configuration vector holds the left arm in
    # 0..6 and the right arm in 7..13, in motor order (checked below).
    from sim_viewer import SDK_JOINTS, PREFIX

    jids = {
        s: [model.getJointId(PREFIX[s] + nm + "_Joint") for nm in SDK_JOINTS]
        for s in ("left", "right")
    }
    assert [model.joints[j].idx_q for j in jids["left"]] == list(range(0, 7)) and [
        model.joints[j].idx_q for j in jids["right"]
    ] == list(range(7, 14)), "unexpected order of the q vector"
    n = len(t_cam)
    P3 = {s: np.full((n, len(CHAIN), 3), np.nan) for s in ("left", "right")}
    P2 = {s: np.full((n, len(CHAIN), 2), np.nan) for s in ("left", "right")}
    q = np.zeros(model.nq)
    for i in range(n):
        for side, off in (("left", 0), ("right", 7)):
            qi, valid = q_at[side]
            if not valid[i]:
                continue
            q[off : off + 7] = qi[i] + q_off[side]
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        for side in ("left", "right"):
            if not q_at[side][1][i]:
                continue
            Tcb = T_cam_base[side]
            pts = np.array([data.oMf[f].translation for f in fids[side]])
            P3[side][i] = pts
            pc = (Tcb[:3, :3] @ pts.T).T + Tcb[:3, 3]
            z = pc[:, 2]
            uv = (K @ pc.T).T
            with np.errstate(invalid="ignore", divide="ignore"):
                uv = uv[:, :2] / z[:, None]
            uv[z <= 1e-6] = np.nan  # behind the camera
            P2[side][i] = uv

    out = a.out or os.path.join(rec, "skeleton_2d.npz")
    np.savez(
        out,
        t_ns=t_cam,
        chain=np.array(CHAIN),
        left_3d=P3["left"],
        right_3d=P3["right"],
        left_2d=P2["left"],
        right_2d=P2["right"],
        K=K,
        T_base_cam=T_base_cam,
        handeye=hetag,
        q_off_left=q_off["left"],
        q_off_right=q_off["right"],
    )
    for side in ("left", "right"):
        uv = P2[side]
        inside = (
            np.isfinite(uv).all(-1)
            & (uv[..., 0] >= 0)
            & (uv[..., 0] < a.w)
            & (uv[..., 1] >= 0)
            & (uv[..., 1] < a.h)
        )
        print(
            f"{side}: joints inside the image {inside.mean()*100:.1f}%  "
            f"wrist flange pixel range u[{np.nanmin(uv[:,-1,0]):.0f},{np.nanmax(uv[:,-1,0]):.0f}] "
            f"v[{np.nanmin(uv[:,-1,1]):.0f},{np.nanmax(uv[:,-1,1]):.0f}]"
        )
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
