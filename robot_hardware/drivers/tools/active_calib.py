#!/usr/bin/env python3
"""Loader for the hand-eye calibration in use.

`drivers/calib/ACTIVE.json` selects the calibration. All code that needs the camera
extrinsics, the joint zero offsets or the hand mounting transform reads them here, so
switching calibration means editing that file only.

Calibrated quantities (solved by `handeye_joint_solve.py`):
    T_base_cam          camera pose in Base_Link (on the floor, at the bottom of the column)
    joint_offsets_rad   joint zero offsets, q_true = q_meas + offset (motor order, radians).
                        J1 of the reference arm and J7 are not observable and stay at 0.

Example:
    from active_calib import load
    cal = load()                 # {"right": Calib, "left": Calib}
    cal["right"].T_cam_base      # 4x4, base -> camera
    cal["right"].q_off           # (7,) rad, add to the measured joint angles
    cal["right"].T_wrist_hand    # 4x4, wrist roll frame -> L6 hand_base_link
    cal["right"].ok              # False: this side has no valid calibration
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np

CALIB_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "calib")
)
ACTIVE = os.path.join(CALIB_DIR, "ACTIVE.json")


@dataclass
class Calib:
    side: str
    T_base_cam: np.ndarray  # camera pose in the base frame
    T_cam_base: np.ndarray  # inverse of T_base_cam (base -> camera)
    q_off: np.ndarray  # (7,) rad, q_true = q_meas + q_off
    label: str  # source file and fit quality, for display
    ok: bool  # False if this side has no valid calibration
    warn: str = ""
    raw: dict = field(default_factory=dict, repr=False)


def _inv(T):
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


def load(path: str | None = None, quiet: bool = False) -> dict[str, Calib]:
    """Read ACTIVE.json and return {"right": Calib, "left": Calib}.

    A side without a valid calibration is returned with zero offsets and ok=False, and a
    warning is printed unless quiet is set.
    """
    p = path or ACTIVE
    with open(p, encoding="utf-8") as f:
        cfg = json.load(f)
    cam_src = os.path.join(CALIB_DIR, cfg["camera"]["from"])
    cam = json.load(open(cam_src, encoding="utf-8"))
    T_base_cam = np.array(cam["T_base_cam"], float)

    out = {}
    for side, ent in cfg["arms"].items():
        src = os.path.join(CALIB_DIR, ent["from"])
        j = json.load(open(src, encoding="utf-8"))
        off = j.get("joint_offsets_rad", {}).get(side)
        ok = bool(ent.get("ok", True)) and off is not None
        q_off = np.array(off, float) if off is not None else np.zeros(7)
        label = f"{os.path.relpath(src, CALIB_DIR)}"
        if "rms_px" in j:
            label += f"  RMS {j['rms_px']:.2f} px"
        warn = (
            ent.get("warn", "")
            if ok
            else (ent.get("warn") or "no joint zero offsets for this side; using zeros")
        )
        if warn and not quiet:
            print(f"  warning, {side}: {warn}")
        out[side] = Calib(side, T_base_cam, _inv(T_base_cam), q_off, label, ok, warn, j)
    # Hand mounting transform; defaults to the values in robot/conventions.py.
    hm = cfg.get("hand_mount")
    if hm is None:
        import sys

        sys.path.insert(
            0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
        )
        from robot.conventions import HAND_MOUNT_XYZ_IN_WRIST_ROLL_M, HAND_MOUNT_RPY, HAND_SHIM_M

        hm = {
            "xyz_in_wrist_roll_m": list(HAND_MOUNT_XYZ_IN_WRIST_ROLL_M),
            "rpy": {k: list(v) for k, v in HAND_MOUNT_RPY.items()},
            "extra_shim_m": HAND_SHIM_M,
        }
    for c in out.values():
        c.hand_mount = hm
        c.T_wrist_hand = hand_mount_T(hm, c.side)
    if not quiet:
        src = os.path.relpath(cam_src, CALIB_DIR)
        pos = T_base_cam[:3, 3]
        print(
            f"  hand-eye: camera from {src}  @({pos[0]:+.3f},{pos[1]:+.3f},{pos[2]:+.3f}) m"
            f"   board square {cam.get('board_square_m', 0.028)*1000:.2f} mm"
        )
        for side, c in out.items():
            print(
                f"    {side:5s} zero offsets (deg) "
                + " ".join(f"{v:+5.2f}" for v in np.degrees(c.q_off))
                + f"   <- {c.label}"
                + ("" if c.ok else "   [not calibrated]")
            )
        extra = float(hm.get("extra_shim_m", hm.get("shim_m", 0.0)))
        d = -hm["xyz_in_wrist_roll_m"][2] * 1000 + extra * 1000
        print(
            f"  hand mount: wrist roll axis to hand base {d:.1f} mm"
            + (f" + extra {extra*1000:.1f} mm" if extra else "")
        )
    return out


def hand_mount_T(hm: dict, side: str) -> np.ndarray:
    """Return the 4x4 transform from the wrist roll joint frame to hand_base_link.

    Follows the URDF convention: rpy gives the orientation and xyz the translation in the
    parent frame. An extra shim moves the hand further out along the tool axis.
    """
    r, p, y = hm["rpy"][side]
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    R = np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(hm["xyz_in_wrist_roll_m"], float) + np.array(
        [0.0, 0.0, -float(hm.get("extra_shim_m", hm.get("shim_m", 0.0)))]
    )
    return T


if __name__ == "__main__":
    load()
