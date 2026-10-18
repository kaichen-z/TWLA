#!/usr/bin/env python3
"""Project the hand skeleton of a recording into the camera image.

    python drivers/tools/project_hand.py recordings/<name>

For every video frame: arm joint angles + zero offsets -> forward kinematics on dual_a7.urdf
-> wrist roll frame -> hand mounting transform (`robot/conventions.py`) -> L6 forward
kinematics from the recorded channel values -> camera frame through the hand-eye calibration
in `drivers/calib/ACTIVE.json` -> pixels. Writes `<rec>/hand_2d.npz`; `overlay_hand.py`
draws it. See `project_arm.py` for the recording layout.

Channel values map linearly onto the URDF joint limits (255 = extended = lower limit); the
distal joints follow the URDF mimic ratios. `--thumb-flip`, `--mount-xyz` and
`--mount-extra` exist to check alternative conventions against the video.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pinocchio as pin

_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
ARM_URDF = os.path.join(_ROOT, "drivers", "robot", "dual_a7.urdf")
HAND_URDF = {s: os.path.join(_ROOT, "models", "l6", s, f"l6_{s}.urdf") for s in ("left", "right")}
CALIB = os.path.join(_ROOT, "drivers", "calib")
# Factory intrinsics of the colour stream at 1920x1080; replace with your own camera.
INTR_1080 = dict(
    fx=1370.053955078125,
    fy=1368.7196044921875,
    ppx=983.3646240234375,
    ppy=567.791748046875,
    width=1920,
    height=1080,
)

# Actuated URDF joint of each channel (without the lh_/rh_ prefix).
CH_JOINT = [
    "thumb_cmc_pitch",
    "thumb_cmc_roll",
    "index_mcp_pitch",
    "middle_mcp_pitch",
    "ring_mcp_pitch",
    "pinky_mcp_pitch",
]
# Passive joints: (actuated joint, ratio), from the <mimic> elements of the URDF.
MIMIC = {
    "index_dip": ("index_mcp_pitch", 1.125676),
    "middle_dip": ("middle_mcp_pitch", 1.125676),
    "ring_dip": ("ring_mcp_pitch", 1.125676),
    "pinky_dip": ("pinky_mcp_pitch", 1.125676),
    "thumb_dip": ("thumb_cmc_pitch", 1.226495),
}
# Channels whose direction is reversed; `--thumb-flip` sets the two thumb channels.
THUMB_FLIP = [False, False, False, False, False, False]

# Hand mounting transform, from `robot/conventions.py`; `--mount-xyz` and `--mount-extra`
# override it on the command line.
import sys as _sys  # noqa: E402

_sys.path.insert(0, os.path.join(_ROOT, "drivers"))
from robot.conventions import (
    HAND_MOUNT_XYZ_IN_WRIST_ROLL_M as _HM_XYZ,
    HAND_MOUNT_RPY as _HM_RPY,
    HAND_SHIM_M as _HM_SHIM,
)  # noqa: E402,E501

MOUNT_XYZ = tuple(_HM_XYZ)
MOUNT_EXTRA_M = float(_HM_SHIM)
MOUNT_RPY = {k: tuple(v) for k, v in _HM_RPY.items()}

# Drawn chains: one per finger, each starting at the hand base (link names without prefix).
FINGERS = {
    "thumb": [
        "hand_base_link",
        "thumb_metacarpals_base1",
        "thumb_metacarpals",
        "thumb_distal",
        "thumb_tip",
    ],
    "index": ["hand_base_link", "index_proximal", "index_distal", "index_tip"],
    "middle": ["hand_base_link", "middle_proximal", "middle_distal", "middle_tip"],
    "ring": ["hand_base_link", "ring_proximal", "ring_distal", "ring_tip"],
    "pinky": ["hand_base_link", "pinky_proximal", "pinky_distal", "pinky_tip"],
}


def load_stream(d, name, keys):
    """Concatenate the given arrays of all chunks of one recorded stream."""
    fs = sorted(glob.glob(os.path.join(d, name, "chunk_*.npz"))) or sorted(
        glob.glob(os.path.join(d, name, "frames_*.npz"))
    )
    if not fs:
        raise SystemExit(f"no data in {d}/{name}")
    o = {k: [] for k in keys}
    for f in fs:
        z = np.load(f)
        for k in keys:
            o[k].append(z[k])
    return {k: np.concatenate(v) for k, v in o.items()}


def mount_T(side: str) -> np.ndarray:
    """4x4 transform from the wrist roll frame to the hand base."""
    r, p, y = MOUNT_RPY[side]
    T = np.eye(4)
    T[:3, :3] = pin.rpy.rpyToMatrix(r, p, y)
    T[:3, 3] = np.asarray(MOUNT_XYZ, float) + np.array([0.0, 0.0, -MOUNT_EXTRA_M])
    return T


class HandFK:
    """Forward kinematics of one L6 hand with pinocchio."""

    def __init__(self, side: str):
        self.side = side
        self.m = pin.buildModelFromUrdf(HAND_URDF[side])
        self.d = self.m.createData()
        self.pre = "lh_" if side == "left" else "rh_"
        self.idx = {}
        for j in range(1, self.m.njoints):
            self.idx[self.m.names[j][len(self.pre) :]] = self.m.joints[j].idx_q
        self.lim = {
            n: (self.m.lowerPositionLimit[i], self.m.upperPositionLimit[i])
            for n, i in self.idx.items()
        }
        self.chain = {k: [self.m.getFrameId(self.pre + n) for n in v] for k, v in FINGERS.items()}
        self.names = [n for v in FINGERS.values() for n in v]
        self.fids = [self.m.getFrameId(self.pre + n) for n in self.names]

    def q_from_l6(self, pos6) -> np.ndarray:
        """Configuration vector from the six channel values (0-255)."""
        q = np.zeros(self.m.nq)
        for ch, name in enumerate(CH_JOINT):
            lo, hi = self.lim[name]
            t = 1.0 - float(np.clip(pos6[ch], 0, 255)) / 255.0
            if THUMB_FLIP[ch]:
                t = 1.0 - t
            q[self.idx[name]] = lo + t * (hi - lo)
        for name, (src, k) in MIMIC.items():
            q[self.idx[name]] = q[self.idx[src]] * k
        return q

    def points(self, pos6) -> np.ndarray:
        """Positions of the chain frames (see FINGERS) in the hand base frame."""
        pin.forwardKinematics(self.m, self.d, self.q_from_l6(pos6))
        pin.updateFramePlacements(self.m, self.d)
        return np.array([self.d.oMf[f].translation for f in self.fids])


def main() -> int:
    """Command line entry point."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("rec")
    ap.add_argument(
        "--handeye",
        default="",
        help="use the closed-form result.json of this directory instead (no zero offsets)",
    )
    ap.add_argument(
        "--thumb-flip",
        action="store_true",
        help="reverse the direction of the two thumb channels",
    )
    ap.add_argument(
        "--active", default="", help="use another ACTIVE.json (default drivers/calib/ACTIVE.json)"
    )
    ap.add_argument(
        "--mount-xyz", default="", help="override the mounting translation, e.g. -0.0225,0,-0.0705"
    )
    ap.add_argument(
        "--mount-extra",
        type=float,
        default=None,
        help="extra shim thickness (m), further out along the tool axis",
    )
    ap.add_argument("--out", default="")
    ap.add_argument("--w", type=int, default=1280)
    ap.add_argument("--h", type=int, default=720)
    ap.add_argument(
        "--frames",
        default="",
        help="compute only these frames (comma separated)",
    )
    a = ap.parse_args()
    if a.thumb_flip:
        THUMB_FLIP[0] = THUMB_FLIP[1] = True
        print("  thumb channels reversed")
    rec = os.path.abspath(a.rec)

    hedir, T_cam_base = {}, {}
    q_off = {"left": np.zeros(7), "right": np.zeros(7)}
    if a.handeye:
        for side in ("left", "right"):
            name = a.handeye
            hd = name if os.path.isdir(name) else os.path.join(CALIB, name)
            he = json.load(open(os.path.join(hd, "result.json")))
            hedir[side] = os.path.basename(hd) + "/result (closed form, no zero offsets)"
            T_cam_base[side] = np.linalg.inv(np.array(he["T_base_cam"], float))
            print(f"{side}: hand-eye {hedir[side]}")
    else:
        from active_calib import load as load_active

        cal = load_active(a.active or None)
        for side in ("left", "right"):
            c = cal[side]
            T_cam_base[side] = c.T_cam_base
            q_off[side] = c.q_off
            hedir[side] = c.label + ("" if c.ok else " [not calibrated]")

    s = a.w / INTR_1080["width"]
    K = np.array(
        [
            [INTR_1080["fx"] * s, 0, INTR_1080["ppx"] * s],
            [0, INTR_1080["fy"] * s, INTR_1080["ppy"] * s],
            [0, 0, 1.0],
        ]
    )

    cam = load_stream(rec, "camera", ["t_ns"])
    t_cam = cam["t_ns"].astype(np.int64)
    sel = np.array([int(x) for x in a.frames.split(",")]) if a.frames else np.arange(len(t_cam))

    arm = pin.buildModelFromUrdf(ARM_URDF)
    adata = arm.createData()
    aflange = {s_: arm.getFrameId(f"{s_.capitalize()}_Wrist_Roll_Link") for s_ in ("left", "right")}

    q_at, p_at, hands = {}, {}, {}
    for side in ("left", "right"):
        st = load_stream(rec, f"{side}_arm", ["t_ns", "q"])
        ok = np.isfinite(st["q"]).all(axis=1)
        tt, qq = st["t_ns"][ok].astype(np.int64), st["q"][ok].astype(float)
        o = np.argsort(tt)
        tt, qq = tt[o], qq[o]
        q_at[side] = (
            np.stack([np.interp(t_cam, tt, qq[:, j]) for j in range(7)], 1),
            (t_cam >= tt[0]) & (t_cam <= tt[-1]),
        )
        hp = load_stream(rec, f"{side}_hand_pos", ["t_ns", "pos"])
        ht, hv = hp["t_ns"].astype(np.int64), hp["pos"].astype(float)
        o = np.argsort(ht)
        ht, hv = ht[o], hv[o]
        p_at[side] = np.stack([np.interp(t_cam, ht, hv[:, j]) for j in range(6)], 1)
        hands[side] = HandFK(side)
        print(
            f"{side}: arm {len(tt)} samples, hand {len(ht)} samples, "
            f"hand value range {hv.min(0).astype(int)}~{hv.max(0).astype(int)}"
        )

    npts = len(hands["left"].names)
    n = len(t_cam)
    P3 = {s_: np.full((n, npts, 3), np.nan) for s_ in ("left", "right")}
    P2 = {s_: np.full((n, npts, 2), np.nan) for s_ in ("left", "right")}
    q = np.zeros(arm.nq)
    global MOUNT_XYZ, MOUNT_EXTRA_M
    if a.mount_extra is not None:
        MOUNT_EXTRA_M = a.mount_extra
    if a.mount_xyz:
        MOUNT_XYZ = tuple(float(x) for x in a.mount_xyz.split(","))
    Tm = {s_: mount_T(s_) for s_ in ("left", "right")}
    mnt = {
        s_: {"xyz": list(MOUNT_XYZ), "rpy": list(MOUNT_RPY[s_]), "extra": MOUNT_EXTRA_M}
        for s_ in ("left", "right")
    }
    for i in sel:
        for side, off in (("left", 0), ("right", 7)):
            if q_at[side][1][i]:
                q[off : off + 7] = q_at[side][0][i] + q_off[side]
        pin.forwardKinematics(arm, adata, q)
        pin.updateFramePlacements(arm, adata)
        for side in ("left", "right"):
            if not q_at[side][1][i]:
                continue
            Tf = np.eye(4)
            Tf[:3, :3] = adata.oMf[aflange[side]].rotation
            Tf[:3, 3] = adata.oMf[aflange[side]].translation
            Tbh = Tf @ Tm[side]
            ph = hands[side].points(p_at[side][i])
            pb = (Tbh[:3, :3] @ ph.T).T + Tbh[:3, 3]
            P3[side][i] = pb
            Tcb = T_cam_base[side]
            pc = (Tcb[:3, :3] @ pb.T).T + Tcb[:3, 3]
            uv = (K @ pc.T).T
            with np.errstate(invalid="ignore", divide="ignore"):
                uv = uv[:, :2] / pc[:, 2:3]
            uv[pc[:, 2] <= 1e-6] = np.nan
            P2[side][i] = uv

    out = a.out or os.path.join(rec, "hand_2d.npz")
    np.savez(
        out,
        t_ns=t_cam,
        names=np.array(hands["left"].names),
        chains=json.dumps(
            {k: [hands["left"].names.index(n) for n in v] for k, v in FINGERS.items()}
        ),
        left_3d=P3["left"],
        right_3d=P3["right"],
        left_2d=P2["left"],
        right_2d=P2["right"],
        K=K,
        mount=json.dumps(mnt),
        handeye=json.dumps(hedir),
    )
    for side in ("left", "right"):
        print(
            f"{side}: hand-eye {hedir[side]}  mount xyz={MOUNT_XYZ} rpy={MOUNT_RPY[side]} "
            f"extra {MOUNT_EXTRA_M*1000:.0f} mm"
        )
    for side in ("left", "right"):
        uv = P2[side][sel]
        fin = np.isfinite(uv).all(-1)
        print(
            f"{side}: valid points {fin.mean()*100:.0f}%  "
            f"u range [{np.nanmin(uv[...,0]):.0f}, {np.nanmax(uv[...,0]):.0f}]"
        )
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
