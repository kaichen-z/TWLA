#!/usr/bin/env python3
"""Forward kinematics of the L6 hand: six joint channels to 21 keypoints in MANO/OpenPose order.

numpy and the standard library only; the kinematics are read from the URDF in `models/l6/`.

    from l6_fk import l6_fk
    pts = l6_fk(np.array([255, 255, 255, 255, 255, 255]))   # (21, 3) metres, right hand

Training labels (robot hand to human hand representation) and the inverse mapping at
inference must both use this file, because the synthetic keypoints are defined only here.
`--selftest` compares it link by link with the pinocchio implementation in
`drivers/tools/project_hand.py`.

Input: the six L6 channels, 0-255 with 255 = extended and 0 = fully flexed, in CAN order
    [thumb flexion, thumb abduction, index, middle, ring, little].
A channel maps linearly onto the limits [lo, hi] of its URDF joint (255 -> lo); the passive
joints follow the URDF `<mimic>` elements.

Output: (21, 3) in metres in the hand_base_link frame, whose origin is on the mounting face
of the hand (123.5 mm from the middle-finger MCP axis), +z along the palm towards the middle
finger, +y towards the thumb, +x the palm normal (fingertips move towards +x when flexing).

Keypoints (MANO/OpenPose order):
    0           wrist       origin of hand_base_link
    1           thumb CMC   thumb_cmc_roll axis
    2           thumb MCP   synthetic: 56 % of the way from the thumb_cmc_pitch axis to the
                            thumb_dip axis (the ratio of the two human thumb segments; the L6
                            thumb has no joint there)
    3           thumb IP    thumb_dip axis
    4           thumb tip
    5/9/13/17   finger MCP  <finger>_mcp_pitch axis
    6/10/14/18  finger PIP  <finger>_dip axis (the L6 fingers have two joints, MCP and DIP)
    7/11/15/19  finger DIP  synthetic: midpoint between the DIP axis and the fingertip
    8/12/16/20  fingertip

Data generated with this file should record JOINTS21_VERSION.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
URDF = {s: os.path.join(_ROOT, "models", "l6", s, f"l6_{s}.urdf") for s in ("left", "right")}

# Actuated URDF joint of each channel (without the lh_/rh_ prefix).
CH_JOINT = [
    "thumb_cmc_pitch",
    "thumb_cmc_roll",
    "index_mcp_pitch",
    "middle_mcp_pitch",
    "ring_mcp_pitch",
    "pinky_mcp_pitch",
]

JOINTS21_VERSION = 2

# Keypoint definitions: (link, second link or None, ratio). With a second link the point is
# origin(link) + ratio * (origin(second link) - origin(link)).
KEYPOINTS = [
    ("hand_base_link", None, 0.0),
    ("thumb_metacarpals_base1", None, 0.0),
    ("thumb_metacarpals", "thumb_distal", 0.56),
    ("thumb_distal", None, 0.0),
    ("thumb_tip", None, 0.0),
    ("index_proximal", None, 0.0),
    ("index_distal", None, 0.0),
    ("index_distal", "index_tip", 0.5),
    ("index_tip", None, 0.0),
    ("middle_proximal", None, 0.0),
    ("middle_distal", None, 0.0),
    ("middle_distal", "middle_tip", 0.5),
    ("middle_tip", None, 0.0),
    ("ring_proximal", None, 0.0),
    ("ring_distal", None, 0.0),
    ("ring_distal", "ring_tip", 0.5),
    ("ring_tip", None, 0.0),
    ("pinky_proximal", None, 0.0),
    ("pinky_distal", None, 0.0),
    ("pinky_distal", "pinky_tip", 0.5),
    ("pinky_tip", None, 0.0),
]
KEYPOINT_NAMES = [
    "wrist",
    "thumb_cmc",
    "thumb_mcp*",
    "thumb_ip",
    "thumb_tip",
    "index_mcp",
    "index_pip",
    "index_dip*",
    "index_tip",
    "middle_mcp",
    "middle_pip",
    "middle_dip*",
    "middle_tip",
    "ring_mcp",
    "ring_pip",
    "ring_dip*",
    "ring_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip*",
    "pinky_tip",
]  # * marks a synthetic point
SYNTHETIC = [2, 7, 11, 15, 19]
# Skeleton connections, for drawing.
BONES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
]


def _rpy(r: float, p: float, y: float) -> np.ndarray:
    """Rotation matrix of URDF roll-pitch-yaw angles: Rz(y) Ry(p) Rx(r)."""
    cr, sr, cp, sp, cy, sy = (
        math.cos(r),
        math.sin(r),
        math.cos(p),
        math.sin(p),
        math.cos(y),
        math.sin(y),
    )
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def _axis_angle(axis: np.ndarray, a: float) -> np.ndarray:
    """Rotation matrix of angle a (rad) about axis."""
    k = axis / np.linalg.norm(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * (K @ K)


class L6FK:
    """Kinematic tree of one L6 hand, read from its URDF."""

    def __init__(self, side: str = "right", urdf: str | None = None) -> None:
        self.side = side
        self.urdf = urdf or URDF[side]
        pre = "lh_" if side == "left" else "rh_"
        root = ET.parse(self.urdf).getroot()
        self.joints = {}  # child link -> joint data
        for j in root.findall("joint"):
            o = j.find("origin")
            xyz = np.array(
                [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
            )
            rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
            T0 = np.eye(4)
            T0[:3, :3] = _rpy(*rpy)
            T0[:3, 3] = xyz
            ax = j.find("axis")
            lim = j.find("limit")
            mi = j.find("mimic")
            name = j.get("name")[len(pre) :]
            self.joints[j.find("child").get("link")[len(pre) :]] = dict(
                name=name,
                parent=j.find("parent").get("link")[len(pre) :],
                T0=T0,
                type=j.get("type"),
                axis=(
                    np.array([float(v) for v in ax.get("xyz").split()]) if ax is not None else None
                ),
                limit=(
                    (float(lim.get("lower")), float(lim.get("upper"))) if lim is not None else None
                ),
                mimic=(
                    (
                        mi.get("joint")[len(pre) :],
                        float(mi.get("multiplier", 1.0)),
                        float(mi.get("offset", 0.0)),
                    )
                    if mi is not None
                    else None
                ),
            )
        self.by_name = {v["name"]: v for v in self.joints.values()}
        self.links = ["hand_base_link"] + list(self.joints.keys())

    def q_from_l6(self, q6) -> dict:
        """Joint angles (rad) by joint name from the six channel values."""
        q6 = np.asarray(q6, float)
        q = {}
        for ch, name in enumerate(CH_JOINT):
            lo, hi = self.by_name[name]["limit"]
            t = 1.0 - float(np.clip(q6[ch], 0, 255)) / 255.0
            q[name] = lo + t * (hi - lo)
        for j in self.by_name.values():
            if j["mimic"] is not None:
                src, k, off = j["mimic"]
                q[j["name"]] = q[src] * k + off
        return q

    def link_T(self, q6) -> dict:
        """4x4 pose of every link in the hand_base_link frame."""
        q = self.q_from_l6(q6)
        T = {"hand_base_link": np.eye(4)}

        def get(link):
            if link in T:
                return T[link]
            j = self.joints[link]
            Tp = get(j["parent"]) @ j["T0"]
            if j["type"] in ("revolute", "continuous"):
                R = np.eye(4)
                R[:3, :3] = _axis_angle(j["axis"], q.get(j["name"], 0.0))
                Tp = Tp @ R
            T[link] = Tp
            return Tp

        for link in self.links:
            get(link)
        return T

    def __call__(self, q6) -> np.ndarray:
        """(21, 3) keypoints in the hand_base_link frame."""
        T = self.link_T(q6)
        out = np.zeros((21, 3))
        for i, (a, b, f) in enumerate(KEYPOINTS):
            p = T[a][:3, 3]
            out[i] = p if b is None else p + f * (T[b][:3, 3] - p)
        return out


_CACHE: dict = {}


def l6_fk(q6, side: str = "right") -> np.ndarray:
    """Keypoints (21, 3) in metres of one hand pose given as six channel values."""
    if side not in _CACHE:
        _CACHE[side] = L6FK(side)
    return _CACHE[side](q6)


def l6_fk_batch(Q: np.ndarray, side: str = "right") -> np.ndarray:
    """Keypoints (N, 21, 3) of N poses; rows with NaN give NaN."""
    Q = np.asarray(Q, float)
    out = np.full((len(Q), 21, 3), np.nan)
    for i in range(len(Q)):
        if np.isfinite(Q[i]).all():
            out[i] = l6_fk(Q[i], side)
    return out


def _selftest() -> int:
    """Compare with the pinocchio implementation and print the extended pose."""
    sys.path.insert(0, os.path.join(_ROOT, "drivers", "tools"))
    sys.path.insert(0, os.path.join(_ROOT, "drivers"))
    from project_hand import HandFK

    ref = HandFK("right")
    fk = L6FK("right")
    names = ref.names
    rng = np.random.default_rng(0)
    worst = 0.0
    for k in range(300):
        q6 = rng.uniform(0, 255, 6) if k > 1 else (np.full(6, 255.0) if k == 0 else np.zeros(6))
        P = ref.points(q6)
        T = fk.link_T(q6)
        for n, p in zip(names, P):
            worst = max(worst, float(np.linalg.norm(T[n][:3, 3] - p)))
    print(f"largest link position difference to the pinocchio HandFK (300 poses): {worst:.2e} m")
    pts = fk(np.full(6, 255.0))
    print("21 points, fully extended (mm):")
    for i, (n, p) in enumerate(zip(KEYPOINT_NAMES, pts)):
        print(f"  {i:2d} {n:12s} {np.round(p * 1000, 1)}")
    d = np.linalg.norm(pts[9] - pts[0]) * 1000
    print(
        f"wrist to middle MCP {d:.1f} mm; thumb CMC y={pts[1][1]*1000:+.1f} mm, "
        f"little MCP y={pts[17][1]*1000:+.1f} mm (thumb on the +y side)"
    )
    T = fk.link_T(np.full(6, 255.0))
    a, b = T["thumb_metacarpals"][:3, 3], T["thumb_distal"][:3, 3]
    f = np.dot(pts[2] - a, b - a) / np.dot(b - a, b - a)
    print(
        f"thumb point 2 at {f:.3f} of the cmc_pitch-dip segment (expected 0.560); thumb segments "
        f"{np.linalg.norm(pts[2]-pts[1])*1000:.1f} / {np.linalg.norm(pts[3]-pts[2])*1000:.1f} / "
        f"{np.linalg.norm(pts[4]-pts[3])*1000:.1f} mm; JOINTS21_VERSION={JOINTS21_VERSION}"
    )
    return 0 if worst < 1e-9 else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    sys.exit(_selftest() if a.selftest else 0)
