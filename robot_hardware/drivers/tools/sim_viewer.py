#!/usr/bin/env python3
"""Browser-based 3D viewer of the A7 dual arm with L6 hands.

Kinematics come from `robot/dual_a7.urdf`, the model the SDK uses for FK and IK. Joint
order is motor order, so angles from `A7.get_angles()` can be passed in directly.

The URDF is loaded without geometry; display meshes are attached separately:
    arm      `models/a7/meshes/`, one mesh per joint, attached to the joint frame
    hand     `models/l6/<side>/meshes/`, posed by `hand/l6_fk.py` and mounted on the wrist
             roll link with `conventions.hand_mount_T()`
    column   drawn as a cylinder
Without the mesh files a skeleton of spheres and cylinders is drawn instead. The spheres
mark the joint origins in either case.

This is a kinematic viewer: it has no dynamics and no collision checking.

Usage:
    python tools/sim_viewer.py [--no-open]

Commands:
    home                     all joints to zero
    calib                    right arm to the calibration start pose
    r4 45                    right arm J4 to 45 deg (l4 for the left arm)
    r 0 -30 0 90 0 0 0       all seven joints of the right arm, in degrees
    rh 255 255 80 80 80 80   six channels of the right hand, 255 = extended (lh: left hand)
    open | fist              both hands extended / closed
    demo [l|r]               swing each joint of one arm in turn
    info | lim | q           print poses / print joint limits / quit
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pinocchio as pin

_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, "hand"))
from robot.conventions import (  # noqa: E402
    HAND_MOUNT_XYZ_IN_WRIST_ROLL_M,
    HAND_SHIM_M,
    JOINT_NAMES,
    LIMITS_LEFT,
    LIMITS_RIGHT,
    hand_mount_T,
)
from l6_fk import L6FK  # noqa: E402

URDF_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "robot"))
URDF = os.path.join(URDF_DIR, "dual_a7.urdf")

# Joint names in the URDF, without the Left_/Right_ prefix and the _Joint suffix.
SDK_JOINTS = [
    "Shoulder_Pitch",
    "Shoulder_Roll",
    "Shoulder_Yaw",
    "Elbow_Pitch",
    "Wrist_Yaw",
    "Wrist_Pitch",
    "Wrist_Roll",
]
PREFIX = {"left": "Left_", "right": "Right_"}

# Mating face of the hand base, in the wrist roll joint frame.
MOUNT_FACE_IN_J7 = np.array(HAND_MOUNT_XYZ_IN_WRIST_ROLL_M) + np.array([0.0, 0.0, -HAND_SHIM_M])
TOOL_ORIGIN_IN_J7 = MOUNT_FACE_IN_J7

# Start pose for hand-eye calibration (motor order, degrees).
CALIB_Q_RIGHT_DEG = [15.8, -31.9, 18.6, 111.8, 36.3, 44.0, 3.1]

USE_ARM_MESHES = True
ARM_MESH_DIR = os.path.join(_ROOT, "models", "a7", "meshes")
# Mesh attached to joint k (file `<side>_<name>.STL`).
ARM_LINK_MESH = [
    "shoulder_pitch_link",
    "shoulder_roll_link",
    "shoulder_yaw_link",
    "elbow_pitch_link",
    "wrist_yaw_link",
    "wrist_roll_link",
    "wrist_pitch_link",
]
# Mesh placement in the joint frame.
#
# Yaw: the wrist meshes are modelled with the first wrist pin along X, while the first wrist
# axis of the URDF (wrist pitch) is along Y. Rotating the three wrist meshes by 90 deg about
# the joint z axis aligns every pin with its joint axis.
#
# Shift: the URDF places some joint origins at an offset along their own rotation axis
# (kinematically irrelevant), whereas each mesh origin lies on the link centre line. The
# shift undoes that offset. It is not a mirror image between the two arms, so both sides
# are listed explicitly.
MESH_YAW_IN_JOINT_DEG = {"right": [0, 0, 0, 0, 90, 90, 90], "left": [0, 0, 0, 0, -90, -90, -90]}
MESH_SHIFT_IN_JOINT = {
    "right": [
        np.array([0, -0.075, 0]),
        np.array([-0.0328, 0, 0]),
        np.zeros(3),
        np.array([0, +0.0325, 0]),
        np.zeros(3),
        np.array([0, -0.022, 0]),
        np.array([-0.0225, 0, 0]),
    ],
    "left": [
        np.array([0, +0.075, 0]),
        np.array([-0.0328, 0, 0]),
        np.zeros(3),
        np.array([0, +0.0325, 0]),
        np.zeros(3),
        np.array([0, +0.022, 0]),
        np.array([-0.0225, 0, 0]),
    ],
}
# L6 hand meshes: one file per link, named after the link without the lh_/rh_ prefix.
HAND_MESH_DIR = {s: os.path.join(_ROOT, "models", "l6", s, "meshes") for s in ("left", "right")}
# Adapter between the wrist mesh and the hand base, drawn as a cylinder (display only).
HAND_ADAPTER_LEN_M = 0.0245
HAND_ADAPTER_RADIUS_M = 0.020
HAND_LINKS = [
    "hand_base_link",
    "thumb_metacarpals_base1",
    "thumb_metacarpals",
    "thumb_distal",
    "index_proximal",
    "index_distal",
    "middle_proximal",
    "middle_distal",
    "ring_proximal",
    "ring_distal",
    "pinky_proximal",
    "pinky_distal",
]

# Skeleton and column dimensions in metres.
LINK_RADIUS = 0.022
JOINT_RADIUS = 0.030
COLUMN_RADIUS = 0.05
SHOULDER_Z = 1.217  # shoulder base height in the URDF


def _rot_z(a):
    """Rotation matrix about z by angle a (rad)."""
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rot_y_to(d):
    """Rotation matrix that maps +Y to direction d (meshcat cylinders point along Y)."""
    d = np.asarray(d, float)
    n = np.linalg.norm(d)
    if n < 1e-9:
        return np.eye(3)
    d = d / n
    y = np.array([0, 1.0, 0])
    v = np.cross(y, d)
    s = np.linalg.norm(v)
    c = float(y @ d)
    if s < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1, -1, -1.0])
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / s**2)


class Viewer:
    """Meshcat scene of the dual arm; `q` holds the joint angles of both arms."""

    # Draw the L6 hands. Subclasses disable this when something else is mounted at the wrist.
    SHOW_HANDS = True

    def __init__(self, open_browser: bool = True):
        if not os.path.isfile(URDF):
            raise FileNotFoundError(f"{URDF} not found: run robot/make_dual_a7_urdf.py first")
        self.model = pin.buildModelFromUrdf(URDF)
        self.data = self.model.createData()
        self.qidx, self.jid = {}, {}
        for side in ("left", "right"):
            ids = [self.model.getJointId(PREFIX[side] + n + "_Joint") for n in SDK_JOINTS]
            self.jid[side] = ids
            self.qidx[side] = [self.model.joints[i].idx_q for i in ids]

        import meshcat
        import meshcat.geometry as g

        self._g = g
        self.viz = _Vis(meshcat.Visualizer())
        if open_browser:
            self.viz.viewer.open()
        self._build_static()
        self._built = {}
        self.q = pin.neutral(self.model)
        self.hand_q = {s: [255.0] * 6 for s in ("left", "right")}  # hand channels, 255 = extended
        self._hand_fk = {s: L6FK(s) for s in ("left", "right")} if self.SHOW_HANDS else {}
        self._hand_T = {s: hand_mount_T(s) for s in ("left", "right")}
        self.display()

    def _cyl(self, node, p0, p1, radius, material):
        """Set node to a cylinder from p0 to p1."""
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        L = float(np.linalg.norm(p1 - p0))
        node.set_object(self._g.Cylinder(max(L, 1e-4), radius), material)
        T = np.eye(4)
        T[:3, :3] = _rot_y_to(p1 - p0)
        T[:3, 3] = (p0 + p1) / 2
        node.set_transform(T)

    def _build_static(self):
        """Create the objects that do not move: column, shoulder bar, base and end markers."""
        g = self._g
        m = g.MeshLambertMaterial(color=0x888888, opacity=0.9)
        self._cyl(self.viz.viewer["body/column"], [0, 0, 0], [0, 0, SHOULDER_Z], COLUMN_RADIUS, m)
        self._cyl(
            self.viz.viewer["body/shoulder_bar"],
            [0, 0.096, SHOULDER_Z],
            [0, -0.096, SHOULDER_Z],
            0.03,
            m,
        )
        self.viz.viewer["body/base"].set_object(g.Box([0.30, 0.30, 0.02]), m)
        for side in ("left", "right"):
            self.viz.viewer[f"marks/{side}/tool_origin"].set_object(
                g.Sphere(0.012), g.MeshLambertMaterial(color=0x00CCCC, opacity=0.85)
            )
            self.viz.viewer[f"marks/{side}/mount_face"].set_object(
                g.Sphere(0.012), g.MeshLambertMaterial(color=0xFF00FF, opacity=0.85)
            )

    def _fk_points(self, side, data):
        """Return the joint origins plus the hand mounting point (base frame), and its SE3."""
        pts = [data.oMi[i].translation.copy() for i in self.jid[side]]
        T7 = data.oMi[self.jid[side][-1]]
        face = T7 * pin.SE3(np.eye(3), MOUNT_FACE_IN_J7)
        pts.append(face.translation.copy())
        return pts, face

    def _mesh_files(self, side):
        """Paths of the seven arm meshes of one side, or None if any is missing."""
        if not USE_ARM_MESHES:
            return None
        paths = [os.path.join(ARM_MESH_DIR, f"{side}_{n}.STL") for n in ARM_LINK_MESH]
        return paths if all(os.path.isfile(p) for p in paths) else None

    def draw_arm(self, node_name, side, q, color=0xDDDDDD, opacity=1.0):
        """Draw one arm at configuration q under the node `node_name/side`."""
        data = self.model.createData()
        pin.forwardKinematics(self.model, data, q)
        pts, _ = self._fk_points(side, data)
        node = self.viz.viewer[f"{node_name}/{side}"]
        key = (node_name, side)
        g = self._g
        mat = g.MeshLambertMaterial(color=color, opacity=opacity)
        meshes = self._mesh_files(side)
        rebuild = self._built.get(key) != color
        if rebuild:
            for k in range(len(pts)):
                node[f"j{k}"].set_object(
                    g.Sphere(0.012 if meshes else (JOINT_RADIUS if k < 7 else 0.008)), mat
                )
            if meshes:
                for k, f in enumerate(meshes):
                    node[f"m{k}"].set_object(g.StlMeshGeometry.from_file(f), mat)
            self._built[key] = color
        for k, p in enumerate(pts):
            T = np.eye(4)
            T[:3, 3] = p
            node[f"j{k}"].set_transform(T)
        if meshes:
            for k, jid in enumerate(self.jid[side]):
                Rk = _rot_z(np.deg2rad(MESH_YAW_IN_JOINT_DEG[side][k]))
                node[f"m{k}"].set_transform(
                    (data.oMi[jid] * pin.SE3(Rk, MESH_SHIFT_IN_JOINT[side][k])).homogeneous
                )
            # Shoulder block between the column and the first motor.
            sgn = 1.0 if side == "left" else -1.0
            self._cyl(
                node["l0"],
                np.array([0.0, 0.0, SHOULDER_Z]),
                np.array([0.0, sgn * 0.110, SHOULDER_Z]),
                0.043,
                mat,
            )
        else:
            base = np.array([0.0, 0.0, SHOULDER_Z])
            segs = [(base, pts[0])] + [(pts[k], pts[k + 1]) for k in range(len(pts) - 1)]
            for k, (a, b) in enumerate(segs):
                self._cyl(node[f"l{k}"], a, b, LINK_RADIUS, mat)

    def draw_hand(self, node_name, side, T_base_hand, q6, color=0x444444, opacity=1.0):
        """Draw one L6 hand under `node_name/side/hand`.

        T_base_hand is the 4x4 pose of hand_base_link in the base frame; q6 are the six
        hand channels (0-255).
        """
        files = {n: os.path.join(HAND_MESH_DIR[side], n + ".STL") for n in HAND_LINKS}
        if not all(os.path.isfile(f) for f in files.values()):
            return
        node = self.viz.viewer[f"{node_name}/{side}/hand"]
        key = (node_name, side, "hand")
        if self._built.get(key) != color:
            mat = self._g.MeshLambertMaterial(color=color, opacity=opacity)
            for n, f in files.items():
                node[n].set_object(self._g.StlMeshGeometry.from_file(f), mat)
            node["adapter"].set_object(
                self._g.Cylinder(HAND_ADAPTER_LEN_M, HAND_ADAPTER_RADIUS_M),
                self._g.MeshLambertMaterial(color=0xB8B8B8, opacity=opacity),
            )
            self._built[key] = color
        link_T = self._hand_fk[side].link_T(q6)  # link poses in the hand_base_link frame
        for n in HAND_LINKS:
            node[n].set_transform(T_base_hand @ link_T[n])
        # The adapter extends from the hand base towards the wrist, along the hand -z axis.
        A = np.eye(4)
        A[:3, :3] = _rot_y_to([0.0, 0.0, -1.0])
        A[2, 3] = -HAND_ADAPTER_LEN_M / 2
        node["adapter"].set_transform(T_base_hand @ A)

    def _tool_frames(self, side: str):
        """Return (tool origin, hand mounting face) as SE3 in the base frame.

        Uses `self.data`, which must hold the forward kinematics of the current `q`.
        """
        T7 = self.data.oMi[self.jid[side][-1]]
        return (
            T7 * pin.SE3(np.eye(3), TOOL_ORIGIN_IN_J7),
            T7 * pin.SE3(np.eye(3), MOUNT_FACE_IN_J7),
        )

    def show_camera(self, T_base_cam, intrinsics=None, depth_m=0.6, label="cam"):
        """Draw a camera: body, axes (blue = optical axis) and view frustum.

        T_base_cam is the 4x4 pose of the camera optical frame (OpenCV convention: +Z forward,
        +X right, +Y down) in the base frame, as written by `handeye_joint_solve.py`.
        Without intrinsics a 69 x 42 deg field of view is drawn.
        """
        g = self._g
        T = np.asarray(T_base_cam, float)
        node = self.viz.viewer[f"marks/camera/{label}"]
        node.set_object(
            g.Box([0.09, 0.025, 0.025]), g.MeshLambertMaterial(color=0x222222, opacity=0.9)
        )
        node.set_transform(T)
        for name, axis, col in (
            ("x", [1, 0, 0], 0xFF0000),
            ("y", [0, 1, 0], 0x00CC00),
            ("z", [0, 0, 1], 0x0066FF),
        ):
            pts = np.array([[0, 0, 0], np.array(axis) * 0.12], float).T
            node[f"axis_{name}"].set_object(
                g.LineSegments(
                    g.PointsGeometry(pts.astype(np.float32)), g.LineBasicMaterial(color=col)
                )
            )
        if intrinsics:
            fx, fy = intrinsics["fx"], intrinsics["fy"]
            w, h = intrinsics["width"], intrinsics["height"]
            cx, cy = intrinsics["ppx"], intrinsics["ppy"]
            xs = ((0 - cx) / fx * depth_m, (w - cx) / fx * depth_m)
            ys = ((0 - cy) / fy * depth_m, (h - cy) / fy * depth_m)
        else:
            hx = np.tan(np.deg2rad(69 / 2)) * depth_m
            hy = np.tan(np.deg2rad(42 / 2)) * depth_m
            xs, ys = (-hx, hx), (-hy, hy)
        c = [
            [xs[0], ys[0], depth_m],
            [xs[1], ys[0], depth_m],
            [xs[1], ys[1], depth_m],
            [xs[0], ys[1], depth_m],
        ]
        o = [0, 0, 0]
        seg = [o, c[0], o, c[1], o, c[2], o, c[3], c[0], c[1], c[1], c[2], c[2], c[3], c[3], c[0]]
        node["frustum"].set_object(
            g.LineSegments(
                g.PointsGeometry(np.array(seg, np.float32).T), g.LineBasicMaterial(color=0xFFAA00)
            )
        )

    def hide_camera(self, label="cam"):
        """Remove the camera marker."""
        self.viz.viewer[f"marks/camera/{label}"].delete()

    def display(self):
        """Redraw both arms, the end markers and the hands at the current `q`."""
        pin.forwardKinematics(self.model, self.data, self.q)
        pin.updateFramePlacements(self.model, self.data)
        for side in ("left", "right"):
            self.draw_arm("robot", side, self.q)
            T, face = self._tool_frames(side)
            self.viz.viewer[f"marks/{side}/tool_origin"].set_transform(T.homogeneous)
            self.viz.viewer[f"marks/{side}/mount_face"].set_transform(face.homogeneous)
            if self.SHOW_HANDS:
                T_base_hand = self.data.oMi[self.jid[side][-1]].homogeneous @ self._hand_T[side]
                self.draw_hand("robot", side, T_base_hand, self.hand_q[side])

    def set_joint(self, side: str, j1based: int, deg: float):
        """Set joint J1..J7 of one arm, in degrees."""
        self.q[self.qidx[side][j1based - 1]] = np.deg2rad(deg)

    def set_arm(self, side: str, degs):
        """Set all joints of one arm, in degrees (motor order)."""
        for i, d in enumerate(degs):
            self.q[self.qidx[side][i]] = np.deg2rad(d)

    def set_hand(self, side: str, q6):
        """Set the six hand channels (0-255, 255 = extended).

        Order: thumb flexion, thumb abduction, index, middle, ring, little.
        """
        self.hand_q[side] = [float(np.clip(v, 0, 255)) for v in q6]

    def get_arm_deg(self, side: str):
        """Joint angles of one arm in degrees (motor order)."""
        return [np.rad2deg(self.q[i]) for i in self.qidx[side]]

    def info(self):
        """Print the joint angles and the end pose of both arms."""
        pin.forwardKinematics(self.model, self.data, self.q)
        for side in ("left", "right"):
            T, face = self._tool_frames(side)
            q = self.get_arm_deg(side)
            print(f"  {side:<6} joints (deg) " + " ".join(f"{v:+7.1f}" for v in q))
            print(
                f"         tool origin (base frame) [{T.translation[0]:+.4f}, "
                f"{T.translation[1]:+.4f}, {T.translation[2]:+.4f}] m"
            )
            print(
                f"         hand mount  (base frame) [{face.translation[0]:+.4f}, "
                f"{face.translation[1]:+.4f}, {face.translation[2]:+.4f}] m"
                f"   height {face.translation[2]*100:.1f} cm"
            )
            print(f"         tool axis points to {np.round(T.rotation @ np.array([0,0,-1]), 3)}")

    def limits(self):
        """Print the joint limits of both arms."""
        print("  joint limits (deg), from conventions.LIMITS_*")
        print(f"  {'joint':<17} {'left arm':>18} {'right arm':>18}")
        for i in range(7):
            lo_l, hi_l = LIMITS_LEFT[i]
            lo_r, hi_r = LIMITS_RIGHT[i]
            print(
                f"  J{i+1} {JOINT_NAMES[i]:<14} "
                f"[{np.rad2deg(lo_l):+7.1f},{np.rad2deg(hi_l):+7.1f}] "
                f"[{np.rad2deg(lo_r):+7.1f},{np.rad2deg(hi_r):+7.1f}]"
            )

    def demo(self, side: str = "right", amp_deg: float = 35.0, step: float = 0.04):
        """Swing each joint of one arm in turn by +-amp_deg around its current angle."""
        base = list(self.get_arm_deg(side))
        for j in range(7):
            print(f"    {side} J{j+1} ({JOINT_NAMES[j]})")
            for d in np.concatenate(
                [
                    np.linspace(0, amp_deg, 18),
                    np.linspace(amp_deg, -amp_deg, 36),
                    np.linspace(-amp_deg, 0, 18),
                ]
            ):
                self.q[self.qidx[side][j]] = np.deg2rad(base[j] + d)
                self.display()
                time.sleep(step)
        self.set_arm(side, base)
        self.display()


class _Vis:
    """Holder exposing the meshcat visualizer as `.viewer`, like pinocchio's MeshcatVisualizer."""

    def __init__(self, viewer):
        self.viewer = viewer


HELP = """  home | calib | info | lim | demo [l|r] | open | fist | q
  r4 45                    right arm J4 to 45 deg (l4 for the left arm)
  r 0 -30 0 90 0 0 0       all seven joints of the right arm
  rh 255 255 80 80 80 80   six channels of the right hand, 255 = extended (lh: left hand)"""


def main() -> int:
    open_browser = "--no-open" not in sys.argv
    print(f"loading {os.path.relpath(URDF)} ...")
    v = Viewer(open_browser=open_browser)
    print(f"\nviewer started: {v.viz.viewer.url()}")
    print("   magenta sphere: mating face of the hand base\n")
    print(HELP)
    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        t = line.split()
        c = t[0].lower()
        try:
            if c in ("q", "quit", "exit"):
                break
            elif c == "home":
                v.q = pin.neutral(v.model)
                v.display()
                v.info()
            elif c == "calib":
                v.set_arm("right", CALIB_Q_RIGHT_DEG)
                v.display()
                v.info()
            elif c == "info":
                v.info()
            elif c in ("lim", "limits"):
                v.limits()
            elif c == "demo":
                v.demo("left" if len(t) > 1 and t[1] in ("l", "left") else "right")
            elif c in ("open", "fist"):
                for s in ("left", "right"):
                    v.set_hand(s, [255.0] * 6 if c == "open" else [0.0] * 6)
                v.display()
            elif c in ("lh", "rh") and len(t) == 7:
                v.set_hand("left" if c[0] == "l" else "right", [float(x) for x in t[1:]])
                v.display()
            elif c in ("l", "r", "left", "right") and len(t) == 8:
                v.set_arm("left" if c[0] == "l" else "right", [float(x) for x in t[1:]])
                v.display()
                v.info()
            elif len(c) >= 2 and c[0] in "lr" and c[1:].isdigit() and len(t) == 2:
                v.set_joint("left" if c[0] == "l" else "right", int(c[1:]), float(t[1]))
                v.display()
                v.info()
            else:
                print(HELP)
        except Exception as e:
            print(f"  error: {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
