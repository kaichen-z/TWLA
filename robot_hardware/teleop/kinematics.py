"""Single-arm kinematics of the A7 for teleoperation: FK, Jacobian and IK (pinocchio).

The model is `drivers/robot/dual_a7.urdf`, the same one used by the viewer and the
hand-eye calibration. Joint angles are in motor order, as returned by `A7.get_angles()`.

Poses are expressed in the frame of the arm's own shoulder (the fixed mounting of J1),
which is parallel to the base frame. The tool point is the mating face of the L6 hand
(`robot.conventions.hand_mount_T`), optionally moved by `tcp_offset`.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np
import pinocchio as pin

_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if os.path.join(_ROOT, "drivers") not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "drivers"))

from robot.conventions import hand_mount_T  # noqa: E402

DEFAULT_URDF = os.path.join(_ROOT, "drivers", "robot", "dual_a7.urdf")

JOINTS = [
    "Shoulder_Pitch",
    "Shoulder_Roll",
    "Shoulder_Yaw",
    "Elbow_Pitch",
    "Wrist_Yaw",
    "Wrist_Pitch",
    "Wrist_Roll",
]
PREFIX = {"left": "Left_", "right": "Right_"}
NUM_JOINTS = 7


@dataclass
class Pose:
    """Tool pose: position (m) and rotation matrix."""

    p: np.ndarray  # (3,)
    R: np.ndarray  # (3, 3)

    @staticmethod
    def from_se3(T: pin.SE3) -> "Pose":
        return Pose(T.translation.copy(), T.rotation.copy())

    def to_se3(self) -> pin.SE3:
        return pin.SE3(self.R, self.p)

    def copy(self) -> "Pose":
        return Pose(self.p.copy(), self.R.copy())


class ArmKinematics:
    """Kinematics of one arm.

    side: "left" or "right"; tcp_offset: translation from the hand mating face to the tool
    point, in the hand base frame (m).
    """

    def __init__(self, side: str, urdf_path: str | None = None, tcp_offset=(0.0, 0.0, 0.0)):
        assert side in ("left", "right"), side
        self.side = side
        self.urdf_path = urdf_path or DEFAULT_URDF
        if not os.path.isfile(self.urdf_path):
            raise FileNotFoundError(f"URDF not found: {self.urdf_path}")

        full = pin.buildModelFromUrdf(self.urdf_path)
        # Keep only this arm: lock the joints of the other one.
        other = PREFIX["right" if side == "left" else "left"]
        lock = [full.getJointId(n) for n in full.names[1:] if n.startswith(other)]
        self.model = pin.buildReducedModel(full, lock, pin.neutral(full))

        self.joint_names = [f"{PREFIX[side]}{n}_Joint" for n in JOINTS]
        missing = [n for n in self.joint_names if not self.model.existJointName(n)]
        if missing:
            raise RuntimeError(f"joints missing from the URDF: {missing}")
        self._qidx = [self.model.joints[self.model.getJointId(n)].idx_q for n in self.joint_names]
        self.limits = [
            (float(self.model.lowerPositionLimit[i]), float(self.model.upperPositionLimit[i]))
            for i in self._qidx
        ]

        # Origin: the fixed placement of the shoulder joint (without the J1 rotation), so
        # that the Jacobian in LOCAL_WORLD_ALIGNED matches the pose frame.
        self._j1 = self.model.getJointId(self.joint_names[0])
        j7 = self.model.getJointId(self.joint_names[-1])
        T_tool = hand_mount_T(side)
        T_tool[:3, 3] += T_tool[:3, :3] @ np.asarray(tcp_offset, dtype=float)
        self._tcp_fid = self.model.addFrame(
            pin.Frame(
                f"{side}_tcp",
                j7,
                self.model.getFrameId(f"{PREFIX[side]}Wrist_Roll_Link"),
                pin.SE3(T_tool[:3, :3], T_tool[:3, 3]),
                pin.FrameType.OP_FRAME,
            )
        )
        self.data = self.model.createData()

    def _q(self, angles) -> np.ndarray:
        q = pin.neutral(self.model)
        for i, a in zip(self._qidx, angles):
            q[i] = float(a)
        return q

    def _shoulder_T(self) -> pin.SE3:
        """Fixed placement of the shoulder joint in the base frame (pure translation)."""
        return self.model.jointPlacements[self._j1]

    def clamp(self, angles) -> np.ndarray:
        """Joint angles clipped to the limits."""
        return np.array([min(max(a, lo), hi) for a, (lo, hi) in zip(angles, self.limits)])

    def in_limits(self, angles) -> bool:
        return all(lo <= a <= hi for a, (lo, hi) in zip(angles, self.limits))

    def fk(self, angles) -> Pose:
        """Tool pose in the shoulder frame for seven joint angles (rad, motor order)."""
        q = self._q(angles)
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        return Pose.from_se3(self._shoulder_T().inverse() * self.data.oMf[self._tcp_fid])

    def jacobian(self, angles) -> np.ndarray:
        """6x7 tool Jacobian in LOCAL_WORLD_ALIGNED."""
        q = self._q(angles)
        pin.computeJointJacobians(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        J = pin.getFrameJacobian(
            self.model, self.data, self._tcp_fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        return J[:, self._qidx]

    # Rest posture for the null-space bias (rad, motor order): elbow bent, away from the
    # straight-arm singularity. It does not affect the tool pose.
    Q_REST = np.array([-0.3, 0.4, 0.0, 1.0, 0.0, 0.0, 0.0])

    def ik(
        self,
        target: Pose,
        seed,
        *,
        max_iter: int = 100,
        pos_tol: float = 1e-4,
        rot_tol: float = 1e-3,
        damping: float = 1e-3,
        max_step: float = 0.2,
        q_rest=None,
        null_gain: float = 0.05,
        manip_thresh: float = 0.02,
    ):
        """Damped least-squares IK with a null-space posture bias.

        target is in the shoulder frame (as fk). seed is the starting guess; in real-time
        control pass the previous solution, so the solution stays on the same branch of the
        redundant arm. max_step limits each iteration (rad); the damping grows when the
        manipulability drops below manip_thresh.

        Returns (angles, info) with info = {ok, iters, pos_err, rot_err, manip}; the best
        solution found is returned even when ok is False.
        """
        q = self.clamp(np.asarray(seed, dtype=float))
        q_rest = self.Q_REST if q_rest is None else np.asarray(q_rest, float)
        T_target = target.to_se3()
        pos_err = rot_err = float("inf")
        w = 0.0
        it = 0
        for it in range(1, max_iter + 1):
            cur = self.fk(q).to_se3()
            err = pin.log6(cur.inverse() * T_target).vector  # local frame
            Rw = cur.rotation
            e = np.concatenate([Rw @ err[:3], Rw @ err[3:]])  # to LOCAL_WORLD_ALIGNED
            pos_err = float(np.linalg.norm(e[:3]))
            rot_err = float(np.linalg.norm(e[3:]))
            if pos_err < pos_tol and rot_err < rot_tol:
                break
            J = self.jacobian(q)

            w = float(np.sqrt(max(np.linalg.det(J @ J.T), 0.0)))
            lam = damping if w >= manip_thresh else damping * (manip_thresh / max(w, 1e-9))

            JJt = J @ J.T + (lam**2) * np.eye(6)
            Jp = J.T @ np.linalg.inv(JJt)  # damped pseudo-inverse
            dq = Jp @ e

            if null_gain > 0.0:
                dq += (np.eye(NUM_JOINTS) - Jp @ J) @ (null_gain * (q_rest - q))

            n = np.linalg.norm(dq)
            if n > max_step:
                dq *= max_step / n
            q = self.clamp(q + dq)
        ok = pos_err < pos_tol and rot_err < rot_tol
        return q, {"ok": ok, "iters": it, "pos_err": pos_err, "rot_err": rot_err, "manip": w}

    def manipulability(self, angles) -> float:
        """sqrt(det(J J^T)); close to 0 near a singularity."""
        J = self.jacobian(angles)
        return float(np.sqrt(max(np.linalg.det(J @ J.T), 0.0)))
