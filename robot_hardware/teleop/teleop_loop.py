"""Arm teleoperation loop for one arm: Vive Tracker motion -> arm tool pose, 50 Hz.

When the clutch engages, the current tool pose of the arm and the tracker pose are taken
as anchors; from then on the tool follows the tracker's displacement (and, depending on
`Knobs.rot_src`, its rotation) relative to the anchors. Tracker displacements are rotated
into the robot base frame with R_base_from_world from `vive/calib/arm_vive_extrinsics.json`
(the Vive and robot frames are assumed to be aligned beforehand); without that file, a fixed
axis relabelling plus a manual heading is used.

Safety behaviour:
    - with live=False the arm is connected but not enabled, and nothing is sent;
    - the target stays inside a sphere around the anchor (`max_reach`);
    - joint motion per frame is limited (`max_step_deg`);
    - if the IK does not converge the previous command is kept (no jump to another branch);
    - below `min_manip` manipulability the arm stops following;
    - an invalid tracker pose disengages the clutch (no reuse of a stale position);
    - on stop, the arm targets are pinned to the current position (the arm stays enabled).
There is no self-collision checking.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from dataclasses import dataclass, fields

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
for _p in (_ROOT, _HERE, os.path.join(_ROOT, "drivers")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from arm import connect  # noqa: E402
from glove import quat  # noqa: E402
from kinematics import ArmKinematics, Pose  # noqa: E402
from tracker_source import vr_to_arm  # noqa: E402

__all__ = ["Knobs", "TeleopLoop", "load_extrinsics", "rot_z", "EXTRINSICS_PATH"]

EXTRINSICS_PATH = os.path.join(_ROOT, "vive", "calib", "arm_vive_extrinsics.json")


def load_extrinsics(path: str = EXTRINSICS_PATH):
    """R_base_from_world (Vive frame -> robot base frame). Returns (R or None, note).

    Only the rotation is used: displacements are applied relative to the anchor taken at
    engagement, so the translation between the frames is not needed. Optional "quality"
    numbers in the file (residual_rms_mm, tracker_rot_max_dev_deg, n_points_used/total) are
    checked when present.
    """
    import json

    if not os.path.isfile(path):
        return None, "not calibrated (no arm_vive_extrinsics.json)"
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        R = np.asarray(d["R_base_from_world"], dtype=float).reshape(3, 3)
        if (
            abs(float(np.linalg.det(R)) - 1.0) > 1e-3
            or float(np.abs(R.T @ R - np.eye(3)).max()) > 1e-3
        ):
            return None, "R in the calibration file is not a rotation; ignored"
        q = d.get("quality", {})
        if not q:
            return R, "aligned (rotation from arm_vive_extrinsics.json)"
        rot_dev = float(q.get("tracker_rot_max_dev_deg", 0.0) or 0.0)
        rms = float(q.get("residual_rms_mm", 0.0) or 0.0)
        used = q.get("n_points_used", 0) or 0
        total = q.get("n_points_total", 0) or 0
        # Reject only clearly broken calibrations: an error of a few degrees is absorbed by
        # the operator, who watches the robot.
        fatal = []
        if rot_dev > 25.0:
            fatal.append(f"tracker rotation drift {rot_dev:.0f} deg (> 25, mount loose)")
        if rms > 60.0:
            fatal.append(f"residual RMS {rms:.0f} mm (> 60)")
        if total and used < max(3, total * 0.6):
            fatal.append(f"only {used}/{total} samples used")
        if fatal:
            return None, "calibration rejected: " + "; ".join(fatal)

        base = (
            f"residual RMS {rms:.1f} mm, scale ratio {q.get('scale_ratio', float('nan')):.4f}, "
            f"{used}/{total} samples, tracker rotation drift {rot_dev:.1f} deg"
        )
        if rot_dev > 2.0 or rms > 5.0:
            return R, f"calibrated, reduced accuracy ({base})"
        return R, f"calibrated ({base})"
    except Exception as exc:  # noqa: BLE001
        return None, f"cannot read the calibration file: {type(exc).__name__}"


@dataclass
class Knobs:
    """Tuning parameters; velocity/acceleration changes are sent to the arm immediately."""

    velocity: float = 0.33  # rad/s
    acceleration: float = 3.3  # rad/s^2
    scale: float = 0.3  # arm displacement per hand displacement
    max_reach: float = 0.12  # m, radius of the sphere around the anchor
    max_step_deg: float = 1.5  # max joint change per frame
    heading_deg: float = 0.0  # manual rotation about z (only without calibration)
    rot_scale: float = 1.0  # orientation following gain; 0 = locked, negative = reversed
    rot_heading_deg: float = 0.0  # rotation of the glove orientation delta about z
    # Orientation source: 0 = glove IMU (yaw drifts), 1 = tracker (absolute),
    # 2 = mixed (roll/pitch from the glove, yaw from the tracker).
    rot_src: float = 1.0
    min_manip: float = 0.020  # stop following below this manipulability
    max_rot_step_deg: float = 3.0  # the total orientation change is limited to 40x this

    def copy(self) -> "Knobs":
        return Knobs(**{f.name: getattr(self, f.name) for f in fields(self)})


def _rotvec_to_mat(rv) -> np.ndarray:
    """Rotation vector -> 3x3 matrix (Rodrigues)."""
    th = float(np.linalg.norm(rv))
    if th < 1e-9:
        return np.eye(3)
    a = np.asarray(rv, dtype=float) / th
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def _mat_to_rotvec(R) -> np.ndarray:
    """3x3 matrix -> rotation vector, robust near 0 and 180 deg."""
    R = np.asarray(R, dtype=float)
    c = (np.trace(R) - 1.0) / 2.0
    th = float(np.arccos(np.clip(c, -1.0, 1.0)))
    if th < 1e-9:
        return np.zeros(3)
    if abs(th - np.pi) < 1e-6:  # the trace formula degenerates; take the axis from R + I
        A = (R + np.eye(3)) / 2.0
        ax = np.sqrt(np.clip(np.diag(A), 0.0, 1.0))
        i = int(np.argmax(ax))
        ax = ax * np.sign(A[i]) if ax[i] > 1e-9 else ax
        n = np.linalg.norm(ax)
        return (ax / n * th) if n > 1e-9 else np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return v / (2.0 * np.sin(th)) * th


def rot_z(deg: float) -> np.ndarray:
    """Rotation matrix about z."""
    c, s = np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class TeleopLoop:
    """Control loop for one arm in its own thread. `start()` connects; `stop()` pins the arm."""

    def __init__(
        self,
        tracker,
        clutch,
        *,
        side: str = "right",
        can: str = "can1",
        live: bool = False,
        rate: float = 50.0,
        min_manip: float = 0.020,
        knobs: Knobs | None = None,
        devices: dict | None = None,
        glove=None,
    ) -> None:
        self.glove = glove  # glove probe (orientation source 0 / 2), or None
        # Status probes; the arm probe on the same bus is suspended while the loop runs.
        self.devices = devices or {}
        self._released: list = []
        self.trk = tracker
        self.clutch = clutch
        self.side, self.can, self.live = side, can, live
        self.rate, self.min_manip = rate, min_manip
        self.lock = threading.Lock()
        self.knobs = knobs or Knobs()
        self._vel_dirty = False
        self._run = False
        self._t: threading.Thread | None = None
        self.state = "idle"  # idle | starting | running | stopped | failed
        self.msg = ""
        self.tel: dict = {}
        self.log: list = []
        self._arm = None
        self._kin = None
        self.R_bw, self.extr_note = load_extrinsics()
        self.use_extr = self.R_bw is not None
        # Start condition: J4 must be bent at least this much (0 = no check), i.e. the arm
        # starts from the ready pose and not from the singular home pose.
        self.require_j4_min_deg = 0.0
        # Pin the arm to its measured position when the clutch releases (otherwise it keeps
        # moving to the last command).
        self.pin_on_release = False
        self.rec = None  # recorder.Recorder, or None

    def set_knob(self, k: str, v: float) -> None:
        """Change one knob."""
        with self.lock:
            if not hasattr(self.knobs, k):
                return
            setattr(self.knobs, k, float(v))
            if k in ("velocity", "acceleration"):
                self._vel_dirty = True

    def _glove_quat(self):
        """Glove wrist orientation (x, y, z, w), or None.

        Falls back to the IMU frame when the wrist transform is unknown; only the change
        since engagement is used, so the constant offset cancels.
        """
        if self.glove is None or not hasattr(self.glove, "frame"):
            return None
        f = self.glove.frame()
        if f is None:
            return None
        q = f.q_wrist if f.q_wrist is not None else f.motion.q_world_body
        return None if q is None else np.asarray(q, dtype=float)

    def _rot_delta(self, src: int, ts, anchor_t_rot, anchor_g, k):
        """Orientation change since engagement as a rotation vector in the base frame."""
        rv_t = None
        if anchor_t_rot is not None and ts is not None and ts.ok:
            dR_w = np.asarray(ts.rot, dtype=float) @ np.asarray(anchor_t_rot, dtype=float).T
            # A rotation changes frame by similarity: R M R^T.
            dR_b = self.R_bw @ dR_w @ self.R_bw.T if self.R_bw is not None else dR_w
            rv_t = _mat_to_rotvec(dR_b)

        rv_g = None
        gq = self._glove_quat()
        if gq is not None and anchor_g is not None:
            dq = quat.multiply(gq, quat.conjugate(anchor_g))
            rv_g = quat.to_rotvec(dq)
            if abs(k.rot_heading_deg) > 1e-6:
                rv_g = rot_z(k.rot_heading_deg) @ rv_g

        if src == 1:
            rv = rv_t
        elif src == 2 and rv_t is not None and rv_g is not None:
            rv = np.array([rv_g[0], rv_g[1], rv_t[2]])  # roll/pitch from glove, yaw from tracker
        else:
            rv = rv_g if rv_g is not None else rv_t
        return None if rv is None else rv * k.rot_scale

    def say(self, s: str) -> None:
        """Append a line to the loop log."""
        with self.lock:
            self.log.append(f"{time.strftime('%H:%M:%S')}  {s}")
            del self.log[:-60]

    def start(self, start_j4_deg: float = 0.0) -> str:
        """Start the loop thread; start_j4_deg > 0 first bends J4 to that angle (live only)."""
        with self.lock:
            if self._run:
                return "already running"
            self.state, self.msg, self.log = "starting", "connecting", []
        self._run = True
        self._t = threading.Thread(target=self._main, args=(start_j4_deg,), daemon=True)
        self._t.start()
        return ""

    def stop(self) -> str:
        """Ask the loop to stop."""
        self._run = False
        return ""

    def snapshot(self) -> dict:
        """State, knobs, telemetry and recent log."""
        with self.lock:
            return {
                "state": self.state,
                "msg": self.msg,
                "live": self.live,
                "use_extr": self.use_extr,
                "extr_note": self.extr_note,
                "knobs": {f.name: getattr(self.knobs, f.name) for f in fields(self.knobs)},
                "tel": dict(self.tel),
                "log": self.log[-25:],
                "side": self.side,
                "can": self.can,
            }

    def _main(self, start_j4_deg: float) -> None:  # noqa: C901
        arm = None
        try:
            ts = self.trk.latest()
            if not ts.ok:
                raise RuntimeError(f"tracker not available: {ts.reason}")

            with self.lock:
                k0 = self.knobs.copy()
            self._released = [
                d
                for d in self.devices.values()
                if getattr(d, "kind", "") == "arm"
                and getattr(d, "can", None) == self.can
                and hasattr(d, "suspend")
                and d.suspend("teleoperation")
            ]
            if self._released:
                self.say(f"status probe released {self.can}")
                time.sleep(1.5)  # let it close its socket

            self.say(
                f"connecting {self.side} arm on {self.can} "
                f"({'enabled' if self.live else 'not enabled'})"
            )
            arm = connect(
                self.side,
                channel=self.can,
                velocity=k0.velocity,
                acceleration=k0.acceleration,
                enable=self.live,
            )
            self._arm = arm
            kin = ArmKinematics(self.side)
            self._kin = kin
            # IK limits = URDF limits intersected with the SDK limits, minus 0.3 deg, so that
            # every IK solution is accepted by `_set_angles(check_limits=True)`.
            try:
                ml = arm.get_joint_limits()  # motor order, same as the model
                m = np.deg2rad(0.3)
                kin.limits = [
                    (max(a[0], b[0]) + m, min(a[1], b[1]) - m) for a, b in zip(kin.limits, ml)
                ]
                self.say(
                    "IK limits (URDF and SDK): "
                    + " ".join(
                        f"J{i+1}[{np.degrees(lo):+.0f},{np.degrees(hi):+.0f}]"
                        for i, (lo, hi) in enumerate(kin.limits)
                    )
                )
            except Exception as exc:  # noqa: BLE001
                self.say(f"cannot read the SDK limits, using the URDF limits: {type(exc).__name__}")
            q_meas = np.array(arm.get_angles(), dtype=float)

            # A single point-to-point motion needs move_j; _set_angles is a streaming command.
            if start_j4_deg > 0 and self.live:
                tgt = list(arm.get_angles())
                tgt[3] = float(np.deg2rad(start_j4_deg))
                lo, hi = arm.get_joint_limits()[3]
                if not (lo <= tgt[3] <= hi):
                    raise RuntimeError(f"J4 target {start_j4_deg:+.1f} deg outside the limits")
                self.say(f"bending J4 {np.degrees(q_meas[3]):+.1f} -> {start_j4_deg:+.1f} deg")
                arm.move_j(tgt, blocking=False)
                t0 = time.monotonic()
                time.sleep(0.5)  # is_moving() is False until the motion has started
                while time.monotonic() - t0 < 20 and arm.is_moving():
                    time.sleep(0.2)
                time.sleep(0.3)
                q_meas = np.array(arm.get_angles(), dtype=float)
                self.say(f"J4 now {np.degrees(q_meas[3]):+.1f} deg")

            if (
                self.require_j4_min_deg
                and self.live
                and np.degrees(q_meas[3]) < self.require_j4_min_deg
            ):
                raise RuntimeError(
                    f"J4={np.degrees(q_meas[3]):+.1f} deg < {self.require_j4_min_deg:.0f} deg: "
                    "move to the ready pose first"
                )

            w0 = kin.manipulability(q_meas)
            self.say(f"start manipulability w = {w0:.5f} (threshold {self.min_manip})")
            if w0 < self.min_manip:
                if self.live:
                    raise RuntimeError(
                        f"w={w0:.5f} < {self.min_manip} at J4={np.degrees(q_meas[3]):+.1f} deg: "
                        "the arm would not follow; bend the elbow beyond 45 deg"
                    )
                self.say("below the threshold; continuing because this is a dry run")

            with self.lock:
                self.state, self.msg = "running", "running: enable the clutch and press space"
            if self.rec is not None:
                threading.Thread(target=self._sample_arm, args=(arm,), daemon=True).start()
            self._loop(arm, kin, q_meas)

        except Exception as exc:  # noqa: BLE001
            with self.lock:
                self.state = "failed"
                self.msg = f"{type(exc).__name__}: {exc}"[:220]
            self.say(f"error: {self.msg}")
        finally:
            self._run = False
            if arm is not None:
                try:
                    arm.emergency_stop()  # pin to the current position; stays enabled
                    self.say("arm pinned at its current position (still enabled)")
                except Exception:  # noqa: BLE001
                    pass
                try:
                    arm.close()
                except Exception:  # noqa: BLE001
                    pass
            self._arm = self._kin = None
            if self._released:
                time.sleep(0.5)
                for d in self._released:
                    d.resume()
                self.say(f"status probe resumed ({len(self._released)})")
                self._released = []
            with self.lock:
                if self.state != "failed":
                    self.state, self.msg = "stopped", "stopped"

    def _sample_arm(self, arm) -> None:
        """Record the arm state (stream <side>_arm, motor order) at the SDK's own rate.

        The SDK polls the motors in the background (position/velocity every 10 ms, torque
        every 50 ms) and caches the values; a sample is recorded whenever the cached
        timestamps change, so nothing extra is sent on the bus.
        """
        last = None
        while self._run and self.rec is not None:
            try:
                ms = arm._motors
                ts = tuple(m.angle.timestamp for m in ms)
                if ts != last:
                    last = ts
                    self.rec.put(
                        f"{self.side}_arm",
                        time.monotonic_ns(),
                        q=np.array([m.angle.angle for m in ms], dtype=np.float64),
                        dq=np.array([m.velocity.velocity for m in ms], dtype=np.float64),
                        tau=np.array([m.torque.torque for m in ms], dtype=np.float64),
                        sdk_ts=np.array(ts, dtype=np.float64),
                    )
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.002)

    def _loop(self, arm, kin, q_meas) -> None:  # noqa: C901
        period = 1.0 / self.rate
        q_cmd = q_meas.copy()
        anchor_p = anchor_R = anchor_t = anchor_g = None
        anchor_t_rot = None
        was_eng = False
        n_loop = n_sent = n_ikfail = 0
        t_start = time.monotonic()
        n_reject = n_jump = 0
        max_dq_deg = 0.0  # largest joint change sent this frame
        max_dq_ever = 0.0
        prev_trk = None
        JUMP_DEG = 15.0  # a larger single-frame change is logged as a jump

        while self._run:
            lt = time.monotonic()
            n_loop += 1
            with self.lock:
                k = self.knobs.copy()
                dirty, self._vel_dirty = self._vel_dirty, False
            if dirty and self.live:
                try:
                    arm.set_velocities([k.velocity] * 7)
                    arm.set_accelerations([k.acceleration] * 7)
                except Exception:  # noqa: BLE001
                    pass

            ts = self.trk.latest()
            eng = self.clutch.get_key_value_by_name(f"{self.side}_grip") > 0.9
            trk_step_mm = 0.0
            if ts.ok and prev_trk is not None:
                trk_step_mm = float(np.linalg.norm(ts.pos - prev_trk) * 1000)
            prev_trk = ts.pos.copy() if ts.ok else None
            max_dq_deg = 0.0
            rejected = False

            # An invalid tracker pose disengages; the last position is never reused.
            if not ts.ok and eng:
                self.clutch.panic()
                self.say(f"tracker lost -> disengaged: {ts.reason[:60]}")
                eng = False

            if eng and not was_eng:
                q_meas = np.array(arm.get_angles(), dtype=float)
                q_cmd = q_meas.copy()
                p0 = kin.fk(q_meas)
                anchor_p, anchor_R = p0.p.copy(), p0.R.copy()
                anchor_t = ts.pos.copy()  # OpenVR frame
                anchor_g = self._glove_quat()
                anchor_t_rot = ts.rot.copy()
                self.say(
                    f"engaged: tool anchor {np.round(anchor_p*1000,1).tolist()} mm"
                    + ("" if anchor_g is not None else " (no glove data)")
                )
            if not eng and was_eng:
                if self.pin_on_release and self.live:
                    try:
                        arm.emergency_stop()  # target = measured angles
                        q_cmd = np.array(arm.get_angles(), dtype=float)
                    except Exception as exc:  # noqa: BLE001
                        self.say(f"pinning failed: {type(exc).__name__}: {exc}"[:90])
                self.say(
                    "disengaged: arm pinned" if self.pin_on_release else "disengaged: arm holds"
                )
            was_eng = eng

            dp = np.zeros(3)
            info = {"ok": False, "reason": "not engaged"}
            if eng and anchor_p is not None:
                d_world = ts.pos - anchor_t
                if self.use_extr:
                    dp = (self.R_bw @ d_world) * k.scale
                else:
                    dp = rot_z(k.heading_deg) @ (vr_to_arm(d_world) * k.scale)
                n = float(np.linalg.norm(dp))
                if n > k.max_reach:
                    dp = dp * (k.max_reach / n)
                w = kin.manipulability(q_cmd)
                if w < k.min_manip:
                    info = {"ok": False, "reason": "near a singularity, not following"}
                else:
                    # Orientation: rotation since engagement applied to the anchor (rotation
                    # vectors avoid the wrap-around and gimbal lock of Euler angles).
                    R_tgt = anchor_R
                    rv = None
                    src = int(round(k.rot_src))
                    if abs(k.rot_scale) > 1e-6:
                        rv = self._rot_delta(src, ts, anchor_t_rot, anchor_g, k)
                    if rv is not None:
                        ang = float(np.linalg.norm(rv))
                        lim = np.deg2rad(k.max_rot_step_deg) * 40
                        if ang > lim:
                            rv = rv * (lim / ang)
                        R_tgt = _rotvec_to_mat(rv) @ anchor_R
                    q_new, info = kin.ik(Pose(anchor_p + dp, R_tgt), q_cmd)
                    if info.get("ok"):
                        dq = q_new - q_cmd
                        mx = float(np.max(np.abs(dq)))
                        lim = np.deg2rad(k.max_step_deg)
                        if mx > lim:
                            dq *= lim / mx
                        q_prev = q_cmd
                        q_cmd = kin.clamp(q_cmd + dq)
                        max_dq_deg = float(np.degrees(np.max(np.abs(q_cmd - q_prev))))
                        max_dq_ever = max(max_dq_ever, max_dq_deg)
                        if max_dq_deg > JUMP_DEG:
                            n_jump += 1
                            self.say(
                                f"jump #{n_jump}: {max_dq_deg:.0f} deg in one frame, dq="
                                + " ".join(f"{v:+.0f}" for v in np.degrees(q_cmd - q_prev))
                                + f", tracker step {trk_step_mm:.0f} mm, "
                                f"IK {info.get('iters','?')} iterations"
                            )
                        if self.rec is not None:
                            self.rec.put(
                                f"{self.side}_arm_cmd",
                                time.monotonic_ns(),
                                q_cmd=np.asarray(q_cmd, dtype=np.float64),
                                sent=bool(self.live),
                            )
                        if self.live:
                            try:
                                arm._set_angles(list(q_cmd), check_limits=True)
                                n_sent += 1
                            except Exception as exc:  # noqa: BLE001
                                rejected = True
                                n_reject += 1
                                self.say(f"command rejected: {type(exc).__name__}: {exc}"[:90])
                    else:
                        n_ikfail += 1
            if self.rec is not None:
                self.rec.put(
                    f"{self.side}_arm_diag",
                    time.monotonic_ns(),
                    engaged=bool(eng),
                    ik_ok=bool(info.get("ok")),
                    max_dq_deg=np.float32(max_dq_deg),
                    rejected=rejected,
                    trk_step_mm=np.float32(trk_step_mm),
                    trk_ok=bool(ts.ok),
                )

            q_act = np.array(arm.get_angles(), dtype=float)
            w_now = kin.manipulability(q_act)
            reach_use = float(np.linalg.norm(dp) / max(k.max_reach, 1e-9))
            src = int(round(k.rot_src))
            if abs(k.rot_scale) < 1e-6:
                rot_txt = "locked (rot_scale = 0)"
            elif anchor_g is None and src == 0:
                rot_txt = "no glove data, locked"
            else:
                rot_txt = f"gain {k.rot_scale:+.2f}" + (
                    "" if abs(k.rot_heading_deg) < 1e-6 else f", heading {k.rot_heading_deg:+.0f}"
                )
            with self.lock:
                self.tel = {
                    "link": [
                        ["Tracker", "valid" if ts.ok else ts.reason[:40], "ok" if ts.ok else "bad"],
                        [
                            "orientation source",
                            {
                                0: "glove (yaw drifts)",
                                1: "tracker",
                                2: "mixed (glove roll/pitch, tracker yaw)",
                            }.get(src, "?"),
                            "ok" if src in (1, 2) else "warn",
                        ],
                        ["orientation", rot_txt, "ok" if abs(k.rot_scale) > 1e-6 else "warn"],
                        [
                            "frame",
                            (
                                "calibrated R"
                                if self.use_extr
                                else f"manual {k.heading_deg:+.0f} deg"
                            ),
                            "ok" if self.use_extr else "warn",
                        ],
                        [
                            "tracker (OpenVR)",
                            " ".join(f"{v*1000:+.0f}" for v in ts.pos) + " mm",
                            "",
                        ],
                        ["displacement (base)", " ".join(f"{v*1000:+.0f}" for v in dp) + " mm", ""],
                        [
                            "reach used",
                            f"{reach_use*100:.0f}%",
                            "bad" if reach_use > 0.98 else "warn" if reach_use > 0.8 else "ok",
                        ],
                    ],
                    "arm": [
                        [
                            "manipulability",
                            f"{w_now:.5f}  (threshold {k.min_manip:.4f})",
                            (
                                "bad"
                                if w_now < k.min_manip
                                else ("warn" if w_now < k.min_manip * 1.5 else "ok")
                            ),
                        ],
                        [
                            "IK",
                            "ok" if info.get("ok") else info.get("reason", "not converged"),
                            "ok" if info.get("ok") else ("" if not eng else "warn"),
                        ],
                        ["IK failures", str(n_ikfail), "warn" if n_ikfail else "ok"],
                        [
                            "jumps (>15 deg/frame)",
                            f"{n_jump}, largest step {max_dq_ever:.1f} deg",
                            "bad" if n_jump else "ok",
                        ],
                        ["rejected by SDK", str(n_reject), "bad" if n_reject else "ok"],
                        [
                            "tracker step",
                            f"{trk_step_mm:.0f} mm/frame",
                            "bad" if trk_step_mm > 50 else ("warn" if trk_step_mm > 20 else "ok"),
                        ],
                        ["sent", str(n_sent), ""],
                        [
                            "joint angles",
                            " ".join(f"{v:+.0f}" for v in np.degrees(q_act)) + " deg",
                            "",
                        ],
                        ["loop rate", f"{n_loop/max(lt-t_start,1e-6):.0f} Hz", ""],
                    ],
                }
            slack = period - (time.monotonic() - lt)
            if slack > 0:
                time.sleep(slack)
