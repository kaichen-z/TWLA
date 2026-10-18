"""Wrist orientation and IMU dead-reckoning from the glove's palm IMU (no SDK dependency).

Input: fused 6-axis IMU frames (orientation quaternion, angular velocity, linear
acceleration including gravity) at 800 Hz. Output: `HandMotionState`.

The IMU has no magnetometer: roll and pitch are absolute (gravity), yaw is relative to the
start of the stream and drifts slowly. The displacement `p_rel` is a double integral with
zero-velocity updates (ZUPT) and drifts within each motion segment; `pos_sigma_m` estimates
its uncertainty. Teleoperation takes the hand position from a Vive Tracker and does not use
`p_rel`.

Gravity is measured during a still period at start-up instead of being assumed along an
axis, so the result does not depend on the axis convention of the SDK's world frame. The
orientation is assumed to be body -> world (`cfg_quat_is_body_to_world`).

    python -m glove.hand_motion --selftest      # checks the integration on synthetic data
"""

from __future__ import annotations

import argparse
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional, Tuple

import numpy as np

try:
    from . import quat
except ImportError:  # pragma: no cover
    import quat  # type: ignore


@dataclass
class MotionConfig:
    """Estimator parameters; conservative defaults for a 16 g / 2000 deg/s / 800 Hz IMU."""

    init_still_s: float = 1.0  # still time at start-up used to measure gravity and bias

    # Still detection (triggers ZUPT).
    still_window_s: float = 0.10
    # Max gravity-free linear acceleration in the window (m/s^2). The norm of the raw
    # reading cannot be used: horizontal accelerations barely change it.
    still_acc_tol: float = 0.30
    still_var_tol: float = 0.20  # max deviation of the raw body-frame reading (m/s^2)
    still_gyr_tol: float = 0.08  # max angular rate (rad/s)

    # Online accelerometer bias estimate.
    bias_update_gain: float = 0.02  # EMA gain per still sample; 0 disables it
    max_bias: float = 1.0  # m/s^2

    # Integration.
    max_dt_s: float = 0.02  # larger gaps are treated as dropped frames and not integrated
    velocity_decay_tau_s: float = 0.0  # > 0 adds velocity damping (s); 0 = none

    # Uncertainty model for pos_sigma_m.
    tilt_error_rad: float = math.radians(1.0)  # assumed roll/pitch error (dominant term)
    accel_noise: float = 0.05  # m/s^2

    freeze_position_when_still: bool = True


@dataclass
class HandMotionState:
    """Hand motion at one instant."""

    timestamp_us: int = 0
    seq: int = 0

    # Orientation (x, y, z, w): roll/pitch absolute, yaw relative to the stream start.
    q_world_body: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0, 1.0]))
    q_rel: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0, 1.0]))
    rpy_deg: np.ndarray = field(default_factory=lambda: np.zeros(3))  # for display only

    # Position relative to the last rezero (m, world frame); drifts, see module docstring.
    p_rel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    v: np.ndarray = field(default_factory=lambda: np.zeros(3))
    a_lin_world: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Raw body-frame readings.
    gyro: np.ndarray = field(default_factory=lambda: np.zeros(3))  # rad/s
    accel: np.ndarray = field(default_factory=lambda: np.zeros(3))  # m/s^2, with gravity
    gyro_norm: float = 0.0

    # Quality of p_rel.
    is_still: bool = True
    since_zupt_s: float = 0.0
    pos_sigma_m: float = 0.0
    speed: float = 0.0

    # Diagnostics.
    accel_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))
    gravity_norm: float = 0.0
    initialized: bool = False
    dropped_frames: int = 0

    def as_dict(self) -> dict:
        """JSON-serialisable dict."""
        return {
            "timestamp_us": int(self.timestamp_us),
            "seq": int(self.seq),
            "q_world_body": [float(x) for x in self.q_world_body],
            "q_rel": [float(x) for x in self.q_rel],
            "rpy_deg": [float(x) for x in self.rpy_deg],
            "p_rel": [float(x) for x in self.p_rel],
            "v": [float(x) for x in self.v],
            "gyro": [float(x) for x in self.gyro],
            "accel": [float(x) for x in self.accel],
            "gyro_norm": float(self.gyro_norm),
            "is_still": bool(self.is_still),
            "since_zupt_s": float(self.since_zupt_s),
            "pos_sigma_m": float(self.pos_sigma_m),
            "speed": float(self.speed),
            "initialized": bool(self.initialized),
        }


class RelativeMotionEstimator:
    """Strapdown integration with zero-velocity updates. Not thread-safe.

    est = RelativeMotionEstimator()
    for frame in imu_stream:
        st = est.update(frame.t_us, frame.quat, frame.accel, frame.gyro)
    """

    def __init__(self, config: Optional[MotionConfig] = None) -> None:
        self.cfg = config or MotionConfig()
        self.reset()

    def reset(self) -> None:
        """Full reset, including the still-period initialisation."""
        self._t_prev_us: Optional[int] = None
        self._init_buf: list[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self._init_t0_us: Optional[int] = None
        self._initialized = False

        self._g_world = np.zeros(3)
        self._g_norm = 9.80665
        self._accel_bias = np.zeros(3)

        self._p = np.zeros(3)
        self._v = np.zeros(3)
        self._a_prev = np.zeros(3)

        self._q_origin = np.array([0.0, 0.0, 0.0, 1.0])
        self._p_origin = np.zeros(3)

        self._win: Deque[Tuple[int, float, float, np.ndarray]] = deque()
        self._is_still = True
        self._last_zupt_us: Optional[int] = None
        self._dropped = 0
        self._seq = 0
        self._state = HandMotionState()

    def rezero(self, state: Optional[HandMotionState] = None) -> None:
        """Make the current pose the origin; gravity and bias estimates are kept."""
        self._p_origin = self._p.copy()
        self._q_origin = self._state.q_world_body.copy()
        self._v[:] = 0.0
        self._last_zupt_us = self._t_prev_us

    def update(
        self,
        timestamp_us: int,
        quat_xyzw,
        accel_body,
        gyro_body,
    ) -> HandMotionState:
        """Process one IMU frame.

        timestamp_us: device time; quat_xyzw: fused orientation; accel_body: m/s^2 including
        gravity (ROS sensor_msgs/Imu convention); gyro_body: rad/s.
        """
        q = quat.normalize(np.asarray(quat_xyzw, dtype=np.float64).reshape(4))
        a_b = np.asarray(accel_body, dtype=np.float64).reshape(3)
        w_b = np.asarray(gyro_body, dtype=np.float64).reshape(3)

        if not self.cfg_quat_is_body_to_world:
            q = quat.conjugate(q)

        R = quat.to_matrix(q)
        a_world_raw = R @ (a_b - self._accel_bias)

        # Before initialisation gravity is unknown; fall back to the norm of the raw reading.
        if self._initialized:
            a_lin_norm = float(np.linalg.norm(a_world_raw - self._g_world))
        else:
            a_lin_norm = abs(float(np.linalg.norm(a_b)) - self._g_norm)
        w_norm = float(np.linalg.norm(w_b))
        self._win.append((timestamp_us, a_lin_norm, w_norm, a_b.copy()))
        win_span_us = self.cfg.still_window_s * 1e6
        while self._win and (timestamp_us - self._win[0][0]) > win_span_us:
            self._win.popleft()

        if not self._initialized:
            self._state = self._do_init(timestamp_us, q, a_b, w_b, a_world_raw)
            self._t_prev_us = timestamp_us
            return self._state

        still = self._detect_still()

        dt = 0.0
        if self._t_prev_us is not None:
            dt = (timestamp_us - self._t_prev_us) * 1e-6
        self._t_prev_us = timestamp_us
        if dt <= 0.0 or dt > self.cfg.max_dt_s:
            # Timestamp went back or the stream paused: update orientation only.
            if dt > self.cfg.max_dt_s:
                self._dropped += 1
            a_lin = a_world_raw - self._g_world
            self._a_prev = a_lin
            return self._emit(timestamp_us, q, a_lin, still, w_b, a_b)

        a_lin = a_world_raw - self._g_world

        # Trapezoidal integration.
        v_new = self._v + 0.5 * (a_lin + self._a_prev) * dt
        if self.cfg.velocity_decay_tau_s > 0.0:
            v_new *= math.exp(-dt / self.cfg.velocity_decay_tau_s)
        p_new = self._p + 0.5 * (self._v + v_new) * dt

        self._a_prev = a_lin

        if still:
            self._v[:] = 0.0  # ZUPT
            self._last_zupt_us = timestamp_us
            if not self.cfg.freeze_position_when_still:
                self._p = p_new
            self._update_bias(R)
        else:
            self._v = v_new
            self._p = p_new

        self._is_still = still
        return self._emit(timestamp_us, q, a_lin, still, w_b, a_b)

    cfg_quat_is_body_to_world: bool = True

    def _do_init(self, t_us, q, a_b, w_b, a_world_raw) -> HandMotionState:
        """Accumulate init_still_s of still data and measure the gravity vector."""
        if self._init_t0_us is None:
            self._init_t0_us = t_us

        # Restart the still period whenever the hand rotates.
        if float(np.linalg.norm(w_b)) > self.cfg.still_gyr_tol * 3.0:
            self._init_buf.clear()
            self._init_t0_us = t_us
        else:
            self._init_buf.append((q.copy(), a_b.copy(), a_world_raw.copy()))

        elapsed = (t_us - self._init_t0_us) * 1e-6
        st = HandMotionState(
            timestamp_us=t_us,
            seq=self._seq,
            q_world_body=q,
            rpy_deg=np.degrees(quat.to_rpy(q)),
            gyro=w_b.copy(),
            accel=a_b.copy(),
            gyro_norm=float(np.linalg.norm(w_b)),
            initialized=False,
            is_still=True,
        )
        self._seq += 1

        if elapsed >= self.cfg.init_still_s and len(self._init_buf) > 10:
            g_samples = np.array([s[2] for s in self._init_buf])
            self._g_world = g_samples.mean(axis=0)
            self._g_norm = float(np.linalg.norm(self._g_world))
            self._q_origin = q.copy()
            self._p_origin = np.zeros(3)
            self._p[:] = 0.0
            self._v[:] = 0.0
            self._a_prev = np.zeros(3)
            self._last_zupt_us = t_us
            self._initialized = True
            st.initialized = True
            st.gravity_norm = self._g_norm
        return st

    def _detect_still(self) -> bool:
        """Still if linear acceleration, angular rate and raw-reading deviation are all small."""
        if len(self._win) < 3:
            return False
        span = (self._win[-1][0] - self._win[0][0]) * 1e-6
        if span < self.cfg.still_window_s * 0.5:
            return False
        for _, a_lin_norm, w_norm, _ in self._win:
            if a_lin_norm > self.cfg.still_acc_tol:
                return False
            if w_norm > self.cfg.still_gyr_tol:
                return False
        a_all = np.array([s[3] for s in self._win])
        dev = np.linalg.norm(a_all - a_all.mean(axis=0), axis=1).max()
        return bool(dev <= self.cfg.still_var_tol)

    def _update_bias(self, R: np.ndarray) -> None:
        """Update the accelerometer bias while still.

        When still, the compensated reading (a_mean - bias) should equal R^T g_world. The
        error must use the compensated reading; with the raw one the update is an open-loop
        integrator that drives the bias away.
        """
        if self.cfg.bias_update_gain <= 0.0 or not self._win:
            return
        a_mean = np.mean([s[3] for s in self._win], axis=0)
        expected = R.T @ self._g_world
        err = (a_mean - self._accel_bias) - expected
        self._accel_bias = self._accel_bias + self.cfg.bias_update_gain * err
        np.clip(self._accel_bias, -self.cfg.max_bias, self.cfg.max_bias, out=self._accel_bias)

    def _emit(
        self,
        t_us: int,
        q: np.ndarray,
        a_lin: np.ndarray,
        still: bool,
        gyro: np.ndarray | None = None,
        accel: np.ndarray | None = None,
    ) -> HandMotionState:
        since = 0.0
        if self._last_zupt_us is not None:
            since = max(0.0, (t_us - self._last_zupt_us) * 1e-6)

        # Position uncertainty: gravity leaking through the tilt error, integrated twice.
        a_err = math.hypot(self.cfg.accel_noise, self._g_norm * math.sin(self.cfg.tilt_error_rad))
        sigma = 0.5 * a_err * since * since

        p_rel = self._p - self._p_origin
        q_rel = quat.multiply(quat.conjugate(self._q_origin), q)

        st = HandMotionState(
            timestamp_us=t_us,
            seq=self._seq,
            q_world_body=q,
            q_rel=q_rel,
            rpy_deg=np.degrees(quat.to_rpy(q)),
            p_rel=p_rel,
            v=self._v.copy(),
            a_lin_world=a_lin,
            gyro=np.zeros(3) if gyro is None else np.asarray(gyro, dtype=np.float64).reshape(3),
            accel=np.zeros(3) if accel is None else np.asarray(accel, dtype=np.float64).reshape(3),
            gyro_norm=0.0 if gyro is None else float(np.linalg.norm(gyro)),
            is_still=still,
            since_zupt_s=since,
            pos_sigma_m=sigma,
            speed=float(np.linalg.norm(self._v)),
            accel_bias=self._accel_bias.copy(),
            gravity_norm=self._g_norm,
            initialized=True,
            dropped_frames=self._dropped,
        )
        self._seq += 1
        self._state = st
        return st


def _selftest() -> int:
    """Three synthetic scenarios checking the integration and ZUPT (not sensor accuracy)."""
    rng = np.random.default_rng(0)
    rate = 800.0
    dt = 1.0 / rate
    g = np.array([0.0, 0.0, 9.80665])
    q_id = np.array([0.0, 0.0, 0.0, 1.0])
    ok = True

    def run(accel_fn, duration, noise=0.0, bias=np.zeros(3)):
        est = RelativeMotionEstimator(MotionConfig(init_still_s=0.5))
        n = int(duration * rate)
        st = None
        for i in range(n):
            t_us = int(i * dt * 1e6)
            a_true = accel_fn(i * dt)
            a_body = a_true + g + bias + rng.normal(0, noise, 3)
            st = est.update(t_us, q_id, a_body, np.zeros(3))
        return st

    # 1. Still for 5 s: no displacement, ZUPT active.
    st = run(lambda t: np.zeros(3), 5.0, noise=0.01)
    err = float(np.linalg.norm(st.p_rel))
    print(
        f"[1] still 5 s          displacement={err*1000:7.3f} mm  still={st.is_still}  "
        f"g={st.gravity_norm:.4f}"
    )
    if err > 1e-3 or not st.is_still:
        print("    FAIL: displacement while still")
        ok = False

    # 2. Still, 200 mm move, still: displacement close to 0.2 m, final velocity zeroed.
    move_t0, move_T, dist = 1.0, 1.0, 0.20

    def profile(t):
        if t < move_t0 or t > move_t0 + move_T:
            return np.zeros(3)
        u = (t - move_t0) / move_T
        a = (2.0 * math.pi * dist / (move_T**2)) * math.sin(2.0 * math.pi * u)
        return np.array([a, 0.0, 0.0])

    st = run(profile, 4.0, noise=0.005)
    print(
        f"[2] move 200 mm        estimate={st.p_rel[0]*1000:7.2f} mm  "
        f"residual speed={st.speed*1000:6.3f} mm/s"
    )
    if abs(st.p_rel[0] - dist) > 0.01:
        print("    FAIL: displacement error > 10 mm")
        ok = False
    if st.speed > 0.01:
        print("    FAIL: final velocity not zeroed")
        ok = False

    # 3. Accelerometer bias: ZUPT and bias estimation must reduce the drift.
    bias = np.array([0.02, 0.0, 0.0])
    st_on = run(lambda t: np.zeros(3), 10.0, noise=0.005, bias=bias)
    est_off = RelativeMotionEstimator(
        MotionConfig(init_still_s=0.5, still_acc_tol=0.0, still_gyr_tol=0.0)  # ZUPT never fires
    )
    st_off = None
    for i in range(int(10.0 * rate)):
        a_body = g + bias + rng.normal(0, 0.005, 3)
        st_off = est_off.update(int(i * dt * 1e6), q_id, a_body, np.zeros(3))
    print(
        f"[3] bias 0.02 m/s^2 10 s   with ZUPT={np.linalg.norm(st_on.p_rel)*1000:8.3f} mm   "
        f"without={np.linalg.norm(st_off.p_rel)*1000:8.1f} mm"
    )
    if np.linalg.norm(st_on.p_rel) >= np.linalg.norm(st_off.p_rel):
        print("    FAIL: ZUPT does not reduce the drift")
        ok = False

    print("\nself-test", "passed" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true", help="run the synthetic-data self-test")
    args = ap.parse_args()
    if args.selftest:
        raise SystemExit(_selftest())
    ap.print_help()
