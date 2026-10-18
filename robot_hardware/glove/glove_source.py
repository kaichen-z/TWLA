"""Wuji data glove: connect, subscribe to its streams and merge them into `GloveFrame`s.

Each frame carries the finger joint angles and 21-point skeleton (from the glove's EMF
sensing, in the wrist frame, 120 Hz), the wrist orientation from the palm IMU (roll and
pitch absolute, yaw relative to start-up) and an IMU-based relative hand displacement
(see hand_motion.py). Teleoperation uses only the finger angles; the hand position comes
from a Vive Tracker.

Requires the Wuji SDK (`wuji_sdk`); it is imported lazily, so this module imports without it.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

try:
    from . import quat
    from .hand_motion import HandMotionState, MotionConfig, RelativeMotionEstimator
except ImportError:  # pragma: no cover
    import quat  # type: ignore
    from hand_motion import HandMotionState, MotionConfig, RelativeMotionEstimator  # type: ignore


FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")


@dataclass
class HandShape:
    """Finger state at one instant."""

    timestamp_us: int = 0
    # (5, 5) rad. The thumb uses all five slots; the other fingers use the first four
    # (MCP x2, PIP, DIP), the fifth is always 0. 21 DoF in total.
    joint_angles: Optional[np.ndarray] = None
    joint_conf: Optional[np.ndarray] = None  # (5,) per-finger confidence
    skeleton_names: tuple = ()
    skeleton_pos: Optional[np.ndarray] = None  # (21, 3) m, wrist frame
    skeleton_conf: Optional[np.ndarray] = None
    tip_pos: Optional[np.ndarray] = None  # (5, 3) m, wrist frame
    tip_quat: Optional[np.ndarray] = None  # (5, 4) (x, y, z, w)
    tip_conf: Optional[np.ndarray] = None


@dataclass
class GloveFrame:
    """Merged frame passed downstream."""

    motion: HandMotionState = field(default_factory=HandMotionState)
    shape: HandShape = field(default_factory=HandShape)
    # Age of the finger data relative to the IMU sample, in ms (normally below ~8 ms).
    hand_age_ms: float = float("nan")
    # Wrist orientation (world <- wrist); None if the IMU mounting transform is unknown.
    q_wrist: Optional[np.ndarray] = None


class WujiGloveSource:
    """One Wuji glove producing `GloveFrame`s.

        with WujiGloveSource(handedness="right") as src:
            src.wait_until_initialized()
            f = src.latest()

    SDK callbacks run in background threads; `latest()` can be called from any thread.
    """

    def __init__(
        self,
        handedness: Optional[str] = None,
        sn: Optional[str] = None,
        address: Optional[str] = None,
        device_name: str = "glove",
        config: Optional[MotionConfig] = None,
        subscribe_shape: bool = True,
        subscribe_tips: bool = False,
        imu_rate_hz: Optional[int] = None,
        on_frame: Optional[Callable[[GloveFrame], None]] = None,
    ) -> None:
        self._handedness = handedness
        self._sn = sn
        self._address = address
        self._device_name = device_name
        self._subscribe_shape = subscribe_shape
        self._subscribe_tips = subscribe_tips
        self._imu_rate_hz = imu_rate_hz
        self._on_frame = on_frame

        self.estimator = RelativeMotionEstimator(config)
        self._lock = threading.Lock()
        self._shape = HandShape()
        self._frame = GloveFrame()
        self._q_imu_from_wrist: Optional[np.ndarray] = None

        self._manager: Any = None
        self.glove: Any = None
        self._subs: list[Any] = []
        self._imu_count = 0
        self._shape_count = 0

    def connect(self) -> "WujiGloveSource":
        """Scan the network and connect to the glove selected by sn, address or handedness."""
        from wuji_sdk import Handedness, SdkManager

        self._manager = SdkManager.instance()
        found = self._manager.scan()
        if not found:
            raise RuntimeError(
                "no glove found: check the cable and power, and that this computer is on "
                "192.168.1.0/24 (default glove addresses: left 192.168.1.100, right "
                "192.168.1.101)"
            )
        for d in found:
            print(f"  found: SN={d.sn} type={getattr(d, 'device_type', '?')} addr={d.address}")

        if self._sn:
            self.glove = self._manager.connect(sn=self._sn, device_name=self._device_name)
        elif self._address:
            self.glove = self._manager.connect(address=self._address, device_name=self._device_name)
        elif self._handedness:
            h = Handedness.Left if self._handedness.lower().startswith("l") else Handedness.Right
            self.glove = self._manager.connect(handedness=h, device_name=self._device_name)
        elif len(found) == 1:
            self.glove = self._manager.auto_connect(device_name=self._device_name)
        else:
            raise RuntimeError(
                f"{len(found)} gloves found; select one by handedness or serial number"
            )

        sn = getattr(self.glove, "serial_number", "?")
        print(f"connected: {sn}")
        self._load_static_tf()
        return self

    def _load_static_tf(self) -> None:
        """Read the fixed wrist -> palm IMU transform; optional (a constant rotation)."""
        try:
            sub = self._manager.tf_static().subscribe()
            deadline = time.monotonic() + 2.0  # published at 1 Hz
            while time.monotonic() < deadline:
                msg = sub.recv()
                if msg is None:
                    time.sleep(0.02)
                    continue
                for tf in msg.transforms:
                    if str(tf.child_frame_id).endswith("palm_imu_link"):
                        r = tf.rotation
                        self._q_imu_from_wrist = quat.normalize(
                            np.array([r.x, r.y, r.z, r.w], dtype=np.float64)
                        )
                        print(
                            f"  tf_static: {tf.parent_frame_id} -> {tf.child_frame_id} "
                            f"t={list(tf.translation)}"
                        )
                        break
                if self._q_imu_from_wrist is not None:
                    break
            sub.close()
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] no tf_static ({exc}); orientation stays in the palm IMU frame")

    def start(self) -> "WujiGloveSource":
        """Connect if needed and subscribe to the IMU and finger streams."""
        if self.glove is None:
            self.connect()

        imu_sub = self.glove.imu_data_palm().subscribe_with_callback(callback=self._on_imu)
        self._subs.append(imu_sub)
        if self._imu_rate_hz:
            try:
                actual = imu_sub.set_rate(self._imu_rate_hz)
                print(f"  IMU rate: requested {self._imu_rate_hz} Hz, got {actual} Hz")
            except Exception as exc:  # noqa: BLE001
                print(f"  [warn] set_rate failed: {exc}")

        if self._subscribe_shape:
            self._subs.append(
                self.glove.hand_joint_angles().subscribe_with_callback(callback=self._on_angles)
            )
            self._subs.append(
                self.glove.hand_skeleton().subscribe_with_callback(callback=self._on_skeleton)
            )
        if self._subscribe_tips:
            self._subs.append(
                self.glove.tip_poses().subscribe_with_callback(callback=self._on_tips)
            )
        print(f"  subscribed to {len(self._subs)} streams")
        return self

    # SDK callbacks swallow exceptions, so each one prints its own errors.

    def _on_imu(self, frame: Any) -> None:
        try:
            cov = getattr(frame, "orientation_covariance", None)
            if cov is not None and len(cov) and float(cov[0]) < 0.0:
                # REP-145: -1 means no orientation, i.e. the raw IMU stream was subscribed.
                if self._imu_count == 0:
                    print("[error] IMU frames have no orientation; subscribe to imu_data_palm()")
                return

            o = frame.orientation
            a = frame.linear_acceleration
            w = frame.angular_velocity
            st = self.estimator.update(
                int(frame.header.timestamp_us),
                (o.x, o.y, o.z, o.w),
                (a.x, a.y, a.z),
                (w.x, w.y, w.z),
            )
            with self._lock:
                self._imu_count += 1
                shape = self._shape
                age = float("nan")
                if shape.timestamp_us:
                    age = (st.timestamp_us - shape.timestamp_us) / 1000.0
                q_wrist = None
                if self._q_imu_from_wrist is not None:
                    # world <- wrist = (world <- imu) * (imu <- wrist)^-1
                    q_wrist = quat.multiply(st.q_world_body, quat.conjugate(self._q_imu_from_wrist))
                self._frame = GloveFrame(motion=st, shape=shape, hand_age_ms=age, q_wrist=q_wrist)
                cb, fr = self._on_frame, self._frame
            if cb is not None:
                cb(fr)
        except Exception as exc:  # noqa: BLE001
            print(f"[error] IMU callback: {type(exc).__name__}: {exc}")

    def _on_angles(self, frame: Any) -> None:
        try:
            angles = np.array([list(f.angles) for f in frame.fingers], dtype=np.float64)
            conf = np.array([float(f.confidence) for f in frame.fingers], dtype=np.float64)
            with self._lock:
                self._shape.timestamp_us = int(frame.header.timestamp_us)
                self._shape.joint_angles = angles
                self._shape.joint_conf = conf
                self._shape_count += 1
        except Exception as exc:  # noqa: BLE001
            print(f"[error] joint angle callback: {type(exc).__name__}: {exc}")

    def _on_skeleton(self, frame: Any) -> None:
        try:
            names = tuple(str(j.name) for j in frame.joints)
            pos = np.array([list(j.pose.position) for j in frame.joints], dtype=np.float64)
            conf = np.array([float(j.confidence) for j in frame.joints], dtype=np.float64)
            with self._lock:
                self._shape.timestamp_us = int(frame.header.timestamp_us)
                self._shape.skeleton_names = names
                self._shape.skeleton_pos = pos
                self._shape.skeleton_conf = conf
        except Exception as exc:  # noqa: BLE001
            print(f"[error] skeleton callback: {type(exc).__name__}: {exc}")

    def _on_tips(self, frame: Any) -> None:
        try:
            pos = np.array([list(p.pose.position) for p in frame.poses], dtype=np.float64)
            qs = np.array(
                [
                    [
                        p.pose.orientation.x,
                        p.pose.orientation.y,
                        p.pose.orientation.z,
                        p.pose.orientation.w,
                    ]
                    for p in frame.poses
                ],
                dtype=np.float64,
            )
            conf = np.array([float(p.confidence) for p in frame.poses], dtype=np.float64)
            with self._lock:
                self._shape.tip_pos = pos
                self._shape.tip_quat = qs
                self._shape.tip_conf = conf
        except Exception as exc:  # noqa: BLE001
            print(f"[error] fingertip callback: {type(exc).__name__}: {exc}")

    def latest(self) -> GloveFrame:
        """Latest merged frame."""
        with self._lock:
            return self._frame

    @property
    def counts(self) -> tuple[int, int]:
        """(IMU frames, finger frames) received."""
        with self._lock:
            return self._imu_count, self._shape_count

    def wait_until_initialized(self, timeout_s: float = 20.0) -> bool:
        """Block until the IMU estimator has measured gravity; keep the hand still."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.latest().motion.initialized:
                return True
            time.sleep(0.05)
        return False

    def rezero(self) -> None:
        """Make the current pose the new origin of the relative motion."""
        with self._lock:
            self.estimator.rezero()

    def close(self) -> None:
        """Unsubscribe and disconnect."""
        for sub in self._subs:
            try:
                sub.close()
            except Exception:  # noqa: BLE001, S110
                pass
        self._subs.clear()
        if self._manager is not None and self.glove is not None:
            try:
                self._manager.disconnect(device_name=self._device_name)
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] disconnect failed: {exc}")
        self.glove = None

    def __enter__(self) -> "WujiGloveSource":
        return self.start()

    def __exit__(self, *exc_info: Any) -> None:
        self.close()
