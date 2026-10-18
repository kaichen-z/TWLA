"""Vive Tracker poses from the tracker server (`vive/tracker_server.py`) over HTTP.

The server runs in its own environment (it needs OpenVR) and serves
`GET http://127.0.0.1:8765/api/poses`. Its open OpenVR connection also keeps SteamVR from
going to standby, which would switch the trackers off.

Poses are in the OpenVR frame: right-handed, +Y up, metres, origin defined by the base
stations. Every pose carries a validity flag (false when the tracker is occluded or out of
view) that must be checked before use.

    src = TrackerSource(serial="LHR-XXXXXXXX").start()
    s = src.latest()
    if s.ok:
        p = s.pos            # (3,) m, OpenVR frame
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

__all__ = ["TrackerSource", "TrackerState", "DEFAULT_URL", "VR_TO_ARM", "vr_to_arm"]

DEFAULT_URL = "http://127.0.0.1:8765/api/poses"

# Axis relabelling from OpenVR (+X right, +Y up, +Z backwards) to the arm base frame
# (+X forward, +Y left, +Z up): arm_x = -vr_z, arm_y = -vr_x, arm_z = vr_y. This is only the
# fallback when no Vive-robot rotation is given (see teleop_loop.load_extrinsics); the
# unknown rotation about the vertical axis is then left to the operator.
VR_TO_ARM = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def vr_to_arm(v: np.ndarray) -> np.ndarray:
    """A vector in the OpenVR frame relabelled to the arm base axes (see VR_TO_ARM)."""
    return VR_TO_ARM @ np.asarray(v, dtype=float).reshape(3)


@dataclass
class TrackerState:
    """Tracker state at one instant; use `ok`, not `pos`, to decide whether it is usable."""

    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))  # m; last valid value
    rot: np.ndarray = field(default_factory=lambda: np.eye(3))
    valid: bool = False
    battery: Optional[float] = None
    serial: str = ""
    age_s: float = float("inf")  # time since the last valid pose
    reason: str = "not started"  # why the pose is not usable; "" if it is

    @property
    def ok(self) -> bool:
        return self.valid and not self.reason


class TrackerSource:
    """Polls the tracker server in a background thread; `latest()` never blocks."""

    def __init__(
        self,
        serial: Optional[str] = None,
        url: str = DEFAULT_URL,
        rate_hz: float = 100.0,
        stale_s: float = 0.15,
    ) -> None:
        """serial: use only this tracker (None: the first one); stale_s: max pose age."""
        self.serial = serial
        self.url = url
        self.period = 1.0 / max(rate_hz, 1.0)
        self.stale_s = stale_s
        self._lock = threading.Lock()
        self._st = TrackerState()
        self._last_ok = 0.0
        self._running = False
        self._t: Optional[threading.Thread] = None
        self._n_poll = 0
        self._n_valid = 0

    def start(self) -> "TrackerSource":
        """Start polling and wait briefly for the first reply."""
        self._running = True
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        for _ in range(40):
            if self._st.reason != "not started":
                break
            time.sleep(0.05)
        s = self.latest()
        if s.ok:
            print(
                f"[tracker] {s.serial}  battery {100*(s.battery or 0):.0f}%  "
                f"position {np.round(s.pos*1000, 1).tolist()} mm"
            )
        else:
            print(f"[tracker] not available: {s.reason}")
            print(
                "  check that the base stations are powered, SteamVR is running and "
                "vive/tracker_server.py is running"
            )
        return self

    def close(self) -> None:
        """Stop polling."""
        self._running = False
        if self._t is not None:
            self._t.join(timeout=0.5)
            self._t = None

    def __enter__(self) -> "TrackerSource":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.close()

    def _loop(self) -> None:
        while self._running:
            t0 = time.monotonic()
            try:
                with urllib.request.urlopen(self.url, timeout=0.5) as r:
                    d = json.loads(r.read().decode())
                self._n_poll += 1
                self._ingest(d)
            except (urllib.error.URLError, OSError, TimeoutError) as exc:
                self._fail(f"tracker server not reachable ({type(exc).__name__})")
            except Exception as exc:  # noqa: BLE001
                self._fail(f"bad reply: {type(exc).__name__}: {exc}")
            slack = self.period - (time.monotonic() - t0)
            if slack > 0:
                time.sleep(slack)

    def _fail(self, reason: str) -> None:
        with self._lock:
            self._st.reason = reason
            self._st.valid = False
            self._st.age_s = time.monotonic() - self._last_ok if self._last_ok else float("inf")

    def _ingest(self, d: dict) -> None:
        devs = [x for x in d.get("devices", []) if x.get("kind") == "tracker"]
        if self.serial:
            devs = [x for x in devs if x.get("serial") == self.serial]
        if not devs:
            self._fail(f"no tracker (serial={self.serial or 'any'}): not paired or offline")
            return
        t = devs[0]
        now = time.monotonic()
        with self._lock:
            self._st.serial = t.get("serial", "")
            self._st.battery = t.get("battery")
            self._st.valid = bool(t.get("valid"))
            if self._st.valid and t.get("pos") is not None:
                self._st.pos = np.asarray(t["pos"], dtype=float).reshape(3)
                if t.get("rot") is not None:
                    self._st.rot = np.asarray(t["rot"], dtype=float).reshape(3, 3)
                self._last_ok = now
                self._n_valid += 1
                self._st.reason = ""
            else:
                # Keep the last position but report it as unusable.
                self._st.reason = "invalid pose (occluded, out of view or base station off)"
            self._st.age_s = now - self._last_ok if self._last_ok else float("inf")
            if not self._st.reason and self._st.age_s > self.stale_s:
                self._st.reason = f"stale pose ({self._st.age_s*1000:.0f} ms)"

    def latest(self) -> TrackerState:
        """Copy of the latest state."""
        with self._lock:
            s = TrackerState(
                pos=self._st.pos.copy(),
                rot=self._st.rot.copy(),
                valid=self._st.valid,
                battery=self._st.battery,
                serial=self._st.serial,
                age_s=self._st.age_s,
                reason=self._st.reason,
            )
        return s

    @property
    def stats(self) -> tuple[int, int]:
        """(number of polls, number of valid poses)."""
        return self._n_poll, self._n_valid
