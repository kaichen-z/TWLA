"""Read-only status probes for the teleoperation devices.

One background thread per subsystem (Vive Tracker, CAN buses, each arm and hand, each
glove) reports a `Status` with a level (ok / warn / bad / skip) for the status strip.

The probes never move anything: arms are connected without enabling, and the L6 hands only
receive read requests. A probe gives up its CAN connection (`suspend`) while a controller
needs the bus, since two connections on one bus steal each other's replies.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
for _p in (_ROOT, os.path.join(_ROOT, "drivers")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

OK, WARN, BAD, SKIP = "ok", "warn", "bad", "skip"

L6_HAND_ID = {"right": 0x27, "left": 0x28}  # CAN ids of the L6 hands

# (key, label, kind, side, default CAN interface). The interfaces are found by can_scan.py
# at start-up; the defaults are only used when the scan is skipped or fails.
DEVICES = [
    ("left_arm", "left arm", "arm", "left", "can0"),
    ("right_arm", "right arm", "arm", "right", "can1"),
    ("left_hand", "left hand", "hand", "left", "can2"),
    ("right_hand", "right hand", "hand", "right", "can3"),
]
CAN_CHOICES = ["can0", "can1", "can2", "can3"]


@dataclass
class Status:
    """State of one subsystem."""

    level: str = BAD
    headline: str = "not started"
    detail: list = field(default_factory=list)
    rows: list = field(default_factory=list)  # [(name, value, level), ...]
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "level": self.level,
            "headline": self.headline,
            "detail": self.detail,
            "rows": self.rows,
            **self.extra,
        }


class Probe(threading.Thread):
    """Polls one subsystem in the background; exceptions become a BAD status."""

    period = 0.5
    STICKY_N = 3  # consecutive failures before a good status turns bad (debouncing)

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self._lock = threading.Lock()
        self._st = Status()
        self._last_good: Status | None = None
        self._bad_run = 0
        self._running = True

    def run(self) -> None:
        while self._running:
            t0 = time.monotonic()
            try:
                st = self.poll()
            except Exception as exc:  # noqa: BLE001
                st = Status(BAD, f"probe error: {type(exc).__name__}", [str(exc)[:200]])
            with self._lock:
                self._st = st
                if st.level == BAD:
                    self._bad_run += 1
                else:
                    self._bad_run = 0
                    if st.level != SKIP:
                        self._last_good = st
            slack = self.period - (time.monotonic() - t0)
            if slack > 0:
                time.sleep(slack)

    def poll(self) -> Status:
        raise NotImplementedError

    def status(self) -> Status:
        """Latest status; a single failed read keeps the last good status."""
        with self._lock:
            st = self._st
            if st.level == BAD and self._bad_run < self.STICKY_N and self._last_good is not None:
                g = Status(
                    self._last_good.level,
                    self._last_good.headline,
                    list(self._last_good.detail)
                    + [f"({self._bad_run} failed read(s): {st.headline})"],
                    list(self._last_good.rows),
                    dict(self._last_good.extra),
                )
                return g
            return st

    def stop(self) -> None:
        self._running = False


# -- Vive Tracker -------------------------------------------------------------------------

_LOG = os.path.expanduser("~/.steam/debian-installation/logs/vrserver.txt")


def _steamvr_powered_off(serial: str) -> str:
    """Last SteamVR log line reporting that this device was powered off, or ""."""
    if not serial or not os.path.exists(_LOG):
        return ""
    try:
        with open(_LOG, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 200_000))
            tail = f.read().decode("utf-8", "replace")
    except OSError:
        return ""
    hits = [ln for ln in tail.splitlines() if serial in ln and "powering off" in ln.lower()]
    return hits[-1].strip()[:160] if hits else ""


def _steamvr_readded_after(serial: str, off_line: str) -> bool:
    """True if SteamVR re-added the device after the power-off line."""
    if not serial or not os.path.exists(_LOG):
        return False
    try:
        with open(_LOG, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 200_000))
            lines = f.read().decode("utf-8", "replace").splitlines()
    except OSError:
        return False
    idx = -1
    if off_line:
        for i, ln in enumerate(lines):
            if off_line[:60] in ln:
                idx = i
    return any(serial in ln and "finished adding tracked device" in ln for ln in lines[idx + 1 :])


class TrackerProbe(Probe):
    """SteamVR, base stations and tracker, read from the tracker server.

    SteamVR powers trackers off after 5 minutes of standby
    (`power.turnOffControllersTimeout`); the base stations then also report invalid poses,
    because their poses are estimated from the tracked devices. Pressing the tracker's
    button wakes it up.
    """

    def __init__(self, url: str, serial: Optional[str]) -> None:
        super().__init__()
        self.url = url
        self.serial = serial

    def poll(self) -> Status:
        import urllib.error
        import urllib.request

        try:
            with urllib.request.urlopen(self.url, timeout=0.6) as r:
                d = json.loads(r.read().decode())
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            return Status(
                BAD,
                "tracker server not reachable",
                [
                    f"{self.url} ({type(exc).__name__})",
                    "start SteamVR and vive/tracker_server.py",
                ],
            )

        devs = d.get("devices", [])
        trk = [x for x in devs if x.get("kind") == "tracker"]
        base = [x for x in devs if x.get("kind") == "base"]
        if self.serial:
            trk = [x for x in trk if x.get("serial") == self.serial] or trk

        rows = []
        for b in base:
            rows.append(
                (
                    f"base {b.get('serial','?')}",
                    "tracking" if b.get("valid") else "invalid",
                    OK if b.get("valid") else WARN,
                )
            )
        for t in trk:
            bat = t.get("battery")
            rows.append(
                (
                    f"tracker {t.get('serial','?')}",
                    ("tracking" if t.get("valid") else "invalid")
                    + (f"  battery {100*bat:.0f}%" if bat is not None else ""),
                    OK if t.get("valid") else BAD,
                )
            )

        if not trk:
            return Status(
                BAD,
                "no tracker",
                ["press the tracker's button; check the dongle and the pairing"],
                rows,
            )

        t0 = trk[0]
        if t0.get("valid"):
            p = t0.get("pos") or [0, 0, 0]
            return Status(
                OK,
                "tracking",
                [f"position {p[0]*1000:+.0f}, {p[1]*1000:+.0f}, {p[2]*1000:+.0f} mm (OpenVR)"],
                rows,
            )

        off = _steamvr_powered_off(t0.get("serial", ""))
        bases_dead = bool(base) and not any(b.get("valid") for b in base)
        if off or bases_dead:
            det = ["press the tracker's button (SteamVR powers trackers off after 5 min idle)"]
            if off:
                det.append(f"SteamVR log: {off}")
            if _steamvr_readded_after(t0.get("serial", ""), off):
                det.insert(
                    0,
                    "SteamVR re-added the tracker but poses are still invalid: restart "
                    "vive/tracker_server.py (its OpenVR connection is stale)",
                )
            return Status(BAD, "tracker powered off: press its button", det, rows)

        det = ["tracker online but not seen by the base stations (occluded / out of view)"]
        if t0.get("battery") is not None:
            det.append(f"battery {100*t0['battery']:.0f}%")
        return Status(BAD, "invalid pose (occluded / out of view)", det, rows)


# -- CAN buses ----------------------------------------------------------------------------


def _can_state() -> dict:
    """Up/down, bitrate and error count of every CAN interface (no root needed)."""
    out: dict = {}
    try:
        txt = subprocess.run(
            ["ip", "-s", "-d", "link", "show", "type", "can"],
            capture_output=True,
            text=True,
            timeout=3,
        ).stdout
    except Exception:  # noqa: BLE001
        return out
    cur = None
    for ln in txt.splitlines():
        m = re.match(r"^\d+:\s+(can\d+):.*?state (\w+)", ln)
        if m:
            cur = m.group(1)
            out[cur] = {"up": m.group(2).upper() == "UP", "bitrate": None, "err": None}
            continue
        if cur and "bitrate" in ln:
            b = re.search(r"bitrate (\d+)", ln)
            if b:
                out[cur]["bitrate"] = int(b.group(1))
        if cur and re.match(r"^\s+\d+\s+\d+\s+\d+", ln) and out[cur]["err"] is None:
            nums = ln.split()
            if len(nums) >= 3:
                try:
                    out[cur]["err"] = int(nums[2])
                except ValueError:
                    pass
    return out


class CanProbe(Probe):
    """The CAN interfaces used by the selected devices."""

    period = 1.0

    def __init__(self, devices: dict) -> None:
        super().__init__()
        self.devices = devices  # key -> DeviceProbe

    def poll(self) -> Status:
        st = _can_state()
        if not st:
            return Status(BAD, "no CAN interface", ["USB-CAN adapter not plugged in?"])
        used: dict = {}
        for k, dev in self.devices.items():
            if dev.enabled:
                used.setdefault(dev.can, []).append(dev.label)

        rows, need = [], []
        for name in sorted(st):
            v = st[name]
            who = used.get(name, [])
            label = name + (f" ({', '.join(who)})" if who else "")
            if v["up"]:
                txt = f"UP  {(v['bitrate'] or 0)//1000} kbps"
                if v["err"]:
                    txt += f"  errors {v['err']}"
                rows.append((label, txt, WARN if v["err"] else OK))
            else:
                rows.append((label, "DOWN", BAD if who else WARN))
                if who:
                    need.append(name)

        det = []
        dup = {c: w for c, w in used.items() if len(w) > 1}
        for c, w in dup.items():
            det.append(f"{c} is assigned to {', '.join(w)}")
        missing = [c for c in used if c not in st]
        if missing:
            return Status(BAD, f"interface missing: {', '.join(missing)}", det, rows)
        if need:
            return Status(
                BAD,
                f"{', '.join(need)} down",
                det
                + [
                    f"sudo ip link set {c} up type can bitrate 1000000 restart-ms 100" for c in need
                ],
                rows,
            )
        if not used:
            return Status(WARN, "no device selected", det, rows)
        return Status(WARN if dup else OK, "buses ready", det, rows)


# -- arms and hands -----------------------------------------------------------------------


class DeviceProbe(Probe):
    """One arm or hand; the user can select it and change its CAN interface at run time."""

    period = 0.5
    MAX_FAILS = 5  # stop retrying after this many failures (see _ensure)

    def __init__(
        self,
        key: str,
        label: str,
        kind: str,
        side: str,
        can: str,
        enabled: bool,
        poll_hand: bool = True,
    ) -> None:
        super().__init__()
        self.key, self.label, self.kind, self.side = key, label, kind, side
        self.poll_hand = poll_hand  # L6Can does not poll by itself
        self._cfg_lock = threading.Lock()
        self._can = can
        self._enabled = enabled  # the user's selection; never changed by the program
        self._suspended = ""  # name of the controller using the bus, "" if none
        self._h = None  # A7 or L6Can
        self._err = ""
        self._open_can = ""
        self._fails = 0
        self._next_try = 0.0

    @property
    def can(self) -> str:
        with self._cfg_lock:
            return self._can

    @property
    def enabled(self) -> bool:
        with self._cfg_lock:
            return self._enabled

    @property
    def suspended(self) -> str:
        with self._cfg_lock:
            return self._suspended

    def suspend(self, who: str) -> bool:
        """Give up the bus to `who`. Returns True if the probe was holding it.

        A probe already suspended for someone else is left alone, so the first owner's
        `resume()` cannot hand the bus back while another controller still uses it.
        """
        with self._cfg_lock:
            if not self._enabled:
                return False
            if self._suspended and self._suspended != who:
                return False
            self._suspended = who
        self._disconnect()
        return True

    def resume(self) -> None:
        """Take the bus back after `suspend`."""
        with self._cfg_lock:
            self._suspended = ""
            self._fails, self._next_try = 0, 0.0

    def configure(self, *, can: Optional[str] = None, enabled: Optional[bool] = None) -> None:
        """Change the CAN interface or the selection; resets the retry back-off."""
        with self._cfg_lock:
            changed = False
            if can is not None and can in CAN_CHOICES and can != self._can:
                self._can, changed = can, True
            if enabled is not None and bool(enabled) != self._enabled:
                self._enabled, changed = bool(enabled), True
            if changed:
                self._fails, self._next_try = 0, 0.0

    def _disconnect(self) -> None:
        h, self._h = self._h, None
        self._open_can = ""
        if h is not None:
            try:
                h.close()
            except Exception:  # noqa: BLE001
                pass

    def _ensure(self, can: str) -> None:
        """Connect if needed, with exponential back-off.

        A failed A7 connection leaks its socket, so unbounded retries would exhaust the
        file descriptors of the process; retrying stops after MAX_FAILS until the device
        is reconfigured.
        """
        if self._h is not None and self._open_can == can:
            return
        now = time.monotonic()
        if now < self._next_try:
            return
        if self._fails >= self.MAX_FAILS:
            return

        self._disconnect()
        cs = _can_state().get(can)
        if not cs or not cs["up"]:
            self._err = f"interface {can} missing" if not cs else f"{can} is down"
            self._backoff()
            return

        h = None
        try:
            if self.kind == "arm":
                # Constructed directly (not arm.connect) so a failed object can be closed.
                # Never enabled: disabled motors ignore motion commands.
                from realhand.arm import A7

                from arm import DEFAULT_ACCELERATION, DEFAULT_VELOCITY, NUM_JOINTS

                h = A7(side=self.side, interface_name=can)
                h.set_velocities([DEFAULT_VELOCITY] * NUM_JOINTS)
                h.set_accelerations([DEFAULT_ACCELERATION] * NUM_JOINTS)
            else:
                from robot.l6_can import L6Can

                h = L6Can(can, hand_id=L6_HAND_ID[self.side])
            self._h, self._open_can, self._err = h, can, ""
            self._fails, self._next_try = 0, 0.0
        except Exception as exc:  # noqa: BLE001
            if h is not None:
                try:
                    h.close()
                except Exception:  # noqa: BLE001
                    pass
            self._err = f"{type(exc).__name__}: {exc}"[:200]
            self._h = None
            self._backoff()

    def _backoff(self) -> None:
        self._fails += 1
        self._next_try = time.monotonic() + min(2.0 * (2 ** (self._fails - 1)), 15.0)

    def retry(self) -> None:
        """Reconnect now (e.g. after the device was power-cycled)."""
        self._fails, self._next_try = 0, 0.0
        self._disconnect()

    def poll(self) -> Status:
        can, enabled, susp = self.can, self.enabled, self.suspended
        extra = {
            "enabled": enabled,
            "can": can,
            "label": self.label,
            "kind": self.kind,
            "key": self.key,
            "suspended": susp,
        }
        if not enabled:
            self._disconnect()
            return Status(SKIP, "not selected", [], [], extra)
        if susp:
            return Status(WARN, f"in use by {susp}", [], [], extra)

        self._ensure(can)
        if self._h is None:
            det = [self._err or "unknown error", f"({self.side} / {can})"]
            if self._fails >= self.MAX_FAILS:
                det.append(f"{self._fails} failures, retrying stopped; reselect to retry")
            elif self._fails:
                wait = max(0.0, self._next_try - time.monotonic())
                det.append(f"failure {self._fails}, retrying in {wait:.0f} s")
            return Status(BAD, "not connected", det, [], extra)
        return (self._poll_arm if self.kind == "arm" else self._poll_hand)(can, extra)

    def _poll_arm(self, can: str, extra: dict) -> Status:
        try:
            import numpy as np

            a = np.degrees(np.asarray(self._h.get_angles(), dtype=float))
        except Exception as exc:  # noqa: BLE001
            self._disconnect()
            return Status(
                BAD, "joint read failed", [f"{type(exc).__name__}: {exc}"[:200]], [], extra
            )
        if len(a) != 7:
            return Status(
                WARN,
                f"{len(a)}/7 joints answered",
                ["motors not powered, emergency stop pressed, or not an arm bus"],
                [],
                extra,
            )
        rows = [(f"J{i+1}", f"{v:+7.2f} deg", OK) for i, v in enumerate(a)]
        det = [f"{self.side} arm on {can}, not enabled"]
        lvl = OK
        # Near J4 = 0 the arm is close to a singularity and the IK cannot move it.
        if abs(a[3]) < 45.0:
            det.append(f"J4 = {a[3]:+.1f} deg: bend the elbow beyond 45 deg before teleoperation")
            lvl = WARN
        return Status(lvl, "connected (not enabled)", det, rows, extra)

    def _poll_hand(self, can: str, extra: dict) -> Status:
        try:
            if self.poll_hand:
                from robot.l6_can import CMD_POSITION

                self._h.request(CMD_POSITION)  # read request, no payload
                time.sleep(0.05)
            snap = self._h.snapshot()
            age = self._h.age()
        except Exception as exc:  # noqa: BLE001
            self._disconnect()
            return Status(
                BAD, "hand read failed", [f"{type(exc).__name__}: {exc}"[:200]], [], extra
            )

        # The position list starts as zeros, so "has a position" does not mean "connected";
        # only a received frame does.
        if age is None:
            det = [
                f"no reply on {can} ({snap.get('tx_count', 0)} requests sent, "
                f"{snap.get('rx_count', 0)} frames received)",
                "hand not powered, or another device on this bus",
            ]
            return Status(BAD, "no reply", det, [], extra)

        pos = list((snap or {}).get("position") or [])
        names = ["thumb flex", "thumb abd", "index", "middle", "ring", "little"]
        rows = [(names[i], str(int(v)), OK) for i, v in enumerate(pos[:6])]
        rows.append(("receive rate", f"{snap.get('rx_hz', 0)} Hz", OK))
        stale = age > 1.0
        det = [f"{can}, read requests only", f"last frame {age:.2f} s ago"]
        if snap.get("firmware"):
            det.append(f"firmware {snap['firmware']}")
        return Status(
            WARN if stale else OK,
            f"stale ({age:.1f} s)" if stale else "connected",
            det,
            rows,
            extra,
        )

    def stop(self) -> None:
        super().stop()
        self._disconnect()


# -- gloves -------------------------------------------------------------------------------


def _glove_net_present() -> bool:
    """True if this computer has an address on the glove network (192.168.1.x)."""
    try:
        txt = subprocess.run(
            ["ip", "-4", "-br", "addr"], capture_output=True, text=True, timeout=3
        ).stdout
    except Exception:  # noqa: BLE001
        return True  # cannot check; let the SDK report
    return "192.168.1." in txt


class GloveProbe(Probe):
    """One Wuji glove. Runs inside glove_worker.py's child process.

    Retries are rate-limited: every failed SDK connection leaves SDK threads behind, and
    the SDK scan blocks the process. Without an address on the glove network the SDK is
    not called at all.
    """

    period = 0.3
    MAX_FAILS = 4

    def __init__(self, handedness: str) -> None:
        super().__init__()
        self.handedness = handedness
        self._src = None
        self._err = ""
        self._prev = (0, 0.0)
        self._fails = 0
        self._next_try = 0.0

    def retry(self) -> None:
        """Clear the back-off and retry now."""
        self._fails, self._next_try = 0, 0.0

    def _ensure(self) -> None:
        if self._src is not None:
            return
        if time.monotonic() < self._next_try or self._fails >= self.MAX_FAILS:
            return
        if not _glove_net_present():
            self._err = "no 192.168.1.x address on this computer: glove cable unplugged?"
            self._backoff()
            return
        src = None
        try:
            from glove.glove_source import WujiGloveSource
            from glove.hand_motion import MotionConfig

            src = WujiGloveSource(handedness=self.handedness, config=MotionConfig(init_still_s=1.0))
            src.start()
            self._src = src
            self._err = ""
            self._fails, self._next_try = 0, 0.0
        except Exception as exc:  # noqa: BLE001
            if src is not None:
                try:
                    src.close()  # otherwise SDK threads are left behind
                except Exception:  # noqa: BLE001
                    pass
            self._err = f"{type(exc).__name__}: {exc}"[:200]
            self._src = None
            self._backoff()

    def _backoff(self) -> None:
        self._fails += 1
        self._next_try = time.monotonic() + min(3.0 * (2 ** (self._fails - 1)), 30.0)

    def poll(self) -> Status:
        self._ensure()
        if self._src is None:
            det = [self._err or "unknown error"]
            if self._fails >= self.MAX_FAILS:
                det.append(f"{self._fails} failures, retrying stopped")
            det.append("check that the USB-C cable is plugged in the right way round")
            return Status(BAD, "glove not connected", det)
        f = self._src.latest()
        m = f.motion
        now = time.monotonic()
        if m.timestamp_us != self._prev[0]:
            self._prev = (m.timestamp_us, now)
        gap = now - self._prev[1]
        if gap > 0.5:
            return Status(BAD, f"no data for {gap:.1f} s", ["device timestamps stopped"])

        rows = [("orientation rpy", "  ".join(f"{v:+.1f}" for v in m.rpy_deg), OK)]
        sk, conf = f.shape.skeleton_pos, f.shape.joint_conf
        if sk is None:
            rows.append(("finger skeleton", "no data", WARN))
        else:
            c = float(min(conf)) if conf is not None else 1.0
            rows.append(("finger skeleton", f"min confidence {c:.2f}", OK if c >= 0.5 else WARN))
        if f.hand_age_ms is not None:
            rows.append(("finger frame age", f"{f.hand_age_ms:.0f} ms", OK))
        return Status(OK, "streaming", [f"{self.handedness} glove"], rows)

    def frame(self):
        """Latest GloveFrame, or None if not connected."""
        return self._src.latest() if self._src is not None else None

    def stop(self) -> None:
        super().stop()
        if self._src is not None:
            try:
                self._src.close()
            except Exception:  # noqa: BLE001
                pass
