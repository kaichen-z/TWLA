"""Bimanual teleoperation: each glove + tracker drives the arm and hand on its side.

Left glove and tracker -> left arm and hand; right -> right. One arm `TeleopLoop` and one
hand `L6Follow` per side share a single clutch.

Software limits are removed for this mode: no per-frame joint limit, no workspace sphere,
no manipulability threshold, no hand smoothing or rate limit, motor speed and acceleration
at the SDK maximum, and 1:1 displacement. Kept: the joint limits, holding the previous
command when the IK does not converge, and disengaging when a tracker is lost.

Instead, a fast-motion guard watches both trackers: if either hand moves faster than the
threshold while engaged, both sides lock and the arms are pinned. Unlocking (space bar)
re-engages after a 5 s delay and re-anchors, so nothing jumps. The guard also catches
tracker pose jumps, which look like very fast motion. Speed statistics (p50/p95/p99/max)
are collected while engaged to choose the threshold.

The tracker assigned to each hand is stored in teleop/tracker_roles.json.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import deque

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
ROLES_PATH = os.path.join(_HERE, "tracker_roles.json")
SIDES = ("left", "right")


def load_roles() -> dict:
    """Tracker serial number assigned to each side ("" if none)."""
    try:
        with open(ROLES_PATH, encoding="utf-8") as f:
            d = json.load(f)
        return {s: str(d.get(s) or "") for s in SIDES}
    except Exception:  # noqa: BLE001
        return {s: "" for s in SIDES}


def save_roles(roles: dict) -> None:
    tmp = ROLES_PATH + ".part"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({s: roles.get(s, "") for s in SIDES}, f, ensure_ascii=False, indent=2)
    os.replace(tmp, ROLES_PATH)


def auto_roles(serials: list, roles: dict) -> dict:
    """Assign trackers by sorted serial number if the stored assignment is not valid."""
    serials = sorted(serials)
    if all(roles.get(s) in serials for s in SIDES) and roles["left"] != roles["right"]:
        return roles
    if len(serials) >= 2:
        return {"left": serials[0], "right": serials[1]}
    return roles


class MotionGuard:
    """Locks the clutch when a tracker moves faster than the threshold while engaged."""

    RATE = 100.0
    WINDOW_S = 0.06  # speed estimation window
    STATS_MAX = 60_000  # speed samples kept (10 min at 100 Hz)

    def __init__(self, sources: dict, clutch, threshold: float = 1.5) -> None:
        self.sources, self.clutch = sources, clutch
        self.threshold = threshold
        self.enabled = True
        self.lock = threading.Lock()
        self._hist = {s: deque() for s in sources}
        self.speed = {s: 0.0 for s in sources}
        self._stats = {s: [] for s in sources}
        self.events: list = []
        self._run = False

    def start(self) -> None:
        self._run = True
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self) -> None:
        self._run = False

    def reset_stats(self) -> None:
        with self.lock:
            self._stats = {s: [] for s in self.sources}

    def _loop(self) -> None:
        period = 1.0 / self.RATE
        while self._run:
            t0 = time.monotonic()
            eng = self.clutch.is_engaged()
            for side, src in self.sources.items():
                st = src.latest()
                h = self._hist[side]
                if not st.ok:
                    h.clear()  # do not compute speed across a tracking loss
                    v = 0.0
                else:
                    now = time.monotonic()
                    p = np.asarray(st.pos, dtype=float)
                    if not h or not np.array_equal(h[-1][1], p):  # skip repeated samples
                        h.append((now, p))
                    while len(h) > 2 and now - h[1][0] >= self.WINDOW_S:
                        h.popleft()
                    v = 0.0
                    if len(h) >= 2 and h[-1][0] - h[0][0] > 1e-3:
                        v = float(np.linalg.norm(h[-1][1] - h[0][1]) / (h[-1][0] - h[0][0]))
                with self.lock:
                    self.speed[side] = v
                    if eng:
                        s = self._stats[side]
                        s.append(v)
                        if len(s) > self.STATS_MAX:
                            del s[: len(s) - self.STATS_MAX]
                    thr, on = self.threshold, self.enabled
                if eng and on and v > thr:
                    why = f"{side} hand {v:.2f} m/s > {thr:.2f}"
                    if self.clutch.guard_lock(why):
                        with self.lock:
                            self.events.append((time.strftime("%H:%M:%S"), why))
                            del self.events[:-20]
                    eng = False
            slack = period - (time.monotonic() - t0)
            if slack > 0:
                time.sleep(slack)

    def snapshot(self) -> dict:
        """Threshold, current speeds, lock events and speed statistics."""
        with self.lock:
            out = {
                "threshold": self.threshold,
                "enabled": self.enabled,
                "speed": dict(self.speed),
                "events": list(self.events),
                "stats": {},
            }
            for s, arr in self._stats.items():
                if arr:
                    a = np.asarray(arr)
                    out["stats"][s] = {
                        "n": int(a.size),
                        "p50": float(np.percentile(a, 50)),
                        "p95": float(np.percentile(a, 95)),
                        "p99": float(np.percentile(a, 99)),
                        "max": float(a.max()),
                    }
            return out


def final_knobs():
    """Knobs with the software limits removed (large finite values, not inf)."""
    from teleop_loop import Knobs

    return Knobs(
        velocity=50.0,  # SDK maximum
        acceleration=50.0,
        scale=1.0,
        max_reach=100.0,
        max_step_deg=360.0,
        max_rot_step_deg=360.0,
        min_manip=0.0,
        rot_src=1.0,  # orientation from the tracker
        rot_scale=1.0,
    )


class FinalTeleop:
    """Bimanual teleoperation: per side one TeleopLoop (arm) and one L6Follow (hand)."""

    # Start condition: J4 bent at least this much (the ready pose). At home J4 = 0 is
    # singular, and without the manipulability threshold the IK would command huge speeds.
    START_MIN_J4_DEG = 30.0

    def __init__(
        self, devices: dict, probes: dict, clutch, tracker_url: str, make_follow, recorder=None
    ) -> None:
        self.rec = recorder
        self.devices, self.probes, self.clutch = devices, probes, clutch
        self.tracker_url = tracker_url
        self.make_follow = make_follow  # side -> L6Follow
        self.lock = threading.Lock()
        self.roles = load_roles()
        self.loops: dict = {}
        self.follows: dict = {}
        self.sources: dict = {}
        self.guard: MotionGuard | None = None
        self.threshold = 1.5
        self.state, self.msg = "idle", ""
        self.live = False
        self.sides = ["left", "right"]  # sides to run; the others are not connected

    def ensure_roles(self, serials: list) -> None:
        """Assign trackers automatically if needed (see auto_roles)."""
        with self.lock:
            r = auto_roles(serials, self.roles)
            if r != self.roles:
                self.roles = r
                save_roles(r)

    def swap(self) -> str:
        """Swap the left and right trackers."""
        with self.lock:
            if self.state in ("running", "starting"):
                return "stop teleoperation first"
            self.roles = {"left": self.roles["right"], "right": self.roles["left"]}
            save_roles(self.roles)
        return ""

    def set_sides(self, v: str) -> str:
        """Select "left", "right" or "both"."""
        sides = {"left": ["left"], "right": ["right"], "both": ["left", "right"]}.get(v)
        if sides is None:
            return f"unknown option {v!r}"
        with self.lock:
            if self.state in ("running", "starting"):
                return "stop teleoperation first"
            self.sides = sides
        return ""

    def set_threshold(self, v: float) -> None:
        """Fast-motion lock threshold in m/s."""
        self.threshold = max(0.2, min(20.0, float(v)))
        if self.guard is not None:
            with self.guard.lock:
                self.guard.threshold = self.threshold

    def reset_stats(self) -> None:
        if self.guard is not None:
            self.guard.reset_stats()

    def start(self, live: bool) -> str:
        """Start the selected sides; live=False is a dry run that sends nothing."""
        from teleop_loop import TeleopLoop
        from tracker_source import TrackerSource

        with self.lock:
            if self.state in ("running", "starting"):
                return "already running"
            roles = dict(self.roles)
            sides = list(self.sides)
        if any(not roles[s] for s in sides) or (
            len(sides) == 2 and roles["left"] == roles["right"]
        ):
            return "trackers not assigned to left and right"
        arms = [s for s in sides if self.devices[f"{s}_arm"].enabled]
        hands = [s for s in sides if self.devices[f"{s}_hand"].enabled]
        if not arms and not hands:
            return f"no arm or hand selected on the {' and '.join(sides)} side"
        with self.lock:
            self.state, self.msg, self.live = "starting", "starting", live

        self.sources = {
            s: TrackerSource(serial=roles[s], url=self.tracker_url).start() for s in sides
        }
        self.loops, self.follows = {}, {}
        for s in arms:
            lp = TeleopLoop(
                self.sources[s],
                self.clutch,
                side=s,
                can=self.devices[f"{s}_arm"].can,
                live=live,
                min_manip=0.0,
                knobs=final_knobs(),
                devices=self.devices,
                glove=self.probes.get(f"glove_{s}"),
            )
            lp.require_j4_min_deg = self.START_MIN_J4_DEG
            lp.pin_on_release = True
            lp.rec = self.rec
            lp.start()
            self.loops[s] = lp
        for s in hands:
            fl = self.make_follow(s)
            fl.smooth, fl.max_step, fl.speed = 1.0, 255, 255  # no smoothing / rate limit
            fl.rate = 100.0  # glove finger data arrives at 120 Hz
            fl.fast_io = True  # interleaved command / position / tactile I/O (see L6Follow)
            fl.rec = self.rec
            err = fl.start(live)
            if not err:
                self.follows[s] = fl
        self.guard = MotionGuard(self.sources, self.clutch, self.threshold)
        self.guard.start()
        if self.rec is not None:
            self.rec.event(
                "final_start",
                live=live,
                arms=arms,
                hands=hands,
                roles=roles,
                threshold=self.threshold,
            )
        with self.lock:
            self.state = "running"
            self.msg = (
                "running: enable the clutch and press space"
                if live
                else "dry run (nothing sent): press space to see the data"
            )
        return ""

    def stop(self) -> str:
        """Stop all sides: arms pinned, hands opened."""
        if self.rec is not None and self.state in ("running", "starting"):
            self.rec.event("final_stop")
        for lp in self.loops.values():
            lp.stop()
        for fl in self.follows.values():
            fl.stop()
        if self.guard is not None:
            self.guard.stop()
        for src in self.sources.values():
            try:
                src.close()
            except Exception:  # noqa: BLE001
                pass
        with self.lock:
            if self.state in ("running", "starting"):
                self.state, self.msg = "stopped", "stopped (arms pinned, hands open)"
        return ""

    def snapshot(self) -> dict:
        """State of both sides and of the guard."""
        with self.lock:
            out = {
                "state": self.state,
                "msg": self.msg,
                "live": self.live,
                "sides": list(self.sides),
                "roles": dict(self.roles),
                "threshold": self.threshold,
                "start_min_j4": self.START_MIN_J4_DEG,
            }
        out["arms"] = {}
        for s, lp in self.loops.items():
            sn = lp.snapshot()
            tel = sn.get("tel") or {}
            out["arms"][s] = {
                "state": sn["state"],
                "msg": sn["msg"],
                "can": sn["can"],
                "log": sn["log"][-4:],
                "arm": tel.get("arm", []),
                "link": tel.get("link", []),
            }
        out["hands"] = {}
        for s, fl in self.follows.items():
            sn = fl.snapshot()
            out["hands"][s] = {"state": sn["state"], "msg": sn["msg"], "tel": sn.get("tel") or {}}
        out["guard"] = self.guard.snapshot() if self.guard is not None else None
        return out
