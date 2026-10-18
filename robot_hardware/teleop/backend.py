"""Teleoperation backend used by the desktop UI (`app.py`).

`build_ctx()` starts everything the UI needs and returns it as a dict:

    probes / devices   status probes (devices.py); devices = the four arms and hands
    clutch             the shared clutch (clutch.py)
    armops, runner     enable / ready pose / home / relax for the arms, one job at a time
    final              bimanual teleoperation (final_teleop.py)
    rec                data recorder (recorder.py)

Only one connection per CAN bus is possible, so every controller suspends the status probe
of the device it drives and resumes it when done. Arm motions (ready, home, enable,
disable) run `drivers/arm.py` in subprocesses.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
from dataclasses import dataclass, field

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
for _p in (_ROOT, _HERE, os.path.join(_ROOT, "drivers")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from clutch import Clutch  # noqa: E402
from devices import (  # noqa: E402
    BAD,
    DEVICES,
    L6_HAND_ID,
    OK,
    SKIP,
    WARN,
    CanProbe,
    DeviceProbe,
    Probe,
    Status,
    TrackerProbe,
)
from final_teleop import FinalTeleop  # noqa: E402
from glove_worker import GloveProxy, _die_with_parent  # noqa: E402
from recorder import Recorder  # noqa: E402

VERSION = "1.0"
PY = sys.executable
ARM_OP = "arm operation"  # name under which arm jobs suspend the arm probes


def l6_calib_path(side: str) -> str:
    """Glove -> L6 calibration file of one hand."""
    from glove.l6_calib import default_path

    return str(default_path(side))


# -- background jobs ----------------------------------------------------------------------


@dataclass
class Job:
    """A background action and its log."""

    name: str = ""
    state: str = "idle"  # idle | running | done | failed
    msg: str = ""
    log: list = field(default_factory=list)
    started: float = 0.0

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "state": self.state,
            "msg": self.msg,
            "log": self.log[-200:],
            "elapsed": (time.time() - self.started) if self.started else 0.0,
        }


class JobRunner:
    """Runs one background job at a time (an arm accepts one command at a time)."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.job = Job()
        self._t: threading.Thread | None = None

    def busy(self) -> bool:
        with self.lock:
            return self.job.state == "running"

    def start(self, name: str, fn) -> str:
        """Run fn(say) in a thread; returns "" or why it did not start."""
        with self.lock:
            if self.job.state == "running":
                return f"busy: {self.job.name}"
            self.job = Job(name=name, state="running", msg="starting", started=time.time())
        self._t = threading.Thread(target=self._run, args=(fn,), daemon=True)
        self._t.start()
        return ""

    def _run(self, fn) -> None:
        try:
            fn(self.say)
            with self.lock:
                if self.job.state == "running":
                    self.job.state, self.job.msg = "done", "done"
        except Exception as exc:  # noqa: BLE001
            with self.lock:
                self.job.state = "failed"
                self.job.msg = f"{type(exc).__name__}: {exc}"[:200]
                self.job.log.append(f"error: {self.job.msg}")

    def say(self, line: str, msg: str | None = None) -> None:
        """Append a log line and optionally set the job message."""
        with self.lock:
            self.job.log.append(line)
            del self.job.log[:-200]
            if msg is not None:
                self.job.msg = msg

    def snapshot(self) -> dict:
        with self.lock:
            return self.job.as_dict()


# -- arm operations -----------------------------------------------------------------------

# Enable an arm with its targets pinned to the current position (arm.connect does this).
_ENABLE_SNIPPET = """
import sys
sys.path.insert(0, "drivers")
from arm import connect, format_angles
side, can = sys.argv[1], sys.argv[2]
a = connect(side, can, enable=True)
print(f"{side} arm on {can}")
print("  current  " + format_angles(a.get_angles()))
print("  enabled, targets pinned to the current position")
a.close()
"""

# Ready pose, reached one joint at a time (joint number, degrees, motor order): J2 away
# from the body, then J4 bends the elbow so the forearm points forward. The right arm also
# turns J5 so that the palm faces outwards. It also keeps the arm away from the singular home
# pose. Moving one joint at a time avoids sweeping the hand over the table.
READY_STEPS = {
    "right": [(2, -60.0), (4, 90.0), (5, 90.0)],
    "left": [(2, +60.0), (4, 90.0)],
}
# From any pose to the ready pose: J2 first, then the joints from the wrist back to the
# shoulder, J4 before J1.
READY_RECOVER = {
    "right": [(2, -60.0), (7, 0.0), (6, 0.0), (5, 90.0), (3, 0.0), (4, 90.0), (1, 0.0)],
    "left": [(2, +60.0), (7, 0.0), (6, 0.0), (5, 0.0), (3, 0.0), (4, 90.0), (1, 0.0)],
}


class ArmOps:
    """Enable, ready pose, home and relax for the selected arms."""

    def __init__(self, devices: dict, runner: JobRunner) -> None:
        self.devices = devices
        self.runner = runner
        self.live_procs: list = []  # running arm.py processes
        self.sides_fn = lambda: ["left", "right"]  # sides selected for teleoperation
        self.busy_fn = lambda: []  # controllers currently driving an arm

    def stop_all(self) -> None:
        """Terminate any running arm.py process."""
        for proc in list(self.live_procs):
            if proc.poll() is None:
                try:
                    proc.terminate()
                    proc.wait(timeout=5)
                except Exception:  # noqa: BLE001
                    try:
                        proc.kill()
                    except Exception:  # noqa: BLE001
                        pass
        self.live_procs = []

    def _arm_probes(self) -> list:
        return [d for d in self.devices.values() if d.kind == "arm"]

    def _release(self) -> list:
        """Suspend the arm probes so that arm.py can use the buses."""
        was = [d for d in self._arm_probes() if d.suspend(ARM_OP)]
        time.sleep(1.2)  # let them close their sockets
        return was

    def _restore(self, was: list) -> None:
        time.sleep(0.5)
        for d in was:
            d.resume()

    def _targets(self) -> list:
        """(side, can) of the arms that are both selected and on a selected side."""
        sides = set(self.sides_fn())
        return [(d.side, d.can) for d in self._arm_probes() if d.enabled and d.side in sides]

    def target_names(self) -> list:
        return [f"{s} arm" for s, _c in self._targets()]

    def _refuse_if_busy(self) -> str:
        busy = self.busy_fn()
        if busy:
            return f"{', '.join(busy)} is driving the arm; stop it first"
        return ""

    def home_all(self) -> str:
        """Ready pose, then back to zero in the reverse order."""
        if e := self._refuse_if_busy():
            return e
        tg = self._targets()
        if not tg:
            return "no arm selected"
        return self.runner.start("home", lambda say: self._do(tg, "home", say))

    def ready_all(self) -> str:
        """From any pose to the ready pose, one joint at a time."""
        if e := self._refuse_if_busy():
            return e
        tg = self._targets()
        if not tg:
            return "no arm selected"
        return self.runner.start("ready pose", lambda say: self._do(tg, "ready", say))

    def enable_all(self) -> str:
        """Enable the arms in place (targets pinned to the current position first)."""
        if e := self._refuse_if_busy():
            return e
        tg = self._targets()
        if not tg:
            return "no arm selected"
        return self.runner.start("enable", lambda say: self._do(tg, "enable", say))

    def relax_all(self) -> str:
        """Disable the arms. Never refused; the caller stops teleoperation first."""
        tg = self._targets()
        if not tg:
            return "no arm selected"
        return self.runner.start("relax", lambda say: self._do(tg, "disable", say))

    def _do(self, targets: list, what: str, say) -> None:
        was = self._release()
        try:
            for side, can in targets:
                label = f"{side} arm on {can}"
                verb = {
                    "home": "home",
                    "disable": "disable",
                    "enable": "enable",
                    "ready": "ready pose",
                }[what]
                say(f"-- {label}: {verb} --", f"{label}: {verb}")
                arm_py = [
                    PY,
                    os.path.join(_ROOT, "drivers", "arm.py"),
                    "--side",
                    side,
                    "--channel",
                    can,
                ]
                if what == "enable":
                    cmds = [[PY, "-c", _ENABLE_SNIPPET, side, can]]
                elif what == "ready":
                    # One joint per command; a failed step stops the sequence.
                    cmds = [
                        arm_py + ["--joint", str(j), "--to", f"{deg:g}"]
                        for j, deg in READY_RECOVER[side]
                    ]
                elif what == "home":
                    cmds = [
                        arm_py + ["--joint", str(j), "--to", f"{deg:g}"]
                        for j, deg in READY_RECOVER[side]
                    ]
                    cmds += [
                        arm_py + ["--joint", str(j), "--to", "0"]
                        for j, _deg in reversed(READY_STEPS[side])
                    ]
                    cmds += [arm_py + ["--home", "--velocity", "0.2", "--acceleration", "2.0"]]
                else:
                    cmds = [arm_py + ["--disable"]]
                for cmd in cmds:
                    self._run_one(cmd, label, what, say)
            if what == "disable":
                say("arms disabled: they can be moved by hand and may drop")
            elif what == "enable":
                say("arms enabled, holding their position")
            elif what == "ready":
                say("ready pose reached; arms stay enabled")
        finally:
            self._restore(was)

    def _run_one(self, cmd: list, label: str, what: str, say) -> None:
        env = dict(os.environ, PYTHONNOUSERSITE="1")
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, cwd=_ROOT
        )
        self.live_procs.append(proc)
        try:
            out, errout = proc.communicate(timeout=180)
        finally:
            self.live_procs = [x for x in self.live_procs if x is not proc]
        for ln in (out or "").splitlines():
            if ln.strip():
                say("  " + ln.rstrip())
        if proc.returncode != 0:
            # arm.py prints its refusals (homing blocked, target outside limits) on stdout.
            err = (
                (errout or "").strip().splitlines()[-1:]
                or [ln for ln in (out or "").splitlines() if "refused" in ln or "aborting" in ln][
                    :1
                ]
                or ["unknown error"]
            )
            say(f"  failed: {err[0][:160]}")
            raise RuntimeError(f"{label} {what} failed: {err[0][:80]}")


# -- glove -> L6 hand ---------------------------------------------------------------------


class L6Follow:
    """The L6 hand follows the glove through the glove calibration (glove/l6_calib.py).

    The hand command is smoothed (EMA `smooth`) and rate-limited (`max_step` per frame);
    on stop the hand opens to 255 so it is not left gripping. While the clutch is
    disengaged the hand keeps its current shape. The hand's status probe is suspended
    while following.

    With `fast_io` a single I/O thread sends, in turn, the hand command, a position request
    and the five fingertip tactile requests. The L6 serialises requests: a tactile request
    takes ~10 ms and is dropped if any other request arrives meanwhile, so interleaving is
    the only way to read tactile data while commanding. The cycle runs at ~14 Hz (positions,
    commands and full-hand tactile alike).
    """

    RATE = 30.0

    def __init__(
        self,
        devices: dict,
        probes: dict,
        clutch=None,
        side: str = "right",
        glove_key: str | None = None,
    ) -> None:
        self.devices, self.probes = devices, probes
        self.glove_key = glove_key or f"glove_{side}"
        self.clutch, self.side = clutch, side  # clutch None: always follow
        self.lock = threading.Lock()
        self.rate = self.RATE
        self.rec = None  # recorder.Recorder, or None
        self.fast_io = False
        self._pending = None  # (out, target, engaged): written by the loop, sent by the I/O thread
        self._t_cur = None  # tactile command currently awaited
        self._tbuf = None
        self._tseen: set = set()
        self._tdone = threading.Event()
        self._pdone = threading.Event()
        self._n_sent = 0
        self.io_stat = {"cycles": 0, "touch_ok": 0, "touch_lost": 0, "t0": 0.0}
        self.smooth, self.max_step = 0.35, 12
        self.speed, self.torque = 150, 150
        self.state, self.msg = "idle", ""
        self.live = False
        self.engaged = False
        self.tel: dict = {}
        self._run = False
        self._released = []

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "state": self.state,
                "msg": self.msg,
                "live": self.live,
                "engaged": self.engaged,
                "smooth": self.smooth,
                "max_step": self.max_step,
                "speed": self.speed,
                "torque": self.torque,
                "tel": dict(self.tel),
            }

    def start(self, live: bool) -> str:
        """Start following; live=False computes the command without sending it."""
        with self.lock:
            if self._run:
                return "already running"
            self.live, self.state, self.msg = live, "starting", "starting"
        self._run = True
        threading.Thread(target=self._main, daemon=True).start()
        return ""

    def stop(self) -> str:
        self._run = False
        return ""

    def _on_frame(self, cmd: int, body: list, t_ns: int) -> None:
        """Called in the L6 receive thread: record positions and complete tactile arrays.

        Only rows of the finger currently requested are accepted, and the buffer is reset
        before each request, so partial reads are never merged into one frame.
        """
        import numpy as np
        from robot.l6_can import _ROW_CODE, CMD_POSITION, TOUCH_CMDS

        rec = self.rec
        if cmd == CMD_POSITION and len(body) == 6:
            if rec is not None:
                rec.put(f"{self.side}_hand_pos", t_ns, pos=np.asarray(body, dtype=np.uint8))
            self._pdone.set()
        elif cmd in TOUCH_CMDS and len(body) == 7 and cmd == self._t_cur:
            row = _ROW_CODE.get(body[0])
            buf = self._tbuf
            if row is None or buf is None:
                return
            buf[row] = body[1:7]
            self._tseen.add(row)
            if len(self._tseen) == 12:
                if rec is not None:
                    rec.put(
                        f"{self.side}_tactile",
                        t_ns,
                        finger=TOUCH_CMDS.index(cmd),
                        data=buf.copy(),
                    )
                self._tdone.set()

    def _io_loop(self, hand) -> None:
        """The only sender on the bus with fast_io: command, position, five tactile reads."""
        import numpy as np
        from robot.l6_can import CMD_POSITION, TOUCH_CMDS

        st = self.io_stat
        st.update(cycles=0, touch_ok=0, touch_lost=0, t0=time.monotonic())
        while self._run:
            p = self._pending
            if p is not None:
                out, tgt, eng = p
                sent = False
                if self.live and eng:
                    try:
                        hand.write_position(out)
                        self._n_sent += 1
                        sent = True
                    except Exception:  # noqa: BLE001
                        pass
                if self.rec is not None:
                    self.rec.put(
                        f"{self.side}_hand_cmd",
                        time.monotonic_ns(),
                        target=np.asarray(tgt, dtype=np.float32),
                        cmd=np.asarray(out, dtype=np.uint8),
                        engaged=bool(eng),
                        sent=sent,
                    )
            self._pdone.clear()
            try:
                hand.request(CMD_POSITION)
            except Exception:  # noqa: BLE001
                pass
            self._pdone.wait(0.005)  # wait for the position reply before the tactile reads
            # Only 0xB1-0xB5 (TOUCH_CMDS); never 0xB6 (see l6_can.py).
            for fi in range(5):
                if not self._run:
                    break
                self._tbuf = np.zeros((12, 6), dtype=np.uint8)
                self._tseen = set()
                self._tdone.clear()
                self._t_cur = TOUCH_CMDS[fi]
                try:
                    hand.request_touch(fi)
                except Exception:  # noqa: BLE001
                    pass
                if self._tdone.wait(0.03):
                    st["touch_ok"] += 1
                else:
                    st["touch_lost"] += 1
            self._t_cur = None
            st["cycles"] += 1

    def _io_rates(self) -> dict:
        st = self.io_stat
        dt = max(time.monotonic() - st["t0"], 1e-6) if st["t0"] else 0
        if not dt:
            return {}
        return {
            "cmd_pos_hz": round(st["cycles"] / dt, 1),
            "tactile_hand_hz": round(st["touch_ok"] / 5 / dt, 1),
            "touch_lost": st["touch_lost"],
        }

    def _main(self) -> None:  # noqa: C901
        import numpy as np

        hand = None
        try:
            from glove.l6_calib import L6Calibration

            cpath = l6_calib_path(self.side)
            if not os.path.isfile(cpath):
                raise RuntimeError(
                    f"no glove calibration for the {self.side} hand: run teleop/hand_panel.py "
                    "or glove/l6_calibrate.py"
                )
            cal = L6Calibration.load(cpath)

            g = self.probes.get(self.glove_key)
            if g is None or not hasattr(g, "frame"):
                raise RuntimeError(f"{self.side} glove not available")

            d = self.devices.get(f"{self.side}_hand")
            if d is None or not d.enabled:
                raise RuntimeError(f"{self.side} hand not selected")
            self._released = [d] if d.suspend("hand following") else []
            time.sleep(1.2)

            from robot.l6_can import CMD_POSITION, L6Can

            hand = L6Can(d.can, hand_id=L6_HAND_ID[self.side])
            if self.fast_io or self.rec is not None:
                hand.on_frame = self._on_frame
            if self.live:
                hand.write_speed([self.speed] * 6)
                hand.write_torque([self.torque] * 6)
                time.sleep(0.2)
                # Open fully and wait until the hand is open, so following starts from a
                # known pose instead of jumping from a half-closed one.
                with self.lock:
                    self.state, self.msg = "opening", "opening to 255"
                t0 = time.monotonic()
                while time.monotonic() - t0 < 4.0 and self._run:
                    hand.write_position([255] * 6)
                    time.sleep(0.15)
                    pos = hand.snapshot().get("position") or []
                    if pos and min(pos) >= 240:
                        break
                pos = hand.snapshot().get("position") or []
                if pos and min(pos) < 200:
                    with self.lock:
                        self.msg = f"hand did not open fully (min {min(pos)}); continuing"

            with self.lock:
                self.state = "running"
                self.msg = "engage the clutch to follow" if self.live else "dry run (nothing sent)"
            if self.fast_io:
                threading.Thread(target=self._io_loop, args=(hand,), daemon=True).start()

            cur = np.full(6, 255.0)
            period, n_sent, n_loop_i = 1.0 / self.rate, 0, 0
            while self._run:
                lt = time.monotonic()
                with self.lock:
                    sm, ms = self.smooth, float(self.max_step)
                eng = True
                if self.clutch is not None and self.live:
                    eng = self.clutch.get_key_value_by_name(f"{self.side}_grip") > 0.9
                with self.lock:
                    self.engaged = eng

                f = g.frame()
                ja = f.shape.joint_angles if f is not None else None
                if ja is not None:
                    tgt = np.array(cal.to_l6(ja), dtype=float)
                    prev = cur.copy()
                    cur = cur + sm * (tgt - cur)  # EMA
                    cur = prev + np.clip(cur - prev, -ms, ms)  # rate limit
                    out = [int(round(x)) for x in np.clip(cur, 0, 255)]
                    if not eng:  # hold the current shape
                        cur = prev
                        out = [int(round(x)) for x in np.clip(prev, 0, 255)]
                    sent = False
                    if self.fast_io:
                        self._pending = (out, tgt, eng)  # sent and recorded by the I/O thread
                    elif self.live and eng:
                        try:
                            hand.write_position(out)
                            n_sent += 1
                            sent = True
                        except Exception as exc:  # noqa: BLE001
                            with self.lock:
                                self.msg = f"send failed: {type(exc).__name__}"
                    if self.rec is not None and not self.fast_io:
                        self.rec.put(
                            f"{self.side}_hand_cmd",
                            time.monotonic_ns(),
                            target=np.asarray(tgt, dtype=np.float32),
                            cmd=np.asarray(out, dtype=np.uint8),
                            engaged=bool(eng),
                            sent=sent,
                        )
                    # L6Can does not poll; request the position every 5th frame.
                    n_loop_i += 1
                    if not self.fast_io and n_loop_i % 5 == 0:
                        try:
                            hand.request(CMD_POSITION)
                        except Exception:  # noqa: BLE001
                            pass
                    snap = hand.snapshot()
                    age = hand.age()
                    with self.lock:
                        self.tel = {
                            "engaged": eng,
                            "target": [int(v) for v in tgt],
                            "age": (None if age is None else round(age, 2)),
                            "smoothed": out,
                            # Raw glove readings and calibrated ranges, to tell a glove
                            # problem from a calibration problem.
                            "raw_deg": [round(float(np.degrees(x)), 1) for x in cal.raw6(ja)],
                            "span_deg": [
                                [round(float(np.degrees(a)), 1), round(float(np.degrees(b)), 1)]
                                for a, b in cal.spans
                            ],
                            "thumb_deg": [
                                round(float(v), 1)
                                for v in np.degrees(np.asarray(ja, float).reshape(5, 5)[0])
                            ],
                            "actual": [int(v) for v in (snap.get("position") or [])],
                            "sent": n_sent + self._n_sent,
                            "io": self._io_rates(),
                            # Tactile arrays (5 x 12 x 6), copied every 5th frame.
                            "touch": (
                                snap.get("touch")
                                if n_loop_i % 5 == 0
                                else (self.tel or {}).get("touch")
                            ),
                            "touch_n": self.io_stat.get("touch_ok", 0),
                        }
                slack = period - (time.monotonic() - lt)
                if slack > 0:
                    time.sleep(slack)
        except Exception as exc:  # noqa: BLE001
            with self.lock:
                self.state, self.msg = "failed", f"{type(exc).__name__}: {exc}"[:150]
        finally:
            self._run = False
            if hand is not None:
                try:
                    if self.live:
                        hand.write_position([255] * 6)  # open on exit
                        time.sleep(0.3)
                    hand.close()
                except Exception:  # noqa: BLE001
                    pass
            for d in self._released:
                d.resume()
            self._released = []
            with self.lock:
                if self.state != "failed":
                    self.state, self.msg = "stopped", "stopped (hand open)"


# -- status strip -------------------------------------------------------------------------


class _SkippedGlove(Probe):
    """Placeholder for a glove when started with --no-glove."""

    def poll(self) -> Status:
        return Status(SKIP, "skipped", ["--no-glove"])


def compact(probes: dict, devices: dict, tracker_raw: dict, roles: dict | None = None) -> list:
    """One entry per status light: [{key, label, level, note, ...}]."""
    out = []
    for side in ("left", "right"):
        g = probes[f"glove_{side}"].status()
        out.append(
            {"key": f"glove_{side}", "label": f"{side} glove", "level": g.level, "note": g.headline}
        )
    role_of = {v: k for k, v in (roles or {}).items() if v}

    devs = tracker_raw.get("devices", [])
    bases = [d for d in devs if d.get("kind") == "base"]
    trks = [d for d in devs if d.get("kind") == "tracker"]
    if not devs:
        out.append({"key": "vive", "label": "tracker server", "level": BAD, "note": "no data"})
    else:
        for i, b in enumerate(bases):
            out.append(
                {
                    "key": f"base{i}",
                    "label": f"base {chr(65+i)}",
                    "level": OK if b.get("valid") else WARN,
                    "note": (b.get("serial") or "")[-8:],
                }
            )
        if not trks:
            out.append({"key": "trk", "label": "tracker", "level": BAD, "note": "none"})
        for t in trks:
            r = role_of.get(t.get("serial") or "")
            out.append(
                {
                    "key": "trk_" + (t.get("serial") or "?"),
                    "label": "tracker" + (f" {r}" if r else ""),
                    "level": OK if t.get("valid") else BAD,
                    "note": (t.get("serial") or "")[-8:],
                }
            )

    for key, label, _kind, _side, _dc in DEVICES:
        st = devices[key].status()
        # enabled/can come from the same status as the level, so they never disagree.
        out.append(
            {
                "key": key,
                "label": label,
                "level": st.level,
                "note": ("not selected" if st.level == SKIP else st.headline),
                "can": st.extra.get("can", devices[key].can),
                "enabled": st.extra.get("enabled", devices[key].enabled),
                "suspended": st.extra.get("suspended", ""),
                "toggleable": True,
            }
        )
    return out


def read_tracker_raw(url: str) -> dict:
    """Raw reply of the tracker server, or {}."""
    import urllib.request

    try:
        with urllib.request.urlopen(url, timeout=0.6) as r:
            return json.loads(r.read().decode())
    except Exception:  # noqa: BLE001
        return {}


# -- tracker server -----------------------------------------------------------------------

VIVE_PY = os.environ.get("VIVE_PYTHON", sys.executable)
VIVE_SERVER = os.path.join(_ROOT, "vive", "tracker_server.py")
VIVE_LOG = os.path.join(tempfile.gettempdir(), "tracker_server.log")


def _url_alive(url: str, timeout: float = 1.0) -> bool:
    import urllib.request

    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


def ensure_vive_service(tracker_url: str):
    """Start vive/tracker_server.py unless it already runs; returns the process or None.

    It runs as a subprocess because OpenVR may live in another Python environment
    (VIVE_PYTHON). Only a server started here is stopped on exit.
    """
    u = urllib.parse.urlparse(tracker_url)
    if u.hostname not in ("127.0.0.1", "localhost"):
        return None
    if _url_alive(tracker_url):
        print(f"  tracker server already running ({u.hostname}:{u.port})")
        return None
    log = open(VIVE_LOG, "w", encoding="utf-8")
    proc = subprocess.Popen(
        [VIVE_PY, VIVE_SERVER, "--port", str(u.port or 8765), "--no-browser"],
        cwd=os.path.dirname(VIVE_SERVER),
        stdout=log,
        stderr=subprocess.STDOUT,
        env=dict(os.environ, PYTHONNOUSERSITE="1"),
        start_new_session=True,  # Ctrl-C in the terminal does not reach it directly
        preexec_fn=_die_with_parent,
    )
    t0 = time.time()
    while time.time() - t0 < 15:
        if proc.poll() is not None:
            log.close()
            tail = open(VIVE_LOG, encoding="utf-8", errors="replace").read().strip()
            print(
                "  tracker server failed to start (trackers will show as disconnected):\n    "
                + "\n    ".join(tail.splitlines()[-4:])
            )
            return None
        if _url_alive(tracker_url):
            print(f"  tracker server started (pid {proc.pid}, log {VIVE_LOG})")
            return proc
        time.sleep(0.5)
    print(f"  tracker server not ready after 15 s, still waiting (log {VIVE_LOG})")
    return proc


def stop_vive_service(proc) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


# -- CAN scan -----------------------------------------------------------------------------


def run_can_scan() -> dict:
    """Run can_scan.py --json in a subprocess (a failed A7 handshake leaks sockets)."""
    try:
        p = subprocess.run(
            [PY, os.path.join(_HERE, "can_scan.py"), "--json"],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=_ROOT,
            env=dict(os.environ, PYTHONNOUSERSITE="1"),
        )
        return json.loads(p.stdout.strip().splitlines()[-1])
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"[:200]}


def apply_can_scan(devices: dict, r: dict, pinned=()) -> str:
    """Set each probe's CAN interface from a scan; the user's selection is not changed.

    Devices given on the command line (`pinned`) keep their interface. A device that was not
    found is moved to an unused bus, so two probes never share one.
    """
    if "error" in r:
        return f"CAN scan failed, keeping the default interfaces: {r['error']}"
    assign = r.get("assign", {})
    for k, can in assign.items():
        if k in devices and k not in pinned:
            devices[k].configure(can=can)
    used = {d.can for d in devices.values()}
    free = [b for b in r.get("buses", {}) if b not in used]
    for k, d in devices.items():
        if k in assign or k in pinned:
            continue
        clash = any(o.can == d.can for kk, o in devices.items() if kk != k)
        if clash and free:
            d.configure(can=free.pop(0))
    found = "  ".join(
        f"{k.replace('_', ' ')} -> {c}" for k, c in sorted(assign.items(), key=lambda x: x[1])
    )
    miss = [k.replace("_", " ") for k in devices if k not in assign]
    out = f"found {found or '(none)'}"
    if miss:
        out += f"; not found: {', '.join(miss)}"
    if r.get("conflicts"):
        out += "; " + "; ".join(r["conflicts"])
    return out


# -- helpers for the UI -------------------------------------------------------------------


def arm_busy(ctx: dict) -> list:
    """Controllers currently driving an arm (names under which the arm probes are suspended)."""
    names = {
        d.suspended
        for d in ctx["devices"].values()
        if d.kind == "arm" and d.suspended and d.suspended != ARM_OP
    }
    return sorted(names)


def stop_controls(ctx: dict, wait_s: float = 4.0) -> None:
    """Stop teleoperation and wait until the arm buses are released (used before relaxing)."""
    ctx["final"].stop()
    t0 = time.monotonic()
    while time.monotonic() - t0 < wait_s and arm_busy(ctx):
        time.sleep(0.1)


def record_meta(ctx: dict) -> dict:
    """Session information written to meta.json, including copies of the calibration files."""
    from dataclasses import asdict

    from final_teleop import final_knobs
    from teleop_loop import EXTRINSICS_PATH

    devs = ctx["devices"]
    fin = ctx["final"]
    extra = {}
    try:
        paths = {
            "extrinsics": EXTRINSICS_PATH,
            "l6_calib_right": l6_calib_path("right"),
            "l6_calib_left": l6_calib_path("left"),
        }
        for k, pth in paths.items():
            if os.path.isfile(pth):
                with open(pth, encoding="utf-8") as f:
                    extra[k] = json.load(f)
    except Exception as exc:  # noqa: BLE001
        extra["calib_error"] = f"{type(exc).__name__}: {exc}"
    return {
        "teleop_version": VERSION,
        "can": {k: d.can for k, d in devs.items()},
        "enabled": {k: d.enabled for k, d in devs.items()},
        "tracker_roles": dict(fin.roles),
        "guard_threshold_mps": fin.threshold,
        "final_knobs": asdict(final_knobs()),
        "conventions": {
            "arm": "q, dq, tau, q_cmd in motor (SDK) order, rad; sdk_ts = SDK time.time()",
            "hand": "L6 positions 0-255, 255 = extended, 0 = fully flexed; tactile data "
            "(12, 6) uint8, finger 0-4 = thumb, index, middle, ring, little",
            "glove": "joint_angles (5, 5) flattened; skeleton (21, 3) flattened, m, wrist "
            "frame; quaternions (x, y, z, w)",
        },
        "calibration_snapshots": extra,
    }


# -- start-up and shutdown ----------------------------------------------------------------


def make_parser() -> argparse.ArgumentParser:
    """Command-line options of the backend."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--devices",
        default="all",
        help="devices selected at start: all, or a list " "such as right_arm,right_hand",
    )
    ap.add_argument("--tracker-url", default="http://127.0.0.1:8765/api/poses")
    ap.add_argument("--tracker-serial", default=None, help="tracker shown in the status probe")
    ap.add_argument("--no-glove", action="store_true", help="do not connect the gloves")
    ap.add_argument("--no-vive", action="store_true", help="do not start the tracker server")
    ap.add_argument("--no-can-scan", action="store_true", help="use the default CAN interfaces")
    for k, label, _kind, _side, dc in DEVICES:
        # None = not given; a given interface is not overridden by the scan.
        ap.add_argument(
            f"--{k.replace('_','-')}-can",
            default=None,
            help=f"CAN interface of the {label} (default: found by scanning, else {dc})",
        )
    return ap


def build_ctx(args) -> dict:
    """Start the probes, clutch, recorder and teleoperation objects."""
    pinned = {k for k, *_ in DEVICES if getattr(args, f"{k}_can") is not None}
    for k, *_rest, dc in DEVICES:
        if getattr(args, f"{k}_can") is None:
            setattr(args, f"{k}_can", dc)

    sel = (
        [d[0] for d in DEVICES]
        if args.devices.strip() == "all"
        else [s.strip() for s in args.devices.split(",") if s.strip()]
    )

    devices = {
        k: DeviceProbe(k, label, kind, side, getattr(args, f"{k}_can"), k in sel)
        for k, label, kind, side, _dc in DEVICES
    }
    if not args.no_can_scan:  # before the probes start, while the buses are free
        print("  scanning the CAN buses ...", flush=True)
        print("  " + apply_can_scan(devices, run_can_scan(), pinned), flush=True)
    vive_proc = None if args.no_vive else ensure_vive_service(args.tracker_url)

    probes = {
        "tracker": TrackerProbe(args.tracker_url, args.tracker_serial),
        "can": CanProbe(devices),
        # Each glove SDK runs in its own child process (see glove_worker.py).
        "glove_left": (GloveProxy("left") if not args.no_glove else _SkippedGlove()),
        "glove_right": (GloveProxy("right") if not args.no_glove else _SkippedGlove()),
        **devices,
    }
    for p in probes.values():
        p.start()

    clutch = Clutch(side="both")
    recorder = Recorder()
    for _k in ("glove_left", "glove_right"):
        if isinstance(probes.get(_k), GloveProxy):
            probes[_k].rec = recorder
    clutch.on_event = lambda m, e, lk: recorder.event("clutch", msg=m, engaged=e, locked=lk)
    runner = JobRunner()
    ctx: dict = {}
    ctx.update(
        {
            "probes": probes,
            "devices": devices,
            "clutch": clutch,
            "runner": runner,
            "armops": ArmOps(devices, runner),
            "final": FinalTeleop(
                devices,
                probes,
                clutch,
                args.tracker_url,
                make_follow=lambda s: L6Follow(devices, probes, clutch=clutch, side=s),
                recorder=recorder,
            ),
            "rec": recorder,
            "arm_targets": lambda: ctx["armops"].target_names(),
            "arm_busy": lambda: arm_busy(ctx),
            "tracker_raw": lambda: read_tracker_raw(args.tracker_url),
            "pinned_can": pinned,
            "_sel": sel,
            "_vive_proc": vive_proc,
        }
    )
    ctx["armops"].sides_fn = lambda: list(ctx["final"].sides)
    ctx["armops"].busy_fn = lambda: arm_busy(ctx)
    return ctx


def shutdown_ctx(ctx: dict) -> None:
    """Stop in order: controllers first, then the probes."""
    ctx["final"].stop()  # arms pinned, hands opened
    ctx["rec"].stop()  # finish the video and merge the chunks
    ctx["armops"].stop_all()
    for p in ctx["probes"].values():
        p.stop()
    stop_vive_service(ctx["_vive_proc"])
    time.sleep(0.6)
