#!/usr/bin/env python3
"""Jog one arm from the 3D viewer and record hand-eye calibration samples.

Every motion is previewed first: sliders and commands only change the preview, and the real
arm moves when `go` is issued. The tool has no home command; it always starts from the
current pose of the arm.

There is no collision checking. Cables, the camera bracket, the table and the calibration
board are not in the model, so watch the robot while it moves.

In the viewer the light arm is the current pose (read from the encoders) and the green arm
is the preview.

Usage:
    python tools/arm_jog.py                 # connect to the arm
    python tools/arm_jog.py --sim-only      # no hardware
    python tools/arm_jog.py --side left     # jog the left arm (default: right)
    python tools/arm_jog.py --max-step 15   # limit the motion of one `go`, in degrees
    python tools/arm_jog.py --continue      # append samples to the latest calib/handeye_*

Control panel: http://127.0.0.1:8840/ with the 3D view, joint sliders, board orientation,
enable / go / undo / sync and the sample button. The terminal accepts the commands below.

Recording a sample (button "record sample" or command `sample`) needs `tools/cam_server.py`
running. It reads the joint angles, has the camera service save the current frame and
detect the board in it, and appends one line to `calib/handeye_<timestamp>/samples.jsonl`
(schema `handeye_sample/v2`); the image is saved next to it. Samples with fewer than six
corners are kept but marked `usable=false`. Solve with `handeye_solve.py` and then
`handeye_joint_solve.py`.

The arm can also be positioned by hand: while it is disabled the motors are unpowered, and
the panel follows the encoders.

Commands (joint numbers are motor numbers, as in arm.py):
    j4 +5         move J4 of the preview by +5 deg
    a4 80         set J4 of the preview to 80 deg
    go            execute the preview
    undo          discard the preview
    sync          re-read the arm
    sample        record a sample
    drop          delete the last sample
    cand          list the candidate poses in calib/handeye_candidates_<side>.json (optional)
    cand 7        load candidate 7 into the preview
    cand next     load the next candidate that is not marked done
    ok            mark the loaded candidate as done
    board 90      board orientation about the tool axis in the viewer, in degrees
    cam | cam off show / hide the calibrated camera in the viewer
    enable | disable | info | q
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time

import numpy as np
import pinocchio as pin

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from robot.conventions import JOINT_NAMES as MOTOR_JOINT_NAMES  # noqa: E402
from robot.conventions import LIMITS_LEFT, LIMITS_RIGHT  # noqa: E402
from sim_viewer import CALIB_Q_RIGHT_DEG, MOUNT_FACE_IN_J7, URDF, Viewer  # noqa: E402

# The arm being jogged; set from --side at start-up.
SIDE = "right"
LIMITS = LIMITS_RIGHT
FIRST_ID = {"right": 51, "left": 61}  # CAN id of the first motor of each arm
SIDE_LABEL = {"right": "right arm", "left": "left arm"}
MAX_STEP_DEG = None  # largest joint motion of one `go`; None means unlimited

# Orientation of the board about the tool axis in the viewer, in degrees (display only).
DEFAULT_BOARD_YAW_DEG = 270.0
DEFAULT_BOARD_YAW_BY_SIDE = {"right": 270.0, "left": 90.0}

# Reduced speed (rad/s, rad/s^2) for moving with the board mounted.
VELOCITY, ACCELERATION = 0.15, 1.5

# Calibration board plate: size in metres, and its centre in the mounting face frame
# (-z points away from the arm).
BOARD_SIZE = (0.216, 0.160, 0.005)
BOARD_CENTER_IN_MOUNT = np.array([-0.077, 0.0, -0.0025])


def rot_z(a):
    """Rotation matrix about z by angle a (rad)."""
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


class Jog(Viewer):
    """Viewer with a preview arm and the calibration board at the end of the arm."""

    SHOW_HANDS = False  # the board is mounted in place of the hand

    def __init__(self, board_yaw_deg=DEFAULT_BOARD_YAW_DEG, open_browser=True):
        super().__init__(open_browser=open_browser)
        self.q_prev = self.q.copy()  # preview configuration
        for tag, col, op in (("real", 0xEEEEEE, 0.95), ("prev", 0x22DD44, 0.45)):
            self.viz.viewer[f"board/{tag}"].set_object(
                self._g.Box(list(BOARD_SIZE)), self._g.MeshLambertMaterial(color=col, opacity=op)
            )
        self.set_board_yaw(board_yaw_deg)

    def set_board_yaw(self, deg):
        """Set the board orientation about the tool axis, in degrees."""
        self.board_yaw = float(deg)

    def _board_world(self, q):
        """4x4 pose of the board centre in the base frame at configuration q."""
        data = self.model.createData()
        pin.forwardKinematics(self.model, data, q)
        T7 = data.oMi[self.jid[SIDE][-1]]
        face = T7 * pin.SE3(rot_z(np.deg2rad(self.board_yaw)), MOUNT_FACE_IN_J7)
        return (face * pin.SE3(np.eye(3), BOARD_CENTER_IN_MOUNT)).homogeneous

    def show(self, ok=True):
        """Redraw the real arm, the preview arm (red if not ok) and both boards."""
        self.display()
        self.draw_arm(
            "preview", SIDE, self.q_prev, color=0x22DD44 if ok else 0xDD2222, opacity=0.45
        )
        self.viz.viewer["board/real"].set_transform(self._board_world(self.q))
        self.viz.viewer["board/prev"].set_transform(self._board_world(self.q_prev))

    def arm_deg(self, q):
        """Joint angles of the jogged arm in degrees."""
        return np.rad2deg(q[self.qidx[SIDE]])


def find_channel(side=None):
    """Find the CAN interface of an arm by probing its first motor on every interface.

    Interface numbers follow USB enumeration and change with the wiring. Returns the
    interface name, or None if the arm does not answer.
    """
    aid = FIRST_ID[side or SIDE]
    import can

    for ch in sorted(c for c in os.listdir("/sys/class/net") if c.startswith("can")):
        try:
            bus = can.interface.Bus(channel=ch, interface="socketcan")
        except Exception:
            continue
        try:
            bus.send(
                can.Message(arbitration_id=aid, data=[0x06], is_extended_id=False), timeout=0.2
            )
            t0 = time.time()
            while time.time() - t0 < 0.3:
                m = bus.recv(timeout=0.1)
                if m is not None and m.arbitration_id == aid and getattr(m, "is_rx", True):
                    return ch
        except Exception:
            pass
        finally:
            bus.shutdown()
    return None


PANEL_PORT = 8840
CAM_URL = "http://127.0.0.1:8850"  # tools/cam_server.py
SAMPLES_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "calib")
PANEL_VERSION = "1.0"
# Serialises all viewer and arm access; the meshcat socket is not thread-safe.
LOCK = threading.RLock()


def latest_handeye_result(path=None):
    """Load a calibration result. Returns (path, dict), or (None, None) if there is none.

    Without a path, the most recently modified joint_result.json is used; failing that, the
    result.json of the newest handeye_* directory.
    """
    import glob

    if path is None:
        joint = glob.glob(os.path.join(SAMPLES_ROOT, "handeye_*", "joint_result.json"))
        single = sorted(glob.glob(os.path.join(SAMPLES_ROOT, "handeye_*", "result.json")))
        if joint:
            path = max(joint, key=os.path.getmtime)
        elif single:
            path = single[-1]
        else:
            return None, None
    with open(path, encoding="utf-8") as f:
        return path, json.load(f)


def show_handeye(J: Jog, path=None):
    """Draw the calibrated camera in the viewer and return a one-line summary."""
    p, r = latest_handeye_result(path)
    if r is None:
        return "no calibration result yet; run tools/handeye_solve.py"
    with LOCK:
        J.show_camera(r["T_base_cam"], r.get("intrinsics"))
    T = np.array(r["T_base_cam"])
    pos = T[:3, 3]
    z = T[:3, :3] @ [0, 0, 1]
    if "rms_px" in r:  # result of handeye_joint_solve.py
        tail = f"joint solve ({'+'.join(r['sides'])}) RMS {r['rms_px']:.1f} px"
    else:
        tail = (
            f"{r['n_samples']} samples, median residual "
            f"{r['residual_trans_mm']['median']:.0f} mm / "
            f"{r['residual_rot_deg']['median']:.1f} deg"
        )
    return (
        f"camera drawn in the viewer: {os.path.relpath(p, SAMPLES_ROOT)}  "
        f"@({pos[0]:+.3f},{pos[1]:+.3f},{pos[2]:+.3f}) m "
        f"pitch {np.rad2deg(np.arcsin(-z[2])):.0f} deg  {tail}"
    )


class Controller:
    """Preview, motion and sampling logic shared by the terminal and the web panel."""

    def __init__(self, J: Jog, arm, channel, max_step_deg=MAX_STEP_DEG):
        self.J, self.arm, self.channel = J, arm, channel
        self.max_step = None if max_step_deg is None else float(max_step_deg)
        self.idx = J.qidx[SIDE]
        self.enabled = False
        self.moving = False
        self.log = []
        self.read_real()
        self.J.q_prev = self.J.q.copy()
        self.J.show()
        # While the arm is disabled it can be moved by hand; a background thread keeps the
        # viewer in sync with the encoders.
        self.relax_follow = True
        self._follow_run = True
        self._preview_dirty = False  # True once the operator has edited the preview
        threading.Thread(target=self._follow_loop, daemon=True).start()

    def _follow_loop(self):
        """Follow the encoders while the arm is disabled and not moving.

        The preview follows as well unless the operator has edited it. A redraw is skipped
        when the arm has moved less than 0.05 deg, because redrawing holds the shared lock.
        """
        last = None
        while self._follow_run:
            time.sleep(0.12)
            if self.arm is None or self.enabled or self.moving or not self.relax_follow:
                continue
            try:
                q = np.asarray(self.arm.get_angles(), dtype=float)  # cached by the SDK
                if last is not None and np.max(np.abs(q - last)) < np.deg2rad(0.05):
                    continue
                last = q
                with LOCK:
                    self.J.q[self.idx] = q
                    if not self._preview_dirty:
                        self.J.q_prev = self.J.q.copy()
                    self.J.show()
            except Exception:  # noqa: BLE001
                pass

    def say(self, msg):
        """Print a message and append it to the panel log."""
        self.log = (self.log + [msg])[-12:]
        print("  " + msg)

    def read_real(self):
        """Copy the measured joint angles into the viewer state."""
        if self.arm is not None:
            self.J.q[self.idx] = np.asarray(self.arm.get_angles(), dtype=float)

    def motor_deg(self, q_vec):
        """Joint angles of the jogged arm in degrees, from a full configuration vector."""
        return np.rad2deg(np.asarray(q_vec[self.idx], dtype=float))

    def preview(self, q_deg_motor, fast=False):
        """Set the preview to the given joint angles (degrees). Returns (ok, message).

        Rejects targets outside the joint limits or further from the arm than the step limit.
        """
        with LOCK:
            for j, v in enumerate(q_deg_motor):
                lo, hi = LIMITS[j]
                if not (lo <= np.deg2rad(v) <= hi):
                    return (
                        False,
                        f"J{j+1} {MOTOR_JOINT_NAMES[j]} outside the limits "
                        f"[{np.rad2deg(lo):+.0f}, {np.rad2deg(hi):+.0f}] deg",
                    )
            qn = self.J.q_prev.copy()
            qn[self.idx] = np.deg2rad(np.asarray(q_deg_motor, dtype=float))
            step = np.abs(self.motor_deg(qn) - self.motor_deg(self.J.q))
            if self.max_step is not None and step.max() > self.max_step + 1e-6:
                return (
                    False,
                    f"J{int(step.argmax())+1} is {step.max():.1f} deg from the arm, above the "
                    f"step limit of {self.max_step:g} deg; move in smaller steps",
                )
            self.J.q_prev = qn
            self._preview_dirty = not np.allclose(qn, self.J.q)
            self.J.show()
            return (
                True,
                "preview "
                + " ".join(f"{v:+.1f}" for v in self.motor_deg(qn))
                + f"  (largest change J{int(step.argmax())+1} {step.max():.1f} deg)",
            )

    def set_board(self, yaw):
        """Set the board orientation in the viewer."""
        with LOCK:
            self.J.set_board_yaw(yaw)
            self.J.show()
            return f"board orientation {self.J.board_yaw:+.0f} deg (display only)"

    def undo(self):
        """Reset the preview to the current pose of the arm."""
        with LOCK:
            self.J.q_prev = self.J.q.copy()
            self._preview_dirty = False
            self.J.show()
            return "preview discarded"

    def sync(self):
        """Re-read the arm and reset the preview."""
        with LOCK:
            self.read_real()
            return self.undo().replace("preview discarded", "synced with the arm")

    def enable(self):
        """Enable the arm after pinning the motor targets to the current pose."""
        if self.arm is None:
            return "sim-only mode"
        with LOCK:
            from arm import pin_target_to_current

            pin_target_to_current(self.arm)
            time.sleep(0.2)
            self.arm.enable()
            time.sleep(0.4)
            self.enabled = True
            self.sync()
            return "enabled"

    def disable(self):
        """Disable the arm; it can then be moved by hand."""
        if self.arm is None:
            return "sim-only mode"
        with LOCK:
            self.arm.disable()
            self.enabled = False
            return "disabled: the arm is no longer holding its position"

    def go(self):
        """Move the arm to the preview and wait until it stops. Returns a message."""
        with LOCK:
            if self.moving:
                return "moving"
            if np.allclose(self.J.q_prev, self.J.q):
                return "no preview"
            if self.arm is None:
                self.J.q = self.J.q_prev.copy()
                self.J.show()
                return "[sim-only] arrived"
            if not self.enabled:
                return "enable first"
            self.moving = True
            target = [float(v) for v in self.J.q_prev[self.idx]]
            self.arm.move_j(target, blocking=False)
        try:
            t0 = time.time()
            while time.time() - t0 < 30:
                with LOCK:
                    self.read_real()
                    self.J.show()
                    if not self.arm.is_moving():
                        break
                time.sleep(0.1)
            time.sleep(0.3)
            with LOCK:
                self.read_real()
                err = float(
                    np.max(np.abs(self.motor_deg(self.J.q) - self.motor_deg(self.J.q_prev)))
                )
                self.J.q_prev = self.J.q.copy()
                self._preview_dirty = False
                self.J.show()
            return f"arrived, largest joint error {err:.2f} deg"
        finally:
            self.moving = False

    def _cam(self, path, body=None, timeout=3.0):
        """Call the camera service; GET without a body, POST with one."""
        import urllib.request

        req = urllib.request.Request(
            CAM_URL + path,
            data=None if body is None else json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="GET" if body is None else "POST",
        )
        return json.loads(urllib.request.urlopen(req, timeout=timeout).read())

    def cam_status(self):
        """Camera service status, cached for 0.7 s."""
        now = time.time()
        if now - getattr(self, "_cam_t", 0) > 0.7:
            try:
                self._cam_cache = self._cam("/api/status", timeout=1.0)
                self._cam_cache["online"] = True
            except Exception as e:
                self._cam_cache = {"online": False, "error": type(e).__name__}
            self._cam_t = now
        return self._cam_cache

    def sample(self):
        """Record one hand-eye sample at the current pose and return a message."""
        from datetime import datetime

        if self.moving:
            return "moving; record after the arm has stopped"
        if self.arm is None:
            return "sim-only mode: nothing recorded"
        with LOCK:
            self.read_real()
            if not hasattr(self, "sample_dir"):
                self.sample_dir = os.path.join(
                    SAMPLES_ROOT, "handeye_" + datetime.now().strftime("%Y%m%d_%H%M%S")
                )
                os.makedirs(self.sample_dir, exist_ok=True)
                self.samples = []
            elif not hasattr(self, "samples"):  # --continue: append to the existing samples
                f = os.path.join(self.sample_dir, "samples.jsonl")
                self.samples = (
                    [json.loads(l) for l in open(f, encoding="utf-8")] if os.path.isfile(f) else []
                )
            n = len(self.samples)
            pin.forwardKinematics(self.J.model, self.J.data, self.J.q)
            pin.updateFramePlacements(self.J.model, self.J.data)
            T7 = self.J.data.oMi[self.J.jid[SIDE][-1]]
            _, face = self.J._tool_frames(SIDE)
            q_motor = self.J.q[self.idx].copy()
        try:
            cam = self._cam(
                "/api/capture", {"dir": self.sample_dir, "name": f"img_{n:03d}"}, timeout=8.0
            )
        except Exception as e:
            return f"camera service not available ({type(e).__name__}): start tools/cam_server.py"
        # The solvers use q_motor_rad and the camera fields; the poses are stored for reference.
        rec = {
            "schema": "handeye_sample/v2",
            "idx": n,
            "t": time.time(),
            "arm": SIDE,
            "fk_model": os.path.relpath(URDF, SAMPLES_ROOT),
            "q_motor_deg": [round(float(v), 4) for v in np.rad2deg(q_motor)],
            "q_motor_rad": [float(v) for v in q_motor],
            "T_base_j7": T7.homogeneous.tolist(),
            "T_base_mountface": face.homogeneous.tolist(),
            "board_yaw_sim_deg": self.J.board_yaw,
            "image": cam.get("image"),
            "n_corners": cam["n_corners"],
            "n_markers": cam["n_markers"],
            "ids": cam.get("ids"),
            "corners_px": cam.get("corners_px"),
            "rvec_cam_board": cam.get("rvec_cam_board"),
            "tvec_cam_board": cam.get("tvec_cam_board"),
            "reproj_err_px": cam.get("reproj_err_px"),
            "intrinsics": cam["intrinsics"],
            "board": cam["board"],
            "usable": cam["n_corners"] >= 6,
        }
        with LOCK:
            self.samples.append(rec)
            with open(os.path.join(self.sample_dir, "samples.jsonl"), "a") as f:
                f.write(json.dumps(rec) + "\n")
        ok = rec["usable"]
        return (
            f"sample #{n}: {rec['n_corners']}/16 corners"
            + (
                f", reprojection {rec['reproj_err_px']:.2f} px"
                if ok
                else ", fewer than 6 corners, not usable"
            )
            + f"  -> usable {sum(r['usable'] for r in self.samples)}/{len(self.samples)}"
        )

    def drop_last_sample(self):
        """Delete the last sample and its image."""
        with LOCK:
            if not getattr(self, "samples", None):
                return "no samples"
            r = self.samples.pop()
            with open(os.path.join(self.sample_dir, "samples.jsonl"), "w") as f:
                for x in self.samples:
                    f.write(json.dumps(x) + "\n")
            try:
                os.remove(r["image"])
            except Exception:
                pass
            return f"deleted #{r['idx']}, {len(self.samples)} left"

    def show_cam(self, show: bool):
        """Show or hide the calibrated camera in the viewer."""
        if show:
            msg = show_handeye(self.J)
            self.cam_shown = not msg.startswith("no ")
            self.cam_info = msg.split(": ", 1)[-1] if self.cam_shown else msg
            return msg
        with LOCK:
            self.J.hide_camera()
        self.cam_shown = False
        return "camera marker hidden"

    def state(self):
        """Full panel state as a json-serialisable dict."""
        cam = self.cam_status()
        with LOCK:
            return {
                "version": PANEL_VERSION,
                "channel": self.channel,
                "sim_only": self.arm is None,
                "enabled": self.enabled,
                "moving": self.moving,
                "board_yaw": self.J.board_yaw,
                "cam_shown": getattr(self, "cam_shown", False),
                "cam_info": getattr(self, "cam_info", ""),
                "relax": bool(self.arm is not None and not self.enabled and not self.moving),
                "preview_dirty": bool(getattr(self, "_preview_dirty", False)),
                "order": "motor",
                "names": list(MOTOR_JOINT_NAMES),
                "limits_deg": [
                    [round(np.rad2deg(a), 1), round(np.rad2deg(b), 1)] for a, b in LIMITS
                ],
                "side": SIDE,
                "side_label": SIDE_LABEL[SIDE],
                "real_deg": [round(float(v), 2) for v in self.motor_deg(self.J.q)],
                "prev_deg": [round(float(v), 2) for v in self.motor_deg(self.J.q_prev)],
                "max_step_deg": self.max_step,
                "model": os.path.basename(URDF),
                "log": self.log,
                "meshcat": self.J.viz.viewer.url(),
                "cam": cam,
                "cam_url": CAM_URL,
                "samples": [
                    {
                        "idx": r["idx"],
                        "n": r["n_corners"],
                        "err": r["reproj_err_px"],
                        "usable": r["usable"],
                    }
                    for r in getattr(self, "samples", [])
                ],
                "sample_dir": getattr(self, "sample_dir", None),
            }


PANEL_HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>arm_jog</title>
<style>
 body{margin:0;font:14px system-ui,sans-serif;background:#f4f5f7;color:#1c1e21}
 .wrap{display:grid;grid-template-columns:1fr 440px;height:100vh}
 .left{display:grid;grid-template-rows:1fr 1fr;min-height:0}
 iframe{width:100%;height:100%;border:0;background:#fff}
 .cam{background:#111;display:flex;align-items:center;justify-content:center;position:relative;min-height:0}
 .cam img{max-width:100%;max-height:100%;object-fit:contain}
 .cam .off{color:#bbb;font-size:13px;text-align:center;padding:20px}
 .samp{margin:8px 0;padding:8px 10px;border:1px solid #cfe3d3;background:#f2faf4;border-radius:8px}
 .samp .big{font-size:15px;font-weight:700}
 button.cap{background:#1d4ed8;color:#fff;border-color:#1d4ed8;font-weight:700;padding:10px 16px}
 .slist{font:12px ui-monospace,monospace;max-height:70px;overflow:auto;color:#334}
 .panel{padding:14px 16px;overflow:auto;border-left:1px solid #ddd;background:#fff}
 h1{font-size:16px;margin:0 0 6px} .muted{color:#667;font-size:12px}
 .j{display:grid;grid-template-columns:128px 1fr 56px 56px;align-items:center;gap:8px;margin:7px 0;border-radius:6px;padding:2px 4px}
 .j.sel{background:#2a3a55;outline:1px solid #6fa0ff}
 .j input[type=range]{width:100%} .n{font-variant-numeric:tabular-nums;text-align:right}
 .real{color:#667} .prev{color:#0a7a2a;font-weight:600}
 .relax{padding:8px 10px;border-radius:6px;margin-bottom:8px;font-weight:700}
.clr{margin:10px 0;padding:10px;border-radius:8px;font-weight:600;white-space:pre-wrap}
 .ok{background:#e6f6ea;color:#0a7a2a} .warn{background:#fff4d6;color:#8a5a00} .bad{background:#fde7e7;color:#a11}
 button{padding:8px 12px;margin:3px 4px 3px 0;border:1px solid #bbb;border-radius:6px;background:#fafafa;cursor:pointer}
 button.go{background:#0a7a2a;color:#fff;border-color:#0a7a2a;font-weight:700;padding:10px 18px}
 button:disabled{opacity:.45;cursor:not-allowed}
 .log{font:12px ui-monospace,monospace;background:#f7f7f9;border:1px solid #e3e3e8;border-radius:6px;padding:8px;height:150px;overflow:auto;white-space:pre-wrap}
 .row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
 @media (max-width:900px){.wrap{grid-template-columns:1fr}iframe{height:45vh}}
</style></head><body><div class="wrap">
<div class="left"><iframe id="mc" src="about:blank"></iframe>
 <div class="cam"><img id="camimg" alt="" hidden><div class="off" id="camoff">camera service not connected<br>python3 tools/cam_server.py</div></div></div>
<div class="panel">
 <h1><span id="sidelabel">right arm</span> jog <span class="muted" id="hdr"></span></h1>
 <div class="muted">Sliders only change the <span class="prev">green preview</span>; the arm moves on <b>go</b>. Grey values are the current joint angles. Joint numbers are motor numbers, as in arm.py.</div>
 <div id="joints"></div>
 <div class="muted" style="margin:4px 0 8px">Keyboard: digits 1-7 select a joint · ← → or ↑ ↓ change by 1° · Shift 5° · Ctrl 0.2° · Esc cancels</div>
 <div class="j"><b>board yaw</b><input id="yaw" type="range" min="0" max="359" step="1"><span class="n" id="yawv"></span><span class="muted">°</span></div>
 <div style="margin:6px 4px"><label><input id="camchk" type="checkbox"> show the calibrated camera</label> <span class="muted" id="caminfo"></span></div>
 <div id="clr" class="clr">—</div>
 <div class="row">
  <button id="go" class="go">go ▶</button><button id="undo">undo</button><button id="sync">sync</button>
  <button id="en">enable</button><button id="dis">disable</button>
 </div>
 <div class="samp">
  <div id="relaxbar" class="relax">—</div>
  <div><span class="big">Camera: </span><span id="camst">-</span></div>
  <div class="row" style="margin-top:6px"><button id="cap" class="cap">record sample</button><button id="drop">delete last</button><span id="sampn" class="muted"></span></div>
  <div class="slist" id="slist"></div>
 </div>
 <div class="muted" id="foot"></div>
 <div class="log" id="log"></div>
</div></div>
<script>
const $=s=>document.querySelector(s); let S=null, dragging=false, t=null;
async function api(p,b){const r=await fetch(p,{method:b?'POST':'GET',headers:{'Content-Type':'application/json'},body:b?JSON.stringify(b):undefined});return r.json();}
let lastSent=0, pending=null;
function sendFast(){const q=[...document.querySelectorAll('#joints input')].map(e=>+e.value);
 const now=Date.now(); if(now-lastSent<50){clearTimeout(pending);pending=setTimeout(sendFast,50-(now-lastSent));return;}
 lastSent=now; api('/api/preview',{q_deg:q,fast:true}).then(render);}
function build(s){
 const J=$('#joints'); J.innerHTML='';
 s.names.forEach((n,i)=>{const [lo,hi]=s.limits_deg[i];
  J.insertAdjacentHTML('beforeend',`<div class="j"><span>J${i+1} ${n}</span><input type="range" data-i="${i}" min="${lo}" max="${hi}" step="0.5" value="${s.prev_deg[i]}"><span class="n real" id="r${i}"></span><span class="n prev" id="p${i}"></span></div>`);});
 J.querySelectorAll('input').forEach(el=>{el.oninput=()=>{dragging=true;$('#p'+el.dataset.i).textContent=(+el.value).toFixed(1)+'°';sendFast();};
  el.onchange=()=>{clearTimeout(pending);send();};});
 $('#mc').src=s.meshcat; $('#yaw').value=Math.round(s.board_yaw);
}
async function send(){const q=[...document.querySelectorAll('#joints input')].map(e=>+e.value);dragging=false;render(await api('/api/preview',{q_deg:q}));}
function render(s){S=s; if(!document.querySelector('#joints input')) build(s);
 s.names.forEach((n,i)=>{$('#r'+i).textContent=s.real_deg[i].toFixed(1)+'°';if(!dragging){$('#p'+i).textContent=s.prev_deg[i].toFixed(1)+'°';document.querySelector(`#joints input[data-i="${i}"]`).value=s.prev_deg[i];}});
 $('#sidelabel').textContent=s.side_label||''; document.title='arm_jog · '+(s.side_label||'');
 $('#camchk').checked=!!s.cam_shown; $('#caminfo').textContent=s.cam_info||'';
 $('#yawv').textContent=Math.round(s.board_yaw); if(document.activeElement!==$('#yaw'))$('#yaw').value=Math.round(s.board_yaw);
 const c=$('#clr'); c.className='clr'; c.textContent=`model ${s.model||''} · no collision checking: watch the robot`;
 $('#hdr').textContent=(s.sim_only?'[sim-only]':'@ '+s.channel)+(s.enabled?' · enabled':' · disabled')+(s.moving?' · moving…':'');
 $('#go').disabled=s.moving||(!s.sim_only&&!s.enabled); $('#en').disabled=s.enabled||s.sim_only; $('#dis').disabled=!s.enabled;
 const rb=$('#relaxbar');
 if(s.sim_only){rb.textContent='sim-only (no arm connected)';rb.style.background='#333';rb.style.color='#ccc';}
 else if(s.relax){rb.textContent='Relaxed sampling: the motors are not powered. Move the arm by hand to the desired pose; when the board is clear in the camera image, click "record sample".'+(s.preview_dirty?' (You edited the preview, so the 3D view has stopped following; press undo to resume.)':'');
   rb.style.background='#1f4a32';rb.style.color='#dfffe9';}
 else if(s.enabled){rb.textContent='Enabled: the motors are powered and the arm cannot be moved by hand. Click disable to move it by hand.';
   rb.style.background='#5a3b10';rb.style.color='#ffe9c2';}
 else {rb.textContent='moving …';rb.style.background='#3a2f1a';rb.style.color='#ffe9c2';}
 const cm=s.cam||{}; const ci=$('#camimg');
 if(cm.online){ if(ci.hidden){ci.src=s.cam_url+'/stream.mjpg';ci.hidden=false;$('#camoff').hidden=true;}
   $('#camst').textContent=`corners ${cm.n_corners}/16 · markers ${cm.n_markers}/12 · ${cm.fps} fps`+(cm.n_corners>=6?'  OK to record':'  (fewer than 6, not usable)');
   $('#camst').style.color=cm.n_corners>=6?'#0a7a2a':'#a11';}
 else{ if(!ci.hidden){ci.hidden=true;ci.src='';$('#camoff').hidden=false;} $('#camst').textContent='not connected';$('#camst').style.color='#a11';}
 $('#cap').disabled=s.moving||s.sim_only||!cm.online; $('#drop').disabled=!(s.samples&&s.samples.length);
 $('#sampn').textContent=s.samples?`usable ${s.samples.filter(x=>x.usable).length} / total ${s.samples.length}`+(s.sample_dir?'  → '+s.sample_dir.split('/').slice(-1)[0]:''):'';
 $('#slist').innerHTML=(s.samples||[]).slice(-6).map(x=>`#${x.idx} corners ${x.n}${x.err!=null?' reprojection '+x.err.toFixed(2)+'px':''}${x.usable?'':' ✗'}`).join('<br>');
 $('#foot').textContent=`${s.max_step_deg==null?'step unlimited':'per joint per go ≤ '+s.max_step_deg+'°'} · v${s.version}`;
 $('#log').textContent=s.log.join('\n'); $('#log').scrollTop=1e9;}
$('#yaw').oninput=async e=>{$('#yawv').textContent=e.target.value;clearTimeout(t);t=setTimeout(async()=>render(await api('/api/board',{yaw:+e.target.value})),120);};
$('#go').onclick=async()=>render(await api('/api/go',{}));
$('#camchk').onchange=async e=>render(await api('/api/cam',{show:e.target.checked}));
$('#undo').onclick=async()=>render(await api('/api/undo',{}));
$('#sync').onclick=async()=>render(await api('/api/sync',{}));
$('#en').onclick=async()=>render(await api('/api/enable',{}));
$('#cap').onclick=async()=>{$('#cap').disabled=true;render(await api('/api/sample',{}));};
$('#drop').onclick=async()=>{if(confirm('Delete the last sample?'))render(await api('/api/sample_drop',{}));};
$('#dis').onclick=async()=>{if(confirm('The arm will stop holding its position once disabled. Continue?'))render(await api('/api/disable',{}));};
let selJ=-1;
function selectJoint(i){selJ=i;document.querySelectorAll('#joints .j').forEach((d,k)=>d.classList.toggle('sel',k===i));}
document.addEventListener('keydown',e=>{
 if(e.target.tagName==='INPUT'&&e.target.type!=='range')return;
 if(e.key>='1'&&e.key<='7'){selectJoint(+e.key-1);e.preventDefault();return;}
 if(e.key==='Escape'){selectJoint(-1);return;}
 if(selJ<0)return;
 const dir={ArrowUp:1,ArrowRight:1,'+':1,'=':1,ArrowDown:-1,ArrowLeft:-1,'-':-1,'_':-1}[e.key]; if(!dir)return;
 e.preventDefault();
 const step=e.shiftKey?5:(e.ctrlKey?0.2:1);
 const el=document.querySelector(`#joints input[data-i="${selJ}"]`);
 const v=Math.min(+el.max,Math.max(+el.min,+el.value+dir*step)); el.value=v;
 dragging=true;$('#p'+selJ).textContent=v.toFixed(1)+'°';sendFast();
 clearTimeout(t);t=setTimeout(()=>{clearTimeout(pending);send();},250);
});
document.addEventListener('click',e=>{const d=e.target.closest('#joints .j');if(d)selectJoint([...d.parentNode.children].indexOf(d));});
setInterval(async()=>{if(!dragging)render(await api('/api/state'));},400);
api('/api/state').then(render);
</script></body></html>"""


def _cand_path(side: str) -> str:
    """Path of the optional candidate pose file of one arm."""
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "calib",
        f"handeye_candidates_{side}.json",
    )


def _load_cands(side: str):
    """Candidate poses, or [] if the file does not exist.

    File format: {"poses": [{"idx", "q_motor_deg", "dist_m", "tilt_deg"}, ...]}.
    """
    p = _cand_path(side)
    if not os.path.isfile(p):
        return []
    return json.load(open(p, encoding="utf-8"))["poses"]


def serve_panel(ctl: Controller, port=PANEL_PORT):
    """Serve the control panel and its json api on localhost in a background thread."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json"):
            b = body.encode() if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", ctype + "; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            if self.path == "/":
                return self._send(200, PANEL_HTML, "text/html")
            if self.path == "/api/state":
                return self._send(200, json.dumps(ctl.state()))
            self._send(404, "{}")

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            try:
                if self.path == "/api/preview":
                    ok, msg = ctl.preview(body["q_deg"], fast=bool(body.get("fast")))
                    if not body.get("fast") or not ok:
                        ctl.say(msg)
                elif self.path == "/api/board":
                    ctl.say(ctl.set_board(float(body["yaw"])))
                elif self.path == "/api/cam":
                    ctl.say(ctl.show_cam(bool(body.get("show", True))))
                elif self.path == "/api/undo":
                    ctl.say(ctl.undo())
                elif self.path == "/api/sync":
                    ctl.say(ctl.sync())
                elif self.path == "/api/enable":
                    ctl.say(ctl.enable())
                elif self.path == "/api/disable":
                    ctl.say(ctl.disable())
                elif self.path == "/api/sample":
                    ctl.say(ctl.sample())
                elif self.path == "/api/sample_drop":
                    ctl.say(ctl.drop_last_sample())
                elif self.path == "/api/go":
                    if ctl.moving:
                        ctl.say("moving")
                    else:
                        threading.Thread(target=lambda: ctl.say(ctl.go()), daemon=True).start()
                        time.sleep(0.15)
                else:
                    return self._send(404, "{}")
            except Exception as e:
                ctl.say(f"error: {type(e).__name__}: {e}")
            self._send(200, json.dumps(ctl.state()))

    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


HELP = """  j4 +5 | a4 80 | go | undo | sync | board <deg> | enable | disable | info | q
  sample | drop | cand [N | next] | ok | cam [off]
  joint numbers are motor numbers, as in arm.py: """ + "  ".join(
    f"J{i+1}={n}" for i, n in enumerate(MOTOR_JOINT_NAMES)
)


def main() -> int:
    """Start the viewer and the panel, then run the terminal command loop."""
    global SIDE, LIMITS
    if "--side" in sys.argv:
        SIDE = sys.argv[sys.argv.index("--side") + 1]
        assert SIDE in ("left", "right"), "--side left|right"
    LIMITS = LIMITS_LEFT if SIDE == "left" else LIMITS_RIGHT
    sim_only = "--sim-only" in sys.argv
    open_browser = "--no-open" not in sys.argv
    J = Jog(board_yaw_deg=DEFAULT_BOARD_YAW_BY_SIDE[SIDE], open_browser=False)
    arm, ch = None, None
    if sim_only:
        q0 = list(CALIB_Q_RIGHT_DEG)
        if SIDE == "left":  # mirror image: the roll and yaw joints change sign
            q0 = [q0[0], -q0[1], -q0[2], q0[3], -q0[4], q0[5], -q0[6]]
        J.q[J.qidx[SIDE]] = np.deg2rad(q0)
        print("[sim-only] no hardware; starting from the default calibration pose")
    else:
        ch = find_channel()
        if ch is None:
            print(
                f"No CAN interface answered for the {SIDE_LABEL[SIDE]} (ID {FIRST_ID[SIDE]}): "
                "check power, USB and the emergency stop, or use --sim-only."
            )
            return 1
        from arm import connect

        print(f"{SIDE_LABEL[SIDE]} on {ch}, connecting without enabling ...")
        arm = connect(SIDE, ch, velocity=VELOCITY, acceleration=ACCELERATION, enable=False)
    max_step = MAX_STEP_DEG
    if "--max-step" in sys.argv:
        max_step = float(sys.argv[sys.argv.index("--max-step") + 1])
    ctl = Controller(J, arm, ch, max_step_deg=max_step)
    if "--continue" in sys.argv:  # append to the latest sample directory
        import glob

        ds = sorted(
            d
            for d in glob.glob(os.path.join(SAMPLES_ROOT, "handeye_*"))
            if os.path.isdir(d) and os.path.isfile(os.path.join(d, "samples.jsonl"))
        )
        if ds:
            ctl.sample_dir = ds[-1]
            f = os.path.join(ds[-1], "samples.jsonl")
            ctl.samples = (
                [json.loads(l) for l in open(f, encoding="utf-8")] if os.path.isfile(f) else []
            )
            print(
                f"   --continue: appending to {os.path.relpath(ds[-1], SAMPLES_ROOT)} "
                f"({len(ctl.samples)} samples so far)"
            )
    port = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else PANEL_PORT
    serve_panel(ctl, port)
    url = f"http://127.0.0.1:{port}/"
    if open_browser:
        import webbrowser

        webbrowser.open(url)
    print(f"\ncontrol panel: {url}   (3D view alone: {J.viz.viewer.url()})")
    print(f"   model {os.path.basename(URDF)} (no collision checking: watch the robot)")
    print("   current pose (deg) " + " ".join(f"{v:+7.1f}" for v in ctl.motor_deg(J.q)))
    print("   " + ctl.show_cam(True) + "\n")
    print(HELP)
    _done, _cur = set(), {"i": None}  # candidates marked done; candidate in the preview
    while True:
        try:
            t = input("\n> ").strip().split()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not t:
            continue
        c0 = t[0].lower()
        try:
            if c0 in ("q", "quit"):
                break
            elif c0 == "help":
                print(HELP)
            elif c0 == "board" and len(t) == 2:
                ctl.say(ctl.set_board(float(t[1])))
            elif c0 == "sync":
                ctl.say(ctl.sync())
            elif c0 == "undo":
                ctl.say(ctl.undo())
            elif c0 == "info":
                J.info()
            elif c0 == "cam":
                print("  " + ctl.show_cam(not (len(t) == 2 and t[1] == "off")))
            elif c0 == "enable":
                ctl.say(ctl.enable())
            elif c0 == "disable":
                prompt = "  The arm will stop holding its position. Type yes to confirm: "
                if input(prompt).strip() == "yes":
                    ctl.say(ctl.disable())
            elif c0 == "go":
                ctl.say(ctl.go())
            elif c0 in ("sample", "cap"):
                ctl.say(ctl.sample())
            elif c0 == "cand":
                # A candidate is only loaded into the preview; it still needs `go`.
                cands = _load_cands(SIDE)
                if not cands:
                    print("  no candidate file " + _cand_path(SIDE))
                    continue
                if len(t) == 1:
                    print(f"  {len(cands)} candidates in {_cand_path(SIDE)} (* = done)")
                    for c in cands:
                        mark = "*" if c["idx"] in _done else " "
                        print(
                            f"   {mark}{c['idx']:3d}  "
                            + " ".join(f"{v:6.1f}" for v in c["q_motor_deg"])
                            + f"   board distance {c['dist_m']:.2f} m, tilt {c['tilt_deg']:.0f} deg"
                        )
                    continue
                if t[1] == "next":
                    left = [c for c in cands if c["idx"] not in _done]
                    if not left:
                        print("  all candidates visited")
                        continue
                    c = left[0]
                else:
                    c = next((x for x in cands if x["idx"] == int(t[1])), None)
                    if c is None:
                        print(f"  no candidate {t[1]}")
                        continue
                ok, msg = ctl.preview(list(c["q_motor_deg"]))
                ctl.say(msg)
                print(
                    f"  candidate #{c['idx']} loaded into the preview (board distance "
                    f"{c['dist_m']:.2f} m, tilt {c['tilt_deg']:.0f} deg); check it, then `go` "
                    "and `sample`"
                )
                _cur["i"] = c["idx"]
            elif c0 == "ok":
                if _cur["i"] is not None:
                    _done.add(_cur["i"])
                    print(
                        f"  candidate #{_cur['i']} marked done, "
                        f"{len([c for c in _load_cands(SIDE) if c['idx'] not in _done])} left"
                    )
                else:
                    print("  no candidate loaded yet")
            elif c0 == "drop":
                ctl.say(ctl.drop_last_sample())
            elif c0[0] in "ja" and c0[1:].isdigit() and len(t) == 2:
                j = int(c0[1:])
                if not 1 <= j <= 7:
                    print("  joint number 1-7")
                    continue
                q = ctl.motor_deg(J.q_prev).tolist()
                q[j - 1] = (q[j - 1] + float(t[1])) if c0[0] == "j" else float(t[1])
                ok, msg = ctl.preview(q)
                ctl.say(msg)
                if ok is not None:
                    print(
                        "  preview (deg) " + " ".join(f"{v:+7.1f}" for v in ctl.motor_deg(J.q_prev))
                    )
            else:
                print(HELP)
        except Exception as e:
            print(f"  error: {type(e).__name__}: {e}")
    if arm is not None:
        arm.close()
        print("disconnected; the enable state of the arm is unchanged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
