#!/usr/bin/env python3
"""Vive Tracker pose server: reads OpenVR and serves all device poses over HTTP.

    python vive/tracker_server.py                # http://127.0.0.1:8765/api/poses
    python vive/tracker_server.py --view         # also serve a 3D view at http://127.0.0.1:8765/

GET /api/poses returns {"devices": [...], "t": seconds, "error": null|str}; each device has
kind (tracker / base / hmd / controller), serial, battery, valid and, when valid, pos (m),
rot (3x3, row-major) and rpy (deg, display only). Poses are in the OpenVR frame:
right-handed, +Y up, metres; with --universe raw (default) the origin is defined by the
base stations, so no SteamVR room setup is needed.

Requires SteamVR running and the `openvr` Python package. While this process holds its
OpenVR connection, SteamVR does not go to standby and switch the trackers off.
In the 3D view: left drag rotates, wheel zooms, right drag pans, R resets, C clears trails.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

try:
    import openvr
except ImportError:
    sys.exit("the openvr package is not installed (pip install openvr)")

HTML_PATH = Path(__file__).with_name("tracker_view.html")

CLASS_NAMES = {
    openvr.TrackedDeviceClass_HMD: "hmd",
    openvr.TrackedDeviceClass_Controller: "controller",
    openvr.TrackedDeviceClass_GenericTracker: "tracker",
    openvr.TrackedDeviceClass_TrackingReference: "base",
}

_state = {"devices": [], "t": 0.0, "error": None}
_lock = threading.Lock()


def mat_to_pos_rot(m):
    """OpenVR HmdMatrix34_t -> (position [x, y, z], rotation as rows)."""
    pos = [m[0][3], m[1][3], m[2][3]]
    rot = [[m[r][c] for c in range(3)] for r in range(3)]
    return pos, rot


def rot_to_rpy_deg(rot):
    """Rotation matrix -> XYZ Euler angles in degrees, for display only."""
    sy = math.sqrt(rot[0][0] ** 2 + rot[1][0] ** 2)
    if sy > 1e-6:
        x = math.atan2(rot[2][1], rot[2][2])
        y = math.atan2(-rot[2][0], sy)
        z = math.atan2(rot[1][0], rot[0][0])
    else:
        x = math.atan2(-rot[1][2], rot[1][1])
        y = math.atan2(-rot[2][0], sy)
        z = 0.0
    return [math.degrees(v) for v in (x, y, z)]


def poll_loop(universe: int, hz: float) -> None:
    """Background thread: read all device poses from OpenVR into _state."""
    try:
        openvr.init(openvr.VRApplication_Other)
    except Exception as e:
        with _lock:
            _state["error"] = f"cannot connect to SteamVR: {e}"
        return
    vr = openvr.VRSystem()
    period = 1.0 / hz
    t0 = time.time()

    while True:
        loop_start = time.perf_counter()
        try:
            poses = vr.getDeviceToAbsoluteTrackingPose(
                universe, 0, openvr.k_unMaxTrackedDeviceCount
            )
            devices = []
            for i in range(openvr.k_unMaxTrackedDeviceCount):
                cls = vr.getTrackedDeviceClass(i)
                if cls == openvr.TrackedDeviceClass_Invalid:
                    continue
                kind = CLASS_NAMES.get(cls, "other")
                try:
                    serial = vr.getStringTrackedDeviceProperty(i, openvr.Prop_SerialNumber_String)
                except Exception:
                    serial = "?"
                try:
                    batt = vr.getFloatTrackedDeviceProperty(
                        i, openvr.Prop_DeviceBatteryPercentage_Float
                    )
                except Exception:
                    batt = None

                p = poses[i]
                entry = {
                    "index": i,
                    "kind": kind,
                    "serial": serial,
                    "battery": batt,
                    "valid": bool(p.bPoseIsValid),
                }
                if p.bPoseIsValid:
                    pos, rot = mat_to_pos_rot(p.mDeviceToAbsoluteTracking)
                    entry["pos"] = pos
                    entry["rot"] = rot
                    entry["rpy"] = rot_to_rpy_deg(rot)
                devices.append(entry)

            with _lock:
                _state["devices"] = devices
                _state["t"] = time.time() - t0
                _state["error"] = None
        except Exception as e:  # a failed read must not end the thread
            with _lock:
                _state["error"] = str(e)

        sleep = period - (time.perf_counter() - loop_start)
        if sleep > 0:
            time.sleep(sleep)


class Handler(BaseHTTPRequestHandler):
    """Serves /api/poses, and the 3D view at / when enabled."""

    view = False

    def log_message(self, *args):
        pass

    def _send(self, code, ctype, body: bytes):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()

    def do_GET(self):
        if self.path.startswith("/api/poses"):
            with _lock:
                body = json.dumps(_state).encode()
            self._send(200, "application/json", body)
        elif self.path in ("/", "/index.html"):
            if self.view and HTML_PATH.exists():
                self._send(200, "text/html; charset=utf-8", HTML_PATH.read_bytes())
            else:
                self._send(
                    200,
                    "text/plain; charset=utf-8",
                    b"GET /api/poses   6-DoF poses of all trackers and base stations (JSON)\n"
                    b"start with --view for a 3D view\n",
                )
        else:
            self._send(404, "text/plain", b"not found")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--hz", type=float, default=90.0, help="OpenVR polling rate")
    ap.add_argument("--universe", choices=["raw", "standing", "seated"], default="raw")
    ap.add_argument("--view", action="store_true", help="serve the 3D view at /")
    ap.add_argument("--no-browser", action="store_true", help="do not open the 3D view")
    args = ap.parse_args()
    Handler.view = args.view

    universe = {
        "raw": openvr.TrackingUniverseRawAndUncalibrated,
        "standing": openvr.TrackingUniverseStanding,
        "seated": openvr.TrackingUniverseSeated,
    }[args.universe]

    threading.Thread(target=poll_loop, args=(universe, args.hz), daemon=True).start()
    time.sleep(0.5)
    with _lock:
        if _state["error"]:
            sys.exit(_state["error"])

    url = f"http://127.0.0.1:{args.port}"
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"tracker server: {url}/api/poses   universe={args.universe}   {args.hz:g} Hz")
    if args.view:
        print(f"3D view: {url}/")
        if not args.no_browser:
            threading.Thread(target=lambda: webbrowser.open(url), daemon=True).start()
    print("Ctrl-C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        openvr.shutdown()


if __name__ == "__main__":
    main()
