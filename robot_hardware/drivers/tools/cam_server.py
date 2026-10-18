#!/usr/bin/env python3
"""Camera service for hand-eye calibration (Intel RealSense).

Serves the live colour image with the ChArUco detection overlaid, and captures full
resolution frames together with the detected board pose. `arm_jog.py` calls it to record
calibration samples.

    python3 tools/cam_server.py        # http://127.0.0.1:8850/

Endpoints:
    GET  /stream.mjpg    live image (960 px wide) with the detection overlay
    GET  /api/status     corner and marker counts, frame rate, intrinsics, board parameters
    POST /api/capture    body {"dir": ..., "name": ...}: saves the current 1920x1080 frame as
                         dir/name.png and returns the corners, marker ids, board pose in the
                         camera frame (rvec, tvec), reprojection error and intrinsics

The board is defined in `calib_target/charuco_5x5_28mm/` (parameters in the json there; set
`square_length_m` to the measured value after printing). The intrinsics are the factory
calibration of the colour stream. Only one process can open the camera at a time.
"""

import json
import os
import sys
import threading
import time

import cv2
import numpy as np
import pyrealsense2 as rs
from flask import Flask, Response, jsonify, request

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
TARGET = os.path.join(ROOT, "calib_target", "charuco_5x5_28mm")
sys.path.insert(0, TARGET)
import make_charuco  # noqa: E402

PORT, W, H, FPS = 8850, 1920, 1080, 30
spec = json.load(open(os.path.join(TARGET, "charuco_5x5_28mm.json")))
make_charuco.SQUARE_MM = spec["square_length_m"] * 1000
make_charuco.MARKER_MM = spec["marker_length_m"] * 1000
BOARD = make_charuco.make_board()
DET = cv2.aruco.CharucoDetector(BOARD)
app = Flask(__name__)


class Cam:
    """RealSense colour stream with a background grab thread and a detection thread."""

    def __init__(self):
        self.pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
        prof = self.pipe.start(cfg)
        i = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.K = np.array([[i.fx, 0, i.ppx], [0, i.fy, i.ppy], [0, 0, 1.0]])
        self.dist = np.array(i.coeffs, dtype=float)
        self.intr = {
            "width": i.width,
            "height": i.height,
            "fx": i.fx,
            "fy": i.fy,
            "ppx": i.ppx,
            "ppy": i.ppy,
            "model": str(i.model),
            "coeffs": list(i.coeffs),
            "source": "realsense factory",
        }
        self.lock = threading.Lock()
        self.frame = None
        self.seq = 0
        self.t_last = 0
        self.fps = 0.0
        self.det = {"n_corners": 0, "n_markers": 0, "corners": None, "ids": None}
        threading.Thread(target=self._loop, daemon=True).start()
        threading.Thread(target=self._detect_loop, daemon=True).start()

    def _loop(self):
        """Grab frames and keep the latest one."""
        n, t0 = 0, time.time()
        while True:
            fr = self.pipe.wait_for_frames().get_color_frame()
            if not fr:
                continue
            img = np.asanyarray(fr.get_data()).copy()
            with self.lock:
                self.frame = img
                self.seq += 1
                self.t_last = time.time()
            n += 1
            if n % 30 == 0:
                self.fps = 30 / (time.time() - t0)
                t0 = time.time()

    def detect(self, img):
        """Detect the ChArUco board in one image."""
        cc, ci, mc, mi = DET.detectBoard(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        return {
            "n_corners": 0 if ci is None else len(ci),
            "n_markers": 0 if mi is None else len(mi),
            "corners": cc,
            "ids": ci,
            "mcorners": mc,
            "mids": mi,
        }

    def _detect_loop(self):
        """Run detection on new frames for the live overlay (about 10 Hz)."""
        last = -1
        while True:
            with self.lock:
                img, seq = self.frame, self.seq
            if img is None or seq == last:
                time.sleep(0.01)
                continue
            last = seq
            d = self.detect(img)
            self.det = d
            time.sleep(0.05)

    def snapshot(self):
        """Return (frame copy, sequence number, timestamp) of the latest frame, or None."""
        with self.lock:
            return None if self.frame is None else (self.frame.copy(), self.seq, self.t_last)

    def overlay_jpeg(self):
        """Latest frame with the detection drawn, downscaled and JPEG-encoded."""
        snap = self.snapshot()
        if snap is None:
            return None
        img, _, _ = snap
        d = self.det
        if d["n_markers"]:
            cv2.aruco.drawDetectedMarkers(img, d["mcorners"], d["mids"])
        if d["n_corners"]:
            cv2.aruco.drawDetectedCornersCharuco(img, d["corners"], d["ids"], (0, 255, 0))
        small = cv2.resize(img, (960, 540), interpolation=cv2.INTER_AREA)
        cv2.putText(
            small,
            f"corners {d['n_corners']}/16  markers {d['n_markers']}/12  {self.fps:.0f} fps",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0) if d["n_corners"] >= 6 else (0, 0, 255),
            2,
        )
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return buf.tobytes() if ok else None

    def pose(self, d):
        """Board pose in the camera frame from a detection.

        Returns (rvec, tvec, reprojection error in px), or None with fewer than 6 corners.
        """
        if d["n_corners"] < 6:
            return None
        obj, imgp = BOARD.matchImagePoints(d["corners"], d["ids"])
        ok, rvec, tvec = cv2.solvePnP(obj, imgp, self.K, self.dist, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return None
        proj, _ = cv2.projectPoints(obj, rvec, tvec, self.K, self.dist)
        err = float(
            np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - imgp.reshape(-1, 2)) ** 2, axis=1)))
        )
        return rvec.ravel().tolist(), tvec.ravel().tolist(), err


cam = Cam()


@app.route("/")
def index():
    return (
        '<img src="/stream.mjpg" style="max-width:100%">'
        f"<pre>{json.dumps(cam.intr, indent=1)}</pre>"
    )


@app.route("/stream.mjpg")
def stream():
    def gen():
        while True:
            b = cam.overlay_jpeg()
            if b:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + b + b"\r\n"
            time.sleep(1 / 15)

    return Response(
        gen(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store"},
    )


@app.route("/api/status")
def status():
    d = cam.det
    return jsonify(
        {
            "n_corners": d["n_corners"],
            "n_markers": d["n_markers"],
            "fps": round(cam.fps, 1),
            "age_s": round(time.time() - cam.t_last, 2),
            "intrinsics": cam.intr,
            "board": {
                k: spec[k]
                for k in (
                    "squares_x",
                    "squares_y",
                    "square_length_m",
                    "marker_length_m",
                    "dictionary",
                )
            },
        }
    )


@app.route("/api/capture", methods=["POST"])
def capture():
    b = request.get_json(force=True) or {}
    d_out, name = b.get("dir"), b.get("name", f"img_{int(time.time())}")
    snap = cam.snapshot()
    if snap is None:
        return jsonify({"ok": False, "error": "no frame"}), 503
    img, seq, t = snap
    d = cam.detect(img)  # detect on the captured frame, not the overlay result
    out = {
        "ok": True,
        "t_frame": t,
        "seq": seq,
        "n_corners": d["n_corners"],
        "n_markers": d["n_markers"],
        "ids": None if d["ids"] is None else d["ids"].ravel().tolist(),
        "corners_px": None if d["corners"] is None else d["corners"].reshape(-1, 2).tolist(),
        "intrinsics": cam.intr,
        "board": {
            k: spec[k]
            for k in ("squares_x", "squares_y", "square_length_m", "marker_length_m", "dictionary")
        },
    }
    p = cam.pose(d)
    if p:
        out.update({"rvec_cam_board": p[0], "tvec_cam_board": p[1], "reproj_err_px": p[2]})
    if d_out:
        os.makedirs(d_out, exist_ok=True)
        path = os.path.join(d_out, name + ".png")
        cv2.imwrite(path, img)
        out["image"] = path
    return jsonify(out)


if __name__ == "__main__":
    print(
        f"camera {W}x{H}@{FPS}, intrinsics fx={cam.intr['fx']:.1f} fy={cam.intr['fy']:.1f} "
        f"ppx={cam.intr['ppx']:.1f} ppy={cam.intr['ppy']:.1f} coeffs={cam.intr['coeffs']}"
    )
    print(
        f"board: {spec['squares_x']}x{spec['squares_y']} squares, "
        f"{spec['square_length_m']*1000:g} mm; serving http://127.0.0.1:{PORT}/"
    )
    app.run(host="127.0.0.1", port=PORT, threaded=True, debug=False, use_reloader=False)
