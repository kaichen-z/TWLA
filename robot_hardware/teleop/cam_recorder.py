#!/usr/bin/env python3
"""Record the RealSense colour stream (RGB 1280x720, 30 fps) to MP4. Started by `Recorder`.

The camera is opened only while recording, so other programs (e.g. the hand-eye
calibration camera service) can use it otherwise. Stops on SIGTERM.

Files in --out:
    rgb.mp4          video (H.264 through ffmpeg)
    frames_*.npz     per frame: t_ns (time.monotonic_ns), hw_ms (camera time), frame_no
    ready.json       written after the first frame ({"ok": true}) or on failure ({"ok": false})
    status.json      frame count, frame rate, dropped frames (updated every second)
    preview.jpg      320x180 preview, replaced every 0.2 s
    cam.log          progress log
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time

import numpy as np

PREVIEW_W, PREVIEW_H = 320, 180
PREVIEW_PERIOD_S = 0.2
OPEN_RETRY = 3


def main() -> int:  # noqa: C901
    ap = argparse.ArgumentParser(description="RealSense colour recorder")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--w", type=int, default=1280)
    ap.add_argument("--h", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--no-preview", action="store_true")
    a = ap.parse_args()

    stop = {"v": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__("v", True))
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__("v", True))
    try:  # exit with the parent process (PR_SET_PDEATHSIG)
        import ctypes

        ctypes.CDLL("libc.so.6").prctl(1, signal.SIGTERM)
    except Exception:  # noqa: BLE001
        pass

    logf = open(os.path.join(a.out, "cam.log"), "a", encoding="utf-8", buffering=1)

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, file=sys.stderr, flush=True)
        logf.write(line + "\n")

    def fail(msg: str) -> int:
        log("error: " + msg)
        _write_json(os.path.join(a.out, "ready.json"), {"ok": False, "err": msg})
        return 2

    log(f"start out={a.out} {a.w}x{a.h}@{a.fps}")
    import cv2
    import pyrealsense2 as rs

    prof = pipe = None
    for attempt in range(1, OPEN_RETRY + 1):
        devs = [
            (d.get_info(rs.camera_info.name), d.get_info(rs.camera_info.serial_number))
            for d in rs.context().devices
        ]
        log(f"attempt {attempt}/{OPEN_RETRY}: devices {devs or '(none)'}")
        if not devs:
            if attempt == OPEN_RETRY:
                return fail("no RealSense found (USB unplugged or still enumerating)")
            time.sleep(1.5)
            continue
        pipe, cfg = rs.pipeline(), rs.config()
        cfg.enable_stream(rs.stream.color, a.w, a.h, rs.format.bgr8, a.fps)
        try:
            log("opening the colour stream")
            prof = pipe.start(cfg)
            log("colour stream open")
            break
        except RuntimeError as exc:
            msg = str(exc)
            log(f"cannot open: {msg[:120]}")
            busy = "busy" in msg.lower() or "resource" in msg.lower()
            if attempt == OPEN_RETRY:
                return fail(
                    "camera in use by another program (e.g. drivers/tools/cam_server.py)"
                    if busy
                    else f"cannot open the colour stream: {msg[:120]}"
                )
            time.sleep(1.5)

    dev = prof.get_device()
    info = {
        "serial": dev.get_info(rs.camera_info.serial_number),
        "name": dev.get_info(rs.camera_info.name),
        "firmware": dev.get_info(rs.camera_info.firmware_version),
    }
    log(f"device {info['name']} sn={info['serial']} fw={info['firmware']}")

    mp4 = os.path.join(a.out, "rgb.mp4")
    ff = subprocess.Popen(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{a.w}x{a.h}",
            "-r",
            str(a.fps),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            mp4,
        ],
        stdin=subprocess.PIPE,
    )
    log(f"ffmpeg started: {mp4}")

    buf = {"t_ns": [], "hw_ms": [], "frame_no": []}
    n = chunk = 0
    t_flush = t_start = time.monotonic()
    t_prev = 0.0
    first_no = last_no = None
    no_frame_since = time.monotonic()

    def flush() -> None:
        nonlocal chunk
        if buf["t_ns"]:
            chunk += 1
            tmp = os.path.join(a.out, f"frames_{chunk:06d}.part.npz")
            np.savez(
                tmp,
                t_ns=np.asarray(buf["t_ns"], dtype=np.int64),
                hw_ms=np.asarray(buf["hw_ms"], dtype=np.float64),
                frame_no=np.asarray(buf["frame_no"], dtype=np.int64),
            )
            os.replace(tmp, os.path.join(a.out, f"frames_{chunk:06d}.npz"))
            for v in buf.values():
                v.clear()
        dropped = (last_no - first_no + 1 - n) if first_no is not None else 0
        _write_json(
            os.path.join(a.out, "status.json"),
            {
                "frames": n,
                "fps": round(n / max(time.monotonic() - t_start, 1e-6), 2),
                "dropped": int(dropped),
                **info,
            },
        )

    try:
        while not stop["v"]:
            try:
                fs = pipe.wait_for_frames(timeout_ms=1000)
            except RuntimeError:
                if time.monotonic() - no_frame_since > 5:
                    log("no frame for 5 s (camera disconnected or USB power?)")
                    no_frame_since = time.monotonic()
                continue
            t_ns = time.monotonic_ns()
            c = fs.get_color_frame()
            if not c:
                continue
            no_frame_since = time.monotonic()
            img = np.asanyarray(c.get_data())
            if n == 0:
                log("first frame received, writing MP4")
                _write_json(os.path.join(a.out, "ready.json"), {"ok": True, "t_ns": t_ns, **info})
            fno = c.get_frame_number()
            if first_no is None:
                first_no = fno
            last_no = fno
            ff.stdin.write(img.tobytes())
            buf["t_ns"].append(t_ns)
            buf["hw_ms"].append(c.get_timestamp())
            buf["frame_no"].append(fno)
            n += 1
            now = time.monotonic()
            if not a.no_preview and now - t_prev >= PREVIEW_PERIOD_S:
                t_prev = now
                small = cv2.resize(img, (PREVIEW_W, PREVIEW_H), interpolation=cv2.INTER_AREA)
                ok, enc = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                if ok:  # atomic replace: readers never see a partial file
                    tmp = os.path.join(a.out, "preview.jpg.part")
                    with open(tmp, "wb") as f:
                        f.write(enc.tobytes())
                    os.replace(tmp, os.path.join(a.out, "preview.jpg"))
            if now - t_flush >= 1.0:
                flush()
                t_flush = now
    finally:
        try:
            pipe.stop()
        except Exception:  # noqa: BLE001
            pass
        flush()
        try:
            ff.stdin.close()
            ff.wait(timeout=30)
        except Exception:  # noqa: BLE001
            ff.kill()
        log(f"done: {n} frames in {mp4}")
        logf.close()
    return 0


def _write_json(path: str, obj: dict) -> None:
    tmp = path + ".part"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, path)


if __name__ == "__main__":
    raise SystemExit(main())
