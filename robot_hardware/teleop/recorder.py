"""Data recording: one directory per session, every stream time-stamped with the same clock.

Recorded: head camera (RGB 1280x720, 30 fps, MP4), arm joint states and commands, L6
positions, commands and tactile arrays, raw glove data, and clutch events. Data are stored
in their raw form; conversion to a training format is a separate step.

The recorder does not read hardware itself: only one program can talk to a device on a CAN
bus, so the control loops that own the devices `put` their readings here. The camera is
recorded by a child process (`cam_recorder.py`), which may run in a separate Python
environment (set CAMERA_PYTHON).

Clock: the timestamp of every sample is `t_ns = time.monotonic_ns()` (CLOCK_MONOTONIC is
shared by all processes). Device timestamps are stored as extra columns. meta.json stores a
(monotonic, wall clock) pair to convert to absolute time.

    recordings/<YYYYmmdd_HHMMSS>/
      meta.json                      start/stop, clock, tracker roles, calibration snapshots
      events.jsonl                   clutch and teleoperation events
      camera/rgb.mp4, frames_*.npz   camera (child process)
      <stream>/chunk_000001.npz      other streams, one chunk per second
      <stream>.npz                   all chunks merged and sorted (written on stop)
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
RECORD_ROOT = os.path.join(_ROOT, "recordings")
CAM_PY = os.environ.get("CAMERA_PYTHON", sys.executable)
FLUSH_S = 1.0


class _Stream:
    """One stream: `put(t_ns, fields)` with scalar or fixed-size array fields."""

    def __init__(self, root: str, name: str) -> None:
        self.dir = os.path.join(root, name)
        os.makedirs(self.dir, exist_ok=True)
        self.name = name
        self.lock = threading.Lock()
        self.buf: dict = {}
        self.n = 0
        self.chunk = 0
        self.t_first = None
        self.t_last = None

    def put(self, t_ns: int, fields: dict) -> None:
        with self.lock:
            if not self.buf:
                self.buf = {"t_ns": []}
                for k in fields:
                    self.buf[k] = []
            self.buf["t_ns"].append(int(t_ns))
            for k, v in fields.items():
                self.buf.setdefault(k, []).append(v)
            self.n += 1
            if self.t_first is None:
                self.t_first = t_ns
            self.t_last = t_ns

    def flush(self) -> None:
        """Write the buffered samples as the next chunk."""
        with self.lock:
            buf, self.buf = self.buf, {}
            if not buf or not buf["t_ns"]:
                return
            self.chunk += 1
            idx = self.chunk
        arrs = {}
        for k, v in buf.items():
            try:
                arrs[k] = np.asarray(v)
            except Exception:  # noqa: BLE001
                arrs[k] = np.asarray([json.dumps(x, default=str) for x in v])
        tmp = os.path.join(self.dir, f"chunk_{idx:06d}.part.npz")
        np.savez(tmp, **arrs)
        os.replace(tmp, os.path.join(self.dir, f"chunk_{idx:06d}.npz"))  # visible once complete

    def rate(self) -> float:
        with self.lock:
            if self.n < 2 or self.t_first is None or self.t_last == self.t_first:
                return 0.0
            return (self.n - 1) / ((self.t_last - self.t_first) / 1e9)


class Recorder:
    """Session recorder; `put(stream, t_ns, **fields)` is a no-op while not recording."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.active = False
        self.dir = ""
        self.streams: dict = {}
        self.t0_ns = 0
        self.cam: subprocess.Popen | None = None
        self.cam_err = ""
        self._events = None
        self._flusher: threading.Thread | None = None
        self.msg = ""
        self.meta_extra: dict = {}

    def put(self, stream: str, t_ns: int, **fields) -> None:
        """Add one sample to a stream."""
        if not self.active:
            return
        s = self.streams.get(stream)
        if s is None:
            with self.lock:
                if not self.active:
                    return
                s = self.streams.get(stream)
                if s is None:
                    s = self.streams[stream] = _Stream(self.dir, stream)
        s.put(t_ns, fields)

    def event(self, kind: str, **info) -> None:
        """Append one line to events.jsonl."""
        if not self.active or self._events is None:
            return
        line = json.dumps(
            {"t_ns": time.monotonic_ns(), "wall": time.time(), "kind": kind, **info},
            ensure_ascii=False,
            default=str,
        )
        with self.lock:
            if self._events is not None:
                self._events.write(line + "\n")
                self._events.flush()

    def start(self, meta: dict | None = None, camera: bool = True) -> str:
        """Start a session; returns "" or the reason it did not start."""
        with self.lock:
            if self.active:
                return "already recording"
            os.makedirs(RECORD_ROOT, exist_ok=True)
            free_gb = shutil.disk_usage(RECORD_ROOT).free / 1e9
            if free_gb < 5:
                return f"only {free_gb:.1f} GB free; not recording"
            name = time.strftime("%Y%m%d_%H%M%S")
            self.dir = os.path.join(RECORD_ROOT, name)
            os.makedirs(self.dir)
            self.streams = {}
            self.t0_ns = time.monotonic_ns()
            self.meta_extra = dict(meta or {})
            self._write_meta(started=True)
            self._events = open(os.path.join(self.dir, "events.jsonl"), "a", encoding="utf-8")
            self.cam, self.cam_err = None, ""
            self.active = True
        if camera:
            self._start_camera()
        self.event("record_start", dir=self.dir)
        self._flusher = threading.Thread(target=self._flush_loop, daemon=True)
        self._flusher.start()
        self.msg = f"recording to {self.dir}"
        return ""

    def stop(self) -> str:
        """Stop, flush, finish the video and merge the chunks."""
        with self.lock:
            if not self.active:
                return ""
        self.event("record_stop")
        with self.lock:
            self.active = False
        self._stop_camera()
        for s in list(self.streams.values()):
            s.flush()
        with self.lock:
            if self._events is not None:
                self._events.close()
                self._events = None
        self._write_meta(started=False)
        try:
            self.merge(self.dir)
            self.msg = f"stopped, merged: {self.dir}"
        except Exception as exc:  # noqa: BLE001
            self.msg = (
                f"stopped; merge failed (chunks are kept, rerun merge): {type(exc).__name__}: {exc}"
            )
        return ""

    def _flush_loop(self) -> None:
        while self.active:
            time.sleep(FLUSH_S)
            for s in list(self.streams.values()):
                try:
                    s.flush()
                except Exception as exc:  # noqa: BLE001
                    self.msg = f"writing {s.name} failed: {type(exc).__name__}: {exc}"

    def _write_meta(self, started: bool) -> None:
        path = os.path.join(self.dir, "meta.json")
        m = {}
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                m = json.load(f)
        now = {
            "monotonic_ns": time.monotonic_ns(),
            "wall": time.time(),
            "wall_str": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        if started:
            m.update(
                {
                    "start": now,
                    "clock": "t_ns = time.monotonic_ns() (CLOCK_MONOTONIC)",
                    "camera": {
                        "model": "RealSense D435i",
                        "stream": "RGB only",
                        "size": [1280, 720],
                        "fps": 30,
                        "codec": "H.264 (libx264)",
                    },
                    **self.meta_extra,
                }
            )
        else:
            m["stop"] = now
            m["streams"] = {
                n: {"samples": s.n, "rate_hz": round(s.rate(), 2)} for n, s in self.streams.items()
            }
            m["camera_error"] = self.cam_err
        tmp = path + ".part"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(m, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)

    def _start_camera(self) -> None:
        from glove_worker import _die_with_parent

        out = os.path.join(self.dir, "camera")
        os.makedirs(out, exist_ok=True)
        # The child writes its own camera/cam.log.
        self.cam = subprocess.Popen(
            [CAM_PY, os.path.join(_HERE, "cam_recorder.py"), "--out", out],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=_die_with_parent,
            env=dict(os.environ, PYTHONNOUSERSITE="1"),
        )

    def _stop_camera(self) -> None:
        p, self.cam = self.cam, None
        if p is None:
            return
        if p.poll() is None:
            p.terminate()  # cam_recorder finishes the MP4 on SIGTERM
            try:
                p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                p.kill()
                self.cam_err = "camera process killed after 15 s; the MP4 may be incomplete"
        elif p.returncode not in (0, None, -15):  # -15: our own terminate
            self.cam_err = f"camera process exited early (code {p.returncode}), see camera/cam.log"

    def camera_status(self) -> dict:
        """Camera state; it counts as ready only once it has produced a frame (ready.json)."""
        out = {
            "running": False,
            "frames": 0,
            "fps": 0.0,
            "err": self.cam_err,
            "ready": False,
            "preview": "",
        }
        cdir = os.path.join(self.dir, "camera") if self.dir else ""
        if cdir:
            try:
                with open(os.path.join(cdir, "ready.json"), encoding="utf-8") as f:
                    r = json.load(f)
                out["ready"] = bool(r.get("ok"))
                if not r.get("ok") and r.get("err"):
                    self.cam_err = r["err"]
                    out["err"] = self.cam_err
            except Exception:  # noqa: BLE001
                pass
            p = os.path.join(cdir, "preview.jpg")
            if os.path.isfile(p):
                out["preview"] = p
        if self.cam is not None and self.cam.poll() is not None and not self.cam_err:
            self.cam_err = f"camera process exited (code {self.cam.returncode}), see camera/cam.log"
            out["err"] = self.cam_err
        st = os.path.join(self.dir, "camera", "status.json") if self.dir else ""
        if st and os.path.isfile(st):
            try:
                with open(st, encoding="utf-8") as f:
                    out.update(json.load(f))
            except Exception:  # noqa: BLE001
                pass
        out["running"] = self.cam is not None and self.cam.poll() is None
        if (
            out["running"]
            and not out["ready"]
            and self.active
            and (time.monotonic_ns() - self.t0_ns) / 1e9 > 8
        ):
            out["err"] = out["err"] or "no camera frame after 8 s, see camera/cam.log"
        return out

    def snapshot(self) -> dict:
        """Recording state for the UI."""
        with self.lock:
            active, d = self.active, self.dir
            streams = {n: {"n": s.n, "hz": round(s.rate(), 1)} for n, s in self.streams.items()}
        return {
            "active": active,
            "dir": d,
            "msg": self.msg,
            "elapsed_s": (time.monotonic_ns() - self.t0_ns) / 1e9 if active else 0.0,
            "streams": streams,
            "camera": self.camera_status() if d else None,
            "free_gb": round(
                shutil.disk_usage(RECORD_ROOT if os.path.isdir(RECORD_ROOT) else _ROOT).free / 1e9,
                1,
            ),
        }

    @staticmethod
    def merge(rec_dir: str) -> None:
        """Merge each stream's chunk_*.npz into <stream>.npz, sorted by t_ns; chunks are kept."""
        for name in sorted(os.listdir(rec_dir)):
            sd = os.path.join(rec_dir, name)
            if not os.path.isdir(sd) or name == "camera":
                continue
            chunks = sorted(
                f
                for f in os.listdir(sd)
                if f.startswith("chunk_") and f.endswith(".npz") and ".part" not in f
            )
            if not chunks:
                continue
            parts: dict = {}
            for c in chunks:
                with np.load(os.path.join(sd, c), allow_pickle=False) as z:
                    for k in z.files:
                        parts.setdefault(k, []).append(z[k])
            out = {k: np.concatenate(v) for k, v in parts.items()}
            order = np.argsort(out["t_ns"], kind="stable")
            out = {k: v[order] for k, v in out.items()}
            np.savez(os.path.join(rec_dir, f"{name}.npz"), **out)
