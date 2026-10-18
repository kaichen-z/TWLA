#!/usr/bin/env python3
"""Run the Wuji glove SDK in a child process.

The SDK holds the GIL while it scans or retries a connection (up to ~20 s), which would
freeze the whole teleoperation process. Each glove is therefore served by a child process
running `devices.GloveProbe`; the main process talks to it through `GloveProxy`, which has
the same interface (status / frame / retry / stop) and never imports the SDK.

    main process                                child process
    GloveProxy  <-- socketpair, length-prefixed pickle --  GloveProbe
      .status()      {"status": Status}            polled every 0.3 s
      .frame()       {"frame": GloveFrame}         new frames only, up to ~500 Hz
      .retry()   --> b"retry\\n"

A frozen child only delays messages; a dead child is restarted with back-off. The child
exits with its parent (PR_SET_PDEATHSIG).
"""

from __future__ import annotations

import argparse
import os
import pickle
import socket
import struct
import subprocess
import sys
import threading
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

_HDR = struct.Struct("!I")


def _send(sock: socket.socket, obj, lock: threading.Lock) -> None:
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    with lock:
        sock.sendall(_HDR.pack(len(data)) + data)


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("peer closed")
        buf += chunk
    return buf


def _die_with_parent() -> None:
    """Ask the kernel to send SIGTERM to this process when its parent dies (Linux)."""
    import ctypes
    import signal

    try:
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG
    except Exception:  # noqa: BLE001
        pass


def worker_main(hand: str, fd: int) -> int:
    """Child process: poll a GloveProbe and send its status and frames to the parent."""
    _die_with_parent()
    from devices import GloveProbe

    sock = socket.socket(fileno=fd)
    lock = threading.Lock()
    probe = GloveProbe(hand)  # driven by the loop below, its own thread is not started
    stop = threading.Event()

    def commands() -> None:
        f = sock.makefile("rb")
        for line in f:
            if line.strip() == b"retry":
                probe.retry()
        stop.set()  # the parent closed the socket

    threading.Thread(target=commands, daemon=True).start()

    last_ts = None
    next_status = 0.0
    try:
        while not stop.is_set():
            now = time.monotonic()
            if now >= next_status:
                from devices import BAD, SKIP, Status

                try:
                    st = probe.poll()  # SDK scan / connect happens here
                except Exception as exc:  # noqa: BLE001
                    st = Status(BAD, f"probe error: {type(exc).__name__}", [str(exc)[:200]])
                # Same bookkeeping as Probe.run, so status() keeps its debouncing.
                with probe._lock:
                    probe._st = st
                    if st.level == BAD:
                        probe._bad_run += 1
                    else:
                        probe._bad_run = 0
                        if st.level != SKIP:
                            probe._last_good = st
                _send(sock, {"status": probe.status()}, lock)
                next_status = time.monotonic() + GloveProbe.period
            f = probe.frame()
            if f is not None:
                ts = (f.motion.timestamp_us, f.shape.timestamp_us)
                if ts != last_ts:
                    last_ts = ts
                    # CLOCK_MONOTONIC is shared by all processes, so this timestamp can be
                    # aligned with the other recorded streams.
                    _send(sock, {"frame": f, "t_ns": time.monotonic_ns()}, lock)
            # Every finger frame (120 Hz) is forwarded; the SDK only exposes the latest
            # IMU frame, so part of the 800 Hz IMU stream is skipped.
            time.sleep(0.002)
    except (BrokenPipeError, ConnectionError, OSError):
        pass
    finally:
        probe.stop()
    return 0


class GloveProxy(threading.Thread):
    """Stand-in for GloveProbe in the main process; owns and restarts the child process."""

    SILENT_S = 25.0  # no message for this long = child busy or stuck
    FRAME_MAX_AGE_S = 0.5  # older frames are not handed out

    def __init__(self, handedness: str) -> None:
        super().__init__(daemon=True)
        self.handedness = handedness
        self._lock = threading.Lock()
        self._st = None
        self._st_t = 0.0
        self._frame = None
        self._frame_t = 0.0
        self._proc: subprocess.Popen | None = None
        self._sock: socket.socket | None = None
        self._running = True
        self._restarts = 0
        self._next_spawn = 0.0
        self.rec = None  # recorder.Recorder, or None

    def _spawn(self) -> None:
        parent, child = socket.socketpair()
        self._proc = subprocess.Popen(
            [
                sys.executable,
                os.path.abspath(__file__),
                "--hand",
                self.handedness,
                "--fd",
                str(child.fileno()),
            ],
            pass_fds=(child.fileno(),),
            cwd=os.path.dirname(_HERE),
            env=dict(os.environ, PYTHONNOUSERSITE="1"),
            stdout=subprocess.DEVNULL,
            preexec_fn=_die_with_parent,
        )
        child.close()
        self._sock = parent
        with self._lock:
            self._st_t = time.monotonic()  # grace period before "not responding"

    def _kill(self) -> None:
        s, self._sock = self._sock, None
        if s is not None:
            try:
                s.close()
            except OSError:
                pass
        p, self._proc = self._proc, None
        if p is not None and p.poll() is None:
            p.terminate()
            try:
                p.wait(timeout=3)
            except subprocess.TimeoutExpired:
                p.kill()

    def run(self) -> None:
        while self._running:
            if self._proc is None or self._proc.poll() is not None:
                if self._proc is not None:
                    self._restarts += 1
                    self._kill()
                    self._next_spawn = time.monotonic() + min(2.0 * self._restarts, 30.0)
                if time.monotonic() < self._next_spawn:
                    time.sleep(0.2)
                    continue
                try:
                    self._spawn()
                except Exception as exc:  # noqa: BLE001
                    self._set_err(f"cannot start the glove process: {type(exc).__name__}: {exc}")
                    self._restarts += 1
                    self._next_spawn = time.monotonic() + 10.0
                    continue
            try:
                n = _HDR.unpack(_recv_exact(self._sock, _HDR.size))[0]
                msg = pickle.loads(_recv_exact(self._sock, n))
            except Exception:  # noqa: BLE001
                # Child exited or socket closed; it is restarted on the next iteration.
                time.sleep(0.2)
                if self._proc is not None and self._proc.poll() is None:
                    self._kill()
                continue
            now = time.monotonic()
            with self._lock:
                if "status" in msg:
                    self._st, self._st_t = msg["status"], now
                elif "frame" in msg:
                    self._frame, self._frame_t = msg["frame"], now
            if "frame" in msg and self.rec is not None:
                self._record(msg["frame"], msg.get("t_ns") or time.monotonic_ns())

    def _record(self, f, t_ns: int) -> None:
        """Stream glove_<side>: one row per frame; missing values are NaN."""
        import numpy as np

        nan = float("nan")

        def arr(x, n):
            return np.full(n, nan) if x is None else np.asarray(x, dtype=np.float64).reshape(-1)[:n]

        m, sh = f.motion, f.shape
        self.rec.put(
            f"glove_{self.handedness}",
            t_ns,
            imu_ts_us=int(m.timestamp_us),
            imu_seq=int(m.seq),
            q_world_body=arr(m.q_world_body, 4),
            rpy_deg=arr(m.rpy_deg, 3),
            gyro=arr(m.gyro, 3),
            accel=arr(m.accel, 3),
            imu_dropped=int(m.dropped_frames),
            q_wrist=arr(f.q_wrist, 4),
            hand_ts_us=int(sh.timestamp_us),
            joint_angles=arr(sh.joint_angles, 25),  # (5, 5) rad, flattened
            joint_conf=arr(sh.joint_conf, 5),
            skeleton=arr(sh.skeleton_pos, 63),  # (21, 3) m, wrist frame, flattened
            hand_age_ms=float(f.hand_age_ms),
        )

    def _set_err(self, text: str) -> None:
        from devices import BAD, Status

        with self._lock:
            self._st, self._st_t = Status(BAD, "glove not connected", [text]), time.monotonic()

    def status(self):
        """Latest status from the child, or why there is none."""
        from devices import BAD, Status

        with self._lock:
            st, age = self._st, time.monotonic() - self._st_t
        if st is None:
            return Status(BAD, "glove process starting", ["the first SDK scan takes a few seconds"])
        if age > self.SILENT_S:
            return Status(
                BAD,
                "glove process not responding",
                [f"no message for {age:.0f} s; the SDK may be stuck scanning or connecting"],
            )
        return st

    def frame(self):
        """Latest frame, or None if it is older than FRAME_MAX_AGE_S."""
        with self._lock:
            if self._frame is None or time.monotonic() - self._frame_t > self.FRAME_MAX_AGE_S:
                return None
            return self._frame

    def retry(self) -> None:
        """Ask the child to retry now; restart it if it is not responding."""
        with self._lock:
            silent = time.monotonic() - self._st_t > self.SILENT_S
        if silent:
            self._kill()
            self._restarts, self._next_spawn = 0, 0.0
            return
        try:
            if self._sock is not None:
                self._sock.sendall(b"retry\n")
        except OSError:
            pass

    def stop(self) -> None:
        """Stop the proxy and its child."""
        self._running = False
        self._kill()


def main() -> int:
    ap = argparse.ArgumentParser(description="glove child process (started by GloveProxy)")
    ap.add_argument("--hand", choices=["left", "right"], required=True)
    ap.add_argument("--fd", type=int, required=True)
    a = ap.parse_args()
    return worker_main(a.hand, a.fd)


if __name__ == "__main__":
    raise SystemExit(main())
