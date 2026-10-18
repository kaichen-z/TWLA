#!/usr/bin/env python3
"""Guided glove -> L6 calibration from six poses of the human hand (command line).

    python glove/l6_calibrate.py --hand right
    python glove/l6_calibrate.py --hand right --live-hand --can can3   # robot hand follows

Poses (see glove.l6_calib.STEPS), each sampled for 1.5 s after pressing Enter:
    1 open hand          2 fist                 -> finger ranges
    3 thumb out          4 thumb in             -> thumb abduction range
    5 thumb straight     6 thumb bent           -> thumb flexion range
Each pair reads only its own glove joints, so the other fingers may be anywhere.

Only the glove is read; the L6 is used only with --live-hand, which drives it with the
previous calibration so the operator can compare while posing. The result is written to
glove/calib/l6_calibration_<hand>.json. teleop/hand_panel.py offers an alternative
calibration from robot/human pose pairs.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parent / "drivers"))

from glove.glove_source import WujiGloveSource  # noqa: E402
from glove.hand_motion import MotionConfig  # noqa: E402
from glove.l6_calib import STEPS, L6Calibration, default_path, identify  # noqa: E402


def sample(src, seconds: float) -> tuple[np.ndarray, float]:
    """Median (5, 5) joint angles over `seconds`, and the lowest confidence seen."""
    buf, confs = [], []
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        f = src.latest()
        ja = f.shape.joint_angles
        if ja is not None:
            buf.append(np.asarray(ja, float).reshape(5, 5).copy())
            if f.shape.joint_conf is not None:
                confs.append(float(np.min(f.shape.joint_conf)))
        time.sleep(0.02)
    if not buf:
        raise RuntimeError("no finger data")
    return np.median(np.stack(buf), axis=0), (min(confs) if confs else 1.0)


class _LiveHand:
    """Drives the L6 from the glove with the previous calibration (none: no following)."""

    RATE = 30.0

    def __init__(self, args, src) -> None:
        import threading

        from robot.l6_can import L6Can

        self.src, self.status, self._run = src, "", False
        self.hand = None
        if not args.can:
            self.status = "--live-hand needs --can; the robot hand will not follow"
            return
        if not os.path.isfile(args.out):
            self.status = f"no previous calibration ({args.out}); the robot hand will not follow"
            return
        self.cal = L6Calibration.load(args.out)
        hid = args.hand_id or (0x28 if args.hand == "left" else 0x27)
        self.hand = L6Can(channel=args.can, hand_id=hid)
        self.hand.write_speed([120] * 6)
        self.hand.write_torque([120] * 6)
        self._run = True
        threading.Thread(target=self._loop, daemon=True).start()
        self.status = f"robot hand following ({args.can} id=0x{hid:02X}, previous calibration)"

    def _loop(self) -> None:
        while self._run:
            f = self.src.latest()
            ja = f.shape.joint_angles if f is not None else None
            if ja is not None:
                try:
                    self.hand.write_position(self.cal.to_l6(ja))
                except Exception:  # noqa: BLE001
                    pass
            time.sleep(1.0 / self.RATE)

    def stop(self) -> None:
        """Stop and open the hand."""
        self._run = False
        if self.hand is not None:
            time.sleep(0.1)
            try:
                self.hand.write_position([255] * 6)
                time.sleep(0.3)
                self.hand.close()
            except Exception:  # noqa: BLE001
                pass


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--hand", choices=["left", "right"], default="right")
    ap.add_argument("--seconds", type=float, default=1.5, help="sampling time per pose")
    ap.add_argument("--out", default="", help="default: glove/calib/l6_calibration_<hand>.json")
    ap.add_argument("--live-hand", action="store_true", help="drive the L6 while calibrating")
    ap.add_argument("--can", default="", help="CAN interface of the hand (with --live-hand)")
    ap.add_argument(
        "--hand-id", type=lambda x: int(x, 0), default=0, help="default: 0x27 right, 0x28 left"
    )
    ap.add_argument("--min-conf", type=float, default=0.5, help="warn below this glove confidence")
    args = ap.parse_args()
    args.out = args.out or str(default_path(args.hand))

    src = WujiGloveSource(handedness=args.hand, config=MotionConfig(init_still_s=1.0))
    src.start()
    print("\nwaiting for finger data ...")
    for _ in range(400):
        if src.latest().shape.joint_angles is not None:
            break
        time.sleep(0.05)
    if src.latest().shape.joint_angles is None:
        print("no finger data after 20 s")
        src.close()
        return 1
    print(
        "finger data OK\nfor each pose: hold it, press Enter and keep still for "
        f"{args.seconds:.1f} s\n"
    )

    follower = None
    if args.live_hand:
        follower = _LiveHand(args, src)
        print("  " + follower.status + "\n")

    poses: dict[str, np.ndarray] = {}
    try:
        for key, _grp, prompt in STEPS:
            input(f"  {prompt}\n    press Enter ")
            m, conf = sample(src, args.seconds)
            poses[key] = m
            flag = "  (low confidence: check the glove fit)" if conf < args.min_conf else ""
            print(f"    sampled (min confidence {conf:.2f}){flag}\n")
    except KeyboardInterrupt:
        print("\naborted, nothing saved")
        if follower is not None:
            follower.stop()
        src.close()
        return 130

    cal, info, warn = identify(poses, hand=args.hand)
    for ln in info:
        print("  " + ln)
    for ln in warn:
        print("  warning: " + ln)
    print()
    print(cal.describe())

    if follower is not None:
        follower.stop()
    p = cal.save(args.out)
    print(f"\nsaved to {p}")
    src.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
