#!/usr/bin/env python3
"""Draw the hand skeleton computed by `project_hand.py` on the recorded video.

    python drivers/tools/overlay_hand.py recordings/<name> --stills --frames 3210
    python drivers/tools/overlay_hand.py recordings/<name> --arm

`--arm` also draws the arm skeleton from `<rec>/skeleton_2d.npz` (`project_arm.py`). Seeing
both tells whether an offset comes from the arm chain or from the hand mounting transform.
Writes an mp4 (needs ffmpeg), or jpg stills with `--stills`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess

import cv2
import numpy as np

FCOLOR = {
    "thumb": (80, 80, 255),
    "index": (80, 200, 255),
    "middle": (80, 255, 120),
    "ring": (255, 200, 80),
    "pinky": (255, 120, 220),
}
ACOLOR = {"left": (120, 255, 160), "right": (120, 190, 255)}  # arm chains (BGR)


def draw_arm(img, uv, color, label=""):
    """Draw one arm chain; the last joint is drawn larger and labelled."""
    ok = np.isfinite(uv).all(-1)
    for a_, b_ in zip(range(len(uv) - 1), range(1, len(uv))):
        if ok[a_] and ok[b_]:
            cv2.line(img, tuple(np.int32(uv[a_])), tuple(np.int32(uv[b_])), color, 3, cv2.LINE_AA)
    for k, p in enumerate(uv):
        if not ok[k]:
            continue
        r = 8 if k == len(uv) - 1 else 5
        cv2.circle(img, tuple(np.int32(p)), r, color, -1, cv2.LINE_AA)
        cv2.circle(img, tuple(np.int32(p)), r, (25, 25, 25), 1, cv2.LINE_AA)
    if label and ok[-1]:
        cv2.putText(
            img,
            label,
            tuple(np.int32(uv[-1]) + [11, -11]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )


def draw(img, uv, chains, dim=False):
    """Draw the finger chains of one hand."""
    for fname, ids in chains.items():
        c = FCOLOR[fname]
        if dim:
            c = tuple(int(v * 0.55) for v in c)
        pts = uv[ids]
        ok = np.isfinite(pts).all(-1)
        for a, b in zip(range(len(pts) - 1), range(1, len(pts))):
            if ok[a] and ok[b]:
                cv2.line(img, tuple(np.int32(pts[a])), tuple(np.int32(pts[b])), c, 2, cv2.LINE_AA)
        for k, p in enumerate(pts):
            if ok[k]:
                cv2.circle(
                    img, tuple(np.int32(p)), 5 if k == len(pts) - 1 else 3, c, -1, cv2.LINE_AA
                )


def main() -> int:
    """Command line entry point."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("rec")
    ap.add_argument("--npz", default="")
    ap.add_argument("--out", default="")
    ap.add_argument(
        "--tag",
        default="",
        help="suffix of the output file names",
    )
    ap.add_argument("--frames", default="")
    ap.add_argument("--stills", action="store_true")
    ap.add_argument("--side", default="both", choices=["both", "left", "right"])
    ap.add_argument(
        "--arm",
        action="store_true",
        help="also draw the arm skeleton (reads <rec>/skeleton_2d.npz)",
    )
    ap.add_argument("--fps", type=float, default=30.0)
    a = ap.parse_args()
    rec = os.path.abspath(a.rec)
    z = np.load(a.npz or os.path.join(rec, "hand_2d.npz"), allow_pickle=True)
    chains = {k: np.array(v) for k, v in json.loads(str(z["chains"])).items()}
    P = {s: z[f"{s}_2d"] for s in ("left", "right")}
    sides = ["left", "right"] if a.side == "both" else [a.side]
    cap = cv2.VideoCapture(os.path.join(rec, "camera", "rgb.mp4"))
    nvid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    A = None
    if a.arm:
        ap_ = os.path.join(rec, "skeleton_2d.npz")
        if not os.path.isfile(ap_):
            raise SystemExit(f"--arm needs {ap_}; run project_arm.py first")
        za = np.load(ap_, allow_pickle=True)
        A = {s: za[f"{s}_2d"] for s in ("left", "right")}
        print(f"arm skeleton {ap_}  hand-eye {str(za['handeye'])}")
    # Only the calibration directory name goes into the image.
    _he = str(z["handeye"])
    _m = re.search(r"handeye_\d{8}_\d{6}", _he)
    info = ("handeye " + _m.group(0).replace("handeye_", "") if _m else "handeye ?") + (
        "  + arm" if a.arm else ""
    )
    print(f"video {nvid} frames {w}x{h}  {info}")

    if a.stills:
        idxs = [int(x) for x in a.frames.split(",")] if a.frames else [nvid // 3, nvid // 2]
        outdir = os.path.join(rec, "overlay_stills")
        os.makedirs(outdir, exist_ok=True)
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, img = cap.read()
            if not ok:
                print("cannot read frame", i)
                continue
            for s in sides:
                if A is not None:
                    draw_arm(img, A[s][i], ACOLOR[s], s[0].upper())
                draw(img, P[s][i], chains)
            cv2.putText(
                img,
                f"f{i}  {info}",
                (12, 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            p = os.path.join(outdir, f"hand_f{i:06d}{a.tag}.jpg")
            cv2.imwrite(p, img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            print("wrote", p)
        return 0

    out = a.out or os.path.join(rec, f"overlay{'_arm_hand' if a.arm else '_hand'}{a.tag}.mp4")
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
            f"{w}x{h}",
            "-r",
            str(a.fps),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            out,
        ],
        stdin=subprocess.PIPE,
    )
    i = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if i < len(P["left"]):
            for s in sides:
                if A is not None and i < len(A[s]):
                    draw_arm(img, A[s][i], ACOLOR[s], s[0].upper())
                draw(img, P[s][i], chains)
        cv2.putText(
            img,
            f"{i/a.fps:6.1f}s  f{i}",
            (12, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            img, info, (12, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (210, 210, 210), 1, cv2.LINE_AA
        )
        ff.stdin.write(img.tobytes())
        i += 1
        if i % 1800 == 0:
            print(f"  {i}/{nvid}", flush=True)
    ff.stdin.close()
    ff.wait()
    print(f"{i} frames -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
