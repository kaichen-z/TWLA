#!/usr/bin/env python3
"""Draw the arm skeleton computed by `project_arm.py` on the recorded video.

    python drivers/tools/overlay_arm.py recordings/<name>
    python drivers/tools/overlay_arm.py recordings/<name> --stills --frames 1800,3600

Writes `<rec>/overlay_arm.mp4` (needs ffmpeg), or jpg stills with `--stills`.
"""

from __future__ import annotations

import argparse
import os
import subprocess

import cv2
import numpy as np

COLOR = {"left": (90, 220, 90), "right": (90, 160, 255)}  # BGR


def draw(img, uv, color, label=""):
    """Draw one skeleton (N, 2) as a polyline with joint markers; NaN points are skipped."""
    ok = np.isfinite(uv).all(-1)
    for a, b in zip(range(len(uv) - 1), range(1, len(uv))):
        if ok[a] and ok[b]:
            cv2.line(img, tuple(np.int32(uv[a])), tuple(np.int32(uv[b])), color, 2, cv2.LINE_AA)
    for k, p in enumerate(uv):
        if not ok[k]:
            continue
        r = 7 if k == len(uv) - 1 else 4
        cv2.circle(img, tuple(np.int32(p)), r, color, -1, cv2.LINE_AA)
        cv2.circle(img, tuple(np.int32(p)), r, (20, 20, 20), 1, cv2.LINE_AA)
    if label and ok[-1]:
        cv2.putText(
            img,
            label,
            tuple(np.int32(uv[-1]) + [10, -10]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )


def main() -> int:
    """Command line entry point."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("rec")
    ap.add_argument("--npz", default="")
    ap.add_argument("--video", default="")
    ap.add_argument("--out", default="")
    ap.add_argument(
        "--frames",
        default="",
        help="comma-separated frame numbers; with --stills only still frames are written",
    )
    ap.add_argument("--stills", action="store_true")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--tag", default="", help="suffix of the output file name")
    a = ap.parse_args()
    rec = os.path.abspath(a.rec)
    z = np.load(a.npz or os.path.join(rec, "skeleton_2d.npz"), allow_pickle=True)
    L2, R2 = z["left_2d"], z["right_2d"]
    video = a.video or os.path.join(rec, "camera", "rgb.mp4")
    cap = cv2.VideoCapture(video)
    nvid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"video {nvid} frames {w}x{h}   skeleton {len(L2)} frames   hand-eye {z['handeye']}")
    if nvid != len(L2):
        print(f"warning: video has {nvid} frames, skeleton {len(L2)}; aligned by index")

    if a.stills:
        idxs = (
            [int(x) for x in a.frames.split(",")]
            if a.frames
            else [0, nvid // 4, nvid // 2, 3 * nvid // 4]
        )
        outdir = os.path.join(rec, "overlay_stills")
        os.makedirs(outdir, exist_ok=True)
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, img = cap.read()
            if not ok:
                print("cannot read frame", i)
                continue
            draw(img, L2[i], COLOR["left"], "L")
            draw(img, R2[i], COLOR["right"], "R")
            cv2.putText(
                img,
                f"frame {i}  {z['handeye']}",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            p = os.path.join(outdir, f"arm_f{i:06d}{a.tag}.jpg")
            cv2.imwrite(p, img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            print("wrote", p)
        return 0

    out = a.out or os.path.join(rec, f"overlay_arm{a.tag}.mp4")
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
        if i < len(L2):
            draw(img, L2[i], COLOR["left"], "L")
            draw(img, R2[i], COLOR["right"], "R")
        cv2.putText(
            img,
            f"{i/a.fps:6.1f}s  f{i}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        ff.stdin.write(img.tobytes())
        i += 1
        if i % 900 == 0:
            print(f"  {i}/{nvid}", flush=True)
    ff.stdin.close()
    ff.wait()
    print(f"{i} frames -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
