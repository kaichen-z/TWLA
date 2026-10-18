#!/usr/bin/env python3
"""L6 hand panel and glove pairing calibration (Tk window).

    python teleop/hand_panel.py --hand left --can can2
    python teleop/hand_panel.py --hand right --can can3 --no-glove     # sliders only

One slider per L6 channel drives the hand directly (255 = extended / open, 0 = flexed /
closed); the two thumb channels are on top with fine-step buttons. The arm and the clutch
are not involved. Closing the window opens the hand.

Pairing calibration of the glove -> L6 mapping:
    1. put the robot hand into a pose with the sliders;
    2. copy the pose with the gloved hand (the whole hand, not only the finger being set);
    3. "record pair": the glove joint angles are sampled for 1.5 s, and their median is
       stored with the six L6 values (glove/calib/l6_pairs_<hand>.json, saved after every
       pair);
    4. "fit and save": glove.l6_calib.fit_pairs() writes glove/calib/l6_calibration_<hand>.json.
Pairs with the robot finger at 255 and at 0 define the two ends of each channel.

"Set OK pinch point" stores the current thumb flexion value as the L6 value at which the
robot thumb meets the index finger (`ok_l6` in the calibration file).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import tkinter as tk

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
for _p in (_HERE, _ROOT, os.path.join(_ROOT, "drivers")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from glove.l6_calib import L6Calibration, default_path, fit_pairs  # noqa: E402
from robot.l6_can import CMD_POSITION, L6Can  # noqa: E402

BG, PANEL, FG, DIM = "#0f1115", "#161a22", "#e6e8ee", "#8791a5"
FONT = ("DejaVu Sans", 10)
FONT_B = ("DejaVu Sans", 11, "bold")
MONO = ("DejaVu Sans Mono", 10)
CH = ["thumb flex", "thumb abd", "index", "middle", "ring", "little"]
HINT = {
    0: "255 = straight, 0 = bent towards the index",
    1: "255 = away from the palm, 0 = against it",
}
RECORD_TXT = "glove in pose: record pair"


class Panel:
    """The panel window."""

    RATE = 20.0  # command rate (Hz)
    SAMPLE_S = 1.5  # glove sampling time per pair

    def __init__(
        self,
        root: tk.Tk,
        hand: L6Can,
        calib_path: str,
        hand_name: str,
        glove=None,
        pairs_path: str = "",
        hand_side: str = "left",
    ) -> None:
        self.root, self.hand, self.calib_path = root, hand, calib_path
        self.glove, self.pairs_path, self.side = glove, pairs_path, hand_side
        self.val = [255] * 6
        self.actual = [0] * 6
        self.msg = tk.StringVar(value="")
        self.pmsg = tk.StringVar(value="")
        self._run = True
        self._sampling = False
        self._notes = []  # messages from worker threads to the UI
        self._nlock = threading.Lock()
        self.pairs = []
        if pairs_path and os.path.isfile(pairs_path):
            try:
                with open(pairs_path, encoding="utf-8") as f:
                    self.pairs = json.load(f).get("pairs", [])
            except Exception:  # noqa: BLE001
                self.pairs = []
        root.title(f"L6 hand panel: {hand_name}")
        root.configure(bg=BG)

        tk.Label(
            root,
            text=f"{hand_name}    255 = extended / open, 0 = flexed / closed",
            bg=BG,
            fg=FG,
            font=FONT_B,
        ).pack(anchor="w", padx=12, pady=(10, 4))

        self.scales, self.labels = [], []
        for i, name in enumerate(CH):
            big = i < 2  # thumb channels
            f = tk.Frame(root, bg=PANEL if big else BG)
            f.pack(fill="x", padx=10, pady=(6 if big else 1, 0))
            head = tk.Frame(f, bg=f["bg"])
            head.pack(fill="x", padx=8, pady=(4 if big else 0, 0))
            tk.Label(
                head,
                text=name,
                bg=f["bg"],
                fg=FG,
                font=FONT_B if big else FONT,
                width=10,
                anchor="w",
            ).pack(side="left")
            lab = tk.Label(
                head, text="255", bg=f["bg"], fg="#ffd479" if big else FG, font=MONO, width=5
            )
            lab.pack(side="left")
            self.labels.append(lab)
            if big:
                for d in (-5, -1, +1, +5):
                    tk.Button(
                        head,
                        text=f"{d:+d}",
                        width=3,
                        takefocus=0,
                        relief="flat",
                        bg="#222836",
                        fg=FG,
                        font=FONT,
                        command=lambda i=i, d=d: self.nudge(i, d),
                    ).pack(side="left", padx=2)
                tk.Label(head, text=HINT[i], bg=f["bg"], fg=DIM, font=FONT).pack(
                    side="left", padx=10
                )
            sc = tk.Scale(
                f,
                from_=0,
                to=255,
                orient="horizontal",
                showvalue=0,
                bg=f["bg"],
                fg=FG,
                troughcolor="#0b0d11",
                highlightthickness=0,
                takefocus=0,
                length=560,
                sliderlength=22 if big else 16,
                command=lambda v, i=i: self.set_val(i, int(float(v))),
            )
            sc.set(255)
            sc.pack(fill="x", padx=8, pady=(0, 4 if big else 0))
            self.scales.append(sc)

        bar = tk.Frame(root, bg=BG)
        bar.pack(fill="x", padx=10, pady=8)
        for txt, cmd, col in (
            ("open all (255)", lambda: self.preset(255), "#222836"),
            ("open fingers, keep thumb", self.fingers_open, "#222836"),
            ("set OK pinch point", self.save_ok, "#1f4a32"),
        ):
            tk.Button(
                bar,
                text=txt,
                command=cmd,
                takefocus=0,
                relief="flat",
                bg=col,
                fg=FG,
                font=FONT,
                padx=10,
                pady=4,
            ).pack(side="left", padx=(0, 8))
        self.readback = tk.Label(
            root, text="", bg=BG, fg=DIM, font=MONO, anchor="w", justify="left"
        )
        self.readback.pack(fill="x", padx=12)
        tk.Label(root, textvariable=self.msg, bg=BG, fg="#8fe3b4", font=FONT, anchor="w").pack(
            fill="x", padx=12, pady=(2, 6)
        )

        if self.glove is not None:
            pf = tk.Frame(root, bg=PANEL)
            pf.pack(fill="x", padx=10, pady=(4, 10))
            tk.Label(
                pf,
                text="Pairing: pose the robot hand, copy it with the glove, record",
                bg=PANEL,
                fg=FG,
                font=FONT_B,
            ).pack(anchor="w", padx=8, pady=(6, 2))
            self.glove_lab = tk.Label(
                pf,
                text="waiting for the glove ...",
                bg=PANEL,
                fg=DIM,
                font=MONO,
                anchor="w",
                justify="left",
            )
            self.glove_lab.pack(fill="x", padx=8)
            pb = tk.Frame(pf, bg=PANEL)
            pb.pack(fill="x", padx=8, pady=6)
            self.rec_btn = tk.Button(
                pb,
                text=RECORD_TXT,
                command=self.record_pair,
                takefocus=0,
                relief="flat",
                bg="#1f4a32",
                fg=FG,
                font=FONT_B,
                padx=12,
                pady=6,
            )
            self.rec_btn.pack(side="left", padx=(0, 8))
            for txt, cmd in (
                ("delete last pair", self.drop_last),
                ("fit and save", self.fit_and_save),
                ("fit thumb only", self.fit_thumb_only),
                ("clear all pairs", self.clear_pairs),
            ):
                tk.Button(
                    pb,
                    text=txt,
                    command=cmd,
                    takefocus=0,
                    relief="flat",
                    bg="#222836",
                    fg=FG,
                    font=FONT,
                    padx=10,
                    pady=4,
                ).pack(side="left", padx=(0, 8))
            self.pairs_lab = tk.Label(
                pf, text="", bg=PANEL, fg=DIM, font=MONO, anchor="w", justify="left"
            )
            self.pairs_lab.pack(fill="x", padx=8)
            tk.Label(
                pf,
                textvariable=self.pmsg,
                bg=PANEL,
                fg="#ffd479",
                font=FONT,
                anchor="w",
                justify="left",
                wraplength=760,
            ).pack(fill="x", padx=8, pady=(2, 8))
            self._refresh_pairs()

        threading.Thread(target=self._io, daemon=True).start()
        self._tick()

    def set_val(self, i: int, v: int) -> None:
        self.val[i] = max(0, min(255, v))
        self.labels[i].config(text=str(self.val[i]))

    def nudge(self, i: int, d: int) -> None:
        self.scales[i].set(max(0, min(255, self.val[i] + d)))

    def preset(self, v: int) -> None:
        for i in range(6):
            self.scales[i].set(v)

    def fingers_open(self) -> None:
        for i in range(2, 6):
            self.scales[i].set(255)

    def save_ok(self) -> None:
        """Write the current thumb flexion value as `ok_l6` into the calibration file."""
        try:
            with open(self.calib_path, encoding="utf-8") as f:
                d = json.load(f)
            old = d.get("ok_l6")
            d["ok_l6"] = int(self.val[0])
            d["note"] = (
                f"{d.get('note', '')} | {time.strftime('%m-%d %H:%M')} "
                f"OK pinch point set to {d['ok_l6']} (was {old})"
            ).strip(" |")
            tmp = self.calib_path + ".part"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(d, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.calib_path)
            self.msg.set(
                f"OK pinch point = {d['ok_l6']} (was {old}) saved to "
                f"{os.path.basename(self.calib_path)}; used from the next start of the follower"
            )
        except Exception as exc:  # noqa: BLE001
            self.msg.set(f"cannot write: {type(exc).__name__}: {exc}")

    def _note(self, text: str) -> None:
        with self._nlock:
            self._notes.append(text)

    def _glove_ja(self):
        f = self.glove.frame()
        return None if f is None else f.shape.joint_angles

    def record_pair(self) -> None:
        """Sample the glove for SAMPLE_S and store the median with the current L6 values."""
        if self._sampling:
            return
        ja = self._glove_ja()
        if ja is None:
            self.pmsg.set("no glove data")
            return
        th = np.degrees(np.asarray(ja, float).reshape(5, 5)[0])
        # Thumb MCP and IP both at the glove's +-15 deg limit: thumb flexion is not being read.
        if abs(abs(th[3]) - 15) < 0.3 and abs(abs(th[4]) - 15) < 0.3:
            self.pmsg.set(
                f"not recorded: thumb mcp={th[3]:+.0f} ip={th[4]:+.0f} deg are at the glove's "
                "joint limit, so thumb flexion is not being read. Bend the thumb away from the "
                "robot and the table and check that flexion (3+4) changes"
            )
            return
        self._sampling = True
        self.rec_btn.config(state="disabled", text=f"sampling {self.SAMPLE_S:.1f} s, hold still")
        l6 = list(self.val)
        actual = list(self.actual)

        def work():
            buf, confs = [], []
            t0 = time.monotonic()
            while time.monotonic() - t0 < self.SAMPLE_S:
                f = self.glove.frame()
                if f is not None and f.shape.joint_angles is not None:
                    buf.append(np.asarray(f.shape.joint_angles, float).reshape(25).copy())
                    if f.shape.joint_conf is not None:
                        confs.append(float(np.min(f.shape.joint_conf)))
                time.sleep(0.02)
            if len(buf) < 10:
                self._note(f"only {len(buf)} frames in {self.SAMPLE_S:.1f} s; pair not recorded")
            else:
                med = np.median(np.stack(buf), axis=0)
                self.pairs.append(
                    {
                        "l6": l6,
                        "l6_actual": actual,
                        "ja": med.tolist(),
                        "n": len(buf),
                        "conf_min": (min(confs) if confs else None),
                        "t": time.strftime("%m-%d %H:%M:%S"),
                    }
                )
                self._save_pairs()
                self._note(
                    f"pair {len(self.pairs)} recorded: L6 {l6} ({len(buf)} frames"
                    + (f", min confidence {min(confs):.2f}" if confs else "")
                    + ")"
                )
            self._sampling = False

        threading.Thread(target=work, daemon=True).start()

    def _save_pairs(self) -> None:
        os.makedirs(os.path.dirname(self.pairs_path), exist_ok=True)
        tmp = self.pairs_path + ".part"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"hand": self.side, "pairs": self.pairs}, f, ensure_ascii=False, indent=1)
        os.replace(tmp, self.pairs_path)

    def drop_last(self) -> None:
        if self.pairs:
            p = self.pairs.pop()
            self._save_pairs()
            self.pmsg.set(f"deleted the last pair (L6 {p['l6']}); {len(self.pairs)} left")
        self._refresh_pairs()

    def clear_pairs(self) -> None:
        self.pairs = []
        self._save_pairs()
        self.pmsg.set("all pairs cleared")
        self._refresh_pairs()

    def _report(self, head: str, info: list, warn: list) -> None:
        self.pmsg.set(
            head
            + "\n"
            + "\n".join(info)
            + ("\n" + "\n".join("warning: " + w for w in warn) if warn else "")
        )

    def fit_and_save(self) -> None:
        """Fit all six channels from the pairs and save the calibration."""
        base = None
        if os.path.isfile(self.calib_path):
            try:
                base = L6Calibration.load(self.calib_path)
            except Exception:  # noqa: BLE001
                base = None
        cal, info, warn = fit_pairs(self.pairs, hand=self.side, base=base)
        cal.note = (
            f"{time.strftime('%m-%d %H:%M')} pairing calibration, {len(self.pairs)} pairs "
            f"({os.path.basename(self.pairs_path)})"
        )
        print("\n".join(info))
        print("\n".join("warning: " + w for w in warn))
        print(cal.describe())
        if len(self.pairs) >= 2 and not any("at least" in w for w in warn):
            cal.save(self.calib_path)
            head = f"saved to {os.path.basename(self.calib_path)}"
        else:
            head = "not saved"
        self._report(head, info, warn)

    def fit_thumb_only(self) -> None:
        """Refit the two thumb channels only; the finger ranges are kept."""
        if not os.path.isfile(self.calib_path):
            self.pmsg.set("no calibration file to keep the fingers from; use 'fit and save' first")
            return
        base = L6Calibration.load(self.calib_path)
        cal, info, warn = fit_pairs(self.pairs, hand=self.side, base=base)
        info = [ln for ln in info if ln.startswith("thumb") or ln.endswith(" pairs")]
        warn = [w for w in warn if "thumb" in w or "at least" in w]
        ok = len(self.pairs) >= 2 and not any("at least" in w for w in warn)
        if ok:
            base.spans[0], base.spans[1] = list(cal.spans[0]), list(cal.spans[1])
            base.ok_raw = None
            base.note = (
                f"{time.strftime('%m-%d %H:%M')} thumb channels refitted "
                f"({len(self.pairs)} pairs, fingers kept)"
            )
            base.save(self.calib_path)
            head = (
                f"thumb channels saved to {os.path.basename(self.calib_path)} (fingers unchanged)"
            )
        else:
            head = "not saved"
        print("\n".join(info))
        print("\n".join("warning: " + w for w in warn))
        print(base.describe())
        self._report(head, info, warn)

    def _refresh_pairs(self) -> None:
        if self.glove is None:
            return
        rows = [f"{len(self.pairs)} pairs   ({os.path.basename(self.pairs_path)})"]
        for k, p in enumerate(self.pairs[-6:], start=max(0, len(self.pairs) - 6) + 1):
            deg = np.degrees(np.asarray(p["ja"]).reshape(5, 5)[0])
            rows.append(
                f"  #{k:2d} L6 {' '.join(f'{v:3d}' for v in p['l6'])}   thumb slots "
                + " ".join(f"{d:+5.0f}" for d in deg)
            )
        self.pairs_lab.config(text="\n".join(rows))

    def _io(self) -> None:
        """Send the slider values and read the hand position."""
        period = 1.0 / self.RATE
        n = 0
        while self._run:
            try:
                self.hand.write_position(list(self.val))
                n += 1
                if n % 5 == 0:  # the L6 does not report by itself
                    self.hand.request(CMD_POSITION)
                self.actual = list(self.hand.snapshot().get("position") or [0] * 6)
            except Exception:  # noqa: BLE001
                pass
            time.sleep(period)

    def _tick(self) -> None:
        self.readback.config(
            text="command  "
            + " ".join(f"{v:6d}" for v in self.val)
            + "\nmeasured "
            + " ".join(f"{v:6d}" for v in self.actual)
            + "\n         "
            + " ".join(f"{n[:6]:>6s}" for n in CH)
        )
        if self.glove is not None:
            ja = self._glove_ja()
            if ja is None:
                st = self.glove.status()
                self.glove_lab.config(
                    text="glove: no data  " + (getattr(st, "headline", "") if st else ""),
                    fg="#ff6b6b",
                )
            else:
                a = np.degrees(np.asarray(ja, float).reshape(5, 5))
                self.glove_lab.config(
                    fg=DIM,
                    text=(
                        "glove thumb slots 0..4 (deg) "
                        + " ".join(f"{v:+6.1f}" for v in a[0])
                        + f"   flexion (3+4) {a[0,3]+a[0,4]:+6.1f}  abduction (2) {a[0,2]:+6.1f}\n"
                        "fingers mcp+pip+dip          "
                        + " ".join(f"{a[r,0]+a[r,2]+a[r,3]:+6.1f}" for r in range(1, 5))
                    ),
                )
            with self._nlock:
                notes, self._notes = self._notes, []
            for n in notes:
                self.pmsg.set(n)
                self._refresh_pairs()
            if not self._sampling and self.rec_btn["state"] == "disabled":
                self.rec_btn.config(state="normal", text=RECORD_TXT)
        self.root.after(150, self._tick)

    def close(self) -> None:
        """Stop sending and open the hand."""
        self._run = False
        time.sleep(0.15)
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
    ap.add_argument("--hand", choices=["left", "right"], default="left")
    ap.add_argument("--can", required=True, help="CAN interface of the hand, e.g. can2")
    ap.add_argument(
        "--hand-id", type=lambda x: int(x, 0), default=0, help="default: 0x27 right, 0x28 left"
    )
    ap.add_argument("--speed", type=int, default=120)
    ap.add_argument("--torque", type=int, default=120)
    ap.add_argument("--no-glove", action="store_true", help="sliders only, no pairing")
    a = ap.parse_args()

    calib_path = str(default_path(a.hand))
    hid = a.hand_id or (0x28 if a.hand == "left" else 0x27)
    hand = L6Can(channel=a.can, hand_id=hid)
    hand.write_speed([a.speed] * 6)
    hand.write_torque([a.torque] * 6)
    print(f"L6 hand panel: {a.hand} on {a.can} id=0x{hid:02X}, calibration file {calib_path}")

    glove = None
    if not a.no_glove:
        from glove_worker import GloveProxy

        glove = GloveProxy(a.hand)
        glove.start()
    pairs_path = os.path.join(os.path.dirname(calib_path), f"l6_pairs_{a.hand}.json")

    root = tk.Tk()
    p = Panel(
        root,
        hand,
        calib_path,
        f"{a.hand} hand @{a.can}",
        glove=glove,
        pairs_path=pairs_path,
        hand_side=a.hand,
    )
    root.protocol("WM_DELETE_WINDOW", lambda: (p.close(), root.destroy()))
    try:
        root.mainloop()
    finally:
        p.close()
        if glove is not None:
            glove.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
