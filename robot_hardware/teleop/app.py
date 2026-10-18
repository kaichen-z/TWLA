#!/usr/bin/env python3
"""Teleoperation console: a desktop window (Tk) for bimanual teleoperation and recording.

    python teleop/app.py                 # all devices
    python teleop/app.py --devices right_arm,right_hand --no-vive

Window: clutch banner, device status lights (click a device to select / deselect it), arm
buttons (enable / ready pose / home / relax), bimanual teleoperation, single-hand glove
following with a tactile heat map, and recording with a camera preview.

Keys: space toggles the clutch (only after "enable clutch" is ticked); Esc is the
emergency stop. Buttons do not take keyboard focus, so space never presses a button.

The UI never blocks on the backend: a collector thread gathers the backend state every
0.2 s and the UI only renders that snapshot; button actions run in a thread pool. The UI
thread sends the clutch heartbeat, so a frozen window disengages the clutch. Dangerous
buttons need a second click within 3 s instead of a modal dialog.
"""

from __future__ import annotations

import os
import sys
import threading
import time
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import backend as C  # noqa: E402

BG, PANEL, FG, DIM = "#0f1115", "#161a22", "#e6e8ee", "#8791a5"
OK, WARN, BAD, SKIP = "#2ecc71", "#f0b429", "#e74c3c", "#4a5164"
LVL = {"ok": OK, "warn": WARN, "bad": BAD, "skip": SKIP}
FONT = ("DejaVu Sans", 10)
FONT_B = ("DejaVu Sans", 10, "bold")
FONT_BIG = ("DejaVu Sans", 18, "bold")
MONO = ("DejaVu Sans Mono", 9)
FINGERS = ["thumb", "index", "middle", "ring", "little"]
CHANNELS = ["th.flex", "th.abd", "index", "middle", "ring", "little"]


class Collector(threading.Thread):
    """Gathers the backend state every 0.2 s; the UI only reads `latest()`."""

    PERIOD = 0.2

    def __init__(self, ctx: dict) -> None:
        super().__init__(daemon=True, name="ui-collector")
        self.ctx = ctx
        self._lock = threading.Lock()
        self._st: dict = {}
        self._t = 0.0
        self._prev_trk: dict = {}

    def latest(self):
        with self._lock:
            return self._st, self._t

    def run(self) -> None:
        while True:
            t0 = time.monotonic()
            st: dict = {}
            for key, fn in (
                ("clutch", lambda: self.ctx["clutch"].state()),
                ("final", lambda: self.ctx["final"].snapshot()),
                ("rec", lambda: self.ctx["rec"].snapshot()),
                ("job", lambda: self.ctx["runner"].snapshot()),
                ("targets", lambda: self.ctx["arm_targets"]()),
                ("busy", lambda: self.ctx["arm_busy"]()),
                ("raw", lambda: self.ctx["tracker_raw"]()),
            ):
                try:
                    st[key] = fn()
                except Exception as exc:  # noqa: BLE001
                    st[key + "_err"] = f"{type(exc).__name__}: {exc}"[:120]
            try:
                roles = self.ctx["final"].roles
                serials = [
                    t.get("serial")
                    for t in (st.get("raw") or {}).get("devices", [])
                    if t.get("kind") == "tracker" and t.get("serial")
                ]
                self.ctx["final"].ensure_roles(serials)
                st["strip"] = C.compact(
                    self.ctx["probes"], self.ctx["devices"], st.get("raw") or {}, roles
                )
            except Exception as exc:  # noqa: BLE001
                st["strip_err"] = f"{type(exc).__name__}: {exc}"[:120]
            st["trk_moving"] = self._moving(st.get("raw") or {})
            with self._lock:
                self._st, self._t = st, time.monotonic()
            time.sleep(max(0.02, self.PERIOD - (time.monotonic() - t0)))

    def _moving(self, raw: dict) -> dict:
        """Speed of each tracker (m/s), to tell which one is being held."""
        out, now = {}, time.monotonic()
        for d in raw.get("devices", []):
            if d.get("kind") != "tracker" or not d.get("pos") or not d.get("valid"):
                continue
            s = d.get("serial")
            p = self._prev_trk.get(s)
            if p and now - p[0] > 1e-3:
                v = sum((a - b) ** 2 for a, b in zip(d["pos"], p[1])) ** 0.5 / (now - p[0])
                out[s] = v
            self._prev_trk[s] = (now, d["pos"])
        return out


def frame(parent, title: str = "") -> tk.Frame:
    """Panel with an optional title."""
    f = tk.Frame(parent, bg=PANEL, highlightthickness=1, highlightbackground="#232733")
    if title:
        tk.Label(f, text=title, bg=PANEL, fg=FG, font=FONT_B).pack(anchor="w", padx=8, pady=(6, 2))
    return f


class Btn(tk.Button):
    """Button without keyboard focus; with confirm=True it needs a second click within 3 s."""

    def __init__(self, parent, text: str, cmd, confirm: bool = False, color: str = "#222836"):
        super().__init__(
            parent,
            text=text,
            command=self._click,
            takefocus=0,
            bg=color,
            fg=FG,
            activebackground="#2c3242",
            activeforeground=FG,
            relief="flat",
            font=FONT,
            padx=10,
            pady=4,
            highlightthickness=0,
        )
        self._cmd, self._confirm, self._text, self._color = cmd, confirm, text, color
        self._armed_until = 0.0

    def set_text(self, t: str) -> None:
        if t != self._text:
            self._text = t
            if time.monotonic() > self._armed_until:
                self.config(text=t)

    def _click(self) -> None:
        if str(self["state"]) == "disabled":
            return
        if self._confirm and time.monotonic() > self._armed_until:
            self._armed_until = time.monotonic() + 3.0
            self.config(text="click again to confirm", bg="#8a2f2f")
            self.after(3000, self._disarm)
            return
        self._armed_until = 0.0
        self.config(text=self._text, bg=self._color)
        self._cmd()

    def _disarm(self) -> None:
        if time.monotonic() >= self._armed_until:
            self._armed_until = 0.0
            self.config(text=self._text, bg=self._color)


class App:
    """Main window."""

    def __init__(self, root: tk.Tk, ctx: dict) -> None:
        self.root, self.ctx = root, ctx
        self.pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ui-act")
        self._hands: dict = {}
        self.col = Collector(ctx)
        self.col.start()
        self.msg = tk.StringVar(value="")
        root.title("Teleoperation console")
        root.configure(bg=BG)
        root.geometry("1400x900")
        self._build()
        root.bind_all("<space>", self._space)
        root.bind_all("<Escape>", lambda e: (self.ctx["clutch"].panic(), "break")[1])
        self._beat()
        self._refresh()

    def act(self, label: str, fn) -> None:
        """Run a button action in the thread pool and report the result."""
        self.msg.set(f"{label} ...")

        def run():
            try:
                r = fn()
                err = r if isinstance(r, str) else ""
                self.msg.set(f"{label}: {err}" if err else f"{label}: done")
            except Exception as exc:  # noqa: BLE001
                self.msg.set(f"{label}: {type(exc).__name__}: {exc}"[:160])

        self.pool.submit(run)

    def _space(self, _e=None):
        self.ctx["clutch"].toggle()
        return "break"

    def _beat(self) -> None:
        # Dead-man switch: the heartbeat comes from the UI thread.
        try:
            self.ctx["clutch"].heartbeat(True)
        except Exception:  # noqa: BLE001
            pass
        self.root.after(200, self._beat)

    def _radio(self, parent, text, value, var, command=None) -> None:
        tk.Radiobutton(
            parent,
            text=text,
            value=value,
            variable=var,
            takefocus=0,
            command=command,
            bg=PANEL,
            fg=FG,
            selectcolor="#2a2312",
            activebackground=PANEL,
            activeforeground=FG,
            font=FONT,
        ).pack(side="left", padx=4)

    def _build(self) -> None:  # noqa: C901
        r = self.root
        top = tk.Frame(r, bg=BG)
        top.pack(fill="x", padx=10, pady=(10, 4))

        # Clutch banner.
        self.cl = tk.Frame(top, bg=PANEL, highlightthickness=2, highlightbackground="#232733")
        self.cl.pack(fill="x")
        self.cl_big = tk.Label(self.cl, text="...", bg=PANEL, fg=FG, font=FONT_BIG)
        self.cl_big.pack(side="left", padx=14, pady=8)
        self.cl_sub = tk.Label(self.cl, text="", bg=PANEL, fg=DIM, font=FONT, justify="left")
        self.cl_sub.pack(side="left", padx=6)
        Btn(self.cl, "STOP (Esc)", lambda: self.ctx["clutch"].panic(), color="#8a2f2f").pack(
            side="right", padx=8
        )
        self.arm_var = tk.BooleanVar(value=False)
        self.arm_cb = tk.Checkbutton(
            self.cl,
            text="enable clutch (space)",
            variable=self.arm_var,
            command=lambda: self.ctx["clutch"].set_armed(self.arm_var.get()),
            takefocus=0,
            bg=PANEL,
            fg=FG,
            selectcolor="#2a2312",
            activebackground=PANEL,
            activeforeground=FG,
            font=FONT_B,
        )
        self.arm_cb.pack(side="right", padx=8)

        # Status lights, in a grid.
        self.strip = tk.Frame(top, bg=BG)
        self.strip.pack(fill="x", pady=(6, 0))
        self.chips: dict = {}
        self.STRIP_COLS = 5

        body = tk.Frame(r, bg=BG)
        body.pack(fill="both", expand=True, padx=10, pady=4)
        body.grid_columnconfigure(0, weight=3)
        body.grid_columnconfigure(1, weight=2, minsize=420)
        body.grid_rowconfigure(0, weight=1)
        left = tk.Frame(body, bg=BG)
        left.grid(row=0, column=0, sticky="nsew")
        right = tk.Frame(body, bg=BG)
        right.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

        # Arms.
        fa = frame(left, "Arms")
        fa.pack(fill="x", pady=(0, 6))
        row = tk.Frame(fa, bg=PANEL)
        row.pack(fill="x", padx=8, pady=4)
        ao = self.ctx["armops"]
        self.b_enable = Btn(row, "enable", lambda: self.act("enable", ao.enable_all))
        self.b_ready = Btn(
            row, "ready pose", lambda: self.act("ready pose", ao.ready_all), confirm=True
        )
        self.b_home = Btn(row, "home", lambda: self.act("home", ao.home_all), confirm=True)
        self.b_relax = Btn(
            row, "relax", lambda: self.act("relax", self._relax), confirm=True, color="#5a2a2a"
        )
        for b in (self.b_enable, self.b_ready, self.b_home, self.b_relax):
            b.pack(side="left", padx=(0, 6))
        self.job_lbl = tk.Label(
            fa, text="", bg=PANEL, fg=DIM, font=FONT, anchor="w", justify="left"
        )
        self.job_lbl.pack(fill="x", padx=8)
        self.job_log = tk.Text(
            fa,
            height=6,
            bg="#0b0d11",
            fg="#c9d3e6",
            font=MONO,
            relief="flat",
            takefocus=0,
            state="disabled",
        )
        self.job_log.pack(fill="x", padx=8, pady=(2, 8))

        # Bimanual teleoperation.
        ff = frame(left, "Teleoperation (arms + hands, 1:1, fast-motion lock)")
        ff.pack(fill="x", pady=(0, 6))
        row = tk.Frame(ff, bg=PANEL)
        row.pack(fill="x", padx=8, pady=4)
        tk.Label(row, text="sides", bg=PANEL, fg=DIM, font=FONT).pack(side="left")
        self.side_var = tk.StringVar(value="both")
        fin = self.ctx["final"]
        for v, t in (("left", "left"), ("right", "right"), ("both", "both")):
            self._radio(
                row,
                t,
                v,
                self.side_var,
                command=lambda: self.act("sides", lambda: fin.set_sides(self.side_var.get())),
            )
        self.b_dry = Btn(row, "dry run", lambda: self.act("dry run", lambda: fin.start(False)))
        self.b_live = Btn(
            row,
            "start LIVE",
            lambda: self.act("start LIVE", lambda: fin.start(True)),
            confirm=True,
            color="#6b2020",
        )
        self.b_stop = Btn(row, "stop", lambda: self.act("stop", fin.stop), color="#5a2a2a")
        for b in (self.b_dry, self.b_live, self.b_stop):
            b.pack(side="left", padx=(12, 0))
        self.fin_lbl = tk.Label(
            ff, text="", bg=PANEL, fg=DIM, font=FONT, anchor="w", justify="left"
        )
        self.fin_lbl.pack(fill="x", padx=8)
        sides = tk.Frame(ff, bg=PANEL)
        sides.pack(fill="x", padx=8, pady=4)
        self.side_txt = {}
        for s in ("left", "right"):
            t = tk.Label(
                sides,
                text="",
                bg="#0b0d11",
                fg="#c9d3e6",
                font=MONO,
                anchor="nw",
                justify="left",
                width=44,
                height=8,
            )
            t.pack(side="left", padx=(0, 6), fill="x", expand=True)
            self.side_txt[s] = t
        row = tk.Frame(ff, bg=PANEL)
        row.pack(fill="x", padx=8, pady=(2, 8))
        self.roles_lbl = tk.Label(
            row, text="", bg=PANEL, fg=FG, font=FONT, anchor="w", justify="left"
        )
        self.roles_lbl.pack(side="left")
        Btn(row, "swap trackers", lambda: self.act("swap trackers", fin.swap)).pack(
            side="left", padx=8
        )
        row = tk.Frame(ff, bg=PANEL)
        row.pack(fill="x", padx=8, pady=(0, 4))
        tk.Label(row, text="fast-motion lock (m/s)", bg=PANEL, fg=DIM, font=FONT).pack(
            side="left", padx=(0, 4)
        )
        self.thr = tk.Spinbox(
            row,
            from_=0.3,
            to=10.0,
            increment=0.1,
            width=5,
            font=FONT,
            command=lambda: fin.set_threshold(float(self.thr.get())),
        )
        self.thr.delete(0, "end")
        self.thr.insert(0, f"{fin.threshold:.1f}")
        self.thr.pack(side="left")
        Btn(row, "reset speed stats", lambda: self.act("reset stats", fin.reset_stats)).pack(
            side="left", padx=8
        )
        self.guard_lbl = tk.Label(
            ff, text="", bg=PANEL, fg=DIM, font=MONO, anchor="w", justify="left"
        )
        self.guard_lbl.pack(fill="x", padx=8, pady=(0, 8))

        # Single hand following the glove (no arm, no clutch).
        fh = frame(left, "Hand only (glove -> L6, no clutch, arm not used)")
        fh.pack(fill="x", pady=(0, 6))
        row = tk.Frame(fh, bg=PANEL)
        row.pack(fill="x", padx=8, pady=4)
        tk.Label(row, text="hand", bg=PANEL, fg=DIM, font=FONT).pack(side="left")
        self.hand_var = tk.StringVar(value="left")
        for v in ("left", "right"):
            self._radio(row, v, v, self.hand_var)
        self.b_hdry = Btn(row, "dry run", lambda: self._hand_start(False))
        self.b_hlive = Btn(
            row, "drive hand LIVE", lambda: self._hand_start(True), confirm=True, color="#6b2020"
        )
        self.b_hstop = Btn(row, "stop", self._hand_stop, color="#5a2a2a")
        for b in (self.b_hdry, self.b_hlive, self.b_hstop):
            b.pack(side="left", padx=(12, 0))
        self.tac_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            row,
            text="read tactile (commands ~14 Hz)",
            variable=self.tac_var,
            takefocus=0,
            bg=PANEL,
            fg=FG,
            selectcolor="#2a2312",
            activebackground=PANEL,
            activeforeground=FG,
            font=FONT,
        ).pack(side="left", padx=(16, 0))
        self.hand_txt = tk.Label(
            fh,
            text="",
            bg="#0b0d11",
            fg="#c9d3e6",
            font=MONO,
            anchor="nw",
            justify="left",
            height=5,
        )
        self.hand_txt.pack(fill="x", padx=8, pady=(2, 4))
        # Tactile heat map: 5 fingers x 12 x 6 cells, created once and recoloured.
        self.tac_c = tk.Canvas(fh, height=150, bg="#0b0d11", highlightthickness=0)
        self.tac_c.pack(fill="x", padx=8, pady=(0, 4))
        self.tac_lbl = tk.Label(fh, text="", bg=PANEL, fg=DIM, font=MONO, anchor="w")
        self.tac_lbl.pack(fill="x", padx=8, pady=(0, 8))
        self._tac_cells: dict = {}

        # Recording.
        fr = frame(right, "Recording")
        fr.pack(fill="both", expand=True)
        row = tk.Frame(fr, bg=PANEL)
        row.pack(fill="x", padx=8, pady=4)
        rec = self.ctx["rec"]
        self.b_rec = Btn(
            row,
            "start recording",
            lambda: self.act("start recording", lambda: rec.start(meta=C.record_meta(self.ctx))),
            color="#1f4a32",
        )
        self.b_rec_stop = Btn(
            row, "stop recording", lambda: self.act("stop recording", rec.stop), color="#5a2a2a"
        )
        self.b_rec.pack(side="left")
        self.b_rec_stop.pack(side="left", padx=6)
        self.rec_lbl = tk.Label(fr, text="", bg=PANEL, fg=FG, font=MONO, anchor="w", justify="left")
        self.rec_lbl.pack(fill="x", padx=8)
        # Camera preview: the camera process writes a small JPEG every 0.2 s while recording.
        self.cam_img = tk.Label(
            fr, bg="#0b0d11", fg=DIM, font=FONT, text="(camera preview while recording)"
        )
        self.cam_img.pack(padx=8, pady=(4, 8))
        self._cam_photo = None
        self._cam_mtime = 0.0

        tk.Label(r, textvariable=self.msg, bg=BG, fg=FG, font=FONT, anchor="w").pack(
            fill="x", padx=12, pady=(0, 8)
        )

    def _hand_follow(self, side: str):
        """One L6Follow per side, without clutch (always follows)."""
        fl = self._hands.get(side)
        if fl is None:
            fl = C.L6Follow(self.ctx["devices"], self.ctx["probes"], clutch=None, side=side)
            fl.rec = self.ctx["rec"]
            self._hands[side] = fl
        return fl

    def _hand_start(self, live: bool):
        side = self.hand_var.get()
        fl = self._hand_follow(side)
        fl.fast_io = bool(self.tac_var.get())  # tactile reading needs the interleaved I/O
        self.act(f"{side} hand {'LIVE' if live else 'dry run'}", lambda: fl.start(live))

    def _hand_stop(self):
        for side, fl in self._hands.items():
            if fl.state in ("running", "starting", "opening"):
                self.act(f"stop {side} hand", fl.stop)

    def _relax(self):
        C.stop_controls(self.ctx)  # stop all control before disabling
        return self.ctx["armops"].relax_all()

    @staticmethod
    def _set(w, **kw) -> None:
        """Configure a widget only if something changed (avoids flicker)."""
        cur = {k: w.cget(k) for k in kw}
        if cur != {k: str(v) for k, v in kw.items()}:
            w.config(**kw)

    @staticmethod
    def _set_text(w: tk.Text, s: str) -> None:
        if getattr(w, "_last", None) == s:
            return
        w._last = s
        w.config(state="normal")
        w.delete("1.0", "end")
        w.insert("1.0", s)
        w.config(state="disabled")
        w.see("end")

    def _refresh(self) -> None:
        try:
            self._render()
        except Exception as exc:  # noqa: BLE001
            self.msg.set(f"display error (control not affected): {type(exc).__name__}: {exc}"[:160])
        self.root.after(200, self._refresh)

    def _render(self) -> None:  # noqa: C901
        st, t = self.col.latest()
        if not st:
            return
        age = time.monotonic() - t
        stale = f"   state not updated for {age:.0f} s (backend busy)" if age > 2 else ""

        # Clutch.
        c = st.get("clutch")
        if c is not None:
            if c.engaged:
                big, col, sub = "ENGAGED", "#ffd479", "space: disengage   Esc: stop"
            elif c.pending_s > 0:
                big, col, sub = f"engaging in {c.pending_s:.1f} s", "#ffd479", "space: cancel"
            elif c.locked:
                big, col, sub = "LOCKED", "#ff8f80", f"{c.lock_reason}; space to unlock"
            else:
                big, col, sub = "disengaged", "#8fe3b4", c.reason or "space: engage"
            self._set(self.cl_big, text=big, fg=col)
            self._set(self.cl_sub, text=sub + stale)
            self._set(self.cl, highlightbackground="#f0b429" if c.engaged else "#232733")
            if self.arm_var.get() != bool(c.armed):
                self.arm_var.set(bool(c.armed))

        # Status lights; rebuilt when the set of lights changes.
        chips_now = st.get("strip") or []
        keys = [c["key"] for c in chips_now]
        if keys != list(self.chips.keys()):
            for w in self.chips.values():
                w.destroy()
            self.chips = {}
            for n, chip in enumerate(chips_now):
                k = chip["key"]
                w = tk.Label(self.strip, bg=PANEL, fg=FG, font=FONT, padx=8, pady=3, anchor="w")
                w.grid(
                    row=n // self.STRIP_COLS,
                    column=n % self.STRIP_COLS,
                    sticky="ew",
                    padx=(0, 5),
                    pady=(0, 4),
                )
                self.strip.grid_columnconfigure(n % self.STRIP_COLS, weight=1, uniform="chip")
                if chip.get("toggleable"):
                    w.bind("<Button-1>", lambda e, key=k: self._toggle_dev(key))
                    w.config(cursor="hand2")
                self.chips[k] = w
        for chip in chips_now:
            w = self.chips[chip["key"]]
            note = chip.get("note") or ""
            can = f" @{chip['can']}" if chip.get("can") else ""
            self._set(w, text=f"● {chip['label']}{can}  {note}", fg=LVL.get(chip.get("level"), FG))

        # Arms.
        who = ", ".join(st.get("targets") or []) or "none"
        busy = st.get("busy") or []
        j = st.get("job") or {}
        running = j.get("state") == "running"
        for b, name in (
            (self.b_enable, "enable"),
            (self.b_ready, "ready pose"),
            (self.b_home, "home"),
        ):
            b.set_text(f"{name} ({who})")
            self._set(b, state="disabled" if (running or busy or who == "none") else "normal")
        self.b_relax.set_text(f"relax ({who})")
        self._set(self.b_relax, state="disabled" if (running or who == "none") else "normal")
        jm = (
            f"{j.get('name')}: {j.get('msg')}"
            if running
            else (
                f"{j.get('name')}: done"
                if j.get("state") == "done"
                else f"{j.get('name')} failed: {j.get('msg')}" if j.get("state") == "failed" else ""
            )
        )
        if busy:
            jm = f"{', '.join(busy)} is driving the arms; enable / ready / home locked   " + jm
        self._set(self.job_lbl, text=jm)
        log = [ln for ln in (j.get("log") or []) if "moving=True" not in ln][-30:]
        self._set_text(self.job_log, "\n".join(log))

        # Bimanual teleoperation.
        F = st.get("final") or {}
        sd = F.get("sides") or []
        sv = "both" if len(sd) == 2 else (sd[0] if sd else "both")
        if self.side_var.get() != sv:
            self.side_var.set(sv)
        fs = F.get("state", "idle")
        fr = fs in ("running", "starting")
        self._set(self.b_dry, state="disabled" if fr else "normal")
        self._set(self.b_live, state="disabled" if fr else "normal")
        self._set(self.b_stop, state="normal" if fr else "disabled")
        self._set(
            self.fin_lbl,
            text=f"state: {fs}{' (LIVE)' if F.get('live') and fr else ''}   {F.get('msg') or ''}",
        )
        for s in ("left", "right"):
            A, H = (F.get("arms") or {}).get(s), (F.get("hands") or {}).get(s)
            lines = [f"{s}" + ("" if s in sd else " (not selected)")]
            if A:
                lines.append(f"arm @{A.get('can')}: {A.get('state')}  {A.get('msg', '')[:40]}")
                for r_ in A.get("arm") or []:
                    if r_[0] in ("IK", "jumps (>15 deg/frame)", "rejected by SDK", "joint angles"):
                        lines.append(f"  {r_[0]}: {r_[1]}")
                for r_ in A.get("link") or []:
                    if r_[0] == "Tracker":
                        lines.append(f"  tracker: {r_[1]}")
            else:
                lines.append("arm: not started")
            if H:
                io = (H.get("tel") or {}).get("io") or {}
                lines.append(f"hand: {H.get('state')}  {H.get('msg', '')[:36]}")
                if io:
                    lines.append(
                        f"  command/position {io.get('cmd_pos_hz')} Hz, "
                        f"tactile {io.get('tactile_hand_hz')} Hz"
                    )
            else:
                lines.append("hand: not started")
            self._set(self.side_txt[s], text="\n".join(lines[:9]))

        roles = F.get("roles") or {}
        mv = st.get("trk_moving") or {}

        def rl(side):
            sn = roles.get(side) or ""
            m = mv.get(sn, 0) > 0.05
            return f"{side} = {sn[-8:] or 'unassigned'}{'  (moving)' if m else ''}"

        self._set(
            self.roles_lbl,
            text=f"trackers   {rl('left')}   {rl('right')}   (move one to identify it)",
        )
        G = F.get("guard")
        if G:
            sp = G.get("speed") or {}
            lines = [
                f"speed  left {sp.get('left', 0):.2f}  right {sp.get('right', 0):.2f} m/s"
                f"   threshold {G.get('threshold', 0):.2f}"
            ]
            for s_, x in (G.get("stats") or {}).items():
                lines.append(
                    f"{s_} while engaged: n={x['n']}  p50 {x['p50']:.2f}  p95 {x['p95']:.2f}"
                    f"  p99 {x['p99']:.2f}  max {x['max']:.2f}"
                )
            for e in (G.get("events") or [])[-2:]:
                lines.append(f"locked {e[0]} {e[1]}")
            self._set(self.guard_lbl, text="\n".join(lines))
        else:
            self._set(self.guard_lbl, text="fast-motion lock: active once teleoperation starts")

        # Single hand.
        side = self.hand_var.get()
        fl = self._hands.get(side)
        sn = fl.snapshot() if fl is not None else {}
        run = sn.get("state") in ("running", "starting", "opening")
        self._set(self.b_hdry, state="disabled" if run else "normal")
        self._set(self.b_hlive, state="disabled" if run else "normal")
        self._set(self.b_hstop, state="normal" if run else "disabled")
        tel = sn.get("tel") or {}
        lines = [
            f"{side} hand: {sn.get('state', 'not started')}"
            f"{' (LIVE)' if sn.get('live') and run else ''}  {sn.get('msg', '')[:44]}"
        ]
        if tel:

            def row6(label, vals):
                return f"{label:<10}" + " ".join(f"{int(v):7d}" for v in (vals or [0] * 6))

            lines.append(" " * 10 + " ".join(f"{n:>7s}" for n in CHANNELS))
            lines.append(row6("command", tel.get("smoothed")))
            lines.append(row6("measured", tel.get("actual")))
            if tel.get("raw_deg"):
                sp = tel.get("span_deg") or [[0, 0]] * 6
                lines.append(row6("glove deg", tel["raw_deg"]))
                lines.append(row6("-> 255", [a for a, _ in sp]))
                lines.append(row6("-> 0", [b for _, b in sp]))
            io = tel.get("io") or {}
            if io:
                lines.append(
                    f"command/position {io.get('cmd_pos_hz')} Hz, "
                    f"tactile {io.get('tactile_hand_hz')} Hz, reply age {tel.get('age')} s"
                )
        else:
            lines.append("(after start: command vs measured hand position)")
        self._set(self.hand_txt, text="\n".join(lines))
        # Tactile: the single-hand follower, else the same hand in teleoperation.
        tac, tn = tel.get("touch"), tel.get("touch_n", 0)
        if tac is None:
            H5 = (F.get("hands") or {}).get(side) if isinstance(F, dict) else None
            t5 = (H5 or {}).get("tel") or {}
            tac, tn = t5.get("touch"), t5.get("touch_n", 0)
        self._draw_tactile(tac, tn)

        # Recording.
        R = st.get("rec") or {}
        if R.get("active"):
            e = int(R.get("elapsed_s") or 0)
            cam = R.get("camera") or {}
            lines = [
                f"RECORDING {e // 60:02d}:{e % 60:02d}   free disk {R.get('free_gb')} GB",
                f"  {R.get('dir', '')}",
                "  camera: "
                + (
                    cam.get("err")
                    or (
                        f"{cam.get('frames', 0)} frames, {cam.get('fps', 0)} fps,"
                        f" {cam.get('dropped', 0)} dropped"
                        if cam.get("ready")
                        else "starting ..."
                    )
                ),
            ]
            for k, v in sorted((R.get("streams") or {}).items()):
                lines.append(f"  {k:18s} {v['n']:7d}  {v['hz']:6.1f} Hz")
            self._set(self.rec_lbl, text="\n".join(lines), fg="#ff9f9f")
        else:
            self._set(
                self.rec_lbl,
                text=(R.get("msg") or "not recording") + f"\nfree disk {R.get('free_gb', '?')} GB",
                fg=DIM,
            )
        self._preview((R.get("camera") or {}).get("preview", ""), bool(R.get("active")))
        self._set(self.b_rec, state="disabled" if R.get("active") else "normal")
        self._set(self.b_rec_stop, state="normal" if R.get("active") else "disabled")

    @staticmethod
    def _tac_color(v: int) -> str:
        """Heat map colour: dark grey at 0, yellow to red with pressure."""
        if v <= 2:
            return "#161a22"
        t = min(1.0, v / 120.0)  # light contact already gives tens; 255 is rarely reached
        r = int(80 + 175 * t)
        g = int(80 + 120 * (1 - t))
        b = int(60 * (1 - t))
        return f"#{r:02x}{g:02x}{b:02x}"

    def _draw_tactile(self, touch: dict | None, n: int) -> None:
        """Five 12 x 6 heat maps; the cells are created once and then recoloured."""
        c = self.tac_c
        if not self._tac_cells:
            w = max(c.winfo_width(), 600)
            cw, ch = 9, 10
            pad = (w - 5 * (6 * cw + 24)) / 6
            for fi, key in enumerate(FINGERS):
                x0 = pad + fi * (6 * cw + 24 + pad)
                c.create_text(x0 + 3 * cw, 10, text=key, fill=DIM, font=FONT, anchor="n")
                for r in range(12):
                    for col in range(6):
                        cid = c.create_rectangle(
                            x0 + col * cw,
                            26 + r * ch,
                            x0 + (col + 1) * cw - 1,
                            26 + (r + 1) * ch - 1,
                            width=0,
                            fill="#161a22",
                        )
                        self._tac_cells[(key, r, col)] = cid
        if not touch:
            self._set(self.tac_lbl, text="tactile: no data (tick 'read tactile' and start)")
            return
        mx = {}
        for key in FINGERS:
            grid = touch.get(key) or []
            m = 0
            for r in range(min(12, len(grid))):
                rowv = grid[r]
                for col in range(min(6, len(rowv))):
                    v = int(rowv[col])
                    m = max(m, v)
                    cid = self._tac_cells.get((key, r, col))
                    if cid:
                        c.itemconfig(cid, fill=self._tac_color(v))
            mx[key] = m
        self._set(
            self.tac_lbl,
            text="tactile max  "
            + "  ".join(f"{k} {mx.get(k, 0):3d}" for k in FINGERS)
            + f"   finger reads {n}",
        )

    def _preview(self, path: str, active: bool) -> None:
        """Show the camera preview; decoded only when the file changes."""
        if not active or not path:
            if self._cam_photo is not None:
                self._cam_photo = None
                self._set(self.cam_img, image="", text="(camera preview while recording)")
            return
        try:
            m = os.path.getmtime(path)
            if m == self._cam_mtime:
                return
            self._cam_mtime = m
            import io

            from PIL import Image, ImageTk

            with open(path, "rb") as f:  # read at once, never a partially written file
                data = f.read()
            img = Image.open(io.BytesIO(data))
            img.load()
            self._cam_photo = ImageTk.PhotoImage(img)  # keep a reference or Tk shows nothing
            self.cam_img.config(image=self._cam_photo, text="")
        except Exception as exc:  # noqa: BLE001
            self._set(self.cam_img, image="", text=f"(preview not readable: {type(exc).__name__})")

    def stop_hands(self) -> None:
        for fl in self._hands.values():
            fl.stop()

    def _toggle_dev(self, key: str) -> None:
        d = self.ctx["devices"].get(key)
        if d is not None:
            self.act(
                f"{'deselect' if d.enabled else 'select'} {key}",
                lambda: d.configure(enabled=not d.enabled),
            )


def main() -> int:
    args = C.make_parser().parse_args()
    print("starting the backend ...", flush=True)
    ctx = C.build_ctx(args)
    root = tk.Tk()
    app = App(root, ctx)

    def on_close():
        app.stop_hands()  # opens the hands
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        print("shutting down ...", flush=True)
        C.shutdown_ctx(ctx)
        print("done.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
