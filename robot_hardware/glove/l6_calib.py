"""Calibration of the glove -> L6 mapping: which glove joints feed each L6 channel, and their range.

The glove reports joint angles as a (5, 5) array, one row per finger, in the joint order of
the Wuji SDK hand model:

    thumb (row 0)                         fingers (rows 1-4)
      0  cmc_rot   (Z)                      0  mcp_flex
      1  cmc_flex  (X)                      1  mcp_abd (spread, not flexion)
      2  cmc_abd   (Y)                      2  pip
      3  mcp       (X)                      3  dip
      4  ip        (X)                      4  unused (always 0)

The slots used by each channel are fixed by anatomy:

    thumb abduction  slot THUMB_ABD_SLOT (2)
    thumb flexion    slots 3 + 4 (MCP + IP)
    finger flexion   slots 0 + 2 + 3 (MCP flexion + PIP + DIP)

The human and robot thumbs bend at different joints, so there is no joint-to-joint mapping:
each channel maps "how far bent" (0 = extended, 1 = fully bent) linearly onto its L6 motor
(255 -> 0). A calibration stores the two ends of each channel's range in rad; the left glove
mirrors some axes, which the measured ends absorb.

Two ways to calibrate, producing the same `L6Calibration`:
    identify()   six guided poses of the human hand (open / fist, thumb out / in, thumb
                 straight / bent);
    fit_pairs()  pairs recorded in `teleop/hand_panel.py`: the robot hand is put into a pose
                 and the operator copies it with the glove.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

__all__ = [
    "L6Calibration",
    "DEFAULT_PATH",
    "default_path",
    "STEPS",
    "GROUPS",
    "identify",
    "fit_pairs",
    "THUMB_ABD_SLOT",
    "THUMB_FLEX_SLOTS",
    "FINGER_SLOTS",
    "THUMB_SLOT_NAMES",
]

CALIB_DIR = Path(__file__).resolve().parent / "calib"


def default_path(side: str) -> Path:
    """Calibration file of one hand: glove/calib/l6_calibration_<side>.json."""
    return CALIB_DIR / f"l6_calibration_{side}.json"


DEFAULT_PATH = default_path("right")

CH_NAMES = ["thumb flex", "thumb abd", "index", "middle", "ring", "little"]

THUMB_SLOT_NAMES = ["cmc_rot(Z)", "cmc_flex(X)", "cmc_abd(Y)", "mcp(X)", "ip(X)"]

THUMB_ABD_SLOT = 2
THUMB_FLEX_SLOTS = [3, 4]
FINGER_SLOTS = [0, 2, 3]


@dataclass
class L6Calibration:
    """Calibration of one glove; angles in rad."""

    finger_slots: list = field(default_factory=lambda: list(FINGER_SLOTS))
    thumb_flex_slots: list = field(default_factory=lambda: list(THUMB_FLEX_SLOTS))
    thumb_abd_slot: int = THUMB_ABD_SLOT

    # Range of the six channels in CAN order [thumb flex, thumb abd, index, middle, ring,
    # little], each [end mapped to 255 (extended / abducted), end mapped to 0]. Either sign.
    spans: list = field(default_factory=lambda: [[0.0, 1.0]] * 6)

    abd_positive: bool = True  # informative: larger value = more abducted

    invert: list = field(default_factory=lambda: [False] * 6)  # per-channel direction flip

    # Optional "OK" pinch anchor for thumb flexion: glove reading at the pinch (ok_raw) and
    # the L6 value at which the robot thumb touches the index finger (ok_l6). With an anchor
    # the thumb flexion mapping is piecewise linear (straight -> 255, OK -> ok_l6, bent -> 0).
    ok_raw: float | None = None
    ok_l6: int = 40

    hand: str = "right"
    created: str = ""
    note: str = ""

    def raw6(self, a: np.ndarray) -> list[float]:
        """The six channel readings (rad) from a (5, 5) joint angle array."""
        a = np.asarray(a, dtype=float).reshape(5, 5)

        def g(f, idxs):
            return float(sum(a[f, i] for i in idxs))

        return [
            g(0, self.thumb_flex_slots),
            g(0, [self.thumb_abd_slot]),
            g(1, self.finger_slots),
            g(2, self.finger_slots),
            g(3, self.finger_slots),
            g(4, self.finger_slots),
        ]

    def to_l6(self, a: np.ndarray) -> list[int]:
        """(5, 5) joint angles -> six L6 values 0-255 in CAN order (255 = extended)."""
        out = []
        inv = list(self.invert) + [False] * 6
        for i, x in enumerate(self.raw6(a)):
            lo, hi = self.spans[i]
            span = hi - lo
            n = 0.0 if abs(span) < 1e-9 else (x - lo) / span  # also valid when lo > hi
            v = float(np.clip(1.0 - np.clip(n, 0.0, 1.0), 0.0, 1.0))
            if inv[i]:
                v = 1.0 - v
            val = v * 255
            if i == 0 and self.ok_raw is not None:
                val = self._thumb_piecewise(x)
            out.append(int(round(float(np.clip(val, 0, 255)))))
        return out

    def _thumb_piecewise(self, x: float) -> float:
        """Thumb flexion through the OK anchor: straight -> 255, OK -> ok_l6, bent -> 0.

        The pinch point sits at different fractions of the human and robot ranges, so a
        single linear map would not make the robot thumb close where the human one does.
        """
        lo, hi = self.spans[0]
        ok = float(self.ok_raw)
        d = hi - lo
        if abs(d) < 1e-9:
            return 255.0
        t = (x - lo) / d  # 0 = straight, 1 = bent
        t_ok = (ok - lo) / d
        t_ok = min(max(t_ok, 1e-3), 1 - 1e-3)
        t = min(max(t, 0.0), 1.0)
        if t <= t_ok:
            return 255 + (self.ok_l6 - 255) * (t / t_ok)
        return self.ok_l6 * (1 - (t - t_ok) / (1 - t_ok))

    def norm6(self, a: np.ndarray) -> list[float]:
        """Normalised channels 0-1 (1 = fully bent), for debugging."""
        out = []
        for i, x in enumerate(self.raw6(a)):
            lo, hi = self.spans[i]
            span = hi - lo
            out.append(0.0 if abs(span) < 1e-9 else float(np.clip((x - lo) / span, 0.0, 1.0)))
        return out

    def save(self, path: Path | str = DEFAULT_PATH) -> Path:
        """Write the calibration as JSON."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")
        return p

    @staticmethod
    def load(path: Path | str = DEFAULT_PATH) -> "L6Calibration":
        """Read a calibration JSON."""
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return L6Calibration(**d)

    def describe(self) -> str:
        """Human-readable summary."""
        lines = [
            f"calibrated {self.created or '?'}   hand {self.hand}",
            f"  finger slots        {self.finger_slots}",
            f"  thumb flexion slots {self.thumb_flex_slots}",
            f"  thumb abduction     slot {self.thumb_abd_slot}"
            f"  ({'positive' if self.abd_positive else 'negative'})",
            "  ranges (deg): [end -> 255, end -> 0]",
        ]
        for i, (lo, hi) in enumerate(self.spans):
            lines.append(
                f"    {CH_NAMES[i]:<10} {np.degrees(lo):+7.1f} -> {np.degrees(hi):+7.1f}"
                f"   range {abs(np.degrees(hi - lo)):6.1f} deg"
            )
        inv = [CH_NAMES[i] for i, v in enumerate(list(self.invert) + [False] * 6) if v and i < 6]
        if inv:
            lines.append(f"  inverted channels: {', '.join(inv)}")
        if self.ok_raw is not None:
            lines.append(f"  OK anchor: {np.degrees(self.ok_raw):+.1f} deg -> L6 {self.ok_l6}")
        if self.note:
            lines.append(f"  note: {self.note}")
        return "\n".join(lines)


FINGER_ROW = {"index": 1, "middle": 2, "ring": 3, "pinky": 4}

# Guided calibration poses: (key, group, instruction). Each group is one channel pair; the
# first pose maps to 255 and the second to 0. Each pair reads only its own slots, so the
# other degrees of freedom may be anywhere.
STEPS = [
    ("fingers_open", "fingers", "1. Open hand: all fingers fully extended"),
    ("fingers_fist", "fingers", "2. Fist: fingers fully closed (thumb anywhere)"),
    ("thumb_out", "thumb abduction", "3. Thumb out: fingers in one plane, thumb away from palm"),
    ("thumb_in", "thumb abduction", "4. Thumb in: fingers in one plane, thumb against the palm"),
    ("thumb_straight", "thumb flexion", "5. Thumb straight: thumb fully extended"),
    ("thumb_bent", "thumb flexion", "6. Thumb bent: thumb fully flexed"),
]

GROUPS = [
    ("fingers", ["fingers_open", "fingers_fist"]),
    ("thumb abduction", ["thumb_out", "thumb_in"]),
    ("thumb flexion", ["thumb_straight", "thumb_bent"]),
]


def identify(poses: dict, hand: str = "right") -> tuple:
    """Calibration from the six guided poses. Returns (L6Calibration, info lines, warnings).

    `poses` maps each key of STEPS (and optionally "thumb_ok") to a (5, 5) joint angle
    array. The info lines list how much each thumb slot changed between the two poses of a
    pair, so the fixed slot assignment can be checked.
    """
    from datetime import datetime

    def deg(x):
        return float(np.degrees(x))

    cal = L6Calibration(hand=hand, created=datetime.now().strftime("%Y-%m-%d %H:%M"))
    info, warn = [], []

    def take(pose, f, idxs):
        return float(sum(poses[pose][f, i] for i in idxs))

    def thumb_delta_line(a, b):
        """Change of the five thumb slots between two poses (deg); used slots marked *."""
        d = np.degrees(poses[b][0] - poses[a][0])

        def used(i):
            return (i == cal.thumb_abd_slot and a == "thumb_out") or (
                i in cal.thumb_flex_slots and a == "thumb_straight"
            )

        return "  ".join(
            f"{'*' if used(i) else ' '}slot{i} {THUMB_SLOT_NAMES[i]}{d[i]:+6.1f}" for i in range(5)
        )

    # Fingers: open -> fist on slots 0 + 2 + 3.
    info.append(f"finger slots = {cal.finger_slots} (MCP flexion + PIP + DIP)")
    d_fist = np.degrees(poses["fingers_fist"] - poses["fingers_open"])
    for name, row in FINGER_ROW.items():
        info.append(
            f"  {name:6s} open -> fist per slot "
            + "  ".join(
                f"{'*' if i in cal.finger_slots else ' '}slot{i}{d_fist[row, i]:+6.1f}"
                for i in range(4)
            )
        )

    # Thumb abduction: slot THUMB_ABD_SLOT only.
    abd_out = take("thumb_out", 0, [cal.thumb_abd_slot])
    abd_in = take("thumb_in", 0, [cal.thumb_abd_slot])
    cal.abd_positive = bool(abd_out > abd_in)
    info.append(
        f"thumb abduction = slot {cal.thumb_abd_slot} ({THUMB_SLOT_NAMES[cal.thumb_abd_slot]})"
        f"  out {deg(abd_out):+.1f} -> in {deg(abd_in):+.1f} deg"
        f"  range {abs(deg(abd_out - abd_in)):.1f} deg"
    )
    info.append("  out -> in " + thumb_delta_line("thumb_out", "thumb_in"))
    d_abd_all = np.abs(np.degrees(poses["thumb_in"][0] - poses["thumb_out"][0]))
    if d_abd_all.argmax() != cal.thumb_abd_slot:
        j = int(d_abd_all.argmax())
        warn.append(
            f"in the abduction pair slot {j} ({THUMB_SLOT_NAMES[j]}) changed most "
            f"({d_abd_all[j]:.0f} deg), not slot {cal.thumb_abd_slot} "
            f"({d_abd_all[cal.thumb_abd_slot]:.0f} deg): keep the fingers in one plane"
        )

    # Thumb flexion: slots 3 + 4.
    fl_straight = take("thumb_straight", 0, cal.thumb_flex_slots)
    fl_bent = take("thumb_bent", 0, cal.thumb_flex_slots)
    info.append(
        f"thumb flexion = slots {cal.thumb_flex_slots} (MCP + IP)"
        f"  straight {deg(fl_straight):+.1f} -> bent {deg(fl_bent):+.1f} deg"
        f"  range {abs(deg(fl_bent - fl_straight)):.1f} deg"
    )
    info.append("  straight -> bent " + thumb_delta_line("thumb_straight", "thumb_bent"))

    # Ranges in CAN order, each [end -> 255, end -> 0].
    spans = [[fl_straight, fl_bent], [abd_out, abd_in]]
    for _name, row in FINGER_ROW.items():
        spans.append(
            [
                take("fingers_open", row, cal.finger_slots),
                take("fingers_fist", row, cal.finger_slots),
            ]
        )
    cal.spans = spans

    # Optional OK pinch anchor for thumb flexion.
    if "thumb_ok" in poses:
        ok = take("thumb_ok", 0, cal.thumb_flex_slots)
        lo, hi = spans[0]
        t = (ok - lo) / (hi - lo) if abs(hi - lo) > 1e-9 else -1.0
        if 0.05 <= t <= 0.95:
            cal.ok_raw = ok
            info.append(f"OK anchor at {t*100:.0f}% of the thumb flexion range -> L6 {cal.ok_l6}")
        else:
            warn.append(f"OK pose at {t*100:.0f}% of the range (needs 5-95%); not used")

    for i, (lo, hi) in enumerate(spans):
        if abs(hi - lo) < np.deg2rad(12):
            warn.append(
                f"{CH_NAMES[i]} range only {abs(deg(hi - lo)):.1f} deg (< 12): pose not reached"
            )
    if abs(spans[0][1] - spans[0][0]) < np.deg2rad(50):
        warn.append(
            f"thumb flexion range {abs(deg(spans[0][1] - spans[0][0])):.0f} deg is small: "
            "bend the thumb fully from the base"
        )
    return cal, info, warn


def fit_pairs(pairs: list, hand: str, base: "L6Calibration | None" = None) -> tuple:
    """Calibration from (glove, robot hand) pose pairs. Returns (L6Calibration, info, warnings).

    Each pair is {"ja": 25 glove joint angles (rad), "l6": six L6 values}. For each channel
    the ends of the range are the median glove readings over the pairs where that motor is
    at 255 and at 0 (for the four fingers, preferably the pairs with all four at 255 / 0).
    Without both ends the channel falls back to a linear least-squares fit
    L6 / 255 = a * x + b over all pairs. Channels that cannot be fitted keep the range of
    `base` (or the default).
    """
    from datetime import datetime

    cal = L6Calibration(hand=hand, created=datetime.now().strftime("%Y-%m-%d %H:%M"))
    if base is not None:
        cal.spans = [list(s) for s in base.spans]
        cal.invert = list(base.invert)
    cal.ok_raw = None
    info, warn = [], []
    if len(pairs) < 2:
        warn.append(f"only {len(pairs)} pairs; at least 2 are needed")
        return cal, info, warn
    X = np.array([cal.raw6(np.asarray(p["ja"], float)) for p in pairs])  # (n, 6) rad
    Y = np.array([p["l6"] for p in pairs], float) / 255.0  # (n, 6) 0-1
    info.append(f"{len(pairs)} pairs")
    # Pairs where the glove thumb MCP and IP both sit at their +-15 deg limit carry no
    # thumb flexion information.
    th = np.degrees(np.array([np.asarray(p["ja"], float).reshape(5, 5)[0] for p in pairs]))
    pinned = np.sum((np.abs(np.abs(th[:, 3]) - 15) < 0.3) & (np.abs(np.abs(th[:, 4]) - 15) < 0.3))
    if pinned >= max(2, len(pairs) // 2):
        warn.append(
            f"in {pinned}/{len(pairs)} pairs the glove thumb MCP and IP are both at their "
            "+-15 deg limit: the glove is not reading thumb flexion"
        )
    for i in range(6):
        x, y = X[:, i], Y[:, i]
        all_open = np.all(Y[:, 2:] > 0.999, axis=1)
        all_fist = np.all(Y[:, 2:] < 0.001, axis=1)
        if i >= 2 and all_open.any() and all_fist.any():
            at255, at0 = x[all_open], x[all_fist]
        else:
            at255, at0 = x[y > 0.999], x[y < 0.001]
        if len(at255) and len(at0):
            lo, hi = float(np.median(at255)), float(np.median(at0))
            if abs(hi - lo) < np.deg2rad(3):
                warn.append(
                    f"{CH_NAMES[i]}: glove readings at 255 and 0 differ by only "
                    f"{np.degrees(abs(hi-lo)):.1f} deg; range kept"
                )
                continue
            cal.spans[i] = [lo, hi]
            sp255 = np.degrees(np.ptp(at255)) if len(at255) > 1 else 0.0
            sp0 = np.degrees(np.ptp(at0)) if len(at0) > 1 else 0.0
            info.append(
                f"{CH_NAMES[i]:<10} {np.degrees(lo):+7.1f} -> {np.degrees(hi):+7.1f} deg"
                f" (range {abs(np.degrees(hi-lo)):5.1f})  at 255: {len(at255)} pairs"
                f" (spread {sp255:.0f})  at 0: {len(at0)} pairs (spread {sp0:.0f})"
            )
            if max(sp255, sp0) > 30:
                warn.append(
                    f"{CH_NAMES[i]}: glove readings for the same robot pose differ by "
                    f"{max(sp255, sp0):.0f} deg; some pairs are inconsistent"
                )
            continue
        warn.append(
            f"{CH_NAMES[i]}: no pairs at both 255 and 0 ({len(at255)} / {len(at0)}); "
            "using a least-squares fit over all pairs"
        )
        if np.ptp(y) < 1e-6:
            warn.append(f"{CH_NAMES[i]}: L6 value constant ({int(y[0]*255)}); range kept")
            continue
        if np.ptp(x) < np.deg2rad(3):
            warn.append(
                f"{CH_NAMES[i]}: glove reading changed only {np.degrees(np.ptp(x)):.1f} deg; "
                "range kept"
            )
            continue
        a, b = np.polyfit(x, y, 1)
        if abs(a) < 1e-9:
            warn.append(f"{CH_NAMES[i]}: no relation between glove and L6 values; range kept")
            continue
        lo, hi = (1.0 - b) / a, -b / a  # ends at L6 = 255 and L6 = 0
        cal.spans[i] = [float(lo), float(hi)]
        r = np.abs((a * x + b) - y) * 255
        info.append(
            f"{CH_NAMES[i]:<10} {np.degrees(lo):+7.1f} -> {np.degrees(hi):+7.1f} deg"
            f" (range {abs(np.degrees(hi-lo)):5.1f})  residual mean {r.mean():4.1f}"
            f" / max {r.max():4.1f} (L6 units)"
        )
        if r.max() > 40:
            warn.append(f"{CH_NAMES[i]}: max residual {r.max():.0f} (> 40); a pair does not match")
        seen_lo, seen_hi = float(x.min()), float(x.max())
        if min(lo, hi) < seen_lo - np.deg2rad(15) or max(lo, hi) > seen_hi + np.deg2rad(15):
            warn.append(
                f"{CH_NAMES[i]}: fitted ends extrapolate more than 15 deg beyond the readings"
            )
    return cal, info, warn
