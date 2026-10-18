"""The test metric: the evaluation plan and the error function.

A copy of the corresponding parts of the main model's `T-WLA_foundation_model/eval_test.py`, so
that the baseline is scored the same way. `GROUPS` slices the keypoints action layout.
"""

from __future__ import annotations

import collections

import numpy as np

from dataloader.sampling import phase_anchors, supervisable_anchors

MAX_DEPTH = 8
MIN_ANCHOR = MAX_DEPTH          # the earliest anchor with MAX_DEPTH steps of history
ANCHORS_PER_UNIT = 4

# (name, left-hand slice, right-hand slice, scale to the reported unit, unit).
# Wrist slots are increments per action step (1/80 s) in metres and radians; finger slots
# are absolute keypoints in metres.
GROUPS = [
    ("wrist_translation", slice(0, 3), slice(69, 72), 100.0, "cm"),
    ("wrist_rotation", slice(3, 6), slice(72, 75), None, "deg"),
    ("fingers", slice(6, 69), slice(75, 138), 100.0, "cm"),
]

def build_plan(ds) -> list[tuple[int, int, int, int]]:
    """One entry per (unit, anchor, depth): (unit index, window start, length, pair seed).

    All depths of one (unit, anchor) share the pair seed, from which the starting noise and
    the paraphrase are drawn, so the depths are compared on paired samples.
    """
    plan, pairs = [], []
    for ui, u in enumerate(ds.units):
        n_sup = supervisable_anchors(phase_anchors(u.n_frames, u.phase, ds.stride),
                                     u.n_frames)
        lo, hi = MIN_ANCHOR, n_sup - 1
        if hi < lo:
            continue
        n = hi - lo + 1
        picks = (list(range(lo, hi + 1)) if n <= ANCHORS_PER_UNIT else
                 sorted({int(round(x)) for x in
                         np.linspace(lo, hi, ANCHORS_PER_UNIT)}))
        for a in picks:
            pairs.append((ui, a))
    # ordered by depth, so that a batch can hold a single depth
    for d in range(MAX_DEPTH + 1):
        for pid, (ui, a) in enumerate(pairs):
            plan.append((ui, a - d, d + 1, pid))
    return plan


def limit_plan_per_source(plan: list[tuple[int, int, int, int]],
                          pair_source: dict[int, str],
                          limit: int) -> list[tuple[int, int, int, int]]:
    """Keep at most ``limit`` paired anchors from every source, including all depths."""
    if limit <= 0:
        return plan
    keep, seen = set(), collections.Counter()
    for pid in sorted({x[3] for x in plan}):
        source = pair_source[pid]
        if seen[source] < limit:
            keep.add(pid)
            seen[source] += 1
    return [x for x in plan if x[3] in keep]


def stride_plan_per_source(plan: list[tuple[int, int, int, int]],
                           pair_source: dict[int, str],
                           stride: int) -> list[tuple[int, int, int, int]]:
    """Take every Nth pair within each source instead of striding one global prefix."""
    if stride <= 1:
        return plan
    by_source: dict[str, list[int]] = collections.defaultdict(list)
    for pid in sorted({x[3] for x in plan}):
        by_source[pair_source[pid]].append(pid)
    keep = {pid for pids in by_source.values() for pid in pids[::stride]}
    return [x for x in plan if x[3] in keep]



VIEWS = ("all", "8", "16")

def errors(pred: np.ndarray, true: np.ndarray, mask: np.ndarray,
           view: str) -> dict[str, float] | None:
    """Error per group in physical units for one view; None when the sample has fewer valid
    steps than the view needs.

    `pred` and `true` are [T, 192], denormalised; `view` is "all" or a number of steps.
    Wrist translation: distance between the accumulated positions (the slots are per-step
    increments). Wrist rotation: geodesic angle between the accumulated rotations.
    Fingers: mean distance over keypoints 1..20.
    """
    from scipy.spatial.transform import Rotation as R
    horizon = pred.shape[0] if view == "all" else int(view)
    out = {}
    for name, ls, rs, scale, _unit in GROUPS:
        vals, n = 0.0, 0
        for sl in (ls, rs):
            m = mask[:horizon, sl]
            if not m.any():
                continue                       # this hand is absent from the episode
            rows = m.all(1)
            if view != "all" and not rows.all():
                return None                    # the chunk is shorter than this view wants
            if not rows.any():
                continue
            p, t = pred[:horizon, sl], true[:horizon, sl]
            if name == "wrist_translation":
                e = np.linalg.norm(np.cumsum(p, 0) - np.cumsum(t, 0), axis=1) * scale
            elif name == "wrist_rotation":
                ep = R.from_euler("xyz", p); et = R.from_euler("xyz", t)
                acc_p = acc_t = R.identity()
                e = np.empty(p.shape[0])
                for k in range(p.shape[0]):
                    acc_p = ep[k] * acc_p
                    acc_t = et[k] * acc_t
                    e[k] = np.degrees((acc_p * acc_t.inv()).magnitude())
            else:
                # Keypoint 0 is the origin of the MANO palm-local frame, zero in every
                # sample, and is left out.
                d = (p - t).reshape(p.shape[0], -1, 3)[:, 1:, :]
                e = np.linalg.norm(d, axis=2).mean(1) * scale
            vals += float(e[rows].sum()); n += int(rows.sum())
        out[name] = vals / n if n else float("nan")
    return out


