"""Normalisation statistics for the padded action (192) and state (212) vectors.

Both vectors hold [ left 69 | right 69 | padding ]. Padding slots keep mean 0 / std 1: they
carry no data and are masked, and a std of 0 would divide by zero. The model stores the
statistics as buffers, so a checkpoint carries its own.
"""

from __future__ import annotations

import hashlib
import json
import os

import numpy as np

SCHEMA_VERSION = 2         # bump when the meaning of a field changes; older files are refused

# Field of the file's `convention` block -> (config section, field) it must equal.
CONVENTION_FIELDS = {
    "action_hz": ("action", "action_hz"),
    "control_hz": ("action", "control_hz"),
    "source_fps": ("action", "source_fps"),
    "chunk_size": ("action", "chunk_size"),
}

PER_HAND_ACTION = 69       # 3 translation + 3 euler rotation + 21*3 keypoints
PER_HAND_STATE = 79        # the same 69, plus 10 MANO beta
BETA = 10
ACTION_DIM = 192
STATE_DIM = 212


def split_digest(path: str) -> str:
    """SHA-256 over the sorted set of names in a split file, so that neither the order nor
    the formatting of the file matters."""
    with open(path) as f:
        names = sorted({ln.strip() for ln in f if ln.strip()})
    return hashlib.sha256("\n".join(names).encode()).hexdigest()


def load_statistics(path: str) -> dict[str, np.ndarray]:
    """Read a VITRA statistics json (per hand: 69-D action, 79-D state) into the padded layout.

    Returns float32 arrays: action_mean/std [192], state_mean/std [212].
    """
    with open(path) as f:
        raw = json.load(f)

    a_mean = np.zeros(ACTION_DIM, np.float32)
    a_std = np.ones(ACTION_DIM, np.float32)
    s_mean = np.zeros(STATE_DIM, np.float32)
    s_std = np.ones(STATE_DIM, np.float32)

    for hi, hand in enumerate(("left", "right")):
        act = raw.get(f"action_{hand}")
        st = raw.get(f"state_{hand}")
        if act is None or st is None:
            raise KeyError(f"{path} has no action_{hand}/state_{hand}")

        am = np.asarray(act["mean"], np.float32)
        asd = np.asarray(act["std"], np.float32)
        sm = np.asarray(st["mean"], np.float32)
        ssd = np.asarray(st["std"], np.float32)
        if am.shape != (PER_HAND_ACTION,) or sm.shape != (PER_HAND_STATE,):
            raise ValueError(
                f"{path}: expected action {PER_HAND_ACTION}-D and state {PER_HAND_STATE}-D "
                f"per hand, got {am.shape} and {sm.shape}")

        lo = hi * PER_HAND_ACTION
        a_mean[lo:lo + PER_HAND_ACTION] = am
        a_std[lo:lo + PER_HAND_ACTION] = asd

        # The state shares the 69 pose dims. Its trailing 10 (MANO beta) are not used.
        s_mean[lo:lo + PER_HAND_ACTION] = sm[:PER_HAND_ACTION]
        s_std[lo:lo + PER_HAND_ACTION] = ssd[:PER_HAND_ACTION]

    return {"action_mean": a_mean, "action_std": a_std,
            "state_mean": s_mean, "state_std": s_std}


def load_v4_statistics(path: str) -> dict[str, np.ndarray]:
    """Load the normalisation statistics file, already in the 192 / 212 layout.

    The file's `convention` and `mixture` blocks are returned under `_convention` and
    `_mixture` for `check_convention` and `check_sources`.
    """
    with open(path) as f:
        raw = json.load(f)
    got = raw.get("schema_version")
    if got != SCHEMA_VERSION:
        raise ValueError(
            f"{path} has schema_version {got!r}, this build writes and reads "
            f"{SCHEMA_VERSION}. Recompute the statistics -- an older file may carry fields "
            f"whose meaning has since changed, which loads without complaint.")
    conv = raw.get("convention", {})
    out = {"action_mean": np.asarray(raw["action"]["mean"], np.float32),
           "action_std": np.asarray(raw["action"]["std"], np.float32),
           "state_mean": np.asarray(raw["state"]["mean"], np.float32),
           "state_std": np.asarray(raw["state"]["std"], np.float32)}
    for k, v in out.items():
        n = ACTION_DIM if k.startswith("action") else STATE_DIM
        if v.shape != (n,):
            raise ValueError(f"{path}: {k} should be {n}-D, got {v.shape}")
    out["_convention"] = conv
    out["_mixture"] = raw.get("mixture")
    return out


def check_convention(stats: dict, cfg) -> None:
    """Raise if the statistics were measured under a different action timebase than `cfg`.

    Wrist channels are per-step increments, so their scale depends on `action_hz`. A file
    without a `convention` block is rejected.
    """
    conv = stats.get("_convention") or {}
    if not conv:
        raise ValueError(
            "the statistics file has no `convention` block, so there is nothing to check "
            "it against. Recompute the statistics; this build records the timebase it "
            "measured at.")
    bad = []
    for key, (section, field) in CONVENTION_FIELDS.items():
        if key not in conv:
            bad.append(f"{key}: missing from the file")
            continue
        want = getattr(getattr(cfg, section), field)
        if abs(float(conv[key]) - float(want)) > 1e-6:
            bad.append(f"{key}: file {conv[key]} vs config {want}")
    if conv.get("action_type") != "keypoints":
        bad.append(f"action_type: file {conv.get('action_type')!r} vs 'keypoints'")
    if bad:
        raise ValueError(
            "statistics were measured under a different convention than this run uses:\n  "
            + "\n  ".join(bad)
            + "\nThe wrist channels are increments -- their sigma does not transfer "
              "between rates. Recompute the statistics.")


def check_sources(stats: dict, sources: list[dict]) -> None:
    """Raise if the statistics cover a different mixture than `sources` (the resolved `data.sources`).

    Names, weights and train-split digests must match. A source may set `statistics_split_dir`
    to name the split its statistics were measured on when that is not the split it trains
    on; the digest is then checked against that split.
    """
    mix = stats.get("_mixture")
    if mix is None:
        raise ValueError(
            "the statistics file records no `mixture`, so it cannot be matched to this "
            "run's sources. Recompute the statistics for this config.")
    want, borrowed = {}, []
    for s in sources:
        d = s.get("statistics_split_dir") or s["split_dir"]
        if s.get("statistics_split_dir"):
            borrowed.append(f"{s['name']}: statistics from {d}, training on {s['split_dir']}")
        want[s["name"]] = (float(s.get("weight", 1.0)),
                           split_digest(os.path.join(d, "train.txt")))
    if borrowed:
        print("  [stats] deliberately foreign normalisation:\n    "
              + "\n    ".join(borrowed), flush=True)
    have = {m["name"]: (float(m["weight"]), m["split_sha256"]) for m in mix}

    bad = []
    for name in sorted(set(want) | set(have)):
        if name not in have:
            bad.append(f"{name}: in the config, not in the statistics")
        elif name not in want:
            bad.append(f"{name}: in the statistics, not in the config")
        else:
            w_w, w_h = want[name], have[name]
            if abs(w_w[0] - w_h[0]) > 1e-9:
                bad.append(f"{name}: weight {w_h[0]} in the statistics, {w_w[0]} here")
            if w_w[1] != w_h[1]:
                bad.append(f"{name}: train split differs "
                           f"({w_h[1][:12]} measured, {w_w[1][:12]} here). If this run "
                           f"deliberately borrows another split's normalisation, name "
                           f"that split in the source's `statistics_split_dir`")
    if bad:
        raise ValueError("the statistics do not describe this run's data:\n  "
                         + "\n  ".join(bad)
                         + "\nRecompute the statistics for this config.")


def report(stats: dict[str, np.ndarray]) -> str:
    """Per-group summary of the action statistics, meant to be printed once per run."""
    groups = [("left_translation", 0, 3), ("left_rotation", 3, 6),
              ("left_articulation", 6, 69), ("right_translation", 69, 72),
              ("right_rotation", 72, 75), ("right_articulation", 75, 138)]
    out = [f"{'group':<20} {'|mean|':>10} {'std':>10} {'std min':>10} {'std max':>10}"]
    for name, lo, hi in groups:
        m = stats["action_mean"][lo:hi]
        s = stats["action_std"][lo:hi]
        out.append(f"{name:<20} {np.abs(m).mean():>10.6f} {s.mean():>10.6f} "
                   f"{s.min():>10.6f} {s.max():>10.6f}")
    dead = int((stats["action_std"][:138] < 1e-6).sum())
    out.append(f"action std < 1e-6 in the real 138 dims: {dead}")
    return "\n".join(out)
