#!/usr/bin/env python3
"""Hand-eye calibration with joint zero offsets, for one or both arms.

A single nonlinear least-squares problem estimates
    Z          T_cam_base, shared by all arms
    X_side     T_board_flange, one per arm (unknown, fixed within a batch)
    d_side     joint zero offsets in rad, q_true = q_meas + d
and optionally
    --scale    the board square size; the arm gives absolute scale, so this absorbs the
               size error of a printed board (recommended)
    --mount    a translation of the left shoulder base relative to the URDF
    --intr     camera intrinsics and distortion (rarely needed with factory intrinsics)

The residual is the reprojection error in pixels of every detected ChArUco corner:
    pixel = project(Z * FK(q + d) * inv(X) * P_board)

Not all offsets are observable. An offset of J7 rotates about the flange axis and is
equivalent to a change of X, so J7 is never solved. The J1 axes of the two arms are
collinear, so rotating both by the same angle is equivalent to rotating the camera;
J1 of the first arm on the command line is therefore fixed to 0, and J1 of the other
arm is solved relative to it.

Validation:
    --holdout          solve on even samples, evaluate on odd ones (interpolation)
    --holdout-by J4    split at the median of one joint angle, solve on the lower half and
                       evaluate on the upper half (extrapolation; the stricter check)

Zero offsets plus board scale are usually sufficient. Adding more kinematic parameters
lowers the training residual but tends to overfit unless the poses cover a wide range.

Usage (run `handeye_solve.py` on every directory first; its result.json is the initial guess):
    python tools/handeye_joint_solve.py calib/handeye_<right> [calib/handeye_<left>]
    python tools/handeye_joint_solve.py calib/handeye_<right> --scale --holdout-by J4
    python tools/handeye_joint_solve.py calib/handeye_<right> --no-offsets   # Z and X only
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pinocchio as pin
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim_viewer import URDF, SDK_JOINTS, PREFIX  # noqa: E402

SIDES = ("right", "left")


def load_batch(d):
    """Load the usable samples of one directory.

    Returns (side, samples, intrinsics, nominal square size). Each sample holds the joint
    angles, the detected corners in pixels and the corner positions on the board.
    """
    rows = [json.loads(l) for l in open(os.path.join(d, "samples.jsonl"), encoding="utf-8")]
    rows = [r for r in rows if r.get("usable")]
    side = rows[0]["arm"]
    b = rows[0]["board"]
    s = b["square_length_m"]
    nx = b["squares_x"] - 1
    samples = []
    for r in rows:
        ids = r["ids"]
        # Corner positions on the board, in units of squares.
        G = np.array([[(i % nx + 1), (i // nx + 1), 0.0] for i in ids], float)
        samples.append(
            {
                "q": np.array(r["q_motor_rad"], float),
                "G": G,
                "P": G * s,
                "uv": np.array(r["corners_px"], float),
                "idx": r["idx"],
                "side": side,
            }
        )
    intr = rows[0]["intrinsics"]
    return side, samples, intr, s


class Robot:
    """Forward kinematics of the dual-arm model."""

    def __init__(self):
        self.model = pin.buildModelFromUrdf(URDF)
        self.data = self.model.createData()
        self.jid = {
            s: [self.model.getJointId(PREFIX[s] + n + "_Joint") for n in SDK_JOINTS] for s in SIDES
        }
        self.qidx = {s: [self.model.joints[i].idx_q for i in self.jid[s]] for s in SIDES}
        # Shoulder base frames, used by --mount.
        self.base_frame = {
            s: self.model.getFrameId(PREFIX[s] + "Shoulder_Base_Joint") for s in SIDES
        }
        self._q = pin.neutral(self.model)

    def flange(self, side, q7, mount_dt=None):
        """Return T_base_flange, the pose of the wrist roll joint frame of one arm."""
        q = self._q
        q[:] = 0.0
        q[self.qidx[side]] = q7
        pin.forwardKinematics(self.model, self.data, q)
        T = self.data.oMi[self.jid[side][-1]].homogeneous.copy()
        if mount_dt is not None:
            T[:3, 3] += mount_dt
        return T


def se3(p6):
    """(rotvec, translation) -> 4x4 transform."""
    T = np.eye(4)
    T[:3, :3] = Rot.from_rotvec(p6[:3]).as_matrix()
    T[:3, 3] = p6[3:]
    return T


def se3_inv(T):
    """Inverse of a 4x4 rigid transform."""
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


def to_p6(T):
    """4x4 transform -> (rotvec, translation)."""
    return np.concatenate([Rot.from_matrix(T[:3, :3]).as_rotvec(), T[:3, 3]])


def project(Pc, intr):
    """Project camera-frame points to pixels (pinhole model with k1, k2, p1, p2)."""
    fx, fy, cx, cy, k1, k2, p1, p2 = intr
    x = Pc[:, 0] / Pc[:, 2]
    y = Pc[:, 1] / Pc[:, 2]
    r2 = x * x + y * y
    rad = 1 + k1 * r2 + k2 * r2 * r2
    xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return np.stack([fx * xd + cx, fy * yd + cy], axis=1)


class Params:
    """Layout of the parameter vector: named slices for each group of unknowns."""

    def __init__(self, sides, offsets=True, mount=False, intr=False, scale=False):
        self.sides, self.offsets, self.mount, self.intr, self.scale = (
            list(sides),
            offsets,
            mount,
            intr,
            scale,
        )
        self.slots = {"Z": (0, 6)}
        n = 6
        for s in self.sides:
            self.slots["X_" + s] = (n, n + 6)
            n += 6
        # Offsets solved per arm: J2..J6 always, J1 only for arms after the first, J7 never.
        self.dj = {}
        if offsets:
            for i, s in enumerate(self.sides):
                js = ([0] if i > 0 else []) + [1, 2, 3, 4, 5]
                self.dj[s] = js
                self.slots["d_" + s] = (n, n + len(js))
                n += len(js)
        if mount and "left" in self.sides:
            self.slots["mount_left"] = (n, n + 3)
            n += 3
        if scale:
            self.slots["scale"] = (n, n + 1)
            n += 1
        if intr:
            self.slots["intr"] = (n, n + 8)
            n += 8
        self.n = n

    def get(self, x, k, default=None):
        if k not in self.slots:
            return default
        a, b = self.slots[k]
        return x[a:b]

    def offsets7(self, x, side):
        """Zero offsets of one arm as 7 values; joints that are not solved are 0."""
        d = np.zeros(7)
        if "d_" + side in self.slots:
            d[self.dj[side]] = self.get(x, "d_" + side)
        return d


def residuals(x, P, rob, batches, intr0, s0=None):
    """Reprojection residuals in pixels of all corners, as a flat vector."""
    Z = se3(P.get(x, "Z"))
    intr = P.get(x, "intr", intr0)
    sc = P.get(x, "scale", [None])[0]
    out = []
    for side, samples in batches:
        Xi = se3_inv(se3(P.get(x, "X_" + side)))
        d = P.offsets7(x, side)
        mdt = P.get(x, "mount_left") if side == "left" else None
        for smp in samples:
            T = Z @ rob.flange(side, smp["q"] + d, mdt) @ Xi  # board pose in the camera
            Pb = smp["P"] if sc is None else smp["G"] * sc
            Pc = Pb @ T[:3, :3].T + T[:3, 3]
            out.append((project(Pc, intr) - smp["uv"]).ravel())
    return np.concatenate(out)


def per_sample_err(x, P, rob, batches, intr0):
    """Per-sample RMS reprojection error as a list of (side, sample index, rms in px)."""
    r = residuals(x, P, rob, batches, intr0)
    out, k = [], 0
    for side, samples in batches:
        for smp in samples:
            n = len(smp["uv"])
            e = r[k : k + 2 * n].reshape(-1, 2)
            k += 2 * n
            out.append((side, smp["idx"], float(np.sqrt((e**2).sum(1).mean()))))
    return out


def init_x(P, rob, batches, intr0, prior):
    """Initial guess from the closed-form results: Z from the prior, X from each batch."""
    x = np.zeros(P.n)
    a, b = P.slots["Z"]
    x[a:b] = to_p6(se3_inv(np.array(prior["T_base_cam"])))
    for side, samples in batches:
        a, b = P.slots["X_" + side]
        x[a:b] = to_p6(se3_inv(np.array(samples[0]["_result"]["T_flange_board"])))
    if P.scale:
        x[P.slots["scale"][0]] = prior.get("_s0", 0.028)
    if P.intr:
        a, b = P.slots["intr"]
        x[a:b] = intr0
    return x


def solve(P, rob, batches, intr0, prior):
    """Run the least-squares solve and return the scipy result."""
    x0 = init_x(P, rob, batches, intr0, prior)
    sol = least_squares(
        residuals,
        x0,
        args=(P, rob, batches, intr0),
        method="trf",
        x_scale="jac",
        max_nfev=200,
        verbose=0,
    )
    return sol


def summarize(tag, sol, P, rob, batches, intr0, holdout=None):
    """Print the solution and its residuals. Returns T_base_cam."""
    r = sol.fun
    rms = float(np.sqrt(np.mean(r**2)))
    errs = per_sample_err(sol.x, P, rob, batches, intr0)
    Z = se3(P.get(sol.x, "Z"))
    Tbc = se3_inv(Z)
    z = Tbc[:3, :3] @ [0, 0, 1]
    print(
        f"\n=== {tag}: {P.n} parameters, {len(r)} residuals, RMS {rms:.3f} px "
        f"(per sample: median {np.median([e[2] for e in errs]):.2f}, "
        f"max {max(e[2] for e in errs):.2f} px)"
    )
    print(
        f"  camera @ base ({Tbc[0,3]:+.4f}, {Tbc[1,3]:+.4f}, {Tbc[2,3]:+.4f}) m  "
        f"pitch down {np.rad2deg(np.arcsin(-z[2])):.1f} deg  "
        f"yaw {np.rad2deg(np.arctan2(z[1], z[0])):+.1f} deg"
    )
    for side, _ in batches:
        if "d_" + side in P.slots:
            d = P.offsets7(sol.x, side)
            print(
                f"  {side:5s} zero offsets (deg): "
                + " ".join(f"J{i+1}{np.rad2deg(v):+6.2f}" for i, v in enumerate(d))
                + (
                    "   (J1 fixed: reference arm)"
                    if 0 not in P.dj[side]
                    else "   (J1 relative to the reference arm)"
                )
                + "  J7 not solved"
            )
    m = P.get(sol.x, "mount_left")
    if m is not None:
        print(
            f"  left shoulder base offset from the URDF "
            f"({m[0]*1000:+.1f}, {m[1]*1000:+.1f}, {m[2]*1000:+.1f}) mm, "
            f"J1 spacing {192 + m[1]*1000:.0f} mm (URDF 192)"
        )
    sc = P.get(sol.x, "scale")
    if sc is not None:
        print(
            f"  board square size {sc[0]*1000:.3f} mm "
            f"(nominal {P.s0*1000:.1f}, ratio {sc[0]/P.s0:.4f})"
        )
    it = P.get(sol.x, "intr")
    if it is not None:
        print(
            f"  intrinsics fx {it[0]:.1f} fy {it[1]:.1f} cx {it[2]:.1f} cy {it[3]:.1f}  "
            f"k1 {it[4]:+.4f} k2 {it[5]:+.4f} p1 {it[6]:+.5f} p2 {it[7]:+.5f}"
        )
    for side, _ in batches:
        es = [e[2] for e in errs if e[0] == side]
        print(f"  {side:5s} per-sample RMS px: " + " ".join(f"{v:.1f}" for v in es))
    if holdout is not None:
        r2 = residuals(sol.x, P, rob, holdout, intr0)
        e2 = per_sample_err(sol.x, P, rob, holdout, intr0)
        print(
            f"  held-out samples: RMS {np.sqrt(np.mean(r2**2)):.3f} px "
            f"(per sample: median {np.median([e[2] for e in e2]):.2f}, "
            f"max {max(e[2] for e in e2):.2f} px)"
        )
    return Tbc


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("dirs", nargs="+", help="handeye_* directories, one arm per directory")
    ap.add_argument("--no-offsets", action="store_true", help="solve Z and X only")
    ap.add_argument("--mount", action="store_true", help="also solve the left shoulder offset")
    ap.add_argument("--intr", action="store_true", help="also solve the camera intrinsics")
    ap.add_argument(
        "--scale",
        action="store_true",
        help="also solve the board square size (recommended)",
    )
    ap.add_argument(
        "--holdout-by",
        default=None,
        metavar="J4",
        help="solve on the samples below the median of this joint angle, evaluate on the rest",
    )
    ap.add_argument(
        "--holdout",
        action="store_true",
        help="solve on even samples, evaluate on odd samples",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="output json (default: joint_result.json in the first directory)",
    )
    a = ap.parse_args()

    rob = Robot()
    batches, intr0, prior = [], None, None
    for d in a.dirs:
        side, samples, intr, s0 = load_batch(d)
        res = json.load(open(os.path.join(d, "result.json")))
        for s in samples:
            s["_result"] = res
        if prior is None or side == "right":
            prior = res
        intr0 = np.array([intr["fx"], intr["fy"], intr["ppx"], intr["ppy"], 0, 0, 0, 0], float)
        prior["_s0"] = s0
        batches.append((side, samples))
        print(f"{d}: {side} {len(samples)} samples")
    sides = [b[0] for b in batches]

    train, test = batches, None
    if a.holdout_by:
        j = int(a.holdout_by.lstrip("Jj")) - 1
        train, test = [], []
        for s, smp in batches:
            med = float(np.median([x["q"][j] for x in smp]))
            train.append((s, [x for x in smp if x["q"][j] <= med]))
            test.append((s, [x for x in smp if x["q"][j] > med]))
        print(f"hold-out by J{j+1}: solving on the lower half, evaluating on the upper half")
    elif a.holdout:
        train = [(s, smp[0::2]) for s, smp in batches]
        test = [(s, smp[1::2]) for s, smp in batches]

    P0 = Params(sides, offsets=False)
    P0.s0 = prior["_s0"]
    sol0 = solve(P0, rob, train, intr0, prior)
    summarize("Z + X only (no zero offsets)", sol0, P0, rob, train, intr0, test)

    if not a.no_offsets:
        P1 = Params(sides, offsets=True, mount=a.mount, intr=a.intr, scale=a.scale)
        P1.s0 = prior["_s0"]
        sol1 = solve(P1, rob, train, intr0, prior)
        Tbc = summarize(
            "Z + X + zero offsets"
            + (" + board scale" if a.scale else "")
            + (" + left shoulder translation" if a.mount else "")
            + (" + intrinsics" if a.intr else ""),
            sol1,
            P1,
            rob,
            train,
            intr0,
            test,
        )
        out = a.out or os.path.join(a.dirs[0], "joint_result.json")
        rec = {
            "schema": "handeye_joint_result/v1",
            "dirs": a.dirs,
            "sides": sides,
            "T_base_cam": Tbc.tolist(),
            "rms_px": float(np.sqrt(np.mean(sol1.fun**2))),
            "joint_offsets_rad": {s: P1.offsets7(sol1.x, s).tolist() for s in sides},
            "note": "J7 is not solved; J1 of the first arm is fixed to 0 as the reference",
            "mount_left_dt_m": (P1.get(sol1.x, "mount_left").tolist() if a.mount else None),
            "board_square_m": (float(P1.get(sol1.x, "scale")[0]) if a.scale else prior["_s0"]),
            "board_square_nominal_m": prior["_s0"],
            "holdout_by": a.holdout_by,
            "intrinsics_opt": (P1.get(sol1.x, "intr").tolist() if a.intr else None),
            "holdout": a.holdout,
        }
        json.dump(rec, open(out, "w"), indent=1)
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
