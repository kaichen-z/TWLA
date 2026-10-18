#!/usr/bin/env python3
"""Connect to one A7 arm: a thin wrapper around the SDK class `A7`.

The wrapper pins the motor targets to the current position before enabling. `A7.enable()`
does not write a target, so a motor whose target register still holds 0 would move to 0 as
soon as it is enabled. The SDK `emergency_stop()` reads the position and writes it back as
the target, which is exactly the pinning needed.

    from arm import connect
    arm = connect("right")            # enabled, targets pinned, 0.2 rad/s
    print(arm.get_angles())
    arm.home(blocking=True)
    arm.close()                       # does not disable; the arm keeps holding its position

Command line (reading the state does not enable the arm):
    python arm.py --side right
    python arm.py --side right --home
    python arm.py --side right --joint 5 --delta 30
    python arm.py --side left  --disable
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from realhand.arm import A7

NUM_JOINTS = 7

# Default CAN interface of each arm. Interface numbers follow USB enumeration and may
# change after re-plugging; a wrong interface fails safely, because `A7()` handshakes with
# every motor (ids 51-57 right, 61-67 left) before any command is sent.
CHANNELS = {"left": "can0", "right": "can1"}

# Motor order, which is also the SDK order.
JOINT_NAMES = [
    "shoulder pitch",
    "shoulder roll",
    "shoulder yaw",
    "elbow pitch",
    "wrist yaw",
    "wrist pitch",
    "wrist roll",
]

DEFAULT_VELOCITY = 0.2  # rad/s; SDK default 1.0
DEFAULT_ACCELERATION = 2.0  # rad/s^2; SDK default 10.0


def home_blocked(side: str) -> str:
    """Reason why homing is blocked, or "" if it is allowed.

    Homing is blocked while the marker file `drivers/NO_HOME_<SIDE>` exists; its content is the
    reason (for example a calibration board mounted on the arm).
    """
    import os

    f = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"NO_HOME_{side.upper()}")
    return open(f, encoding="utf-8").read().strip() if os.path.isfile(f) else ""


def pin_target_to_current(arm: A7) -> None:
    """Write the measured angle of every motor as its target (see the module docstring)."""
    arm.emergency_stop()


def connect(
    side: str,
    channel: str | None = None,
    *,
    velocity: float = DEFAULT_VELOCITY,
    acceleration: float = DEFAULT_ACCELERATION,
    enable: bool = True,
) -> A7:
    """Connect to one arm; with `enable=False` it is only read and never moves.

    Velocity and acceleration are set before enabling, so that the first motion after
    enabling does not use the SDK defaults.
    """
    if side not in CHANNELS:
        raise ValueError(f"side must be left/right, got {side!r}")
    arm = A7(side=side, interface_name=channel or CHANNELS[side])
    arm.set_velocities([velocity] * NUM_JOINTS)
    arm.set_accelerations([acceleration] * NUM_JOINTS)
    time.sleep(0.2)
    if enable:
        pin_target_to_current(arm)
        time.sleep(0.2)
        arm.enable()
        time.sleep(0.4)
    return arm


def format_angles(angles) -> str:
    """Joint angles (rad) as one line in degrees."""
    return "  ".join(f"J{i+1}={v * 57.2958:+7.2f}°" for i, v in enumerate(angles))


def watch_until_done(arm: A7, timeout_s: float = 30.0, period_s: float = 0.5) -> None:
    """Print the joint angles until the motion ends or the timeout expires."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        print(
            f"  t={time.time()-t0:5.1f}s  {format_angles(arm.get_angles())}"
            f"  moving={arm.is_moving()}"
        )
        if not arm.is_moving():
            return
        time.sleep(period_s)
    print(f"  still moving after {timeout_s}s, giving up waiting")


def main() -> int:
    """Command line: read the state, home, move one joint or disable."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--side", choices=["left", "right"], required=True)
    ap.add_argument("--channel", default=None, help="default: the CHANNELS mapping")
    ap.add_argument("--velocity", type=float, default=DEFAULT_VELOCITY)
    ap.add_argument("--acceleration", type=float, default=DEFAULT_ACCELERATION)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--home", action="store_true", help="move all 7 joints to zero")
    g.add_argument(
        "--joint",
        type=int,
        choices=range(1, 8),
        metavar="1-7",
        help="move only this joint (with --delta or --to)",
    )
    g.add_argument(
        "--disable", action="store_true", help="disable (the arm can then be moved by hand)"
    )
    ap.add_argument("--delta", type=float, help="degrees to move relative to the current angle")
    ap.add_argument("--to", type=float, help="absolute target angle in degrees")
    args = ap.parse_args()

    if args.home and home_blocked(args.side):
        print(f"Homing refused: NO_HOME_{args.side.upper()} exists:\n" + home_blocked(args.side))
        return 3
    moves = args.home or args.joint is not None
    arm = connect(
        args.side,
        args.channel,
        velocity=args.velocity,
        acceleration=args.acceleration,
        enable=moves,
    )
    try:
        start = arm.get_angles()
        print(f"{args.side} arm @ {args.channel or CHANNELS[args.side]}")
        print(f"  current      {format_angles(start)}")
        print(f"  temperature  {' '.join(f'{t:.0f}' for t in arm.get_temperatures())} degC")

        if args.disable:
            arm.disable()
            print("  disabled: the arm can now be moved by hand; it may drop")
            return 0
        if not moves:
            return 0

        if args.home:
            target = [0.0] * NUM_JOINTS
            desc = "home(): all joints to zero"
        else:
            idx = args.joint - 1
            if args.delta is None and args.to is None:
                ap.error("--joint requires --delta or --to")
            target = list(start)
            target[idx] = (
                start[idx] + np.deg2rad(args.delta)
                if args.delta is not None
                else np.deg2rad(args.to)
            )
            lo, hi = arm.get_joint_limits()[idx]
            if not (lo <= target[idx] <= hi):
                print(
                    f"  target {np.rad2deg(target[idx]):+.1f} deg is outside the limits "
                    f"[{np.rad2deg(lo):+.1f} deg, {np.rad2deg(hi):+.1f} deg], aborting"
                )
                return 1
            desc = (
                f"J{args.joint}({JOINT_NAMES[idx]}) "
                f"{np.rad2deg(start[idx]):+.1f}° → {np.rad2deg(target[idx]):+.1f}°"
            )

        will_move = [
            (i, np.rad2deg(target[i] - start[i]))
            for i in range(NUM_JOINTS)
            if abs(target[i] - start[i]) > np.deg2rad(1.0)
        ]
        print(f"\n  {desc}")
        print(
            "  will move: "
            + (
                ", ".join(f"J{i+1} {d:+.1f} deg" for i, d in will_move)
                or "nothing (already at the target)"
            )
        )
        print(f"  velocity {args.velocity} rad/s, acceleration {args.acceleration} rad/s^2\n")

        arm.move_j(target, blocking=False)
        watch_until_done(arm)

        time.sleep(0.3)
        end = arm.get_angles()
        print(f"\n  final        {format_angles(end)}")
        worst = max(abs(e - t) for e, t in zip(end, target))
        print(f"  largest deviation from the target {np.rad2deg(worst):.2f} deg")
        print("\nThe arm is still enabled and holds the target. Use --disable to release it.")
    finally:
        arm.close()  # the enable state of the motors is unchanged
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
