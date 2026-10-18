"""Joint order and limits of the A7 dual arm, and the mounting transform of the L6 hand.

Data only; calibration, visualisation and deployment all read these values from here.

Joint angles are always in motor order, which is the SDK order of `A7.get_angles()` and
`A7.move_j()` and the joint order of `dual_a7.urdf`:

    J1 shoulder pitch  J2 shoulder roll  J3 shoulder yaw  J4 elbow pitch
    J5 wrist yaw       J6 wrist pitch    J7 wrist roll

Some A7 URDFs list the wrist as roll first, then pitch, with the pitch axis flipped;
`urdf_to_motor()` and `motor_to_urdf()` convert to and from that order.
"""

from __future__ import annotations

NUM_JOINTS = 7

JOINT_NAMES = [
    "shoulder pitch",
    "shoulder roll",
    "shoulder yaw",
    "elbow pitch",
    "wrist yaw",
    "wrist pitch",
    "wrist roll",
]

# CAN motor ids; the reliable way to tell the arms apart (interface names change with the wiring).
LEFT_IDS = list(range(61, 68))
RIGHT_IDS = list(range(51, 58))


def urdf_to_motor(q):
    """Joint angles of a "wrist roll first" URDF to motor order."""
    q = list(q)
    return [q[0], q[1], q[2], q[3], q[4], -q[6], q[5]]


def motor_to_urdf(q):
    """Joint angles in motor order to the order of a "wrist roll first" URDF."""
    q = list(q)
    return [q[0], q[1], q[2], q[3], q[4], q[6], -q[5]]


# Joint limits (rad, motor order): the intersection of the SDK limits and the mechanical
# limits, so they are more conservative than what `A7.move_j()` checks.
LIMITS_LEFT = [
    (-2.970, 1.050),  # J1 shoulder pitch
    (-0.100, 3.200),  # J2 shoulder roll (see POLE_CLEARANCE_RAD)
    (-2.355, 2.355),  # J3 shoulder yaw
    (-0.100, 2.355),  # J4 elbow pitch
    (-2.355, 2.355),  # J5 wrist yaw
    (-1.570, 1.570),  # J6 wrist pitch
    (-1.570, 1.570),  # J7 wrist roll
]
LIMITS_RIGHT = [
    (-1.050, 2.970),  # J1 shoulder pitch (mirror of the left arm)
    (-3.200, 0.100),  # J2 shoulder roll (mirror of the left arm)
    (-2.355, 2.355),  # J3 shoulder yaw
    (-0.100, 2.355),  # J4 elbow pitch
    (-2.355, 2.355),  # J5 wrist yaw
    (-1.570, 1.570),  # J6 wrist pitch
    (-1.570, 1.570),  # J7 wrist roll
]

# J2 = 0 is a kinematic singularity. Cartesian control and IK keep J2 at least 10 deg away
# from it (left arm J2 >= +10 deg, right arm J2 <= -10 deg); joint-space motion is not
# affected. Note that `A7.home()` moves J2 to exactly 0.
POLE_CLEARANCE_RAD = 0.1745  # 10 deg


def safe_limits(side: str):
    """Joint limits with the J2 pole clearance applied, for Cartesian control and IK.

    The home pose (J2 = 0) is outside these limits.
    """
    lim = list(LIMITS_LEFT if side == "left" else LIMITS_RIGHT)
    lo, hi = lim[1]
    if side == "left":
        lim[1] = (max(lo, POLE_CLEARANCE_RAD), hi)
    else:
        lim[1] = (lo, min(hi, -POLE_CLEARANCE_RAD))
    return lim


# Mounting of the L6 hand on the arm, as a fixed joint:
#     parent  {Left,Right}_Wrist_Roll_Link of `dual_a7.urdf` (frame of the last motor)
#     child   {l,r}h_hand_base_link of the L6 URDF (origin on the mating face of the hand base)
#     xyz [-0.0225, 0, -0.0705]   rpy [pi, 0, +-pi/2] (right +, left -)
# The -22.5 mm moves the joint origin back onto the centre line of the arm; the 70.5 mm along
# the tool axis is the distance from there to the mating face, including the adapter parts.
# The rotation about the tool axis is equivalent to a zero offset of the wrist roll joint and
# cannot be separated from it by hand-eye calibration.
HAND_MOUNT_XYZ_IN_WRIST_ROLL_M = (-0.0225, 0.0, -0.0705)
HAND_MOUNT_RPY = {"right": (3.1416, 0.0, 1.5708), "left": (3.1416, 0.0, -1.5708)}
# Thickness of any extra shim added along the tool axis; 0 for the standard mounting.
HAND_SHIM_M = 0.0


def hand_mount_T(side: str, shim_m: float | None = None):
    """4x4 transform from the wrist roll frame to the L6 hand_base_link (URDF rpy convention)."""
    import numpy as _np

    r, p, y = HAND_MOUNT_RPY[side]
    cr, sr, cp, sp, cy, sy = _np.cos(r), _np.sin(r), _np.cos(p), _np.sin(p), _np.cos(y), _np.sin(y)
    R = _np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )
    T = _np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = _np.asarray(HAND_MOUNT_XYZ_IN_WRIST_ROLL_M, float) + _np.array(
        [0.0, 0.0, -(HAND_SHIM_M if shim_m is None else shim_m)]
    )
    return T
