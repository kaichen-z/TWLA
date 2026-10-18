"""Minimal quaternion utilities (numpy only).

Quaternions are stored as (x, y, z, w) arrays of shape (4,), the order used by the Wuji SDK.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "normalize",
    "to_matrix",
    "conjugate",
    "multiply",
    "rotate",
    "to_rpy",
    "to_rotvec",
    "from_two_vectors",
    "angle_between",
]


def normalize(q: np.ndarray) -> np.ndarray:
    """Unit quaternion; a zero quaternion becomes the identity."""
    q = np.asarray(q, dtype=np.float64).reshape(4)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return q / n


def to_matrix(q: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix R with v_out = R @ v_in.

    The glove orientation is treated as body -> world (see hand_motion.py).
    """
    x, y, z, w = normalize(q)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def conjugate(q: np.ndarray) -> np.ndarray:
    """Conjugate (the inverse rotation of a unit quaternion)."""
    x, y, z, w = normalize(q)
    return np.array([-x, -y, -z, w])


def multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product: `multiply(a, b)` applies b first, then a."""
    ax, ay, az, aw = normalize(a)
    bx, by, bz, bw = normalize(b)
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ]
    )


def rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by q."""
    return to_matrix(q) @ np.asarray(v, dtype=np.float64).reshape(3)


def to_rpy(q: np.ndarray) -> np.ndarray:
    """ZYX Euler angles (roll, pitch, yaw) in rad; for display only."""
    x, y, z, w = normalize(q)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw])


def from_two_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest rotation taking direction a to direction b."""
    a = np.asarray(a, dtype=np.float64).reshape(3)
    b = np.asarray(b, dtype=np.float64).reshape(3)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    a, b = a / na, b / nb
    d = float(np.dot(a, b))
    if d > 1.0 - 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    if d < -1.0 + 1e-12:
        # Opposite directions: 180 deg about any axis perpendicular to a.
        axis = np.cross(a, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, np.array([0.0, 1.0, 0.0]))
        axis /= np.linalg.norm(axis)
        return np.array([axis[0], axis[1], axis[2], 0.0])
    axis = np.cross(a, b)
    s = float(np.sqrt((1.0 + d) * 2.0))
    return normalize(np.array([axis[0] / s, axis[1] / s, axis[2] / s, s * 0.5]))


def to_rotvec(q: np.ndarray) -> np.ndarray:
    """Rotation vector (axis * angle, rad).

    Use this rather than Euler angles for orientation mapping: it is continuous for angles
    below 180 deg, with no wrap-around and no gimbal lock.
    """
    x, y, z, w = normalize(q)
    if w < 0.0:  # q and -q are the same rotation; take the short arc
        x, y, z, w = -x, -y, -z, -w
    v = np.array([x, y, z])
    s = float(np.linalg.norm(v))
    if s < 1e-12:
        return np.zeros(3)
    return v / s * (2.0 * float(np.arctan2(s, w)))


def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    """Rotation angle between two quaternions, in rad (0 to pi)."""
    d = abs(float(np.dot(normalize(a), normalize(b))))
    return float(2.0 * np.arccos(np.clip(d, -1.0, 1.0)))
