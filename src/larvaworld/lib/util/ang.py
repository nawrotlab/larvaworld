"""
Methods for managing angular metrics
"""

from __future__ import annotations

import math

import numpy as np

__all__: list[str] = [
    "wrap_angle_to_0",
    "angles_between_vectors",
    "angle_dif",
    "rotationMatrix",
    "rotate_points_around_point",
]


def wrap_angle_to_0(angle: float, in_deg: bool = False) -> float:
    """
    Wrap an angle to range around zero.

    Wraps an angle within absolute range of π (radians) or 180° (degrees) around 0.

    Args:
        angle: The angle to wrap
        in_deg: If True, angle is in degrees (default: False for radians)

    Returns:
        Wrapped angle in same units as input, range (-π, π] or (-180°, 180°]

    Example:
        >>> wrap_angle_to_0(270, in_deg=True)
        -90.0
        >>> wrap_angle_to_0(3*np.pi/2)  # radians
        -1.5707...
    """
    if in_deg:
        angle = angle % 360.0
        if angle > 180.0:
            angle -= 360.0
    else:
        angle = angle % (2 * math.pi)
        if angle > math.pi:
            angle -= 2 * math.pi

    return angle


def angles_between_vectors(
    xy_front: np.ndarray,
    xy_mid: np.ndarray = None,
    xy_rear: np.ndarray = None,
    in_deg: bool = True,
    wrap_to_0: bool = True,
) -> np.ndarray:
    """
    Calculate angles between front and rear vectors defined by triplets of 2D points.

    Each row defines two vectors:
    - Front vector: from midpoint to frontpoint
    - Rear vector: from rearpoint to midpoint

    Args:
        xy_front: Coordinates of frontpoints, shape (N, 2)
        xy_mid: Coordinates of midpoints, shape (N, 2). Defaults to (0,0) if None
        xy_rear: Coordinates of rearpoints, shape (N, 2). If None, rear vectors are parallel to x-axis
        in_deg: If True, return angles in degrees (default: True)
        wrap_to_0: If True, normalize to range (-lim, lim) where lim=π or 180°. If False, range (0, 2*lim)

    Returns:
        Array of pairwise angles between front and rear vectors

    Example:
        >>> front = np.array([[1, 1], [0, 1]])
        >>> mid = np.array([[0, 0], [0, 0]])
        >>> angles = angles_between_vectors(front, mid)
        >>> angles[0]  # 45 degrees
        45.0
    """
    xy_front = xy_front.astype(float)
    if xy_mid is None:
        xy_mid = np.zeros_like(xy_front)

    xy_mid = xy_mid.astype(float)
    xy_front -= xy_mid
    a1 = np.arctan2(xy_front[:, 1], xy_front[:, 0])

    if xy_rear is not None:
        xy_rear = xy_rear.astype(float)
        xy_rear = xy_mid - xy_rear
        a2 = np.arctan2(xy_rear[:, 1], xy_rear[:, 0])
        a = a1 - a2
    else:
        a = a1

    if wrap_to_0:
        a = np.remainder(a, 2 * np.pi)
        a[a > np.pi] -= 2 * np.pi
    else:
        a[a < 0] += 2 * np.pi
    if in_deg:
        a = np.degrees(a)
    return a


def angle_dif(angle_1: float, angle_2: float, in_deg: bool = True) -> float:
    """
    Compute the signed difference between two angles.

    Computes angle_1 - angle_2 with proper wrapping to range (-π, π] or (-180°, 180°].

    Args:
        angle_1: First angle
        angle_2: Second angle
        in_deg: If True, angles are in degrees (default: True)

    Returns:
        Signed angular difference in same units as input

    Example:
        >>> angle_dif(350, 10, in_deg=True)
        -20.0
        >>> angle_dif(10, 350, in_deg=True)
        20.0
    """
    a = angle_1 - angle_2
    if in_deg:
        return (a + 180) % 360 - 180
    else:
        return (a + np.pi) % (np.pi * 2) - np.pi


def rotationMatrix(a: float) -> np.ndarray:
    """
    Create a 2D rotation matrix for given angle.

    Args:
        a: Rotation angle in radians

    Returns:
        2x2 rotation matrix as numpy array

    Example:
        >>> R = rotationMatrix(np.pi/4)  # 45 degrees
        >>> R.shape
        (2, 2)
    """
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def rotate_points_around_point(
    points: np.ndarray, radians: float, origin: tuple[float, float] | None = None
) -> np.ndarray:
    """
    Rotate multiple points around a given origin point.

    Applies 2D rotation transformation using rotation matrix.

    Args:
        points: XY coordinates of points to rotate, shape (N, 2) or (2,)
        radians: Rotation angle in radians (positive = counter-clockwise)
        origin: XY coordinates of rotation center. Defaults to (0, 0)

    Returns:
        Rotated XY coordinates, same shape as input

    Example:
        >>> pts = np.array([[1, 0], [0, 1]])
        >>> rotated = rotate_points_around_point(pts, np.pi/2)  # 90° rotation
        >>> np.allclose(rotated, [[0, 1], [-1, 0]])
        True
    """
    if origin is None:
        origin = (0, 0)
    origin = np.array(origin)
    return (points - origin) @ rotationMatrix(radians) + origin
