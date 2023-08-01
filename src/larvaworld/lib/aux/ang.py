import math
import numpy as np


def wrap_angle_to_0(angle: float, in_deg: bool = False) -> float:
    """
    Wraps an angle within an absolute range of pi if in radians,
    or 180 if in degrees around 0.

    Args:
    - angle (float): The angle to wrap.
    - in_deg (bool): If True, the angle is assumed to be in degrees.

    Returns:
    - float: The wrapped angle.
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



def angles_between_vectors(xy_front: np.ndarray, xy_mid: np.ndarray = None, xy_rear: np.ndarray = None,
                           in_deg: bool = True, wrap_to_0: bool = True) -> np.ndarray:
    """
        Calculate the angles defined by 3 arrays of 2D points.
        Each line of the 3 arrays defines a pair of vectors :
            - front vector starting at the midpoint and ending at the frontpoint.
            - rear vector starting at the rearpoint and ending at the midpoint.

        Parameters:
            xy_front (np.ndarray):
                The coordinates of the frontpoints as an array of shape (N,2).
            xy_mid (np.ndarray):
                The coordinates of the midpoints as an array of shape (N,2). Defaults to array of (0,0) if not provided.
            xy_rear (np.ndarray):
                The coordinates of the rearpoints as an array of shape (N,2). Default to rear vectors parallel to the x-axis if not provided.
            in_deg (bool):
                If True, the angle is returned in degrees.
            wrap_to_0 (bool):
                If True, the angle is normalized within a range (-lim,lim) where lim=π (180 if in_deg is True).
                Otherwise the angle is normalized within a range (0,2*lim)

        Returns:
            np.ndarray: The array of pairwise angles of the front and rear vectors. Range [-π, π).

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


def angle_dif(angle_1, angle_2, in_deg=True):
    """Computes the difference between two angles

        Parameters
        ----------
        angle_1, angle_2 : float
            The angles
        in_deg : bool, optional
            Whether angles are in degrees (default is True)

        Returns
        -------
        a
            the angle
        """

    a = angle_1 - angle_2
    if in_deg:
        return (a + 180) % 360 - 180
    else :
        return (a + np.pi) % (np.pi * 2) - np.pi

def rotationMatrix(a):
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


# def rotate_point_around_point(point, radians, origin=None):
#     """Rotate a point around a given point clockwise
#
#         Parameters
#         ----------
#         point : Tuple[float]
#             The XY coordinates of the point to be rotated
#         radians : float
#             The rotation angle
#         origin : Tuple[float], optional
#             The XY coordinates of the rotation point (default is [0, 0])
#
#         Returns
#         -------
#         p : Tuple[float]
#             The XY coordinates of the rotated point
#         """
#
#     if origin is None:
#         origin = [0, 0]
#     x, y = point
#     x0, y0 = origin
#     xx = (x - x0)
#     yy = (y - y0)
#     cos_rad = math.cos(radians)
#     sin_rad = math.sin(radians)
#     qx = x0 + cos_rad * xx + sin_rad * yy
#     qy = y0 + -sin_rad * xx + cos_rad * yy
#     return np.array([qx, qy])


def rotate_points_around_point(points, radians, origin=None):
    """Rotate multiple points around a given point clockwise

        Parameters
        ----------
        points : List[Tuple[float]]
            The XY coordinates of the points to be rotated
        radians : float
            The rotation angle
        origin : Tuple[float], optional
            The XY coordinates of the rotation point (default is [0, 0])

        Returns
        -------
        ps : List[Tuple[float]]
            The XY coordinates of the rotated points
        """

    if origin is None:
        origin = (0, 0)
    origin=np.array(origin)
    return (points - origin) @ rotationMatrix(radians) + origin

