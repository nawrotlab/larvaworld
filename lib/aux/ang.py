import copy
import math
import numpy as np


def wrap_angle_to_0(a, in_deg=False):
    """Converts an angle to be around 0 meaning within [-lim, +lim]
        where lim is pi for radians and 180 for degrees

        Parameters
        ----------
        a : float
            The angle to be wrapped
        in_deg : bool, optional
            Whether angles are in degrees (default is False)

        Returns
        -------
        a
            the angle wrapped around 0
        """
    lim=np.pi if not in_deg else 180
    if np.abs(a) > lim:
        a = (a + lim) % (lim * 2) - lim
    return a


def rear_orientation_change(bend, d, l, correction_coef=1.0):
    k0 = 2*d*correction_coef/ l
    if 0 <= k0 < 1:
        return bend * k0
    elif 1 <= k0:
        return bend
    elif k0 < 0:
        return 0


def angle_from_3points(p1, pmid, p2, in_deg=True):
    """Computes an angle from 3 2D points, meaning between 2 line segments :p1->pmid and pmid->p2

        Parameters
        ----------
        p1, pmid, p2 : Tuple[float]
            The XY coordinates of the first, middle and final point
        in_deg : bool, optional
            Whether angles are in degrees (default is False)

        Returns
        -------
        a
            the angle
        """
    if np.isnan(p1).any() or np.isnan(pmid).any() or np.isnan(p2).any():
        return np.nan
    if in_deg:
        a = (math.degrees(math.atan2(p2[1] - pmid[1], p2[0] - pmid[0]) - math.atan2(p1[1] - pmid[1], p1[0] - pmid[0])) - 180) % 360
        return a if a <= 180 else a - 360
    else:
        a = (math.degrees(math.atan2(p2[1] - pmid[1], p2[0] - pmid[0]) - math.atan2(p1[1] - pmid[1], p1[0] - pmid[0])) - np.pi) % (
                2 * np.pi)
        return a if a <= np.pi else a - 2 * np.pi


def angle_to_x_axis(point_1, point_2, in_deg=True):
    """Computes the angle of the line segment p1->p2 relative to the x axis

        Parameters
        ----------
        p1, p2 : Tuple[float]
            The XY coordinates of the start and end point of vector
        in_deg : bool, optional
            Whether angles are in degrees (default is True)

        Returns
        -------
        a
            the angle
        """

    dx, dy = np.array(point_2) - np.array(point_1)
    a = math.atan2(dy, dx)
    a %= 2 * np.pi
    if in_deg:
        return math.degrees(a)
    else:
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




def rotate_point_around_point(point, radians, origin=None):
    """Rotate a point around a given point clockwise

        Parameters
        ----------
        point : Tuple[float]
            The XY coordinates of the point to be rotated
        radians : float
            The rotation angle
        origin : Tuple[float], optional
            The XY coordinates of the rotation point (default is [0, 0])

        Returns
        -------
        p : Tuple[float]
            The XY coordinates of the rotated point
        """

    if origin is None:
        origin = [0, 0]
    x, y = point
    x0, y0 = origin
    xx = (x - x0)
    yy = (y - y0)
    cos_rad = math.cos(radians)
    sin_rad = math.sin(radians)
    qx = x0 + cos_rad * xx + sin_rad * yy
    qy = y0 + -sin_rad * xx + cos_rad * yy
    p = (qx, qy)
    return p



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
        origin = [0, 0]
    qx, qy = rotate_point_around_point(np.array(points).T, radians, origin=origin)
    ps=np.vstack((qx, qy)).T
    return ps


def unwrap_rad(s, in_deg=False) :
    """
    Unwraps an angular timeseries

    Parameters
    ----------
    s : pd.Dataframe
        Index levels : ['Step', 'AgentID']
        single column : the angular timeseries
    in_deg : bool, optional
        Whether angles are in degrees (default is False)

    Returns
    -------
    s: np.array of appropriate shape to be inserted in the pd.Dataframe
        the unwrapped timeseries
    """
    ids = s.index.unique('AgentID').values
    UP = np.zeros([s.index.unique('Step').size, len(ids)]) * np.nan
    for j, id in enumerate(ids):
        b = s.xs(id, level='AgentID').values
        if in_deg:
            b = np.deg2rad(b)
        b[~np.isnan(b)] = np.unwrap(b[~np.isnan(b)])
        if in_deg:
            b = np.rad2deg(b)
        UP[:, j] = b
    return UP.flatten()




