import math
import time

import numpy as np


def _restore_angle(a, d, l, n, num_segments, correction_coef):
    k0 = (l * n / num_segments) / correction_coef
    k1 = (l * (n + 1) / num_segments) / correction_coef
    if d <= k0:
        return a, 0
    elif k0 < d < k1:
        da = 1.0 * a * d / (l / num_segments)
        return a - da, da
    elif k1 <= d:
        return 0, a


def restore_bend(state, d, l, num_segments, correction_coef=1.0):
    nstate = []
    da = 0
    for i, a in enumerate(state):
        na, k = _restore_angle(a + da, d, l, i, num_segments, correction_coef=correction_coef)
        da = k
        nstate.append(na)
    return nstate


def restore_bend_2seg(bend, d, l, correction_coef=1.0):
    # print(bend,d,l)
    k0 = 2*d*correction_coef/ l
    if 0 <= k0 < 1:
        return bend * (1 - k0)
    elif 1 <= k0:
        return 0
    elif k0 < 0:
        return bend


def angle(a, b, c, in_deg=True):
    if np.isnan(a).any() or np.isnan(b).any() or np.isnan(c).any():
        return np.nan
    if in_deg:
        ang = (math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])) - 180) % 360
        return ang if ang <= 180 else ang - 360
    else:
        ang = (math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])) - np.pi) % (
                2 * np.pi)
        return ang if ang <= np.pi else ang - 2 * np.pi


def angle_to_x_axis(point_1, point_2, in_deg=True):
    # Point 1 is start, point 2 is end of vector
    # print(point_2, point_1)
    # print(type(point_2), type(point_1), type(point_1[0]),type(point_1[1]))
    # if np.isnan(point_1).any() or np.isnan(point_2).any():
    #
    #     return np.nan
    dx, dy = np.array(point_2) - np.array(point_1)
    rads = math.atan2(dy, dx)
    rads %= 2 * np.pi
    if in_deg:
        return math.degrees(rads)
    else:
        return rads


def angle_dif(angle_1, angle_2, in_deg=True):
    a = angle_1 - angle_2
    if in_deg:
        return (a + 180) % 360 - 180
    else :
        return (a + np.pi) % (np.pi * 2) - np.pi


def rotate_around_point(point, radians, origin=[0, 0]):
    """Rotate a point around a given point.

    I call this the "high performance" version since we're caching some
    values that are needed >1 time. It'sigma less readable than the previous
    function but it'sigma faster.
    """
    x, y = point
    offset_x, offset_y = origin
    adjusted_x = (x - offset_x)
    adjusted_y = (y - offset_y)
    cos_rad = math.cos(radians)
    sin_rad = math.sin(radians)
    qx = offset_x + cos_rad * adjusted_x + sin_rad * adjusted_y
    qy = offset_y + -sin_rad * adjusted_x + cos_rad * adjusted_y

    return qx, qy


def rotate_around_center(point, radians):
    x, y = point
    cos_rad = math.cos(radians)
    sin_rad = math.sin(radians)
    qx = cos_rad * x + sin_rad * y
    qy = -sin_rad * x + cos_rad * y
    return np.array([qx, qy])


def rotate_around_center_multi(points : np.array, radians):
    cos_rad = math.cos(radians)
    sin_rad = math.sin(radians)
    # def func(point):
    #     # x,y=point
    #     return cos_rad * point[0] + sin_rad * point[1], -sin_rad * point[0] + cos_rad * point[1]
    # # print(points)
    # # print(type(points))
    # # print(type(points[0]))
    # tt=time.time()
    # b=np.array([(cos_rad * x + sin_rad * y, -sin_rad * x + cos_rad * y) for x, y in points])
    # # print(b)
    # ttt = time.time()
    #
    #
    # a=np.apply_along_axis(func, 1, points)
    # tttt = time.time()
    # print(ttt-tt, tttt-ttt)
    # # print(a)
    # # print(a==b)
    # raise
    return np.array([(cos_rad * x + sin_rad * y, -sin_rad * x + cos_rad * y) for x, y in points])


def rotate_multiple_points(points, radians, origin=[0, 0]):
    # points have the form :
    # points=np.array([[1,2],[3,4], [5,6], [7,8]])
    qx, qy = rotate_around_point(points.T, radians, origin=origin)
    return np.vstack((qx, qy)).T


def unwrap_deg(ts):
    b = np.copy(ts)
    b[~np.isnan(b)] = np.unwrap(b[~np.isnan(b)] * np.pi / 180) * 180 / np.pi
    return b

def line_through_point(pos, angle, length, pos_as_start=False) :
    import math
    from shapely.geometry import LineString, Point

    if not pos_as_start :
        length=-length

    start = Point(pos)
    end = Point(start.x + length * math.cos(angle),
                start.y + length * math.sin(angle))
    return LineString([start, end])

