"""
Methods for managing shapely-based metrics
"""

from __future__ import annotations

from typing import Any

from shapely import geometry

__all__: list[str] = [
    "segments_intersection",
    "detect_nearest_obstacle",
]


def segments_intersection(
    segment_1: tuple[geometry.Point, geometry.Point],
    segment_2: tuple[geometry.Point, geometry.Point],
) -> geometry.Point | None:
    """
    Compute intersection point of two line segments.

    Determines if two line segments intersect and returns the intersection
    point if they do, using coordinate-based calculation.

    Args:
        segment_1: First segment as (start_point, end_point)
        segment_2: Second segment as (start_point, end_point)

    Returns:
        Intersection Point if segments intersect, None otherwise

    Example:
        >>> from shapely.geometry import Point
        >>> seg1 = (Point(0, 0), Point(2, 2))
        >>> seg2 = (Point(0, 2), Point(2, 0))
        >>> p = segments_intersection(seg1, seg2)
        >>> p  # Point at (1, 1)
    """
    return segments_intersection_p(
        segment_1[0].x,
        segment_1[0].y,
        segment_1[1].x,
        segment_1[1].y,
        segment_2[0].x,
        segment_2[0].y,
        segment_2[1].x,
        segment_2[1].y,
    )


def segments_intersection_p(
    p0_x: float,
    p0_y: float,
    p1_x: float,
    p1_y: float,
    p2_x: float,
    p2_y: float,
    p3_x: float,
    p3_y: float,
):
    EPSILON = 0.000001
    s1_x = p1_x - p0_x
    s1_y = p1_y - p0_y
    s2_x = p3_x - p2_x
    s2_y = p3_y - p2_y

    divisore_1 = -s2_x * s1_y + s1_x * s2_y
    if divisore_1 == 0:
        divisore_1 = EPSILON

    divisore_2 = -s2_x * s1_y + s1_x * s2_y
    if divisore_2 == 0:
        divisore_2 = EPSILON

    s = (-s1_y * (p0_x - p2_x) + s1_x * (p0_y - p2_y)) / divisore_1
    t = (s2_x * (p0_y - p2_y) - s2_y * (p0_x - p2_x)) / divisore_2

    if s >= 0 and s <= 1 and t >= 0 and t <= 1:
        intersection_x = p0_x + (t * s1_x)
        intersection_y = p0_y + (t * s1_y)
        return geometry.Point(intersection_x, intersection_y)
    else:
        return None


def detect_nearest_obstacle(
    obstacles: list[Any],
    sensor_ray: tuple[geometry.Point, geometry.Point],
    p0: geometry.Point,
) -> tuple[float | None, Any]:
    """
    Find nearest obstacle intersected by a sensor ray.

    Checks all obstacle edges for intersections with sensor ray and returns
    the closest obstacle and its distance from the sensor origin point.

    Args:
        obstacles: List of obstacle objects with .edges attribute
        sensor_ray: Ray segment as (start_point, end_point)
        p0: Sensor origin point for distance calculation

    Returns:
        Tuple of (distance, obstacle) where distance is to nearest intersection,
        or (None, None) if no intersections found

    Example:
        >>> from shapely.geometry import Point
        >>> ray = (Point(0, 0), Point(10, 0))
        >>> dist, obs = detect_nearest_obstacle(obstacles, ray, Point(0, 0))
    """
    Dmin = None
    Onearest = None

    for obj in obstacles:
        # check collision between obstacle edges and sensor ray
        for edge in obj.edges:
            p = segments_intersection(sensor_ray, edge)

            if p is not None:
                if Dmin is None or p0.distance(p) < Dmin:
                    Dmin = p0.distance(p)
                    Onearest = obj

    return Dmin, Onearest
