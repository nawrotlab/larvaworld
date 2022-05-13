import math
import shapely.geometry as geometry
from typing import Optional, List
import numpy as np


def move_point(point: geometry.Point, angle: float, distance: float) -> geometry.Point:
    return geometry.Point(
        point.x + math.cos(angle) * distance,
        point.y + math.sin(angle) * distance)


def radar_line(starting_point: geometry.Point, angle: float, distance: float) -> geometry.LineString:
    distant_point = move_point(starting_point, angle, distance)
    return geometry.LineString((starting_point, distant_point))


def distance(point: tuple, angle: float, way: geometry.LineString, max_distance: float = 1000) -> Optional[float]:
    starting_point = geometry.Point(point)
    radar = radar_line(starting_point, angle, max_distance)

    intersection_points = radar.intersection(way)
    if intersection_points.is_empty:
        return None
    return starting_point.distance(intersection_points)

def distance_multi(point: tuple, angle: float, ways: List[geometry.LineString], max_distance: float = 1000) -> Optional[float]:
    starting_point = geometry.Point(point)
    radar = radar_line(starting_point, angle, max_distance)

    min_dst=None
    # dsts=[]
    for way in ways :
        intersection_points = radar.intersection(way)
        if not intersection_points.is_empty:
            dst=starting_point.distance(intersection_points)
            if min_dst is None or dst<min_dst:
                min_dst=dst
    return min_dst