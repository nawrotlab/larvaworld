from shapely import geometry, measurement
from typing import Optional, List
import math

def move_point(p0: geometry.Point, angle: float, distance: float) -> geometry.Point:
    return geometry.Point(
        p0.x + math.cos(angle) * distance,
        p0.y + math.sin(angle) * distance)


def radar_line(p0: geometry.Point, angle: float, distance: float) -> geometry.LineString:
    p1 = move_point(p0, angle, distance)
    return geometry.LineString((p0, p1))

def radar_tuple(p0: geometry.Point, angle: float, distance: float):
    p1 = geometry.Point(
        p0.x + math.cos(angle) * distance,
        p0.y + math.sin(angle) * distance)
    return p0, p1

def eudis5(p1, p2):
    return geometry.Point(p1).distance(geometry.Point(p2))

def distance(p0: tuple, angle: float, way: geometry.LineString, max_distance: float = 1000) -> Optional[float]:
    p00 = geometry.Point(p0)
    radar = radar_line(p00, angle, max_distance)

    ps = radar.intersection(way)
    if ps.is_empty:
        return None
    return p00.distance(ps)

def min_dst_to_lines_along_vector(point: tuple, angle: float, target_lines: List[geometry.LineString], max_distance: float = 1000) -> Optional[float]:
    p0 = geometry.Point(point)
    radar = radar_line(p0, angle, max_distance)

    min_dst=None
    for line in target_lines :
        ps = radar.intersection(line)
        if not ps.is_empty:
            dst=p0.distance(ps)
            if min_dst is None or dst<min_dst:
                min_dst=dst
    return min_dst

def segments_intersection(segment_1, segment_2):
    return segments_intersection_p(segment_1[0].x, segment_1[0].y, segment_1[1].x, segment_1[1].y,
                                   segment_2[0].x, segment_2[0].y, segment_2[1].x, segment_2[1].y)


def segments_intersection_p(p0_x, p0_y, p1_x, p1_y, p2_x, p2_y, p3_x, p3_y):
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
    t = ( s2_x * (p0_y - p2_y) - s2_y * (p0_x - p2_x)) / divisore_2

    if s >= 0 and s <= 1 and t >= 0 and t <= 1:
        # Collision detected
        intersection_x = p0_x + (t * s1_x)
        intersection_y = p0_y + (t * s1_y)
        return geometry.Point(intersection_x, intersection_y)
    else:
        return None



def detect_nearest_obstacle(obstacles, sensor_ray, p0) :

    min_dst = None
    nearest_obstacle = None

    for obj in obstacles:
        # check collision between obstacle edges and sensor ray
        for edge in obj.edges:
            intersection_point = segments_intersection(sensor_ray, edge)

            if intersection_point is not None:
                dst = distance(p0, intersection_point)

                if min_dst is None or dst < min_dst:
                    min_dst = dst
                    nearest_obstacle = obj

    return min_dst, nearest_obstacle


def line_through_point(pos, angle, length, pos_as_start=False) :
    if not pos_as_start :
        length=-length
    p0 = geometry.Point(pos)
    p1 = geometry.Point(p0.x + length * math.cos(angle),
                p0.y + length * math.sin(angle))
    return geometry.LineString([p0, p1])
