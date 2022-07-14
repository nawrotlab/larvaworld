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

def radar_tuple(p0: geometry.Point, angle: float, distance: float):
    p1 = geometry.Point(
        p0.x + math.cos(angle) * distance,
        p0.y + math.sin(angle) * distance)
    return p0, p1


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

def segments_intersection(segment_1, segment_2):
    return segments_intersection_p(segment_1[0].x, segment_1[0].y, segment_1[1].x, segment_1[1].y,
                                   segment_2[0].x, segment_2[0].y, segment_2[1].x, segment_2[1].y)


def segments_intersection_p(p0_x, p0_y, p1_x, p1_y, p2_x, p2_y, p3_x, p3_y):
    EPSILON = 0.000001
    s1_x = p1_x - p0_x

    # print("s1_y = " + str(p1_y) + " - " + str(p0_y))

    s1_y = p1_y - p0_y
    s2_x = p3_x - p2_x

    # print("s2_y = " + str(p3_y) + " - " + str(p2_y))

    s2_y = p3_y - p2_y

    # print("divisore: " + str(-s2_x) + " * " + str(s1_y) + " + " + str(s1_x) + " * " + str(s2_y))

    # s = (-s1_y * (p0_x - p2_x) + s1_x * (p0_y - p2_y)) / (-s2_x * s1_y + s1_x * s2_y)
    # t = ( s2_x * (p0_y - p2_y) - s2_y * (p0_x - p2_x)) / (-s2_x * s1_y + s1_x * s2_y)

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
    from lib.model.space.obstacle import Obstacle
    min_dst = None
    nearest_obstacle = None

    for obj in obstacles:
        if issubclass(type(obj), Obstacle):
            obstacle = obj

            # check collision between obstacle edges and sensor ray
            # print(obstacle.edges)
            # raise
            for edge in obstacle.edges:
                # print("obstacle_edge:", geom_utils.segment_to_string(obstacle_edge))
                intersection_point = segments_intersection(sensor_ray, edge)

                if intersection_point is not None:
                    # print("intersection_point:", geom_utils.point_to_string(intersection_point))
                    dst = distance(p0, intersection_point)

                    if min_dst is None or dst < min_dst:
                        min_dst = dst
                        nearest_obstacle = obstacle
                        # print("new distance_from_nearest_obstacle:", distance_from_nearest_obstacle)
    return min_dst, nearest_obstacle


class Point:

    def __init__(self, x, y):
        self.x = x
        self.y = y