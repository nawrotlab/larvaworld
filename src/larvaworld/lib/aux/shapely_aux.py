from shapely import geometry
import math




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
                dst = geometry.Point(p0).distance(geometry.Point(intersection_point))
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


