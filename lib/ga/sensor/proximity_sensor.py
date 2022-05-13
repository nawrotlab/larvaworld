import math
import random
import pygame
from math import sin, cos

from lib.ga.exception.collision_exception import Collision
from lib.ga.geometry.point import Point
from lib.ga.geometry.util import detect_nearest_obstacle, radar_tuple
from lib.ga.sensor.sensor import Sensor
from lib.ga.util.color import Color


class ProximitySensor(Sensor):

    # COLLISION_DISTANCE = 12  # px

    def __init__(self, robot, delta_direction, saturation_value, error, max_distance, scene, collision_distance=12):
        super().__init__(robot, delta_direction, saturation_value, error, scene)
        self.max_distance = max_distance
        self.collision_distance = collision_distance
        # print(max_distance)

        # raise

    def get_value(self, pos=None, direction=None):
        if pos is None:
            pos = [self.robot.x, self.robot.y]

        if direction is None:
            direction = self.robot.direction

        x, y = pos
        angle = -direction - self.delta_direction
        p0 = Point(x, y)
        p1 = Point(
            x + math.cos(angle) * self.max_distance,
            y + math.sin(angle) * self.max_distance)
        # sensor_ray = radar_tuple(p0=p0, angle=angle, distance=self.max_distance)
        # print('m',x, y,sensor_ray[1].x,sensor_ray[1].y, self.max_distance/self.robot.model.scene._scale[0, 0],self.robot.real_length)
        min_dst, nearest_obstacle = detect_nearest_obstacle(self.scene.objects, (p0,p1), p0)

        if min_dst is None:
            # no obstacle detected
            return 0
        else:
            # check collision
            if min_dst < self.collision_distance:
                raise Collision(self.robot, nearest_obstacle)

            proximity_value = 1 / random.gauss(min_dst, self.error * min_dst)

            if proximity_value > self.saturation_value:
                return self.saturation_value
            else:
                return proximity_value

    def draw(self, pos=None, direction=None):
        if pos is None:
            pos = [self.robot.x, self.robot.y]
        if direction is None:
            direction = self.robot.direction
        x0, y0 = pos
        angle = -direction - self.delta_direction

        x1=x0 + math.cos(angle) * self.max_distance
        y1=y0 + math.sin(angle) * self.max_distance

        # self.scene.draw_line((x, y), (x_sensor_eol, y_sensor_eol),Color.RED, width=0.0005)
        pygame.draw.line(self.scene.screen, Color.RED, (x0, y0), (x1, y1))
