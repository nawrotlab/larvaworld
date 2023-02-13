# import math
import random
import pygame
from math import atan2, sin, cos
from shapely import geometry

from larvaworld.lib import aux
from larvaworld.lib.model.modules.rot_surface import LightSource


class Sensor2:

    def __init__(self, robot, delta_direction, saturation_value, error):
        self.robot = robot
        self.delta_direction = delta_direction
        self.saturation_value = saturation_value
        self.error = error
        self.value = 0

    def get_value(self):
        # defined by subclasses
        pass

    def draw(self):
        # defined by subclasses
        pass



class LightSensor(Sensor2):

    LENGTH_SENSOR_LINE = 100

    def __init__(self, robot, delta_direction, saturation_value, error, scene):
        super().__init__(robot, delta_direction, saturation_value, error)

    def get_value(self):
        dir_sensor = self.robot.direction + self.delta_direction
        total_value = 0

        for obj in self.robot.model.obstacles:
        # for obj in self.robot.model.viewer.objects:
            if issubclass(type(obj), LightSource):
                light = obj

                # cambio SDR
                x_robot = self.robot.xx
                y_robot = -self.robot.yy
                x_light = light.x
                y_light = -light.y

                x_light -= x_robot
                y_light -= y_robot

                dir_light = atan2(y_light, x_light)
                difference_dir = dir_sensor - dir_light
                angle_sensor_light = atan2(sin(difference_dir), cos(difference_dir))
                value = cos(angle_sensor_light) * light.emitting_power

                if value > 0:
                    total_value += value

        if total_value > self.saturation_value:
            return self.saturation_value
        else:
            # percentage standard deviation
            percentage_std_dev = self.error * total_value
            total_value_with_error = random.gauss(total_value, percentage_std_dev)
            return total_value_with_error

    def draw(self):
        dir_sensor = self.robot.direction + self.delta_direction
        x_sensor_eol = self.robot.x + self.LENGTH_SENSOR_LINE * cos(dir_sensor)
        y_sensor_eol = self.robot.y + self.LENGTH_SENSOR_LINE * -sin(dir_sensor)
        # viewer.draw_line(pos, xy_aux.xy_projection(pos, orientation, radius * 3),
        #                  color=color, width=radius / 10)
        pygame.draw.line(self.robot.model.viewer._window, aux.Color.YELLOW, (self.robot.x, self.robot.y), (x_sensor_eol, y_sensor_eol))


class ProximitySensor(Sensor2):

    # COLLISION_DISTANCE = 12  # px

    def __init__(self, robot, delta_direction, saturation_value, error, max_distance, collision_distance=12):
        super().__init__(robot, delta_direction, saturation_value, error)
        self.max_distance = max_distance
        self.collision_distance = collision_distance
        # print(max_distance)

        # raise

    def get_value(self, pos=None, direction=None):
        if pos is None:
            pos = [self.robot.x, self.robot.y]

        if direction is None:
            direction = self.robot.direction

        angle = -direction - self.delta_direction
        p0 = geometry.Point(pos)
        p1 = geometry.Point(
            p0.x + cos(angle) * self.max_distance,
            p0.y + sin(angle) * self.max_distance)
        min_dst, nearest_obstacle = aux.detect_nearest_obstacle(self.robot.model.obstacles, (p0,p1), p0)

        if min_dst is None:
            # no obstacle detected
            return 0
        else:
            # check collision
            if min_dst < self.collision_distance:
                raise aux.Collision(self.robot, nearest_obstacle)

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

        x1=x0 + cos(angle) * self.max_distance
        y1=y0 + sin(angle) * self.max_distance

        # self.scene.draw_line((x, y), (x_sensor_eol, y_sensor_eol),Color.RED, width=0.0005)
        pygame.draw.line(self.robot.model.viewer._window, aux.Color.RED, (x0, y0), (x1, y1))
