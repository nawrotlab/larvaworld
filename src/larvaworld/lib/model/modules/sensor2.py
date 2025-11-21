from __future__ import annotations
from typing import Any

# import math
import random
from math import atan2, cos, sin

import pygame
from shapely import geometry

from ... import util
from .rot_surface import LightSource

__all__: list[str] = [
    "Sensor2",
    "LightSensor",
    "ProximitySensor",
]


class Sensor2:
    """
    Base class for robot sensors in 2D simulation.

    Provides interface for sensor value reading and visualization
    with configurable orientation, saturation, and noise.

    Attributes:
        robot: Parent robot instance.
        delta_direction: Sensor orientation offset in radians.
        saturation_value: Maximum sensor reading value.
        error: Sensor noise/error magnitude.
        value: Current sensor reading.

    Example:
        >>> sensor = Sensor2(
        ...     robot=my_robot,
        ...     delta_direction=0.5,
        ...     saturation_value=1.0,
        ...     error=0.1
        ... )
    """

    def __init__(
        self, robot: Any, delta_direction: float, saturation_value: float, error: float
    ) -> None:
        self.robot = robot
        self.delta_direction = delta_direction
        self.saturation_value = saturation_value
        self.error = error
        self.value = 0

    def get_value(self) -> float:
        # defined by subclasses
        return 0.0

    def draw(self) -> None:
        # defined by subclasses
        return None


class LightSensor(Sensor2):
    """
    Light intensity sensor for robot navigation.

    Detects light sources in simulation environment, calculating
    intensity based on distance and direction using ray-casting.

    Attributes:
        LENGTH_SENSOR_LINE: Ray-casting distance for sensor (100 units).

    Example:
        >>> light_sensor = LightSensor(
        ...     robot=my_robot,
        ...     delta_direction=0.0,
        ...     saturation_value=100.0,
        ...     error=0.05,
        ...     scene=sim_scene
        ... )
    """

    LENGTH_SENSOR_LINE = 100

    def __init__(
        self,
        robot: Any,
        delta_direction: float,
        saturation_value: float,
        error: float,
        scene: Any,
    ):
        super().__init__(robot, delta_direction, saturation_value, error)

    def get_value(self) -> float:
        dir_sensor = self.robot.direction + self.delta_direction
        total_value = 0

        for obj in self.robot.model.obstacles:
            # for obj in self.robot.model.viewer.objects:
            if issubclass(type(obj), LightSource):
                light = obj

                # cambio SDR
                x_robot, y_robot = self.robot.model.viewer._transform(self.robot.pos)

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

    def draw(self) -> None:
        dir_sensor = self.robot.direction + self.delta_direction
        x_sensor_eol = self.robot.x + self.LENGTH_SENSOR_LINE * cos(dir_sensor)
        y_sensor_eol = self.robot.y + self.LENGTH_SENSOR_LINE * -sin(dir_sensor)

        pygame.draw.line(
            self.robot.model.viewer.v,
            util.Color.YELLOW,
            (self.robot.x, self.robot.y),
            (x_sensor_eol, y_sensor_eol),
        )


class ProximitySensor(Sensor2):
    """
    Proximity sensor for obstacle detection.

    Detects nearby obstacles using ray-casting, returning distance
    to nearest obstacle in sensor direction.

    Attributes:
        max_distance: Maximum detection range.
        collision_distance: Distance threshold for collision (default: 12).

    Example:
        >>> prox_sensor = ProximitySensor(
        ...     robot=my_robot,
        ...     delta_direction=-0.785,
        ...     saturation_value=50.0,
        ...     error=0.1,
        ...     max_distance=100
        ... )
    """

    # COLLISION_DISTANCE = 12  # px

    def __init__(
        self,
        robot: Any,
        delta_direction: float,
        saturation_value: float,
        error: float,
        max_distance: int,
        collision_distance: int = 12,
    ) -> None:
        super().__init__(robot, delta_direction, saturation_value, error)
        self.max_distance = max_distance
        self.collision_distance = collision_distance
        # print(max_distance)

        # raise

    def get_value(
        self, pos: list[float] | None = None, direction: float | None = None
    ) -> float:
        if pos is None:
            pos = [self.robot.x, self.robot.y]

        if direction is None:
            direction = self.robot.direction

        angle = -direction - self.delta_direction
        p0 = geometry.Point(pos)
        p1 = geometry.Point(
            p0.x + cos(angle) * self.max_distance, p0.y + sin(angle) * self.max_distance
        )
        min_dst, nearest_obstacle = util.detect_nearest_obstacle(
            self.robot.model.obstacles, (p0, p1), p0
        )

        if min_dst is None:
            # no obstacle detected
            return 0
        else:
            # check collision
            if min_dst < self.collision_distance:
                raise util.Collision(self.robot, nearest_obstacle)

            proximity_value = 1 / random.gauss(min_dst, self.error * min_dst)

            if proximity_value > self.saturation_value:
                return self.saturation_value
            else:
                return proximity_value

    def draw(
        self, pos: list[float] | None = None, direction: float | None = None
    ) -> None:
        if pos is None:
            pos = [self.robot.x, self.robot.y]
        if direction is None:
            direction = self.robot.direction
        x0, y0 = pos
        angle = -direction - self.delta_direction

        x1 = x0 + cos(angle) * self.max_distance
        y1 = y0 + sin(angle) * self.max_distance

        # self.scene.draw_line((x, y), (x_sensor_eol, y_sensor_eol),Color.RED, width=0.0005)
        pygame.draw.line(self.robot.model.viewer.v, util.Color.RED, (x0, y0), (x1, y1))
