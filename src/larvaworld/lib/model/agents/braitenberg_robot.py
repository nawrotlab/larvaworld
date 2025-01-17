import random

import numpy as np

from ... import util
from ...model.modules.rot_surface import RotTriangle

__all__ = [
    "DifferentialDriveRobot",
    "SensorDrivenRobot",
]

__displayname__ = "Braitenberg agent"


class DifferentialDriveRobot(RotTriangle):
    def __init__(self, unique_id, model, x, y, length, wheel_radius):
        direction = random.uniform(-np.pi, np.pi)
        super().__init__(
            x,
            y,
            length,
            util.Color.random_color(127, 127, 127),
            util.Color.BLACK,
            direction,
        )
        self.model = model
        self.unique_id = unique_id
        self.length = length
        self.wheel_radius = wheel_radius
        self.speed_left_wheel = 0.0  # angular velocity of left wheel
        self.speed_right_wheel = 0.0  # angular velocity of left wheel
        self._delta = 0.01
        self.deltax = None
        self.deltay = None

    def step(self):
        """Updates x, y and direction"""
        self.delta_x()
        self.delta_y()
        self.delta_direction()

    def move_duration(self, seconds):
        """Moves the robot for an 's' amount of seconds"""
        for i in range(int(seconds / self._delta)):
            self.step()

    def print_xyd(self):
        """Prints the x,y position and direction"""
        print("x = " + str(self.x) + " " + "y = " + str(self.y))
        print("direction = " + str(self.direction))

    def delta_x(self):
        self.deltax = (
            self._delta
            * (self.wheel_radius * 0.5)
            * (self.speed_right_wheel + self.speed_left_wheel)
            * np.cos(-self.direction)
        )
        self.x += self.deltax

    def delta_y(self):
        self.deltay = (
            self._delta
            * (self.wheel_radius * 0.5)
            * (self.speed_right_wheel + self.speed_left_wheel)
            * np.sin(-self.direction)
        )
        self.y += self.deltay

    def delta_direction(self):
        self.direction += (
            self._delta
            * (self.wheel_radius / self.length)
            * (self.speed_right_wheel - self.speed_left_wheel)
        )

        if self.direction > np.pi:
            self.direction -= 2 * np.pi
        elif self.direction < -np.pi:
            self.direction += 2 * np.pi


class SensorDrivenRobot(DifferentialDriveRobot):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.collision_with_object = False
        self.left_motor_controller = None
        self.right_motor_controller = None
        self.label = None

    def step(self):
        if not self.collision_with_object:
            try:
                self.left_motor_controller.sense_and_act()
                self.right_motor_controller.sense_and_act()
                self.speed_left_wheel = self.left_motor_controller.get_actuator_value()
                self.speed_right_wheel = (
                    self.right_motor_controller.get_actuator_value()
                )
                self.step()
            except util.Collision:
                self.collision_with_object = True
                self.speed_left_wheel = 0
                self.speed_right_wheel = 0
        else:
            # a collision has already occured
            self.speed_left_wheel = 0
            self.speed_right_wheel = 0

    def set_left_motor_controller(self, left_motor_controller):
        self.left_motor_controller = left_motor_controller

    def set_right_motor_controller(self, right_motor_controller):
        self.right_motor_controller = right_motor_controller

    def draw(self, scene):
        # draw the sensor lines

        # in scene_loader a robot doesn't have sensors
        if self.left_motor_controller is not None:
            self.left_motor_controller.sensor.draw()
            self.right_motor_controller.sensor.draw()

        # call super method to draw the robot
        super().draw(scene)
