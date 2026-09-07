"""
Braitenberg-style robot agents.

Implements the classic sensor-to-motor crossed and uncrossed wirings, used as a
minimal reference controller for the 2D robot simulation.
"""

from __future__ import annotations
from typing import Any
import random

import numpy as np

from ... import util
from ...model.modules import RotTriangle

__all__: list[str] = [
    "DifferentialDriveRobot",
    "SensorDrivenRobot",
]

__displayname__ = "Braitenberg agent"


class DifferentialDriveRobot(RotTriangle):
    """
    Two-wheeled differential drive robot with triangular body.

    Implements differential steering kinematics where left/right wheel
    speeds control translation and rotation. Used as base for Braitenberg
    vehicle simulations with sensor-motor coupling.

    Attributes:
        speed_left_wheel: Angular velocity of left wheel (rad/s)
        speed_right_wheel: Angular velocity of right wheel (rad/s)
        wheel_radius: Radius of drive wheels (meters)
        length: Robot body length (meters)
        deltax: X displacement in last timestep
        deltay: Y displacement in last timestep

    Example:
        >>> robot = DifferentialDriveRobot(
        ...     unique_id='robot_1',
        ...     model=sim_model,
        ...     x=0.5, y=0.5,
        ...     length=0.01,
        ...     wheel_radius=0.002
        ... )
        >>> robot.speed_left_wheel = 1.0
        >>> robot.speed_right_wheel = 0.8
        >>> robot.step()  # Update position based on wheel speeds
    """

    def __init__(
        self,
        unique_id: str,
        model: Any,
        x: float,
        y: float,
        length: float,
        wheel_radius: float,
    ) -> None:
        """Build the robot with a random initial heading.

        Args:
            unique_id: The robot identifier.
            model: The simulation model.
            x: The initial x coordinate.
            y: The initial y coordinate.
            length: The wheelbase length.
            wheel_radius: The drive wheel radius.
        """
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

    def step(self) -> None:
        """Updates x, y and direction"""
        self.delta_x()
        self.delta_y()
        self.delta_direction()

    def move_duration(self, seconds: float) -> None:
        """Moves the robot for an 's' amount of seconds"""
        for i in range(int(seconds / self._delta)):
            self.step()

    def print_xyd(self) -> None:
        """Prints the x,y position and direction"""
        print("x = " + str(self.x) + " " + "y = " + str(self.y))
        print("direction = " + str(self.direction))

    def delta_x(self) -> None:
        """Update the x displacement from the two wheel speeds."""
        self.deltax = (
            self._delta
            * (self.wheel_radius * 0.5)
            * (self.speed_right_wheel + self.speed_left_wheel)
            * np.cos(-self.direction)
        )
        self.x += self.deltax

    def delta_y(self) -> None:
        """Update the y displacement from the two wheel speeds."""
        self.deltay = (
            self._delta
            * (self.wheel_radius * 0.5)
            * (self.speed_right_wheel + self.speed_left_wheel)
            * np.sin(-self.direction)
        )
        self.y += self.deltay

    def delta_direction(self) -> None:
        """Update the heading from the difference in wheel speeds."""
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
    """
    Differential drive robot with sensorimotor control.

    Extends DifferentialDriveRobot with bilateral motor controllers
    and sensors, implementing reactive Braitenberg vehicle behaviors
    (obstacle avoidance, phototaxis, etc.) via sensor-motor coupling.

    Attributes:
        left_motor_controller: Left motor controller with sensor
        right_motor_controller: Right motor controller with sensor
        collision_with_object: Collision detection flag
        label: Optional robot label for visualization

    Example:
        >>> robot = SensorDrivenRobot(
        ...     unique_id='braitenberg_1',
        ...     model=sim_model,
        ...     x=0.5, y=0.5,
        ...     length=0.01,
        ...     wheel_radius=0.002
        ... )
        >>> robot.set_left_motor_controller(left_motor)
        >>> robot.set_right_motor_controller(right_motor)
        >>> robot.step()  # Sense and act
    """

    def __init__(self, **kwargs: Any) -> None:
        """Build the robot without motor controllers attached.

        Args:
            **kwargs: Robot attributes, forwarded to the parent class.
        """
        super().__init__(**kwargs)
        self.collision_with_object = False
        self.left_motor_controller = None
        self.right_motor_controller = None
        self.label = None

    def step(self) -> None:
        """Sense, drive the wheels and move, unless already collided."""
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

    def set_left_motor_controller(self, left_motor_controller: Any) -> None:
        """Attach the controller driving the left wheel.

        Args:
            left_motor_controller: The controller to attach.
        """
        self.left_motor_controller = left_motor_controller

    def set_right_motor_controller(self, right_motor_controller: Any) -> None:
        """Attach the controller driving the right wheel.

        Args:
            right_motor_controller: The controller to attach.
        """
        self.right_motor_controller = right_motor_controller

    def draw(self, scene: Any) -> None:
        # draw the sensor lines

        # in scene_loader a robot doesn't have sensors
        """Render the robot and its sensors.

        Args:
            scene: The scene to draw into.
        """
        if self.left_motor_controller is not None:
            self.left_motor_controller.sensor.draw()
            self.right_motor_controller.sensor.draw()

        # call super method to draw the robot
        super().draw(scene)
