from __future__ import annotations
from typing import Any
import os
import warnings

# Deprecation: discourage deep imports from internal module paths
if os.getenv("LARVAWORLD_STRICT_DEPRECATIONS") == "1":
    raise ImportError(
        "Deep import path deprecated. Use public API: 'from larvaworld.lib.model.agents import LarvaRobot, ObstacleLarvaRobot'"
    )
else:
    warnings.warn(
        "Deep import path deprecated. Use public API: 'from larvaworld.lib.model.agents import LarvaRobot, ObstacleLarvaRobot'",
        DeprecationWarning,
        stacklevel=2,
    )
from ... import util
from ...model.modules import Actuator, MotorController, ProximitySensor
from ...param import PositiveNumber

# ScreenManager import deferred due to circular dependency - will be imported when needed
from . import LarvaSim

__all__: list[str] = [
    "LarvaRobot",
    "ObstacleLarvaRobot",
]

__displayname__ = "Braitenberg-like larva"


class LarvaRobot(LarvaSim):
    """
    Virtual larva agent for genetic algorithm optimization.

    Extends LarvaSim with genome-based parameter encoding, enabling
    evolutionary optimization of behavioral parameters via GA. The genome
    represents evolvable traits (locomotor gains, sensor weights, etc.).

    Attributes:
        genome: Parameter genome vector for GA optimization (optional)

    Args:
        larva_pars: Dictionary of larva configuration parameters
        genome: Optional genome encoding of evolvable parameters
        **kwargs: Additional agent configuration

    Example:
        >>> genome = np.array([0.5, 1.2, 0.8])  # Evolvable parameters
        >>> robot = LarvaRobot(larva_pars={'model': 'explorer'}, genome=genome)
    """

    def __init__(
        self, larva_pars: dict[str, Any], genome: Any | None = None, **kwargs: Any
    ) -> None:
        super().__init__(**larva_pars, **kwargs)
        self.genome = genome


class ObstacleLarvaRobot(LarvaRobot):
    """
    Obstacle-avoiding larva robot with Braitenberg-like sensorimotor coupling.

    Extends LarvaRobot with bilateral proximity sensors and motor controllers
    that implement reactive obstacle avoidance through differential actuation.
    Uses sensor-motor coupling to generate turning away from obstacles.

    Attributes:
        Lmotor: Left motor controller with proximity sensor
        Rmotor: Right motor controller with proximity sensor
        sensor_delta_direction: Angular offset of sensors from midline (radians)
        sensor_saturation_value: Max sensor response value
        obstacle_sensor_error: Sensor noise/error magnitude
        sensor_max_distance: Maximum sensing distance
        motor_coefficient: Motor gain coefficient
        min_actuator_value: Minimum motor output value

    Example:
        >>> robot = ObstacleLarvaRobot(
        ...     larva_pars={'model': 'navigator'},
        ...     sensor_delta_direction=0.4,
        ...     motor_coefficient=8770.0
        ... )
        >>> robot.sense()  # Detect obstacles and adjust motors
    """

    sensor_delta_direction = PositiveNumber(0.4, doc="Sensor delta_direction")
    sensor_saturation_value = PositiveNumber(40.0, doc="Sensor saturation value")
    obstacle_sensor_error = PositiveNumber(0.35, doc="Proximity sensor error")
    sensor_max_distance = PositiveNumber(0.9, doc="Sensor max_distance")
    motor_coefficient = PositiveNumber(8770.0, doc="Motor ctrl_coefficient")
    min_actuator_value = PositiveNumber(35.0, doc="Motor ctrl_min_actuator_value")

    def __init__(self, larva_pars: dict[str, Any], **kwargs: Any) -> None:
        kws = larva_pars.sensorimotor
        larva_pars.pop("sensorimotor", None)
        super().__init__(larva_pars=larva_pars, **kws, **kwargs)
        S_kws = {
            "robot": self,
            "saturation_value": self.sensor_saturation_value,
            "error": self.obstacle_sensor_error,
            "max_distance": int(
                self.model.screen_manager._scale[0, 0]
                * self.sensor_max_distance
                * self.length
            ),
            "collision_distance": int(
                self.model.screen_manager._scale[0, 0] * self.length / 5
            ),
        }

        M_kws = {
            "coefficient": self.motor_coefficient,
            "min_actuator_value": self.min_actuator_value,
        }

        self.Lmotor = MotorController(
            sensor=ProximitySensor(
                delta_direction=self.sensor_delta_direction, **S_kws
            ),
            actuator=Actuator(),
            **M_kws,
        )
        self.Rmotor = MotorController(
            sensor=ProximitySensor(
                delta_direction=-self.sensor_delta_direction, **S_kws
            ),
            actuator=Actuator(),
            **M_kws,
        )

    def sense(self) -> None:
        """
        This method allows the larva robot to sense its environment and act accordingly.
        If there is no collision with an object, it transforms the olfactor position and
        uses the left and right motors to sense and act based on the position and direction.
        It then calculates the torque difference between the right and left motors and
        adjusts the neural oscillator's parameters accordingly. If a collision is detected,
        it sets the collision flag to True and interrupts the locomotion.

        Raises:
            util.Collision: If a collision with an object is detected.
        """
        if not self.collision_with_object:
            pos = self.model.screen_manager._transform(self.olfactor_pos)
            try:
                self.Lmotor.sense_and_act(pos=pos, direction=self.direction)
                self.Rmotor.sense_and_act(pos=pos, direction=self.direction)
                Ltorque = self.Lmotor.get_actuator_value()
                Rtorque = self.Rmotor.get_actuator_value()
                dRL = Rtorque - Ltorque
                if dRL > 0:
                    self.brain.locomotor.turner.neural_oscillator.E_r += (
                        dRL * self.model.dt
                    )
                else:
                    self.brain.locomotor.turner.neural_oscillator.E_l -= (
                        dRL * self.model.dt
                    )
            except util.Collision:
                self.collision_with_object = True
                self.brain.locomotor.intermitter.interrupt_locomotion()
        else:
            pass

    def draw(self, v: Any, **kwargs: Any) -> None:
        """
        Draws the larva robot and its sensors on the screen.

        Args:
            v (ScreenManager): The screen manager responsible for rendering.
            **kwargs: Additional keyword arguments passed to the drawing method.

        Notes:
            - The method transforms the olfactor position using the screen manager.
            - If the robot has motors, it draws the sensors for both the left and right motors.
            - Finally, it calls the superclass's draw method to render the robot itself.
        """
        pos = v._transform(self.olfactor_pos)

        # draw the sensor lines
        if self.Lmotor is not None:
            self.Lmotor.sensor.draw(pos=pos, direction=self.direction)
            self.Rmotor.sensor.draw(pos=pos, direction=self.direction)

        # call super method to draw the robot
        super().draw(v, **kwargs)
