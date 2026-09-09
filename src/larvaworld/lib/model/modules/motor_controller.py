"""
Sensorimotor coupling for the 2D robot agents.

Provides the controller mapping a sensor reading onto a motor command, and the
actuator that smooths that command over time.
"""

from __future__ import annotations

from typing import Any

__all__: list[str] = [
    "MotorController",
    "Actuator",
]


class MotorController:
    """
    Maps a sensor reading onto a motor command.

    Scales the sensor's value by a coefficient and adds a baseline, then drives
    an :class:`Actuator` that smooths the result over time.
    """

    def __init__(
        self,
        sensor: Any,
        coefficient: float,
        actuator: "Actuator",
        min_actuator_value: float,
    ) -> None:
        """Build the controller.

        Args:
            sensor: The sensor driving this motor.
            coefficient: The gain applied to the sensor reading.
            actuator: The actuator smoothing the command.
            min_actuator_value: The baseline command added to the scaled
                sensor reading.
        """
        self.sensor = sensor
        self.actuator = actuator
        self.coefficient = coefficient
        self.min_actuator_value = min_actuator_value

    def sense_and_act(self, **kwargs: Any) -> None:
        """Read the sensor and drive the actuator with the scaled value.

        Args:
            **kwargs: Forwarded to the sensor reading.
        """
        sensor_value = self.sensor.get_value(**kwargs)
        weighted_value = self.coefficient * sensor_value
        self.actuator.value = weighted_value + self.min_actuator_value

    def get_actuator_value(self) -> float:
        """Return the actuator's current value."""
        return float(self.actuator.value)


class Actuator:
    """
    Smooths a motor command over successive timesteps.

    Accumulates the commanded values and decays them, so that the driven motor
    responds gradually rather than tracking the sensor instantaneously.
    """

    def __init__(self) -> None:
        """Build the actuator with a zeroed value."""
        self.value: float = 0.0
