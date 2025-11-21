from __future__ import annotations

from typing import Any

__all__: list[str] = [
    "MotorController",
    "Actuator",
]


class MotorController:
    def __init__(
        self,
        sensor: Any,
        coefficient: float,
        actuator: "Actuator",
        min_actuator_value: float,
    ) -> None:
        self.sensor = sensor
        self.actuator = actuator
        self.coefficient = coefficient
        self.min_actuator_value = min_actuator_value

    def sense_and_act(self, **kwargs: Any) -> None:
        sensor_value = self.sensor.get_value(**kwargs)
        weighted_value = self.coefficient * sensor_value
        self.actuator.value = weighted_value + self.min_actuator_value

    def get_actuator_value(self) -> float:
        return float(self.actuator.value)


class Actuator:
    def __init__(self) -> None:
        self.value: float = 0.0
