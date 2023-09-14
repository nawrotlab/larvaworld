
__all__ = [
    'MotorController',
    'Actuator',
]

class MotorController:

    def __init__(self, sensor, coefficient, actuator, min_actuator_value):
        self.sensor = sensor
        self.actuator = actuator
        self.coefficient = coefficient
        self.min_actuator_value = min_actuator_value

    def sense_and_act(self, **kwargs):
        sensor_value = self.sensor.get_value(**kwargs)
        weighted_value = self.coefficient * sensor_value
        self.actuator.value = weighted_value + self.min_actuator_value

    def get_actuator_value(self):
        return self.actuator.value

class Actuator:

    def __init__(self):
        self.value = 0
