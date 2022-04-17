import math

from lib.conf.stored.conf import copyConf
from lib.ga.robot.larva_robot import ObstacleLarvaRobot
from lib.ga.robot.sensor_driven_robot import SensorDrivenRobot


class GaRobot(SensorDrivenRobot):

    def __init__(self, genome, **kwargs):
        super().__init__(wheel_radius=genome.robot_wheel_radius, **kwargs)
        self.genome = genome
        self.mileage = 0

    def step(self):
        super().step()
        distance = math.sqrt(math.pow(self.deltax, 2) + math.pow(self.deltay, 2))
        # print('mileage', self.mileage, 'deltax', self.deltax, 'deltay', self.deltay, 'distance', distance)
        self.mileage += distance

class GaLarvaRobot(ObstacleLarvaRobot):

    def __init__(self, genome, **kwargs):
        super().__init__(conf=copyConf('Sakagiannis2022', 'Model').brain, **kwargs)
        self.genome = genome
        self.mileage = 0

    def step(self):
        super().step()
        # distance = math.sqrt(math.pow(self.deltax, 2) + math.pow(self.deltay, 2))
        # print('mileage', self.mileage, 'deltax', self.deltax, 'deltay', self.deltay, 'distance', distance)
        self.mileage += self.dst
