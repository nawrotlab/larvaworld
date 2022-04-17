import random
import multiprocessing
import math

from lib.conf.base.dtypes import null_dict
from lib.ga.obstacle_avoidance.ga_robot import GaRobot, GaLarvaRobot
from lib.ga.obstacle_avoidance.genome import Genome
from lib.ga.robot.actuator import Actuator
from lib.ga.robot.motor_controller import MotorController
from lib.ga.scene.box import Box
from lib.ga.sensor.proximity_sensor import ProximitySensor
from lib.ga.util.color import Color
from lib.ga.util.templates import GAEngineTemplate
from lib.ga.util.thread_ga_robot import ThreadGaRobot
from lib.ga.util.time_util import TimeUtil


class GaEngine(GAEngineTemplate):
    ROBOT_SIZE = 25

    def __init__(self, **kwargs):
        super().__init__(genome_class=Genome, **kwargs)

    def build_robot(self, x, y, genome, label):
        if self.robot_class=='larva':
            robot = GaLarvaRobot(unique_id=label, model=self,x=x, y=y, genome=genome)
        elif self.robot_class=='triangle':
            robot = GaRobot(unique_id=label, model=self,x=x, y=y, length=self.ROBOT_SIZE, genome=genome)

        if not self.robot_random_direction:
            robot.direction = 0

        left_obstacle_sensor = ProximitySensor(robot, genome.sensor_delta_direction, genome.sensor_saturation_value,
                                               self.obstacle_sensor_error, genome.sensor_max_distance, self.scene)
        right_obstacle_sensor = ProximitySensor(robot, -genome.sensor_delta_direction, genome.sensor_saturation_value,
                                                self.obstacle_sensor_error, genome.sensor_max_distance, self.scene)
        left_wheel_actuator = Actuator()
        right_wheel_actuator = Actuator()
        left_motor_controller = MotorController(left_obstacle_sensor, genome.motor_ctrl_coefficient,
                                                left_wheel_actuator, genome.motor_ctrl_min_actuator_value)
        right_motor_controller = MotorController(right_obstacle_sensor, genome.motor_ctrl_coefficient,
                                                 right_wheel_actuator, genome.motor_ctrl_min_actuator_value)

        robot.set_left_motor_controller(left_motor_controller)
        robot.set_right_motor_controller(right_motor_controller)
        robot.label = label

        return robot

    def get_fitness(self, robot):
        # robot.finalize()
        # return -np.sum([ks_2samp(self.eval[p], robot.eval[p]) for p in self.eval_shorts])
        return robot.mileage
