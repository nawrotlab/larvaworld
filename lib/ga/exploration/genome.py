import random
import math

from unflatten import unflatten

import lib.aux.dictsNlists as dNl
from lib.conf.stored.conf import expandConf, copyConf
from lib.ga.robot.actuator import Actuator
from lib.ga.robot.larvaConfDic import LarvaConfDic
from lib.ga.robot.larva_robot import LarvaRobot, ObstacleLarvaRobot
from lib.ga.robot.motor_controller import MotorController
from lib.ga.sensor.proximity_sensor import ProximitySensor



class LarvaGenome:

    def __init__(self, generation_num=None,base_model='Sakagiannis2022', **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.generation_num = generation_num
        self.fitness = None
        self.base_conf= copyConf(base_model, 'Model').brain

    @staticmethod
    def random(generation_num):
        kws = {key: random.uniform(r0, r1) for key, [v0, (r0, r1), lab] in LarvaConfDic.items()}
        return LarvaGenome(**kws, generation_num=generation_num)

    def crossover(self, other_parent, generation_num):
        kws = {key: getattr(self, key) if random.random() < 0.5 else getattr(other_parent, key) for key in
               LarvaConfDic.keys()}
        # apply uniform crossover to generate a new genome
        return LarvaGenome(**kws, generation_num=generation_num)

    def mutation(self, mutation_probability, mutation_coefficient):
        for key in LarvaConfDic.keys():
            v = getattr(self, key)
            setattr(self, key, self.mutate_with_probability(v, mutation_probability, mutation_coefficient))
        self.check_parameter_bounds()

    def mutate_with_probability(self, value, mutation_probability, mutation_coefficient):
        if random.random() < mutation_probability:
            mutation_std_dev = mutation_coefficient * value
            return random.gauss(value, mutation_std_dev)
        else:
            return value

    def check_parameter_bounds(self):
        for key, [v0, (r0, r1), lab] in LarvaConfDic.items():
            v = getattr(self, key)
            if v < r0:
                setattr(self, key, r0)
            elif v > r1:
                setattr(self, key, r1)

    def __repr__(self):
        fitness_value = None if self.fitness is None else round(self.fitness, 2)
        return self.__class__.__name__ + '(fitness:' + repr(fitness_value) + ' generation_num:' + repr(
            self.generation_num) + ')'

    def to_string(self):
        fitness_value = None if self.fitness is None else round(self.fitness, 2)
        kwstrings = [f' {key}:' + repr(round(getattr(self, key), 2)) for key in LarvaConfDic.keys()]
        kwstr = ''
        for ii in kwstrings:
            kwstr = kwstr + ii

        return self.__class__.__name__ + '(fitness:' + repr(fitness_value) + ' generation_num:' + repr(
            self.generation_num) + kwstr + ')'

    def get_saved_genome_repr(self):
        kwstr = ''
        for key in LarvaConfDic.keys():
            kwstr = kwstr + str(getattr(self, key)) + ' '
        return kwstr + str(self.generation_num) + ' ' + str(self.fitness)

    def build_larva_robot(self, unique_id, model, conf=None, robot_class=LarvaRobot):
        if conf is None:
            conf = self.base_conf

        conf_f = dNl.flatten_dict(conf, parent_key='conf', sep='.')
        for key in LarvaConfDic.keys():
            conf_f[key] = getattr(self, key)
        kws = dNl.AttrDict.from_nested_dicts(unflatten(conf_f))
        robot = robot_class(unique_id=unique_id, model=model, **kws)
        return robot

    def build_obstacle_larva_robot(self,sensor_delta_direction,sensor_saturation_value,sensor_max_distance,
                                   sensor_error,motor_ctrl_coefficient,motor_ctrl_min_actuator_value, scene, **kwargs):

        robot = self.build_larva_robot(robot_class=ObstacleLarvaRobot, **kwargs)
        # robot = ObstacleLarvaRobot(**kwargs)

        left_obstacle_sensor = ProximitySensor(robot, sensor_delta_direction, sensor_saturation_value,
                                               sensor_error, sensor_max_distance, scene)
        right_obstacle_sensor = ProximitySensor(robot, -sensor_delta_direction, sensor_saturation_value,
                                                sensor_error, sensor_max_distance, scene)
        left_wheel_actuator = Actuator()
        right_wheel_actuator = Actuator()
        left_motor_controller = MotorController(left_obstacle_sensor, motor_ctrl_coefficient, left_wheel_actuator,
                                                motor_ctrl_min_actuator_value)
        right_motor_controller = MotorController(right_obstacle_sensor, motor_ctrl_coefficient, right_wheel_actuator,
                                                 motor_ctrl_min_actuator_value)

        robot.set_left_motor_controller(left_motor_controller)
        robot.set_right_motor_controller(right_motor_controller)
        return robot

