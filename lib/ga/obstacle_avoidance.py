import sys
import pygame
import random
# import util.cli_parser

from pygame.locals import *

from lib.ga.obstacle_avoidance.genome import Genome
from lib.ga.util.templates import RunTemplate
from lib.ga.util.scene_type import SceneType






class ObstacleAvoidance(RunTemplate):
    WHEEL_RADIUS = ROBOT_SIZE = 25
    MOTOR_CONTROLLER_COEFFICIENT = 500
    MOTOR_CONTROLLER_MIN_ACTUATOR_VALUE = 10
    SENSOR_DELTA_DIRECTION = 0.44
    SENSOR_SATURATION_VALUE = 85
    SENSOR_MAX_DISTANCE = 120
    SENSOR_ERROR = 0.2

    # DEFAULT_SCENE_FILE = 'saved_scenes/obstacle_avoidance_900.txt'
    SAVED_SCENE_FILENAME = 'obstacle_avoidance_scene'

    def __init__(self, scene_file='saved_scenes/obstacle_avoidance_900.txt', **kwargs):
        super().__init__(scene_file=scene_file, caption="Obstacle avoidance - BRAVE", **kwargs)

    def add_robots(self, number_to_add=1):
        genome = Genome(self.WHEEL_RADIUS, self.MOTOR_CONTROLLER_COEFFICIENT, self.MOTOR_CONTROLLER_MIN_ACTUATOR_VALUE,
                        self.SENSOR_DELTA_DIRECTION, self.SENSOR_SATURATION_VALUE, self.SENSOR_MAX_DISTANCE)
        for i in range(number_to_add):
            x = self.scene.width / 2
            y = self.scene.height / 2
            robot = genome.build_obstacle_avoidance_robot(i,x, y, self.ROBOT_SIZE, self.SENSOR_ERROR, self.scene)
            self.scene.put(robot)
            self.robots.append(robot)
        print('Number of robots:', len(self.robots))

    # def parse_cli_arguments(self):
    #     from lib.ga.util.cli_parser import CliParser
    #     parser = CliParser()
    #     parser.parse_args(self.DEFAULT_SCENE_FILE, self.DEFAULT_SCENE_SPEED, self.scene_type)
    #
    #     self.scene_speed = parser.scene_speed
    #     self.scene_file = parser.scene_file
    #     self.genome_file = parser.genome_file
    #     self.load_all_genomes = parser.load_all_genomes

    def load_genomes_from_file(self):
        n_genomes_loaded = 0
        x = self.scene.width / 2
        y = self.scene.height / 2

        with open(self.genome_file) as f:
            line_number = 1

            for line in f:
                # load only the first N_GENOMES_TO_LOAD genomes (genomes file could be very large)
                if not self.load_all_genomes and n_genomes_loaded == self.N_GENOMES_TO_LOAD_FROM_FILE:
                    print('Loaded ' + str(self.N_GENOMES_TO_LOAD_FROM_FILE) +
                          ' genomes. To load all of them, use --load_all_genomes parameter')
                    break

                values = line.split()

                # skip empty lines
                if len(values) == 0:
                    line_number += 1
                    continue

                # skip comments in file
                if values[0][0] == '#':
                    line_number += 1
                    continue

                robot_wheel_radius = float(values[0])
                motor_ctrl_coefficient = float(values[1])
                motor_ctrl_min_actuator_value = float(values[2])
                sensor_delta_direction = float(values[3])
                sensor_saturation_value = float(values[4])
                sensor_max_distance = float(values[5])

                genome = Genome(robot_wheel_radius, motor_ctrl_coefficient, motor_ctrl_min_actuator_value,
                                sensor_delta_direction, sensor_saturation_value, sensor_max_distance)

                robot = genome.build_obstacle_avoidance_robot(line_number,x, y, self.ROBOT_SIZE, self.SENSOR_ERROR,
                                                              self.scene)
                # robot.label = line_number
                self.robots.append(robot)
                self.scene.put(robot)
                n_genomes_loaded += 1
                line_number += 1
        f.closed

        print('Number of robots:', len(self.robots))


if __name__ == '__main__':
    ObstacleAvoidance(robot_class='larva')
