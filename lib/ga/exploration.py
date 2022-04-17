import sys
import pygame
import random
import util.cli_parser

from pygame.locals import *

from lib.conf.base.dtypes import null_dict
from lib.ga.exploration.genome import LarvaGenome
from lib.ga.geometry.point import Point
from lib.ga.robot.larvaConfDic import LarvaConfDic
from lib.ga.robot.larva_robot import LarvaRobot
from lib.ga.scene.box import Box
from lib.ga.scene.scene import Scene
from lib.ga.scene.wall import Wall
from lib.ga.util.color import Color
from lib.ga.util.templates import RunTemplate
from lib.ga.util.scene_type import SceneType
from lib.ga.util.side_panel import SidePanel


class Exploration(RunTemplate):


    # DEFAULT_GENOMES_FILE = None
    DEFAULT_GENOMES_FILE = 'saved_genomes/Sakagiannis2022_optimization_30generations_x_30larvae_x_1000steps.txt'
    DEFAULT_SCENE_FILE = 'saved_scenes/no_boxes.txt'
    SAVED_SCENE_FILENAME = 'exploration_scene'


    def __init__(self, scene_file='saved_scenes/no_boxes.txt', **kwargs):
        super().__init__(scene_file=scene_file,  caption="Larva free exploration", **kwargs)



    def add_robots(self, number_to_add=1):
        genome = LarvaGenome(**{key : v0 for key,[v0, (r0,r1), lab] in LarvaConfDic.items()})
        for i in range(number_to_add):
            robot = genome.build_larva_robot(unique_id=i, model=self)
            self.scene.put(robot)
            self.robots.append(robot)
        print('Number of robots:', len(self.robots))


    def parse_cli_arguments(self):
        from lib.ga.util.cli_parser import CliParser
        parser = CliParser()
        parser.parse_args(self.DEFAULT_SCENE_FILE, self.DEFAULT_SCENE_SPEED, self.scene_type)

        self.scene_speed = parser.scene_speed
        self.scene_file = parser.scene_file
        self.genome_file = parser.genome_file
        self.load_all_genomes = parser.load_all_genomes

    def load_genomes_from_file(self):
        n_genomes_loaded = 0
        # x = self.scene.width / 2
        # y = self.scene.height / 2

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

                genome = LarvaGenome(**{key : float(values[i]) for i, key in enumerate(LarvaConfDic.keys())})

                robot = genome.build_larva_robot(unique_id=line_number,model=self)
                self.robots.append(robot)
                self.scene.put(robot)
                n_genomes_loaded += 1
                line_number += 1
        f.closed

        print('Number of loaded robots:', len(self.robots))


if __name__ == '__main__':
    Exploration(robot_class='larva', dt=1/16, arena=null_dict('arena', arena_dims=(0.2, 0.2), arena_shape='rectangular'))
