import sys
import pygame


from pygame.locals import *

from lib.ga.obstacle_avoidance.ga_engine import GaEngine
from lib.ga.obstacle_avoidance.ga_robot import GaRobot

from lib.ga.util.templates import GALauncher
from lib.ga.util.scene_type import SceneType


class ObstacleAvoidanceGA(GALauncher):

    # DEFAULT_SCENE_FILE = 'saved_scenes/obstacle_avoidance_900.txt'

    def __init__(self,scene_file='saved_scenes/obstacle_avoidance_900.txt',**kwargs):
        super().__init__(scene_file=scene_file,
                         caption="GA Obstacle avoidance",
                         GA_engine=GaEngine, **kwargs)

    def display_info(self):
        self.side_panel.display_ga_info()


if __name__ == '__main__':
    ObstacleAvoidanceGA()
