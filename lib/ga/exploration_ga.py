import sys
import pygame


from pygame.locals import *

from lib.conf.base.dtypes import null_dict
from lib.ga.exploration.ga_engine import GaLarvaEngine
from lib.ga.robot.larvaConfDic import LarvaConfDic
from lib.ga.robot.larva_robot import LarvaRobot
from lib.ga.util.templates import GATemplate



class ExplorationGA(GATemplate):


    def __init__(self,robot_class='larva',scene_file='saved_scenes/no_boxes.txt',**kwargs):
        super().__init__(scene_file = scene_file,robot_class=robot_class,
                         caption="GA Larva free exploration",
                         GA_engine=GaLarvaEngine, **kwargs)

    def display_info(self):
        self.side_panel.display_ga_info_larva(LarvaConfDic)



if __name__ == '__main__':
    ExplorationGA(population_num=10, dt=1/16, arena=null_dict('arena', arena_dims=(0.5, 0.5), arena_shape='rectangular'),
                  GA_engine_kws={'eval_shorts':['b', 'fov', 'foa','tur_fou','tur_fov_max', 'v', 'a','run_d', 'run_t','pau_t','tor5', 'tor20', ]})
