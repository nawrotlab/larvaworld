import math
import os
import sys

import numpy as np

from lib.aux.sim_aux import get_source_xy
from lib.registry.pars import preg
from lib.registry import reg
from lib.model.space.scene import Scene
from lib.aux.color_util import Color
from lib.sim.ga.ga_engine import GAbuilder
from lib.aux.time_util import TimeUtil
from lib.model.envs._base_larvaworld import BaseLarvaWorld


class BaseGAlauncher(BaseLarvaWorld):
    SCENE_MAX_SPEED = 3000

    SCENE_MIN_SPEED = 1
    SCENE_SPEED_CHANGE_COEFF = 1.5

    SIDE_PANEL_WIDTH = 600

    # SCREEN_MARGIN = 12

    def __init__(self, sim_params, scene='no_boxes', scene_speed=0, env_params=None, experiment='exploration',
                 caption=None, save_to=None, offline=False, **kwargs):
        self.offline = offline
        id = sim_params.sim_ID
        self.sim_params = sim_params
        dt = sim_params.timestep
        Nsteps = int(sim_params.duration * 60 / dt)
        if save_to is None:
            save_to = reg.Path.SIM
        self.save_to = save_to
        self.dir_path = f'{save_to}/{sim_params.path}/{id}'
        self.plot_dir = f'{self.dir_path}/plots'
        os.makedirs(self.plot_dir, exist_ok=True)
        if not self.offline:
            super().__init__(id=id, dt=dt, Box2D=sim_params.Box2D, env_params=env_params,
                             save_to=f'{self.dir_path}/visuals',
                             Nsteps=Nsteps, experiment=experiment, **kwargs)
            self.arena_width, self.arena_height = self.env_pars.arena.arena_dims
        else:
            self.env_pars = env_params
            self.scaling_factor = 1
            X, Y = self.arena_width, self.arena_height = self.env_pars.arena.arena_dims
            self.space_edges_for_screen = np.array([-X / 2, X / 2, -Y / 2, Y / 2])
            self.experiment = experiment
            self.dt = dt
            self.Nticks = 0
            self.Nsteps = Nsteps
            self.id = id
            self.save_to = save_to
            self.Box2D = False
        self.source_xy = get_source_xy(self.env_pars.food_params)
        if caption is None:
            caption = f'GA {experiment} : {self.id}'
        self.caption = caption
        self.scene_file = f'{reg.Path.ga_scene}/{scene}.txt'
        self.scene_speed = scene_speed
        self.obstacles = []

    def build_box(self, x, y, size, color):
        from lib.model.space.obstacle import Box

        box = Box(x, y, size, color)
        self.obstacles.append(box)
        return box

    def build_wall(self, point1, point2, color):
        from lib.model.space.obstacle import Wall
        wall = Wall(point1, point2, color)
        self.obstacles.append(wall)
        return wall

    def get_larvaworld_food(self):

        for label,ff in self.env_pars.food_params.source_units.items():
            # pos=dic['pos']
            x, y = self.screen_pos(ff.pos)
            size = ff.radius * self.scaling_factor
            col = ff.default_color
            box = self.build_box(x, y, size, col)
            box.label = label
            self.scene.put(box)

    def screen_pos(self, real_pos):
        return np.array(real_pos) * self.scaling_factor + np.array([self.scene.width / 2, self.scene.height / 2])

    def init_scene(self):
        self.scene = Scene.load_from_file(self.scene_file, self.scene_speed, self.SIDE_PANEL_WIDTH)

        self.scene.set_bounds(*self.space_edges_for_screen)

    def increase_scene_speed(self):
        if self.scene.speed < self.SCENE_MAX_SPEED:
            self.scene.speed *= self.SCENE_SPEED_CHANGE_COEFF
        print('scene.speed:', self.scene.speed)

    def decrease_scene_speed(self):
        if self.scene.speed > self.SCENE_MIN_SPEED:
            self.scene.speed /= self.SCENE_SPEED_CHANGE_COEFF
        print('scene.speed:', self.scene.speed)


class GAlauncher(BaseGAlauncher):
    def __init__(self, ga_build_kws, ga_select_kws, show_screen=True, **kwargs):
        super().__init__(**kwargs)
        if self.offline:
            from lib.model.robot.larva_offline import LarvaOffline
            ga_build_kws.robot_class = LarvaOffline
            show_screen = False
        elif ga_build_kws.robot_class is None :
            from lib.model.robot.larva_robot import LarvaRobot
            ga_build_kws.robot_class = LarvaRobot
        self.ga_build_kws = ga_build_kws
        self.ga_select_kws = ga_select_kws
        self.show_screen = show_screen
        if show_screen:
            import pygame

            pygame.init()
            pygame.display.set_caption(self.caption)
            self.clock = pygame.time.Clock()
        self.initialize(**ga_build_kws, **ga_select_kws)

    def run(self):
        while True and self.engine.is_running:
            from pygame import KEYDOWN, K_ESCAPE, K_r, K_MINUS, K_PLUS, K_s, K_e, QUIT,event, Rect, draw, display

            # t0 = TimeUtil.current_time_millis()
            self.engine.step()
            # t1 = TimeUtil.current_time_millis()
            # self.printd(2, 'Step duration: ', t1 - t0)
            if self.show_screen:
                for e in event.get():
                    if e.type == QUIT or (e.type == KEYDOWN and e.key == K_ESCAPE):
                        sys.exit()
                    elif e.type == KEYDOWN and e.key == K_r:
                        self.initialize(**self.ga_select_kws, **self.ga_build_kws)
                    elif e.type == KEYDOWN and (e.key == K_PLUS or e.key == 93 or e.key == 270):
                        self.increase_scene_speed()
                    elif e.type == KEYDOWN and (e.key == K_MINUS or e.key == 47 or e.key == 269):
                        self.decrease_scene_speed()
                    elif e.type == KEYDOWN and e.key == K_s:
                        pass
                        # self.engine.save_genomes()
                    # elif e.type == KEYDOWN and e.key == K_e:
                    #     self.engine.evaluation_mode = 'preparing'

                if self.side_panel.generation_num < self.engine.generation_num:
                    self.side_panel.update_ga_data(self.engine.generation_num, self.engine.best_genome)

                # update statistics time
                cur_t = TimeUtil.current_time_millis()
                cum_t = math.floor((cur_t - self.engine.start_total_time) / 1000)
                gen_t = math.floor((cur_t - self.engine.start_generation_time) / 1000)
                self.side_panel.update_ga_time(cum_t, gen_t, self.engine.generation_sim_time)
                self.side_panel.update_ga_population(len(self.engine.robots), self.engine.Nagents)
                self.screen.fill(Color.BLACK)

                for obj in self.scene.objects:
                    obj.draw(self.scene)

                # draw a black background for the side panel
                side_panel_bg_rect = Rect(self.scene.width, 0, self.SIDE_PANEL_WIDTH, self.scene.height)
                draw.rect(self.screen, Color.BLACK, side_panel_bg_rect)

                self.display_info()

                display.flip()
                self.clock.tick(int(round(self.scene.speed)))
        return self.engine.best_genome

    def printd(self, min_debug_level, *args):
        if self.engine.verbose >= min_debug_level:
            msg = ''

            for arg in args:
                msg += str(arg) + ' '

            print(msg)

    def display_info(self):
        self.side_panel.display_ga_info()

    def initialize(self, **kwargs):
        self.init_scene()

        self.engine = GAbuilder(scene=self.scene, model=self, **kwargs)
        if self.show_screen:
            from lib.screen.side_panel import SidePanel

            from pygame import display
            if not self.offline:
                self.get_larvaworld_food()
            self.scene.screen = display.set_mode((self.scene.width + self.scene.panel_width, self.scene.height))
            self.screen = self.scene.screen
            self.side_panel = SidePanel(self.scene, self.engine.space_dict)
            self.side_panel.update_ga_data(self.engine.generation_num, None)
            self.side_panel.update_ga_population(len(self.engine.robots), self.engine.Nagents)
            self.side_panel.update_ga_time(0, 0, 0)
            # self.side_panel.space_dict=self.engine.space_dict
