import math
import os
import sys
import warnings

import numpy as np


from lib import reg, aux
from lib.screen.rendering import  Viewer
# from lib.sim.base import BaseRun
from lib.sim.ga_engine import GAbuilder
from lib.model.envs.base_world import BaseWorld

# class GenAlgRun(BaseRun):
#     def __init__(self, sim_params, env_params=None, experiment='exploration',
#                  offline=False, **kwargs):
#
#         kws = {
#             # 'dt': dt,
#             # 'model_class': WorldSim,
#             # 'progress_bar': progress_bar,
#             # 'save_to': save_to,
#             # 'store_data': store_data,
#             # 'analysis': analysis,
#             # 'show': show,
#             # 'Nsteps': int(sim_params.duration * 60 / dt),
#             # 'output': output,
#             'id': sim_params.sim_ID,
#             # 'Box2D': sim_params.Box2D,
#             # 'larva_groups': larva_groups,
#             # **kwargs
#         }
#         # super().__init__(runtype='exp', **kws)
#
#
#         super().__init__(runtype='ga', **kwargs)
#         self.offline = offline
#         id = sim_params.sim_ID
#         self.sim_params = sim_params
#         dt = sim_params.timestep
#         Nsteps = int(sim_params.duration * 60 / dt)
#         if not self.offline:
#             super().__init__(id=id, dt=dt, Box2D=sim_params.Box2D, env_params=env_params,
#                              save_to=f'{self.dir_path}/visuals',
#                              Nsteps=Nsteps, experiment=experiment, **kwargs)
#             self.arena_width, self.arena_height = self.env_pars.arena.arena_dims
#         else:
#             self.env_pars = env_params
#             self.scaling_factor = 1
#             X, Y = self.arena_width, self.arena_height = self.env_pars.arena.arena_dims
#             self.space_edges_for_screen = np.array([-X / 2, X / 2, -Y / 2, Y / 2])
#             # self.experiment = experiment
#             self.dt = dt
#             self.Nticks = 0
#             self.Nsteps = Nsteps
#             self.id = id
#             # self.save_to = save_to
#             self.Box2D = False





class BaseGAlauncher(BaseWorld):

    def __init__(self, sim_params,  env_params=None, experiment='exploration',
                  save_to=None,offline=False,  **kwargs):
        self.offline = offline
        id = sim_params.sim_ID
        self.sim_params = sim_params
        dt = sim_params.timestep
        Nsteps = int(sim_params.duration * 60 / dt)
        if save_to is None:
            save_to = f'{reg.SIM_DIR}/ga_runs'
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
            self._id_counter = -1
            self.p = env_params
            self.env_pars = env_params
            self.scaling_factor = 1
            X,Y=self.arena_dims = self.arena_width, self.arena_height = self.env_pars.arena.arena_dims
            self.space_edges_for_screen = np.array([-X / 2, X / 2, -Y / 2, Y / 2])
            self.experiment = experiment
            self.dt = dt
            self.Nticks = 0
            self.Nsteps = Nsteps
            self.id = id
            self.save_to = save_to
            self.Box2D = False
            self.source_xy = aux.get_source_xy(self.env_pars.food_params)
            self.foodtypes = aux.get_all_foodtypes(self.env_pars.food_params)






class GAlauncher(BaseGAlauncher):
    SCENE_MAX_SPEED = 3000

    SCENE_MIN_SPEED = 1
    SCENE_SPEED_CHANGE_COEFF = 1.5

    SIDE_PANEL_WIDTH = 600
    def __init__(self, ga_build_kws, ga_select_kws, show_screen=True,
                 caption=None, scene='no_boxes', scene_speed=0, **kwargs):
        super().__init__(**kwargs)


        self.ga_build_kws = ga_build_kws
        self.ga_select_kws = ga_select_kws
        self.show_screen = show_screen
        if caption is None:
            caption = f'GA {self.experiment} : {self.id}'
        self.caption = caption
        self.scene_file = f'{reg.ROOT_DIR}/lib/sim/ga_scenes/{scene}.txt'
        self.scene_speed = scene_speed
        self.obstacles = []

        self.initialize(**ga_build_kws, **ga_select_kws)




    def run(self):
        while True and self.engine.is_running:
            self.engine.step()
            if self.show_screen:
                from pygame import KEYDOWN, K_ESCAPE, K_r, K_MINUS, K_PLUS, K_s, QUIT, event, Rect, draw, display
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
                cur_t = aux.TimeUtil.current_time_millis()
                cum_t = math.floor((cur_t - self.engine.start_total_time) / 1000)
                gen_t = math.floor((cur_t - self.engine.start_generation_time) / 1000)
                self.side_panel.update_ga_time(cum_t, gen_t, self.engine.generation_sim_time)
                self.side_panel.update_ga_population(len(self.engine.robots), self.engine.Nagents)
                self.screen.fill(aux.Color.BLACK)

                for obj in self.viewer.objects:
                    obj.draw(self.viewer)

                # draw a black background for the side panel
                side_panel_bg_rect = Rect(self.viewer.width, 0, self.SIDE_PANEL_WIDTH, self.viewer.height)
                draw.rect(self.screen, aux.Color.BLACK, side_panel_bg_rect)

                self.display_info()

                display.flip()
                self.viewer._t.tick(int(round(self.viewer.speed)))
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
        self.viewer = Viewer.load_from_file(self.scene_file, scene_speed=self.scene_speed,
                                           panel_width=self.SIDE_PANEL_WIDTH,
                                           space_bounds=aux.get_arena_bounds(self.arena_dims, self.scaling_factor))

        self.engine = GAbuilder(viewer=self.viewer, model=self, **kwargs)
        if self.show_screen:
            from lib.screen.side_panel import SidePanel

            from pygame import display
            if not self.offline:
                self.get_larvaworld_food()
            self.screen = self.viewer._window
            self.side_panel = SidePanel(self.viewer, self.engine.space_dict)
            self.side_panel.update_ga_data(self.engine.generation_num, None)
            self.side_panel.update_ga_population(len(self.engine.robots), self.engine.Nagents)
            self.side_panel.update_ga_time(0, 0, 0)

    def build_box(self, x, y, size, color):
        from lib.model.envs.obstacle import Box
        box = Box(x, y, size, color=color)
        self.obstacles.append(box)
        return box

    def build_wall(self, point1, point2, color):
        from lib.model.envs.obstacle import Wall
        wall = Wall(point1, point2, color=color)
        self.obstacles.append(wall)
        return wall

    def get_larvaworld_food(self):
        for label,ff in self.env_pars.food_params.source_units.items():
            x, y = self.screen_pos(ff.pos)
            size = ff.radius * self.scaling_factor
            col = ff.default_color
            box = self.build_box(x, y, size, col)
            box.label = label
            self.viewer.put(box)

    def screen_pos(self, real_pos):
        return np.array(real_pos) * self.scaling_factor + np.array([self.viewer.width / 2, self.viewer.height / 2])

    def init_scene(self):
        self.viewer = Viewer.load_from_file(self.scene_file, scene_speed=self.scene_speed,
                                           panel_width=self.SIDE_PANEL_WIDTH, space_bounds = aux.get_arena_bounds(self.arena_dims, self.scaling_factor))


    def increase_scene_speed(self):
        if self.viewer.speed < self.SCENE_MAX_SPEED:
            self.viewer.speed *= self.SCENE_SPEED_CHANGE_COEFF
        print('viewer.speed:', self.viewer.speed)

    def decrease_scene_speed(self):
        if self.viewer.speed > self.SCENE_MIN_SPEED:
            self.viewer.speed /= self.SCENE_SPEED_CHANGE_COEFF
        print('viewer.speed:', self.viewer.speed)


def optimize_mID(mID0, mID1=None, fit_dict=None, refID=None, space_mkeys=['turner', 'interference'], init='model',
                 offline=False,show_screen=False,exclusion_mode=False,experiment='exploration',
                 sim_ID=None, dt=1 / 16, dur=0.5, save_to=None, store_data=False, Nagents=30, Nelits=6, Ngenerations=20,
                 **kwargs):

    warnings.filterwarnings('ignore')
    if mID1 is None:
        mID1 = mID0

    if sim_ID is None:
        sim_ID = f'{experiment}_{reg.next_idx(id=experiment, conftype="Ga")}'

    kws = {
        'sim_params': reg.get_null('sim_params', duration=dur, sim_ID=sim_ID, store_data=store_data, timestep=dt,
                                   path = f'ga_runs/{experiment}'),
        'show_screen': show_screen,
        'offline': offline,
        'save_to': save_to,
        'experiment': experiment,
        'env_params': 'arena_200mm',
        'ga_select_kws': reg.get_null('ga_select_kws', Nagents=Nagents, Nelits=Nelits, Ngenerations=Ngenerations, selection_ratio=0.1),
        'ga_build_kws': reg.get_null('ga_build_kws', init_mode=init, space_mkeys=space_mkeys, base_model=mID0,exclusion_mode=exclusion_mode,
                                      bestConfID=mID1, fitness_target_refID=refID)
    }

    conf = reg.get_null('Ga', **kws)
    conf.env_params = reg.expandConf(id=conf.env_params, conftype='Env')

    conf.ga_build_kws.fit_dict = fit_dict

    GA = GAlauncher(**conf)
    best_genome = GA.run()
    entry = {mID1: best_genome.mConf}
    return entry
