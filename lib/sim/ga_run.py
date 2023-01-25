import math
import sys
import warnings

import numpy as np


from lib import reg, aux, util
from lib.screen.rendering import  Viewer
from lib.sim.ga_engine import GAbuilder
from lib.sim.base_run import BaseRun

class GAlauncher(BaseRun):
    SCENE_MAX_SPEED = 3000

    SCENE_MIN_SPEED = 1
    SCENE_SPEED_CHANGE_COEFF = 1.5

    SIDE_PANEL_WIDTH = 600

    def __init__(self, **kwargs):
        super().__init__(runtype = 'Ga', **kwargs)

    def setup(self):

        self.odor_ids = aux.get_all_odors({}, self.p.env_params.food_params)
        self.build_env(self.p.env_params)

        self.scene_file = f'{reg.ROOT_DIR}/lib/sim/ga_scenes/{self.p.scene}.txt'
        self.scene_speed = 0


        self.initialize(**self.p.ga_build_kws, **self.p.ga_select_kws)


    def simulate(self):
        self.setup(**self._setup_kwargs)
        while True and self.engine.is_running:
            self.engine.step()
            if self.viewer.show_display:
                from pygame import KEYDOWN, K_ESCAPE, K_r, K_MINUS, K_PLUS, K_s, QUIT, event, Rect, draw, display
                for e in event.get():
                    if e.type == QUIT or (e.type == KEYDOWN and e.key == K_ESCAPE):
                        sys.exit()
                    elif e.type == KEYDOWN and e.key == K_r:
                        self.initialize(**self.p.ga_select_kws, **self.p.ga_build_kws)
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

                self.side_panel.display_ga_info()

                display.flip()
                self.viewer._t.tick(int(round(self.viewer.speed)))
        return self.engine.best_genome

    def printd(self, min_debug_level, *args):
        if self.engine.verbose >= min_debug_level:
            msg = ''

            for arg in args:
                msg += str(arg) + ' '

            print(msg)

    def initialize(self, **kwargs):
        self.viewer = Viewer.load_from_file(self.scene_file, scene_speed=self.scene_speed,show_display=self.p.show_screen and not self.p.offline,
                                           panel_width=self.SIDE_PANEL_WIDTH,caption = f'GA {self.p.experiment} : {self.id}',
                                           space_bounds=aux.get_arena_bounds(self.space.dims, self.scaling_factor))

        self.engine = GAbuilder(viewer=self.viewer, model=self, **kwargs)
        if self.viewer.show_display:
            from lib.screen.side_panel import SidePanel

            from pygame import display
            self.get_larvaworld_food()
            self.screen = self.viewer._window
            self.side_panel = SidePanel(self.viewer, self.engine.space_dict)
            self.side_panel.update_ga_data(self.engine.generation_num, None)
            self.side_panel.update_ga_population(len(self.engine.robots), self.engine.Nagents)
            self.side_panel.update_ga_time(0, 0, 0)



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
                 id=None, dt=1 / 16, dur=0.5, save_to=None, store_data=False, Nagents=30, Nelits=6, Ngenerations=20,
                 **kwargs):

    warnings.filterwarnings('ignore')
    if mID1 is None:
        mID1 = mID0



    kws = {
        'sim_params': reg.get_null('sim_params', duration=dur, store_data=store_data, timestep=dt),
        'show_screen': show_screen,
        'offline': offline,
        # 'save_to': save_to,
        'experiment': experiment,
        'env_params': 'arena_200mm',
        'ga_select_kws': reg.get_null('ga_select_kws', Nagents=Nagents, Nelits=Nelits, Ngenerations=Ngenerations, selection_ratio=0.1),
        'ga_build_kws': reg.get_null('ga_build_kws', init_mode=init, space_mkeys=space_mkeys, base_model=mID0,exclusion_mode=exclusion_mode,
                                      bestConfID=mID1, fitness_target_refID=refID)
    }

    conf = reg.get_null('Ga', **kws)
    conf.env_params = reg.expandConf(id=conf.env_params, conftype='Env')

    conf.ga_build_kws.fit_dict = fit_dict

    GA = GAlauncher(parameters=conf, save_to=save_to, id=id)
    best_genome = GA.simulate()
    entry = {mID1: best_genome.mConf}
    return entry

