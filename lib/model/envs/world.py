import warnings
import numpy as np
import pandas as pd
import progressbar
import os


import lib.aux.sample_aux
from lib.screen.drawing import ScreenManager

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

from lib.registry import reg

import lib.screen.rendering as ren

from lib.sim.single.input_lib import evaluate_input, evaluate_graphs
from lib.model.envs.base_world import BaseWorld
from lib.aux import naming as nam, dictsNlists as dNl, colsNstr as cNs, sim_aux, xy_aux
from lib.sim.single.conditions import get_exp_condition

class World(BaseWorld):

    def __init__(self, vis_kwargs=None,odor_aura = False,
                 background_motion=None, traj_color=None, allow_clicks=True,
                 progress_bar=None, show_conf_text=False, **kwargs):

        super().__init__(**kwargs)

        if progress_bar is None:
            progress_bar = progressbar.ProgressBar(self.Nsteps)
            progress_bar.start()
        self.progress_bar = progress_bar

        self.screen_manager=ScreenManager(model=self, vis_kwargs=vis_kwargs, show_conf_text=show_conf_text,odor_aura = odor_aura,
                                          background_motion=background_motion, traj_color=traj_color, allow_clicks=allow_clicks)



        self.step_collector=None



        k = get_exp_condition(self.experiment)
        self.exp_condition = k(self) if k is not None else None

    @ property
    def end_condition_met(self):
        if self.Nsteps is not None and self.Nticks >= self.Nsteps:
            return True
        if self.exp_condition is not None:
            return self.exp_condition.check(self)
        return False


    def close(self):
        self.is_running = False


    def step(self):

        if not self.larva_collisions:
            self.larva_bodies = self.get_larva_bodies()
        # Update value_layers
        for id, layer in self.odor_layers.items():
            layer.update_values()  # Currently doing something only for the DiffusionValueLayer
        if self.windscape is not None:
            self.windscape.update()
        self.active_larva_schedule.step()
        self.active_food_schedule.step()
        if self.Box2D:
            self.space.Step(self.dt, self._sim_velocity_iterations, self._sim_position_iterations)
            for fly in self.get_flies():
                fly.update_trajectory()
        if self.step_collector is not None:
            self.step_collector.collect(self)
        # if self.step_group_collector is not None:
        #     self.step_group_collector.collect()
        self.Nticks += 1

    # def step(self):
    #     # Overriden by subclasses
    #     self.Nticks += 1
    #     # print(self.Nticks)
    #     # Tick sim_clock
    #     self.sim_clock.tick_clock()
    #     if self.windscape is not None:
    #         self.windscape.update()

    def run(self):


        self.is_running = True
        # completed=False
        warnings.filterwarnings('ignore')
        while self.is_running and not self.end_condition_met:
            if not self.is_paused:
                self.step()
                if self.progress_bar:
                    self.progress_bar.update(self.Nticks)
            self.screen_manager.step(self.Nticks)
        self.screen_manager.finalize(self.Nticks)


        return self.is_running


















if __name__ == '__main__':

    ww=World()
    ww.run()
#     RefPars = lib.aux.dictsNlists.load_dict(paths.path('ParRef'), use_pickle=False)
#     print(RefPars)
#     sample_ps=list(RefPars.keys())
#
#     from lib.conf.stored.conf import loadConf
#     sample=loadConf('None.200_controls', 'Ref')
#     dic=sample_group(sample, 10, sample_ps)
# print(dic)
