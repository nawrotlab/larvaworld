import os
import time
import agentpy
import numpy as np

from lib import reg, aux, util, plot
from lib.screen.drawing import ScreenManager
from lib.model import envs, agents
from lib.model.envs.conditions import get_exp_condition

class BaseRun(agentpy.Model):
    def __init__(self, runtype, save_to=None, id=None, **kwargs):
        super().__init__(**kwargs)
        self.experiment = self.p.experiment
        if id is None:
            idx = reg.next_idx(self.experiment, conftype=runtype)
            id = f'{self.experiment}_{idx}'
        self.id = id
        # Define directories
        if save_to is None:
            save_to = f'{reg.SIM_DIR}/{runtype.lower()}_runs'
        self.dir = f'{save_to}/{id}'
        self.plot_dir = f'{self.dir}/plots'
        self.data_dir = f'{self.dir}/data'
        self.save_to = self.dir

        # Define sim params
        self.store_data = self.p.sim_params.store_data
        self.dt = self.p.sim_params.timestep
        self.duration = self.p.sim_params.duration
        self.Nsteps = int(self.duration * 60 / self.dt) if self.duration is not None else None
        self._steps = self.Nsteps

        self.Box2D = self.p.sim_params.Box2D
        self.scaling_factor = 1000.0 if self.Box2D else 1.0

        self.is_paused = False
        self.datasets = None
        self.results = None
        self.figs = {}
