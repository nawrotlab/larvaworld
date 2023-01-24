import os
import time
import agentpy
import numpy as np

from lib import reg, aux, util, plot
from lib.screen.drawing import ScreenManager
from lib.model import envs, agents
from lib.model.envs.conditions import get_exp_condition

class BaseRun(agentpy.Model):
    def __init__(self, runtype, parameters={},  save_to=None, id=None,experiment=None, **kwargs):
        if experiment is None:
            experiment = parameters.experiment
        self.experiment = experiment


        if 'sim_params' in parameters.keys() :
            # Define sim params
            self.store_data = parameters.sim_params.store_data
            self.dt = parameters.sim_params.timestep
            self.duration = parameters.sim_params.duration
            self.Nsteps = int(self.duration * 60 / self.dt) if self.duration is not None else None


            self.Box2D = parameters.sim_params.Box2D
            self.scaling_factor = 1000.0 if self.Box2D else 1.0

            parameters.steps = self.Nsteps




        super().__init__(parameters=parameters, **kwargs)

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

        self.is_paused = False
        self.datasets = None
        self.results = None
        self.figs = {}
        self.obstacles = []

    @property
    def configuration_text(self):
        text = f"Simulation configuration : \n" \
               "\n" \
               f"Experiment : {self.experiment}\n" \
               f"Simulation ID : {self.id}\n" \
               f"Duration (min) : {self.duration}\n" \
               f"Timestep (sec) : {self.dt}\n" \
               f"Plot path : {self.plot_dir}\n" \
               f"Parent path : {self.dir}"
        return text

    @property
    def Nticks(self):
        return self.t

    def build_box(self, x, y, size, color):
        box = envs.Box(x, y, size, color=color)
        self.obstacles.append(box)
        return box

    def build_wall(self, point1, point2, color):
        wall = envs.Wall(point1, point2, color=color)
        self.obstacles.append(wall)
        return wall