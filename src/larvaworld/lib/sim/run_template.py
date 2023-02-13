import os
import time
import agentpy
import numpy as np

from larvaworld.lib import reg, aux, util, plot
from larvaworld.lib.screen.drawing import ScreenManager
from larvaworld.lib.model import envs, agents
from larvaworld.lib.model.envs.conditions import get_exp_condition

class BaseRun(agentpy.Model):
    def __init__(self, runtype, parameters={},  store_data=True, save_to=None, id=None,experiment=None,
                 Box2D=False, **kwargs):
        self.experiment = experiment if experiment is not None else parameters.experiment
        self.store_data = store_data
        self.Box2D = Box2D
        self.scaling_factor = 1000.0 if self.Box2D else 1.0

        if 'sim_params' in parameters.keys() :
            # Define sim params
            self.dt = parameters.sim_params.timestep
            self.duration = parameters.sim_params.duration
            self.Nsteps = int(self.duration * 60 / self.dt) if self.duration is not None else None
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

    def build_env(self, env_params):
        # Define environment
        self.space = envs.Arena(self, **env_params.arena)

        self.place_obstacles(env_params.border_list)
        self.place_food(p=env_params.food_params)

        '''
        Sensory landscapes of the simulation environment arranged per modality
        - Olfactory landscapes : odorscape
        - Wind landscape : windscape
        - Temperature landscape : thermoscape
        '''

        self.odor_layers = envs.create_odor_layers(model=self, sources=self.sources, pars=env_params.odorscape)
        self.windscape = envs.WindScape(model=self, **env_params.windscape) if env_params.windscape else None
        self.thermoscape = envs.ThermoScape(**env_params.thermoscape) if env_params.thermoscape else None


    def place_obstacles(self, barriers={}):
        self.borders, self.border_lines = [], []
        for id, pars in barriers.items():
            b = envs.Border(unique_id=id, **pars)
            self.borders.append(b)
            self.border_lines += b.border_lines

    def place_food(self, p):
        self.food_grid = envs.FoodGrid(**p.food_grid, model=self) if p.food_grid else None
        sourceConfs = util.generate_sourceConfs(p.source_groups, p.source_units)
        source_list = [agents.Food(model=self, **conf) for conf in sourceConfs]
        self.space.add_agents(source_list, positions=[a.pos for a in source_list])
        self.sources = agentpy.AgentList(model=self, objs=source_list)
        self.foodtypes = aux.get_all_foodtypes(p)
        self.source_xy = aux.get_source_xy(p)