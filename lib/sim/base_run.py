import os
import time
import agentpy
import numpy as np

from lib import reg, aux, util, plot
from lib.screen.drawing import ScreenManager
from lib.model import envs, agents
from lib.model.envs.conditions import get_exp_condition

class BaseRun(agentpy.Model):
    def __init__(self, runtype, parameters={},  store_data=None, save_to=None, id=None,experiment=None, **kwargs):
        if experiment is None:
            experiment = parameters.experiment
        self.experiment = experiment
        if store_data is not None:
            self.store_data = store_data


        if 'sim_params' in parameters.keys() :
            # Define sim params
            if store_data is None :
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

    def build_env(self, env_params):
        # Define environment
        self.env_pars = env_params

        self.space = envs.Arena(self, **env_params.arena)
        self.arena_dims = self.space.dims

        self.place_obstacles(env_params.border_list)
        self.place_food(**env_params.food_params)

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

    def place_food(self, food_grid=None, source_groups={}, source_units={}):
        self.food_grid = envs.FoodGrid(**food_grid, model=self) if food_grid else None
        sourceConfs = util.generate_sourceConfs(source_groups, source_units)
        source_list = [agents.Food(model=self, **conf) for conf in sourceConfs]
        self.space.add_agents(source_list, positions=[a.pos for a in source_list])
        self.sources = agentpy.AgentList(model=self, objs=source_list)
        self.foodtypes = aux.get_all_foodtypes(self.env_pars.food_params)
        self.source_xy = aux.get_source_xy(self.env_pars.food_params)