import os
import agentpy
import numpy as np

from larvaworld.lib import reg, aux, util, plot
from larvaworld.lib.model import envs, agents

class BaseRun(agentpy.Model):
    def __init__(self, runtype, parameters=None,
                 store_data=True, save_to=None,
                 id=None,experiment=None,offline=False,show_display=True,
                 Box2D=False, larva_collisions=True,
                 dt=0.1,duration=None,Nsteps=None,
                 **kwargs):

        if parameters is None :
            if experiment is not None :
                parameters = reg.expandConf('Exp', experiment)
            else :
                raise ValueError('Either a parameter dictionary or the name of the experiment must be provided')

        self.experiment = experiment if experiment is not None else parameters.experiment

        # Define N timesteps
        self.dt = dt
        if Nsteps is None and duration is not None :
            Nsteps = int(duration * 60 / dt)
        if duration is None and Nsteps is not None :
            duration = Nsteps* dt/60
        self.Nsteps = Nsteps
        self.duration = duration
        parameters.steps = self.Nsteps
        super().__init__(parameters=parameters, **kwargs)

        # Define constant parameters
        self.offline = offline
        self.show_display = show_display and not offline
        self.larva_collisions = larva_collisions
        self.Box2D = Box2D
        self.scaling_factor = 1000.0 if self.Box2D else 1.0

        # Define ID
        if id is None:
            idx = reg.next_idx(self.experiment, conftype=runtype)
            id = f'{self.experiment}_{idx}'
        self.id = id



        # Define directories
        if save_to is None:
            save_to = f'{reg.SIM_DIR}/{runtype.lower()}_runs'
        self.dir = f'{save_to}/{self.experiment}/{id}'
        self.plot_dir = f'{self.dir}/plots'
        self.data_dir = f'{self.dir}/data'
        self.save_to = self.dir
        self.store_data = store_data
        if self.store_data :
            os.makedirs(self.data_dir, exist_ok=True)
            os.makedirs(self.plot_dir, exist_ok=True)
        self.agentpy_output_kws = {'exp_name': self.experiment, 'exp_id': self.id,
                                   'path': f'{self.data_dir}/agentpy_output'}


        self.report(['agentpy_output_kws', 'id', 'dir', 'Box2D', 'offline', 'show_display',
                     'experiment', 'save_to', 'dt', 'duration', 'Nsteps'])

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

    # def build_box(self, x, y, size, color):
    #     box = envs.Box(x, y, size, color=color)
    #     self.obstacles.append(box)
    #     return box
    #
    # def build_wall(self, point1, point2, color):
    #     wall = envs.Wall(point1, point2, color=color)
    #     self.obstacles.append(wall)
    #     return wall

    def build_env(self, p):
        reg.vprint(f'--- Simulation {self.id} : Building environment!--- ', 1)
        # Define environment
        self.space = envs.Arena(self, **p.arena)

        self.place_obstacles(p.border_list)
        self.place_food(p=p.food_params)

        '''
        Sensory landscapes of the simulation environment arranged per modality
        - Olfactory landscapes : odorscape
        - Wind landscape : windscape
        - Temperature landscape : thermoscape
        '''

        self.odor_layers = envs.create_odor_layers(model=self, sources=self.sources, pars=p.odorscape)
        self.windscape = envs.WindScape(model=self, **p.windscape) if p.windscape else None
        self.thermoscape = envs.ThermoScape(**p.thermoscape) if p.thermoscape else None


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
        self.space.add_sources(source_list, positions=[a.pos for a in source_list])
        self.sources = agentpy.AgentList(model=self, objs=source_list)
        self.foodtypes = aux.get_all_foodtypes(p)
        self.source_xy = aux.get_source_xy(p)

    # def get_flies(self, ids=None, group=None):
    #     ls = self.agents
    #     if ids is not None:
    #         ls = [l for l in ls if l.unique_id in ids]
    #     if group is not None:
    #         ls = [l for l in ls if l.group == group]
    #     return ls

    def get_all_objects(self):
        return self.sources + self.agents + self.borders

    def place_agents(self, confs, agent_class):
        agent_list = [agent_class(model=self, **conf) for conf in confs]
        self.space.add_agents(agent_list, positions=[a.pos for a in agent_list])
        self.agents = agentpy.AgentList(model=self, objs=agent_list)

    def delete_agent(self, a):
        self.agents.remove(a)
        self.space.remove_agents([a])

    def delete_agents(self, agent_list=None):
        if agent_list is None :
            agent_list = self.agents
        for a in agent_list:
            self.delete_agent(a)

    def set_collectors(self, collections):
        self.collectors = reg.get_reporters(collections=collections, agents=self.agents)
        self.step_output_keys = list(self.collectors['step'].keys())
        self.end_output_keys = list(self.collectors['end'].keys())