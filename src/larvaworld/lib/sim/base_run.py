import os
import agentpy
import numpy as np
import param

from larvaworld.lib import reg, aux, util, plot
from larvaworld.lib.model import envs, agents
from larvaworld.lib.process.dataset import RefDataset




class BaseRun(reg.SimOps, agentpy.Model):

    def __init__(self, runtype,parameters=None, **kwargs):
        '''
        Basic simulation class that extends the agentpy.Model class and creates a larvaworld agent-based model (ABM).
        Further extended by classes supporting the various simulation modes in larvaworld.
        Specifies the simulation mode, type of experiment and simulation duration and timestep.
        Specifies paths for saving simulated data and results.

        Args:
            runtype: The simulation mode as defined by a subclass
            parameters: Dictionary of configuration parameters to be passed to the ABM model
            store_data: Whether to store simulation data. Defaults to True
            save_to: Path to store data. If not specified, it is automatically set to the runtype-specific subdirectory under the platform's ROOT/DATA directory
            id: Unique ID of the simulation. If not specified it is automatically set according to the simulation mode and experiment type.
            experiment: The experiment simulated
            offline: Whether to perform the simulation without launching a spatial arena. Defaults to False
            show_display: Whether to launch the pygame-visualization. Defaults to True
            Box2D: Whether to implement the Box2D physics engine. Defaults to False
            larva_collisions: Whether to allow overlap between larva bodies. Defaults to True
            dt: The simulation timestep in seconds. Defaults to 0.1
            duration: The simulation duration in seconds. Defaults to None for unlimited duration. Computed from Nsteps if specified.
            Nsteps: The number of simulation timesteps. Defaults to None for unlimited timesteps. Computed from duration if specified.
            **kwargs: Arguments passed to the setup method
        '''
        agentpy.Model.__init__(self, parameters=parameters)

        reg.SimOps.__init__(self, runtype=runtype,**kwargs)
        self.agent_class = self.define_agent_class()
        # print(self.id)
        # raise
        # self.experiment = experiment if experiment is not None else parameters.experiment
        # self.runtype = runtype

        self.p.steps = self.Nsteps

        # print(self.id)
        # raise

        # Define ID
        # if id is None:
        #     idx = reg.next_idx(self.experiment, conftype=runtype)
        #     id = f'{self.experiment}_{idx}'
        # self.id = id



        # # Define directories
        # if save_to is None:
        #     save_to = f'{reg.SIM_DIR}/{runtype.lower()}_runs'
        # self.dir = f'{save_to}/{self.experiment}/{self.id}'
        # self.plot_dir = f'{self.dir}/plots'
        # self.data_dir = f'{self.dir}/data'
        # self.save_to = self.dir
        # if self.store_data :
        #     os.makedirs(self.data_dir, exist_ok=True)
        #     os.makedirs(self.plot_dir, exist_ok=True)
        self.agentpy_output_kws = {'exp_name': self.experiment, 'exp_id': self.id,
                                   'path': f'{self.data_dir}/agentpy_output'}


        self.report(['agentpy_output_kws', 'id', 'dir', 'Box2D', 'offline', 'show_display',
                     'experiment', 'save_to', 'dt', 'duration', 'Nsteps'])

        self.is_paused = False
        self.datasets = None
        self.results = None
        self.figs = {}
        self.obstacles = []
        self._odor_ids=None


    @property
    def configuration_text(self):
        pref0 = '     '
        text = f"Simulation configuration : \n" \
               f"{pref0}Simulation mode : {self.runtype}\n" \
               f"{pref0}Experiment : {self.experiment}\n" \
               f"{pref0}Simulation ID : {self.id}\n" \
               f"{pref0}Duration (min) : {self.duration}\n" \
               f"{pref0}Timestep (sec) : {self.dt}\n" \
               f"{pref0}Ticks (#) : {self.Nsteps}\n" \
               f"{pref0}Box2D active : {self.Box2D}\n" \
               f"{pref0}Display active : {self.show_display}\n" \
               f"{pref0}Offline mode : {self.offline}\n" \
               f"{pref0}Data storage : {self.store_data}\n" \
               f"{pref0}Parent path : {self.dir}"
        return text

    @property
    def Nticks(self):
        return self.t


    # def get_all_odors(self, larva_groups={}):
    #     fp=self.p.env_params.food_params
    #
    #     lg = [conf.odor.id for conf in larva_groups.values()]
    #     su = [conf.odor.id for conf in fp.source_units.values()]
    #     sg = [conf.odor.id for conf in fp.source_groups.values()]
    #     ids = aux.unique_list([id for id in lg + su + sg if id is not None])
    #     return ids

    def build_env(self, p):
        reg.vprint(f'--- Simulation {self.id} : Building environment!--- ', 1)
        # Define environment
        self.space = envs.Arena(model=self, **p.arena)

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

    @ property
    def odor_ids(self):
        if self._odor_ids is None :
            ids=[]
            if hasattr(self,'agents'):
                ids += self.agents.odor.id
            if hasattr(self,'sources'):
                ids += self.sources.odor.id
            ids=aux.unique_list(ids)
            self._odor_ids=[id for id in ids if id is not None]
        return self._odor_ids

    def place_obstacles(self, barriers={}):
        border_list = [envs.Border(model=self, unique_id=id, **pars) for id, pars in barriers.items()]
        self.borders = agentpy.AgentList(model=self, objs=border_list)
        self.border_lines=self.borders.border_lines

    def place_food(self, p):
        self.food_grid = envs.FoodGrid(**p.food_grid, model=self) if p.food_grid else None
        c1 = reg.gen.FoodGroup.from_entries(p.source_groups)
        c2 = reg.gen.FoodUnit.from_entries(p.source_units)
        sourceConfs=c1+c2
        # sourceConfs = util.generate_sourceConfs(p.source_groups, p.source_units)
        source_list = [agents.Food(model=self, **conf) for conf in sourceConfs]
        self.source_xy = aux.AttrDict({a.id: a.pos for a in source_list})
        self.space.add_sources(source_list, positions=[a.pos for a in source_list])
        self.sources = agentpy.AgentList(model=self, objs=source_list)
  

    def get_all_objects(self):
        return self.sources + self.agents + self.borders

    def place_agents(self, confs):
        agent_list = [self.agent_class(model=self, **conf) for conf in confs]
        self.space.add_agents(agent_list, positions=[a.pos for a in agent_list])
        self.agents = agentpy.AgentList(model=self, objs=agent_list)

    def define_agent_class(self):
        if self.runtype=='Replay' :
            return agents.LarvaReplay
        elif self.Box2D :
            return agents.LarvaBox2D
        elif self.offline :
            return agents.LarvaOffline
        elif self.runtype=='Ga' :
            if self.experiment=='obstacle_avoidance':
                return agents.ObstacleLarvaRobot
            else:
                return agents.LarvaRobot
        else:
            return agents.LarvaSim


    def delete_agent(self, a):
        self.agents.remove(a)
        self.space.remove_agents([a])

    def delete_agents(self, agent_list=None):
        if agent_list is None :
            agent_list = self.agents
        for a in agent_list:
            self.delete_agent(a)

    def set_collectors(self, collections):
        self.collectors = reg.par.get_reporters(collections=collections, agents=self.agents)
        self.step_output_keys = list(self.collectors['step'].keys())
        self.end_output_keys = list(self.collectors['end'].keys())
        # print(self.step_output_keys)
        # raise

    def convert_output_to_dataset(self, df, agents=None,to_Geo=False, **kwargs):
        kws = {
            'load_data' : False,
            'env_params': self.p.env_params,
            'source_xy': aux.AttrDict({s.unique_id : s.pos for s in self.sources}),
            'fr': 1 / self.dt,
            'dt': self.dt,
            **kwargs
        }
        if not to_Geo :
            from larvaworld.lib.process.dataset import LarvaDataset
            d = LarvaDataset(**kws)
        else:
            from larvaworld.lib.process.larva_trajectory_collection import LarvaTrajectoryCollection
            d = LarvaTrajectoryCollection(**kws)

        df.index.set_names(['AgentID', 'Step'], inplace=True)
        df = df.reorder_levels(order=['Step', 'AgentID'], axis=0)
        df.sort_index(level=['Step', 'AgentID'], inplace=True)

        end = df[self.end_output_keys].xs(df.index.get_level_values('Step').max(), level='Step')
        step = df[self.step_output_keys]
        d.set_data(step=step, end=end)
        if agents and not to_Geo:
            ls = aux.AttrDict({l.unique_id: l for l in agents if l.unique_id in d.agent_ids})
            d.larva_dicts = aux.get_larva_dicts(ls)

        return d




# class RefRun(BaseRun,RefDataset):

