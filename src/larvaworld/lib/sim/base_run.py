import os
import agentpy
import numpy as np
import param

from larvaworld.lib import reg, aux, util, plot
from larvaworld.lib.model import envs, agents


# class BaseRunConf(reg.SimOps):
#     def __init__(self, runtype, **kwargs):
#         reg.SimOps.__init__(self, runtype=runtype, **kwargs)

class BaseRun(agentpy.Model,reg.SimOps):

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
        reg.SimOps.__init__(self, runtype=runtype, **kwargs)
        c=reg.SimOps(runtype=runtype,**kwargs)
        self.agent_class = self.define_agent_class(c)
        self.agentpy_output_kws = {'exp_name': c.experiment, 'exp_id': c.id,
                                   'path': f'{c.data_dir}/agentpy_output'}
        self.p.conf=c




        # print(self.id)
        # raise
        # reg.SimOps.__init__(self, runtype=runtype,**kwargs)

        # print(self.id)
        # raise
        # self.experiment = experiment if experiment is not None else parameters.experiment
        # self.runtype = runtype

        self.p.steps = self.p.conf.Nsteps



        self.report(['agentpy_output_kws', 'id', 'dir', 'Box2D', 'offline', 'show_display',
                     'experiment', 'dt', 'duration', 'Nsteps'])

        self.is_paused = False
        self.datasets = None
        self.results = None
        self.figs = {}
        self.obstacles = []
        self._odor_ids=None


    @property
    def configuration_text(self):
        c=self.p.conf
        pref0 = '     '
        text = f"Simulation configuration : \n" \
               f"{pref0}Simulation mode : {c.runtype}\n" \
               f"{pref0}Experiment : {c.experiment}\n" \
               f"{pref0}Simulation ID : {c.id}\n" \
               f"{pref0}Duration (min) : {c.duration}\n" \
               f"{pref0}Timestep (sec) : {c.dt}\n" \
               f"{pref0}Ticks (#) : {c.Nsteps}\n" \
               f"{pref0}Box2D active : {c.Box2D}\n" \
               f"{pref0}Display active : {c.show_display}\n" \
               f"{pref0}Offline mode : {c.offline}\n" \
               f"{pref0}Data storage : {c.store_data}\n" \
               f"{pref0}Parent path : {c.dir}"
        return text

    @property
    def Nticks(self):
        return self.t

    # @property
    # def dt(self):
    #     return self.p.conf.dt
    # def get_all_odors(self, larva_groups={}):
    #     fp=self.p.env_params.food_params
    #
    #     lg = [conf.odor.id for conf in larva_groups.values()]
    #     su = [conf.odor.id for conf in fp.source_units.values()]
    #     sg = [conf.odor.id for conf in fp.source_groups.values()]
    #     ids = aux.unique_list([id for id in lg + su + sg if id is not None])
    #     return ids

    def build_env(self, p):
        # reg.vprint(f'--- Simulation {self.id} : Building environment!--- ', 1)
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
        c2 = reg.gen.Food.from_entries(p.source_units)
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

    def define_agent_class(self, c):
        if c.runtype=='Replay' :
            return agents.LarvaReplay
        elif c.Box2D :
            return agents.LarvaBox2D
        elif c.offline :
            return agents.LarvaOffline
        elif c.runtype=='Ga' :
            if c.experiment=='obstacle_avoidance':
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

    def create_config(self, **kwargs):
        # source_xy = aux.AttrDict({s.unique_id: s.pos for s in self.sources})
        p = self.p
        config = aux.AttrDict({
            'env_params': p.env_params,
            'larva_groups': p.larva_groups,
            'source_xy': self.source_xy,
            **p.conf.nestedConf
        })
        config.update(**kwargs)
        return config

    def convert_group_output_to_dataset(self,df, collectors):
        step_output_keys = list(collectors['step'].keys())
        end_output_keys = list(collectors['end'].keys())

        df.index.set_names(['AgentID', 'Step'], inplace=True)
        df = df.reorder_levels(order=['Step', 'AgentID'], axis=0)
        df.sort_index(level=['Step', 'AgentID'], inplace=True)

        end = df[end_output_keys].xs(df.index.get_level_values('Step').max(), level='Step')
        step = df[step_output_keys]

        return step, end

    def convert_output_to_dataset(self, df,agents=None,to_Geo=False, **kwargs):
        config=self.create_config(**kwargs)
        step, end = self.convert_group_output_to_dataset(df, self.collectors)

        from larvaworld.lib.process.dataset import BaseLarvaDataset
        d=BaseLarvaDataset.initGeo(to_Geo=to_Geo,config=config,load_data=False,step=step,end=end,agents=agents)

        return d




# class RefRun(BaseRun,RefDataset):

