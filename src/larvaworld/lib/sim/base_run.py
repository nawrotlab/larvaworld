import os
import agentpy
import numpy as np

from larvaworld.lib import reg, aux, util, plot
from larvaworld.lib.model import envs, agents

class BaseRun(agentpy.Model):

    def __init__(self, runtype, parameters=None, store_data=True, save_to=None,
                 id=None,experiment=None,offline=False,show_display=True,
                 Box2D=False, larva_collisions=True,
                 dt=0.1,duration=None,Nsteps=None,
                 **kwargs):
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

        # if parameters is None :
        #     if experiment is not None :
        #         parameters = reg.expandConf('Exp', experiment)
        #     else :
        #         raise ValueError('Either a parameter dictionary or the name of the experiment must be provided')

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


    def get_all_odors(self, larva_groups={}):
        fp=self.p.env_params.food_params

        lg = [conf.odor.id for conf in larva_groups.values()]
        su = [conf.odor.id for conf in fp.source_units.values()]
        sg = [conf.odor.id for conf in fp.source_groups.values()]
        ids = aux.unique_list([id for id in lg + su + sg if id is not None])
        return ids

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
        self.foodtypes = self.get_all_foodtypes(p)
        self.source_xy = self.get_source_xy(p)

    def get_source_xy(self, p):
        sources_u = {k: v['pos'] for k, v in p['source_units'].items()}
        sources_g = {k: v['distribution']['loc'] for k, v in p['source_groups'].items()}
        return {**sources_u, **sources_g}

    def get_all_foodtypes(self, p):
        sg = {k: v.default_color for k, v in p.source_groups.items()}
        su = {conf.group: conf.default_color for conf in p.source_units.values()}
        gr = {
            p.food_grid.unique_id: p.food_grid.default_color} if p.food_grid is not None else {}
        ids = {**gr, **su, **sg}
        ks = aux.unique_list(list(ids.keys()))
        try:
            ids = {k: list(np.array(ids[k]) / 255) for k in ks}
        except:
            ids = {k: ids[k] for k in ks}
        return ids
  

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
        self.collectors = reg.par.get_reporters(collections=collections, agents=self.agents)
        self.step_output_keys = list(self.collectors['step'].keys())
        self.end_output_keys = list(self.collectors['end'].keys())

    def convert_output_to_dataset(self, df, agents=None, **kwargs):
        kws = {
            'load_data' : False,
            'env_params': self.p.env_params,
            'source_xy': self.source_xy,
            'fr': 1 / self.dt,
            **kwargs
        }

        from larvaworld.lib.process.dataset import LarvaDataset

        df.index.set_names(['AgentID', 'Step'], inplace=True)
        df = df.reorder_levels(order=['Step', 'AgentID'], axis=0)
        df.sort_index(level=['Step', 'AgentID'], inplace=True)

        end = df[self.end_output_keys].xs(df.index.get_level_values('Step').max(), level='Step')
        d = LarvaDataset(**kws)
        d.set_data(step=df[self.step_output_keys], end=end)
        if agents:
            ls = aux.AttrDict({l.unique_id: l for l in agents if l.unique_id in d.agent_ids})
            d.larva_dicts = aux.get_larva_dicts(ls)
        return d