import agentpy

from larvaworld.lib import reg, aux
from larvaworld.lib.model import envs, agents
from larvaworld.lib.sim import ABModel


class BaseRun(ABModel):

    def __init__(self, **kwargs):
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
            Box2D: Whether to implement the Box2D physics engine. Defaults to False
            larva_collisions: Whether to allow overlap between larva bodies. Defaults to True
            dt: The simulation timestep in seconds. Defaults to 0.1
            duration: The simulation duration in seconds. Defaults to None for unlimited duration. Computed from Nsteps if specified.
            Nsteps: The number of simulation timesteps. Defaults to None for unlimited timesteps. Computed from duration if specified.
            **kwargs: Arguments passed to the setup method
        '''

        super().__init__(**kwargs)
        self.p.update(**self.nestedConf)
        self.agent_class = self.define_agent_class()
        self.is_paused = False
        self.datasets = None
        self.results = None
        self.figs = {}
        self.obstacles = []
        self._odor_ids=None




    @property
    def Nticks(self):
        return self.t



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
        sourceConfs=reg.gen.FoodGroup.from_entries(p.source_groups)+reg.gen.Food.from_entries(p.source_units)
        source_list = [agents.Food(model=self, **conf) for conf in sourceConfs]
        self.p.source_xy = aux.AttrDict({a.id: a.pos for a in source_list})
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
            if self.p.draw_Nsegs is None :
                return agents.LarvaReplayContoured
            else:
                return agents.LarvaReplaySegmented
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
        self.p.collectors=aux.AttrDict({'step': list(self.collectors['step'].keys()),
                                 'end' : list(self.collectors['end'].keys())})


