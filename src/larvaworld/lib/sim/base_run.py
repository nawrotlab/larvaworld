from __future__ import annotations
from typing import Any
import agentpy

from .. import reg, screen, util
from .ABM_model import ABModel
from ..model import agents, envs

__all__: list[str] = [
    "BaseRun",
]


class BaseRun(ABModel):
    def __init__(self, screen_kws: dict[str, Any] = {}, **kwargs: Any):
        """
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

        """
        super().__init__(**kwargs)
        self.p.update(**self.nestedConf)
        self.agent_class = self._agent_class
        self.is_paused = False
        self.datasets = None
        self.results = None
        self.exp_condition = None
        self.figs = {}

        self.agents = []
        self.sources = []
        self.obstacles = []
        self._odor_ids = None
        self.screen_kws = screen_kws
        self.screen_manager = self.screen_class(model=self, **self.screen_kws)

    @property
    def end_condition_met(self) -> bool:
        if self.exp_condition is not None:
            return self.exp_condition.check()
        return False

    def sim_step(self) -> None:
        """
        Proceeds the simulation by one step, incrementing `Model.t` by 1
        and then calling :func:`Model.step` and :func:`Model.update`.
        """
        # If the screen was closed by the user, stop immediately.
        if getattr(self.screen_manager, "closed", False):
            self.running = False
            return

        if self.is_paused:
            # Still process user input while paused so shortcuts (e.g., space) work.
            if self.screen_manager.show_display:
                self.screen_manager.evaluate_input()
                # Render once so the pause/unpause message is visible.
                try:
                    self.screen_manager.render()
                except Exception:
                    pass
            return

        self.step_env()
        self.step()
        self.screen_manager.step()
        self.update()
        self.t += 1
        if self.t >= self._steps or self.end_condition_met:
            self.running = False

    def step_env(self) -> None:
        for id, layer in self.odor_layers.items():
            layer.update_values()  # Currently doing something only for the DiffusionValueLayer
        if self.windscape is not None:
            self.windscape.update()

    @property
    def Nticks(self) -> int:
        return self.t

    @property
    def sensorscapes(self):
        ls = [self.windscape, self.thermoscape, self.food_grid] + list(
            self.odor_layers.values()
        )
        ls = [l for l in ls if l is not None]
        return ls

    def set_obj_visibility(self, objs, vis: bool = True) -> None:
        for obj in objs:
            obj.visible = vis

    def build_env(self, p: Any) -> None:
        # reg.vprint(f'--- Simulation {self.id} : Building environment!--- ', 1)
        # Define environment
        if self.Box2D:
            from ..model.box2d import ArenaBox2D

            self.space = ArenaBox2D(model=self, **p.arena)
        else:
            self.space = envs.Arena(model=self, **p.arena)

        self.place_obstacles(p.border_list)
        self.place_food(p=p.food_params)

        """
        Sensory landscapes of the simulation environment arranged per modality
        - Olfactory landscapes : odorscape
        - Wind landscape : windscape
        - Temperature landscape : thermoscape
        """

        self.odor_layers = self.create_odor_layers(**p.odorscape) if p.odorscape else {}
        self.windscape = (
            envs.WindScape(model=self, **p.windscape) if p.windscape else None
        )
        self.thermoscape = envs.ThermoScape(**p.thermoscape) if p.thermoscape else None

    def create_odor_layers(self, odorscape: str, **kwargs: Any):
        odor_layers = {}
        ids = util.unique_list(
            [s.odor.id for s in self.sources if s.odor.id is not None]
        )
        for id in ids:
            od_sources = [f for f in self.sources if f.odor.id == id]
            temp = util.unique_list([s.color for s in od_sources])
            if len(temp) == 1:
                c0 = temp[0]
            elif len(temp) == 3 and all([type(k) == float] for k in temp):
                c0 = temp
            else:
                c0 = util.random_colors(1)[0]
            kwargs0 = {
                "model": self,
                "unique_id": id,
                "sources": od_sources,
                "color": c0,
                **kwargs,
            }
            if odorscape == "Diffusion":
                odor_layers[id] = envs.DiffusionValueLayer(**kwargs0)
            elif odorscape == "Gaussian":
                odor_layers[id] = envs.GaussianValueLayer(**kwargs0)
        return odor_layers

    @property
    def odor_ids(self):
        if self._odor_ids is None:
            ids = []
            if hasattr(self, "agents"):
                ids += self.agents.odor.id
            if hasattr(self, "sources"):
                ids += self.sources.odor.id
            ids = util.unique_list(ids)
            self._odor_ids = [id for id in ids if id is not None]
        return self._odor_ids

    def place_obstacles(self, barriers: dict = {}) -> None:
        borderConfs = reg.gen.Border.from_entries(barriers)
        border_list = [envs.Border(model=self, **conf) for conf in borderConfs]
        # border_list = [envs.Border(model=self, **pars) for pars in barriers]
        # border_list = [envs.Border(model=self, unique_id=id, **pars) for id, pars in barriers.items()]
        self.borders = agentpy.AgentList(model=self, objs=border_list)
        self.border_lines = util.SuperList(self.borders.border_lines).flatten

    def place_food(self, p: Any) -> None:
        self.food_grid = (
            envs.FoodGrid(**p.food_grid, model=self) if p.food_grid else None
        )
        sourceConfs = reg.gen.FoodGroup.from_entries(
            p.source_groups
        ) + reg.gen.Food.from_entries(p.source_units)
        source_list = [agents.Food(model=self, **conf) for conf in sourceConfs]
        self.p.source_xy = util.AttrDict({a.id: a.pos for a in source_list})
        self.space.add_sources(source_list, positions=[a.pos for a in source_list])
        self.sources = agentpy.AgentList(model=self, objs=source_list)

    def get_all_objects(self):
        return self.sources + self.agents + self.borders

    def place_agents(self, confs: list[Any]) -> None:
        agent_list = [self.agent_class(model=self, **conf) for conf in confs]
        self.space.add_agents(agent_list, positions=[a.pos for a in agent_list])
        self.agents = agentpy.AgentList(model=self, objs=agent_list)

    @property
    def _agent_class(self):
        if self.runtype == "Replay":
            if self.p.draw_Nsegs is None:
                return agents.LarvaReplayContoured
            else:
                return agents.LarvaReplaySegmented
        elif self.Box2D:
            from ..model.box2d import LarvaBox2D

            return LarvaBox2D
        elif self.offline:
            return agents.LarvaOffline
        elif self.runtype == "Ga":
            if self.experiment == "obstacle_avoidance":
                return agents.ObstacleLarvaRobot
            else:
                return agents.LarvaRobot
        else:
            return agents.LarvaSim

    @property
    def screen_class(self):
        return screen.GA_ScreenManager if self.runtype == "Ga" else screen.ScreenManager

    def delete_agent(self, a: Any) -> None:
        self.agents.remove(a)
        self.space.remove_agents([a])

    def delete_source(self, a: Any) -> None:
        self.sources.remove(a)
        # self.space.remove_agents([a])

    def delete_agents(self, agent_list: Any | None = None) -> None:
        if agent_list is None:
            agent_list = self.agents
        for a in agent_list:
            self.delete_agent(a)

    def set_collectors(self, cs: Any) -> None:
        self.collectors = reg.par.get_reporters(cs=cs, agents=self.agents)
        self.p.collectors = util.AttrDict(
            {
                "step": list(self.collectors["step"].keys()),
                "end": list(self.collectors["end"].keys()),
            }
        )
        # print(self.collectors['end'])
        # raise

    @property
    def configuration_text(self):
        c = self.p
        pref0 = "     "
        text = (
            f"Simulation configuration : \n"
            f"{pref0}Simulation mode : {c.runtype}\n"
            f"{pref0}Experiment : {c.experiment}\n"
            f"{pref0}Simulation ID : {c.id}\n"
            f"{pref0}Duration (min) : {c.duration}\n"
            f"{pref0}Timestep (sec) : {c.dt}\n"
            f"{pref0}Ticks (#) : {c.Nsteps}\n"
            f"{pref0}Box2D active : {c.Box2D}\n"
            f"{pref0}Offline mode : {c.offline}\n"
            f"{pref0}Data storage : {c.store_data}\n"
            f"{pref0}Parent path : {c.dir}"
        )
        return text

    @classmethod
    def visualize_Env(
        cls,
        envID: Any | None = None,
        envConf: Any | None = None,
        id: Any | None = None,
        duration: int = 1,
        screen_kws: dict[str, Any] = {},
        func: Any | None = None,
        **kwargs: Any,
    ) -> None:
        if envConf is None:
            if envID and envID in reg.conf.Env.confIDs:
                envConf = reg.conf.Env.get(envID).nestedConf
        if id is None:
            if envID:
                id = envID
            else:
                id = "Env_visualization"

        kws = util.AttrDict(
            {
                "screen_kws": {
                    "show_display": True,
                    "vis_mode": "video",
                    "odor_aura": True,
                    "intro_text": False,
                    "fps": 60,
                },
                "parameters": util.AttrDict({"env_params": envConf}),
                "duration": duration,
                "id": id,
                "runtype": "Exp",
                "experiment": "dish",
                **kwargs,
            }
        )
        kws.screen_kws.update(**screen_kws)

        m = cls(**kws)
        m.sim_setup(steps=m.p.steps, seed=None)
        m.build_env(m.p.env_params)
        m.set_obj_visibility(m.sensorscapes, True)
        while m.running:
            if func is not None:
                func(model=m)
            # if Wm == 'direction':
            #     m.windscape.set_wind_direction((m.t / 10 / np.pi) % (2 * np.pi))
            # elif Wm == 'speed':
            #     m.windscape.wind_speed = m.t % 100
            m.sim_step()
        m.end()
        m.screen_manager.close()
