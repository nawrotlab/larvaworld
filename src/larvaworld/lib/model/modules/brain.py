"""
The brain, coupling sensory input to locomotory output.

Assembles the configured sensor, memory and locomotor modules into one
controller, and steps them in order each timestep so that the sensed
environment drives the agent's motion.
"""

from __future__ import annotations

import numpy as np

from ... import util
from ...param import ClassAttr, NestedConf
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Type-only import to avoid runtime cycle
    from .locomotor import Locomotor
from .module_modes import moduleDB as MD

__all__: list[str] = [
    "Brain",
    "DefaultBrain",
]


class Brain(NestedConf):
    """
    Base brain class for agent behavioral control.

    Orchestrates sensory processing and locomotor control for autonomous
    agents in simulation. Integrates multiple sensory modalities (olfaction,
    touch, thermosensation, wind) with memory and locomotor modules.

    Attributes:
        olfactor: Olfactory sensor module for odor detection
        toucher: Tactile sensor module for contact sensing
        windsensor: Wind sensor module for air flow detection
        thermosensor: Temperature sensor module for thermal gradients
        locomotor: Locomotor module for movement control
        agent: Parent agent instance (polymorphic: LarvaRobot, LarvaSim, etc.)
        dt: Simulation time step (seconds)
        modalities: Dict of sensory modalities with sensors and processing functions

    Example:
        >>> brain = Brain(conf=brain_conf, agent=my_agent, dt=0.1)
        >>> brain.sense(pos=agent.pos, reward=False)
        >>> print(f"Total sensory input: {brain.A_in}")
    """

    olfactor = ClassAttr(
        class_=MD.parent_class("olfactor"), default=None, doc="The olfactory sensor"
    )
    toucher = ClassAttr(
        class_=MD.parent_class("toucher"), default=None, doc="The tactile sensor"
    )
    windsensor = ClassAttr(
        class_=MD.parent_class("windsensor"), default=None, doc="The wind sensor"
    )
    thermosensor = ClassAttr(
        class_=MD.parent_class("thermosensor"),
        default=None,
        doc="The temperature sensor",
    )

    def __init__(
        self,
        conf: Any,
        agent: Any | None = None,
        dt: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.agent = agent
        if dt is None:
            dt = self.agent.model.dt
        self.dt = dt
        # Local import to avoid import-time cycle with modules package
        from .locomotor import Locomotor as _Locomotor

        self.locomotor: Locomotor = _Locomotor(conf=conf, dt=self.dt)
        self.modalities = util.AttrDict(
            {
                "olfaction": {
                    "sensor": self.olfactor,
                    "func": self.sense_odors,
                    "A": 0.0,
                    "mem": None,
                },
                "touch": {
                    "sensor": self.toucher,
                    "func": self.sense_food_multi,
                    "A": 0.0,
                    "mem": None,
                },
                "thermosensation": {
                    "sensor": self.thermosensor,
                    "func": self.sense_thermo,
                    "A": 0.0,
                    "mem": None,
                },
                "windsensation": {
                    "sensor": self.windsensor,
                    "func": self.sense_wind,
                    "A": 0.0,
                    "mem": None,
                },
            }
        )
        self.init_sensors()

        m = conf["memory"]
        if m is not None:
            M = self.modalities[m.modality]
            if M.sensor:
                m.gain = M.sensor.gain
                kws = {"dt": dt, "brain": self}
                M.mem = MD.build_memory_module(conf=m, **kws)

    def init_sensors(self) -> None:
        """Mount the touch sensors on the agent's body."""
        if self.toucher is not None and self.agent is not None:
            self.agent.add_touch_sensors(self.toucher.touch_sensors)
            for s in self.agent.touch_sensorIDs:
                if s not in self.toucher.gain:
                    self.toucher.add_novel_gain(id=s, gain=self.toucher.initial_gain)

    def sense_odors(self, pos: Any | None = None) -> dict[str, Any]:
        """Sample the odor concentrations at a position.

        Args:
            pos: The position to sample. Defaults to the agent's olfactor
                position.

        Returns:
            The concentration of each odor, empty when no odor layers exist.
        """
        try:
            a = self.agent
            if pos is None:
                pos = a.olfactor_pos
            return {id: l.get_value(pos) for id, l in a.model.odor_layers.items()}
        except:
            return {}

    def sense_food_multi(self, **kwargs: Any) -> dict[Any, int]:
        """Sample food contact at each of the agent's touch sensors.

        Args:
            **kwargs: Forwarded to the agent's contact detection.

        Returns:
            One flag per sensor, indicating contact with food.
        """
        try:
            a = self.agent
            kws = {
                "sources": a.model.sources,
                "grid": a.model.food_grid,
                "radius": a.radius,
            }
            return {
                s: int(util.sense_food(pos=a.get_sensor_position(s), **kws) is not None)
                for s in a.touch_sensorIDs
            }
        except:
            return {}

    def sense_wind(self, **kwargs: Any) -> dict[str, float]:
        """Sample the wind the agent currently feels.

        Args:
            **kwargs: Accepted for signature compatibility; unused.

        Returns:
            The perceived wind speed, empty when the environment has no wind.
        """
        try:
            a = self.agent
            return {"windsensor": a.model.windscape.get_value(a)}
        except:
            return {"windsensor": 0.0}

    def sense_thermo(self, pos: Any | None = None) -> dict[str, float]:
        """Sample the thermal stimulus at a position.

        Args:
            pos: The position to sample, in relative arena coordinates.
                Defaults to the agent's own position.

        Returns:
            The cooling and warming stimulus magnitudes, empty when the
            environment has no thermal field.
        """
        try:
            a = self.agent
            if pos is None:
                pos = a.pos
            ad = a.model.space.dims
            return a.model.thermoscape.get_value(
                [(pos[0] + (ad[0] * 0.5)) / ad[0], (pos[1] + (ad[1] * 0.5)) / ad[1]]
            )
        except AttributeError:
            return {"cool": 0, "warm": 0}

    def sense(self, pos: Any | None = None, reward: bool = False) -> None:
        """Sample every configured modality and update its sensor.

        Each sensor's gains are first adapted by its memory module, so that
        learning takes effect before the stimulus is transduced.

        Args:
            pos: The position to sample from. Defaults to the agent's own.
            reward: Whether the agent is currently rewarded, which drives the
                associative memories.
        """
        kws = {"pos": pos}
        for m, M in self.modalities.items():
            if M.sensor:
                M.sensor.update_gain_via_memory(mem=M.mem, reward=reward)
                M.A = M.sensor.step(M.func(**kws))

    @property
    def A_in(self) -> float:
        """The summed activation from every sensory modality."""
        return np.sum([M.A for m, M in self.modalities.items()])
        # return self.A_olf + self.A_touch + self.A_thermo + self.A_wind

    @property
    def A_olf(self) -> float:
        """The activation contributed by olfaction."""
        return self.modalities["olfaction"].A

    @property
    def A_touch(self) -> float:
        """The activation contributed by touch."""
        return self.modalities["touch"].A

    @property
    def A_thermo(self) -> float:
        """The activation contributed by thermosensation."""
        return self.modalities["thermosensation"].A

    @property
    def A_wind(self) -> float:
        """The activation contributed by wind sensing."""
        return self.modalities["windsensation"].A


class DefaultBrain(Brain):
    """
    Default larva brain implementation with full sensory integration.

    Extends base Brain with automatic sensor module construction
    and integrated locomotor stepping. Provides complete sensorimotor
    loop for autonomous larva agents.

    Args:
        conf: Brain configuration dict with keys:
              - 'olfactor': Olfactory sensor config (or None)
              - 'toucher': Tactile sensor config (or None)
              - 'windsensor': Wind sensor config (or None)
              - 'thermosensor': Thermal sensor config (or None)
              - 'memory': Memory module config (or None)
              - 'locomotor': Locomotor module config
        agent: Parent agent instance (LarvaRobot, LarvaSim, or compatible).
               Must have attributes: model, radius, pos, touch_sensorIDs
        dt: Simulation time step (seconds). Defaults to agent.model.dt
        **kwargs: Additional keyword arguments passed to parent Brain

    Returns:
        Tuple of (linear_velocity, angular_velocity, crawl_flag)

    Example:
        >>> brain = DefaultBrain(conf=brain_config, agent=my_larva, dt=0.1)
        >>> lin_vel, ang_vel, crawling = brain.step(pos=larva.pos, on_food=False)
    """

    def __init__(
        self,
        conf: Any,
        agent: Any | None = None,
        dt: float | None = None,
        **kwargs: Any,
    ) -> None:
        if dt is None:
            dt = agent.model.dt
        kws = {"dt": dt, "brain": self}
        kwargs.update(MD.build_sensormodules(conf=conf, **kws))
        super().__init__(agent=agent, dt=dt, conf=conf, **kwargs)

    def step(
        self, pos: Any, on_food: bool = False, **kwargs: Any
    ) -> tuple[float, float, bool]:
        """Advance the brain by one timestep.

        Senses the environment, then drives the locomotor with the summed
        sensory activation.

        Args:
            pos: The agent's current position.
            on_food: Whether the agent sits on food.
            **kwargs: Forwarded to the locomotor step.

        Returns:
            The linear velocity, the angular velocity, and whether a feeding
            motion completed this timestep.
        """
        self.sense(pos=pos, reward=on_food)
        return self.locomotor.step(A_in=self.A_in, on_food=on_food, **kwargs)
