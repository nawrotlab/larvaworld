from __future__ import annotations
from typing import Any
import os
import warnings

# Deprecation: discourage deep imports from internal module paths
if os.getenv("LARVAWORLD_STRICT_DEPRECATIONS") == "1":
    raise ImportError(
        "Deep import path deprecated. Use public API: 'from larvaworld.lib.model.modules import Sensor'"
    )
else:
    warnings.warn(
        "Deep import path deprecated. Use public API: 'from larvaworld.lib.model.modules import Sensor'",
        DeprecationWarning,
        stacklevel=2,
    )
import numpy as np
import param

from .... import vprint
from ....lib import util
from ...param import PositiveNumber, RangeRobust
from .basic import Effector
from .remote_brian_interface import RemoteBrianModelInterface

__all__: list[str] = [
    "Sensor",
    "Olfactor",
    "OSNOlfactor",
    "Toucher",
    "Windsensor",
    "Thermosensor",
]


class Sensor(Effector):
    """
    Base sensor module for agent sensory processing.

    Abstract base class providing sensory input processing with gain control,
    temporal dynamics (decay), and optional memory integration. Supports
    multiple perception modes (log, linear, null) for sensory transduction.

    Attributes:
        output_range: Valid output range for sensor activation (-1 to 1)
        perception: Sensory transduction mode ('log', 'linear', or 'null')
        decay_coef: Linear decay coefficient for sensory activation
        brute_force: If True, apply direct locomotor modulation (bypass normal output)
        gain_dict: Dictionary mapping stimulus IDs to gain coefficients
        brain: Parent brain instance (for locomotor access)
        X: Current sensory input values per stimulus ID
        dX: Perceived sensory changes per stimulus ID
        gain: Current gain coefficients per stimulus ID

    Args:
        brain: Parent brain instance (polymorphic: Brain or subclasses).
               Provides access to locomotor for brute_force modulation
        **kwargs: Additional keyword arguments passed to parent Effector

    Example:
        >>> sensor = Sensor(brain=my_brain, decay_coef=0.1, perception='log')
        >>> sensor.step(input={'odor1': 0.5, 'odor2': 0.3})
    """

    output_range = RangeRobust((-1.0, 1.0), readonly=True)
    perception = param.Selector(
        objects=["log", "linear", "null"],
        label="sensory transduction mode",
        doc="The method used to calculate the perceived sensory activation from the current and previous sensory input.",
    )
    decay_coef = PositiveNumber(
        0.1,
        softmax=2.0,
        step=0.01,
        label="sensory decay coef",
        doc="The linear decay coefficient of the olfactory sensory activation.",
    )
    brute_force = param.Boolean(
        False, doc="Whether to apply direct rule-based modulation on locomotion or not."
    )
    gain_dict = param.Dict(
        default=util.AttrDict(), doc="Dictionary of sensor gain per stimulus ID"
    )

    def __init__(self, brain: Any | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.brain = brain
        self.exp_decay_coef = np.exp(-self.dt * self.decay_coef)
        self.X = util.AttrDict({id: 0.0 for id in self.gain_ids})
        self.dX = util.AttrDict({id: 0.0 for id in self.gain_ids})
        self.gain = self.gain_dict

    def compute_dif(self, input: Any) -> None:
        pass

    def update_gain_via_memory(self, mem: Any | None = None, **kwargs: Any) -> None:
        if mem is not None:
            self.gain = mem.step(dx=self.get_dX(), **kwargs)

    def update(self) -> None:
        if len(self.input) == 0:
            self.output = 0
        elif self.brute_force:
            if self.brain is not None:
                self.affect_locomotion(L=self.brain.locomotor)
            self.output = 0
        else:
            self.compute_dX(self.input)
            self.output *= self.exp_decay_coef
            self.output += self.dt * np.sum(
                [self.gain[id] * self.dX[id] for id in self.gain_ids]
            )

    def affect_locomotion(self, L: Any) -> None:
        pass

    def get_dX(self) -> dict[str, float]:
        return self.dX

    def get_X_values(self, t: float, N: int) -> list[float]:
        return list(self.X.values())

    def get_gain(self) -> dict[str, float]:
        return self.gain

    def set_gain(self, value: float, gain_id: str) -> None:
        self.gain[gain_id] = value

    def reset_gain(self, gain_id: str) -> None:
        self.gain[gain_id] = self.gain_dict[gain_id]

    def reset_all_gains(self) -> None:
        self.gain = self.gain_dict

    def compute_single_dx(self, cur: float, prev: float) -> float:
        if self.perception == "log":
            return cur / prev - 1 if prev != 0 else 0
        elif self.perception == "linear":
            return cur - prev if prev != 0 else 0
        elif self.perception == "null":
            return cur

    def compute_dX(self, input: dict[str, float]) -> None:
        for id, cur in input.items():
            if id not in self.X:
                self.add_novel_gain(id, con=cur)
            else:
                prev = self.X[id]
                self.dX[id] = self.compute_single_dx(cur, prev)
        self.X = input

    def add_novel_gain(self, id: str, con: float = 0.0, gain: float = 0.0) -> None:
        self.gain_dict[id] = gain
        self.gain[id] = gain
        self.dX[id] = 0.0
        self.X[id] = con

    @property
    def gain_ids(self) -> list[str]:
        return list(self.gain_dict.keys())


class Olfactor(Sensor):
    """
    Olfactory sensor module for odor detection and chemotaxis.

    Extends base Sensor with olfaction-specific locomotor modulation.
    When negative gradients detected and stride completes, probabilistically
    triggers locomotor interruption (reorientation behavior).

    Attributes:
        Inherits all Sensor attributes
        Provides odor concentration and change properties for first/second odors

    Example:
        >>> olfactor = Olfactor(brain=my_brain, decay_coef=0.15, perception='log')
        >>> olfactor.step(input={'odor_A': 0.6, 'odor_B': 0.2})
        >>> print(f"First odor: {olfactor.first_odor_concentration}")
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def affect_locomotion(self, L: Any) -> None:
        if self.output < 0 and L.stride_completed:
            if np.random.uniform(0, 1, 1) <= np.abs(self.output):
                L.intermitter.interrupt_locomotion()

    @property
    def first_odor_concentration(self) -> float:
        return list(self.X.values())[0]

    @property
    def second_odor_concentration(self) -> float:
        return list(self.X.values())[1]

    @property
    def first_odor_concentration_change(self) -> float:
        return list(self.dX.values())[0]

    @property
    def second_odor_concentration_change(self) -> float:
        return list(self.dX.values())[1]


class Toucher(Sensor):
    """
    Tactile sensor module for contact detection.

    Extends base Sensor with touch-specific locomotor modulation.
    Triggers locomotion on contact detection (+1) and interrupts on
    contact loss (-1). Uses multiple sensor points around body contour.

    Attributes:
        initial_gain: Initial tactile sensitivity coefficient (default: 40.0)
        touch_sensors: List of sensor location indices around body contour

    Example:
        >>> toucher = Toucher(brain=my_brain, initial_gain=40.0, touch_sensors=[0, 5, 10])
        >>> toucher.step(input={'sensor_0': 1, 'sensor_5': 0})
    """

    initial_gain = PositiveNumber(
        40.0,
        label="tactile sensitivity coef",
        doc="The initial gain of the tactile sensor.",
    )
    touch_sensors = param.List(
        default=[],
        item_type=int,
        doc="The location indexes of sensors around body contour.",
    )

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def affect_locomotion(self, L: Any) -> None:
        for id in self.gain_ids:
            if self.dX[id] == 1:
                L.intermitter.trigger_locomotion()
                break
            elif self.dX[id] == -1:
                L.intermitter.interrupt_locomotion()
                break


class Windsensor(Sensor):
    """
    Wind sensor module for air flow detection.

    Extends base Sensor for wind stimulus processing with fixed gain
    and null perception mode (direct transduction without adaptation).

    Attributes:
        gain_dict: Fixed gain for wind sensor (default: 1.0)
        perception: Fixed to 'null' mode (direct input)
        weights: Wind response weight coefficients (polymorphic structure)

    Example:
        >>> windsensor = Windsensor(weights=wind_weights, brain=my_brain)
        >>> windsensor.step(input={'windsensor': 0.7})
    """

    gain_dict = param.Dict(default=util.AttrDict({"windsensor": 1.0}))
    perception = param.Selector(default="null")

    def __init__(self, weights: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.weights = weights


class Thermosensor(Sensor):
    """
    Temperature sensor module for thermal gradient detection.

    Extends base Sensor for thermotaxis with separate warm/cool
    sensory channels. Provides dual-channel temperature perception
    for thermal navigation behaviors.

    Attributes:
        gain_dict: Gains for warm and cool sensors (default: both 1.0)
        Properties for warm/cool sensor inputs, perceptions, and gains

    Example:
        >>> thermosensor = Thermosensor(brain=my_brain, decay_coef=0.1)
        >>> thermosensor.step(input={'warm': 0.4, 'cool': 0.6})
        >>> print(f"Warm gain: {thermosensor.warm_gain}")
    """

    gain_dict = param.Dict(default=util.AttrDict({"warm": 1.0, "cool": 1.0}))
    # cool_gain = PositiveNumber(0.0, label='cool sensitivity coef', doc='The gain of the cool sensor.')
    # warm_gain = PositiveNumber(0.0, label='warm sensitivity coef', doc='The gain of the warm sensor.')

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @property
    def warm_sensor_input(self) -> float:
        return self.X["warm"]

    @property
    def warm_sensor_perception(self) -> float:
        return self.dX["warm"]

    @property
    def cool_sensor_input(self) -> float:
        return self.X["cool"]

    @property
    def cool_sensor_perception(self) -> float:
        return self.dX["cool"]

    @property
    def cool_gain(self) -> float:
        return self.gain["cool"]

    @property
    def warm_gain(self) -> float:
        return self.gain["warm"]


class OSNOlfactor(Olfactor):
    """
    Olfactory Sensory Neuron (OSN) olfactor with Brian2 neural simulation.

    Extends Olfactor with biologically realistic OSN dynamics via remote
    Brian2 server. Converts odor concentrations to neural spike rates
    through detailed OSN model, then applies sigmoid normalization.

    Attributes:
        brianInterface: Remote Brian2 model interface for OSN simulation
        brian_warmup: Warmup steps for neural model initialization
        response_key: Key for extracting response from Brian2 model (default: 'OSN_rate')
        remote_dt: Time step for remote Brian2 simulation (ms)
        agent_id: Unique agent identifier for Brian2 tracking
        sim_id: Unique simulation identifier for Brian2 tracking

    Args:
        response_key: Brian2 response parameter to extract (default: 'OSN_rate')
        server_host: Brian2 server hostname (default: 'localhost')
        server_port: Brian2 server port (default: 5795)
        remote_dt: Brian2 simulation time step in ms (default: 100)
        remote_warmup: Brian2 warmup steps before data collection (default: 500)
        **kwargs: Additional keyword arguments passed to parent Olfactor

    Example:
        >>> osn_olfactor = OSNOlfactor(
        ...     brain=my_brain,
        ...     server_host='localhost',
        ...     server_port=5795,
        ...     remote_dt=100
        ... )
        >>> osn_olfactor.step(input={'odor_A': 0.8})
    """

    def __init__(
        self,
        response_key: str = "OSN_rate",
        server_host: str = "localhost",
        server_port: int = 5795,
        remote_dt: int = 100,
        remote_warmup: int = 500,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.brianInterface = RemoteBrianModelInterface(
            server_host, server_port, remote_dt
        )
        self.brian_warmup = remote_warmup
        self.response_key = response_key
        self.remote_dt = remote_dt
        self.agent_id = RemoteBrianModelInterface.getRandomModelId()
        self.sim_id = RemoteBrianModelInterface.getRandomModelId()

    def normalized_sigmoid(self, a: float, b: float, x: float) -> float:
        """
        Returns array of a horizontal mirrored normalized sigmoid function
        output between 0 and 1
        Function parameters a = center; b = width
        """
        s = 1 / (1 + np.exp(b * (x - a)))
        # return 1 * (s - min(s)) / (max(s) - min(s))  # normalize function to 0-1
        return s

    def update(self) -> None:
        agent_id = (
            self.brain.agent.unique_id if self.brain is not None else self.agent_id
        )
        sim_id = self.brain.agent.model.id if self.brain is not None else self.sim_id

        msg_kws = {
            # Default :
            # TODO: can we get this info from somewhere ?
            # yes: self.X.values() provides an array of all odor types, the index could be used as odor_id
            "odor_id": 0,
            # The concentration change :
            "concentration_mmol": self.first_odor_concentration,  # 1st ODOR concentration
            "concentration_change_mmol": self.first_odor_concentration_change,  # 1st ODOR concentration change
        }

        try:
            response = self.brianInterface.executeRemoteModelStep(
                sim_id, agent_id, self.remote_dt, t_warmup=self.brian_warmup, **msg_kws
            )
            self.output = self.normalized_sigmoid(
                15, 8, response.param(self.response_key)
            )
        except ConnectionRefusedError:
            vprint(
                f"WARNING: Unable to reach remote brian server at {self.brianInterface.server_host} - is the server running ?"
            )
            pass

        super().update()
