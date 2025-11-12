from __future__ import annotations
from typing import Any
import os
import warnings

# Deprecation: discourage deep imports from internal module paths
if os.getenv("LARVAWORLD_STRICT_DEPRECATIONS") == "1":
    raise ImportError(
        "Deep import path deprecated. Use public API: 'from larvaworld.lib.model.modules import Memory'"
    )
else:
    warnings.warn(
        "Deep import path deprecated. Use public API: 'from larvaworld.lib.model.modules import Memory'",
        DeprecationWarning,
        stacklevel=2,
    )
import itertools
import random

import numpy as np
import param

from .... import vprint
from ...ipc import BrianInterfaceMessage, Client
from ...param import PositiveInteger, PositiveNumber
from .oscillator import Timer

__all__: list[str] = [
    "Memory",
    "RLmemory",
    "RLOlfMemory",
    "RLTouchMemory",
    "RemoteBrianModelMemory",
]


class Memory(Timer):
    """
    Base memory module for reinforcement learning and plasticity.

    Abstract base class providing memory-based gain adaptation for
    sensory processing. Supports reinforcement learning (RL) and
    mushroom body (MB) algorithms for sensory gain modulation.

    Attributes:
        mode: Memory algorithm type ('RL' or 'MB')
        modality: Sensory modality ('olfaction' or 'touch')
        brain: Parent brain instance (polymorphic)
        gain: Current gain values per stimulus ID
        rewardSum: Cumulative reward for RL updates

    Args:
        brain: Parent brain instance (polymorphic: Brain or subclasses).
               Provides access to agent for state tracking
        gain: Initial gain dictionary mapping stimulus IDs to coefficients
        **kwargs: Additional keyword arguments passed to parent Timer

    Example:
        >>> memory = Memory(brain=my_brain, gain={'odor1': 1.0}, modality='olfaction')
        >>> updated_gain = memory.step(reward=True, dx={'odor1': 0.3})
    """

    mode = param.Selector(objects=["RL", "MB"], doc="The memory algorithm")
    modality = param.Selector(
        objects=["olfaction", "touch"], doc="The sensory modality"
    )

    def __init__(
        self, brain: Any | None = None, gain: dict[str, float] = {}, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.brain = brain
        self.gain = gain
        self.rewardSum = 0

    def step(self, reward: bool = False, **kwargs: Any) -> dict[str, float]:
        if self.active:
            self.count_time()
        self.rewardSum += int(reward) - 0.01
        self.update_gain(**kwargs)
        return self.gain

    def update_gain(self, dx: dict[str, float] | None = None, **kwargs: Any) -> None:
        pass


class RLmemory(Memory):
    """
    Reinforcement learning memory module with Q-learning.

    Implements Q-learning algorithm for sensory gain adaptation based
    on reward feedback. Discretizes state space and learns optimal
    gain values through exploration and exploitation.

    Attributes:
        mode: Fixed to 'RL' (reinforcement learning)
        update_dt: Time interval between gain updates (seconds)
        train_dur: Training duration before stopping learning (seconds)
        Delta: Input sensitivity for state discretization
        alpha: Learning rate for Q-table updates (0-1)
        gamma: Discount factor for future rewards (0-1)
        epsilon: Exploration rate for random action selection (0-1)
        state_spacePerSide: Number of discrete states per side of zero
        state_specific_best: If True, use state-specific best actions
        gain_space: Possible gain values to choose from
        q_table: Q-learning table (states × actions)

    Example:
        >>> rl_memory = RLmemory(
        ...     brain=my_brain,
        ...     gain={'odor1': 0.0},
        ...     alpha=0.05,
        ...     gamma=0.6,
        ...     gain_space=[-300, -50, 50, 300]
        ... )
        >>> updated_gain = rl_memory.step(reward=True, dx={'odor1': 0.5})
    """

    mode = param.Selector(default="RL", readonly=True)
    update_dt = PositiveNumber(
        1.0,
        precedence=2,
        softmax=10.0,
        step=0.01,
        label="gain-update timestep",
        doc="The interval duration between gain switches.",
    )
    train_dur = PositiveNumber(
        30.0,
        precedence=2,
        step=0.01,
        label="training duration",
        doc="The duration of the training period after which no further learning will take place.",
    )
    Delta = PositiveNumber(0.1, doc="The input sensitivity of the memory.")
    alpha = PositiveNumber(
        0.05, doc="The alpha parameter of reinforcement learning algorithm."
    )
    gamma = PositiveNumber(
        0.6,
        doc="The probability of sampling a random gain rather than exploiting the currently highest evaluated gain for the current state.",
    )
    epsilon = PositiveNumber(
        0.15, doc="The epsilon parameter of reinforcement learning algorithm."
    )
    state_spacePerSide = PositiveInteger(
        0,
        doc="The number of discrete states to parse the state space on either side of 0.",
    )
    state_specific_best = param.Boolean(
        True, doc="Whether to select the best action for each state"
    )
    gain_space = param.List(
        default=[-300.0, -50.0, 50.0, 300.0],
        item_type=float,
        doc="The possible values for memory gain to choose from.",
    )

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        n = len(self.gain)
        self.Niters = int(self.update_dt * 60 / self.dt)
        self.iterator = self.Niters
        self.state_space = np.array(
            [
                ii
                for ii in itertools.product(
                    range(2 * self.state_spacePerSide + 1), repeat=n
                )
            ]
        )
        self.actions = [ii for ii in itertools.product(self.gain_space, repeat=n)]
        self.q_table = np.zeros((self.state_space.shape[0], len(self.actions)))

        self.lastAction = 0
        self.lastState = 0

    def update_q_table(self, state: int, reward: float) -> None:
        self.q_table[self.lastState, self.lastAction] = (1 - self.alpha) * self.q_table[
            self.lastState, self.lastAction
        ] + self.alpha * (reward + self.gamma * np.max(self.q_table[state]))
        self.lastState = state

    def state_collapse(self, dx: dict[str, float]) -> int:
        k = self.state_spacePerSide
        if len(dx) > 0:
            dx = [dx]
        v = []
        for j in range(len(dx)):
            for i in dx[j]:
                dxI = dx[j][i]
                su = np.sum([np.abs(dxI) > (ii + 1) * self.Delta for ii in range(k)])
            v.append(int(np.sign(dxI) * su + k))
        state = np.where((self.state_space == v).all(axis=1))[0][0]
        return state

    def update_ext_gain(
        self,
        gain: dict[str, float] = {},
        dx: dict[str, float] = {},
        randomize: bool = True,
    ) -> dict[str, float]:
        gain_ids = list(gain.keys())
        if randomize and random.uniform(0, 1) < self.epsilon:
            actionID = random.randrange(len(self.actions))
        else:
            state = self.state_collapse(dx)
            self.update_q_table(state, self.rewardSum)
            actionID = np.argmax(self.q_table[state])  # Exploit learned values
        self.lastAction = actionID
        for ii, id in enumerate(gain_ids):
            gain[id] = self.actions[actionID][ii]
        return gain

    def update_gain(self, dx: dict[str, float] | None = None, **kwargs: Any) -> None:
        if dx is None:
            dx = {}
        if self.learning_on:
            self.iterator += 1
            if self.condition(dx):
                self.gain = self.update_ext_gain(self.gain, dx=dx, randomize=True)
                self.rewardSum = 0
                self.iterator = 0
        else:
            if not self.state_specific_best:
                self.gain = self.best_gain
            else:
                self.gain = self.update_ext_gain(self.gain, dx=dx, randomize=False)

    def condition(self, dx: dict[str, float]) -> bool:
        return self.iterator >= self.Niters

    @property
    def best_actions(self) -> tuple[float, ...]:
        return self.actions[np.argmax(np.mean(self.q_table, axis=0))]

    @property
    def best_gain(self) -> dict[str, float]:
        gain_ids = list(self.gain.keys())
        return dict(zip(gain_ids, self.best_actions))

    @property
    def learning_on(self) -> bool:
        return self.active and self.total_t <= self.train_dur * 60


class RLOlfMemory(RLmemory):
    """
    Reinforcement learning memory for olfactory stimuli.

    Specializes RLmemory for olfaction modality with properties
    for accessing best gain values for first/second odors.

    Attributes:
        modality: Fixed to 'olfaction'

    Example:
        >>> olf_memory = RLOlfMemory(brain=my_brain, gain={'odor_A': 0.0, 'odor_B': 0.0})
        >>> print(f"Best gain for first odor: {olf_memory.first_odor_best_gain}")
    """

    modality = param.Selector(default="olfaction", readonly=True)

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @property
    def first_odor_best_gain(self) -> float:
        return list(self.best_gain.values())[0]

    @property
    def second_odor_best_gain(self) -> float:
        return list(self.best_gain.values())[1]


class RLTouchMemory(RLmemory):
    """
    Reinforcement learning memory for tactile stimuli.

    Specializes RLmemory for touch modality with custom condition
    logic that triggers updates on contact detection (±1 changes).

    Attributes:
        modality: Fixed to 'touch'

    Example:
        >>> touch_memory = RLTouchMemory(brain=my_brain, gain={'sensor_0': 0.0})
        >>> updated_gain = touch_memory.step(reward=False, dx={'sensor_0': 1})
    """

    modality = param.Selector(default="touch", readonly=True)

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def condition(self, dx: dict[str, int]) -> bool:
        if 1 in dx.values() or -1 in dx.values():
            if 1 in dx.values():
                self.rewardSum = 1 / self.iterator
            elif -1 in dx.values():
                self.rewardSum = self.iterator
            return True
        else:
            return False


class RemoteBrianModelMemory(Memory):
    """
    Mushroom body memory using remote Brian2 neural simulation.

    Implements biologically realistic mushroom body (MB) plasticity
    via remote Brian2 server. Computes gain modulation based on
    MBON (mushroom body output neuron) differential activity.

    Attributes:
        mode: Fixed to 'MB' (mushroom body)
        server_host: Brian2 server hostname
        server_port: Brian2 server port
        sim_id: Simulation identifier for Brian2 tracking
        G: Gain scaling coefficient for MBON output
        t_sim: Simulation time step in milliseconds
        step_id: Current step counter for Brian2 synchronization

    Args:
        G: Gain scaling coefficient (default: 0.001)
        server_host: Brian2 server hostname (default: 'localhost')
        server_port: Brian2 server port (default: 5795)
        **kwargs: Additional keyword arguments passed to parent Memory

    Example:
        >>> mb_memory = RemoteBrianModelMemory(
        ...     brain=my_brain,
        ...     gain={'Odor': 0.0},
        ...     G=0.001,
        ...     server_host='localhost'
        ... )
        >>> updated_gain = mb_memory.step(reward=True, dx={'Odor': 0.8})
    """

    mode = param.Selector(default="MB", readonly=True)

    def __init__(
        self,
        G: float = 0.001,
        server_host: str = "localhost",
        server_port: int = 5795,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.server_host = server_host
        self.server_port = server_port
        self.sim_id = self.brain.agent.model.id
        self.G = G
        self.t_sim = int(self.dt * 1000)
        self.step_id = 0

    def runRemoteModel(
        self,
        model_instance_id: str,
        odor_id: int,
        t_sim: int = 100,
        t_warmup: int = 0,
        concentration: float = 1,
        **kwargs: Any,
    ) -> float:
        # T: duration of remote model simulation in ms
        # warmup: duration of remote model warmup in ms
        msg = BrianInterfaceMessage(
            self.sim_id,
            model_instance_id,
            self.step_id,
            odor_id=odor_id,
            odor_concentration=concentration,
            T=t_sim,
            warmup=t_warmup,
            step_id=self.step_id,
            **kwargs,
        )
        # send model parameters to remote model server & wait for result response
        try:
            with Client((self.server_host, self.server_port)) as client:
                [response] = client.send([msg])  # this is a LarvaMessage object again
                # extract returned model results
                mbon_p = response.param("MBONp")
                mbon_n = response.param("MBONn")
                mbon_dif = mbon_p - mbon_n
                return mbon_dif
                # return response.param('preference_index')
        except ConnectionRefusedError:
            vprint(
                f"**** WARNING ****: Unable to connect to RemoteBrianInterface (host={self.server_host} port={self.server_port})"
            )
            vprint(
                f"Verify server instance is up and running at host={self.server_host} port={self.server_port}"
            )
            return 0

    def step(
        self,
        dx: dict[str, float] | None = None,
        reward: bool = False,
        t_warmup: int = 0,
    ):
        # Default message arguments
        if dx is None:
            dx = {}
        msg_kws0 = {
            "model_instance_id": self.brain.agent.unique_id,
            "t_sim": self.t_sim,
            "t_warmup": t_warmup,
        }

        # Let's focus on the CS odor only :
        msg_kws = {
            # Default :
            "odor_id": 0,
            # The concentration change :
            "concentration": dx["Odor"],
            # reward as 0 or 1
            "reward": int(reward),
        }

        mbon_dif = self.runRemoteModel(**msg_kws0, **msg_kws)
        self.gain["Odor"] = self.G * mbon_dif
        self.step_id += 1
        return self.gain
