from __future__ import annotations
from typing import Any
import os
import warnings

# Deprecation: discourage deep imports from internal module paths
if os.getenv("LARVAWORLD_STRICT_DEPRECATIONS") == "1":
    raise ImportError(
        "Deep import path deprecated. Use public API: 'from larvaworld.lib.model.modules import Oscillator'"
    )
else:
    warnings.warn(
        "Deep import path deprecated. Use public API: 'from larvaworld.lib.model.modules import Oscillator'",
        DeprecationWarning,
        stacklevel=2,
    )
import numpy as np
import param

from ...param import PositiveNumber, RandomizedPhase

__all__: list[str] = [
    "Timer",
    "Oscillator",
]


# class Timer(NestedConf):
class Timer(param.Parameterized):
    """
    Base timer module for time tracking and activation control.

    Provides time-step counting, activation state management, and
    iteration tracking. Base class for all time-dependent modules.

    Attributes:
        dt: Simulation time step in seconds
        ticks: Current elapsed ticks since last reset
        total_ticks: Total ticks since initialization
        active: Whether timer/module is currently active
        complete_iteration: Flag for iteration completion

    Example:
        >>> timer = Timer(dt=0.1)
        >>> timer.count_time()
        >>> print(f"Elapsed: {timer.t} seconds")
    """

    dt = PositiveNumber(
        0.1,
        precedence=2,
        softmax=1.0,
        step=0.01,
        label="simulation timestep",
        doc="The timestep of the simulation in seconds.",
    )

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.ticks = 0
        self.total_ticks = 0

        self.active = True
        self.complete_iteration = False

    def count_time(self) -> None:
        self.ticks += 1
        self.total_ticks += 1

    @property
    def t(self) -> float:
        return self.ticks * self.dt

    @property
    def total_t(self) -> float:
        return self.total_ticks * self.dt

    def reset(self) -> None:
        self.ticks = 0
        self.total_ticks = 0

    def start_effector(self) -> None:
        self.active = True

    def stop_effector(self) -> None:
        self.active = False
        self.ticks = 0


class Oscillator(Timer):
    """
    Oscillator module for phase-based periodic behaviors.

    Extends Timer with phase tracking and oscillation mechanics.
    Manages phase progression, iteration detection, and frequency control
    for all oscillatory behavioral modules.

    Attributes:
        freq: Oscillation frequency in Hz
        phi: Current oscillation phase in radians (0-2π)
        initial_freq: Frequency at initialization (stored for reset)
        iteration_counter: Number of completed oscillation cycles
        complete_iteration: True when phase completes full 2π cycle

    Args:
        random_phi: If True, randomize initial phase (default: True)
        **kwargs: Additional keyword arguments passed to parent Timer

    Example:
        >>> oscillator = Oscillator(freq=1.5, random_phi=False)
        >>> oscillator.oscillate()
        >>> print(f"Phase: {oscillator.phi}, Completed: {oscillator.complete_iteration}")
    """

    freq = PositiveNumber(
        label="oscillation frequency", doc="The initial frequency of the oscillator."
    )
    phi = RandomizedPhase(
        precedence=-1, label="oscillation phase", doc="The phase of the oscillation."
    )

    def __init__(self, random_phi: bool = True, **kwargs: Any) -> None:
        if "phi" not in kwargs and not random_phi:
            kwargs["phi"] = 0.0
        super().__init__(**kwargs)
        self.initial_freq = self.freq

        self.iteration_counter = 0
        # self.complete_iteration = False

    def set_freq(self, v: float) -> None:
        self.freq = v

    def get_freq(self, t: float) -> float:
        return self.freq

    def oscillate(self) -> None:
        self.complete_iteration = False
        phi = self.phi + 2 * np.pi * self.dt * self.freq
        if phi >= 2 * np.pi:
            phi %= 2 * np.pi
            self.complete_iteration = True
            self.act_on_complete_iteration()
            self.iteration_counter += 1
        self.phi = phi

    def act_on_complete_iteration(self) -> None:
        pass

    def reset(self) -> None:
        # self.ticks = 0
        # self.total_ticks = 0
        self.phi = 0
        self.complete_iteration = False
        self.iteration_counter = 0

    def update(self) -> None:
        self.complete_iteration = False

    def phi_in_range(self, phi_range: tuple[float, float]) -> bool:
        return phi_range[0] < self.phi < phi_range[1]

    @property
    def Act_Phi(self) -> float:
        return self.phi
