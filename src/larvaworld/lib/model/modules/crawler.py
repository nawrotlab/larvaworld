from __future__ import annotations
from typing import Any
import os
import warnings

# Deprecation: discourage deep imports from internal module paths
if os.getenv("LARVAWORLD_STRICT_DEPRECATIONS") == "1":
    raise ImportError(
        "Deep import path deprecated. Use public API: 'from larvaworld.lib.model.modules import Crawler'"
    )
else:
    warnings.warn(
        "Deep import path deprecated. Use public API: 'from larvaworld.lib.model.modules import Crawler'",
        DeprecationWarning,
        stacklevel=2,
    )
import numpy as np
import param
from scipy import signal

from ...param import Phase, PositiveNumber
from .basic import StepEffector, StepOscillator

__all__: list[str] = [
    "Crawler",
    "StrideOscillator",
    "GaussOscillator",
    "SquareOscillator",
    "PhaseOscillator",
]


class Crawler(StepEffector):
    """
    Base crawler module for peristaltic locomotion.

    Abstract base class for crawling behavior modules that generate
    forward locomotion through peristaltic waves. Extends StepEffector
    to provide stride-based movement with oscillatory patterns.

    Example:
        >>> # Use concrete subclasses like StrideOscillator, GaussOscillator
        >>> crawler = GaussOscillator(freq=1.5, stride_dst_mean=0.25)
    """

    pass


class StrideOscillator(Crawler, StepOscillator):
    """
    Stride-based oscillatory crawler with variable step length.

    Implements peristaltic crawling using frequency-based oscillation
    with stochastic stride lengths. Each stride distance is sampled
    from a normal distribution, providing realistic locomotion variability.

    Attributes:
        freq: Oscillation frequency in Hz (0.5-3.0)
        stride_dst_mean: Mean stride distance (fraction of body length)
        stride_dst_std: Stride distance standard deviation
        step_to_length: Current stride distance (resampled each stride)

    Example:
        >>> crawler = StrideOscillator(freq=1.42, stride_dst_mean=0.23, stride_dst_std=0.04)
        >>> velocity = crawler.step()
    """

    freq = PositiveNumber(1.42, bounds=(0.5, 3.0))
    stride_dst_mean = PositiveNumber(
        0.23,
        softmax=1.0,
        step=0.01,
        label="stride distance mean",
        doc="The mean displacement achieved in a single peristaltic stride as a fraction of the body length.",
    )
    stride_dst_std = PositiveNumber(
        0.04,
        softmax=1.0,
        step=0.001,
        label="stride distance std",
        doc="The standard deviation of the displacement achieved in a single peristaltic stride as a fraction of the body length.",
    )

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # print(self.stride_dst_std,self.stride_dst_mean)
        self.step_to_length = self.new_stride

    @property
    def new_stride(self) -> float:
        return np.random.normal(loc=self.stride_dst_mean, scale=self.stride_dst_std)

    @property
    def Act(self) -> float:
        return self.freq * self.step_to_length * (1 + self.Act_coef * self.Act_Phi)

    # def act(self):
    #     self.oscillate()
    #     self.output = self.Act

    def act_on_complete_iteration(self) -> None:
        self.step_to_length = self.new_stride

    def suppresion_relief(self, phi_range: tuple[float, float]) -> bool:
        return self.phi_in_range(phi_range)


class GaussOscillator(StrideOscillator):
    """
    Gaussian-windowed oscillatory crawler.

    Extends StrideOscillator with gaussian-shaped velocity modulation
    within each stride cycle. Provides smooth, biologically realistic
    acceleration/deceleration profiles during peristaltic crawling.

    Attributes:
        std: Standard deviation of gaussian window (fraction of cycle, 0-1)
        gauss_w: Precomputed 360-point gaussian window for cycle modulation

    Example:
        >>> crawler = GaussOscillator(freq=1.5, std=0.6, stride_dst_mean=0.25)
        >>> velocity = crawler.step()
    """

    # mode = param.Selector(default='gaussian', readonly=True)
    std = PositiveNumber(
        0.6,
        softmax=1.0,
        step=0.01,
        label="gaussian stride cycle std",
        doc="The std of the gaussian window for the velocity oscillation during a stride cycle.",
    )

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.gauss_w = signal.gaussian(360, std=self.std * 360, sym=False)

    @property
    def Act_Phi(self) -> float:
        return self.gauss_w[int(np.rad2deg(self.phi))]


class SquareOscillator(StrideOscillator):
    """
    Square-wave oscillatory crawler.

    Extends StrideOscillator with square-wave velocity modulation,
    creating distinct power/recovery phases in each stride cycle.

    Attributes:
        duty: Duty cycle fraction (0-1) for square wave modulation
              Controls percentage of time at maximum velocity

    Example:
        >>> crawler = SquareOscillator(freq=1.2, duty=0.6, stride_dst_mean=0.20)
        >>> velocity = crawler.step()
    """

    # mode = param.Selector(default='square', readonly=True)
    duty = param.Magnitude(
        0.6,
        step=0.01,
        label="square signal duty",
        doc="The duty parameter(%time at the upper end) of the square signal.",
    )

    @property
    def Act_Phi(self) -> float:
        return float(signal.square(self.phi, duty=self.duty))

    def suppresion_relief(self, phi_range: tuple[float, float]) -> bool:
        return self.phi <= 2 * np.pi * self.duty


class PhaseOscillator(StrideOscillator):
    """
    Phase-modulated oscillatory crawler with realistic velocity profile.

    Extends StrideOscillator with cosine-based phase modulation,
    producing the most biologically realistic peristaltic crawling patterns.
    Velocity peaks at a configurable phase within each stride cycle.

    Attributes:
        max_vel_phase: Phase angle (radians) where velocity is maximum
        max_scaled_vel: Maximum scaled forward velocity coefficient

    Example:
        >>> crawler = PhaseOscillator(freq=1.42, max_vel_phase=3.49, max_scaled_vel=0.51)
        >>> velocity = crawler.step()
    """

    # mode = param.Selector(default='realistic', readonly=True)
    max_vel_phase = Phase(
        3.49,
        label="max velocity phase",
        doc="The phase of the crawling oscillation cycle where forward velocity is maximum.",
    )
    max_scaled_vel = PositiveNumber(
        0.51,
        softmax=1.5,
        step=0.01,
        label="maximum scaled velocity",
        doc="The maximum scaled forward velocity.",
    )

    @property
    def Act_Phi(self) -> float:
        return np.cos(self.phi - self.max_vel_phase)

    @property
    def Act_coef(self) -> float:
        return self.max_scaled_vel
