"""
Base classes shared by the behavioural effector modules.

Defines the effector interface, its stepwise and oscillatory specializations,
and the adapter that lets an effector be driven by a Nengo network.
"""

from __future__ import annotations
from typing import Any

import numpy as np
import param

from ...param import PositiveNumber
from ..modules.oscillator import Oscillator, Timer

__all__: list[str] = [
    "Effector",
    "StepEffector",
    "StepOscillator",
    "SinOscillator",
    "NengoEffector",
]


class Effector(Timer):
    """
    Base effector module for behavioral output generation.

    Abstract base class for all behavioral modules that produce motor
    outputs (crawlers, turners, feeders, sensors). Provides noise
    application, activation control, and input/output processing.

    Attributes:
        input_noise: Gaussian noise magnitude applied to input (0-1)
        output_noise: Gaussian noise magnitude applied to output (0-1)
        input_range: Valid input range (min, max)
        output_range: Valid output range (min, max)
        input: Current input value
        output: Current output value
        active: Whether effector is currently active

    Example:
        >>> # Use concrete subclasses like StepEffector, Crawler, Turner
        >>> effector = StepEffector(amp=0.5, input_noise=0.1)
        >>> output = effector.step(A_in=0.3)
    """

    input_noise = param.Magnitude(
        0.0,
        step=0.01,
        precedence=-3,
        label="input noise",
        doc="The noise applied at the input of the module.",
    )
    output_noise = param.Magnitude(
        0.0,
        step=0.01,
        precedence=-3,
        label="output noise",
        doc="The noise applied at the output of the module.",
    )
    input_range = param.Range(
        precedence=-3, label="input range", doc="The input range of the module."
    )
    output_range = param.Range(
        precedence=-3, label="output range", doc="The output range of the module."
    )

    def __init__(self, **kwargs: Any) -> None:
        """Build the effector with zeroed input and output.

        Args:
            **kwargs: Effector parameters, forwarded to the parent class.
        """
        super().__init__(**kwargs)
        self.input = 0
        self.output = 0

    def update_output(self, output: Any) -> Any:
        """Apply the configured output noise and range.

        Args:
            output: The computed output.

        Returns:
            The noisy, range-limited output.
        """
        return self.apply_noise(output, self.output_noise, self.output_range)

    def update_input(self, input: Any) -> Any:
        """Apply the configured input noise and range.

        Args:
            input: The incoming signal.

        Returns:
            The noisy, range-limited input.
        """
        return self.apply_noise(input, self.input_noise, self.input_range)

    def apply_noise(
        self, value: Any, noise: float = 0, range: Any | None = None
    ) -> Any:
        """Add multiplicative Gaussian noise and clip to a range.

        Args:
            value: The value to perturb. Scalars and arrays are both accepted.
            noise: The relative noise magnitude. Zero leaves the value intact.
            range: The ``(min, max)`` limits to clip to, if any.

        Returns:
            The perturbed value.
        """
        if type(value) in [int, float]:
            value *= 1 + np.random.normal(scale=noise)
            if range is not None and len(range) == 2:
                A0, A1 = range
                if value > A1:
                    value = A1
                elif value < A0:
                    value = A0
        elif isinstance(value, dict):
            for k, v in value.items():
                value[k] = self.apply_noise(v, noise)
        else:
            pass
        return value

    def get_output(self, t: float) -> float:
        """Return the current output.

        Args:
            t: The current time. Accepted for signature compatibility.

        Returns:
            The effector's output.
        """
        return self.output

    def update(self) -> None:
        """Hook run each timestep before acting. Does nothing by default."""
        pass

    def act(self, **kwargs: Any) -> None:
        """Produce the output while the effector is active.

        Args:
            **kwargs: Subclass-specific arguments.
        """
        pass

    def inact(self, **kwargs: Any) -> None:
        """Produce the output while the effector is inactive.

        Args:
            **kwargs: Subclass-specific arguments.
        """
        pass

    def step(self, A_in: float = 0, **kwargs: Any) -> Any:
        """Advance the effector by one timestep.

        Applies input noise, runs the per-step update, then acts or idles
        according to whether the effector is active, and applies output noise.

        Args:
            A_in: The incoming activation.
            **kwargs: Forwarded to :meth:`act` or :meth:`inact`.

        Returns:
            The effector's output for this timestep.
        """
        self.input = self.update_input(A_in)
        self.update()
        if self.active:
            self.act(**kwargs)
        else:
            self.inact(**kwargs)
        self.output = self.update_output(self.output)
        return self.output


class StepEffector(Effector):
    """
    Step-based effector with amplitude control.

    Extends Effector with amplitude-based activation (Act = amp × phase).
    Base class for step-driven behaviors (crawling, constant turning).

    Attributes:
        amp: Oscillation amplitude coefficient
        Act_coef: Activation coefficient (returns amp)
        Act_Phi: Activation phase modulation (returns 1 for constant)
        Act: Total activation (Act_coef × Act_Phi)

    Example:
        >>> step_eff = StepEffector(amp=0.8)
        >>> output = step_eff.step()
    """

    amp = PositiveNumber(
        1.0,
        allow_None=True,
        label="oscillation amplitude",
        doc="The initial amplitude of the oscillation.",
    )

    def __init__(self, **kwargs: Any) -> None:
        """Build the step effector.

        Args:
            **kwargs: Effector parameters, forwarded to the parent class.
        """
        super().__init__(**kwargs)

    @property
    def Act_coef(self) -> float:
        """The amplitude scaling of the activation."""
        return self.amp

    @property
    def Act_Phi(self) -> float:
        """The phase-dependent factor. Constant for a non-oscillating step."""
        return 1

    @property
    def Act(self) -> float:
        """The activation: the amplitude scaled by the phase factor."""
        return self.Act_coef * self.Act_Phi

    def set_amp(self, v: float) -> None:
        """Set the activation amplitude.

        Args:
            v: The new amplitude.
        """
        self.amp = v

    def get_amp(self, t: float) -> float:
        """Return the activation amplitude.

        Args:
            t: The current time. Accepted for signature compatibility.

        Returns:
            The amplitude.
        """
        return self.amp

    def act(self) -> None:
        """Emit the activation."""
        self.output = self.Act

    def inact(self) -> None:
        """Emit nothing while inactive."""
        self.output = 0


class StepOscillator(Oscillator, StepEffector):
    """
    Step oscillator combining oscillation with step-based activation.

    Merges Oscillator phase tracking with StepEffector amplitude control.
    Base class for oscillatory behaviors (peristaltic crawling, sinusoidal turning).

    Example:
        >>> step_osc = StepOscillator(freq=1.5, amp=0.7)
        >>> output = step_osc.step()
    """

    def act(self) -> None:
        """Advance the phase, then emit the activation."""
        self.oscillate()
        self.output = self.Act


class SinOscillator(StepOscillator):
    """
    Sinusoidal oscillator with sine-wave phase modulation.

    Extends StepOscillator with sinusoidal activation (Act_Phi = sin(φ)).
    Used for smooth oscillatory behaviors like sinusoidal turning.

    Example:
        >>> sin_osc = SinOscillator(freq=0.58, amp=0.4)
        >>> output = sin_osc.step()
    """

    @property
    def Act_Phi(self) -> float:
        """The phase-dependent factor: a sinusoid of the phase."""
        return np.sin(self.phi)


class NengoEffector(StepOscillator):
    """
    Nengo-compatible effector with frequency-based activation control.

    Extends StepOscillator with automatic frequency setting on start/stop.
    Used for Nengo neural simulator integration.

    Example:
        >>> nengo_eff = NengoEffector(freq=1.2, amp=0.5)
        >>> nengo_eff.start_effector()  # Sets freq to initial_freq
        >>> nengo_eff.stop_effector()   # Sets freq to 0
    """

    def start_effector(self) -> None:
        """Activate the effector and restore its resting frequency."""
        super().start_effector()
        self.set_freq(self.initial_freq)

    def stop_effector(self) -> None:
        """Deactivate the effector by driving its frequency to zero."""
        super().stop_effector()
        self.set_freq(0)
