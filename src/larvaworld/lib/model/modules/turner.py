"""
Turner modules generating lateral bending.

Provides the turner interface and its alternative implementations, from a
constant or sinusoidal torque to a neural oscillator whose intrinsic dynamics
produce the alternating head casts.
"""

from __future__ import annotations
from typing import Any

import random

import param

from ...param import PositiveInteger, PositiveNumber
from .basic import Effector, SinOscillator, StepEffector

__all__: list[str] = [
    "Turner",
    "ConstantTurner",
    "SinTurner",
    "NeuralOscillator",
]


class Turner(Effector):
    """
    Base turner module for directional control.

    Abstract base class for body-bending modules that control angular
    velocity and directional changes during locomotion.

    Attributes:
        input_range: Valid input range for sensory activation (-1 to 1)

    Example:
        >>> # Use concrete subclasses like ConstantTurner, SinTurner, NeuralOscillator
        >>> turner = SinTurner(freq=0.58, amp=0.5)
    """

    input_range = param.Range(
        (-1, 1),
        bounds=(-1, 1),
        step=0.01,
        precedence=-2,
        label="input range",
        doc="The input range of the oscillator.",
        readonly=True,
    )


class ConstantTurner(Turner, StepEffector):
    """
    Constant-output turner for simple directional control.

    Provides step-based turning behavior with constant angular velocity
    when active. Simplest turner implementation for baseline locomotion.

    Example:
        >>> turner = ConstantTurner(amp=0.3)
        >>> angular_velocity = turner.step()
    """

    pass


class SinTurner(Turner, SinOscillator):
    """
    Sinusoidal turner for oscillatory directional control.

    Extends Turner with sinusoidal oscillation, producing periodic
    left-right turning patterns. Commonly used for exploratory behavior.

    Attributes:
        freq: Oscillation frequency in Hz (0.0-2.0)

    Example:
        >>> turner = SinTurner(freq=0.58, amp=0.4)
        >>> angular_velocity = turner.step()
    """

    freq = PositiveNumber(0.58, bounds=(0.0, 2.0))


class NeuralOscillator(Turner):
    """
    Neural oscillator turner using Wilson-Cowan dynamics.

    Implements biologically realistic central pattern generator (CPG)
    for directional control using coupled excitatory/inhibitory neural
    populations. Produces emergent oscillatory turning behavior from
    neural dynamics rather than prescribed waveforms.

    Attributes:
        base_activation: Baseline neural activation level (10-40 Hz)
        activation_range: Valid activation range bounds (0-100 Hz)
        tau: Neural time constant for population dynamics (seconds)
        w_ee, w_ce, w_ec, w_cc: Synaptic connection weights between populations
        m: Maximum neural spike rate (Hz)
        n: Spike-rate response steepness coefficient
        E_l, E_r: Left/right excitatory population activities
        C_l, C_r: Left/right inhibitory population activities
        H_E_l, H_E_r, H_C_l, H_C_r: Hysteresis variables for populations

    Example:
        >>> neural_turner = NeuralOscillator(
        ...     base_activation=20.0,
        ...     tau=0.1,
        ...     w_ee=3.0,
        ...     w_ec=4.0
        ... )
        >>> angular_velocity = neural_turner.step(A_in=0.5)
    """

    base_activation = PositiveNumber(
        20.0,
        bounds=(10.0, 40.0),
        step=1.0,
        precedence=1,
        label="baseline activation",
        doc="The baseline activation of the oscillator.",
    )
    activation_range = param.Range(
        (10.0, 40.0),
        bounds=(0.0, 100.0),
        step=1.0,
        precedence=1,
        label="activation range",
        doc="The activation range of the oscillator.",
    )
    # input_range = param.Range((-1, 1), bounds=(-1, 1), precedence=-2, label='input range',
    #                           doc='The input range of the oscillator.', readonly=True)
    tau = PositiveNumber(
        0.1,
        step=0.01,
        precedence=2,
        label="time constant",
        doc="The time constant of the oscillator.",
    )
    w_ee = PositiveNumber(
        3.0, step=0.01, label="E->E weigths", doc="The E->E synapse connection weights."
    )
    w_ce = PositiveNumber(
        0.1, step=0.01, label="C->E weigths", doc="The C->E synapse connection weights."
    )
    w_ec = PositiveNumber(
        4.0, step=0.01, label="E->C weigths", doc="The E->C synapse connection weights."
    )
    w_cc = PositiveNumber(
        4.0, step=0.01, label="C->C weigths", doc="The C->C synapse connection weights."
    )
    m = PositiveInteger(
        100,
        softmax=1000,
        label="maximum spike-rate",
        doc="The maximum allowed spike rate.",
    )
    n = PositiveNumber(
        2.0,
        softmax=10.0,
        step=0.1,
        label="spike response steepness",
        doc="The neuron spike-rate response steepness coefficient.",
    )

    def __init__(self, **kwargs: Any) -> None:
        """Build the oscillator and run it to its limit cycle.

        Args:
            **kwargs: Turner parameters, forwarded to the parent class.
        """
        super().__init__(**kwargs)
        self.param.base_activation.bounds = self.activation_range
        self.r1 = self.activation_range[1] - self.base_activation
        self.r0 = self.base_activation - self.activation_range[0]
        self.activation = self.base_activation

        # Neural populations
        self.E_r = 0  # 28
        self.H_E_r = 0  # 10

        self.E_l = 0  # 30
        self.H_E_l = 0  # 10

        self.C_r = 0
        self.H_C_r = 0  # 10

        self.C_l = 0
        self.H_C_l = 0  # 10

        self.scaled_tau = self.dt / self.tau
        self.warm_up()

    def warm_up(self) -> None:
        """Run the network unforced so it settles onto its limit cycle.

        Without this the first simulated timesteps would reflect the arbitrary
        initial conditions rather than the oscillator's own dynamics.
        """
        for i in range(1000):
            if random.uniform(0, 1) < 0.5:
                self.step()

    def update(self) -> None:
        """Convert the incoming activation into the network's drive.

        Negative and positive inputs are scaled by separate coefficients, so
        the network can respond asymmetrically to opposing stimuli.
        """
        if self.input < 0:
            a = self.r0 * self.input
        elif self.input >= 0:
            a = self.r1 * self.input
        self.activation = self.base_activation + a

    def act(self) -> None:
        """Advance the network and emit the left-right activity difference."""
        self.oscillate()
        self.output = self.E_r - self.E_l

    def inact(self) -> None:
        """Emit no torque while inactive."""
        self.output = 0

    def oscillate(self) -> None:
        """Integrate the coupled excitatory-inhibitory network one timestep.

        Two mutually inhibiting excitatory-inhibitory pairs, one per body side,
        each with its own slow adaptation variable. The adaptation makes the
        active side fatigue and yield to the other, producing the alternating
        head casts. The drive sets both the integration gain and the adaptation
        time constant.
        """
        A = self.activation
        t = self.scaled_tau
        tau_h = 3 / (1 + (0.04 * A) ** 2)
        t_h = self.dt / tau_h
        g = 6 + (0.09 * A) ** 2

        self.E_l += t * (
            -self.E_l
            + self.compute_R(
                A + self.w_ee * self.E_l - self.w_ec * self.C_r, 64 + g * self.H_E_l
            )
        )
        self.E_r += t * (
            -self.E_r
            + self.compute_R(
                A + self.w_ee * self.E_r - self.w_ec * self.C_l, 64 + g * self.H_E_r
            )
        )
        self.H_E_l += t_h * (-self.H_E_l + self.E_l)
        self.H_E_r += t_h * (-self.H_E_r + self.E_r)

        self.C_l += t * (
            -self.C_l
            + self.compute_R(
                A + self.w_ce * self.E_l - self.w_cc * self.C_r, 64 + g * self.H_C_l
            )
        )
        self.C_r += t * (
            -self.C_r
            + self.compute_R(
                A + self.w_ce * self.E_r - self.w_cc * self.C_l, 64 + g * self.H_C_r
            )
        )
        self.H_C_l += t_h * (-self.H_C_l + self.E_l)
        self.H_C_r += t_h * (-self.H_C_r + self.E_r)

    def compute_R(self, x: float, h: float) -> float:
        """Apply the network's saturating activation function.

        Args:
            x: The net input to a population.
            h: The half-activation threshold, raised by adaptation.

        Returns:
            The firing rate, zero for non-positive input.
        """
        if x > 0:
            r = self.m * x**self.n / (x**self.n + h**self.n)
            return r
        else:
            return 0.0

    def get_state(self) -> list[float]:
        """Return the full network state.

        Returns:
            The activity and adaptation of both excitatory and both inhibitory
            populations.
        """
        state = [
            self.E_l,
            self.H_E_l,
            self.E_r,
            self.H_E_r,
            self.C_l,
            self.H_C_l,
            self.C_r,
            self.H_C_r,
        ]
        return state
