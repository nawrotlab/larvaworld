from __future__ import annotations
from typing import Any
import os
import warnings

# Deprecation: discourage deep imports from internal module paths
if os.getenv("LARVAWORLD_STRICT_DEPRECATIONS") == "1":
    raise ImportError(
        "Deep import path deprecated. Use public API: 'from larvaworld.lib.model.modules import Feeder'"
    )
else:
    warnings.warn(
        "Deep import path deprecated. Use public API: 'from larvaworld.lib.model.modules import Feeder'",
        DeprecationWarning,
        stacklevel=2,
    )
import param

from ...param import PositiveNumber
from .oscillator import Oscillator

__all__: list[str] = [
    "Feeder",
]


class Feeder(Oscillator):
    """
    Feeder module for feeding behavior and head-sweeping motions.

    Implements oscillatory feeding behavior with configurable frequency,
    feeding radius, and bite volume. Controls mouth-hook movements
    during food consumption periods.

    Attributes:
        freq: Feeding oscillation frequency in Hz (1.0-3.0)
        feed_radius: Accessible feeding radius (fraction of body length)
        V_bite: Volume consumed per feeding motion (fraction of body volume)

    Example:
        >>> feeder = Feeder(freq=2.0, feed_radius=0.05, V_bite=0.001)
        >>> feeder.start_effector()
        >>> feeder.step()
    """

    freq = PositiveNumber(2.0, bounds=(1.0, 3.0))
    feed_radius = param.Magnitude(
        0.05,
        step=0.001,
        label="feeding radius",
        doc="The accessible radius for a feeding motion as fraction of body length.",
    )
    V_bite = param.Magnitude(
        0.001,
        step=0.0001,
        label="mouthook capacity",
        doc="The volume of a feeding motion as fraction of body volume.",
    )

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.stop_effector()

    def step(self) -> None:
        self.complete_iteration = False
        if self.active:
            self.oscillate()
        # return self.complete_iteration

    def suppresion_relief(self, phi_range: tuple[float, float]) -> bool:
        return self.phi_in_range(phi_range)
