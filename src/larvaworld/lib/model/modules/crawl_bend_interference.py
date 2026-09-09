"""
Coupling between the crawling and turning rhythms.

Crawling suppresses turning within each stride cycle. These modules define how
strong that suppression is and how it varies with stride phase, from a constant
attenuation to an explicitly phase-dependent one.
"""

from __future__ import annotations
from typing import Any, Tuple

import numpy as np
import param

from ...param import Phase, PhaseRange

__all__: list[str] = [
    "Coupling",
    "DefaultCoupling",
    "SquareCoupling",
    "PhasicCoupling",
]


class Coupling(param.Parameterized):
    """
    Base class for the crawl-to-turn suppression.

    While the agent crawls, its angular motion is attenuated. Subclasses decide
    how the attenuation varies over the stride cycle; this class holds the
    attenuation coefficients and applies the resulting factor according to the
    configured suppression mode, which targets the turner's oscillation, its
    amplitude, or both.
    """

    attenuation = param.Magnitude(
        0.0,
        step=0.01,
        label="crawl-induced angular attenuation",
        doc="The attenuation coefficient for the crawl-interference to the angular motion.",
    )
    attenuation_max = param.Magnitude(
        1.0,
        step=0.01,
        label="crawl-induced maximum angular attenuation",
        doc="The suppression relief coefficient for the crawl-interference to the angular motion.",
    )
    suppression_mode = param.Selector(
        objects=["amplitude", "oscillation", "both"],
        label="crawl-induced suppression mode",
        doc="The suppression mode for the crawl-interference to the angular motion.",
    )

    def __init__(self, dt: float = 0.1, **kwargs: Any) -> None:
        """Build the coupling with suppression initially lifted.

        Args:
            dt: The simulation timestep in seconds.
            **kwargs: Coupling parameters, forwarded to the parent class.
        """
        super().__init__(**kwargs)
        self.cur_attenuation = 1

    def apply_attenuation(self, cur_att: float) -> tuple[float, float]:
        """Split the attenuation across the turner's input and output.

        Args:
            cur_att: The attenuation factor for this timestep.

        Returns:
            The factors applied to the turner's input and to its output.
            ``"oscillation"`` attenuates only the input, ``"amplitude"`` only
            the output, and ``"both"`` attenuates each.

        Raises:
            RuntimeError: If the suppression mode is unrecognized.
        """
        if self.suppression_mode == "oscillation":
            return cur_att, 1
        elif self.suppression_mode == "amplitude":
            return 1, cur_att
        elif self.suppression_mode == "both":
            return cur_att, cur_att
        else:
            raise

    def check_module(self, obj: Any, module: str) -> None:
        """Set this timestep's attenuation from a driving module.

        The base coupling attenuates constantly, ignoring the module's phase.

        Args:
            obj: The driving module, typically the crawler or feeder.
            module: Its name.
        """
        self.cur_attenuation = self.attenuation


class DefaultCoupling(Coupling):
    """
    Constant crawl-to-turn suppression.

    Applies the base attenuation throughout the stride cycle, with no
    phase-dependent relief.
    """

    pass


class SquareCoupling(Coupling):
    """
    Crawl-to-turn suppression relieved over a phase interval.

    Suppression is lifted by the maximum relief coefficient whenever the driving
    oscillator's phase falls inside the configured interval, giving a square
    relief profile over the stride cycle.
    """

    crawler_phi_range = PhaseRange(
        label="crawler suppression relief phase interval",
        doc="CRAWLER phase range for TURNER suppression lift.",
    )
    feeder_phi_range = PhaseRange(
        label="feeder suppression relief phase interval",
        doc="FEEDER phase range for TURNER suppression lift.",
    )

    def check_module(self, obj: Any, module: str) -> None:
        """Set this timestep's attenuation, relieved over a phase interval.

        Args:
            obj: The driving module, typically the crawler or feeder.
            module: Its name, selecting which relief interval applies.
        """
        phi_dic = {
            "Crawler": self.crawler_phi_range,
            "Feeder": self.feeder_phi_range,
        }
        A = self.attenuation
        if hasattr(obj, "phi") and obj.suppresion_relief(phi_dic[module]):
            A += self.attenuation_max
        self.cur_attenuation = A


class PhasicCoupling(Coupling):
    """
    Crawl-to-turn suppression varying smoothly with stride phase.

    Relief follows a Gaussian centred on the phase of minimum suppression, so
    that the attenuation eases in and out over the stride cycle rather than
    switching abruptly. The result is clipped to the unit range.
    """

    max_attenuation_phase = Phase(
        3.4,
        label="max relief phase",
        doc="CRAWLER phase of minimum TURNER suppression.",
    )

    def get(self, x: float) -> float:
        """Evaluate the attenuation at a given phase.

        Args:
            x: The driving module's phase in radians.

        Returns:
            The attenuation factor, clipped to the unit range.
        """

        def gaussian(x, mu, sig):
            """Evaluate an unnormalized Gaussian."""
            return np.exp(-np.power(x - mu, 2.0) / (2 * np.power(sig, 2.0)))

        # A = gaussian(x, self.max_attenuation_phase, 1) * self.attenuation_max + self.attenuation
        A = (
            np.exp(-np.power(x - self.max_attenuation_phase, 2.0) / 2)
            * self.attenuation_max
            + self.attenuation
        )
        if A >= 1:
            A = 1
        elif A <= 0:
            A = 0
        return A

    def check_module(self, obj: Any, module: str) -> None:
        """Set this timestep's attenuation from the driving module's phase.

        Args:
            obj: The driving module, typically the crawler or feeder.
            module: Its name. Unused; the phase alone decides.
        """
        x = obj.phi if hasattr(obj, "phi") else 0
        self.cur_attenuation = self.get(x)
