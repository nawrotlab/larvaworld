from __future__ import annotations
from typing import Any, Tuple
import os
import warnings

# Deprecation: discourage deep imports from internal module paths
if os.getenv("LARVAWORLD_STRICT_DEPRECATIONS") == "1":
    raise ImportError(
        "Deep import path deprecated. Use public API from 'larvaworld.lib.model.modules'"
    )
else:
    warnings.warn(
        "Deep import path deprecated. Use public API from 'larvaworld.lib.model.modules'",
        DeprecationWarning,
        stacklevel=2,
    )
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
        super().__init__(**kwargs)
        self.cur_attenuation = 1

    def apply_attenuation(self, cur_att: float) -> tuple[float, float]:
        if self.suppression_mode == "oscillation":
            return cur_att, 1
        elif self.suppression_mode == "amplitude":
            return 1, cur_att
        elif self.suppression_mode == "both":
            return cur_att, cur_att
        else:
            raise

    def check_module(self, obj: Any, module: str) -> None:
        self.cur_attenuation = self.attenuation


class DefaultCoupling(Coupling):
    pass


class SquareCoupling(Coupling):
    crawler_phi_range = PhaseRange(
        label="crawler suppression relief phase interval",
        doc="CRAWLER phase range for TURNER suppression lift.",
    )
    feeder_phi_range = PhaseRange(
        label="feeder suppression relief phase interval",
        doc="FEEDER phase range for TURNER suppression lift.",
    )

    def check_module(self, obj: Any, module: str) -> None:
        phi_dic = {
            "Crawler": self.crawler_phi_range,
            "Feeder": self.feeder_phi_range,
        }
        A = self.attenuation
        if hasattr(obj, "phi") and obj.suppresion_relief(phi_dic[module]):
            A += self.attenuation_max
        self.cur_attenuation = A


class PhasicCoupling(Coupling):
    max_attenuation_phase = Phase(
        3.4,
        label="max relief phase",
        doc="CRAWLER phase of minimum TURNER suppression.",
    )

    def get(self, x: float) -> float:
        def gaussian(x, mu, sig):
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
        x = obj.phi if hasattr(obj, "phi") else 0
        self.cur_attenuation = self.get(x)
