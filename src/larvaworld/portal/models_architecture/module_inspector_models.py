"""Dataclasses and errors for the portal Module Inspector.

Covers three module "kinds":
- "effector": crawler/turner, driven by a constant scalar ``A_in``
- "feeder":   self-oscillator, no external input
- "sensor":   driven by a time-varying stimulus converted to a dict input
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

__all__ = [
    "ModuleInspectorError",
    "ModuleTraceResult",
    "ModuleVariantSpec",
    "StimulusSpec",
]


@dataclass(frozen=True)
class StimulusSpec:
    """Time-varying stimulus configuration for sensor modules."""

    waveform: str  # "step" | "sinusoid"
    baseline: float
    amplitude: float
    frequency: float  # Hz, used by "sinusoid"
    onset: float  # seconds, used by "step"


@dataclass(frozen=True)
class ModuleVariantSpec:
    """One inspectable module variant (module id + mode)."""

    module_id: str
    mode: str
    kind: str  # "effector" | "feeder" | "sensor"
    display_name: str
    available_signals: tuple[str, ...]


@dataclass(frozen=True)
class ModuleTraceResult:
    """Time series produced by stepping a standalone module."""

    module_id: str
    mode: str
    kind: str
    steps: int
    dt: float
    a_in: float
    signals: tuple[str, ...]
    dataframe: pd.DataFrame  # columns: "time" + signals
    input_range: tuple[float, float]
    stimulus: StimulusSpec | None = None  # set only for kind == "sensor"


class ModuleInspectorError(RuntimeError):
    """Raised when module inspection or trace sampling cannot proceed."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.context = context or {}
