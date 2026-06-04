"""Dataclasses and errors for the portal Module Inspector (standalone crawler/turner)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

__all__ = [
    "ModuleInspectorError",
    "ModuleTraceResult",
    "ModuleVariantSpec",
]


@dataclass(frozen=True)
class ModuleVariantSpec:
    """One inspectable module variant (module id + mode)."""

    module_id: str  # "crawler" | "turner"
    mode: str  # e.g. "realistic"
    display_name: str  # e.g. "Crawler / realistic"
    available_signals: tuple[
        str, ...
    ]  # subset of ("input","activation","phi","output")


@dataclass(frozen=True)
class ModuleTraceResult:
    """Time series from stepping a standalone module with constant A_in."""

    module_id: str
    mode: str
    steps: int
    dt: float
    a_in: float
    signals: tuple[str, ...]
    dataframe: pd.DataFrame  # columns: "time" + signals
    input_range: tuple[float, float]


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
