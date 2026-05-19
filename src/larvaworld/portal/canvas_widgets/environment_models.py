from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

CanvasObjectType = Literal[
    "source_unit",
    "source_group",
    "border_segment",
    "larva_group",
]


@dataclass(frozen=True)
class CanvasArena:
    geometry: str
    dims: tuple[float, float]
    torus: bool = False


@dataclass(frozen=True)
class CanvasObject:
    object_id: str
    object_type: CanvasObjectType
    x: float | None = None
    y: float | None = None
    x2: float | None = None
    y2: float | None = None
    radius: float | None = None
    width: float | None = None
    color: str | None = None
    amount: float | None = None
    odor_id: str | None = None
    odor_intensity: float | None = None
    odor_spread: float | None = None
    distribution_mode: str | None = None
    distribution_shape: str | None = None
    distribution_n: int | None = None
    distribution_scale_x: float | None = None
    distribution_scale_y: float | None = None
    distribution_show_shape: bool | None = None


@dataclass(frozen=True)
class EnvironmentCanvasState:
    arena: CanvasArena
    objects: tuple[CanvasObject, ...] = ()
    food_grid: dict[str, Any] | None = None
    odorscape: dict[str, Any] | None = None
    windscape: dict[str, Any] | None = None
    thermoscape: dict[str, Any] | None = None
    show_arena_outline: bool = True


@dataclass(frozen=True)
class LarvaPreviewFrame:
    tick: int
    centroids: tuple[tuple[float, float], ...] = ()
    heads: tuple[tuple[float, float], ...] = ()
    midlines: tuple[tuple[tuple[float, float], ...], ...] = ()
    trails: tuple[tuple[tuple[float, float], ...], ...] = ()
    colors: tuple[str, ...] = ()
    labels: tuple[str, ...] = ()


@dataclass(frozen=True)
class CanvasRingOverlay:
    x: float
    y: float
    radius: float
    color: str = "#2f4858"
    line_width: float = 3.0
    line_alpha: float = 0.95
    line_dash: str = "solid"
