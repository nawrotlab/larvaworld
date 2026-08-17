from __future__ import annotations

from .environment_mapping import env_params_to_canvas_state
from .environment_models import (
    CanvasArena,
    CanvasRingOverlay,
    CanvasObject,
    CanvasObjectType,
    EnvironmentCanvasState,
    LarvaPreviewFrame,
)
from .placement_controller import (
    HitCandidate,
    ObjectTable,
    SelectionSync,
    TapDispatcher,
    pick_nearest,
)

__all__ = [
    "CanvasArena",
    "CanvasRingOverlay",
    "CanvasObject",
    "CanvasObjectType",
    "EnvironmentCanvasState",
    "LarvaPreviewFrame",
    "env_params_to_canvas_state",
    "HitCandidate",
    "ObjectTable",
    "SelectionSync",
    "TapDispatcher",
    "pick_nearest",
]


def __getattr__(name: str):
    if name == "EnvironmentCanvas":
        from .environment_canvas import EnvironmentCanvas

        return EnvironmentCanvas
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
