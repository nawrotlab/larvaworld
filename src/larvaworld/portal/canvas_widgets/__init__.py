"""
Interactive canvas widgets for the portal.

Exposes the arena canvas, the objects placed on it and the state model that
keeps the two in step.
"""

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
    """Resolve a public name by importing its module on first access.

    Args:
        name: The attribute being accessed.

    Returns:
        The resolved object.

    Raises:
        AttributeError: If no submodule exports that name.
    """
    if name == "EnvironmentCanvas":
        from .environment_canvas import EnvironmentCanvas

        return EnvironmentCanvas
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
