"""Shared config helpers for the models_architecture inspector data layers.

These helpers are reused by both the Model Inspector and the Module Inspector
data layers to avoid duplicating canonical defaults and copy semantics.
"""

from __future__ import annotations

import copy
from typing import Any

__all__ = [
    "WINDSENSOR_DEFAULT_WEIGHTS",
    "copy_config_value",
]

# Canonical zero-valued wind-sensor response weights (keys used by the nengo
# wind pathway). Required to construct a standalone ``Windsensor`` instance.
WINDSENSOR_DEFAULT_WEIGHTS: dict[str, float] = {
    "hunch_lin": 0.0,
    "hunch_ang": 0.0,
    "bend_lin": 0.0,
    "bend_ang": 0.0,
}


def copy_config_value(value: Any) -> Any:
    """Return a copy of a config value, honoring ``AttrDict.get_copy`` if present."""
    if hasattr(value, "get_copy"):
        return value.get_copy()
    return copy.deepcopy(value)
