"""
Agent classes for the agent-based-modeling simulations.
"""

from __future__ import annotations

from typing import Any

__displayname__ = "Agents"

__all__: list[str] = [
    "NonSpatialAgent",
    "PointAgent",
    "OrientedAgent",
    "MobilePointAgent",
    "MobileAgent",
    "Source",
    "Food",
    "Larva",
    "LarvaContoured",
    "LarvaSegmented",
    "LarvaMotile",
    "LarvaReplay",
    "LarvaReplayContoured",
    "LarvaReplaySegmented",
    "BaseController",
    "LarvaSim",
    "LarvaRobot",
    "ObstacleLarvaRobot",
    "LarvaOffline",
]

_NAME_TO_MODULE: dict[str, str] = {
    # Agent bases
    "NonSpatialAgent": "larvaworld.lib.model.agents._agent",
    "PointAgent": "larvaworld.lib.model.agents._agent",
    "OrientedAgent": "larvaworld.lib.model.agents._agent",
    "MobilePointAgent": "larvaworld.lib.model.agents._agent",
    "MobileAgent": "larvaworld.lib.model.agents._agent",
    # Sources
    "Source": "larvaworld.lib.model.agents._source",
    "Food": "larvaworld.lib.model.agents._source",
    # Larvae
    "Larva": "larvaworld.lib.model.agents._larva",
    "LarvaContoured": "larvaworld.lib.model.agents._larva",
    "LarvaSegmented": "larvaworld.lib.model.agents._larva",
    "LarvaMotile": "larvaworld.lib.model.agents._larva",
    # Replay & sim
    "LarvaReplay": "larvaworld.lib.model.agents._larva_replay",
    "LarvaReplayContoured": "larvaworld.lib.model.agents._larva_replay",
    "LarvaReplaySegmented": "larvaworld.lib.model.agents._larva_replay",
    "BaseController": "larvaworld.lib.model.agents._larva_sim",
    "LarvaSim": "larvaworld.lib.model.agents._larva_sim",
    # Robots
    "LarvaRobot": "larvaworld.lib.model.agents.larva_robot",
    "ObstacleLarvaRobot": "larvaworld.lib.model.agents.larva_robot",
    # Offline
    "LarvaOffline": "larvaworld.lib.model.agents.larva_offline",
}


def __getattr__(name: str) -> Any:
    module_path = _NAME_TO_MODULE.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    mod = import_module(module_path)
    obj = getattr(mod, name)
    globals()[name] = obj
    return obj


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
