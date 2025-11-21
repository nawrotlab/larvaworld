"""
All classes supporting objects, agents and environment of the agent-based-modeling
simulations, as well as the modules comprising the layered behavioral architecture
modeling the nervous system, body and metabolism.
"""

from __future__ import annotations

from typing import Any

__displayname__ = "Modeling"

# Provide a lazy facade for common classes historically re-exported from here.
__all__: list[str] = [
    # Subpackages (lazy loaded)
    "deb",
    "modules",
    "agents",
    "envs",
    "object",
    # object.py classes
    "Object",
    "GroupedObject",
    # envs.valuegrid
    "AnalyticalValueLayer",
    "GaussianValueLayer",
    "DiffusionValueLayer",
    "OdorScape",
    "FoodGrid",
    "WindScape",
    "ThermoScape",
    # envs.obstacle
    "Border",
    # agents._source
    "Source",
    "Food",
    # modules
    "moduleDB",
    # agents (newly added)
    "Larva",
    "BaseController",
    "LarvaSim",
    # envs (newly added)
    "Arena",
    "Maze",
    "Box",
    "Wall",
    # modules (newly added)
    "Brain",
    "DefaultBrain",
    "Locomotor",
    "SpaceDict",
    "Effector",
    "Crawler",
    "PhaseOscillator",
    "NeuralOscillator",
    "ConstantTurner",
    "SinTurner",
    "Timer",
    "LightSource",
    "NengoBrain",
    # deb (newly added)
    "DEB",
]

_NAME_TO_MODULE: dict[str, str] = {
    # object.py classes
    "Object": "larvaworld.lib.model.object",
    "GroupedObject": "larvaworld.lib.model.object",
    # envs.valuegrid
    "AnalyticalValueLayer": "larvaworld.lib.model.envs.valuegrid",
    "GaussianValueLayer": "larvaworld.lib.model.envs.valuegrid",
    "DiffusionValueLayer": "larvaworld.lib.model.envs.valuegrid",
    "OdorScape": "larvaworld.lib.model.envs.valuegrid",
    "FoodGrid": "larvaworld.lib.model.envs.valuegrid",
    "WindScape": "larvaworld.lib.model.envs.valuegrid",
    "ThermoScape": "larvaworld.lib.model.envs.valuegrid",
    # envs.obstacle
    "Border": "larvaworld.lib.model.envs.obstacle",
    # agents._source
    "Source": "larvaworld.lib.model.agents._source",
    "Food": "larvaworld.lib.model.agents._source",
    # modules
    "moduleDB": "larvaworld.lib.model.modules.module_modes",
    "Larva": "larvaworld.lib.model.agents.larva_robot",
    "BaseController": "larvaworld.lib.model.agents._larva_sim",
    "LarvaSim": "larvaworld.lib.model.agents._larva_sim",
    "Arena": "larvaworld.lib.model.envs.arena",
    "Maze": "larvaworld.lib.model.envs.maze",
    "Box": "larvaworld.lib.model.envs.obstacle",
    "Wall": "larvaworld.lib.model.envs.obstacle",
    "Brain": "larvaworld.lib.model.modules.brain",
    "DefaultBrain": "larvaworld.lib.model.modules.brain",
    "Locomotor": "larvaworld.lib.model.modules.locomotor",
    "SpaceDict": "larvaworld.lib.model.modules.module_modes",
    # modules.basic / oscillator provide these bases
    "Effector": "larvaworld.lib.model.modules.basic",
    "Crawler": "larvaworld.lib.model.modules.crawler",
    "PhaseOscillator": "larvaworld.lib.model.modules.crawler",
    "NeuralOscillator": "larvaworld.lib.model.modules.turner",
    "ConstantTurner": "larvaworld.lib.model.modules.turner",
    "SinTurner": "larvaworld.lib.model.modules.turner",
    "Timer": "larvaworld.lib.model.modules.oscillator",
    "LightSource": "larvaworld.lib.model.modules.rot_surface",
    "NengoBrain": "larvaworld.lib.model.modules.nengobrain",
    "DEB": "larvaworld.lib.model.deb.deb",
}

_SUBPACKAGES: dict[str, str] = {
    "deb": "larvaworld.lib.model.deb",
    "modules": "larvaworld.lib.model.modules",
    "agents": "larvaworld.lib.model.agents",
    "envs": "larvaworld.lib.model.envs",
    "object": "larvaworld.lib.model.object",
}


def __getattr__(name: str) -> Any:
    from importlib import import_module

    # Check if it's a subpackage
    if name in _SUBPACKAGES:
        module_path = _SUBPACKAGES[name]
        mod = import_module(module_path)
        globals()[name] = mod
        return mod

    # Check if it's a class/symbol
    module_path = _NAME_TO_MODULE.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    mod = import_module(module_path)
    obj = getattr(mod, name)
    globals()[name] = obj
    return obj


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
