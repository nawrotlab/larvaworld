"""
Launchers of the diverse available simulation modes
"""

from __future__ import annotations

from typing import Any

__displayname__ = "Simulation"

__all__: list[str] = [
    "ABModel",
    "BaseRun",
    "ReplayRun",
    "ExpRun",
    "Exec",
    "BatchRun",
    "OptimizationOps",
    "GAlauncher",
    "EvalRun",
    "sim_model",
]

_NAME_TO_MODULE = {
    "ABModel": "larvaworld.lib.sim.ABM_model",
    "BaseRun": "larvaworld.lib.sim.base_run",
    "ReplayRun": "larvaworld.lib.sim.dataset_replay",
    "ExpRun": "larvaworld.lib.sim.single_run",
    "Exec": "larvaworld.lib.sim.subprocess_run",
    "BatchRun": "larvaworld.lib.sim.batch_run",
    "OptimizationOps": "larvaworld.lib.sim.batch_run",
    "GAlauncher": "larvaworld.lib.sim.genetic_algorithm",
    "EvalRun": "larvaworld.lib.sim.model_evaluation",
    "sim_model": "larvaworld.lib.sim.agent_simulations",
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
