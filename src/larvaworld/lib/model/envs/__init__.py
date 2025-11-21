"""
Environment classes for the agent-based-modeling simulations,
including the arena and any objects and impassable obstacles located within it,
as well as any existing sensory landscapes.
"""

from __future__ import annotations

from typing import Any

__displayname__ = "Environment"

__all__: list[str] = ["Arena"]

_NAME_TO_MODULE: dict[str, str] = {"Arena": "larvaworld.lib.model.envs.arena"}


def __getattr__(name: str) -> Any:
    module_path = _NAME_TO_MODULE.get(name)
    if module_path is None:
        # Defer legacy star-imports on first unknown attribute access
        from importlib import import_module

        for m in (
            "larvaworld.lib.model.envs.obstacle",
            "larvaworld.lib.model.envs.valuegrid",
        ):
            mod = import_module(m)
            # export public names
            names = getattr(mod, "__all__", None)
            if names is None:
                names = [n for n in dir(mod) if not n.startswith("_")]
            for n in names:
                globals().setdefault(n, getattr(mod, n))
        if name in globals():
            return globals()[name]
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    mod = import_module(module_path)
    obj = getattr(mod, name)
    globals()[name] = obj
    return obj


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
