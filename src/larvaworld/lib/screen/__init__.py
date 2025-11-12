"""
Rendering and visualization on a pygame display
"""

from __future__ import annotations

from typing import Any

from importlib import import_module

__all__: list[str] = [
    "drawing",
    "rendering",
    "side_panel",
    # Commonly used symbols expected at package level (legacy surface)
    "ScreenOps",
    "ScreenManager",
    "GA_ScreenManager",
    "MediaDrawOps",
    "AgentDrawOps",
    "ColorDrawOps",
    # Additional rendering symbols
    "IDBox",
    "ScreenTextBox",
]

_NAME_TO_MODULE = {
    "drawing": "larvaworld.lib.screen.drawing",
    "rendering": "larvaworld.lib.screen.rendering",
    "side_panel": "larvaworld.lib.screen.side_panel",
}

# Map expected public symbols to their defining modules (lazy resolution)
_SYMBOL_TO_MODULE = {
    "ScreenOps": "larvaworld.lib.screen.drawing",
    "ScreenManager": "larvaworld.lib.screen.drawing",
    "GA_ScreenManager": "larvaworld.lib.screen.drawing",
    "MediaDrawOps": "larvaworld.lib.screen.drawing",
    "AgentDrawOps": "larvaworld.lib.screen.drawing",
    "ColorDrawOps": "larvaworld.lib.screen.drawing",
    "IDBox": "larvaworld.lib.screen.rendering",
    "ScreenTextBox": "larvaworld.lib.screen.rendering",
}


def __getattr__(name: str) -> Any:
    # First treat submodule names
    module_path = _NAME_TO_MODULE.get(name)
    if module_path is not None:
        mod = import_module(module_path)
        globals()[name] = mod
        return mod
    # Then treat symbol names exposed at package level
    module_path = _SYMBOL_TO_MODULE.get(name)
    if module_path is not None:
        mod = import_module(module_path)
        obj = getattr(mod, name)
        globals()[name] = obj
        return obj
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)


__displayname__ = "Visualization"
