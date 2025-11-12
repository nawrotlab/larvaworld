"""
Modeling of the energetics/metabolism based on the Dynamic Energy Budget theory
"""

from __future__ import annotations

from typing import Any

__displayname__ = "Dynamic Energy Budget"

__all__: list[str] = [
    # expose primary public classes/functions explicitly if needed later
]

_LOADED: bool = False


def _load_all() -> None:
    global _LOADED
    if _LOADED:
        return
    from importlib import import_module

    def _export_all_from(mod):
        names = getattr(mod, "__all__", None)
        if names is None:
            names = [n for n in dir(mod) if not n.startswith("_")]
        for n in names:
            globals()[n] = getattr(mod, n)

    for sub in ("gut", "deb"):
        _export_all_from(import_module(f"{__name__}.{sub}"))
    _LOADED = True


def __getattr__(name: str) -> Any:
    if not _LOADED:
        _load_all()
    if name in globals():
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
