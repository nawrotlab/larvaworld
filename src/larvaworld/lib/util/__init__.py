"""
Collection of standalone methods, custom classes and other tools, all of them
independent of the larvaworld registry.

This file used to eagerly star-import many submodules, which made importing
`larvaworld` heavier and obscured the public surface. We now lazily import the
same symbols on first access to preserve backwards compatibility without the
eager cost.
"""

from __future__ import annotations

from typing import Any

__displayname__ = "Auxilliary methods"

_LOADED = False


def _load_all() -> None:
    """Populate this module's namespace with legacy star-imports on demand."""
    global _LOADED, nam
    if _LOADED:
        return

    from importlib import import_module

    def _export_all_from(mod):
        names = getattr(mod, "__all__", None)
        if names is None:
            names = [n for n in dir(mod) if not n.startswith("_")]
        for n in names:
            globals()[n] = getattr(mod, n)

    for sub in (
        "ang",
        "fitting",
        "color",
        "dictsNlists",
        "nan_interpolation",
        "stdout",
        "shapely_aux",
        "xy",
    ):
        _export_all_from(import_module(f"{__name__}.{sub}"))

    # Explicit single-symbols
    globals()["combine_pdfs"] = import_module(f"{__name__}.combining").combine_pdfs
    NamingRegistry = import_module(f"{__name__}.naming").NamingRegistry
    globals()["NamingRegistry"] = NamingRegistry
    nam = NamingRegistry()

    _LOADED = True


def __getattr__(name: str) -> Any:
    # Defer legacy star-imports until the first attribute access
    if not _LOADED:
        _load_all()
    if name in globals():
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
