"""
Custom parameters and parameterized classes extending those existing in the
`param` package.

Previous versions eagerly re-exported everything with star-imports. We now lazy
load these to avoid heavy import chains and potential cycles while preserving
the public API surface.
"""

from __future__ import annotations
from typing import Any

__displayname__ = "Parameters"

_LOADED = False


def _load_all() -> None:
    global _LOADED
    if _LOADED:
        return
    from importlib import import_module

    def _export_all_from(mod: Any) -> None:
        names = getattr(mod, "__all__", None)
        if names is None:
            names = [n for n in dir(mod) if not n.startswith("_")]
        for n in names:
            globals()[n] = getattr(mod, n)

    for sub in (
        "custom",
        "xy_distro",
        "nested_parameter_group",
        "enrichment",
        "spatial",
        "drawable",
        "body_shape",
        "grouped",
        "composition",
    ):
        _export_all_from(import_module(f"{__name__}.{sub}"))
    _LOADED = True


def __getattr__(name: str) -> Any:
    if not _LOADED:
        _load_all()
    if name in globals():
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    if not _LOADED:
        _load_all()
    return sorted(list(globals().keys()))
