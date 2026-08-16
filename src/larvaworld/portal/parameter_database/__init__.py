from __future__ import annotations

from importlib import import_module
from typing import Any

__all__: list[str] = [
    "build_parameter_db_content",
    "build_param_detail_popup",
    "build_standalone_page",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        module = import_module("larvaworld.portal.parameter_database.parameter_db_app")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
