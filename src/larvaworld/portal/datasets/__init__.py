from __future__ import annotations

from importlib import import_module
from typing import Any

__all__: list[str] = [
    "ImportRequest",
    "RawDatasetCandidate",
    "WorkspaceDatasetRecord",
    "build_workspace_proc_folder",
    "discover_raw_datasets",
    "get_workspace_dataset",
    "import_into_workspace",
    "list_workspace_datasets",
]


def __getattr__(name: str) -> Any:
    if name in {"ImportRequest", "WorkspaceDatasetRecord"}:
        module = import_module("larvaworld.portal.datasets.models")
        return getattr(module, name)
    if name in {"RawDatasetCandidate", "discover_raw_datasets"}:
        module = import_module("larvaworld.portal.datasets.discovery")
        return getattr(module, name)
    if name in {"build_workspace_proc_folder", "import_into_workspace"}:
        module = import_module("larvaworld.portal.datasets.import_adapter")
        return getattr(module, name)
    if name in {"get_workspace_dataset", "list_workspace_datasets"}:
        module = import_module("larvaworld.portal.datasets.workspace_index")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
