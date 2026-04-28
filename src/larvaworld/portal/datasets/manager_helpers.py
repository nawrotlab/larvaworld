from __future__ import annotations

import shutil
from pathlib import Path

from larvaworld.portal.datasets.models import WorkspaceDatasetRecord
from larvaworld.portal.workspace import WorkspaceState, get_workspace_dir


def imported_workspace_root(workspace: WorkspaceState) -> Path:
    return (get_workspace_dir("datasets", workspace=workspace) / "imported").resolve()


def format_relative_imported_location(
    record: WorkspaceDatasetRecord, workspace: WorkspaceState
) -> str:
    dataset_dir = record.dataset_dir.expanduser().resolve()
    datasets_root = get_workspace_dir("datasets", workspace=workspace).resolve()
    try:
        return dataset_dir.relative_to(datasets_root).as_posix()
    except ValueError:
        return dataset_dir.name


def delete_imported_workspace_dataset(
    record: WorkspaceDatasetRecord, workspace: WorkspaceState
) -> None:
    dataset_dir = record.dataset_dir.expanduser().resolve()
    imported_root = imported_workspace_root(workspace)
    try:
        dataset_dir.relative_to(imported_root)
    except ValueError as exc:
        raise RuntimeError(
            "Delete failed: dataset path resolved outside the active workspace imported root"
        ) from exc
    if dataset_dir == imported_root:
        raise RuntimeError("Delete failed: refusing to delete the imported root")
    if not dataset_dir.is_dir():
        raise RuntimeError("Delete failed: dataset directory no longer exists")
    shutil.rmtree(dataset_dir)
