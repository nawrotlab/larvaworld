from __future__ import annotations

from pathlib import Path
from typing import Any

from larvaworld.lib import reg
from larvaworld.portal.datasets.models import ImportRequest, WorkspaceDatasetRecord
from larvaworld.portal.datasets.workspace_index import get_workspace_dataset
from larvaworld.portal.workspace import (
    WorkspaceState,
    get_active_workspace,
    get_workspace_dir,
)


def _validated_text(value: str, *, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"Import request requires a non-empty {field_name}.")
    return normalized


def _validate_workspace(
    workspace: WorkspaceState | None,
) -> WorkspaceState:
    resolved = workspace or get_active_workspace()
    if resolved is None:
        raise RuntimeError("Import failed: no valid active workspace is configured")
    return resolved


def _validate_proc_root(proc_root: Path, datasets_root: Path) -> Path:
    try:
        proc_root.relative_to(datasets_root)
    except ValueError as exc:
        raise RuntimeError(
            "Import failed: workspace proc folder resolved outside the workspace datasets directory"
        ) from exc
    return proc_root


def build_workspace_proc_folder(workspace: WorkspaceState, lab_id: str) -> Path:
    normalized_lab_id = _validated_text(lab_id, field_name="lab_id")
    datasets_root = get_workspace_dir("datasets", workspace=workspace).resolve()
    proc_root = (datasets_root / "imported" / normalized_lab_id).resolve()
    return _validate_proc_root(proc_root, datasets_root)


def import_into_workspace(
    request: ImportRequest, workspace: WorkspaceState | None = None
) -> WorkspaceDatasetRecord:
    active_workspace = _validate_workspace(workspace)
    lab_id = _validated_text(request.lab_id, field_name="lab_id")
    parent_dir = _validated_text(request.parent_dir, field_name="parent_dir")
    proc_root = build_workspace_proc_folder(active_workspace, lab_id)

    lab = reg.conf.LabFormat.get(lab_id)
    import_kwargs: dict[str, Any] = {
        "parent_dir": parent_dir,
        "raw_folder": (
            str(request.raw_folder) if request.raw_folder is not None else None
        ),
        "merged": request.merged,
        "proc_folder": str(proc_root),
        "group_id": request.group_id,
        "id": request.dataset_id,
        "color": request.color,
        "enrich_conf": request.enrich_conf,
        "save_dataset": True,
        **request.extra_kwargs,
    }
    if request.ref_id is not None:
        import_kwargs["refID"] = request.ref_id
    imported_dataset = lab.import_dataset(**import_kwargs)
    if imported_dataset is None:
        raise RuntimeError("Import failed: backend returned no dataset")

    dataset_dir_raw = getattr(getattr(imported_dataset, "config", None), "dir", None)
    if not isinstance(dataset_dir_raw, str) or not dataset_dir_raw.strip():
        raise RuntimeError(
            "Import failed: saved dataset was not found in portal-supported workspace layout"
        )

    dataset_dir = Path(dataset_dir_raw).expanduser().resolve()
    record = get_workspace_dataset(dataset_dir)
    if record is not None:
        return record

    data_dir = dataset_dir / "data"
    conf_path = data_dir / "conf.txt"
    h5_path = data_dir / "data.h5"
    if not conf_path.is_file() or not h5_path.is_file():
        raise RuntimeError(
            "Import failed: saved dataset was not found in portal-supported workspace layout"
        )
    raise RuntimeError(
        "Import failed: workspace dataset record could not be built from saved output"
    )
