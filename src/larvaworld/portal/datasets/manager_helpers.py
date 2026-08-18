from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from larvaworld.lib.process.dataset import LarvaDataset
from larvaworld.portal.datasets.models import (
    WorkspaceDatasetRecord,
    WorkspaceReplayDatasetRecord,
)
from larvaworld.portal.datasets.workspace_index import (
    list_all_workspace_datasets,
    list_data_dir_datasets,
)
from larvaworld.portal.workspace import WorkspaceState, get_workspace_dir


@dataclass(frozen=True)
class UnifiedDatasetRecord:
    #: "bundled" datasets ship with the package itself (DATA_DIR, e.g. the
    #: default reference dataset) -- never workspace-deletable, same as
    #: "simulation_run" (see _DatasetManagerController's delete gating,
    #: keyed off `origin == "imported"`).
    origin: Literal["imported", "simulation_run", "bundled"]
    dataset_id: str
    dataset_dir: Path
    data_dir: Path
    conf_path: Path
    h5_path: Path
    lab_id: str | None
    group_id: str | None
    ref_id: str | None
    n_agents: int | None
    run_id: str | None = None
    member_id: str | None = None

    @staticmethod
    def from_imported(record: WorkspaceDatasetRecord) -> UnifiedDatasetRecord:
        return UnifiedDatasetRecord(
            origin="imported",
            dataset_id=record.dataset_id,
            dataset_dir=record.dataset_dir,
            data_dir=record.data_dir,
            conf_path=record.conf_path,
            h5_path=record.h5_path,
            lab_id=record.lab_id,
            group_id=record.group_id,
            ref_id=record.ref_id,
            n_agents=record.n_agents,
        )

    @staticmethod
    def from_simulated(record: WorkspaceReplayDatasetRecord) -> UnifiedDatasetRecord:
        return UnifiedDatasetRecord(
            origin="simulation_run",
            dataset_id=record.dataset_id,
            dataset_dir=record.dataset_dir,
            data_dir=record.data_dir,
            conf_path=record.conf_path,
            h5_path=record.h5_path,
            lab_id=None,
            group_id=record.group_id,
            ref_id=record.ref_id,
            n_agents=record.n_agents,
            run_id=record.run_id,
            member_id=record.member_id,
        )

    @staticmethod
    def from_bundled(record: WorkspaceDatasetRecord) -> UnifiedDatasetRecord:
        return UnifiedDatasetRecord(
            origin="bundled",
            dataset_id=record.dataset_id,
            dataset_dir=record.dataset_dir,
            data_dir=record.data_dir,
            conf_path=record.conf_path,
            h5_path=record.h5_path,
            lab_id=record.lab_id,
            group_id=record.group_id,
            ref_id=record.ref_id,
            n_agents=record.n_agents,
        )


def list_all_unified_datasets(
    workspace: WorkspaceState | None = None,
) -> list[UnifiedDatasetRecord]:
    imported, simulated = list_all_workspace_datasets(workspace=workspace)
    records = [UnifiedDatasetRecord.from_imported(r) for r in imported]
    records.extend([UnifiedDatasetRecord.from_simulated(r) for r in simulated])
    records.extend(
        UnifiedDatasetRecord.from_bundled(r) for r in list_data_dir_datasets()
    )
    return sorted(records, key=lambda r: (r.origin, r.dataset_id))


def get_processing_status(record: UnifiedDatasetRecord) -> dict[str, bool]:
    """
    Check which processing/annotation steps have been applied to a dataset.
    Returns a dict mapping step name to whether it's been done.
    """
    try:
        ds = LarvaDataset(dir=str(record.dataset_dir), refID=record.ref_id)
        status = {
            "preprocessing_applied": ds.c.get("preprocessing", {}) != {},
            "spatial_computed": "v" in ds.step_ps or "angular_velocity" in ds.step_ps,
            "angular_computed": "phi" in ds.step_ps or "angular_velocity" in ds.step_ps,
            "dispersal_computed": "d" in ds.step_ps,
            "tortuosity_computed": "tor" in ds.step_ps,
            "bouts_detected": "b_id" in ds.step_ps or "b_start" in ds.step_ps,
            "interference_computed": "interf" in ds.end_ps,
        }
        return status
    except Exception:
        return {}


def preprocess_dataset(record: UnifiedDatasetRecord, **kwargs) -> tuple[bool, str]:
    """Run preprocessing on a dataset. Returns (success, message)."""
    try:
        ds = LarvaDataset(dir=str(record.dataset_dir), refID=record.ref_id)
        ds.preprocess(**kwargs)
        return True, f"Preprocessing completed for '{record.dataset_id}'."
    except Exception as exc:
        return False, f"Preprocessing failed: {str(exc)}"


def process_dataset(
    record: UnifiedDatasetRecord, proc_keys: list[str] | None = None, **kwargs
) -> tuple[bool, str]:
    """Run processing on a dataset. Returns (success, message)."""
    if proc_keys is None:
        proc_keys = ["angular", "spatial"]
    try:
        ds = LarvaDataset(dir=str(record.dataset_dir), refID=record.ref_id)
        ds.process(proc_keys=proc_keys, is_last=True, **kwargs)
        keys_str = ", ".join(proc_keys)
        return True, f"Processing completed for '{record.dataset_id}' ({keys_str})."
    except Exception as exc:
        return False, f"Processing failed: {str(exc)}"


def annotate_dataset(
    record: UnifiedDatasetRecord, anot_keys: list[str] | None = None, **kwargs
) -> tuple[bool, str]:
    """Run annotation on a dataset. Returns (success, message)."""
    if anot_keys is None:
        anot_keys = ["bout_detection", "bout_distribution"]
    try:
        ds = LarvaDataset(dir=str(record.dataset_dir), refID=record.ref_id)
        ds.annotate(anot_keys=anot_keys, is_last=True, **kwargs)
        keys_str = ", ".join(anot_keys)
        return True, f"Annotation completed for '{record.dataset_id}' ({keys_str})."
    except Exception as exc:
        return False, f"Annotation failed: {str(exc)}"


def update_dataset_refid(
    record: UnifiedDatasetRecord, new_ref_id: str
) -> tuple[bool, str]:
    """Update or create a reference ID for a dataset. Returns (success, message)."""
    try:
        ds = LarvaDataset(dir=str(record.dataset_dir), refID=new_ref_id)
        ds.save_config(refID=new_ref_id)
        return True, f"Reference ID updated to '{new_ref_id}'."
    except Exception as exc:
        return False, f"Failed to update reference ID: {str(exc)}"


def subsample_dataset(
    record: UnifiedDatasetRecord,
    n_agents: int,
    output_name: str,
    workspace: WorkspaceState,
) -> tuple[bool, str]:
    """
    Create a subsampled dataset with fewer agents.
    Returns (success, message)
    """
    try:
        ds = LarvaDataset(dir=str(record.dataset_dir), refID=record.ref_id)
        output_dir = (
            get_workspace_dir("datasets", workspace=workspace)
            / "imported"
            / output_name
        )
        if output_dir.exists():
            raise FileExistsError(f"Output dataset already exists: {output_dir}")
        ds.derive_subsample(
            n_agents,
            new_id=output_name,
            new_dir=str(output_dir),
        )
        return True, f"Subsampled dataset saved to '{output_name}'."
    except Exception as exc:
        return False, f"Subsampling failed: {str(exc)}"


def timeslice_dataset(
    record: UnifiedDatasetRecord,
    time_range: tuple[float, float],
    output_name: str,
    workspace: WorkspaceState,
) -> tuple[bool, str]:
    """
    Create a time-sliced dataset.
    time_range: (start_time, end_time) tuple in seconds
    Returns (success, message)
    """
    try:
        ds = LarvaDataset(dir=str(record.dataset_dir), refID=record.ref_id)
        output_dir = (
            get_workspace_dir("datasets", workspace=workspace)
            / "imported"
            / output_name
        )
        if output_dir.exists():
            raise FileExistsError(f"Output dataset already exists: {output_dir}")
        ds.derive_timeseries_slice(
            time_range,
            new_id=output_name,
            new_dir=str(output_dir),
        )
        return True, f"Time-sliced dataset saved to '{output_name}'."
    except Exception as exc:
        return False, f"Time-slicing failed: {str(exc)}"


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
