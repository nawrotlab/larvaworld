from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class WorkspaceDatasetRecord:
    dataset_id: str
    dataset_dir: Path
    data_dir: Path
    conf_path: Path
    h5_path: Path
    lab_id: str | None
    group_id: str | None
    ref_id: str | None
    n_agents: int | None


@dataclass(frozen=True)
class ImportRequest:
    lab_id: str
    parent_dir: str
    raw_folder: Path | None = None
    group_id: str | None = None
    dataset_id: str | None = None
    merged: bool = False
    color: str = "black"
    ref_id: str | None = None
    enrich_conf: dict[str, object] | None = None
    extra_kwargs: dict[str, object] = field(default_factory=dict)
