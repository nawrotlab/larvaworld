from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd

from larvaworld.portal.datasets.models import (
    WorkspaceDatasetRecord,
    WorkspaceReplayDatasetRecord,
)


ReplaySourceType = Literal[
    "workspace_dataset",
    "workspace_group",
    "workspace_simulation_run",
    "registry_reference",
    "registry_reference_group",
]

ReplayCoordinateOrigin = Literal["corner", "centered"]


@dataclass(frozen=True)
class ReplaySourceMember:
    token: str
    label: str
    source_type: ReplaySourceType
    workspace_record: WorkspaceDatasetRecord | WorkspaceReplayDatasetRecord | None = (
        None
    )
    registry_ref_id: str | None = None


@dataclass(frozen=True)
class ReplaySource:
    token: str
    label: str
    source_type: ReplaySourceType
    members: tuple[ReplaySourceMember, ...]


@dataclass(frozen=True)
class PreparedReplayMember:
    token: str
    label: str
    color: str
    xy_default: pd.DataFrame
    arena_dims: tuple[float, float]
    dt: float
    nticks: int
    env_conf_id: str | None = None
    env_params: dict[str, Any] | None = None
    coordinate_origin: ReplayCoordinateOrigin = "corner"
    xy_by_track_point: dict[int, pd.DataFrame] = field(default_factory=dict)
    agent_ids: tuple[object, ...] = ()


@dataclass
class PreparedReplaySource:
    source: ReplaySource
    members: dict[str, PreparedReplayMember] = field(default_factory=dict)
    nticks: int = 0
    dt: float = 0.1
    arena_dims: tuple[float, float] = (0.2, 0.2)
