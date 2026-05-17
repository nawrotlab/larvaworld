from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from larvaworld.portal.datasets.dataset_replay_app import (
    _DatasetReplayController,
    dataset_replay_app,
)
from larvaworld.portal.workspace import initialize_workspace, set_active_workspace_path


@pytest.fixture(autouse=True)
def workspace_config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LARVAWORLD_PORTAL_CONFIG_DIR", str(tmp_path / "config"))


def _write_workspace_dataset(
    dataset_dir: Path, *, dataset_id: str, group_id: str
) -> None:
    data_dir = dataset_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    conf = {
        "id": dataset_id,
        "dir": str(dataset_dir),
        "group_id": group_id,
        "dt": 0.1,
        "fr": 10.0,
        "Nticks": 4,
        "Npoints": 3,
        "Ncontour": 0,
        "agent_ids": [f"{dataset_id}_a0"],
        "N": 1,
        "env_params": {
            "id": "dish",
            "arena": {"geometry": "rectangular", "dims": [0.2, 0.1]},
        },
        "larva_group": {"group_id": group_id},
        "x": "x",
        "y": "y",
    }
    (data_dir / "conf.txt").write_text(json.dumps(conf), encoding="utf-8")
    rows = [
        {
            "Step": t,
            "AgentID": f"{dataset_id}_a0",
            "x": float(t) * 0.01,
            "y": float(t) * 0.02,
        }
        for t in range(4)
    ]
    pd.DataFrame(rows).set_index(["Step", "AgentID"]).to_hdf(
        data_dir / "data.h5", key="step"
    )
    pd.DataFrame(index=[f"{dataset_id}_a0"]).to_hdf(data_dir / "data.h5", key="end")


def test_dataset_replay_controller_loads_workspace_source(tmp_path: Path) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    set_active_workspace_path(workspace.root)
    dataset_dir = workspace.datasets_dir / "imported" / "Schleyer" / "grp1" / "ds1"
    _write_workspace_dataset(dataset_dir, dataset_id="ds1", group_id="grp1")

    controller = _DatasetReplayController()
    assert controller.source_select.value is not None
    assert len(controller.member_visibility.options) >= 1
    assert controller.tick_player.end >= 0


def test_dataset_replay_controller_origin_mode_builds_ring(tmp_path: Path) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    set_active_workspace_path(workspace.root)
    dataset_dir = workspace.datasets_dir / "imported" / "Schleyer" / "grp1" / "ds1"
    _write_workspace_dataset(dataset_dir, dataset_id="ds1", group_id="grp1")
    controller = _DatasetReplayController()

    controller.transposition.value = "origin"
    controller.show_dispersal.value = True
    controller.tick_player.value = 2
    controller._render()
    assert len(controller.canvas.dynamic_ring_source.data["r"]) <= 1


def test_dataset_replay_app_returns_viewable() -> None:
    view = dataset_replay_app()
    assert view is not None
