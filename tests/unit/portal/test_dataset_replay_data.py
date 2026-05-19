from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from larvaworld.lib import reg
from larvaworld.lib.process.dataset import LarvaDataset
from larvaworld.portal.datasets.replay_data import (
    build_render_state,
    build_source_catalog,
    prepare_replay_source,
    _resolve_xy_columns,
)
from larvaworld.portal.workspace import initialize_workspace


@pytest.fixture(autouse=True)
def workspace_config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LARVAWORLD_PORTAL_CONFIG_DIR", str(tmp_path / "config"))


def _write_workspace_dataset(
    dataset_dir: Path,
    *,
    dataset_id: str,
    lab_id: str = "Schleyer",
    group_id: str | None = None,
    n_agents: int = 2,
    n_ticks: int = 5,
) -> None:
    data_dir = dataset_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    conf = {
        "id": dataset_id,
        "dir": str(dataset_dir),
        "refID": None,
        "group_id": group_id,
        "dt": 0.1,
        "fr": 10.0,
        "Nticks": n_ticks,
        "Npoints": 3,
        "Ncontour": 0,
        "agent_ids": [f"{dataset_id}_a{i}" for i in range(n_agents)],
        "N": n_agents,
        "env_params": {
            "id": "dish",
            "arena": {"geometry": "rectangular", "dims": [0.2, 0.1]},
        },
        "larva_group": {"group_id": group_id, "default_color": "#2f4858"},
        "x": "x",
        "y": "y",
    }
    (data_dir / "conf.txt").write_text(json.dumps(conf), encoding="utf-8")
    rows = []
    for tick in range(n_ticks):
        for agent_idx in range(n_agents):
            rows.append(
                {
                    "Step": tick,
                    "AgentID": f"{dataset_id}_a{agent_idx}",
                    "x": 0.01 * tick + agent_idx * 0.005,
                    "y": 0.02 * tick + agent_idx * 0.003,
                }
            )
    step = pd.DataFrame(rows).set_index(["Step", "AgentID"]).sort_index()
    step.to_hdf(data_dir / "data.h5", key="step")
    end = pd.DataFrame(index=[f"{dataset_id}_a{i}" for i in range(n_agents)])
    end.to_hdf(data_dir / "data.h5", key="end")


def test_source_catalog_workspace_groups_by_lab_and_group(tmp_path: Path) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    ds_a = workspace.datasets_dir / "imported" / "LabA" / "grp1" / "a"
    ds_b = workspace.datasets_dir / "imported" / "LabA" / "grp1" / "b"
    ds_c = workspace.datasets_dir / "imported" / "LabB" / "grp1" / "c"
    _write_workspace_dataset(ds_a, dataset_id="a", group_id="grp1")
    _write_workspace_dataset(ds_b, dataset_id="b", group_id="grp1")
    _write_workspace_dataset(ds_c, dataset_id="c", group_id="grp1")

    sources = build_source_catalog(workspace)
    group_sources = [s for s in sources if s.source_type == "workspace_group"]
    labels = {s.label for s in group_sources}

    assert "Workspace / Group / LabA:grp1" in labels
    assert "Workspace / Group / LabB:grp1" not in labels


def test_prepare_replay_source_does_not_call_write_capable_methods(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    ds_a = workspace.datasets_dir / "imported" / "Schleyer" / "grp1" / "a"
    _write_workspace_dataset(ds_a, dataset_id="a", group_id="grp1")
    source = next(
        s
        for s in build_source_catalog(workspace)
        if s.source_type == "workspace_dataset"
    )

    def _boom(*_args, **_kwargs):
        raise AssertionError("write-capable method must not be called")

    monkeypatch.setattr(LarvaDataset, "load_traj", _boom)
    monkeypatch.setattr(LarvaDataset, "store", _boom)
    monkeypatch.setattr(LarvaDataset, "save", _boom)
    monkeypatch.setattr(LarvaDataset, "save_config", _boom)

    prepared = prepare_replay_source(source)
    assert prepared.nticks >= 1
    assert len(prepared.members) == 1


def test_build_render_state_origin_dispersal_and_labels(tmp_path: Path) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    ds_a = workspace.datasets_dir / "imported" / "Schleyer" / "grp1" / "a"
    _write_workspace_dataset(
        ds_a, dataset_id="a", group_id="grp1", n_agents=1, n_ticks=4
    )
    source = next(
        s
        for s in build_source_catalog(workspace)
        if s.source_type == "workspace_dataset"
    )
    prepared = prepare_replay_source(source)
    member_token = next(iter(prepared.members.keys()))

    state = build_render_state(
        prepared,
        tick=2,
        member_tokens=[member_token],
        show_positions=True,
        show_ids=True,
        show_tracks=True,
        trail_length=10,
        transposition="origin",
        track_point=-1,
        time_range=None,
        show_dispersal_ring=True,
    )

    assert len(state.frame.centroids) == 1
    assert len(state.frame.labels) == 1
    assert state.frame.labels[0] == "a_a0"
    assert len(state.rings) == 1
    assert state.rings[0].radius > 0


def test_resolve_xy_columns_falls_back_to_point_xy_for_registry_reference() -> None:
    dataset = reg.conf.Ref.loadRef(id="exploration.dish01", load=False)
    dataset.load(step=True)

    cols = _resolve_xy_columns(dataset, dataset.s, track_point=-1)

    assert cols == list(dataset.c.point_xy)
