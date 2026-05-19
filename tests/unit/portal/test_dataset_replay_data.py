from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from larvaworld.lib import reg
from larvaworld.lib.process.dataset import LarvaDataset
from larvaworld.portal.datasets.replay_data import (
    build_environment_state_for_member,
    build_render_state,
    build_source_catalog,
    prepare_replay_source,
    _resolve_xy_columns,
)
from larvaworld.portal.datasets.replay_models import (
    PreparedReplayMember,
    PreparedReplaySource,
    ReplaySource,
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


def _write_simulation_workspace_dataset(
    run_dir: Path,
    *,
    member_id: str,
    dataset_id: str,
    group_id: str | None = None,
    n_agents: int = 1,
    n_ticks: int = 5,
) -> None:
    dataset_dir = run_dir / "data" / member_id
    data_dir = dataset_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    conf = {
        "id": dataset_id,
        "dir": str(dataset_dir),
        "refID": None,
        "group_id": group_id,
        "dt": 0.2,
        "fr": 5.0,
        "Nticks": n_ticks,
        "agent_ids": [f"{dataset_id}_a{i}" for i in range(n_agents)],
        "N": n_agents,
        "env_params": {"arena": {"geometry": "rectangular", "dims": [0.3, 0.2]}},
        "larva_group": {"group_id": group_id},
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
                    "x": 0.03 * tick + agent_idx * 0.01,
                    "y": 0.01 * tick + agent_idx * 0.02,
                }
            )
    pd.DataFrame(rows).set_index(["Step", "AgentID"]).sort_index().to_hdf(
        data_dir / "data.h5", key="step"
    )
    pd.DataFrame(index=[f"{dataset_id}_a{i}" for i in range(n_agents)]).to_hdf(
        data_dir / "data.h5", key="end"
    )


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

    assert "Workspace / Imported group / LabA:grp1" in labels
    assert "Workspace / Imported group / LabB:grp1" not in labels


def test_source_catalog_labels_imported_dataset_prefix(tmp_path: Path) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    ds_a = workspace.datasets_dir / "imported" / "LabA" / "grp1" / "alpha"
    _write_workspace_dataset(ds_a, dataset_id="alpha", group_id="grp1")

    sources = build_source_catalog(workspace)
    dataset_source = next(s for s in sources if s.source_type == "workspace_dataset")

    assert dataset_source.label == "Workspace / Imported dataset / alpha"


def test_source_catalog_adds_workspace_simulation_run_sources(tmp_path: Path) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    run_dir = workspace.experiments_dir / "run_alpha"
    _write_simulation_workspace_dataset(
        run_dir, member_id="g1", dataset_id="run_alpha_g1", group_id="grp1"
    )
    _write_simulation_workspace_dataset(
        run_dir, member_id="g2", dataset_id="run_alpha_g2", group_id="grp1"
    )

    sources = build_source_catalog(workspace)
    run_sources = [s for s in sources if s.source_type == "workspace_simulation_run"]

    assert len(run_sources) == 1
    assert run_sources[0].token == "workspace_simulation_run:run_alpha"
    assert [m.label for m in run_sources[0].members] == ["run_alpha_g1", "run_alpha_g2"]


def test_simulation_sources_do_not_merge_into_workspace_groups(
    tmp_path: Path,
) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    imported_a = workspace.datasets_dir / "imported" / "LabA" / "grp1" / "a"
    imported_b = workspace.datasets_dir / "imported" / "LabA" / "grp1" / "b"
    _write_workspace_dataset(imported_a, dataset_id="a", group_id="grp1")
    _write_workspace_dataset(imported_b, dataset_id="b", group_id="grp1")
    _write_simulation_workspace_dataset(
        workspace.experiments_dir / "run_mix",
        member_id="grp1",
        dataset_id="run_mix_grp1",
        group_id="grp1",
    )

    sources = build_source_catalog(workspace)
    group_sources = [s for s in sources if s.source_type == "workspace_group"]
    run_sources = [s for s in sources if s.source_type == "workspace_simulation_run"]

    assert [s.label for s in group_sources] == [
        "Workspace / Imported group / LabA:grp1"
    ]
    assert len(run_sources) == 1


def test_prepare_replay_source_supports_workspace_simulation_member(
    tmp_path: Path,
) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    _write_simulation_workspace_dataset(
        workspace.experiments_dir / "run_delta",
        member_id="explorer",
        dataset_id="run_delta_explorer",
        n_ticks=6,
    )
    source = next(
        s
        for s in build_source_catalog(workspace)
        if s.source_type == "workspace_simulation_run"
    )

    prepared = prepare_replay_source(source)

    assert prepared.nticks >= 6
    assert len(prepared.members) == 1
    assert next(iter(prepared.members.values())).coordinate_origin == "centered"


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


def test_environment_state_shows_outline_without_static_layers_by_default() -> None:
    step = pd.DataFrame([{"Step": 0, "AgentID": "a0", "x": 0.0, "y": 0.0}]).set_index(
        ["Step", "AgentID"]
    )
    member = PreparedReplayMember(
        token="m1",
        label="m1",
        color="#000000",
        xy_default=step,
        arena_dims=(0.2, 0.1),
        dt=0.1,
        nticks=1,
        env_conf_id=None,
    )

    state = build_environment_state_for_member(member, allow_static_layers=False)

    assert state.show_arena_outline is True
    assert state.objects == ()


def test_build_environment_state_for_member_can_hide_outline_when_requested() -> None:
    step = pd.DataFrame([{"Step": 0, "AgentID": "a0", "x": 0.0, "y": 0.0}]).set_index(
        ["Step", "AgentID"]
    )
    member = PreparedReplayMember(
        token="m1",
        label="m1",
        color="#000000",
        xy_default=step,
        arena_dims=(0.2, 0.1),
        dt=0.1,
        nticks=1,
        env_conf_id=None,
    )

    state = build_environment_state_for_member(
        member, allow_static_layers=False, show_arena_outline=False
    )

    assert state.show_arena_outline is False
    assert state.objects == ()


def test_build_environment_state_for_member_shows_outline_when_static_enabled() -> None:
    step = pd.DataFrame([{"Step": 0, "AgentID": "a0", "x": 0.0, "y": 0.0}]).set_index(
        ["Step", "AgentID"]
    )
    member = PreparedReplayMember(
        token="m1",
        label="m1",
        color="#000000",
        xy_default=step,
        arena_dims=(0.2, 0.1),
        dt=0.1,
        nticks=1,
        env_conf_id=None,
    )

    state = build_environment_state_for_member(member, allow_static_layers=True)

    assert state.show_arena_outline is True


def test_build_environment_state_uses_embedded_env_params_for_borders() -> None:
    step = pd.DataFrame([{"Step": 0, "AgentID": "a0", "x": 0.0, "y": 0.0}]).set_index(
        ["Step", "AgentID"]
    )
    member = PreparedReplayMember(
        token="m1",
        label="m1",
        color="#000000",
        xy_default=step,
        arena_dims=(0.2, 0.2),
        dt=0.1,
        nticks=1,
        env_params={
            "arena": {"geometry": "rectangular", "dims": (0.2, 0.2)},
            "border_list": {
                "border_001": {
                    "vertices": [(0.0007, 0.0497), (-0.0494, -0.0003)],
                    "width": 0.001,
                    "color": "#eb3110",
                }
            },
        },
    )

    state = build_environment_state_for_member(member, allow_static_layers=True)

    assert state.show_arena_outline is True
    assert [obj.object_type for obj in state.objects] == ["border_segment"]
    assert state.objects[0].color == "#eb3110"


def test_build_render_state_arena_keeps_centered_simulation_coordinates() -> None:
    step = pd.DataFrame(
        [
            {"Step": 0, "AgentID": "a0", "x": -0.04, "y": 0.02},
            {"Step": 1, "AgentID": "a0", "x": 0.03, "y": -0.01},
        ]
    ).set_index(["Step", "AgentID"])
    member = PreparedReplayMember(
        token="sim",
        label="sim",
        color="#000000",
        xy_default=step,
        arena_dims=(0.2, 0.2),
        dt=0.1,
        nticks=2,
        coordinate_origin="centered",
    )
    source = PreparedReplaySource(
        source=ReplaySource(
            token="workspace_simulation_run:run",
            label="Workspace / Simulation run / run",
            source_type="workspace_simulation_run",
            members=(),
        ),
        members={"sim": member},
        nticks=2,
        dt=0.1,
    )

    state = build_render_state(
        source,
        tick=1,
        member_tokens=["sim"],
        show_positions=True,
        show_ids=False,
        show_tracks=False,
        trail_length=0,
        transposition="arena",
        track_point=-1,
        time_range=None,
        show_dispersal_ring=False,
    )

    assert state.frame.centroids == ((0.03, -0.01),)
