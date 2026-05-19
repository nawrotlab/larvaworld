from __future__ import annotations

import json
from pathlib import Path

import panel as pn
import pandas as pd
import pytest

from larvaworld.portal.datasets.dataset_replay_app import (
    _DatasetReplayController,
    dataset_replay_app,
)
from larvaworld.portal.datasets.replay_models import ReplaySource
from larvaworld.portal.workspace import (
    clear_active_workspace_path,
    initialize_workspace,
    set_active_workspace_path,
)


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


def _write_workspace_simulation_dataset(
    run_dir: Path,
    *,
    member_id: str,
    dataset_id: str,
) -> None:
    dataset_dir = run_dir / "data" / member_id
    data_dir = dataset_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    conf = {
        "id": dataset_id,
        "dir": str(dataset_dir),
        "group_id": member_id,
        "dt": 0.1,
        "fr": 10.0,
        "Nticks": 4,
        "agent_ids": [f"{dataset_id}_a0"],
        "N": 1,
        "env_params": {
            "arena": {"geometry": "rectangular", "dims": [0.2, 0.1]},
        },
        "larva_group": {"group_id": member_id},
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
    assert not hasattr(controller, "source_type_select")
    assert controller.source_select.value is not None
    assert len(controller.member_visibility.options) >= 1
    assert controller.tick_player.end >= 0
    assert controller.transposition.value == "origin"


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
    assert controller.canvas._arena_rect_renderer.visible is False
    assert controller.canvas._arena_circle_renderer.visible is False


def test_dataset_replay_controller_center_mode_hides_arena_outline(
    tmp_path: Path,
) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    set_active_workspace_path(workspace.root)
    dataset_dir = workspace.datasets_dir / "imported" / "Schleyer" / "grp1" / "ds1"
    _write_workspace_dataset(dataset_dir, dataset_id="ds1", group_id="grp1")
    controller = _DatasetReplayController()

    controller.transposition.value = "center"
    controller._render()

    assert controller.canvas._arena_rect_renderer.visible is False
    assert controller.canvas._arena_circle_renderer.visible is False


def test_dataset_replay_controller_stored_corner_coordinates_hide_arena_outline(
    tmp_path: Path,
) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    set_active_workspace_path(workspace.root)
    dataset_dir = workspace.datasets_dir / "imported" / "Schleyer" / "grp1" / "ds1"
    _write_workspace_dataset(dataset_dir, dataset_id="ds1", group_id="grp1")
    controller = _DatasetReplayController()

    controller.transposition.value = None
    controller._render()

    assert controller.canvas._arena_rect_renderer.visible is False
    assert controller.canvas._arena_circle_renderer.visible is False


def test_dataset_replay_controller_stored_centered_coordinates_show_arena_outline(
    tmp_path: Path,
) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    set_active_workspace_path(workspace.root)
    _write_workspace_simulation_dataset(
        workspace.experiments_dir / "run_one",
        member_id="explorer",
        dataset_id="run_one_explorer",
    )
    controller = _DatasetReplayController()
    simulation_option = next(
        token
        for label, token in controller.source_select.options.items()
        if label.startswith("Workspace / Simulation run / ")
    )

    controller.source_select.value = simulation_option
    controller.transposition.value = None
    controller._render()

    assert controller.canvas._arena_rect_renderer.visible is True
    assert controller.canvas._arena_circle_renderer.visible is False


def test_dataset_replay_controller_arena_mode_shows_arena_outline(
    tmp_path: Path,
) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    set_active_workspace_path(workspace.root)
    dataset_dir = workspace.datasets_dir / "imported" / "Schleyer" / "grp1" / "ds1"
    _write_workspace_dataset(dataset_dir, dataset_id="ds1", group_id="grp1")
    controller = _DatasetReplayController()

    controller.transposition.value = "arena"
    controller._render()

    assert (
        controller.canvas._arena_rect_renderer.visible
        or controller.canvas._arena_circle_renderer.visible
    )


def test_dataset_replay_controller_lists_simulation_run_source(
    tmp_path: Path,
) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    set_active_workspace_path(workspace.root)
    _write_workspace_simulation_dataset(
        workspace.experiments_dir / "run_one",
        member_id="explorer",
        dataset_id="run_one_explorer",
    )

    controller = _DatasetReplayController()
    simulation_option = next(
        (
            token
            for label, token in controller.source_select.options.items()
            if label.startswith("Workspace / Simulation run / ")
        ),
        None,
    )
    assert simulation_option is not None
    controller.source_select.value = simulation_option
    assert len(controller.member_visibility.options) >= 1
    assert controller.tick_player.end >= 0


def test_render_applies_dynamic_state_after_static_refresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    set_active_workspace_path(workspace.root)
    dataset_dir = workspace.datasets_dir / "imported" / "Schleyer" / "grp1" / "ds1"
    _write_workspace_dataset(dataset_dir, dataset_id="ds1", group_id="grp1")
    controller = _DatasetReplayController()

    call_order: list[str] = []
    orig_set_state = controller.canvas.set_state
    orig_set_larva_frame = controller.canvas.set_larva_frame
    orig_set_dynamic_overlays = controller.canvas.set_dynamic_overlays

    def _set_state(*args, **kwargs):
        call_order.append("set_state")
        return orig_set_state(*args, **kwargs)

    def _set_larva_frame(*args, **kwargs):
        call_order.append("set_larva_frame")
        return orig_set_larva_frame(*args, **kwargs)

    def _set_dynamic_overlays(*args, **kwargs):
        call_order.append("set_dynamic_overlays")
        return orig_set_dynamic_overlays(*args, **kwargs)

    monkeypatch.setattr(controller.canvas, "set_state", _set_state)
    monkeypatch.setattr(controller.canvas, "set_larva_frame", _set_larva_frame)
    monkeypatch.setattr(
        controller.canvas, "set_dynamic_overlays", _set_dynamic_overlays
    )

    controller._last_static_state_key = None
    controller._render()

    assert call_order.index("set_state") < call_order.index("set_larva_frame")
    assert call_order.index("set_larva_frame") < call_order.index(
        "set_dynamic_overlays"
    )


def test_dataset_replay_controller_view_has_no_source_type_widget(
    tmp_path: Path,
) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    set_active_workspace_path(workspace.root)
    dataset_dir = workspace.datasets_dir / "imported" / "Schleyer" / "grp1" / "ds1"
    _write_workspace_dataset(dataset_dir, dataset_id="ds1", group_id="grp1")

    controller = _DatasetReplayController()
    names = [widget.name for widget in controller.view().select(pn.widgets.Widget)]

    assert "Source type" not in names
    assert "Source" in names


def test_dataset_replay_controller_preserves_source_on_reload(tmp_path: Path) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    set_active_workspace_path(workspace.root)
    imported_dir = workspace.datasets_dir / "imported" / "Schleyer" / "grp1" / "ds1"
    _write_workspace_dataset(imported_dir, dataset_id="ds1", group_id="grp1")
    _write_workspace_simulation_dataset(
        workspace.experiments_dir / "run_one",
        member_id="explorer",
        dataset_id="run_one_explorer",
    )
    controller = _DatasetReplayController()
    selected = next(
        token
        for label, token in controller.source_select.options.items()
        if label.startswith("Workspace / Simulation run / ")
    )
    controller.source_select.value = selected

    controller._reload_source_options()

    assert controller.source_select.value == selected


def test_dataset_replay_controller_disambiguates_duplicate_source_labels(
    tmp_path: Path,
) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    set_active_workspace_path(workspace.root)
    controller = _DatasetReplayController()
    source_a = ReplaySource(
        token="workspace:/tmp/a/ds",
        label="Workspace / Imported dataset / ds1",
        source_type="workspace_dataset",
        members=(),
    )
    source_b = ReplaySource(
        token="workspace:/tmp/b/ds",
        label="Workspace / Imported dataset / ds1",
        source_type="workspace_dataset",
        members=(),
    )
    controller._sources = [source_a, source_b]
    controller._source_by_token = {
        source.token: source for source in controller._sources
    }

    controller._reload_source_options()

    labels = list(controller.source_select.options.keys())
    assert len(labels) == 2
    assert labels[0] != labels[1]
    assert all(
        label.startswith("Workspace / Imported dataset / ds1 (") for label in labels
    )


def test_dataset_replay_controller_registry_only_without_workspace() -> None:
    clear_active_workspace_path()

    controller = _DatasetReplayController()

    assert len(controller.source_select.options) >= 1
    assert any(
        label.startswith("Registry / Reference")
        for label in controller.source_select.options
    )


def test_dataset_replay_app_returns_viewable() -> None:
    view = dataset_replay_app()
    assert view is not None
