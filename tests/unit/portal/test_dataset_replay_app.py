from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Any

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
    dataset_dir: Path, *, dataset_id: str, group_id: str, n_agents: int = 1
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
        "agent_ids": [f"{dataset_id}_a{i}" for i in range(n_agents)],
        "N": n_agents,
        "env_params": {
            "id": "dish",
            "arena": {"geometry": "rectangular", "dims": [0.2, 0.1]},
        },
        "larva_group": {"group_id": group_id},
        "x": "x",
        "y": "y",
    }
    (data_dir / "conf.txt").write_text(json.dumps(conf), encoding="utf-8")
    rows = []
    for t in range(4):
        for i in range(n_agents):
            rows.append(
                {
                    "Step": t,
                    "AgentID": f"{dataset_id}_a{i}",
                    "x": float(t) * 0.01 + (i * 0.003),
                    "y": float(t) * 0.02 + (i * 0.004),
                    "front_orientation": 0.0,
                    "rear_orientation": 0.0,
                }
            )
    pd.DataFrame(rows).set_index(["Step", "AgentID"]).to_hdf(
        data_dir / "data.h5", key="step"
    )
    pd.DataFrame(index=[f"{dataset_id}_a{i}" for i in range(n_agents)]).to_hdf(
        data_dir / "data.h5", key="end"
    )


def _write_workspace_dataset_without_xy(
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
        "agent_ids": [f"{dataset_id}_a0"],
        "N": 1,
        "env_params": {
            "arena": {"geometry": "rectangular", "dims": [0.2, 0.1]},
        },
        "larva_group": {"group_id": group_id},
        "x": "missing_x",
        "y": "missing_y",
    }
    (data_dir / "conf.txt").write_text(json.dumps(conf), encoding="utf-8")
    rows = [
        {
            "Step": t,
            "AgentID": f"{dataset_id}_a0",
            "front_orientation": 0.0,
            "rear_orientation": 0.0,
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
            "front_orientation": 0.0,
            "rear_orientation": 0.0,
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
    assert controller.agent_indices.name == "Agent indices"
    assert controller.show_heads.name == "Heads"
    assert controller.show_midlines.name == "Midlines"
    assert controller.show_segments.name == "Body segments"
    assert controller.show_body_contours.name == "Body contours"


def test_dataset_replay_controller_handles_invalid_workspace_source_gracefully(
    tmp_path: Path,
) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    set_active_workspace_path(workspace.root)
    valid_dir = workspace.datasets_dir / "imported" / "Schleyer" / "grp1" / "ds1"
    invalid_dir = (
        workspace.datasets_dir / "imported" / "Schleyer" / "grp2" / "broken_ds"
    )
    _write_workspace_dataset(valid_dir, dataset_id="ds1", group_id="grp1")
    _write_workspace_dataset_without_xy(
        invalid_dir, dataset_id="broken_ds", group_id="grp2"
    )

    controller = _DatasetReplayController()
    broken_token = next(
        token
        for label, token in controller.source_select.options.items()
        if "broken_ds" in label
    )

    controller.source_select.value = broken_token

    assert controller._prepared is None
    assert controller.member_visibility.options == {}
    assert controller.member_visibility.value == []
    assert "Replay source load failed:" in controller.status_pane.object
    assert "Workspace dataset is missing xy columns: broken_ds" in (
        controller.status_pane.object
    )


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


def test_dataset_replay_controller_view_groups_controls_in_tiles(
    tmp_path: Path,
) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    set_active_workspace_path(workspace.root)
    dataset_dir = workspace.datasets_dir / "imported" / "Schleyer" / "grp1" / "ds1"
    _write_workspace_dataset(dataset_dir, dataset_id="ds1", group_id="grp1")

    controller = _DatasetReplayController()
    view = controller.view()
    cards = view.select(pn.Card)

    titles = {card.title for card in cards}
    assert {
        "Source",
        "Display",
        "Motion",
        "Coordinates",
        "Pygame replay",
    }.issubset(titles)
    assert "Native Close Inspection" not in titles
    assert "Time" not in titles
    assert "Media / Output" not in titles
    assert {
        "Time range",
        "Close inspection",
        "Body rendering",
        "Output",
    }.issubset(titles)
    assert not view.select(pn.Accordion)

    assert controller.use_time_range.name == "Limit replay time range"
    assert controller.time_start.name == "Start (s)"
    assert controller.time_end.name == "End (s)"
    assert controller.fix_point.name == "Fix point"
    assert controller.close_view.name == "Close view"
    assert controller.fix_segment.name == "Fix orientation"
    assert controller.native_body_rendering.name == "Body rendering"
    assert controller.native_segment_count.name == "Segment count"
    assert controller.show_display.name == "Show display"
    assert controller.display_every_n_steps.name == "Display every N steps"
    assert controller.save_video.name == "Save video"
    assert controller.video_filename.name == "Video filename"
    assert controller.video_fps.name == "Video speed-up"
    assert controller.display_shortcuts_link.name == "Display Shortcuts"
    assert controller.open_pygame_replay_btn.name == "Run replay"


def test_dataset_replay_controller_display_shortcuts_dialog_open_close(
    tmp_path: Path,
) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    set_active_workspace_path(workspace.root)
    dataset_dir = workspace.datasets_dir / "imported" / "Schleyer" / "grp1" / "ds1"
    _write_workspace_dataset(dataset_dir, dataset_id="ds1", group_id="grp1")

    controller = _DatasetReplayController()
    assert controller.display_shortcuts_dialog.visible is False
    controller.display_shortcuts_dialog_controller.open()
    assert controller.display_shortcuts_dialog.visible is True
    controller.display_shortcuts_dialog_controller.close()
    assert controller.display_shortcuts_dialog.visible is False


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


def test_dataset_replay_controller_invalid_agent_indices_sets_status(
    tmp_path: Path,
) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    set_active_workspace_path(workspace.root)
    dataset_dir = workspace.datasets_dir / "imported" / "Schleyer" / "grp1" / "ds1"
    _write_workspace_dataset(dataset_dir, dataset_id="ds1", group_id="grp1")
    controller = _DatasetReplayController()

    controller.agent_indices.value = "1,,2"
    controller._render()

    assert "Invalid Agent indices" in controller.status_pane.object


def test_dataset_replay_controller_show_display_toggles_runtime_controls(
    tmp_path: Path,
) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    set_active_workspace_path(workspace.root)
    dataset_dir = workspace.datasets_dir / "imported" / "Schleyer" / "grp1" / "ds1"
    _write_workspace_dataset(dataset_dir, dataset_id="ds1", group_id="grp1")
    controller = _DatasetReplayController()

    controller.show_display.value = True
    controller._on_show_display_change()
    assert controller.display_every_n_steps.disabled is False
    assert controller.open_pygame_replay_btn.disabled is False
    assert controller.video_filename.disabled is True
    assert controller.video_fps.disabled is True

    controller.show_display.value = False
    controller._on_show_display_change()
    assert controller.display_every_n_steps.disabled is True
    assert controller.open_pygame_replay_btn.disabled is True

    controller.save_video.value = True
    controller._on_save_video_change()
    assert controller.open_pygame_replay_btn.disabled is False
    assert controller.video_filename.disabled is False
    assert controller.video_fps.disabled is False


def test_dataset_replay_controller_missing_native_columns_disable_action(
    tmp_path: Path,
) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    set_active_workspace_path(workspace.root)
    dataset_dir = workspace.datasets_dir / "imported" / "Schleyer" / "grp1" / "ds1"
    _write_workspace_dataset(dataset_dir, dataset_id="ds1", group_id="grp1")
    controller = _DatasetReplayController()
    selected_member = next(iter(controller.member_visibility.options.values()))
    controller.member_visibility.value = [selected_member]
    controller._prepared.members[selected_member] = replace(
        controller._prepared.members[selected_member],
        native_replay_missing_columns=("front_orientation", "rear_orientation"),
    )

    controller._on_any_control_change()

    assert controller.open_pygame_replay_btn.disabled is True
    assert (
        "Native replay is unavailable for this member: missing "
        "front_orientation, rear_orientation." in controller.status_pane.object
    )


def test_dataset_replay_controller_native_action_enabled_for_capable_member(
    tmp_path: Path,
) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    set_active_workspace_path(workspace.root)
    dataset_dir = workspace.datasets_dir / "imported" / "Schleyer" / "grp1" / "ds1"
    _write_workspace_dataset(dataset_dir, dataset_id="ds1", group_id="grp1")
    controller = _DatasetReplayController()
    selected_member = next(iter(controller.member_visibility.options.values()))
    controller.member_visibility.value = [selected_member]
    controller.show_display.value = True
    controller.save_video.value = False

    controller._refresh_native_replay_control_state()

    assert controller.open_pygame_replay_btn.disabled is False


def test_dataset_replay_controller_native_action_disabled_without_output_mode(
    tmp_path: Path,
) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    set_active_workspace_path(workspace.root)
    dataset_dir = workspace.datasets_dir / "imported" / "Schleyer" / "grp1" / "ds1"
    _write_workspace_dataset(dataset_dir, dataset_id="ds1", group_id="grp1")
    controller = _DatasetReplayController()
    selected_member = next(iter(controller.member_visibility.options.values()))
    controller.member_visibility.value = [selected_member]
    controller.show_display.value = False
    controller.save_video.value = False

    controller._refresh_native_replay_control_state()

    assert controller.open_pygame_replay_btn.disabled is True


def test_dataset_replay_controller_native_lock_disables_shortcuts_and_hides_dialog(
    tmp_path: Path,
) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    set_active_workspace_path(workspace.root)
    dataset_dir = workspace.datasets_dir / "imported" / "Schleyer" / "grp1" / "ds1"
    _write_workspace_dataset(dataset_dir, dataset_id="ds1", group_id="grp1")
    controller = _DatasetReplayController()
    controller.display_shortcuts_dialog_controller.open()
    assert controller.display_shortcuts_dialog.visible is True

    controller._set_native_replay_controls_disabled(True)

    assert controller.display_shortcuts_link.disabled is True
    assert controller.display_shortcuts_close_btn.disabled is True
    assert controller.display_shortcuts_dialog.visible is False

    controller._set_native_replay_controls_disabled(False)
    assert controller.display_shortcuts_link.disabled is False
    assert controller.display_shortcuts_close_btn.disabled is False


def test_dataset_replay_controller_member_visibility_refreshes_native_action(
    tmp_path: Path,
) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    set_active_workspace_path(workspace.root)
    ds1 = workspace.datasets_dir / "imported" / "Schleyer" / "grp1" / "ds1"
    ds2 = workspace.datasets_dir / "imported" / "Schleyer" / "grp1" / "ds2"
    _write_workspace_dataset(ds1, dataset_id="ds1", group_id="grp1")
    _write_workspace_dataset(ds2, dataset_id="ds2", group_id="grp1")
    controller = _DatasetReplayController()
    source_token = next(
        token
        for label, token in controller.source_select.options.items()
        if label.startswith("Workspace / Imported group / ")
    )
    controller.source_select.value = source_token
    tokens = list(controller.member_visibility.options.values())
    unsupported, supported = tokens[0], tokens[1]
    controller._prepared.members[unsupported] = replace(
        controller._prepared.members[unsupported],
        native_replay_missing_columns=("front_orientation",),
    )
    controller._prepared.members[supported] = replace(
        controller._prepared.members[supported],
        native_replay_missing_columns=(),
    )

    controller.member_visibility.value = [unsupported]
    controller._on_any_control_change()
    assert controller.open_pygame_replay_btn.disabled is True

    controller.member_visibility.value = [supported]
    controller._on_any_control_change()
    assert controller.open_pygame_replay_btn.disabled is False


def test_dataset_replay_controller_invalid_track_point_sets_status(
    tmp_path: Path,
) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    set_active_workspace_path(workspace.root)
    dataset_dir = workspace.datasets_dir / "imported" / "Schleyer" / "grp1" / "ds1"
    _write_workspace_dataset(dataset_dir, dataset_id="ds1", group_id="grp1")
    controller = _DatasetReplayController()

    controller.track_point.value = 99
    controller._render()

    assert "Replay render error" in controller.status_pane.object


def test_dataset_replay_controller_valid_agent_indices_filters_agents(
    tmp_path: Path,
) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    set_active_workspace_path(workspace.root)
    dataset_dir = workspace.datasets_dir / "imported" / "Schleyer" / "grp1" / "ds1"
    _write_workspace_dataset(dataset_dir, dataset_id="ds1", group_id="grp1", n_agents=2)
    controller = _DatasetReplayController()

    controller.show_ids.value = True
    controller.agent_indices.value = "1"
    controller.tick_player.value = 1
    controller._render()

    assert controller.canvas.sim_larva_label_source.data["label"] == ["ds1_a1"]


def test_dataset_replay_controller_passes_body_visibility_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    set_active_workspace_path(workspace.root)
    dataset_dir = workspace.datasets_dir / "imported" / "Schleyer" / "grp1" / "ds1"
    _write_workspace_dataset(dataset_dir, dataset_id="ds1", group_id="grp1")
    controller = _DatasetReplayController()

    captured: dict[str, object] = {}

    def _stub_build_render_state(*args, **kwargs):
        captured.update(kwargs)
        from larvaworld.portal.datasets.replay_data import ReplayRenderState
        from larvaworld.portal.canvas_widgets.environment_models import (
            LarvaPreviewFrame,
        )

        return ReplayRenderState(frame=LarvaPreviewFrame(tick=0))

    monkeypatch.setattr(
        "larvaworld.portal.datasets.dataset_replay_app.build_render_state",
        _stub_build_render_state,
    )

    controller.show_heads.value = False
    controller.show_midlines.value = False
    controller.show_segments.value = False
    controller.show_body_contours.value = True
    controller._render()

    assert captured["show_heads"] is False
    assert captured["show_midlines"] is False
    assert captured["show_segments"] is False
    assert captured["show_body_contours"] is True


def test_dataset_replay_controller_pygame_replay_requires_one_visible_member(
    tmp_path: Path,
) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    set_active_workspace_path(workspace.root)
    dataset_dir = workspace.datasets_dir / "imported" / "Schleyer" / "grp1" / "ds1"
    _write_workspace_dataset(dataset_dir, dataset_id="ds1", group_id="grp1", n_agents=2)
    controller = _DatasetReplayController()

    controller.member_visibility.value = []
    controller._on_open_pygame_replay()
    assert (
        "Select one visible member for native replay." in controller.status_pane.object
    )

    controller.member_visibility.value = list(
        controller.member_visibility.options.values()
    )
    if len(controller.member_visibility.value) < 2:
        second = next(iter(controller.member_visibility.options.values()))
        controller.member_visibility.value = [second, second]
    controller._on_open_pygame_replay()
    assert (
        "Native replay supports one visible member in this version."
        in controller.status_pane.object
    )


def test_dataset_replay_controller_native_replay_requires_display_or_video(
    tmp_path: Path,
) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    set_active_workspace_path(workspace.root)
    dataset_dir = workspace.datasets_dir / "imported" / "Schleyer" / "grp1" / "ds1"
    _write_workspace_dataset(dataset_dir, dataset_id="ds1", group_id="grp1")
    controller = _DatasetReplayController()

    controller.show_display.value = False
    controller.save_video.value = False
    controller._on_open_pygame_replay()

    assert (
        "Enable Show display or Save video to run native replay."
        in controller.status_pane.object
    )


def test_dataset_replay_controller_pygame_replay_invalid_agent_indices_does_not_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    set_active_workspace_path(workspace.root)
    dataset_dir = workspace.datasets_dir / "imported" / "Schleyer" / "grp1" / "ds1"
    _write_workspace_dataset(dataset_dir, dataset_id="ds1", group_id="grp1")
    controller = _DatasetReplayController()
    controller.member_visibility.value = [
        next(iter(controller.member_visibility.options.values()))
    ]
    controller.agent_indices.value = "1,,2"

    called = {"value": False}

    class _FakeReplayRun:
        def __init__(self, **kwargs: Any):
            called["value"] = True

        def run(self) -> None:
            called["value"] = True

    monkeypatch.setattr(
        "larvaworld.portal.datasets.dataset_replay_app.sim.ReplayRun",
        _FakeReplayRun,
    )
    controller._on_open_pygame_replay()

    assert "Invalid Agent indices" in controller.status_pane.object
    assert called["value"] is False


def test_dataset_replay_controller_pygame_replay_registry_invocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    set_active_workspace_path(workspace.root)
    dataset_dir = workspace.datasets_dir / "imported" / "Schleyer" / "grp1" / "ds1"
    _write_workspace_dataset(dataset_dir, dataset_id="ds1", group_id="grp1")
    controller = _DatasetReplayController()
    registry_token = next(
        token
        for label, token in controller.source_select.options.items()
        if label.startswith("Registry / Reference dataset / ")
    )
    controller.source_select.value = registry_token
    selected_member = next(iter(controller.member_visibility.options.values()))
    controller.member_visibility.value = [selected_member]
    expected_draw_nsegs = max(
        2, len(controller._prepared.members[selected_member].body_xy_by_point) - 1
    )
    controller._prepared.members[selected_member] = replace(
        controller._prepared.members[selected_member],
        native_replay_missing_columns=(),
    )
    controller.show_display.value = True
    controller.display_every_n_steps.value = 3

    captured: dict[str, Any] = {}

    class _FakeScreenManager:
        def __init__(self) -> None:
            self.close_called = False

        def close(self) -> None:
            self.close_called = True

    class _FakeReplayRun:
        def __init__(self, **kwargs: Any):
            captured.update(kwargs)
            self.screen_manager = _FakeScreenManager()

        def run(self) -> None:
            captured["run_called"] = True

    monkeypatch.setattr(
        "larvaworld.portal.datasets.dataset_replay_app.sim.ReplayRun",
        _FakeReplayRun,
    )
    monkeypatch.setattr(
        "larvaworld.portal.datasets.dataset_replay_app.pn.state.curdoc",
        None,
        raising=False,
    )

    controller._on_open_pygame_replay()

    assert captured["dataset"] is None
    assert captured["store_data"] is False
    assert captured["run_called"] is True
    assert captured["screen_kws"]["show_display"] is True
    assert captured["screen_kws"]["vis_mode"] == "video"
    assert captured["screen_kws"]["display_every_n_steps"] == 3
    assert "pygame_keys" in captured["screen_kws"]
    assert captured["screen_kws"]["pygame_keys"]["pause"] == "K_SPACE"
    assert "save_video" not in captured["screen_kws"]
    prepared_member = controller._prepared.members[selected_member]
    assert captured["parameters"].track_point == int(
        prepared_member.native_default_track_point
    )
    assert "refDir" in captured["parameters"]
    assert captured["parameters"].draw_Nsegs == expected_draw_nsegs
    assert "Native pygame replay finished." in controller.status_pane.object


def test_dataset_replay_controller_native_replay_headless_video_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    set_active_workspace_path(workspace.root)
    dataset_dir = workspace.datasets_dir / "imported" / "Schleyer" / "grp1" / "ds1"
    _write_workspace_dataset(dataset_dir, dataset_id="ds1", group_id="grp1")
    controller = _DatasetReplayController()
    selected_member = next(iter(controller.member_visibility.options.values()))
    controller.member_visibility.value = [selected_member]
    controller.show_display.value = False
    controller.save_video.value = True
    controller.video_filename.value = "My Replay.mp4"
    controller.video_fps.value = 4

    captured: dict[str, Any] = {}

    class _FakeScreenManager:
        def __init__(self) -> None:
            self.close_called = False

        def close(self) -> None:
            self.close_called = True

    class _FakeReplayRun:
        def __init__(self, **kwargs: Any):
            captured.update(kwargs)
            self.screen_manager = _FakeScreenManager()

        def run(self) -> None:
            captured["run_called"] = True

    monkeypatch.setattr(
        "larvaworld.portal.datasets.dataset_replay_app.sim.ReplayRun",
        _FakeReplayRun,
    )
    monkeypatch.setattr(
        controller,
        "_build_native_replay_parameters",
        lambda **kwargs: ({"mock": "parameters"}, None),
    )
    monkeypatch.setattr(
        "larvaworld.portal.datasets.dataset_replay_app.pn.state.curdoc",
        None,
        raising=False,
    )

    controller._on_open_pygame_replay()

    assert captured["screen_kws"]["show_display"] is False
    assert captured["screen_kws"]["vis_mode"] == "video"
    assert "pygame_keys" in captured["screen_kws"]
    assert captured["screen_kws"]["save_video"] is True
    assert captured["screen_kws"]["video_file"] == "My_Replay"
    assert captured["screen_kws"]["fps"] == 4
    assert "dataset_replay_media" in captured["screen_kws"]["media_dir"]
    assert "Video target:" in controller.status_pane.object


def test_dataset_replay_controller_native_replay_explicit_track_point_mapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    set_active_workspace_path(workspace.root)
    dataset_dir = workspace.datasets_dir / "imported" / "Schleyer" / "grp1" / "ds1"
    _write_workspace_dataset(dataset_dir, dataset_id="ds1", group_id="grp1")
    data_h5 = dataset_dir / "data" / "data.h5"
    step = pd.read_hdf(data_h5, "step")
    step["head_x"] = step["x"]
    step["head_y"] = step["y"]
    step["point2_x"] = step["x"] + 0.001
    step["point2_y"] = step["y"] + 0.001
    step["tail_x"] = step["x"] + 0.002
    step["tail_y"] = step["y"] + 0.002
    step.to_hdf(data_h5, key="step")
    controller = _DatasetReplayController()
    selected_member = next(iter(controller.member_visibility.options.values()))
    controller.member_visibility.value = [selected_member]
    controller.show_display.value = True
    controller.track_point.value = 0

    monkeypatch.setattr(
        "larvaworld.portal.datasets.dataset_replay_app.LarvaDataset",
        lambda **kwargs: {"dataset_dir": kwargs.get("dir")},
    )

    parameters, _dataset = controller._build_native_replay_parameters(
        selected_member_token=selected_member,
        agent_indices=None,
        time_range=None,
    )

    prepared_member = controller._prepared.members[selected_member]
    assert parameters.track_point == int(
        prepared_member.native_track_point_by_ui_track_point[0]
    )


def test_dataset_replay_controller_native_close_inspection_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    set_active_workspace_path(workspace.root)
    dataset_dir = workspace.datasets_dir / "imported" / "Schleyer" / "grp1" / "ds1"
    _write_workspace_dataset(dataset_dir, dataset_id="ds1", group_id="grp1")
    controller = _DatasetReplayController()
    selected_member = next(iter(controller.member_visibility.options.values()))
    controller.member_visibility.value = [selected_member]
    controller._on_any_control_change()
    monkeypatch.setattr(
        "larvaworld.portal.datasets.dataset_replay_app.LarvaDataset",
        lambda **kwargs: {"dataset_dir": kwargs.get("dir")},
    )

    parameters, _dataset = controller._build_native_replay_parameters(
        selected_member_token=selected_member,
        agent_indices=None,
        time_range=None,
    )

    assert parameters.close_view is False
    assert parameters.fix_point is None
    assert parameters.fix_segment is None
    assert parameters.draw_Nsegs == 2


def test_dataset_replay_controller_native_contour_rendering_passes_no_draw_nsegs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    set_active_workspace_path(workspace.root)
    dataset_dir = workspace.datasets_dir / "imported" / "Schleyer" / "grp1" / "ds1"
    _write_workspace_dataset(dataset_dir, dataset_id="ds1", group_id="grp1")
    controller = _DatasetReplayController()
    selected_member = next(iter(controller.member_visibility.options.values()))
    controller.member_visibility.value = [selected_member]
    controller.native_body_rendering.value = "contour"
    controller._on_any_control_change()
    monkeypatch.setattr(
        "larvaworld.portal.datasets.dataset_replay_app.LarvaDataset",
        lambda **kwargs: {"dataset_dir": kwargs.get("dir")},
    )

    parameters, _dataset = controller._build_native_replay_parameters(
        selected_member_token=selected_member,
        agent_indices=None,
        time_range=None,
    )

    assert parameters.draw_Nsegs is None


def test_dataset_replay_controller_native_body_rendering_presets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    set_active_workspace_path(workspace.root)
    dataset_dir = workspace.datasets_dir / "imported" / "Schleyer" / "grp1" / "ds1"
    _write_workspace_dataset(dataset_dir, dataset_id="ds1", group_id="grp1")
    controller = _DatasetReplayController()
    selected_member = next(iter(controller.member_visibility.options.values()))
    controller.member_visibility.value = [selected_member]
    monkeypatch.setattr(
        "larvaworld.portal.datasets.dataset_replay_app.LarvaDataset",
        lambda **kwargs: {"dataset_dir": kwargs.get("dir")},
    )

    controller.native_body_rendering.value = "midline"
    parameters, _dataset = controller._build_native_replay_parameters(
        selected_member_token=selected_member,
        agent_indices=None,
        time_range=None,
    )
    screen_kws, _video_target = controller._native_replay_screen_kws(selected_member)
    assert parameters.draw_Nsegs is None
    assert screen_kws["draw_contour"] is False
    assert screen_kws["draw_midline"] is True


def test_dataset_replay_controller_native_segment_count_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    set_active_workspace_path(workspace.root)
    dataset_dir = workspace.datasets_dir / "imported" / "Schleyer" / "grp1" / "ds1"
    _write_workspace_dataset(dataset_dir, dataset_id="ds1", group_id="grp1")
    controller = _DatasetReplayController()
    selected_member = next(iter(controller.member_visibility.options.values()))
    prepared_member = controller._prepared.members[selected_member]
    controller._prepared.members[selected_member] = replace(
        prepared_member,
        body_xy_by_point={idx: pd.DataFrame() for idx in range(5)},
    )
    controller.member_visibility.value = [selected_member]
    controller._refresh_native_segment_count_options()
    monkeypatch.setattr(
        "larvaworld.portal.datasets.dataset_replay_app.LarvaDataset",
        lambda **kwargs: {"dataset_dir": kwargs.get("dir")},
    )

    assert controller.native_segment_count.options == {
        "2": 2,
        "All available (4)": 4,
    }
    assert controller.native_segment_count.value == 4

    controller.native_segment_count.value = 2
    parameters, _dataset = controller._build_native_replay_parameters(
        selected_member_token=selected_member,
        agent_indices=None,
        time_range=None,
    )
    assert parameters.draw_Nsegs == 2

    controller.native_segment_count.value = 4
    parameters, _dataset = controller._build_native_replay_parameters(
        selected_member_token=selected_member,
        agent_indices=None,
        time_range=None,
    )
    assert parameters.draw_Nsegs == 4


def test_dataset_replay_controller_browser_render_ignores_pygame_time_range(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    set_active_workspace_path(workspace.root)
    dataset_dir = workspace.datasets_dir / "imported" / "Schleyer" / "grp1" / "ds1"
    _write_workspace_dataset(dataset_dir, dataset_id="ds1", group_id="grp1")
    controller = _DatasetReplayController()
    captured: dict[str, Any] = {}

    def _stub_build_render_state(*args, **kwargs):
        captured.update(kwargs)

        class _Frame:
            tick = 0
            centroids = ()
            heads = ()
            midlines = ()
            trails = ()
            segment_polygons = ()
            body_contours = ()
            colors = ()
            labels = ()

        class _State:
            frame = _Frame()
            rings = ()

        return _State()

    monkeypatch.setattr(
        "larvaworld.portal.datasets.dataset_replay_app.build_render_state",
        _stub_build_render_state,
    )

    controller.use_time_range.value = True
    controller.time_start.value = 30.0
    controller.time_end.value = 40.0
    controller.tick_player.value = 0
    controller._render()

    assert captured["time_range"] is None


def test_dataset_replay_controller_pygame_replay_still_receives_time_range(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    set_active_workspace_path(workspace.root)
    dataset_dir = workspace.datasets_dir / "imported" / "Schleyer" / "grp1" / "ds1"
    _write_workspace_dataset(dataset_dir, dataset_id="ds1", group_id="grp1")
    controller = _DatasetReplayController()
    selected_member = next(iter(controller.member_visibility.options.values()))
    controller.member_visibility.value = [selected_member]
    monkeypatch.setattr(
        "larvaworld.portal.datasets.dataset_replay_app.LarvaDataset",
        lambda **kwargs: {"dataset_dir": kwargs.get("dir")},
    )

    parameters, _dataset = controller._build_native_replay_parameters(
        selected_member_token=selected_member,
        agent_indices=None,
        time_range=(30.0, 40.0),
    )

    assert parameters.time_range == (30.0, 40.0)


def test_dataset_replay_controller_native_close_inspection_gating(
    tmp_path: Path,
) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    set_active_workspace_path(workspace.root)
    ds1 = workspace.datasets_dir / "imported" / "Schleyer" / "grp1" / "ds1"
    ds2 = workspace.datasets_dir / "imported" / "Schleyer" / "grp1" / "ds2"
    _write_workspace_dataset(ds1, dataset_id="ds1", group_id="grp1")
    _write_workspace_dataset(ds2, dataset_id="ds2", group_id="grp1")
    controller = _DatasetReplayController()
    source_token = next(
        token
        for label, token in controller.source_select.options.items()
        if label.startswith("Workspace / Imported group / ")
    )
    controller.source_select.value = source_token
    tokens = list(controller.member_visibility.options.values())

    controller.member_visibility.value = [tokens[0], tokens[1]]
    controller._on_any_control_change()
    assert controller.fix_point.disabled is True
    assert controller.close_view.disabled is True
    assert controller.fix_segment.disabled is True

    controller.member_visibility.value = [tokens[0]]
    controller._on_any_control_change()
    assert controller.fix_point.disabled is False
    assert controller.close_view.disabled is True
    assert controller.fix_segment.disabled is True


def test_dataset_replay_controller_native_close_view_requires_fix_point(
    tmp_path: Path,
) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    set_active_workspace_path(workspace.root)
    dataset_dir = workspace.datasets_dir / "imported" / "Schleyer" / "grp1" / "ds1"
    _write_workspace_dataset(dataset_dir, dataset_id="ds1", group_id="grp1")
    data_h5 = dataset_dir / "data" / "data.h5"
    step = pd.read_hdf(data_h5, "step")
    step["head_x"] = step["x"]
    step["head_y"] = step["y"]
    step.to_hdf(data_h5, key="step")
    controller = _DatasetReplayController()
    selected_member = next(iter(controller.member_visibility.options.values()))
    controller.member_visibility.value = [selected_member]
    controller._on_any_control_change()

    first_fix_option = next(
        value for value in controller.fix_point.options.values() if value is not None
    )
    controller.fix_point.value = first_fix_option
    controller._on_any_control_change()
    controller.close_view.value = True
    assert controller.close_view.disabled is False

    controller.fix_point.value = None
    controller._on_any_control_change()
    assert controller.close_view.disabled is True
    assert controller.close_view.value is False


def test_dataset_replay_controller_native_fix_point_mapping_and_segment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    set_active_workspace_path(workspace.root)
    dataset_dir = workspace.datasets_dir / "imported" / "Schleyer" / "grp1" / "ds1"
    _write_workspace_dataset(dataset_dir, dataset_id="ds1", group_id="grp1")
    data_h5 = dataset_dir / "data" / "data.h5"
    step = pd.read_hdf(data_h5, "step")
    step["head_x"] = step["x"]
    step["head_y"] = step["y"]
    step["point2_x"] = step["x"] + 0.001
    step["point2_y"] = step["y"] + 0.001
    step["tail_x"] = step["x"] + 0.002
    step["tail_y"] = step["y"] + 0.002
    step.to_hdf(data_h5, key="step")

    controller = _DatasetReplayController()
    selected_member = next(iter(controller.member_visibility.options.values()))
    controller.member_visibility.value = [selected_member]
    controller._on_any_control_change()

    body_labels = [k for k in controller.fix_point.options.keys() if k != "None"]
    assert body_labels == ["Body point 1", "Body point 2", "Body point 3"]
    assert all(v is None or int(v) > 0 for v in controller.fix_point.options.values())

    controller.fix_point.value = controller.fix_point.options["Body point 2"]
    controller.transposition.value = "origin"
    controller._on_any_control_change()
    orientation_labels = list(controller.fix_segment.options.keys())
    assert "Front segment" in orientation_labels
    assert "Rear segment" in orientation_labels
    controller.fix_segment.value = "rear"
    monkeypatch.setattr(
        "larvaworld.portal.datasets.dataset_replay_app.LarvaDataset",
        lambda **kwargs: {"dataset_dir": kwargs.get("dir")},
    )

    parameters, _dataset = controller._build_native_replay_parameters(
        selected_member_token=selected_member,
        agent_indices=None,
        time_range=None,
    )
    assert parameters.fix_point == int(controller.fix_point.options["Body point 2"])
    assert parameters.fix_segment == "rear"
    assert parameters.transposition is None


def test_dataset_replay_controller_native_fix_orientation_invalid_state_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    set_active_workspace_path(workspace.root)
    dataset_dir = workspace.datasets_dir / "imported" / "Schleyer" / "grp1" / "ds1"
    _write_workspace_dataset(dataset_dir, dataset_id="ds1", group_id="grp1")
    data_h5 = dataset_dir / "data" / "data.h5"
    step = pd.read_hdf(data_h5, "step")
    step["head_x"] = step["x"]
    step["head_y"] = step["y"]
    step["point2_x"] = step["x"] + 0.001
    step["point2_y"] = step["y"] + 0.001
    step["tail_x"] = step["x"] + 0.002
    step["tail_y"] = step["y"] + 0.002
    step.to_hdf(data_h5, key="step")

    controller = _DatasetReplayController()
    selected_member = next(iter(controller.member_visibility.options.values()))
    controller.member_visibility.value = [selected_member]
    controller._on_any_control_change()
    controller.fix_point.value = controller.fix_point.options["Body point 1"]
    controller._on_any_control_change()
    assert "Front segment" not in controller.fix_segment.options
    controller.fix_segment.value = "front"
    monkeypatch.setattr(
        "larvaworld.portal.datasets.dataset_replay_app.LarvaDataset",
        lambda **kwargs: {"dataset_dir": kwargs.get("dir")},
    )

    with pytest.raises(ValueError, match="Fix orientation is unavailable"):
        controller._build_native_replay_parameters(
            selected_member_token=selected_member,
            agent_indices=None,
            time_range=None,
        )


def test_dataset_replay_controller_native_replay_unavailable_track_point_sets_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    set_active_workspace_path(workspace.root)
    dataset_dir = workspace.datasets_dir / "imported" / "Schleyer" / "grp1" / "ds1"
    _write_workspace_dataset(dataset_dir, dataset_id="ds1", group_id="grp1")
    controller = _DatasetReplayController()
    selected_member = next(iter(controller.member_visibility.options.values()))
    controller.member_visibility.value = [selected_member]
    controller.show_display.value = True
    controller.track_point.value = 99

    called = {"value": False}

    class _FakeReplayRun:
        def __init__(self, **kwargs: Any):
            called["value"] = True

        def run(self) -> None:
            called["value"] = True

    monkeypatch.setattr(
        "larvaworld.portal.datasets.dataset_replay_app.sim.ReplayRun",
        _FakeReplayRun,
    )
    monkeypatch.setattr(
        "larvaworld.portal.datasets.dataset_replay_app.LarvaDataset",
        lambda **kwargs: {"dataset_dir": kwargs.get("dir")},
    )

    controller._on_open_pygame_replay()

    assert (
        "Track point 99 is unavailable for native replay."
        in controller.status_pane.object
    )
    assert called["value"] is False


def test_dataset_replay_controller_native_replay_missing_columns_does_not_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    set_active_workspace_path(workspace.root)
    dataset_dir = workspace.datasets_dir / "imported" / "Schleyer" / "grp1" / "ds1"
    _write_workspace_dataset(dataset_dir, dataset_id="ds1", group_id="grp1")
    controller = _DatasetReplayController()
    selected_member = next(iter(controller.member_visibility.options.values()))
    controller.member_visibility.value = [selected_member]
    controller.show_display.value = True
    controller._prepared.members[selected_member] = replace(
        controller._prepared.members[selected_member],
        native_replay_missing_columns=("front_orientation", "rear_orientation"),
    )
    called = {"value": False}

    class _FakeReplayRun:
        def __init__(self, **kwargs: Any):
            called["value"] = True

        def run(self) -> None:
            called["value"] = True

    monkeypatch.setattr(
        "larvaworld.portal.datasets.dataset_replay_app.sim.ReplayRun",
        _FakeReplayRun,
    )
    monkeypatch.setattr(
        "larvaworld.portal.datasets.dataset_replay_app.LarvaDataset",
        lambda **kwargs: {"dataset_dir": kwargs.get("dir")},
    )

    controller._on_open_pygame_replay()

    assert (
        "Native replay is unavailable for this member: missing "
        "front_orientation, rear_orientation." in controller.status_pane.object
    )
    assert called["value"] is False


def test_dataset_replay_controller_native_replay_invalid_shortcuts_does_not_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    set_active_workspace_path(workspace.root)
    dataset_dir = workspace.datasets_dir / "imported" / "Schleyer" / "grp1" / "ds1"
    _write_workspace_dataset(dataset_dir, dataset_id="ds1", group_id="grp1")
    controller = _DatasetReplayController()
    selected_member = next(iter(controller.member_visibility.options.values()))
    controller.member_visibility.value = [selected_member]
    controller.show_display.value = True

    called = {"value": False}

    class _FakeReplayRun:
        def __init__(self, **kwargs: Any):
            called["value"] = True

        def run(self) -> None:
            called["value"] = True

    monkeypatch.setattr(
        "larvaworld.portal.datasets.dataset_replay_app.sim.ReplayRun",
        _FakeReplayRun,
    )
    monkeypatch.setattr(
        controller.display_shortcuts,
        "validate",
        lambda: ["duplicate key: pause and snapshot"],
    )

    controller._on_open_pygame_replay()

    assert "Display shortcut errors:" in controller.status_pane.object
    assert called["value"] is False


def test_dataset_replay_controller_pygame_replay_workspace_invocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    set_active_workspace_path(workspace.root)
    dataset_dir = workspace.datasets_dir / "imported" / "Schleyer" / "grp1" / "ds1"
    _write_workspace_dataset(dataset_dir, dataset_id="ds1", group_id="grp1")
    controller = _DatasetReplayController()
    selected_member = next(iter(controller.member_visibility.options.values()))
    controller.member_visibility.value = [selected_member]
    controller.show_display.value = True

    captured: dict[str, Any] = {}

    class _FakeScreenManager:
        def __init__(self) -> None:
            self.close_called = False

        def close(self) -> None:
            self.close_called = True

    class _FakeReplayRun:
        def __init__(self, **kwargs: Any):
            captured.update(kwargs)
            self.screen_manager = _FakeScreenManager()

        def run(self) -> None:
            captured["run_called"] = True

    monkeypatch.setattr(
        "larvaworld.portal.datasets.dataset_replay_app.sim.ReplayRun",
        _FakeReplayRun,
    )
    monkeypatch.setattr(
        "larvaworld.portal.datasets.dataset_replay_app.LarvaDataset",
        lambda **kwargs: {"dataset_dir": kwargs.get("dir")},
    )
    monkeypatch.setattr(
        "larvaworld.portal.datasets.dataset_replay_app.pn.state.curdoc",
        None,
        raising=False,
    )

    controller._on_open_pygame_replay()

    assert captured["dataset"] is not None
    assert captured["run_called"] is True
    assert "Native pygame replay finished." in controller.status_pane.object


def test_dataset_replay_controller_pygame_replay_closes_screen_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    set_active_workspace_path(workspace.root)
    dataset_dir = workspace.datasets_dir / "imported" / "Schleyer" / "grp1" / "ds1"
    _write_workspace_dataset(dataset_dir, dataset_id="ds1", group_id="grp1")
    controller = _DatasetReplayController()
    selected_member = next(iter(controller.member_visibility.options.values()))
    controller.member_visibility.value = [selected_member]
    controller.show_display.value = True

    captured: dict[str, Any] = {}

    class _FakeScreenManager:
        def __init__(self) -> None:
            self.close_called = False

        def close(self) -> None:
            self.close_called = True
            captured["close_called"] = True

    class _FailingReplayRun:
        def __init__(self, **kwargs: Any):
            captured.update(kwargs)
            self.screen_manager = _FakeScreenManager()

        def run(self) -> None:
            raise RuntimeError("boom")

    monkeypatch.setattr(
        "larvaworld.portal.datasets.dataset_replay_app.sim.ReplayRun",
        _FailingReplayRun,
    )
    monkeypatch.setattr(
        "larvaworld.portal.datasets.dataset_replay_app.LarvaDataset",
        lambda **kwargs: {"dataset_dir": kwargs.get("dir")},
    )
    monkeypatch.setattr(
        "larvaworld.portal.datasets.dataset_replay_app.pn.state.curdoc",
        None,
        raising=False,
    )

    controller._on_open_pygame_replay()

    assert captured.get("close_called") is True
    assert "Native replay failed: boom" in controller.status_pane.object


def test_dataset_replay_app_returns_viewable() -> None:
    view = dataset_replay_app()
    assert view is not None
