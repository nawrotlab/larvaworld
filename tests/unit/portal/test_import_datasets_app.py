from __future__ import annotations

from html import unescape
from pathlib import Path
from types import SimpleNamespace

import panel as pn
import pytest

from larvaworld.portal.datasets import import_datasets_app
from larvaworld.portal.datasets.discovery import RawDatasetCandidate
from larvaworld.portal.datasets.models import WorkspaceDatasetRecord
from larvaworld.portal.workspace import (
    clear_active_workspace_path,
    initialize_workspace,
    set_active_workspace_path,
)


@pytest.fixture(autouse=True)
def workspace_config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LARVAWORLD_PORTAL_CONFIG_DIR", str(tmp_path / "config"))
    clear_active_workspace_path()


def _record(path: Path) -> WorkspaceDatasetRecord:
    return WorkspaceDatasetRecord(
        dataset_id=path.name,
        dataset_dir=path,
        data_dir=path / "data",
        conf_path=path / "data" / "conf.txt",
        h5_path=path / "data" / "data.h5",
        lab_id="Schleyer",
        group_id="exploration",
        ref_id=None,
        n_agents=12,
    )


def _section_widgets(section: pn.viewable.Viewable) -> dict[str, pn.widgets.Widget]:
    return {
        widget.name: widget
        for widget in section.select(pn.widgets.Widget)
        if getattr(widget, "name", None)
    }


def _contains_widget(section: pn.viewable.Viewable, widget_name: str) -> bool:
    return widget_name in _section_widgets(section)


def _is_config_family_section(section: pn.viewable.Viewable) -> bool:
    css_classes = getattr(section, "css_classes", None) or []
    return "lw-import-datasets-config-family" in css_classes


def _find_section_with_widget(
    controller: import_datasets_app._ImportDatasetsController, widget_name: str
) -> pn.viewable.Viewable:
    def _descend(section: pn.viewable.Viewable) -> pn.viewable.Viewable | None:
        children = getattr(section, "objects", None) or []
        for child in children:
            if hasattr(child, "select") and _contains_widget(child, widget_name):
                nested = _descend(child)
                if nested is not None:
                    return nested
                if _is_config_family_section(child):
                    return child
        return None

    for section in controller.lab_editor_sections.objects:
        if _contains_widget(section, widget_name):
            nested = _descend(section)
            if nested is not None:
                return nested
            if _is_config_family_section(section):
                return section
            return section
    raise AssertionError(f"Could not find section containing widget {widget_name!r}")


def _find_widget(
    controller: import_datasets_app._ImportDatasetsController, widget_name: str
) -> pn.widgets.Widget:
    section = _find_section_with_widget(controller, widget_name)
    return _section_widgets(section)[widget_name]


def _scape_switches(section: pn.viewable.Viewable) -> list[pn.widgets.Switch]:
    return [widget for widget in section.select(pn.widgets.Switch)]


def _card_titles(section: pn.viewable.Viewable) -> set[str]:
    return {
        card.title
        for card in section.select(pn.Card)
        if isinstance(getattr(card, "title", None), str)
    }


def test_import_datasets_controller_requires_active_workspace() -> None:
    controller = import_datasets_app._ImportDatasetsController()

    assert controller.discover_button.disabled is True
    assert controller.import_button.disabled is True
    assert controller.merged_checkbox.disabled is True
    assert "Configure an active workspace" in controller.status.object


def test_import_datasets_lab_config_panel_loads_selected_configuration() -> None:
    controller = import_datasets_app._ImportDatasetsController()

    assert isinstance(
        controller.lab_actions,
        import_datasets_app.ConftypeActionsController,
    )
    assert controller.lab_config_name_input.value == controller.lab_select.value
    assert len(controller.lab_editor_sections.objects) == 3
    assert "Loaded LabFormat" in controller.lab_status.object


def test_import_datasets_view_uses_three_column_layout_with_environment_below() -> None:
    controller = import_datasets_app._ImportDatasetsController()
    view = controller.view()

    top_row = next(
        section
        for section in view.objects
        if isinstance(section, pn.Row) and len(getattr(section, "objects", [])) == 3
    )
    top_columns = top_row.objects
    col_one = top_columns[0]
    col_two = top_columns[1]
    col_three = top_columns[2]

    assert _contains_widget(col_one, "Lab format")
    assert _contains_widget(col_one, "Configuration ID")
    assert _contains_widget(col_one, "Raw root")
    assert _contains_widget(col_one, "Dataset ID")
    assert _contains_widget(col_one, "Group ID override")
    assert _contains_widget(col_one, "Import into workspace")
    assert controller.raw_root_input.sizing_mode == "stretch_width"
    assert controller.candidate_select.sizing_mode == "stretch_width"
    raw_root_row = next(
        section
        for section in col_one.select(pn.Row)
        if _contains_widget(section, "Raw root")
    )
    source_action_row = next(
        section
        for section in col_one.select(pn.Row)
        if _contains_widget(section, "Browse")
        and _contains_widget(section, "Discover datasets")
    )
    assert source_action_row is not raw_root_row
    assert not _contains_widget(raw_root_row, "Browse")
    candidate_row = next(
        section
        for section in col_one.select(pn.Row)
        if _contains_widget(section, "Candidate")
    )
    merged_row = next(
        section
        for section in col_one.select(pn.Row)
        if _contains_widget(section, "Merged")
    )
    assert merged_row is not candidate_row
    assert not _contains_widget(candidate_row, "Merged")
    dataset_id_row = next(
        section
        for section in col_one.select(pn.Column)
        if controller.dataset_id_input in (getattr(section, "objects", []) or [])
    )
    assert not _contains_widget(dataset_id_row, "Group ID override")
    html_panes = list(col_one.select(pn.pane.HTML))
    assert controller.status in html_panes
    assert controller.candidate_summary not in html_panes
    assert controller.lab_status not in html_panes
    import_section = col_one.objects[1]
    import_section_objects = list(getattr(import_section, "objects", ()))
    assert controller.status in import_section_objects
    assert controller.workspace_summary in import_section_objects
    assert import_section_objects.index(
        controller.status
    ) < import_section_objects.index(controller.workspace_summary)

    lab_source_section = col_one.objects[0]
    lab_source_objects = list(getattr(lab_source_section, "objects", ()))
    lab_config_controls = next(
        section
        for section in lab_source_objects
        if _contains_widget(section, "Lab format")
        and _contains_widget(section, "Configuration ID")
    )
    lab_actions_row = next(
        section
        for section in lab_source_objects
        if _contains_widget(section, "Load")
        and _contains_widget(section, "Save")
        and _contains_widget(section, "Delete")
        and _contains_widget(section, "Reset configurations")
    )
    assert lab_source_objects.index(lab_config_controls) < lab_source_objects.index(
        lab_actions_row
    )
    assert lab_source_objects.index(lab_actions_row) < lab_source_objects.index(
        raw_root_row
    )

    lab_actions_widgets = controller.lab_actions.view.select(pn.widgets.Button)
    lab_actions_names = {widget.name for widget in lab_actions_widgets}
    source_section_widgets = _section_widgets(col_one)
    assert lab_actions_names.issubset(source_section_widgets)

    assert _contains_widget(col_two, "Front vector")
    assert _contains_widget(col_two, "framerate")

    col_three_widget_names = set(_section_widgets(col_three))
    assert {"labID", "Lab ID", "LabID"} & col_three_widget_names
    assert {"File pref", "Folder pref", "File sep"} & col_three_widget_names
    assert {"Rescale by", "rescale_by"} & col_three_widget_names

    assert len(controller.lab_editor_sections.objects) == 3
    environment_section = controller.lab_editor_sections.objects[2]
    assert environment_section in view.objects
    assert environment_section not in top_columns
    assert _contains_widget(environment_section, "Arena width")
    environment_row = next(
        section
        for section in environment_section.select(pn.Row)
        if len(getattr(section, "objects", [])) == 3
    )
    env_left_col, env_middle_col, env_right_col = environment_row.objects
    assert _contains_widget(env_left_col, "Arena width")
    assert not _contains_widget(env_left_col, "Enable Food grid")
    assert _contains_widget(env_middle_col, "Enable Food grid")
    assert not _contains_widget(env_middle_col, "New border ID")
    assert _contains_widget(env_right_col, "New border ID")
    assert "Border list" in _card_titles(env_right_col)
    assert "Sensory landscapes" in _card_titles(env_right_col)
    assert len(_scape_switches(env_right_col)) >= 1


def test_import_datasets_environment_panel_uses_typed_env_helpers() -> None:
    controller = import_datasets_app._ImportDatasetsController()
    environment_section = _find_section_with_widget(controller, "Arena width")

    assert _find_section_with_widget(controller, "Arena width") is environment_section
    assert (
        _find_section_with_widget(controller, "Enable Food grid") is environment_section
    )
    assert _find_section_with_widget(controller, "New border ID") is environment_section
    assert len(_scape_switches(environment_section)) == 3


def test_import_datasets_environment_panel_updates_working_lab() -> None:
    controller = import_datasets_app._ImportDatasetsController()
    environment_section = _find_section_with_widget(controller, "Arena width")
    env_conf = controller._working_lab.env_params

    width_widget = _section_widgets(environment_section)["Arena width"]
    height_widget = _section_widgets(environment_section)["Arena height"]
    enable_grid = _section_widgets(environment_section)["Enable Food grid"]
    enable_odorscape = _scape_switches(environment_section)[0]
    new_border_id = _section_widgets(environment_section)["New border ID"]
    add_border = _section_widgets(environment_section)["Add border"]

    width_widget.value = 0.27
    height_widget.value = 0.19
    enable_grid.value = True
    enable_odorscape.value = True
    new_border_id.value = "import_border"
    add_border.clicks += 1

    assert env_conf.arena.dims == pytest.approx((0.27, 0.19))
    assert env_conf.food_params.food_grid is not None
    assert env_conf.odorscape is not None
    assert "import_border" in env_conf.border_list


def test_import_datasets_tracker_panel_uses_safe_numeric_widgets() -> None:
    controller = import_datasets_app._ImportDatasetsController()
    metric_section = _find_section_with_widget(controller, "Front vector")
    framerate_section = _find_section_with_widget(controller, "framerate")

    float_inputs = {
        widget.name for widget in framerate_section.select(pn.widgets.FloatInput)
    }
    int_inputs = {widget.name for widget in metric_section.select(pn.widgets.IntInput)}

    assert "framerate" in float_inputs
    assert "timestep" in float_inputs
    assert "# midline 2D points" in int_inputs
    assert "# contour 2D points" in int_inputs


def test_import_datasets_tracker_vector_sliders_follow_points_and_bend() -> None:
    controller = import_datasets_app._ImportDatasetsController()
    tracker = controller._working_lab.tracker
    tracker_section = _find_section_with_widget(controller, "Front vector")
    widgets = _section_widgets(tracker_section)
    front_slider = widgets["Front vector"]
    rear_slider = widgets["Rear vector"]

    tracker.Npoints = 6
    tracker.front_vector = (1, 3)
    tracker.rear_vector = (-2, -1)
    assert front_slider.start == 1
    assert front_slider.end == 6
    assert front_slider.value == (1, 3)
    assert rear_slider.start == -6
    assert rear_slider.end == -1
    assert rear_slider.value == (-2, -1)

    tracker.Npoints = 2
    assert front_slider.end == 2
    assert front_slider.value == (1, 2)
    assert rear_slider.start == -2
    assert rear_slider.end == -1
    assert rear_slider.value == (-2, -1)

    tracker.bend = "from_angles"
    assert front_slider.disabled is True
    assert rear_slider.disabled is True

    tracker.Npoints = 0
    assert tracker.front_vector is None
    assert tracker.rear_vector is None
    assert front_slider.disabled is True
    assert rear_slider.disabled is True

    tracker.Npoints = 5
    tracker.bend = "from_vectors"
    assert tracker.front_vector == (1, 2)
    assert tracker.rear_vector == (-2, -1)
    assert front_slider.end == 5
    assert rear_slider.start == -5

    tracker.bend = "from_vectors"
    assert front_slider.disabled is False
    assert rear_slider.disabled is False


def test_import_datasets_tracker_panel_surfaces_param_docs() -> None:
    controller = import_datasets_app._ImportDatasetsController()
    tracker_section = _find_section_with_widget(controller, "Front vector")
    docs = unescape(
        " ".join(
            pane.object
            for pane in tracker_section.select(pn.pane.HTML)
            if isinstance(pane.object, str)
        )
    )

    assert "The initial & final segment of the front body vector." in docs
    assert "The initial & final segment of the rear body vector." in docs
    assert "Whether to use the component velocity" in docs
    assert "Whether bending angle is computed" not in docs


def test_import_datasets_tracker_places_bend_above_vector_sliders() -> None:
    controller = import_datasets_app._ImportDatasetsController()
    tracker_section = _find_section_with_widget(controller, "Front vector")
    widget_names = [
        widget.name
        for widget in tracker_section.select(pn.widgets.Widget)
        if getattr(widget, "name", None)
    ]

    assert widget_names == [
        "XY unit",
        "# midline 2D points",
        "# contour 2D points",
        "Point idx",
        "Bend",
        "Front vector",
        "Rear vector",
        "Front body ratio",
        "Use component vel",
    ]


def test_import_datasets_tracker_framerate_panel_is_separate() -> None:
    controller = import_datasets_app._ImportDatasetsController()
    framerate_section = _find_section_with_widget(controller, "framerate")

    assert list(_section_widgets(framerate_section)) == [
        "framerate",
        "timestep",
        "Constant framerate",
    ]


def test_import_datasets_merged_checkbox_builds_schleyer_merge_target_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    set_active_workspace_path(workspace.root)
    raw_root = tmp_path / "raw"

    controller = import_datasets_app._ImportDatasetsController()
    controller.lab_select.value = "Schleyer"
    controller.raw_root_input.value = str(raw_root)

    candidates = [
        RawDatasetCandidate(
            candidate_id="box01",
            parent_dir="exploration/box01",
            display_name="exploration/box01",
            source_path=raw_root / "exploration" / "box01",
            warnings=[],
        ),
        RawDatasetCandidate(
            candidate_id="box02",
            parent_dir="exploration/box02",
            display_name="exploration/box02",
            source_path=raw_root / "exploration" / "box02",
            warnings=[],
        ),
    ]
    monkeypatch.setattr(
        import_datasets_app,
        "discover_raw_datasets",
        lambda _lab_id, _raw_root: candidates,
    )
    monkeypatch.setattr(
        import_datasets_app,
        "_candidate_import_overrides",
        lambda _lab_id, _raw_root, _candidate: (_ for _ in ()).throw(
            AssertionError("merged imports must not use candidate overrides")
        ),
    )

    controller._handle_discover()
    controller.merged_checkbox.value = True
    merge_key = next(
        value
        for label, value in controller.candidate_select.options.items()
        if label == "exploration (2 datasets)"
    )
    controller.candidate_select.value = merge_key

    request = controller._build_import_request()

    assert controller.candidate_select.disabled is False
    assert controller.candidate_select.name == "Merge target"
    assert request.parent_dir == "exploration"
    assert request.dataset_id == "exploration"
    assert request.merged is True
    assert request.extra_kwargs == {}
    assert (
        "Child candidates</strong>: 2 datasets" in controller.candidate_summary.object
    )


def test_import_datasets_merged_checkbox_rejects_jovanic_source_id_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    set_active_workspace_path(workspace.root)
    raw_root = tmp_path / "raw"
    day1 = raw_root / "day1"

    candidates = [
        RawDatasetCandidate(
            candidate_id="dishA",
            parent_dir="day1",
            display_name="day1 / dishA",
            source_path=day1,
            warnings=[],
        ),
        RawDatasetCandidate(
            candidate_id="dishB",
            parent_dir="day1",
            display_name="day1 / dishB",
            source_path=day1,
            warnings=[],
        ),
    ]
    monkeypatch.setattr(
        import_datasets_app,
        "discover_raw_datasets",
        lambda _lab_id, _raw_root: candidates,
    )

    controller = import_datasets_app._ImportDatasetsController()
    controller.lab_select.value = "Jovanic"
    controller.raw_root_input.value = str(raw_root)
    controller._handle_discover()

    assert controller.merged_checkbox.disabled is True
    assert controller.candidate_select.name == "Candidate"
    assert "day1 / dishA" in controller.candidate_select.options
    assert "day1 (2 datasets)" not in controller.candidate_select.options

    controller.merged_checkbox.value = True

    assert controller.merged_checkbox.value is False
    assert "Merged import is not supported" in controller.status.object
    assert controller.candidate_select.name == "Candidate"


def test_import_datasets_merged_toggle_rebuilds_selector_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    set_active_workspace_path(workspace.root)
    raw_root = tmp_path / "raw"
    candidates = [
        RawDatasetCandidate(
            candidate_id="box01",
            parent_dir="exploration/box01",
            display_name="exploration/box01",
            source_path=raw_root / "exploration" / "box01",
            warnings=[],
        ),
        RawDatasetCandidate(
            candidate_id="box02",
            parent_dir="exploration/box02",
            display_name="exploration/box02",
            source_path=raw_root / "exploration" / "box02",
            warnings=[],
        ),
    ]
    monkeypatch.setattr(
        import_datasets_app,
        "discover_raw_datasets",
        lambda _lab_id, _raw_root: candidates,
    )

    controller = import_datasets_app._ImportDatasetsController()
    controller.lab_select.value = "Schleyer"
    controller.raw_root_input.value = str(raw_root)
    controller._handle_discover()
    candidate_key = next(
        value
        for label, value in controller.candidate_select.options.items()
        if label == "exploration/box01"
    )
    controller.candidate_select.value = candidate_key

    assert controller.candidate_select.name == "Candidate"
    assert controller.dataset_id_input.value == "box01"

    controller.merged_checkbox.value = True

    assert controller.candidate_select.value == ""
    assert controller.dataset_id_input.value == ""
    assert controller.candidate_select.name == "Merge target"
    assert "exploration (2 datasets)" in controller.candidate_select.options
    assert "exploration/box01" not in controller.candidate_select.options

    controller.merged_checkbox.value = False

    assert controller.candidate_select.value == ""
    assert controller.dataset_id_input.value == ""
    assert controller.candidate_select.name == "Candidate"
    assert "exploration/box01" in controller.candidate_select.options
    assert "exploration/box02" in controller.candidate_select.options


def test_import_datasets_lab_config_save_and_delete_use_registry_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = import_datasets_app._ImportDatasetsController()
    saved: list[tuple[str, object]] = []
    deleted: list[str] = []
    original_lab_id = controller._working_lab.labID
    controller._working_lab.labID = "EditedLab"

    monkeypatch.setattr(
        import_datasets_app.reg.conf.LabFormat,
        "setID",
        lambda config_id, conf: saved.append((config_id, conf)),
    )
    monkeypatch.setattr(
        import_datasets_app.reg.conf.LabFormat,
        "delete",
        lambda config_id: deleted.append(config_id),
    )
    monkeypatch.setattr(controller.lab_actions, "refresh_registry", lambda: None)
    controller.lab_actions.on_save = None
    controller.lab_actions.on_delete = None

    controller.lab_config_name_input.value = "LabCopy"
    controller._handle_lab_save()
    controller._handle_lab_delete()

    assert saved[0][0] == "LabCopy"
    assert saved[0][1].labID == "LabCopy"
    assert original_lab_id != "EditedLab"
    assert deleted == [controller.lab_select.value]
    assert (
        "saved to the registry" in controller.lab_status.object
        or "deleted from the registry" in controller.lab_status.object
    )


def test_import_datasets_lab_config_reset_recreates_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = import_datasets_app._ImportDatasetsController()
    selected_lab_id = controller.lab_select.value
    reset_calls: list[dict[str, object]] = []
    refreshed: list[dict[str, object]] = []
    loaded: list[str | None] = []

    monkeypatch.setattr(
        import_datasets_app.reg.conf.LabFormat,
        "reset",
        lambda **kwargs: reset_calls.append(kwargs),
    )
    monkeypatch.setattr(
        controller.lab_actions,
        "refresh_registry",
        lambda: refreshed.append({}),
    )
    controller.lab_actions.on_reset = lambda lab_id: loaded.append(lab_id)

    controller._handle_lab_reset()

    assert reset_calls == [{"recreate": True}]
    assert refreshed == [{}]
    assert loaded == [selected_lab_id]
    assert "registry recreated" in controller.lab_status.object


def test_import_datasets_controller_discovers_candidates_and_enables_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    set_active_workspace_path(workspace.root)
    raw_root = tmp_path / "raw"
    candidate = RawDatasetCandidate(
        candidate_id="dish01",
        parent_dir="exploration/dish01",
        display_name="exploration/dish01",
        source_path=raw_root / "exploration" / "dish01",
        warnings=[],
    )
    monkeypatch.setattr(
        import_datasets_app,
        "discover_raw_datasets",
        lambda _lab_id, _raw_root: [candidate],
    )

    controller = import_datasets_app._ImportDatasetsController()

    assert controller.discover_button.disabled is True
    controller.raw_root_input.value = str(raw_root)
    assert controller.discover_button.disabled is False

    controller._handle_discover()

    assert "Discovered 1 candidate" in controller.status.object
    assert controller.candidate_select.disabled is False
    option_values = [value for value in controller.candidate_select.options.values()]
    candidate_key = next(value for value in option_values if value)
    controller.candidate_select.value = candidate_key

    assert controller.import_button.disabled is False
    assert controller.dataset_id_input.value == "dish01"
    assert "exploration/dish01" in controller.candidate_summary.object


def test_import_datasets_discover_failure_sets_danger_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    set_active_workspace_path(workspace.root)
    raw_root = tmp_path / "raw"
    raw_root.mkdir()

    monkeypatch.setattr(
        import_datasets_app,
        "discover_raw_datasets",
        lambda _lab_id, _raw_root: (_ for _ in ()).throw(RuntimeError("scan failed")),
    )

    controller = import_datasets_app._ImportDatasetsController()
    controller.raw_root_input.value = str(raw_root)
    controller._handle_discover()

    assert "Discovery failed: scan failed" in controller.status.object
    assert controller.discover_button.disabled is False


def test_import_datasets_import_shows_running_state_before_deferred_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from unittest.mock import PropertyMock

    workspace = initialize_workspace(tmp_path / "workspace")
    set_active_workspace_path(workspace.root)
    raw_root = tmp_path / "raw"
    candidate = RawDatasetCandidate(
        candidate_id="dish01",
        parent_dir="exploration/dish01",
        display_name="exploration/dish01",
        source_path=raw_root / "exploration" / "dish01",
        warnings=[],
    )
    seen_requests: list[tuple[object, object]] = []
    record = _record(
        workspace.datasets_dir / "imported" / "Schleyer" / "exploration" / "dish01"
    )

    monkeypatch.setattr(
        import_datasets_app,
        "discover_raw_datasets",
        lambda _lab_id, _raw_root: [candidate],
    )
    monkeypatch.setattr(
        import_datasets_app,
        "_candidate_import_overrides",
        lambda _lab_id, _raw_root, _candidate: {},
    )
    monkeypatch.setattr(
        import_datasets_app,
        "import_into_workspace",
        lambda request, workspace=None: seen_requests.append((request, workspace))
        or record,
    )

    class _DummyCurdoc:
        def __init__(self) -> None:
            self.pending: list[object] = []
            self.session_context = SimpleNamespace(
                request=SimpleNamespace(arguments={})
            )

        def add_next_tick_callback(self, callback: object) -> None:
            self.pending.append(callback)

    dummy = _DummyCurdoc()
    state_obj = import_datasets_app.pn.state
    monkeypatch.setattr(
        type(state_obj),
        "curdoc",
        PropertyMock(return_value=dummy),
    )

    controller = import_datasets_app._ImportDatasetsController()
    controller.raw_root_input.value = str(raw_root)
    controller._handle_discover()
    candidate_key = next(
        value for value in controller.candidate_select.options.values() if value
    )
    controller.candidate_select.value = candidate_key

    controller._handle_import()

    assert seen_requests == []
    assert "Importing dataset into the active workspace" in controller.status.object
    assert controller.import_button.disabled is True
    assert len(dummy.pending) == 1
    dummy.pending[0]()
    assert len(seen_requests) == 1
    assert "imported into the active workspace" in controller.status.object
    assert controller.import_button.disabled is False


def test_import_datasets_browse_raw_root_clears_existing_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    set_active_workspace_path(workspace.root)
    original_root = tmp_path / "raw"
    new_root = tmp_path / "raw-next"
    candidate = RawDatasetCandidate(
        candidate_id="dish01",
        parent_dir="exploration/dish01",
        display_name="exploration/dish01",
        source_path=original_root / "exploration" / "dish01",
        warnings=[],
    )
    monkeypatch.setattr(
        import_datasets_app,
        "discover_raw_datasets",
        lambda _lab_id, _raw_root: [candidate],
    )
    monkeypatch.setattr(
        import_datasets_app,
        "pick_directory",
        lambda *args, **kwargs: (new_root, None),
    )

    controller = import_datasets_app._ImportDatasetsController()
    controller.raw_root_input.value = str(original_root)
    controller._handle_discover()
    candidate_key = next(
        value for value in controller.candidate_select.options.values() if value
    )
    controller.candidate_select.value = candidate_key

    assert controller.import_button.disabled is False

    controller._handle_browse_raw_root()

    assert controller.raw_root_input.value == str(new_root)
    assert controller.candidate_select.disabled is True
    assert controller.candidate_select.value == ""
    assert controller.dataset_id_input.value == ""
    assert "Source changed." in controller.status.object


def test_import_datasets_browse_raw_root_cancel_reports_cancel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    set_active_workspace_path(workspace.root)
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    monkeypatch.setattr(
        import_datasets_app,
        "pick_directory",
        lambda *args, **kwargs: (None, None),
    )

    controller = import_datasets_app._ImportDatasetsController()
    controller.raw_root_input.value = str(raw_root)

    controller._handle_browse_raw_root()

    assert controller.raw_root_input.value == str(raw_root)
    assert "Browse cancelled" in controller.status.object


def test_import_datasets_browse_raw_root_surfaces_picker_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    set_active_workspace_path(workspace.root)
    monkeypatch.setattr(
        import_datasets_app,
        "pick_directory",
        lambda *args, **kwargs: (
            None,
            "No folder picker is available in this environment.",
        ),
    )

    controller = import_datasets_app._ImportDatasetsController()

    controller._handle_browse_raw_root()

    assert (
        "No folder picker is available in this environment." in controller.status.object
    )


def test_import_datasets_controller_builds_request_and_reports_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    set_active_workspace_path(workspace.root)
    raw_root = tmp_path / "raw"
    candidate = RawDatasetCandidate(
        candidate_id="dish01",
        parent_dir="exploration/dish01",
        display_name="exploration/dish01",
        source_path=raw_root / "exploration" / "dish01",
        warnings=[],
    )
    seen_requests = []
    record = _record(
        workspace.datasets_dir / "imported" / "Schleyer" / "exploration" / "dish01"
    )
    monkeypatch.setattr(
        import_datasets_app,
        "discover_raw_datasets",
        lambda _lab_id, _raw_root: [candidate],
    )
    monkeypatch.setattr(
        import_datasets_app,
        "_candidate_import_overrides",
        lambda _lab_id, _raw_root, _candidate: {},
    )
    monkeypatch.setattr(
        import_datasets_app,
        "import_into_workspace",
        lambda request, workspace=None: seen_requests.append((request, workspace))
        or record,
    )

    controller = import_datasets_app._ImportDatasetsController()
    controller.lab_select.value = "Schleyer"
    controller.raw_root_input.value = str(raw_root)
    controller._handle_discover()
    candidate_key = next(
        value for value in controller.candidate_select.options.values() if value
    )
    controller.candidate_select.value = candidate_key
    controller.group_id_input.value = "exploration"
    controller.color_input.value = "blue"

    controller._handle_import()

    request, resolved_workspace = seen_requests[0]
    assert resolved_workspace == workspace
    assert request.lab_id == "Schleyer"
    assert request.parent_dir == "exploration/dish01"
    assert request.raw_folder == raw_root
    assert request.dataset_id == "dish01"
    assert request.group_id == "exploration"
    assert request.color == "blue"
    assert request.extra_kwargs == {}
    assert "imported into the active workspace" in controller.status.object
    assert str(record.dataset_dir) in controller.status.object


def test_import_datasets_controller_surfaces_adapter_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    set_active_workspace_path(workspace.root)
    raw_root = tmp_path / "raw"
    candidate = RawDatasetCandidate(
        candidate_id="dish01",
        parent_dir="exploration/dish01",
        display_name="exploration/dish01",
        source_path=raw_root / "exploration" / "dish01",
        warnings=[],
    )
    monkeypatch.setattr(
        import_datasets_app,
        "discover_raw_datasets",
        lambda _lab_id, _raw_root: [candidate],
    )
    monkeypatch.setattr(
        import_datasets_app,
        "_candidate_import_overrides",
        lambda _lab_id, _raw_root, _candidate: {},
    )
    monkeypatch.setattr(
        import_datasets_app,
        "import_into_workspace",
        lambda _request, workspace=None: (_ for _ in ()).throw(
            RuntimeError("Import failed: backend returned no dataset")
        ),
    )

    controller = import_datasets_app._ImportDatasetsController()
    controller.raw_root_input.value = str(raw_root)
    controller._handle_discover()
    candidate_key = next(
        value for value in controller.candidate_select.options.values() if value
    )
    controller.candidate_select.value = candidate_key

    controller._handle_import()

    assert "Import failed: backend returned no dataset" in controller.status.object
