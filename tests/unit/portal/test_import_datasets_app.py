from __future__ import annotations

from html import unescape
from pathlib import Path

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


def _find_section_with_widget(
    controller: import_datasets_app._ImportDatasetsController, widget_name: str
) -> pn.viewable.Viewable:
    for section in controller.lab_editor_sections.objects:
        widgets = _section_widgets(section)
        if widget_name in widgets:
            return section
    raise AssertionError(f"Could not find section containing widget {widget_name!r}")


def test_import_datasets_controller_requires_active_workspace() -> None:
    controller = import_datasets_app._ImportDatasetsController()

    assert controller.discover_button.disabled is True
    assert controller.import_button.disabled is True
    assert "Configure an active workspace" in controller.status.object


def test_import_datasets_lab_config_panel_loads_selected_configuration() -> None:
    controller = import_datasets_app._ImportDatasetsController()

    assert controller.lab_config_name_input.value == controller.lab_select.value
    assert len(controller.lab_editor_sections.objects) == 6
    assert "Loaded LabFormat" in controller.lab_status.object


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
    monkeypatch.setattr(controller, "_refresh_lab_options", lambda **kwargs: None)
    monkeypatch.setattr(controller, "_load_working_lab", lambda _lab_id: None)

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


def test_import_datasets_browse_raw_root_cancel_is_silent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    set_active_workspace_path(workspace.root)
    raw_root = tmp_path / "raw"
    monkeypatch.setattr(
        import_datasets_app,
        "pick_directory",
        lambda *args, **kwargs: (None, None),
    )

    controller = import_datasets_app._ImportDatasetsController()
    controller.raw_root_input.value = str(raw_root)
    initial_status = controller.status.object

    controller._handle_browse_raw_root()

    assert controller.raw_root_input.value == str(raw_root)
    assert controller.status.object == initial_status


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
