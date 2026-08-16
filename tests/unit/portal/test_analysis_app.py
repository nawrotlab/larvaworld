"""Tests for the Analysis portal app -- real (non-mocked) end-to-end
coverage against the package's bundled DATA_DIR reference dataset, since
`_AnalysisController` relies on `GraphRegistry`/`analysis_helpers`
functions whose failures (registry construction, plot availability,
figure saving/rendering) are all silently swallowed by broad excepts
elsewhere in the call chain.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from larvaworld.lib import reg
from larvaworld.portal.datasets import analysis_app
from larvaworld.portal.workspace import (
    clear_active_workspace_path,
    initialize_workspace,
    set_active_workspace_path,
)


@pytest.fixture(autouse=True)
def workspace_config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LARVAWORLD_PORTAL_CONFIG_DIR", str(tmp_path / "config"))
    clear_active_workspace_path()


def test_analysis_requires_active_workspace() -> None:
    controller = analysis_app._AnalysisController()

    assert controller.workspace is None
    assert controller._all_records == []
    assert "requires an active workspace" in controller.main_content.objects[0].object


def test_analysis_detects_bundled_data_dir_dataset(tmp_path: Path) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    set_active_workspace_path(workspace.root)

    controller = analysis_app._AnalysisController()

    dataset_ids = [record.dataset_id for record in controller._all_records]
    assert "30controls" in dataset_ids


def test_analysis_graph_registry_constructs_and_has_entries(tmp_path: Path) -> None:
    # The real regression: GraphRegistry() previously raised during
    # __init__ (a missing "epochs" graph function, only registered once
    # lib.plot.epochs is actually imported) -- caught silently by
    # _AnalysisController, leaving graph_registry as None and the whole
    # Analysis app inert. Assert it's a real, populated registry, not None.
    workspace = initialize_workspace(tmp_path / "workspace")
    set_active_workspace_path(workspace.root)

    controller = analysis_app._AnalysisController()

    assert controller.graph_registry is not None
    assert controller.graph_registry.exists("epochs")
    assert len(controller.graph_registry.dict) > 10


def test_analysis_refresh_plots_finds_real_valid_plots_for_bundled_dataset(
    tmp_path: Path,
) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    set_active_workspace_path(workspace.root)

    controller = analysis_app._AnalysisController()
    bundled_idx = next(
        i
        for i, record in enumerate(controller._all_records)
        if record.dataset_id == "30controls"
    )
    controller.dataset_table.selection = [bundled_idx]
    controller._on_dataset_selection_change()

    controller._handle_refresh_plots()

    assert controller.plot_group_select.options
    assert controller.plot_function_select.options
    assert "danger" not in controller.status_pane.object
    total_valid = sum(len(v) for v in controller._valid_plots_by_group.values())
    assert total_valid > 0


def test_analysis_run_plot_renders_and_saves_to_workspace_analysis_folder(
    tmp_path: Path,
) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    set_active_workspace_path(workspace.root)

    controller = analysis_app._AnalysisController()
    bundled_idx = next(
        i
        for i, record in enumerate(controller._all_records)
        if record.dataset_id == "30controls"
    )
    controller.dataset_table.selection = [bundled_idx]
    controller._on_dataset_selection_change()
    controller._handle_refresh_plots()
    assert controller.plot_function_select.options

    plot_id = next(iter(controller.plot_function_select.options.values()))
    controller.plot_function_select.value = plot_id

    controller._handle_run_plot()

    assert "danger" not in controller.status_pane.object
    assert controller._current_figure is not None
    # Rendered as an embedded image (plain matplotlib Figure -- see
    # _render_figure's savefig branch), not a bare str(fig) fallback.
    assert "<img" in controller.figure_pane.object

    analysis_dir = workspace.analysis_dir
    saved_files = list(analysis_dir.rglob("*"))
    saved_files = [p for p in saved_files if p.is_file()]
    assert saved_files, f"Expected a saved plot under {analysis_dir}"


def test_analysis_default_ref_dataset_matches_bundled_record(tmp_path: Path) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    set_active_workspace_path(workspace.root)

    controller = analysis_app._AnalysisController()
    bundled = next(
        record
        for record in controller._all_records
        if record.dataset_id == "30controls"
    )
    assert bundled.ref_id == reg.default_refID
    assert bundled.origin == "bundled"
