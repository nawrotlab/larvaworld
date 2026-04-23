from __future__ import annotations

import json
from pathlib import Path

import pytest

from larvaworld.portal.datasets import dataset_manager_app
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


def _write_dataset(
    workspace,
    *,
    lab_id: str = "Schleyer",
    group_id: str = "exploration",
    dataset_slug: str = "dish01",
    dataset_id: str | None = None,
    ref_id: str | None = None,
    n_agents: int | None = 12,
) -> WorkspaceDatasetRecord:
    dataset_dir = workspace.datasets_dir / "imported" / lab_id / group_id / dataset_slug
    data_dir = dataset_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "id": dataset_id or dataset_slug,
        "dir": str(dataset_dir),
        "refID": ref_id,
        "group_id": group_id,
        "N": n_agents,
        "larva_group": {"group_id": group_id},
    }
    (data_dir / "conf.txt").write_text(json.dumps(payload), encoding="utf-8")
    (data_dir / "data.h5").write_bytes(b"placeholder")
    return WorkspaceDatasetRecord(
        dataset_id=(dataset_id or dataset_slug),
        dataset_dir=dataset_dir.resolve(),
        data_dir=data_dir.resolve(),
        conf_path=(data_dir / "conf.txt").resolve(),
        h5_path=(data_dir / "data.h5").resolve(),
        lab_id=lab_id,
        group_id=group_id,
        ref_id=ref_id,
        n_agents=n_agents,
    )


def _select_first_row(
    controller: dataset_manager_app._DatasetManagerController,
) -> None:
    controller.table.selection = [0]
    controller._on_table_selection_change()


def test_dataset_manager_requires_active_workspace() -> None:
    controller = dataset_manager_app._DatasetManagerController()

    assert "requires an active workspace" in controller.empty_state.object
    assert controller._all_records == []


def test_dataset_manager_empty_state_points_to_import_app(tmp_path: Path) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    set_active_workspace_path(workspace.root)

    controller = dataset_manager_app._DatasetManagerController()

    assert (
        "No imported datasets found in this workspace" in controller.empty_state.object
    )
    assert "/wf.open_dataset" in controller.empty_state.object


def test_dataset_manager_renders_records_from_workspace_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    set_active_workspace_path(workspace.root)
    alpha = _write_dataset(workspace, dataset_slug="alpha", dataset_id="alpha")
    monkeypatch.setattr(
        dataset_manager_app, "list_workspace_datasets", lambda workspace=None: [alpha]
    )

    controller = dataset_manager_app._DatasetManagerController()

    assert controller.table.value.iloc[0]["Dataset ID"] == "alpha"
    assert (
        controller.table.value.iloc[0]["Location"]
        == "imported/Schleyer/exploration/alpha"
    )


def test_dataset_manager_search_filters_by_dataset_group_and_ref_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    set_active_workspace_path(workspace.root)
    alpha = _write_dataset(
        workspace,
        dataset_slug="alpha",
        dataset_id="alpha_dataset",
        group_id="controls",
        ref_id="ref.alpha",
    )
    beta = _write_dataset(
        workspace,
        dataset_slug="beta",
        dataset_id="beta_dataset",
        group_id="treated",
        ref_id="ref.beta",
    )
    monkeypatch.setattr(
        dataset_manager_app,
        "list_workspace_datasets",
        lambda workspace=None: [alpha, beta],
    )

    controller = dataset_manager_app._DatasetManagerController()

    controller.search_input.value = "controls"
    assert [record.dataset_id for record in controller._filtered_records] == [
        "alpha_dataset"
    ]

    controller.search_input.value = "ref.beta"
    assert [record.dataset_id for record in controller._filtered_records] == [
        "beta_dataset"
    ]

    controller.search_input.value = "alpha_dataset"
    assert [record.dataset_id for record in controller._filtered_records] == [
        "alpha_dataset"
    ]


def test_dataset_manager_lab_filter_narrows_catalog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    set_active_workspace_path(workspace.root)
    alpha = _write_dataset(workspace, lab_id="Schleyer", dataset_slug="alpha")
    beta = _write_dataset(workspace, lab_id="Arguello", dataset_slug="beta")
    monkeypatch.setattr(
        dataset_manager_app,
        "list_workspace_datasets",
        lambda workspace=None: [alpha, beta],
    )

    controller = dataset_manager_app._DatasetManagerController()
    controller.lab_filter.value = "Arguello"

    assert [record.dataset_id for record in controller._filtered_records] == ["beta"]


def test_dataset_manager_selection_populates_details_and_actions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    set_active_workspace_path(workspace.root)
    record = _write_dataset(workspace, dataset_slug="alpha", ref_id="ref.alpha")
    monkeypatch.setattr(
        dataset_manager_app, "list_workspace_datasets", lambda workspace=None: [record]
    )

    controller = dataset_manager_app._DatasetManagerController()
    _select_first_row(controller)

    assert str(record.dataset_dir) in controller.details_pane.object
    assert str(record.conf_path) in controller.details_pane.object
    assert str(record.h5_path) in controller.details_pane.object
    assert controller.copy_path_button.disabled is False
    assert controller.delete_button.disabled is False


def test_dataset_manager_refresh_reloads_catalog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    set_active_workspace_path(workspace.root)
    alpha = _write_dataset(workspace, dataset_slug="alpha")
    beta = _write_dataset(workspace, dataset_slug="beta")
    calls = {"count": 0}

    def _list_records(workspace=None):
        calls["count"] += 1
        if calls["count"] == 1:
            return [alpha]
        return [alpha, beta]

    monkeypatch.setattr(dataset_manager_app, "list_workspace_datasets", _list_records)

    controller = dataset_manager_app._DatasetManagerController()
    assert len(controller._all_records) == 1

    controller._handle_refresh()

    assert len(controller._all_records) == 2
    assert "2 imported dataset(s) found." in controller.action_status.object


def test_dataset_manager_copy_feedback_handles_success_and_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    set_active_workspace_path(workspace.root)
    record = _write_dataset(workspace, dataset_slug="alpha")
    monkeypatch.setattr(
        dataset_manager_app, "list_workspace_datasets", lambda workspace=None: [record]
    )

    controller = dataset_manager_app._DatasetManagerController()
    _select_first_row(controller)

    assert str(record.dataset_dir) in controller.details_pane.object

    controller._apply_copy_feedback(f"copied|1|{record.dataset_dir}")
    assert "copied to the clipboard" in controller.action_status.object

    controller._apply_copy_feedback(f"fallback|2|{record.dataset_dir}")
    assert "Clipboard copy is unavailable" in controller.action_status.object
    assert str(record.dataset_dir) in controller.details_pane.object


def test_dataset_manager_delete_requires_confirmation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    set_active_workspace_path(workspace.root)
    record = _write_dataset(workspace, dataset_slug="alpha")
    monkeypatch.setattr(
        dataset_manager_app, "list_workspace_datasets", lambda workspace=None: [record]
    )

    controller = dataset_manager_app._DatasetManagerController()
    _select_first_row(controller)

    controller._handle_request_delete()

    assert controller.delete_confirm_panel.visible is True
    assert record.dataset_id in controller.delete_confirm_text.object
    assert str(record.dataset_dir) in controller.delete_confirm_text.object


def test_dataset_manager_confirm_delete_removes_selected_dataset(
    tmp_path: Path,
) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    set_active_workspace_path(workspace.root)
    record = _write_dataset(workspace, dataset_slug="alpha")

    controller = dataset_manager_app._DatasetManagerController()
    _select_first_row(controller)
    controller._handle_request_delete()
    controller._handle_confirm_delete()

    assert record.dataset_dir.exists() is False
    assert controller._all_records == []
    assert "Deleted dataset" in controller.action_status.object


def test_dataset_manager_cancel_delete_leaves_dataset_intact(tmp_path: Path) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    set_active_workspace_path(workspace.root)
    record = _write_dataset(workspace, dataset_slug="alpha")

    controller = dataset_manager_app._DatasetManagerController()
    _select_first_row(controller)
    controller._handle_request_delete()
    controller._handle_cancel_delete()

    assert record.dataset_dir.exists() is True
    assert controller.delete_confirm_panel.visible is False
    assert "cancelled" in controller.action_status.object


def test_dataset_manager_delete_rejects_records_outside_imported_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    set_active_workspace_path(workspace.root)
    outside_dir = workspace.datasets_dir / "custom" / "rogue"
    record = _write_dataset(workspace, dataset_slug="alpha")
    outside_dir.mkdir(parents=True, exist_ok=True)
    rogue_record = WorkspaceDatasetRecord(
        dataset_id="rogue",
        dataset_dir=outside_dir.resolve(),
        data_dir=(outside_dir / "data").resolve(),
        conf_path=(outside_dir / "data" / "conf.txt").resolve(),
        h5_path=(outside_dir / "data" / "data.h5").resolve(),
        lab_id="Schleyer",
        group_id="rogue",
        ref_id=None,
        n_agents=1,
    )
    monkeypatch.setattr(
        dataset_manager_app, "list_workspace_datasets", lambda workspace=None: [record]
    )

    controller = dataset_manager_app._DatasetManagerController()
    controller._selected_record = rogue_record
    controller._pending_delete_record = rogue_record

    controller._handle_confirm_delete()

    assert (
        "outside the active workspace imported root" in controller.action_status.object
    )
