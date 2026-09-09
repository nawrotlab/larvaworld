from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from larvaworld import DATA_DIR
from larvaworld.lib import reg
from larvaworld.lib.sim.manifest import RunManifestSession
from larvaworld.portal.datasets import dataset_manager_app
from larvaworld.portal.datasets.manager_helpers import UnifiedDatasetRecord
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
) -> UnifiedDatasetRecord:
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
    record = WorkspaceDatasetRecord(
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
    return UnifiedDatasetRecord.from_imported(record)


def _select_first_row(
    controller: dataset_manager_app._DatasetManagerController,
) -> None:
    controller.table.selection = [0]
    controller._on_table_selection_change()


def test_dataset_manager_requires_active_workspace() -> None:
    controller = dataset_manager_app._DatasetManagerController()

    assert "requires an active workspace" in controller.empty_state.object
    assert controller._all_records == []


def test_dataset_manager_empty_state_points_to_import_app(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    set_active_workspace_path(workspace.root)
    # An empty *workspace* still has the package's own bundled DATA_DIR
    # dataset(s) available -- isolate from those here to test the
    # genuinely-empty-catalog state specifically.
    monkeypatch.setattr(
        dataset_manager_app, "list_workspace_datasets", lambda workspace=None: []
    )

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


def test_list_data_dir_datasets_detects_default_ref_dataset_by_folder() -> None:
    # Detection by folder layout alone (DATA_DIR/<Lab>Group/processed/...),
    # not by refID lookup -- confirms the package's own bundled reference
    # dataset (reg.default_refID == "exploration.30controls") is found by
    # scanning real files on disk, the same way workspace datasets are
    # found by scanning the workspace's own directories.
    from larvaworld.portal.datasets.workspace_index import list_data_dir_datasets

    records = list_data_dir_datasets()

    thirty_controls = next((r for r in records if r.dataset_id == "30controls"), None)
    assert thirty_controls is not None, (
        f"Expected a '30controls' dataset under {DATA_DIR}, found: "
        f"{[r.dataset_id for r in records]}"
    )
    assert thirty_controls.lab_id == "Schleyer"
    assert thirty_controls.group_id == "exploration"
    assert thirty_controls.ref_id == reg.default_refID
    assert thirty_controls.dataset_dir.is_dir()
    assert (thirty_controls.data_dir / "conf.txt").is_file()
    assert (thirty_controls.data_dir / "data.h5").is_file()


def test_dataset_manager_shows_bundled_data_dir_datasets_as_editable(
    tmp_path: Path,
) -> None:
    # The manager must surface DATA_DIR-bundled datasets alongside
    # workspace ones (found by folder detection, via the real
    # list_data_dir_datasets -- not mocked here), labeled distinctly, and
    # editable like any other dataset (preprocess/process/annotate/
    # update_refid); only delete stays workspace-"imported"-only, per
    # delete_imported_workspace_dataset's own path validation.
    workspace = initialize_workspace(tmp_path / "workspace")
    set_active_workspace_path(workspace.root)

    controller = dataset_manager_app._DatasetManagerController()

    bundled = next((r for r in controller._all_records if r.origin == "bundled"), None)
    assert bundled is not None, "Expected a bundled DATA_DIR dataset to be detected"
    assert bundled.dataset_id == "30controls"

    row = controller.table.value[controller.table.value["Dataset ID"] == "30controls"]
    assert row.iloc[0]["Source"] == "Bundled"

    idx = controller._filtered_records.index(bundled)
    controller.table.selection = [idx]
    controller._on_table_selection_change()
    assert controller.delete_button.disabled is True
    assert controller.preprocess_button.disabled is False
    assert controller.process_button.disabled is False
    assert controller.annotate_button.disabled is False
    assert controller.update_refid_button.disabled is False


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


def test_dataset_manager_inspects_simulated_dataset_source_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    set_active_workspace_path(workspace.root)
    run_dir = workspace.experiments_dir / "source-run"
    session = RunManifestSession(
        run=SimpleNamespace(
            dir=str(run_dir),
            id="source-run",
            runtype="Exp",
            experiment="dish",
            store_data=True,
            parameters={},
            screen_kws={},
        ),
        seed=3,
    )
    session.finish()
    dataset_dir = run_dir / "data" / "group"
    data_dir = dataset_dir / "data"
    data_dir.mkdir(parents=True)
    conf_path = data_dir / "conf.txt"
    conf_path.write_text(
        json.dumps(
            {
                "id": "group",
                "dir": str(dataset_dir),
                "refID": None,
                "group_id": "group",
                "agent_ids": [],
                "provenance": {
                    "origin": "simulation",
                    "run_manifest": {
                        "manifest_id": session.manifest["run"]["manifest_id"],
                        "workspace_id": workspace.workspace_id,
                        "path": "../../run_manifest.json",
                    },
                    "lineage": [],
                },
            }
        ),
        encoding="utf-8",
    )
    h5_path = data_dir / "data.h5"
    h5_path.write_bytes(b"")
    record = UnifiedDatasetRecord(
        origin="simulation_run",
        dataset_id="group",
        dataset_dir=dataset_dir,
        data_dir=data_dir,
        conf_path=conf_path,
        h5_path=h5_path,
        lab_id=None,
        group_id="group",
        ref_id=None,
        n_agents=0,
        run_id="source-run",
        member_id="group",
    )
    monkeypatch.setattr(
        dataset_manager_app, "list_workspace_datasets", lambda workspace=None: [record]
    )

    controller = dataset_manager_app._DatasetManagerController()
    _select_first_row(controller)
    controller._handle_inspect_manifest()

    assert session.manifest["run"]["manifest_id"] in controller.details_pane.object
    assert controller.inspect_manifest_button.disabled is False
    assert controller.manifest_inspector.visible is True
    assert (
        controller.manifest_inspector.object["run"]["manifest_id"]
        == (session.manifest["run"]["manifest_id"])
    )


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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    set_active_workspace_path(workspace.root)
    record = _write_dataset(workspace, dataset_slug="alpha")
    # Real workspace scanning is exercised below (to confirm "alpha" is
    # actually gone from disk after delete); isolate from the package's
    # own bundled DATA_DIR dataset(s), which delete must never touch and
    # which would otherwise survive to break the "nothing left" assertion.
    monkeypatch.setattr(
        "larvaworld.portal.datasets.manager_helpers.list_data_dir_datasets",
        lambda: [],
    )

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
