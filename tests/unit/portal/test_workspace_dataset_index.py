from __future__ import annotations

import json
from pathlib import Path

import pytest

from larvaworld.portal.datasets.models import WorkspaceDatasetRecord
from larvaworld.portal.datasets import workspace_index
from larvaworld.portal.workspace import initialize_workspace


@pytest.fixture(autouse=True)
def workspace_config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LARVAWORLD_PORTAL_CONFIG_DIR", str(tmp_path / "config"))


def _write_dataset(
    dataset_dir: Path,
    *,
    dataset_id: str,
    ref_id: str | None = None,
    group_id: str | None = None,
    larva_group_id: str | None = None,
    n_agents: int | None = None,
    agent_ids: list[str] | None = None,
) -> None:
    data_dir = dataset_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "id": dataset_id,
        "dir": str(dataset_dir),
        "refID": ref_id,
        "group_id": group_id,
        "N": n_agents,
        "agent_ids": agent_ids,
        "larva_group": {"group_id": larva_group_id},
    }
    (data_dir / "conf.txt").write_text(json.dumps(payload), encoding="utf-8")
    (data_dir / "data.h5").write_bytes(b"placeholder")


def test_list_workspace_datasets_returns_empty_list_for_empty_workspace(
    tmp_path: Path,
) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")

    records = workspace_index.list_workspace_datasets(workspace=workspace)

    assert records == []


def test_list_workspace_datasets_returns_valid_portal_imported_record(
    tmp_path: Path,
) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    dataset_dir = (
        workspace.datasets_dir / "imported" / "Schleyer" / "exploration" / "dish01"
    )
    _write_dataset(
        dataset_dir,
        dataset_id="dish01_dataset",
        ref_id="exploration.dish01",
        group_id=None,
        larva_group_id="exploration",
        n_agents=12,
    )

    records = workspace_index.list_workspace_datasets(workspace=workspace)

    assert len(records) == 1
    record = records[0]
    assert record == WorkspaceDatasetRecord(
        dataset_id="dish01_dataset",
        dataset_dir=dataset_dir.resolve(),
        data_dir=(dataset_dir / "data").resolve(),
        conf_path=(dataset_dir / "data" / "conf.txt").resolve(),
        h5_path=(dataset_dir / "data" / "data.h5").resolve(),
        lab_id="Schleyer",
        group_id="exploration",
        ref_id="exploration.dish01",
        n_agents=12,
    )


def test_list_workspace_datasets_ignores_dataset_without_conf(
    tmp_path: Path,
) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    data_dir = (
        workspace.datasets_dir / "imported" / "Schleyer" / "group" / "ds" / "data"
    )
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "data.h5").write_bytes(b"placeholder")

    records = workspace_index.list_workspace_datasets(workspace=workspace)

    assert records == []


def test_list_workspace_datasets_ignores_dataset_without_h5(
    tmp_path: Path,
) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    data_dir = (
        workspace.datasets_dir / "imported" / "Schleyer" / "group" / "ds" / "data"
    )
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "conf.txt").write_text("{}", encoding="utf-8")

    records = workspace_index.list_workspace_datasets(workspace=workspace)

    assert records == []


def test_list_workspace_datasets_ignores_malformed_conf_without_crashing(
    tmp_path: Path,
) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    data_dir = (
        workspace.datasets_dir / "imported" / "Schleyer" / "group" / "ds" / "data"
    )
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "conf.txt").write_text("{not valid json", encoding="utf-8")
    (data_dir / "data.h5").write_bytes(b"placeholder")

    records = workspace_index.list_workspace_datasets(workspace=workspace)

    assert records == []


def test_list_workspace_datasets_ignores_unsupported_layout(
    tmp_path: Path,
) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    dataset_dir = workspace.datasets_dir / "custom" / "dataset_a"
    _write_dataset(
        dataset_dir,
        dataset_id="dataset_a",
        ref_id=None,
        group_id=None,
        larva_group_id=None,
        n_agents=3,
    )

    records = workspace_index.list_workspace_datasets(workspace=workspace)

    assert records == []


def test_list_workspace_datasets_returns_records_in_deterministic_order(
    tmp_path: Path,
) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    beta_dir = workspace.datasets_dir / "imported" / "Schleyer" / "group" / "beta"
    alpha_dir = workspace.datasets_dir / "imported" / "Schleyer" / "group" / "alpha"
    _write_dataset(alpha_dir, dataset_id="alpha", larva_group_id="group")
    _write_dataset(beta_dir, dataset_id="beta", larva_group_id="group")

    records = workspace_index.list_workspace_datasets(workspace=workspace)

    assert [record.dataset_id for record in records] == ["alpha", "beta"]


def test_get_workspace_dataset_uses_shared_record_builder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sentinel = WorkspaceDatasetRecord(
        dataset_id="sentinel",
        dataset_dir=tmp_path / "dataset",
        data_dir=tmp_path / "dataset" / "data",
        conf_path=tmp_path / "dataset" / "data" / "conf.txt",
        h5_path=tmp_path / "dataset" / "data" / "data.h5",
        lab_id="Schleyer",
        group_id="group",
        ref_id=None,
        n_agents=1,
    )
    monkeypatch.setattr(
        workspace_index, "_record_from_dataset_dir", lambda _path: sentinel
    )

    record = workspace_index.get_workspace_dataset(tmp_path / "dataset")

    assert record is sentinel
