from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from larvaworld.portal.datasets import import_adapter
from larvaworld.portal.datasets.models import ImportRequest, WorkspaceDatasetRecord
from larvaworld.portal.workspace import initialize_workspace, set_active_workspace_path


@pytest.fixture(autouse=True)
def workspace_config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LARVAWORLD_PORTAL_CONFIG_DIR", str(tmp_path / "config"))


class _FakeLab:
    def __init__(self, returned_dir: Path | None) -> None:
        self.returned_dir = returned_dir
        self.import_calls: list[dict] = []

    def import_dataset(self, **kwargs):
        self.import_calls.append(kwargs)
        if self.returned_dir is None:
            return None
        return SimpleNamespace(config=SimpleNamespace(dir=str(self.returned_dir)))

    def import_datasets(self, **kwargs):
        raise AssertionError("import_datasets() must not be used by the adapter")


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
        n_agents=4,
    )


def test_build_workspace_proc_folder_targets_workspace_import_tree(
    tmp_path: Path,
) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")

    proc_root = import_adapter.build_workspace_proc_folder(workspace, "Schleyer")

    assert proc_root == (workspace.datasets_dir / "imported" / "Schleyer").resolve()


def test_import_into_workspace_uses_active_workspace_and_explicit_backend_call_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    set_active_workspace_path(workspace.root)
    dataset_dir = (
        workspace.datasets_dir / "imported" / "Schleyer" / "exploration" / "dish01"
    )
    fake_lab = _FakeLab(dataset_dir)
    seen_dataset_dirs: list[Path] = []
    sentinel = _record(dataset_dir.resolve())
    monkeypatch.setattr(
        import_adapter.reg.conf.LabFormat, "get", lambda _lab_id: fake_lab
    )
    monkeypatch.setattr(
        import_adapter,
        "get_workspace_dataset",
        lambda path: seen_dataset_dirs.append(path) or sentinel,
    )

    request = ImportRequest(
        lab_id="Schleyer",
        parent_dir="exploration/dish01",
        raw_folder=Path("/tmp/raw"),
        group_id="exploration",
        dataset_id="dish01",
        merged=True,
        color="black",
        enrich_conf={"mode": "minimal"},
        extra_kwargs={"sample": "sample_ref"},
    )

    record = import_adapter.import_into_workspace(request)

    assert record is sentinel
    assert seen_dataset_dirs == [dataset_dir.resolve()]
    assert len(fake_lab.import_calls) == 1
    assert {k: v for k, v in fake_lab.import_calls[0].items() if k != "raw_folder"} == {
        "parent_dir": "exploration/dish01",
        "merged": True,
        "proc_folder": str(
            (workspace.datasets_dir / "imported" / "Schleyer").resolve()
        ),
        "group_id": "exploration",
        "id": "dish01",
        "color": "black",
        "enrich_conf": {"mode": "minimal"},
        "save_dataset": True,
        "sample": "sample_ref",
    }
    assert Path(fake_lab.import_calls[0]["raw_folder"]) == Path("/tmp/raw")


def test_import_into_workspace_passes_refid_only_when_requested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    dataset_dir = (
        workspace.datasets_dir / "imported" / "Schleyer" / "exploration" / "dish01"
    )
    fake_lab = _FakeLab(dataset_dir)
    monkeypatch.setattr(
        import_adapter.reg.conf.LabFormat, "get", lambda _lab_id: fake_lab
    )
    monkeypatch.setattr(
        import_adapter,
        "get_workspace_dataset",
        lambda _path: _record(dataset_dir.resolve()),
    )

    import_adapter.import_into_workspace(
        ImportRequest(
            lab_id="Schleyer",
            parent_dir="exploration/dish01",
            ref_id="exploration.dish01",
        ),
        workspace=workspace,
    )

    assert fake_lab.import_calls[0]["refID"] == "exploration.dish01"


def test_import_into_workspace_raises_when_backend_returns_no_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    fake_lab = _FakeLab(None)
    monkeypatch.setattr(
        import_adapter.reg.conf.LabFormat, "get", lambda _lab_id: fake_lab
    )

    with pytest.raises(
        RuntimeError, match="Import failed: backend returned no dataset"
    ):
        import_adapter.import_into_workspace(
            ImportRequest(lab_id="Schleyer", parent_dir="exploration/dish01"),
            workspace=workspace,
        )


def test_import_into_workspace_raises_when_saved_output_is_not_supported_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    dataset_dir = (
        workspace.datasets_dir / "imported" / "Schleyer" / "exploration" / "dish01"
    )
    fake_lab = _FakeLab(dataset_dir)
    monkeypatch.setattr(
        import_adapter.reg.conf.LabFormat, "get", lambda _lab_id: fake_lab
    )

    with pytest.raises(
        RuntimeError,
        match="Import failed: saved dataset was not found in portal-supported workspace layout",
    ):
        import_adapter.import_into_workspace(
            ImportRequest(lab_id="Schleyer", parent_dir="exploration/dish01"),
            workspace=workspace,
        )


def test_import_into_workspace_raises_when_record_cannot_be_built_from_saved_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    dataset_dir = (
        workspace.datasets_dir / "imported" / "Schleyer" / "exploration" / "dish01"
    )
    data_dir = dataset_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "conf.txt").write_text("{}", encoding="utf-8")
    (data_dir / "data.h5").write_bytes(b"placeholder")
    fake_lab = _FakeLab(dataset_dir)
    monkeypatch.setattr(
        import_adapter.reg.conf.LabFormat, "get", lambda _lab_id: fake_lab
    )
    monkeypatch.setattr(import_adapter, "get_workspace_dataset", lambda _path: None)

    with pytest.raises(
        RuntimeError,
        match="Import failed: workspace dataset record could not be built from saved output",
    ):
        import_adapter.import_into_workspace(
            ImportRequest(lab_id="Schleyer", parent_dir="exploration/dish01"),
            workspace=workspace,
        )
