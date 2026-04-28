from __future__ import annotations

import json
from pathlib import Path

import pytest

from larvaworld.portal.workspace import (
    WorkspaceError,
    clear_active_workspace_path,
    get_active_workspace,
    get_active_workspace_path,
    get_notebook_workspace_dir,
    get_workspace_dir,
    initialize_workspace,
    load_workspace,
    require_active_workspace,
    set_active_workspace_path,
    validate_workspace,
)


@pytest.fixture(autouse=True)
def workspace_config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LARVAWORLD_PORTAL_CONFIG_DIR", str(tmp_path / "config"))


def test_validate_workspace_for_new_path_is_creatable(tmp_path: Path) -> None:
    candidate = tmp_path / "new-workspace"

    validation = validate_workspace(candidate)

    assert validation.path == candidate.resolve()
    assert validation.exists is False
    assert validation.writable is True
    assert validation.initialized is False
    assert "environments" in validation.missing_dirs
    assert validation.errors == []


def test_initialize_workspace_creates_expected_layout(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"

    state = initialize_workspace(workspace_root, name="Portal Workspace")

    assert state.root == workspace_root.resolve()
    assert state.name == "Portal Workspace"
    assert state.environments_dir.is_dir()
    assert state.experiments_dir.is_dir()
    assert state.datasets_dir.is_dir()
    assert state.analysis_dir.is_dir()
    assert state.metadata_dir.is_dir()
    assert state.metadata_path.is_file()

    metadata = json.loads(state.metadata_path.read_text(encoding="utf-8"))
    assert metadata["workspace_name"] == "Portal Workspace"
    assert metadata["schema_version"] == 1


def test_load_workspace_returns_resolved_paths(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)

    state = load_workspace(workspace_root)

    assert state.environments_dir == workspace_root.resolve() / "environments"
    assert state.metadata_dir == workspace_root.resolve() / "metadata"


def test_active_workspace_roundtrip(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)

    set_active_workspace_path(workspace_root)

    assert get_active_workspace_path() == workspace_root.resolve()
    assert get_active_workspace() is not None
    assert require_active_workspace().root == workspace_root.resolve()


def test_clear_active_workspace_resets_config() -> None:
    clear_active_workspace_path()

    assert get_active_workspace_path() is None
    assert get_active_workspace() is None
    with pytest.raises(WorkspaceError):
        require_active_workspace()


def test_get_workspace_dir_and_notebook_dir_use_active_workspace(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    assert (
        get_workspace_dir("environments") == workspace_root.resolve() / "environments"
    )
    notebook_dir = get_notebook_workspace_dir()
    assert notebook_dir == workspace_root.resolve() / "metadata" / "notebooks"
    assert notebook_dir.is_dir()


def test_validate_workspace_rejects_repo_root() -> None:
    repo_root = Path(__file__).resolve().parents[3]

    validation = validate_workspace(repo_root)

    assert validation.errors
    assert any("source tree" in error for error in validation.errors)


def test_load_workspace_requires_initialized_metadata(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir(parents=True)

    with pytest.raises(WorkspaceError, match="not initialized"):
        load_workspace(workspace_root)
