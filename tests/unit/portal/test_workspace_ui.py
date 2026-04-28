from __future__ import annotations

from pathlib import Path

import pytest

from larvaworld.portal import workspace_ui
from larvaworld.portal.workspace import get_active_workspace_path, initialize_workspace


@pytest.fixture(autouse=True)
def workspace_config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LARVAWORLD_PORTAL_CONFIG_DIR", str(tmp_path / "config"))


def test_browse_activates_initialized_workspace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root, name="Workspace")
    controller = workspace_ui.WorkspaceUiController()

    monkeypatch.setattr(
        workspace_ui,
        "pick_directory",
        lambda *args, **kwargs: (workspace_root, None),
    )

    controller._on_browse(None)

    assert controller.path_input.value == str(workspace_root)
    assert get_active_workspace_path() == workspace_root.resolve()
    assert "Active workspace updated." in controller.status_pane.object


def test_browse_keeps_uninitialized_workspace_pending(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    controller = workspace_ui.WorkspaceUiController()

    monkeypatch.setattr(
        workspace_ui,
        "pick_directory",
        lambda *args, **kwargs: (workspace_root, None),
    )

    controller._on_browse(None)

    assert controller.path_input.value == str(workspace_root)
    assert get_active_workspace_path() is None
    assert "Folder is not initialized yet." in controller.status_pane.object
