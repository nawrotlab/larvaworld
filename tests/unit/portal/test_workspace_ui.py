from __future__ import annotations

from pathlib import Path

import pytest

from larvaworld.portal import workspace_ui
from larvaworld.portal.workspace import (
    get_active_workspace_path,
    initialize_workspace,
    set_active_workspace_path,
)


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


def test_default_candidate_is_a_ready_to_use_home_folder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A beginner must be able to accept the suggestion without typing a path.
    monkeypatch.setattr(workspace_ui.Path, "home", classmethod(lambda cls: tmp_path))

    assert workspace_ui._default_workspace_candidate() == tmp_path / "larvaworld"


def test_default_candidate_prefers_the_remembered_workspace(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    assert workspace_ui._default_workspace_candidate() == workspace_root.resolve()


def test_path_input_is_prefilled_so_setup_is_one_press(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(workspace_ui.Path, "home", classmethod(lambda cls: tmp_path))

    controller = workspace_ui.WorkspaceUiController()

    assert controller.path_input.value.strip()
    assert controller.init_button.name == "Use this folder"
    assert controller.init_button.button_type == "primary"


def test_initialize_creates_the_folder_and_remembers_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(workspace_ui.Path, "home", classmethod(lambda cls: tmp_path))
    controller = workspace_ui.WorkspaceUiController()
    target = Path(controller.path_input.value).expanduser()
    assert not target.exists()

    controller._on_initialize(None)

    assert target.is_dir()
    assert get_active_workspace_path() == target.resolve()
    assert "Workspace initialized and activated." in controller.status_pane.object


def test_workspace_survives_a_simulated_restart(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(workspace_ui.Path, "home", classmethod(lambda cls: tmp_path))
    workspace_ui.WorkspaceUiController()._on_initialize(None)
    remembered = get_active_workspace_path()

    # A fresh controller stands in for the next portal launch: it must not
    # re-prompt, and must point back at the same folder.
    assert workspace_ui.WorkspaceUiController().path_input.value == str(remembered)
    assert get_active_workspace_path() == remembered
