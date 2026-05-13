from __future__ import annotations

import json
from pathlib import Path

import pytest

from larvaworld.portal.simulation.display_shortcuts import DisplayShortcutsController
from larvaworld.portal.workspace import (
    clear_active_workspace_path,
    get_workspace_dir,
    initialize_workspace,
    set_active_workspace_path,
)


@pytest.fixture(autouse=True)
def workspace_config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LARVAWORLD_PORTAL_CONFIG_DIR", str(tmp_path / "config"))
    clear_active_workspace_path()


def test_display_shortcuts_defaults_load_without_workspace_file(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = DisplayShortcutsController()
    assert controller.config.pause == "space"
    assert controller.config.visible_ids == "tab"
    assert controller.dirty is False


def test_display_shortcuts_save_and_load_workspace_overrides(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = DisplayShortcutsController()
    controller.config.pause = "a"
    controller.config.snapshot = "d"
    controller._on_save()

    path = get_workspace_dir("metadata") / "display_shortcuts.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["version"] == 1
    assert payload["keys"]["simulation"]["pause"] == "a"
    assert "pygame_keys" not in payload

    reloaded = DisplayShortcutsController()
    assert reloaded.config.pause == "a"
    assert reloaded.config.snapshot == "d"


def test_display_shortcuts_reset_marks_dirty_and_restores_defaults(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = DisplayShortcutsController()
    controller.config.pause = "tab"
    controller._on_reset()

    assert controller.config.pause == "space"
    assert controller.dirty is True
    assert "Save shortcuts" in controller.status


def test_display_shortcuts_duplicate_key_blocks_save(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = DisplayShortcutsController()
    controller.config.pause = "p"
    controller.config.visible_trails = "p"
    controller._on_save()

    assert "Cannot save shortcuts" in controller.status
    assert "already assigned" in controller.status


def test_display_shortcuts_runtime_pygame_keys_reflect_edits(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = DisplayShortcutsController()
    controller.config.pause = "tab"
    pygame_keys = controller.runtime_pygame_keys()
    assert pygame_keys["pause"] == "K_TAB"


def test_display_shortcuts_save_without_workspace_reports_error() -> None:
    controller = DisplayShortcutsController()
    controller.config.pause = "a"
    controller._on_save()

    assert "no active workspace" in controller.status
