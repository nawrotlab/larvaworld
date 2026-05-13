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
    assert controller.editing is False


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


def test_display_shortcuts_toggle_editing_enables_key_buttons(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = DisplayShortcutsController()
    controller.view()
    assert controller._key_buttons["pause"].disabled is True
    assert controller._edit_btn.name == "Edit shortcuts"

    controller._toggle_editing()
    assert controller.editing is True
    assert controller._key_buttons["pause"].disabled is False
    assert controller._edit_btn.name == "Done editing"


def test_display_shortcuts_capture_apply_tab_updates_pause(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = DisplayShortcutsController()
    controller.view()
    controller._toggle_editing()
    controller._start_capture("pause")
    assert controller.capturing_field == "pause"
    assert "Press a key for" in controller.status
    assert controller._key_buttons["pause"].name == "Press key..."

    controller._apply_captured_key("pause", "Tab")
    assert controller.config.pause == "space"
    assert controller.capturing_field == ""
    assert "already used by Agent IDs" in controller.status


def test_display_shortcuts_capture_space_updates_pause(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = DisplayShortcutsController()
    controller.view()
    controller.config.pause = "a"
    controller._toggle_editing()
    controller._start_capture("pause")
    controller._apply_captured_key("pause", " ")

    assert controller.config.pause == "space"
    assert controller.dirty is True
    assert controller.capturing_field == ""
    assert controller._key_buttons["pause"].name == "Space"


def test_display_shortcuts_capture_escape_cancels(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = DisplayShortcutsController()
    controller.view()
    controller._toggle_editing()
    controller._start_capture("pause")
    original = controller.config.pause

    controller._apply_captured_key("pause", "Escape")
    assert controller.config.pause == original
    assert controller.capturing_field == ""
    assert "cancelled" in controller.status


def test_display_shortcuts_capture_unsupported_key(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = DisplayShortcutsController()
    controller.view()
    controller._toggle_editing()
    controller._start_capture("pause")
    controller.config.pause = "a"
    controller._apply_captured_key("pause", "α")
    assert controller.config.pause == ""
    assert controller.dirty is True
    assert controller.capturing_field == ""
    assert "Unsupported key" in controller.status


def test_display_shortcuts_save_blocks_blank_shortcut(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = DisplayShortcutsController()
    controller.config.pause = ""
    controller._on_save()

    assert "Cannot save shortcuts" in controller.status


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
