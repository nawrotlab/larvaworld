from __future__ import annotations

from pathlib import Path

import pytest

from larvaworld.portal import path_picker


def test_pick_directory_uses_macos_picker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    selected = tmp_path / "workspace"

    monkeypatch.delenv("WSL_DISTRO_NAME", raising=False)
    monkeypatch.setattr(path_picker.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(
        path_picker.shutil,
        "which",
        lambda command: "/usr/bin/osascript" if command == "osascript" else None,
    )
    monkeypatch.setattr(
        path_picker,
        "_pick_directory_via_osascript",
        lambda initial_dir=None, *, fallback_dir=None, title="Select folder": selected,
    )

    path, error = path_picker.pick_directory(
        tmp_path, title="Select Larvaworld workspace folder"
    )

    assert path == selected
    assert error is None


def test_pick_directory_macos_cancel_is_silent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("WSL_DISTRO_NAME", raising=False)
    monkeypatch.setattr(path_picker.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(
        path_picker.shutil,
        "which",
        lambda command: "/usr/bin/osascript" if command == "osascript" else None,
    )
    monkeypatch.setattr(
        path_picker,
        "_pick_directory_via_osascript",
        lambda initial_dir=None, *, fallback_dir=None, title="Select folder": None,
    )

    path, error = path_picker.pick_directory(
        tmp_path, title="Select Larvaworld workspace folder"
    )

    assert path is None
    assert error is None


def test_pick_directory_reports_missing_picker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("WSL_DISTRO_NAME", raising=False)
    monkeypatch.setattr(path_picker.platform, "system", lambda: "Linux")
    monkeypatch.setattr(path_picker.shutil, "which", lambda _command: None)
    monkeypatch.setattr(
        path_picker,
        "_pick_directory_via_tk",
        lambda initial_dir=None, *, fallback_dir=None, title="Select folder": None,
    )

    path, error = path_picker.pick_directory(tmp_path)

    assert path is None
    assert error == "No folder picker is available in this environment."
