from __future__ import annotations

from pathlib import Path

import pytest

from larvaworld.portal import landing_app as landing_module
from larvaworld.portal import serve, workspace_ui
from larvaworld.portal.workspace import (
    get_active_workspace_path,
    initialize_workspace,
    set_active_workspace_path,
)


@pytest.fixture(autouse=True)
def workspace_config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LARVAWORLD_PORTAL_CONFIG_DIR", str(tmp_path / "config"))


def test_landing_redirects_to_workspace_setup_without_active_workspace() -> None:
    view = landing_module.landing_app()
    html = view.objects[0].object

    assert 'window.location.replace("/")' in html
    assert "Workspace setup required" in html


def test_loading_app_shows_workspace_setup_when_bootstrap_ready_without_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(serve, "_start_bootstrap_once", lambda: None)
    monkeypatch.setattr(
        serve._BOOTSTRAP_STATE,
        "snapshot",
        lambda: {
            "ready": True,
            "error": None,
            "step": "Ready",
            "completed_steps": 3,
            "total_steps": 3,
            "percent": 100,
            "elapsed": 1.0,
            "remaining": 0.0,
        },
    )
    root = serve.loading_app()
    row = root.objects[1]
    loading_card = row.objects[1]
    workspace_card = row.objects[2]

    assert loading_card.visible is False
    assert workspace_card.visible is True


def test_loading_app_clears_persisted_workspace_until_user_reselects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    monkeypatch.setattr(serve, "_start_bootstrap_once", lambda: None)
    monkeypatch.setattr(
        serve._BOOTSTRAP_STATE,
        "snapshot",
        lambda: {
            "ready": True,
            "error": None,
            "step": "Ready",
            "completed_steps": 3,
            "total_steps": 3,
            "percent": 100,
            "elapsed": 1.0,
            "remaining": 0.0,
        },
    )

    root = serve.loading_app()
    row = root.objects[1]
    loading_card = row.objects[1]
    workspace_card = row.objects[2]

    assert get_active_workspace_path() is None
    assert loading_card.visible is False
    assert workspace_card.visible is True


def test_landing_renders_normally_with_active_workspace(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    view = landing_module.landing_app()

    assert getattr(view, "title", None) == "Larvaworld Portal"
