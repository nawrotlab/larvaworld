"""Tests for the portal landing page."""

from __future__ import annotations

from pathlib import Path

import panel as pn
import pytest

from larvaworld.portal.landing_app import landing_app
from larvaworld.portal.workspace import (
    clear_active_workspace_path,
    initialize_workspace,
    set_active_workspace_path,
)


@pytest.fixture(autouse=True)
def workspace_config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LARVAWORLD_PORTAL_CONFIG_DIR", str(tmp_path / "config"))
    clear_active_workspace_path()


def test_landing_app_without_workspace_redirects() -> None:
    app = landing_app()

    assert isinstance(app, pn.Column)
    assert "window.location.replace" in app.objects[0].object


def test_landing_app_with_active_workspace_renders_template(tmp_path: Path) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    set_active_workspace_path(workspace.root)

    app = landing_app()

    assert isinstance(app, pn.template.MaterialTemplate)
    assert app.main.objects


def test_landing_app_banner_renders_below_quick_start(tmp_path: Path) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    set_active_workspace_path(workspace.root)

    app = landing_app()
    root = next(
        child
        for child in app.main
        if "lw-portal-root" in getattr(child, "css_classes", [])
    )
    css_classes = [tuple(getattr(child, "css_classes", ())) for child in root.objects]
    quick_start_index = next(
        i
        for i, classes in enumerate(css_classes)
        if any(c.startswith("lw-portal-quick-start") for c in classes)
    )
    banner_index = next(
        i for i, classes in enumerate(css_classes) if "lw-portal-banner" in classes
    )
    assert quick_start_index < banner_index
