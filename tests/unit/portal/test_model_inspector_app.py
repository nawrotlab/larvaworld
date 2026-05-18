from __future__ import annotations

import importlib
from pathlib import Path

import pandas as pd
import pytest

from larvaworld.lib import reg
from larvaworld.portal.models_architecture.model_inspector_data import BASELINE_MODULES
from larvaworld.portal.models_architecture.model_inspector_app import (
    _ModelInspectorController,
    model_inspector_app,
)
from larvaworld.portal.workspace import clear_active_workspace_path


@pytest.fixture(autouse=True)
def workspace_config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LARVAWORLD_PORTAL_CONFIG_DIR", str(tmp_path / "config"))
    clear_active_workspace_path()


def _guard_registry_writes(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*_args, **_kwargs):
        raise AssertionError("registry write path must not be called")

    monkeypatch.setattr(reg.conf.Model, "save", _boom, raising=True)
    monkeypatch.setattr(reg.conf.Model, "setID", _boom, raising=True)
    monkeypatch.setattr(reg.conf.Model, "delete", _boom, raising=True)
    monkeypatch.setattr(reg.conf.Model, "set_dict", _boom, raising=True)
    monkeypatch.setattr(reg.conf.Model, "reset", _boom, raising=True)


def test_model_inspector_app_returns_viewable(monkeypatch: pytest.MonkeyPatch) -> None:
    _guard_registry_writes(monkeypatch)
    view = model_inspector_app()
    assert view is not None


def test_controller_initializes_primary_only(monkeypatch: pytest.MonkeyPatch) -> None:
    _guard_registry_writes(monkeypatch)
    controller = _ModelInspectorController()
    assert controller.primary_select.value is not None
    assert controller.compare_select.value == ""
    assert isinstance(controller.primary_table.object, pd.DataFrame)
    assert controller.settings_grid.objects
    assert len(controller.settings_grid.objects) >= len(BASELINE_MODULES)
    assert controller.compare_table.object.empty


def test_controller_shows_comparison_when_second_model_selected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _guard_registry_writes(monkeypatch)
    controller = _ModelInspectorController()
    options = [value for value in controller.compare_select.options.values() if value]
    if not options:
        pytest.skip("Need at least two models for comparison.")
    controller.compare_select.value = options[0]
    controller._on_compare_change()
    assert not controller.compare_table.object.empty


def test_settings_cards_are_read_only(monkeypatch: pytest.MonkeyPatch) -> None:
    _guard_registry_writes(monkeypatch)
    controller = _ModelInspectorController()
    param_panes: list[object] = []
    for card in controller.settings_grid.objects:
        for child in getattr(card, "objects", []):
            if hasattr(child, "_widgets"):
                param_panes.append(child)
    assert param_panes, "Expected at least one module settings Param pane."
    for pane in param_panes:
        for widget in pane._widgets.values():
            assert getattr(widget, "disabled", False)


def test_controller_probe_updates_table_and_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _guard_registry_writes(monkeypatch)
    controller = _ModelInspectorController()
    controller._on_run_probe()
    assert isinstance(controller.probe_table.object, pd.DataFrame)
    assert not controller.probe_table.object.empty
    assert controller.probe_plots.objects
    assert "Probe completed" in controller.status.object


def test_app_source_does_not_import_legacy_dashboard() -> None:
    model_inspector_app_module = importlib.import_module(
        "larvaworld.portal.models_architecture.model_inspector_app"
    )
    source = Path(model_inspector_app_module.__file__).read_text(encoding="utf-8")
    assert "larvaworld.dashboards.model_inspector" not in source
