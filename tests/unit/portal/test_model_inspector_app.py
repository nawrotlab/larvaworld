from __future__ import annotations

import importlib
from pathlib import Path

import pandas as pd
import pytest

from larvaworld.lib import reg
from larvaworld.portal.models_architecture.model_inspector_app import (
    _ModelInspectorController,
    model_inspector_app,
)
from larvaworld.portal.models_architecture.model_inspector_data import (
    BASELINE_MODULES,
    inspect_model,
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
    assert not hasattr(controller, "optional_table")
    assert controller.settings_grid.objects
    assert len(controller.settings_grid.objects) >= len(BASELINE_MODULES)


def test_baseline_cards_are_editable(monkeypatch: pytest.MonkeyPatch) -> None:
    _guard_registry_writes(monkeypatch)
    controller = _ModelInspectorController()
    editable_widgets = []
    for card in controller.settings_grid.objects[: len(BASELINE_MODULES)]:
        for child in getattr(card, "objects", []):
            if hasattr(child, "_widgets"):
                for widget in child._widgets.values():
                    editable_widgets.append(widget)
    assert editable_widgets, "Expected editable widgets for baseline modules."
    assert any(not getattr(widget, "disabled", False) for widget in editable_widgets)


def test_controller_comparison_hidden_after_local_edits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _guard_registry_writes(monkeypatch)
    controller = _ModelInspectorController()
    controller._on_local_parameter_edit()
    assert controller.compare_select.disabled is True
    assert "hidden during local edits" in controller.compare_title.object


def test_controller_merges_optional_modules_into_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _guard_registry_writes(monkeypatch)
    controller = _ModelInspectorController()
    for model_id in reg.conf.Model.confIDs:
        inspection = inspect_model(model_id)
        if not inspection.optional_modules:
            continue
        controller.primary_select.value = model_id
        summary = controller.primary_table.object
        assert isinstance(summary, pd.DataFrame)
        assert "Category" in summary.columns
        assert set(summary["Category"]) >= {"Baseline", "Optional"}
        optional_ids = {module.module_id for module in inspection.optional_modules}
        assert optional_ids.issubset(set(summary["Module"]))
        return
    pytest.skip("No model with configured optional modules was found.")


def test_controller_live_run_updates_trace_and_pause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _guard_registry_writes(monkeypatch)
    controller = _ModelInspectorController()
    controller._on_run()
    controller._tick_live_preview()
    controller._tick_live_preview()
    assert isinstance(controller.probe_table.object, pd.DataFrame)
    assert not controller.probe_table.object.empty
    assert controller._step >= 2
    controller._on_pause()
    assert controller._is_running is False


def test_controller_run_resume_preserves_transient_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _guard_registry_writes(monkeypatch)
    controller = _ModelInspectorController()
    brain = controller._brain
    controller._on_local_parameter_edit()
    controller._on_run()
    controller._on_pause()
    controller._on_run()
    assert controller._brain is brain


def test_controller_preview_settings_are_editable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _guard_registry_writes(monkeypatch)
    controller = _ModelInspectorController()
    controller.max_steps_input.value = 2
    controller.a_in_input.value = 0.25
    controller.trace_window_input.value = 1
    controller._on_run()
    controller._tick_live_preview()
    controller._tick_live_preview()
    assert controller._step == 2
    assert len(controller.probe_table.object) == 1
    controller._tick_live_preview()
    assert controller._is_running is False


def test_controller_reset_required_settings_lock_while_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _guard_registry_writes(monkeypatch)
    controller = _ModelInspectorController()
    assert controller.dt_input.disabled is False
    controller._on_run()
    assert controller.dt_input.disabled is True
    controller._on_pause()
    assert controller.dt_input.disabled is False


def test_controller_dt_change_resets_local_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _guard_registry_writes(monkeypatch)
    controller = _ModelInspectorController()
    old_brain = controller._brain
    controller._on_local_parameter_edit()
    controller._on_run()
    controller._tick_live_preview()
    controller._on_pause()
    controller.dt_input.value = 0.2
    assert controller._brain is not old_brain
    assert controller._has_local_edits is False
    assert controller._step == 0
    assert controller.probe_table.object.empty


def test_controller_clear_trace_preserves_local_edits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _guard_registry_writes(monkeypatch)
    controller = _ModelInspectorController()
    controller._on_local_parameter_edit()
    controller._on_run()
    controller._tick_live_preview()
    assert not controller.probe_table.object.empty
    controller._on_clear_trace()
    assert controller._has_local_edits is True
    assert controller.probe_table.object.empty


def test_controller_reset_to_preset_clears_local_edit_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _guard_registry_writes(monkeypatch)
    controller = _ModelInspectorController()
    controller._on_local_parameter_edit()
    controller._on_reset_to_preset()
    assert controller._has_local_edits is False
    assert controller.compare_select.disabled is False


def test_controller_auto_stops_at_max_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _guard_registry_writes(monkeypatch)
    controller = _ModelInspectorController()
    controller._on_run()
    controller._step = controller.max_steps_input.value
    controller._tick_live_preview()
    assert controller._is_running is False


def test_app_source_does_not_import_legacy_dashboard() -> None:
    model_inspector_app_module = importlib.import_module(
        "larvaworld.portal.models_architecture.model_inspector_app"
    )
    source = Path(model_inspector_app_module.__file__).read_text(encoding="utf-8")
    assert "larvaworld.dashboards.model_inspector" not in source
