from __future__ import annotations

import copy
import io
import json
from collections.abc import Mapping
from html import escape
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import pandas as pd
import panel as pn
import param
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure

from larvaworld.lib import reg, util
from larvaworld.lib.model import Effector
from larvaworld.lib.model import agents, deb
from larvaworld.lib.model import moduleDB as MD
from larvaworld.lib.param import class_objs
from larvaworld.portal.config_widgets.preset_controls import (
    ADVANCED_PRESET_POLICY,
    USER_PRESET_POLICY,
    PresetControlsController,
    PresetRef,
    WorkspacePresetStore,
)
from larvaworld.portal.config_widgets.widget_base import param_controls
from larvaworld.portal.models_architecture.model_inspector_data import (
    BASELINE_MODULES,
    DEFAULT_LIVE_PREVIEW_REPORTER_KEYS,
    LIVE_PREVIEW_REPORTER_KEYS,
    build_inspection_brain_from_config,
    compare_model_inspections,
    inspect_model,
    inspect_model_from_config,
    inspect_model_modules_from_config,
    load_model_draft,
    list_model_ids,
    set_draft_brain_module_mode,
    set_draft_memory_config,
    set_draft_module_enabled,
    set_draft_module_parameter,
    validate_draft_module_config,
)
from larvaworld.portal.models_architecture.model_inspector_models import (
    DraftValidationIssue,
    ModelModuleSpec,
    ModelInspectorError,
    ModuleInspection,
)
from larvaworld.portal.panel_components import PORTAL_RAW_CSS, build_app_header
from larvaworld.portal.workspace import WorkspaceError, get_workspace_dir

__all__ = ["_ModelInspectorController", "model_inspector_app"]


LIVE_ROLLOVER = 100
LIVE_MAX_STEPS = 501
LIVE_DT = 0.1
LIVE_A_IN = 0.0
CONTROLS_COLUMN_WIDTH = 340
SECTION_COLUMN_GAP_PX = 10
UIRefreshScope = Literal["parameter", "mode", "enabled", "full"]


MODEL_INSPECTOR_RAW_CSS = """
.lw-model-inspector-root {
  padding: 14px 12px 20px 12px;
}

.lw-model-inspector-intro {
  border-left: 4px solid #c1b0c2;
  background: rgba(193, 176, 194, 0.16);
  border-radius: 10px;
  padding: 10px 12px;
  margin: 0 0 10px 0;
}

.lw-model-inspector-status {
  font-size: 12px;
  line-height: 1.45;
  border-radius: 10px;
  padding: 10px 12px;
  border: 1px solid rgba(17, 17, 17, 0.1);
  background: rgba(248, 250, 252, 0.94);
}

.lw-model-inspector-controls-box {
  border-radius: 10px;
  border: 1px solid rgba(17, 17, 17, 0.12);
  background: rgba(193, 176, 194, 0.12);
  padding: 10px;
}

.lw-model-inspector-section-box {
  border-radius: 10px;
  border: 1px solid rgba(17, 17, 17, 0.1);
  background: rgba(248, 250, 252, 0.96);
  padding: 10px;
  margin-top: 8px;
}

.lw-model-inspector-subsection-box {
  border-radius: 8px;
  border: 1px solid rgba(17, 17, 17, 0.08);
  background: rgba(248, 250, 252, 0.96);
  padding: 8px;
  margin-bottom: 10px;
}

.lw-model-inspector-subsection--locomotion {
  background: rgba(248, 250, 252, 0.96);
}

.lw-model-inspector-subsection--sensation {
  background: rgba(248, 250, 252, 0.96);
}

.lw-model-inspector-subsection--feeding {
  background: rgba(248, 250, 252, 0.96);
}

.lw-model-inspector-subsection--memory {
  background: rgba(248, 250, 252, 0.96);
}

.lw-model-inspector-subsection--core {
  background: rgba(248, 250, 252, 0.96);
}

.lw-model-inspector-subsection--optional {
  background: rgba(248, 250, 252, 0.96);
}

.lw-model-inspector-live-box {
  border-radius: 10px;
  border: 1px solid rgba(17, 17, 17, 0.1);
  background: rgba(255, 255, 255, 0.98);
  padding: 10px;
}

.lw-model-inspector-live-sidebar {
  border-radius: 10px;
  border: 1px solid rgba(17, 17, 17, 0.08);
  background: rgba(193, 176, 194, 0.16);
  padding: 10px;
}

.lw-model-inspector-live-table .slick-cell,
.lw-model-inspector-live-table .slick-header-column,
.lw-model-inspector-live-table .slick-headerrow-column {
  font-size: 11px;
}

.lw-model-inspector-live-table .slick-cell {
  line-height: 1.2;
}

.lw-model-inspector-validation-warning {
  border-left: 4px solid #b7791f;
  background: rgba(251, 191, 36, 0.16);
  padding: 8px 10px;
  border-radius: 8px;
  font-size: 12px;
}

.lw-model-inspector-validation-error {
  border-left: 4px solid #b91c1c;
  background: rgba(248, 113, 113, 0.16);
  padding: 8px 10px;
  border-radius: 8px;
  font-size: 12px;
}
""".strip()


def _status_html(text: str) -> str:
    return f'<div class="lw-model-inspector-status">{escape(text)}</div>'


def _reporter_plot_label(key: str) -> str:
    try:
        entry = reg.par.kdict[key]
        desc = getattr(entry, "d", None) or key
        return f"{key} ({desc})"
    except Exception:
        return key


def _ordered_selected_reporters(widget: pn.widgets.CheckBoxGroup) -> tuple[str, ...]:
    selected = set(widget.value or ())
    if not selected:
        return tuple(DEFAULT_LIVE_PREVIEW_REPORTER_KEYS)
    return tuple(k for k in LIVE_PREVIEW_REPORTER_KEYS if k in selected)


def _json_ready(value: Any) -> Any:
    nested_conf = getattr(type(value), "nestedConf", None)
    if nested_conf is not None:
        return _json_ready(value.nestedConf)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def _coerce_like_template(template: Any, value: Any) -> Any:
    if isinstance(template, dict) and isinstance(value, dict):
        coerced: dict[Any, Any] = {}
        for key, item in value.items():
            if key in template:
                coerced[key] = _coerce_like_template(template[key], item)
            else:
                coerced[key] = item
        return coerced
    if isinstance(template, tuple) and isinstance(value, list):
        if not template:
            return tuple(value)
        if len(template) == len(value):
            return tuple(
                _coerce_like_template(template[idx], item)
                for idx, item in enumerate(value)
            )
        exemplar = template[0]
        return tuple(_coerce_like_template(exemplar, item) for item in value)
    return value


class _ModelInspectorController:
    def __init__(self, *, advanced_preset_controls: bool = False) -> None:
        model_ids = list_model_ids()
        if not model_ids:
            raise ModelInspectorError("no_models", "No model presets are available.")
        self._model_ids = model_ids
        self._advanced_preset_controls = bool(advanced_preset_controls)
        self._model_preset_workspace_available = False
        self._brain = None
        self._runtime = None
        self._callback = None
        self._draft_model_id: str | None = None
        self._draft_model: Any | None = None
        self._draft_validation_issues: tuple[DraftValidationIssue, ...] = ()
        self._is_running = False
        self._has_local_edits = False
        self._status_message = "Ready."
        self._step = 0
        self._active_dt = LIVE_DT
        self._reporter_paths: dict[str, str] = {}
        self._reporter_available: dict[str, bool] = {
            key: False for key in LIVE_PREVIEW_REPORTER_KEYS
        }
        self._watched_param_tokens: set[tuple[int, str]] = set()
        self.plot_reporters_checkbox = pn.widgets.CheckBoxGroup(
            name="Plot signals (live preview)",
            value=list(DEFAULT_LIVE_PREVIEW_REPORTER_KEYS),
            options=list(LIVE_PREVIEW_REPORTER_KEYS),
            inline=True,
        )
        self._probe_df = pd.DataFrame(
            columns=[
                "time",
                "lin",
                "ang",
                "feed_motion",
                *_ordered_selected_reporters(self.plot_reporters_checkbox),
            ]
        )

        self.primary_select = pn.widgets.Select(
            name="Primary model",
            options={model_id: model_id for model_id in model_ids},
            value="explorer" if "explorer" in model_ids else model_ids[0],
        )
        compare_options = {"(None)": ""} | {
            model_id: model_id
            for model_id in model_ids
            if model_id != self.primary_select.value
        }
        self.compare_select = pn.widgets.Select(
            name="Compare with",
            options=compare_options,
            value="",
        )
        self.run_button = pn.widgets.Button(name="Run", button_type="success")
        self.pause_button = pn.widgets.Button(name="Pause", button_type="primary")
        self.clear_trace_button = pn.widgets.Button(
            name="Clear trace", button_type="primary"
        )
        self.reset_preset_button = pn.widgets.Button(
            name="Reset to model preset",
            button_type="warning",
            sizing_mode="stretch_width",
        )
        self.max_steps_input = pn.widgets.IntInput(
            name="Max steps",
            value=LIVE_MAX_STEPS,
            start=1,
        )
        self.a_in_input = pn.widgets.FloatInput(
            name="A_in",
            value=LIVE_A_IN,
            step=0.1,
        )
        self.trace_window_input = pn.widgets.IntInput(
            name="Trace window",
            value=LIVE_ROLLOVER,
            start=1,
        )
        self.dt_input = pn.widgets.FloatInput(
            name="dt (resets runtime)",
            value=LIVE_DT,
            start=0.001,
            step=0.01,
        )

        self.primary_table = pn.pane.DataFrame(
            pd.DataFrame(),
            sizing_mode="stretch_width",
        )
        self.module_sections_box = pn.Column(sizing_mode="stretch_width")
        self._module_card_slots: dict[str, pn.Column] = {}
        self._module_specs_by_id: dict[str, ModelModuleSpec] = {}
        self.compare_table = pn.pane.DataFrame(
            pd.DataFrame(),
            sizing_mode="stretch_width",
        )
        self.compare_title = pn.pane.Markdown("", margin=(0, 0, 6, 0))
        self.summary_sections_box = pn.Column(
            sizing_mode="stretch_width",
            css_classes=["lw-model-inspector-section-box"],
        )
        self.probe_table = pn.pane.DataFrame(
            pd.DataFrame(),
            height=300,
            css_classes=["lw-model-inspector-live-table"],
        )
        self.probe_meta = pn.pane.HTML("", margin=0)
        self.live_plot_view = pn.Column(sizing_mode="stretch_width")
        self.status_pane = pn.pane.HTML(_status_html("Ready."), margin=(8, 0, 0, 0))
        self.validation_pane = pn.Column(
            sizing_mode="stretch_width", margin=(6, 0, 0, 0)
        )
        self.model_preset_controls = self._build_model_preset_controls()
        self.download_json_button = pn.widgets.FileDownload(
            name="Download JSON",
            button_type="default",
            callback=self._export_draft_json,
            filename=self._draft_download_filename(),
            sizing_mode="stretch_width",
        )

        self._sources = {
            key: ColumnDataSource(data={"time": [], key: []})
            for key in LIVE_PREVIEW_REPORTER_KEYS
        }

        self.plot_reporters_checkbox.param.watch(
            self._on_plot_reporters_change, "value"
        )
        self.primary_select.param.watch(self._on_primary_change, "value")
        self.compare_select.param.watch(self._on_compare_change, "value")
        self.run_button.on_click(self._on_run)
        self.pause_button.on_click(self._on_pause)
        self.clear_trace_button.on_click(self._on_clear_trace)
        self.reset_preset_button.on_click(self._on_reset_to_preset)
        self.max_steps_input.param.watch(self._on_live_preview_setting_change, "value")
        self.a_in_input.param.watch(self._on_live_preview_setting_change, "value")
        self.trace_window_input.param.watch(
            self._on_live_preview_setting_change, "value"
        )
        self.dt_input.param.watch(self._on_dt_change, "value")
        self._reset_draft_to_selected_model()
        self._refresh_model_preset_controls()
        self._refresh_draft_validation()
        if not self._has_validation_errors():
            self._ensure_brain_for_selected_model()
        self._refresh_preview_controls()
        self._set_status(self._with_validation_status("Ready."))
        self._refresh_inspection()
        self._init_live_plots()
        self._update_probe_meta()
        self._update_summary_sections()

    def _set_status(self, message: str) -> None:
        self._status_message = message
        self.status_pane.object = _status_html(message)

    def _set_running(self, value: bool) -> None:
        self._is_running = value
        self._refresh_preview_controls()

    def _model_preset_workspace_dir(self) -> Path:
        return get_workspace_dir("metadata") / "model_presets"

    def _fallback_model_preset_workspace_dir(self) -> Path:
        return Path.cwd() / ".larvaworld_model_presets_unavailable"

    def _draft_payload_for_storage(self, _name: str | None = None) -> dict[str, Any]:
        return _json_ready(self._require_draft_model())

    def _draft_json_text(self) -> str:
        return json.dumps(self._draft_payload_for_storage(), indent=2) + "\n"

    def _export_draft_json(self) -> io.StringIO:
        return io.StringIO(self._draft_json_text())

    def _draft_download_filename(self) -> str:
        base = str(self.primary_select.value or "model").strip() or "model"
        safe = WorkspacePresetStore.normalize_name(base)
        return f"{safe}_draft.json"

    def _build_model_preset_controls(self) -> PresetControlsController:
        policy = (
            ADVANCED_PRESET_POLICY
            if self._advanced_preset_controls
            else USER_PRESET_POLICY
        )
        workspace_dir = self._fallback_model_preset_workspace_dir()
        try:
            workspace_dir = self._model_preset_workspace_dir()
            self._model_preset_workspace_available = True
        except WorkspaceError:
            self._model_preset_workspace_available = False
        kwargs: dict[str, Any] = {}
        if self._advanced_preset_controls:
            kwargs["build_registry_payload"] = self._draft_payload_for_storage
        return PresetControlsController(
            conftype="Model",
            workspace_store=WorkspacePresetStore(
                workspace_dir,
                directory_key="model-inspector-models",
            ),
            policy=policy,
            build_workspace_payload=self._draft_payload_for_storage,
            on_load=self._on_model_preset_loaded,
            on_save=self._on_model_preset_saved,
            on_status=self._on_model_preset_status,
            title="Model Presets",
            preset_name_after_refresh=True,
            **kwargs,
        )

    def _refresh_model_preset_controls(self) -> None:
        try:
            workspace_store = WorkspacePresetStore(
                self._model_preset_workspace_dir(),
                directory_key="model-inspector-models",
            )
            self.model_preset_controls.workspace_store = workspace_store
            self.model_preset_controls.load_button.disabled = False
            self.model_preset_controls.save_button.disabled = False
            self.model_preset_controls.delete_button.disabled = False
            self._model_preset_workspace_available = True
            self.model_preset_controls.refresh_list()
        except WorkspaceError as exc:
            self.model_preset_controls.workspace_store = WorkspacePresetStore(
                self._fallback_model_preset_workspace_dir(),
                directory_key="model-inspector-models",
            )
            self.model_preset_controls.load_button.disabled = True
            self.model_preset_controls.save_button.disabled = True
            self.model_preset_controls.delete_button.disabled = True
            self._model_preset_workspace_available = False
            self.model_preset_controls.refresh_list()
            self.model_preset_controls._set_status(
                f"Workspace model presets unavailable: {exc}",
                tone="danger",
            )

    def _on_model_preset_status(self, message: str, *, tone: str = "neutral") -> None:
        if tone in {"warning", "danger"}:
            self._set_status(self._with_validation_status(message))

    def _on_model_preset_saved(self, ref: PresetRef, payload: Any) -> None:
        del payload
        self._set_status(self._with_validation_status(f"Saved {ref.display_label}."))

    def _on_model_preset_loaded(self, ref: PresetRef, payload: Any) -> None:
        self._replace_draft_from_loaded_preset(ref, payload)

    def _replace_draft_from_loaded_preset(self, ref: PresetRef, payload: Any) -> None:
        copied = util.AttrDict(payload).get_copy()
        brain_payload = copied.get("brain") if isinstance(copied, Mapping) else None
        if not isinstance(copied, Mapping) or not isinstance(brain_payload, Mapping):
            raise ModelInspectorError(
                "invalid_model_preset",
                f'Loaded preset "{ref.display_label}" is missing a valid "brain" payload.',
            )
        template = load_model_draft(str(self.primary_select.value))
        self._draft_model = util.AttrDict(
            _coerce_like_template(template, copied)
        ).get_copy()
        self._draft_model_id = str(self.primary_select.value)
        if hasattr(self.model_preset_controls, "preset_name"):
            self.model_preset_controls.preset_name.value = ref.name
        self._sync_preview_after_draft_change(
            message=f"Loaded {ref.display_label}.",
            clear_trace=True,
            mark_dirty=True,
            ui_scope="full",
        )

    def _on_primary_change(self, _event=None) -> None:
        self._pause_callback()
        self._reset_draft_to_selected_model()
        self._refresh_model_preset_controls()
        self.download_json_button.filename = self._draft_download_filename()
        compare_options = {"(None)": ""} | {
            model_id: model_id
            for model_id in self._model_ids
            if model_id != self.primary_select.value
        }
        previous = self.compare_select.value
        self.compare_select.options = compare_options
        self.compare_select.value = (
            previous if previous in compare_options.values() else ""
        )
        self._sync_preview_after_draft_change(
            message=f'Model changed to "{self.primary_select.value}".',
            clear_trace=True,
            mark_dirty=False,
            ui_scope="full",
        )

    def _on_compare_change(self, _event=None) -> None:
        self._refresh_inspection_tables()

    def _on_run(self, _event=None) -> None:
        if self._is_running:
            return
        self._refresh_draft_validation()
        if self._has_validation_errors():
            self._brain = None
            self._runtime = None
            self._refresh_preview_controls()
            self._set_status(
                self._with_validation_status(
                    "Live preview blocked by draft validation errors."
                )
            )
            return
        if self._brain is None:
            self._ensure_brain_for_selected_model()
        self._start_callback()
        if self._draft_validation_issues:
            self._set_status(
                self._with_validation_status(
                    "Live preview running with validation warnings."
                )
            )
        else:
            self._set_status(
                f'Live preview running for model "{self.primary_select.value}".'
            )

    def _on_pause(self, _event=None) -> None:
        self._pause_callback()
        self._set_status(f"Live preview paused at step {self._step}.")

    def _on_clear_trace(self, _event=None) -> None:
        self._clear_trace_data()
        self._update_probe_meta()
        self._set_status(
            self._with_validation_status(
                "Trace cleared. Local edited runtime state preserved."
            )
        )

    def _on_reset_to_preset(self, _event=None) -> None:
        self._pause_callback()
        self._reset_draft_to_selected_model()
        self._has_local_edits = False
        self._sync_preview_after_draft_change(
            message=f'Reset local state to model preset "{self.primary_select.value}".',
            clear_trace=True,
            mark_dirty=False,
            ui_scope="full",
        )

    def _start_callback(self) -> None:
        self._pause_callback()
        self._set_running(True)
        self._callback = pn.state.add_periodic_callback(
            self._tick_live_preview, period=40
        )

    def _pause_callback(self) -> None:
        callback = self._callback
        self._callback = None
        if callback is not None:
            callback.stop()
        self._set_running(False)

    def _ensure_brain_for_selected_model(self) -> None:
        self._active_dt = self._dt()
        draft = self._require_draft_model()
        self._brain = build_inspection_brain_from_config(
            str(self.primary_select.value), draft, dt=self._active_dt
        )
        self._runtime = SimpleNamespace(brain=self._brain)
        self._watched_param_tokens.clear()
        self._prepare_reporters()

    def _reset_draft_to_selected_model(self) -> None:
        model_id = str(self.primary_select.value)
        self._draft_model = load_model_draft(model_id)
        self._draft_model_id = model_id
        self._draft_validation_issues = ()
        self._has_local_edits = False

    def _require_draft_model(self) -> Any:
        if self._draft_model is None:
            raise ModelInspectorError(
                "draft_not_initialized", "Model draft is not initialized."
            )
        return self._draft_model

    def _refresh_draft_validation(self) -> tuple[DraftValidationIssue, ...]:
        draft = self._require_draft_model()
        self._draft_validation_issues = validate_draft_module_config(draft)
        self._refresh_validation_pane()
        self._refresh_preview_controls()
        return self._draft_validation_issues

    def _has_validation_errors(self) -> bool:
        return any(issue.severity == "error" for issue in self._draft_validation_issues)

    def _validation_counts(self) -> tuple[int, int]:
        errors = sum(
            1 for issue in self._draft_validation_issues if issue.severity == "error"
        )
        warnings = sum(
            1 for issue in self._draft_validation_issues if issue.severity == "warning"
        )
        return errors, warnings

    def _validation_summary_text(self) -> str:
        errors, warnings = self._validation_counts()
        if errors == 0 and warnings == 0:
            return ""
        return f"Validation: {errors} error(s), {warnings} warning(s)."

    def _validation_detail_text(self) -> str:
        if not self._draft_validation_issues:
            return ""
        return " ".join(
            f"[{issue.severity}] {issue.module_id}: {issue.message}"
            for issue in self._draft_validation_issues
        )

    def _with_validation_status(self, message: str) -> str:
        summary = self._validation_summary_text()
        details = self._validation_detail_text()
        return " ".join(bit for bit in (message, summary, details) if bit).strip()

    def _refresh_validation_pane(self) -> None:
        if not self._draft_validation_issues:
            self.validation_pane.objects = []
            return
        panes: list[pn.viewable.Viewable] = []
        for issue in self._draft_validation_issues:
            css = (
                "lw-model-inspector-validation-error"
                if issue.severity == "error"
                else "lw-model-inspector-validation-warning"
            )
            level = "error" if issue.severity == "error" else "warning"
            panes.append(
                pn.pane.Markdown(
                    f"Validation {level}: {issue.message}",
                    css_classes=[css],
                    sizing_mode="stretch_width",
                    margin=(0, 0, 4, 0),
                )
            )
        self.validation_pane.objects = panes

    def _refresh_preview_controls(self) -> None:
        has_errors = self._has_validation_errors()
        self.run_button.disabled = self._is_running or has_errors
        self.pause_button.disabled = not self._is_running
        self.dt_input.disabled = self._is_running
        if has_errors:
            self.run_button.button_type = "danger"
        elif self._draft_validation_issues:
            self.run_button.button_type = "warning"
        else:
            self.run_button.button_type = "success"

    def _sync_preview_after_draft_change(
        self,
        *,
        message: str,
        clear_trace: bool = True,
        mark_dirty: bool,
        ui_scope: UIRefreshScope = "full",
        module_id: str | None = None,
    ) -> None:
        if ui_scope not in {"parameter", "mode", "enabled", "full"}:
            raise ValueError(f"Unsupported UI refresh scope: {ui_scope!r}")
        previous_issues = self._draft_validation_issues
        self._pause_callback()
        if mark_dirty:
            self._has_local_edits = True
        self._refresh_draft_validation()
        module_specs = self._refresh_inspection_tables()
        if module_specs is not None:
            self._refresh_targeted_module_cards(
                module_specs=module_specs,
                module_id=module_id,
                ui_scope=ui_scope,
                previous_issues=previous_issues,
            )
        if self._has_validation_errors():
            self._brain = None
            self._runtime = None
            self._reporter_available = {
                key: False for key in LIVE_PREVIEW_REPORTER_KEYS
            }
            if clear_trace:
                self._clear_trace_data()
            self._update_probe_meta()
            self._set_status(
                self._with_validation_status(
                    f"{message} Preview blocked by draft validation errors."
                )
            )
            return
        self._ensure_brain_for_selected_model()
        if clear_trace:
            self._clear_trace_data()
        self._update_probe_meta()
        if self._draft_validation_issues:
            self._set_status(
                self._with_validation_status(
                    f"{message} Preview rebuilt with validation warnings."
                )
            )
        else:
            self._set_status(f"{message} Preview rebuilt from current draft.")

    def _set_module_enabled(self, module_id: str, enabled: bool) -> None:
        draft = self._require_draft_model()
        set_draft_module_enabled(draft, module_id, enabled)
        action = "enabled" if enabled else "disabled"
        self._sync_preview_after_draft_change(
            message=f'Module "{module_id}" {action}.',
            clear_trace=True,
            mark_dirty=True,
            ui_scope="enabled",
            module_id=module_id,
        )

    def _set_brain_module_mode(self, module_id: str, mode: str) -> None:
        draft = self._require_draft_model()
        set_draft_brain_module_mode(draft, module_id, mode)
        self._sync_preview_after_draft_change(
            message=f'Module "{module_id}" mode changed to "{mode}".',
            clear_trace=True,
            mark_dirty=True,
            ui_scope="mode",
            module_id=module_id,
        )

    def _set_memory_mode(self, mode: str) -> None:
        draft = self._require_draft_model()
        set_draft_memory_config(draft, enabled=True, mode=mode, modality=None)
        self._sync_preview_after_draft_change(
            message=f'Memory mode changed to "{mode}".',
            clear_trace=True,
            mark_dirty=True,
            ui_scope="mode",
            module_id="memory",
        )

    def _set_memory_modality(self, modality: str) -> None:
        draft = self._require_draft_model()
        current_memory = draft.brain["memory"]
        current_mode = current_memory["mode"] if current_memory is not None else None
        set_draft_memory_config(
            draft, enabled=True, mode=current_mode, modality=modality
        )
        self._sync_preview_after_draft_change(
            message=f'Memory modality changed to "{modality}".',
            clear_trace=True,
            mark_dirty=True,
            ui_scope="mode",
            module_id="memory",
        )

    def _set_module_parameter(
        self,
        module_id: str,
        parameter_path: tuple[str, ...],
        value: Any,
    ) -> None:
        draft = self._require_draft_model()
        set_draft_module_parameter(draft, module_id, parameter_path, value)
        path_label = ".".join(parameter_path)
        self._sync_preview_after_draft_change(
            message=f'Module "{module_id}" parameter "{path_label}" changed.',
            clear_trace=True,
            mark_dirty=True,
            ui_scope="parameter",
            module_id=module_id,
        )

    def _selected_plot_reporter_keys(self) -> tuple[str, ...]:
        return _ordered_selected_reporters(self.plot_reporters_checkbox)

    def _on_plot_reporters_change(self, event) -> None:
        if not self.plot_reporters_checkbox.value:
            if event.old:
                self.plot_reporters_checkbox.value = list(
                    DEFAULT_LIVE_PREVIEW_REPORTER_KEYS
                )
            return
        self._pause_callback()
        self._step = 0
        for key in LIVE_PREVIEW_REPORTER_KEYS:
            self._sources[key].data = {"time": [], key: []}
        self._probe_df = pd.DataFrame(
            columns=[
                "time",
                "lin",
                "ang",
                "feed_motion",
                *self._selected_plot_reporter_keys(),
            ]
        )
        self._refresh_probe_table()
        self._init_live_plots()
        if (
            self._brain is not None
            and self._runtime is not None
            and not self._has_validation_errors()
        ):
            self._prepare_reporters()
        self._update_probe_meta()

    def _prepare_reporters(self) -> None:
        assert self._runtime is not None
        available = reg.par.output_reporters(
            ks=list(self._selected_plot_reporter_keys()), agents=[self._runtime]
        )
        available_paths = set(available.values())
        reporter_paths: dict[str, str] = {}
        reporter_available: dict[str, bool] = {}
        for key in self._selected_plot_reporter_keys():
            try:
                path = reg.par.kdict[key].codename
            except Exception:
                path = ""
            reporter_paths[key] = path
            reporter_available[key] = bool(path) and path in available_paths
        self._reporter_paths = reporter_paths
        self._reporter_available = reporter_available

    def _tick_live_preview(self) -> None:
        if self._brain is None or self._runtime is None:
            self._pause_callback()
            return
        if self._step >= self._max_steps():
            self._pause_callback()
            self._set_status(f"Live preview auto-stopped at step {self._max_steps()}.")
            return

        lin, ang, feed_motion = self._brain.locomotor.step(A_in=self._a_in())
        time_now = self._step * self._active_dt
        row: dict[str, Any] = {
            "time": time_now,
            "lin": lin,
            "ang": ang,
            "feed_motion": bool(feed_motion),
        }
        for key, path in self._reporter_paths.items():
            if not self._reporter_available.get(key, False):
                row[key] = None
                continue
            try:
                row[key] = util.rgetattr(self._runtime, path)
            except Exception:
                self._reporter_available[key] = False
                row[key] = None

        for key in self._selected_plot_reporter_keys():
            value = row.get(key)
            if value is None:
                continue
            self._sources[key].stream(
                {"time": [time_now], key: [float(value)]},
                rollover=self._trace_window(),
            )

        self._probe_df = pd.concat(
            [self._probe_df, pd.DataFrame([row])], ignore_index=True
        )
        self._probe_df = self._probe_df.tail(self._trace_window()).reset_index(
            drop=True
        )
        self._refresh_probe_table()
        self._step += 1
        self._update_probe_meta()

    def _clear_trace_data(self) -> None:
        self._step = 0
        for key in LIVE_PREVIEW_REPORTER_KEYS:
            self._sources[key].data = {"time": [], key: []}
        self._probe_df = pd.DataFrame(
            columns=[
                "time",
                "lin",
                "ang",
                "feed_motion",
                *self._selected_plot_reporter_keys(),
            ]
        )
        self._refresh_probe_table()

    def _refresh_probe_table(self) -> None:
        rename = {
            k: _reporter_plot_label(k)
            for k in self._selected_plot_reporter_keys()
            if k in self._probe_df.columns
        }
        self.probe_table.object = self._probe_df.rename(columns=rename)

    def _max_steps(self) -> int:
        return max(1, int(self.max_steps_input.value))

    def _a_in(self) -> float:
        return float(self.a_in_input.value)

    def _trace_window(self) -> int:
        return max(1, int(self.trace_window_input.value))

    def _dt(self) -> float:
        return max(0.001, float(self.dt_input.value))

    def _trim_trace_data(self) -> None:
        trace_window = self._trace_window()
        for key in self._selected_plot_reporter_keys():
            source_data = self._sources[key].data
            self._sources[key].data = {
                "time": list(source_data["time"])[-trace_window:],
                key: list(source_data[key])[-trace_window:],
            }
        self._probe_df = self._probe_df.tail(trace_window).reset_index(drop=True)
        self._refresh_probe_table()

    def _on_live_preview_setting_change(self, _event=None) -> None:
        self._trim_trace_data()
        self._update_probe_meta()
        if self._is_running and self._step >= self._max_steps():
            self._pause_callback()
            self._set_status(f"Live preview auto-stopped at step {self._max_steps()}.")

    def _on_dt_change(self, _event=None) -> None:
        if self._is_running:
            return
        self._sync_preview_after_draft_change(
            message=f"dt changed to {self._dt()}.",
            clear_trace=True,
            mark_dirty=False,
            ui_scope="full",
        )

    def _update_probe_meta(self) -> None:
        if self._brain is None or self._runtime is None:
            runtime_state = "unavailable"
        elif self._has_validation_errors():
            runtime_state = "blocked by validation errors"
        elif self._draft_validation_issues:
            runtime_state = "ready with validation warnings"
        elif self._is_running:
            runtime_state = "running"
        else:
            runtime_state = "ready"
        reporter_bits = [
            f'{_reporter_plot_label(k)}={"yes" if self._reporter_available.get(k, False) else "no"}'
            for k in self._selected_plot_reporter_keys()
        ]
        self.probe_meta.object = (
            '<div class="lw-model-inspector-status">'
            f"<strong>Preview runtime:</strong> {runtime_state}<br>"
            f"<strong>Preview settings:</strong> dt={self._active_dt}, a_in={self._a_in()}, rollover={self._trace_window()}, max_steps={self._max_steps()}<br>"
            f"<strong>Current step:</strong> {self._step}<br>"
            f"<strong>Reporter availability:</strong> {'; '.join(reporter_bits)}"
            "</div>"
        )

    def _update_summary_sections(self) -> None:
        children: list[pn.viewable.Viewable] = [
            pn.pane.Markdown("#### Configured modules (summary)", margin=(0, 0, 6, 0)),
            self.primary_table,
        ]
        if self.compare_title.object:
            children.append(pn.Spacer(height=8))
            children.append(self.compare_title)
        if not self.compare_table.object.empty:
            children.append(self.compare_table)
        self.summary_sections_box.objects = children

    def _refresh_inspection(self) -> None:
        self._refresh_draft_validation()
        module_specs = self._refresh_inspection_tables()
        if module_specs is not None:
            self._refresh_all_module_cards(module_specs)

    def _refresh_inspection_tables(self) -> tuple[ModelModuleSpec, ...] | None:
        primary_id = str(self.primary_select.value)
        try:
            draft = self._require_draft_model()
            primary = inspect_model_from_config(primary_id, draft)
            module_specs = inspect_model_modules_from_config(primary_id, draft)
        except ModelInspectorError as exc:
            self._set_status(f"Inspection failed ({exc.code}): {exc}")
            self.primary_table.object = pd.DataFrame()
            self.compare_table.object = pd.DataFrame()
            self.compare_title.object = ""
            self.module_sections_box.objects = [
                pn.pane.Markdown("Module inspection unavailable.", margin=0)
            ]
            self._module_card_slots.clear()
            self._module_specs_by_id.clear()
            self._update_summary_sections()
            return None

        self.primary_table.object = _modules_to_dataframe(
            primary.baseline_modules, primary.optional_modules
        )
        self._module_specs_by_id = {spec.module_id: spec for spec in module_specs}

        if self._has_local_edits:
            self.compare_select.disabled = True
            self.compare_title.object = "#### Comparison hidden during local edits"
            self.compare_table.object = pd.DataFrame()
            self._update_summary_sections()
            return module_specs
        self.compare_select.disabled = False

        compare_id = str(self.compare_select.value or "")
        if not compare_id:
            self.compare_title.object = ""
            self.compare_table.object = pd.DataFrame()
            self._update_summary_sections()
            return module_specs

        try:
            comparison = inspect_model(compare_id)
            diffs = compare_model_inspections(primary, comparison)
        except ModelInspectorError as exc:
            self._set_status(f"Comparison failed ({exc.code}): {exc}")
            self.compare_title.object = ""
            self.compare_table.object = pd.DataFrame()
            self._update_summary_sections()
            return module_specs

        self.compare_title.object = f"#### Comparison: `{primary_id}` vs `{compare_id}`"
        self.compare_table.object = pd.DataFrame(
            [
                {
                    "Module": item.module_id,
                    "Primary present": item.primary.present,
                    "Comparison present": item.comparison.present,
                    "Primary mode": item.primary.mode or "—",
                    "Comparison mode": item.comparison.mode or "—",
                    "Changed fields": ", ".join(item.changed_fields)
                    if item.changed_fields
                    else "none",
                    "Equal": item.equal,
                }
                for item in diffs
            ]
        )
        self._update_summary_sections()
        return module_specs

    def _refresh_all_module_cards(
        self,
        module_specs: tuple[ModelModuleSpec, ...],
    ) -> None:
        self._module_card_slots.clear()
        self._module_specs_by_id = {spec.module_id: spec for spec in module_specs}
        self.module_sections_box.objects = _build_module_sections(
            controller=self,
            specs=module_specs,
            validation_issues=self._draft_validation_issues,
            card_slots=self._module_card_slots,
        )

    def _refresh_targeted_module_cards(
        self,
        *,
        module_specs: tuple[ModelModuleSpec, ...],
        module_id: str | None,
        ui_scope: UIRefreshScope,
        previous_issues: tuple[DraftValidationIssue, ...],
    ) -> None:
        self._module_specs_by_id = {spec.module_id: spec for spec in module_specs}
        if ui_scope == "full":
            self._refresh_all_module_cards(module_specs)
            return
        if ui_scope == "parameter":
            affected_cards: set[str] = set()
        elif ui_scope in {"mode", "enabled"}:
            affected_cards = {module_id} if module_id else set()
        else:
            raise ValueError(f"Unsupported UI refresh scope: {ui_scope!r}")

        if module_id in {"memory", "olfactor", "toucher"} and ui_scope in {
            "mode",
            "enabled",
        }:
            affected_cards.add("memory")

        old_sig = _validation_issue_signature(previous_issues)
        new_sig = _validation_issue_signature(self._draft_validation_issues)
        changed_issue_modules = {
            m_id
            for m_id in set(old_sig) | set(new_sig)
            if old_sig.get(m_id, ()) != new_sig.get(m_id, ())
        }
        affected_cards |= changed_issue_modules

        for affected_module_id in affected_cards:
            spec = self._module_specs_by_id.get(affected_module_id)
            slot = self._module_card_slots.get(affected_module_id)
            if spec is None or slot is None:
                self._refresh_all_module_cards(module_specs)
                return
            new_card = _module_editor_card(
                controller=self,
                spec=spec,
                validation_issues=self._draft_validation_issues,
            )
            slot.objects = [new_card]

    def _build_settings_cards(self, inspection) -> list[pn.viewable.Viewable]:
        if self._brain is None:
            return [
                pn.Card(
                    pn.pane.Markdown("Could not build transient runtime brain."),
                    title="Module settings unavailable",
                    sizing_mode="stretch_width",
                )
            ]

        inspections = {
            module.module_id: module
            for module in (*inspection.baseline_modules, *inspection.optional_modules)
        }
        ordered_ids = [
            module_id for module_id in BASELINE_MODULES if module_id in inspections
        ]
        return [
            _module_settings_card(
                brain=self._brain,
                module_id=module_id,
                inspection=inspections[module_id],
                editable=module_id in BASELINE_MODULES
                and inspections[module_id].present,
                on_edit=self._on_local_parameter_edit,
                watched_tokens=self._watched_param_tokens,
            )
            for module_id in ordered_ids
        ]

    def _on_local_parameter_edit(self, *_args: Any, **_kwargs: Any) -> None:
        # Legacy compatibility path for old runtime-object editor; visible UI uses draft helpers.
        self._has_local_edits = True
        self._refresh_inspection()
        if self._is_running:
            self._set_status(
                f"Local parameters changed at step {self._step}; live preview continues."
            )
        else:
            self._set_status("Local parameters changed. Press Run to preview.")

    def _init_live_plots(self) -> None:
        plots: list[pn.viewable.Viewable] = []
        for reporter in self._selected_plot_reporter_keys():
            reporter_label = _reporter_plot_label(reporter)
            fig = figure(
                title=reporter_label,
                height=220,
                width=900,
                x_axis_label="time (sec)",
                y_axis_label=reporter_label,
                tools="pan,wheel_zoom,box_zoom,save,reset",
                active_drag=None,
                sizing_mode="stretch_width",
            )
            fig.line("time", reporter, source=self._sources[reporter], line_width=2)
            plots.append(pn.pane.Bokeh(fig, sizing_mode="stretch_width"))
        self.live_plot_view.objects = plots

    def view(self) -> pn.viewable.Viewable:
        intro = pn.pane.HTML(
            (
                '<div class="lw-model-inspector-intro">'
                "Inspect canonical larva model presets, edit baseline modules locally, and run live reporter preview."
                "</div>"
            ),
            margin=0,
        )
        action_buttons = pn.Column(
            pn.Row(
                self.run_button,
                self.pause_button,
                self.clear_trace_button,
                sizing_mode="stretch_width",
            ),
            self.reset_preset_button,
            sizing_mode="stretch_width",
        )
        primary_controls = pn.Column(
            self.primary_select,
            self.compare_select,
            self.status_pane,
            self.validation_pane,
            pn.pane.Markdown("#### Preview settings", margin=(8, 0, 2, 0)),
            self.max_steps_input,
            self.a_in_input,
            self.trace_window_input,
            self.dt_input,
            action_buttons,
            sizing_mode="stretch_width",
            css_classes=["lw-model-inspector-controls-box"],
        )
        preset_controls = pn.Column(
            pn.pane.Markdown("#### Draft presets", margin=(0, 0, 2, 0)),
            self.model_preset_controls.view,
            self.download_json_button,
            sizing_mode="stretch_width",
            css_classes=["lw-model-inspector-controls-box"],
        )
        controls_box = pn.Column(
            primary_controls,
            preset_controls,
            width=CONTROLS_COLUMN_WIDTH,
            styles={
                "flex": f"0 0 {CONTROLS_COLUMN_WIDTH}px",
                "margin-bottom": "10px",
            },
        )
        probe_sidebar = pn.Column(
            self.probe_meta,
            self.probe_table,
            sizing_mode="stretch_width",
            css_classes=["lw-model-inspector-live-sidebar"],
            styles={
                "flex": "0 0 auto",
                "margin-top": "6px",
            },
        )
        probe_body = pn.Column(
            self.plot_reporters_checkbox,
            self.live_plot_view,
            probe_sidebar,
            sizing_mode="stretch_width",
        )
        probe = pn.Column(
            pn.pane.Markdown("#### Live preview", margin=(0, 0, 6, 0)),
            probe_body,
            css_classes=["lw-model-inspector-live-box"],
            sizing_mode="stretch_width",
        )
        probe_cell = pn.Column(
            probe,
            sizing_mode="stretch_width",
            styles={
                "margin-left": f"{SECTION_COLUMN_GAP_PX}px",
                "flex": "1 1 0",
                "min-width": "0",
            },
        )
        top_row = pn.Row(
            controls_box,
            probe_cell,
            sizing_mode="stretch_width",
            styles={"align-items": "flex-start"},
        )
        return pn.Column(
            intro,
            top_row,
            self.module_sections_box,
            self.summary_sections_box,
            css_classes=["lw-model-inspector-root"],
            sizing_mode="stretch_width",
        )


def _modules_to_dataframe(
    baseline_modules: tuple[ModuleInspection, ...],
    optional_modules: tuple[ModuleInspection, ...],
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Category": "Baseline" if module.is_baseline else "Optional",
                "Module": module.module_id,
                "Present": module.present,
                "Mode": module.mode or "—",
                "Parameters": repr(module.parameters),
            }
            for module in (*baseline_modules, *optional_modules)
        ]
    )


def _module_settings_card(
    *,
    brain,
    module_id: str,
    inspection: ModuleInspection,
    editable: bool,
    on_edit,
    watched_tokens: set[tuple[int, str]],
) -> pn.Card:
    if not inspection.present:
        body = pn.pane.Markdown("Not configured in this model.", margin=0)
        title = f"{inspection.display_name} | absent"
        return pn.Card(body, title=title, sizing_mode="stretch_width")

    module_obj = None
    if module_id in MD.LocoMods:
        module_obj = getattr(brain.locomotor, module_id, None)
    elif module_id in MD.ids:
        module_obj = getattr(brain, module_id, None)

    if module_obj is None:
        body = pn.pane.Markdown(
            "Configured, but runtime module is unavailable.", margin=0
        )
        title = f"{inspection.display_name} | unavailable"
        return pn.Card(body, title=title, sizing_mode="stretch_width")

    parameter_names = class_objs(
        module_obj.__class__, excluded=[Effector, "phi", "name"]
    ).keylist
    parameter_names = [name for name in parameter_names if name != "mode"]
    widgets = {}
    if not editable:
        widgets = {name: {"disabled": True} for name in parameter_names}
    pane = pn.Param(
        module_obj,
        parameters=parameter_names,
        widgets=widgets,
        show_name=False,
        expand_button=True,
        default_precedence=3,
        sizing_mode="stretch_width",
    )
    if editable:
        for param_name in parameter_names:
            token = (id(module_obj), param_name)
            if token in watched_tokens:
                continue
            try:
                module_obj.param.watch(on_edit, param_name)
                watched_tokens.add(token)
            except Exception:
                continue

    module_name = getattr(module_obj.__class__, "name", module_obj.__class__.__name__)
    edit_tag = "editable" if editable else "read-only"
    title = f"{module_id} | {module_name} | {edit_tag}"
    return pn.Card(pane, title=title, sizing_mode="stretch_width")


def _module_spec_title(spec: ModelModuleSpec) -> str:
    if not spec.present:
        return f"{spec.module_id} | absent"
    if spec.module_id == "memory":
        if spec.current_mode and spec.current_modality:
            return (
                f"memory | {spec.current_mode} / {spec.current_modality} | configured"
            )
        if spec.current_mode:
            return f"memory | {spec.current_mode} | configured"
        return "memory | configured"
    if spec.current_mode:
        return f"{spec.module_id} | {spec.current_mode} | configured"
    return f"{spec.module_id} | configured"


_PROTECTED_PARAMETER_ROOTS = {"mode", "modality", "name"}


def _module_draft_config(spec: ModelModuleSpec) -> Any:
    if spec.module_kind == "brain":
        return spec.parameters if spec.present else None
    if spec.module_kind == "memory":
        return spec.parameters if spec.present else None
    if spec.module_kind == "larva":
        return spec.parameters if spec.present else None
    return None


def _clone_parameter_object(
    parameter: param.Parameter, default: Any
) -> param.Parameter:
    clone = copy.copy(parameter)
    try:
        clone.default = default
    except Exception:
        pass
    return clone


def _is_supported_parameter_for_editor(parameter: param.Parameter) -> bool:
    # Dict-valued parameters (e.g. intermitter distribution blobs) need
    # dedicated editors; rendering them as scalar controls raises Param errors.
    return not isinstance(parameter, param.Dict)


def _make_parameter_proxy(
    *,
    module_id: str,
    scope: str,
    parameter_objects: dict[str, param.Parameter],
    values: dict[str, Any],
) -> param.Parameterized:
    attrs: dict[str, param.Parameter] = {}
    for name, pobj in parameter_objects.items():
        if name.split(".", 1)[0] in _PROTECTED_PARAMETER_ROOTS:
            continue
        if not _is_supported_parameter_for_editor(pobj):
            continue
        if name not in values:
            continue
        default = values.get(name, getattr(pobj, "default", None))
        attrs[name] = _clone_parameter_object(pobj, default)
    proxy_cls = type(
        f"ModelInspectorProxy_{module_id}_{scope}",
        (param.Parameterized,),
        attrs,
    )
    kwargs = {name: values[name] for name in attrs if name in values}
    return proxy_cls(**kwargs)


def _parameter_editor_group(
    *,
    controller: _ModelInspectorController,
    module_id: str,
    title: str,
    parameter_objects: dict[str, param.Parameter],
    values: dict[str, Any],
    path_prefix: tuple[str, ...] = (),
) -> pn.viewable.Viewable:
    names = [
        name
        for name in parameter_objects
        if (
            name
            and name.split(".", 1)[0] not in _PROTECTED_PARAMETER_ROOTS
            and _is_supported_parameter_for_editor(parameter_objects[name])
            and name in values
        )
    ]
    if not names:
        return pn.pane.Markdown("No editable parameters.", margin=(2, 0, 0, 0))

    proxy = _make_parameter_proxy(
        module_id=module_id,
        scope=title.replace(" ", "_"),
        parameter_objects=parameter_objects,
        values=values,
    )
    controls = param_controls(obj=proxy, parameters=names)
    for control in getattr(controls, "objects", []):
        widgets = getattr(control, "_widgets", {})
        if not isinstance(widgets, dict):
            continue
        for parameter_name, widget in widgets.items():
            if parameter_name not in names:
                continue
            widget.name = (
                ".".join((*path_prefix, parameter_name))
                if path_prefix
                else parameter_name
            )

    for name in names:

        def _on_change(event, *, parameter_name=name) -> None:
            if event.old == event.new:
                return
            controller._set_module_parameter(
                module_id,
                (*path_prefix, parameter_name),
                event.new,
            )

        proxy.param.watch(_on_change, name)

    return pn.Column(
        pn.pane.Markdown(f"**{title}**", margin=(4, 0, 2, 0)),
        controls,
        sizing_mode="stretch_width",
    )


def _canonical_editor_groups_for_spec(
    *,
    controller: _ModelInspectorController,
    spec: ModelModuleSpec,
) -> list[pn.viewable.Viewable]:
    if not spec.present:
        return []

    parameter_views: list[pn.viewable.Viewable] = []
    parameters = dict(spec.parameters or {})

    if spec.module_kind == "brain":
        if not spec.current_mode:
            return []
        objects = dict(
            MD.module_objects(
                mID=spec.module_id, mode=spec.current_mode, as_entry=False
            )
        )
        parameter_views.append(
            _parameter_editor_group(
                controller=controller,
                module_id=spec.module_id,
                title="Parameters",
                parameter_objects=objects,
                values=parameters,
            )
        )
        return parameter_views

    if spec.module_kind == "memory":
        if not spec.current_mode or not spec.current_modality:
            return []
        cls = MD.get_memory_class(spec.current_mode, spec.current_modality)
        if cls is None:
            return []
        objects = dict(class_objs(cls, excluded=["dt"]))
        parameter_views.append(
            _parameter_editor_group(
                controller=controller,
                module_id="memory",
                title="Parameters",
                parameter_objects=objects,
                values=parameters,
            )
        )
        return parameter_views

    if spec.module_kind == "larva":
        if spec.module_id == "body":
            objects = dict(
                class_objs(
                    agents.LarvaSegmented,
                    excluded=[
                        agents.OrientedAgent,
                        "vertices",
                        "base_vertices",
                        "width",
                        "guide_points",
                        "segs",
                    ],
                )
            )
            parameter_views.append(
                _parameter_editor_group(
                    controller=controller,
                    module_id=spec.module_id,
                    title="Parameters",
                    parameter_objects=objects,
                    values=parameters,
                )
            )
        elif spec.module_id == "physics":
            objects = dict(class_objs(agents.BaseController))
            parameter_views.append(
                _parameter_editor_group(
                    controller=controller,
                    module_id=spec.module_id,
                    title="Parameters",
                    parameter_objects=objects,
                    values=parameters,
                )
            )
        elif spec.module_id == "sensorimotor":
            objects = dict(
                class_objs(agents.ObstacleLarvaRobot, excluded=[agents.LarvaRobot])
            )
            parameter_views.append(
                _parameter_editor_group(
                    controller=controller,
                    module_id=spec.module_id,
                    title="Parameters",
                    parameter_objects=objects,
                    values=parameters,
                )
            )
        elif spec.module_id == "energetics":
            deb_values = dict(parameters.get("DEB", {}) or {})
            gut_values = dict(parameters.get("gut", {}) or {})
            deb_objects = dict(
                class_objs(
                    deb.DEB,
                    excluded=[deb.DEB_model, "substrate", "id"],
                )
            )
            gut_objects = dict(class_objs(deb.Gut))
            parameter_views.append(
                _parameter_editor_group(
                    controller=controller,
                    module_id=spec.module_id,
                    title="DEB",
                    parameter_objects=deb_objects,
                    values=deb_values,
                    path_prefix=("DEB",),
                )
            )
            parameter_views.append(
                _parameter_editor_group(
                    controller=controller,
                    module_id=spec.module_id,
                    title="gut",
                    parameter_objects=gut_objects,
                    values=gut_values,
                    path_prefix=("gut",),
                )
            )
        elif spec.module_id == "Box2D":
            if "joint_types" in parameters:
                joint_types = pn.widgets.LiteralInput(
                    name="joint_types",
                    value=parameters["joint_types"],
                    sizing_mode="stretch_width",
                )

                def _on_joint_types_change(event) -> None:
                    if event.old == event.new:
                        return
                    controller._set_module_parameter(
                        "Box2D", ("joint_types",), event.new
                    )

                joint_types.param.watch(_on_joint_types_change, "value")
                parameter_views.append(
                    pn.Column(
                        pn.pane.Markdown("**Parameters**", margin=(4, 0, 2, 0)),
                        joint_types,
                        sizing_mode="stretch_width",
                    )
                )
        return parameter_views

    return parameter_views


def _issues_for_card(
    validation_issues: tuple[DraftValidationIssue, ...],
    module_id: str,
) -> tuple[DraftValidationIssue, ...]:
    return tuple(issue for issue in validation_issues if issue.module_id == module_id)


def _validation_issue_signature(
    issues: tuple[DraftValidationIssue, ...],
) -> dict[str, tuple[tuple[str, str, tuple[str, ...], str], ...]]:
    by_module: dict[str, list[tuple[str, str, tuple[str, ...], str]]] = {}
    for issue in issues:
        by_module.setdefault(issue.module_id, []).append(
            (
                issue.severity,
                issue.code,
                tuple(issue.path),
                issue.message,
            )
        )
    return {module_id: tuple(sorted(values)) for module_id, values in by_module.items()}


def _module_editor_card(
    *,
    controller: _ModelInspectorController,
    spec: ModelModuleSpec,
    validation_issues: tuple[DraftValidationIssue, ...],
) -> pn.Card:
    controls: list[pn.viewable.Viewable] = []

    is_optional = not spec.is_core
    is_memory = spec.module_id == "memory"
    is_brain_non_memory = spec.module_kind == "brain"
    is_optional_brain_non_memory = is_brain_non_memory and is_optional
    is_optional_larva = spec.module_kind == "larva" and is_optional

    if is_optional or is_memory:
        enabled_checkbox = pn.widgets.Checkbox(name="Enabled", value=spec.present)
        enabled_checkbox.param.watch(
            lambda event, module_id=spec.module_id: controller._set_module_enabled(
                module_id, bool(event.new)
            ),
            "value",
        )
        controls.append(enabled_checkbox)

    if is_brain_non_memory:
        mode_value = spec.current_mode or (
            spec.mode_options[0] if spec.mode_options else None
        )
        mode_select = pn.widgets.Select(
            name="Mode",
            options=list(spec.mode_options),
            value=mode_value,
            disabled=(is_optional_brain_non_memory and not spec.present),
        )
        mode_select.param.watch(
            lambda event, module_id=spec.module_id: controller._set_brain_module_mode(
                module_id, str(event.new)
            ),
            "value",
        )
        controls.append(mode_select)

    if is_memory:
        mode_options = list(spec.mode_options)
        mode_value = (
            spec.current_mode
            if spec.current_mode in mode_options
            else (
                "RL"
                if "RL" in mode_options
                else (mode_options[0] if mode_options else None)
            )
        )
        mode_select = pn.widgets.Select(
            name="Memory mode",
            options=mode_options,
            value=mode_value,
            disabled=not spec.present,
            sizing_mode="fixed",
            width=170,
        )
        selected_mode = str(mode_value) if mode_value is not None else ""
        modality_options = list(spec.modality_options_by_mode.get(selected_mode, ()))
        modality_value = (
            spec.current_modality
            if spec.current_modality in modality_options
            else (modality_options[0] if modality_options else None)
        )
        modality_select = pn.widgets.Select(
            name="Memory modality",
            options=modality_options,
            value=modality_value,
            disabled=not spec.present,
            sizing_mode="fixed",
            width=160,
        )
        mode_select.param.watch(
            lambda event: controller._set_memory_mode(str(event.new)),
            "value",
        )
        modality_select.param.watch(
            lambda event: controller._set_memory_modality(str(event.new)),
            "value",
        )
        controls.extend([mode_select, modality_select])

    body_children: list[pn.viewable.Viewable] = []
    if controls:
        if is_memory:
            body_children.append(
                pn.Row(
                    *controls,
                    sizing_mode="stretch_width",
                    styles={"flex-wrap": "wrap", "row-gap": "6px"},
                )
            )
        else:
            body_children.append(pn.Row(*controls, sizing_mode="stretch_width"))
    if not spec.present and (
        is_optional_brain_non_memory or is_optional_larva or is_memory
    ):
        body_children.append(
            pn.pane.Markdown("Not configured in this draft.", margin=(4, 0, 0, 0))
        )
    if spec.present:
        parameter_editors = _canonical_editor_groups_for_spec(
            controller=controller,
            spec=spec,
        )
        if parameter_editors:
            body_children.extend(parameter_editors)
    for issue in _issues_for_card(validation_issues, spec.module_id):
        css = (
            "lw-model-inspector-validation-error"
            if issue.severity == "error"
            else "lw-model-inspector-validation-warning"
        )
        level = "error" if issue.severity == "error" else "warning"
        body_children.append(
            pn.pane.Markdown(
                f"Validation {level}: {issue.message}",
                css_classes=[css],
                margin=(4, 0, 0, 0),
            )
        )

    return pn.Card(
        pn.Column(*body_children, sizing_mode="stretch_width"),
        title=_module_spec_title(spec),
        sizing_mode="stretch_width",
    )


def _build_module_sections(
    *,
    controller: _ModelInspectorController,
    specs: tuple[ModelModuleSpec, ...],
    validation_issues: tuple[DraftValidationIssue, ...],
    card_slots: dict[str, pn.Column] | None = None,
) -> list[pn.viewable.Viewable]:
    specs_by_id = {spec.module_id: spec for spec in specs}
    sections: list[pn.viewable.Viewable] = []

    nervous_system = _build_nervous_system_section(
        controller=controller,
        specs_by_id=specs_by_id,
        validation_issues=validation_issues,
        card_slots=card_slots,
    )
    if nervous_system is not None:
        sections.append(nervous_system)

    larva_modules = _build_larva_modules_section(
        controller=controller,
        specs_by_id=specs_by_id,
        validation_issues=validation_issues,
        card_slots=card_slots,
    )
    if larva_modules is not None:
        sections.append(larva_modules)
    return sections


def _build_nervous_system_section(
    *,
    controller: _ModelInspectorController,
    specs_by_id: dict[str, ModelModuleSpec],
    validation_issues: tuple[DraftValidationIssue, ...],
    card_slots: dict[str, pn.Column] | None,
) -> pn.viewable.Viewable | None:
    locomotion = _build_locomotion_subsection(
        controller=controller,
        specs_by_id=specs_by_id,
        validation_issues=validation_issues,
        card_slots=card_slots,
    )
    right_column = pn.Column(
        *(
            section
            for section in (
                _build_vertical_subsection(
                    title="Sensation",
                    css_modifier="sensation",
                    module_ids=("olfactor", "toucher", "windsensor", "thermosensor"),
                    controller=controller,
                    specs_by_id=specs_by_id,
                    validation_issues=validation_issues,
                    card_slots=card_slots,
                ),
                _build_vertical_subsection(
                    title="Memory",
                    css_modifier="memory",
                    module_ids=("memory",),
                    controller=controller,
                    specs_by_id=specs_by_id,
                    validation_issues=validation_issues,
                    card_slots=card_slots,
                ),
            )
            if section is not None
        ),
        sizing_mode="stretch_width",
        styles={"margin-left": f"{SECTION_COLUMN_GAP_PX}px"},
    )
    if locomotion is None and not right_column.objects:
        return None
    layout = pn.GridSpec(
        ncols=3,
        nrows=1,
        sizing_mode="stretch_width",
    )
    if locomotion is not None:
        layout[0, 0:2] = locomotion
    if right_column.objects:
        layout[0, 2] = right_column
    inner = pn.Column(
        layout,
        sizing_mode="stretch_width",
    )
    return pn.Card(
        inner,
        title="Nervous System",
        collapsed=False,
        collapsible=True,
        css_classes=["lw-model-inspector-section-box"],
        sizing_mode="stretch_width",
    )


def _build_larva_modules_section(
    *,
    controller: _ModelInspectorController,
    specs_by_id: dict[str, ModelModuleSpec],
    validation_issues: tuple[DraftValidationIssue, ...],
    card_slots: dict[str, pn.Column] | None,
) -> pn.viewable.Viewable | None:
    core = _build_larva_core_subsection(
        controller=controller,
        specs_by_id=specs_by_id,
        validation_issues=validation_issues,
        card_slots=card_slots,
    )
    optional = _build_vertical_subsection(
        title="Optional",
        css_modifier="optional",
        module_ids=("energetics", "sensorimotor", "Box2D"),
        controller=controller,
        specs_by_id=specs_by_id,
        validation_issues=validation_issues,
        card_slots=card_slots,
    )
    if core is None and optional is None:
        return None
    optional_cell: pn.viewable.Viewable | None = None
    if optional is not None:
        optional_cell = pn.Column(
            optional,
            sizing_mode="stretch_width",
            styles={"margin-left": f"{SECTION_COLUMN_GAP_PX}px"},
        )
    layout = pn.GridSpec(
        ncols=3,
        nrows=1,
        sizing_mode="stretch_width",
    )
    if core is not None:
        layout[0, 0:2] = core
    if optional_cell is not None:
        layout[0, 2] = optional_cell
    return pn.Column(
        pn.pane.Markdown("#### Body and Metabolism", margin=(0, 0, 6, 0)),
        layout,
        css_classes=["lw-model-inspector-section-box"],
        sizing_mode="stretch_width",
    )


def _build_locomotion_subsection(
    *,
    controller: _ModelInspectorController,
    specs_by_id: dict[str, ModelModuleSpec],
    validation_issues: tuple[DraftValidationIssue, ...],
    card_slots: dict[str, pn.Column] | None,
) -> pn.viewable.Viewable | None:
    first_column = _build_module_slot_column(
        module_ids=("crawler", "turner"),
        controller=controller,
        specs_by_id=specs_by_id,
        validation_issues=validation_issues,
        card_slots=card_slots,
    )
    second_column = _build_module_slot_column(
        module_ids=("interference", "intermitter", "feeder"),
        controller=controller,
        specs_by_id=specs_by_id,
        validation_issues=validation_issues,
        card_slots=card_slots,
    )
    if first_column is None and second_column is None:
        return None
    columns = [column for column in (first_column, second_column) if column is not None]
    return pn.Card(
        pn.GridBox(*columns, ncols=len(columns), sizing_mode="stretch_width"),
        title="Locomotion",
        collapsed=False,
        css_classes=[
            "lw-model-inspector-subsection-box",
            "lw-model-inspector-subsection--locomotion",
        ],
        sizing_mode="stretch_width",
    )


def _build_larva_core_subsection(
    *,
    controller: _ModelInspectorController,
    specs_by_id: dict[str, ModelModuleSpec],
    validation_issues: tuple[DraftValidationIssue, ...],
    card_slots: dict[str, pn.Column] | None,
) -> pn.viewable.Viewable | None:
    first_column = _build_module_slot_column(
        module_ids=("body",),
        controller=controller,
        specs_by_id=specs_by_id,
        validation_issues=validation_issues,
        card_slots=card_slots,
    )
    second_column = _build_module_slot_column(
        module_ids=("physics",),
        controller=controller,
        specs_by_id=specs_by_id,
        validation_issues=validation_issues,
        card_slots=card_slots,
    )
    if first_column is None and second_column is None:
        return None
    columns = [column for column in (first_column, second_column) if column is not None]
    return pn.Card(
        pn.GridBox(*columns, ncols=len(columns), sizing_mode="stretch_width"),
        title="Core",
        collapsed=False,
        css_classes=[
            "lw-model-inspector-subsection-box",
            "lw-model-inspector-subsection--core",
        ],
        sizing_mode="stretch_width",
    )


def _build_vertical_subsection(
    *,
    title: str,
    css_modifier: str | None,
    module_ids: tuple[str, ...],
    controller: _ModelInspectorController,
    specs_by_id: dict[str, ModelModuleSpec],
    validation_issues: tuple[DraftValidationIssue, ...],
    card_slots: dict[str, pn.Column] | None,
) -> pn.viewable.Viewable | None:
    column = _build_module_slot_column(
        module_ids=module_ids,
        controller=controller,
        specs_by_id=specs_by_id,
        validation_issues=validation_issues,
        card_slots=card_slots,
    )
    if column is None:
        return None
    css_classes = ["lw-model-inspector-subsection-box"]
    if css_modifier:
        css_classes.append(f"lw-model-inspector-subsection--{css_modifier}")
    return pn.Card(
        column,
        title=title,
        collapsed=False,
        css_classes=css_classes,
        sizing_mode="stretch_width",
    )


def _build_module_slot_column(
    *,
    module_ids: tuple[str, ...],
    controller: _ModelInspectorController,
    specs_by_id: dict[str, ModelModuleSpec],
    validation_issues: tuple[DraftValidationIssue, ...],
    card_slots: dict[str, pn.Column] | None,
) -> pn.viewable.Viewable | None:
    slots = [
        _build_module_card_slot(
            controller=controller,
            spec=specs_by_id[module_id],
            validation_issues=validation_issues,
            card_slots=card_slots,
        )
        for module_id in module_ids
        if module_id in specs_by_id
    ]
    if not slots:
        return None
    return pn.GridBox(
        *slots,
        ncols=1,
        sizing_mode="stretch_width",
    )


def _build_module_card_slot(
    *,
    controller: _ModelInspectorController,
    spec: ModelModuleSpec,
    validation_issues: tuple[DraftValidationIssue, ...],
    card_slots: dict[str, pn.Column] | None,
) -> pn.Column:
    card = _module_editor_card(
        controller=controller,
        spec=spec,
        validation_issues=validation_issues,
    )
    slot = pn.Column(card, sizing_mode="stretch_width")
    if card_slots is not None:
        card_slots[spec.module_id] = slot
    return slot


def model_inspector_app() -> pn.viewable.Viewable:
    pn.extension(raw_css=[PORTAL_RAW_CSS, MODEL_INSPECTOR_RAW_CSS])
    controller = _ModelInspectorController()
    template = pn.template.MaterialTemplate(
        title="",
        header_background="#c1b0c2",
        header_color="#111111",
    )
    template.header.append(build_app_header(title="Model Inspector"))
    template.main.append(controller.view())
    return template
