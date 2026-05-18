from __future__ import annotations

from html import escape
from types import SimpleNamespace
from typing import Any

import pandas as pd
import panel as pn
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure

from larvaworld.lib import reg, util
from larvaworld.lib.model import Effector
from larvaworld.lib.model import moduleDB as MD
from larvaworld.lib.param import class_objs
from larvaworld.portal.models_architecture.model_inspector_data import (
    BASELINE_MODULES,
    build_inspection_brain,
    compare_model_inspections,
    inspect_model,
    list_model_ids,
)
from larvaworld.portal.models_architecture.model_inspector_models import (
    ModelInspectorError,
    ModuleInspection,
)
from larvaworld.portal.panel_components import PORTAL_RAW_CSS, build_app_header

__all__ = ["_ModelInspectorController", "model_inspector_app"]


LIVE_ROLLOVER = 100
LIVE_MAX_STEPS = 501
LIVE_DT = 0.1
LIVE_A_IN = 0.0
LIVE_REPORTERS = ("A_T", "A_C")
LIVE_REPORTER_LABELS = {
    "A_T": "Turn activation (A_T)",
    "A_C": "Crawl activation (A_C)",
}
LIVE_PREVIEW_SIDEBAR_TOP_OFFSET = 28
LIVE_PREVIEW_SIDEBAR_HEIGHT = 354
CONTROLS_COLUMN_WIDTH = 340


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
  background: rgba(255, 255, 255, 0.82);
  padding: 10px;
  margin-top: 8px;
}

.lw-model-inspector-live-box {
  border-radius: 10px;
  border: 1px solid rgba(17, 17, 17, 0.1);
  background: rgba(255, 255, 255, 0.98);
  padding: 10px;
  margin-top: 8px;
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
""".strip()


def _status_html(text: str) -> str:
    return f'<div class="lw-model-inspector-status">{escape(text)}</div>'


class _ModelInspectorController:
    def __init__(self) -> None:
        model_ids = list_model_ids()
        if not model_ids:
            raise ModelInspectorError("no_models", "No model presets are available.")
        self._model_ids = model_ids
        self._brain = None
        self._runtime = None
        self._callback = None
        self._is_running = False
        self._has_local_edits = False
        self._status_message = "Ready."
        self._step = 0
        self._active_dt = LIVE_DT
        self._reporter_paths: dict[str, str] = {}
        self._reporter_available: dict[str, bool] = {
            key: False for key in LIVE_REPORTERS
        }
        self._watched_param_tokens: set[tuple[int, str]] = set()
        self._probe_df = pd.DataFrame(
            columns=["time", "lin", "ang", "feed_motion", *LIVE_REPORTERS]
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

        self.primary_table = pn.pane.DataFrame(pd.DataFrame(), height=240)
        self.optional_table = pn.pane.DataFrame(pd.DataFrame(), height=200)
        self.settings_grid = pn.GridBox(ncols=2, sizing_mode="stretch_width")
        self.compare_table = pn.pane.DataFrame(pd.DataFrame(), height=260)
        self.compare_title = pn.pane.Markdown("", margin=(0, 0, 6, 0))
        self.summary_sections_box = pn.Column(
            sizing_mode="stretch_width",
            styles={"flex": "1 1 0", "min-width": "0"},
        )
        self.probe_table = pn.pane.DataFrame(
            pd.DataFrame(),
            height=220,
            css_classes=["lw-model-inspector-live-table"],
        )
        self.probe_meta = pn.pane.HTML("", margin=0)
        self.live_plot_view = pn.Column(sizing_mode="stretch_width")

        self._sources = {
            key: ColumnDataSource(data={"time": [], key: []}) for key in LIVE_REPORTERS
        }

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
        self._ensure_brain_for_selected_model()
        self._refresh_inspection()
        self._init_live_plots()
        self._update_probe_meta()
        self._update_summary_sections()

    def _set_status(self, message: str) -> None:
        self._status_message = message

    def _set_running(self, value: bool) -> None:
        self._is_running = value
        self.run_button.disabled = value
        self.pause_button.disabled = not value
        self.dt_input.disabled = value

    def _on_primary_change(self, _event=None) -> None:
        self._pause_callback()
        self._has_local_edits = False
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
        self._ensure_brain_for_selected_model()
        self._clear_trace_data()
        self._refresh_inspection()

    def _on_compare_change(self, _event=None) -> None:
        self._refresh_inspection()

    def _on_run(self, _event=None) -> None:
        if self._is_running:
            return
        if self._brain is None:
            self._ensure_brain_for_selected_model()
        self._start_callback()
        self._set_status(
            f'Live preview running for model "{self.primary_select.value}".'
        )

    def _on_pause(self, _event=None) -> None:
        self._pause_callback()
        self._set_status(f"Live preview paused at step {self._step}.")

    def _on_clear_trace(self, _event=None) -> None:
        self._clear_trace_data()
        self._update_probe_meta()
        self._set_status("Trace cleared. Local edited runtime state preserved.")

    def _on_reset_to_preset(self, _event=None) -> None:
        self._pause_callback()
        self._has_local_edits = False
        self._ensure_brain_for_selected_model()
        self._clear_trace_data()
        self._refresh_inspection()
        self._update_probe_meta()
        self._set_status(
            f'Reset local state to model preset "{self.primary_select.value}".'
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
        self._brain = build_inspection_brain(
            str(self.primary_select.value), dt=self._active_dt
        )
        self._runtime = SimpleNamespace(brain=self._brain)
        self._watched_param_tokens.clear()
        self._prepare_reporters()

    def _prepare_reporters(self) -> None:
        assert self._runtime is not None
        available = reg.par.output_reporters(
            ks=list(LIVE_REPORTERS), agents=[self._runtime]
        )
        available_paths = set(available.values())
        reporter_paths: dict[str, str] = {}
        reporter_available: dict[str, bool] = {}
        for key in LIVE_REPORTERS:
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

        for key in LIVE_REPORTERS:
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
        for key in LIVE_REPORTERS:
            self._sources[key].data = {"time": [], key: []}
        self._probe_df = pd.DataFrame(
            columns=["time", "lin", "ang", "feed_motion", *LIVE_REPORTERS]
        )
        self._refresh_probe_table()

    def _refresh_probe_table(self) -> None:
        self.probe_table.object = self._probe_df.rename(columns=LIVE_REPORTER_LABELS)

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
        for key in LIVE_REPORTERS:
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
        self._has_local_edits = False
        self._ensure_brain_for_selected_model()
        self._clear_trace_data()
        self._refresh_inspection()
        self._update_probe_meta()
        self._set_status(
            f"dt changed to {self._active_dt}; local runtime reset to the canonical model preset."
        )

    def _update_probe_meta(self) -> None:
        reporter_bits = [
            f'{LIVE_REPORTER_LABELS[k]}={"yes" if self._reporter_available.get(k, False) else "no"}'
            for k in LIVE_REPORTERS
        ]
        self.probe_meta.object = (
            '<div class="lw-model-inspector-status">'
            f"<strong>Preview settings:</strong> dt={self._active_dt}, a_in={self._a_in()}, rollover={self._trace_window()}, max_steps={self._max_steps()}<br>"
            f"<strong>Current step:</strong> {self._step}<br>"
            f"<strong>Reporter availability:</strong> {'; '.join(reporter_bits)}"
            "</div>"
        )

    def _update_summary_sections(self) -> None:
        baseline_summary_box = pn.Column(
            pn.pane.Markdown(
                "#### Baseline locomotor modules (summary)", margin=(0, 0, 6, 0)
            ),
            self.primary_table,
            css_classes=["lw-model-inspector-section-box"],
            sizing_mode="stretch_width",
        )
        optional_summary_box = pn.Column(
            pn.pane.Markdown(
                "#### Optional configured modules (summary)", margin=(0, 0, 6, 0)
            ),
            self.optional_table,
            css_classes=["lw-model-inspector-section-box"],
            sizing_mode="stretch_width",
        )
        comparison_box = pn.Column(
            self.compare_title,
            self.compare_table,
            css_classes=["lw-model-inspector-section-box"],
            sizing_mode="stretch_width",
        )
        table_sections: list[pn.viewable.Viewable] = [baseline_summary_box]
        if not self.optional_table.object.empty:
            table_sections.append(optional_summary_box)
        if not self.compare_table.object.empty:
            table_sections.append(comparison_box)
        self.summary_sections_box.objects = table_sections

    def _refresh_inspection(self) -> None:
        primary_id = str(self.primary_select.value)
        try:
            primary = inspect_model(primary_id)
        except ModelInspectorError as exc:
            self._set_status(f"Inspection failed ({exc.code}): {exc}")
            self.primary_table.object = pd.DataFrame()
            self.optional_table.object = pd.DataFrame()
            self.compare_table.object = pd.DataFrame()
            self.compare_title.object = ""
            self._update_summary_sections()
            return

        self.primary_table.object = _modules_to_dataframe(primary.baseline_modules)
        self.optional_table.object = _modules_to_dataframe(primary.optional_modules)
        self.settings_grid.objects = self._build_settings_cards(primary)

        if self._has_local_edits:
            self.compare_select.disabled = True
            self.compare_title.object = "#### Comparison hidden during local edits"
            self.compare_table.object = pd.DataFrame(
                [{"Status": "Reset to model preset to re-enable canonical comparison."}]
            )
            self._update_summary_sections()
            return
        self.compare_select.disabled = False

        compare_id = str(self.compare_select.value or "")
        if not compare_id:
            self.compare_title.object = ""
            self.compare_table.object = pd.DataFrame()
            self._update_summary_sections()
            return

        try:
            comparison = inspect_model(compare_id)
            diffs = compare_model_inspections(primary, comparison)
        except ModelInspectorError as exc:
            self._set_status(f"Comparison failed ({exc.code}): {exc}")
            self.compare_title.object = ""
            self.compare_table.object = pd.DataFrame()
            self._update_summary_sections()
            return

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
        for reporter in LIVE_REPORTERS:
            reporter_label = LIVE_REPORTER_LABELS[reporter]
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
        controls = pn.Column(
            self.primary_select,
            self.compare_select,
            action_buttons,
            pn.pane.Markdown("#### Preview settings", margin=(8, 0, 2, 0)),
            self.max_steps_input,
            self.a_in_input,
            self.trace_window_input,
            self.dt_input,
            sizing_mode="stretch_width",
            css_classes=["lw-model-inspector-controls-box"],
        )
        cards = list(self.settings_grid.objects)
        cards_by_id = {
            card.title.split(" | ", 1)[0]: card
            for card in cards
            if getattr(card, "title", "")
        }
        controls_box = pn.Column(
            controls,
            width=CONTROLS_COLUMN_WIDTH,
            styles={
                "flex": f"0 0 {CONTROLS_COLUMN_WIDTH}px",
                "margin-bottom": "10px",
            },
        )
        tables_box = self.summary_sections_box
        top_row = pn.Row(
            controls_box,
            tables_box,
            sizing_mode="stretch_width",
            styles={"align-items": "stretch"},
        )
        crawler_interference_cards = [
            cards_by_id.get("crawler"),
            cards_by_id.get("interference"),
        ]
        intermitter_cards = [cards_by_id.get("intermitter")]
        turner_cards = [cards_by_id.get("turner")]
        tile_row = pn.Row(
            pn.Column(
                *[card for card in crawler_interference_cards if card is not None],
                sizing_mode="stretch_width",
                styles={"flex": "1 1 0"},
            ),
            pn.Column(
                *[card for card in intermitter_cards if card is not None],
                sizing_mode="stretch_width",
                styles={"flex": "1 1 0"},
            ),
            pn.Column(
                *[card for card in turner_cards if card is not None],
                sizing_mode="stretch_width",
                styles={"flex": "1 1 0"},
            ),
            sizing_mode="stretch_width",
            styles={"align-items": "stretch"},
        )
        probe_sidebar = pn.Column(
            self.probe_meta,
            self.probe_table,
            margin=(LIVE_PREVIEW_SIDEBAR_TOP_OFFSET, 0, 0, 0),
            height=LIVE_PREVIEW_SIDEBAR_HEIGHT,
            sizing_mode="stretch_width",
            css_classes=["lw-model-inspector-live-sidebar"],
            styles={
                "flex": "1 1 0",
                "overflow-y": "auto",
            },
        )
        probe_body = pn.Row(
            pn.Column(
                self.live_plot_view,
                sizing_mode="stretch_width",
                styles={"flex": "3 1 0"},
            ),
            probe_sidebar,
            sizing_mode="stretch_width",
            styles={"align-items": "stretch"},
        )
        probe = pn.Column(
            pn.pane.Markdown("#### Live preview", margin=(0, 0, 6, 0)),
            probe_body,
            css_classes=["lw-model-inspector-live-box"],
            sizing_mode="stretch_width",
        )
        return pn.Column(
            intro,
            top_row,
            tile_row,
            probe,
            css_classes=["lw-model-inspector-root"],
            sizing_mode="stretch_width",
        )


def _modules_to_dataframe(modules: tuple[ModuleInspection, ...]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Module": module.module_id,
                "Present": module.present,
                "Mode": module.mode or "—",
                "Parameters": repr(module.parameters),
            }
            for module in modules
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
