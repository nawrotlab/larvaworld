"""Portal Module Inspector: standalone crawler/turner stepping and signal plots."""

from __future__ import annotations

from html import escape
from typing import Any

import panel as pn
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure

from larvaworld.lib import util
from larvaworld.lib.model import moduleDB as MD
from larvaworld.portal.config_widgets.widget_base import param_controls
from larvaworld.portal.models_architecture import module_inspector_data as data
from larvaworld.portal.models_architecture.module_inspector_data import (
    DEFAULT_A_IN,
    DEFAULT_DT,
    DEFAULT_STEPS,
)
from larvaworld.portal.models_architecture.module_inspector_models import (
    ModuleInspectorError,
    ModuleTraceResult,
)
from larvaworld.portal.panel_components import PORTAL_RAW_CSS, build_app_header

__all__ = ["_ModuleInspectorController", "module_inspector_app"]

CONTROLS_COLUMN_WIDTH = 340
SECTION_COLUMN_GAP_PX = 10

# Minimal subset of Model Inspector CSS for matching look (root, intro, status, controls).
MODULE_INSPECTOR_RAW_CSS = """
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
""".strip()


def _status_html(text: str) -> str:
    return f'<div class="lw-model-inspector-status">{escape(text)}</div>'


class _ModuleInspectorController:
    """Panel controller: module/mode selection, param editing, trace recompute."""

    def __init__(self) -> None:
        self._module_id = "crawler"
        self._mode = str(data.module_modes(self._module_id)[0])
        self._editor: Any = None
        self._editable_params: list[str] = []
        self._editor_watchers: list[Any] = []

        self._sources = {
            sig: ColumnDataSource(data={"time": [], sig: []})
            for sig in data.CANDIDATE_SIGNALS
        }

        self.module_select = pn.widgets.Select(
            name="Module",
            options=list(data.INSPECTABLE_MODULES),
            value=self._module_id,
        )
        self.mode_select = pn.widgets.Select(
            name="Mode",
            options=self._mode_option_labels(),
            value=self._mode,
        )
        lo, hi = data.FALLBACK_INPUT_RANGE
        self.a_in_slider = pn.widgets.FloatSlider(
            name="A_in",
            start=lo,
            end=hi,
            value=DEFAULT_A_IN,
            step=0.01,
        )
        self.steps_input = pn.widgets.IntInput(
            name="N steps",
            value=DEFAULT_STEPS,
            start=1,
        )
        self.dt_input = pn.widgets.FloatInput(
            name="dt",
            value=DEFAULT_DT,
            start=0.001,
            step=0.01,
        )
        self.signal_checkbox = pn.widgets.CheckBoxGroup(
            name="Signals",
            options=[],
            value=[],
        )
        self.status_pane = pn.pane.HTML(_status_html("Ready."), margin=(8, 0, 0, 0))
        self.param_box = pn.Column(sizing_mode="stretch_width")
        self.plot_view = pn.Column(sizing_mode="stretch_width")

        self.determinism_note = pn.pane.Markdown(
            (
                "**Preview note:** Initial phase is fixed to 0 for reproducibility. "
                "Turner **neural** mode uses stochastic warm-up; seed RNGs for repeatable tests."
            ),
            margin=(0, 0, 6, 0),
            sizing_mode="stretch_width",
        )

        self.module_select.param.watch(self._on_module_change, "value")
        self.mode_select.param.watch(self._on_mode_change, "value")
        _slider_attr = (
            "value_throttled"
            if "value_throttled" in self.a_in_slider.param
            else "value"
        )
        self.a_in_slider.param.watch(self._on_setting_change, _slider_attr)
        self.steps_input.param.watch(self._on_setting_change, "value")
        self.dt_input.param.watch(self._on_dt_change, "value")
        self.signal_checkbox.param.watch(self._on_signals_change, "value")

        self._rebuild_editor()
        self._recompute()

    def _mode_option_labels(self) -> dict[str, str]:
        return {
            data.mode_label(self._module_id, m): m
            for m in data.module_modes(self._module_id)
        }

    def _clear_editor_watchers(self) -> None:
        editor = self._editor
        if editor is None:
            self._editor_watchers.clear()
            return
        for watcher in self._editor_watchers:
            try:
                editor.param.unwatch(watcher)
            except Exception:
                pass
        self._editor_watchers.clear()

    def _watch_editor_params(self) -> None:
        def _on_param_change(_event) -> None:
            self._recompute()

        for name in self._editable_params:
            try:
                w = self._editor.param.watch(_on_param_change, name)
                self._editor_watchers.append(w)
            except Exception:
                continue

    def _rebuild_editor(self) -> None:
        self._clear_editor_watchers()
        raw_names = list(MD.brainDB[self._module_id].module_pars(mode=self._mode))
        self._editable_params = [n for n in raw_names if n != "mode"]
        self._editor = data.build_standalone_module(
            self._module_id,
            self._mode,
            dt=self._dt(),
        )
        lo, hi = data.module_input_range(self._editor)
        self.a_in_slider.start = lo
        self.a_in_slider.end = hi
        if self.a_in_slider.value < lo:
            self.a_in_slider.value = lo
        elif self.a_in_slider.value > hi:
            self.a_in_slider.value = hi

        self.param_box.objects = [
            param_controls(self._editor, parameters=self._editable_params)
        ]
        self._watch_editor_params()

        available = list(data.detect_signals(self._editor))
        self.signal_checkbox.options = available
        prev = set(self.signal_checkbox.value or ())
        self.signal_checkbox.value = [s for s in available if s in prev] or list(
            available
        )

    def _conf_from_editor(self) -> util.AttrDict:
        conf = util.AttrDict(
            {name: getattr(self._editor, name) for name in self._editable_params}
        )
        conf["mode"] = self._mode
        return conf

    def _dt(self) -> float:
        return max(0.001, float(self.dt_input.value))

    def _steps(self) -> int:
        return max(1, int(self.steps_input.value))

    def _a_in(self) -> float:
        return float(self.a_in_slider.value)

    def _on_module_change(self, event) -> None:
        self._module_id = str(event.new)
        labels = self._mode_option_labels()
        self.mode_select.options = labels
        modes = data.module_modes(self._module_id)
        first = str(modes[0])
        if self.mode_select.value != first:
            self.mode_select.value = first
        else:
            self._mode = first
            self._rebuild_editor()
            self._recompute()

    def _on_mode_change(self, event) -> None:
        self._mode = str(event.new)
        self._rebuild_editor()
        self._recompute()

    def _on_setting_change(self, _event=None) -> None:
        self._recompute()

    def _on_dt_change(self, _event=None) -> None:
        self._rebuild_editor()
        self._recompute()

    def _on_signals_change(self, _event=None) -> None:
        self._recompute()

    def _recompute(self) -> None:
        try:
            result = data.run_module_trace(
                self._module_id,
                self._mode,
                self._conf_from_editor(),
                steps=self._steps(),
                dt=self._dt(),
                a_in=self._a_in(),
            )
        except ModuleInspectorError as exc:
            self.status_pane.object = _status_html(f"Trace failed ({exc.code}): {exc}")
            return
        self._update_sources(result)
        self._rebuild_plots(result)
        self.status_pane.object = _status_html(
            f"{self._module_id} / {self._mode}: {result.steps} steps, "
            f"dt={result.dt}, A_in={result.a_in}."
        )

    def _update_sources(self, result: ModuleTraceResult) -> None:
        df = result.dataframe
        for sig in data.CANDIDATE_SIGNALS:
            if sig in df.columns:
                self._sources[sig].data = {
                    "time": list(df["time"]),
                    sig: list(df[sig]),
                }
            else:
                self._sources[sig].data = {"time": [], sig: []}

    def _rebuild_plots(self, result: ModuleTraceResult) -> None:
        selected = tuple(self.signal_checkbox.value or ())
        plots: list[pn.viewable.Viewable] = []
        for sig in selected:
            if sig not in result.signals:
                continue
            fig = figure(
                title=sig,
                height=220,
                x_axis_label="time (sec)",
                y_axis_label=sig,
                tools="pan,wheel_zoom,box_zoom,save,reset",
                active_drag=None,
                sizing_mode="stretch_width",
            )
            fig.line("time", sig, source=self._sources[sig], line_width=2)
            plots.append(pn.pane.Bokeh(fig, sizing_mode="stretch_width"))
        self.plot_view.objects = plots

    def view(self) -> pn.viewable.Viewable:
        intro = pn.pane.HTML(
            (
                '<div class="lw-model-inspector-intro">'
                "Inspect <strong>crawler</strong> and <strong>turner</strong> module "
                "implementations per mode: edit parameters, set constant "
                "<strong>A_in</strong>, and view time series of available signals."
                "</div>"
            ),
            margin=0,
        )
        controls = pn.Column(
            self.module_select,
            self.mode_select,
            self.a_in_slider,
            self.steps_input,
            self.dt_input,
            self.determinism_note,
            self.signal_checkbox,
            self.status_pane,
            sizing_mode="stretch_width",
            css_classes=["lw-model-inspector-controls-box"],
            width=CONTROLS_COLUMN_WIDTH,
            styles={
                "flex": f"0 0 {CONTROLS_COLUMN_WIDTH}px",
                "margin-bottom": "10px",
            },
        )
        body = pn.Column(
            self.param_box,
            self.plot_view,
            sizing_mode="stretch_width",
            css_classes=["lw-model-inspector-live-box"],
        )
        body_cell = pn.Column(
            body,
            sizing_mode="stretch_width",
            styles={
                "margin-left": f"{SECTION_COLUMN_GAP_PX}px",
                "flex": "1 1 0",
                "min-width": "0",
            },
        )
        row = pn.Row(
            controls,
            body_cell,
            sizing_mode="stretch_width",
            styles={"align-items": "flex-start"},
        )
        return pn.Column(
            intro,
            row,
            css_classes=["lw-model-inspector-root"],
            sizing_mode="stretch_width",
        )


def module_inspector_app() -> pn.viewable.Viewable:
    pn.extension(raw_css=[PORTAL_RAW_CSS, MODULE_INSPECTOR_RAW_CSS])
    controller = _ModuleInspectorController()
    template = pn.template.MaterialTemplate(
        title="",
        header_background="#c1b0c2",
        header_color="#111111",
    )
    template.header.append(build_app_header(title="Module Inspector"))
    template.main.append(controller.view())
    return template
