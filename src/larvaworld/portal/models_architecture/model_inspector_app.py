from __future__ import annotations

from html import escape

import holoviews as hv
import pandas as pd
import panel as pn

from larvaworld.portal.models_architecture.model_inspector_data import (
    BASELINE_MODULES,
    OPTIONAL_MODULES,
    build_inspection_brain,
    compare_model_inspections,
    inspect_model,
    list_model_ids,
    run_model_probe,
)
from larvaworld.portal.models_architecture.model_inspector_models import (
    ModelInspectorError,
    ModuleInspection,
    ProbeResult,
)
from larvaworld.portal.panel_components import PORTAL_RAW_CSS, build_app_header
from larvaworld.lib.model import Effector
from larvaworld.lib.model import moduleDB as MD
from larvaworld.lib.param import class_objs


__all__ = ["_ModelInspectorController", "model_inspector_app"]


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

.lw-model-inspector-card-grid {
  align-items: start;
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
        self.run_probe_button = pn.widgets.Button(
            name="Run probe", button_type="primary"
        )
        self.status = pn.pane.HTML(_status_html("Ready."), margin=0)
        self.primary_table = pn.pane.DataFrame(pd.DataFrame(), height=280)
        self.optional_table = pn.pane.DataFrame(pd.DataFrame(), height=220)
        self.settings_grid = pn.GridBox(ncols=2, sizing_mode="stretch_width")
        self.compare_table = pn.pane.DataFrame(pd.DataFrame(), height=260)
        self.compare_title = pn.pane.Markdown("", margin=(0, 0, 6, 0))
        self.probe_table = pn.pane.DataFrame(pd.DataFrame(), height=260)
        self.probe_meta = pn.pane.HTML("", margin=0)
        self.probe_plots = pn.Column(sizing_mode="stretch_width")

        self.primary_select.param.watch(self._on_primary_change, "value")
        self.compare_select.param.watch(self._on_compare_change, "value")
        self.run_probe_button.on_click(self._on_run_probe)
        self._refresh_inspection()

    def _set_status(self, message: str) -> None:
        self.status.object = _status_html(message)

    def _on_primary_change(self, _event=None) -> None:
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
        self._refresh_inspection()

    def _on_compare_change(self, _event=None) -> None:
        self._refresh_inspection()

    def _on_run_probe(self, _event=None) -> None:
        model_id = str(self.primary_select.value)
        try:
            result = run_model_probe(model_id)
        except ModelInspectorError as exc:
            self._set_status(f"Probe failed ({exc.code}): {exc}")
            self.probe_table.object = pd.DataFrame()
            self.probe_meta.object = ""
            self.probe_plots.objects = []
            return
        except Exception as exc:
            self._set_status(f"Probe failed: {exc}")
            self.probe_table.object = pd.DataFrame()
            self.probe_meta.object = ""
            self.probe_plots.objects = []
            return

        self.probe_table.object = result.dataframe
        self.probe_plots.objects = _build_probe_plots(result)
        reporter_bits = []
        for reporter, available in result.reporter_available.items():
            reporter_bits.append(f"{reporter}={'yes' if available else 'no'}")
        issues_text = ", ".join(issue.code for issue in result.issues) or "none"
        self.probe_meta.object = (
            '<div class="lw-model-inspector-status">'
            f"<strong>Probe settings:</strong> steps={result.steps}, dt={result.dt}, a_in={result.a_in}<br>"
            f"<strong>Reporter availability:</strong> {'; '.join(reporter_bits)}<br>"
            f"<strong>Probe issues:</strong> {escape(issues_text)}"
            "</div>"
        )
        self._set_status(f'Probe completed for model "{model_id}".')

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
            return

        self.primary_table.object = _modules_to_dataframe(primary.baseline_modules)
        self.optional_table.object = _modules_to_dataframe(primary.optional_modules)
        self.settings_grid.objects = self._build_settings_cards(primary_id, primary)

        compare_id = str(self.compare_select.value or "")
        if not compare_id:
            self.compare_title.object = ""
            self.compare_table.object = pd.DataFrame()
            self._set_status(f'Loaded primary model "{primary_id}".')
            return

        try:
            comparison = inspect_model(compare_id)
            diffs = compare_model_inspections(primary, comparison)
        except ModelInspectorError as exc:
            self._set_status(f"Comparison failed ({exc.code}): {exc}")
            self.compare_title.object = ""
            self.compare_table.object = pd.DataFrame()
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
        self._set_status(f'Loaded comparison "{primary_id}" vs "{compare_id}".')

    def view(self) -> pn.viewable.Viewable:
        intro = pn.pane.HTML(
            (
                '<div class="lw-model-inspector-intro">'
                "Inspect canonical larva model presets, compare module composition, and run a finite locomotor probe."
                "</div>"
            ),
            margin=0,
        )
        controls = pn.Column(
            self.primary_select,
            self.compare_select,
            self.run_probe_button,
            width=340,
            sizing_mode="fixed",
        )
        inspection = pn.Column(
            pn.pane.Markdown("#### Module settings", margin=(0, 0, 6, 0)),
            self.settings_grid,
            pn.pane.Markdown("#### Baseline locomotor modules", margin=(0, 0, 6, 0)),
            self.primary_table,
            pn.pane.Markdown("#### Optional configured modules", margin=(10, 0, 6, 0)),
            self.optional_table,
            sizing_mode="stretch_width",
        )
        compare = pn.Column(
            self.compare_title,
            self.compare_table,
            sizing_mode="stretch_width",
        )
        probe = pn.Column(
            pn.pane.Markdown("#### Finite locomotor probe", margin=(0, 0, 6, 0)),
            self.probe_plots,
            self.probe_meta,
            self.probe_table,
            sizing_mode="stretch_width",
        )
        return pn.Column(
            intro,
            controls,
            inspection,
            probe,
            compare,
            self.status,
            css_classes=["lw-model-inspector-root"],
            sizing_mode="stretch_width",
        )

    def _build_settings_cards(
        self,
        model_id: str,
        inspection,
    ) -> list[pn.viewable.Viewable]:
        try:
            brain = build_inspection_brain(model_id)
        except Exception as exc:
            return [
                pn.Card(
                    pn.pane.Markdown(
                        f"Could not build inspection brain: `{escape(str(exc))}`"
                    ),
                    title="Module settings unavailable",
                    sizing_mode="stretch_width",
                )
            ]

        inspections = {
            module.module_id: module
            for module in (*inspection.baseline_modules, *inspection.optional_modules)
        }
        ordered_ids = [
            *BASELINE_MODULES,
            *[m for m in OPTIONAL_MODULES if m in inspections],
        ]
        return [
            _module_settings_card(
                brain=brain,
                module_id=module_id,
                inspection=inspections[module_id],
            )
            for module_id in ordered_ids
        ]


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
        module_obj.__class__,
        excluded=[Effector, "phi", "name"],
    ).keylist
    widgets = {name: {"disabled": True} for name in parameter_names}
    body = pn.Param(
        module_obj,
        parameters=parameter_names,
        widgets=widgets,
        show_name=False,
        expand_button=True,
        default_precedence=3,
        sizing_mode="stretch_width",
    )
    module_name = getattr(module_obj.__class__, "name", module_obj.__class__.__name__)
    title = f"{module_id} | {module_name}"
    return pn.Card(body, title=title, sizing_mode="stretch_width")


def _build_probe_plots(result: ProbeResult) -> list[pn.viewable.Viewable]:
    # Controller tests can invoke probe plotting without going through
    # model_inspector_app(), so ensure a plotting backend is available here.
    if not hv.Store.renderers:
        hv.extension("bokeh")
    plots: list[pn.viewable.Viewable] = []
    for reporter in ("A_T", "A_C"):
        if not result.reporter_available.get(reporter, False):
            plots.append(
                pn.pane.HTML(
                    _status_html(
                        f'Reporter "{reporter}" is unavailable for this model.'
                    ),
                    margin=(0, 0, 8, 0),
                )
            )
            continue
        if reporter not in result.dataframe.columns:
            plots.append(
                pn.pane.HTML(
                    _status_html(
                        f'Reporter "{reporter}" is missing from probe output.'
                    ),
                    margin=(0, 0, 8, 0),
                )
            )
            continue
        curve = hv.Curve(
            result.dataframe[["time", reporter]],
            kdims=["time"],
            vdims=[reporter],
        ).opts(
            xlabel="time (sec)",
            ylabel=reporter,
            width=900,
            height=220,
            responsive=True,
        )
        plots.append(pn.pane.HoloViews(curve, sizing_mode="stretch_width"))
    return plots


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
