from __future__ import annotations

from html import escape
from typing import Any

import pandas as pd
import panel as pn

from larvaworld.lib.reg.graph import GraphRegistry
from larvaworld.lib.process.dataset import LarvaDataset
from larvaworld.portal.datasets.analysis_helpers import (
    get_valid_plots_for_datasets,
    run_plot_for_datasets,
)
from larvaworld.portal.datasets.manager_helpers import list_all_unified_datasets
from larvaworld.portal.panel_components import PORTAL_RAW_CSS, build_app_header
from larvaworld.portal.workspace import (
    WorkspaceError,
    get_active_workspace,
    get_workspace_dir,
)


__all__ = ["_AnalysisController", "analysis_app"]


ANALYSIS_RAW_CSS = """
.lw-analysis-root {
  padding: 14px 12px 20px 12px;
}

.lw-analysis-intro {
  border-left: 4px solid #7aa6c2;
  background: rgba(122, 166, 194, 0.16);
  border-radius: 10px;
  padding: 10px 12px;
  margin: 0 0 10px 0;
}

.lw-analysis-scope {
  display: inline-flex;
  align-items: center;
  padding: 5px 10px;
  border-radius: 999px;
  border: 1px solid rgba(0,0,0,0.12);
  background: rgba(255,255,255,0.94);
  color: rgba(17,17,17,0.82);
  font-size: 12px;
  font-weight: 600;
  margin: 0 0 10px 0;
}

.lw-analysis-status {
  font-size: 12px;
  line-height: 1.45;
  border-radius: 10px;
  padding: 10px 12px;
  border: 1px solid rgba(17, 17, 17, 0.1);
  background: rgba(248, 250, 252, 0.94);
}

.lw-analysis-status--success {
  border-color: rgba(62,124,67,0.24);
  background: rgba(62,124,67,0.10);
}

.lw-analysis-status--warning {
  border-color: rgba(176,112,33,0.28);
  background: rgba(245,161,66,0.12);
}

.lw-analysis-status--danger {
  border-color: rgba(160,40,40,0.24);
  background: rgba(160,40,40,0.10);
}

.lw-analysis-main {
  gap: 14px;
  align-items: flex-start;
}

.lw-analysis-table .tabulator {
  border-radius: 10px;
  border: 1px solid rgba(90, 71, 96, 0.12);
  overflow: hidden;
}

.lw-analysis-table .tabulator .tabulator-col,
.lw-analysis-table .tabulator .tabulator-cell {
  font-size: 12px;
}

.lw-analysis-empty {
  max-width: 640px;
  margin: 18px auto 0 auto;
  padding: 22px 24px;
  border-radius: 12px;
  border: 1px solid rgba(90, 71, 96, 0.12);
  background: rgba(255,255,255,0.97);
  text-align: center;
}

.lw-analysis-empty-title {
  font-size: 18px;
  font-weight: 650;
  margin: 0 0 6px 0;
  color: rgba(17,17,17,0.92);
}

.lw-analysis-empty-copy {
  font-size: 13px;
  line-height: 1.5;
  color: rgba(17,17,17,0.72);
}
""".strip()


def _status_html(text: str, *, tone: str = "neutral", detail: str | None = None) -> str:
    detail_html = ""
    if detail:
        detail_html = (
            '<div style="margin-top:4px;font-size:11px;opacity:0.84;word-break:break-word;">'
            f"{escape(detail)}"
            "</div>"
        )
    tone_class = ""
    if tone in {"success", "warning", "danger"}:
        tone_class = f" lw-analysis-status--{escape(tone)}"
    return (
        f'<div class="lw-analysis-status{tone_class}">'
        f"{escape(text)}"
        f"{detail_html}"
        "</div>"
    )


class _AnalysisController:
    def __init__(self) -> None:
        self.workspace = get_active_workspace()
        try:
            self.graph_registry = GraphRegistry()
        except Exception:
            self.graph_registry = None
        self._all_records = []
        self._selected_datasets = []
        self._loaded_datasets: dict[str, LarvaDataset] = {}
        self._valid_plots_by_group: dict[str, list[str]] = {}
        self._current_plot_id: str | None = None
        self._current_figure = None

        self.dataset_table = pn.widgets.Tabulator(
            pd.DataFrame(),
            show_index=False,
            selectable="checkbox",
            height=400,
            sizing_mode="stretch_width",
            css_classes=["lw-analysis-table"],
        )
        self.refresh_plots_button = pn.widgets.Button(
            name="Check plot availability",
            button_type="primary",
            width=180,
        )
        self.plot_group_select = pn.widgets.Select(
            name="Plot category",
            options={},
            width=220,
        )
        self.plot_function_select = pn.widgets.Select(
            name="Plot function",
            options={},
            width=220,
        )
        self.run_plot_button = pn.widgets.Button(
            name="Generate plot",
            button_type="primary",
            width=120,
        )
        self.status_pane = pn.pane.HTML("", margin=0)
        self.figure_pane = pn.pane.Markdown("", margin=0)
        self.main_content = pn.Column(sizing_mode="stretch_width", margin=0)

        self.dataset_table.param.watch(self._on_dataset_selection_change, "selection")
        self.refresh_plots_button.on_click(self._handle_refresh_plots)
        self.plot_group_select.param.watch(self._on_plot_group_change, "value")
        self.run_plot_button.on_click(self._handle_run_plot)

        self._load_records()

    def _load_records(self) -> None:
        if self.workspace is None:
            self._all_records = []
            self._set_status(
                "Configure an active workspace before opening Analysis.",
                tone="warning",
            )
            self._refresh_body()
            return

        self._all_records = list_all_unified_datasets(workspace=self.workspace)
        if self._all_records:
            self._set_status(
                f"Loaded {len(self._all_records)} dataset(s). Select one or more to analyze.",
                tone="success",
            )
        else:
            self.status_pane.object = ""
        self._update_dataset_table()
        self._refresh_body()

    def _set_status(
        self, text: str, *, tone: str = "neutral", detail: str | None = None
    ) -> None:
        self.status_pane.object = _status_html(text, tone=tone, detail=detail)

    def _update_dataset_table(self) -> None:
        rows = []
        for i, record in enumerate(self._all_records):
            source_label = "Imported" if record.origin == "imported" else "Simulated"
            rows.append(
                {
                    "ID": i,
                    "Dataset ID": record.dataset_id,
                    "Source": source_label,
                    "Group": record.group_id or "—",
                    "N agents": record.n_agents if record.n_agents is not None else "—",
                }
            )
        columns = ["ID", "Dataset ID", "Source", "Group", "N agents"]
        if rows:
            df = pd.DataFrame(rows, columns=columns)
        else:
            df = pd.DataFrame(columns=columns)
        self.dataset_table.value = df

    def _on_dataset_selection_change(self, *_events) -> None:
        selection = list(self.dataset_table.selection or [])
        self._selected_datasets = [self._all_records[i] for i in selection]
        self.refresh_plots_button.disabled = len(self._selected_datasets) == 0

    def _load_selected_datasets(self) -> bool:
        self._loaded_datasets.clear()
        for record in self._selected_datasets:
            try:
                ds = LarvaDataset(
                    dir=str(record.dataset_dir),
                    refID=record.ref_id,
                )
                self._loaded_datasets[record.dataset_id] = ds
            except Exception as exc:
                self._set_status(
                    f"Failed to load dataset {record.dataset_id}.",
                    tone="danger",
                    detail=str(exc),
                )
                return False
        return True

    def _handle_refresh_plots(self, _event=None) -> None:
        if not self._selected_datasets:
            self._set_status("Select at least one dataset first.", tone="warning")
            return

        if self.graph_registry is None:
            self._set_status(
                "Plot registry is not available. Please check your larvaworld installation.",
                tone="danger",
            )
            return

        self._set_status(
            "Loading datasets and checking plot availability…", tone="neutral"
        )

        if not self._load_selected_datasets():
            return

        dataset_ids = [record.dataset_id for record in self._selected_datasets]
        datasets = [self._loaded_datasets[did] for did in dataset_ids]

        self._set_status(
            "Checking which plots work for the selected dataset(s)…", tone="neutral"
        )

        try:
            self._valid_plots_by_group = get_valid_plots_for_datasets(
                self.graph_registry,
                dataset_ids,
                datasets,
            )
        except Exception as exc:
            self._set_status(
                "Error while checking plot availability.",
                tone="danger",
                detail=str(exc),
            )
            return

        group_options = {}
        for group_id, valid_fids in self._valid_plots_by_group.items():
            if valid_fids:
                group_options[group_id] = group_id

        if not group_options:
            self._set_status(
                "No plots are available for the selected dataset(s).",
                tone="warning",
            )
            self.plot_group_select.options = {}
            self.plot_function_select.options = {}
            return

        self.plot_group_select.options = group_options
        if group_options:
            self.plot_group_select.value = list(group_options.keys())[0]
            self._on_plot_group_change()

        self._set_status(
            f"Found {sum(len(v) for v in self._valid_plots_by_group.values())} plot(s) for the selected dataset(s).",
            tone="success",
        )

    def _on_plot_group_change(self, *_events) -> None:
        group_id = self.plot_group_select.value
        if not group_id or group_id not in self._valid_plots_by_group:
            self.plot_function_select.options = {}
            return

        valid_fids = self._valid_plots_by_group[group_id]
        func_options = {fid: fid for fid in valid_fids}
        self.plot_function_select.options = func_options
        if func_options:
            self.plot_function_select.value = list(func_options.keys())[0]

    def _plot_save_kwargs(self, plot_id: str, dataset_ids: list[str]) -> dict[str, Any]:
        """Save kwargs pointing at the workspace's "analysis" folder, same convention as `save_param_config_to_workspace`'s "parameters" folder."""
        try:
            analysis_dir = get_workspace_dir("analysis", workspace=self.workspace)
        except WorkspaceError:
            return {}
        subfolder = "_".join(sorted(dataset_ids)) or "dataset"
        return {
            "save_to": str(analysis_dir),
            "subfolder": subfolder,
            "save_as": plot_id,
        }

    def _handle_run_plot(self, _event=None) -> None:
        if not self._selected_datasets or not self._loaded_datasets:
            self._set_status(
                "Select and refresh dataset availability first.", tone="warning"
            )
            return

        if self.graph_registry is None:
            self._set_status(
                "Plot registry is not available. Please check your larvaworld installation.",
                tone="danger",
            )
            return

        plot_id = self.plot_function_select.value
        if not plot_id:
            self._set_status("Select a plot function first.", tone="warning")
            return

        dataset_ids = [record.dataset_id for record in self._selected_datasets]
        datasets = [self._loaded_datasets[did] for did in dataset_ids]

        self._set_status("Generating plot…", tone="neutral")

        try:
            fig = run_plot_for_datasets(
                self.graph_registry,
                plot_id,
                datasets,
                dataset_ids,
                default_kwargs=self._plot_save_kwargs(plot_id, dataset_ids),
            )
            self._current_figure = fig
            self._current_plot_id = plot_id
            self._render_figure(fig)
            self._set_status(
                f"Plot '{plot_id}' generated successfully.",
                tone="success",
            )
        except Exception as exc:
            self._set_status(
                f"Failed to generate plot '{plot_id}'.",
                tone="danger",
                detail=str(exc),
            )

    def _render_figure(self, fig: Any) -> None:
        try:
            html_repr = fig._repr_html_() if hasattr(fig, "_repr_html_") else None
            if html_repr:
                self.figure_pane.object = (
                    '<div style="border: 1px solid #ccc; border-radius: 8px; padding: 12px; overflow-x: auto;">'
                    f"{html_repr}"
                    "</div>"
                )
            elif hasattr(fig, "to_html"):
                self.figure_pane.object = (
                    '<div style="border: 1px solid #ccc; border-radius: 8px; padding: 12px; overflow-x: auto;">'
                    f"{fig.to_html()}"
                    "</div>"
                )
            elif hasattr(fig, "savefig"):
                # matplotlib Figure._repr_html_() exists but returns None
                # (a Jupyter rich-display fallback), so it can't gate this
                # branch by hasattr alone -- render as an embedded base64 PNG.
                import base64
                from io import BytesIO

                buf = BytesIO()
                fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
                buf.seek(0)
                b64 = base64.b64encode(buf.read()).decode("ascii")
                self.figure_pane.object = (
                    '<div style="border: 1px solid #ccc; border-radius: 8px; padding: 12px; overflow-x: auto;">'
                    f'<img src="data:image/png;base64,{b64}" style="max-width:100%;" />'
                    "</div>"
                )
            else:
                self.figure_pane.object = str(fig)
        except Exception as exc:
            self._set_status(
                "Could not render the figure.",
                tone="warning",
                detail=str(exc),
            )

    def _refresh_body(self) -> None:
        if self.workspace is None:
            empty = pn.pane.HTML(
                '<div class="lw-analysis-empty">'
                '<div class="lw-analysis-empty-title">Analysis requires an active workspace</div>'
                '<div class="lw-analysis-empty-copy">Configure an active workspace to analyze datasets.</div>'
                "</div>",
                margin=0,
            )
            self.main_content.objects = [empty]
            return

        if not self._all_records:
            empty = pn.pane.HTML(
                '<div class="lw-analysis-empty">'
                '<div class="lw-analysis-empty-title">No datasets found</div>'
                '<div class="lw-analysis-empty-copy">Import or generate datasets to analyze.</div>'
                "</div>",
                margin=0,
            )
            self.main_content.objects = [empty]
            return

        dataset_card = pn.Card(
            self.dataset_table,
            title="Select datasets",
            collapsed=False,
            sizing_mode="stretch_width",
        )
        controls_row = pn.Row(
            self.refresh_plots_button,
            self.plot_group_select,
            self.plot_function_select,
            self.run_plot_button,
            sizing_mode="stretch_width",
            margin=(10, 0, 10, 0),
        )
        plot_card = pn.Card(
            self.figure_pane,
            title="Plot",
            collapsed=False,
            sizing_mode="stretch_width",
        )
        main_col = pn.Column(
            dataset_card,
            self.status_pane,
            controls_row,
            plot_card,
            sizing_mode="stretch_width",
            margin=0,
        )
        self.main_content.objects = [main_col]

    def view(self) -> pn.viewable.Viewable:
        intro_text = pn.pane.HTML(
            (
                "<p>Select one or more datasets to analyze. "
                "Choose a plot category and function to generate analysis visualizations. "
                "Only compatible plots will be available for your dataset selection.</p>"
                "<p>Works with preprocessed and processed datasets. "
                "Use Dataset Manager to apply preprocessing/processing steps first.</p>"
            ),
            margin=0,
        )
        info_panel = pn.Card(
            intro_text,
            title="ℹ️ About Analysis",
            collapsed=True,
            collapsible=True,
            css_classes=["lw-portal-app-info"],
            sizing_mode="stretch_width",
            margin=(0, 0, 12, 0),
        )
        scope = pn.pane.HTML(
            '<div class="lw-analysis-scope">Scope: Dataset analysis & plotting</div>',
            margin=0,
        )
        return pn.Column(
            info_panel,
            scope,
            self.main_content,
            css_classes=["lw-analysis-root"],
            sizing_mode="stretch_width",
        )


def analysis_app() -> pn.viewable.Viewable:
    pn.extension("tabulator", raw_css=[PORTAL_RAW_CSS, ANALYSIS_RAW_CSS])
    controller = _AnalysisController()
    template = pn.template.MaterialTemplate(
        title="",
        header_background="#b0b4c2",
        header_color="#111111",
    )
    template.header.append(build_app_header(title="Analysis"))
    template.main.append(controller.view())
    return template
