from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd
import panel as pn

from larvaworld.portal.datasets.manager_helpers import (
    delete_imported_workspace_dataset,
    format_relative_imported_location,
)
from larvaworld.portal.datasets.models import WorkspaceDatasetRecord
from larvaworld.portal.datasets.workspace_index import list_workspace_datasets
from larvaworld.portal.panel_components import PORTAL_RAW_CSS, build_app_header
from larvaworld.portal.workspace import get_active_workspace


__all__ = ["_DatasetManagerController", "dataset_manager_app"]


DATASET_MANAGER_RAW_CSS = """
.lw-dataset-manager-root {
  padding: 14px 12px 20px 12px;
}

.lw-dataset-manager-intro {
  border-left: 4px solid #7aa6c2;
  background: rgba(122, 166, 194, 0.16);
  border-radius: 10px;
  padding: 10px 12px;
  margin: 0 0 10px 0;
}

.lw-dataset-manager-scope {
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

.lw-dataset-manager-control-strip {
  display: flex;
  align-items: flex-end;
  gap: 12px;
  margin: 0 0 10px 0;
}

.lw-dataset-manager-count {
  font-size: 12px;
  color: rgba(17,17,17,0.72);
  padding: 0 0 8px 0;
  white-space: nowrap;
}

.lw-dataset-manager-status {
  font-size: 12px;
  line-height: 1.45;
  border-radius: 10px;
  padding: 10px 12px;
  border: 1px solid rgba(17, 17, 17, 0.1);
  background: rgba(248, 250, 252, 0.94);
}

.lw-dataset-manager-status--success {
  border-color: rgba(62,124,67,0.24);
  background: rgba(62,124,67,0.10);
}

.lw-dataset-manager-status--warning {
  border-color: rgba(176,112,33,0.28);
  background: rgba(245,161,66,0.12);
}

.lw-dataset-manager-status--danger {
  border-color: rgba(160,40,40,0.24);
  background: rgba(160,40,40,0.10);
}

.lw-dataset-manager-main {
  gap: 14px;
  align-items: flex-start;
}

.lw-dataset-manager-table .tabulator {
  border-radius: 10px;
  border: 1px solid rgba(90, 71, 96, 0.12);
  overflow: hidden;
}

.lw-dataset-manager-table .tabulator .tabulator-col,
.lw-dataset-manager-table .tabulator .tabulator-cell {
  font-size: 12px;
}

.lw-dataset-manager-empty {
  max-width: 640px;
  margin: 18px auto 0 auto;
  padding: 22px 24px;
  border-radius: 12px;
  border: 1px solid rgba(90, 71, 96, 0.12);
  background: rgba(255,255,255,0.97);
  text-align: center;
}

.lw-dataset-manager-empty-title {
  font-size: 18px;
  font-weight: 650;
  margin: 0 0 6px 0;
  color: rgba(17,17,17,0.92);
}

.lw-dataset-manager-empty-copy {
  font-size: 13px;
  line-height: 1.5;
  color: rgba(17,17,17,0.72);
  margin: 0 0 16px 0;
}

.lw-dataset-manager-cta {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-height: 38px;
  padding: 0 16px;
  border-radius: 10px;
  background: #4e9bcc;
  color: #ffffff !important;
  text-decoration: none;
  font-weight: 600;
}

.lw-dataset-manager-cta:hover {
  text-decoration: none;
  background: #4089b8;
}

.lw-dataset-manager-delete-confirm {
  font-size: 12px;
  line-height: 1.45;
  padding: 10px 12px;
  border-radius: 10px;
  border: 1px solid rgba(160,40,40,0.18);
  background: rgba(255,235,235,0.96);
  color: rgba(95,20,20,0.95);
}
""".strip()


TABLE_COLUMNS = ["Dataset ID", "Lab", "Group", "Ref ID", "N agents", "Location"]


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
        tone_class = f" lw-dataset-manager-status--{escape(tone)}"
    return (
        f'<div class="lw-dataset-manager-status{tone_class}">'
        f"{escape(text)}"
        f"{detail_html}"
        "</div>"
    )


def _details_html(record: WorkspaceDatasetRecord | None) -> str:
    if record is None:
        return (
            '<div class="lw-dataset-manager-status">'
            "Select a dataset from the catalog to inspect its stored artifacts."
            "</div>"
        )
    conf_present = "yes" if record.conf_path.is_file() else "no"
    h5_present = "yes" if record.h5_path.is_file() else "no"
    return (
        '<div class="lw-dataset-manager-status">'
        f"<div><strong>Dataset ID</strong>: {escape(record.dataset_id)}</div>"
        f"<div><strong>Lab</strong>: {escape(record.lab_id or '—')}</div>"
        f"<div><strong>Group ID</strong>: {escape(record.group_id or '—')}</div>"
        f"<div><strong>Ref ID</strong>: {escape(record.ref_id or '—')}</div>"
        f"<div><strong>Agents</strong>: {escape(str(record.n_agents) if record.n_agents is not None else '—')}</div>"
        f'<div style="margin-top:8px;"><strong>Dataset directory</strong>: {escape(str(record.dataset_dir))}</div>'
        f"<div><strong>conf.txt</strong>: {escape(str(record.conf_path))}</div>"
        f"<div><strong>data.h5</strong>: {escape(str(record.h5_path))}</div>"
        f'<div style="margin-top:8px;"><strong>Portal-supported imported layout</strong>: yes</div>'
        f"<div><strong>Config file present</strong>: {conf_present}</div>"
        f"<div><strong>HDF file present</strong>: {h5_present}</div>"
        "</div>"
    )


def _empty_state_html(title: str, copy: str, *, cta_href: str | None = None) -> str:
    cta_html = ""
    if cta_href:
        cta_html = (
            f'<a class="lw-dataset-manager-cta" href="{escape(cta_href)}">'
            "Go to Import Experimental Datasets"
            "</a>"
        )
    return (
        '<div class="lw-dataset-manager-empty">'
        f'<div class="lw-dataset-manager-empty-title">{escape(title)}</div>'
        f'<div class="lw-dataset-manager-empty-copy">{escape(copy)}</div>'
        f"{cta_html}"
        "</div>"
    )


def _records_frame(records: list[WorkspaceDatasetRecord], workspace) -> pd.DataFrame:
    rows = []
    for record in records:
        rows.append(
            {
                "Dataset ID": record.dataset_id,
                "Lab": record.lab_id or "—",
                "Group": record.group_id or "—",
                "Ref ID": record.ref_id or "—",
                "N agents": record.n_agents if record.n_agents is not None else "—",
                "Location": format_relative_imported_location(record, workspace),
            }
        )
    if not rows:
        return pd.DataFrame(columns=TABLE_COLUMNS)
    return pd.DataFrame(rows, columns=TABLE_COLUMNS)


class _DatasetManagerController:
    def __init__(self) -> None:
        self.workspace = get_active_workspace()
        self._all_records: list[WorkspaceDatasetRecord] = []
        self._filtered_records: list[WorkspaceDatasetRecord] = []
        self._selected_record: WorkspaceDatasetRecord | None = None
        self._pending_delete_record: WorkspaceDatasetRecord | None = None

        self.search_input = pn.widgets.TextInput(
            name="Search",
            placeholder="dataset ID / group ID / ref ID",
            width=280,
        )
        self.lab_filter = pn.widgets.Select(
            name="Lab filter",
            options={"All": ""},
            value="",
            width=220,
        )
        self.refresh_button = pn.widgets.Button(
            name="Refresh",
            button_type="primary",
            width=120,
        )
        self.refresh_list_button = pn.widgets.Button(
            name="Refresh list",
            button_type="default",
            width=140,
        )
        self.copy_path_button = pn.widgets.Button(
            name="Copy dataset path",
            button_type="default",
            width=160,
            disabled=True,
        )
        self.delete_button = pn.widgets.Button(
            name="Delete dataset",
            button_type="danger",
            width=140,
            disabled=True,
        )
        self.confirm_delete_button = pn.widgets.Button(
            name="Yes, delete",
            button_type="danger",
            sizing_mode="stretch_width",
        )
        self.cancel_delete_button = pn.widgets.Button(
            name="Cancel",
            button_type="default",
            sizing_mode="stretch_width",
        )
        self.table = pn.widgets.Tabulator(
            pd.DataFrame(columns=TABLE_COLUMNS),
            show_index=False,
            selectable=1,
            editors={column: None for column in TABLE_COLUMNS},
            height=560,
            sizing_mode="stretch_width",
            css_classes=["lw-dataset-manager-table"],
        )

        self.count_pane = pn.pane.HTML("", margin=(0, 0, 0, 0))
        self.details_pane = pn.pane.HTML(_details_html(None), margin=0)
        self.action_status = pn.pane.HTML("", margin=0)
        self.empty_state = pn.pane.HTML("", margin=0)
        self.main_content = pn.Column(sizing_mode="stretch_width", margin=0)
        self.delete_confirm_text = pn.pane.HTML("", margin=0)
        self.delete_confirm_panel = pn.Column(
            self.delete_confirm_text,
            pn.Row(
                self.confirm_delete_button,
                self.cancel_delete_button,
                sizing_mode="stretch_width",
                margin=(6, 0, 0, 0),
            ),
            visible=False,
            sizing_mode="stretch_width",
            margin=(4, 0, 0, 0),
        )

        self._copy_source = pn.widgets.TextInput(value="", visible=False)
        self._copy_result = pn.widgets.TextInput(value="", visible=False)

        self.search_input.param.watch(self._on_filters_change, "value")
        self.lab_filter.param.watch(self._on_filters_change, "value")
        self.table.param.watch(self._on_table_selection_change, "selection")
        self.refresh_button.on_click(self._handle_refresh)
        self.refresh_list_button.on_click(self._handle_refresh)
        self.delete_button.on_click(self._handle_request_delete)
        self.confirm_delete_button.on_click(self._handle_confirm_delete)
        self.cancel_delete_button.on_click(self._handle_cancel_delete)
        self._copy_result.param.watch(self._on_copy_result, "value")
        self.copy_path_button.js_on_click(
            args={"copy_source": self._copy_source, "copy_result": self._copy_result},
            code="""
                const text = String(copy_source.value || '');
                const stamp = Date.now().toString();
                if (!text) {
                    copy_result.setv({value: `missing|${stamp}|`});
                    return;
                }
                if (navigator.clipboard && navigator.clipboard.writeText) {
                    navigator.clipboard.writeText(text).then(
                        () => copy_result.setv({value: `copied|${stamp}|${text}`}),
                        () => copy_result.setv({value: `fallback|${stamp}|${text}`}),
                    );
                } else {
                    copy_result.setv({value: `fallback|${stamp}|${text}`});
                }
            """,
        )

        self._load_records()

    def _blocked_workspace(self) -> bool:
        return self.workspace is None

    def _set_action_status(
        self, text: str, *, tone: str = "neutral", detail: str | None = None
    ) -> None:
        self.action_status.object = _status_html(text, tone=tone, detail=detail)

    def _clear_selection(self) -> None:
        self._selected_record = None
        self._pending_delete_record = None
        self.table.selection = []
        self.details_pane.object = _details_html(None)
        self._copy_source.value = ""
        self.delete_confirm_panel.visible = False
        self.delete_confirm_text.object = ""
        self.copy_path_button.disabled = True
        self.delete_button.disabled = True

    def _load_records(self) -> None:
        if self.workspace is None:
            self._all_records = []
            self._filtered_records = []
            self.count_pane.object = ""
            self._clear_selection()
            self._set_action_status(
                "Configure an active workspace before opening Dataset Manager.",
                tone="warning",
            )
            self._refresh_body()
            return

        self._all_records = list_workspace_datasets(workspace=self.workspace)
        lab_options = {"All": ""}
        for lab_id in sorted(
            {record.lab_id for record in self._all_records if record.lab_id}
        ):
            lab_options[str(lab_id)] = str(lab_id)
        self.lab_filter.options = lab_options
        if self.lab_filter.value not in set(lab_options.values()):
            self.lab_filter.value = ""
        self._apply_filters()
        if self._all_records:
            self._set_action_status(
                f"Scanned the active workspace. {len(self._all_records)} imported dataset(s) found.",
                tone="success",
            )
        else:
            self.action_status.object = ""
        self._refresh_body()

    def _apply_filters(self) -> None:
        query = self.search_input.value.strip().lower()
        lab_filter = self.lab_filter.value.strip()

        def matches(record: WorkspaceDatasetRecord) -> bool:
            if lab_filter and record.lab_id != lab_filter:
                return False
            if not query:
                return True
            haystack = " ".join(
                filter(
                    None,
                    [record.dataset_id, record.group_id, record.ref_id],
                )
            ).lower()
            return query in haystack

        self._filtered_records = [
            record for record in self._all_records if matches(record)
        ]
        if self.workspace is None:
            frame = pd.DataFrame(columns=TABLE_COLUMNS)
        else:
            frame = _records_frame(self._filtered_records, self.workspace)
        self.table.value = frame
        self._clear_selection()
        total = len(self._all_records)
        shown = len(self._filtered_records)
        label = f"{shown} dataset{'s' if shown != 1 else ''}"
        if shown != total:
            label = f"{shown} of {total} datasets"
        self.count_pane.object = (
            f'<div class="lw-dataset-manager-count">{escape(label)}</div>'
        )
        self._refresh_body()

    def _refresh_body(self) -> None:
        if self.workspace is None:
            self.empty_state.object = _empty_state_html(
                "Dataset Manager requires an active workspace",
                "Configure an active workspace to browse imported dataset records.",
            )
            self.main_content.objects = [self.empty_state]
            return

        if not self._all_records:
            self.empty_state.object = _empty_state_html(
                "No imported datasets found in this workspace",
                "This view lists imported datasets recognized under the current workspace imported layout.",
                cta_href="/wf.open_dataset",
            )
            self.main_content.objects = [self.empty_state]
            return

        controls = pn.Row(
            self.search_input,
            self.lab_filter,
            self.refresh_button,
            pn.Spacer(sizing_mode="stretch_width"),
            self.count_pane,
            css_classes=["lw-dataset-manager-control-strip"],
            sizing_mode="stretch_width",
            margin=0,
        )
        catalog_card = pn.Card(
            self.table,
            title="Imported datasets",
            collapsed=False,
            sizing_mode="stretch_width",
        )
        details_card = pn.Card(
            self.details_pane,
            title="Dataset details",
            collapsed=False,
            sizing_mode="stretch_width",
        )
        actions_card = pn.Card(
            pn.Column(
                pn.Row(
                    self.refresh_list_button,
                    self.copy_path_button,
                    self.delete_button,
                    sizing_mode="stretch_width",
                ),
                self.delete_confirm_panel,
                self.action_status,
                sizing_mode="stretch_width",
            ),
            title="Actions",
            collapsed=False,
            sizing_mode="stretch_width",
        )
        main_row = pn.Row(
            pn.Column(catalog_card, sizing_mode="stretch_width"),
            pn.Column(
                details_card,
                actions_card,
                sizing_mode="stretch_width",
                min_width=430,
            ),
            css_classes=["lw-dataset-manager-main"],
            sizing_mode="stretch_width",
            margin=0,
        )
        self.main_content.objects = [controls, main_row]

    def _on_filters_change(self, *_events) -> None:
        self._apply_filters()

    def _on_table_selection_change(self, *_events) -> None:
        selection = list(self.table.selection or [])
        if not selection:
            self._clear_selection()
            return
        idx = selection[0]
        if idx < 0 or idx >= len(self._filtered_records):
            self._clear_selection()
            return
        record = self._filtered_records[idx]
        self._selected_record = record
        self._pending_delete_record = None
        self.delete_confirm_panel.visible = False
        self.delete_confirm_text.object = ""
        self.details_pane.object = _details_html(record)
        self._copy_source.value = str(record.dataset_dir)
        self.copy_path_button.disabled = False
        self.delete_button.disabled = False

    def _handle_refresh(self, _event=None) -> None:
        self._load_records()

    def _apply_copy_feedback(self, payload: str) -> None:
        if not payload:
            return
        kind, _sep, rest = payload.partition("|")
        if not kind:
            return
        parts = payload.split("|", 2)
        if len(parts) < 3:
            return
        kind, _stamp, path = parts
        if kind == "copied":
            self._set_action_status(
                "Dataset path copied to the clipboard.",
                tone="success",
                detail=path,
            )
        elif kind == "fallback":
            self._set_action_status(
                "Clipboard copy is unavailable in this environment. Use the path shown in the details pane.",
                tone="warning",
                detail=path,
            )
        elif kind == "missing":
            self._set_action_status(
                "Select a dataset before copying its path.",
                tone="warning",
            )

    def _on_copy_result(self, event) -> None:
        self._apply_copy_feedback(str(event.new or ""))

    def _handle_request_delete(self, _event=None) -> None:
        if self._selected_record is None:
            return
        self._pending_delete_record = self._selected_record
        self.delete_confirm_text.object = (
            '<div class="lw-dataset-manager-delete-confirm">'
            "Delete this imported dataset from the active workspace?<br><br>"
            f"<strong>{escape(self._selected_record.dataset_id)}</strong><br>"
            f"{escape(str(self._selected_record.dataset_dir))}"
            "</div>"
        )
        self.delete_confirm_panel.visible = True
        self._set_action_status(
            "Delete requested. Confirm to remove the selected dataset from the active workspace.",
            tone="warning",
        )

    def _handle_cancel_delete(self, _event=None) -> None:
        if self._pending_delete_record is None:
            return
        self._pending_delete_record = None
        self.delete_confirm_text.object = ""
        self.delete_confirm_panel.visible = False
        self._set_action_status("Delete dataset cancelled.")

    def _handle_confirm_delete(self, _event=None) -> None:
        if self.workspace is None or self._pending_delete_record is None:
            return
        record = self._pending_delete_record
        try:
            delete_imported_workspace_dataset(record, self.workspace)
        except Exception as exc:
            self._set_action_status(str(exc), tone="danger")
            return
        deleted_label = record.dataset_id
        self._pending_delete_record = None
        self._load_records()
        self._set_action_status(
            f'Deleted dataset "{deleted_label}" from the active workspace.',
            tone="success",
        )

    def view(self) -> pn.viewable.Viewable:
        intro = pn.pane.HTML(
            (
                '<div class="lw-dataset-manager-intro">'
                "Browse and inspect imported datasets stored in the active workspace. "
                "This first manager pass is a read-first catalog for portal-recognized imported dataset records, with lightweight selection, path inspection, and safe workspace actions."
                "</div>"
            ),
            margin=0,
        )
        scope = pn.pane.HTML(
            '<div class="lw-dataset-manager-scope">Scope: Imported datasets only</div>',
            margin=0,
        )
        hidden_proxies = pn.Column(
            self._copy_source,
            self._copy_result,
            visible=False,
            sizing_mode="fixed",
            width=0,
            height=0,
            margin=0,
        )
        return pn.Column(
            intro,
            scope,
            self.main_content,
            hidden_proxies,
            css_classes=["lw-dataset-manager-root"],
            sizing_mode="stretch_width",
        )


def dataset_manager_app() -> pn.viewable.Viewable:
    pn.extension("tabulator", raw_css=[PORTAL_RAW_CSS, DATASET_MANAGER_RAW_CSS])
    controller = _DatasetManagerController()
    template = pn.template.MaterialTemplate(
        title="",
        header_background="#b0b4c2",
        header_color="#111111",
    )
    template.header.append(build_app_header(title="Dataset Manager"))
    template.main.append(controller.view())
    return template
