from __future__ import annotations

from html import escape
from pathlib import Path

import panel as pn

from larvaworld.lib import reg
from larvaworld.portal.landing_registry import DOCS_DATA_PROCESSING
from larvaworld.portal.datasets.discovery import (
    RawDatasetCandidate,
    _candidate_import_overrides,
    discover_raw_datasets,
)
from larvaworld.portal.datasets.import_adapter import (
    build_workspace_proc_folder,
    import_into_workspace,
)
from larvaworld.portal.datasets.models import ImportRequest
from larvaworld.portal.panel_components import PORTAL_RAW_CSS, build_app_header
from larvaworld.portal.path_picker import pick_directory
from larvaworld.portal.workspace import get_active_workspace


__all__ = ["_ImportDatasetsController", "import_datasets_app"]


IMPORT_DATASETS_RAW_CSS = """
.lw-import-datasets-root {
  padding: 14px 12px 20px 12px;
}

.lw-import-datasets-intro {
  border-left: 4px solid #7aa6c2;
  background: rgba(122, 166, 194, 0.16);
  border-radius: 10px;
  padding: 10px 12px;
  margin: 0 0 10px 0;
}

.lw-import-datasets-summary,
.lw-import-datasets-status {
  font-size: 12px;
  line-height: 1.5;
  border-radius: 10px;
  padding: 10px 12px;
  border: 1px solid rgba(17, 17, 17, 0.1);
  background: rgba(248, 250, 252, 0.94);
}

.lw-import-datasets-status--success {
  border-color: rgba(62,124,67,0.24);
  background: rgba(62,124,67,0.10);
}

.lw-import-datasets-status--warning {
  border-color: rgba(176,112,33,0.28);
  background: rgba(245,161,66,0.12);
}

.lw-import-datasets-status--danger {
  border-color: rgba(160,40,40,0.24);
  background: rgba(160,40,40,0.10);
}

.lw-import-datasets-flow-section {
  background: rgba(252, 252, 253, 0.99);
  border: 1px solid rgba(90, 71, 96, 0.10);
  border-radius: 10px;
  padding: 10px 12px 8px 12px;
  margin-top: 4px;
}

.lw-import-datasets-flow-title {
  margin: 0 0 6px 0;
  color: #4f2f5f;
  font-weight: 700;
}

.lw-import-datasets-flow-title p {
  margin: 0;
}

.lw-import-datasets-source-row {
  gap: 0;
  align-items: flex-end;
}

.lw-import-datasets-source-input .bk-input,
.lw-import-datasets-source-input input {
  border-top-right-radius: 0 !important;
  border-bottom-right-radius: 0 !important;
  border-right: 0 !important;
}

.lw-import-datasets-source-browse .bk-btn,
.lw-import-datasets-source-browse button {
  border-top-left-radius: 0 !important;
  border-bottom-left-radius: 0 !important;
  min-height: 40px;
  padding-left: 16px;
  padding-right: 16px;
}

.lw-import-datasets-source-browse {
  margin-top: 13px;
}

.lw-import-datasets-color-picker {
  width: 52px;
  min-width: 52px;
  margin-top: 0;
}

.lw-import-datasets-config-intro {
  border-left: 4px solid #c1b0c2;
  background: rgba(193, 176, 194, 0.16);
  border-radius: 10px;
  padding: 10px 12px;
  margin: 0 0 10px 0;
}

.lw-import-datasets-config-actions {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
}

.lw-import-datasets-config-family {
  background: rgba(248, 248, 250, 0.96);
  border: 1px solid rgba(90, 71, 96, 0.12);
  border-radius: 10px;
  padding: 10px 12px 8px 12px;
  margin-top: 4px;
}

.lw-import-datasets-config-family-title {
  margin: 0 0 6px 0;
  color: #4f2f5f;
  font-weight: 700;
}

.lw-import-datasets-config-field {
  margin: 0 0 4px 0;
}

.lw-import-datasets-config-help {
  font-size: 11px;
  line-height: 1.4;
  color: rgba(63, 51, 73, 0.82);
  margin-top: 4px;
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
        tone_class = f" lw-import-datasets-status--{escape(tone)}"
    return (
        f'<div class="lw-import-datasets-status{tone_class}">'
        f"{escape(text)}"
        f"{detail_html}"
        "</div>"
    )


def _candidate_summary_html(candidate: RawDatasetCandidate | None) -> str:
    if candidate is None:
        return (
            '<div class="lw-import-datasets-summary">'
            "No candidate selected yet."
            "</div>"
        )
    warnings_html = (
        "<ul>"
        + "".join(f"<li>{escape(warning)}</li>" for warning in candidate.warnings)
        + "</ul>"
        if candidate.warnings
        else "<div>No warnings.</div>"
    )
    return (
        '<div class="lw-import-datasets-summary">'
        f"<div><strong>Candidate</strong>: {escape(candidate.candidate_id)}</div>"
        f"<div><strong>Parent dir</strong>: {escape(candidate.parent_dir)}</div>"
        f"<div><strong>Source path</strong>: {escape(str(candidate.source_path))}</div>"
        f'<div style="margin-top:6px;"><strong>Warnings</strong>:</div>{warnings_html}'
        "</div>"
    )


def _flow_section(title: str, *children: object) -> pn.Column:
    return pn.Column(
        pn.pane.Markdown(
            f"**{title}**",
            css_classes=["lw-import-datasets-flow-title"],
            margin=(0, 0, 4, 0),
        ),
        *children,
        css_classes=["lw-import-datasets-flow-section"],
        sizing_mode="stretch_width",
        margin=0,
    )


def _config_family_box(title: str, *children: object) -> pn.Column:
    return pn.Column(
        pn.pane.Markdown(
            f"**{title}**",
            css_classes=["lw-import-datasets-config-family-title"],
            margin=(0, 0, 4, 0),
        ),
        *children,
        css_classes=["lw-import-datasets-config-family"],
        sizing_mode="stretch_width",
        margin=0,
    )


class _ImportDatasetsController:
    def __init__(self) -> None:
        self.workspace = get_active_workspace()
        self._candidate_by_key: dict[str, RawDatasetCandidate] = {}
        self._selected_record_path: Path | None = None
        self._working_lab_id: str | None = None
        self._working_lab = None
        self._tracker_widget_syncing = False

        self.lab_select = pn.widgets.Select(
            name="Lab format",
            options=self._lab_options(),
            value=self._default_lab_value(),
            width=260,
        )
        self.lab_config_name_input = pn.widgets.TextInput(
            name="Configuration ID",
            width=260,
        )
        self.lab_load_button = pn.widgets.Button(
            name="Load",
            button_type="default",
            width=90,
        )
        self.lab_save_button = pn.widgets.Button(
            name="Save",
            button_type="primary",
            width=90,
        )
        self.lab_delete_button = pn.widgets.Button(
            name="Delete",
            button_type="danger",
            width=90,
        )
        self.lab_reset_button = pn.widgets.Button(
            name="Reset",
            button_type="default",
            width=90,
        )
        self.raw_root_input = pn.widgets.TextInput(
            name="Raw root",
            placeholder="/path/to/raw/data",
            width=520,
            css_classes=["lw-import-datasets-source-input"],
        )
        self.browse_raw_root_button = pn.widgets.Button(
            name="Browse",
            button_type="default",
            width=110,
            css_classes=["lw-import-datasets-source-browse"],
        )
        self.reset_button = pn.widgets.Button(
            name="Reset source",
            button_type="default",
            width=140,
        )
        self.discover_button = pn.widgets.Button(
            name="Discover datasets",
            button_type="primary",
            width=170,
        )
        self.candidate_select = pn.widgets.Select(
            name="Candidate",
            options={"Select a candidate": ""},
            value="",
            width=520,
        )
        self.candidate_select.description = "Select one discovered candidate to inspect its source path and warnings before importing it into the active workspace."
        self.dataset_id_input = pn.widgets.TextInput(name="Dataset ID", width=260)
        self.group_id_input = pn.widgets.TextInput(
            name="Group ID override", placeholder="optional", width=260
        )
        self.color_input = pn.widgets.ColorPicker(
            name="Color",
            value="#000000",
            width=52,
            css_classes=["lw-import-datasets-color-picker"],
        )
        self.import_button = pn.widgets.Button(
            name="Import into workspace",
            button_type="primary",
            width=180,
        )
        self.workspace_summary = pn.pane.HTML("", margin=0)
        self.candidate_summary = pn.pane.HTML(
            _candidate_summary_html(None), margin=(0, 0, 0, 0)
        )
        self.lab_status = pn.pane.HTML("", margin=0)
        self.lab_editor_sections = pn.Column(sizing_mode="stretch_width", margin=0)
        self.status = pn.pane.HTML("", margin=0)

        self.lab_select.param.watch(self._on_lab_select_change, "value")
        self.raw_root_input.param.watch(self._on_raw_root_change, "value")
        self.candidate_select.param.watch(self._on_candidate_change, "value")
        self.lab_load_button.on_click(self._handle_lab_load)
        self.lab_save_button.on_click(self._handle_lab_save)
        self.lab_delete_button.on_click(self._handle_lab_delete)
        self.lab_reset_button.on_click(self._handle_lab_reset)
        self.browse_raw_root_button.on_click(self._handle_browse_raw_root)
        self.reset_button.on_click(self._handle_reset)
        self.discover_button.on_click(self._handle_discover)
        self.import_button.on_click(self._handle_import)

        self._load_working_lab(self.lab_select.value)
        self._refresh_workspace_summary()
        if self.workspace is None:
            self._set_status(
                "Configure an active workspace before importing datasets.",
                tone="warning",
            )
        else:
            self.status.object = ""
        self._sync_controls()

    @staticmethod
    def _lab_options() -> dict[str, str]:
        return {lab_id: lab_id for lab_id in sorted(reg.conf.LabFormat.confIDs)}

    def _default_lab_value(self) -> str | None:
        options = self._lab_options()
        if not options:
            return None
        return next(iter(options.values()))

    def _set_lab_status(
        self, text: str, *, tone: str = "neutral", detail: str | None = None
    ) -> None:
        self.lab_status.object = _status_html(text, tone=tone, detail=detail)

    def _refresh_lab_options(self, *, select_id: str | None = None) -> None:
        options = self._lab_options()
        current = select_id or self.lab_select.value
        self.lab_select.options = options
        if current in options.values():
            self.lab_select.value = current
        elif options:
            self.lab_select.value = next(iter(options.values()))
        else:
            self.lab_select.value = None

    @staticmethod
    def _widget_has_native_help(widget: object) -> bool:
        description = getattr(widget, "description", None)
        return isinstance(description, str) and description.strip() != ""

    @staticmethod
    def _doc_pane(doc: str | None) -> pn.pane.HTML | None:
        if not doc:
            return None
        return pn.pane.HTML(
            f'<div class="lw-import-datasets-config-help">{escape(doc)}</div>',
            margin=0,
        )

    @classmethod
    def _widget_block(cls, widget: object, *, doc: str | None = None) -> pn.Column:
        children = [widget]
        doc_pane = None if cls._widget_has_native_help(widget) else cls._doc_pane(doc)
        if doc_pane is not None:
            children.append(doc_pane)
        return pn.Column(*children, sizing_mode="stretch_width", margin=0)

    @classmethod
    def _param_controls(
        cls,
        obj: object,
        *,
        parameters: list[str],
        widget_overrides: dict[str, dict[str, object]] | None = None,
    ) -> pn.Column:
        param_pane = pn.Param(
            obj,
            parameters=parameters,
            widgets=widget_overrides or {},
            sizing_mode="stretch_width",
            show_name=False,
            expand_button=False,
            expand=False,
        )
        controls = []
        for name in parameters:
            widget = param_pane._widgets.get(name)
            if widget is None:
                continue
            controls.append(
                cls._widget_block(widget, doc=getattr(obj.param[name], "doc", None))
            )
        container = pn.Column(*controls, sizing_mode="stretch_width", margin=0)
        container._param_pane = param_pane
        return container

    @staticmethod
    def _param_section(
        title: str,
        obj: object,
        *,
        parameters: list[str] | None = None,
        widget_overrides: dict[str, dict[str, object]] | None = None,
    ) -> pn.Column:
        if parameters is None:
            parameters = [name for name in obj.param if name != "name"]
        return _config_family_box(
            title,
            _ImportDatasetsController._param_controls(
                obj,
                parameters=parameters,
                widget_overrides=widget_overrides,
            ),
        )

    def _sync_tracker_vector_widgets(self, *_events) -> None:
        if self._working_lab is None or not hasattr(
            self, "_tracker_front_vector_slider"
        ):
            return
        tracker = self._working_lab.tracker
        if tracker.Npoints > 0 and tracker.bend == "from_vectors":
            updates = {}
            if tracker.front_vector is None:
                updates["front_vector"] = (1, min(2, tracker.Npoints))
            if tracker.rear_vector is None:
                updates["rear_vector"] = (-min(2, tracker.Npoints), -1)
            if updates:
                tracker.param.update(**updates)
                return
        self._tracker_widget_syncing = True
        try:
            if tracker.Npoints > 0:
                front_value = tracker.front_vector or (1, min(2, tracker.Npoints))
                rear_tail = min(2, tracker.Npoints)
                rear_value = tracker.rear_vector or (-rear_tail, -1)
                self._tracker_front_vector_slider.start = 1
                self._tracker_front_vector_slider.end = tracker.Npoints
                self._tracker_front_vector_slider.value = front_value
                self._tracker_rear_vector_slider.start = -tracker.Npoints
                self._tracker_rear_vector_slider.end = -1
                self._tracker_rear_vector_slider.value = rear_value
            else:
                self._tracker_front_vector_slider.start = 1
                self._tracker_front_vector_slider.end = 1
                self._tracker_front_vector_slider.value = (1, 1)
                self._tracker_rear_vector_slider.start = -1
                self._tracker_rear_vector_slider.end = -1
                self._tracker_rear_vector_slider.value = (-1, -1)
            sliders_disabled = tracker.Npoints <= 0 or tracker.bend != "from_vectors"
            self._tracker_front_vector_slider.disabled = sliders_disabled
            self._tracker_rear_vector_slider.disabled = sliders_disabled
        finally:
            self._tracker_widget_syncing = False

    def _handle_tracker_front_vector_change(self, event) -> None:
        if self._tracker_widget_syncing or self._working_lab is None:
            return
        tracker = self._working_lab.tracker
        if tracker.Npoints <= 0 or tracker.bend != "from_vectors":
            return
        tracker.front_vector = tuple(event.new)

    def _handle_tracker_rear_vector_change(self, event) -> None:
        if self._tracker_widget_syncing or self._working_lab is None:
            return
        tracker = self._working_lab.tracker
        if tracker.Npoints <= 0 or tracker.bend != "from_vectors":
            return
        tracker.rear_vector = tuple(event.new)

    def _build_tracker_metric_section(self) -> pn.Column:
        tracker = self._working_lab.tracker
        tracker_top_controls = self._param_controls(
            tracker,
            parameters=[
                "XY_unit",
                "Npoints",
                "Ncontour",
                "point_idx",
            ],
            widget_overrides={
                "Npoints": {"type": pn.widgets.IntInput},
                "Ncontour": {"type": pn.widgets.IntInput},
                "point_idx": {"type": pn.widgets.IntInput},
            },
        )
        bend_control = self._param_controls(tracker, parameters=["bend"])
        tracker_tail_controls = self._param_controls(
            tracker,
            parameters=["front_body_ratio", "use_component_vel"],
        )
        self._tracker_front_vector_slider = pn.widgets.RangeSlider(name="Front vector")
        self._tracker_rear_vector_slider = pn.widgets.RangeSlider(name="Rear vector")
        self._tracker_front_vector_slider.param.watch(
            self._handle_tracker_front_vector_change, "value"
        )
        self._tracker_rear_vector_slider.param.watch(
            self._handle_tracker_rear_vector_change, "value"
        )
        tracker.param.watch(
            self._sync_tracker_vector_widgets,
            ["Npoints", "bend", "front_vector", "rear_vector"],
        )
        self._sync_tracker_vector_widgets()
        return _config_family_box(
            "Tracker Metrics",
            tracker_top_controls,
            bend_control,
            self._widget_block(
                self._tracker_front_vector_slider,
                doc=getattr(tracker.param["front_vector"], "doc", None),
            ),
            self._widget_block(
                self._tracker_rear_vector_slider,
                doc=getattr(tracker.param["rear_vector"], "doc", None),
            ),
            tracker_tail_controls,
        )

    def _build_tracker_framerate_section(self) -> pn.Column:
        tracker = self._working_lab.tracker
        return _config_family_box(
            "Tracker Framerate",
            self._param_controls(
                tracker,
                parameters=["fr", "dt", "constant_framerate"],
                widget_overrides={
                    "fr": {"type": pn.widgets.FloatInput},
                    "dt": {"type": pn.widgets.FloatInput},
                },
            ),
        )

    def _rebuild_lab_editor(self) -> None:
        if self._working_lab is None:
            self.lab_editor_sections.objects = [
                _config_family_box(
                    "Lab Format Configuration",
                    pn.pane.HTML(
                        '<div class="lw-import-datasets-summary">No LabFormat configuration is loaded.</div>',
                        margin=0,
                    ),
                )
            ]
            return
        self.lab_editor_sections.objects = [
            self._param_section("General", self._working_lab, parameters=["labID"]),
            self._build_tracker_metric_section(),
            self._build_tracker_framerate_section(),
            self._param_section("Filesystem", self._working_lab.filesystem),
            self._param_section("Environment", self._working_lab.env_params),
            self._param_section(
                "Preprocess",
                self._working_lab.preprocess,
                widget_overrides={
                    "rescale_by": {"type": pn.widgets.FloatInput},
                    "filter_f": {"type": pn.widgets.FloatInput},
                },
            ),
        ]

    def _load_working_lab(self, lab_id: str | None) -> None:
        if not lab_id:
            self._working_lab_id = None
            self._working_lab = None
            self.lab_config_name_input.value = ""
            self._rebuild_lab_editor()
            return
        self._working_lab_id = lab_id
        self._working_lab = reg.conf.LabFormat.get(lab_id)
        self.lab_config_name_input.value = lab_id
        self._rebuild_lab_editor()
        self._set_lab_status(f'Loaded LabFormat "{lab_id}".')

    def _build_working_lab_conf(self):
        if self._working_lab_id is None:
            raise RuntimeError("No LabFormat configuration is loaded.")
        rebuilt = self._working_lab.nestedConf.get_copy()
        target_id = self.lab_config_name_input.value.strip() or self._working_lab_id
        rebuilt["labID"] = target_id
        return rebuilt

    def _active_workspace_ready(self) -> bool:
        return self.workspace is not None

    def _raw_root_text(self) -> str:
        return self.raw_root_input.value.strip()

    def _raw_root_path(self) -> Path | None:
        raw_text = self._raw_root_text()
        if not raw_text:
            return None
        return Path(raw_text).expanduser()

    def _selected_candidate(self) -> RawDatasetCandidate | None:
        return self._candidate_by_key.get(self.candidate_select.value)

    def _set_status(
        self, text: str, *, tone: str = "neutral", detail: str | None = None
    ) -> None:
        self.status.object = _status_html(text, tone=tone, detail=detail)

    def _refresh_workspace_summary(self) -> None:
        if self.workspace is None:
            self.workspace_summary.object = (
                '<div class="lw-import-datasets-summary">'
                "No active workspace is configured."
                "</div>"
            )
            return
        proc_root = None
        if self.lab_select.value:
            try:
                proc_root = build_workspace_proc_folder(
                    self.workspace, self.lab_select.value
                )
            except Exception:
                proc_root = None
        target_html = ""
        if proc_root is not None:
            target_html = (
                f"<div><strong>Import target</strong>: {escape(str(proc_root))}</div>"
            )
        self.workspace_summary.object = (
            '<div class="lw-import-datasets-summary">'
            f"<div><strong>Workspace</strong>: {escape(str(self.workspace.root))}</div>"
            f"{target_html}"
            "</div>"
        )

    def _candidate_option_key(self, candidate: RawDatasetCandidate) -> str:
        return (
            f"{candidate.parent_dir}::{candidate.candidate_id}::{candidate.source_path}"
        )

    def _clear_candidates(self) -> None:
        self._candidate_by_key.clear()
        self.candidate_select.options = {"Select a candidate": ""}
        self.candidate_select.value = ""
        self.dataset_id_input.value = ""
        self.candidate_summary.object = _candidate_summary_html(None)
        self._selected_record_path = None

    def _sync_controls(self) -> None:
        workspace_ready = self._active_workspace_ready()
        source_ready = bool(
            workspace_ready and self.lab_select.value and self._raw_root_text()
        )
        candidate_ready = self._selected_candidate() is not None
        self.discover_button.disabled = not source_ready
        self.candidate_select.disabled = not bool(self._candidate_by_key)
        self.dataset_id_input.disabled = not candidate_ready
        self.group_id_input.disabled = not candidate_ready
        self.color_input.disabled = not candidate_ready
        self.import_button.disabled = not (workspace_ready and candidate_ready)
        self.lab_select.disabled = (
            not bool(self.lab_select.options) or not workspace_ready
        )
        self.raw_root_input.disabled = not workspace_ready
        self.browse_raw_root_button.disabled = not workspace_ready
        self.reset_button.disabled = not (
            workspace_ready and (self._raw_root_text() or self._candidate_by_key)
        )
        lab_ready = bool(self.lab_select.options)
        self.lab_config_name_input.disabled = not lab_ready
        self.lab_load_button.disabled = not lab_ready
        self.lab_save_button.disabled = not lab_ready
        self.lab_delete_button.disabled = not lab_ready or not self.lab_select.value
        self.lab_reset_button.disabled = not lab_ready or self._working_lab is None

    def _on_lab_select_change(self, *_events) -> None:
        self._load_working_lab(self.lab_select.value)
        self._clear_candidates()
        self._refresh_workspace_summary()
        if self.workspace is not None:
            self._set_status(
                "Source changed. Discover datasets again to refresh the candidate list."
            )
        self._sync_controls()

    def _on_raw_root_change(self, *_events) -> None:
        self._clear_candidates()
        self._refresh_workspace_summary()
        if self.workspace is not None:
            self._set_status(
                "Source changed. Discover datasets again to refresh the candidate list."
            )
        self._sync_controls()

    def _on_candidate_change(self, *_events) -> None:
        candidate = self._selected_candidate()
        if candidate is None:
            self.candidate_summary.object = _candidate_summary_html(None)
            self.dataset_id_input.value = ""
            self._sync_controls()
            return
        self.candidate_summary.object = _candidate_summary_html(candidate)
        self.dataset_id_input.value = candidate.candidate_id
        self._set_status(
            "Candidate selected. Review the import options and start the workspace import."
        )
        self._sync_controls()

    def _handle_lab_load(self, _event=None) -> None:
        self._load_working_lab(self.lab_select.value)
        self._sync_controls()

    def _handle_lab_save(self, _event=None) -> None:
        config_id = self.lab_config_name_input.value.strip()
        if not config_id:
            self._set_lab_status(
                "Enter a configuration ID before saving.", tone="warning"
            )
            return
        try:
            conf = self._build_working_lab_conf()
            reg.conf.LabFormat.setID(config_id, conf)
        except Exception as exc:
            self._set_lab_status(str(exc), tone="danger")
            return
        self._refresh_lab_options(select_id=config_id)
        self._load_working_lab(config_id)
        self._set_lab_status(
            f'LabFormat "{config_id}" saved to the registry.',
            tone="success",
        )
        self._refresh_workspace_summary()
        self._sync_controls()

    def _handle_lab_delete(self, _event=None) -> None:
        lab_id = self.lab_select.value
        if not lab_id:
            self._set_lab_status(
                "Select a LabFormat configuration first.", tone="warning"
            )
            return
        try:
            reg.conf.LabFormat.delete(lab_id)
        except Exception as exc:
            self._set_lab_status(str(exc), tone="danger")
            return
        self._set_lab_status(
            f'LabFormat "{lab_id}" deleted from the registry.',
            tone="success",
        )
        self._refresh_lab_options()
        self._load_working_lab(self.lab_select.value)
        self._clear_candidates()
        self._refresh_workspace_summary()
        self._sync_controls()

    def _handle_lab_reset(self, _event=None) -> None:
        self._load_working_lab(self.lab_select.value)
        if self.lab_select.value:
            self._set_lab_status(
                f'LabFormat "{self.lab_select.value}" reset to the stored registry version.',
                tone="success",
            )
        self._sync_controls()

    def _handle_reset(self, _event=None) -> None:
        self.raw_root_input.value = ""
        self.group_id_input.value = ""
        self.color_input.value = "#000000"
        self._clear_candidates()
        if self.workspace is not None:
            self._set_status(
                "Source state cleared. Enter a raw root path to start a new discovery pass."
            )
        self._sync_controls()

    def _handle_browse_raw_root(self, _event=None) -> None:
        fallback_dir = self.workspace.root if self.workspace is not None else None
        selected, error = pick_directory(
            initial_dir=self._raw_root_path(),
            fallback_dir=fallback_dir,
            title="Select raw dataset root",
        )
        if selected is not None:
            self.raw_root_input.value = str(selected)
            return
        if error is not None:
            self._set_status(error, tone="warning")
            self._sync_controls()

    def _handle_discover(self, _event=None) -> None:
        raw_root = self._raw_root_path()
        if self.workspace is None:
            self._set_status(
                "Configure an active workspace before importing datasets.",
                tone="warning",
            )
            self._sync_controls()
            return
        if raw_root is None:
            self._set_status("Enter a raw root path before discovery.", tone="warning")
            self._sync_controls()
            return
        candidates = discover_raw_datasets(self.lab_select.value, raw_root)
        self._clear_candidates()
        if not candidates:
            self._set_status(
                "No import candidates were found under the selected raw root.",
                tone="warning",
                detail=str(raw_root),
            )
            self._sync_controls()
            return
        options: dict[str, str] = {"Select a candidate": ""}
        for candidate in candidates:
            key = self._candidate_option_key(candidate)
            self._candidate_by_key[key] = candidate
            options[candidate.display_name] = key
        self.candidate_select.options = options
        self._set_status(
            f"Discovered {len(candidates)} candidate(s). Select one candidate to continue.",
            tone="success",
            detail=str(raw_root),
        )
        self._sync_controls()

    def _build_import_request(self) -> ImportRequest:
        candidate = self._selected_candidate()
        raw_root = self._raw_root_path()
        if candidate is None or raw_root is None:
            raise RuntimeError(
                "Import is not ready: select a discovered candidate first"
            )
        dataset_id = self.dataset_id_input.value.strip() or candidate.candidate_id
        group_id = self.group_id_input.value.strip() or None
        extra_kwargs = _candidate_import_overrides(
            self.lab_select.value,
            raw_root,
            candidate,
        )
        return ImportRequest(
            lab_id=self.lab_select.value,
            parent_dir=candidate.parent_dir,
            raw_folder=raw_root,
            group_id=group_id,
            dataset_id=dataset_id,
            color=(self.color_input.value or "#000000"),
            extra_kwargs=extra_kwargs,
        )

    def _handle_import(self, _event=None) -> None:
        if self.workspace is None:
            self._set_status(
                "Configure an active workspace before importing datasets.",
                tone="warning",
            )
            self._sync_controls()
            return
        try:
            request = self._build_import_request()
            record = import_into_workspace(request, workspace=self.workspace)
        except Exception as exc:
            self._set_status(str(exc), tone="danger")
            self._sync_controls()
            return
        self._selected_record_path = record.dataset_dir
        self._set_status(
            f'Dataset "{record.dataset_id}" imported into the active workspace.',
            tone="success",
            detail=str(record.dataset_dir),
        )
        self._sync_controls()

    def view(self) -> pn.viewable.Viewable:
        config_intro = pn.pane.HTML(
            (
                '<div class="lw-import-datasets-config-intro">'
                "Inspect and edit the selected `LabFormat` configuration before running dataset discovery and import. "
                "This embedded panel exposes the registry-backed tracker, filesystem, environment, and preprocess structure used by the import lane."
                "</div>"
            ),
            margin=0,
        )
        config_section = pn.Card(
            pn.Column(
                config_intro,
                pn.Row(
                    self.lab_select,
                    self.lab_config_name_input,
                    sizing_mode="stretch_width",
                    margin=0,
                ),
                pn.Row(
                    self.lab_load_button,
                    self.lab_save_button,
                    self.lab_delete_button,
                    self.lab_reset_button,
                    css_classes=["lw-import-datasets-config-actions"],
                    sizing_mode="stretch_width",
                    margin=(4, 0, 0, 0),
                ),
                self.lab_status,
                self.lab_editor_sections,
                sizing_mode="stretch_width",
            ),
            title="Lab Format Configuration",
            collapsed=False,
            sizing_mode="stretch_width",
        )
        raw_root_row = pn.Row(
            self.raw_root_input,
            self.browse_raw_root_button,
            css_classes=["lw-import-datasets-source-row"],
            sizing_mode="stretch_width",
        )
        source_section = _flow_section(
            "Source",
            self.lab_select,
            raw_root_row,
            self.discover_button,
            self.status,
        )
        discovery_section = _flow_section(
            "Discovery",
            self.candidate_select,
            self.candidate_summary,
        )
        import_section = _flow_section(
            "Import Options",
            pn.Row(
                pn.Column(
                    self.dataset_id_input,
                    self.color_input,
                    margin=(0, 0, 0, 0),
                    width=260,
                ),
                self.group_id_input,
                sizing_mode="stretch_width",
            ),
            pn.Row(
                self.import_button,
                self.reset_button,
                sizing_mode="stretch_width",
            ),
            pn.Spacer(height=8),
            self.workspace_summary,
        )
        workflow_section = pn.Card(
            pn.Column(
                source_section,
                discovery_section,
                import_section,
                sizing_mode="stretch_width",
            ),
            title="Import Workflow",
            collapsed=False,
            sizing_mode="stretch_width",
        )
        intro = pn.pane.HTML(
            (
                '<div class="lw-import-datasets-intro">'
                "Import one experimental raw dataset into the active workspace through a small workspace-first pipeline. "
                "Use the Source step to choose a lab format and point the app at a local raw-data folder, then run Discovery to resolve one import candidate and review its candidate-specific warnings before importing. "
                "The app writes into workspace-owned dataset storage, reuses the central Larvaworld import backend, and does not register references or set global active-dataset state. "
                f'See the data-processing documentation on Read the Docs for the broader dataset pipeline: <a href="{escape(DOCS_DATA_PROCESSING)}" target="_blank">Read the Docs</a>.'
                "</div>"
            ),
            margin=0,
        )
        return pn.Column(
            intro,
            config_section,
            workflow_section,
            css_classes=["lw-import-datasets-root"],
            sizing_mode="stretch_width",
        )


def import_datasets_app() -> pn.viewable.Viewable:
    pn.extension(raw_css=[PORTAL_RAW_CSS, IMPORT_DATASETS_RAW_CSS])
    controller = _ImportDatasetsController()

    template = pn.template.MaterialTemplate(
        title="",
        header_background="#b0b4c2",
        header_color="#111111",
    )
    template.header.append(build_app_header(title="Import Experimental Datasets"))
    template.main.append(controller.view())
    return template
