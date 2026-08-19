from __future__ import annotations

from dataclasses import dataclass
from html import escape
import inspect
from pathlib import Path

import panel as pn

from larvaworld.lib import reg
from larvaworld.lib.reg.generators import LabFormat
from larvaworld.portal.buttons import reset_button
from larvaworld.portal.config_widgets import (
    ConftypeActionsController,
    build_env_params_widget,
)
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


@dataclass(frozen=True)
class _MergeTarget:
    target_id: str
    parent_dir: str
    display_name: str
    source_path: Path
    children: tuple[RawDatasetCandidate, ...]


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

.lw-import-datasets-source-action-row {
  gap: 10px;
  align-items: center;
}

.lw-import-datasets-candidate-row {
  gap: 10px;
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

.lw-import-datasets-config-subfamily {
  background: rgba(243, 245, 248, 0.92);
  border: 1px solid rgba(90, 71, 96, 0.10);
  border-radius: 9px;
  padding: 9px 10px 8px 10px;
  margin-top: 6px;
}

.lw-import-datasets-config-subfamily-title {
  margin: 0 0 6px 0;
  color: #5c4668;
  font-weight: 700;
}

.lw-import-datasets-config-subfamily-card {
  margin-top: 6px;
}

.lw-import-datasets-config-compact-card .bk-card-header,
.lw-import-datasets-config-compact-card .bk-Card-header,
.lw-import-datasets-config-compact-card .card-header {
  font-size: 12px;
}

.lw-import-datasets-config-compact-card .bk-card-title,
.lw-import-datasets-config-compact-card .bk-Card-title,
.lw-import-datasets-config-compact-card .card-title,
.lw-import-datasets-config-compact-card .bk-btn-group > .bk-btn,
.lw-import-datasets-config-compact-card .bk-btn-group > button,
.lw-import-datasets-config-compact-card button.bk-btn {
  font-size: 12px !important;
  line-height: 1.2 !important;
}

.lw-import-datasets-config-compact-card .bk-card-header .bk-btn,
.lw-import-datasets-config-compact-card .bk-Card-header .bk-btn,
.lw-import-datasets-config-compact-card .card-header .bk-btn {
  font-size: 12px;
}

.lw-import-datasets-config-family-header {
  align-items: center;
}

.lw-import-datasets-inline-action-row {
  align-items: flex-end;
  gap: 10px;
}

.lw-import-datasets-config-grid {
  gap: 12px;
  align-items: flex-start;
}

.lw-import-datasets-config-grid-col {
  flex: 1 1 0;
  min-width: 0;
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


def _merge_target_summary_html(target: _MergeTarget | None) -> str:
    if target is None:
        return (
            '<div class="lw-import-datasets-summary">'
            "No merge target selected yet."
            "</div>"
        )
    unit = "dataset" if len(target.children) == 1 else "datasets"
    return (
        '<div class="lw-import-datasets-summary">'
        f"<div><strong>Merge target</strong>: {escape(target.target_id)}</div>"
        f"<div><strong>Parent dir</strong>: {escape(target.parent_dir)}</div>"
        f"<div><strong>Source path</strong>: {escape(str(target.source_path))}</div>"
        f"<div><strong>Child candidates</strong>: {len(target.children)} {unit}</div>"
        "</div>"
    )


def _relative_source_dir(raw_root: Path, source_path: Path) -> str:
    rel = source_path.resolve().relative_to(raw_root.resolve())
    return "." if str(rel) == "." else rel.as_posix()


def _merge_target_dataset_id(target: _MergeTarget, raw_root: Path) -> str:
    if target.parent_dir == ".":
        return raw_root.name
    return Path(target.parent_dir).name


def _merge_target_display_name(
    parent_dir: str, raw_root: Path, children: tuple[RawDatasetCandidate, ...]
) -> str:
    name = raw_root.name if parent_dir == "." else parent_dir
    unit = "dataset" if len(children) == 1 else "datasets"
    return f"{name} ({len(children)} {unit})"


def _lab_supports_merged_import(lab_id: str | None) -> bool:
    if not lab_id:
        return False
    lab = reg.conf.LabFormat.get(str(lab_id).strip())
    core_supplied = {"tracker", "filesystem", "source_dir", "source_files"}
    for name, parameter in inspect.signature(lab.import_func).parameters.items():
        if name in core_supplied:
            continue
        if parameter.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            continue
        if parameter.default is inspect.Parameter.empty:
            return False
    return True


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


def _import_failure_status_message(exc: BaseException) -> str:
    message = str(exc)
    if message.startswith("Import failed:"):
        return message
    return f"Import failed: {message}"


class _ImportDatasetsController:
    def __init__(self) -> None:
        self.workspace = get_active_workspace()
        self._candidate_by_key: dict[str, RawDatasetCandidate] = {}
        self._selected_record_path: Path | None = None
        self._working_lab_id: str | None = None
        self._working_lab = None
        self._tracker_widget_syncing = False
        self._action_running = False
        self._discovered_candidates: tuple[RawDatasetCandidate, ...] = ()
        self._merge_target_by_key: dict[str, _MergeTarget] = {}

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
        self.lab_actions = ConftypeActionsController(
            LabFormat,
            conftype="LabFormat",
            build_save_payload=self._build_working_lab_conf,
            get_selected_id=lambda: self.lab_select.value,
            get_save_id=lambda: (
                self.lab_config_name_input.value.strip() or self.lab_select.value
            ),
            on_load=self._apply_loaded_lab_config,
            on_save=self._after_lab_save,
            on_delete=self._after_lab_delete,
            on_reset=self._after_lab_reset,
            on_status=self._set_lab_status,
            confirm_reset=False,
        )
        self.lab_load_button = self.lab_actions.load_button
        self.lab_save_button = self.lab_actions.save_button
        self.lab_delete_button = self.lab_actions.delete_button
        self.lab_reset_button = self.lab_actions.reset_button
        self.raw_root_input = pn.widgets.TextInput(
            name="Raw root",
            placeholder="/path/to/raw/data",
            sizing_mode="stretch_width",
            css_classes=["lw-import-datasets-source-input"],
        )
        self.browse_raw_root_button = pn.widgets.Button(
            name="Browse",
            button_type="default",
            width=110,
            css_classes=["lw-import-datasets-source-browse"],
        )
        self.reset_button = reset_button(
            name="Reset source", button_type="default", width=140
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
            sizing_mode="stretch_width",
        )
        self.candidate_select.description = "Select one discovered candidate to inspect its source path and warnings before importing it into the active workspace."
        self.merged_checkbox = pn.widgets.Checkbox(
            name="Merged",
            value=False,
            width=90,
        )
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
        self.workspace_summary = pn.pane.HTML(
            "", margin=(5, 10), sizing_mode="stretch_width"
        )
        self.candidate_summary = pn.pane.HTML(
            _candidate_summary_html(None),
            margin=(5, 10),
            sizing_mode="stretch_width",
        )
        self.lab_status = pn.pane.HTML("", margin=(5, 10), sizing_mode="stretch_width")
        self.lab_editor_sections = pn.Column(sizing_mode="stretch_width", margin=0)
        self.status = pn.pane.HTML("", margin=(5, 10), sizing_mode="stretch_width")

        self.lab_select.param.watch(self._on_lab_select_change, "value")
        self.raw_root_input.param.watch(self._on_raw_root_change, "value")
        self.candidate_select.param.watch(self._on_candidate_change, "value")
        self.merged_checkbox.param.watch(self._on_merged_change, "value")
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
            self._set_status(
                "Select a raw root folder, discover candidates, then import one "
                "dataset into the active workspace."
            )
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

    def _build_environment_section(self) -> pn.Column:
        env_content = build_env_params_widget(self._working_lab.env_params, wrap=False)
        env_children = list(getattr(env_content, "objects", []) or [])
        if len(env_children) >= 4:
            left_col = pn.Column(
                env_children[0],
                css_classes=["lw-import-datasets-config-grid-col"],
                sizing_mode="stretch_width",
                margin=0,
            )
            middle_col = pn.Column(
                env_children[1],
                css_classes=["lw-import-datasets-config-grid-col"],
                sizing_mode="stretch_width",
                margin=0,
            )
            right_col = pn.Column(
                env_children[2],
                env_children[3],
                css_classes=["lw-import-datasets-config-grid-col"],
                sizing_mode="stretch_width",
                margin=0,
            )
            body = pn.Row(
                left_col,
                middle_col,
                right_col,
                css_classes=["lw-import-datasets-config-grid"],
                sizing_mode="stretch_width",
                margin=0,
            )
        else:
            body = env_content
        return _config_family_box("Environment", body)

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
        tracker_column = pn.Column(
            self._build_tracker_metric_section(),
            self._build_tracker_framerate_section(),
            css_classes=["lw-import-datasets-config-grid-col"],
            sizing_mode="stretch_width",
            margin=0,
        )
        general_column = pn.Column(
            self._param_section("General", self._working_lab, parameters=["labID"]),
            self._param_section("Filesystem", self._working_lab.filesystem),
            self._param_section(
                "Preprocess",
                self._working_lab.preprocess,
                widget_overrides={
                    "rescale_by": {"type": pn.widgets.FloatInput},
                    "filter_f": {"type": pn.widgets.FloatInput},
                },
            ),
            css_classes=["lw-import-datasets-config-grid-col"],
            sizing_mode="stretch_width",
            margin=0,
        )
        self.lab_editor_sections.objects = [
            tracker_column,
            general_column,
            self._build_environment_section(),
        ]

    def _load_working_lab(self, lab_id: str | None) -> None:
        if not lab_id:
            self._working_lab_id = None
            self._working_lab = None
            self.lab_config_name_input.value = ""
            self._rebuild_lab_editor()
            self._set_lab_status("No LabFormat selected.")
            return
        self._apply_loaded_lab_config(lab_id, reg.conf.LabFormat.get(lab_id))
        self._set_lab_status(
            f'✓ Loaded LabFormat "{lab_id}". Configuration panel updated above.'
        )

    def _apply_loaded_lab_config(self, lab_id: str, lab_config: object) -> None:
        self._working_lab_id = lab_id
        self._working_lab = lab_config
        self.lab_config_name_input.value = lab_id
        self._rebuild_lab_editor()

    def _build_working_lab_conf(self, config_id: str | None = None):
        if self._working_lab_id is None:
            raise RuntimeError("No LabFormat configuration is loaded.")
        rebuilt = self._working_lab.nestedConf.get_copy()
        thermoscape = self._working_lab.env_params.thermoscape
        if thermoscape is not None:
            thermoscape_payload = rebuilt["env_params"].get("thermoscape") or {}
            if hasattr(thermoscape, "plate_temp"):
                thermoscape_payload["plate_temp"] = thermoscape.plate_temp
            if hasattr(thermoscape, "thermo_spread"):
                thermoscape_payload["spread"] = thermoscape.thermo_spread
            if hasattr(thermoscape, "thermo_sources"):
                thermoscape_payload["thermo_sources"] = thermoscape.thermo_sources
            if hasattr(thermoscape, "thermo_source_dTemps"):
                thermoscape_payload["thermo_source_dTemps"] = (
                    thermoscape.thermo_source_dTemps
                )
            rebuilt["env_params"]["thermoscape"] = thermoscape_payload
        target_id = config_id or self.lab_config_name_input.value.strip()
        target_id = target_id or self._working_lab_id
        rebuilt["labID"] = target_id
        return rebuilt

    def _after_lab_save(self, config_id: str, _payload: object) -> None:
        self._refresh_lab_options(select_id=config_id)
        self._load_working_lab(config_id)
        self._refresh_workspace_summary()
        self._sync_controls()

    def _after_lab_delete(self, _config_id: str) -> None:
        self._refresh_lab_options()
        self._load_working_lab(self.lab_select.value)
        self._clear_candidates()
        self._refresh_workspace_summary()
        self._sync_controls()

    def _after_lab_reset(self, selected_lab_id: str | None) -> None:
        self._refresh_lab_options(select_id=selected_lab_id)
        self._load_working_lab(self.lab_select.value)
        self._clear_candidates()
        self._refresh_workspace_summary()
        self._sync_controls()

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

    def _selected_merge_target(self) -> _MergeTarget | None:
        return self._merge_target_by_key.get(self.candidate_select.value)

    def _merged_mode(self) -> bool:
        return bool(self.merged_checkbox.value)

    def _merged_import_supported(self) -> bool:
        return _lab_supports_merged_import(self.lab_select.value)

    def _set_status(
        self, text: str, *, tone: str = "neutral", detail: str | None = None
    ) -> None:
        self.status.object = _status_html(text, tone=tone, detail=detail)

    def _begin_action_status(
        self, text: str, *, detail: str | None = None, tone: str = "neutral"
    ) -> None:
        self._action_running = True
        self._set_status(text, tone=tone, detail=detail)
        self._sync_controls()

    def _finish_action_status(self) -> None:
        self._action_running = False
        self._sync_controls()

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

    def _merge_target_option_key(self, target: _MergeTarget) -> str:
        return f"{target.parent_dir}::{target.source_path}"

    def _clear_candidates(self) -> None:
        self._discovered_candidates = ()
        self._candidate_by_key.clear()
        self._merge_target_by_key.clear()
        if self._merged_mode():
            self.candidate_select.name = "Merge target"
            self.candidate_select.options = {"Select a merge target": ""}
            self.candidate_summary.object = _merge_target_summary_html(None)
        else:
            self.candidate_select.name = "Candidate"
            self.candidate_select.options = {"Select a candidate": ""}
            self.candidate_summary.object = _candidate_summary_html(None)
        self.candidate_select.value = ""
        self.dataset_id_input.value = ""
        self._selected_record_path = None

    def _clear_selection(self) -> None:
        placeholder = (
            "Select a merge target" if self._merged_mode() else "Select a candidate"
        )
        self.candidate_select.value = ""
        self.dataset_id_input.value = ""
        self.candidate_summary.object = (
            _merge_target_summary_html(None)
            if self._merged_mode()
            else _candidate_summary_html(None)
        )
        if not self.candidate_select.options:
            self.candidate_select.options = {placeholder: ""}

    def _build_merge_targets(
        self, raw_root: Path, candidates: tuple[RawDatasetCandidate, ...]
    ) -> tuple[_MergeTarget, ...]:
        if not candidates:
            return ()
        lab = reg.conf.LabFormat.get(str(self.lab_select.value).strip())
        filesystem = lab.filesystem
        folder_based = bool(filesystem.folder_pref or filesystem.folder_suff)
        grouped: dict[str, list[RawDatasetCandidate]] = {}
        source_paths: dict[str, Path] = {}
        for candidate in candidates:
            if folder_based:
                source_path = candidate.source_path.resolve().parent
                parent_dir = _relative_source_dir(raw_root, source_path)
            else:
                parent_dir = candidate.parent_dir
                source_path = (
                    raw_root.resolve()
                    if raw_root.is_file() or parent_dir == "."
                    else (raw_root / parent_dir).resolve()
                )
            grouped.setdefault(parent_dir, []).append(candidate)
            source_paths[parent_dir] = source_path

        targets: list[_MergeTarget] = []
        for parent_dir in sorted(grouped):
            children = tuple(
                sorted(
                    grouped[parent_dir],
                    key=lambda candidate: (
                        candidate.parent_dir,
                        candidate.candidate_id,
                        str(candidate.source_path),
                    ),
                )
            )
            targets.append(
                _MergeTarget(
                    target_id=parent_dir,
                    parent_dir=parent_dir,
                    display_name=_merge_target_display_name(
                        parent_dir, raw_root, children
                    ),
                    source_path=source_paths[parent_dir],
                    children=children,
                )
            )
        return tuple(targets)

    def _refresh_candidate_options(self) -> None:
        previous_value = self.candidate_select.value
        if self._merged_mode():
            options: dict[str, str] = {"Select a merge target": ""}
            self._merge_target_by_key.clear()
            for target in self._merge_targets():
                key = self._merge_target_option_key(target)
                self._merge_target_by_key[key] = target
                options[target.display_name] = key
            self.candidate_select.name = "Merge target"
        else:
            options = {"Select a candidate": ""}
            self._candidate_by_key.clear()
            for candidate in self._discovered_candidates:
                key = self._candidate_option_key(candidate)
                self._candidate_by_key[key] = candidate
                options[candidate.display_name] = key
            self.candidate_select.name = "Candidate"
        self.candidate_select.options = options
        if previous_value in options.values():
            self.candidate_select.value = previous_value
        else:
            self.candidate_select.value = ""

    def _merge_targets(self) -> tuple[_MergeTarget, ...]:
        raw_root = self._raw_root_path()
        if raw_root is None:
            return ()
        return self._build_merge_targets(raw_root, self._discovered_candidates)

    def _sync_controls(self) -> None:
        workspace_ready = self._active_workspace_ready()
        source_ready = bool(
            workspace_ready and self.lab_select.value and self._raw_root_text()
        )
        selection_ready = (
            self._selected_merge_target() is not None
            if self._merged_mode()
            else self._selected_candidate() is not None
        )
        options_ready = (
            bool(self._merge_target_by_key)
            if self._merged_mode()
            else bool(self._candidate_by_key)
        )
        busy = self._action_running
        merged_supported = self._merged_import_supported()

        self.discover_button.disabled = busy or not source_ready
        self.candidate_select.disabled = busy or not options_ready
        self.dataset_id_input.disabled = busy or not selection_ready
        self.group_id_input.disabled = busy or not selection_ready
        self.color_input.disabled = busy or not selection_ready
        self.import_button.disabled = busy or not (workspace_ready and selection_ready)
        self.lab_select.disabled = (
            not bool(self.lab_select.options) or not workspace_ready
        )
        self.raw_root_input.disabled = not workspace_ready or busy
        self.browse_raw_root_button.disabled = not workspace_ready or busy
        self.merged_checkbox.disabled = (
            not workspace_ready or busy or not merged_supported
        )
        self.reset_button.disabled = busy or not (
            workspace_ready and (self._raw_root_text() or self._candidate_by_key)
        )
        lab_ready = bool(self.lab_select.options)
        self.lab_config_name_input.disabled = not lab_ready or busy
        self.lab_load_button.disabled = not lab_ready or busy
        self.lab_save_button.disabled = not lab_ready or busy
        self.lab_delete_button.disabled = (
            not lab_ready or not self.lab_select.value or busy
        )
        self.lab_reset_button.disabled = busy

    def _on_lab_select_change(self, *_events) -> None:
        self._load_working_lab(self.lab_select.value)
        if self._merged_mode() and not self._merged_import_supported():
            self.merged_checkbox.value = False
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
        if self._merged_mode():
            target = self._selected_merge_target()
            if target is None:
                self.candidate_summary.object = _merge_target_summary_html(None)
                self.dataset_id_input.value = ""
                self._sync_controls()
                return
            self.candidate_summary.object = _merge_target_summary_html(target)
            raw_root = self._raw_root_path()
            if raw_root is not None:
                self.dataset_id_input.value = _merge_target_dataset_id(target, raw_root)
            self._set_status(
                "Merge target selected. Review the import options and start the workspace import."
            )
            self._sync_controls()
            return

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

    def _on_merged_change(self, *_events) -> None:
        if self._merged_mode() and not self._merged_import_supported():
            self.merged_checkbox.value = False
            self._refresh_candidate_options()
            self._clear_selection()
            self._set_status(
                f'Merged import is not supported for "{self.lab_select.value}" '
                "because this lab format requires a source id.",
                tone="warning",
            )
            self._sync_controls()
            return
        self._refresh_candidate_options()
        self._clear_selection()
        if self.workspace is not None and self._discovered_candidates:
            if self._merged_mode():
                self._set_status(
                    "Merged mode enabled. Select a source folder whose child datasets should be merged."
                )
            else:
                self._set_status(
                    "Merged mode disabled. Select one candidate to continue."
                )
        self._sync_controls()

    def _handle_lab_load(self, _event=None) -> None:
        self.lab_actions.load_selected()
        self._sync_controls()

    def _handle_lab_save(self, _event=None) -> None:
        self.lab_actions.save_current()

    def _handle_lab_delete(self, _event=None) -> None:
        self.lab_actions.delete_selected()

    def _handle_lab_reset(self, _event=None) -> None:
        self.lab_actions.reset_store()

    def _handle_reset(self, _event=None) -> None:
        self.raw_root_input.value = ""
        self.group_id_input.value = ""
        self.color_input.value = "#000000"
        self.merged_checkbox.value = False
        self._clear_candidates()
        if self.workspace is not None:
            self._set_status(
                "Source state cleared. Enter a raw root path to start a new discovery pass."
            )
        self._sync_controls()

    def _handle_browse_raw_root(self, _event=None) -> None:
        if self.workspace is None:
            self._set_status(
                "Configure an active workspace before importing datasets.",
                tone="warning",
            )
            self._sync_controls()
            return
        self._set_status(
            "Opening folder picker...",
            detail="Select the raw dataset root folder, or enter a DeepLabCut ZIP path manually.",
        )
        self._sync_controls()
        selected, error = pick_directory(
            self._raw_root_path(),
            fallback_dir=self.workspace.root,
            title="Select raw dataset root",
        )
        if selected is not None:
            self.raw_root_input.value = str(selected)
            self._sync_controls()
            return
        if error is not None:
            self._set_status(error, tone="warning")
            self._sync_controls()
            return
        self._set_status("Browse cancelled. Raw root was not changed.")
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
        self._begin_action_status(
            "Discovering raw dataset candidates...",
            detail=str(raw_root),
        )
        try:
            candidates = discover_raw_datasets(self.lab_select.value, raw_root)
        except Exception as exc:
            self._set_status(f"Discovery failed: {exc}", tone="danger")
            return
        finally:
            self._finish_action_status()

        self._clear_candidates()
        if not candidates:
            self._set_status(
                "No import candidates were found under the selected raw root.",
                tone="warning",
                detail=str(raw_root),
            )
            self._sync_controls()
            return
        self._discovered_candidates = tuple(candidates)
        self._refresh_candidate_options()
        self._set_status(
            (
                f"Discovered {len(candidates)} candidate(s). Select one candidate to continue."
                if not self._merged_mode()
                else "Merged mode enabled. Select a source folder whose child datasets should be merged."
            ),
            tone="success",
            detail=str(raw_root),
        )
        self._sync_controls()

    def _build_import_request(self) -> ImportRequest:
        raw_root = self._raw_root_path()
        if raw_root is None:
            raise RuntimeError(
                "Import is not ready: select a raw root before importing"
            )
        group_id = self.group_id_input.value.strip() or None

        if self._merged_mode():
            if not self._merged_import_supported():
                raise RuntimeError(
                    f'Merged import is not supported for "{self.lab_select.value}" '
                    "because this lab format requires a source id."
                )
            target = self._selected_merge_target()
            if target is None:
                raise RuntimeError(
                    "Import is not ready: select a discovered merge target first"
                )
            dataset_id = (
                self.dataset_id_input.value.strip()
                or _merge_target_dataset_id(target, raw_root)
            )
            return ImportRequest(
                lab_id=self.lab_select.value,
                parent_dir=target.parent_dir,
                raw_folder=raw_root,
                group_id=group_id,
                dataset_id=dataset_id,
                merged=True,
                color=(self.color_input.value or "#000000"),
                extra_kwargs={},
            )

        candidate = self._selected_candidate()
        if candidate is None:
            raise RuntimeError(
                "Import is not ready: select a discovered candidate first"
            )
        dataset_id = self.dataset_id_input.value.strip() or candidate.candidate_id
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
            merged=self.merged_checkbox.value,
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
        except Exception as exc:
            self._set_status(str(exc), tone="danger")
            self._sync_controls()
            return
        dataset_id = request.dataset_id
        raw_folder = request.raw_folder
        detail_lines = [
            f"Dataset ID: {dataset_id}",
            f"Raw root: {raw_folder}",
        ]
        self._begin_action_status(
            "Importing dataset into the active workspace...",
            detail="\n".join(detail_lines),
        )

        def _run() -> None:
            self._execute_import(request)

        curdoc = pn.state.curdoc
        if curdoc is not None:
            curdoc.add_next_tick_callback(_run)
        else:
            _run()

    def _execute_import(self, request: ImportRequest) -> None:
        try:
            record = import_into_workspace(request, workspace=self.workspace)
        except Exception as exc:
            self._set_status(_import_failure_status_message(exc), tone="danger")
        else:
            self._selected_record_path = record.dataset_dir
            self._set_status(
                f'Dataset "{record.dataset_id}" imported into the active workspace.',
                tone="success",
                detail=str(record.dataset_dir),
            )
        finally:
            self._finish_action_status()

    def view(self) -> pn.viewable.Viewable:
        raw_root_row = pn.Row(
            self.raw_root_input,
            css_classes=["lw-import-datasets-source-row"],
            sizing_mode="stretch_width",
        )
        source_action_row = pn.Row(
            self.browse_raw_root_button,
            self.discover_button,
            css_classes=["lw-import-datasets-source-action-row"],
            sizing_mode="stretch_width",
        )
        candidate_row = pn.Row(
            self.candidate_select,
            css_classes=["lw-import-datasets-candidate-row"],
            sizing_mode="stretch_width",
        )
        merged_row = pn.Row(
            self.merged_checkbox,
            css_classes=["lw-import-datasets-candidate-row"],
            sizing_mode="stretch_width",
        )
        import_section = _flow_section(
            "Import Options",
            pn.Column(
                pn.Column(
                    self.dataset_id_input,
                    self.color_input,
                    margin=(0, 0, 0, 0),
                    width=260,
                ),
                self.group_id_input,
                sizing_mode="stretch_width",
                margin=0,
            ),
            pn.Row(
                self.import_button,
                self.reset_button,
                sizing_mode="stretch_width",
            ),
            self.status,
            pn.Spacer(height=8),
            self.workspace_summary,
        )
        lab_source_section = _flow_section(
            "Lab Format Setup",
            pn.Column(
                self.lab_select,
                self.lab_config_name_input,
                sizing_mode="stretch_width",
                margin=0,
            ),
            pn.Row(
                pn.Column(
                    self.lab_actions.view,
                    sizing_mode="stretch_width",
                    margin=0,
                ),
                pn.Spacer(sizing_mode="stretch_width"),
                sizing_mode="stretch_width",
                margin=0,
            ),
            raw_root_row,
            source_action_row,
            candidate_row,
            merged_row,
        )
        intro_text = pn.pane.HTML(
            (
                "<p>Import experimental raw datasets into the active workspace. "
                "Edit the LabFormat configuration, discover candidates, inspect warnings, and import.</p>"
                "<p><strong>Workflow:</strong> Select lab format → Browse/enter raw root → Discover datasets → "
                "Select candidate → Configure import options → Import</p>"
                f'<p><a href="{escape(DOCS_DATA_PROCESSING)}" target="_blank">📚 Dataset Processing Guide</a> — '
                "View the complete dataset pipeline documentation.</p>"
            ),
            margin=0,
        )
        intro = pn.Card(
            intro_text,
            title="ℹ️ About Dataset Import",
            collapsed=True,
            collapsible=True,
            css_classes=["lw-portal-app-info"],
            sizing_mode="stretch_width",
            margin=(0, 0, 12, 0),
        )
        column_one = pn.Column(
            lab_source_section,
            import_section,
            css_classes=["lw-import-datasets-config-grid-col"],
            sizing_mode="stretch_width",
            margin=0,
        )
        top_row = pn.Row(
            column_one,
            self.lab_editor_sections.objects[0],
            self.lab_editor_sections.objects[1],
            css_classes=["lw-import-datasets-config-grid"],
            sizing_mode="stretch_width",
            margin=0,
        )
        return pn.Column(
            intro,
            top_row,
            self.lab_editor_sections.objects[2],
            css_classes=["lw-import-datasets-root"],
            sizing_mode="stretch_width",
        )


def import_datasets_app() -> pn.viewable.Viewable:
    pn.extension("tabulator", raw_css=[PORTAL_RAW_CSS, IMPORT_DATASETS_RAW_CSS])
    controller = _ImportDatasetsController()

    template = pn.template.MaterialTemplate(
        title="",
        header_background="#b0b4c2",
        header_color="#111111",
    )
    template.header.append(build_app_header(title="Import Experimental Datasets"))
    template.main.append(controller.view())
    return template
