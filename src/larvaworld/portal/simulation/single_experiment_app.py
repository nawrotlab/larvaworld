from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import holoviews as hv
import panel as pn
import param

from larvaworld.lib import reg, screen, sim, util
from larvaworld.lib.param.custom import ClassAttr, ClassDict
from larvaworld.lib.sim.validation import (
    CompatibilityIssue,
    CompatibilityReport,
    validate_experiment_environment_compatibility,
)
from larvaworld.portal.canvas_widgets import (
    EnvironmentCanvas,
    LarvaPreviewFrame,
    env_params_to_canvas_state,
)
from larvaworld.portal.landing_registry import (
    DOCS_EXPERIMENT_TYPES,
    DOCS_SINGLE_EXPERIMENTS,
)
from larvaworld.portal.panel_components import PORTAL_RAW_CSS, build_app_header
from larvaworld.portal.config_widgets.collections_widget import (
    build_collections_widget,
)
from larvaworld.portal.config_widgets.env_widget import build_env_params_widget
from larvaworld.portal.config_widgets.enrichment_widget import build_enrichment_widget
from larvaworld.portal.config_widgets.larvagroup_widget import (
    build_larva_groups_widget,
)
from larvaworld.portal.config_widgets.sim_ops_widget import build_sim_ops_widget
from larvaworld.portal.config_widgets.preset_controls import (
    USER_PRESET_POLICY,
    PresetControlsController,
    PresetRef,
    PresetSource,
    WorkspacePresetStore,
)
from larvaworld.portal.config_widgets.trials_widget import build_trials_widget
from larvaworld.portal.simulation.parameter_resolution import (
    _builder_obstacle_border_vertices,
    _coerce_xy_sequences,
    _normalize_scalar,
    apply_environment_payload,
    resolve_base_experiment_parameters,
)
from larvaworld.portal.simulation.preview_frames import generate_preview_frames
from larvaworld.portal.workspace import WorkspaceError, get_workspace_dir


__all__ = [
    "_SingleExperimentController",
    "_default_run_name",
    "_editor_group_title",
    "_safe_slug",
    "single_experiment_app",
]


SINGLE_EXPERIMENT_RAW_CSS = """
.lw-single-exp-root {
  padding: 14px 12px 20px 12px;
}

.lw-single-exp-intro {
  border-left: 4px solid #b5c2b0;
  background: rgba(181,194,176,0.14);
  border-radius: 10px;
  padding: 10px 12px;
  margin: 0 0 10px 0;
}

.lw-single-exp-intro a {
  color: #2f4858;
}

.lw-single-exp-preview-placeholder {
  padding: 22px 20px;
  border: 1px dashed rgba(17, 17, 17, 0.18);
  border-radius: 12px;
  background: rgba(248, 250, 252, 0.9);
  color: rgba(17, 17, 17, 0.72);
  line-height: 1.55;
}

.lw-single-exp-preview-body {
  min-height: 480px;
}

.lw-single-exp-preview-canvas-row {
  align-items: flex-start;
  gap: 12px;
}

.lw-single-exp-run-info-box {
  font-size: 12px;
  line-height: 1.55;
  color: rgba(17, 17, 17, 0.82);
  padding: 10px 12px;
  background: rgba(181,194,176,0.14);
  border-left: 3px solid #b5c2b0;
  border-radius: 8px;
}

.lw-inline-help-link .bk-btn {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  color: #2f4858 !important;
  font-size: 11px !important;
  line-height: 1.3 !important;
  padding: 0 !important;
  min-height: 0 !important;
  text-decoration: underline;
}

.lw-inline-help-link .bk-btn:hover,
.lw-inline-help-link .bk-btn:focus {
  color: #1f3542 !important;
  text-decoration: underline;
}

.lw-single-exp-shortcuts-overlay {
  position: fixed;
  inset: 0;
  z-index: 2000;
  background: rgba(15, 23, 42, 0.36);
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 24px;
}

.lw-single-exp-shortcuts-dialog {
  width: min(560px, 95vw);
  max-height: 85vh;
  overflow: auto;
  border-radius: 10px;
  border: 1px solid rgba(17, 17, 17, 0.14);
  background: #fff;
  color: rgba(17, 17, 17, 0.86);
  padding: 12px 14px;
}

.lw-single-exp-shortcuts-note {
  font-size: 12px;
  line-height: 1.45;
  margin: 0 0 8px 0;
}

.lw-single-exp-shortcuts-table-wrap {
  border-radius: 8px;
  border: 1px solid rgba(17, 17, 17, 0.1);
  background: rgba(181,194,176,0.14);
  padding: 10px 10px 8px 10px;
}

.lw-single-exp-shortcuts-table-wrap h4 {
  margin: 0 0 8px 0;
  font-size: 13px;
  line-height: 1.2;
}

.lw-single-exp-shortcuts-table-wrap table {
  width: 100%;
  border-collapse: collapse;
}

.lw-single-exp-shortcuts-table-wrap td {
  padding: 3px 0;
  vertical-align: top;
}

.lw-single-exp-shortcuts-table-wrap kbd {
  display: inline-block;
  min-width: 28px;
  border-radius: 5px;
  border: 1px solid rgba(17, 17, 17, 0.18);
  background: rgba(255, 255, 255, 0.82);
  padding: 1px 5px;
  text-align: center;
  font: 11px/1.35 monospace;
  color: rgba(17, 17, 17, 0.78);
}

.lw-single-exp-media {
  border-radius: 10px;
  border: 1px solid rgba(17, 17, 17, 0.08);
  background: rgba(226, 232, 240, 0.28);
  padding: 10px 10px 8px 10px;
}

.lw-single-exp-params-group .bk-input,
.lw-single-exp-params-group textarea,
.lw-single-exp-params-group .bk-input-group {
  font-size: 12px;
}

.lw-single-exp-param-summary {
  font-size: 12px;
  line-height: 1.5;
  color: rgba(17, 17, 17, 0.72);
  background: rgba(241, 245, 249, 0.95);
  border: 1px solid rgba(17, 17, 17, 0.1);
  border-radius: 8px;
  padding: 8px 10px;
}

.lw-single-exp-param-family {
  border-radius: 12px;
  border: 1px solid rgba(17, 17, 17, 0.08);
  padding: 10px 10px 8px 10px;
  margin: 0 0 10px 0;
}

.lw-single-exp-param-family-title {
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.01em;
  color: rgba(17, 17, 17, 0.8);
  margin: 0;
}

.lw-single-exp-param-family--arena {
  background: rgba(248, 250, 252, 0.96);
}

.lw-single-exp-param-family--food {
  background: rgba(250, 250, 249, 0.96);
}

.lw-single-exp-param-family--scape {
  background: rgba(249, 248, 250, 0.96);
}

.lw-single-exp-param-family--border {
  background: rgba(248, 250, 252, 0.96);
}

.lw-single-exp-param-family--larva {
  background: rgba(248, 251, 248, 0.96);
}

.lw-single-exp-param-family--life {
  background: rgba(250, 249, 247, 0.96);
}

.lw-single-exp-param-family--odor {
  background: rgba(250, 248, 249, 0.96);
}

.lw-single-exp-param-family--runtime {
  background: rgba(248, 250, 252, 0.96);
}

.lw-single-exp-param-family--enrichment {
  background: rgba(248, 249, 251, 0.96);
}

.lw-single-exp-param-family--trials {
  background: rgba(250, 251, 247, 0.96);
}

.lw-single-exp-param-group-card {
  margin: 0 0 10px 0;
  background: rgba(181,194,176,0.14);
}

.lw-single-exp-params-columns {
  gap: 6px;
}

.lw-single-exp-env-preset-box {
  background: rgba(255,255,255,0.96);
  border: 1px solid rgba(17, 17, 17, 0.12);
  border-radius: 8px;
  padding: 8px 10px;
  margin: 0 0 8px 0;
}

.lw-single-exp-env-save-hint {
  font-size: 11px;
  line-height: 1.4;
  color: rgba(17, 17, 17, 0.62);
  margin-top: 6px;
}

.lw-single-exp-env-save-inline {
  font-size: 11px;
  line-height: 1.4;
  margin-top: 6px;
}

.lw-single-exp-template-save-box {
  background: rgba(255,255,255,0.96);
  border: 1px solid rgba(17, 17, 17, 0.12);
  border-radius: 8px;
  padding: 8px 10px;
  margin: 0 0 8px 0;
}
""".strip()

_EDITOR_EXCLUDED_PATHS = {"experiment", "parameter_dict"}
_ARENA_GEOMETRY_OPTIONS = ["circular", "rectangular"]
_SPATIAL_DISTRO_MODE_OPTIONS = ["uniform", "normal", "periphery", "grid"]
_SPATIAL_DISTRO_SHAPE_OPTIONS = ["circle", "rect", "oval", "rectangular"]
_ENRICHMENT_MODE_OPTIONS = ["minimal", "full"]
_ENRICHMENT_PROC_KEY_OPTIONS = ["angular", "spatial", "source", "PI", "wind"]
_ENRICHMENT_ANOT_KEY_OPTIONS = [
    "bout_detection",
    "bout_distribution",
    "interference",
    "source_attraction",
    "patch_residency",
]
_ODORSCAPE_OPTIONS = ["Analytical", "Gaussian", "Diffusion"]
_PREVIEW_STEP_CAP = 300
_PREVIEW_CANVAS_WIDTH = 920
_PREVIEW_CANVAS_HEIGHT = 760
_REGISTRY_ENV_PRESET_PREFIX = "__registry__:"
_WORKSPACE_ENV_PRESET_PREFIX = "__workspace__:"
_REGISTRY_EXPERIMENT_PREFIX = "__registry__:"
_WORKSPACE_EXPERIMENT_PREFIX = "__workspace__:"
_NONE_OPTION_LABEL = "None"
_SIM_OPS_FIELDS = (
    "duration",
    "Nsteps",
    "fr",
    "dt",
    "constant_framerate",
    "Box2D",
    "larva_collisions",
)
_EXPERIMENT_TEMPLATE_SAVE_KEYS = (
    "env_params",
    "larva_groups",
    "trials",
    "enrichment",
    "collections",
)


def _safe_slug(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip()).strip("._-")
    return cleaned or "single_experiment"


def _safe_preset_slug(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip()).strip("._-")
    return cleaned


def _default_run_name(experiment_id: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{_safe_slug(experiment_id)}_{stamp}"


def _default_experiment_template() -> str | None:
    experiment_ids = list(reg.conf.Exp.confIDs)
    if not experiment_ids:
        return None
    return "dish" if "dish" in experiment_ids else experiment_ids[0]


@dataclass(frozen=True)
class WorkspaceExperimentTemplateRecord:
    name: str
    filename: str
    path: Path


class _SingleExperimentSelection(param.Parameterized):
    experiment_template = param.Selector(
        default=(
            f"{_REGISTRY_EXPERIMENT_PREFIX}{_default_experiment_template()}"
            if _default_experiment_template() is not None
            else None
        ),
        objects={},
        doc="Selected experiment template value.",
    )
    environment_preset = param.Selector(
        default="__template__",
        objects={"Template default environment": "__template__"},
        doc=(
            "Environment preset from the template default, workspace JSON presets, "
            "or registry Env configurations."
        ),
    )

    def __init__(
        self,
        *,
        experiment_options: dict[str, str],
        default_experiment: str,
        environment_options: dict[str, str] | None = None,
    ) -> None:
        super().__init__(
            experiment_template=f"{_REGISTRY_EXPERIMENT_PREFIX}{default_experiment}"
        )
        self.param["experiment_template"].objects = experiment_options
        self.param["experiment_template"].label = "Experiment template"
        self.param["environment_preset"].label = "Environment preset"
        if environment_options is not None:
            self.set_environment_options(environment_options)

    def set_environment_options(self, options: dict[str, str]) -> None:
        self.param["environment_preset"].objects = options
        if self.environment_preset not in options.values():
            self.environment_preset = "__template__"

    def set_experiment_options(self, options: dict[str, str]) -> None:
        self.param["experiment_template"].objects = options
        if self.experiment_template not in options.values():
            registry_values = [
                value
                for value in options.values()
                if str(value).startswith(_REGISTRY_EXPERIMENT_PREFIX)
            ]
            self.experiment_template = (
                registry_values[0] if registry_values else next(iter(options.values()))
            )


def _editor_group_title(key: str) -> str:
    if key == "sim_ops":
        return "Simulation Settings"
    return key.replace("_", " ").title()


def _editor_field_title(path: str) -> str:
    return path.replace("_", " ").replace(".", " / ").title()


def _title_case_name(value: str) -> str:
    return value.replace("_", " ").title()


def _family_spec_for_path(path: str) -> tuple[str, str, str]:
    if path.startswith("env_params.arena."):
        return "env_arena", "Arena", "arena"
    if path == "env_params.food_params.source_units":
        return "env_food_source_units", "Source Units", "food"
    if path == "env_params.food_params.source_groups":
        return "env_food_source_groups", "Source Groups", "food"
    if path == "env_params.food_params.food_grid":
        return "env_food_grid", "Food Grid", "food"
    if path.startswith("env_params.food_params.source_units."):
        return "env_food_source_units", "Source Units", "food"
    if path.startswith("env_params.food_params.source_groups."):
        return "env_food_source_groups", "Source Groups", "food"
    if path.startswith("env_params.food_params.food_grid"):
        return "env_food_grid", "Food Grid", "food"
    if path.startswith("env_params.food_params."):
        return "env_food", "Food Params", "food"
    if path == "env_params.odorscape":
        return "env_odorscape", "Odorscape", "scape"
    if path == "env_params.windscape":
        return "env_windscape", "Windscape", "scape"
    if path == "env_params.thermoscape":
        return "env_thermoscape", "Thermoscape", "scape"
    if path.startswith("env_params.odorscape."):
        return "env_odorscape", "Odorscape", "scape"
    if path.startswith("env_params.windscape."):
        return "env_windscape", "Windscape", "scape"
    if path.startswith("env_params.thermoscape."):
        return "env_thermoscape", "Thermoscape", "scape"
    if path == "env_params.border_list":
        return "env_borders", "Borders", "border"
    if path.startswith("env_params.border_list."):
        return "env_borders", "Borders", "border"
    if path.startswith("larva_groups."):
        parts = path.split(".")
        group_id = parts[1]
        family = parts[2] if len(parts) > 2 else "core"
        if family == "distribution":
            return (
                f"larva_{group_id}_distribution",
                f"{group_id} / Distribution",
                "larva",
            )
        if family == "life_history":
            return (
                f"larva_{group_id}_life_history",
                f"{group_id} / Life History",
                "life",
            )
        if family == "odor":
            return (f"larva_{group_id}_odor", f"{group_id} / Odor", "odor")
        return (f"larva_{group_id}_core", f"{group_id} / Identity", "larva")
    if path.startswith("enrichment.pre_kws."):
        return "enrichment_preprocessing", "Enrichment / Preprocessing", "enrichment"
    if path.startswith("enrichment."):
        return "enrichment", "Enrichment", "enrichment"
    if path.startswith("trials."):
        return "trials", "Trials", "trials"
    if path in {
        "dt",
        "duration",
        "fr",
        "Nsteps",
        "Box2D",
        "constant_framerate",
        "larva_collisions",
        "collections",
        "parameter_dict",
    }:
        return "runtime", "Runtime / Outputs", "runtime"
    return (
        path.split(".", 1)[0],
        _editor_group_title(path.split(".", 1)[0]),
        "runtime",
    )


def _display_label_for_path(path: str) -> str:
    if path == "env_params.food_params.source_groups":
        return "Source Groups"
    if path == "env_params.food_params.source_units":
        return "Source Units"
    if path == "env_params.food_params.food_grid":
        return "Food Grid"
    if path == "env_params.odorscape":
        return "Odorscape"
    if path == "env_params.windscape":
        return "Windscape"
    if path == "env_params.thermoscape":
        return "Thermoscape"
    if path == "env_params.border_list":
        return "Borders"
    if path.startswith("env_params.arena."):
        leaf = path.split(".")[-1]
        return {
            "geometry": "Geometry",
            "dims": "Dimensions",
            "torus": "Torus",
        }.get(leaf, _title_case_name(leaf))
    if path.startswith("env_params.border_list."):
        return {
            "color": "Color",
            "group": "Group",
            "unique_id": "Unique ID",
            "vertices": "Vertices",
            "width": "Width",
        }.get(path.split(".")[-1], _title_case_name(path.split(".")[-1]))
    if path.startswith("larva_groups."):
        parts = path.split(".")
        if len(parts) >= 4 and parts[2] == "distribution":
            leaf = parts[-1]
            return {
                "N": "Larvae count",
                "loc": "Start position",
                "scale": "Spread",
                "mode": "Placement mode",
                "shape": "Placement shape",
                "orientation_range": "Initial orientation range",
            }.get(leaf, _title_case_name(leaf))
        if len(parts) >= 4 and parts[2] == "life_history":
            leaf = parts[-1]
            return {
                "age": "Initial age",
                "epochs": "Epochs",
                "reach_pupation": "Reach pupation",
            }.get(leaf, _title_case_name(leaf))
        if len(parts) >= 4 and parts[2] == "odor":
            leaf = parts[-1]
            return {
                "id": "Odor ID",
                "intensity": "Odor intensity",
                "spread": "Odor spread",
            }.get(leaf, _title_case_name(leaf))
        if len(parts) >= 3:
            leaf = parts[-1]
            return {
                "group_id": "Group ID",
                "model": "Model",
                "sample": "Sample",
                "color": "Color",
                "imitation": "Imitation",
            }.get(leaf, _title_case_name(leaf))
    if path.startswith("env_params.food_params.source_units.") or path.startswith(
        "env_params.food_params.source_groups."
    ):
        leaf = path.split(".")[-1]
        return {
            "group_id": "Group ID",
            "unique_id": "Unique ID",
            "odor": "Odor",
            "substrate": "Substrate",
        }.get(leaf, _title_case_name(leaf))
    if path.startswith("env_params.food_params.food_grid."):
        return _title_case_name(path.split(".")[-1])
    if (
        path.startswith("env_params.odorscape.")
        or path.startswith("env_params.windscape.")
        or path.startswith("env_params.thermoscape.")
    ):
        return _title_case_name(path.split(".")[-1])
    if path.startswith("enrichment.pre_kws."):
        return _title_case_name(path.split(".")[-1])
    if path.startswith("enrichment."):
        return _title_case_name(path.split(".")[-1])
    if path.startswith("trials."):
        return _title_case_name(path.split(".")[-1])
    return _editor_field_title(path)


def _sequence_component_labels(
    path: str, value: list[Any] | tuple[Any, ...]
) -> list[str]:
    length = len(value)
    if path.endswith((".dims", ".loc", ".scale")) and length == 2:
        return ["x", "y"]
    if path.endswith((".orientation_range", ".age_range")) and length == 2:
        return ["min", "max"]
    if path == "enrichment.dsp_starts":
        return [f"Start {index}" for index in range(1, length + 1)]
    if path == "enrichment.dsp_stops":
        return [f"Stop {index}" for index in range(1, length + 1)]
    if path == "enrichment.tor_durs":
        return [f"Duration {index}" for index in range(1, length + 1)]
    return [f"Value {index}" for index in range(1, length + 1)]


def _attach_scroll_restore(button: pn.widgets.Button) -> None:
    button._lw_scroll_restore_callback = button.jscallback(
        clicks="""
        const scrollY = window.scrollY || window.pageYOffset || 0;
        [40, 140, 320, 700].forEach((delay) => {
          setTimeout(() => {
            window.scrollTo({top: scrollY, left: 0, behavior: 'auto'});
          }, delay);
        });
        """
    )


def _is_absent_optional_value(value: Any, absent_value: Any) -> bool:
    if absent_value == "empty_dict":
        return value == "empty_dict"
    if isinstance(absent_value, list):
        return value == absent_value
    return value is None


def _summary_label(text: str) -> str:
    return f'<div class="lw-single-exp-param-summary">{text}</div>'


def _display_shortcuts_content() -> pn.pane.HTML:
    return pn.pane.HTML(
        """
        <div class="lw-single-exp-shortcuts-table-wrap">
          <h4>Run display shortcuts</h4>
          <table>
            <tr><td><kbd>Space</kbd></td><td>Pause / resume</td></tr>
            <tr><td><kbd>Wheel</kbd></td><td>Zoom around cursor</td></tr>
            <tr><td><kbd>Arrows</kbd></td><td>Pan display</td></tr>
            <tr><td><kbd>]</kbd> <kbd>/</kbd></td><td>FPS up / down</td></tr>
            <tr><td><kbd>T</kbd></td><td>Clock</td></tr>
            <tr><td><kbd>N</kbd></td><td>Scale</td></tr>
            <tr><td><kbd>S</kbd></td><td>State overlay</td></tr>
            <tr><td><kbd>Tab</kbd></td><td>Agent IDs</td></tr>
            <tr><td><kbd>P</kbd></td><td>Toggle trails</td></tr>
            <tr><td><kbd>+</kbd> <kbd>-</kbd></td><td>Trail duration</td></tr>
            <tr><td><kbd>X</kbd></td><td>Trail color</td></tr>
            <tr><td><kbd>G</kbd></td><td>Black background</td></tr>
            <tr><td><kbd>R</kbd></td><td>Random colors</td></tr>
            <tr><td><kbd>B</kbd></td><td>Color by behavior</td></tr>
            <tr><td><kbd>I</kbd></td><td>Snapshot</td></tr>
            <tr><td><kbd>Y</kbd></td><td>Larva collisions</td></tr>
          </table>
        </div>
        """,
        margin=0,
    )


def _preview_canvas_row(canvas_view: pn.viewable.Viewable) -> pn.Row:
    return pn.Row(
        canvas_view,
        css_classes=["lw-single-exp-preview-canvas-row"],
        sizing_mode="stretch_width",
        margin=0,
    )


def _field_label_html(label: str) -> str:
    escaped_label = html.escape(label)
    return f'<div class="lw-single-exp-param-family-title">{escaped_label}</div>'


def _apply_widget_help(
    widget: pn.viewable.Viewable, label: str, help_text: str | None
) -> pn.viewable.Viewable:
    if hasattr(widget, "name"):
        widget.name = label
    if help_text and hasattr(widget, "description"):
        widget.description = help_text
    return widget


def _optional_family_control(
    path: str, enabled: bool, help_text: str | None
) -> pn.viewable.Viewable:
    widget = pn.widgets.Switch(
        name="",
        value=enabled,
        width=19,
        margin=0,
    )
    if help_text and hasattr(widget, "description"):
        widget.description = help_text
    return widget


def _family_title_row(
    title: str, toggle: pn.viewable.Viewable | None
) -> pn.viewable.Viewable:
    if toggle is None:
        return pn.pane.HTML(
            _field_label_html(title),
            margin=(0, 0, 8, 0),
        )
    grid = pn.GridSpec(
        ncols=12,
        nrows=1,
        sizing_mode="stretch_width",
        margin=(0, 0, 8, 0),
    )
    grid[0, 0:11] = pn.pane.HTML(_field_label_html(title), margin=0)
    grid[0, 11] = pn.Row(toggle, align="end", margin=0)
    return grid


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


def _deep_merge_attrdict(base: Any, override: Any) -> Any:
    if isinstance(base, dict) and isinstance(override, dict):
        merged = util.AttrDict(base).get_copy()
        for key, value in override.items():
            if key in merged:
                merged[key] = _deep_merge_attrdict(merged[key], value)
            else:
                merged[key] = util.AttrDict(value) if isinstance(value, dict) else value
        return util.AttrDict(merged)
    return util.AttrDict(override) if isinstance(override, dict) else override


def _join_help_parts(*parts: str | None) -> str | None:
    cleaned = []
    seen = set()
    for part in parts:
        if not part or not part.strip():
            continue
        normalized = part.strip()
        if normalized in seen:
            continue
        seen.add(normalized)
        cleaned.append(normalized)
    if not cleaned:
        return None
    return "\n\n".join(cleaned)


class _ExperimentPreview:
    _FORWARD_ONLY_MESSAGE = (
        "Preview is forward-only; prepare the preview again to replay from the start."
    )

    def __init__(
        self,
        launcher: sim.ExpRun,
        *,
        size: int = 620,
        preview_step_cap: int = _PREVIEW_STEP_CAP,
        launcher_ready: bool = False,
    ) -> None:
        self.launcher = launcher
        self.size = size
        self.draw_ops = screen.AgentDrawOps(draw_centroid=True, draw_segs=False)
        preview_steps = max(2, min(int(self.launcher.p.steps), preview_step_cap))
        if not launcher_ready:
            self.launcher.sim_setup(steps=preview_steps)
        self.Nfade = max(1, int(self.draw_ops.trail_dt / self.launcher.dt))
        self.env = self.launcher.p.env_params
        xdim, ydim = self.env.arena.dims
        self.image_kws = {
            "title": "Arena preview",
            "xlim": (-xdim / 2, xdim / 2),
            "ylim": (-ydim / 2, ydim / 2),
            "width": self.size,
            "height": int(self.size * ydim / xdim) if xdim else self.size,
            "xlabel": "X (m)",
            "ylabel": "Y (m)",
        }
        self.progress_bar = pn.widgets.Progress(
            name="Simulation timestep",
            bar_color="primary",
            width=int(self.size / 2),
            max=self.launcher.Nsteps - 1,
            value=self.launcher.t,
        )
        self.time_slider = pn.widgets.IntSlider(
            name="Tick",
            width=int(self.size / 2),
            start=0,
            end=preview_steps - 1,
            step=1,
            value=0,
        )
        self.forward_only_note = pn.pane.HTML("", sizing_mode="stretch_width")
        self._syncing_slider = False
        self.tank_plot = self._get_tank_plot()
        self.static_overlay = self._build_static_overlay()

    def _get_tank_plot(self) -> hv.element.Overlay:
        arena = self.env.arena
        if arena.geometry == "circular":
            return hv.Ellipse(0, 0, arena.dims[0]).opts(
                line_width=5,
                bgcolor="lightgrey",
            )
        if arena.geometry == "rectangular":
            return hv.Box(0, 0, spec=arena.dims).opts(
                line_width=5,
                bgcolor="lightgrey",
            )
        raise ValueError(f"Unsupported arena geometry: {arena.geometry}")

    def _source_layers(self) -> list[Any]:
        return [
            hv.Ellipse(source.pos[0], source.pos[1], source.radius * 2).opts(
                line_width=5,
                color=source.color,
                bgcolor=source.color,
            )
            for source in self.launcher.sources
        ]

    def _odor_layers(self) -> list[Any]:
        odor_layers = []
        for source in self.launcher.sources:
            odor = getattr(source, "odor", None)
            spread = getattr(odor, "spread", None)
            odor_id = getattr(odor, "id", None)
            if odor_id in {None, ""} or spread in {None, ""}:
                continue
            try:
                spread_value = float(spread)
            except Exception:
                continue
            if spread_value <= 0:
                continue
            odor_layers.append(
                hv.Ellipse(source.pos[0], source.pos[1], spread_value * 2).opts(
                    line_width=2,
                    color=source.color,
                    alpha=0.22,
                )
            )
        return odor_layers

    def _border_layers(self) -> list[Any]:
        border_layers = []
        for border in getattr(self.launcher, "borders", []):
            line_width = max(1.0, float(getattr(border, "width", 0.001)) * 1800)
            for segment in getattr(border, "border_xy", []):
                border_layers.append(
                    hv.Path(segment).opts(
                        color=getattr(border, "color", "#d94841"),
                        line_width=line_width,
                    )
                )
        return border_layers

    def _build_static_overlay(self) -> hv.Overlay:
        return hv.Overlay(
            [self.tank_plot]
            + self._odor_layers()
            + self._source_layers()
            + self._border_layers()
        )

    def _dynamic_agent_layers(self) -> list[Any]:
        agents = self.launcher.agents
        layers = []
        if self.draw_ops.draw_segs:
            layers.append(
                hv.Overlay(
                    [
                        hv.Polygons([seg.vertices for seg in agent.segs]).opts(
                            color=agent.color
                        )
                        for agent in agents
                    ]
                )
            )
        if self.draw_ops.draw_centroid:
            layers.append(
                hv.Points(agents.get_position()).opts(
                    size=5,
                    color="black",
                )
            )
        if self.draw_ops.draw_head:
            layers.append(
                hv.Points(agents.head.front_end).opts(
                    size=5,
                    color="red",
                )
            )
        if self.draw_ops.draw_midline:
            layers.append(
                hv.Overlay(
                    [
                        hv.Path(agent.midline_xy).opts(color="blue", line_width=2)
                        for agent in agents
                    ]
                )
            )
        if self.draw_ops.visible_trails:
            layers.append(
                hv.Contours([agent.trajectory[-self.Nfade :] for agent in agents]).opts(
                    color="black"
                )
            )
        return layers

    def _draw_overlay(self) -> hv.Overlay:
        agent_layers = self._dynamic_agent_layers()
        overlay = self.static_overlay
        if agent_layers:
            overlay = overlay * hv.Overlay(agent_layers)
        return overlay.opts(
            responsive=False,
            **self.image_kws,
        )

    def _sync_slider_to_current_tick(self) -> None:
        self._syncing_slider = True
        try:
            self.time_slider.value = self.launcher.t
        finally:
            self._syncing_slider = False

    def _image_for_tick(self, i: int) -> hv.Overlay:
        if self._syncing_slider:
            return self._draw_overlay()
        if i < self.launcher.t:
            self.forward_only_note.object = self._FORWARD_ONLY_MESSAGE
            self._sync_slider_to_current_tick()
            self.progress_bar.value = self.launcher.t
            return self._draw_overlay()
        self.forward_only_note.object = ""
        if i == self.launcher.t:
            return self._draw_overlay()
        while i > self.launcher.t:
            self.launcher.sim_step()
        self.progress_bar.value = self.launcher.t
        return self._draw_overlay()

    def view(self) -> pn.viewable.Viewable:
        @pn.depends(i=self.time_slider)
        def _image(i: int) -> hv.Overlay:
            return self._image_for_tick(i)

        preview = hv.DynamicMap(_image)
        return pn.Row(
            preview,
            pn.Column(
                pn.Row(pn.Column("Tick", self.time_slider)),
                pn.Row(pn.Column("Computed timestep", self.progress_bar)),
                self.forward_only_note,
                pn.Param(self.draw_ops),
            ),
            sizing_mode="stretch_width",
        )


class _FrameSimulationPreview:
    def __init__(
        self,
        *,
        canvas: EnvironmentCanvas,
        frames: list[LarvaPreviewFrame],
        dt: float,
    ) -> None:
        if not frames:
            raise ValueError("frames must not be empty")
        self.canvas = canvas
        self.frames = list(frames)
        self.dt = max(0.0, float(dt))
        self.frame_player = pn.widgets.Player(
            name="Frame",
            start=0,
            end=len(self.frames) - 1,
            value=0,
            step=1,
            interval=max(50, min(1000, int(self.dt * 1000))),
            loop_policy="once",
            show_loop_controls=False,
            width=420,
        )
        self.metadata = pn.pane.HTML("", sizing_mode="stretch_width")
        self.frame_player.param.watch(self._on_player_change, "value")
        self._show_frame(0)

    def _set_metadata(self, index: int) -> None:
        frame = self.frames[index]
        end_index = len(self.frames) - 1
        timestamp = frame.tick * self.dt
        end_time = self.frames[-1].tick * self.dt
        self.metadata.object = (
            '<div class="lw-single-exp-preview-meta">'
            f"<strong>Frame:</strong> {index}/{end_index}; "
            f"<strong>Tick:</strong> {frame.tick}; "
            f"<strong>Time:</strong> {timestamp:.1f} s; "
            f"<strong>Displayed range:</strong> 0.0-{end_time:.1f} s."
            "</div>"
        )

    def _show_frame(self, index: int) -> None:
        clamped = max(0, min(index, len(self.frames) - 1))
        if int(self.frame_player.value) != clamped:
            self.frame_player.value = clamped
            return
        self.canvas.set_larva_frame(self.frames[clamped])
        self._set_metadata(clamped)

    def _on_player_change(self, event: Any) -> None:
        self._show_frame(int(event.new))

    def view(self) -> pn.viewable.Viewable:
        return pn.Column(
            _preview_canvas_row(self.canvas.view()),
            pn.Row(pn.Column("Frame", self.frame_player), sizing_mode="stretch_width"),
            self.metadata,
            sizing_mode="stretch_width",
        )


class _SingleExperimentController:
    @staticmethod
    def _new_preview_canvas() -> EnvironmentCanvas:
        try:
            return EnvironmentCanvas(
                editable=False,
                width=_PREVIEW_CANVAS_WIDTH,
                height=_PREVIEW_CANVAS_HEIGHT,
                snap_heads_to_midline=True,
            )
        except TypeError:
            return EnvironmentCanvas(editable=False)

    @staticmethod
    def _initial_preview_content() -> list[pn.viewable.Viewable]:
        canvas = _SingleExperimentController._new_preview_canvas()
        clear = getattr(canvas, "clear", None)
        if callable(clear):
            clear()
        return [
            _preview_canvas_row(canvas.view()),
            pn.pane.HTML(
                (
                    '<div class="lw-single-exp-preview-placeholder">'
                    "Choose an experiment template, optionally apply a workspace environment preset, "
                    "and prepare the configuration preview here."
                    "</div>"
                ),
                margin=(8, 0, 0, 0),
            ),
        ]

    def __init__(self) -> None:
        experiment_ids = list(reg.conf.Exp.confIDs)
        default_experiment = "dish" if "dish" in experiment_ids else experiment_ids[0]
        experiment_options = {
            f"Registry / {name}": f"{_REGISTRY_EXPERIMENT_PREFIX}{name}"
            for name in experiment_ids
        }
        self.selection = _SingleExperimentSelection(
            experiment_options=experiment_options,
            default_experiment=default_experiment,
        )
        self.experiment = pn.widgets.Select.from_param(
            self.selection.param.experiment_template
        )
        self.run_name = pn.widgets.TextInput(
            name="Run name",
            value=_default_run_name(self.selection.experiment_template),
        )
        self.environment_template_default_btn = pn.widgets.Button(
            name="Reset to defaults",
            button_type="default",
        )
        self.environment_preset_controls = PresetControlsController(
            conftype="Env",
            workspace_store=WorkspacePresetStore(
                Path.cwd(),
                directory_key="single-experiment-environments",
            ),
            policy=USER_PRESET_POLICY,
            build_workspace_payload=lambda _name: _json_ready(
                self._environment_payload_from_owner()
            ),
            on_load=self._on_environment_preset_loaded,
            on_save=self._on_environment_preset_saved,
            on_status=self._on_environment_preset_status,
            title="Stored Configurations",
            preset_name_after_refresh=True,
        )
        self.environment_select = self.environment_preset_controls.preset_select
        self.refresh_environments_btn = self.environment_preset_controls.refresh_button
        self.refresh_environments_btn.width = None
        self.refresh_environments_btn.sizing_mode = "stretch_width"
        self.environment_save_name = self.environment_preset_controls.preset_name
        self.environment_save_btn = self.environment_preset_controls.save_button
        self.environment_template_default_btn.width = None
        self.environment_template_default_btn.sizing_mode = "stretch_width"
        self.environment_preset_view = pn.Column(
            self.environment_preset_controls.preset_select,
            pn.Row(
                self.environment_preset_controls.refresh_button,
                self.environment_template_default_btn,
                sizing_mode="stretch_width",
            ),
            self.environment_preset_controls.preset_name,
            pn.Row(
                self.environment_preset_controls.save_button,
                self.environment_preset_controls.load_button,
                self.environment_preset_controls.delete_button,
                sizing_mode="stretch_width",
            ),
            self.environment_preset_controls.confirmation_host,
            self.environment_preset_controls.status,
            sizing_mode="stretch_width",
        )
        self.experiment_template_preset_controls = PresetControlsController(
            conftype="Exp",
            workspace_store=WorkspacePresetStore(
                Path.cwd(),
                directory_key="single-experiment-templates",
            ),
            policy=USER_PRESET_POLICY,
            build_workspace_payload=lambda _name: self._experiment_template_payload(),
            before_save=self._before_save_experiment_template_preset,
            on_load=self._on_experiment_template_preset_loaded,
            on_save=self._on_experiment_template_preset_saved,
            on_status=self._on_experiment_template_preset_status,
            title=None,
            preset_name_after_refresh=True,
        )
        self.experiment_template_select = (
            self.experiment_template_preset_controls.preset_select
        )
        self.refresh_experiment_templates_btn = (
            self.experiment_template_preset_controls.refresh_button
        )
        self.experiment_template_save_name = (
            self.experiment_template_preset_controls.preset_name
        )
        self.experiment_template_save_btn = (
            self.experiment_template_preset_controls.save_button
        )
        self.experiment_template_save_hint = pn.pane.HTML(
            "", css_classes=["lw-single-exp-env-save-hint"], margin=0
        )
        self.experiment_template_save_inline = pn.pane.HTML(
            "", css_classes=["lw-single-exp-env-save-inline"], margin=0
        )
        self.experiment_template_save_box = pn.Column(
            self.experiment_template_preset_controls.view,
            self.experiment_template_save_hint,
            self.experiment_template_save_inline,
            css_classes=["lw-single-exp-template-save-box"],
            sizing_mode="stretch_width",
            margin=(6, 0, 4, 0),
        )
        self.prepare_btn = pn.widgets.Button(
            name="Arena Preview",
            button_type="primary",
        )
        self.simulation_preview_btn = pn.widgets.Button(
            name="Generate simulation preview",
            button_type="primary",
        )
        self.run_btn = pn.widgets.Button(
            name="Run experiment",
            button_type="success",
        )
        self.preview_frames_input = pn.widgets.IntInput(
            name="Preview frames",
            value=_PREVIEW_STEP_CAP,
            start=1,
            end=1000,
            step=50,
        )
        self.save_video = pn.widgets.Checkbox(name="Save video", value=False)
        self.video_filename = pn.widgets.TextInput(
            name="Video filename",
            value=self.run_name.value,
            disabled=True,
        )
        self.video_fps = pn.widgets.IntInput(
            name="Video speed-up",
            value=1,
            step=1,
            start=1,
            end=120,
            disabled=True,
            description=(
                "Controls video playback speed, not the encoder frame rate. "
                "1x keeps the saved video close to simulated real time. 2x plays "
                "twice as fast and makes the video about half as long; 5x plays "
                "five times as fast and makes it about one fifth as long."
            ),
        )
        self.show_display = pn.widgets.Checkbox(name="Show display", value=False)
        self.display_every_n_steps = pn.widgets.IntInput(
            name="Display every N steps",
            value=1,
            step=1,
            start=1,
            end=20,
            disabled=True,
            description=(
                "Live display redraw cadence. 1 updates every simulation step; "
                "higher values redraw less often to reduce display overhead."
            ),
        )
        self.prepare_btn.width = None
        self.prepare_btn.sizing_mode = "stretch_width"
        self.simulation_preview_btn.width = None
        self.simulation_preview_btn.sizing_mode = "stretch_width"
        self.run_btn.width = None
        self.run_btn.sizing_mode = "stretch_width"
        self.preview_frames_input.width = None
        self.preview_frames_input.sizing_mode = "stretch_width"
        self.preview_action_row = pn.Row(
            self.prepare_btn,
            sizing_mode="stretch_width",
            margin=0,
        )
        self.preview_options_row = pn.Row(
            self.preview_frames_input,
            sizing_mode="stretch_width",
            margin=0,
        )
        self.preview_generate_row = pn.Row(
            self.simulation_preview_btn,
            sizing_mode="stretch_width",
            margin=(4, 0, 0, 0),
        )
        self.execution_action_row = pn.Row(
            self.run_btn,
            sizing_mode="stretch_width",
            margin=(6, 0, 0, 0),
        )
        self.summary = pn.pane.HTML(
            "", sizing_mode="stretch_width", margin=(0, 0, 4, 0)
        )
        self.parameters_editor = pn.Column(
            sizing_mode="stretch_width",
            margin=0,
            styles={
                "font-size": "12px",
                "line-height": "1.45",
            },
        )
        self.environment_parameters_editor = pn.Column(
            sizing_mode="stretch_width",
            margin=0,
        )
        self._parameter_groups: dict[str, list[str]] = {}
        self._parameter_widgets: dict[str, tuple[str, Any]] = {}
        self._parameter_widget_specs: dict[
            str, tuple[str, Any, pn.viewable.Viewable]
        ] = {}
        self._typed_experiment_for_larva_groups: Any | None = None
        self._larva_groups_group_view: pn.viewable.Viewable | None = None
        self._typed_experiment_for_enrichment: Any | None = None
        self._enrichment_group_view: pn.viewable.Viewable | None = None
        self._typed_experiment_for_env_params: Any | None = None
        self._env_params_group_view: pn.viewable.Viewable | None = None
        self._typed_experiment_for_sim_ops: Any | None = None
        self._sim_ops_group_view: pn.viewable.Viewable | None = None
        self._typed_experiment_for_collections: Any | None = None
        self._collections_group_view: pn.viewable.Viewable | None = None
        self._typed_experiment_for_trials: Any | None = None
        self._trials_group_view: pn.viewable.Viewable | None = None
        self._editor_context: dict[str, list[str]] = {"odor_ids": []}
        self._parameter_seed_overrides = util.AttrDict()
        self._optional_family_meta: dict[str, dict[str, Any]] = {}
        self.summary.styles = {
            "font-size": "12px",
            "line-height": "1.55",
            "color": "rgba(17, 17, 17, 0.82)",
            "padding": "10px 12px",
            "background": "rgba(181,194,176,0.14)",
            "border-left": "3px solid #b5c2b0",
            "border-radius": "8px",
        }
        self.status = pn.pane.Markdown("", margin=0)
        self.display_shortcuts_link = pn.widgets.Button(
            name="Display Shortcuts",
            button_type="light",
            css_classes=["lw-inline-help-link"],
            margin=(4, 0, 0, 0),
            width_policy="min",
        )
        self.display_shortcuts_close_btn = pn.widgets.Button(
            name="Close",
            button_type="default",
            width=88,
            margin=(8, 0, 0, 0),
        )
        self.display_shortcuts_dialog = pn.Column(
            pn.Column(
                pn.pane.Markdown(
                    (
                        "These shortcuts apply only to the live pygame display opened by "
                        "Run experiment when Show display is enabled. "
                        "They do not control the preview canvas."
                    ),
                    css_classes=["lw-single-exp-shortcuts-note"],
                    margin=0,
                ),
                _display_shortcuts_content(),
                self.display_shortcuts_close_btn,
                css_classes=["lw-single-exp-shortcuts-dialog"],
                sizing_mode="fixed",
            ),
            visible=False,
            css_classes=["lw-single-exp-shortcuts-overlay"],
            sizing_mode="stretch_width",
            margin=0,
        )
        self.run_info = pn.Column(
            self.status,
            self.display_shortcuts_link,
            css_classes=["lw-single-exp-run-info-box"],
            sizing_mode="stretch_width",
            margin=0,
        )
        self.preview = pn.Column(
            *self._initial_preview_content(),
            sizing_mode="stretch_width",
            css_classes=["lw-single-exp-preview-body"],
        )
        self._parameter_group_views: dict[str, pn.viewable.Viewable] = {}
        self._environment_baseline_signature: str | None = None
        self._environment_watcher_handles: list[Any] = []
        self._active_environment_preset_ref: PresetRef | None = None
        self._active_environment_preset_payload: util.AttrDict | None = None
        self._experiment_template_baseline_signature: str | None = None
        self._experiment_template_watcher_handles: list[Any] = []
        self._active_workspace_template_payload: util.AttrDict | None = None
        self._active_workspace_template_filename: str | None = None
        self._experiment_template_workspace_available = False
        self._loading_experiment_template_preset = False
        self._last_valid_experiment_template: str | None = str(
            self.selection.experiment_template
        )
        self._suspend_experiment_change = False
        self._run_controls_locked = False
        self._pending_run_warning_note = ""
        self._pending_template_save_warning_note = ""

        self.selection.param.watch(self._on_experiment_change, "experiment_template")
        self.run_name.param.watch(self._on_run_name_change, "value")
        self.selection.param.watch(
            self._on_parameter_override_change, "environment_preset"
        )
        self.save_video.param.watch(self._on_save_video_change, "value")
        self.show_display.param.watch(self._on_show_display_change, "value")
        self.environment_save_name.param.watch(
            self._on_environment_save_name_change, "value"
        )
        self.experiment_template_save_name.param.watch(
            self._on_experiment_template_save_name_change, "value"
        )
        self.experiment_template_select.param.watch(
            self._on_experiment_template_preset_select_change, "value"
        )
        self.environment_template_default_btn.on_click(
            self._on_use_template_default_environment
        )
        self.refresh_environments_btn.on_click(self._on_refresh_environments)
        self.prepare_btn.on_click(self._on_prepare_preview)
        self.simulation_preview_btn.on_click(self._on_generate_simulation_preview)
        self.run_btn.on_click(self._on_run_experiment)
        self.display_shortcuts_link.on_click(self._on_open_display_shortcuts)
        self.display_shortcuts_close_btn.on_click(self._on_close_display_shortcuts)

        self._on_show_display_change()
        self._refresh_environment_options()
        self._refresh_experiment_template_options()
        self._refresh_summary()
        self._refresh_parameter_editor()
        self.status.object = "Select a template and prepare a single-run preview."

    def _environment_dir(self) -> Path:
        return get_workspace_dir("environments")

    def _experiment_dir(self) -> Path:
        return get_workspace_dir("experiments")

    def _registry_experiment_from_token(self, selected: str) -> str | None:
        if selected.startswith(_REGISTRY_EXPERIMENT_PREFIX):
            return selected[len(_REGISTRY_EXPERIMENT_PREFIX) :]
        ref = self.experiment_template_preset_controls.catalog.resolve(selected)
        if ref is not None and ref.source == PresetSource.REGISTRY:
            return ref.name
        if selected.startswith("registry:"):
            parts = selected.split(":", 2)
            if len(parts) == 3:
                return parts[2]
        return None

    def _workspace_experiment_filename_from_token(self, selected: str) -> str | None:
        if selected.startswith(_WORKSPACE_EXPERIMENT_PREFIX):
            return selected[len(_WORKSPACE_EXPERIMENT_PREFIX) :]
        ref = self.experiment_template_preset_controls.catalog.resolve(selected)
        if ref is not None and ref.source == PresetSource.WORKSPACE:
            return ref.workspace_filename
        if selected.startswith("workspace:"):
            parts = selected.split(":", 2)
            if len(parts) == 3:
                return parts[2]
        return None

    def _list_workspace_experiment_templates(
        self,
    ) -> list[WorkspaceExperimentTemplateRecord]:
        records: list[WorkspaceExperimentTemplateRecord] = []
        for (
            record
        ) in self.experiment_template_preset_controls.workspace_store.list_presets():
            path = record.path
            records.append(
                WorkspaceExperimentTemplateRecord(
                    name=record.name,
                    filename=record.filename,
                    path=path,
                )
            )
        return records

    def _experiment_template_options(self) -> dict[str, str]:
        options: dict[str, str] = {}
        for ref in self.experiment_template_preset_controls.catalog.refs:
            options[ref.display_label] = ref.token
        return options

    def _refresh_experiment_template_options(self) -> None:
        selected = str(self.selection.experiment_template)
        selected_registry = self._registry_experiment_from_token(selected)
        selected_workspace = self._workspace_experiment_filename_from_token(selected)
        workspace_error: WorkspaceError | None = None
        try:
            self.experiment_template_preset_controls.workspace_store = (
                WorkspacePresetStore(
                    self._workspace_experiment_templates_dir(),
                    directory_key="single-experiment-templates",
                )
            )
        except WorkspaceError as exc:
            workspace_error = exc
            self.experiment_template_preset_controls.workspace_store = (
                WorkspacePresetStore(
                    Path.cwd(),
                    directory_key="single-experiment-templates",
                )
            )
            self.experiment_template_preset_controls.save_button.disabled = True
            self.experiment_template_preset_controls.delete_button.disabled = True
            self._experiment_template_workspace_available = False
        else:
            self.experiment_template_preset_controls.delete_button.disabled = False
            self._experiment_template_workspace_available = True

        self.experiment_template_preset_controls.refresh_list()
        run_locked = bool(getattr(self, "_run_controls_locked", False))
        self.experiment_template_preset_controls.load_button.disabled = (
            run_locked
            or not bool(self.experiment_template_preset_controls.catalog.refs)
        )
        if not run_locked and self._experiment_template_workspace_available:
            self.experiment_template_preset_controls.delete_button.disabled = False
        options = self._experiment_template_options()
        self.selection.set_experiment_options(options)
        resolved_selected = None
        if selected in options.values():
            resolved_selected = selected
        elif selected_registry is not None:
            for ref in self.experiment_template_preset_controls.catalog.refs:
                if (
                    ref.source == PresetSource.REGISTRY
                    and ref.name == selected_registry
                ):
                    resolved_selected = ref.token
                    break
        elif selected_workspace is not None:
            for ref in self.experiment_template_preset_controls.catalog.refs:
                if (
                    ref.source == PresetSource.WORKSPACE
                    and ref.workspace_filename == selected_workspace
                ):
                    resolved_selected = ref.token
                    break
        if resolved_selected is not None:
            self.selection.experiment_template = resolved_selected
        if (
            self.selection.experiment_template
            in self.experiment_template_preset_controls.catalog.by_token
        ):
            self.experiment_template_preset_controls.preset_select.value = (
                self.selection.experiment_template
            )
        if workspace_error is not None:
            self.status.object = f"Workspace templates unavailable: {workspace_error}"

    def _selected_experiment(self) -> str:
        selected = str(self.selection.experiment_template)
        registry_experiment = self._registry_experiment_from_token(selected)
        if registry_experiment is not None:
            return registry_experiment
        workspace_filename = self._workspace_experiment_filename_from_token(selected)
        if workspace_filename is not None:
            if self._active_workspace_template_payload:
                experiment = self._active_workspace_template_payload.get("experiment")
                if isinstance(experiment, str) and experiment.strip():
                    return experiment.strip()
            if self._last_valid_experiment_template is not None:
                registry_experiment = self._registry_experiment_from_token(
                    self._last_valid_experiment_template
                )
                if registry_experiment is not None:
                    return registry_experiment
            default_experiment = _default_experiment_template()
            if default_experiment is not None:
                return default_experiment
        # Backward-compatible fallback for legacy unprefixed values.
        return selected

    def _parameters_from_selected_template(self) -> util.AttrDict:
        base_parameters = resolve_base_experiment_parameters(
            self._selected_experiment(),
            self._load_selected_environment(),
        )
        return util.AttrDict(base_parameters.get_copy())

    def _environment_options(self) -> dict[str, str]:
        options = {"Template default environment": "__template__"}
        for ref in self.environment_preset_controls.catalog.refs:
            options[ref.display_label] = ref.token
        return options

    def _sync_environment_selection_options(self) -> None:
        self.selection.param.environment_preset.objects = self._environment_options()

    def _sync_environment_preset_select(self) -> None:
        options = self._environment_options()
        selected = self.selection.environment_preset
        self.environment_preset_controls.preset_select.options = options
        self.environment_preset_controls.preset_select.value = (
            selected if selected in options.values() else "__template__"
        )

    def _load_selected_environment(self) -> util.AttrDict | None:
        selected = self.selection.environment_preset
        if selected in {None, "", "__template__"}:
            return None
        selected_text = str(selected)
        if (
            self._active_environment_preset_ref is not None
            and self._active_environment_preset_payload is not None
            and self._active_environment_preset_ref.token == selected_text
        ):
            return util.AttrDict(self._active_environment_preset_payload.get_copy())
        ref = self.environment_preset_controls.catalog.resolve(selected_text)
        if ref is None:
            if str(selected).startswith(_WORKSPACE_ENV_PRESET_PREFIX):
                filename = str(selected)[len(_WORKSPACE_ENV_PRESET_PREFIX) :]
                payload = self.environment_preset_controls.workspace_store.load(
                    filename
                )
                return util.AttrDict(payload)
            if str(selected).startswith(_REGISTRY_ENV_PRESET_PREFIX):
                registry_id = str(selected)[len(_REGISTRY_ENV_PRESET_PREFIX) :]
                return util.AttrDict(reg.conf.Env.getID(registry_id)).get_copy()
            if selected_text.endswith(".json"):
                payload = self.environment_preset_controls.workspace_store.load(
                    selected_text
                )
                return util.AttrDict(payload)
            return None
        if ref.source == PresetSource.WORKSPACE:
            assert ref.workspace_filename is not None
            payload = self.environment_preset_controls.workspace_store.load(
                ref.workspace_filename
            )
            return util.AttrDict(payload)
        payload = self.environment_preset_controls.registry_store.load(ref.name)
        return util.AttrDict(payload)

    def _selected_environment_label(self) -> str:
        selected = self.selection.environment_preset
        if selected == "__template__":
            return "template default"
        selected_text = str(selected)
        if (
            self._active_environment_preset_ref is not None
            and self._active_environment_preset_ref.token == selected_text
        ):
            return self._active_environment_preset_ref.display_label
        ref = self.environment_preset_controls.catalog.resolve(selected_text)
        if ref is not None:
            return ref.display_label
        if selected is not None and str(selected).startswith(
            _WORKSPACE_ENV_PRESET_PREFIX
        ):
            filename = str(selected)[len(_WORKSPACE_ENV_PRESET_PREFIX) :]
            return f"Workspace / {Path(filename).stem}"
        if selected is not None and str(selected).startswith(
            _REGISTRY_ENV_PRESET_PREFIX
        ):
            registry_id = str(selected)[len(_REGISTRY_ENV_PRESET_PREFIX) :]
            return f"Registry / {registry_id}"
        selected_text = str(selected)
        if selected_text.endswith(".json"):
            return f"Workspace / {Path(selected_text).stem}"
        return selected_text

    def _environment_payload_from_owner(self) -> util.AttrDict:
        if self._typed_experiment_for_env_params is not None:
            nested = util.AttrDict(self._typed_experiment_for_env_params.nestedConf)
            env_payload = nested.get("env_params")
            if env_payload is not None:
                return util.AttrDict(_coerce_xy_sequences(util.AttrDict(env_payload)))
        parameters = self._parameters_from_selected_template()
        return util.AttrDict(_coerce_xy_sequences(util.AttrDict(parameters.env_params)))

    @staticmethod
    def _canonical_env_signature(env_payload: util.AttrDict) -> str:
        return json.dumps(
            _json_ready(env_payload), sort_keys=True, separators=(",", ":")
        )

    def _clear_environment_watchers(self) -> None:
        while self._environment_watcher_handles:
            watcher = self._environment_watcher_handles.pop()
            owner = getattr(watcher, "inst", None)
            if owner is None:
                continue
            try:
                owner.param.unwatch(watcher)
            except Exception:
                continue

    def _clear_pending_environment_overwrite(self) -> None:
        if self.environment_preset_controls._pending_confirmation is not None:
            self.environment_preset_controls.cancel_pending_action()

    def _default_environment_preset_name(self) -> str:
        return "my_environment"

    def _refresh_environment_save_state(self, *, reset_baseline: bool = False) -> None:
        current_payload = self._environment_payload_from_owner()
        current_signature = self._canonical_env_signature(current_payload)
        if reset_baseline or self._environment_baseline_signature is None:
            self._environment_baseline_signature = current_signature
            self.environment_save_name.disabled = True
            self.environment_save_btn.disabled = True
            return

        dirty = current_signature != self._environment_baseline_signature
        safe_name = _safe_preset_slug((self.environment_save_name.value or "").strip())
        self.environment_save_name.disabled = not dirty or self._run_controls_locked
        self.environment_save_btn.disabled = (
            (not dirty) or (not safe_name) or self._run_controls_locked
        )

    def _bind_environment_watchers(self) -> None:
        self._clear_environment_watchers()
        env_view = self._env_params_group_view
        if env_view is None:
            return
        widgets = env_view.select(pn.widgets.Widget)
        for widget in widgets:
            if not hasattr(widget, "param"):
                continue
            if "value" not in widget.param:
                continue
            try:
                watcher = widget.param.watch(
                    self._on_environment_parameter_widget_change, "value"
                )
            except Exception:
                continue
            self._environment_watcher_handles.append(watcher)

    def _save_environment_payload_to_workspace(self, safe_name: str) -> Path:
        payload = _json_ready(self._environment_payload_from_owner())
        return self.environment_preset_controls.workspace_store.save(safe_name, payload)

    def _on_environment_parameter_widget_change(self, *_: object) -> None:
        if not (self.environment_save_name.value or "").strip():
            self.environment_save_name.value = self._default_environment_preset_name()
        self._refresh_environment_save_state(reset_baseline=False)

    def _on_environment_save_name_change(self, *_: object) -> None:
        self._refresh_environment_save_state(reset_baseline=False)

    def _on_save_environment_preset(self, *_: object) -> None:
        self.environment_preset_controls.save_current()

    def _on_confirm_overwrite_environment(self, *_: object) -> None:
        self.environment_preset_controls.confirm_pending_action()

    def _on_cancel_overwrite_environment(self, *_: object) -> None:
        self.environment_preset_controls.cancel_pending_action()

    def _on_environment_preset_status(
        self, message: str, *, tone: str = "neutral"
    ) -> None:
        if hasattr(self, "status") and tone in {"warning", "danger"}:
            self.status.object = message

    def _on_environment_preset_loaded(self, ref: PresetRef, payload: Any) -> None:
        self._active_environment_preset_ref = ref
        self._active_environment_preset_payload = util.AttrDict(payload).get_copy()
        self._sync_environment_selection_options()
        self.selection.environment_preset = ref.token
        self._sync_environment_preset_select()
        self.environment_save_name.value = ref.name
        self._parameter_seed_overrides = util.AttrDict()
        self._refresh_summary()
        self._refresh_parameter_editor()

    def _on_environment_preset_saved(self, ref: PresetRef, payload: Any) -> None:
        self._active_environment_preset_ref = ref
        self._active_environment_preset_payload = util.AttrDict(payload).get_copy()
        self._sync_environment_selection_options()
        self.selection.environment_preset = ref.token
        self._sync_environment_preset_select()
        self.environment_save_name.value = ref.name
        self._parameter_seed_overrides = util.AttrDict()
        self._refresh_summary()
        self._refresh_parameter_editor()

    def _on_use_template_default_environment(self, *_: object) -> None:
        self._active_environment_preset_ref = None
        self._active_environment_preset_payload = None
        self.selection.environment_preset = "__template__"
        self._sync_environment_preset_select()
        self._parameter_seed_overrides = util.AttrDict()
        self._refresh_summary()
        self._refresh_parameter_editor()

    def _workspace_experiment_templates_dir(self) -> Path:
        metadata_dir = get_workspace_dir("metadata")
        return metadata_dir / "experiment_templates"

    def _load_workspace_experiment_template_payload(
        self, filename: str
    ) -> util.AttrDict:
        path = self._workspace_experiment_templates_dir() / filename
        if not path.is_file():
            raise ValueError("Workspace template file not found.")
        try:
            raw_payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError("Workspace template is not valid JSON.") from exc
        return self._normalize_workspace_experiment_template_payload(raw_payload)

    @staticmethod
    def _apply_workspace_template_overrides(
        base_parameters: util.AttrDict, payload: util.AttrDict
    ) -> util.AttrDict:
        merged = util.AttrDict(base_parameters.get_copy())
        for key in _EXPERIMENT_TEMPLATE_SAVE_KEYS:
            if key in payload:
                value = payload[key]
                if key == "collections" and isinstance(value, list):
                    merged[key] = list(value)
                else:
                    merged[key] = _deep_merge_attrdict(merged.get(key), value)
        for key in _SIM_OPS_FIELDS:
            if key in payload:
                merged[key] = _normalize_scalar(payload[key])
        return util.AttrDict(merged)

    @staticmethod
    def _workspace_template_seed_overrides(payload: util.AttrDict) -> util.AttrDict:
        flat = util.AttrDict()
        for key in _EXPERIMENT_TEMPLATE_SAVE_KEYS:
            if key not in payload:
                continue
            if key == "collections":
                flat[key] = (
                    list(payload[key])
                    if isinstance(payload[key], list)
                    else payload[key]
                )
                continue
            value = payload[key]
            if isinstance(value, dict):
                nested = util.AttrDict(value).flatten()
                for nested_key, nested_value in nested.items():
                    flat[f"{key}.{nested_key}"] = _normalize_scalar(nested_value)
            else:
                flat[key] = _normalize_scalar(value)
        for key in _SIM_OPS_FIELDS:
            if key in payload:
                flat[key] = _normalize_scalar(payload[key])
        return flat

    @staticmethod
    def _workspace_override_value(existing: Any, value: Any) -> Any:
        normalized = _normalize_scalar(value)
        if normalized == "empty_dict":
            if isinstance(existing, dict):
                return {}
            if existing is None:
                return None
        if isinstance(existing, tuple) and isinstance(normalized, (list, tuple)):
            return tuple(normalized)
        return normalized

    @staticmethod
    def _set_nested_value(target: Any, path: list[str], value: Any) -> None:
        current = target
        for part in path[:-1]:
            if isinstance(current, dict):
                if part not in current:
                    return
                current = current[part]
                continue
            if not hasattr(current, part):
                return
            current = getattr(current, part)
        leaf = path[-1]
        if isinstance(current, dict):
            existing = current.get(leaf)
            current[leaf] = _SingleExperimentController._workspace_override_value(
                existing, value
            )
            return
        if hasattr(current, leaf):
            existing = getattr(current, leaf)
            setattr(
                current,
                leaf,
                _SingleExperimentController._workspace_override_value(existing, value),
            )

    def _apply_workspace_overrides_to_typed_owner(
        self,
        *,
        exp_owner: Any,
        section: str,
    ) -> None:
        payload = self._active_workspace_template_payload
        if payload is None or section not in payload:
            return
        section_payload = payload.get(section)
        if section == "collections":
            if isinstance(section_payload, list) and hasattr(exp_owner, "collections"):
                exp_owner.collections = list(section_payload)
            return
        if not isinstance(section_payload, dict):
            return
        section_target = getattr(exp_owner, section, None)
        if section_target is None:
            return
        flat = util.AttrDict(section_payload).flatten()
        for path, value in flat.items():
            self._set_nested_value(
                section_target,
                path.split("."),
                _normalize_scalar(value),
            )

    def _apply_workspace_sim_settings_to_typed_owner(self, exp_owner: Any) -> None:
        payload = self._active_workspace_template_payload
        if payload is None:
            return
        for key in _SIM_OPS_FIELDS:
            if key in payload and hasattr(exp_owner, key):
                setattr(exp_owner, key, _normalize_scalar(payload[key]))

    @staticmethod
    def _compatibility_summary(issues: tuple[CompatibilityIssue, ...]) -> str:
        if not issues:
            return ""
        rendered: list[str] = []
        for issue in issues[:3]:
            if issue.path:
                rendered.append(f"{issue.path}: {issue.message}")
            else:
                rendered.append(issue.message)
        if len(issues) > 3:
            rendered.append(f"... and {len(issues) - 3} more")
        return "; ".join(rendered)

    def _compatibility_warning_suffix(self, report: CompatibilityReport) -> str:
        if not report.warnings:
            return ""
        return f" Warning: {self._compatibility_summary(report.warnings)}"

    def _validate_resolved_parameters_for_action(
        self,
        *,
        parameters: util.AttrDict,
        action_label: str,
        show_preview_failure: bool,
    ) -> tuple[bool, str]:
        selected_token = str(self.selection.experiment_template)
        is_registry_selection = (
            self._registry_experiment_from_token(selected_token) is not None
        )
        report = validate_experiment_environment_compatibility(
            parameters,
            allow_registry_legacy=is_registry_selection,
            experiment_id=self._selected_experiment(),
        )
        if report.has_errors:
            message = (
                f"Experiment configuration is incompatible for {action_label}: "
                f"{self._compatibility_summary(report.errors)}"
            )
            self.status.object = message
            if show_preview_failure:
                self.preview[:] = [
                    pn.pane.HTML(
                        (
                            '<div class="lw-single-exp-preview-placeholder">'
                            f"{html.escape(message)}"
                            "</div>"
                        ),
                        margin=0,
                    )
                ]
            return False, ""
        return True, self._compatibility_warning_suffix(report)

    def _before_save_experiment_template_preset(
        self,
        _target_name: str,
        target_source: str,
    ) -> None:
        self._pending_template_save_warning_note = ""
        if target_source != PresetSource.WORKSPACE:
            return
        parameters = self._resolve_experiment_parameters()
        selected_token = str(self.selection.experiment_template)
        is_registry_selection = (
            self._registry_experiment_from_token(selected_token) is not None
        )
        report = validate_experiment_environment_compatibility(
            parameters,
            allow_registry_legacy=is_registry_selection,
            experiment_id=self._selected_experiment(),
        )
        if report.has_errors:
            raise ValueError(
                "Experiment configuration is incompatible for saving this "
                f"template: {self._compatibility_summary(report.errors)}"
            )
        self._pending_template_save_warning_note = self._compatibility_warning_suffix(
            report
        )

    def _experiment_template_payload(self) -> dict[str, Any]:
        parameters = self._resolve_experiment_parameters()
        payload: dict[str, Any] = {"experiment": self._selected_experiment()}
        for key in _EXPERIMENT_TEMPLATE_SAVE_KEYS:
            if key in parameters:
                payload[key] = _json_ready(_normalize_scalar(parameters[key]))
        for key in _SIM_OPS_FIELDS:
            if key in parameters:
                payload[key] = _json_ready(_normalize_scalar(parameters[key]))
        return payload

    @staticmethod
    def _canonical_experiment_template_signature(payload: dict[str, Any]) -> str:
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def _experiment_template_relative_path(filename: str) -> str:
        return f"metadata/experiment_templates/{filename}"

    def _clear_pending_experiment_template_overwrite(self) -> None:
        if self.experiment_template_preset_controls._pending_confirmation is not None:
            self.experiment_template_preset_controls.cancel_pending_action()
        self.experiment_template_save_inline.object = ""

    def _refresh_experiment_template_save_state(
        self, *, reset_baseline: bool = False
    ) -> None:
        try:
            current_payload = self._experiment_template_payload()
        except (WorkspaceError, OSError, json.JSONDecodeError) as exc:
            self.experiment_template_save_name.disabled = True
            self.experiment_template_save_btn.disabled = True
            self.experiment_template_save_hint.object = (
                f"Cannot prepare template payload: {exc}"
            )
            self._clear_pending_experiment_template_overwrite()
            return

        current_signature = self._canonical_experiment_template_signature(
            current_payload
        )
        if reset_baseline or self._experiment_template_baseline_signature is None:
            self._experiment_template_baseline_signature = current_signature
            self.experiment_template_save_name.disabled = True
            self.experiment_template_save_btn.disabled = True
            self.experiment_template_save_hint.object = ""
            self._clear_pending_experiment_template_overwrite()
            return

        dirty = current_signature != self._experiment_template_baseline_signature
        safe_name = _safe_preset_slug(
            (self.experiment_template_save_name.value or "").strip()
        )
        filename = f"{safe_name}.json" if safe_name else ""
        self.experiment_template_save_name.disabled = (
            (not self._experiment_template_workspace_available)
            or (not dirty)
            or self._run_controls_locked
        )
        if dirty:
            if not filename:
                self.experiment_template_save_hint.object = (
                    "Enter a template name to save this experiment template."
                )
            else:
                self.experiment_template_save_hint.object = ""
        else:
            self.experiment_template_save_hint.object = ""
            self._clear_pending_experiment_template_overwrite()
        self.experiment_template_save_btn.disabled = (
            (not self._experiment_template_workspace_available)
            or (not dirty)
            or (not safe_name)
            or self._run_controls_locked
        )

    def _clear_experiment_template_watchers(self) -> None:
        while self._experiment_template_watcher_handles:
            watcher = self._experiment_template_watcher_handles.pop()
            owner = getattr(watcher, "inst", None)
            if owner is None:
                continue
            try:
                owner.param.unwatch(watcher)
            except Exception:
                continue

    def _bind_experiment_template_watchers(self) -> None:
        self._clear_experiment_template_watchers()
        for root in (self.environment_parameters_editor, self.parameters_editor):
            widgets = root.select(pn.widgets.Widget)
            for widget in widgets:
                if not hasattr(widget, "param") or "value" not in widget.param:
                    continue
                try:
                    watcher = widget.param.watch(
                        self._on_experiment_template_parameter_widget_change, "value"
                    )
                except Exception:
                    continue
                self._experiment_template_watcher_handles.append(watcher)

    def _save_experiment_template_payload_to_workspace(self, safe_name: str) -> Path:
        payload = self._experiment_template_payload()
        return self.experiment_template_preset_controls.workspace_store.save(
            safe_name, payload
        )

    def _on_experiment_template_parameter_widget_change(self, *_: object) -> None:
        if self._loading_experiment_template_preset:
            return
        self._refresh_experiment_template_save_state(reset_baseline=False)

    def _on_experiment_template_save_name_change(self, *_: object) -> None:
        if self._loading_experiment_template_preset:
            return
        self._refresh_experiment_template_save_state(reset_baseline=False)

    def _on_experiment_template_preset_select_change(self, *_: object) -> None:
        if self._loading_experiment_template_preset:
            return
        ref = self.experiment_template_preset_controls.catalog.resolve(
            self.experiment_template_select.value
        )
        if ref is None:
            return
        self.experiment_template_save_name.value = ref.name
        self._refresh_experiment_template_save_state(reset_baseline=False)

    def _on_save_experiment_template(self, *_: object) -> None:
        self.experiment_template_preset_controls.save_current()

    def _on_confirm_overwrite_experiment_template(self, *_: object) -> None:
        self.experiment_template_preset_controls.confirm_pending_action()

    def _on_cancel_overwrite_experiment_template(self, *_: object) -> None:
        self.experiment_template_preset_controls.cancel_pending_action()
        self._refresh_experiment_template_save_state(reset_baseline=False)

    def _on_experiment_template_preset_status(
        self, message: str, *, tone: str = "neutral"
    ) -> None:
        if hasattr(self, "status") and tone in {"warning", "danger"}:
            self.status.object = message

    @staticmethod
    def _normalize_workspace_experiment_template_payload(payload: Any) -> util.AttrDict:
        normalized = util.AttrDict(payload)
        experiment = normalized.get("experiment")
        if not isinstance(experiment, str) or not experiment.strip():
            raise ValueError(
                "Workspace experiment template is missing required field: experiment."
            )
        experiment = experiment.strip()
        if experiment not in [str(exp_id) for exp_id in reg.conf.Exp.confIDs]:
            raise ValueError(
                f"Workspace experiment template refers to unknown registry experiment: {experiment}."
            )
        normalized["experiment"] = experiment
        return normalized

    def _on_experiment_template_preset_loaded(
        self, ref: PresetRef, payload: Any
    ) -> None:
        self._loading_experiment_template_preset = True
        self.experiment_template_save_name.disabled = True
        self.experiment_template_save_btn.disabled = True
        try:
            loaded_payload: util.AttrDict | None = None
            if ref.source == PresetSource.WORKSPACE:
                loaded_payload = self._normalize_workspace_experiment_template_payload(
                    payload
                )
                self._active_workspace_template_payload = loaded_payload
                self._active_workspace_template_filename = ref.workspace_filename
                self._parameter_seed_overrides = (
                    self._workspace_template_seed_overrides(loaded_payload)
                )
                selected_experiment = str(loaded_payload["experiment"])
                self.experiment_template_save_name.value = ref.name
            else:
                self._active_workspace_template_payload = None
                self._active_workspace_template_filename = None
                self._parameter_seed_overrides = util.AttrDict()
                selected_experiment = ref.name
                self.experiment_template_save_name.value = ref.name

            self.selection.set_experiment_options(self._experiment_template_options())
            self._suspend_experiment_change = True
            try:
                self.selection.experiment_template = ref.token
            finally:
                self._suspend_experiment_change = False

            self._last_valid_experiment_template = ref.token
            self.run_name.value = _default_run_name(selected_experiment)
            self._active_environment_preset_ref = None
            self._active_environment_preset_payload = None
            self._refresh_environment_options()
            self.selection.environment_preset = "__template__"
            self._refresh_summary()
            self._refresh_parameter_editor()
            self._refresh_experiment_template_save_state(reset_baseline=True)
            self.experiment_template_save_hint.object = ""
            self.experiment_template_save_inline.object = ""
            self.status.object = f'Template "{selected_experiment}" loaded.'
        finally:
            self._loading_experiment_template_preset = False

    def _on_experiment_template_preset_saved(
        self, ref: PresetRef, payload: Any
    ) -> None:
        self.selection.set_experiment_options(self._experiment_template_options())
        self.selection.experiment_template = ref.token
        self.experiment_template_save_name.value = ref.name
        self._refresh_experiment_template_save_state(reset_baseline=True)
        warning_note = self._pending_template_save_warning_note
        self._pending_template_save_warning_note = ""
        self.status.object = f'Experiment template "{ref.name}" saved.' + (
            warning_note if warning_note else ""
        )

    @staticmethod
    def _apply_environment_payload(
        env_params: util.AttrDict,
        environment_payload: util.AttrDict,
    ) -> util.AttrDict:
        return apply_environment_payload(env_params, environment_payload)

    def _build_parameters(self) -> util.AttrDict:
        parameters = self._parameters_from_selected_template()
        flat = self._merge_seed_overrides(parameters.flatten())
        for path, (kind, widget) in self._parameter_widgets.items():
            if kind == "toggle_factory":
                continue
            flat[path] = self._parse_widget_value(kind, widget)
        flat = self._apply_optional_family_states(flat)
        resolved = util.AttrDict(flat.unflatten())
        if self._typed_experiment_for_larva_groups is not None:
            nested_conf = util.AttrDict(
                self._typed_experiment_for_larva_groups.nestedConf
            )
            larva_groups = nested_conf.get("larva_groups")
            if larva_groups is not None:
                resolved["larva_groups"] = util.AttrDict(larva_groups)
        if self._typed_experiment_for_enrichment is not None:
            nested_conf = util.AttrDict(
                self._typed_experiment_for_enrichment.nestedConf
            )
            enrichment = nested_conf.get("enrichment")
            if enrichment is not None:
                resolved["enrichment"] = util.AttrDict(enrichment)
        if self._typed_experiment_for_env_params is not None:
            nested_conf = util.AttrDict(
                self._typed_experiment_for_env_params.nestedConf
            )
            env_params = nested_conf.get("env_params")
            if env_params is not None:
                resolved["env_params"] = util.AttrDict(env_params)
        if self._typed_experiment_for_sim_ops is not None:
            nested_conf = util.AttrDict(self._typed_experiment_for_sim_ops.nestedConf)
            for key in _SIM_OPS_FIELDS:
                if key in nested_conf:
                    resolved[key] = _normalize_scalar(nested_conf[key])
        if self._typed_experiment_for_collections is not None:
            nested_conf = util.AttrDict(
                self._typed_experiment_for_collections.nestedConf
            )
            collections = nested_conf.get("collections")
            if collections is not None:
                resolved["collections"] = list(collections)
        if self._typed_experiment_for_trials is not None:
            nested_conf = util.AttrDict(self._typed_experiment_for_trials.nestedConf)
            trials = nested_conf.get("trials")
            if trials is not None:
                resolved["trials"] = util.AttrDict(trials)
        return util.AttrDict(_coerce_xy_sequences(resolved))

    def _resolve_experiment_parameters(self) -> util.AttrDict:
        return self._build_parameters()

    def _refresh_environment_options(self) -> None:
        try:
            self.environment_preset_controls.workspace_store = WorkspacePresetStore(
                self._environment_dir(),
                directory_key="single-experiment-environments",
            )
        except WorkspaceError as exc:
            self.environment_preset_controls.load_button.disabled = True
            self.environment_preset_controls.save_button.disabled = True
            self.environment_preset_controls.delete_button.disabled = True
            self.selection.set_environment_options(
                {"Template default environment": "__template__"}
            )
            self.selection.environment_preset = "__template__"
            self._sync_environment_preset_select()
            self._active_environment_preset_ref = None
            self._active_environment_preset_payload = None
            self.status.object = f"Cannot load workspace environment presets: {exc}"
            return
        self.environment_preset_controls.load_button.disabled = False
        self.environment_preset_controls.delete_button.disabled = False
        self.environment_preset_controls.refresh_list()
        options = self._environment_options()
        selected = self.selection.environment_preset
        self.selection.set_environment_options(options)
        self.selection.environment_preset = (
            selected if selected in options.values() else "__template__"
        )
        self._sync_environment_preset_select()

    def _refresh_summary(self) -> None:
        experiment = self._selected_experiment()
        parameters = self._parameters_from_selected_template()
        larva_groups = list(parameters.get("larva_groups", {}).keys())
        env = util.AttrDict(parameters.env_params)
        epochs = parameters.get("trials", {}).get("epochs", {})
        dims = getattr(env.arena, "dims", ("?", "?"))
        geometry = str(getattr(env.arena, "geometry", ""))
        if isinstance(dims, (list, tuple)) and len(dims) >= 2:
            dims_text = f"{float(dims[0]):.3f} x {float(dims[1]):.3f} m"
            if geometry == "circular":
                radius_text = f"{(float(dims[0]) / 2.0):.3f} m"
            else:
                radius_text = None
        else:
            dims_text = str(dims)
            radius_text = None
        if radius_text is not None:
            arena_size_line = (
                f"<strong>Arena radius:</strong> {radius_text} "
                f"(<strong>diameter:</strong> {float(dims[0]):.3f} m)<br>"
            )
        else:
            arena_size_line = f"<strong>Arena dims:</strong> {dims_text}<br>"
        selected_env = self._selected_environment_label()
        self.summary.object = (
            '<div class="lw-single-exp-summary">'
            f"<strong>Template:</strong> <code>{experiment}</code><br>"
            f"<strong>Environment source:</strong> {selected_env}<br>"
            f"<strong>Default duration:</strong> {float(parameters.get('duration', 0.0)):.2f} min<br>"
            f"<strong>Arena geometry:</strong> {env.arena.geometry}<br>"
            f"{arena_size_line}"
            f"<strong>Larva groups:</strong> {', '.join(larva_groups) if larva_groups else 'None'}<br>"
            f"<strong>Epochs:</strong> {len(epochs)}<br>"
            "<strong>Parameter editing:</strong> all resolved experiment parameters are editable below; preview uses this resolved configuration."
            "</div>"
        )

    def _editable_flat_parameters(self) -> util.AttrDict:
        parameters = self._parameters_from_selected_template()
        flat = parameters.flatten()
        flat = self._merge_seed_overrides(flat)
        flat = self._augment_optional_family_entries(flat)
        filtered = util.AttrDict()
        for path, value in flat.items():
            if path in _EDITOR_EXCLUDED_PATHS:
                continue
            filtered[path] = _normalize_scalar(value)
        return filtered

    def _merge_seed_overrides(self, flat: util.AttrDict) -> util.AttrDict:
        merged = util.AttrDict(flat)
        seed_keys = list(self._parameter_seed_overrides.keys())
        for seed_key in seed_keys:
            parts = seed_key.split(".")
            for idx in range(1, len(parts)):
                prefix = ".".join(parts[:idx])
                merged.pop(prefix, None)
        merged.update(self._parameter_seed_overrides)
        return merged

    def _optional_family_specs(
        self, flat: util.AttrDict | None = None
    ) -> dict[str, dict[str, Any]]:
        from larvaworld.lib.reg.generators import gen

        specs = {
            "env_params.odorscape": {
                "absent": None,
                "payload": gen.GaussianValueLayer().nestedConf,
            },
            "env_params.windscape": {
                "absent": None,
                "payload": gen.WindScape().nestedConf,
            },
            "env_params.thermoscape": {
                "absent": None,
                "payload": gen.ThermoScape().nestedConf,
            },
            "env_params.food_params.food_grid": {
                "absent": None,
                "payload": gen.FoodGrid().nestedConf,
            },
            "env_params.food_params.source_groups": {
                "absent": "empty_dict",
                "payload": gen.FoodGroup().entry("SourceGroup"),
            },
            "env_params.food_params.source_units": {
                "absent": "empty_dict",
                "payload": gen.Food().entry("Source"),
            },
            "env_params.border_list": {
                "absent": "empty_dict",
                "payload": gen.Border(
                    vertices=[(-0.01, 0.0), (0.01, 0.0)],
                    width=0.001,
                ).entry("Border"),
            },
        }
        if flat is not None:
            windscape_enabled = flat.get("env_params.windscape", None) is not None
            if windscape_enabled or any(
                key.startswith("env_params.windscape.")
                for key in flat.keys()
                if key != "env_params.windscape.puffs"
            ):
                specs["env_params.windscape.puffs"] = {
                    "absent": "empty_dict",
                    "payload": {"Puff": gen.AirPuff().nestedConf},
                }
            epoch_payload = [gen.Epoch().nestedConf]
            if "trials.epochs" in flat:
                specs["trials.epochs"] = {
                    "absent": [],
                    "payload": epoch_payload,
                }
            for path in flat.keys():
                if path.startswith("larva_groups.") and path.endswith(
                    ".life_history.epochs"
                ):
                    specs[path] = {
                        "absent": [],
                        "payload": epoch_payload,
                    }
        return specs

    def _augment_optional_family_entries(self, flat: util.AttrDict) -> util.AttrDict:
        augmented = util.AttrDict(flat)
        self._optional_family_meta = {}
        for root, spec in self._optional_family_specs(augmented).items():
            descendants = {
                key: value
                for key, value in augmented.items()
                if key.startswith(f"{root}.")
            }
            root_value = augmented.get(root, spec["absent"])
            enabled = bool(descendants) or not _is_absent_optional_value(
                root_value, spec["absent"]
            )
            self._optional_family_meta[root] = {
                "absent": spec["absent"],
                "payload": spec["payload"],
                "enabled": enabled,
            }
            augmented[root] = root_value
            if not descendants:
                for key, value in self._flatten_seed_payload(
                    root, spec["payload"]
                ).items():
                    if key not in augmented:
                        augmented[key] = value
        return augmented

    def _optional_root_for_path(self, path: str) -> str | None:
        for root in sorted(self._optional_family_meta.keys(), key=len, reverse=True):
            if path == root:
                return root
            if path.startswith(f"{root}."):
                return root
        return None

    @staticmethod
    def _set_control_disabled(kind: str, control: Any, disabled: bool) -> None:
        if kind in {"readonly", "factory", "toggle_factory"}:
            return
        if kind == "sequence":
            for widget in control["widgets"]:
                widget.disabled = disabled
            return
        if kind == "optional_float":
            control["enabled"].disabled = disabled
            control["widget"].disabled = disabled or not bool(control["enabled"].value)
            return
        if hasattr(control, "disabled"):
            control.disabled = disabled

    @staticmethod
    def _optional_toggle_enabled_value(widget: Any) -> bool:
        return bool(getattr(widget, "value", None))

    def _apply_optional_family_disabled_state(self, root: str, enabled: bool) -> None:
        for path, (kind, control) in self._parameter_widgets.items():
            if path == root or not path.startswith(f"{root}."):
                continue
            self._set_control_disabled(kind, control, not enabled)

    def _wire_optional_family_toggles(self) -> None:
        for root in self._optional_family_meta:
            kind, control = self._parameter_widgets.get(root, (None, None))
            if kind != "toggle_factory":
                continue
            enabled_widget = control["enabled"]
            self._apply_optional_family_disabled_state(
                root, self._optional_toggle_enabled_value(enabled_widget)
            )

            def _sync(event: Any, root: str = root) -> None:
                self._apply_optional_family_disabled_state(root, bool(event.new))

            enabled_widget.param.watch(_sync, "value")

    def _apply_optional_family_states(self, flat: util.AttrDict) -> util.AttrDict:
        updated = util.AttrDict(flat)
        for root, meta in self._optional_family_meta.items():
            kind, control = self._parameter_widgets.get(root, (None, None))
            if kind != "toggle_factory":
                continue
            enabled = self._optional_toggle_enabled_value(control["enabled"])
            descendant_keys = [
                key for key in list(updated.keys()) if key.startswith(f"{root}.")
            ]
            if enabled:
                if descendant_keys:
                    updated.pop(root, None)
                elif _is_absent_optional_value(
                    updated.get(root, meta["absent"]), meta["absent"]
                ):
                    updated[root] = _normalize_scalar(meta["payload"])
                continue
            for key in descendant_keys:
                updated.pop(key, None)
            updated[root] = meta["absent"]
        return updated

    @staticmethod
    def _runtime_note_for_path(path: str) -> str | None:
        if path == "duration":
            return "Runtime: total simulated duration in minutes."
        if path == "dt":
            return "Runtime: simulation timestep in seconds."
        if path.startswith("env_params."):
            return "Runtime: consumed during ExpRun.setup() when build_env(self.p.env_params) constructs the environment."
        if path.startswith("larva_groups."):
            return "Runtime: consumed during ExpRun.build_agents() to generate and place larva groups."
        if path == "collections" or path.startswith("collections."):
            return "Runtime: consumed during ExpRun.setup() when set_collectors(self.p.collections) defines recorded outputs."
        if path.startswith("trials.epochs"):
            return "Runtime: converted in ExpRun.setup() from age ranges to start/stop simulation steps."
        if path == "enrichment" or path.startswith("enrichment."):
            return "Runtime: applied after simulate() completes, during post-simulation dataset enrichment."
        if path == "larva_collisions":
            return "Runtime: when disabled, ExpRun.setup() runs overlap elimination before the simulation starts."
        return None

    @staticmethod
    def _resolve_doc_from_class(cls: type[Any], parts: list[str]) -> str | None:
        if not hasattr(cls, "param") or not parts:
            return None
        objects = cls.param.objects(instance=False)
        name = parts[0]
        if name not in objects:
            return None
        p = objects[name]
        current_doc = getattr(p, "doc", None)
        if len(parts) == 1:
            return current_doc
        rest = parts[1:]
        if isinstance(p, ClassDict):
            if not rest:
                return current_doc
            item_cls = p.item_type
            if item_cls is None or len(rest) < 2:
                return current_doc
            nested_doc = _SingleExperimentController._resolve_doc_from_class(
                item_cls, rest[1:]
            )
            return nested_doc or current_doc
        if isinstance(p, ClassAttr):
            nested_cls = p.class_[0] if isinstance(p.class_, tuple) else p.class_
            nested_doc = _SingleExperimentController._resolve_doc_from_class(
                nested_cls, rest
            )
            return nested_doc or current_doc
        if name == "trials" and rest and rest[0] == "epochs":
            from larvaworld.lib.reg.generators import gen

            epoch_doc = getattr(
                gen.Epoch.param.objects(instance=False)["age_range"], "doc", None
            )
            return _join_help_parts(current_doc, epoch_doc)
        return current_doc

    @staticmethod
    def _param_doc_for_path(path: str) -> str | None:
        from larvaworld.lib.param.enrichment import EnrichConf
        from larvaworld.lib.reg.generators import EnvConf, ExpConf
        from larvaworld.lib.reg.larvagroup import LarvaGroup

        if path.startswith("env_params."):
            return _SingleExperimentController._resolve_doc_from_class(
                EnvConf, path.split(".")[1:]
            )
        if path.startswith("larva_groups."):
            return _SingleExperimentController._resolve_doc_from_class(
                LarvaGroup, path.split(".")[2:]
            )
        if path.startswith("enrichment."):
            return _SingleExperimentController._resolve_doc_from_class(
                EnrichConf, path.split(".")[1:]
            )
        return _SingleExperimentController._resolve_doc_from_class(
            ExpConf, path.split(".")
        )

    @staticmethod
    def _help_text_for_path(path: str) -> str | None:
        return _join_help_parts(
            _SingleExperimentController._param_doc_for_path(path),
            _SingleExperimentController._runtime_note_for_path(path),
        )

    @staticmethod
    def _factory_spec_for_path(path: str, value: Any) -> tuple[str, Any] | None:
        from larvaworld.lib.reg.generators import gen

        if path == "env_params.odorscape" and value is None:
            return "Add odorscape", gen.GaussianValueLayer().nestedConf
        if path == "env_params.windscape" and value is None:
            return "Add windscape", gen.WindScape().nestedConf
        if path == "env_params.thermoscape" and value is None:
            return "Add thermoscape", gen.ThermoScape().nestedConf
        if path == "env_params.food_params.food_grid" and value is None:
            return "Add food grid", gen.FoodGrid().nestedConf
        if path == "env_params.food_params.source_groups" and value == "empty_dict":
            return "Add source group", gen.FoodGroup().entry("SourceGroup")
        if path == "env_params.food_params.source_units" and value == "empty_dict":
            return "Add source unit", gen.Food().entry("Source")
        if path == "env_params.border_list" and value == "empty_dict":
            return (
                "Add border",
                gen.Border(vertices=[(-0.01, 0.0), (0.01, 0.0)], width=0.001).entry(
                    "Border"
                ),
            )
        if path == "env_params.windscape.puffs" and value == "empty_dict":
            return "Add air puff", {"Puff": gen.AirPuff().nestedConf}
        return None

    @staticmethod
    def _flatten_seed_payload(path: str, payload: Any) -> util.AttrDict:
        payload = _normalize_scalar(payload)
        if isinstance(payload, dict) and payload:
            items = util.AttrDict()
            for key, value in payload.items():
                nested = _SingleExperimentController._flatten_seed_payload(
                    f"{path}.{key}", value
                )
                items.update(nested)
            return items
        return util.AttrDict({path: payload})

    def _activate_factory_value(self, path: str, payload: Any, message: str) -> None:
        keys_to_remove = [
            key
            for key in self._parameter_seed_overrides
            if key == path or key.startswith(f"{path}.")
        ]
        for key in keys_to_remove:
            self._parameter_seed_overrides.pop(key, None)
        self._parameter_seed_overrides.update(self._flatten_seed_payload(path, payload))
        self._refresh_parameter_editor()
        self.status.object = message

    def _options_for_path(self, path: str, value: Any) -> tuple[str, list[Any]] | None:
        if path == "collections":
            return "multi", list(reg.parDB.output_keys)
        if path == "enrichment.proc_keys":
            return "multi", list(_ENRICHMENT_PROC_KEY_OPTIONS)
        if path == "enrichment.anot_keys":
            return "multi", list(_ENRICHMENT_ANOT_KEY_OPTIONS)
        if path.endswith("arena.geometry"):
            return "single", list(_ARENA_GEOMETRY_OPTIONS)
        if path.endswith("distribution.mode"):
            return "single", list(_SPATIAL_DISTRO_MODE_OPTIONS)
        if path.endswith("distribution.shape"):
            return "single", list(_SPATIAL_DISTRO_SHAPE_OPTIONS)
        if path.endswith(".model"):
            return "single", list(reg.conf.Model.confIDs)
        if path.endswith(".sample"):
            return "single_optional", [None] + list(reg.conf.Ref.confIDs)
        if path.endswith(".odor.id"):
            return "single_optional", [None] + list(
                self._editor_context.get("odor_ids", [])
            )
        if path == "enrichment.mode":
            return "single", list(_ENRICHMENT_MODE_OPTIONS)
        if path == "env_params.odorscape":
            return "single_optional", [None] + list(_ODORSCAPE_OPTIONS)
        if path.endswith(".odorscape.odorscape"):
            return "single", list(_ODORSCAPE_OPTIONS)
        return None

    @staticmethod
    def _optional_scalar_kind(path: str, value: Any) -> str | None:
        if path.endswith((".odor.intensity", ".odor.spread")) and (
            value is None or isinstance(value, (int, float))
        ):
            return "float"
        return None

    @staticmethod
    def _summarize_value(path: str, value: Any) -> str:
        if value == "empty_dict":
            return "Empty collection. This family should be edited via a dedicated structured editor, not free text."
        if value is None:
            return "Not configured."
        if isinstance(value, list):
            if path.endswith(".epochs"):
                return f"{len(value)} epoch item(s). This should be edited with a dedicated epoch editor."
            if path.endswith(".vertices"):
                return f"{len(value)} vertex points. This should be edited with the canvas/drawing tools."
            return f"List with {len(value)} item(s)."
        if isinstance(value, dict):
            return f"Object with {len(value)} field(s)."
        return str(value)

    def _widget_for_value(
        self, path: str, value: Any
    ) -> tuple[str, Any, pn.viewable.Viewable]:
        label = _display_label_for_path(path)
        help_text = self._help_text_for_path(path)
        optional_meta = self._optional_family_meta.get(path)
        if optional_meta is not None:
            toggle = _optional_family_control(
                path, bool(optional_meta["enabled"]), help_text
            )
            view = pn.Row(
                pn.Column(
                    pn.pane.HTML(_field_label_html(label), margin=0),
                    sizing_mode="stretch_width",
                    width_policy="max",
                    margin=0,
                ),
                pn.Spacer(width=16),
                pn.Row(
                    toggle,
                    align="end",
                    margin=0,
                ),
                sizing_mode="stretch_width",
                width_policy="max",
                margin=(0, 0, 6, 0),
            )
            return "toggle_factory", {"enabled": toggle}, view
        factory = self._factory_spec_for_path(path, value)
        if factory is not None:
            button_label, payload = factory
            button = pn.widgets.Button(
                name=button_label,
                button_type="primary",
                width=None,
                sizing_mode="stretch_width",
            )
            if help_text and hasattr(button, "description"):
                button.description = help_text
            _attach_scroll_restore(button)

            def _create_value(*_: object) -> None:
                self._activate_factory_value(
                    path,
                    payload,
                    f'Initialized "{label}" with a default configuration.',
                )

            button.on_click(_create_value)
            summary = pn.Column(
                pn.pane.HTML(_field_label_html(label), margin=0),
                pn.pane.HTML(
                    _summary_label(self._summarize_value(path, _json_ready(value))),
                    margin=0,
                ),
                button,
                sizing_mode="stretch_width",
                margin=(0, 0, 8, 0),
            )
            return "factory", {"value": value, "button": button}, summary
        options = self._options_for_path(path, value)
        if options is not None:
            mode, option_values = options
            if mode == "multi":
                widget = _apply_widget_help(
                    pn.widgets.MultiChoice(
                        name="",
                        value=list(value or []),
                        options=option_values,
                        sizing_mode="stretch_width",
                    ),
                    label,
                    help_text,
                )
                return "multichoice", widget, widget
            normalized_options = [
                _NONE_OPTION_LABEL if item is None else item for item in option_values
            ]
            selected_value = _NONE_OPTION_LABEL if value is None else value
            widget = _apply_widget_help(
                pn.widgets.Select(
                    name="",
                    value=selected_value
                    if selected_value in normalized_options
                    else normalized_options[0],
                    options=normalized_options,
                    sizing_mode="stretch_width",
                ),
                label,
                help_text,
            )
            return "option", widget, widget
        optional_scalar_kind = _SingleExperimentController._optional_scalar_kind(
            path, value
        )
        if optional_scalar_kind == "float":
            enabled = value is not None
            scalar = _apply_widget_help(
                pn.widgets.FloatInput(
                    name="",
                    value=float(value) if value is not None else 0.0,
                    step=0.1,
                    disabled=not enabled,
                    width=None,
                    sizing_mode="stretch_width",
                ),
                label,
                help_text,
            )
            enabled_toggle = pn.widgets.Checkbox(
                name="Enabled",
                value=enabled,
                margin=(5, 0, 0, 8),
            )

            def _sync_optional_float(event: Any) -> None:
                scalar.disabled = not bool(event.new)

            enabled_toggle.param.watch(_sync_optional_float, "value")
            view = pn.Column(
                pn.Row(scalar, enabled_toggle, sizing_mode="stretch_width", margin=0),
                sizing_mode="stretch_width",
                margin=(0, 0, 8, 0),
            )
            control = {"widget": scalar, "enabled": enabled_toggle}
            return "optional_float", control, view
        if isinstance(value, bool):
            widget = _apply_widget_help(
                pn.widgets.Checkbox(name="", value=value),
                label,
                help_text,
            )
            return "bool", widget, widget
        if isinstance(value, int):
            widget = _apply_widget_help(
                pn.widgets.IntInput(
                    name="",
                    value=value,
                    step=1,
                    sizing_mode="stretch_width",
                ),
                label,
                help_text,
            )
            return "int", widget, widget
        if isinstance(value, float):
            widget = _apply_widget_help(
                pn.widgets.FloatInput(
                    name="",
                    value=value,
                    step=0.1,
                    sizing_mode="stretch_width",
                ),
                label,
                help_text,
            )
            return "float", widget, widget
        if (
            isinstance(value, (list, tuple))
            and 1 <= len(value) <= 4
            and all(isinstance(item, (int, float)) for item in value)
        ):
            scalar_kind = (
                "int" if all(isinstance(item, int) for item in value) else "float"
            )
            component_labels = _sequence_component_labels(path, value)
            subwidgets: list[pn.viewable.Viewable] = []
            for index, item in enumerate(value, start=1):
                component_label = component_labels[index - 1]
                if scalar_kind == "int":
                    subwidgets.append(
                        _apply_widget_help(
                            pn.widgets.IntInput(
                                name=component_label,
                                value=int(item),
                                step=1,
                                width=None,
                                sizing_mode="stretch_width",
                            ),
                            component_label,
                            help_text,
                        )
                    )
                else:
                    subwidgets.append(
                        _apply_widget_help(
                            pn.widgets.FloatInput(
                                name=component_label,
                                value=float(item),
                                step=0.1,
                                width=None,
                                sizing_mode="stretch_width",
                            ),
                            component_label,
                            help_text,
                        )
                    )
            view = pn.Column(
                pn.pane.HTML(_field_label_html(label), margin=0),
                pn.Row(*subwidgets, sizing_mode="stretch_width", margin=0),
                sizing_mode="stretch_width",
                margin=(0, 0, 8, 0),
            )
            control = {
                "widgets": subwidgets,
                "container": tuple if isinstance(value, tuple) else list,
                "scalar_kind": scalar_kind,
            }
            return "sequence", control, view
        if isinstance(value, str):
            if path.endswith(".color"):
                widget = _apply_widget_help(
                    pn.widgets.ColorPicker(
                        name="",
                        value=value,
                        sizing_mode="stretch_width",
                    ),
                    label,
                    help_text,
                )
                return "color", widget, widget
            summary = pn.Column(
                pn.pane.HTML(_field_label_html(label), margin=0),
                pn.pane.HTML(_summary_label(str(value)), margin=0),
                sizing_mode="stretch_width",
                margin=(0, 0, 8, 0),
            )
            return "readonly", {"value": value}, summary
        summary = pn.Column(
            pn.pane.HTML(_field_label_html(label), margin=0),
            pn.pane.HTML(
                _summary_label(self._summarize_value(path, _json_ready(value))),
                margin=0,
            ),
            sizing_mode="stretch_width",
            margin=(0, 0, 8, 0),
        )
        return "readonly", {"value": _json_ready(value)}, summary

    @staticmethod
    def _parse_widget_value(kind: str, control: Any) -> Any:
        if kind == "toggle_factory":
            return bool(control["enabled"].value)
        if kind == "factory":
            return control["value"]
        if kind == "readonly":
            return control["value"]
        if kind == "sequence":
            scalar_kind = control["scalar_kind"]
            parsed = []
            for widget in control["widgets"]:
                value = getattr(widget, "value")
                parsed.append(int(value) if scalar_kind == "int" else float(value))
            return control["container"](parsed)
        if kind == "optional_float":
            return (
                None if not control["enabled"].value else float(control["widget"].value)
            )
        value = getattr(control, "value")
        if kind == "bool":
            return bool(value)
        if kind == "int":
            return int(value)
        if kind == "float":
            return float(value)
        if kind == "color":
            return str(value)
        if kind == "option":
            return None if value == _NONE_OPTION_LABEL else value
        if kind == "multichoice":
            return list(value)
        raise ValueError(f"Unsupported widget kind: {kind}")

    def _render_single_parameter_group(
        self, group_key: str
    ) -> pn.viewable.Viewable | None:
        if group_key == "sim_ops":
            return self._sim_ops_group_view
        if group_key == "collections":
            return self._collections_group_view
        if group_key == "trials":
            return self._trials_group_view
        if group_key == "env_params":
            if self._env_params_group_view is None:
                return None
            return self._env_params_group_view
        if group_key == "enrichment":
            return self._enrichment_group_view
        if group_key == "larva_groups":
            return self._larva_groups_group_view
        paths = self._parameter_groups.get(group_key, [])
        if not paths:
            return None
        families: dict[str, dict[str, Any]] = {}
        ordered_family_ids: list[str] = []
        for path in paths:
            family_id, family_title, color_key = _family_spec_for_path(path)
            if family_id not in families:
                families[family_id] = {
                    "title": family_title,
                    "color_key": color_key,
                    "views": [],
                    "title_toggle": None,
                }
                ordered_family_ids.append(family_id)
            kind, control, view = self._parameter_widget_specs[path]
            if (
                kind == "toggle_factory"
                and _display_label_for_path(path) == family_title
            ):
                families[family_id]["title_toggle"] = control["enabled"]
                continue
            families[family_id]["views"].append(view)

        family_views = [
            pn.Column(
                _family_title_row(
                    families[family_id]["title"],
                    families[family_id]["title_toggle"],
                ),
                *families[family_id]["views"],
                css_classes=[
                    "lw-single-exp-param-family",
                    f'lw-single-exp-param-family--{families[family_id]["color_key"]}',
                ],
                sizing_mode="stretch_width",
                margin=0,
            )
            for family_id in ordered_family_ids
        ]
        return pn.Column(
            *family_views,
            sizing_mode="stretch_width",
            margin=0,
        )

    @staticmethod
    def _apply_parameter_group_card_style(view: pn.viewable.Viewable) -> None:
        for child in getattr(view, "objects", []):
            if not isinstance(child, pn.Card):
                continue
            if "lw-single-exp-param-group-card" not in child.css_classes:
                child.css_classes.append("lw-single-exp-param-group-card")

    def _render_all_parameter_groups(self) -> None:
        self._parameter_group_views = {}
        environment_column = pn.Column(sizing_mode="stretch_width", margin=0)
        experiment_columns = [
            pn.Column(sizing_mode="stretch_width", margin=0),
            pn.Column(sizing_mode="stretch_width", margin=0),
        ]
        used_keys: set[str] = set()

        def _add_group(
            group_key: str,
            column_index: int | None = None,
            *,
            target: list[pn.Column] | pn.Column,
        ) -> None:
            if group_key in used_keys or group_key not in self._parameter_groups:
                return
            group_view = self._render_single_parameter_group(group_key)
            if group_view is None:
                return
            title = _editor_group_title(group_key)
            if group_key == "env_params":
                self._apply_parameter_group_card_style(group_view)
                group_view = pn.Column(
                    pn.Column(
                        self.environment_preset_view,
                        css_classes=["lw-single-exp-env-preset-box"],
                        sizing_mode="stretch_width",
                        margin=(0, 0, 8, 0),
                    ),
                    group_view,
                    css_classes=["lw-single-exp-env-params-content"],
                    sizing_mode="stretch_width",
                    margin=0,
                )
            used_keys.add(group_key)
            self._parameter_group_views[group_key] = group_view
            if group_key == "env_params" and not isinstance(target, list):
                target.append(group_view)
                return
            card = pn.Card(
                group_view,
                title=title,
                collapsed=False,
                sizing_mode="stretch_width",
                css_classes=["lw-single-exp-param-group-card"],
            )
            if isinstance(target, list):
                assert column_index is not None
                target[column_index].append(card)
            else:
                target.append(card)

        _add_group("env_params", target=environment_column)
        for key in ["larva_groups", "trials"]:
            _add_group(key, 0, target=experiment_columns)
        for key in ["enrichment", "sim_ops", "collections"]:
            _add_group(key, 1, target=experiment_columns)

        fallback_column = 0
        for group_key in self._parameter_groups:
            if group_key in used_keys:
                continue
            _add_group(group_key, fallback_column, target=experiment_columns)
            fallback_column = (fallback_column + 1) % len(experiment_columns)
        self.environment_parameters_editor[:] = [environment_column]
        self.parameters_editor[:] = [
            pn.Row(
                *experiment_columns,
                css_classes=["lw-single-exp-params-columns"],
                sizing_mode="stretch_width",
                margin=0,
            )
        ]

    def _get_parameter_group_view(self, group_key: str) -> pn.viewable.Viewable | None:
        return self._parameter_group_views.get(group_key)

    def _refresh_parameter_editor(self) -> None:
        flat = self._editable_flat_parameters()
        self._refresh_typed_larva_groups_owner()
        self._refresh_typed_enrichment_owner()
        self._refresh_typed_env_params_owner()
        self._refresh_typed_sim_ops_owner()
        self._refresh_typed_collections_owner()
        self._refresh_typed_trials_owner()
        self._editor_context = {
            "odor_ids": sorted(
                {
                    str(value)
                    for path, value in flat.items()
                    if path.endswith(".odor.id")
                    and isinstance(value, str)
                    and value.strip()
                }
            )
        }
        self._parameter_widget_specs = {}
        grouped: dict[str, list[str]] = {}
        self._parameter_widgets = {}
        # Keep paths in lexical order so newly materialized nested configs
        # (e.g. source units/groups) appear close to their family instead of
        # being appended at the end of the editor after factory activation.
        for path in sorted(flat.keys()):
            if path in _SIM_OPS_FIELDS:
                continue
            if path == "env_params" or path.startswith("env_params."):
                continue
            if path == "enrichment" or path.startswith("enrichment."):
                continue
            if path.startswith("larva_groups."):
                continue
            if path == "collections" or path.startswith("collections."):
                continue
            if path == "trials" or path.startswith("trials."):
                continue
            value = flat[path]
            group_key = path.split(".", 1)[0]
            kind, control, view = self._widget_for_value(path, value)
            grouped.setdefault(group_key, []).append(path)
            self._parameter_widgets[path] = (kind, control)
            self._parameter_widget_specs[path] = (kind, control, view)
        grouped.setdefault("enrichment", [])
        grouped.setdefault("larva_groups", [])
        grouped.setdefault("env_params", [])
        grouped.setdefault("sim_ops", [])
        grouped.setdefault("collections", [])
        grouped.setdefault("trials", [])
        self._parameter_groups = grouped
        self._wire_optional_family_toggles()
        self._render_all_parameter_groups()
        self._bind_environment_watchers()
        self._bind_experiment_template_watchers()
        self._refresh_environment_save_state(reset_baseline=True)
        self._refresh_experiment_template_save_state(reset_baseline=True)

    def _refresh_typed_larva_groups_owner(self) -> None:
        self._typed_experiment_for_larva_groups = (
            self._build_typed_experiment_from_selected_template()
        )
        self._apply_workspace_overrides_to_typed_owner(
            exp_owner=self._typed_experiment_for_larva_groups,
            section="larva_groups",
        )
        self._larva_groups_group_view = build_larva_groups_widget(
            self._typed_experiment_for_larva_groups
        )

    def _build_typed_experiment_from_selected_template(self) -> Any:
        from larvaworld.lib.reg.generators import ExpConf

        parameters = util.AttrDict(self._parameters_from_selected_template().get_copy())
        larva_groups = parameters.get("larva_groups")
        if isinstance(larva_groups, dict):
            typed_larva_groups = util.AttrDict()
            for group_id, group_payload in larva_groups.items():
                if isinstance(group_payload, reg.gen.LarvaGroup):
                    typed_larva_groups[group_id] = group_payload
                    continue
                if isinstance(group_payload, dict):
                    group_kwargs = dict(group_payload)
                    model_payload = group_kwargs.pop("model", None)
                    if "group_id" not in group_kwargs:
                        group_kwargs["group_id"] = str(group_id)
                    if isinstance(model_payload, dict):
                        group_obj = reg.gen.LarvaGroup(**group_kwargs)
                        group_obj.param.model.check_on_set = False
                        group_obj.model = model_payload
                        group_obj.param.model.check_on_set = True
                        typed_larva_groups[group_id] = group_obj
                    else:
                        if model_payload is not None:
                            group_kwargs["model"] = model_payload
                        typed_larva_groups[group_id] = reg.gen.LarvaGroup(
                            **group_kwargs
                        )
                    continue
                typed_larva_groups[group_id] = group_payload
            parameters["larva_groups"] = typed_larva_groups
        return ExpConf(**dict(parameters))

    def _refresh_typed_enrichment_owner(self) -> None:
        self._typed_experiment_for_enrichment = (
            self._build_typed_experiment_from_selected_template()
        )
        self._apply_workspace_overrides_to_typed_owner(
            exp_owner=self._typed_experiment_for_enrichment,
            section="enrichment",
        )
        self._enrichment_group_view = build_enrichment_widget(
            self._typed_experiment_for_enrichment.enrichment,
            wrap=False,
        )

    def _refresh_typed_env_params_owner(self) -> None:
        self._typed_experiment_for_env_params = (
            self._build_typed_experiment_from_selected_template()
        )
        self._apply_workspace_overrides_to_typed_owner(
            exp_owner=self._typed_experiment_for_env_params,
            section="env_params",
        )
        self._env_params_group_view = build_env_params_widget(
            self._typed_experiment_for_env_params.env_params,
            wrap=False,
        )

    def _refresh_typed_sim_ops_owner(self) -> None:
        self._typed_experiment_for_sim_ops = (
            self._build_typed_experiment_from_selected_template()
        )
        self._apply_workspace_sim_settings_to_typed_owner(
            self._typed_experiment_for_sim_ops
        )
        self._sim_ops_group_view = build_sim_ops_widget(
            self._typed_experiment_for_sim_ops,
            wrap=False,
        )

    def _refresh_typed_collections_owner(self) -> None:
        self._typed_experiment_for_collections = (
            self._build_typed_experiment_from_selected_template()
        )
        self._apply_workspace_overrides_to_typed_owner(
            exp_owner=self._typed_experiment_for_collections,
            section="collections",
        )
        self._collections_group_view = build_collections_widget(
            self._typed_experiment_for_collections,
            wrap=False,
        )

    def _refresh_typed_trials_owner(self) -> None:
        self._typed_experiment_for_trials = (
            self._build_typed_experiment_from_selected_template()
        )
        self._apply_workspace_overrides_to_typed_owner(
            exp_owner=self._typed_experiment_for_trials,
            section="trials",
        )
        self._trials_group_view = build_trials_widget(
            self._typed_experiment_for_trials,
            wrap=False,
        )

    def _on_experiment_change(self, *_: object) -> None:
        if self._suspend_experiment_change:
            return
        selected_value = str(self.selection.experiment_template)
        workspace_filename = self._workspace_experiment_filename_from_token(
            selected_value
        )
        payload: util.AttrDict | None = None
        if workspace_filename is not None:
            try:
                payload = self._load_workspace_experiment_template_payload(
                    workspace_filename
                )
            except (WorkspaceError, OSError, ValueError) as exc:
                self.status.object = str(exc)
                return
            self._active_workspace_template_payload = payload
            self._active_workspace_template_filename = workspace_filename
            self.experiment_template_save_name.value = Path(workspace_filename).stem
        else:
            self._active_workspace_template_payload = None
            self._active_workspace_template_filename = None

        experiment = self._selected_experiment()
        self._last_valid_experiment_template = selected_value
        self.run_name.value = _default_run_name(experiment)
        self._active_environment_preset_ref = None
        self._active_environment_preset_payload = None
        self._refresh_environment_options()
        self._refresh_experiment_template_options()
        if (
            self.selection.experiment_template
            in self.experiment_template_preset_controls.catalog.by_token
        ):
            self.experiment_template_preset_controls.preset_select.value = (
                self.selection.experiment_template
            )
        self.selection.environment_preset = "__template__"
        if payload is not None:
            self._parameter_seed_overrides = self._workspace_template_seed_overrides(
                payload
            )
        else:
            self._parameter_seed_overrides = util.AttrDict()
        self._refresh_summary()
        self._refresh_parameter_editor()
        self.status.object = f'Template "{experiment}" loaded.'

    def _on_run_name_change(self, *_: object) -> None:
        current = (self.video_filename.value or "").strip()
        if not current or current == _safe_slug(current):
            self.video_filename.value = _safe_slug(
                self.run_name.value or self._selected_experiment()
            )

    def _on_save_video_change(self, *_: object) -> None:
        enabled = bool(self.save_video.value)
        self.video_filename.disabled = not enabled
        self.video_fps.disabled = not enabled

    def _on_show_display_change(self, *_: object) -> None:
        self.display_every_n_steps.disabled = not bool(self.show_display.value)

    def _on_parameter_override_change(self, *_: object) -> None:
        self._parameter_seed_overrides = util.AttrDict()
        self._refresh_summary()
        self._refresh_parameter_editor()

    def _on_refresh_environments(self, *_: object) -> None:
        self._refresh_environment_options()
        self._refresh_summary()
        self._refresh_parameter_editor()
        self.status.object = "Refreshed environment presets."

    def _build_run_directory(self) -> Path:
        run_id = _safe_slug(self.run_name.value or self._selected_experiment())
        base_dir = self._experiment_dir()
        candidate = base_dir / run_id
        if not candidate.exists():
            return candidate
        suffix = 2
        while (base_dir / f"{run_id}_{suffix}").exists():
            suffix += 1
        return base_dir / f"{run_id}_{suffix}"

    @staticmethod
    def _resolved_plan_payload(
        *,
        experiment: str,
        run_name: str,
        selected_env: str,
        parameters: util.AttrDict,
    ) -> dict[str, Any]:
        return {
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "experiment": experiment,
            "run_name": run_name,
            "selected_environment": selected_env,
            "parameters": _json_ready(parameters.get_copy()),
        }

    def _write_resolved_plan(
        self,
        *,
        run_dir: Path,
        parameters: util.AttrDict,
        selected_env: str,
    ) -> Path:
        run_dir.mkdir(parents=True, exist_ok=False)
        plan_path = run_dir / "resolved_experiment.json"
        payload = self._resolved_plan_payload(
            experiment=self._selected_experiment(),
            run_name=self.run_name.value or run_dir.name,
            selected_env=selected_env,
            parameters=parameters,
        )
        plan_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return plan_path

    def _runtime_screen_kws(self, run_dir: Path) -> dict[str, Any]:
        kws: dict[str, Any] = {
            "show_display": bool(self.show_display.value),
            "display_every_n_steps": int(self.display_every_n_steps.value),
        }
        if self.show_display.value:
            kws["vis_mode"] = "video"
        if self.save_video.value:
            video_name = _safe_slug(
                (self.video_filename.value or "").strip() or run_dir.name
            )
            kws.update(
                {
                    "save_video": True,
                    "vis_mode": "video",
                    "video_file": video_name,
                    "media_dir": str(run_dir),
                    "fps": int(self.video_fps.value),
                }
            )
        return kws

    def _set_run_controls_disabled(self, disabled: bool) -> None:
        self._run_controls_locked = bool(disabled)
        self.prepare_btn.disabled = disabled
        self.simulation_preview_btn.disabled = disabled
        self.run_btn.disabled = disabled
        self.preview_frames_input.disabled = disabled
        self.experiment_template_save_name.disabled = disabled
        self.experiment_template_save_btn.disabled = disabled
        self.display_shortcuts_link.disabled = disabled
        self.display_shortcuts_close_btn.disabled = disabled
        if disabled:
            self.display_shortcuts_dialog.visible = False
        self.display_every_n_steps.disabled = disabled or not bool(
            self.show_display.value
        )
        self.refresh_environments_btn.disabled = disabled
        if disabled:
            self.experiment_template_preset_controls.load_button.disabled = True
            self.experiment_template_preset_controls.delete_button.disabled = True
            self.environment_save_name.disabled = True
            self.environment_save_btn.disabled = True
            self.environment_preset_controls.load_button.disabled = True
            self.environment_preset_controls.delete_button.disabled = True
        else:
            self._refresh_experiment_template_options()
            self.experiment_template_preset_controls.load_button.disabled = False
            self.environment_preset_controls.load_button.disabled = False
            self.environment_preset_controls.delete_button.disabled = False
            self._refresh_environment_save_state(reset_baseline=False)
            self._refresh_experiment_template_save_state(reset_baseline=False)

    def _execute_run_experiment(
        self,
        *,
        parameters: util.AttrDict,
        run_dir: Path,
        selected_env: str,
    ) -> None:
        launcher = None
        warning_note = self._pending_run_warning_note
        self._pending_run_warning_note = ""
        try:
            plan_path = self._write_resolved_plan(
                run_dir=run_dir,
                parameters=parameters,
                selected_env=selected_env,
            )
            screen_kws = self._runtime_screen_kws(run_dir)
            launcher = sim.ExpRun(
                experiment=self._selected_experiment(),
                parameters=parameters,
                id=run_dir.name,
                dir=str(run_dir),
                store_data=True,
                screen_kws=screen_kws,
            )
            datasets = launcher.simulate()
        except Exception as exc:
            self.status.object = f"Single experiment run failed: {exc}{warning_note}"
            self._set_run_controls_disabled(False)
            return
        finally:
            try:
                if launcher is not None and getattr(launcher, "screen_manager", None):
                    launcher.screen_manager.close()
            except Exception:
                pass

        dataset_count = len(datasets or [])
        self.status.object = (
            f'Completed run "{self._selected_experiment()}". '
            f"Stored outputs in <code>{run_dir}</code>. "
            f"Saved resolved config to <code>{plan_path.name}</code>. "
            f"Datasets produced: {dataset_count}."
            + (
                f' Video target: <code>{run_dir / ((_safe_slug((self.video_filename.value or "").strip() or run_dir.name)) + ".mp4")}</code>.'
                if self.save_video.value
                else ""
            )
            + warning_note
        )
        self._set_run_controls_disabled(False)

    @staticmethod
    def _preview_metadata_html(parameters: util.AttrDict, selected_env: str) -> str:
        env = util.AttrDict(parameters.env_params)
        larva_groups = util.AttrDict(parameters.get("larva_groups", {}))
        counts = []
        total = 0
        for group_id, group in larva_groups.items():
            try:
                count = int(group.distribution.N)
            except Exception:
                count = 0
            total += count
            counts.append(f"{group_id}: {count}")
        dims = getattr(env.arena, "dims", ("?", "?"))
        if isinstance(dims, (list, tuple)) and len(dims) >= 2:
            dims_text = f"{float(dims[0]):.3f} x {float(dims[1]):.3f} m"
        else:
            dims_text = str(dims)
        return (
            '<div class="lw-single-exp-preview-meta">'
            f"<strong>Applied preview config:</strong> "
            f"environment = {selected_env}; "
            f"arena = {env.arena.geometry} ({dims_text}); "
            f"duration = {float(parameters.duration):.2f} min; "
            f"larvae = {total}"
            + (f" ({', '.join(counts)})" if counts else "")
            + ".</div>"
        )

    @staticmethod
    def _preview_runtime_parameters(parameters: util.AttrDict) -> util.AttrDict:
        preview_parameters = util.AttrDict(_coerce_xy_sequences(parameters.get_copy()))
        preview_parameters["collections"] = []
        preview_parameters["enrichment"] = None
        return preview_parameters

    @staticmethod
    def _prepare_preview_launcher(
        experiment: str,
        parameters: util.AttrDict,
        run_dir: Path,
    ) -> tuple[sim.ExpRun, str | None]:
        preview_parameters = _SingleExperimentController._preview_runtime_parameters(
            parameters
        )
        fallback_note = None
        try:
            launcher = sim.ExpRun(
                experiment=experiment,
                parameters=preview_parameters,
                id=run_dir.name,
                dir=str(run_dir),
                store_data=False,
            )
            launcher.sim_setup(steps=_PREVIEW_STEP_CAP)
            return launcher, fallback_note
        except Exception as exc:
            if "get_polygon" not in str(exc):
                raise
            preview_parameters = preview_parameters.get_copy()
            preview_parameters["larva_collisions"] = True
            launcher = sim.ExpRun(
                experiment=experiment,
                parameters=preview_parameters,
                id=run_dir.name,
                dir=str(run_dir),
                store_data=False,
            )
            launcher.sim_setup(steps=_PREVIEW_STEP_CAP)
            fallback_note = "Preview fallback disabled larva overlap elimination for this visualization."
            return launcher, fallback_note

    def _on_prepare_preview(self, *_: object) -> None:
        try:
            parameters = self._resolve_experiment_parameters()
        except (WorkspaceError, OSError, json.JSONDecodeError, ValueError) as exc:
            self.status.object = f"Cannot prepare the configuration preview: {exc}"
            return
        ok, warning_note = self._validate_resolved_parameters_for_action(
            parameters=parameters,
            action_label="Arena Preview",
            show_preview_failure=True,
        )
        if not ok:
            return
        try:
            run_dir = self._build_run_directory()
        except (WorkspaceError, OSError, json.JSONDecodeError) as exc:
            self.status.object = f"Cannot prepare the configuration preview: {exc}"
            return

        selected_env = self._selected_environment_label()
        self._set_run_controls_disabled(True)
        try:
            state = env_params_to_canvas_state(
                parameters.env_params,
                larva_groups=parameters.get("larva_groups", {}),
                show_group_shapes=False,
            )
            canvas = self._new_preview_canvas()
            canvas.set_state(state)
        except Exception as exc:
            self.preview[:] = [
                pn.pane.HTML(
                    (
                        '<div class="lw-single-exp-preview-placeholder">'
                        f"Configuration preview failed: {exc}"
                        "</div>"
                    ),
                    margin=0,
                )
            ]
            self.status.object = f"Cannot prepare the configuration preview: {exc}"
            self._set_run_controls_disabled(False)
            return

        self.preview[:] = [_preview_canvas_row(canvas.view())]
        self.status.object = (
            f'Prepared configuration preview for "{self._selected_experiment()}" using '
            f"{selected_env}. Reserved output directory for a future run: "
            f"<code>{run_dir}</code>. No simulation has been run."
            f"{warning_note}"
        )
        self._set_run_controls_disabled(False)

    def _on_generate_simulation_preview(self, *_: object) -> None:
        try:
            parameters = self._resolve_experiment_parameters()
        except (WorkspaceError, OSError, json.JSONDecodeError, ValueError) as exc:
            self.status.object = f"Cannot generate the simulation preview: {exc}"
            return
        ok, warning_note = self._validate_resolved_parameters_for_action(
            parameters=parameters,
            action_label="Generate simulation preview",
            show_preview_failure=True,
        )
        if not ok:
            return
        try:
            run_dir = self._build_run_directory()
        except (WorkspaceError, OSError, json.JSONDecodeError) as exc:
            self.status.object = f"Cannot generate the simulation preview: {exc}"
            return

        selected_env = self._selected_environment_label()
        self.preview[:] = [
            pn.pane.HTML(
                (
                    '<div class="lw-single-exp-preview-placeholder">'
                    "Generating simulation preview. The environment and agents are being initialized."
                    "</div>"
                ),
                margin=0,
            )
        ]
        experiment = self._selected_experiment()
        self.status.object = (
            f'Generating simulation preview for "{experiment}" using {selected_env}.'
        )
        self._set_run_controls_disabled(True)
        launcher = None
        try:
            launcher, fallback_note = self._prepare_preview_launcher(
                experiment,
                parameters,
                run_dir,
            )
            requested_steps = int(self.preview_frames_input.value)
            preview_steps = max(1, min(requested_steps, int(launcher.p.steps)))
            frames = generate_preview_frames(
                launcher,
                preview_steps=preview_steps,
            )
            if not frames:
                raise ValueError("No preview frames were generated.")
            state = env_params_to_canvas_state(
                parameters.env_params,
                larva_groups=None,
                show_group_shapes=False,
            )
            canvas = self._new_preview_canvas()
            canvas.set_state(state)
            preview = _FrameSimulationPreview(
                canvas=canvas,
                frames=frames,
                dt=float(launcher.dt),
            ).view()
        except Exception as exc:
            self.preview[:] = [
                pn.pane.HTML(
                    (
                        '<div class="lw-single-exp-preview-placeholder">'
                        f"Preview preparation failed: {exc}"
                        "</div>"
                    ),
                    margin=0,
                )
            ]
            self.status.object = f"Cannot generate the simulation preview: {exc}"
            self._set_run_controls_disabled(False)
            return
        finally:
            try:
                if launcher is not None and getattr(launcher, "screen_manager", None):
                    launcher.screen_manager.close()
            except Exception:
                pass

        self.preview[:] = [preview]
        displayed_end = frames[-1].tick * float(launcher.dt)
        status = (
            f'Generated simulation preview for "{experiment}" using {selected_env}. '
            f"Reserved output directory for a future run: <code>{run_dir}</code>. "
            f"Simulation preview ready: {len(frames)} frames generated. "
            f"Displayed range: 0.0-{displayed_end:.1f} s simulated time. "
            "Generated from the actual simulation engine. Outputs are not stored; use Run experiment for the full run."
        )
        if fallback_note:
            status = f"{status} {fallback_note}"
        self.status.object = f"{status}{warning_note}"
        self._set_run_controls_disabled(False)

    def _on_run_experiment(self, *_: object) -> None:
        self._pending_run_warning_note = ""
        try:
            parameters = self._resolve_experiment_parameters()
        except (WorkspaceError, OSError, json.JSONDecodeError, ValueError) as exc:
            self.status.object = f"Cannot start the single experiment run: {exc}"
            return
        ok, warning_note = self._validate_resolved_parameters_for_action(
            parameters=parameters,
            action_label="Run experiment",
            show_preview_failure=False,
        )
        if not ok:
            return
        try:
            run_dir = self._build_run_directory()
        except (WorkspaceError, OSError, json.JSONDecodeError) as exc:
            self.status.object = f"Cannot start the single experiment run: {exc}"
            return

        selected_env = self._selected_environment_label()
        experiment = self._selected_experiment()
        self.status.object = (
            f'Running "{experiment}" using {selected_env}. '
            f"Outputs will be stored under <code>{run_dir}</code>. "
            "The UI will be unresponsive until the simulation finishes."
            f"{warning_note}"
        )
        self.preview[:] = [
            pn.pane.HTML(
                (
                    '<div class="lw-single-exp-preview-placeholder">'
                    f'Running "{html.escape(experiment)}" and writing outputs to '
                    f"<code>{html.escape(str(run_dir))}</code>. "
                    "This view will update again after the simulation finishes."
                    "</div>"
                ),
                margin=0,
            )
        ]
        self._pending_run_warning_note = warning_note
        self._set_run_controls_disabled(True)
        document = pn.state.curdoc
        if document is not None:
            document.add_next_tick_callback(
                lambda: self._execute_run_experiment(
                    parameters=parameters,
                    run_dir=run_dir,
                    selected_env=selected_env,
                )
            )
            return
        self._execute_run_experiment(
            parameters=parameters,
            run_dir=run_dir,
            selected_env=selected_env,
        )

    def _on_open_display_shortcuts(self, *_: object) -> None:
        self.display_shortcuts_dialog.visible = True

    def _on_close_display_shortcuts(self, *_: object) -> None:
        self.display_shortcuts_dialog.visible = False

    def view(self) -> pn.viewable.Viewable:
        intro = pn.pane.Markdown(
            (
                "### Single Experiment\n"
                "Prepare one Larvaworld `Exp` run in the portal: select an experiment template, "
                "optionally override its environment with a workspace preset, and inspect the arena "
                "dynamics in an interactive preview before broader simulation workflows are added. "
                f"References: [Single Experiments]({DOCS_SINGLE_EXPERIMENTS}) and "
                f"[Experiment Types]({DOCS_EXPERIMENT_TYPES})."
            ),
            css_classes=["lw-single-exp-intro"],
            margin=0,
        )
        media_controls = pn.Column(
            pn.pane.Markdown("#### Media / Output", margin=(0, 0, 6, 0)),
            self.save_video,
            self.video_filename,
            self.video_fps,
            self.show_display,
            self.display_every_n_steps,
            css_classes=["lw-single-exp-media"],
            sizing_mode="stretch_width",
            margin=0,
        )
        controls = pn.Card(
            pn.Column(
                self.experiment_template_save_box,
                self.summary,
                self.preview_action_row,
                self.preview_options_row,
                self.preview_generate_row,
                media_controls,
                self.run_name,
                self.execution_action_row,
                self.run_info,
                sizing_mode="stretch_width",
                margin=0,
            ),
            title="Configuration",
            collapsed=False,
            sizing_mode="stretch_width",
        )
        preview = pn.Card(
            self.preview,
            title="Preview",
            collapsed=False,
            sizing_mode="stretch_width",
        )
        environment_parameters = pn.Card(
            pn.Column(
                self.environment_parameters_editor,
                sizing_mode="stretch_width",
                margin=0,
            ),
            title="Environment Parameters",
            collapsed=False,
            sizing_mode="stretch_width",
            css_classes=["lw-single-exp-params-group"],
        )
        experiment_parameters = pn.Card(
            pn.Column(
                self.parameters_editor,
                sizing_mode="stretch_width",
                margin=0,
            ),
            title="Experiment Parameters",
            collapsed=False,
            sizing_mode="stretch_width",
            css_classes=["lw-single-exp-params-group"],
        )
        left_column = pn.Column(controls, sizing_mode="stretch_width")
        content = pn.GridSpec(
            ncols=4,
            nrows=1,
            sizing_mode="stretch_width",
        )
        content[0, 0] = left_column
        content[0, 1:4] = preview
        lower = pn.GridSpec(
            ncols=3,
            nrows=1,
            sizing_mode="stretch_width",
            margin=(8, 0, 0, 0),
        )
        lower[0, 0] = environment_parameters
        lower[0, 1:3] = experiment_parameters
        return pn.Column(
            intro,
            self.display_shortcuts_dialog,
            content,
            lower,
            css_classes=["lw-single-exp-root"],
            sizing_mode="stretch_width",
        )


def single_experiment_app() -> pn.viewable.Viewable:
    pn.extension(raw_css=[PORTAL_RAW_CSS, SINGLE_EXPERIMENT_RAW_CSS])
    hv.extension("bokeh")
    controller = _SingleExperimentController()

    template = pn.template.MaterialTemplate(
        title="",
        header_background="#b5c2b0",
        header_color="#111111",
    )
    template.header.append(build_app_header(title="Single Experiment"))
    template.main.append(controller.view())
    return template
