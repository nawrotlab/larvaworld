from __future__ import annotations

import html
import io
import json
from dataclasses import dataclass
import math
from numbers import Number
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
import panel as pn
import param
from bokeh.events import Tap
from bokeh.models import ColumnDataSource, WheelZoomTool
from bokeh.plotting import figure

from larvaworld.lib import reg, util
from larvaworld.lib.reg import config as reg_config
from larvaworld.lib.model.envs.valuegrid import FoodGrid
from larvaworld.lib.param.custom import ClassAttr, ClassDict
from larvaworld.lib.param.composition import Substrate, substrate_dict
from larvaworld.lib.param.spatial import Area
from larvaworld.lib.param.xy_distro import Spatial_Distro
from larvaworld.portal.landing_registry import DOCS_ARENAS_SUBSTRATES
from larvaworld.portal.panel_components import PORTAL_RAW_CSS, build_app_header
from larvaworld.portal.workspace import WorkspaceError, get_workspace_dir


LANE_MODELS_COLOR = "#c1b0c2"
LANE_MODELS_COLOR_DARK = "#5a4760"
SUBSTRATE_TYPE_OPTIONS = ["standard"] + [
    key for key in substrate_dict.keys() if key != "standard"
]
ENV_CANVAS_WIDTH = 760
ENV_CANVAS_HEIGHT = 620
ENV_CANVAS_Y_HALF_RANGE = 0.30
ENV_CANVAS_X_HALF_RANGE = ENV_CANVAS_Y_HALF_RANGE * (
    ENV_CANVAS_WIDTH / ENV_CANVAS_HEIGHT
)

ENV_BUILDER_RAW_CSS = """
.lw-env-builder-root {
  padding: 14px 12px 20px 12px;
}

.lw-env-builder-divider {
  width: 100%;
  height: 1px;
  background: rgba(17, 17, 17, 0.38);
  margin: 6px 0 8px 0;
}

.lw-env-builder-actions {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
  width: 100%;
}

.lw-env-builder-actions-primary {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
}

.lw-env-builder-intro {
  border-left: 4px solid #c1b0c2;
  background: rgba(193, 176, 194, 0.18);
  border-radius: 10px;
  padding: 10px 12px;
  margin: 0 0 10px 0;
}

.lw-env-builder-intro a {
  color: #4f2f5f;
}

.lw-env-builder-note {
  font-size: 12px;
  line-height: 1.45;
  color: rgba(17, 17, 17, 0.72);
}

.lw-env-builder-objects-table .tabulator {
  font-size: 12px;
}

.lw-env-builder-objects-table .tabulator .tabulator-col,
.lw-env-builder-objects-table .tabulator .tabulator-cell {
  font-size: 12px;
}

.lw-env-builder-family {
  background: rgba(248, 248, 250, 0.96);
  border: 1px solid rgba(90, 71, 96, 0.12);
  border-radius: 10px;
  padding: 10px 12px 8px 12px;
  margin-top: 4px;
}

.lw-env-builder-scape-family {
  background: rgba(252, 252, 253, 0.99);
  border-color: rgba(90, 71, 96, 0.10);
}

.lw-env-builder-family-title {
  margin: 0 0 4px 0;
  color: #4f2f5f;
  font-weight: 700;
}

.lw-env-builder-family-title p {
  margin: 0;
}

.lw-env-builder-param-header {
  display: flex;
  align-items: center;
  gap: 6px;
  margin: 0 0 4px 0;
}

.lw-env-builder-param-header-label {
  font-size: 12px;
  font-weight: 600;
  color: rgba(17, 17, 17, 0.78);
}

.lw-env-builder-param-help {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 11px;
  height: 11px;
  border-radius: 999px;
  background: #5a4760;
  color: white;
  font-size: 8px;
  font-weight: 700;
  cursor: help;
  user-select: none;
}

.lw-env-builder-param-field {
  margin: 0 0 4px 0;
  padding: 0 8px;
}

.lw-env-builder-family-titlebar {
  display: flex;
  align-items: center;
  gap: 6px;
  margin: 0 0 6px 0;
}

.lw-env-builder-family-titletext {
  color: #4f2f5f;
  font-weight: 700;
  font-size: 14px;
  line-height: 1.2;
}

""".strip()


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


def _field_header_html(label: str, help_text: str | None) -> str:
    escaped_label = html.escape(label)
    if not help_text:
        return (
            '<div class="lw-env-builder-param-header">'
            f'<span class="lw-env-builder-param-header-label">{escaped_label}</span>'
            "</div>"
        )
    escaped_help = html.escape(help_text, quote=True)
    return (
        '<div class="lw-env-builder-param-header">'
        f'<span class="lw-env-builder-param-header-label">{escaped_label}</span>'
        f'<span class="lw-env-builder-param-help" title="{escaped_help}">i</span>'
        "</div>"
    )


def _title_with_help_html(
    label: str, help_text: str | None, *, title_class: str
) -> str:
    escaped_label = html.escape(label)
    help_html = ""
    if help_text:
        escaped_help = html.escape(help_text, quote=True)
        help_html = (
            f'<span class="lw-env-builder-param-help" title="{escaped_help}">i</span>'
        )
    return (
        '<div class="lw-env-builder-family-titlebar">'
        f'<span class="{title_class}">{escaped_label}</span>'
        f"{help_html}"
        "</div>"
    )


def _editor_family_box(
    title: str, *children: object, help_text: str | None = None
) -> pn.Column:
    return pn.Column(
        pn.pane.Markdown(
            f"**{title}**",
            css_classes=["lw-env-builder-family-title"],
            margin=(0, 0, 6, 0),
        ),
        *children,
        css_classes=["lw-env-builder-family"],
        sizing_mode="stretch_width",
        margin=0,
    )


@dataclass(frozen=True)
class _ObjectRow:
    object_id: str
    object_type: str
    x: float | None
    y: float | None
    x2: float | None = None
    y2: float | None = None
    radius: float | None = None
    width: float | None = None
    color: str | None = None
    amount: float | None = None
    odor_id: str | None = None
    odor_intensity: float | None = None
    odor_spread: float | None = None
    can_be_carried: bool | None = None
    can_be_displaced: bool | None = None
    regeneration: bool | None = None
    substrate_type: str | None = None
    substrate_quality: float | None = None
    distribution_mode: str | None = None
    distribution_shape: str | None = None
    distribution_n: int | None = None
    distribution_scale_x: float | None = None
    distribution_scale_y: float | None = None
    distribution_show_shape: bool | None = None


_WIND_PUFF_COLUMNS = [
    "id",
    "duration",
    "speed",
    "direction",
    "start_time",
    "N",
    "interval",
]

_THERMO_SOURCE_COLUMNS = [
    "id",
    "x",
    "y",
    "dTemp",
]


_REGISTRY_PRESET_PREFIX = "__registry__:"


def _coerce_xy_sequences(value: object) -> object:
    if isinstance(value, dict):
        return util.AttrDict(
            {str(key): _coerce_xy_sequences(item) for key, item in value.items()}
        )
    if isinstance(value, tuple):
        return tuple(_coerce_xy_sequences(item) for item in value)
    if isinstance(value, list):
        if len(value) == 2 and all(isinstance(item, Number) for item in value):
            return tuple(value)
        if value and all(
            isinstance(item, (list, tuple))
            and len(item) == 2
            and all(isinstance(coord, Number) for coord in item)
            for item in value
        ):
            return [tuple(item) for item in value]
        return [_coerce_xy_sequences(item) for item in value]
    return value


def _translate_builder_environment_payload(
    environment_payload: dict[str, object],
) -> util.AttrDict:
    return util.AttrDict(
        _coerce_xy_sequences(util.AttrDict(environment_payload).get_copy())
    )


def _normalize_group_shape(shape: str | None) -> str:
    normalized = str(shape or "circle").strip().lower()
    if normalized in {"circle", "circular"}:
        return "circle"
    if normalized in {"oval", "ellipse", "elliptical"}:
        return "oval"
    if normalized in {"rect", "rectangle", "rectangular"}:
        return "rect"
    return "circle"


def _source_visual_state(
    *,
    amount: float | None,
    color: str | None,
) -> tuple[str, str, float, float, float]:
    base_color = str(color or "#4caf50")
    has_food = amount is not None and float(amount) > 0
    fill_color = _mix_hex_colors(base_color, "#ffffff", 0.0 if has_food else 0.68)
    line_color = _mix_hex_colors(base_color, "#111111", 0.08 if has_food else 0.02)
    return (
        fill_color,
        line_color,
        0.94 if has_food else 0.34,
        0.98 if has_food else 0.58,
        2.6 if has_food else 1.6,
    )


def _mix_hex_colors(color_a: str, color_b: str, ratio: float) -> str:
    ratio = max(0.0, min(1.0, float(ratio)))

    def _parse(color: str) -> tuple[int, int, int]:
        cleaned = color.strip().lstrip("#")
        if len(cleaned) != 6:
            return (76, 175, 80)
        try:
            return tuple(int(cleaned[idx : idx + 2], 16) for idx in (0, 2, 4))
        except ValueError:
            return (76, 175, 80)

    rgb_a = _parse(color_a)
    rgb_b = _parse(color_b)
    mixed = tuple(
        int(round((1.0 - ratio) * component_a + ratio * component_b))
        for component_a, component_b in zip(rgb_a, rgb_b)
    )
    return "#{:02x}{:02x}{:02x}".format(*mixed)


def _stable_preview_seed(*parts: object) -> int:
    seed = 2166136261
    for part in parts:
        for char in str(part):
            seed ^= ord(char)
            seed = (seed * 16777619) & 0xFFFFFFFF
    return seed


def _build_odor_layers(
    *,
    x: float | None,
    y: float | None,
    source_radius: float | None,
    odor_id: str | None,
    odor_intensity: float | None,
    odor_spread: float | None,
    color: str | None,
    source_id: str | None,
) -> list[dict[str, object]]:
    if x is None or y is None or not odor_id:
        return []
    if odor_intensity is None or odor_spread is None:
        return []
    spread = float(odor_spread)
    intensity = float(odor_intensity)
    if spread <= 0 or intensity <= 0:
        return []

    intensity_scale = min(1.0, 0.35 + 0.18 * intensity)
    source_r = max(float(source_radius or 0.0), 0.002)
    aura_color = _mix_hex_colors(str(color or "#4caf50"), "#ffffff", 0.2)
    sigmas = [0.45, 0.9, 1.4, 2.0, 2.8, 3.8]
    alphas = [0.18, 0.13, 0.09, 0.055, 0.03, 0.014]
    rows: list[dict[str, object]] = []
    for sigma, alpha in zip(sigmas, alphas):
        rows.append(
            {
                "x": float(x),
                "y": float(y),
                "r": source_r + spread * sigma,
                "color": aura_color,
                "fill_alpha": alpha * intensity_scale,
                "id": str(source_id or ""),
            }
        )
    return rows


def _build_odor_peak(
    *,
    x: float | None,
    y: float | None,
    source_radius: float | None,
    odor_id: str | None,
    odor_intensity: float | None,
    odor_spread: float | None,
    color: str | None,
    source_id: str | None,
) -> dict[str, object] | None:
    if x is None or y is None or not odor_id:
        return None
    if odor_intensity is None or odor_spread is None:
        return None
    spread = float(odor_spread)
    intensity = float(odor_intensity)
    if spread <= 0 or intensity <= 0:
        return None

    source_r = max(float(source_radius or 0.0), 0.002)
    peak_radius = max(source_r * 0.42, min(spread * 0.3, source_r * 0.72))
    peak_color = _mix_hex_colors(str(color or "#4caf50"), "#ffffff", 0.08)
    return {
        "x": float(x),
        "y": float(y),
        "r": peak_radius,
        "color": peak_color,
        "fill_alpha": min(0.72, 0.44 + 0.1 * intensity),
        "id": str(source_id or ""),
    }


def _table_dataframe(rows: list[dict[str, Any]], columns: list[str]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(rows)
    for column in columns:
        if column not in frame.columns:
            frame[column] = None
    return frame[columns]


def _rotate_point(x: float, y: float, angle: float) -> tuple[float, float]:
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    return (x * cos_a - y * sin_a, x * sin_a + y * cos_a)


def _rad_to_deg(angle: float) -> float:
    return round(math.degrees(float(angle)), 4)


def _deg_to_rad(angle: float) -> float:
    return float(math.radians(float(angle)))


class _EnvironmentBuilderController:
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
            nested_doc = _EnvironmentBuilderController._resolve_doc_from_class(
                item_cls, rest[1:]
            )
            return nested_doc or current_doc
        if isinstance(p, ClassAttr):
            nested_cls = p.class_[0] if isinstance(p.class_, tuple) else p.class_
            nested_doc = _EnvironmentBuilderController._resolve_doc_from_class(
                nested_cls, rest
            )
            return nested_doc or current_doc
        return current_doc

    @staticmethod
    def _class_doc_summary(cls: type[Any]) -> str | None:
        raw_doc = getattr(cls, "__doc__", None)
        if not raw_doc:
            return None
        lines = [line.strip() for line in raw_doc.strip().splitlines()]
        summary = []
        for line in lines:
            if not line:
                if summary:
                    break
                continue
            summary.append(line)
        return " ".join(summary) if summary else None

    @staticmethod
    def _param_doc_for_key(key: str) -> str | None:
        from larvaworld.lib.model.agents._source import Food, Source
        from larvaworld.lib.model.envs.valuegrid import (
            DiffusionValueLayer,
            FoodGrid,
            Grid,
            OdorScape,
            ValueGrid,
            WindScape,
        )
        from larvaworld.lib.param.composition import AirPuff, Odor, Substrate
        from larvaworld.lib.param.spatial import Area
        from larvaworld.lib.param.xy_distro import Spatial_Distro
        from larvaworld.lib.reg.generators import EnvConf, FoodConf, gen

        if key == "arena_shape":
            return _EnvironmentBuilderController._resolve_doc_from_class(
                Area, ["geometry"]
            )
        if key in {"arena_width", "arena_height"}:
            return _EnvironmentBuilderController._resolve_doc_from_class(Area, ["dims"])
        if key == "arena_torus":
            return _EnvironmentBuilderController._resolve_doc_from_class(
                Area, ["torus"]
            )
        if key in {"object_radius", "selected_radius"}:
            return _EnvironmentBuilderController._resolve_doc_from_class(
                Food, ["radius"]
            )
        if key in {"border_width", "selected_width"}:
            return _EnvironmentBuilderController._resolve_doc_from_class(
                gen.Border, ["width"]
            )
        if key in {"group_count", "selected_distribution_n"}:
            return _EnvironmentBuilderController._resolve_doc_from_class(
                Spatial_Distro, ["N"]
            )
        if key in {"group_shape", "selected_distribution_shape"}:
            return _EnvironmentBuilderController._resolve_doc_from_class(
                Spatial_Distro, ["shape"]
            )
        if key in {"group_mode", "selected_distribution_mode"}:
            return _EnvironmentBuilderController._resolve_doc_from_class(
                Spatial_Distro, ["mode"]
            )
        if key in {
            "group_spread_x",
            "group_spread_y",
            "selected_distribution_scale_x",
            "selected_distribution_scale_y",
        }:
            return _EnvironmentBuilderController._resolve_doc_from_class(
                Spatial_Distro, ["scale"]
            )
        if key == "selected_amount":
            return _EnvironmentBuilderController._resolve_doc_from_class(
                Food, ["amount"]
            )
        if key == "selected_can_be_carried":
            return _EnvironmentBuilderController._resolve_doc_from_class(
                Source, ["can_be_carried"]
            )
        if key == "selected_can_be_displaced":
            return _EnvironmentBuilderController._resolve_doc_from_class(
                Source, ["can_be_displaced"]
            )
        if key == "selected_regeneration":
            return _EnvironmentBuilderController._resolve_doc_from_class(
                Source, ["regeneration"]
            )
        if key == "selected_odor_id":
            return _EnvironmentBuilderController._resolve_doc_from_class(Odor, ["id"])
        if key == "selected_odor_intensity":
            return _EnvironmentBuilderController._resolve_doc_from_class(
                Odor, ["intensity"]
            )
        if key == "selected_odor_spread":
            return _EnvironmentBuilderController._resolve_doc_from_class(
                Odor, ["spread"]
            )
        if key in {"selected_substrate_quality", "food_grid_substrate_quality"}:
            return _EnvironmentBuilderController._resolve_doc_from_class(
                Substrate, ["quality"]
            )
        if key == "food_grid_enabled":
            return _EnvironmentBuilderController._resolve_doc_from_class(
                FoodConf, ["food_grid"]
            )
        if key in {
            "food_grid_dims_x",
            "food_grid_dims_y",
            "odorscape_grid_dims_x",
            "odorscape_grid_dims_y",
        }:
            return _EnvironmentBuilderController._resolve_doc_from_class(
                Grid, ["grid_dims"]
            )
        if key == "food_grid_initial_value":
            return _join_help_parts(
                _EnvironmentBuilderController._resolve_doc_from_class(
                    ValueGrid, ["initial_value"]
                ),
                _EnvironmentBuilderController._class_doc_summary(FoodGrid),
            )
        if key == "odorscape_enabled":
            return _EnvironmentBuilderController._resolve_doc_from_class(
                EnvConf, ["odorscape"]
            )
        if key == "odorscape_mode":
            return _EnvironmentBuilderController._resolve_doc_from_class(
                OdorScape, ["odorscape"]
            )
        if key == "odorscape_initial_value":
            return _EnvironmentBuilderController._resolve_doc_from_class(
                ValueGrid, ["initial_value"]
            )
        if key == "odorscape_fixed_max":
            return _EnvironmentBuilderController._resolve_doc_from_class(
                ValueGrid, ["fixed_max"]
            )
        if key == "odorscape_evap_const":
            return _EnvironmentBuilderController._resolve_doc_from_class(
                DiffusionValueLayer, ["evap_const"]
            )
        if key in {"odorscape_sigma_x", "odorscape_sigma_y"}:
            return _EnvironmentBuilderController._resolve_doc_from_class(
                DiffusionValueLayer, ["gaussian_sigma"]
            )
        if key == "windscape_enabled":
            return _EnvironmentBuilderController._resolve_doc_from_class(
                EnvConf, ["windscape"]
            )
        if key == "windscape_direction":
            return _EnvironmentBuilderController._resolve_doc_from_class(
                WindScape, ["wind_direction"]
            )
        if key == "windscape_speed":
            return _EnvironmentBuilderController._resolve_doc_from_class(
                WindScape, ["wind_speed"]
            )
        if key == "wind_puffs_table":
            return _join_help_parts(
                _EnvironmentBuilderController._resolve_doc_from_class(
                    WindScape, ["puffs"]
                ),
                _EnvironmentBuilderController._class_doc_summary(AirPuff),
            )
        if key == "thermoscape_enabled":
            return _EnvironmentBuilderController._resolve_doc_from_class(
                EnvConf, ["thermoscape"]
            )
        if key == "family_food":
            return _EnvironmentBuilderController._class_doc_summary(Food)
        if key == "family_substrate":
            return _EnvironmentBuilderController._class_doc_summary(Substrate)
        if key == "family_odor":
            return _EnvironmentBuilderController._class_doc_summary(Odor)
        if key == "family_distribution":
            return _EnvironmentBuilderController._class_doc_summary(Spatial_Distro)
        if key == "family_source_behavior":
            return _EnvironmentBuilderController._class_doc_summary(Source)
        if key == "family_odorscape":
            return _EnvironmentBuilderController._class_doc_summary(OdorScape)
        if key == "family_windscape":
            return _EnvironmentBuilderController._class_doc_summary(WindScape)
        return None

    @staticmethod
    def _builder_note_for_key(key: str) -> str | None:
        notes = {
            "select_mode": "Toggle selection mode to click existing objects on the canvas instead of inserting new ones.",
            "object_type": "Choose which canonical EnvConf object family to place on the canvas: source unit, source group, or border segment.",
            "arena_width": "Builder note: in circular arenas this field edits the arena radius. The exported config still stores full arena dimensions in meters.",
            "arena_height": "Builder note: used only when the arena shape is rectangular.",
            "object_radius": "Builder note: this controls the source unit radius, not the overall footprint of a source group distribution.",
            "selected_radius": "Builder note: this controls the source unit radius, not the overall footprint of a source group distribution.",
            "object_color": "Preview color used for the object and exported as the object's color.",
            "selected_color": "Preview color used for the object and exported as the object's color.",
            "group_spread_x": "Builder note: for circles this becomes the group radius. For oval and rect groups it represents the horizontal size of the group footprint.",
            "group_spread_y": "Builder note: hidden for circles. For oval and rect groups it represents the vertical size of the group footprint.",
            "selected_distribution_scale_x": "Builder note: for circles this becomes the group radius. For oval and rect groups it represents the horizontal size of the group footprint.",
            "selected_distribution_scale_y": "Builder note: hidden for circles. For oval and rect groups it represents the vertical size of the group footprint.",
            "selected_object": "Pick one placed object to inspect and edit its exported environment parameters.",
            "selected_id": "Unique identifier exported for this object inside source_units, source_groups, or border_list.",
            "selected_x": "X coordinate in meters. For border segments this is the first endpoint.",
            "selected_y": "Y coordinate in meters. For border segments this is the first endpoint.",
            "selected_x2": "Second endpoint X coordinate in meters, used only for border segments.",
            "selected_y2": "Second endpoint Y coordinate in meters, used only for border segments.",
            "selected_odor_id": "Free-text identifier for the odorant. Existing odor IDs from the current environment are offered as suggestions, but custom IDs are allowed.",
            "selected_substrate_type": f"Selects one of the predefined substrate profiles used by the codebase. See Arenas and Substrates docs: {DOCS_ARENAS_SUBSTRATES}",
            "food_grid_enabled": "Adds an optional FoodGrid covering the arena.",
            "food_grid_color": "Preview and export color for the food grid overlay.",
            "food_grid_initial_value": "Builder note: this is the initial food amount per grid cell in simulation units. The current codebase default is 1e-6.",
            "food_grid_substrate_type": f"Selects one of the predefined substrate profiles used by the codebase. See Arenas and Substrates docs: {DOCS_ARENAS_SUBSTRATES}",
            "odorscape_enabled": "Preview contours appear only when at least one placed source has odor configured.",
            "odorscape_color": "Preview and export color for the odorscape layer.",
            "odorscape_initial_value": "Baseline value over the odorscape grid before source contributions are applied.",
            "odorscape_sigma_x": "Builder note: X component of the diffusion Gaussian sigma tuple.",
            "odorscape_sigma_y": "Builder note: Y component of the diffusion Gaussian sigma tuple.",
            "windscape_enabled": "Wind arrows appear in the preview only when wind speed is greater than zero.",
            "windscape_color": "Preview and export color for wind arrows.",
            "windscape_direction": "Builder note: edited in degrees here for readability and converted back to radians when exported to the codebase.",
            "windscape_speed": "Builder note: shown here in meters per second.",
            "wind_puffs_table": "Each row defines one AirPuff event with duration, speed, direction, start time, repetition count, and interval. Puff direction is stored in radians.",
            "thermoscape_enabled": "Thermoscape preview appears only after at least one thermal source row is added.",
            "thermoscape_plate_temp": "Baseline plate temperature in degrees Celsius.",
            "thermoscape_spread": "Gaussian spread parameter controlling how broadly each thermal source diffuses across the arena.",
            "thermo_sources_table": "Each row adds one thermal source at x,y with dTemp relative to the plate temperature. Positive dTemp is warmer; negative dTemp is cooler.",
            "preset_name": "Name used for the saved workspace JSON file and for the Env registry entry created from this builder preset.",
            "preset_select": "Choose a preset from the active workspace or from the shared Env registry.",
            "family_windscape": "Includes constant wind settings and optional AirPuff events.",
            "family_thermoscape": "Thermal source rows define hotspots or cold spots relative to the plate temperature.",
        }
        return notes.get(key)

    @staticmethod
    def _help_text_for_key(key: str) -> str | None:
        return _join_help_parts(
            _EnvironmentBuilderController._param_doc_for_key(key),
            _EnvironmentBuilderController._builder_note_for_key(key),
        )

    def _register_field(
        self,
        widget: pn.viewable.Viewable,
        key: str,
        *,
        label: str | None = None,
    ) -> None:
        help_text = self._help_text_for_key(key)
        if not hasattr(widget, "param") or "description" not in widget.param:
            return None
        widget.description = help_text

    def _field_view(self, widget: pn.viewable.Viewable) -> pn.viewable.Viewable:
        return widget

    def _set_field_visible(self, widget: pn.viewable.Viewable, visible: bool) -> None:
        if hasattr(widget, "visible"):
            widget.visible = visible

    def __init__(self) -> None:
        self._objects: list[_ObjectRow] = []
        self._border_start: tuple[float, float] | None = None
        self._selected_object_id: str | None = None
        self._syncing_selection = False
        self._counter = 1
        self._loaded_config = util.AttrDict()
        self._field_wrappers: dict[int, pn.viewable.Viewable] = {}

        self.arena_shape = pn.widgets.Select(
            name="Arena shape",
            value="rectangular",
            options=["rectangular", "circular"],
        )
        self.arena_width = pn.widgets.FloatSlider(
            name="Arena width (m)",
            start=0.05,
            end=0.50,
            step=0.01,
            value=0.20,
        )
        self.arena_height = pn.widgets.FloatSlider(
            name="Arena height (m)",
            start=0.05,
            end=0.50,
            step=0.01,
            value=0.20,
        )
        self.arena_torus = pn.widgets.Checkbox(name="Arena torus", value=False)
        self.object_type = pn.widgets.Select(
            name="Insert object",
            value="Source unit",
            options=["Source unit", "Source group", "Border segment"],
        )
        self.object_radius = pn.widgets.FloatSlider(
            name="Source unit radius (mm)",
            start=1.0,
            end=30.0,
            step=1.0,
            value=3.0,
            format="0.0",
        )
        self.border_width = pn.widgets.FloatSlider(
            name="Border width (mm)",
            start=0.5,
            end=10.0,
            step=0.5,
            value=1.0,
            format="0.0",
        )
        self.object_color = pn.widgets.ColorPicker(name="Object color", value="#4caf50")
        self.group_count = pn.widgets.IntSlider(
            name="Number of units",
            start=1,
            end=100,
            step=1,
            value=30,
        )
        self.group_shape = pn.widgets.Select(
            name="Group shape",
            value="circle",
            options=["circle", "oval", "rect"],
        )
        self.group_mode = pn.widgets.Select(
            name="Group mode",
            value="uniform",
            options=["uniform", "normal", "periphery", "grid"],
        )
        self.group_spread_x = pn.widgets.FloatSlider(
            name="Group spread X (mm)",
            start=0.0,
            end=100.0,
            step=1.0,
            value=12.0,
            format="0.0",
        )
        self.group_spread_y = pn.widgets.FloatSlider(
            name="Group spread Y (mm)",
            start=0.0,
            end=100.0,
            step=1.0,
            value=12.0,
            format="0.0",
        )
        self.select_mode = pn.widgets.Toggle(
            name="Select on canvas",
            value=False,
            button_type="primary",
        )
        self.selected_object = pn.widgets.Select(
            name="Inspect object",
            options={},
            value=None,
        )
        self.selected_id = pn.widgets.TextInput(name="Object ID", value="")
        self.selected_x = pn.widgets.FloatInput(name="X (m)", value=0.0, step=0.001)
        self.selected_y = pn.widgets.FloatInput(name="Y (m)", value=0.0, step=0.001)
        self.selected_x2 = pn.widgets.FloatInput(name="X2 (m)", value=0.0, step=0.001)
        self.selected_y2 = pn.widgets.FloatInput(name="Y2 (m)", value=0.0, step=0.001)
        self.selected_radius = pn.widgets.FloatSlider(
            name="Source unit radius (mm)",
            start=1.0,
            end=30.0,
            step=0.5,
            value=3.0,
        )
        self.selected_width = pn.widgets.FloatInput(
            name="Border width (mm)", value=1.0, step=0.5
        )
        self.selected_color = pn.widgets.ColorPicker(name="Color", value="#4caf50")
        self.selected_amount = pn.widgets.FloatInput(
            name="Food amount", value=0.0, step=0.5
        )
        self.selected_can_be_carried = pn.widgets.Checkbox(
            name="Can be carried",
            value=False,
        )
        self.selected_can_be_displaced = pn.widgets.Checkbox(
            name="Can be displaced",
            value=False,
        )
        self.selected_regeneration = pn.widgets.Checkbox(
            name="Regeneration",
            value=False,
        )
        self.selected_odor_id = pn.widgets.AutocompleteInput(
            name="Odor ID",
            value="",
            options=[],
            case_sensitive=False,
            restrict=False,
            placeholder="Existing odor ID or new custom value",
        )
        self.selected_odor_intensity = pn.widgets.FloatInput(
            name="Odor intensity",
            value=1.0,
            step=0.1,
        )
        self.selected_odor_spread = pn.widgets.FloatInput(
            name="Odor spread",
            value=0.02,
            step=0.005,
        )
        self.selected_distribution_n = pn.widgets.IntInput(
            name="Group count", value=30, step=1
        )
        self.selected_distribution_shape = pn.widgets.Select(
            name="Group shape",
            value="circle",
            options=["circle", "oval", "rect"],
        )
        self.selected_distribution_mode = pn.widgets.Select(
            name="Group mode",
            value="uniform",
            options=["uniform", "normal", "periphery", "grid"],
        )
        self.selected_distribution_show_shape = pn.widgets.Checkbox(
            name="Show group shape",
            value=True,
        )
        self.selected_distribution_scale_x = pn.widgets.FloatSlider(
            name="Group spread X (mm)",
            start=0.0,
            end=100.0,
            step=1.0,
            value=12.0,
            format="0.0",
        )
        self.selected_distribution_scale_y = pn.widgets.FloatSlider(
            name="Group spread Y (mm)",
            start=0.0,
            end=100.0,
            step=1.0,
            value=12.0,
            format="0.0",
        )
        self.selected_substrate_type = pn.widgets.Select(
            name="Substrate type",
            value="standard",
            options=SUBSTRATE_TYPE_OPTIONS,
        )
        self.selected_substrate_quality = pn.widgets.FloatInput(
            name="Substrate quality",
            value=1.0,
            step=0.1,
        )
        self.food_grid_enabled = pn.widgets.Checkbox(
            name="Enable food grid", value=False
        )
        self.food_grid_color = pn.widgets.ColorPicker(
            name="Food grid color", value="#ffd1f3"
        )
        self.food_grid_dims_x = pn.widgets.IntInput(
            name="Grid cells X", value=51, step=1
        )
        self.food_grid_dims_y = pn.widgets.IntInput(
            name="Grid cells Y", value=51, step=1
        )
        self.food_grid_initial_value = pn.widgets.FloatInput(
            name="Grid initial value",
            value=1e-6,
            step=1e-6,
        )
        self.food_grid_fixed_max = pn.widgets.Checkbox(
            name="Grid fixed max",
            value=True,
        )
        self.food_grid_substrate_type = pn.widgets.Select(
            name="Grid substrate type",
            value="standard",
            options=SUBSTRATE_TYPE_OPTIONS,
        )
        self.food_grid_substrate_quality = pn.widgets.FloatInput(
            name="Grid substrate quality",
            value=1.0,
            step=0.1,
        )
        self.odorscape_enabled = pn.widgets.Checkbox(
            name="Enable odorscape", value=False
        )
        self.odorscape_mode = pn.widgets.Select(
            name="Odorscape mode",
            value="Gaussian",
            options=["Gaussian", "Diffusion", "Analytical"],
        )
        self.odorscape_color = pn.widgets.ColorPicker(
            name="Odorscape color",
            value="#4caf50",
        )
        self.odorscape_grid_dims_x = pn.widgets.IntInput(
            name="Odorscape grid X", value=51, step=1
        )
        self.odorscape_grid_dims_y = pn.widgets.IntInput(
            name="Odorscape grid Y", value=51, step=1
        )
        self.odorscape_initial_value = pn.widgets.FloatInput(
            name="Odorscape initial value",
            value=0.0,
            step=1e-6,
        )
        self.odorscape_fixed_max = pn.widgets.Checkbox(
            name="Odorscape fixed max", value=False
        )
        self.odorscape_evap_const = pn.widgets.FloatInput(
            name="Diffusion evaporation",
            value=0.9,
            step=0.01,
        )
        self.odorscape_sigma_x = pn.widgets.FloatInput(
            name="Diffusion sigma X",
            value=0.95,
            step=0.05,
        )
        self.odorscape_sigma_y = pn.widgets.FloatInput(
            name="Diffusion sigma Y",
            value=0.95,
            step=0.05,
        )
        self.windscape_enabled = pn.widgets.Checkbox(
            name="Enable windscape", value=False
        )
        self.windscape_color = pn.widgets.ColorPicker(
            name="Wind color",
            value="#ff0000",
        )
        self.windscape_direction = pn.widgets.FloatInput(
            name="Wind direction (deg)",
            value=180.0,
            step=5.0,
        )
        self.windscape_speed = pn.widgets.FloatInput(
            name="Wind speed (m/s)",
            value=0.0,
            step=0.5,
        )
        self.wind_puffs_table = pn.widgets.Tabulator(
            _table_dataframe([], _WIND_PUFF_COLUMNS),
            show_index=False,
            selectable=1,
            editors={
                "id": None,
                "duration": {"type": "number", "step": 0.1, "min": 0},
                "speed": {"type": "number", "step": 0.1, "min": 0},
                "direction": {"type": "number", "step": 0.1},
                "start_time": {"type": "number", "step": 0.1, "min": 0},
                "N": {"type": "number", "step": 1, "min": 1},
                "interval": {"type": "number", "step": 0.1, "min": 0},
            },
            height=180,
            sizing_mode="stretch_width",
        )
        self.add_wind_puff_btn = pn.widgets.Button(
            name="Add air puff",
            button_type="default",
        )
        self.remove_wind_puff_btn = pn.widgets.Button(
            name="Remove selected air puff",
            button_type="warning",
        )
        self.thermoscape_enabled = pn.widgets.Checkbox(
            name="Enable thermoscape", value=False
        )
        self.thermoscape_plate_temp = pn.widgets.FloatInput(
            name="Plate temperature (deg C)",
            value=22.0,
            step=0.5,
        )
        self.thermoscape_spread = pn.widgets.FloatInput(
            name="Thermal spread",
            value=0.1,
            step=0.01,
        )
        self.thermo_sources_table = pn.widgets.Tabulator(
            _table_dataframe([], _THERMO_SOURCE_COLUMNS),
            show_index=False,
            selectable=1,
            editors={
                "id": None,
                "x": {"type": "number", "step": 0.001},
                "y": {"type": "number", "step": 0.001},
                "dTemp": {"type": "number", "step": 0.1},
            },
            height=180,
            sizing_mode="stretch_width",
        )
        self.add_thermo_source_btn = pn.widgets.Button(
            name="Add thermal source",
            button_type="default",
        )
        self.remove_thermo_source_btn = pn.widgets.Button(
            name="Remove selected thermal source",
            button_type="warning",
        )
        self.apply_selected_btn = pn.widgets.Button(
            name="Apply changes",
            button_type="primary",
        )
        self.delete_selected_btn = pn.widgets.Button(
            name="Delete selected",
            button_type="warning",
        )
        for key, widget in (
            ("select_mode", self.select_mode),
            ("arena_shape", self.arena_shape),
            ("arena_width", self.arena_width),
            ("arena_height", self.arena_height),
            ("arena_torus", self.arena_torus),
            ("object_type", self.object_type),
            ("object_radius", self.object_radius),
            ("group_count", self.group_count),
            ("group_shape", self.group_shape),
            ("group_mode", self.group_mode),
            ("group_spread_x", self.group_spread_x),
            ("group_spread_y", self.group_spread_y),
            ("border_width", self.border_width),
            ("object_color", self.object_color),
            ("food_grid_enabled", self.food_grid_enabled),
            ("food_grid_color", self.food_grid_color),
            ("food_grid_dims_x", self.food_grid_dims_x),
            ("food_grid_dims_y", self.food_grid_dims_y),
            ("food_grid_initial_value", self.food_grid_initial_value),
            ("food_grid_fixed_max", self.food_grid_fixed_max),
            ("food_grid_substrate_type", self.food_grid_substrate_type),
            ("food_grid_substrate_quality", self.food_grid_substrate_quality),
            ("odorscape_enabled", self.odorscape_enabled),
            ("odorscape_mode", self.odorscape_mode),
            ("odorscape_color", self.odorscape_color),
            ("odorscape_grid_dims_x", self.odorscape_grid_dims_x),
            ("odorscape_grid_dims_y", self.odorscape_grid_dims_y),
            ("odorscape_initial_value", self.odorscape_initial_value),
            ("odorscape_fixed_max", self.odorscape_fixed_max),
            ("odorscape_evap_const", self.odorscape_evap_const),
            ("odorscape_sigma_x", self.odorscape_sigma_x),
            ("odorscape_sigma_y", self.odorscape_sigma_y),
            ("windscape_enabled", self.windscape_enabled),
            ("windscape_color", self.windscape_color),
            ("windscape_direction", self.windscape_direction),
            ("windscape_speed", self.windscape_speed),
            ("wind_puffs_table", self.wind_puffs_table),
            ("thermoscape_enabled", self.thermoscape_enabled),
            ("thermoscape_plate_temp", self.thermoscape_plate_temp),
            ("thermoscape_spread", self.thermoscape_spread),
            ("thermo_sources_table", self.thermo_sources_table),
            ("selected_object", self.selected_object),
            ("selected_id", self.selected_id),
            ("selected_x", self.selected_x),
            ("selected_y", self.selected_y),
            ("selected_x2", self.selected_x2),
            ("selected_y2", self.selected_y2),
            ("selected_radius", self.selected_radius),
            ("selected_width", self.selected_width),
            ("selected_color", self.selected_color),
            ("selected_amount", self.selected_amount),
            ("selected_can_be_carried", self.selected_can_be_carried),
            ("selected_can_be_displaced", self.selected_can_be_displaced),
            ("selected_regeneration", self.selected_regeneration),
            ("selected_odor_id", self.selected_odor_id),
            ("selected_odor_intensity", self.selected_odor_intensity),
            ("selected_odor_spread", self.selected_odor_spread),
            ("selected_distribution_n", self.selected_distribution_n),
            ("selected_distribution_shape", self.selected_distribution_shape),
            ("selected_distribution_mode", self.selected_distribution_mode),
            ("selected_distribution_show_shape", self.selected_distribution_show_shape),
            ("selected_distribution_scale_x", self.selected_distribution_scale_x),
            ("selected_distribution_scale_y", self.selected_distribution_scale_y),
            ("selected_substrate_type", self.selected_substrate_type),
            ("selected_substrate_quality", self.selected_substrate_quality),
        ):
            label = None
            if key == "wind_puffs_table":
                label = "Air puffs"
            elif key == "thermo_sources_table":
                label = "Thermal sources"
            self._register_field(widget, key, label=label)
        self._editor_food_family = _editor_family_box(
            "Food",
            self._field_view(self.selected_amount),
            help_text=self._help_text_for_key("family_food"),
        )
        self._editor_source_behavior_family = _editor_family_box(
            "Source behavior",
            self._field_view(self.selected_can_be_carried),
            self._field_view(self.selected_can_be_displaced),
            self._field_view(self.selected_regeneration),
            help_text=self._help_text_for_key("family_source_behavior"),
        )
        self._editor_substrate_family = _editor_family_box(
            "Substrate",
            self._field_view(self.selected_substrate_type),
            self._field_view(self.selected_substrate_quality),
            help_text=self._help_text_for_key("family_substrate"),
        )
        self._editor_odor_family = _editor_family_box(
            "Odor",
            self._field_view(self.selected_odor_id),
            self._field_view(self.selected_odor_intensity),
            self._field_view(self.selected_odor_spread),
            help_text=self._help_text_for_key("family_odor"),
        )
        self._editor_distribution_family = _editor_family_box(
            "Distribution",
            self._field_view(self.selected_distribution_n),
            self._field_view(self.selected_distribution_shape),
            self._field_view(self.selected_distribution_mode),
            self._field_view(self.selected_distribution_show_shape),
            self._field_view(self.selected_distribution_scale_x),
            self._field_view(self.selected_distribution_scale_y),
            help_text=self._help_text_for_key("family_distribution"),
        )
        self._odorscape_family = _editor_family_box(
            "Odorscape",
            self._field_view(self.odorscape_enabled),
            self._field_view(self.odorscape_mode),
            self._field_view(self.odorscape_color),
            self._field_view(self.odorscape_grid_dims_x),
            self._field_view(self.odorscape_grid_dims_y),
            self._field_view(self.odorscape_initial_value),
            self._field_view(self.odorscape_fixed_max),
            self._field_view(self.odorscape_evap_const),
            self._field_view(self.odorscape_sigma_x),
            self._field_view(self.odorscape_sigma_y),
            help_text=self._help_text_for_key("family_odorscape"),
        )
        self._odorscape_family.css_classes = [
            *self._odorscape_family.css_classes,
            "lw-env-builder-scape-family",
        ]
        self._windscape_family = _editor_family_box(
            "Windscape",
            self._field_view(self.windscape_enabled),
            self._field_view(self.windscape_color),
            self._field_view(self.windscape_direction),
            self._field_view(self.windscape_speed),
            self._field_view(self.wind_puffs_table),
            pn.Column(
                self.add_wind_puff_btn,
                self.remove_wind_puff_btn,
                sizing_mode="stretch_width",
                margin=0,
            ),
            help_text=self._help_text_for_key("family_windscape"),
        )
        self._windscape_family.css_classes = [
            *self._windscape_family.css_classes,
            "lw-env-builder-scape-family",
        ]
        self._thermoscape_family = _editor_family_box(
            "Thermoscape",
            self._field_view(self.thermoscape_enabled),
            self._field_view(self.thermoscape_plate_temp),
            self._field_view(self.thermoscape_spread),
            self._field_view(self.thermo_sources_table),
            pn.Column(
                self.add_thermo_source_btn,
                self.remove_thermo_source_btn,
                sizing_mode="stretch_width",
                margin=0,
            ),
            help_text=self._help_text_for_key("family_thermoscape"),
        )
        self._thermoscape_family.css_classes = [
            *self._thermoscape_family.css_classes,
            "lw-env-builder-scape-family",
        ]
        self.preset_name = pn.widgets.TextInput(
            name="Preset name",
            value="environment_builder_config",
            placeholder="environment_builder_config",
        )
        self.preset_select = pn.widgets.Select(
            name="Saved presets",
            options={},
            value=None,
        )
        self.save_preset_btn = pn.widgets.Button(
            name="Save",
            button_type="primary",
        )
        self.load_preset_btn = pn.widgets.Button(
            name="Load",
            button_type="default",
        )
        self.delete_preset_btn = pn.widgets.Button(
            name="Delete",
            button_type="warning",
        )
        self.load_file_btn = pn.widgets.Button(
            name="Load file",
            button_type="default",
        )
        self.download_file_btn = pn.widgets.Button(
            name="Download file",
            button_type="default",
        )
        self.load_file_input = pn.widgets.FileInput(
            name="",
            accept=".json,application/json",
            multiple=False,
        )
        self.load_file_input.css_classes = ["lw-env-builder-hidden-file-input"]
        self.refresh_presets_btn = pn.widgets.Button(
            name="Refresh list",
            button_type="default",
        )
        self.clear_last_btn = pn.widgets.Button(name="Undo last", button_type="default")
        self.clear_all_btn = pn.widgets.Button(
            name="Clear canvas", button_type="danger"
        )
        self.clear_arena_btn = pn.widgets.Button(
            name="Clear arena", button_type="warning"
        )
        self.export_btn = pn.widgets.FileDownload(
            name="",
            label="Download file",
            button_type="default",
            callback=self._export_json,
            filename="environment_builder_config.json",
        )
        self.export_btn.css_classes = ["lw-env-builder-hidden-download-proxy"]
        self._register_field(self.preset_name, "preset_name")
        self._register_field(self.preset_select, "preset_select")
        self.preset_meta = pn.pane.HTML(
            "",
            sizing_mode="stretch_width",
            margin=(0, 0, 4, 0),
            styles={
                "font-size": "11px",
                "line-height": "1.45",
                "color": "rgba(17, 17, 17, 0.72)",
                "padding": "8px 10px",
                "text-align": "left",
                "background": "rgba(252, 252, 253, 0.99)",
                "border": "1px solid rgba(90, 71, 96, 0.10)",
                "border-radius": "8px",
                "overflow-wrap": "anywhere",
            },
        )
        self._pending_overwrite_id: str | None = None
        self._suspend_arena_update = False
        self._last_valid_arena_controls = {
            "shape": str(self.arena_shape.value),
            "width": float(self.arena_width.value),
            "height": float(self.arena_height.value),
        }
        self.preset_overwrite_text = pn.pane.Markdown(
            "",
            sizing_mode="stretch_width",
            margin=0,
            styles={
                "font-size": "12px",
                "line-height": "1.4",
                "padding": "8px 10px",
                "background": "rgba(255, 244, 214, 0.95)",
                "border": "1px solid rgba(180, 120, 0, 0.22)",
                "border-radius": "8px",
                "color": "rgba(65, 45, 0, 0.92)",
            },
        )
        self.confirm_overwrite_btn = pn.widgets.Button(
            name="Yes, overwrite",
            button_type="warning",
            sizing_mode="stretch_width",
        )
        self.cancel_overwrite_btn = pn.widgets.Button(
            name="No, cancel",
            button_type="default",
            sizing_mode="stretch_width",
        )
        self.preset_overwrite_confirm = pn.Column(
            self.preset_overwrite_text,
            pn.Row(
                self.confirm_overwrite_btn,
                self.cancel_overwrite_btn,
                sizing_mode="stretch_width",
                margin=(6, 0, 0, 0),
            ),
            sizing_mode="stretch_width",
            margin=(4, 0, 0, 0),
            visible=False,
        )
        self._pending_reset_confirmation = False
        self.reset_confirm_text = pn.pane.Markdown(
            "",
            sizing_mode="stretch_width",
            margin=0,
            styles={
                "font-size": "12px",
                "line-height": "1.4",
                "padding": "8px 10px",
                "background": "rgba(255, 235, 235, 0.96)",
                "border": "1px solid rgba(170, 40, 40, 0.28)",
                "border-radius": "8px",
                "color": "rgba(95, 20, 20, 0.95)",
            },
        )
        self.confirm_reset_btn = pn.widgets.Button(
            name="Yes, reset",
            button_type="danger",
            sizing_mode="stretch_width",
        )
        self.cancel_reset_btn = pn.widgets.Button(
            name="No, cancel",
            button_type="default",
            sizing_mode="stretch_width",
        )
        self.reset_confirm_panel = pn.Column(
            self.reset_confirm_text,
            pn.Row(
                self.confirm_reset_btn,
                self.cancel_reset_btn,
                sizing_mode="stretch_width",
                margin=(6, 0, 0, 0),
            ),
            sizing_mode="stretch_width",
            margin=(4, 0, 0, 0),
            visible=False,
        )
        self.clear_last_btn.width = None
        self.clear_last_btn.sizing_mode = "stretch_width"
        self.preset_name.width = None
        self.preset_name.sizing_mode = "stretch_width"
        self.preset_select.width = None
        self.preset_select.sizing_mode = "stretch_width"
        for widget in (
            self.arena_shape,
            self.arena_width,
            self.arena_height,
            self.arena_torus,
            self.object_type,
            self.object_radius,
            self.group_count,
            self.group_shape,
            self.group_mode,
            self.group_spread_x,
            self.group_spread_y,
            self.border_width,
            self.object_color,
            self.food_grid_enabled,
            self.food_grid_color,
            self.food_grid_dims_x,
            self.food_grid_dims_y,
            self.food_grid_initial_value,
            self.food_grid_fixed_max,
            self.food_grid_substrate_type,
            self.food_grid_substrate_quality,
        ):
            widget.width = None
            widget.sizing_mode = "stretch_width"
        self.save_preset_btn.width = None
        self.save_preset_btn.sizing_mode = "stretch_width"
        self.load_preset_btn.width = None
        self.load_preset_btn.sizing_mode = "stretch_width"
        self.delete_preset_btn.width = None
        self.delete_preset_btn.sizing_mode = "stretch_width"
        self.load_file_btn.width = None
        self.load_file_btn.sizing_mode = "stretch_width"
        self.download_file_btn.width = None
        self.download_file_btn.sizing_mode = "stretch_width"
        self.load_file_input.width = None
        self.load_file_input.sizing_mode = "stretch_width"
        self.load_file_input.styles = {
            "position": "absolute",
            "left": "-10000px",
            "top": "auto",
            "width": "1px",
            "height": "1px",
            "opacity": "0",
            "overflow": "hidden",
            "pointer-events": "none",
        }
        self.apply_selected_btn.width = None
        self.apply_selected_btn.sizing_mode = "stretch_width"
        self.delete_selected_btn.width = None
        self.delete_selected_btn.sizing_mode = "stretch_width"
        for widget in (
            self.selected_object,
            self.selected_id,
            self.selected_x,
            self.selected_y,
            self.selected_x2,
            self.selected_y2,
            self.selected_radius,
            self.selected_width,
            self.selected_color,
            self.selected_amount,
            self.selected_can_be_carried,
            self.selected_can_be_displaced,
            self.selected_regeneration,
            self.selected_odor_id,
            self.selected_odor_intensity,
            self.selected_odor_spread,
            self.selected_distribution_n,
            self.selected_distribution_shape,
            self.selected_distribution_mode,
            self.selected_distribution_show_shape,
            self.selected_distribution_scale_x,
            self.selected_distribution_scale_y,
            self.selected_substrate_type,
            self.selected_substrate_quality,
        ):
            widget.width = None
            widget.sizing_mode = "stretch_width"
        for widget in (
            self.odorscape_enabled,
            self.odorscape_mode,
            self.odorscape_color,
            self.odorscape_grid_dims_x,
            self.odorscape_grid_dims_y,
            self.odorscape_initial_value,
            self.odorscape_fixed_max,
            self.odorscape_evap_const,
            self.odorscape_sigma_x,
            self.odorscape_sigma_y,
            self.windscape_enabled,
            self.windscape_color,
            self.windscape_direction,
            self.windscape_speed,
            self.thermoscape_enabled,
            self.thermoscape_plate_temp,
            self.thermoscape_spread,
        ):
            widget.width = None
            widget.sizing_mode = "stretch_width"
        self.add_wind_puff_btn.width = None
        self.add_wind_puff_btn.sizing_mode = "stretch_width"
        self.remove_wind_puff_btn.width = None
        self.remove_wind_puff_btn.sizing_mode = "stretch_width"
        self.add_thermo_source_btn.width = None
        self.add_thermo_source_btn.sizing_mode = "stretch_width"
        self.remove_thermo_source_btn.width = None
        self.remove_thermo_source_btn.sizing_mode = "stretch_width"
        self.clear_all_btn.width = None
        self.clear_all_btn.sizing_mode = "stretch_width"
        self.clear_arena_btn.width = None
        self.clear_arena_btn.sizing_mode = "stretch_width"
        self.export_btn.width = None
        self.export_btn.sizing_mode = "stretch_width"
        self.export_btn.styles = {
            "position": "absolute",
            "left": "-10000px",
            "top": "auto",
            "width": "1px",
            "height": "1px",
            "opacity": "0",
            "overflow": "hidden",
            "pointer-events": "none",
        }
        self.refresh_presets_btn.width = None
        self.refresh_presets_btn.sizing_mode = "stretch_width"
        self.status = pn.pane.Markdown("Click on the canvas to place an object.")
        self._table_columns = [
            "id",
            "type",
            "x",
            "y",
            "x2",
            "y2",
            "radius",
            "spread_x",
            "spread_y",
            "count",
            "width",
            "color",
            "amount",
            "odor_id",
        ]
        self.table = pn.widgets.Tabulator(
            pd.DataFrame(columns=self._table_columns),
            show_index=False,
            selectable=1,
            editors={column: None for column in self._table_columns},
            height=620,
            sizing_mode="stretch_width",
            css_classes=["lw-env-builder-objects-table"],
        )

        self.food_source = ColumnDataSource(
            {
                "x": [],
                "y": [],
                "r": [],
                "fill_color": [],
                "line_color": [],
                "id": [],
                "fill_alpha": [],
                "line_alpha": [],
                "line_width": [],
            }
        )
        self.odor_layer_source = ColumnDataSource(
            {"x": [], "y": [], "r": [], "color": [], "fill_alpha": [], "id": []}
        )
        self.odor_peak_source = ColumnDataSource(
            {"x": [], "y": [], "r": [], "color": [], "fill_alpha": [], "id": []}
        )
        self.food_highlight_source = ColumnDataSource(
            {"x": [], "y": [], "r": [], "color": []}
        )
        self.odorscape_contour_source = ColumnDataSource(
            {
                "x": [],
                "y": [],
                "r": [],
                "color": [],
                "line_alpha": [],
                "line_width": [],
                "id": [],
            }
        )
        self.source_group_circle_source = ColumnDataSource(
            self._empty_source_group_circle_data()
        )
        self.source_group_ellipse_source = ColumnDataSource(
            self._empty_source_group_xy_data()
        )
        self.source_group_rect_source = ColumnDataSource(
            self._empty_source_group_xy_data()
        )
        self.source_group_member_source = ColumnDataSource(
            self._empty_source_group_member_data()
        )
        self.source_group_circle_highlight_source = ColumnDataSource(
            self._empty_source_group_circle_highlight_data()
        )
        self.source_group_ellipse_highlight_source = ColumnDataSource(
            self._empty_source_group_xy_highlight_data()
        )
        self.source_group_rect_highlight_source = ColumnDataSource(
            self._empty_source_group_xy_highlight_data()
        )
        self.border_source = ColumnDataSource(
            {"x0": [], "y0": [], "x1": [], "y1": [], "w": [], "color": [], "id": []}
        )
        self.border_highlight_source = ColumnDataSource(
            {"x0": [], "y0": [], "x1": [], "y1": [], "w": [], "color": []}
        )
        self.border_preview_source = ColumnDataSource({"x": [], "y": [], "color": []})
        self.windscape_segment_source = ColumnDataSource(
            {"x0": [], "y0": [], "x1": [], "y1": [], "color": [], "line_alpha": []}
        )
        self.windscape_head_source = ColumnDataSource(
            {"x": [], "y": [], "angle": [], "color": [], "size": []}
        )
        self.thermoscape_aura_source = ColumnDataSource(
            {
                "x": [],
                "y": [],
                "r": [],
                "color": [],
                "fill_alpha": [],
                "line_alpha": [],
                "id": [],
            }
        )
        self.thermoscape_marker_source = ColumnDataSource(
            {
                "x": [],
                "y": [],
                "color": [],
                "size": [],
                "id": [],
            }
        )
        self._food_grid_overlay_source = ColumnDataSource(
            {
                "x": [0.0],
                "y": [0.0],
                "w": [self.arena_width.value],
                "h": [self.arena_height.value],
                "color": [self.food_grid_color.value],
                "fill_alpha": [0.0],
            }
        )
        self._food_grid_cell_source = ColumnDataSource(
            self._empty_food_grid_cell_data()
        )
        self._arena_source = ColumnDataSource(
            {
                "x": [0.0],
                "y": [0.0],
                "w": [self.arena_width.value],
                "h": [self.arena_height.value],
            }
        )

        self.fig = figure(
            title="Environment canvas",
            x_range=(-ENV_CANVAS_X_HALF_RANGE, ENV_CANVAS_X_HALF_RANGE),
            y_range=(-ENV_CANVAS_Y_HALF_RANGE, ENV_CANVAS_Y_HALF_RANGE),
            match_aspect=True,
            width=ENV_CANVAS_WIDTH,
            height=ENV_CANVAS_HEIGHT,
            tools="pan,wheel_zoom,reset,save",
            active_scroll="wheel_zoom",
            toolbar_location="right",
        )
        wheel_zoom = self.fig.select_one(WheelZoomTool)
        if wheel_zoom is not None:
            wheel_zoom.dimensions = "both"
            wheel_zoom.zoom_on_axis = False
        self.fig.background_fill_color = "#ffffff"
        self.fig.border_fill_color = "#fafafa"
        self.fig.xaxis.axis_label = "X (m)"
        self.fig.yaxis.axis_label = "Y (m)"

        self._arena_rect_renderer = self.fig.rect(
            x="x",
            y="y",
            width="w",
            height="h",
            source=self._arena_source,
            line_color=LANE_MODELS_COLOR_DARK,
            line_width=3,
            fill_alpha=0.0,
            visible=True,
        )
        self._food_grid_rect_renderer = self.fig.rect(
            x="x",
            y="y",
            width="w",
            height="h",
            source=self._food_grid_overlay_source,
            line_color=None,
            fill_color="color",
            fill_alpha="fill_alpha",
            visible=True,
        )
        self._arena_circle_renderer = self.fig.ellipse(
            x="x",
            y="y",
            width="w",
            height="h",
            source=self._arena_source,
            line_color=LANE_MODELS_COLOR_DARK,
            line_width=3,
            fill_alpha=0.0,
            visible=False,
        )
        self._food_grid_circle_renderer = self.fig.ellipse(
            x="x",
            y="y",
            width="w",
            height="h",
            source=self._food_grid_overlay_source,
            line_color=None,
            fill_color="color",
            fill_alpha="fill_alpha",
            visible=False,
        )
        self._food_grid_cell_renderer = self.fig.rect(
            x="x",
            y="y",
            width="w",
            height="h",
            source=self._food_grid_cell_source,
            line_color="line_color",
            line_alpha="line_alpha",
            line_width="line_width",
            fill_color="fill_color",
            fill_alpha="fill_alpha",
            visible=False,
        )
        self.fig.circle(
            x="x",
            y="y",
            radius="r",
            source=self.thermoscape_aura_source,
            line_color="color",
            line_alpha="line_alpha",
            fill_color="color",
            fill_alpha="fill_alpha",
            line_width=1,
            legend_label="Thermoscape",
        )
        self.fig.segment(
            x0="x0",
            y0="y0",
            x1="x1",
            y1="y1",
            source=self.windscape_segment_source,
            line_color="color",
            line_alpha="line_alpha",
            line_width=2,
            legend_label="Windscape",
        )
        self.fig.scatter(
            x="x",
            y="y",
            source=self.windscape_head_source,
            marker="triangle",
            angle="angle",
            size="size",
            line_color="color",
            fill_color="color",
            fill_alpha=0.75,
            line_alpha=0.85,
        )
        self.fig.circle(
            x="x",
            y="y",
            radius="r",
            source=self.odorscape_contour_source,
            line_color="color",
            line_alpha="line_alpha",
            line_width="line_width",
            fill_alpha=0.0,
            legend_label="Odorscape",
        )
        self.fig.scatter(
            x="x",
            y="y",
            source=self.thermoscape_marker_source,
            marker="diamond",
            size="size",
            line_color="color",
            fill_color="color",
            fill_alpha=0.9,
            line_alpha=0.95,
        )
        self.fig.circle(
            x="x",
            y="y",
            radius="r",
            source=self.odor_layer_source,
            line_color=None,
            fill_color="color",
            fill_alpha="fill_alpha",
            legend_label="Odor aura",
        )
        self.fig.circle(
            x="x",
            y="y",
            radius="r",
            source=self.food_source,
            line_color="line_color",
            fill_color="fill_color",
            fill_alpha="fill_alpha",
            line_alpha="line_alpha",
            line_width="line_width",
            legend_label="Source units",
        )
        self.fig.circle(
            x="x",
            y="y",
            radius="r",
            source=self.odor_peak_source,
            line_color=None,
            fill_color="color",
            fill_alpha="fill_alpha",
        )
        self.fig.circle(
            x="x",
            y="y",
            radius="r",
            source=self.food_highlight_source,
            line_color="#f97316",
            fill_color=None,
            line_width=4,
        )
        self.fig.circle(
            x="x",
            y="y",
            radius="r",
            source=self.source_group_circle_source,
            line_color="color",
            fill_color="color",
            fill_alpha="fill_alpha",
            line_alpha="line_alpha",
            line_width=2,
            legend_label="Source groups",
        )
        self.fig.ellipse(
            x="x",
            y="y",
            width="w",
            height="h",
            source=self.source_group_ellipse_source,
            line_color="color",
            fill_color="color",
            fill_alpha="fill_alpha",
            line_alpha="line_alpha",
            line_width=2,
        )
        self.fig.rect(
            x="x",
            y="y",
            width="w",
            height="h",
            source=self.source_group_rect_source,
            line_color="color",
            fill_color="color",
            fill_alpha="fill_alpha",
            line_alpha="line_alpha",
            line_width=2,
        )
        self.fig.circle(
            x="x",
            y="y",
            radius="r",
            source=self.source_group_member_source,
            line_color="line_color",
            fill_color="fill_color",
            fill_alpha="fill_alpha",
            line_alpha="line_alpha",
            line_width="line_width",
        )
        self.fig.circle(
            x="x",
            y="y",
            radius="r",
            source=self.source_group_circle_highlight_source,
            line_color="#f97316",
            fill_color=None,
            line_width=4,
        )
        self.fig.ellipse(
            x="x",
            y="y",
            width="w",
            height="h",
            source=self.source_group_ellipse_highlight_source,
            line_color="#f97316",
            fill_color=None,
            line_width=4,
        )
        self.fig.rect(
            x="x",
            y="y",
            width="w",
            height="h",
            source=self.source_group_rect_highlight_source,
            line_color="#f97316",
            fill_color=None,
            line_width=4,
        )
        self.fig.segment(
            x0="x0",
            y0="y0",
            x1="x1",
            y1="y1",
            source=self.border_source,
            line_color="color",
            line_width="w",
            legend_label="Borders",
        )
        self.fig.segment(
            x0="x0",
            y0="y0",
            x1="x1",
            y1="y1",
            source=self.border_highlight_source,
            line_color="#f97316",
            line_width="w",
        )
        self.fig.scatter(
            x="x",
            y="y",
            source=self.border_preview_source,
            marker="cross",
            size=16,
            line_width=3,
            line_color="color",
        )
        legend_order = [
            "Source units",
            "Source groups",
            "Borders",
            "Odor aura",
            "Odorscape",
            "Windscape",
            "Thermoscape",
        ]
        if self.fig.legend:
            items_by_label = {}
            for item in self.fig.legend[0].items:
                label = getattr(item.label, "value", None)
                if isinstance(label, str):
                    items_by_label[label] = item
            self.fig.legend[0].items = [
                items_by_label[label]
                for label in legend_order
                if label in items_by_label
            ]
        self.fig.legend.location = "top_left"
        self.fig.legend.background_fill_alpha = 0.85

        self.fig.on_event(Tap, self._on_tap)
        self.arena_shape.param.watch(self._on_arena_shape_change, "value")
        self.arena_width.param.watch(self._update_arena, "value")
        self.arena_height.param.watch(self._update_arena, "value")
        self.object_type.param.watch(self._update_insert_hint, "value")
        self.group_shape.param.watch(self._sync_group_shape_controls, "value")
        self.group_spread_x.param.watch(self._on_group_spread_x_change, "value")
        self.food_grid_enabled.param.watch(self._sync_food_grid_overlay, "value")
        self.food_grid_color.param.watch(self._sync_food_grid_overlay, "value")
        self.food_grid_dims_x.param.watch(self._sync_food_grid_overlay, "value")
        self.food_grid_dims_y.param.watch(self._sync_food_grid_overlay, "value")
        self.odorscape_mode.param.watch(self._sync_odorscape_controls, "value")
        for widget in [
            self.odorscape_enabled,
            self.odorscape_mode,
            self.odorscape_color,
            self.odorscape_grid_dims_x,
            self.odorscape_grid_dims_y,
            self.odorscape_initial_value,
            self.odorscape_fixed_max,
            self.odorscape_evap_const,
            self.odorscape_sigma_x,
            self.odorscape_sigma_y,
            self.windscape_enabled,
            self.windscape_color,
            self.windscape_direction,
            self.windscape_speed,
            self.thermoscape_enabled,
            self.thermoscape_plate_temp,
            self.thermoscape_spread,
        ]:
            widget.param.watch(self._sync_scape_preview, "value")
        self.wind_puffs_table.param.watch(self._sync_scape_preview, "value")
        self.thermo_sources_table.param.watch(self._sync_scape_preview, "value")
        self.select_mode.param.watch(self._on_select_mode_change, "value")
        self.selected_object.param.watch(self._on_selected_object_change, "value")
        self.selected_distribution_shape.param.watch(
            self._sync_selected_group_shape_controls, "value"
        )
        self.selected_distribution_scale_x.param.watch(
            self._on_selected_distribution_scale_x_change, "value"
        )
        self.table.param.watch(self._on_table_selection_change, "selection")
        self.apply_selected_btn.on_click(self._on_apply_selected_object)
        self.delete_selected_btn.on_click(self._on_delete_selected_object)
        self.save_preset_btn.on_click(self._on_save_preset)
        self.load_preset_btn.on_click(self._on_load_preset)
        self.delete_preset_btn.on_click(self._on_delete_preset)
        self.load_file_btn.js_on_click(
            args={"load_input": self.load_file_input},
            code="""
                const picker = document.createElement('input');
                picker.type = 'file';
                picker.accept = '.json,application/json';
                picker.multiple = false;
                picker.onchange = async () => {
                    const file = picker.files && picker.files[0];
                    if (!file) {
                        return;
                    }
                    const dataUrl = await new Promise((resolve, reject) => {
                        const reader = new FileReader();
                        reader.onload = () => resolve(reader.result || '');
                        reader.onerror = () => reject(reader.error || new Error(`unable to read '${file.name}'`));
                        reader.readAsDataURL(file);
                    });
                    const [, mime_type = '', , value = ''] = String(dataUrl).split(/[:;,]/, 4);
                    load_input.setv({value, filename: file.name, mime_type});
                };
                picker.click();
            """,
        )
        self.download_file_btn.js_on_click(
            args={"download_proxy": self.export_btn},
            code="""
                download_proxy.setv({clicks: download_proxy.clicks + 1});
            """,
        )
        self.load_file_input.param.watch(self._on_load_file, "value")
        self.refresh_presets_btn.on_click(self._on_refresh_presets)
        self.confirm_overwrite_btn.on_click(self._on_confirm_overwrite_preset)
        self.cancel_overwrite_btn.on_click(self._on_cancel_overwrite_preset)
        self.clear_last_btn.on_click(self._on_clear_last)
        self.confirm_reset_btn.on_click(self._on_confirm_reset_configurations)
        self.cancel_reset_btn.on_click(self._on_cancel_reset_configurations)
        self.clear_all_btn.on_click(self._on_clear_all)
        self.clear_arena_btn.on_click(self._on_clear_arena)
        self.add_wind_puff_btn.on_click(self._on_add_wind_puff)
        self.remove_wind_puff_btn.on_click(self._on_remove_wind_puff)
        self.add_thermo_source_btn.on_click(self._on_add_thermo_source)
        self.remove_thermo_source_btn.on_click(self._on_remove_thermo_source)
        self.preset_name.param.watch(self._on_preset_name_change, "value")

        self._update_insert_hint()
        self._sync_arena_controls()
        self._sync_group_shape_controls()
        self._sync_selected_group_shape_controls()
        self._sync_food_grid_overlay()
        self._sync_odorscape_controls()
        self._sync_scape_preview()
        self._refresh_preset_controls()
        self._refresh_object_controls()
        self._sync_arena_lock_state()

    def view(self) -> pn.viewable.Viewable:
        intro = pn.pane.Markdown(
            (
                "### Environment Builder\n"
                "Use this app to compose and inspect Larvaworld environments visually. Define the arena, place "
                "source units, source groups, and borders on the canvas, configure optional food-grid and sensory-scape "
                "layers, and then refine every object from the `Inspect / Edit` panel. The builder follows the same "
                "canonical environment structure used by the codebase, mapping directly onto `FoodConf` / `EnvConf`: "
                "`source_units`, `source_groups`, `food_grid`, `border_list`, `odorscape`, `windscape`, and `thermoscape`. "
                "Food amount controls whether a source is visually filled, while odor is rendered as a Gaussian-like aura "
                "around the source. "
                "`Odorscape` contours appear in the preview only when at least one placed source has odor configured, "
                "`Thermoscape` preview appears only after thermal-source rows are added, and `Windscape` arrows appear only when wind speed is greater than zero. "
                "`Wind direction` is edited in degrees in the builder UI and converted back to radians for the exported config. "
                f"Reference: [Arenas and Substrates]({DOCS_ARENAS_SUBSTRATES})."
            ),
            css_classes=["lw-env-builder-intro"],
            margin=0,
        )
        divider = pn.pane.HTML('<div class="lw-env-builder-divider"></div>', margin=0)
        controls = pn.Card(
            pn.Column(
                self._field_view(self.arena_shape),
                self._field_view(self.arena_width),
                self._field_view(self.arena_height),
                self._field_view(self.arena_torus),
                divider,
                pn.Row(
                    self.select_mode,
                    self.clear_last_btn,
                    sizing_mode="stretch_width",
                    margin=0,
                ),
                self._field_view(self.object_type),
                self._field_view(self.object_radius),
                self._field_view(self.group_count),
                self._field_view(self.group_shape),
                self._field_view(self.group_mode),
                self._field_view(self.group_spread_x),
                self._field_view(self.group_spread_y),
                self._field_view(self.border_width),
                self._field_view(self.object_color),
                pn.layout.Divider(margin=(4, 0, 0, 0)),
                self._field_view(self.food_grid_enabled),
                self._field_view(self.food_grid_color),
                self._field_view(self.food_grid_dims_x),
                self._field_view(self.food_grid_dims_y),
                self._field_view(self.food_grid_initial_value),
                self._field_view(self.food_grid_fixed_max),
                self._field_view(self.food_grid_substrate_type),
                self._field_view(self.food_grid_substrate_quality),
                self.status,
                sizing_mode="stretch_width",
                margin=0,
            ),
            title="Controls",
            collapsed=False,
            sizing_mode="stretch_width",
        )
        presets_card = pn.Card(
            pn.Column(
                self._field_view(self.preset_name),
                self._field_view(self.preset_select),
                self.refresh_presets_btn,
                self.preset_meta,
                pn.Row(
                    self.save_preset_btn,
                    self.load_preset_btn,
                    self.delete_preset_btn,
                    sizing_mode="stretch_width",
                    margin=0,
                ),
                self.export_btn,
                sizing_mode="stretch_width",
                margin=0,
            ),
            title="Presets",
            collapsed=False,
            sizing_mode="stretch_width",
        )
        table_card = pn.Card(
            self.table,
            title="Placed objects",
            collapsed=False,
            sizing_mode="stretch_width",
        )
        editor_card = pn.Card(
            pn.Column(
                self._field_view(self.selected_object),
                self._field_view(self.selected_id),
                self._field_view(self.selected_x),
                self._field_view(self.selected_y),
                self._field_view(self.selected_x2),
                self._field_view(self.selected_y2),
                self._field_view(self.selected_radius),
                self._field_view(self.selected_width),
                self._field_view(self.selected_color),
                self._editor_food_family,
                self._editor_source_behavior_family,
                self._editor_substrate_family,
                self._editor_odor_family,
                self._editor_distribution_family,
                pn.Column(
                    pn.Row(
                        self.apply_selected_btn,
                        self.delete_selected_btn,
                        sizing_mode="stretch_width",
                        margin=0,
                    ),
                    self.clear_all_btn,
                    sizing_mode="stretch_width",
                    margin=(4, 0, 0, 0),
                ),
                sizing_mode="stretch_width",
                margin=0,
            ),
            title="Inspect / Edit",
            collapsed=False,
            sizing_mode="stretch_width",
        )
        scapes_card = pn.Card(
            self._odorscape_family,
            self._windscape_family,
            self._thermoscape_family,
            title="Environment Scapes",
            collapsed=False,
            sizing_mode="stretch_width",
        )
        side = pn.Column(
            controls,
            presets_card,
            width=360,
        )

        canvas = pn.pane.Bokeh(self.fig, sizing_mode="stretch_both")
        center = pn.Column(
            canvas,
            table_card,
            width=760,
        )
        right = pn.Column(
            editor_card,
            scapes_card,
            width=360,
        )
        main = pn.Row(side, center, right, sizing_mode="stretch_width")

        return pn.Column(
            intro, main, css_classes=["lw-env-builder-root"], sizing_mode="stretch_both"
        )

    def _update_insert_hint(self, *_: object) -> None:
        is_border_segment = self.object_type.value == "Border segment"
        is_source_group = self.object_type.value == "Source group"
        self._set_field_visible(self.border_width, is_border_segment)
        self._set_field_visible(self.object_radius, not is_border_segment)
        self._set_field_visible(self.group_count, is_source_group)
        self._set_field_visible(self.group_shape, is_source_group)
        self._set_field_visible(self.group_mode, is_source_group)
        self._set_field_visible(self.group_spread_x, is_source_group)
        self._set_field_visible(self.group_spread_y, is_source_group)
        self.object_radius.name = "Source unit radius (mm)"
        self._sync_group_shape_controls()
        if self.object_type.value == "Border segment":
            if self._border_start is None:
                self.status.object = "Click first point for border segment."
            else:
                x0, y0 = self._border_start
                self.status.object = (
                    f"First point captured at ({x0:.3f}, {y0:.3f}). Click second point."
                )
            return
        self._border_start = None
        self._clear_border_preview()
        self.status.object = f"Click canvas to add a {self.object_type.value.lower()}."

    def _arena_dimensions(self) -> tuple[float, float]:
        if self.arena_shape.value == "circular":
            diameter = max(float(self.arena_width.value) * 2.0, 0.0)
            return diameter, diameter
        return float(self.arena_width.value), float(self.arena_height.value)

    def _sync_arena_controls(self) -> None:
        is_circular = self.arena_shape.value == "circular"
        self.arena_width.name = "Arena radius (m)" if is_circular else "Arena width (m)"
        self._set_field_visible(self.arena_height, not is_circular)

    def _sync_arena_lock_state(self) -> None:
        locked = bool(self._objects)
        self.arena_shape.disabled = locked
        self.arena_width.disabled = locked
        self.arena_height.disabled = locked
        self.arena_torus.disabled = locked

    def _on_arena_shape_change(self, event: object) -> None:
        old = getattr(event, "old", None)
        new = getattr(event, "new", self.arena_shape.value)
        if self._suspend_arena_update:
            self._sync_arena_controls()
            self._apply_arena_visuals()
            return
        if old == new:
            self._sync_arena_controls()
            self._update_arena()
            return
        if old == "rectangular" and new == "circular":
            radius = (
                min(float(self.arena_width.value), float(self.arena_height.value)) / 2.0
            )
            self.arena_width.value = radius
            self.arena_height.value = radius
        elif old == "circular" and new == "rectangular":
            diameter = float(self.arena_width.value) * 2.0
            self.arena_width.value = diameter
            self.arena_height.value = diameter
        self._sync_arena_controls()
        self._update_arena()

    def _sync_odorscape_controls(self, *_: object) -> None:
        is_diffusion = self.odorscape_mode.value == "Diffusion"
        self._set_field_visible(self.odorscape_evap_const, is_diffusion)
        self._set_field_visible(self.odorscape_sigma_x, is_diffusion)
        self._set_field_visible(self.odorscape_sigma_y, is_diffusion)

    def _empty_scape_preview_sources(self) -> None:
        self.odorscape_contour_source.data = {
            "x": [],
            "y": [],
            "r": [],
            "color": [],
            "line_alpha": [],
            "line_width": [],
            "id": [],
        }
        self.windscape_segment_source.data = {
            "x0": [],
            "y0": [],
            "x1": [],
            "y1": [],
            "color": [],
            "line_alpha": [],
        }
        self.windscape_head_source.data = {
            "x": [],
            "y": [],
            "angle": [],
            "color": [],
            "size": [],
        }
        self.thermoscape_aura_source.data = {
            "x": [],
            "y": [],
            "r": [],
            "color": [],
            "fill_alpha": [],
            "line_alpha": [],
            "id": [],
        }
        self.thermoscape_marker_source.data = {
            "x": [],
            "y": [],
            "color": [],
            "size": [],
            "id": [],
        }

    def _iter_odor_preview_rows(self) -> list[_ObjectRow]:
        return [
            obj
            for obj in self._objects
            if obj.object_type in {"Source unit", "Source group"}
            and obj.x is not None
            and obj.y is not None
            and obj.odor_id
            and obj.odor_intensity is not None
            and obj.odor_spread is not None
            and float(obj.odor_intensity or 0.0) > 0
            and float(obj.odor_spread or 0.0) > 0
        ]

    def _build_odorscape_preview(self) -> None:
        if not self.odorscape_enabled.value:
            self.odorscape_contour_source.data = {
                "x": [],
                "y": [],
                "r": [],
                "color": [],
                "line_alpha": [],
                "line_width": [],
                "id": [],
            }
            return
        rows = []
        contour_color = self.odorscape_color.value
        mode = self.odorscape_mode.value
        line_alpha_scale = {"Analytical": 0.42, "Gaussian": 0.32, "Diffusion": 0.24}
        mults = [0.5, 1.0, 1.5, 2.2]
        for obj in self._iter_odor_preview_rows():
            spread = float(obj.odor_spread or 0.0)
            source_r = max(float(obj.radius or 0.003), 0.002)
            for index, mult in enumerate(mults):
                rows.append(
                    {
                        "x": float(obj.x),
                        "y": float(obj.y),
                        "r": source_r + spread * mult,
                        "color": contour_color,
                        "line_alpha": max(
                            0.08,
                            line_alpha_scale.get(mode, 0.3) - index * 0.07,
                        ),
                        "line_width": max(1.0, 2.2 - index * 0.35),
                        "id": str(obj.object_id),
                    }
                )
        if rows:
            self.odorscape_contour_source.data = {
                key: [row[key] for row in rows] for key in rows[0]
            }
        else:
            self.odorscape_contour_source.data = {
                "x": [],
                "y": [],
                "r": [],
                "color": [],
                "line_alpha": [],
                "line_width": [],
                "id": [],
            }

    def _build_windscape_preview(self) -> None:
        if not self.windscape_enabled.value or float(self.windscape_speed.value) <= 0:
            self.windscape_segment_source.data = {
                "x0": [],
                "y0": [],
                "x1": [],
                "y1": [],
                "color": [],
                "line_alpha": [],
            }
            self.windscape_head_source.data = {
                "x": [],
                "y": [],
                "angle": [],
                "color": [],
                "size": [],
            }
            return

        width, height = self._arena_dimensions()
        D = max(width, height) * 0.48
        N = 9
        ds = D / N * math.sqrt(2)
        direction_rad = _deg_to_rad(float(self.windscape_direction.value))
        angle = -direction_rad
        color = self.windscape_color.value
        speed = float(self.windscape_speed.value)
        segment_rows = []
        head_rows = []
        for i in range(N):
            y_offset = (i - N / 2) * ds
            p0 = _rotate_point(-D, y_offset, angle)
            p1 = _rotate_point(D, y_offset, angle)
            segment_rows.append(
                {
                    "x0": p0[0],
                    "y0": p0[1],
                    "x1": p1[0],
                    "y1": p1[1],
                    "color": color,
                    "line_alpha": min(0.9, 0.25 + speed / 80.0),
                }
            )
            head_rows.append(
                {
                    "x": p1[0],
                    "y": p1[1],
                    "angle": direction_rad - math.pi / 2.0,
                    "color": color,
                    "size": min(14.0, 8.0 + speed / 8.0),
                }
            )
        self.windscape_segment_source.data = {
            key: [row[key] for row in segment_rows] for key in segment_rows[0]
        }
        self.windscape_head_source.data = {
            key: [row[key] for row in head_rows] for key in head_rows[0]
        }

    def _build_thermoscape_preview(self) -> None:
        if not self.thermoscape_enabled.value:
            self.thermoscape_aura_source.data = {
                "x": [],
                "y": [],
                "r": [],
                "color": [],
                "fill_alpha": [],
                "line_alpha": [],
                "id": [],
            }
            self.thermoscape_marker_source.data = {
                "x": [],
                "y": [],
                "color": [],
                "size": [],
                "id": [],
            }
            return
        frame = self.thermo_sources_table.value
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            self.thermoscape_aura_source.data = {
                "x": [],
                "y": [],
                "r": [],
                "color": [],
                "fill_alpha": [],
                "line_alpha": [],
                "id": [],
            }
            self.thermoscape_marker_source.data = {
                "x": [],
                "y": [],
                "color": [],
                "size": [],
                "id": [],
            }
            return
        spread = max(float(self.thermoscape_spread.value), 0.001)
        aura_rows = []
        marker_rows = []
        for row in frame.to_dict(orient="records"):
            source_id = str(row.get("id") or "").strip()
            x = float(row.get("x", 0.0))
            y = float(row.get("y", 0.0))
            dtemp = float(row.get("dTemp", 0.0))
            color = "#d94841" if dtemp >= 0 else "#356ae6"
            marker_rows.append(
                {
                    "x": x,
                    "y": y,
                    "color": color,
                    "size": min(18.0, 10.0 + abs(dtemp) * 0.8),
                    "id": source_id,
                }
            )
            for mult, alpha in zip([0.45, 0.9, 1.5], [0.18, 0.10, 0.05]):
                aura_rows.append(
                    {
                        "x": x,
                        "y": y,
                        "r": spread * mult,
                        "color": color,
                        "fill_alpha": alpha,
                        "line_alpha": max(0.12, alpha + 0.03),
                        "id": source_id,
                    }
                )
        self.thermoscape_aura_source.data = {
            key: [row[key] for row in aura_rows] for key in aura_rows[0]
        }
        self.thermoscape_marker_source.data = {
            key: [row[key] for row in marker_rows] for key in marker_rows[0]
        }

    def _sync_scape_preview(self, *_: object) -> None:
        self._build_odorscape_preview()
        self._build_windscape_preview()
        self._build_thermoscape_preview()

    def _next_wind_puff_id(self) -> str:
        ids = []
        frame = self.wind_puffs_table.value
        if isinstance(frame, pd.DataFrame) and "id" in frame.columns:
            ids = [str(value) for value in frame["id"].tolist()]
        highest = 0
        for puff_id in ids:
            match = re.search(r"_(\d+)$", puff_id)
            if match:
                highest = max(highest, int(match.group(1)))
        return f"puff_{highest + 1:03d}"

    def _next_thermo_source_id(self) -> str:
        ids = []
        frame = self.thermo_sources_table.value
        if isinstance(frame, pd.DataFrame) and "id" in frame.columns:
            ids = [str(value) for value in frame["id"].tolist()]
        highest = 0
        for source_id in ids:
            match = re.search(r"_(\d+)$", source_id)
            if match:
                highest = max(highest, int(match.group(1)))
        return f"thermal_{highest + 1:03d}"

    def _on_add_wind_puff(self, _: object) -> None:
        frame = self.wind_puffs_table.value
        if not isinstance(frame, pd.DataFrame):
            frame = _table_dataframe([], _WIND_PUFF_COLUMNS)
        row = {
            "id": self._next_wind_puff_id(),
            "duration": 1.0,
            "speed": 10.0,
            "direction": 0.0,
            "start_time": 0.0,
            "N": 1,
            "interval": 5.0,
        }
        self.wind_puffs_table.value = pd.concat(
            [frame, pd.DataFrame([row])], ignore_index=True
        )[_WIND_PUFF_COLUMNS]

    def _on_remove_wind_puff(self, _: object) -> None:
        frame = self.wind_puffs_table.value
        selection = list(self.wind_puffs_table.selection or [])
        if not isinstance(frame, pd.DataFrame) or not selection:
            return
        self.wind_puffs_table.value = frame.drop(index=selection).reset_index(
            drop=True
        )[_WIND_PUFF_COLUMNS]
        self.wind_puffs_table.selection = []

    def _on_add_thermo_source(self, _: object) -> None:
        frame = self.thermo_sources_table.value
        if not isinstance(frame, pd.DataFrame):
            frame = _table_dataframe([], _THERMO_SOURCE_COLUMNS)
        row = {
            "id": self._next_thermo_source_id(),
            "x": 0.0,
            "y": 0.0,
            "dTemp": 8.0,
        }
        self.thermo_sources_table.value = pd.concat(
            [frame, pd.DataFrame([row])], ignore_index=True
        )[_THERMO_SOURCE_COLUMNS]

    def _on_remove_thermo_source(self, _: object) -> None:
        frame = self.thermo_sources_table.value
        selection = list(self.thermo_sources_table.selection or [])
        if not isinstance(frame, pd.DataFrame) or not selection:
            return
        self.thermo_sources_table.value = frame.drop(index=selection).reset_index(
            drop=True
        )[_THERMO_SOURCE_COLUMNS]
        self.thermo_sources_table.selection = []

    def _sync_group_shape_controls(self, *_: object) -> None:
        is_source_group = self.object_type.value == "Source group"
        shape = _normalize_group_shape(self.group_shape.value)
        is_circle = shape == "circle"
        if shape == "circle":
            self.group_spread_x.name = "Group radius (mm)"
            self.group_spread_y.name = "Group radius (mm)"
        elif shape == "oval":
            self.group_spread_x.name = "Group width (mm)"
            self.group_spread_y.name = "Group height (mm)"
        else:
            self.group_spread_x.name = "Group width (mm)"
            self.group_spread_y.name = "Group height (mm)"
        self._set_field_visible(self.group_spread_y, is_source_group and not is_circle)
        if is_source_group and is_circle:
            self.group_spread_y.value = float(self.group_spread_x.value)

    def _on_group_spread_x_change(self, event: object) -> None:
        if _normalize_group_shape(self.group_shape.value) == "circle":
            self.group_spread_y.value = float(
                getattr(event, "new", self.group_spread_x.value)
            )

    def _sync_selected_group_shape_controls(self, *_: object) -> None:
        selected = self._selected_row()
        is_source_group = (
            selected is not None and selected.object_type == "Source group"
        )
        shape = _normalize_group_shape(self.selected_distribution_shape.value)
        is_circle = shape == "circle"
        if shape == "circle":
            self.selected_distribution_scale_x.name = "Group radius (mm)"
            self.selected_distribution_scale_y.name = "Group radius (mm)"
        elif shape == "oval":
            self.selected_distribution_scale_x.name = "Group width (mm)"
            self.selected_distribution_scale_y.name = "Group height (mm)"
        else:
            self.selected_distribution_scale_x.name = "Group width (mm)"
            self.selected_distribution_scale_y.name = "Group height (mm)"
        self._set_field_visible(
            self.selected_distribution_scale_y, is_source_group and not is_circle
        )
        if is_source_group and is_circle:
            self.selected_distribution_scale_y.value = float(
                self.selected_distribution_scale_x.value
            )

    def _on_selected_distribution_scale_x_change(self, event: object) -> None:
        if _normalize_group_shape(self.selected_distribution_shape.value) == "circle":
            self.selected_distribution_scale_y.value = float(
                getattr(event, "new", self.selected_distribution_scale_x.value)
            )

    def _group_display_value_mm(self, shape: str, scale_m: float | None) -> float:
        value_m = float(scale_m or 0.012)
        if _normalize_group_shape(shape) == "circle":
            return round(value_m * 1000.0, 4)
        return round(value_m * 2000.0, 4)

    def _group_scale_from_display_mm(self, shape: str, display_mm: float) -> float:
        value_mm = float(display_mm)
        if _normalize_group_shape(shape) == "circle":
            return round(value_mm / 1000.0, 4)
        return round(value_mm / 2000.0, 4)

    def _preset_dir(self) -> Path:
        return get_workspace_dir("environments")

    def _preset_filename(self, name: str) -> str:
        cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", name.strip()).strip("._-")
        if not cleaned:
            cleaned = "environment_builder_config"
        if not cleaned.endswith(".json"):
            cleaned += ".json"
        return cleaned

    def _preset_label_from_filename(self, filename: str) -> str:
        return Path(filename).stem

    def _registry_preset_value(self, name: str) -> str:
        return f"{_REGISTRY_PRESET_PREFIX}{name}"

    def _is_registry_preset(self, selected: str | None) -> bool:
        return bool(selected and str(selected).startswith(_REGISTRY_PRESET_PREFIX))

    def _registry_preset_name_from_value(self, selected: str) -> str:
        return str(selected)[len(_REGISTRY_PRESET_PREFIX) :]

    def _next_counter_seed(self) -> int:
        highest = 0
        for obj in self._objects:
            match = re.search(r"_(\d+)$", obj.object_id)
            if match:
                highest = max(highest, int(match.group(1)))
        return highest + 1 if highest else len(self._objects) + 1

    def _refresh_preset_controls(self, *, selected_filename: str | None = None) -> None:
        workspace_message = ""
        workspace_options: dict[str, str] = {}
        workspace_labels: set[str] = set()
        workspace_available = True
        try:
            preset_dir = self._preset_dir()
            preset_dir.mkdir(parents=True, exist_ok=True)
        except WorkspaceError as exc:
            preset_dir = None
            workspace_available = False
            workspace_message = f"Workspace environments directory unavailable: {exc}"
        else:
            preset_files = sorted(preset_dir.glob("*.json"))
            workspace_labels = {
                self._preset_label_from_filename(path.name) for path in preset_files
            }
            workspace_options = {
                f"Workspace / {self._preset_label_from_filename(path.name)}": path.name
                for path in preset_files
            }

        registry_options = {
            f"Registry / {name}": self._registry_preset_value(name)
            for name in sorted(str(key) for key in reg.conf.Env.dict.keys())
            if name not in workspace_labels
        }
        options = {**workspace_options, **registry_options}
        self.preset_select.options = options
        self.preset_select.disabled = not bool(options)
        self.save_preset_btn.disabled = not workspace_available
        self.refresh_presets_btn.disabled = False

        if not options:
            self.preset_select.value = None
            self.load_preset_btn.disabled = True
        else:
            self.load_preset_btn.disabled = False
            if selected_filename in options.values():
                self.preset_select.value = selected_filename
            elif self.preset_select.value not in options.values():
                self.preset_select.value = next(iter(options.values()))

        meta_lines = ["<div>"]
        if workspace_available and preset_dir is not None:
            meta_lines.append(
                f'Workspace preset directory:<br><code style="color: #4a8fd8;">{preset_dir}</code><br>'
            )
        elif workspace_message:
            meta_lines.append(f"{workspace_message}<br>")
        meta_lines.append(
            f"{len(registry_options)} Registry environments from:<br>"
            f'<code style="color: #4a8fd8;">{reg.conf.Env.path_to_dict}</code>'
        )
        meta_lines.append("</div>")
        self.preset_meta.object = "".join(meta_lines)
        if (
            self._pending_overwrite_id is not None
            and not self._preset_id_exists_in_visible_options(
                self._pending_overwrite_id
            )
        ):
            self._clear_overwrite_confirmation()

    def _preset_id_exists_in_visible_options(self, preset_id: str) -> bool:
        normalized_id = str(preset_id).strip()
        if not normalized_id:
            return False
        options = self.preset_select.options
        labels = options.keys() if isinstance(options, dict) else (options or [])
        for label in labels:
            label_text = str(label)
            if " / " in label_text:
                label_text = label_text.split(" / ", 1)[1]
            if label_text == normalized_id:
                return True
        return False

    def _show_overwrite_confirmation(self, preset_id: str) -> None:
        self._pending_overwrite_id = preset_id
        self.preset_overwrite_text.object = (
            f'Unique ID "{preset_id}" already exists. Overwrite the existing preset?'
        )
        self.preset_overwrite_confirm.visible = True
        self.status.object = (
            f'Unique ID "{preset_id}" already exists. Confirm overwrite or cancel.'
        )

    def _clear_overwrite_confirmation(self) -> None:
        self._pending_overwrite_id = None
        self.preset_overwrite_text.object = ""
        self.preset_overwrite_confirm.visible = False

    def _on_preset_name_change(self, _: param.parameterized.Event) -> None:
        self._clear_overwrite_confirmation()

    def _on_confirm_overwrite_preset(self, _: object) -> None:
        if not self._pending_overwrite_id:
            return
        self._perform_save_preset()

    def _on_cancel_overwrite_preset(self, _: object) -> None:
        pending_id = self._pending_overwrite_id
        self._clear_overwrite_confirmation()
        if pending_id:
            self.status.object = f'Overwrite cancelled for "{pending_id}".'

    def _perform_save_preset(self) -> None:
        self._clear_overwrite_confirmation()
        try:
            preset_dir = self._preset_dir()
            preset_dir.mkdir(parents=True, exist_ok=True)
        except WorkspaceError as exc:
            self.status.object = (
                f"Cannot save preset without an active workspace: {exc}"
            )
            return

        raw_name = self.preset_name.value.strip() or "environment_builder_config"
        filename = self._preset_filename(raw_name)
        target = preset_dir / filename
        registry_id = self._preset_label_from_filename(filename)
        validation_error = self._validate_configuration_for_storage()
        if validation_error is not None:
            self.status.object = f"{validation_error} Preset was not saved."
            return
        payload = self._build_export_config()
        target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        env_dict = util.AttrDict(reg.conf.Env.dict).get_copy()
        env_dict[registry_id] = self._build_registry_config()
        reg.conf.Env.set_dict(env_dict)
        self._refresh_preset_controls(selected_filename=filename)
        self.preset_name.value = registry_id
        self.status.object = (
            f'Saved environment preset "{self.preset_name.value}" to the workspace '
            "and registered it in Env.txt."
        )

    def _selected_row(self) -> _ObjectRow | None:
        selected_id = self._selected_object_id or self.selected_object.value
        if not selected_id:
            return None
        for obj in self._objects:
            if obj.object_id == selected_id:
                return obj
        return None

    def _selected_row_index(self) -> int | None:
        selected = self._selected_row()
        if selected is None:
            return None
        for index, obj in enumerate(self._objects):
            if obj.object_id == selected.object_id:
                return index
        return None

    def _object_options(self) -> dict[str, str]:
        return {
            f"{obj.object_id} ({obj.object_type})": obj.object_id
            for obj in self._objects
        }

    def _odor_id_options(self) -> list[str]:
        odor_ids = {
            str(obj.odor_id).strip()
            for obj in self._objects
            if obj.object_type in {"Source unit", "Source group"} and obj.odor_id
        }
        return sorted(odor_id for odor_id in odor_ids if odor_id)

    def _set_editor_disabled(self, disabled: bool) -> None:
        widgets = [
            self.selected_object,
            self.selected_id,
            self.selected_x,
            self.selected_y,
            self.selected_x2,
            self.selected_y2,
            self.selected_radius,
            self.selected_width,
            self.selected_color,
            self.selected_amount,
            self.selected_odor_id,
            self.selected_odor_intensity,
            self.selected_odor_spread,
            self.selected_distribution_n,
            self.selected_distribution_shape,
            self.selected_distribution_mode,
            self.selected_distribution_scale_x,
            self.selected_distribution_scale_y,
            self.selected_substrate_type,
            self.selected_substrate_quality,
            self.apply_selected_btn,
            self.delete_selected_btn,
        ]
        for widget in widgets:
            widget.disabled = disabled

    def _clear_canvas_highlight(self) -> None:
        self.food_highlight_source.data = {"x": [], "y": [], "r": [], "color": []}
        self.source_group_circle_highlight_source.data = (
            self._empty_source_group_circle_highlight_data()
        )
        self.source_group_ellipse_highlight_source.data = (
            self._empty_source_group_xy_highlight_data()
        )
        self.source_group_rect_highlight_source.data = (
            self._empty_source_group_xy_highlight_data()
        )
        self.border_highlight_source.data = {
            "x0": [],
            "y0": [],
            "x1": [],
            "y1": [],
            "w": [],
            "color": [],
        }

    def _update_canvas_highlight(self, obj: _ObjectRow | None) -> None:
        self._clear_canvas_highlight()
        if obj is None:
            return
        if obj.object_type == "Source unit":
            self.food_highlight_source.data = {
                "x": [obj.x],
                "y": [obj.y],
                "r": [max(float(obj.radius or 0.008) * 1.35, 0.004)],
                "color": ["#f97316"],
            }
        elif obj.object_type == "Source group":
            shape = _normalize_group_shape(obj.distribution_shape)
            width = max(float(obj.distribution_scale_x or 0.008) * 2.1, 0.01)
            height = max(float(obj.distribution_scale_y or 0.008) * 2.1, 0.01)
            if shape == "circle":
                self.source_group_circle_highlight_source.data = {
                    "x": [obj.x],
                    "y": [obj.y],
                    "r": [max(width, height) / 2.0],
                    "color": ["#f97316"],
                }
            elif shape == "oval":
                self.source_group_ellipse_highlight_source.data = {
                    "x": [obj.x],
                    "y": [obj.y],
                    "w": [width],
                    "h": [height],
                    "color": ["#f97316"],
                }
            else:
                self.source_group_rect_highlight_source.data = {
                    "x": [obj.x],
                    "y": [obj.y],
                    "w": [width],
                    "h": [height],
                    "color": ["#f97316"],
                }
        else:
            self.border_highlight_source.data = {
                "x0": [obj.x],
                "y0": [obj.y],
                "x1": [obj.x2],
                "y1": [obj.y2],
                "w": [max(4, int((obj.width or 0.001) * 3000))],
                "color": ["#f97316"],
            }

    def _set_selected_object(self, object_id: str | None) -> None:
        self._syncing_selection = True
        try:
            self._selected_object_id = object_id
            options = self._object_options()
            if object_id is None or object_id not in options.values():
                self.selected_object.value = None
                self.table.selection = []
                self._populate_editor(None)
                self.selected_object.disabled = not bool(options)
                self._update_canvas_highlight(None)
                return
            self.selected_object.value = object_id
            row_index = self._selected_row_index()
            self.table.selection = [] if row_index is None else [row_index]
            obj = self._selected_row()
            self._populate_editor(obj)
            self._update_canvas_highlight(obj)
        finally:
            self._syncing_selection = False

    def _sync_editor_visibility(self, obj: _ObjectRow | None) -> None:
        object_type = obj.object_type if obj is not None else None
        is_border = object_type == "Border segment"
        is_source_unit = object_type == "Source unit"
        is_source_group = object_type == "Source group"
        is_source = is_source_unit or is_source_group
        self._set_field_visible(self.selected_x2, is_border)
        self._set_field_visible(self.selected_y2, is_border)
        self._set_field_visible(self.selected_width, is_border)
        self._set_field_visible(
            self.selected_radius, object_type in {"Source unit", "Source group"}
        )
        self._set_field_visible(self.selected_amount, is_source)
        self._set_field_visible(self.selected_can_be_carried, is_source)
        self._set_field_visible(self.selected_can_be_displaced, is_source)
        self._set_field_visible(self.selected_regeneration, is_source)
        self._set_field_visible(self.selected_odor_id, is_source)
        self._set_field_visible(self.selected_odor_intensity, is_source)
        self._set_field_visible(self.selected_odor_spread, is_source)
        self._set_field_visible(self.selected_distribution_n, is_source_group)
        self._set_field_visible(self.selected_distribution_shape, is_source_group)
        self._set_field_visible(self.selected_distribution_mode, is_source_group)
        self._set_field_visible(self.selected_distribution_show_shape, is_source_group)
        self._set_field_visible(self.selected_distribution_scale_x, is_source_group)
        self._set_field_visible(self.selected_distribution_scale_y, is_source_group)
        self._set_field_visible(self.selected_substrate_type, is_source)
        self._set_field_visible(self.selected_substrate_quality, is_source)
        self._editor_food_family.visible = is_source
        self._editor_source_behavior_family.visible = is_source
        self._editor_substrate_family.visible = is_source
        self._editor_odor_family.visible = is_source
        self._editor_distribution_family.visible = is_source_group
        self._sync_selected_group_shape_controls()

    def _set_substrate_type_options(self, substrate_type: str | None = None) -> None:
        options = list(SUBSTRATE_TYPE_OPTIONS)
        if substrate_type:
            substrate_value = str(substrate_type)
            if substrate_value not in options:
                options.append(substrate_value)
        self.selected_substrate_type.options = options

    def _populate_editor(self, obj: _ObjectRow | None) -> None:
        self._sync_editor_visibility(obj)
        if obj is None:
            self._set_editor_disabled(True)
            self._set_substrate_type_options("standard")
            return

        self._set_editor_disabled(False)
        self.selected_id.value = obj.object_id
        self.selected_x.value = float(obj.x or 0.0)
        self.selected_y.value = float(obj.y or 0.0)
        self.selected_x2.value = float(obj.x2 or 0.0)
        self.selected_y2.value = float(obj.y2 or 0.0)
        self.selected_radius.value = round(
            float(obj.radius if obj.radius is not None else 0.003) * 1000.0, 4
        )
        self.selected_width.value = round(float(obj.width or 0.001) * 1000.0, 4)
        self.selected_color.value = str(obj.color or "#4caf50")
        self.selected_amount.value = float(
            obj.amount if obj.amount is not None else 0.0
        )
        self.selected_can_be_carried.value = bool(obj.can_be_carried)
        self.selected_can_be_displaced.value = bool(obj.can_be_displaced)
        self.selected_regeneration.value = bool(obj.regeneration)
        self.selected_odor_id.value = str(obj.odor_id or "")
        self.selected_odor_intensity.value = float(
            obj.odor_intensity if obj.odor_intensity is not None else 1.0
        )
        self.selected_odor_spread.value = float(
            obj.odor_spread if obj.odor_spread is not None else 0.02
        )
        self.selected_distribution_n.value = int(obj.distribution_n or 30)
        self.selected_distribution_shape.value = _normalize_group_shape(
            obj.distribution_shape
        )
        self.selected_distribution_mode.value = str(obj.distribution_mode or "uniform")
        self.selected_distribution_show_shape.value = bool(
            True if obj.distribution_show_shape is None else obj.distribution_show_shape
        )
        self.selected_distribution_scale_x.value = self._group_display_value_mm(
            self.selected_distribution_shape.value, obj.distribution_scale_x
        )
        self.selected_distribution_scale_y.value = self._group_display_value_mm(
            self.selected_distribution_shape.value, obj.distribution_scale_y
        )
        self._sync_selected_group_shape_controls()
        self._set_substrate_type_options(obj.substrate_type or "standard")
        self.selected_substrate_type.value = str(obj.substrate_type or "standard")
        self.selected_substrate_quality.value = float(obj.substrate_quality or 1.0)

    def _refresh_object_controls(
        self, *, selected_object_id: str | None = None
    ) -> None:
        self._sync_arena_lock_state()
        self.selected_odor_id.options = self._odor_id_options()
        options = self._object_options()
        self.selected_object.options = options
        if not options:
            self._set_selected_object(None)
            return

        self.selected_object.disabled = False
        target_id = selected_object_id
        if target_id not in options.values():
            target_id = (
                self._selected_object_id
                if self._selected_object_id in options.values()
                else next(iter(options.values()))
            )
        self._set_selected_object(target_id)

    def _on_selected_object_change(self, *_: object) -> None:
        if self._syncing_selection:
            return
        self._set_selected_object(self.selected_object.value)

    def _on_table_selection_change(self, *_: object) -> None:
        if self._syncing_selection:
            return
        selection = list(self.table.selection or [])
        if not selection:
            self._set_selected_object(None)
            return
        row_index = selection[0]
        if row_index < 0 or row_index >= len(self._objects):
            self._set_selected_object(None)
            return
        self._set_selected_object(self._objects[row_index].object_id)

    def _on_select_mode_change(self, *_: object) -> None:
        if self.select_mode.value:
            self.select_mode.button_type = "success"
            self.status.object = (
                "Select mode enabled. Click an object on the canvas to inspect it."
            )
        else:
            self.select_mode.button_type = "primary"
            self.status.object = (
                f"Click canvas to add a {self.object_type.value.lower()}."
            )

    def _iter_loaded_objects(self, config: dict[str, object]) -> list[_ObjectRow]:
        loaded: list[_ObjectRow] = []
        food_params = config.get("food_params")
        if isinstance(food_params, dict):
            source_units = food_params.get("source_units", {})
            if isinstance(source_units, dict):
                for object_id, entry in source_units.items():
                    if not isinstance(entry, dict):
                        continue
                    pos = entry.get("pos")
                    if not isinstance(pos, (list, tuple)) or len(pos) < 2:
                        continue
                    loaded.append(
                        _ObjectRow(
                            object_id=str(object_id),
                            object_type="Source unit",
                            x=float(pos[0]),
                            y=float(pos[1]),
                            radius=float(entry.get("radius", 0.003)),
                            color=str(entry.get("color", "#4caf50")),
                            amount=(
                                float(entry.get("amount"))
                                if entry.get("amount") is not None
                                else 0.0
                            ),
                            can_be_carried=bool(entry.get("can_be_carried", False)),
                            can_be_displaced=bool(entry.get("can_be_displaced", False)),
                            regeneration=bool(entry.get("regeneration", False)),
                            odor_id=str(
                                entry.get("odor", {}).get("id", f"{object_id}_odor")
                            )
                            if entry.get("odor", {}).get("id") is not None
                            else None,
                            odor_intensity=(
                                float(entry.get("odor", {}).get("intensity"))
                                if entry.get("odor", {}).get("intensity") is not None
                                else None
                            ),
                            odor_spread=(
                                float(entry.get("odor", {}).get("spread"))
                                if entry.get("odor", {}).get("spread") is not None
                                else None
                            ),
                            substrate_type=str(
                                entry.get("substrate", {}).get("type", "standard")
                            ),
                            substrate_quality=float(
                                entry.get("substrate", {}).get("quality", 1.0)
                            ),
                        )
                    )
            source_groups = food_params.get("source_groups", {})
            if isinstance(source_groups, dict):
                for object_id, entry in source_groups.items():
                    if not isinstance(entry, dict):
                        continue
                    distribution = entry.get("distribution", {})
                    if not isinstance(distribution, dict):
                        distribution = {}
                    loc = distribution.get("loc", (0.0, 0.0))
                    scale = distribution.get("scale", (0.012, 0.012))
                    if not isinstance(loc, (list, tuple)) or len(loc) < 2:
                        continue
                    if not isinstance(scale, (list, tuple)) or len(scale) < 2:
                        scale = (0.012, 0.012)
                    loaded.append(
                        _ObjectRow(
                            object_id=str(object_id),
                            object_type="Source group",
                            x=float(loc[0]),
                            y=float(loc[1]),
                            radius=float(entry.get("radius", 0.003)),
                            color=str(entry.get("color", "#4caf50")),
                            amount=(
                                float(entry.get("amount"))
                                if entry.get("amount") is not None
                                else 0.0
                            ),
                            can_be_carried=bool(entry.get("can_be_carried", False)),
                            can_be_displaced=bool(entry.get("can_be_displaced", False)),
                            regeneration=bool(entry.get("regeneration", False)),
                            odor_id=str(entry.get("odor", {}).get("id"))
                            if entry.get("odor", {}).get("id") is not None
                            else None,
                            odor_intensity=entry.get("odor", {}).get("intensity"),
                            odor_spread=entry.get("odor", {}).get("spread"),
                            substrate_type=str(
                                entry.get("substrate", {}).get("type", "standard")
                            ),
                            substrate_quality=float(
                                entry.get("substrate", {}).get("quality", 1.0)
                            ),
                            distribution_mode=str(distribution.get("mode", "uniform")),
                            distribution_shape=_normalize_group_shape(
                                distribution.get("shape", "circle")
                            ),
                            distribution_n=int(distribution.get("N", 30)),
                            distribution_show_shape=True,
                            distribution_scale_x=float(scale[0]),
                            distribution_scale_y=float(scale[1]),
                        )
                    )

        border_list = config.get("border_list", {})
        if isinstance(border_list, dict):
            for object_id, entry in border_list.items():
                if not isinstance(entry, dict):
                    continue
                vertices = entry.get("vertices", entry.get("points"))
                if not isinstance(vertices, (list, tuple)) or len(vertices) < 2:
                    continue
                valid_vertices = [
                    tuple(point)
                    for point in vertices
                    if isinstance(point, (list, tuple)) and len(point) >= 2
                ]
                if len(valid_vertices) < 2:
                    continue
                for segment_index in range(0, len(valid_vertices) - 1, 2):
                    p0 = valid_vertices[segment_index]
                    p1 = valid_vertices[segment_index + 1]
                    segment_id = (
                        str(object_id)
                        if len(valid_vertices) == 2
                        else f"{object_id}_{segment_index // 2 + 1:03d}"
                    )
                    loaded.append(
                        _ObjectRow(
                            object_id=segment_id,
                            object_type="Border segment",
                            x=float(p0[0]),
                            y=float(p0[1]),
                            x2=float(p1[0]),
                            y2=float(p1[1]),
                            width=float(entry.get("width", 0.001)),
                            color=str(entry.get("color", "#111111")),
                        )
                    )

        return loaded

    def _apply_config(self, config: dict[str, object]) -> None:
        translated_config = _translate_builder_environment_payload(config)
        self._loaded_config = util.AttrDict(translated_config).get_copy()
        arena = translated_config.get("arena", {})
        if isinstance(arena, dict):
            geometry = str(arena.get("geometry", arena.get("shape", "rectangular")))
            geometry = {
                "rect": "rectangular",
                "rectangle": "rectangular",
                "circle": "circular",
            }.get(geometry, geometry)
            if geometry in {"rectangular", "circular"}:
                self.arena_shape.value = geometry
            self.arena_torus.value = bool(arena.get("torus", False))
            dims = arena.get("dims")
            if isinstance(dims, (list, tuple)) and len(dims) >= 2:
                if geometry == "circular":
                    self.arena_width.value = float(dims[0]) / 2.0
                    self.arena_height.value = float(dims[0]) / 2.0
                else:
                    self.arena_width.value = float(dims[0])
                    self.arena_height.value = float(dims[1])
            self._sync_arena_controls()
            self._update_arena()

        food_params = translated_config.get("food_params", {})
        if isinstance(food_params, dict):
            food_grid = food_params.get("food_grid")
            if isinstance(food_grid, dict):
                self.food_grid_enabled.value = True
                self.food_grid_color.value = str(food_grid.get("color", "#4caf50"))
                grid_dims = food_grid.get("grid_dims", (51, 51))
                if isinstance(grid_dims, (list, tuple)) and len(grid_dims) >= 2:
                    self.food_grid_dims_x.value = int(grid_dims[0])
                    self.food_grid_dims_y.value = int(grid_dims[1])
                self.food_grid_initial_value.value = float(
                    food_grid.get("initial_value", 1e-6)
                )
                self.food_grid_fixed_max.value = bool(food_grid.get("fixed_max", True))
                self.food_grid_substrate_type.value = str(
                    food_grid.get("substrate", {}).get("type", "standard")
                )
                self.food_grid_substrate_quality.value = float(
                    food_grid.get("substrate", {}).get("quality", 1.0)
                )
            else:
                self.food_grid_enabled.value = False

        odorscape = translated_config.get("odorscape")
        if isinstance(odorscape, dict):
            odorscape_mode = str(odorscape.get("odorscape", "Gaussian"))
            if odorscape_mode not in self.odorscape_mode.options:
                odorscape_mode = "Gaussian"
            self.odorscape_enabled.value = True
            self.odorscape_mode.value = odorscape_mode
            self.odorscape_color.value = str(odorscape.get("color", "#4caf50"))
            odor_grid_dims = odorscape.get("grid_dims", (51, 51))
            if isinstance(odor_grid_dims, (list, tuple)) and len(odor_grid_dims) >= 2:
                self.odorscape_grid_dims_x.value = int(odor_grid_dims[0])
                self.odorscape_grid_dims_y.value = int(odor_grid_dims[1])
            self.odorscape_initial_value.value = float(
                odorscape.get("initial_value", 0.0)
            )
            self.odorscape_fixed_max.value = bool(odorscape.get("fixed_max", False))
            self.odorscape_evap_const.value = float(odorscape.get("evap_const", 0.9))
            gaussian_sigma = odorscape.get("gaussian_sigma", (0.95, 0.95))
            if isinstance(gaussian_sigma, (list, tuple)) and len(gaussian_sigma) >= 2:
                self.odorscape_sigma_x.value = float(gaussian_sigma[0])
                self.odorscape_sigma_y.value = float(gaussian_sigma[1])
        else:
            self.odorscape_enabled.value = False

        windscape = translated_config.get("windscape")
        if isinstance(windscape, dict):
            self.windscape_enabled.value = True
            self.windscape_color.value = str(windscape.get("color", "#ff0000"))
            self.windscape_direction.value = _rad_to_deg(
                float(windscape.get("wind_direction", math.pi))
            )
            self.windscape_speed.value = float(windscape.get("wind_speed", 0.0))
            puffs = windscape.get("puffs", {})
            puff_rows: list[dict[str, object]] = []
            if isinstance(puffs, dict):
                for puff_id, puff in puffs.items():
                    if not isinstance(puff, dict):
                        continue
                    puff_rows.append(
                        {
                            "id": str(puff_id),
                            "duration": float(puff.get("duration", 1.0)),
                            "speed": float(puff.get("speed", 10.0)),
                            "direction": float(puff.get("direction", 0.0)),
                            "start_time": float(puff.get("start_time", 0.0)),
                            "N": (
                                int(puff.get("N"))
                                if puff.get("N") is not None
                                else None
                            ),
                            "interval": float(puff.get("interval", 5.0)),
                        }
                    )
            self.wind_puffs_table.value = _table_dataframe(
                puff_rows, _WIND_PUFF_COLUMNS
            )
            self.wind_puffs_table.selection = []
        else:
            self.windscape_enabled.value = False
            self.wind_puffs_table.value = _table_dataframe([], _WIND_PUFF_COLUMNS)
            self.wind_puffs_table.selection = []

        thermoscape = translated_config.get("thermoscape")
        if isinstance(thermoscape, dict):
            self.thermoscape_enabled.value = True
            self.thermoscape_plate_temp.value = float(
                thermoscape.get("plate_temp", 22.0)
            )
            self.thermoscape_spread.value = float(thermoscape.get("spread", 0.1))
            thermo_sources = thermoscape.get("thermo_sources", {})
            thermo_source_dtemps = thermoscape.get("thermo_source_dTemps", {})
            thermo_rows: list[dict[str, object]] = []
            if isinstance(thermo_sources, dict):
                for source_id, pos in thermo_sources.items():
                    if not isinstance(pos, (list, tuple)) or len(pos) < 2:
                        continue
                    dtemp = 0.0
                    if isinstance(thermo_source_dtemps, dict):
                        dtemp = float(thermo_source_dtemps.get(source_id, 0.0))
                    thermo_rows.append(
                        {
                            "id": str(source_id),
                            "x": float(pos[0]),
                            "y": float(pos[1]),
                            "dTemp": dtemp,
                        }
                    )
            self.thermo_sources_table.value = _table_dataframe(
                thermo_rows, _THERMO_SOURCE_COLUMNS
            )
            self.thermo_sources_table.selection = []
        else:
            self.thermoscape_enabled.value = False
            self.thermo_sources_table.value = _table_dataframe(
                [], _THERMO_SOURCE_COLUMNS
            )
            self.thermo_sources_table.selection = []

        self._objects = self._iter_loaded_objects(translated_config)
        self._border_start = None
        self._counter = self._next_counter_seed()
        self._rebuild_sources()
        self._sync_food_grid_overlay()
        self._sync_odorscape_controls()
        self._sync_scape_preview()
        self._refresh_table()
        self._refresh_object_controls()
        self._update_insert_hint()

    def _on_save_preset(self, _: object) -> None:
        raw_name = self.preset_name.value.strip() or "environment_builder_config"
        if (
            self._preset_id_exists_in_visible_options(raw_name)
            and self._pending_overwrite_id != raw_name
        ):
            self._show_overwrite_confirmation(raw_name)
            return
        self._perform_save_preset()

    def _apply_loaded_preset_payload(
        self,
        payload: object,
        *,
        loaded_name: str,
        status_prefix: str,
    ) -> None:
        if not isinstance(payload, dict):
            self.status.object = "Preset file is not a valid environment configuration."
            return
        self._apply_config(payload)
        self.preset_name.value = loaded_name
        self.status.object = f'{status_prefix} "{loaded_name}".'

    def _on_load_preset(self, _: object) -> None:
        self._clear_overwrite_confirmation()
        selected = self.preset_select.value
        if not selected:
            self.status.object = "Select a saved preset first."
            return

        if self._is_registry_preset(str(selected)):
            registry_name = self._registry_preset_name_from_value(str(selected))
            try:
                payload = util.AttrDict(reg.conf.Env.getID(registry_name)).get_copy()
            except Exception as exc:
                self.status.object = f"Failed to load registry environment: {exc}"
                return
            loaded_name = registry_name
            status_prefix = "Loaded registry environment"
        else:
            try:
                path = self._preset_dir() / str(selected)
            except WorkspaceError as exc:
                self.status.object = (
                    f"Cannot load workspace preset without an active workspace: {exc}"
                )
                return

            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                self._refresh_preset_controls()
                self.status.object = "Selected preset file no longer exists."
                return
            except (OSError, json.JSONDecodeError) as exc:
                self.status.object = f"Failed to read preset: {exc}"
                return
            loaded_name = path.stem
            status_prefix = "Loaded environment preset"

        self._apply_loaded_preset_payload(
            payload,
            loaded_name=loaded_name,
            status_prefix=status_prefix,
        )

    def _on_load_file(self, _: param.parameterized.Event) -> None:
        self._clear_overwrite_confirmation()
        raw_value = self.load_file_input.value
        if raw_value in (None, b"", ""):
            return
        filename = str(getattr(self.load_file_input, "filename", "") or "").strip()
        loaded_name = Path(filename).stem if filename else "imported_environment"
        try:
            if isinstance(raw_value, (bytes, bytearray)):
                text = raw_value.decode("utf-8")
            elif isinstance(raw_value, str):
                text = raw_value
            else:
                raise ValueError("Unsupported uploaded file payload type.")
            payload = json.loads(text)
        except Exception as exc:
            self.status.object = f"Failed to load file: {exc}"
            return

        self._apply_loaded_preset_payload(
            payload,
            loaded_name=loaded_name,
            status_prefix="Loaded environment file",
        )

    def _on_delete_preset(self, _: object) -> None:
        self._clear_overwrite_confirmation()
        selected = self.preset_select.value
        if not selected:
            self.status.object = "Select a saved preset first."
            return

        deleted_workspace = False
        deleted_registry = False

        if self._is_registry_preset(str(selected)):
            registry_name = self._registry_preset_name_from_value(str(selected))
            reg.conf.Env.delete(registry_name)
            deleted_registry = registry_name not in reg.conf.Env.dict
            deleted_label = registry_name
        else:
            try:
                path = self._preset_dir() / str(selected)
            except WorkspaceError as exc:
                self.status.object = (
                    f"Cannot delete workspace preset without an active workspace: {exc}"
                )
                return

            deleted_label = path.stem
            try:
                path.unlink()
                deleted_workspace = True
            except FileNotFoundError:
                deleted_workspace = False
            except OSError as exc:
                self.status.object = f"Failed to delete preset file: {exc}"
                return

            reg.conf.Env.delete(deleted_label)
            deleted_registry = deleted_label not in reg.conf.Env.dict

        self._refresh_preset_controls()
        removed_targets: list[str] = []
        if deleted_workspace:
            removed_targets.append("workspace")
        if deleted_registry:
            removed_targets.append("registry")
        if not removed_targets:
            self.status.object = (
                f'Preset "{deleted_label}" was already missing from the selected store.'
            )
            return
        removed_text = " and ".join(removed_targets)
        self.status.object = f'Deleted preset "{deleted_label}" from {removed_text}.'

    def _on_refresh_presets(self, _: object) -> None:
        self._clear_overwrite_confirmation()
        self._refresh_preset_controls(
            selected_filename=str(self.preset_select.value or "")
        )
        self.status.object = "Refreshed environment preset list."

    def _on_apply_selected_object(self, _: object) -> None:
        current = self._selected_row()
        if current is None:
            self.status.object = "Select an object to edit first."
            return

        new_id = self.selected_id.value.strip()
        if not new_id:
            self.status.object = "Object ID cannot be empty."
            return
        if any(obj.object_id == new_id and obj is not current for obj in self._objects):
            self.status.object = f'Object ID "{new_id}" is already in use.'
            return

        x = float(self.selected_x.value)
        y = float(self.selected_y.value)
        if not self._inside_arena(x, y):
            self.status.object = (
                "Primary object coordinates must stay inside the arena."
            )
            return

        updated = _ObjectRow(
            object_id=new_id,
            object_type=current.object_type,
            x=x,
            y=y,
            color=self.selected_color.value,
        )

        if current.object_type == "Border segment":
            x2 = float(self.selected_x2.value)
            y2 = float(self.selected_y2.value)
            if not self._inside_arena(x2, y2):
                self.status.object = (
                    "Border end coordinates must stay inside the arena."
                )
                return
            updated = _ObjectRow(
                object_id=new_id,
                object_type=current.object_type,
                x=x,
                y=y,
                x2=x2,
                y2=y2,
                width=round(float(self.selected_width.value) / 1000.0, 4),
                color=self.selected_color.value,
            )
        else:
            odor_id = self.selected_odor_id.value.strip() or None
            group_shape = _normalize_group_shape(self.selected_distribution_shape.value)
            group_scale_x = self._group_scale_from_display_mm(
                group_shape, float(self.selected_distribution_scale_x.value)
            )
            group_scale_y = (
                group_scale_x
                if group_shape == "circle"
                else self._group_scale_from_display_mm(
                    group_shape, float(self.selected_distribution_scale_y.value)
                )
            )
            updated = _ObjectRow(
                object_id=new_id,
                object_type=current.object_type,
                x=x,
                y=y,
                radius=round(float(self.selected_radius.value) / 1000.0, 4),
                color=self.selected_color.value,
                amount=float(self.selected_amount.value),
                can_be_carried=bool(self.selected_can_be_carried.value),
                can_be_displaced=bool(self.selected_can_be_displaced.value),
                regeneration=bool(self.selected_regeneration.value),
                odor_id=odor_id,
                odor_intensity=(
                    float(self.selected_odor_intensity.value)
                    if odor_id is not None
                    else None
                ),
                odor_spread=(
                    float(self.selected_odor_spread.value)
                    if odor_id is not None
                    else None
                ),
                substrate_type=self.selected_substrate_type.value.strip() or "standard",
                substrate_quality=float(self.selected_substrate_quality.value),
                distribution_n=(
                    int(self.selected_distribution_n.value)
                    if current.object_type == "Source group"
                    else None
                ),
                distribution_shape=(
                    group_shape if current.object_type == "Source group" else None
                ),
                distribution_mode=(
                    self.selected_distribution_mode.value
                    if current.object_type == "Source group"
                    else None
                ),
                distribution_show_shape=(
                    bool(self.selected_distribution_show_shape.value)
                    if current.object_type == "Source group"
                    else None
                ),
                distribution_scale_x=(
                    group_scale_x if current.object_type == "Source group" else None
                ),
                distribution_scale_y=(
                    group_scale_y if current.object_type == "Source group" else None
                ),
            )

        validation_error = self._validate_object_within_arena(updated)
        if validation_error is not None:
            self.status.object = validation_error
            return

        self._objects = [
            updated if obj.object_id == current.object_id else obj
            for obj in self._objects
        ]
        self._counter = self._next_counter_seed()
        self._rebuild_sources()
        self._refresh_table()
        self._refresh_object_controls(selected_object_id=updated.object_id)
        self.status.object = f'Updated object "{updated.object_id}".'

    def _on_delete_selected_object(self, _: object) -> None:
        current = self._selected_row()
        if current is None:
            self.status.object = "Select an object to delete first."
            return
        self._objects = [
            obj for obj in self._objects if obj.object_id != current.object_id
        ]
        self._counter = self._next_counter_seed()
        self._rebuild_sources()
        self._refresh_table()
        self._refresh_object_controls()
        self.status.object = f'Deleted object "{current.object_id}".'

    def _apply_arena_visuals(self) -> None:
        self._sync_arena_controls()
        width, height = self._arena_dimensions()
        self._arena_source.data = {"x": [0.0], "y": [0.0], "w": [width], "h": [height]}
        self._food_grid_overlay_source.data = {
            "x": [0.0],
            "y": [0.0],
            "w": [width],
            "h": [height],
            "color": [self.food_grid_color.value],
            "fill_alpha": [0.08 if self.food_grid_enabled.value else 0.0],
        }
        is_rect = self.arena_shape.value == "rectangular"
        self._arena_rect_renderer.visible = is_rect
        self._arena_circle_renderer.visible = not is_rect
        self._food_grid_rect_renderer.visible = False
        self._food_grid_circle_renderer.visible = False
        self._food_grid_cell_renderer.visible = bool(self.food_grid_enabled.value)
        self._rebuild_food_grid_cells()
        self._sync_scape_preview()

    def _restore_last_valid_arena_controls(self) -> None:
        previous = dict(self._last_valid_arena_controls)
        self._suspend_arena_update = True
        try:
            self.arena_shape.value = str(previous["shape"])
            self.arena_width.value = float(previous["width"])
            self.arena_height.value = float(previous["height"])
        finally:
            self._suspend_arena_update = False
        self._apply_arena_visuals()

    def _update_arena(self, *_: object) -> None:
        if self._suspend_arena_update:
            self._apply_arena_visuals()
            return
        validation_error = self._validate_configuration_for_storage()
        if validation_error is not None:
            self._restore_last_valid_arena_controls()
            self.status.object = f"{validation_error} Arena size change was cancelled."
            return
        self._last_valid_arena_controls = {
            "shape": str(self.arena_shape.value),
            "width": float(self.arena_width.value),
            "height": float(self.arena_height.value),
        }
        self._apply_arena_visuals()

    def _sync_food_grid_overlay(self, *_: object) -> None:
        width, height = self._arena_dimensions()
        self._food_grid_overlay_source.data = {
            "x": [0.0],
            "y": [0.0],
            "w": [width],
            "h": [height],
            "color": [self.food_grid_color.value],
            "fill_alpha": [0.08 if self.food_grid_enabled.value else 0.0],
        }
        self._food_grid_cell_renderer.visible = bool(self.food_grid_enabled.value)
        self._rebuild_food_grid_cells()

    def _inside_arena(self, x: float, y: float) -> bool:
        width, height = self._arena_dimensions()
        if self.arena_shape.value == "rectangular":
            return abs(x) <= width / 2 and abs(y) <= height / 2
        if width <= 0 or height <= 0:
            return False
        nx = x / (width / 2)
        ny = y / (height / 2)
        return (nx * nx + ny * ny) <= 1.0

    def _inside_arena_with_margin(self, x: float, y: float, margin: float) -> bool:
        width, height = self._arena_dimensions()
        safe_margin = max(float(margin), 0.0)
        half_width = (width / 2.0) - safe_margin
        half_height = (height / 2.0) - safe_margin
        if half_width <= 0 or half_height <= 0:
            return False
        if self.arena_shape.value == "rectangular":
            return abs(x) <= half_width and abs(y) <= half_height
        nx = x / half_width
        ny = y / half_height
        return (nx * nx + ny * ny) <= 1.0

    def _validate_source_group_members_within_arena(
        self, obj: _ObjectRow
    ) -> str | None:
        expected_count = int(obj.distribution_n or 0)
        positions = self._group_member_positions(obj)
        if expected_count > 0 and len(positions) != expected_count:
            return (
                f'Source group "{obj.object_id}" could not generate a valid member '
                "layout for the current distribution settings."
            )
        member_radius = max(float(obj.radius or 0.003), 0.0015)
        for member_x, member_y in positions:
            if not self._inside_arena_with_margin(member_x, member_y, member_radius):
                return (
                    f'Source group "{obj.object_id}" places member units outside the '
                    "arena. Move it inward or reduce its footprint."
                )
        return None

    def _validate_object_within_arena(self, obj: _ObjectRow) -> str | None:
        if obj.x is None or obj.y is None:
            return f'Object "{obj.object_id}" is missing primary coordinates.'
        if not self._inside_arena(float(obj.x), float(obj.y)):
            return (
                f'Object "{obj.object_id}" has primary coordinates outside the arena.'
            )
        if obj.object_type == "Border segment":
            if obj.x2 is None or obj.y2 is None:
                return f'Border segment "{obj.object_id}" is missing an end coordinate.'
            if not self._inside_arena(float(obj.x2), float(obj.y2)):
                return (
                    f'Border segment "{obj.object_id}" has an end coordinate outside '
                    "the arena."
                )
            return None
        if obj.object_type == "Source group":
            return self._validate_source_group_members_within_arena(obj)
        return None

    def _validate_configuration_for_storage(self) -> str | None:
        for obj in self._objects:
            error = self._validate_object_within_arena(obj)
            if error is not None:
                return error
        return None

    def _pick_object_at(self, x: float, y: float) -> _ObjectRow | None:
        nearest: tuple[float, _ObjectRow] | None = None
        group_lookup = {
            obj.object_id: obj
            for obj in self._objects
            if obj.object_type == "Source group"
        }
        member_data = self.source_group_member_source.data
        member_xs = member_data.get("x", [])
        member_ys = member_data.get("y", [])
        member_rs = member_data.get("r", [])
        member_parent_ids = member_data.get("parent_id", [])
        for member_x, member_y, member_radius, parent_id in zip(
            member_xs, member_ys, member_rs, member_parent_ids
        ):
            parent = group_lookup.get(str(parent_id))
            if parent is None:
                continue
            distance = math.hypot(x - float(member_x), y - float(member_y))
            tolerance = max(float(member_radius or 0.0), 0.004)
            if distance <= tolerance and (nearest is None or distance < nearest[0]):
                nearest = (distance, parent)
        for obj in self._objects:
            if obj.object_type == "Border segment":
                x0 = float(obj.x or 0.0)
                y0 = float(obj.y or 0.0)
                x1 = float(obj.x2 or 0.0)
                y1 = float(obj.y2 or 0.0)
                dx = x1 - x0
                dy = y1 - y0
                segment_len_sq = dx * dx + dy * dy
                if segment_len_sq <= 0:
                    continue
                t = max(0.0, min(1.0, ((x - x0) * dx + (y - y0) * dy) / segment_len_sq))
                px = x0 + t * dx
                py = y0 + t * dy
                distance = math.hypot(x - px, y - py)
                tolerance = max(float(obj.width or 0.001) * 3.0, 0.008)
                if distance <= tolerance and (nearest is None or distance < nearest[0]):
                    nearest = (distance, obj)
                continue

            ox = float(obj.x or 0.0)
            oy = float(obj.y or 0.0)
            if obj.object_type == "Source group":
                radius = max(
                    float(obj.distribution_scale_x or 0.008),
                    float(obj.distribution_scale_y or 0.008),
                    float(obj.radius or 0.003),
                    0.006,
                )
            else:
                radius = max(float(obj.radius or 0.008), 0.004)
            distance = math.hypot(x - ox, y - oy)
            if distance <= radius and (nearest is None or distance < nearest[0]):
                nearest = (distance, obj)

        return None if nearest is None else nearest[1]

    def _on_tap(self, event: Tap) -> None:
        x = round(float(event.x), 4)
        y = round(float(event.y), 4)
        if self.select_mode.value:
            selected = self._pick_object_at(x, y)
            if selected is None:
                self.status.object = "No object found at that location."
                return
            self._set_selected_object(selected.object_id)
            self.status.object = f'Selected "{selected.object_id}" from canvas.'
            return
        if not self._inside_arena(x, y):
            self.status.object = "Click inside arena bounds."
            return

        object_type = self.object_type.value
        if object_type == "Border segment":
            self._tap_border(x, y)
            return
        self._add_point_object(
            object_type=object_type,
            x=x,
            y=y,
            radius=round(float(self.object_radius.value) / 1000.0, 4),
            color=self.object_color.value,
        )

    def _tap_border(self, x: float, y: float) -> None:
        if self._border_start is None:
            self._border_start = (x, y)
            self._show_border_preview(x=x, y=y, color=self.object_color.value)
            self._update_insert_hint()
            return
        x0, y0 = self._border_start
        self._border_start = None
        self._clear_border_preview()
        self._add_border_object(
            x0=x0,
            y0=y0,
            x1=x,
            y1=y,
            width=round(float(self.border_width.value) / 1000.0, 4),
            color=self.object_color.value,
        )
        self._update_insert_hint()

    def _next_id(self, prefix: str) -> str:
        object_id = f"{prefix}_{self._counter:03d}"
        self._counter += 1
        return object_id

    def _add_point_object(
        self, *, object_type: str, x: float, y: float, radius: float, color: str
    ) -> None:
        if object_type == "Source unit":
            object_id = self._next_id("food")
            candidate = _ObjectRow(
                object_id=object_id,
                object_type="Source unit",
                x=x,
                y=y,
                radius=radius,
                color=color,
                amount=0.0,
                can_be_carried=False,
                can_be_displaced=False,
                regeneration=False,
                odor_id=None,
                odor_intensity=None,
                odor_spread=None,
                substrate_type="standard",
                substrate_quality=1.0,
            )
            validation_error = self._validate_object_within_arena(candidate)
            if validation_error is not None:
                self.status.object = validation_error
                return
            fill_color, line_color, fill_alpha, line_alpha, line_width = (
                _source_visual_state(amount=0.0, color=color)
            )
            self._append_source_row(
                self.food_source,
                {
                    "x": x,
                    "y": y,
                    "r": radius,
                    "fill_color": fill_color,
                    "line_color": line_color,
                    "id": object_id,
                    "fill_alpha": fill_alpha,
                    "line_alpha": line_alpha,
                    "line_width": line_width,
                },
            )
            self._objects.append(candidate)
        elif object_type == "Source group":
            object_id = self._next_id("group")
            shape = _normalize_group_shape(self.group_shape.value)
            scale_x = self._group_scale_from_display_mm(
                shape, float(self.group_spread_x.value)
            )
            scale_y = (
                scale_x
                if shape == "circle"
                else self._group_scale_from_display_mm(
                    shape, float(self.group_spread_y.value)
                )
            )
            candidate = _ObjectRow(
                object_id=object_id,
                object_type="Source group",
                x=x,
                y=y,
                radius=radius,
                color=color,
                amount=0.0,
                odor_id=None,
                odor_intensity=None,
                odor_spread=None,
                can_be_carried=False,
                can_be_displaced=False,
                regeneration=False,
                substrate_type="standard",
                substrate_quality=1.0,
                distribution_n=int(self.group_count.value),
                distribution_shape=shape,
                distribution_mode=self.group_mode.value,
                distribution_show_shape=True,
                distribution_scale_x=scale_x,
                distribution_scale_y=scale_y,
            )
            validation_error = self._validate_object_within_arena(candidate)
            if validation_error is not None:
                self.status.object = validation_error
                return
            self._objects.append(candidate)
            self._rebuild_sources()
        else:
            raise ValueError(f"Unsupported object type: {object_type}")

        self.status.object = f"Added {object_type.lower()} at ({x:.3f}, {y:.3f})."
        self._refresh_table()
        self._refresh_object_controls(selected_object_id=object_id)

    def _add_border_object(
        self, *, x0: float, y0: float, x1: float, y1: float, width: float, color: str
    ) -> None:
        object_id = self._next_id("border")
        candidate = _ObjectRow(
            object_id=object_id,
            object_type="Border segment",
            x=x0,
            y=y0,
            x2=x1,
            y2=y1,
            width=width,
            color=color,
        )
        validation_error = self._validate_object_within_arena(candidate)
        if validation_error is not None:
            self.status.object = validation_error
            return
        self._append_source_row(
            self.border_source,
            {
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
                "w": max(1, int(width * 1500)),
                "color": color,
                "id": object_id,
            },
        )
        self._objects.append(candidate)
        self.status.object = (
            f"Added border segment from ({x0:.3f}, {y0:.3f}) to ({x1:.3f}, {y1:.3f})."
        )
        self._refresh_table()
        self._refresh_object_controls(selected_object_id=object_id)

    def _show_border_preview(self, *, x: float, y: float, color: str) -> None:
        self.border_preview_source.data = {"x": [x], "y": [y], "color": [color]}

    def _clear_border_preview(self) -> None:
        self.border_preview_source.data = {"x": [], "y": [], "color": []}

    def _append_source_row(
        self, source: ColumnDataSource, row: dict[str, object]
    ) -> None:
        data = {key: list(value) for key, value in source.data.items()}
        for key, value in row.items():
            data[key].append(value)
        source.data = data

    def _append_rows(
        self, source: ColumnDataSource, rows: list[dict[str, object]]
    ) -> None:
        for row in rows:
            self._append_source_row(source, row)

    def _empty_source_group_circle_data(self) -> dict[str, list[object]]:
        return {
            "x": [],
            "y": [],
            "r": [],
            "color": [],
            "fill_alpha": [],
            "line_alpha": [],
            "id": [],
        }

    def _empty_source_group_xy_data(self) -> dict[str, list[object]]:
        return {
            "x": [],
            "y": [],
            "w": [],
            "h": [],
            "color": [],
            "fill_alpha": [],
            "line_alpha": [],
            "id": [],
        }

    def _empty_source_group_circle_highlight_data(self) -> dict[str, list[object]]:
        return {"x": [], "y": [], "r": [], "color": []}

    def _empty_source_group_xy_highlight_data(self) -> dict[str, list[object]]:
        return {"x": [], "y": [], "w": [], "h": [], "color": []}

    def _empty_food_grid_cell_data(self) -> dict[str, list[object]]:
        return {
            "x": [],
            "y": [],
            "w": [],
            "h": [],
            "fill_color": [],
            "line_color": [],
            "fill_alpha": [],
            "line_alpha": [],
            "line_width": [],
        }

    def _rebuild_food_grid_cells(self) -> None:
        if not bool(self.food_grid_enabled.value):
            self._food_grid_cell_source.data = self._empty_food_grid_cell_data()
            return
        width, height = self._arena_dimensions()
        nx = max(int(self.food_grid_dims_x.value), 1)
        ny = max(int(self.food_grid_dims_y.value), 1)
        if width <= 0 or height <= 0:
            self._food_grid_cell_source.data = self._empty_food_grid_cell_data()
            return
        cell_w = width / nx
        cell_h = height / ny
        xs = np.linspace(-width / 2 + cell_w / 2, width / 2 - cell_w / 2, nx)
        ys = np.linspace(-height / 2 + cell_h / 2, height / 2 - cell_h / 2, ny)
        fill_color = _mix_hex_colors(self.food_grid_color.value, "#ffffff", 0.32)
        line_color = _mix_hex_colors(self.food_grid_color.value, "#111111", 0.20)
        rows: list[dict[str, object]] = []
        for center_x in xs:
            for center_y in ys:
                if self.arena_shape.value != "rectangular":
                    nx_ellipse = float(center_x) / (width / 2)
                    ny_ellipse = float(center_y) / (height / 2)
                    if (nx_ellipse * nx_ellipse + ny_ellipse * ny_ellipse) > 1.0:
                        continue
                rows.append(
                    {
                        "x": float(center_x),
                        "y": float(center_y),
                        "w": float(cell_w),
                        "h": float(cell_h),
                        "fill_color": fill_color,
                        "line_color": line_color,
                        "fill_alpha": 0.10,
                        "line_alpha": 0.42,
                        "line_width": 0.9,
                    }
                )
        data = self._empty_food_grid_cell_data()
        for row in rows:
            for key, value in row.items():
                data[key].append(value)
        self._food_grid_cell_source.data = data

    def _empty_source_group_member_data(self) -> dict[str, list[object]]:
        return {
            "x": [],
            "y": [],
            "r": [],
            "fill_color": [],
            "line_color": [],
            "fill_alpha": [],
            "line_alpha": [],
            "line_width": [],
            "parent_id": [],
        }

    def _group_member_positions(self, obj: _ObjectRow) -> list[tuple[float, float]]:
        if obj.object_type != "Source group" or obj.x is None or obj.y is None:
            return []
        count = int(obj.distribution_n or 0)
        if count <= 0:
            return []
        distribution = Spatial_Distro(
            N=count,
            shape=_normalize_group_shape(obj.distribution_shape),
            mode=str(obj.distribution_mode or "uniform"),
            loc=(float(obj.x), float(obj.y)),
            scale=(
                float(obj.distribution_scale_x or 0.012),
                float(obj.distribution_scale_y or 0.012),
            ),
        )
        state = np.random.get_state()
        try:
            np.random.seed(
                _stable_preview_seed(
                    obj.object_id,
                    obj.distribution_mode,
                    obj.distribution_shape,
                    obj.distribution_n,
                    obj.x,
                    obj.y,
                    obj.distribution_scale_x,
                    obj.distribution_scale_y,
                )
            )
            return [
                (float(member_x), float(member_y))
                for member_x, member_y in distribution()
            ]
        except Exception:
            return []
        finally:
            np.random.set_state(state)

    def _build_group_member_rows(self, obj: _ObjectRow) -> list[dict[str, object]]:
        positions = self._group_member_positions(obj)
        if not positions:
            return []
        base_color = str(obj.color or "#4caf50")
        fill_color = _mix_hex_colors(base_color, "#ffffff", 0.18)
        line_color = _mix_hex_colors(base_color, "#111111", 0.04)
        radius = max(float(obj.radius or 0.003), 0.0015)
        return [
            {
                "x": member_x,
                "y": member_y,
                "r": radius,
                "fill_color": fill_color,
                "line_color": line_color,
                "fill_alpha": 0.78,
                "line_alpha": 0.92,
                "line_width": 1.4,
                "parent_id": obj.object_id,
            }
            for member_x, member_y in positions
        ]

    def _on_clear_last(self, _: object) -> None:
        if self._border_start is not None:
            self._border_start = None
            self._clear_border_preview()
            self._update_insert_hint()
            self.status.object = "Cancelled pending border segment."
            return
        if not self._objects:
            self.status.object = "Nothing to undo."
            return
        self._objects.pop()
        self._rebuild_sources()
        self._refresh_table()
        self._refresh_object_controls()
        self.status.object = "Removed last object."

    def _show_reset_confirmation(self) -> None:
        self._pending_reset_confirmation = True
        self.reset_confirm_text.object = (
            "This action will clear all placed objects and recreate the Env registry "
            "(resetting `Env.txt` to defaults). Continue?"
        )
        self.reset_confirm_panel.visible = True
        self.status.object = (
            "Reset requested. Confirm to clear the builder state and recreate Env.txt."
        )

    def _clear_reset_confirmation(self) -> None:
        self._pending_reset_confirmation = False
        self.reset_confirm_text.object = ""
        self.reset_confirm_panel.visible = False

    def _on_confirm_reset_configurations(self, _: object) -> None:
        if not self._pending_reset_confirmation:
            return
        self._on_clear_all(None)

    def _on_cancel_reset_configurations(self, _: object) -> None:
        if not self._pending_reset_confirmation:
            return
        self._clear_reset_confirmation()
        self.status.object = "Reset configurations cancelled."

    def _clear_arena_contents(self) -> None:
        self._objects.clear()
        self._border_start = None
        self._counter = 1
        self._clear_border_preview()
        self.food_source.data = {
            "x": [],
            "y": [],
            "r": [],
            "fill_color": [],
            "line_color": [],
            "id": [],
            "fill_alpha": [],
            "line_alpha": [],
            "line_width": [],
        }
        self.odor_layer_source.data = {
            "x": [],
            "y": [],
            "r": [],
            "color": [],
            "fill_alpha": [],
            "id": [],
        }
        self.odor_peak_source.data = {
            "x": [],
            "y": [],
            "r": [],
            "color": [],
            "fill_alpha": [],
            "id": [],
        }
        self.source_group_circle_source.data = self._empty_source_group_circle_data()
        self.source_group_ellipse_source.data = self._empty_source_group_xy_data()
        self.source_group_rect_source.data = self._empty_source_group_xy_data()
        self.source_group_member_source.data = self._empty_source_group_member_data()
        self.border_source.data = {
            "x0": [],
            "y0": [],
            "x1": [],
            "y1": [],
            "w": [],
            "color": [],
            "id": [],
        }
        self._refresh_table()
        self._refresh_object_controls()
        self._update_insert_hint()
        self._sync_scape_preview()

    def _on_clear_arena(self, _: object) -> None:
        self._clear_overwrite_confirmation()
        self._clear_reset_confirmation()
        self._clear_arena_contents()
        self.status.object = "Cleared all arena objects."

    def _on_clear_all(self, _: object) -> None:
        if not self._pending_reset_confirmation:
            self._show_reset_confirmation()
            return
        self._clear_reset_confirmation()
        self._clear_arena_contents()
        try:
            reg_config.resetConfs(conftypes=["Env"], recreate=True)
        except Exception as exc:
            self._refresh_preset_controls()
            self.status.object = (
                "Cleared all placed objects, but failed to recreate the Env registry: "
                f"{exc}"
            )
            return

        self._refresh_preset_controls()
        self.status.object = (
            "Cleared all placed objects and recreated the Env registry."
        )

    def _rebuild_sources(self) -> None:
        self._clear_border_preview()
        self.food_source.data = {
            "x": [],
            "y": [],
            "r": [],
            "fill_color": [],
            "line_color": [],
            "id": [],
            "fill_alpha": [],
            "line_alpha": [],
            "line_width": [],
        }
        self.odor_layer_source.data = {
            "x": [],
            "y": [],
            "r": [],
            "color": [],
            "fill_alpha": [],
            "id": [],
        }
        self.odor_peak_source.data = {
            "x": [],
            "y": [],
            "r": [],
            "color": [],
            "fill_alpha": [],
            "id": [],
        }
        self.source_group_circle_source.data = self._empty_source_group_circle_data()
        self.source_group_ellipse_source.data = self._empty_source_group_xy_data()
        self.source_group_rect_source.data = self._empty_source_group_xy_data()
        self.source_group_member_source.data = self._empty_source_group_member_data()
        self.border_source.data = {
            "x0": [],
            "y0": [],
            "x1": [],
            "y1": [],
            "w": [],
            "color": [],
            "id": [],
        }
        for obj in self._objects:
            if obj.object_type == "Source unit":
                fill_color, line_color, fill_alpha, line_alpha, line_width = (
                    _source_visual_state(amount=obj.amount, color=obj.color)
                )
                self._append_source_row(
                    self.food_source,
                    {
                        "x": obj.x,
                        "y": obj.y,
                        "r": obj.radius,
                        "fill_color": fill_color,
                        "line_color": line_color,
                        "id": obj.object_id,
                        "fill_alpha": fill_alpha,
                        "line_alpha": line_alpha,
                        "line_width": line_width,
                    },
                )
                self._append_rows(
                    self.odor_layer_source,
                    _build_odor_layers(
                        x=obj.x,
                        y=obj.y,
                        source_radius=obj.radius,
                        odor_id=obj.odor_id,
                        odor_intensity=obj.odor_intensity,
                        odor_spread=obj.odor_spread,
                        color=obj.color,
                        source_id=obj.object_id,
                    ),
                )
                odor_peak = _build_odor_peak(
                    x=obj.x,
                    y=obj.y,
                    source_radius=obj.radius,
                    odor_id=obj.odor_id,
                    odor_intensity=obj.odor_intensity,
                    odor_spread=obj.odor_spread,
                    color=obj.color,
                    source_id=obj.object_id,
                )
                if odor_peak is not None:
                    self._append_source_row(self.odor_peak_source, odor_peak)
            elif obj.object_type == "Source group":
                member_radius = max(float(obj.radius or 0.003), 0.002)
                for member_x, member_y in self._group_member_positions(obj):
                    self._append_rows(
                        self.odor_layer_source,
                        _build_odor_layers(
                            x=member_x,
                            y=member_y,
                            source_radius=member_radius,
                            odor_id=obj.odor_id,
                            odor_intensity=obj.odor_intensity,
                            odor_spread=obj.odor_spread,
                            color=obj.color,
                            source_id=obj.object_id,
                        ),
                    )
                    odor_peak = _build_odor_peak(
                        x=member_x,
                        y=member_y,
                        source_radius=member_radius,
                        odor_id=obj.odor_id,
                        odor_intensity=obj.odor_intensity,
                        odor_spread=obj.odor_spread,
                        color=obj.color,
                        source_id=obj.object_id,
                    )
                    if odor_peak is not None:
                        self._append_source_row(self.odor_peak_source, odor_peak)
                if bool(
                    True
                    if obj.distribution_show_shape is None
                    else obj.distribution_show_shape
                ):
                    width = max(float(obj.distribution_scale_x or 0.012) * 2.0, 0.002)
                    height = max(float(obj.distribution_scale_y or 0.012) * 2.0, 0.002)
                    shape = _normalize_group_shape(obj.distribution_shape)
                    if shape == "circle":
                        self._append_source_row(
                            self.source_group_circle_source,
                            {
                                "x": obj.x,
                                "y": obj.y,
                                "r": max(width, height) / 2.0,
                                "color": obj.color,
                                "fill_alpha": 0.08,
                                "line_alpha": 0.9,
                                "id": obj.object_id,
                            },
                        )
                    elif shape == "oval":
                        self._append_source_row(
                            self.source_group_ellipse_source,
                            {
                                "x": obj.x,
                                "y": obj.y,
                                "w": width,
                                "h": height,
                                "color": obj.color,
                                "fill_alpha": 0.08,
                                "line_alpha": 0.9,
                                "id": obj.object_id,
                            },
                        )
                    else:
                        self._append_source_row(
                            self.source_group_rect_source,
                            {
                                "x": obj.x,
                                "y": obj.y,
                                "w": width,
                                "h": height,
                                "color": obj.color,
                                "fill_alpha": 0.08,
                                "line_alpha": 0.9,
                                "id": obj.object_id,
                            },
                        )
                self._append_rows(
                    self.source_group_member_source,
                    self._build_group_member_rows(obj),
                )
            else:
                self._append_source_row(
                    self.border_source,
                    {
                        "x0": obj.x,
                        "y0": obj.y,
                        "x1": obj.x2,
                        "y1": obj.y2,
                        "w": max(1, int((obj.width or 0.001) * 1500)),
                        "color": obj.color,
                        "id": obj.object_id,
                    },
                )
        self._sync_scape_preview()

    def _refresh_table(self) -> None:
        rows = [
            {
                "id": obj.object_id,
                "type": obj.object_type,
                "x": obj.x,
                "y": obj.y,
                "x2": obj.x2,
                "y2": obj.y2,
                "radius": obj.radius,
                "spread_x": obj.distribution_scale_x,
                "spread_y": obj.distribution_scale_y,
                "count": obj.distribution_n,
                "width": obj.width,
                "color": obj.color,
                "amount": obj.amount,
                "odor_id": obj.odor_id,
            }
            for obj in self._objects
        ]
        self.table.value = pd.DataFrame(rows)

    def _build_export_config(self) -> dict[str, object]:
        base_config = util.AttrDict(self._loaded_config).get_copy()
        source_units: dict[str, dict[str, object]] = {}
        source_groups: dict[str, dict[str, object]] = {}
        border_list: dict[str, dict[str, object]] = {}
        food_grid: dict[str, object] | None = None
        odorscape: dict[str, object] | None = None
        windscape: dict[str, object] | None = None
        thermoscape: dict[str, object] | None = None

        for obj in self._objects:
            if obj.object_type == "Source unit":
                source_units[obj.object_id] = {
                    "pos": [obj.x, obj.y],
                    "radius": obj.radius,
                    "amount": obj.amount if obj.amount is not None else 0.0,
                    "can_be_carried": bool(obj.can_be_carried),
                    "can_be_displaced": bool(obj.can_be_displaced),
                    "regeneration": bool(obj.regeneration),
                    "odor": {
                        "id": obj.odor_id,
                        "intensity": obj.odor_intensity,
                        "spread": obj.odor_spread,
                    },
                    "substrate": {
                        "type": obj.substrate_type or "standard",
                        "quality": obj.substrate_quality
                        if obj.substrate_quality is not None
                        else 1.0,
                    },
                    "color": obj.color,
                }
            elif obj.object_type == "Source group":
                distribution_mode = obj.distribution_mode or "uniform"
                source_groups[obj.object_id] = {
                    "radius": obj.radius,
                    "amount": obj.amount if obj.amount is not None else 0.0,
                    "can_be_carried": bool(obj.can_be_carried),
                    "can_be_displaced": bool(obj.can_be_displaced),
                    "regeneration": bool(obj.regeneration),
                    "distribution": {
                        "N": obj.distribution_n
                        if obj.distribution_n is not None
                        else 30,
                        "loc": [obj.x, obj.y],
                        "mode": obj.distribution_mode or "uniform",
                        "scale": [
                            obj.distribution_scale_x
                            if obj.distribution_scale_x is not None
                            else 0.012,
                            obj.distribution_scale_y
                            if obj.distribution_scale_y is not None
                            else 0.012,
                        ],
                        "shape": _normalize_group_shape(obj.distribution_shape),
                    },
                    "odor": {
                        "id": obj.odor_id,
                        "intensity": obj.odor_intensity,
                        "spread": obj.odor_spread,
                    },
                    "substrate": {
                        "type": obj.substrate_type or "standard",
                        "quality": obj.substrate_quality
                        if obj.substrate_quality is not None
                        else 1.0,
                    },
                    "color": obj.color,
                }
            elif obj.object_type == "Border segment":
                border_list[obj.object_id] = {
                    "vertices": [[obj.x, obj.y], [obj.x2, obj.y2]],
                    "width": obj.width,
                    "color": obj.color,
                }

        if self.food_grid_enabled.value:
            food_grid = {
                "unique_id": "FoodGrid",
                "color": self.food_grid_color.value,
                "fixed_max": bool(self.food_grid_fixed_max.value),
                "grid_dims": [
                    int(self.food_grid_dims_x.value),
                    int(self.food_grid_dims_y.value),
                ],
                "initial_value": float(self.food_grid_initial_value.value),
                "substrate": {
                    "type": self.food_grid_substrate_type.value or "standard",
                    "quality": float(self.food_grid_substrate_quality.value),
                },
            }

        if self.odorscape_enabled.value:
            odorscape = {
                "unique_id": f"{self.odorscape_mode.value}ValueLayer",
                "odorscape": self.odorscape_mode.value,
                "color": self.odorscape_color.value,
                "grid_dims": [
                    int(self.odorscape_grid_dims_x.value),
                    int(self.odorscape_grid_dims_y.value),
                ],
                "initial_value": float(self.odorscape_initial_value.value),
                "fixed_max": bool(self.odorscape_fixed_max.value),
            }
            if self.odorscape_mode.value == "Diffusion":
                odorscape["evap_const"] = float(self.odorscape_evap_const.value)
                odorscape["gaussian_sigma"] = (
                    float(self.odorscape_sigma_x.value),
                    float(self.odorscape_sigma_y.value),
                )

        if self.windscape_enabled.value:
            puff_rows = self.wind_puffs_table.value
            puffs: dict[str, dict[str, object]] = {}
            if isinstance(puff_rows, pd.DataFrame):
                for row in puff_rows.to_dict(orient="records"):
                    puff_id = str(row.get("id") or "").strip()
                    if not puff_id:
                        continue
                    N_value = row.get("N")
                    puffs[puff_id] = {
                        "duration": float(row.get("duration", 1.0)),
                        "speed": float(row.get("speed", 10.0)),
                        "direction": float(row.get("direction", 0.0)),
                        "start_time": float(row.get("start_time", 0.0)),
                        "N": (
                            int(N_value)
                            if pd.notna(N_value) and N_value not in ("", None)
                            else None
                        ),
                        "interval": float(row.get("interval", 5.0)),
                    }
            windscape = {
                "unique_id": "WindScape",
                "color": self.windscape_color.value,
                "wind_direction": _deg_to_rad(float(self.windscape_direction.value)),
                "wind_speed": float(self.windscape_speed.value),
                "puffs": puffs,
            }

        if self.thermoscape_enabled.value:
            thermo_rows = self.thermo_sources_table.value
            thermo_sources: dict[str, tuple[float, float]] = {}
            thermo_source_dtemps: dict[str, float] = {}
            if isinstance(thermo_rows, pd.DataFrame):
                for row in thermo_rows.to_dict(orient="records"):
                    source_id = str(row.get("id") or "").strip()
                    if not source_id:
                        continue
                    thermo_sources[source_id] = (
                        float(row.get("x", 0.0)),
                        float(row.get("y", 0.0)),
                    )
                    thermo_source_dtemps[source_id] = float(row.get("dTemp", 0.0))
            thermoscape = {
                "unique_id": "ThermoScape",
                "plate_temp": float(self.thermoscape_plate_temp.value),
                "spread": float(self.thermoscape_spread.value),
                "thermo_sources": thermo_sources,
                "thermo_source_dTemps": thermo_source_dtemps,
            }

        arena_width, arena_height = self._arena_dimensions()
        base_config["arena"] = {
            "geometry": self.arena_shape.value,
            "dims": [arena_width, arena_height],
            "torus": bool(self.arena_torus.value),
        }
        base_config["food_params"] = {
            "source_units": source_units,
            "source_groups": source_groups,
            "food_grid": food_grid,
        }
        base_config["border_list"] = border_list
        base_config["odorscape"] = odorscape
        base_config["windscape"] = windscape
        base_config["thermoscape"] = thermoscape
        base_config.pop("obstacles", None)
        return dict(base_config)

    def _build_registry_config(self) -> util.AttrDict:
        return _translate_builder_environment_payload(self._build_export_config())

    def _export_json(self) -> io.StringIO:
        payload = self._build_export_config()
        return io.StringIO(json.dumps(payload, indent=2))


class _InsertSection(param.Parameterized):
    object_type = param.Selector(
        default="Source unit",
        objects=["Source unit", "Source group", "Border segment"],
        label="Insert object",
        doc="Choose which canonical EnvConf object family to place on the canvas: source unit, source group, or border segment.",
    )
    radius = param.Number(
        default=3.0,
        bounds=(1.0, 30.0),
        label="Source unit radius (mm)",
        doc="The spatial radius of the source unit core. Builder UI is shown in millimeters.",
    )
    border_width = param.Number(
        default=1.0,
        bounds=(0.5, 10.0),
        label="Border width (mm)",
        doc="The width of the border segment line. Builder UI is shown in millimeters.",
    )
    color = param.String(
        default="#4caf50",
        label="Object color",
        doc="Preview color used for the object and exported as the object's color.",
    )
    enabled = param.Boolean(
        default=False,
        label="Enable food grid",
        doc="Adds an optional FoodGrid covering the arena.",
    )


class _OdorScapeSection(param.Parameterized):
    enabled = param.Boolean(
        default=False,
        label="Enable odorscape",
        doc="The sensory odor landscape in the arena. Preview contours appear only when at least one placed source has odor configured.",
    )
    mode = param.Selector(
        default="Gaussian",
        objects=["Gaussian", "Diffusion", "Analytical"],
        label="Odorscape mode",
        doc="The odorscape algorithm.",
    )
    color = param.String(
        default="#4caf50",
        label="Odorscape color",
        doc="Preview and export color for the odorscape layer.",
    )
    grid_dims_x = param.Integer(
        default=51,
        bounds=(1, None),
        label="Odorscape grid X",
        doc="The spatial resolution of the odorscape grid in the X direction.",
    )
    grid_dims_y = param.Integer(
        default=51,
        bounds=(1, None),
        label="Odorscape grid Y",
        doc="The spatial resolution of the odorscape grid in the Y direction.",
    )
    initial_value = param.Number(
        default=0.0,
        label="Odorscape initial value",
        doc="Initial value over the odorscape grid before source contributions are applied.",
    )
    fixed_max = param.Boolean(
        default=False,
        label="Odorscape fixed max",
        doc="Whether the maximum odorscape value is kept constant.",
    )
    evap_const = param.Number(
        default=0.9,
        bounds=(0.0, None),
        label="Diffusion evaporation",
        doc="The evaporation constant of the diffusion algorithm.",
    )
    sigma_x = param.Number(
        default=0.95,
        bounds=(0.0, None),
        label="Diffusion sigma X",
        doc="The X component of the gaussian diffusion sigma.",
    )
    sigma_y = param.Number(
        default=0.95,
        bounds=(0.0, None),
        label="Diffusion sigma Y",
        doc="The Y component of the gaussian diffusion sigma.",
    )


class _WindScapeSection(param.Parameterized):
    enabled = param.Boolean(
        default=False,
        label="Enable windscape",
        doc="The wind landscape in the arena. Wind arrows appear in the preview only when wind speed is greater than zero.",
    )
    color = param.String(
        default="#ff0000",
        label="Wind color",
        doc="Preview and export color for wind arrows.",
    )
    direction = param.Number(
        default=180.0,
        label="Wind direction (deg)",
        doc="The absolute polar direction of the wind or air puff. Builder UI edits this in degrees and exports it in radians.",
    )
    speed = param.Number(
        default=0.0,
        bounds=(0.0, None),
        label="Wind speed (m/s)",
        doc="The speed of the wind or air puff.",
    )


class _ThermoScapeSection(param.Parameterized):
    enabled = param.Boolean(
        default=False,
        label="Enable thermoscape",
        doc="The thermal landscape in the arena. Preview appears only after at least one thermal source row is added.",
    )
    plate_temp = param.Number(
        default=22.0,
        label="Plate temperature (deg C)",
        doc="Baseline plate temperature in degrees Celsius.",
    )
    spread = param.Number(
        default=0.1,
        bounds=(0.0, None),
        label="Thermal spread",
        doc="Spread parameter controlling how broadly each thermal source diffuses across the arena.",
    )


class _PresetSection(param.Parameterized):
    preset_name = param.String(
        default="environment_builder_config",
        label="Unique ID",
        doc="Name used for the saved workspace JSON file and for the Env registry entry created from this builder preset.",
    )


_FAMILY_BOX_STYLES = {
    "background": "rgba(252, 252, 253, 0.99)",
    "border": "1px solid rgba(90, 71, 96, 0.08)",
    "border-radius": "10px",
    "padding": "10px 12px 8px 12px",
}

_FAMILY_BOX_COMPACT_STYLES = {
    **_FAMILY_BOX_STYLES,
    "padding": "5px 12px 1px 12px",
}

_FAMILY_BG = "rgba(252, 252, 253, 0.99)"
_FAMILY_FIELD_STYLESHEET = f"""
:host {{
  --lw-family-bg: {_FAMILY_BG};
}}

label,
legend,
.bk-input-group label,
.bk-clearfix label,
.bk-input-group legend {{
  background: var(--lw-family-bg) !important;
}}
""".strip()

_FAMILY_TITLE_STYLES = {
    "color": "#4f2f5f",
    "font-weight": "700",
    "margin": "0 0 6px 0",
}


def _sync_param_with_widget(
    section: param.Parameterized,
    parameter_name: str,
    widget: pn.viewable.Viewable,
) -> None:
    state = {"syncing_param": False, "syncing_widget": False}

    def _from_param(event: param.parameterized.Event) -> None:
        if state["syncing_widget"] or not hasattr(widget, "value"):
            return
        if getattr(widget, "value") == event.new:
            return
        state["syncing_param"] = True
        try:
            widget.value = event.new
        finally:
            state["syncing_param"] = False

    def _from_widget(event: param.parameterized.Event) -> None:
        if state["syncing_param"]:
            return
        if getattr(section, parameter_name) == event.new:
            return
        state["syncing_widget"] = True
        try:
            setattr(section, parameter_name, event.new)
        finally:
            state["syncing_widget"] = False

    section.param.watch(_from_param, parameter_name)
    widget.param.watch(_from_widget, "value")


class _EnvironmentBuilderV2Controller:
    def __init__(self) -> None:
        self.runtime = _EnvironmentBuilderController()
        self._syncing_area_dims = False
        self._syncing_group_scale = False
        self._syncing_food_grid_dims = False
        self.runtime.clear_all_btn.name = "Reset configurations"
        self.runtime.selected_color.width = 52
        self.runtime.selected_color.sizing_mode = "fixed"
        self.runtime.select_mode.width = None
        self.runtime.select_mode.sizing_mode = "stretch_width"
        self.runtime.clear_last_btn.width = None
        self.runtime.clear_last_btn.sizing_mode = "stretch_width"
        arena_width, arena_height = self.runtime._arena_dimensions()
        self.arena = Area(
            dims=(arena_width, arena_height),
            geometry=self.runtime.arena_shape.value,
            torus=self.runtime.arena_torus.value,
        )
        self.insert = _InsertSection(
            object_type=self.runtime.object_type.value,
            radius=self.runtime.object_radius.value,
            border_width=self.runtime.border_width.value,
            color=self.runtime.object_color.value,
            enabled=self.runtime.food_grid_enabled.value,
        )
        self.group_distribution = Spatial_Distro(
            N=self.runtime.group_count.value,
            shape=self.runtime.group_shape.value,
            mode=self.runtime.group_mode.value,
            scale=(
                self.runtime._group_scale_from_display_mm(
                    self.runtime.group_shape.value,
                    float(self.runtime.group_spread_x.value),
                ),
                self.runtime._group_scale_from_display_mm(
                    self.runtime.group_shape.value,
                    float(self.runtime.group_spread_y.value),
                ),
            ),
        )
        self.food_grid = FoodGrid(
            color=self.runtime.food_grid_color.value,
            grid_dims=(
                self.runtime.food_grid_dims_x.value,
                self.runtime.food_grid_dims_y.value,
            ),
            initial_value=self.runtime.food_grid_initial_value.value,
            substrate=Substrate(
                type=self.runtime.food_grid_substrate_type.value,
                quality=self.runtime.food_grid_substrate_quality.value,
            ),
            fixed_max=self.runtime.food_grid_fixed_max.value,
        )
        self.food_grid_dims_x = pn.widgets.IntSlider(
            name="Grid X",
            start=1,
            end=500,
            step=1,
            value=int(self.food_grid.grid_dims[0]),
            sizing_mode="stretch_width",
        )
        self.food_grid_dims_y = pn.widgets.IntSlider(
            name="Grid Y",
            start=1,
            end=500,
            step=1,
            value=int(self.food_grid.grid_dims[1]),
            sizing_mode="stretch_width",
        )
        self.food_grid_substrate_quality = pn.widgets.FloatInput(
            name="Grid substrate quality",
            value=float(self.food_grid.substrate.quality),
            step=0.01,
            start=0.0,
            end=1.0,
            sizing_mode="stretch_width",
        )
        self.food_grid_fixed_max = pn.widgets.Checkbox(
            name="Fixed max",
            value=bool(self.food_grid.fixed_max),
            sizing_mode="stretch_width",
        )
        self.food_grid_fixed_max_note = pn.pane.Markdown(
            "Keep the grid maximum fixed instead of letting it grow dynamically.",
            styles={"font-size": "11px", "color": "rgba(17, 17, 17, 0.68)"},
            margin=(0, 0, 4, 0),
        )
        self.odorscape = _OdorScapeSection(
            enabled=self.runtime.odorscape_enabled.value,
            mode=self.runtime.odorscape_mode.value,
            color=self.runtime.odorscape_color.value,
            grid_dims_x=self.runtime.odorscape_grid_dims_x.value,
            grid_dims_y=self.runtime.odorscape_grid_dims_y.value,
            initial_value=self.runtime.odorscape_initial_value.value,
            fixed_max=self.runtime.odorscape_fixed_max.value,
            evap_const=self.runtime.odorscape_evap_const.value,
            sigma_x=self.runtime.odorscape_sigma_x.value,
            sigma_y=self.runtime.odorscape_sigma_y.value,
        )
        self.odorscape_grid_dims_x = pn.widgets.IntSlider(
            name="Odorscape grid X",
            start=1,
            end=500,
            step=1,
            value=int(self.odorscape.grid_dims_x),
            sizing_mode="stretch_width",
        )
        self.odorscape_grid_dims_y = pn.widgets.IntSlider(
            name="Odorscape grid Y",
            start=1,
            end=500,
            step=1,
            value=int(self.odorscape.grid_dims_y),
            sizing_mode="stretch_width",
        )
        self.windscape = _WindScapeSection(
            enabled=self.runtime.windscape_enabled.value,
            color=self.runtime.windscape_color.value,
            direction=self.runtime.windscape_direction.value,
            speed=self.runtime.windscape_speed.value,
        )
        self.thermoscape = _ThermoScapeSection(
            enabled=self.runtime.thermoscape_enabled.value,
            plate_temp=self.runtime.thermoscape_plate_temp.value,
            spread=self.runtime.thermoscape_spread.value,
        )
        self.presets = _PresetSection(preset_name=self.runtime.preset_name.value)
        self.odorscape_enabled_switch = self._make_bound_switch(
            self.odorscape, "enabled"
        )
        self.windscape_enabled_switch = self._make_bound_switch(
            self.windscape, "enabled"
        )
        self.thermoscape_enabled_switch = self._make_bound_switch(
            self.thermoscape, "enabled"
        )
        self._bind_sections()
        self._sync_canvas_title()

    def _bind_sections(self) -> None:
        for section, parameter_name, widget in (
            (self.arena, "torus", self.runtime.arena_torus),
            (self.insert, "object_type", self.runtime.object_type),
            (self.insert, "radius", self.runtime.object_radius),
            (self.insert, "border_width", self.runtime.border_width),
            (self.insert, "color", self.runtime.object_color),
            (self.presets, "preset_name", self.runtime.preset_name),
            (self.odorscape, "enabled", self.runtime.odorscape_enabled),
            (self.odorscape, "mode", self.runtime.odorscape_mode),
            (self.odorscape, "color", self.runtime.odorscape_color),
            (self.odorscape, "grid_dims_x", self.runtime.odorscape_grid_dims_x),
            (self.odorscape, "grid_dims_y", self.runtime.odorscape_grid_dims_y),
            (self.odorscape, "initial_value", self.runtime.odorscape_initial_value),
            (self.odorscape, "fixed_max", self.runtime.odorscape_fixed_max),
            (self.odorscape, "evap_const", self.runtime.odorscape_evap_const),
            (self.odorscape, "sigma_x", self.runtime.odorscape_sigma_x),
            (self.odorscape, "sigma_y", self.runtime.odorscape_sigma_y),
            (self.windscape, "enabled", self.runtime.windscape_enabled),
            (self.windscape, "color", self.runtime.windscape_color),
            (self.windscape, "direction", self.runtime.windscape_direction),
            (self.windscape, "speed", self.runtime.windscape_speed),
            (self.thermoscape, "enabled", self.runtime.thermoscape_enabled),
            (self.thermoscape, "plate_temp", self.runtime.thermoscape_plate_temp),
            (self.thermoscape, "spread", self.runtime.thermoscape_spread),
        ):
            _sync_param_with_widget(section, parameter_name, widget)
        _sync_param_with_widget(self.arena, "geometry", self.runtime.arena_shape)
        _sync_param_with_widget(self.insert, "enabled", self.runtime.food_grid_enabled)
        _sync_param_with_widget(self.group_distribution, "N", self.runtime.group_count)
        _sync_param_with_widget(
            self.group_distribution, "shape", self.runtime.group_shape
        )
        _sync_param_with_widget(
            self.group_distribution, "mode", self.runtime.group_mode
        )
        _sync_param_with_widget(self.food_grid, "color", self.runtime.food_grid_color)
        _sync_param_with_widget(
            self.food_grid, "initial_value", self.runtime.food_grid_initial_value
        )
        _sync_param_with_widget(self.food_grid, "fixed_max", self.food_grid_fixed_max)
        _sync_param_with_widget(
            self.food_grid, "fixed_max", self.runtime.food_grid_fixed_max
        )
        _sync_param_with_widget(
            self.food_grid.substrate, "quality", self.food_grid_substrate_quality
        )
        _sync_param_with_widget(
            self.food_grid.substrate,
            "quality",
            self.runtime.food_grid_substrate_quality,
        )
        _sync_param_with_widget(
            self.odorscape, "grid_dims_x", self.odorscape_grid_dims_x
        )
        _sync_param_with_widget(
            self.odorscape, "grid_dims_y", self.odorscape_grid_dims_y
        )
        self.presets.param.watch(self._sync_canvas_title, "preset_name")
        self.runtime.arena_width.param.watch(self._sync_area_dims_from_widgets, "value")
        self.runtime.arena_height.param.watch(
            self._sync_area_dims_from_widgets, "value"
        )
        self.runtime.group_spread_x.param.watch(
            self._sync_group_distribution_scale_from_widgets, "value"
        )
        self.runtime.group_spread_y.param.watch(
            self._sync_group_distribution_scale_from_widgets, "value"
        )
        self.runtime.group_shape.param.watch(
            self._sync_group_distribution_scale_from_widgets, "value"
        )
        self.group_distribution.param.watch(
            self._sync_group_distribution_widgets_from_param, "scale"
        )
        self.food_grid_dims_x.param.watch(self._sync_food_grid_dims_from_v2, "value")
        self.food_grid_dims_y.param.watch(self._sync_food_grid_dims_from_v2, "value")
        self.runtime.food_grid_dims_x.param.watch(
            self._sync_food_grid_dims_from_runtime, "value"
        )
        self.runtime.food_grid_dims_y.param.watch(
            self._sync_food_grid_dims_from_runtime, "value"
        )
        self.food_grid.param.watch(self._sync_food_grid_dims_from_param, "grid_dims")
        self._sync_area_dims_from_widgets()
        self._sync_group_distribution_scale_from_widgets()
        self._sync_food_grid_dims_from_runtime()

    def _sync_canvas_title(self, *_: object) -> None:
        unique_id = (
            self.presets.preset_name or ""
        ).strip() or "environment_builder_config"
        self.runtime.fig.title.text = f"Environment: {unique_id}"

    def _sync_area_dims_from_widgets(self, *_: object) -> None:
        if self._syncing_area_dims:
            return
        dims = (
            float(self.runtime.arena_width.value),
            float(self.runtime.arena_height.value),
        )
        if tuple(self.arena.dims) == dims:
            return
        self._syncing_area_dims = True
        try:
            self.arena.dims = dims
        finally:
            self._syncing_area_dims = False

    def _sync_group_distribution_scale_from_widgets(self, *_: object) -> None:
        if self._syncing_group_scale:
            return
        shape = _normalize_group_shape(self.runtime.group_shape.value)
        scale_x = self.runtime._group_scale_from_display_mm(
            shape, float(self.runtime.group_spread_x.value)
        )
        scale_y = (
            scale_x
            if shape == "circle"
            else self.runtime._group_scale_from_display_mm(
                shape, float(self.runtime.group_spread_y.value)
            )
        )
        scale = (scale_x, scale_y)
        if tuple(self.group_distribution.scale) == scale:
            return
        self._syncing_group_scale = True
        try:
            self.group_distribution.scale = scale
        finally:
            self._syncing_group_scale = False

    def _sync_group_distribution_widgets_from_param(
        self, *_: param.parameterized.Event
    ) -> None:
        if self._syncing_group_scale:
            return
        shape = _normalize_group_shape(self.runtime.group_shape.value)
        scale_x, scale_y = tuple(self.group_distribution.scale)
        display_x = self.runtime._group_display_value_mm(shape, float(scale_x))
        display_y = (
            display_x
            if shape == "circle"
            else self.runtime._group_display_value_mm(shape, float(scale_y))
        )
        self._syncing_group_scale = True
        try:
            if float(self.runtime.group_spread_x.value) != float(display_x):
                self.runtime.group_spread_x.value = float(display_x)
            if float(self.runtime.group_spread_y.value) != float(display_y):
                self.runtime.group_spread_y.value = float(display_y)
        finally:
            self._syncing_group_scale = False

    def _sync_food_grid_dims_from_v2(self, *_: object) -> None:
        if self._syncing_food_grid_dims:
            return
        dims = (int(self.food_grid_dims_x.value), int(self.food_grid_dims_y.value))
        self._syncing_food_grid_dims = True
        try:
            if tuple(self.food_grid.grid_dims) != dims:
                self.food_grid.grid_dims = dims
            if int(self.runtime.food_grid_dims_x.value) != dims[0]:
                self.runtime.food_grid_dims_x.value = dims[0]
            if int(self.runtime.food_grid_dims_y.value) != dims[1]:
                self.runtime.food_grid_dims_y.value = dims[1]
        finally:
            self._syncing_food_grid_dims = False

    def _sync_food_grid_dims_from_runtime(self, *_: object) -> None:
        if self._syncing_food_grid_dims:
            return
        dims = (
            int(self.runtime.food_grid_dims_x.value),
            int(self.runtime.food_grid_dims_y.value),
        )
        self._syncing_food_grid_dims = True
        try:
            if tuple(self.food_grid.grid_dims) != dims:
                self.food_grid.grid_dims = dims
            if int(self.food_grid_dims_x.value) != dims[0]:
                self.food_grid_dims_x.value = dims[0]
            if int(self.food_grid_dims_y.value) != dims[1]:
                self.food_grid_dims_y.value = dims[1]
        finally:
            self._syncing_food_grid_dims = False

    def _sync_food_grid_dims_from_param(self, *_: param.parameterized.Event) -> None:
        if self._syncing_food_grid_dims:
            return
        dims = tuple(int(value) for value in self.food_grid.grid_dims)
        self._syncing_food_grid_dims = True
        try:
            if int(self.food_grid_dims_x.value) != dims[0]:
                self.food_grid_dims_x.value = dims[0]
            if int(self.food_grid_dims_y.value) != dims[1]:
                self.food_grid_dims_y.value = dims[1]
            if int(self.runtime.food_grid_dims_x.value) != dims[0]:
                self.runtime.food_grid_dims_x.value = dims[0]
            if int(self.runtime.food_grid_dims_y.value) != dims[1]:
                self.runtime.food_grid_dims_y.value = dims[1]
        finally:
            self._syncing_food_grid_dims = False

    @staticmethod
    def _param_pane(
        section: param.Parameterized,
        parameters: list[str],
        *,
        widget_overrides: dict[str, object] | None = None,
    ) -> pn.Param:
        return pn.Param(
            section.param,
            parameters=parameters,
            widgets=widget_overrides or {},
            show_name=False,
            sizing_mode="stretch_width",
            margin=0,
            stylesheets=[_FAMILY_FIELD_STYLESHEET],
        )

    @staticmethod
    def _family_box(
        title: str,
        *children: object,
        header_right: pn.viewable.Viewable | None = None,
        compact: bool = False,
    ) -> pn.Column:
        title_styles = dict(_FAMILY_TITLE_STYLES)
        if header_right is not None:
            title_styles["margin"] = "-6px 0 0 0"
        title_pane = pn.pane.Markdown(f"**{title}**", styles=title_styles, margin=0)
        if header_right is None:
            header = pn.Row(
                title_pane, sizing_mode="stretch_width", margin=(0, 0, 6, 0)
            )
        else:
            header_right.margin = (0, 0, 0, 0)
            header = pn.Row(
                pn.Column(title_pane, sizing_mode="stretch_width", margin=0),
                pn.Column(header_right, width=22, align="start", margin=0),
                sizing_mode="stretch_width",
                align="start",
                margin=(0, 0, 6, 0),
            )
        return pn.Column(
            header,
            *children,
            styles=_FAMILY_BOX_COMPACT_STYLES if compact else _FAMILY_BOX_STYLES,
            sizing_mode="stretch_width",
            margin=(0, 0, 8, 0),
        )

    @staticmethod
    def _make_bound_switch(
        section: param.Parameterized, parameter_name: str
    ) -> pn.widgets.Switch:
        widget = pn.widgets.Switch(
            name="", value=bool(getattr(section, parameter_name)), width=18, margin=0
        )
        _sync_param_with_widget(section, parameter_name, widget)
        return widget

    def _arena_panel(self) -> pn.viewable.Viewable:
        @pn.depends(self.arena.param.geometry, self.runtime.table.param.value)
        def _view(geometry: str, _table_value: object) -> pn.Column:
            locked = bool(self.runtime._objects)
            return self._family_box(
                "Arena",
                pn.Param(
                    self.arena.param,
                    parameters=["geometry", "torus"],
                    widgets={
                        "geometry": {
                            "type": pn.widgets.Select,
                            "disabled": locked,
                        },
                        "torus": {
                            "type": pn.widgets.Checkbox,
                            "disabled": locked,
                        },
                    },
                    show_name=False,
                    sizing_mode="stretch_width",
                    margin=0,
                    stylesheets=[_FAMILY_FIELD_STYLESHEET],
                ),
                self.runtime.arena_width,
                self.runtime.arena_height
                if geometry != "circular"
                else pn.Spacer(height=0),
                self.runtime.clear_arena_btn,
            )

        return _view

    def _insert_panel(self) -> pn.viewable.Viewable:
        @pn.depends(self.insert.param.object_type)
        def _view(object_type: str) -> pn.Column:
            children: list[object] = [
                self._param_pane(
                    self.insert,
                    ["object_type"],
                    widget_overrides={"object_type": {"type": pn.widgets.Select}},
                )
            ]
            if object_type == "Border segment":
                children.append(
                    self._param_pane(
                        self.insert,
                        ["border_width", "color"],
                        widget_overrides={
                            "border_width": {
                                "type": pn.widgets.FloatInput,
                                "step": 0.5,
                            },
                            "color": {"type": pn.widgets.ColorPicker},
                        },
                    )
                )
            else:
                distro_pane = None
                if object_type == "Source group":
                    distro_pane = pn.Column(
                        pn.Param(
                            self.group_distribution.param,
                            parameters=["N", "shape", "mode"],
                            widgets={
                                "shape": {"type": pn.widgets.Select},
                                "mode": {"type": pn.widgets.Select},
                            },
                            show_name=False,
                            sizing_mode="stretch_width",
                            margin=0,
                        ),
                        self.runtime.group_spread_x,
                        self.runtime.group_spread_y,
                        sizing_mode="stretch_width",
                        margin=0,
                    )
                children.append(
                    pn.Column(
                        self._param_pane(
                            self.insert,
                            ["radius", "color"],
                            widget_overrides={
                                "radius": {
                                    "type": pn.widgets.FloatSlider,
                                    "start": 1.0,
                                    "end": 30.0,
                                    "step": 1.0,
                                    "sizing_mode": "stretch_width",
                                    "width": None,
                                },
                                "color": {"type": pn.widgets.ColorPicker},
                            },
                        ),
                        *([distro_pane] if distro_pane is not None else []),
                        sizing_mode="stretch_width",
                        margin=0,
                    )
                )
            return self._family_box("Insert object", *children, self.runtime.status)

        return _view

    def _food_grid_panel(self) -> pn.viewable.Viewable:
        @pn.depends(self.insert.param.enabled)
        def _view(enabled: bool) -> pn.Column:
            children: list[object] = [self._param_pane(self.insert, ["enabled"])]
            if enabled:
                children.extend(
                    [
                        pn.Param(
                            self.food_grid.param,
                            parameters=["color", "initial_value"],
                            widgets={
                                "color": {"type": pn.widgets.ColorPicker},
                                "initial_value": {
                                    "type": pn.widgets.FloatInput,
                                    "step": 1e-6,
                                },
                            },
                            show_name=False,
                            sizing_mode="stretch_width",
                            margin=0,
                        ),
                        self.food_grid_dims_x,
                        self.food_grid_dims_y,
                        self.food_grid_fixed_max,
                        self.food_grid_fixed_max_note,
                        pn.layout.Divider(margin=(4, 0, 2, 0)),
                        self.runtime.food_grid_substrate_type,
                        self.food_grid_substrate_quality,
                    ]
                )
            return self._family_box("Food grid", *children)

        return _view

    def _odorscape_panel(self) -> pn.viewable.Viewable:
        @pn.depends(self.odorscape.param.enabled, self.odorscape.param.mode)
        def _view(enabled: bool, mode: str) -> pn.Column:
            children: list[object] = []
            if enabled:
                params = ["mode", "color", "initial_value", "fixed_max"]
                if mode == "Diffusion":
                    params.extend(["evap_const", "sigma_x", "sigma_y"])
                children.extend(
                    [
                        self.odorscape_grid_dims_x,
                        self.odorscape_grid_dims_y,
                        self._param_pane(
                            self.odorscape,
                            params,
                            widget_overrides={
                                "mode": {"type": pn.widgets.Select},
                                "color": {"type": pn.widgets.ColorPicker},
                            },
                        ),
                    ]
                )
            return self._family_box(
                "Odorscape",
                *children,
                header_right=self.odorscape_enabled_switch,
                compact=not enabled,
            )

        return _view

    def _windscape_panel(self) -> pn.viewable.Viewable:
        @pn.depends(self.windscape.param.enabled)
        def _view(enabled: bool) -> pn.Column:
            children: list[object] = []
            if enabled:
                children.extend(
                    [
                        self._param_pane(
                            self.windscape,
                            ["color", "direction", "speed"],
                            widget_overrides={
                                "color": {"type": pn.widgets.ColorPicker}
                            },
                        ),
                        self.runtime.wind_puffs_table,
                        pn.Column(
                            self.runtime.add_wind_puff_btn,
                            self.runtime.remove_wind_puff_btn,
                            sizing_mode="stretch_width",
                            margin=0,
                        ),
                    ]
                )
            return self._family_box(
                "Windscape",
                *children,
                header_right=self.windscape_enabled_switch,
                compact=not enabled,
            )

        return _view

    def _thermoscape_panel(self) -> pn.viewable.Viewable:
        @pn.depends(self.thermoscape.param.enabled)
        def _view(enabled: bool) -> pn.Column:
            children: list[object] = []
            if enabled:
                children.extend(
                    [
                        self._param_pane(self.thermoscape, ["plate_temp", "spread"]),
                        self.runtime.thermo_sources_table,
                        pn.Column(
                            self.runtime.add_thermo_source_btn,
                            self.runtime.remove_thermo_source_btn,
                            sizing_mode="stretch_width",
                            margin=0,
                        ),
                    ]
                )
            return self._family_box(
                "Thermoscape",
                *children,
                header_right=self.thermoscape_enabled_switch,
                compact=not enabled,
            )

        return _view

    def view(self) -> pn.viewable.Viewable:
        intro = pn.pane.Markdown(
            (
                "### Environment Builder\n"
                "Use this app to compose and inspect Larvaworld environments visually. Define the arena, place "
                "source units, source groups, and borders on the canvas, configure optional food-grid and sensory-scape "
                "layers, and then refine every object from the `Inspect / Edit` panel. The builder follows the same "
                "canonical environment structure used by the codebase, mapping directly onto `FoodConf` / `EnvConf`: "
                "`source_units`, `source_groups`, `food_grid`, `border_list`, `odorscape`, `windscape`, and `thermoscape`. "
                "Food amount controls whether a source is visually filled, while odor is rendered as a Gaussian-like aura "
                "around the source. "
                "`Odorscape` contours appear in the preview only when at least one placed source has odor configured, "
                "`Thermoscape` preview appears only after thermal-source rows are added, and `Windscape` arrows appear only when wind speed is greater than zero. "
                "`Wind direction` is edited in degrees in the builder UI and converted back to radians for the exported config. "
                f"Reference: [Arenas and Substrates]({DOCS_ARENAS_SUBSTRATES})."
            ),
            css_classes=["lw-env-builder-intro"],
            margin=0,
        )
        controls = pn.Card(
            pn.Column(
                self._arena_panel(),
                self._insert_panel(),
                self._food_grid_panel(),
                sizing_mode="stretch_width",
                margin=0,
            ),
            title="Configuration",
            collapsed=False,
            sizing_mode="stretch_width",
        )
        presets = pn.Card(
            pn.Column(
                self._param_pane(
                    self.presets,
                    ["preset_name"],
                    widget_overrides={"preset_name": {"type": pn.widgets.TextInput}},
                ),
                self.runtime.preset_select,
                self.runtime.refresh_presets_btn,
                pn.Row(
                    self.runtime.save_preset_btn,
                    self.runtime.load_preset_btn,
                    self.runtime.delete_preset_btn,
                    sizing_mode="stretch_width",
                    margin=(4, 0, 0, 0),
                ),
                self.runtime.preset_meta,
                self.runtime.load_file_input,
                self.runtime.export_btn,
                pn.Row(
                    self.runtime.load_file_btn,
                    self.runtime.download_file_btn,
                    sizing_mode="stretch_width",
                    margin=(4, 0, 0, 0),
                ),
                pn.Row(
                    self.runtime.clear_all_btn,
                    sizing_mode="stretch_width",
                    margin=(4, 0, 0, 0),
                ),
                self.runtime.reset_confirm_panel,
                self.runtime.preset_overwrite_confirm,
                sizing_mode="stretch_width",
                margin=0,
            ),
            title="Stored Configurations",
            collapsed=False,
            sizing_mode="stretch_width",
        )
        table_card = pn.Card(
            self.runtime.table,
            title="Placed objects",
            collapsed=False,
            sizing_mode="stretch_width",
        )
        editor_card = pn.Card(
            pn.Column(
                pn.Row(
                    self.runtime.apply_selected_btn,
                    self.runtime.delete_selected_btn,
                    sizing_mode="stretch_width",
                    margin=(0, 0, 4, 0),
                ),
                pn.Row(
                    self.runtime.select_mode,
                    sizing_mode="stretch_width",
                    margin=0,
                ),
                self.runtime.selected_object,
                self.runtime.selected_id,
                self.runtime.selected_x,
                self.runtime.selected_y,
                self.runtime.selected_x2,
                self.runtime.selected_y2,
                self.runtime.selected_radius,
                self.runtime.selected_width,
                pn.Row(self.runtime.selected_color, pn.Spacer(), margin=0),
                self.runtime._editor_distribution_family,
                self.runtime._editor_food_family,
                self.runtime._editor_substrate_family,
                self.runtime._editor_source_behavior_family,
                self.runtime._editor_odor_family,
                sizing_mode="stretch_width",
                margin=0,
            ),
            title="Inspect / Edit",
            collapsed=False,
            sizing_mode="stretch_width",
        )
        scapes_card = pn.Card(
            pn.Column(
                self._odorscape_panel(),
                self._windscape_panel(),
                self._thermoscape_panel(),
                sizing_mode="stretch_width",
                margin=0,
            ),
            title="Environment Scapes",
            collapsed=False,
            sizing_mode="stretch_width",
        )
        side = pn.Column(presets, controls, min_width=320, sizing_mode="stretch_width")
        center = pn.Column(
            pn.pane.Bokeh(self.runtime.fig, sizing_mode="stretch_both"),
            table_card,
            min_width=760,
            sizing_mode="stretch_width",
        )
        right = pn.Column(
            editor_card, scapes_card, min_width=360, sizing_mode="stretch_width"
        )
        main = pn.Row(side, center, right, sizing_mode="stretch_width")
        return pn.Column(
            intro,
            main,
            css_classes=["lw-env-builder-root"],
            sizing_mode="stretch_width",
        )


def environment_builder_app() -> pn.viewable.Viewable:
    pn.extension("tabulator", raw_css=[PORTAL_RAW_CSS, ENV_BUILDER_RAW_CSS])
    controller = _EnvironmentBuilderV2Controller()
    template = pn.template.MaterialTemplate(
        title="",
        header_background=LANE_MODELS_COLOR,
        header_color="#1f1f1f",
    )
    template.header.append(build_app_header(title="Environment Builder"))
    template.main.append(controller.view())
    return template
