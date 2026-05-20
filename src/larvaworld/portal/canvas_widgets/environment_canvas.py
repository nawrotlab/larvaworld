from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np
import panel as pn
from bokeh.colors import named as named_colors
from bokeh.models import ColumnDataSource, LabelSet, Legend, LegendItem, WheelZoomTool
from bokeh.plotting import figure

from larvaworld.lib.param.xy_distro import Spatial_Distro

from .environment_models import (
    CanvasArena,
    CanvasRingOverlay,
    CanvasObject,
    EnvironmentCanvasState,
    LarvaPreviewFrame,
)


LANE_MODELS_COLOR_DARK = "#5a4760"
DEFAULT_SOURCE_COLOR = "#4caf50"
DEFAULT_LARVA_COLOR = "#2f4858"
HIGHLIGHT_COLOR = "#f97316"
STATIC_LARVA_GROUP_MEMBER_HALF_LENGTH = 0.0015
ENV_CANVAS_WIDTH = 760
ENV_CANVAS_HEIGHT = 620
ENV_CANVAS_Y_HALF_RANGE = 0.30
ENV_CANVAS_X_HALF_RANGE = ENV_CANVAS_Y_HALF_RANGE * (
    ENV_CANVAS_WIDTH / ENV_CANVAS_HEIGHT
)


def _empty(keys: Iterable[str]) -> dict[str, list[Any]]:
    return {key: [] for key in keys}


def _rows_to_data(
    rows: Iterable[dict[str, Any]], keys: Iterable[str]
) -> dict[str, list[Any]]:
    data = _empty(keys)
    for row in rows:
        for key in data:
            data[key].append(row.get(key))
    return data


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _valid_xy(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        x = float(value[0])
        y = float(value[1])
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    return (x, y)


def _valid_path(value: Any) -> tuple[tuple[float, float], ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    points = tuple(
        xy for xy in (_valid_xy(candidate) for candidate in value) if xy is not None
    )
    return points if len(points) >= 2 else ()


def _closed_path(
    points: tuple[tuple[float, float], ...],
) -> tuple[tuple[float, float], ...]:
    if len(points) >= 3 and points[0] != points[-1]:
        return (*points, points[0])
    return points


def _nearest_endpoint(
    point: tuple[float, float],
    path: tuple[tuple[float, float], ...],
) -> tuple[float, float]:
    first = path[0]
    last = path[-1]
    first_dist_sq = (point[0] - first[0]) ** 2 + (point[1] - first[1]) ** 2
    last_dist_sq = (point[0] - last[0]) ** 2 + (point[1] - last[1]) ** 2
    return first if first_dist_sq <= last_dist_sq else last


def _optional_float(value: Any, default: float) -> float:
    if value is None:
        return default
    return _safe_float(value, default)


def _distribution_scale_pair(obj: CanvasObject) -> tuple[float, float]:
    if obj.object_type == "source_group":
        return (
            _safe_float(obj.distribution_scale_x or 0.012, 0.012),
            _safe_float(obj.distribution_scale_y or 0.012, 0.012),
        )
    return (
        _optional_float(obj.distribution_scale_x, 0.012),
        _optional_float(obj.distribution_scale_y, 0.012),
    )


def _mix_hex_colors(color_a: Any, color_b: Any, ratio: float) -> str:
    ratio = max(0.0, min(1.0, float(ratio)))

    def _parse(color: Any) -> tuple[int, int, int]:
        if isinstance(color, (list, tuple)) and len(color) >= 3:
            try:
                channels = [float(component) for component in color[:3]]
            except (TypeError, ValueError):
                return (76, 175, 80)
            if all(0.0 <= channel <= 1.0 for channel in channels):
                return tuple(int(round(channel * 255)) for channel in channels)
            return tuple(
                int(round(max(0.0, min(255.0, channel)))) for channel in channels
            )
        raw = str(color).strip()
        if not raw:
            return (76, 175, 80)
        if not raw.startswith("#"):
            named = getattr(named_colors, raw.lower().replace(" ", ""), None)
            if named is not None:
                rgb = named.to_rgb()
                return (int(rgb.r), int(rgb.g), int(rgb.b))
        cleaned = raw.lstrip("#")
        if len(cleaned) == 3:
            cleaned = "".join(char * 2 for char in cleaned)
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


def _source_visual_state(
    *,
    amount: float | None,
    color: str | None,
) -> tuple[str, str, float, float, float]:
    base_color = str(color or DEFAULT_SOURCE_COLOR)
    has_food = amount is not None and _safe_float(amount) > 0
    fill_color = _mix_hex_colors(base_color, "#ffffff", 0.0 if has_food else 0.68)
    line_color = _mix_hex_colors(base_color, "#111111", 0.08 if has_food else 0.02)
    return (
        fill_color,
        line_color,
        0.94 if has_food else 0.34,
        0.98 if has_food else 0.58,
        2.6 if has_food else 1.6,
    )


def _stable_preview_seed(*parts: object) -> int:
    seed = 2166136261
    for part in parts:
        for char in str(part):
            seed ^= ord(char)
            seed = (seed * 16777619) & 0xFFFFFFFF
    return seed


def _stable_member_angle(obj: CanvasObject, member_index: int) -> float:
    seed = _stable_preview_seed(
        "larva_member_angle",
        obj.object_id,
        member_index,
        obj.distribution_mode,
        obj.distribution_shape,
        obj.distribution_n,
        obj.x,
        obj.y,
        obj.distribution_scale_x,
        obj.distribution_scale_y,
    )
    rng = np.random.default_rng(seed)
    return float(rng.uniform(0.0, 2.0 * math.pi))


def _normalize_group_shape(value: str | None) -> str:
    shape = str(value or "circle").strip().lower()
    if shape in {"oval", "ellipse", "elliptic"}:
        return "oval"
    if shape in {"rect", "rectangle", "square"}:
        return "rect"
    return "circle"


def _rotate_point(x: float, y: float, angle: float) -> tuple[float, float]:
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    return (x * cos_a - y * sin_a, x * sin_a + y * cos_a)


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
    spread = _safe_float(odor_spread)
    intensity = _safe_float(odor_intensity)
    if spread <= 0 or intensity <= 0:
        return []

    intensity_scale = min(1.0, 0.35 + 0.18 * intensity)
    source_r = max(_safe_float(source_radius), 0.002)
    aura_color = _mix_hex_colors(str(color or DEFAULT_SOURCE_COLOR), "#ffffff", 0.2)
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
    spread = _safe_float(odor_spread)
    intensity = _safe_float(odor_intensity)
    if spread <= 0 or intensity <= 0:
        return None

    source_r = max(_safe_float(source_radius), 0.002)
    peak_radius = max(source_r * 0.42, min(spread * 0.3, source_r * 0.72))
    peak_color = _mix_hex_colors(str(color or DEFAULT_SOURCE_COLOR), "#ffffff", 0.08)
    return {
        "x": float(x),
        "y": float(y),
        "r": peak_radius,
        "color": peak_color,
        "fill_alpha": min(0.72, 0.44 + 0.1 * intensity),
        "id": str(source_id or ""),
    }


class EnvironmentCanvas:
    """Reusable read-only Bokeh canvas for portal environment previews."""

    def __init__(
        self,
        *,
        width: int = ENV_CANVAS_WIDTH,
        height: int = ENV_CANVAS_HEIGHT,
        editable: bool = False,
        snap_heads_to_midline: bool = False,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.editable = bool(editable)
        self.snap_heads_to_midline = bool(snap_heads_to_midline)
        self._state: EnvironmentCanvasState | None = None
        self._arena = CanvasArena("rectangular", (0.2, 0.2))

        self.arena_source = ColumnDataSource({"x": [], "y": [], "w": [], "h": []})
        self.food_grid_overlay_source = ColumnDataSource(
            {"x": [], "y": [], "w": [], "h": [], "color": [], "fill_alpha": []}
        )
        self.food_grid_cell_source = ColumnDataSource(
            _empty(
                [
                    "x",
                    "y",
                    "w",
                    "h",
                    "fill_color",
                    "line_color",
                    "fill_alpha",
                    "line_alpha",
                    "line_width",
                ]
            )
        )
        self.thermoscape_aura_source = ColumnDataSource(
            _empty(["x", "y", "r", "color", "fill_alpha", "line_alpha", "id"])
        )
        self.thermoscape_marker_source = ColumnDataSource(
            _empty(["x", "y", "color", "size", "id"])
        )
        self.windscape_segment_source = ColumnDataSource(
            _empty(["x0", "y0", "x1", "y1", "color", "line_alpha"])
        )
        self.windscape_head_source = ColumnDataSource(
            _empty(["x", "y", "angle", "color", "size"])
        )
        self.odorscape_contour_source = ColumnDataSource(
            _empty(["x", "y", "r", "color", "line_alpha", "line_width", "id"])
        )
        self.odor_layer_source = ColumnDataSource(
            _empty(["x", "y", "r", "color", "fill_alpha", "id"])
        )
        self.odor_peak_source = ColumnDataSource(
            _empty(["x", "y", "r", "color", "fill_alpha", "id"])
        )
        self.food_source = ColumnDataSource(
            _empty(
                [
                    "x",
                    "y",
                    "r",
                    "fill_color",
                    "line_color",
                    "id",
                    "fill_alpha",
                    "line_alpha",
                    "line_width",
                ]
            )
        )
        self.food_highlight_source = ColumnDataSource(
            {"x": [], "y": [], "r": [], "color": []}
        )
        self.source_group_circle_source = ColumnDataSource(
            _empty(["x", "y", "r", "color", "fill_alpha", "line_alpha", "id"])
        )
        self.source_group_ellipse_source = ColumnDataSource(
            _empty(["x", "y", "w", "h", "color", "fill_alpha", "line_alpha", "id"])
        )
        self.source_group_rect_source = ColumnDataSource(
            _empty(["x", "y", "w", "h", "color", "fill_alpha", "line_alpha", "id"])
        )
        self.source_group_member_source = ColumnDataSource(
            _empty(
                [
                    "x",
                    "y",
                    "r",
                    "fill_color",
                    "line_color",
                    "fill_alpha",
                    "line_alpha",
                    "line_width",
                    "parent_id",
                ]
            )
        )
        self.source_group_circle_highlight_source = ColumnDataSource(
            {"x": [], "y": [], "r": [], "color": []}
        )
        self.source_group_ellipse_highlight_source = ColumnDataSource(
            {"x": [], "y": [], "w": [], "h": [], "color": []}
        )
        self.source_group_rect_highlight_source = ColumnDataSource(
            {"x": [], "y": [], "w": [], "h": [], "color": []}
        )
        self.border_source = ColumnDataSource(
            _empty(["x0", "y0", "x1", "y1", "w", "color", "id"])
        )
        self.border_highlight_source = ColumnDataSource(
            {"x0": [], "y0": [], "x1": [], "y1": [], "w": [], "color": []}
        )
        self.larva_group_circle_source = ColumnDataSource(
            _empty(["x", "y", "r", "color", "fill_alpha", "line_alpha", "id"])
        )
        self.larva_group_ellipse_source = ColumnDataSource(
            _empty(["x", "y", "w", "h", "color", "fill_alpha", "line_alpha", "id"])
        )
        self.larva_group_rect_source = ColumnDataSource(
            _empty(["x", "y", "w", "h", "color", "fill_alpha", "line_alpha", "id"])
        )
        self.larva_group_member_source = ColumnDataSource(
            _empty(
                [
                    "x0",
                    "y0",
                    "x1",
                    "y1",
                    "fill_color",
                    "line_color",
                    "fill_alpha",
                    "line_alpha",
                    "line_width",
                    "parent_id",
                ]
            )
        )
        self.larva_group_circle_highlight_source = ColumnDataSource(
            {"x": [], "y": [], "r": [], "color": []}
        )
        self.larva_group_ellipse_highlight_source = ColumnDataSource(
            {"x": [], "y": [], "w": [], "h": [], "color": []}
        )
        self.larva_group_rect_highlight_source = ColumnDataSource(
            {"x": [], "y": [], "w": [], "h": [], "color": []}
        )
        self.sim_larva_centroid_source = ColumnDataSource(
            _empty(["x", "y", "color", "id"])
        )
        self.sim_larva_head_source = ColumnDataSource(_empty(["x", "y", "color", "id"]))
        self.sim_larva_midline_source = ColumnDataSource(
            _empty(["xs", "ys", "color", "id"])
        )
        self.sim_larva_trail_source = ColumnDataSource(
            _empty(["xs", "ys", "color", "id"])
        )
        self.sim_larva_segment_source = ColumnDataSource(
            _empty(["xs", "ys", "color", "id"])
        )
        self.sim_larva_body_contour_source = ColumnDataSource(
            _empty(["xs", "ys", "color", "id"])
        )
        self.sim_larva_label_source = ColumnDataSource(
            _empty(["x", "y", "label", "color", "id"])
        )
        self.dynamic_ring_source = ColumnDataSource(
            _empty(["x", "y", "r", "color", "line_width", "line_alpha", "line_dash"])
        )

        self.fig = figure(
            title="Environment canvas",
            x_range=(-ENV_CANVAS_X_HALF_RANGE, ENV_CANVAS_X_HALF_RANGE),
            y_range=(-ENV_CANVAS_Y_HALF_RANGE, ENV_CANVAS_Y_HALF_RANGE),
            match_aspect=True,
            width=self.width,
            height=self.height,
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
            source=self.arena_source,
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
            source=self.food_grid_overlay_source,
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
            source=self.arena_source,
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
            source=self.food_grid_overlay_source,
            line_color=None,
            fill_color="color",
            fill_alpha="fill_alpha",
            visible=False,
        )
        self.fig.rect(
            x="x",
            y="y",
            width="w",
            height="h",
            source=self.food_grid_cell_source,
            line_color="line_color",
            line_alpha="line_alpha",
            line_width="line_width",
            fill_color="fill_color",
            fill_alpha="fill_alpha",
        )
        self._thermoscape_aura_renderer = self.fig.circle(
            x="x",
            y="y",
            radius="r",
            source=self.thermoscape_aura_source,
            line_color="color",
            line_alpha="line_alpha",
            fill_color="color",
            fill_alpha="fill_alpha",
            line_width=1,
        )
        self._windscape_segment_renderer = self.fig.segment(
            x0="x0",
            y0="y0",
            x1="x1",
            y1="y1",
            source=self.windscape_segment_source,
            line_color="color",
            line_alpha="line_alpha",
            line_width=2,
        )
        self._windscape_head_renderer = self.fig.scatter(
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
        self._odorscape_renderer = self.fig.circle(
            x="x",
            y="y",
            radius="r",
            source=self.odorscape_contour_source,
            line_color="color",
            line_alpha="line_alpha",
            line_width="line_width",
            fill_alpha=0.0,
        )
        self._thermoscape_marker_renderer = self.fig.scatter(
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
        self._odor_aura_renderer = self.fig.circle(
            x="x",
            y="y",
            radius="r",
            source=self.odor_layer_source,
            line_color=None,
            fill_color="color",
            fill_alpha="fill_alpha",
        )
        self._source_units_renderer = self.fig.circle(
            x="x",
            y="y",
            radius="r",
            source=self.food_source,
            line_color="line_color",
            fill_color="fill_color",
            fill_alpha="fill_alpha",
            line_alpha="line_alpha",
            line_width="line_width",
        )
        self._odor_peak_renderer = self.fig.circle(
            x="x",
            y="y",
            radius="r",
            source=self.odor_peak_source,
            line_color=None,
            fill_color="color",
            fill_alpha="fill_alpha",
        )
        self._source_group_circle_renderer = self.fig.circle(
            x="x",
            y="y",
            radius="r",
            source=self.source_group_circle_source,
            line_color="color",
            fill_color="color",
            fill_alpha="fill_alpha",
            line_alpha="line_alpha",
            line_width=2,
        )
        self._source_group_ellipse_renderer = self.fig.ellipse(
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
        self._source_group_rect_renderer = self.fig.rect(
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
        self._source_group_member_renderer = self.fig.circle(
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
        self._border_renderer = self.fig.segment(
            x0="x0",
            y0="y0",
            x1="x1",
            y1="y1",
            source=self.border_source,
            line_color="color",
            line_width="w",
        )
        self._larva_group_circle_renderer = self.fig.circle(
            x="x",
            y="y",
            radius="r",
            source=self.larva_group_circle_source,
            line_color="color",
            fill_color="color",
            fill_alpha="fill_alpha",
            line_alpha="line_alpha",
            line_dash="dashed",
            line_width=2,
        )
        self._larva_group_ellipse_renderer = self.fig.ellipse(
            x="x",
            y="y",
            width="w",
            height="h",
            source=self.larva_group_ellipse_source,
            line_color="color",
            fill_color="color",
            fill_alpha="fill_alpha",
            line_alpha="line_alpha",
            line_dash="dashed",
            line_width=2,
        )
        self._larva_group_rect_renderer = self.fig.rect(
            x="x",
            y="y",
            width="w",
            height="h",
            source=self.larva_group_rect_source,
            line_color="color",
            fill_color="color",
            fill_alpha="fill_alpha",
            line_alpha="line_alpha",
            line_dash="dashed",
            line_width=2,
        )
        self._larva_group_member_renderer = self.fig.segment(
            x0="x0",
            y0="y0",
            x1="x1",
            y1="y1",
            source=self.larva_group_member_source,
            line_color="line_color",
            line_alpha="line_alpha",
            line_width="line_width",
        )
        self._sim_larva_trail_renderer = self.fig.multi_line(
            xs="xs",
            ys="ys",
            source=self.sim_larva_trail_source,
            line_color="color",
            line_alpha=0.32,
            line_width=1.5,
        )
        self._sim_larva_segment_renderer = self.fig.patches(
            xs="xs",
            ys="ys",
            source=self.sim_larva_segment_source,
            fill_color="color",
            line_color="#111111",
            fill_alpha=0.55,
            line_alpha=0.80,
            line_width=0.8,
        )
        self._sim_larva_body_contour_renderer = self.fig.multi_line(
            xs="xs",
            ys="ys",
            source=self.sim_larva_body_contour_source,
            line_color="color",
            line_alpha=0.95,
            line_width=1.2,
        )
        self._sim_larva_midline_renderer = self.fig.multi_line(
            xs="xs",
            ys="ys",
            source=self.sim_larva_midline_source,
            line_color="color",
            line_alpha=0.85,
            line_width=2.0,
        )
        self._sim_larva_centroid_renderer = self.fig.circle(
            x="x",
            y="y",
            source=self.sim_larva_centroid_source,
            size=6,
            fill_color="color",
            line_color="#111111",
            fill_alpha=0.90,
            line_alpha=0.75,
        )
        self._sim_larva_head_renderer = self.fig.circle(
            x="x",
            y="y",
            source=self.sim_larva_head_source,
            size=4,
            fill_color="color",
            line_color="#111111",
            fill_alpha=1.0,
            line_alpha=0.80,
        )
        self._sim_larva_label_renderer = self.fig.add_layout(
            LabelSet(
                x="x",
                y="y",
                text="label",
                source=self.sim_larva_label_source,
                text_font_size="8pt",
                text_color="color",
                x_offset=6,
                y_offset=4,
            )
        )
        self._dynamic_ring_renderer = self.fig.circle(
            x="x",
            y="y",
            radius="r",
            source=self.dynamic_ring_source,
            line_color="color",
            line_width="line_width",
            line_alpha="line_alpha",
            line_dash="line_dash",
            fill_alpha=0.0,
        )
        self._food_highlight_renderer = self.fig.circle(
            x="x",
            y="y",
            radius="r",
            source=self.food_highlight_source,
            line_color=HIGHLIGHT_COLOR,
            fill_color=None,
            line_width=4,
        )
        self.fig.circle(
            x="x",
            y="y",
            radius="r",
            source=self.source_group_circle_highlight_source,
            line_color=HIGHLIGHT_COLOR,
            fill_color=None,
            line_width=4,
        )
        self.fig.ellipse(
            x="x",
            y="y",
            width="w",
            height="h",
            source=self.source_group_ellipse_highlight_source,
            line_color=HIGHLIGHT_COLOR,
            fill_color=None,
            line_width=4,
        )
        self.fig.rect(
            x="x",
            y="y",
            width="w",
            height="h",
            source=self.source_group_rect_highlight_source,
            line_color=HIGHLIGHT_COLOR,
            fill_color=None,
            line_width=4,
        )
        self.fig.segment(
            x0="x0",
            y0="y0",
            x1="x1",
            y1="y1",
            source=self.border_highlight_source,
            line_color=HIGHLIGHT_COLOR,
            line_width="w",
        )
        self.fig.circle(
            x="x",
            y="y",
            radius="r",
            source=self.larva_group_circle_highlight_source,
            line_color=HIGHLIGHT_COLOR,
            fill_color=None,
            line_width=4,
        )
        self.fig.ellipse(
            x="x",
            y="y",
            width="w",
            height="h",
            source=self.larva_group_ellipse_highlight_source,
            line_color=HIGHLIGHT_COLOR,
            fill_color=None,
            line_width=4,
        )
        self.fig.rect(
            x="x",
            y="y",
            width="w",
            height="h",
            source=self.larva_group_rect_highlight_source,
            line_color=HIGHLIGHT_COLOR,
            fill_color=None,
            line_width=4,
        )
        self._environment_legend = Legend(
            items=self._environment_legend_items(),
            click_policy="hide",
            background_fill_alpha=0.85,
            location="top_left",
        )
        self._larva_legend = Legend(
            items=self._larva_legend_items(),
            click_policy="hide",
            background_fill_alpha=0.85,
            location="top_right",
        )
        self.fig.add_layout(self._environment_legend)
        self.fig.add_layout(self._larva_legend)

        self._pane = pn.pane.Bokeh(self.fig, sizing_mode="stretch_width")

    def _environment_legend_items(self) -> list[LegendItem]:
        return [
            LegendItem(
                label="Source units",
                renderers=[self._source_units_renderer],
            ),
            LegendItem(
                label="Source groups",
                renderers=[
                    self._source_group_circle_renderer,
                    self._source_group_ellipse_renderer,
                    self._source_group_rect_renderer,
                    self._source_group_member_renderer,
                ],
            ),
            LegendItem(
                label="Borders",
                renderers=[self._border_renderer],
            ),
            LegendItem(
                label="Odor aura",
                renderers=[self._odor_aura_renderer, self._odor_peak_renderer],
            ),
            LegendItem(
                label="Odorscape",
                renderers=[self._odorscape_renderer],
            ),
            LegendItem(
                label="Windscape",
                renderers=[
                    self._windscape_segment_renderer,
                    self._windscape_head_renderer,
                ],
            ),
            LegendItem(
                label="Thermoscape",
                renderers=[
                    self._thermoscape_aura_renderer,
                    self._thermoscape_marker_renderer,
                ],
            ),
        ]

    def _larva_legend_items(self) -> list[LegendItem]:
        return [
            LegendItem(
                label="Larva groups",
                renderers=[
                    self._larva_group_circle_renderer,
                    self._larva_group_ellipse_renderer,
                    self._larva_group_rect_renderer,
                    self._larva_group_member_renderer,
                ],
            ),
            LegendItem(
                label="Larva trails",
                renderers=[self._sim_larva_trail_renderer],
            ),
            LegendItem(
                label="Larva body segments",
                renderers=[self._sim_larva_segment_renderer],
            ),
            LegendItem(
                label="Body contour",
                renderers=[self._sim_larva_body_contour_renderer],
            ),
            LegendItem(
                label="Larva midline",
                renderers=[self._sim_larva_midline_renderer],
            ),
            LegendItem(
                label="Larva markers",
                renderers=[
                    self._sim_larva_centroid_renderer,
                    self._sim_larva_head_renderer,
                ],
            ),
            LegendItem(
                label="Replay overlays",
                renderers=[self._dynamic_ring_renderer],
            ),
        ]

    def view(self) -> pn.viewable.Viewable:
        return self._pane

    def set_larva_frame(self, frame: LarvaPreviewFrame) -> None:
        centroid_rows: list[dict[str, Any]] = []
        head_rows: list[dict[str, Any]] = []
        midline_rows: list[dict[str, Any]] = []
        trail_rows: list[dict[str, Any]] = []
        segment_rows: list[dict[str, Any]] = []
        body_contour_rows: list[dict[str, Any]] = []
        label_rows: list[dict[str, Any]] = []

        for index, centroid in enumerate(frame.centroids):
            centroid_xy = _valid_xy(centroid)
            if centroid_xy is None:
                continue
            raw_color = frame.colors[index] if index < len(frame.colors) else None
            color = str(raw_color) if raw_color else DEFAULT_LARVA_COLOR
            larva_id = f"larva_{index}"
            centroid_rows.append(
                {
                    "x": centroid_xy[0],
                    "y": centroid_xy[1],
                    "color": color,
                    "id": larva_id,
                }
            )
            if index < len(frame.labels):
                label = str(frame.labels[index]).strip()
                if label:
                    label_rows.append(
                        {
                            "x": centroid_xy[0],
                            "y": centroid_xy[1],
                            "label": label,
                            "color": color,
                            "id": larva_id,
                        }
                    )
            midline_points: tuple[tuple[float, float], ...] = ()
            if index < len(frame.midlines):
                midline_points = _valid_path(frame.midlines[index])
                if midline_points:
                    midline_rows.append(
                        {
                            "xs": [point[0] for point in midline_points],
                            "ys": [point[1] for point in midline_points],
                            "color": color,
                            "id": larva_id,
                        }
                    )
            if index < len(frame.heads):
                head_xy = _valid_xy(frame.heads[index])
                if (
                    head_xy is not None
                    and self.snap_heads_to_midline
                    and midline_points
                ):
                    head_xy = _nearest_endpoint(head_xy, midline_points)
                if head_xy is not None:
                    head_rows.append(
                        {
                            "x": head_xy[0],
                            "y": head_xy[1],
                            "color": color,
                            "id": larva_id,
                        }
                    )
            if index < len(frame.trails):
                trail_points = _valid_path(frame.trails[index])
                if trail_points:
                    trail_rows.append(
                        {
                            "xs": [point[0] for point in trail_points],
                            "ys": [point[1] for point in trail_points],
                            "color": color,
                            "id": larva_id,
                        }
                    )
            if index < len(frame.segment_polygons):
                for segment_index, polygon in enumerate(frame.segment_polygons[index]):
                    polygon_points = _valid_path(polygon)
                    if len(polygon_points) >= 3:
                        segment_rows.append(
                            {
                                "xs": [point[0] for point in polygon_points],
                                "ys": [point[1] for point in polygon_points],
                                "color": color,
                                "id": f"{larva_id}_seg_{segment_index}",
                            }
                        )
            if index < len(frame.body_contours):
                contour_points = _valid_path(frame.body_contours[index])
                if len(contour_points) >= 3:
                    contour_points = _closed_path(contour_points)
                    body_contour_rows.append(
                        {
                            "xs": [point[0] for point in contour_points],
                            "ys": [point[1] for point in contour_points],
                            "color": color,
                            "id": f"{larva_id}_contour",
                        }
                    )

        self.sim_larva_centroid_source.data = _rows_to_data(
            centroid_rows, ["x", "y", "color", "id"]
        )
        self.sim_larva_head_source.data = _rows_to_data(
            head_rows, ["x", "y", "color", "id"]
        )
        self.sim_larva_midline_source.data = _rows_to_data(
            midline_rows, ["xs", "ys", "color", "id"]
        )
        self.sim_larva_trail_source.data = _rows_to_data(
            trail_rows, ["xs", "ys", "color", "id"]
        )
        self.sim_larva_segment_source.data = _rows_to_data(
            segment_rows, ["xs", "ys", "color", "id"]
        )
        self.sim_larva_body_contour_source.data = _rows_to_data(
            body_contour_rows, ["xs", "ys", "color", "id"]
        )
        self.sim_larva_label_source.data = _rows_to_data(
            label_rows, ["x", "y", "label", "color", "id"]
        )

    def set_dynamic_overlays(
        self, *, rings: tuple[CanvasRingOverlay, ...] = ()
    ) -> None:
        ring_rows: list[dict[str, Any]] = []
        for ring in rings:
            if not (math.isfinite(ring.x) and math.isfinite(ring.y)):
                continue
            if not math.isfinite(ring.radius) or ring.radius <= 0.0:
                continue
            ring_rows.append(
                {
                    "x": float(ring.x),
                    "y": float(ring.y),
                    "r": float(ring.radius),
                    "color": str(ring.color or DEFAULT_LARVA_COLOR),
                    "line_width": max(float(ring.line_width), 0.5),
                    "line_alpha": max(0.0, min(float(ring.line_alpha), 1.0)),
                    "line_dash": str(ring.line_dash or "solid"),
                }
            )
        self.dynamic_ring_source.data = _rows_to_data(
            ring_rows, ["x", "y", "r", "color", "line_width", "line_alpha", "line_dash"]
        )

    def clear_dynamic_overlays(self) -> None:
        self.dynamic_ring_source.data = _empty(
            ["x", "y", "r", "color", "line_width", "line_alpha", "line_dash"]
        )

    def clear_larva_frame(self) -> None:
        self.sim_larva_centroid_source.data = _empty(["x", "y", "color", "id"])
        self.sim_larva_head_source.data = _empty(["x", "y", "color", "id"])
        self.sim_larva_midline_source.data = _empty(["xs", "ys", "color", "id"])
        self.sim_larva_trail_source.data = _empty(["xs", "ys", "color", "id"])
        self.sim_larva_segment_source.data = _empty(["xs", "ys", "color", "id"])
        self.sim_larva_body_contour_source.data = _empty(["xs", "ys", "color", "id"])
        self.sim_larva_label_source.data = _empty(["x", "y", "label", "color", "id"])

    def clear(self) -> None:
        self.arena_source.data = {"x": [], "y": [], "w": [], "h": []}
        self.food_grid_overlay_source.data = {
            "x": [],
            "y": [],
            "w": [],
            "h": [],
            "color": [],
            "fill_alpha": [],
        }
        self.food_grid_cell_source.data = _empty(
            [
                "x",
                "y",
                "w",
                "h",
                "fill_color",
                "line_color",
                "fill_alpha",
                "line_alpha",
                "line_width",
            ]
        )
        self.thermoscape_aura_source.data = _empty(
            ["x", "y", "r", "color", "fill_alpha", "line_alpha", "id"]
        )
        self.thermoscape_marker_source.data = _empty(["x", "y", "color", "size", "id"])
        self.windscape_segment_source.data = _empty(
            ["x0", "y0", "x1", "y1", "color", "line_alpha"]
        )
        self.windscape_head_source.data = _empty(["x", "y", "angle", "color", "size"])
        self.odorscape_contour_source.data = _empty(
            ["x", "y", "r", "color", "line_alpha", "line_width", "id"]
        )
        self.odor_layer_source.data = _empty(
            ["x", "y", "r", "color", "fill_alpha", "id"]
        )
        self.odor_peak_source.data = _empty(
            ["x", "y", "r", "color", "fill_alpha", "id"]
        )
        self.food_source.data = _empty(
            [
                "x",
                "y",
                "r",
                "fill_color",
                "line_color",
                "id",
                "fill_alpha",
                "line_alpha",
                "line_width",
            ]
        )
        self.source_group_circle_source.data = _empty(
            ["x", "y", "r", "color", "fill_alpha", "line_alpha", "id"]
        )
        self.source_group_ellipse_source.data = _empty(
            ["x", "y", "w", "h", "color", "fill_alpha", "line_alpha", "id"]
        )
        self.source_group_rect_source.data = _empty(
            ["x", "y", "w", "h", "color", "fill_alpha", "line_alpha", "id"]
        )
        self.source_group_member_source.data = _empty(
            [
                "x",
                "y",
                "r",
                "fill_color",
                "line_color",
                "fill_alpha",
                "line_alpha",
                "line_width",
                "parent_id",
            ]
        )
        self.border_source.data = _empty(["x0", "y0", "x1", "y1", "w", "color", "id"])
        self.larva_group_circle_source.data = _empty(
            ["x", "y", "r", "color", "fill_alpha", "line_alpha", "id"]
        )
        self.larva_group_ellipse_source.data = _empty(
            ["x", "y", "w", "h", "color", "fill_alpha", "line_alpha", "id"]
        )
        self.larva_group_rect_source.data = _empty(
            ["x", "y", "w", "h", "color", "fill_alpha", "line_alpha", "id"]
        )
        self.larva_group_member_source.data = _empty(
            [
                "x0",
                "y0",
                "x1",
                "y1",
                "fill_color",
                "line_color",
                "fill_alpha",
                "line_alpha",
                "line_width",
                "parent_id",
            ]
        )
        self.clear_larva_frame()
        self.clear_dynamic_overlays()
        self.set_selected_object(None)
        self._state = None

    def set_state(self, state: EnvironmentCanvasState) -> None:
        self._arena = state.arena
        self.clear()
        self._state = state
        self._apply_arena(state.arena, show_arena_outline=state.show_arena_outline)
        self._apply_food_grid(state.food_grid)
        self._apply_scapes(state)
        self._apply_objects(state.objects)

    def set_selected_object(self, object_id: str | None) -> None:
        self.food_highlight_source.data = {"x": [], "y": [], "r": [], "color": []}
        self.source_group_circle_highlight_source.data = {
            "x": [],
            "y": [],
            "r": [],
            "color": [],
        }
        self.source_group_ellipse_highlight_source.data = {
            "x": [],
            "y": [],
            "w": [],
            "h": [],
            "color": [],
        }
        self.source_group_rect_highlight_source.data = {
            "x": [],
            "y": [],
            "w": [],
            "h": [],
            "color": [],
        }
        self.border_highlight_source.data = {
            "x0": [],
            "y0": [],
            "x1": [],
            "y1": [],
            "w": [],
            "color": [],
        }
        self.larva_group_circle_highlight_source.data = {
            "x": [],
            "y": [],
            "r": [],
            "color": [],
        }
        self.larva_group_ellipse_highlight_source.data = {
            "x": [],
            "y": [],
            "w": [],
            "h": [],
            "color": [],
        }
        self.larva_group_rect_highlight_source.data = {
            "x": [],
            "y": [],
            "w": [],
            "h": [],
            "color": [],
        }
        if not object_id or self._state is None:
            return
        obj = next(
            (
                candidate
                for candidate in self._state.objects
                if candidate.object_id == object_id
            ),
            None,
        )
        if obj is None:
            return
        if obj.object_type == "source_unit" and obj.x is not None and obj.y is not None:
            self.food_highlight_source.data = {
                "x": [obj.x],
                "y": [obj.y],
                "r": [max(_safe_float(obj.radius, 0.003) * 1.35, 0.004)],
                "color": [HIGHLIGHT_COLOR],
            }
        elif obj.object_type == "source_group":
            self._set_group_highlight(
                obj,
                self.source_group_circle_highlight_source,
                self.source_group_ellipse_highlight_source,
                self.source_group_rect_highlight_source,
            )
        elif obj.object_type == "larva_group":
            self._set_group_highlight(
                obj,
                self.larva_group_circle_highlight_source,
                self.larva_group_ellipse_highlight_source,
                self.larva_group_rect_highlight_source,
            )
        elif obj.object_type == "border_segment":
            self.border_highlight_source.data = {
                "x0": [obj.x],
                "y0": [obj.y],
                "x1": [obj.x2],
                "y1": [obj.y2],
                "w": [max(4, int(_safe_float(obj.width, 0.001) * 1500) + 2)],
                "color": [HIGHLIGHT_COLOR],
            }

    def _arena_dimensions(self) -> tuple[float, float]:
        dims = self._arena.dims
        width = abs(_safe_float(dims[0] if len(dims) > 0 else 0.2, 0.2))
        height = abs(_safe_float(dims[1] if len(dims) > 1 else width, width))
        if width <= 0:
            width = 0.2
        if height <= 0:
            height = width
        return width, height

    def _apply_arena(self, arena: CanvasArena, *, show_arena_outline: bool) -> None:
        self._arena = arena
        width, height = self._arena_dimensions()
        self.arena_source.data = {"x": [0.0], "y": [0.0], "w": [width], "h": [height]}
        is_circular = str(arena.geometry).lower().startswith("circ")
        self._arena_circle_renderer.visible = is_circular and show_arena_outline
        self._food_grid_circle_renderer.visible = is_circular
        self._arena_rect_renderer.visible = (not is_circular) and show_arena_outline
        self._food_grid_rect_renderer.visible = not is_circular
        pad_scale = 1.24
        required_x_span = width * pad_scale
        required_y_span = height * pad_scale
        canvas_ratio = self.width / self.height
        if required_x_span / required_y_span < canvas_ratio:
            y_span = required_y_span
            x_span = y_span * canvas_ratio
        else:
            x_span = required_x_span
            y_span = x_span / canvas_ratio
        self.fig.x_range.start = -x_span / 2
        self.fig.x_range.end = x_span / 2
        self.fig.y_range.start = -y_span / 2
        self.fig.y_range.end = y_span / 2

    def _apply_food_grid(self, food_grid: dict[str, Any] | None) -> None:
        if not isinstance(food_grid, dict):
            return
        width, height = self._arena_dimensions()
        color = str(food_grid.get("color") or DEFAULT_SOURCE_COLOR)
        self.food_grid_overlay_source.data = {
            "x": [0.0],
            "y": [0.0],
            "w": [width],
            "h": [height],
            "color": [color],
            "fill_alpha": [0.08],
        }
        grid_dims = food_grid.get("grid_dims", (0, 0))
        try:
            nx = max(int(grid_dims[0]), 1)
            ny = max(int(grid_dims[1]), 1)
        except Exception:
            return
        cell_w = width / nx
        cell_h = height / ny
        fill_color = _mix_hex_colors(color, "#ffffff", 0.32)
        line_color = _mix_hex_colors(color, "#111111", 0.20)
        xs = np.linspace(-width / 2 + cell_w / 2, width / 2 - cell_w / 2, nx)
        ys = np.linspace(-height / 2 + cell_h / 2, height / 2 - cell_h / 2, ny)
        rows: list[dict[str, Any]] = []
        for center_x in xs:
            for center_y in ys:
                if str(self._arena.geometry).lower().startswith("circ"):
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
        self.food_grid_cell_source.data = _rows_to_data(
            rows,
            [
                "x",
                "y",
                "w",
                "h",
                "fill_color",
                "line_color",
                "fill_alpha",
                "line_alpha",
                "line_width",
            ],
        )

    def _apply_scapes(self, state: EnvironmentCanvasState) -> None:
        try:
            self._apply_odorscape(state.odorscape, state.objects)
        except Exception:
            self.odorscape_contour_source.data = _empty(
                ["x", "y", "r", "color", "line_alpha", "line_width", "id"]
            )
        try:
            self._apply_windscape(state.windscape)
        except Exception:
            self.windscape_segment_source.data = _empty(
                ["x0", "y0", "x1", "y1", "color", "line_alpha"]
            )
            self.windscape_head_source.data = _empty(
                ["x", "y", "angle", "color", "size"]
            )
        try:
            self._apply_thermoscape(state.thermoscape)
        except Exception:
            self.thermoscape_aura_source.data = _empty(
                ["x", "y", "r", "color", "fill_alpha", "line_alpha", "id"]
            )
            self.thermoscape_marker_source.data = _empty(
                ["x", "y", "color", "size", "id"]
            )

    def _apply_odorscape(
        self, odorscape: dict[str, Any] | None, objects: tuple[CanvasObject, ...]
    ) -> None:
        if not isinstance(odorscape, dict):
            return
        color = str(odorscape.get("color") or DEFAULT_SOURCE_COLOR)
        mode = str(
            odorscape.get("odorscape") or odorscape.get("unique_id") or "Gaussian"
        )
        line_alpha_scale = {"Analytical": 0.42, "Gaussian": 0.32, "Diffusion": 0.24}
        mults = [0.5, 1.0, 1.5, 2.2]
        rows: list[dict[str, Any]] = []
        for obj in objects:
            if obj.object_type not in {"source_unit", "source_group"}:
                continue
            if obj.x is None or obj.y is None or not obj.odor_id:
                continue
            spread = _safe_float(obj.odor_spread)
            intensity = _safe_float(obj.odor_intensity)
            if spread <= 0 or intensity <= 0:
                continue
            source_r = max(_safe_float(obj.radius, 0.003), 0.002)
            for index, mult in enumerate(mults):
                rows.append(
                    {
                        "x": float(obj.x),
                        "y": float(obj.y),
                        "r": source_r + spread * mult,
                        "color": color,
                        "line_alpha": max(
                            0.08,
                            line_alpha_scale.get(mode, 0.3) - index * 0.07,
                        ),
                        "line_width": max(1.0, 2.2 - index * 0.35),
                        "id": str(obj.object_id),
                    }
                )
        self.odorscape_contour_source.data = _rows_to_data(
            rows, ["x", "y", "r", "color", "line_alpha", "line_width", "id"]
        )

    def _apply_windscape(self, windscape: dict[str, Any] | None) -> None:
        if not isinstance(windscape, dict):
            return
        speed = _safe_float(windscape.get("wind_speed"), 0.0)
        if speed <= 0:
            return
        width, height = self._arena_dimensions()
        d = max(width, height) * 0.48
        n = 9
        ds = d / n * math.sqrt(2)
        direction_rad = _safe_float(windscape.get("wind_direction"), math.pi)
        angle = -direction_rad
        color = str(windscape.get("color") or "#ff0000")
        segment_rows = []
        head_rows = []
        for i in range(n):
            y_offset = (i - n / 2) * ds
            p0 = _rotate_point(-d, y_offset, angle)
            p1 = _rotate_point(d, y_offset, angle)
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
        self.windscape_segment_source.data = _rows_to_data(
            segment_rows, ["x0", "y0", "x1", "y1", "color", "line_alpha"]
        )
        self.windscape_head_source.data = _rows_to_data(
            head_rows, ["x", "y", "angle", "color", "size"]
        )

    def _apply_thermoscape(self, thermoscape: dict[str, Any] | None) -> None:
        if not isinstance(thermoscape, dict):
            return
        spread = max(_safe_float(thermoscape.get("spread"), 0.1), 0.001)
        source_map = thermoscape.get("thermo_sources") or {}
        dtemp_map = thermoscape.get("thermo_source_dTemps") or {}
        source_rows = thermoscape.get("sources") or []
        records: list[dict[str, Any]] = []
        if isinstance(source_map, dict):
            for source_id, position in source_map.items():
                try:
                    x, y = position
                except Exception:
                    continue
                dtemp = (
                    dtemp_map.get(source_id, 0.0)
                    if isinstance(dtemp_map, dict)
                    else 0.0
                )
                records.append(
                    {
                        "id": str(source_id),
                        "x": _safe_float(x),
                        "y": _safe_float(y),
                        "dTemp": dtemp,
                    }
                )
        if isinstance(source_rows, list):
            for row in source_rows:
                if isinstance(row, dict):
                    records.append(row)
        aura_rows: list[dict[str, Any]] = []
        marker_rows: list[dict[str, Any]] = []
        for row in records:
            source_id = str(row.get("id") or "").strip()
            if not source_id:
                source_id = "thermal_source"
            x = _safe_float(row.get("x"), 0.0)
            y = _safe_float(row.get("y"), 0.0)
            dtemp = _safe_float(row.get("dTemp"), _safe_float(row.get("dtemp"), 0.0))
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
        self.thermoscape_aura_source.data = _rows_to_data(
            aura_rows, ["x", "y", "r", "color", "fill_alpha", "line_alpha", "id"]
        )
        self.thermoscape_marker_source.data = _rows_to_data(
            marker_rows, ["x", "y", "color", "size", "id"]
        )

    def _apply_objects(self, objects: tuple[CanvasObject, ...]) -> None:
        food_rows: list[dict[str, Any]] = []
        odor_rows: list[dict[str, Any]] = []
        peak_rows: list[dict[str, Any]] = []
        circle_rows: list[dict[str, Any]] = []
        ellipse_rows: list[dict[str, Any]] = []
        rect_rows: list[dict[str, Any]] = []
        member_rows: list[dict[str, Any]] = []
        border_rows: list[dict[str, Any]] = []
        larva_circle_rows: list[dict[str, Any]] = []
        larva_ellipse_rows: list[dict[str, Any]] = []
        larva_rect_rows: list[dict[str, Any]] = []
        larva_member_rows: list[dict[str, Any]] = []

        for obj in objects:
            if (
                obj.object_type == "source_unit"
                and obj.x is not None
                and obj.y is not None
            ):
                fill_color, line_color, fill_alpha, line_alpha, line_width = (
                    _source_visual_state(amount=obj.amount, color=obj.color)
                )
                food_rows.append(
                    {
                        "x": obj.x,
                        "y": obj.y,
                        "r": max(_safe_float(obj.radius, 0.003), 0.001),
                        "fill_color": fill_color,
                        "line_color": line_color,
                        "id": obj.object_id,
                        "fill_alpha": fill_alpha,
                        "line_alpha": line_alpha,
                        "line_width": line_width,
                    }
                )
                odor_rows.extend(self._odor_rows_for(obj, obj.x, obj.y))
                peak = self._odor_peak_for(obj, obj.x, obj.y)
                if peak is not None:
                    peak_rows.append(peak)
            elif obj.object_type == "source_group":
                self._append_group_footprint(
                    obj, circle_rows, ellipse_rows, rect_rows, 0.08, 0.9
                )
                members = self._group_member_rows(obj)
                member_rows.extend(members)
                for member in members:
                    x = member.get("x")
                    y = member.get("y")
                    odor_rows.extend(self._odor_rows_for(obj, x, y))
                    peak = self._odor_peak_for(obj, x, y)
                    if peak is not None:
                        peak_rows.append(peak)
            elif obj.object_type == "border_segment":
                border_rows.append(
                    {
                        "x0": obj.x,
                        "y0": obj.y,
                        "x1": obj.x2,
                        "y1": obj.y2,
                        "w": max(1, int(_safe_float(obj.width, 0.001) * 1500)),
                        "color": obj.color or LANE_MODELS_COLOR_DARK,
                        "id": obj.object_id,
                    }
                )
            elif obj.object_type == "larva_group":
                self._append_group_footprint(
                    obj,
                    larva_circle_rows,
                    larva_ellipse_rows,
                    larva_rect_rows,
                    0.06,
                    0.72,
                    default_color=DEFAULT_LARVA_COLOR,
                )
                larva_member_rows.extend(
                    self._group_member_rows(obj, default_color=DEFAULT_LARVA_COLOR)
                )

        self.food_source.data = _rows_to_data(
            food_rows,
            [
                "x",
                "y",
                "r",
                "fill_color",
                "line_color",
                "id",
                "fill_alpha",
                "line_alpha",
                "line_width",
            ],
        )
        self.odor_layer_source.data = _rows_to_data(
            odor_rows, ["x", "y", "r", "color", "fill_alpha", "id"]
        )
        self.odor_peak_source.data = _rows_to_data(
            peak_rows, ["x", "y", "r", "color", "fill_alpha", "id"]
        )
        self.source_group_circle_source.data = _rows_to_data(
            circle_rows, ["x", "y", "r", "color", "fill_alpha", "line_alpha", "id"]
        )
        self.source_group_ellipse_source.data = _rows_to_data(
            ellipse_rows,
            ["x", "y", "w", "h", "color", "fill_alpha", "line_alpha", "id"],
        )
        self.source_group_rect_source.data = _rows_to_data(
            rect_rows, ["x", "y", "w", "h", "color", "fill_alpha", "line_alpha", "id"]
        )
        self.source_group_member_source.data = _rows_to_data(
            member_rows,
            [
                "x",
                "y",
                "r",
                "fill_color",
                "line_color",
                "fill_alpha",
                "line_alpha",
                "line_width",
                "parent_id",
            ],
        )
        self.border_source.data = _rows_to_data(
            border_rows, ["x0", "y0", "x1", "y1", "w", "color", "id"]
        )
        self.larva_group_circle_source.data = _rows_to_data(
            larva_circle_rows,
            ["x", "y", "r", "color", "fill_alpha", "line_alpha", "id"],
        )
        self.larva_group_ellipse_source.data = _rows_to_data(
            larva_ellipse_rows,
            ["x", "y", "w", "h", "color", "fill_alpha", "line_alpha", "id"],
        )
        self.larva_group_rect_source.data = _rows_to_data(
            larva_rect_rows,
            ["x", "y", "w", "h", "color", "fill_alpha", "line_alpha", "id"],
        )
        self.larva_group_member_source.data = _rows_to_data(
            larva_member_rows,
            [
                "x0",
                "y0",
                "x1",
                "y1",
                "fill_color",
                "line_color",
                "fill_alpha",
                "line_alpha",
                "line_width",
                "parent_id",
            ],
        )

    def _odor_rows_for(
        self, obj: CanvasObject, x: Any, y: Any
    ) -> list[dict[str, object]]:
        return _build_odor_layers(
            x=None if x is None else _safe_float(x),
            y=None if y is None else _safe_float(y),
            source_radius=obj.radius,
            odor_id=obj.odor_id,
            odor_intensity=obj.odor_intensity,
            odor_spread=obj.odor_spread,
            color=obj.color,
            source_id=obj.object_id,
        )

    def _odor_peak_for(
        self, obj: CanvasObject, x: Any, y: Any
    ) -> dict[str, object] | None:
        return _build_odor_peak(
            x=None if x is None else _safe_float(x),
            y=None if y is None else _safe_float(y),
            source_radius=obj.radius,
            odor_id=obj.odor_id,
            odor_intensity=obj.odor_intensity,
            odor_spread=obj.odor_spread,
            color=obj.color,
            source_id=obj.object_id,
        )

    def _append_group_footprint(
        self,
        obj: CanvasObject,
        circle_rows: list[dict[str, Any]],
        ellipse_rows: list[dict[str, Any]],
        rect_rows: list[dict[str, Any]],
        fill_alpha: float,
        line_alpha: float,
        *,
        default_color: str = DEFAULT_SOURCE_COLOR,
    ) -> None:
        if obj.x is None or obj.y is None:
            return
        if obj.distribution_show_shape is False:
            return
        scale_x, scale_y = _distribution_scale_pair(obj)
        width = max(scale_x * 2.0, 0.002)
        height = max(scale_y * 2.0, 0.002)
        color = str(obj.color or default_color)
        row = {
            "x": float(obj.x),
            "y": float(obj.y),
            "color": color,
            "fill_alpha": fill_alpha,
            "line_alpha": line_alpha,
            "id": obj.object_id,
        }
        shape = _normalize_group_shape(obj.distribution_shape)
        if shape == "circle" and (
            obj.object_type == "source_group"
            or math.isclose(width, height, rel_tol=1e-6, abs_tol=1e-9)
        ):
            circle_rows.append({**row, "r": max(width, height) / 2.0})
        elif shape in {"circle", "oval"}:
            ellipse_rows.append({**row, "w": width, "h": height})
        else:
            rect_rows.append({**row, "w": width, "h": height})

    def _set_group_highlight(
        self,
        obj: CanvasObject,
        circle_source: ColumnDataSource,
        ellipse_source: ColumnDataSource,
        rect_source: ColumnDataSource,
    ) -> None:
        if obj.x is None or obj.y is None:
            return
        scale_x, scale_y = _distribution_scale_pair(obj)
        width = max(scale_x * 2.0, 0.002)
        height = max(scale_y * 2.0, 0.002)
        shape = _normalize_group_shape(obj.distribution_shape)
        if shape == "circle" and (
            obj.object_type == "source_group"
            or math.isclose(width, height, rel_tol=1e-6, abs_tol=1e-9)
        ):
            circle_source.data = {
                "x": [obj.x],
                "y": [obj.y],
                "r": [max(width, height) / 2.0],
                "color": [HIGHLIGHT_COLOR],
            }
        elif shape in {"circle", "oval"}:
            ellipse_source.data = {
                "x": [obj.x],
                "y": [obj.y],
                "w": [width],
                "h": [height],
                "color": [HIGHLIGHT_COLOR],
            }
        else:
            rect_source.data = {
                "x": [obj.x],
                "y": [obj.y],
                "w": [width],
                "h": [height],
                "color": [HIGHLIGHT_COLOR],
            }

    def _group_member_positions(self, obj: CanvasObject) -> list[tuple[float, float]]:
        if (
            obj.object_type not in {"source_group", "larva_group"}
            or obj.x is None
            or obj.y is None
        ):
            return []
        count = _safe_int(obj.distribution_n, 0)
        if count <= 0:
            return []
        distribution = Spatial_Distro(
            N=count,
            shape=_normalize_group_shape(obj.distribution_shape),
            mode=str(obj.distribution_mode or "uniform"),
            loc=(float(obj.x), float(obj.y)),
            scale=_distribution_scale_pair(obj),
        )
        state = np.random.get_state()
        seed = _stable_preview_seed(
            obj.object_id,
            obj.distribution_mode,
            obj.distribution_shape,
            obj.distribution_n,
            obj.x,
            obj.y,
            obj.distribution_scale_x,
            obj.distribution_scale_y,
        )
        try:
            np.random.seed(seed)
            return [
                (float(member_x), float(member_y))
                for member_x, member_y in distribution()
            ]
        except Exception:
            return []
        finally:
            np.random.set_state(state)

    def _group_member_rows(
        self, obj: CanvasObject, *, default_color: str = DEFAULT_SOURCE_COLOR
    ) -> list[dict[str, Any]]:
        positions = self._group_member_positions(obj)
        if not positions:
            return []
        base_color = str(obj.color or default_color)
        fill_color = _mix_hex_colors(base_color, "#ffffff", 0.18)
        line_color = _mix_hex_colors(base_color, "#111111", 0.04)
        if obj.object_type == "larva_group":
            rows: list[dict[str, Any]] = []
            for idx, (member_x, member_y) in enumerate(positions):
                angle = _stable_member_angle(obj, idx)
                dx = STATIC_LARVA_GROUP_MEMBER_HALF_LENGTH * math.cos(angle)
                dy = STATIC_LARVA_GROUP_MEMBER_HALF_LENGTH * math.sin(angle)
                rows.append(
                    {
                        "x0": member_x - dx,
                        "y0": member_y - dy,
                        "x1": member_x + dx,
                        "y1": member_y + dy,
                        "fill_color": fill_color,
                        "line_color": line_color,
                        "fill_alpha": 0.0,
                        "line_alpha": 0.95,
                        "line_width": 1.4,
                        "parent_id": obj.object_id,
                    }
                )
            return rows
        radius = max(_safe_float(obj.radius, 0.0018), 0.0012)
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
