"""
Shared logic behind the environment builder.

Normalizes the builder's object rows against the stored environment payload,
so the canvas, the form and the saved configuration stay consistent.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
import re
from typing import Any

from larvaworld.lib import util
from larvaworld.portal.canvas_widgets.environment_mapping import (
    env_params_to_canvas_state,
)
from larvaworld.portal.canvas_widgets.environment_models import EnvironmentCanvasState

SOURCE_UNIT = "source_unit"
SOURCE_GROUP = "source_group"
BORDER_SEGMENT = "border_segment"

DEFAULT_SOURCE_UNIT_RADIUS = 0.003
DEFAULT_SOURCE_GROUP_RADIUS = 0.003
DEFAULT_BORDER_WIDTH = 0.001
DEFAULT_GROUP_N = 30
DEFAULT_GROUP_SCALE = 0.012

_REGEX_PRESET_NAME = re.compile(r"[^a-zA-Z0-9._-]+")


@dataclass(frozen=True)
class EnvBuilderObjectRow:
    """One object row shared by the builder's views."""

    object_id: str
    object_type: str
    x: float | None = None
    y: float | None = None
    x2: float | None = None
    y2: float | None = None
    radius: float | None = None
    width: float | None = None
    color: str | None = None
    amount: float | None = None
    odor_id: str | None = None
    odor_intensity: float | None = None
    odor_spread: float | None = None
    substrate_type: str | None = None
    substrate_quality: float | None = None
    can_be_carried: bool | None = None
    can_be_displaced: bool | None = None
    regeneration: bool | None = None
    distribution_mode: str | None = None
    distribution_shape: str | None = None
    distribution_n: int | None = None
    distribution_scale_x: float | None = None
    distribution_scale_y: float | None = None
    distribution_show_shape: bool | None = None


def normalize_group_shape(shape: str | None) -> str:
    normalized = str(shape or "circle").strip().lower()
    if normalized in {"circle", "circular"}:
        return "circle"
    if normalized in {"oval", "ellipse", "elliptical"}:
        return "oval"
    if normalized in {"rect", "rectangle", "rectangular"}:
        return "rect"
    return "circle"


def normalize_preset_filename(
    name: str, *, default: str = "environment_builder_config"
) -> str:
    cleaned = _REGEX_PRESET_NAME.sub("_", str(name).strip()).strip("._-")
    if not cleaned:
        cleaned = default
    if not cleaned.endswith(".json"):
        cleaned = f"{cleaned}.json"
    return cleaned


def translate_environment_payload(payload: Any) -> util.AttrDict:
    plain = _to_plain(payload)
    if not isinstance(plain, dict):
        plain = {}

    arena = plain.get("arena") or {}
    food_params = plain.get("food_params") or {}
    border_list = plain.get("border_list") or {}

    normalized = {
        "arena": _ensure_mapping(
            arena, {"geometry": "rectangular", "dims": (0.2, 0.2), "torus": False}
        ),
        "food_params": _ensure_mapping(
            food_params,
            {"source_units": {}, "source_groups": {}, "food_grid": None},
        ),
        "border_list": _ensure_mapping(border_list, {}),
        "odorscape": plain.get("odorscape"),
        "windscape": plain.get("windscape"),
        "thermoscape": plain.get("thermoscape"),
    }
    normalized_food = normalized["food_params"]
    if "source_units" not in normalized_food or normalized_food["source_units"] is None:
        normalized_food["source_units"] = {}
    if (
        "source_groups" not in normalized_food
        or normalized_food["source_groups"] is None
    ):
        normalized_food["source_groups"] = {}
    if "food_grid" not in normalized_food:
        normalized_food["food_grid"] = None
    return util.AttrDict(normalized)


def object_rows_from_payload(payload: Any) -> tuple[EnvBuilderObjectRow, ...]:
    data = translate_environment_payload(payload)
    rows: list[EnvBuilderObjectRow] = []

    food_params = data["food_params"] or {}
    source_units = food_params.get("source_units") or {}
    if _is_mapping(source_units):
        for object_id, source in source_units.items():
            rows.append(_source_unit_row(str(object_id), source))

    source_groups = food_params.get("source_groups") or {}
    if _is_mapping(source_groups):
        for object_id, group in source_groups.items():
            rows.append(_source_group_row(str(object_id), group))

    border_list = data["border_list"] or {}
    if _is_mapping(border_list):
        for object_id, border in border_list.items():
            rows.extend(_border_rows(str(object_id), border))

    return tuple(rows)


def payload_with_object_rows(
    base_payload: Any, rows: list[EnvBuilderObjectRow] | tuple[EnvBuilderObjectRow, ...]
) -> dict[str, Any]:
    payload = _to_plain(translate_environment_payload(base_payload))
    rows_by_type: dict[str, list[EnvBuilderObjectRow]] = {
        SOURCE_UNIT: [],
        SOURCE_GROUP: [],
        BORDER_SEGMENT: [],
    }
    for row in rows:
        rows_by_type.setdefault(row.object_type, []).append(row)

    food_params = payload.setdefault("food_params", {})
    food_params["source_units"] = {
        row.object_id: _row_to_source_unit_payload(row)
        for row in rows_by_type[SOURCE_UNIT]
    }
    food_params["source_groups"] = {
        row.object_id: _row_to_source_group_payload(row)
        for row in rows_by_type[SOURCE_GROUP]
    }
    payload["border_list"] = _border_rows_to_payload(rows_by_type[BORDER_SEGMENT])
    return payload


def add_source_unit(base_payload: Any, row: EnvBuilderObjectRow) -> dict[str, Any]:
    rows = list(object_rows_from_payload(base_payload))
    rows.append(row)
    return payload_with_object_rows(base_payload, rows)


def update_source_unit(base_payload: Any, row: EnvBuilderObjectRow) -> dict[str, Any]:
    return _replace_row(base_payload, row, SOURCE_UNIT)


def delete_source_unit(base_payload: Any, object_id: str) -> dict[str, Any]:
    return _delete_row(base_payload, object_id, SOURCE_UNIT)


def add_source_group(base_payload: Any, row: EnvBuilderObjectRow) -> dict[str, Any]:
    rows = list(object_rows_from_payload(base_payload))
    rows.append(row)
    return payload_with_object_rows(base_payload, rows)


def update_source_group(base_payload: Any, row: EnvBuilderObjectRow) -> dict[str, Any]:
    return _replace_row(base_payload, row, SOURCE_GROUP)


def delete_source_group(base_payload: Any, object_id: str) -> dict[str, Any]:
    return _delete_row(base_payload, object_id, SOURCE_GROUP)


def add_border_segment(base_payload: Any, row: EnvBuilderObjectRow) -> dict[str, Any]:
    rows = list(object_rows_from_payload(base_payload))
    rows.append(row)
    return payload_with_object_rows(base_payload, rows)


def update_border_segment(
    base_payload: Any, row: EnvBuilderObjectRow
) -> dict[str, Any]:
    return _replace_row(base_payload, row, BORDER_SEGMENT)


def delete_border_segment(base_payload: Any, object_id: str) -> dict[str, Any]:
    return _delete_row(base_payload, object_id, BORDER_SEGMENT)


def build_canvas_state(payload: Any) -> EnvironmentCanvasState:
    return env_params_to_canvas_state(translate_environment_payload(payload))


def _replace_row(
    base_payload: Any, row: EnvBuilderObjectRow, object_type: str
) -> dict[str, Any]:
    rows = [
        item
        for item in object_rows_from_payload(base_payload)
        if not (item.object_type == object_type and item.object_id == row.object_id)
    ]
    rows.append(row)
    return payload_with_object_rows(base_payload, rows)


def _delete_row(base_payload: Any, object_id: str, object_type: str) -> dict[str, Any]:
    rows = [
        item
        for item in object_rows_from_payload(base_payload)
        if not (item.object_type == object_type and item.object_id == object_id)
    ]
    return payload_with_object_rows(base_payload, rows)


def _source_unit_row(object_id: str, source: Any) -> EnvBuilderObjectRow:
    pos = _pair(_get(source, "pos"), default=(None, None))
    odor = _get(source, "odor") or {}
    substrate = _get(source, "substrate") or {}
    return EnvBuilderObjectRow(
        object_id=object_id,
        object_type=SOURCE_UNIT,
        x=pos[0],
        y=pos[1],
        radius=_float_or_none(_get(source, "radius"), DEFAULT_SOURCE_UNIT_RADIUS),
        color=_str_or_none(_get(source, "color")),
        amount=_float_or_none(_get(source, "amount")),
        odor_id=_str_or_none(_get(odor, "id")),
        odor_intensity=_float_or_none(_get(odor, "intensity")),
        odor_spread=_float_or_none(_get(odor, "spread")),
        substrate_type=_str_or_none(_get(substrate, "type")),
        substrate_quality=_float_or_none(_get(substrate, "quality")),
        can_be_carried=_bool_or_none(_get(source, "can_be_carried")),
        can_be_displaced=_bool_or_none(_get(source, "can_be_displaced")),
        regeneration=_bool_or_none(_get(source, "regeneration")),
    )


def _source_group_row(object_id: str, group: Any) -> EnvBuilderObjectRow:
    distribution = _get(group, "distribution") or {}
    pos = _pair(_get(distribution, "loc", _get(group, "pos")), default=(None, None))
    odor = _get(group, "odor") or {}
    substrate = _get(group, "substrate") or {}
    scale = _pair(
        _get(distribution, "scale"), default=(DEFAULT_GROUP_SCALE, DEFAULT_GROUP_SCALE)
    )
    return EnvBuilderObjectRow(
        object_id=object_id,
        object_type=SOURCE_GROUP,
        x=pos[0],
        y=pos[1],
        radius=_float_or_none(_get(group, "radius"), DEFAULT_SOURCE_GROUP_RADIUS),
        color=_str_or_none(_get(group, "color")),
        amount=_float_or_none(_get(group, "amount")),
        odor_id=_str_or_none(_get(odor, "id")),
        odor_intensity=_float_or_none(_get(odor, "intensity")),
        odor_spread=_float_or_none(_get(odor, "spread")),
        substrate_type=_str_or_none(_get(substrate, "type")),
        substrate_quality=_float_or_none(_get(substrate, "quality")),
        can_be_carried=_bool_or_none(_get(group, "can_be_carried")),
        can_be_displaced=_bool_or_none(_get(group, "can_be_displaced")),
        regeneration=_bool_or_none(_get(group, "regeneration")),
        distribution_mode=_str_or_none(_get(distribution, "mode", "uniform")),
        distribution_shape=normalize_group_shape(_get(distribution, "shape", "circle")),
        distribution_n=_int_or_none(_get(distribution, "N"), DEFAULT_GROUP_N),
        distribution_scale_x=scale[0],
        distribution_scale_y=scale[1],
        distribution_show_shape=_bool_or_none(
            _get(group, "distribution_show_shape", True), default=True
        ),
    )


def _border_rows(object_id: str, border: Any) -> list[EnvBuilderObjectRow]:
    vertices = _get(border, "vertices")
    segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    if isinstance(vertices, (list, tuple)):
        points = [_pair(item, default=(None, None)) for item in vertices]
        filtered = [point for point in points if None not in point]
        for idx in range(0, len(filtered) - 1, 2):
            segments.append((filtered[idx], filtered[idx + 1]))
    if not segments:
        border_xy = _get(border, "border_xy")
        if isinstance(border_xy, (list, tuple)):
            for path in border_xy:
                points = [_pair(item, default=(None, None)) for item in path]
                filtered = [point for point in points if None not in point]
                for idx in range(0, len(filtered) - 1, 2):
                    segments.append((filtered[idx], filtered[idx + 1]))
    rows: list[EnvBuilderObjectRow] = []
    for idx, (start, end) in enumerate(segments):
        rows.append(
            EnvBuilderObjectRow(
                object_id=object_id if len(segments) == 1 else f"{object_id}:{idx}",
                object_type=BORDER_SEGMENT,
                x=start[0],
                y=start[1],
                x2=end[0],
                y2=end[1],
                width=_float_or_none(_get(border, "width"), DEFAULT_BORDER_WIDTH),
                color=_str_or_none(_get(border, "color")),
            )
        )
    return rows


def _row_to_source_unit_payload(row: EnvBuilderObjectRow) -> dict[str, Any]:
    return {
        "pos": [row.x, row.y],
        "radius": row.radius if row.radius is not None else DEFAULT_SOURCE_UNIT_RADIUS,
        "amount": row.amount if row.amount is not None else 0.0,
        "can_be_carried": bool(row.can_be_carried)
        if row.can_be_carried is not None
        else False,
        "can_be_displaced": bool(row.can_be_displaced)
        if row.can_be_displaced is not None
        else False,
        "regeneration": bool(row.regeneration)
        if row.regeneration is not None
        else False,
        "odor": {
            "id": row.odor_id,
            "intensity": row.odor_intensity,
            "spread": row.odor_spread,
        },
        "substrate": {
            "type": row.substrate_type or "standard",
            "quality": row.substrate_quality
            if row.substrate_quality is not None
            else 1.0,
        },
        "color": row.color,
    }


def _row_to_source_group_payload(row: EnvBuilderObjectRow) -> dict[str, Any]:
    shape = normalize_group_shape(row.distribution_shape)
    scale_x = (
        row.distribution_scale_x
        if row.distribution_scale_x is not None
        else DEFAULT_GROUP_SCALE
    )
    scale_y = (
        row.distribution_scale_y
        if row.distribution_scale_y is not None
        else DEFAULT_GROUP_SCALE
    )
    if shape == "circle":
        scale_y = scale_x
    return {
        "radius": row.radius if row.radius is not None else DEFAULT_SOURCE_GROUP_RADIUS,
        "amount": row.amount if row.amount is not None else 0.0,
        "can_be_carried": bool(row.can_be_carried)
        if row.can_be_carried is not None
        else False,
        "can_be_displaced": bool(row.can_be_displaced)
        if row.can_be_displaced is not None
        else False,
        "regeneration": bool(row.regeneration)
        if row.regeneration is not None
        else False,
        "distribution": {
            "N": row.distribution_n
            if row.distribution_n is not None
            else DEFAULT_GROUP_N,
            "loc": [row.x, row.y],
            "mode": row.distribution_mode or "uniform",
            "shape": shape,
            "scale": [scale_x, scale_y],
        },
        "odor": {
            "id": row.odor_id,
            "intensity": row.odor_intensity,
            "spread": row.odor_spread,
        },
        "substrate": {
            "type": row.substrate_type or "standard",
            "quality": row.substrate_quality
            if row.substrate_quality is not None
            else 1.0,
        },
        "color": row.color,
        "distribution_show_shape": True
        if row.distribution_show_shape is None
        else bool(row.distribution_show_shape),
    }


def _border_rows_to_payload(rows: list[EnvBuilderObjectRow]) -> dict[str, Any]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        base_id = row.object_id.split(":", 1)[0]
        entry = grouped.setdefault(
            base_id,
            {
                "vertices": [],
                "width": row.width if row.width is not None else DEFAULT_BORDER_WIDTH,
                "color": row.color,
            },
        )
        entry["vertices"].extend([[row.x, row.y], [row.x2, row.y2]])
        if row.width is not None:
            entry["width"] = row.width
        if row.color is not None:
            entry["color"] = row.color
    return grouped


def _ensure_mapping(value: Any, default: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        return copy.deepcopy(default)
    return copy.deepcopy(value)


def _to_plain(value: Any) -> Any:
    if isinstance(value, util.AttrDict):
        return {str(key): _to_plain(item) for key, item in value.items()}
    if isinstance(value, dict):
        return {str(key): _to_plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_to_plain(item) for item in value)
    if isinstance(value, list):
        return [_to_plain(item) for item in value]
    if hasattr(value, "__array__"):
        try:
            return [_to_plain(item) for item in list(value)]
        except Exception:
            return value
    return value


def _is_mapping(value: Any) -> bool:
    return isinstance(value, (dict, util.AttrDict)) or hasattr(value, "items")


def _get(value: Any, key: str, default: Any = None) -> Any:
    if value is None:
        return default
    if _is_mapping(value):
        try:
            return value.get(key, default)
        except Exception:
            return default
    return getattr(value, key, default)


def _pair(value: Any, *, default: tuple[Any, Any]) -> tuple[Any, Any]:
    if value is None:
        return default
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return (_float_or_none(value[0]), _float_or_none(value[1]))
    return default


def _float_or_none(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _int_or_none(value: Any, default: int | None = None) -> int | None:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _bool_or_none(value: Any, default: bool | None = None) -> bool | None:
    if value is None:
        return default
    return bool(value)


def _str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
