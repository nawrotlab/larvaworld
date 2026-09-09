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
    """Normalize a source group's distribution shape.

    Args:
        shape: The stored shape name.

    Returns:
        The canonical shape name.
    """
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
    """Normalize a preset name into a safe file name.

    Args:
        name: The preset name as entered.

    Returns:
        The file name.
    """
    cleaned = _REGEX_PRESET_NAME.sub("_", str(name).strip()).strip("._-")
    if not cleaned:
        cleaned = default
    if not cleaned.endswith(".json"):
        cleaned = f"{cleaned}.json"
    return cleaned


def translate_environment_payload(payload: Any) -> util.AttrDict:
    """Convert a stored environment into the builder's payload form.

    Args:
        payload: The stored environment configuration.

    Returns:
        The builder payload.
    """
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
    """Build the editable object rows from a payload.

    Args:
        payload: The builder payload.

    Returns:
        One row per placed object.
    """
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
    """Write a set of object rows back into a payload.

    Args:
        base_payload: The payload to update.
        rows: The edited object rows.

    Returns:
        The updated payload.
    """
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
    """Add one food source to the payload.

    Args:
        base_payload: The payload to update.
        row: The source to add.

    Returns:
        The updated payload.
    """
    rows = list(object_rows_from_payload(base_payload))
    rows.append(row)
    return payload_with_object_rows(base_payload, rows)


def update_source_unit(base_payload: Any, row: EnvBuilderObjectRow) -> dict[str, Any]:
    """Update one food source in the payload.

    Args:
        base_payload: The payload to update.
        row: The edited source.

    Returns:
        The updated payload.
    """
    return _replace_row(base_payload, row, SOURCE_UNIT)


def delete_source_unit(base_payload: Any, object_id: str) -> dict[str, Any]:
    """Remove one food source from the payload.

    Args:
        base_payload: The payload to update.
        object_id: The source to remove.

    Returns:
        The updated payload.
    """
    return _delete_row(base_payload, object_id, SOURCE_UNIT)


def add_source_group(base_payload: Any, row: EnvBuilderObjectRow) -> dict[str, Any]:
    """Add one source group to the payload.

    Args:
        base_payload: The payload to update.
        row: The group to add.

    Returns:
        The updated payload.
    """
    rows = list(object_rows_from_payload(base_payload))
    rows.append(row)
    return payload_with_object_rows(base_payload, rows)


def update_source_group(base_payload: Any, row: EnvBuilderObjectRow) -> dict[str, Any]:
    """Update one source group in the payload.

    Args:
        base_payload: The payload to update.
        row: The edited group.

    Returns:
        The updated payload.
    """
    return _replace_row(base_payload, row, SOURCE_GROUP)


def delete_source_group(base_payload: Any, object_id: str) -> dict[str, Any]:
    """Remove one source group from the payload.

    Args:
        base_payload: The payload to update.
        object_id: The group to remove.

    Returns:
        The updated payload.
    """
    return _delete_row(base_payload, object_id, SOURCE_GROUP)


def add_border_segment(base_payload: Any, row: EnvBuilderObjectRow) -> dict[str, Any]:
    """Add one border to the payload.

    Args:
        base_payload: The payload to update.
        row: The border to add.

    Returns:
        The updated payload.
    """
    rows = list(object_rows_from_payload(base_payload))
    rows.append(row)
    return payload_with_object_rows(base_payload, rows)


def update_border_segment(
    base_payload: Any, row: EnvBuilderObjectRow
) -> dict[str, Any]:
    """Update one border in the payload.

    Args:
        base_payload: The payload to update.
        row: The edited border.

    Returns:
        The updated payload.
    """
    return _replace_row(base_payload, row, BORDER_SEGMENT)


def delete_border_segment(base_payload: Any, object_id: str) -> dict[str, Any]:
    """Remove one border from the payload.

    Args:
        base_payload: The payload to update.
        object_id: The border to remove.

    Returns:
        The updated payload.
    """
    return _delete_row(base_payload, object_id, BORDER_SEGMENT)


def build_canvas_state(payload: Any) -> EnvironmentCanvasState:
    """Build the canvas state a payload describes.

    Args:
        payload: The builder payload.

    Returns:
        The state the canvas renders.
    """
    return env_params_to_canvas_state(translate_environment_payload(payload))


def _replace_row(
    base_payload: Any, row: EnvBuilderObjectRow, object_type: str
) -> dict[str, Any]:
    """Replace one object row of a given type in the payload.

    Args:
        base_payload: The payload to update.
        row: The replacement row.
        object_type: The row's object type.

    Returns:
        The updated payload.
    """
    rows = [
        item
        for item in object_rows_from_payload(base_payload)
        if not (item.object_type == object_type and item.object_id == row.object_id)
    ]
    rows.append(row)
    return payload_with_object_rows(base_payload, rows)


def _delete_row(base_payload: Any, object_id: str, object_type: str) -> dict[str, Any]:
    """Remove one object row of a given type from the payload.

    Args:
        base_payload: The payload to update.
        object_id: The row to remove.
        object_type: The row's object type.

    Returns:
        The updated payload.
    """
    rows = [
        item
        for item in object_rows_from_payload(base_payload)
        if not (item.object_type == object_type and item.object_id == object_id)
    ]
    return payload_with_object_rows(base_payload, rows)


def _source_unit_row(object_id: str, source: Any) -> EnvBuilderObjectRow:
    """Build the editable row for one food source.

    Args:
        object_id: The source's identifier.
        source: The stored source.

    Returns:
        The row.
    """
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
    """Build the editable row for one source group.

    Args:
        object_id: The group's identifier.
        group: The stored group.

    Returns:
        The row.
    """
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
    """Build the editable rows for one border.

    Args:
        object_id: The border's identifier.
        border: The stored border.

    Returns:
        One row per border segment.
    """
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
    """Convert an edited row back into a source.

    Args:
        row: The edited row.

    Returns:
        The source configuration.
    """
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
    """Convert an edited row back into a source group.

    Args:
        row: The edited row.

    Returns:
        The group configuration.
    """
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
    """Convert edited rows back into border configurations.

    Args:
        rows: The edited rows.

    Returns:
        The border configurations.
    """
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
    """Return a value as a mapping, substituting a default.

    Args:
        value: The value to coerce.
        default: Returned when it is not a mapping.

    Returns:
        The mapping.
    """
    if not isinstance(value, dict):
        return copy.deepcopy(default)
    return copy.deepcopy(value)


def _to_plain(value: Any) -> Any:
    """Convert a nested config object into plain dicts and lists.

    Args:
        value: The value to convert.

    Returns:
        The plain equivalent.
    """
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
    """Report whether a value behaves as a mapping.

    Args:
        value: The value to test.

    Returns:
        True when it can be read by key.
    """
    return isinstance(value, (dict, util.AttrDict)) or hasattr(value, "items")


def _get(value: Any, key: str, default: Any = None) -> Any:
    """Read a key from a value that may not be a mapping.

    Args:
        value: The value to read.
        key: The key to look up.
        default: Returned when the key or the mapping is absent.

    Returns:
        The value found, or the default.
    """
    if value is None:
        return default
    if _is_mapping(value):
        try:
            return value.get(key, default)
        except Exception:
            return default
    return getattr(value, key, default)


def _pair(value: Any, *, default: tuple[Any, Any]) -> tuple[Any, Any]:
    """Coerce a value into a coordinate pair.

    Args:
        value: The value to coerce.

    Returns:
        The pair, or None when it is not one.
    """
    if value is None:
        return default
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return (_float_or_none(value[0]), _float_or_none(value[1]))
    return default


def _float_or_none(value: Any, default: float | None = None) -> float | None:
    """Coerce a value into a float.

    Args:
        value: The value to coerce.
        default: Returned when it cannot be coerced.

    Returns:
        The float, or the default.
    """
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _int_or_none(value: Any, default: int | None = None) -> int | None:
    """Coerce a value into an integer.

    Args:
        value: The value to coerce.
        default: Returned when it cannot be coerced.

    Returns:
        The integer, or the default.
    """
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _bool_or_none(value: Any, default: bool | None = None) -> bool | None:
    """Coerce a value into a boolean.

    Args:
        value: The value to coerce.
        default: Returned when it cannot be coerced.

    Returns:
        The boolean, or the default.
    """
    if value is None:
        return default
    return bool(value)


def _str_or_none(value: Any) -> str | None:
    """Coerce a value into a non-empty string.

    Args:
        value: The value to coerce.

    Returns:
        The string, or None when it is empty or absent.
    """
    if value is None:
        return None
    text = str(value).strip()
    return text or None
