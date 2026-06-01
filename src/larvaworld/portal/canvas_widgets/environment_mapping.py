from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .environment_models import CanvasArena, CanvasObject, EnvironmentCanvasState


def env_params_to_canvas_state(
    env_params: Any,
    *,
    larva_groups: Mapping[str, Any] | None = None,
    show_group_shapes: bool = True,
) -> EnvironmentCanvasState:
    """Map resolved environment parameters into normalized canvas state."""

    arena_conf = _get(env_params, "arena", {})
    arena = CanvasArena(
        geometry=str(_get(arena_conf, "geometry", "rectangular") or "rectangular"),
        dims=_pair(_get(arena_conf, "dims", (0.2, 0.2)), default=(0.2, 0.2)),
        torus=bool(_get(arena_conf, "torus", False)),
    )

    objects: list[CanvasObject] = []
    food_params = _get(env_params, "food_params", {})
    source_units = _get(food_params, "source_units", {}) or {}
    source_groups = _get(food_params, "source_groups", {}) or {}
    border_list = _get(env_params, "border_list", {}) or {}

    if _is_mapping(source_units):
        for source_id, source in source_units.items():
            obj = _source_unit_to_canvas_object(str(source_id), source)
            if obj is not None:
                objects.append(obj)

    if _is_mapping(source_groups):
        for group_id, group in source_groups.items():
            obj = _source_group_to_canvas_object(
                str(group_id), group, show_shape=show_group_shapes
            )
            if obj is not None:
                objects.append(obj)

    if _is_mapping(border_list):
        for border_id, border in border_list.items():
            objects.extend(_border_to_canvas_objects(str(border_id), border))

    if _is_mapping(larva_groups):
        for group_id, group in larva_groups.items():
            obj = _larva_group_to_canvas_object(
                str(group_id), group, show_shape=show_group_shapes
            )
            if obj is not None:
                objects.append(obj)

    return EnvironmentCanvasState(
        arena=arena,
        objects=tuple(objects),
        food_grid=_optional_mapping(_get(food_params, "food_grid", None)),
        odorscape=_optional_mapping(_get(env_params, "odorscape", None)),
        windscape=_optional_mapping(_get(env_params, "windscape", None)),
        thermoscape=_optional_mapping(_get(env_params, "thermoscape", None)),
    )


def _source_unit_to_canvas_object(object_id: str, source: Any) -> CanvasObject | None:
    pos = _pair(_get(source, "pos", None), default=(None, None))
    x, y = pos
    if x is None or y is None:
        return None
    odor = _get(source, "odor", {}) or {}
    return CanvasObject(
        object_id=object_id,
        object_type="source_unit",
        x=x,
        y=y,
        radius=_float_or_none(_get(source, "radius", 0.003)),
        color=_str_or_none(_get(source, "color", None)),
        amount=_float_or_none(_get(source, "amount", None)),
        odor_id=_str_or_none(_get(odor, "id", None)),
        odor_intensity=_float_or_none(_get(odor, "intensity", None)),
        odor_spread=_float_or_none(_get(odor, "spread", None)),
    )


def _source_group_to_canvas_object(
    object_id: str, group: Any, *, show_shape: bool
) -> CanvasObject | None:
    distribution = _get(group, "distribution", {}) or {}
    pos = _pair(
        _get(distribution, "loc", _get(group, "pos", None)), default=(None, None)
    )
    x, y = pos
    if x is None or y is None:
        return None
    scale = _pair(_get(distribution, "scale", None), default=(0.012, 0.012))
    odor = _get(group, "odor", {}) or {}
    return CanvasObject(
        object_id=object_id,
        object_type="source_group",
        x=x,
        y=y,
        radius=_float_or_none(_get(group, "radius", 0.003)),
        color=_str_or_none(_get(group, "color", None)),
        amount=_float_or_none(_get(group, "amount", None)),
        odor_id=_str_or_none(_get(odor, "id", None)),
        odor_intensity=_float_or_none(_get(odor, "intensity", None)),
        odor_spread=_float_or_none(_get(odor, "spread", None)),
        distribution_mode=_str_or_none(_get(distribution, "mode", "uniform")),
        distribution_shape=_str_or_none(_get(distribution, "shape", "circle")),
        distribution_n=_int_or_none(_get(distribution, "N", None)),
        distribution_scale_x=scale[0],
        distribution_scale_y=scale[1],
        distribution_show_shape=(
            bool(_get(group, "distribution_show_shape", True)) if show_shape else False
        ),
    )


def _border_to_canvas_objects(object_id: str, border: Any) -> list[CanvasObject]:
    vertices = _get(border, "vertices", None)
    segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    if vertices is not None:
        segments.extend(_paired_segments(vertices))
    if not segments:
        border_xy = _get(border, "border_xy", None)
        if isinstance(border_xy, (list, tuple)):
            for path in border_xy:
                segments.extend(_paired_segments(path))
    width = _float_or_none(_get(border, "width", 0.001))
    color = _str_or_none(_get(border, "color", "#333333"))
    objects: list[CanvasObject] = []
    for index, ((x0, y0), (x1, y1)) in enumerate(segments):
        objects.append(
            CanvasObject(
                object_id=object_id if len(segments) == 1 else f"{object_id}:{index}",
                object_type="border_segment",
                x=x0,
                y=y0,
                x2=x1,
                y2=y1,
                width=width,
                color=color,
            )
        )
    return objects


def _larva_group_to_canvas_object(
    object_id: str, group: Any, *, show_shape: bool
) -> CanvasObject | None:
    distribution = _get(group, "distribution", {}) or {}
    pos = _pair(_get(distribution, "loc", None), default=(None, None))
    x, y = pos
    if x is None or y is None:
        return None
    scale = _pair(_get(distribution, "scale", None), default=(0.012, 0.012))
    return CanvasObject(
        object_id=object_id,
        object_type="larva_group",
        x=x,
        y=y,
        color=_str_or_none(_get(group, "color", None)),
        distribution_mode=_str_or_none(_get(distribution, "mode", "uniform")),
        distribution_shape=_str_or_none(_get(distribution, "shape", "circle")),
        distribution_n=_int_or_none(_get(distribution, "N", None)),
        distribution_scale_x=scale[0],
        distribution_scale_y=scale[1],
        distribution_show_shape=show_shape,
    )


def _optional_mapping(value: Any) -> dict[str, Any] | None:
    if value in (None, {}, "empty_dict"):
        return None
    if not _is_mapping(value):
        return None
    plain = _to_plain(value)
    return plain if isinstance(plain, dict) and plain else None


def _to_plain(value: Any) -> Any:
    if _is_mapping(value):
        return {str(key): _to_plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_to_plain(item) for item in value)
    if isinstance(value, list):
        return [_to_plain(item) for item in value]
    if hasattr(value, "__array__"):
        try:
            return [_to_plain(item) for item in list(value)]
        except Exception:
            return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return value


def _is_mapping(value: Any) -> bool:
    return isinstance(value, Mapping) or hasattr(value, "items")


def _get(value: Any, key: str, default: Any = None) -> Any:
    if value is None:
        return default
    if _is_mapping(value):
        try:
            return value.get(key, default)
        except Exception:
            pass
    return getattr(value, key, default)


def _pair(value: Any, *, default: tuple[Any, Any]) -> tuple[Any, Any]:
    try:
        if value is None:
            return default
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            return (_float_or_none(value[0]), _float_or_none(value[1]))
        if hasattr(value, "__array__"):
            items = list(value)
            if len(items) >= 2:
                return (_float_or_none(items[0]), _float_or_none(items[1]))
    except Exception:
        return default
    return default


def _paired_segments(
    value: Any,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    try:
        points = list(value)
    except Exception:
        return []
    if len(points) < 2:
        return []
    segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for index in range(0, len(points) - 1, 2):
        first = _pair(points[index], default=(None, None))
        second = _pair(points[index + 1], default=(None, None))
        if None in first or None in second:
            continue
        segments.append(((first[0], first[1]), (second[0], second[1])))
    return segments


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None
