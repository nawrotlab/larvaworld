from __future__ import annotations

import math
from numbers import Number
from pathlib import Path
from typing import Any

from larvaworld.lib import reg, util

__all__ = [
    "_builder_obstacle_border_vertices",
    "_coerce_like",
    "_coerce_xy_sequences",
    "_merge_collection_like",
    "_merge_object_like",
    "_normalize_scalar",
    "_translate_builder_environment_payload",
    "apply_environment_payload",
    "resolve_base_experiment_parameters",
]


def _normalize_scalar(value: Any) -> Any:
    nested_conf = getattr(type(value), "nestedConf", None)
    if nested_conf is not None:
        return _normalize_scalar(value.nestedConf)
    if isinstance(value, tuple):
        return tuple(_normalize_scalar(item) for item in value)
    if isinstance(value, list):
        return [_normalize_scalar(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _normalize_scalar(item) for key, item in value.items()}
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def _coerce_like(template: Any, value: Any) -> Any:
    if isinstance(template, tuple) and isinstance(value, list):
        return tuple(
            _coerce_like(item_template, item_value)
            for item_template, item_value in zip(template, value)
        )
    if isinstance(template, list) and isinstance(value, list) and len(template) > 0:
        item_template = template[0]
        return [_coerce_like(item_template, item) for item in value]
    if isinstance(template, dict) and isinstance(value, dict):
        coerced = {}
        for key, item in value.items():
            key_str = str(key)
            coerced[key_str] = (
                _coerce_like(template[key_str], item) if key_str in template else item
            )
        return coerced
    return value


def _coerce_xy_sequences(value: Any) -> Any:
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


def _builder_obstacle_border_vertices(
    pos: tuple[float, float] | list[float],
    radius: float,
    *,
    segments: int = 18,
) -> list[tuple[float, float]]:
    x0, y0 = float(pos[0]), float(pos[1])
    n_segments = max(6, int(segments))
    points = [
        (
            x0 + math.cos(2 * math.pi * idx / n_segments) * float(radius),
            y0 + math.sin(2 * math.pi * idx / n_segments) * float(radius),
        )
        for idx in range(n_segments)
    ]
    vertices: list[tuple[float, float]] = []
    for idx, point in enumerate(points):
        point_next = points[(idx + 1) % len(points)]
        vertices.extend(
            [
                (round(point[0], 6), round(point[1], 6)),
                (round(point_next[0], 6), round(point_next[1], 6)),
            ]
        )
    return vertices


def _translate_builder_environment_payload(
    environment_payload: util.AttrDict,
) -> util.AttrDict:
    payload = _coerce_xy_sequences(environment_payload.get_copy())
    obstacles = payload.get("obstacles", {})
    if not isinstance(obstacles, dict) or not obstacles:
        return payload

    border_list = payload.get("border_list", {})
    if not isinstance(border_list, dict):
        border_list = {}
    else:
        border_list = dict(border_list)

    for object_id, entry in obstacles.items():
        if not isinstance(entry, dict):
            continue
        pos = entry.get("pos")
        radius = entry.get("radius")
        if not isinstance(pos, (list, tuple)) or len(pos) != 2 or radius in {None, ""}:
            continue
        border_id = f"Obstacle_{object_id}"
        border_list[border_id] = {
            "vertices": _builder_obstacle_border_vertices(pos, float(radius)),
            "width": max(float(radius) * 0.18, 0.001),
            "color": entry.get("color") or "#ff6b35",
        }

    payload["border_list"] = util.AttrDict(border_list)
    return payload


def _merge_object_like(template: Any, payload: Any) -> Any:
    if isinstance(template, dict) and isinstance(payload, dict):
        if not template:
            return util.AttrDict(_normalize_scalar(payload))
        merged = util.AttrDict(template).get_copy()
        for key, value in payload.items():
            key_str = str(key)
            if key_str not in merged:
                continue
            merged[key_str] = _merge_object_like(merged[key_str], value)
        return merged
    return _coerce_like(template, _normalize_scalar(payload))


def _merge_collection_like(template: Any, payload: Any) -> util.AttrDict:
    if not isinstance(payload, dict):
        return util.AttrDict()
    if not isinstance(template, dict) or not template:
        return util.AttrDict(_normalize_scalar(payload))

    prototype = next(iter(template.values()))
    merged = util.AttrDict()
    for key, value in payload.items():
        key_str = str(key)
        item_template = template.get(key_str, prototype)
        merged[key_str] = _merge_object_like(item_template, value)
    return merged


def apply_environment_payload(
    env_params: util.AttrDict,
    environment_payload: util.AttrDict,
) -> util.AttrDict:
    merged = env_params.get_copy()
    payload = _translate_builder_environment_payload(environment_payload)

    if isinstance(payload.get("arena"), dict):
        merged["arena"] = _merge_object_like(merged.arena, payload["arena"])

    if isinstance(payload.get("food_params"), dict):
        food_params = util.AttrDict(merged.food_params).get_copy()
        payload_food = util.AttrDict(payload["food_params"])
        if isinstance(payload_food.get("source_units"), dict):
            food_params["source_units"] = _merge_collection_like(
                food_params.get("source_units", {}),
                payload_food["source_units"],
            )
        if isinstance(payload_food.get("source_groups"), dict):
            food_params["source_groups"] = _merge_collection_like(
                food_params.get("source_groups", {}),
                payload_food["source_groups"],
            )
        if "food_grid" in payload_food:
            incoming_food_grid = payload_food["food_grid"]
            if isinstance(incoming_food_grid, dict):
                food_params["food_grid"] = _merge_object_like(
                    food_params.get("food_grid", {}),
                    incoming_food_grid,
                )
            else:
                food_params["food_grid"] = incoming_food_grid
        merged["food_params"] = food_params

    if isinstance(payload.get("border_list"), dict):
        merged["border_list"] = _merge_collection_like(
            merged.get("border_list", {}),
            payload["border_list"],
        )

    for scape_key in ("odorscape", "windscape", "thermoscape"):
        if scape_key not in payload:
            continue
        incoming = payload[scape_key]
        if incoming is None:
            merged[scape_key] = None
        elif isinstance(incoming, dict):
            current = merged.get(scape_key)
            merged[scape_key] = (
                _merge_object_like(current, incoming)
                if isinstance(current, dict)
                else util.AttrDict(_normalize_scalar(incoming))
            )
        else:
            merged[scape_key] = incoming

    return merged


def resolve_base_experiment_parameters(
    experiment_id: str,
    environment_payload: util.AttrDict | None = None,
) -> util.AttrDict:
    parameters = reg.conf.Exp.getID(experiment_id).get_copy()
    parameters["duration"] = float(parameters.get("duration", 5.0))
    if environment_payload is not None:
        env_params = util.AttrDict(parameters.env_params).get_copy()
        parameters["env_params"] = apply_environment_payload(
            env_params, environment_payload
        )
    return parameters
