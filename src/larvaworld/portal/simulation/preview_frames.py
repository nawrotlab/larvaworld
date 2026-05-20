from __future__ import annotations

from math import isfinite, nan
from typing import Any

from larvaworld.portal.canvas_widgets.environment_models import LarvaPreviewFrame

_INVALID_XY: tuple[float, float] = (nan, nan)


def _copy_xy(value: Any) -> tuple[float, float] | None:
    try:
        x_raw = value[0]
        y_raw = value[1]
    except (TypeError, IndexError, KeyError):
        return None
    try:
        x = float(x_raw)
        y = float(y_raw)
    except (TypeError, ValueError):
        return None
    if not (isfinite(x) and isfinite(y)):
        return None
    return (x, y)


def _copy_path(value: Any) -> tuple[tuple[float, float], ...]:
    try:
        iterator = iter(value)
    except TypeError:
        return ()
    points: list[tuple[float, float]] = []
    for candidate in iterator:
        xy = _copy_xy(candidate)
        if xy is not None:
            points.append(xy)
    return tuple(points)


def _copy_positions(values: Any, count: int) -> tuple[tuple[float, float], ...]:
    copied: list[tuple[float, float]] = []
    for index in range(count):
        point = None
        try:
            point = _copy_xy(values[index])
        except (TypeError, IndexError, KeyError):
            point = None
        copied.append(point if point is not None else _INVALID_XY)
    return tuple(copied)


def _copy_segment_polygons(agent: Any) -> tuple[tuple[tuple[float, float], ...], ...]:
    polygons: list[tuple[tuple[float, float], ...]] = []
    for seg in getattr(agent, "segs", ()):
        polygon = _copy_path(getattr(seg, "vertices", ()))
        if len(polygon) >= 3:
            polygons.append(polygon)
    return tuple(polygons)


def _is_explicit_contour_agent(agent: Any) -> bool:
    class_names = {cls.__name__ for cls in type(agent).mro()}
    return bool({"LarvaContoured", "LarvaReplayContoured"} & class_names)


def _copy_explicit_body_contour(agent: Any) -> tuple[tuple[float, float], ...]:
    """Copy explicit runtime contour geometry from an agent.

    This intentionally avoids inferred or derived outlines. Generic vertices are
    only accepted for known contoured/replay-contoured agents.
    """
    try:
        contour = getattr(agent, "contour_xy")
    except (TypeError, AttributeError, IndexError, KeyError, ValueError):
        contour = None

    copied = _copy_path(contour) if contour is not None else ()
    if len(copied) >= 3:
        return copied

    if _is_explicit_contour_agent(agent):
        try:
            vertices = getattr(agent, "vertices")
        except (TypeError, AttributeError, IndexError, KeyError, ValueError):
            vertices = None

        copied = _copy_path(vertices) if vertices is not None else ()
        if len(copied) >= 3:
            return copied

    return ()


def capture_larva_frame(
    launcher: Any,
    *,
    tick: int | None = None,
    trail_length: int = 30,
    include_heads: bool = True,
    include_midlines: bool = True,
    include_trails: bool = True,
) -> LarvaPreviewFrame:
    agents = launcher.agents
    n_agents = len(agents)
    centroids = _copy_positions(agents.get_position(), n_agents)

    heads: tuple[tuple[float, float], ...] = ()
    if include_heads:
        raw_heads = agents.head.front_end
        head_rows: list[tuple[float, float]] = []
        for index in range(n_agents):
            point = None
            try:
                point = _copy_xy(raw_heads[index])
            except (TypeError, IndexError, KeyError):
                point = None
            head_rows.append(point if point is not None else _INVALID_XY)
        heads = tuple(head_rows)

    midlines: tuple[tuple[tuple[float, float], ...], ...] = ()
    if include_midlines:
        midline_rows: list[tuple[tuple[float, float], ...]] = []
        for index in range(n_agents):
            path: tuple[tuple[float, float], ...] = ()
            try:
                path = _copy_path(agents[index].midline_xy)
            except (TypeError, IndexError, KeyError, AttributeError):
                path = ()
            midline_rows.append(path)
        midlines = tuple(midline_rows)

    trails: tuple[tuple[tuple[float, float], ...], ...] = ()
    if include_trails:
        trail_rows: list[tuple[tuple[float, float], ...]] = []
        for index in range(n_agents):
            path: tuple[tuple[float, float], ...] = ()
            try:
                trajectory = agents[index].trajectory
                path = _copy_path(trajectory[-trail_length:])
            except (TypeError, IndexError, KeyError, AttributeError):
                path = ()
            trail_rows.append(path)
        trails = tuple(trail_rows)

    segment_rows: list[tuple[tuple[tuple[float, float], ...], ...]] = []
    for index in range(n_agents):
        try:
            segment_rows.append(_copy_segment_polygons(agents[index]))
        except (TypeError, IndexError, KeyError, AttributeError):
            segment_rows.append(())

    body_contour_rows: list[tuple[tuple[float, float], ...]] = []
    for index in range(n_agents):
        try:
            body_contour_rows.append(_copy_explicit_body_contour(agents[index]))
        except (TypeError, IndexError, KeyError, AttributeError, ValueError):
            body_contour_rows.append(())

    colors = tuple(
        "" if getattr(agent, "color", None) is None else str(agent.color)
        for agent in agents
    )

    resolved_tick = launcher.t if tick is None else tick
    return LarvaPreviewFrame(
        tick=int(resolved_tick),
        centroids=centroids,
        heads=heads,
        midlines=midlines,
        trails=trails,
        segment_polygons=tuple(segment_rows),
        body_contours=tuple(body_contour_rows),
        colors=colors,
    )


def generate_preview_frames(
    launcher: Any,
    *,
    preview_steps: int,
    trail_length: int = 30,
    include_heads: bool = True,
    include_midlines: bool = True,
    include_trails: bool = True,
) -> list[LarvaPreviewFrame]:
    if preview_steps <= 0:
        return []

    frames: list[LarvaPreviewFrame] = []
    for index in range(preview_steps):
        frames.append(
            capture_larva_frame(
                launcher,
                trail_length=trail_length,
                include_heads=include_heads,
                include_midlines=include_midlines,
                include_trails=include_trails,
            )
        )
        if index < preview_steps - 1:
            launcher.sim_step()
    return frames
