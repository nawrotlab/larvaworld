from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from larvaworld.lib import reg, util
from larvaworld.lib.process.dataset import LarvaDataset
from larvaworld.portal.canvas_widgets.environment_models import (
    CanvasArena,
    CanvasRingOverlay,
    EnvironmentCanvasState,
    LarvaPreviewFrame,
)
from larvaworld.portal.datasets.replay_models import (
    PreparedReplayMember,
    PreparedReplaySource,
    ReplayCoordinateOrigin,
    ReplaySource,
    ReplaySourceMember,
)
from larvaworld.portal.datasets.workspace_index import (
    list_workspace_datasets,
    list_workspace_simulation_datasets,
)
from larvaworld.portal.workspace import WorkspaceState


@dataclass(frozen=True)
class ReplayRenderState:
    frame: LarvaPreviewFrame
    rings: tuple[CanvasRingOverlay, ...] = ()


def build_source_catalog(workspace: WorkspaceState | None) -> list[ReplaySource]:
    sources: list[ReplaySource] = []
    if workspace is not None:
        records = list_workspace_datasets(workspace=workspace)
        by_group: dict[tuple[str, str], list] = {}
        for record in records:
            member = ReplaySourceMember(
                token=f"workspace:{record.dataset_dir}",
                label=record.dataset_id,
                source_type="workspace_dataset",
                workspace_record=record,
            )
            sources.append(
                ReplaySource(
                    token=member.token,
                    label=f"Workspace / Imported dataset / {record.dataset_id}",
                    source_type="workspace_dataset",
                    members=(member,),
                )
            )
            if record.lab_id and record.group_id:
                by_group.setdefault((record.lab_id, record.group_id), []).append(record)
        for (lab_id, group_id), group_records in sorted(by_group.items()):
            if len(group_records) < 2:
                continue
            members = tuple(
                ReplaySourceMember(
                    token=f"workspace:{record.dataset_dir}",
                    label=record.dataset_id,
                    source_type="workspace_group",
                    workspace_record=record,
                )
                for record in sorted(group_records, key=lambda r: r.dataset_id)
            )
            sources.append(
                ReplaySource(
                    token=f"workspace_group:{lab_id}:{group_id}",
                    label=f"Workspace / Imported group / {lab_id}:{group_id}",
                    source_type="workspace_group",
                    members=members,
                )
            )
        simulation_records = list_workspace_simulation_datasets(workspace=workspace)
        by_run: dict[str, list] = {}
        for record in simulation_records:
            if not record.run_id:
                continue
            by_run.setdefault(record.run_id, []).append(record)
        for run_id, run_records in sorted(by_run.items()):
            members = tuple(
                ReplaySourceMember(
                    token=(
                        "workspace_simulation:"
                        f"{run_id}:{record.member_id or record.dataset_id}"
                    ),
                    label=record.dataset_id,
                    source_type="workspace_simulation_run",
                    workspace_record=record,
                )
                for record in sorted(
                    run_records, key=lambda r: (r.dataset_id, str(r.dataset_dir))
                )
            )
            sources.append(
                ReplaySource(
                    token=f"workspace_simulation_run:{run_id}",
                    label=f"Workspace / Simulation run / {run_id}",
                    source_type="workspace_simulation_run",
                    members=members,
                )
            )

    for ref_id in sorted(reg.conf.Ref.confIDs):
        member = ReplaySourceMember(
            token=f"registry_ref:{ref_id}",
            label=ref_id,
            source_type="registry_reference",
            registry_ref_id=ref_id,
        )
        sources.append(
            ReplaySource(
                token=member.token,
                label=f"Registry / Reference dataset / {ref_id}",
                source_type="registry_reference",
                members=(member,),
            )
        )

    for group_id in sorted(reg.conf.Ref.RefGroupIDs):
        group = reg.conf.Ref.getRefGroup(group_id)
        members = tuple(
            ReplaySourceMember(
                token=f"registry_ref:{ref_id}",
                label=ref_id,
                source_type="registry_reference_group",
                registry_ref_id=ref_id,
            )
            for ref_id in sorted(group.keys())
        )
        sources.append(
            ReplaySource(
                token=f"registry_group:{group_id}",
                label=f"Registry / Reference group / {group_id}",
                source_type="registry_reference_group",
                members=members,
            )
        )
    return sources


def prepare_replay_source(source: ReplaySource) -> PreparedReplaySource:
    prepared = PreparedReplaySource(source=source)
    for member in source.members:
        loaded_member = _prepare_member(member)
        prepared.members[member.token] = loaded_member
        prepared.nticks = max(prepared.nticks, loaded_member.nticks)
        if loaded_member.dt > 0:
            prepared.dt = min(prepared.dt, loaded_member.dt)
        prepared.arena_dims = loaded_member.arena_dims
    return prepared


def build_environment_state_for_member(
    member: PreparedReplayMember,
    *,
    allow_static_layers: bool,
    show_arena_outline: bool = True,
    coordinate_origin: ReplayCoordinateOrigin | None = None,
) -> EnvironmentCanvasState:
    if not allow_static_layers:
        state_coordinate_origin = coordinate_origin or member.coordinate_origin
        arena = _canvas_arena_from_member(member)
        if arena is None:
            arena = CanvasArena(
                "rectangular",
                member.arena_dims,
                coordinate_origin=state_coordinate_origin,
            )
            show_arena_outline = False
        elif arena.coordinate_origin != state_coordinate_origin:
            arena = CanvasArena(
                arena.geometry,
                arena.dims,
                torus=arena.torus,
                coordinate_origin=state_coordinate_origin,
            )
        return EnvironmentCanvasState(
            arena=arena,
            objects=(),
            show_arena_outline=show_arena_outline,
        )
    if member.env_params is not None:
        env_conf = member.env_params
    elif member.env_conf_id and member.env_conf_id in reg.conf.Env.confIDs:
        env_conf = reg.conf.Env.get(member.env_conf_id)
    else:
        env_conf = reg.gen.Env(
            arena={"dims": member.arena_dims, "geometry": "rectangular"}
        )
    from larvaworld.portal.canvas_widgets.environment_mapping import (
        env_params_to_canvas_state,
    )

    return env_params_to_canvas_state(env_conf)


def member_has_arena_geometry(member: PreparedReplayMember) -> bool:
    return _canvas_arena_from_member(member) is not None


def _canvas_arena_from_member(member: PreparedReplayMember) -> CanvasArena | None:
    env_conf = _member_env_conf(member)
    arena_conf = _get_env_value(env_conf, "arena", None)
    if arena_conf is None:
        return None
    dims = _get_env_value(arena_conf, "dims", None)
    if not (isinstance(dims, (list, tuple)) and len(dims) >= 2):
        return None
    try:
        arena_dims = (float(dims[0]), float(dims[1]))
    except (TypeError, ValueError):
        return None
    geometry = str(_get_env_value(arena_conf, "geometry", "rectangular") or "")
    if not geometry:
        return None
    return CanvasArena(
        geometry,
        arena_dims,
        torus=bool(_get_env_value(arena_conf, "torus", False)),
        coordinate_origin=member.coordinate_origin,
    )


def _member_env_conf(member: PreparedReplayMember) -> Any | None:
    if member.env_params is not None:
        return member.env_params
    if member.env_conf_id and member.env_conf_id in reg.conf.Env.confIDs:
        return reg.conf.Env.get(member.env_conf_id)
    return None


def _get_env_value(value: Any, key: str, default: Any = None) -> Any:
    if value is None:
        return default
    if hasattr(value, "get"):
        try:
            return value.get(key, default)
        except Exception:
            pass
    return getattr(value, key, default)


def build_render_state(
    prepared: PreparedReplaySource,
    *,
    tick: int,
    member_tokens: Iterable[str],
    show_positions: bool,
    show_ids: bool,
    show_tracks: bool,
    trail_length: int,
    transposition: str | None,
    track_point: int,
    agent_indices: tuple[int, ...] | None,
    time_range: tuple[float, float] | None,
    show_dispersal_ring: bool,
    show_heads: bool = True,
    show_midlines: bool = True,
    show_segments: bool = True,
    show_body_contours: bool = False,
) -> ReplayRenderState:
    body_geometry_enabled = bool(show_heads or show_midlines or show_segments)
    contour_geometry_enabled = bool(show_body_contours)
    needs_aligned_history = bool(
        show_tracks or (show_dispersal_ring and transposition == "origin")
    )

    centroids: list[tuple[float, float]] = []
    heads: list[tuple[float, float]] = []
    midlines: list[tuple[tuple[float, float], ...]] = []
    trails: list[tuple[tuple[float, float], ...]] = []
    segment_polygons: list[tuple[tuple[tuple[float, float], ...], ...]] = []
    body_contours: list[tuple[tuple[float, float], ...]] = []
    colors: list[str] = []
    labels: list[str] = []
    dispersal_distances: list[float] = []

    if not show_positions:
        return ReplayRenderState(
            frame=LarvaPreviewFrame(
                tick=max(int(tick), 0),
                centroids=(),
                heads=(),
                midlines=(),
                trails=(),
                segment_polygons=(),
                body_contours=(),
                colors=(),
                labels=(),
            ),
            rings=(),
        )

    for token in member_tokens:
        member = prepared.members.get(token)
        if member is None:
            continue
        xy = select_member_xy(member, track_point=track_point)
        xy = filter_xy_by_agent_indices(
            xy,
            member.agent_ids,
            agent_indices=agent_indices,
        )
        if xy.empty:
            continue
        if time_range is not None and member.dt > 0:
            s0 = int(time_range[0] / member.dt)
            s1 = int(time_range[1] / member.dt)
            xy = xy.loc[(slice(s0, s1), slice(None)), :]
        if xy.empty:
            continue
        offsets_by_agent = _alignment_offsets_by_agent(
            xy,
            transposition=transposition,
            arena_dims=member.arena_dims,
            coordinate_origin=member.coordinate_origin,
        )

        aligned_xy = None
        grouped = None
        if needs_aligned_history:
            aligned_xy = _apply_alignment_offsets(xy, offsets_by_agent=offsets_by_agent)
            grouped = aligned_xy.groupby(level="AgentID")

        try:
            at_tick = xy.xs(tick, level="Step")
        except KeyError:
            continue

        emitted_rows: list[tuple[object, float, float]] = []
        for agent_id, at_row in at_tick.iterrows():
            x_raw = float(at_row["x"])
            y_raw = float(at_row["y"])
            dx, dy = offsets_by_agent.get(agent_id, (0.0, 0.0))
            x = float(x_raw - dx)
            y = float(y_raw - dy)
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            emitted_rows.append((agent_id, x, y))
            centroids.append((x, y))
            colors.append(member.color)
            labels.append(str(agent_id) if show_ids else "")

        if not emitted_rows:
            continue

        emitted_agent_ids = tuple(agent_id for agent_id, _, _ in emitted_rows)
        body_points_by_agent: dict[object, tuple[tuple[float, float], ...]] = {}
        if body_geometry_enabled:
            body_points_by_agent = _geometry_points_for_agents_at_tick(
                member.body_xy_by_point,
                tick=tick,
                emitted_agent_ids=emitted_agent_ids,
                offsets_by_agent=offsets_by_agent,
                min_points=2,
            )

        contour_points_by_agent: dict[object, tuple[tuple[float, float], ...]] = {}
        if contour_geometry_enabled:
            contour_points_by_agent = _geometry_points_for_agents_at_tick(
                member.contour_xy_by_point,
                tick=tick,
                emitted_agent_ids=emitted_agent_ids,
                offsets_by_agent=offsets_by_agent,
                min_points=3,
            )

        for agent_id, x, y in emitted_rows:
            larva_midline = body_points_by_agent.get(agent_id, ())
            if show_midlines:
                midlines.append(larva_midline)
            else:
                midlines.append(())
            if show_heads and larva_midline:
                heads.append(larva_midline[0])
            else:
                heads.append(())
            if show_segments and larva_midline:
                segment_polygons.append(_segment_polygons_from_midline(larva_midline))
            else:
                segment_polygons.append(())
            if contour_geometry_enabled:
                body_contours.append(contour_points_by_agent.get(agent_id, ()))
            else:
                body_contours.append(())
            if show_tracks:
                assert grouped is not None
                history = grouped.get_group(agent_id).loc[:tick]
                history = history.dropna()
                if trail_length > 0 and history.shape[0] > trail_length:
                    history = history.iloc[-trail_length:]
                trails.append(
                    tuple((float(px), float(py)) for px, py in history.values)
                )
            else:
                trails.append(())
            if show_dispersal_ring and transposition == "origin":
                assert grouped is not None
                path = grouped.get_group(agent_id).dropna()
                if path.empty:
                    continue
                x0, y0 = path.iloc[0]["x"], path.iloc[0]["y"]
                if not (np.isfinite(x0) and np.isfinite(y0)):
                    continue
                dx = x - float(x0)
                dy = y - float(y0)
                distance = float((dx * dx + dy * dy) ** 0.5)
                if np.isfinite(distance):
                    dispersal_distances.append(distance)

    rings: tuple[CanvasRingOverlay, ...] = ()
    if show_dispersal_ring and transposition == "origin" and dispersal_distances:
        radius = float(np.mean(np.asarray(dispersal_distances, dtype=float)))
        if np.isfinite(radius) and radius > 0:
            rings = (CanvasRingOverlay(x=0.0, y=0.0, radius=radius, color="#1f4e79"),)

    frame = LarvaPreviewFrame(
        tick=max(int(tick), 0),
        centroids=tuple(centroids),
        heads=tuple(heads),
        midlines=tuple(midlines),
        trails=tuple(trails),
        segment_polygons=tuple(segment_polygons),
        body_contours=tuple(body_contours),
        colors=tuple(colors),
        labels=tuple(labels),
    )
    return ReplayRenderState(frame=frame, rings=rings)


def _prepare_member(member: ReplaySourceMember) -> PreparedReplayMember:
    if member.workspace_record is not None:
        return _prepare_workspace_member(member)
    dataset = _load_dataset(member)
    dataset.load(step=True)
    xy = dataset.s[_resolve_xy_columns(dataset, dataset.s, track_point=-1)].copy()
    xy.columns = ["x", "y"]
    xy_by_track_point = _build_registry_xy_by_track_point(dataset, dataset.s)
    native_track_point_by_ui_track_point = _build_registry_native_track_point_mapping(
        dataset, dataset.s
    )
    native_default_track_point = _choose_native_default_track_point(
        step=dataset.s,
        native_track_point_by_ui_track_point=native_track_point_by_ui_track_point,
        configured_point_idx=int(getattr(dataset.c, "point_idx", -1) or -1),
        traj_xy=list(dataset.c.traj_xy),
        centroid_xy=list(dataset.c.centroid_xy),
        point_name_for_native_index=dataset.c.get_track_point,
    )
    body_xy_by_point = _build_body_xy_by_point(dataset.s, int(dataset.c.Npoints or 0))
    contour_xy_by_point = _build_contour_xy_by_point(
        dataset.s, int(dataset.c.Ncontour or 0)
    )
    arena_dims = _resolve_arena_dims(dataset)
    dt = float(dataset.c.dt or 0.1)
    nticks = int(dataset.c.Nticks or (int(xy.index.unique("Step").max()) + 1))
    color = str(dataset.c.color or "#2f4858")
    env_conf_id = None
    agent_ids = tuple(dataset.c.agent_ids or ())
    if not agent_ids:
        agent_ids = _agent_ids_from_step_index(xy)
    coordinate_origin = _infer_coordinate_origin(xy, arena_dims)
    return PreparedReplayMember(
        token=member.token,
        label=member.label,
        color=color,
        xy_default=xy,
        arena_dims=arena_dims,
        dt=dt,
        nticks=nticks,
        env_conf_id=env_conf_id,
        env_params=getattr(dataset.c, "env_params", None),
        coordinate_origin=coordinate_origin,
        xy_by_track_point=xy_by_track_point,
        native_default_track_point=native_default_track_point,
        native_track_point_by_ui_track_point=native_track_point_by_ui_track_point,
        native_replay_missing_columns=_native_replay_missing_columns(dataset.s),
        body_xy_by_point=body_xy_by_point,
        contour_xy_by_point=contour_xy_by_point,
        agent_ids=agent_ids,
    )


def _prepare_workspace_member(member: ReplaySourceMember) -> PreparedReplayMember:
    assert member.workspace_record is not None
    record = member.workspace_record
    conf = util.load_dict(str(record.conf_path))
    step = pd.read_hdf(record.h5_path, "step")
    spatial_step = _build_workspace_spatial_step(
        step,
        h5_path=record.h5_path,
        dataset_id=record.dataset_id,
    )
    xy_cols = _resolve_workspace_xy_columns(
        spatial_step,
        conf,
        dataset_id=record.dataset_id,
    )
    xy = spatial_step[xy_cols].copy()
    xy.columns = ["x", "y"]
    xy_by_track_point = _build_workspace_xy_by_track_point(spatial_step, conf)
    native_track_point_by_ui_track_point = _build_workspace_native_track_point_mapping(
        spatial_step, conf
    )
    configured_point_idx = int(conf.get("point_idx") or -1)
    if int(conf.get("Npoints") or 0) <= 0:
        configured_point_idx = -1
    native_default_track_point = _choose_native_default_track_point(
        step=spatial_step,
        native_track_point_by_ui_track_point=native_track_point_by_ui_track_point,
        configured_point_idx=configured_point_idx,
        traj_xy=list(util.nam.xy("")),
        centroid_xy=list(util.nam.xy("centroid")),
        point_name_for_native_index=lambda idx: _workspace_point_name(conf, idx),
    )
    body_xy_by_point = _build_body_xy_by_point(
        spatial_step, int(conf.get("Npoints") or 0)
    )
    contour_xy_by_point = _build_contour_xy_by_point(
        spatial_step, int(conf.get("Ncontour") or 0)
    )
    conf_agent_ids = conf.get("agent_ids") if isinstance(conf, dict) else None
    if isinstance(conf_agent_ids, list) and conf_agent_ids:
        agent_ids = tuple(conf_agent_ids)
    else:
        agent_ids = _agent_ids_from_step_index(step)
    dt = float(conf.get("dt") or (1.0 / float(conf.get("fr") or 10.0)))
    nticks = int(conf.get("Nticks") or (int(xy.index.unique("Step").max()) + 1))
    env_conf = conf.get("env_params") if isinstance(conf, dict) else None
    arena_dims = (0.2, 0.2)
    if isinstance(env_conf, dict):
        arena = env_conf.get("arena")
        if isinstance(arena, dict):
            dims = arena.get("dims")
            if isinstance(dims, (list, tuple)) and len(dims) >= 2:
                arena_dims = (float(dims[0]), float(dims[1]))
    coordinate_origin = (
        "centered"
        if getattr(record, "origin", None) == "simulation_run"
        else _infer_coordinate_origin(xy, arena_dims)
    )
    return PreparedReplayMember(
        token=member.token,
        label=member.label,
        color=str(conf.get("color") or "#2f4858"),
        xy_default=xy,
        arena_dims=arena_dims,
        dt=dt,
        nticks=nticks,
        env_conf_id=str(env_conf.get("id")) if isinstance(env_conf, dict) else None,
        env_params=_plain_mapping(env_conf),
        coordinate_origin=coordinate_origin,
        xy_by_track_point=xy_by_track_point,
        native_default_track_point=native_default_track_point,
        native_track_point_by_ui_track_point=native_track_point_by_ui_track_point,
        native_replay_missing_columns=_native_replay_missing_columns(step),
        body_xy_by_point=body_xy_by_point,
        contour_xy_by_point=contour_xy_by_point,
        agent_ids=agent_ids,
    )


def _load_dataset(member: ReplaySourceMember) -> LarvaDataset:
    if member.registry_ref_id:
        return reg.conf.Ref.loadRef(id=member.registry_ref_id, load=False)
    raise ValueError(f"Cannot load replay source member: {member.token}")


def _resolve_arena_dims(dataset: LarvaDataset) -> tuple[float, float]:
    try:
        dims = dataset.c.env_params.arena.dims
        return (float(dims[0]), float(dims[1]))
    except Exception:
        return (0.2, 0.2)


def _resolve_xy_columns(
    dataset: LarvaDataset, df: pd.DataFrame, *, track_point: int
) -> list[str]:
    if track_point >= 0:
        point = dataset.c.get_track_point(track_point)
        exact = list(util.nam.xy(point))
        if all(col in df.columns for col in exact):
            return exact
        raise ValueError(f"Missing exact xy columns for track_point={track_point}.")
    candidates: list[list[str]] = []
    candidates.extend(
        [
            list(dataset.c.traj_xy),
            list(dataset.c.point_xy),
            list(dataset.c.centroid_xy),
        ]
    )
    for cols in candidates:
        if all(col in df.columns for col in cols):
            return cols
    for cols in dataset.c.all_xy.group_by_n(2):
        if all(col in df.columns for col in cols):
            return list(cols)
    if all(col in df.columns for col in ("x", "y")):
        return ["x", "y"]
    raise ValueError("Could not resolve trajectory xy columns for replay.")


def _plain_mapping(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    return dict(value)


def _aligned_xy(
    xy: pd.DataFrame,
    *,
    transposition: str | None,
    arena_dims: tuple[float, float],
    coordinate_origin: str = "corner",
) -> pd.DataFrame:
    offsets_by_agent = _alignment_offsets_by_agent(
        xy,
        transposition=transposition,
        arena_dims=arena_dims,
        coordinate_origin=coordinate_origin,
    )
    return _apply_alignment_offsets(xy, offsets_by_agent=offsets_by_agent)


def _infer_coordinate_origin(
    xy: pd.DataFrame, arena_dims: tuple[float, float]
) -> ReplayCoordinateOrigin:
    if xy.empty:
        return "corner"
    try:
        width = abs(float(arena_dims[0]))
        height = abs(float(arena_dims[1]))
        x_min = float(xy["x"].min(skipna=True))
        y_min = float(xy["y"].min(skipna=True))
        x_max = float(xy["x"].max(skipna=True))
        y_max = float(xy["y"].max(skipna=True))
    except Exception:
        return "corner"
    values = (width, height, x_min, y_min, x_max, y_max)
    if not all(np.isfinite(value) for value in values):
        return "corner"
    if width <= 0.0 or height <= 0.0:
        return "corner"
    centered_tol_x = max(width * 0.08, 1e-9)
    centered_tol_y = max(height * 0.08, 1e-9)
    fits_centered = (
        x_min >= -width / 2.0 - centered_tol_x
        and x_max <= width / 2.0 + centered_tol_x
        and y_min >= -height / 2.0 - centered_tol_y
        and y_max <= height / 2.0 + centered_tol_y
    )
    has_negative_coordinates = x_min < -centered_tol_x or y_min < -centered_tol_y
    if fits_centered and has_negative_coordinates:
        return "centered"
    return "corner"


def _alignment_offsets_by_agent(
    xy: pd.DataFrame,
    *,
    transposition: str | None,
    arena_dims: tuple[float, float],
    coordinate_origin: str,
) -> dict[object, tuple[float, float]]:
    mode = transposition
    if mode in {"", "none", "stored"}:
        mode = None
    if mode not in {None, "origin", "center", "arena"}:
        raise ValueError(f"Unsupported transposition mode: {mode!r}")

    grouped = xy.groupby(level="AgentID")
    offsets: dict[object, tuple[float, float]] = {}
    for agent_id, g in grouped:
        offsets[agent_id] = (0.0, 0.0)
        if mode is None:
            continue
        if mode == "arena":
            if coordinate_origin == "centered":
                continue
            offsets[agent_id] = (arena_dims[0] / 2.0, arena_dims[1] / 2.0)
            continue
        valid = g.dropna()
        if valid.empty:
            continue
        if mode == "origin":
            x0, y0 = valid.iloc[0]["x"], valid.iloc[0]["y"]
        else:
            x0 = (float(valid["x"].min()) + float(valid["x"].max())) / 2.0
            y0 = (float(valid["y"].min()) + float(valid["y"].max())) / 2.0
        if np.isfinite(x0) and np.isfinite(y0):
            offsets[agent_id] = (float(x0), float(y0))
    return offsets


def _apply_alignment_offsets(
    xy: pd.DataFrame,
    *,
    offsets_by_agent: dict[object, tuple[float, float]],
) -> pd.DataFrame:
    if xy.empty:
        return xy.copy()
    out = []
    for agent_id, g in xy.groupby(level="AgentID"):
        gg = g.copy()
        dx, dy = offsets_by_agent.get(agent_id, (0.0, 0.0))
        gg.loc[:, "x"] = gg["x"] - float(dx)
        gg.loc[:, "y"] = gg["y"] - float(dy)
        out.append(gg)
    if not out:
        return xy.copy()
    return pd.concat(out).sort_index()


def parse_agent_indices(raw: str) -> tuple[int, ...] | None:
    text = str(raw).strip()
    if text == "":
        return None
    parts = [part.strip() for part in text.split(",")]
    if any(part == "" for part in parts):
        raise ValueError("Agent indices must be a comma-separated integer list.")
    values: list[int] = []
    for part in parts:
        if not part.isdigit():
            raise ValueError(f"Invalid agent index: {part!r}")
        value = int(part)
        if value < 0:
            raise ValueError(f"Agent index must be non-negative: {value}")
        values.append(value)
    return tuple(values)


def select_member_xy(member: PreparedReplayMember, *, track_point: int) -> pd.DataFrame:
    if track_point == -1:
        return member.xy_default.copy()
    if track_point < -1:
        raise ValueError("Track point must be -1 or a non-negative integer.")
    selected = member.xy_by_track_point.get(track_point)
    if selected is None:
        raise ValueError(
            f"Track point {track_point} is unavailable for member {member.label}."
        )
    xy = selected.copy()
    xy.columns = ["x", "y"]
    return xy


def filter_xy_by_agent_indices(
    xy: pd.DataFrame,
    member_agent_ids: tuple[object, ...],
    *,
    agent_indices: tuple[int, ...] | None,
) -> pd.DataFrame:
    if agent_indices is None:
        return xy
    if not member_agent_ids:
        raise ValueError("Cannot apply agent index filtering: member has no agent IDs.")
    max_index = len(member_agent_ids) - 1
    selected_agent_ids: list[object] = []
    for idx in agent_indices:
        if idx < 0 or idx > max_index:
            raise ValueError(
                f"Agent index {idx} out of range for member (valid 0..{max_index})."
            )
        selected_agent_ids.append(member_agent_ids[idx])
    return xy[xy.index.get_level_values("AgentID").isin(selected_agent_ids)]


def _select_and_align_geometry_by_point(
    xy_by_point: dict[int, pd.DataFrame],
    *,
    agent_ids: tuple[object, ...],
    agent_indices: tuple[int, ...] | None,
    time_range: tuple[float, float] | None,
    dt: float,
    offsets_by_agent: dict[object, tuple[float, float]],
) -> dict[int, pd.DataFrame]:
    # Deprecated in the replay frame hot path; retained for compatibility.
    selected: dict[int, pd.DataFrame] = {}
    for idx, xy in xy_by_point.items():
        filtered = filter_xy_by_agent_indices(
            xy,
            agent_ids,
            agent_indices=agent_indices,
        )
        if filtered.empty:
            continue
        if time_range is not None and dt > 0:
            s0 = int(time_range[0] / dt)
            s1 = int(time_range[1] / dt)
            filtered = filtered.loc[(slice(s0, s1), slice(None)), :]
        if filtered.empty:
            continue
        selected[idx] = _apply_alignment_offsets(
            filtered,
            offsets_by_agent=offsets_by_agent,
        )
    return selected


def _geometry_points_for_agents_at_tick(
    xy_by_point: dict[int, pd.DataFrame],
    *,
    tick: int,
    emitted_agent_ids: tuple[object, ...],
    offsets_by_agent: dict[object, tuple[float, float]],
    min_points: int,
) -> dict[object, tuple[tuple[float, float], ...]]:
    if not emitted_agent_ids:
        return {}
    emitted_set = set(emitted_agent_ids)
    points_by_agent: dict[object, list[tuple[float, float]]] = {
        agent_id: [] for agent_id in emitted_agent_ids
    }
    for point_idx in sorted(xy_by_point.keys()):
        xy = xy_by_point[point_idx]
        try:
            at_tick = xy.xs(tick, level="Step")
        except KeyError:
            continue
        for agent_id, row in at_tick.iterrows():
            if agent_id not in emitted_set:
                continue
            x_raw = float(row["x"])
            y_raw = float(row["y"])
            dx, dy = offsets_by_agent.get(agent_id, (0.0, 0.0))
            x = float(x_raw - dx)
            y = float(y_raw - dy)
            if np.isfinite(x) and np.isfinite(y):
                points_by_agent[agent_id].append((x, y))
    out: dict[object, tuple[tuple[float, float], ...]] = {}
    for agent_id in emitted_agent_ids:
        points = tuple(points_by_agent.get(agent_id, ()))
        if len(points) >= min_points:
            out[agent_id] = points
    return out


def _segment_polygons_from_midline(
    midline: tuple[tuple[float, float], ...],
) -> tuple[tuple[tuple[float, float], ...], ...]:
    if len(midline) < 2:
        return ()
    lengths: list[float] = []
    pairs: list[
        tuple[tuple[float, float], tuple[float, float], float, float, float]
    ] = []
    for p0, p1 in zip(midline[:-1], midline[1:]):
        dx = float(p1[0] - p0[0])
        dy = float(p1[1] - p0[1])
        length = float((dx * dx + dy * dy) ** 0.5)
        if not np.isfinite(length) or length <= 0:
            continue
        lengths.append(length)
        pairs.append((p0, p1, dx, dy, length))
    if not lengths:
        return ()
    half_width = 0.25 * float(np.median(np.asarray(lengths, dtype=float)))
    if not np.isfinite(half_width) or half_width <= 0:
        return ()
    polygons: list[tuple[tuple[float, float], ...]] = []
    for p0, p1, dx, dy, length in pairs:
        nx = -dy / length
        ny = dx / length
        polygon = (
            (p0[0] + (nx * half_width), p0[1] + (ny * half_width)),
            (p1[0] + (nx * half_width), p1[1] + (ny * half_width)),
            (p1[0] - (nx * half_width), p1[1] - (ny * half_width)),
            (p0[0] - (nx * half_width), p0[1] - (ny * half_width)),
        )
        polygons.append(tuple((float(x), float(y)) for x, y in polygon))
    return tuple(polygons)


def _agent_ids_from_step_index(step: pd.DataFrame) -> tuple[object, ...]:
    return tuple(step.index.get_level_values("AgentID").unique())


def _build_registry_xy_by_track_point(
    dataset: LarvaDataset, step: pd.DataFrame
) -> dict[int, pd.DataFrame]:
    xy_by_track_point: dict[int, pd.DataFrame] = {}
    npoints = int(dataset.c.Npoints or 0)
    for idx in range(max(npoints, 0)):
        point_name = dataset.c.get_track_point(idx)
        cols = list(util.nam.xy(point_name))
        if all(col in step.columns for col in cols):
            xy_tp = step[cols].copy()
            xy_tp.columns = ["x", "y"]
            xy_by_track_point[idx] = xy_tp
    return xy_by_track_point


def _build_registry_native_track_point_mapping(
    dataset: LarvaDataset, step: pd.DataFrame
) -> dict[int, int]:
    mapping: dict[int, int] = {}
    npoints = int(dataset.c.Npoints or 0)
    for ui_idx in range(max(npoints, 0)):
        point_name = dataset.c.get_track_point(ui_idx)
        cols = list(util.nam.xy(point_name))
        if all(col in step.columns for col in cols):
            mapping[ui_idx] = ui_idx
    return mapping


def _native_replay_missing_columns(step: pd.DataFrame) -> tuple[str, ...]:
    required = ("front_orientation", "rear_orientation")
    return tuple(col for col in required if col not in step.columns)


def _workspace_point_name(conf: dict[str, Any], native_idx: int) -> str:
    npoints = int(conf.get("Npoints") or 0)
    if native_idx <= 0 or npoints <= 0:
        return "centroid"
    points = list(util.nam.midline(npoints, type="point"))
    if not points:
        return "centroid"
    clamped = max(1, min(native_idx, len(points)))
    return points[clamped - 1]


def _read_workspace_hdf_group(h5_path: Path, key: str) -> pd.DataFrame | None:
    full_key = f"/{key}"
    with pd.HDFStore(h5_path, mode="r") as store:
        if full_key not in set(store.keys()):
            return None
    return pd.read_hdf(h5_path, key)


def _join_workspace_spatial_group(
    base: pd.DataFrame,
    group: pd.DataFrame | None,
    *,
    group_name: str,
    dataset_id: str,
) -> pd.DataFrame:
    if group is None:
        return base
    if not group.index.equals(base.index):
        raise ValueError(
            f"Workspace dataset has incompatible {group_name} index: {dataset_id}"
        )
    new_columns = [col for col in group.columns if col not in base.columns]
    if not new_columns:
        return base
    return base.join(group[new_columns])


def _build_workspace_spatial_step(
    step: pd.DataFrame,
    *,
    h5_path: Path,
    dataset_id: str,
) -> pd.DataFrame:
    spatial_step = step.copy()
    base_spatial = _read_workspace_hdf_group(h5_path, "base_spatial")
    spatial_step = _join_workspace_spatial_group(
        spatial_step,
        base_spatial,
        group_name="base_spatial",
        dataset_id=dataset_id,
    )
    midline = _read_workspace_hdf_group(h5_path, "midline")
    spatial_step = _join_workspace_spatial_group(
        spatial_step,
        midline,
        group_name="midline",
        dataset_id=dataset_id,
    )
    contour = _read_workspace_hdf_group(h5_path, "contour")
    spatial_step = _join_workspace_spatial_group(
        spatial_step,
        contour,
        group_name="contour",
        dataset_id=dataset_id,
    )
    return spatial_step


def _resolve_workspace_xy_columns(
    spatial_step: pd.DataFrame,
    conf: dict[str, Any],
    *,
    dataset_id: str,
) -> list[str]:
    x_name = conf.get("x")
    y_name = conf.get("y")
    if (
        isinstance(x_name, str)
        and isinstance(y_name, str)
        and x_name in spatial_step.columns
        and y_name in spatial_step.columns
    ):
        return [x_name, y_name]
    if "x" in spatial_step.columns and "y" in spatial_step.columns:
        return ["x", "y"]
    point_name = conf.get("point")
    if isinstance(point_name, str) and point_name:
        point_cols = list(util.nam.xy(point_name))
        if all(col in spatial_step.columns for col in point_cols):
            return point_cols
    point_idx = conf.get("point_idx")
    try:
        parsed_point_idx = int(point_idx)
    except (TypeError, ValueError):
        parsed_point_idx = None
    if parsed_point_idx is not None:
        idx_point_cols = list(
            util.nam.xy(_workspace_point_name(conf, parsed_point_idx))
        )
        if all(col in spatial_step.columns for col in idx_point_cols):
            return idx_point_cols
    centroid_cols = ["centroid_x", "centroid_y"]
    if all(col in spatial_step.columns for col in centroid_cols):
        return centroid_cols
    npoints = int(conf.get("Npoints") or 0)
    for point in util.nam.midline(npoints, type="point"):
        cols = list(util.nam.xy(point))
        if all(col in spatial_step.columns for col in cols):
            return cols
    raise ValueError(f"Workspace dataset is missing xy columns: {dataset_id}")


def _build_workspace_native_track_point_mapping(
    step: pd.DataFrame, conf: dict[str, Any]
) -> dict[int, int]:
    mapping: dict[int, int] = {}
    npoints = int(conf.get("Npoints") or 0)
    for ui_idx, point_name in enumerate(util.nam.midline(npoints, type="point")):
        cols = list(util.nam.xy(point_name))
        if all(col in step.columns for col in cols):
            mapping[ui_idx] = ui_idx + 1
    return mapping


def _has_native_point_xy(
    step: pd.DataFrame,
    *,
    native_track_point: int,
    point_name_for_native_index: Any,
) -> bool:
    point_name = point_name_for_native_index(int(native_track_point))
    cols = list(util.nam.xy(point_name))
    return all(col in step.columns for col in cols)


def _choose_native_default_track_point(
    *,
    step: pd.DataFrame,
    native_track_point_by_ui_track_point: dict[int, int],
    configured_point_idx: int,
    traj_xy: list[str],
    centroid_xy: list[str],
    point_name_for_native_index: Any,
) -> int | None:
    if _has_native_point_xy(
        step,
        native_track_point=configured_point_idx,
        point_name_for_native_index=point_name_for_native_index,
    ):
        return int(configured_point_idx)
    if all(col in step.columns for col in traj_xy) or all(
        col in step.columns for col in centroid_xy
    ):
        return -1
    native_body_points = sorted(
        native
        for native in native_track_point_by_ui_track_point.values()
        if int(native) > 0
    )
    if native_body_points:
        return int(native_body_points[0])
    return None


def _build_body_xy_by_point(
    step: pd.DataFrame, npoints: int
) -> dict[int, pd.DataFrame]:
    xy_by_point: dict[int, pd.DataFrame] = {}
    for idx, point_name in enumerate(util.nam.midline(max(npoints, 0), type="point")):
        cols = list(util.nam.xy(point_name))
        if all(col in step.columns for col in cols):
            xy_tp = step[cols].copy()
            xy_tp.columns = ["x", "y"]
            xy_by_point[idx] = xy_tp
    return xy_by_point


def _build_contour_xy_by_point(
    step: pd.DataFrame, ncontour: int
) -> dict[int, pd.DataFrame]:
    xy_by_point: dict[int, pd.DataFrame] = {}
    for idx, point_name in enumerate(util.nam.contour(max(ncontour, 0))):
        cols = list(util.nam.xy(point_name))
        if all(col in step.columns for col in cols):
            xy_tp = step[cols].copy()
            xy_tp.columns = ["x", "y"]
            xy_by_point[idx] = xy_tp
    return xy_by_point


def _build_workspace_xy_by_track_point(
    step: pd.DataFrame, conf: dict[str, Any]
) -> dict[int, pd.DataFrame]:
    xy_by_track_point: dict[int, pd.DataFrame] = {}
    npoints = int(conf.get("Npoints") or 0)
    for idx, point_name in enumerate(util.nam.midline(npoints, type="point")):
        cols = list(util.nam.xy(point_name))
        if all(col in step.columns for col in cols):
            xy_tp = step[cols].copy()
            xy_tp.columns = ["x", "y"]
            xy_by_track_point[idx] = xy_tp
    return xy_by_track_point
