from __future__ import annotations

from dataclasses import dataclass
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
) -> EnvironmentCanvasState:
    if not allow_static_layers:
        return EnvironmentCanvasState(
            arena=CanvasArena("rectangular", member.arena_dims),
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
) -> ReplayRenderState:
    centroids: list[tuple[float, float]] = []
    trails: list[tuple[tuple[float, float], ...]] = []
    colors: list[str] = []
    labels: list[str] = []
    dispersal_distances: list[float] = []

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
        xy = _aligned_xy(
            xy,
            transposition=transposition,
            arena_dims=member.arena_dims,
            coordinate_origin=member.coordinate_origin,
        )
        try:
            at_tick = xy.xs(tick, level="Step")
        except KeyError:
            continue
        grouped = xy.groupby(level="AgentID")
        for agent_id, at_row in at_tick.iterrows():
            x = float(at_row["x"])
            y = float(at_row["y"])
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            if show_positions:
                centroids.append((x, y))
                colors.append(member.color)
                labels.append(str(agent_id) if show_ids else "")
            if show_tracks:
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
        trails=tuple(trails),
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
    arena_dims = _resolve_arena_dims(dataset)
    dt = float(dataset.c.dt or 0.1)
    nticks = int(dataset.c.Nticks or (int(xy.index.unique("Step").max()) + 1))
    color = str(dataset.c.color or "#2f4858")
    env_conf_id = None
    agent_ids = tuple(dataset.c.agent_ids or ())
    if not agent_ids:
        agent_ids = _agent_ids_from_step_index(xy)
    return PreparedReplayMember(
        token=member.token,
        label=member.label,
        color=color,
        xy_default=xy,
        arena_dims=arena_dims,
        dt=dt,
        nticks=nticks,
        env_conf_id=env_conf_id,
        coordinate_origin="corner",
        xy_by_track_point=xy_by_track_point,
        agent_ids=agent_ids,
    )


def _prepare_workspace_member(member: ReplaySourceMember) -> PreparedReplayMember:
    assert member.workspace_record is not None
    record = member.workspace_record
    conf = util.load_dict(str(record.conf_path))
    step = pd.read_hdf(record.h5_path, "step")
    x_name = str(conf.get("x") or "x")
    y_name = str(conf.get("y") or "y")
    if x_name not in step.columns or y_name not in step.columns:
        if "x" in step.columns and "y" in step.columns:
            x_name, y_name = "x", "y"
        else:
            raise ValueError(
                f"Workspace dataset is missing xy columns: {record.dataset_id}"
            )
    xy = step[[x_name, y_name]].copy()
    xy.columns = ["x", "y"]
    xy_by_track_point = _build_workspace_xy_by_track_point(step, conf)
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
        coordinate_origin=(
            "centered"
            if getattr(record, "origin", None) == "simulation_run"
            else "corner"
        ),
        xy_by_track_point=xy_by_track_point,
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
    mode = transposition
    if mode in {"", "none", "stored"}:
        mode = None
    xy = xy.copy()
    if mode is None:
        return xy
    assert mode in {"origin", "center", "arena"}
    if mode == "arena":
        if coordinate_origin == "centered":
            return xy
        xy.loc[:, "x"] = xy["x"] - (arena_dims[0] / 2.0)
        xy.loc[:, "y"] = xy["y"] - (arena_dims[1] / 2.0)
        return xy
    out = []
    for _, g in xy.groupby(level="AgentID"):
        gg = g.copy()
        valid = gg.dropna()
        if valid.empty:
            out.append(gg)
            continue
        if mode == "origin":
            x0, y0 = valid.iloc[0]["x"], valid.iloc[0]["y"]
        else:
            x0 = (float(valid["x"].min()) + float(valid["x"].max())) / 2.0
            y0 = (float(valid["y"].min()) + float(valid["y"].max())) / 2.0
        gg.loc[:, "x"] = gg["x"] - x0
        gg.loc[:, "y"] = gg["y"] - y0
        out.append(gg)
    if not out:
        return xy
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
