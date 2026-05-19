from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

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
from larvaworld.portal.datasets.workspace_index import list_workspace_datasets
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
                    label=f"Workspace / Dataset / {record.dataset_id}",
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
                    label=f"Workspace / Group / {lab_id}:{group_id}",
                    source_type="workspace_group",
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
    member: PreparedReplayMember, *, allow_static_layers: bool
) -> EnvironmentCanvasState:
    if not allow_static_layers:
        return EnvironmentCanvasState(
            arena=CanvasArena("rectangular", member.arena_dims), objects=()
        )
    if member.env_conf_id and member.env_conf_id in reg.conf.Env.confIDs:
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
        xy = _aligned_xy(
            member.xy_default,
            transposition=transposition,
            track_point=track_point,
            arena_dims=member.arena_dims,
        )
        if time_range is not None and member.dt > 0:
            s0 = int(time_range[0] / member.dt)
            s1 = int(time_range[1] / member.dt)
            xy = xy.loc[(slice(s0, s1), slice(None)), :]
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
    arena_dims = _resolve_arena_dims(dataset)
    dt = float(dataset.c.dt or 0.1)
    nticks = int(dataset.c.Nticks or (int(xy.index.unique("Step").max()) + 1))
    color = str(dataset.c.color or "#2f4858")
    env_conf_id = None
    return PreparedReplayMember(
        token=member.token,
        label=member.label,
        color=color,
        xy_default=xy,
        arena_dims=arena_dims,
        dt=dt,
        nticks=nticks,
        env_conf_id=env_conf_id,
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
    candidates: list[list[str]] = []
    if track_point >= 0:
        point = dataset.c.get_track_point(track_point)
        candidates.append(list(util.nam.xy(point)))
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


def _aligned_xy(
    xy_default: pd.DataFrame,
    *,
    transposition: str | None,
    track_point: int,
    arena_dims: tuple[float, float],
) -> pd.DataFrame:
    mode = transposition
    if mode in {"", "none", "stored"}:
        mode = None
    xy = xy_default.copy()
    if mode is None:
        return xy
    assert mode in {"origin", "center", "arena"}
    if mode == "arena":
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
