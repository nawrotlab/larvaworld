"""
Agent 2D trajectory-related plotting
"""

from __future__ import annotations
from typing import Any, Dict, Optional, Sequence, Tuple

import copy

import numpy as np

from .. import plot, reg, util, funcs
from ..util import nam

__all__: list[str] = [
    "traj_1group",
    "traj_grouped",
    "track_annotated",
    "annotated_strideplot",
    "annotated_turnplot",
    "track_annotated_data",
    "annotated_strideplot_data",
    "annotated_turnplot_data",
    "plot_marked_strides",
    "plot_sample_tracks",
]


def traj_1group(
    d: Any,
    unit: str = "mm",
    mode: str = "default",
    title: Optional[str] = None,
    single_color: bool = False,
    time_range: Optional[Tuple[int, int]] = None,
    **kwargs: Any,
) -> Any:
    """
    Plot trajectories for a single dataset.

    Creates 2D trajectory plot showing movement paths of all individuals
    within arena boundaries with food sources marked.

    Args:
        d: Dataset containing trajectory data
        unit: Distance unit ('mm' or 'm'). Defaults to 'mm'
        mode: Trajectory mode. Defaults to 'default'
        title: Plot title. Defaults to None
        single_color: Use single color for all tracks. Defaults to False
        time_range: Time range (start, end) to slice. Defaults to None
        **kwargs: Additional arguments passed to AutoBasePlot

    Returns:
        Plot output (figure object or None based on return_fig setting)

    Example:
        >>> fig = traj_1group(dataset, unit='mm', single_color=True)
    """
    df = d.load_traj(mode=mode)
    xy = d.timeseries_slice(time_range=time_range, df=df)[d.c.traj_xy]
    c = d.config
    color = c.color if single_color else None
    scale = 1000 if unit == "mm" else 1
    P = plot.AutoBasePlot(name="trajectories", **kwargs)
    ax = P.axs[0]
    tank = c.arena_vertices * scale
    for xy0 in d.data_by_ID(xy).values():
        ax.plot(xy0[:, 0] * scale, xy0[:, 1] * scale, color=color)

    ax.fill(
        tank[:, 0],
        tank[:, 1],
        fill=True,
        color="lightgrey",
        edgecolor="black",
        linewidth=4,
    )
    from matplotlib import pyplot as plt

    for sdic in c.env_params.food_params.source_units.values():
        px, py = sdic.pos
        circle = plt.Circle(
            (px * scale, py * scale), sdic.radius * scale, color=sdic.color
        )
        ax.add_patch(circle)
    P.conf_ax(
        xMaxN=3,
        yMaxN=3,
        title=title,
        titlefontsize=25,
        xlab=f"X ({unit})",
        ylab=f"Y ({unit})",
        equal_aspect=True,
    )
    return P.get()


@funcs.graph("trajectories", required={"traj": []})
def traj_grouped(
    unit: str = "mm",
    name: Optional[str] = None,
    subfolder: str = "trajectories",
    range: Optional[Tuple[int, int]] = None,
    mode: str = "default",
    single_color: bool = False,
    **kwargs: Any,
) -> Any:
    """
    Plot trajectories grouped by dataset.

    Creates multi-panel trajectory plots with one panel per dataset,
    showing all individual tracks within each dataset.

    Args:
        unit: Distance unit ('mm' or 'm'). Defaults to 'mm'
        name: Plot name for saving. Auto-generated if None
        subfolder: Subfolder for saving. Defaults to 'trajectories'
        range: Time range (start, end) to slice. Defaults to None
        mode: Trajectory mode. Defaults to 'default'
        single_color: Use single color per dataset. Defaults to False
        **kwargs: Additional arguments passed to AutoPlot

    Returns:
        Plot output (figure object or None based on return_fig setting)

    Example:
        >>> fig = traj_grouped(datasets=[d1, d2], unit='mm', range=(0, 60))
    """
    if name is None:
        name = f"comparative_trajectories_{mode}"

    P = plot.AutoPlot(
        name=name,
        subfolder=subfolder,  # subplot_kw=dict(projection='polar'),
        build_kws={"Ncols": "Ndatasets", "wh": 5, "sharex": True, "sharey": True},
        **kwargs,
    )
    for ii, (l, d) in enumerate(P.data_dict.items()):
        _ = traj_1group(
            d=d,
            unit=unit,
            mode=mode,
            fig=P.fig,
            axs=P.axs[ii],
            title=l,
            single_color=single_color,
            save_to=None,
            time_range=range,
        )
        if ii != 0:
            P.axs[ii].yaxis.set_visible(False)
    P.adjust((0.1, 0.9), (0.2, 0.9), 0.1, 0.01)
    return P.get()


def ax_conf_kws(
    kws: Any,
    trange: Any,
    Ndatasets: int,
    Nrows: int,
    i: int = 0,
    ylab: Optional[str] = None,
    ylim: Optional[Sequence[float]] = None,
    xlim: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    conf_kws = {
        "ylab": kws.ylab if ylab is None else ylab,
        "ylim": kws.ylim if ylim is None else ylim,
        "xlim": (0, trange[-1]) if xlim is None else xlim,
        "xlab": r"time $(sec)$",
        "xvis": True if i == Nrows - 1 else False,
    }

    from matplotlib import patches as mpatches

    leg_kws = {
        "leg_loc": "upper right",
        "leg_handles": [
            mpatches.Patch(color=col, label=l)
            for l, col in zip(kws.labels, kws.chunk_cols)
        ],
        "leg_labels": kws.labels,
        "legfontsize": 15,
    }

    return {**conf_kws, **leg_kws}


def track_annotated(
    epoch: str = "stride",
    a: Any = None,
    dt: float = 0.1,
    a2plot: Any = None,
    ylab: Optional[str] = None,
    ylim: Optional[Sequence[float]] = None,
    xlim: Optional[Sequence[float]] = None,
    slice: Optional[Tuple[int, int]] = None,
    agent_idx: int = 0,
    agent_id: Optional[str] = None,
    subfolder: str = "tracks",
    moving_average_interval: Optional[float] = None,
    epoch_boundaries: bool = True,
    show_extrema: bool = True,
    min_amp: Optional[float] = None,
    **kwargs: Any,
) -> Any:
    """
    Create annotated track with behavioral epochs highlighted.

    Generates time series plot with stride/turn epochs shown as colored
    background regions and extrema marked.

    Args:
        epoch: Epoch type ('stride' or 'turn'). Defaults to 'stride'
        a: Parameter array to plot. Defaults to None
        dt: Time step in seconds. Defaults to 0.1
        a2plot: Alternative parameter to overlay. Defaults to None
        ylab: Y-axis label. Auto-generated if None
        ylim: Y-axis limits. Defaults to None
        xlim: X-axis limits. Defaults to None
        slice: Time slice (start, end). Defaults to None
        agent_idx: Agent index. Defaults to 0
        agent_id: Specific agent ID. Uses agent_idx if None
        subfolder: Subfolder for saving. Defaults to 'tracks'
        moving_average_interval: Smoothing window. Defaults to None
        epoch_boundaries: Show epoch boundaries. Defaults to True
        show_extrema: Mark velocity extrema. Defaults to True
        min_amp: Minimum amplitude for turns. Defaults to None
        **kwargs: Additional arguments passed to AutoPlot

    Returns:
        Plot output (figure object or None based on return_fig setting)

    Example:
        >>> fig = track_annotated(datasets=[d1], epoch='stride', agent_idx=0)
    """
    temp = f"track_{slice[0]}-{slice[1]}" if slice is not None else "track"
    name = f"{temp}_{agent_id}" if agent_id is not None else f"{temp}_{agent_idx}"
    P = plot.AutoPlot(
        name=name,
        subfolder=subfolder,
        build_kws={
            "Nrows": "Ndatasets",
            "w": 20,
            "h": 5,
            "sharex": True,
            "sharey": True,
        },
        **kwargs,
    )
    d = P.datasets[0]
    Nticks = a.shape[0]
    trange = np.arange(0, Nticks * dt, dt)

    ax = P.axs[0]

    def stride_epochs(a: np.ndarray, ax: Any) -> Tuple[list, np.ndarray]:
        i_min, i_max, strides = d.detect_strides(a=a)
        runs, run_counts = d.detect_stridechains(strides)
        if show_extrema and a2plot is None:
            ax.plot(
                trange[i_max],
                a[i_max],
                linestyle="None",
                lw=10,
                color="green",
                marker="v",
            )
            ax.plot(
                trange[i_min],
                a[i_min],
                linestyle="None",
                lw=10,
                color="red",
                marker="^",
            )
        pauses = d.detect_pauses(a, runs=runs)
        return [runs, pauses], strides

    def turn_epochs(a: np.ndarray, ax: Any) -> Tuple[list, np.ndarray]:
        ax.axhline(0, color="black", alpha=1, linestyle="dashed", linewidth=1)
        Lturns, Rturns = d.detect_turns(a)
        if min_amp is not None:
            # Ldurs = d.epoch_durs(Lturns)
            # Rdurs = d.epoch_durs(Rturns)
            Lamps = d.epoch_amps(Lturns, a)
            Ramps = d.epoch_amps(Rturns, a)
            # Lmaxs = d.epoch_maxs(Lturns, a.values)
            # Rmaxs = d.epoch_maxs(Rturns, a.values)
            Lturns = Lturns[np.abs(Lamps) > min_amp]
            Rturns = Rturns[np.abs(Ramps) > min_amp]
        return [Lturns, Rturns], np.vstack([Lturns, Rturns])

    epoch_dict = util.AttrDict(
        {
            "stride": {
                "ylab": "velocity (1/sec)",
                "labels": ["runs", "pauses"],
                "chunk_cols": ["lightblue", "grey"],
                "func": stride_epochs,
            },
            "turn": {
                "ylab": "angular velocity (deg/sec)",
                "labels": ["L turns", "R turns"],
                "chunk_cols": ["lightgreen", "orange"],
                "func": turn_epochs,
            },
        }
    )

    kws = epoch_dict[epoch]

    conf_kws = {
        "ylab": kws.ylab if ylab is None else ylab,
        "ylim": ylim,
        "xlim": (0, trange[-1]) if xlim is None else xlim,
        "xlab": r"time $(sec)$" if 0 == P.Ndatasets - 1 else None,
    }

    if a2plot is not None:
        aa2plot = a2plot
    else:
        if moving_average_interval:
            a = util.moving_average(a, n=int(moving_average_interval / dt))
        aa2plot = a

    ax.plot(trange, aa2plot)

    epochs, epochs0 = kws.func(a, ax=ax)
    # plot.color_epochs(epochs=epochs0, epoch_area=False,epoch_boundaries=epoch_boundaries, edgecolor=f'{0.4 * (0 + 1)}',ax=ax,trange=trange)
    # for color, epoch in zip(kws.chunk_cols, epochs):
    #     plot.color_epochs(epochs=epoch, epoch_boundaries=False, facecolor=color,ax=ax,trange=trange)
    #
    if epoch_boundaries:
        for s0, s1 in epochs0:
            for s01 in [s0, s1]:
                ax.axvline(
                    trange[s01],
                    color=f"{0.4 * (0 + 1)}",
                    alpha=0.3,
                    linestyle="dashed",
                    linewidth=1,
                )

    for color, epoch in zip(kws.chunk_cols, epochs):
        for s0, s1 in epoch:
            ax.axvspan(trange[s0], trange[s1], color=color, alpha=1.0)

    from matplotlib import patches as mpatches

    leg_kws = {
        "leg_loc": "upper right",
        "leg_handles": [
            mpatches.Patch(color=col, label=l)
            for l, col in zip(kws.labels, kws.chunk_cols)
        ],
        "leg_labels": kws.labels,
        "legfontsize": 15,
    }

    P.conf_ax(0, **conf_kws, **leg_kws)
    return P.get()


def annotated_strideplot(**kwargs: Any) -> Any:
    """
    Create stride-annotated track plot.

    Wrapper function that creates track with stride epochs highlighted.

    Args:
        **kwargs: Arguments passed to track_annotated

    Returns:
        Plot output (figure object or None based on return_fig setting)

    Example:
        >>> fig = annotated_strideplot(datasets=[d1], agent_idx=0)
    """
    return track_annotated(epoch="stride", **kwargs)


def annotated_turnplot(**kwargs: Any) -> Any:
    """
    Create turn-annotated track plot.

    Wrapper function that creates track with turn epochs highlighted.

    Args:
        **kwargs: Arguments passed to track_annotated

    Returns:
        Plot output (figure object or None based on return_fig setting)

    Example:
        >>> fig = annotated_turnplot(datasets=[d1], agent_idx=0)
    """
    return track_annotated(epoch="turn", **kwargs)


def track_annotated_data(
    name: Optional[str] = None,
    subfolder: str = "tracks",
    epoch: str = "stride",
    a2plot_k: Optional[str] = None,
    agent_idx: Sequence[int] = [3, 4, 5, 6],
    dur: float = 1,
    **kwargs: Any,
) -> Any:
    """
    Create annotated tracks for multiple individuals from datasets.

    Generates multi-panel figure with annotated tracks for multiple agents,
    showing behavioral epochs and trajectory metrics.

    Args:
        name: Plot name for saving. Auto-generated if None
        subfolder: Subfolder for saving. Defaults to 'tracks'
        epoch: Epoch type ('stride' or 'turn'). Defaults to 'stride'
        a2plot_k: Additional parameter key to overlay. Defaults to None
        agent_idx: List of agent indices to plot. Defaults to [3, 4, 5, 6]
        dur: Duration in minutes. Defaults to 1
        **kwargs: Additional arguments passed to AutoPlot

    Returns:
        Plot output (figure object or None based on return_fig setting)

    Example:
        >>> fig = track_annotated_data(datasets=[d1, d2], epoch='stride', agent_idx=[0, 1, 2])
    """
    if name is None:
        name = f"annotated_{epoch}plot"
    Nidx = len(agent_idx)

    P = plot.AutoPlot(
        name=name,
        subfolder=subfolder,
        build_kws={
            "Nrows": "Ndatasets",
            "Nrows_coef": Nidx,
            "Ncols": 1,
            "w": 15,
            "h": 3,
            "sharex": True,
            "sharey": True,
        },
        **kwargs,
    )
    epoch_kdic = {"stride": "sv", "turn": "fov"}
    apar = reg.getPar(epoch_kdic[epoch])

    def get_a(ss: Any) -> np.ndarray:
        return ss[apar].values

    if a2plot_k is not None:
        par, lab = reg.getPar(a2plot_k, to_return=["d", "symbol"])
    else:
        par, lab = None, None

    def get_a2plot(ss: Any) -> Optional[np.ndarray]:
        return ss[par].values if par is not None else None

    def get_title(idx: int, c: Any, e: Any, l: str) -> str:
        id = c.agent_ids[idx]
        ee = e.loc[id]

        length = np.round(ee["length"] * 1000, 2)
        cum_sd = np.round(ee[reg.getPar("cum_sd")], 2)
        run_tr = int(ee[reg.getPar("run_tr")] * 100)
        title = f"{l}  # {idx} track, l : {length} mm, pathlength {cum_sd}xl , {run_tr}% time crawling"
        return title

    for jj, (l, d) in enumerate(P.data_dict.items()):
        s, e, c = d.step_data, d.endpoint_data, d.config
        Nticks = int(dur * 60 / c.dt)
        kws0 = util.AttrDict(
            {
                "datasets": [d],
                "labels": [l],
                "slice": (0, dur * 60),
                "dt": c.dt,
                "fig": P.fig,
                "show": False,
                "epoch": epoch,
                "ylab": lab,
            }
        )

        for i, idx in enumerate(agent_idx):
            ii = Nidx * jj + i
            id = c.agent_ids[idx]
            ss = s.xs(id, level="AgentID", drop_level=True).loc[:Nticks]
            title = get_title(idx, c, e, l)
            # try:
            #     chunk_dict=d.chunk_dicts[id]
            # except:
            #     chunk_dict=None
            kws1 = util.AttrDict(
                {
                    "agent_id": id,
                    "a": get_a(ss),
                    "axs": P.axs[ii],
                    "a2plot": get_a2plot(ss),
                    # 'chunk_dict':chunk_dict
                    **kws0,
                }
            )
            track_annotated(**kws1)
            P.conf_ax(
                ii, xvis=True if ii == P.Nrows - 1 else False, ylab=lab, title=title
            )
    P.adjust((0.1, 0.98), (0.05, 0.95), 0.001, 0.2)
    P.fig.align_ylabels(P.axs[:])
    return P.get()


@funcs.graph("stride track")
def annotated_strideplot_data(**kwargs: Any) -> Any:
    """
    Create stride-annotated tracks for multiple individuals.

    Wrapper that generates annotated tracks with stride epochs for
    multiple agents from datasets.

    Args:
        **kwargs: Arguments passed to track_annotated_data

    Returns:
        Plot output (figure object or None based on return_fig setting)

    Example:
        >>> fig = annotated_strideplot_data(datasets=[d1, d2], agent_idx=[0, 1])
    """
    return track_annotated_data(epoch="stride", **kwargs)


@funcs.graph("turn track")
def annotated_turnplot_data(**kwargs: Any) -> Any:
    """
    Create turn-annotated tracks for multiple individuals.

    Wrapper that generates annotated tracks with turn epochs for
    multiple agents from datasets.

    Args:
        **kwargs: Arguments passed to track_annotated_data

    Returns:
        Plot output (figure object or None based on return_fig setting)

    Example:
        >>> fig = annotated_turnplot_data(datasets=[d1, d2], agent_idx=[0, 1])
    """
    return track_annotated_data(epoch="turn", **kwargs)


@funcs.graph("marked strides")
def plot_marked_strides(
    agent_idx: int = 0,
    agent_id: Optional[str] = None,
    slice: Sequence[int] = [20, 40],
    subfolder: str = "individuals",
    **kwargs: Any,
) -> Any:
    """
    Plot velocity track with marked stride epochs.

    Creates time series showing velocity with stride and pause epochs
    highlighted and extrema marked for individual analysis.

    Args:
        agent_idx: Agent index. Defaults to 0
        agent_id: Specific agent ID. Uses agent_idx if None
        slice: Time slice [start, end] in seconds. Defaults to [20, 40]
        subfolder: Subfolder for saving. Defaults to 'individuals'
        **kwargs: Additional arguments passed to AutoPlot

    Returns:
        Plot output (figure object or None based on return_fig setting)

    Example:
        >>> fig = plot_marked_strides(datasets=[d1, d2], agent_idx=0, slice=[0, 60])
    """
    temp = (
        f"marked_strides_{slice[0]}-{slice[1]}"
        if slice is not None
        else "marked_strides"
    )
    name = f"{temp}_{agent_id}" if agent_id is not None else f"{temp}_{agent_idx}"
    figx = 15 * 6 * 3 if slice is None else int((slice[1] - slice[0]) / 3)
    chunks = ["stride", "pause"]
    chunk_cols = ["lightblue", "grey"]
    p, ylab = reg.getPar("sv", to_return=["d", "l"])
    P = plot.AutoPlot(
        name=name,
        subfolder=subfolder,
        build_kws={
            "Nrows": "Ndatasets",
            "sharex": True,
            "sharey": True,
            "w": figx,
            "h": 5,
        },
        **kwargs,
    )
    from matplotlib import patches as mpatches

    handles = [
        mpatches.Patch(color=col, label=n)
        for n, col in zip(["stride", "pause"], chunk_cols)
    ]

    for ii, (d, l) in enumerate(zip(P.datasets, P.labels)):
        ax = P.axs[ii]
        P.conf_ax(
            ii,
            xlab=r"time $(sec)$" if ii == P.Ndatasets - 1 else None,
            ylab=ylab,
            ylim=[0, 1.0],
            xlim=slice,
            leg_loc="upper right",
            leg_handles=handles,
        )
        temp_id = d.agent_ids[agent_idx] if agent_id is None else agent_id
        s = copy.deepcopy(d.read("step").xs(temp_id, level="AgentID", drop_level=True))
        s.set_index(s.index * d.dt, inplace=True)
        ax.plot(s[p], color="blue")
        for i, (c, col) in enumerate(zip(chunks, chunk_cols)):
            s0s = s.index[s[nam.start(c)] == True]
            s1s = s.index[s[nam.stop(c)] == True]
            for s0, s1 in zip(s0s, s1s):
                ax.axvspan(s0, s1, color=col, alpha=1.0)
                ax.axvline(
                    s0,
                    color=f"{0.4 * (i + 1)}",
                    alpha=0.6,
                    linestyle="dashed",
                    linewidth=1,
                )
                ax.axvline(
                    s1,
                    color=f"{0.4 * (i + 1)}",
                    alpha=0.6,
                    linestyle="dashed",
                    linewidth=1,
                )
        ax.plot(
            s[p].loc[s[nam.max(p)] == True],
            linestyle="None",
            lw=10,
            color="green",
            marker="v",
        )
        ax.plot(
            s[p].loc[s[nam.min(p)] == True],
            linestyle="None",
            lw=10,
            color="red",
            marker="^",
        )
    P.adjust((0.08, 0.95), (0.15, 0.95), H=0.1)
    return P.get()


@funcs.graph("sample tracks")
def plot_sample_tracks(
    mode: Sequence[str] = ["strides", "turns"],
    agent_idx: int = 4,
    agent_id: Optional[str] = None,
    slice: Sequence[int] = [0, 160],
    subfolder: str = "individuals",
    **kwargs: Any,
) -> Any:
    """
    Plot sample tracks with stride and/or turn annotations.

    Creates multi-panel figure showing strides and/or turns for sample
    individuals across datasets with detailed epoch marking.

    Args:
        mode: Plot modes ('strides', 'turns', or both). Defaults to both
        agent_idx: Agent index. Defaults to 4
        agent_id: Specific agent ID. Uses agent_idx if None
        slice: Time slice [start, end] in seconds. Defaults to [0, 160]
        subfolder: Subfolder for saving. Defaults to 'individuals'
        **kwargs: Additional arguments passed to AutoPlot

    Returns:
        Plot output (figure object or None based on return_fig setting)

    Example:
        >>> fig = plot_sample_tracks(datasets=[d1, d2], mode=['strides'], slice=[0, 120])
    """
    Nrows = len(mode)
    if Nrows == 2:
        suf = "stridesVSturns"
    else:
        suf = mode[0]
    t0, t1 = slice
    figx = 15 * 6 * 3 if slice is None else int((t1 - t0) / 3)
    temp = f"sample_marked_{suf}_{t0}-{t1}"
    name = f"{temp}_{agent_id}" if agent_id is not None else f"{temp}_{agent_idx}"
    P = plot.AutoPlot(
        name=name,
        subfolder=subfolder,
        build_kws={
            "Ncols": "Ndatasets",
            "Nrows": Nrows,
            "w": figx,
            "h": 5,
            "sharey": True,
        },
        **kwargs,
    )

    for ii, (l, d) in enumerate(P.data_dict.items()):
        for jj, key in enumerate(mode):
            kk = ii + Nrows * jj
            ax = P.axs[kk]
            print(kk)
            if key == "strides":
                chunks = ["stride", "pause"]
                chunk_cols = ["lightblue", "grey"]

                p, ylab, ylim = reg.getPar("sv", to_return=["d", "l", "lim"])
                ylim = (0.0, 1.0)
            elif key == "turns":
                chunks = ["Rturn", "Lturn"]
                chunk_cols = ["lightgreen", "orange"]

                b = "bend"
                bv = nam.vel(b)
                ho = nam.orient("front")
                hov = nam.vel(ho)
                p, ylab, ylim = reg.getPar("fov", to_return=["d", "l", "lim"])
            else:
                raise

            from matplotlib import patches as mpatches

            handles = [
                mpatches.Patch(color=col, label=n) for n, col in zip(chunks, chunk_cols)
            ]
            P.conf_ax(
                kk,
                xlab=r"time $(sec)$" if jj == Nrows - 1 else None,
                ylab=ylab,
                ylim=ylim,
                xlim=slice,
                leg_loc="upper right",
                leg_handles=handles,
            )

            temp_id = d.agent_ids[agent_idx] if agent_id is None else agent_id
            s = copy.deepcopy(d.s.xs(temp_id, level="AgentID", drop_level=True))
            s.set_index(s.index * d.dt, inplace=True)
            ax.plot(s[p], color="blue")
            for i, (c, col) in enumerate(zip(chunks, chunk_cols)):
                s0s = s.index[s[nam.start(c)] == True]
                s1s = s.index[s[nam.stop(c)] == True]
                for s0, s1 in zip(s0s, s1s):
                    ax.axvspan(s0, s1, color=col, alpha=1.0)
                    ax.axvline(
                        s0,
                        color=f"{0.4 * (i + 1)}",
                        alpha=0.6,
                        linestyle="dashed",
                        linewidth=1,
                    )
                    ax.axvline(
                        s1,
                        color=f"{0.4 * (i + 1)}",
                        alpha=0.6,
                        linestyle="dashed",
                        linewidth=1,
                    )
            ax.plot(
                s[p].loc[s[nam.max(p)] == True],
                linestyle="None",
                lw=10,
                color="green",
                marker="v",
            )
            ax.plot(
                s[p].loc[s[nam.min(p)] == True],
                linestyle="None",
                lw=10,
                color="red",
                marker="^",
            )
    P.adjust((0.08, 0.95), (0.12, 0.95), H=0.2)
    return P.get()
