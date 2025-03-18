"""
Agent 2D trajectory-related plotting
"""

import copy

import numpy as np
from matplotlib import patches
from matplotlib import pyplot as plt

from .. import plot, reg, util, funcs
from ..util import nam

__all__ = [
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
    d, unit="mm", mode="default", title=None, single_color=False, time_range=None, **kwargs
):
    df=d.load_traj(mode=mode)
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
    unit="mm",
    name=None,
    subfolder="trajectories",
    range=None,
    mode="default",
    single_color=False,
    **kwargs,
):
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


def ax_conf_kws(kws, trange, Ndatasets, Nrows, i=0, ylab=None, ylim=None, xlim=None):
    conf_kws = {
        "ylab": kws.ylab if ylab is None else ylab,
        "ylim": kws.ylim if ylim is None else ylim,
        "xlim": (0, trange[-1]) if xlim is None else xlim,
        "xlab": r"time $(sec)$",
        "xvis": True if i == Nrows - 1 else False,
    }

    leg_kws = {
        "leg_loc": "upper right",
        "leg_handles": [
            patches.Patch(color=col, label=l)
            for l, col in zip(kws.labels, kws.chunk_cols)
        ],
        "leg_labels": kws.labels,
        "legfontsize": 15,
    }

    return {**conf_kws, **leg_kws}


def track_annotated(
    epoch="stride",
    a=None,
    dt=0.1,
    a2plot=None,
    ylab=None,
    ylim=None,
    xlim=None,
    slice=None,
    agent_idx=0,
    agent_id=None,
    subfolder="tracks",
    moving_average_interval=None,
    epoch_boundaries=True,
    show_extrema=True,
    min_amp=None,
    **kwargs,
):
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

    def stride_epochs(a, ax):
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

    def turn_epochs(a, ax):
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

    leg_kws = {
        "leg_loc": "upper right",
        "leg_handles": [
            patches.Patch(color=col, label=l)
            for l, col in zip(kws.labels, kws.chunk_cols)
        ],
        "leg_labels": kws.labels,
        "legfontsize": 15,
    }

    P.conf_ax(0, **conf_kws, **leg_kws)
    return P.get()


def annotated_strideplot(**kwargs):
    return track_annotated(epoch="stride", **kwargs)


def annotated_turnplot(**kwargs):
    return track_annotated(epoch="turn", **kwargs)


def track_annotated_data(
    name=None,
    subfolder="tracks",
    epoch="stride",
    a2plot_k=None,
    agent_idx=[3, 4, 5, 6],
    dur=1,
    **kwargs,
):
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

    def get_a(ss):
        return ss[apar].values

    if a2plot_k is not None:
        par, lab = reg.getPar(a2plot_k, to_return=["d", "symbol"])
    else:
        par, lab = None, None

    def get_a2plot(ss):
        return ss[par].values if par is not None else None

    def get_title(idx, c, e, l):
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
def annotated_strideplot_data(**kwargs):
    return track_annotated_data(epoch="stride", **kwargs)


@funcs.graph("turn track")
def annotated_turnplot_data(**kwargs):
    return track_annotated_data(epoch="turn", **kwargs)


@funcs.graph("marked strides")
def plot_marked_strides(
    agent_idx=0, agent_id=None, slice=[20, 40], subfolder="individuals", **kwargs
):
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
    handles = [
        patches.Patch(color=col, label=n)
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
    mode=["strides", "turns"],
    agent_idx=4,
    agent_id=None,
    slice=[0, 160],
    subfolder="individuals",
    **kwargs,
):
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

            handles = [
                patches.Patch(color=col, label=n) for n, col in zip(chunks, chunk_cols)
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
