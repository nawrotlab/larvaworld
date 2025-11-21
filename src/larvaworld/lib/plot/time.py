"""
Timeseries plotting
"""

from __future__ import annotations
from typing import Any, Optional, Sequence, Tuple

import numpy as np

from .. import plot, reg, util, funcs

__all__: list[str] = [
    "plot_ethogram",
    "plot_nengo_network",
    "timeplot",
    "timeplots",
    "plot_navigation_index",
    "plot_pathlength",
    "plot_dispersal",
]


@funcs.graph("ethogram", required={"dicts": ["chunk_dicts"]})
def plot_ethogram(subfolder: str = "timeplots", **kwargs: Any) -> Any:
    """
    Create ethogram showing behavioral bouts over time.

    Generates dual-panel time series showing runs/pauses and left/right turns
    as colored horizontal bars representing behavioral epochs.

    Args:
        subfolder: Subfolder for saving. Defaults to 'timeplots'
        **kwargs: Additional arguments passed to AutoPlot

    Returns:
        Plot output (figure object or None based on return_fig setting)

    Example:
        >>> fig = plot_ethogram(datasets=[d1, d2])
    """
    P = plot.AutoPlot(
        name="ethogram",
        subfolder=subfolder,
        build_kws={"Nrows": "Ndatasets", "Ncols": 2, "sharex": True},
        **kwargs,
    )
    Cbouts = {
        "lin": {"exec": "green", "pause": "red", "feedchain": "blue"},
        "ang": {"Lturn": "cyan", "Rturn": "orange"},
    }
    for i, (d, dlab) in enumerate(zip(P.datasets, P.labels)):
        c = d.config
        for j, dic in enumerate(d.chunk_dicts.values()):
            for k, (n, title) in enumerate(
                zip(
                    ["lin", "ang"],
                    [r"$\bf{runs & pauses}$", r"$\bf{left & right turns}$"],
                )
            ):
                idx = 2 * i + k
                # ax = P.axs[idx]

                for b, bcol in Cbouts[n].items():
                    try:
                        bbs = dic[b] * c.dt
                        b0s, b1s = bbs[:, 0], bbs[:, 1]

                        lines = [[(b0, j + 1), (b1, j + 1)] for b0, b1 in zip(b0s, b1s)]
                        from matplotlib import collections as mc

                        lc = mc.LineCollection(lines, colors=bcol, linewidths=2)
                        P.axs[idx].add_collection(lc)

                    except:
                        pass
                P.conf_ax(
                    idx,
                    xlab="time $(sec)$" if i == P.Ndatasets - 1 else None,
                    ylab=f"{dlab} Individuals $(idx)$" if k == 0 else None,
                    ylim=(0, c.N + 2),
                    xlim=(0, c.Nticks * d.dt),
                    title=title if i == 0 else None,
                )
                P.data_leg(
                    idx, labels=list(Cbouts[n].keys()), colors=list(Cbouts[n].values())
                )
    P.adjust((0.1, 0.95), (0.15, 0.92), 0.05, 0.05)
    return P.get()


@funcs.graph("nengo")
def plot_nengo_network(
    datasets: Sequence[Any],
    group: Optional[str] = None,
    probes: Optional[Sequence[str]] = None,
    same_plot: bool = False,
    subfolder: str = "nengo",
    **kwargs: Any,
) -> Any:
    """
    Plot Nengo neural network probe outputs over time.

    Creates multi-panel time series showing neural network activity from
    Nengo probes for different modules (crawler, turner, feeder, etc.).

    Args:
        datasets: List of datasets with Nengo probe data
        group: Probe group name ('anemotaxis', 'frequency', etc.). Required if probes is None
        probes: Individual probe names. Uses group probes if None
        same_plot: Plot all probes on same axis. Defaults to False
        subfolder: Subfolder for saving. Defaults to 'nengo'
        **kwargs: Additional arguments passed to AutoPlot

    Returns:
        Plot output (figure object or None based on return_fig setting)

    Example:
        >>> fig = plot_nengo_network(datasets=[d1], group='anemotaxis', same_plot=True)
    """
    probe_groups = {
        "anemotaxis": ["Ch", "LNa", "LNb", "Ha", "Hb", "B1", "B2", "Bend", "Hunch"],
        "frequency": ["linFrIn", "angFrIn", "linFr", "angFr"],
        "frequency_x3": ["linFrIn", "angFrIn", "feeFrIn", "linFr", "angFr", "feeFr"],
        "velocity": ["Vs", "linV", "angV"],
        "velocity_x3": ["Vs", "linV", "angV", "feeV"],
        "interference": ["Vs", "interference"],
        "crawler": ["linFrIn", "linFr", "linV"],
        "turner": ["angFrIn", "angFr", "angV"],
        "feeder": ["feeFrIn", "feeFr", "feeV"],
        "feeding": ["feeFrIn", "feeFr", "feeV", "f_cur", "f_suc"],
        "wind_effect_on_V": ["Bend", "Hunch", "linV", "angV"],
        "wind_effect_on_Fr": ["Bend", "Hunch", "linFr", "angFr"],
    }
    if group is not None:
        probes = probe_groups[group]
        name = f"{group}_network"
    elif probes is None:
        raise ValueError("Either a probe group or individual probes have to be defined")
    else:
        name = f"{probes[0]}_VS_{probes[1]}"
    N = len(probes)
    Cprobes = util.N_colors(N)

    Nds = len(datasets)
    Nids = np.max([len(d.agent_ids) for d in datasets])
    if same_plot:
        Nrows = Nds
        yMaxN = 8
    else:
        Nrows = N * Nds
        yMaxN = 3

    P = plot.AutoPlot(
        name=name,
        subfolder=subfolder,
        datasets=datasets,
        build_kws={"Nrows": Nrows, "Ncols": Nids, "sharex": True, "w": 30, "h": 15},
        **kwargs,
    )

    for i, d in enumerate(P.datasets):
        dics = d.load_dicts("nengo")
        for j, dic in enumerate(dics):
            for k, (p, c) in enumerate(zip(probes, Cprobes)):
                Nrow = i if same_plot else i * P.Ndatasets + k
                idx = j + Nrow * Nids
                y = np.array(dic[p])
                dim = y.shape[1]
                if dim == 1:
                    P.axs[idx].plot(P.trange(), y, color=c, label=p)
                else:
                    for jj in range(dim):
                        P.axs[idx].plot(P.trange(), y[:, jj], label=f"{p}_{jj}")
                P.conf_ax(
                    idx,
                    xlab=r"time $min$" if Nrow == Nrows - 1 else None,
                    ylab="activity" if j == 0 else None,
                    yticks=[] if j != 0 else None,
                    yticklabels=[] if j != 0 else None,
                    yMaxN=yMaxN,
                    leg_loc="upper right",
                )
    P.adjust((0.1, 0.95), (0.1, 0.95), 0.01, 0.05)
    return P.get()


@funcs.graph("timeplot", required={"ks": []})
def timeplot(
    ks: Sequence[str] = [],
    pars: Sequence[str] = [],
    name: Optional[str] = None,
    same_plot: bool = True,
    individuals: bool = False,
    table: Optional[Any] = None,
    unit: str = "sec",
    absolute: bool = True,
    show_legend: bool = True,
    show_first: bool = False,
    subfolder: str = "timeplots",
    legend_loc: str = "upper left",
    leg_fontsize: int = 15,
    figsize: Tuple[float, float] = (7.5, 5),
    **kwargs: Any,
) -> Any:
    """
    Create time series plot for one or more parameters.

    Generates single-panel time series with quantile bands showing parameter
    evolution over time, with optional individual trajectories.

    Args:
        ks: Parameter shortcut keys. Required if pars is empty
        pars: Direct parameter names. Uses ks if empty
        name: Plot name for saving. Auto-generated if None
        same_plot: Plot all parameters on same axes. Defaults to True
        individuals: Show individual trajectories. Defaults to False
        table: Table data for custom x-axis. Defaults to None
        unit: Time unit ('sec', 'min', 'hour'). Defaults to 'sec'
        absolute: Use absolute values. Defaults to True
        show_legend: Show dataset legend. Defaults to True
        show_first: Highlight first individual. Defaults to False
        subfolder: Subfolder for saving. Defaults to 'timeplots'
        legend_loc: Legend location. Defaults to 'upper left'
        leg_fontsize: Legend font size. Defaults to 15
        figsize: Figure size. Defaults to (7.5, 5)
        **kwargs: Additional arguments passed to AutoPlot

    Returns:
        Plot output (figure object or None based on return_fig setting)

    Example:
        >>> fig = timeplot(ks=['v', 'fov'], datasets=[d1, d2], individuals=True)
    """
    unit_coefs = {"sec": 1, "min": 1 / 60, "hour": 1 / 60 / 60}
    if len(pars) == 0:
        if len(ks) == 0:
            raise ValueError("Either parameter names or shortcuts must be provided")
        else:
            pars, symbols, ylabs, ylims = reg.getPar(
                ks, to_return=["d", "disp", "l", "lim"]
            )

    else:
        symbols = pars
        ylabs = pars
        ylims = [None] * len(pars)
    N = len(pars)
    cols = ["grey"] if N == 1 else util.N_colors(N)
    if not same_plot:
        raise NotImplementedError
    if name is None:
        if N == 1:
            name = f"{pars[0]}"
        elif N == 2:
            name = f"{pars[0]}_VS_{pars[1]}"
        else:
            name = f"{N}_pars"
    P = plot.AutoPlot(name=name, subfolder=subfolder, figsize=figsize, **kwargs)

    ax = P.axs[0]
    counter = 0
    for p, symbol, ylab, ylim, c in zip(pars, symbols, ylabs, ylims, cols):
        P.conf_ax(
            xlab=f"time, ${unit}$" if table is None else "timesteps",
            ylab=ylab,
            ylim=ylim,
            yMaxN=4,
        )
        for d, d_col, d_lab in zip(P.datasets, P.colors, P.labels):
            if P.Ndatasets > 1:
                c = d_col
            try:
                dc = d.get_par(p)
                if absolute:
                    dc = dc.abs()
                dc_m = dc.groupby(level="Step").quantile(q=0.5)
                Nticks = len(dc_m)
                x = (
                    np.linspace(0, int(Nticks / d.fr) * unit_coefs[unit], Nticks)
                    if table is None
                    else np.arange(Nticks)
                )
                ax.set_xlim([x[0], x[-1]])

                if individuals:
                    for id in dc.index.get_level_values("AgentID"):
                        dc_single = dc.xs(id, level="AgentID")
                        ax.plot(x, dc_single, color=c, linewidth=1)
                    ax.plot(x, dc_m, color=c, linewidth=2)
                else:
                    plot.plot_quantiles(
                        df=dc, x=x, axis=ax, color=c, label=symbol, linewidth=2
                    )
                    if show_first:
                        cc = "red" if P.Ndatasets == 1 else c
                        dc0 = dc.xs(
                            dc.index.get_level_values("AgentID")[0], level="AgentID"
                        )
                        ax.plot(x, dc0, color=cc, linestyle="dashed", linewidth=1)
                counter += 1
            except:
                pass
    if counter == 0:
        raise ValueError("None of the parameters exist in any dataset")
    if N > 1:
        ax.legend()
    if P.Ndatasets > 1 and show_legend:
        P.data_leg(0, loc=legend_loc, fontsize=leg_fontsize)
    P.adjust((0.15, 0.95), (0.15, 0.95))
    return P.get()


@funcs.graph("timeplots", required={"ks": []})
def timeplots(
    ks: Sequence[str],
    subfolder: str = "timeplots",
    name: Optional[str] = None,
    unit: str = "sec",
    xlim: Optional[Sequence[float]] = None,
    individuals: bool = False,
    absolute: bool = False,
    show_first: bool = False,
    **kwargs: Any,
) -> Any:
    """
    Create multi-panel time series plots.

    Generates stacked time series panels (one per parameter) showing
    quantile bands for multiple parameters over time.

    Args:
        ks: Parameter shortcut keys to plot
        subfolder: Subfolder for saving. Defaults to 'timeplots'
        name: Plot name for saving. Auto-generated if None
        unit: Time unit ('sec', 'min', 'hour'). Defaults to 'sec'
        xlim: X-axis limits. Defaults to None
        individuals: Show individual trajectories. Defaults to False
        absolute: Use absolute values. Defaults to False
        show_first: Highlight first individual. Defaults to False
        **kwargs: Additional arguments passed to AutoPlot

    Returns:
        Plot output (figure object or None based on return_fig setting)

    Example:
        >>> fig = timeplots(ks=['v', 'fov', 'b'], datasets=[d1, d2], unit='min')
    """
    Nks = len(ks)
    if name is None:
        name = f"timeplots_x{Nks}"
    P = plot.AutoPlot(
        name=name,
        subfolder=subfolder,
        build_kws={"Nrows": Nks, "sharex": True, "w": 15, "h": 5},
        **kwargs,
    )

    for i, k in enumerate(ks):
        P.plot_quantiles(
            k=k,
            idx=i,
            unit=unit,
            xvis=True if i == Nks - 1 else False,
            xlim=xlim,
            individuals=individuals,
            absolute=absolute,
            show_first=show_first,
        )
    P.adjust((0.1, 0.95), (0.15, 0.95), H=0.05)
    P.fig.align_ylabels(P.axs[:])
    return P.get()


@funcs.graph("navigation index", required={"traj": ["default"]})
def plot_navigation_index(subfolder: str = "source", **kwargs: Any) -> Any:
    """
    Plot navigation index time series.

    Creates time series showing navigation efficiency index over time,
    measuring directed movement toward source/target.

    Args:
        subfolder: Subfolder for saving. Defaults to 'source'
        **kwargs: Additional arguments passed to AutoPlot

    Returns:
        Plot output (figure object or None based on return_fig setting)

    Example:
        >>> fig = plot_navigation_index(datasets=[d1, d2])
    """
    P = plot.AutoPlot(
        name="nav_index",
        subfolder=subfolder,
        build_kws={"Nrows": 2, "w": 20, "h": 10, "sharex": True, "sharey": True},
        **kwargs,
    )

    for l, d, c in P.data_palette:
        Nsec = int(P.Nticks * d.config.dt)
        trange = np.linspace(0, Nsec, P.Nticks)
        vxs, vys = [], []
        for s0 in d.traj_xy_data_byID.values():
            dxy = np.diff(s0, axis=0)
            rads = np.arctan2(dxy[:, 1], dxy[:, 0])
            rads = np.insert(rads, 0, 0)
            vxs.append(np.cos(rads))
            vys.append(-np.sin(rads))
        P.axs[0].plot(trange, np.nanmean(np.array(vxs), axis=0), color=c, label=l)
        P.axs[1].plot(trange, np.nanmean(np.array(vys), axis=0), color=c, label=l)
    P.adjust((0.1, 0.95), (0.2, 0.98), H=0.15)
    P.conf_ax(0, ylab="X index", leg_loc="upper right")
    P.conf_ax(1, xlab="time (sec)", ylab="Y index", xlim=[0, Nsec], ylim=[-1.0, 1.0])
    P.axs[0].axhline(0.0, color="green", alpha=0.5, linestyle="dashed", linewidth=1)
    P.axs[1].axhline(0.0, color="green", alpha=0.5, linestyle="dashed", linewidth=1)
    return P.get()


@funcs.graph("pathlength", required={"ks": ["cum_d", "cum_sd"]})
def plot_pathlength(scaled: bool = False, **kwargs: Any) -> Any:
    """
    Plot cumulative path length over time.

    Creates time series showing total distance traveled (scaled or absolute)
    accumulated over the experiment duration.

    Args:
        scaled: Use body-length-scaled distance. Defaults to False
        **kwargs: Additional arguments passed to timeplots

    Returns:
        Plot output (figure object or None based on return_fig setting)

    Example:
        >>> fig = plot_pathlength(datasets=[d1, d2], scaled=True)
    """
    k = "cum_sd" if scaled else "cum_d"
    return timeplots(ks=[k], **kwargs)


@funcs.graph("dispersal")
def plot_dispersal(
    range: Tuple[int, int] = (0, 40), scaled: bool = False, **kwargs: Any
) -> Any:
    """
    Plot dispersal metric over time.

    Creates time series showing spatial dispersal from origin over
    specified time range (scaled or absolute distance).

    Args:
        range: Time range (start, end) in minutes. Defaults to (0, 40)
        scaled: Use body-length-scaled distance. Defaults to False
        **kwargs: Additional arguments passed to timeplots

    Returns:
        Plot output (figure object or None based on return_fig setting)

    Example:
        >>> fig = plot_dispersal(datasets=[d1, d2], range=(0, 60), scaled=True)
    """
    t0, t1 = range
    k = f"dsp_{int(t0)}_{int(t1)}"
    if scaled:
        k = f"s{k}"
    return timeplots(name=reg.getPar(k), ks=[k], xlim=range, **kwargs)
