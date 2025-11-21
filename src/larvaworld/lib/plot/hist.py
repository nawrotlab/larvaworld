"""
Histograms
"""

from __future__ import annotations
from typing import Any, Optional, Sequence

import itertools

import numpy as np
import pandas as pd
import seaborn as sns

from .. import plot, reg, funcs
from ..model import moduleDB as MD

__all__: list[str] = [
    "module_endpoint_hists",
    "plot_ang_pars",
    "plot_distros",
    "plot_crawl_pars",
    "plot_turn_amp_VS_Ypos",
    "plot_turn_duration",
    "plot_bout_ang_pars",
    "plot_endpoint_scatter",
    "plot_turns",
    "plot_endpoint_hist",
    "plot_endpoint_box",
    "plot_endpoint_params",
]


@funcs.graph("module hists")
def module_endpoint_hists(
    e: Any,
    mkey: str = "crawler",
    mode: str = "realistic",
    Nbins: Optional[int] = None,
    show_median: bool = True,
    **kwargs: Any,
) -> Any:
    """
    Create histograms of module endpoint parameters.

    Generates multi-panel histogram figure showing distribution of module
    parameters at experiment endpoints, useful for model configuration analysis.

    Args:
        e: Endpoint data containing parameter values
        mkey: Module key identifier. Defaults to 'crawler'
        mode: Parameter mode ('realistic', etc.). Defaults to 'realistic'
        Nbins: Number of histogram bins. Auto-calculated if None
        show_median: Whether to show median line. Defaults to True
        **kwargs: Additional arguments passed to AutoBasePlot

    Returns:
        Plot output (figure object or None based on return_fig setting)

    Example:
        >>> fig = module_endpoint_hists(endpoint_data, mkey='crawler', Nbins=20)
    """
    if Nbins is None:
        Nbins = int(e.index.values.shape[0] / 10)

    pars = MD.module_pars(mID=mkey, mode=mode)
    pars = reg.sample_ps(pars, e)
    N = len(pars)

    P = plot.AutoBasePlot(
        name=f"{mkey}_endpoint_hists",
        build_kws={"Ncols": N, "w": 7, "h": 6, "sharey": True},
        **kwargs,
    )
    for i, par in enumerate(pars):
        vs = e[par].dropna().values
        plab, psym = reg.getPar(d=par, to_return=["label", "symbol"])
        P.axs[i].hist(vs, bins=Nbins)
        P.conf_ax(
            i,
            xlab=plab,
            ylab="# larvae" if i == 0 else None,
            xMaxN=3,
            xlabelfontsize=18,
            xticklabelsize=18,
            yvis=False if i != 0 else True,
        )

        if show_median:
            v_mu = np.median(vs)
            text = f"{psym} = {np.round(v_mu, 2)}"
            P.axs[i].axvline(
                v_mu, color="red", alpha=1, linestyle="dashed", linewidth=3
            )
            P.axs[i].annotate(
                text,
                rotation=0,
                fontsize=15,
                va="center",
                ha="left",
                xy=(0.55, 0.8),
                xycoords="axes fraction",
            )
    P.adjust((0.2, 0.9), (0.2, 0.9), 0.01)
    return P.get()


@funcs.graph(
    "angular pars", required={"ks": ["b", "bv", "ba", "fov", "foa", "rov", "roa"]}
)
def plot_ang_pars(
    absolute: bool = False,
    rad2deg: bool = True,
    include_rear: bool = False,
    name: str = "ang_pars",
    half_circles: bool = True,
    kde: bool = True,
    Npars: int = 5,
    Nbins: int = 100,
    **kwargs: Any,
) -> Any:
    """
    Create histograms of angular parameters.

    Generates multi-panel histograms showing distributions of bend, angular
    velocity, angular acceleration, and optionally rear body angles.

    Args:
        absolute: Use absolute values. Defaults to False
        rad2deg: Convert radians to degrees. Defaults to True
        include_rear: Include rear body parameters. Defaults to False
        name: Plot name for saving. Defaults to 'ang_pars'
        half_circles: Use half-circle plots. Defaults to True
        kde: Include kernel density estimate. Defaults to True
        Npars: Number of parameters (3 or 5). Defaults to 5
        Nbins: Number of histogram bins. Defaults to 100
        **kwargs: Additional arguments passed to plot_step_params

    Returns:
        Plot output (figure object or None based on return_fig setting)

    Example:
        >>> fig = plot_ang_pars(datasets=[d1, d2], include_rear=True, Npars=5)
    """
    if Npars == 5:
        ks = ["b", "bv", "ba", "fov", "foa"]
        rs = [100, 200, 2000, 200, 2000]
    elif Npars == 3:
        ks = ["b", "bv", "fov"]
        rs = [100, 200, 200]
    else:
        raise ValueError("3 or 5 pars allowed")

    if include_rear:
        ks += ["rov", "roa"]
        rs += [200, 2000]

    if not rad2deg:
        rs = np.deg2rad(rs).tolist()

    return plot_step_params(
        type="hist",
        ks=ks,
        name=name,
        Ncols="Nks",
        plot_kws={"kde": kde, "half_circles": half_circles, "nbins": Nbins},
        ranges=rs,
        absolute=absolute,
        rad2deg=rad2deg,
        **kwargs,
    )


# ks=['v', 'a','sv', 'sa', 'b', 'bv', 'ba', 'fov', 'foa']
@funcs.graph("distros", required={"ks": []})
def plot_distros(
    name: Optional[str] = None,
    ks: Sequence[str] = ("v", "a", "sv", "sa", "b", "bv", "ba", "fov", "foa"),
    mode: str = "hist",
    half_circles: bool = True,
    annotation: bool = False,
    target_only: Optional[str] = None,
    show_ns: bool = False,
    subfolder: str = "distro",
    Nbins: int = 100,
    **kwargs: Any,
) -> Any:
    """
    Create distribution plots for kinematic parameters.

    Generates multi-panel histograms or boxplots showing distributions of
    velocity, acceleration, and angular parameters across datasets.

    Args:
        name: Plot name for saving. Auto-generated if None
        ks: Parameter keys to plot. Defaults to velocity and angular parameters
        mode: Plot type ('hist' or 'box'). Defaults to 'hist'
        half_circles: Use half-circle layout. Defaults to True
        annotation: Add statistical annotations. Defaults to False
        target_only: Compare only to target dataset. Defaults to None
        show_ns: Show non-significant comparisons. Defaults to False
        subfolder: Subfolder for saving. Defaults to 'distro'
        Nbins: Number of histogram bins. Defaults to 100
        **kwargs: Additional arguments passed to AutoPlot

    Returns:
        Plot output (figure object or None based on return_fig setting)

    Example:
        >>> fig = plot_distros(datasets=[d1, d2], mode='hist', ks=['v', 'a', 'fov'])
    """
    Nps = len(ks)
    if name is None:
        name = f"distros_{mode}_{Nps}"
    legloc = "upper left" if half_circles else "upper right"
    build_kws = {"N": "Nks", "wh": 8}
    if mode == "box":
        build_kws["sharex"] = True
    elif mode == "hist":
        build_kws["sharey"] = True
    P = plot.AutoPlot(
        ks=ks, name=name, subfolder=subfolder, build_kws=build_kws, **kwargs
    )
    palette = dict(zip(P.labels, P.colors))
    Ddata = {}
    lims = {}
    parlabs = {}
    for sh, par in zip(ks, P.pars):
        Ddata[par] = {}
        vs = []
        for d, l in zip(P.datasets, P.labels):
            x = d.get_par(par).dropna().values
            Ddata[par][l] = x
            vs.append(x)
        vvs = np.hstack(vs)
        vmin, vmax = np.quantile(vvs, 0.005), np.quantile(vvs, 0.995)
        lims[par] = (vmin, vmax)
        parlabs[par] = reg.getPar(sh, to_return="l")
    for i, (par, dic) in enumerate(Ddata.items()):
        if mode == "box":
            dfs = []
            for l, x in dic.items():
                df = pd.DataFrame(x, columns=[par])
                df["DatasetID"] = l
                dfs.append(df)
            df0 = pd.concat(dfs)
            kws = {
                "x": "DatasetID",
                "y": par,
                "palette": palette,
                "hue": None,
                "data": df0,
                "ax": P.axs[i],
                "width": 0.8,
                "fliersize": 3,
                "whis": 1.5,
                "linewidth": None,
            }
            g1 = sns.boxplot(**kws)  # RUN PLOT
            try:
                g1.get_legend().remove()
            except:
                pass
            if annotation:
                try:
                    plot.annotate_plot(
                        box=g1, show_ns=show_ns, target_only=target_only, **kws
                    )
                except:
                    pass

            P.conf_ax(
                i, xticklabelrotation=30, ylab=parlabs[par], yMaxN=4, ylim=lims[par]
            )

        else:
            vmin, vmax = lims[par]
            bins = np.linspace(vmin, vmax, Nbins)
            dic = {}
            for l, x in dic.items():
                ws = np.ones_like(x) / float(len(x))
                dic[l] = {
                    "weights": ws,
                    "color": palette[l],
                    "label": l,
                    "x": x,
                    "alpha": 0.6,
                }
                P.axs[i].hist(bins=bins, **dic[l])
            P.conf_ax(
                i,
                ylab="probability",
                yvis=True if i % P.Ncols == 0 else False,
                xlab=parlabs[par],
                yMaxN=3,
                xMaxN=5,
                leg_loc=legloc,
            )

    if mode == "box":
        P.adjust((0.15, 0.95), (0.15, 0.95), 0.3, 0.01)
    else:
        P.data_leg(0, loc="upper left" if half_circles else "upper right")
        P.adjust((0.15, 0.95), (0.15, 0.95), 0.01, 0.1)
    return P.get()


@funcs.graph("crawl pars", required={"ks": ["str_N", "run_tr", "cum_sd"]})
def plot_crawl_pars(kde: bool = True, **kwargs: Any) -> Any:
    """
    Create histograms of crawling parameters.

    Generates histograms for stride count, run duration, and cumulative
    distance - key metrics for crawling locomotion analysis.

    Args:
        kde: Include kernel density estimate. Defaults to True
        **kwargs: Additional arguments passed to plot_endpoint_params

    Returns:
        Plot output (figure object or None based on return_fig setting)

    Example:
        >>> fig = plot_crawl_pars(datasets=[d1, d2], kde=True)
    """
    return plot_endpoint_params(
        type="hist",
        ks=["str_N", "run_tr", "cum_sd"],
        name="crawl_pars",
        plot_kws={"kde": kde},
        **kwargs,
    )


@funcs.graph("turn amplitude VS Y pos", required={"ks": ["tur_y0"]})
def plot_turn_amp_VS_Ypos(**kwargs: Any) -> Any:
    """
    Plot turn amplitude as function of Y position.

    Creates histogram showing distribution of body Y-position during turns,
    useful for analyzing body curvature patterns.

    Args:
        **kwargs: Additional arguments passed to plot_turn_amp

    Returns:
        Plot output (figure object or None based on return_fig setting)

    Example:
        >>> fig = plot_turn_amp_VS_Ypos(datasets=[d1, d2])
    """
    return plot_turn_amp(k="tur_y0", **kwargs)


@funcs.graph("turn duration", required={"ks": ["tur_t"]})
def plot_turn_duration(absolute: bool = True, **kwargs: Any) -> Any:
    """
    Plot turn duration distribution.

    Creates histogram showing distribution of turn durations across datasets.

    Args:
        absolute: Use absolute durations. Defaults to True
        **kwargs: Additional arguments passed to plot_turn_amp

    Returns:
        Plot output (figure object or None based on return_fig setting)

    Example:
        >>> fig = plot_turn_duration(datasets=[d1, d2], absolute=True)
    """
    return plot_turn_amp(k="tur_t", absolute=absolute, **kwargs)


def plot_turn_amp(
    name: Optional[str] = None,
    k: str = "tur_t",
    ref_angle: Optional[float] = None,
    subfolder: str = "turn",
    absolute: bool = True,
    **kwargs: Any,
) -> Any:
    if name is None:
        nn = "turn_amp" if ref_angle is None else "rel_turn_angle"
        name = f"{nn}_VS_{k}_scatter"

    P = plot.AutoPlot(name=name, subfolder=subfolder, **kwargs)
    ypar, ylab, ylim = reg.getPar("tur_fou", to_return=["d", "l", "lim"])

    if ref_angle is not None:
        A0 = float(ref_angle)
        p_ref = reg.getPar(["tur_fo0", "tur_fo1"])
        ys = []
        ylab = r"$\Delta\theta_{bearing} (deg)$"
        cumylab = r"$\bar{\Delta\theta}_{bearing} (deg)$"
        for d in P.datasets:
            y0 = d.get_par(p_ref[0]).dropna().values.flatten() - A0
            y1 = d.get_par(p_ref[1]).dropna().values.flatten() - A0
            y0 %= 360
            y1 %= 360
            y0[y0 > 180] -= 360
            y1[y1 > 180] -= 360
            y = np.abs(y0) - np.abs(y1)
            ys.append(y)

    else:
        cumylab = r"$\bar{\Delta\theta}_{or} (deg)$"
        ys = [d.get_par(ypar).dropna().values.flatten() for d in P.datasets]
        if absolute:
            ys = [np.abs(y) for y in ys]
    xpar, xlab = reg.getPar(k, to_return=["d", "l"])
    xs = [d.get_par(xpar).dropna().values.flatten() for d in P.datasets]

    ax = P.axs[0]
    for x, y, l, c in zip(xs, ys, P.labels, P.colors):
        ax.scatter(x=x, y=y, marker="o", s=5.0, color=c, alpha=0.5)
        m, k = np.polyfit(x, y, 1)
        ax.plot(x, m * x + k, linewidth=4, color=c, label=l)
        P.conf_ax(xlab=xlab, ylab=ylab, ylim=ylim, yMaxN=4, leg_loc="upper left")
        P.adjust((0.15, 0.95), (0.1, 0.95), 0.01)
    return P.get()


@funcs.graph("angular/epoch", required={"ks": ["bv", "fov", "rov", "ba", "foa", "roa"]})
def plot_bout_ang_pars(
    name: str = "bout_ang_pars",
    absolute: bool = True,
    include_rear: bool = True,
    subfolder: str = "turn",
    **kwargs: Any,
) -> Any:
    """
    Plot angular parameters during behavioral bouts.

    Creates histograms of angular velocity and acceleration for front and
    optionally rear body segments during runs vs pauses.

    Args:
        name: Plot name for saving. Defaults to 'bout_ang_pars'
        absolute: Use absolute values. Defaults to True
        include_rear: Include rear body parameters. Defaults to True
        subfolder: Subfolder for saving. Defaults to 'turn'
        **kwargs: Additional arguments passed to AutoPlot

    Returns:
        Plot output (figure object or None based on return_fig setting)

    Example:
        >>> fig = plot_bout_ang_pars(datasets=[d1, d2], include_rear=True)
    """
    ks = (
        ["bv", "fov", "rov", "ba", "foa", "roa"]
        if include_rear
        else ["bv", "fov", "ba", "foa"]
    )
    Nps = len(ks)
    P = plot.AutoPlot(
        name=name,
        subfolder=subfolder,
        build_kws={"N": Nps, "Nrows": 2, "wh": 7, "sharey": True},
        **kwargs,
    )
    ranges = [250, 250, 50, 2000, 2000, 500] if include_rear else [200, 200, 2000, 2000]
    chunks = ["run", "pause"]
    chunk_cols = ["green", "purple"]
    for i, k in enumerate(ks):
        pdisp, par = reg.getPar(k, to_return=["disp", "d"])
        r = ranges[i]
        xlim = (r0, r1) = (0, r) if absolute else (-r, r)
        bins = np.linspace(r0, r1, 200)

        P.conf_ax(i, xlab=pdisp, xlim=xlim, yMaxN=3)
        for d, l in zip(P.datasets, P.labels):
            vs = [d.get_chunk_par(chunk=c, par=par) for c in chunks]
            if absolute:
                vs = [np.abs(v) for v in vs]
            plot.prob_hist(
                vs,
                chunk_cols,
                chunks,
                ax=P.axs[i],
                bins=bins,
                alpha=1.0,
                histtype="step",
                linewidth=2,
            )

    P.conf_ax(0, ylab="probability", ylim=[0, 0.04], leg_loc="upper left")
    P.conf_ax(ylab="probability", leg_loc="upper left")
    P.adjust((0.1, 0.95), (0.1, 0.9), 0.1, 0.3)
    return P.get()


@funcs.graph("endpoint pars (scatter)", required={"ks": []})
def plot_endpoint_scatter(
    subfolder: str = "endpoint", ks: Sequence[str] | None = None, **kwargs: Any
) -> Any:
    """
    Create scatter plots of endpoint parameter pairs.

    Generates pairwise scatter plots showing correlations between endpoint
    parameters across datasets.

    Args:
        subfolder: Subfolder for saving. Defaults to 'endpoint'
        ks: Parameter keys to plot. Required
        **kwargs: Additional arguments passed to AutoPlot

    Returns:
        Plot output (figure object or None based on return_fig setting)

    Example:
        >>> fig = plot_endpoint_scatter(datasets=[d1, d2], ks=['cum_sd', 'run_tr', 'pau_tr'])
    """
    pairs = list(itertools.combinations(ks, 2))
    Npairs = len(pairs)
    if Npairs % 3 == 0:
        Nx, Ny = 3, int(Npairs / 3)
    elif Npairs % 2 == 0:
        Nx, Ny = 2, int(Npairs / 2)
    elif Npairs % 5 == 0:
        Nx, Ny = 5, int(Npairs / 5)
    else:
        Nx, Ny = Npairs, 1
    if Nx * Ny > 1:
        name = "endpoint_scatterplot"
    else:
        name = f"{ks[1]}_vs_{ks[0]}"
    P = plot.AutoPlot(
        name=name,
        subfolder=subfolder,
        build_kws={"Nrows": Nx, "Ncols": Ny, "wh": 10},
        **kwargs,
    )
    for i, (p0, p1) in enumerate(pairs):
        pars, labs = reg.getPar([p0, p1], to_return=["d", "l"])

        v0_all = [d.endpoint_data[pars[0]].values for d in P.datasets]
        v1_all = [d.endpoint_data[pars[1]].values for d in P.datasets]
        r0, r1 = 0.9, 1.1
        v0_r = [np.min(np.array(v0_all)) * r0, np.max(np.array(v0_all)) * r1]
        v1_r = [np.min(np.array(v1_all)) * r0, np.max(np.array(v1_all)) * r1]

        for v0, v1, l, c in zip(v0_all, v1_all, P.labels, P.colors):
            P.axs[i].scatter(v0, v1, color=c, label=l)
        P.conf_ax(
            i,
            xlab=labs[0],
            ylab=labs[1],
            xlim=v0_r,
            ylim=v1_r,
            tickMath=(0, 0),
            title=f"{pars[1]}_vs_{pars[0]}",
            leg_loc="upper right",
        )

    return P.get()


@funcs.graph("turn amplitude", required={"ks": ["tur_fou"]})
def plot_turns(
    name: str = "turn_amplitude",
    absolute: bool = False,
    subfolder: str = "turn",
    **kwargs: Any,
) -> Any:
    """
    Plot turn amplitude distribution.

    Creates violin plots showing distribution of turn amplitudes (bearing changes)
    during turns across datasets.

    Args:
        name: Plot name for saving. Defaults to 'turn_amplitude'
        absolute: Use absolute amplitudes. Defaults to False
        subfolder: Subfolder for saving. Defaults to 'turn'
        **kwargs: Additional arguments passed to AutoPlot

    Returns:
        Plot output (figure object or None based on return_fig setting)

    Example:
        >>> fig = plot_turns(datasets=[d1, d2], absolute=False)
    """
    P = plot.AutoPlot(
        ks=["tur_fou"],
        ranges=[100],
        absolute=absolute,
        rad2deg=True,
        name=name,
        subfolder=subfolder,
        **kwargs,
    )
    P.plot_hist(par_legend=True, nbins=30, alpha=1.0, histtype="step")
    P.adjust((0.25, 0.95), (0.15, 0.92), 0.05, 0.005)
    return P.get()


@funcs.graph("endpoint hist", required={"ks": []})
def plot_endpoint_hist(**kwargs: Any) -> Any:
    """
    Create histograms of endpoint parameters.

    Wrapper function that generates histogram plots for experiment endpoint
    metrics across datasets.

    Args:
        **kwargs: Arguments passed to plot_endpoint_params

    Returns:
        Plot output (figure object or None based on return_fig setting)

    Example:
        >>> fig = plot_endpoint_hist(datasets=[d1, d2], ks=['cum_sd', 'run_tr'])
    """
    return plot_endpoint_params(type="hist", **kwargs)


@funcs.graph("endpoint box", required={"ks": []})
def plot_endpoint_box(**kwargs: Any) -> Any:
    """
    Create boxplots of endpoint parameters.

    Wrapper function that generates boxplot comparisons for experiment endpoint
    metrics across datasets.

    Args:
        **kwargs: Arguments passed to plot_endpoint_params

    Returns:
        Plot output (figure object or None based on return_fig setting)

    Example:
        >>> fig = plot_endpoint_box(datasets=[d1, d2], ks=['cum_sd', 'run_tr'])
    """
    return plot_endpoint_params(type="box", **kwargs)


def plot_endpoint_params(type: str, mode: str = "basic", **kwargs: Any) -> Any:
    """
    Create plots of endpoint parameters with specified type.

    General function for plotting endpoint parameters using different
    visualization types (histogram, boxplot, etc.).

    Args:
        type: Plot type ('hist', 'box', etc.)
        mode: Parameter mode. Defaults to 'basic'
        **kwargs: Additional arguments passed to plot_params

    Returns:
        Plot output (figure object or None based on return_fig setting)

    Example:
        >>> fig = plot_endpoint_params(type='hist', datasets=[d1, d2], ks=['cum_sd'])
    """
    return plot_params(key="end", type=type, mode=mode, **kwargs)


@funcs.graph("step hist", required={"ks": []})
def plot_step_hist(**kwargs: Any) -> Any:
    return plot_step_params(type="hist", **kwargs)


@funcs.graph("step box", required={"ks": []})
def plot_step_box(**kwargs: Any) -> Any:
    return plot_step_params(type="box", **kwargs)


def plot_step_params(
    type: str,
    ks: Sequence[str] = ("v", "a", "sv", "sa", "b", "bv", "ba", "fov", "foa"),
    **kwargs: Any,
) -> Any:
    return plot_params(key="step", type=type, ks=ks, **kwargs)


def plot_params(
    key: str,
    type: str,
    name: Optional[str] = None,
    mode: Optional[str] = None,
    ks: Optional[Sequence[str]] = None,
    Ncols: Optional[int] = None,
    plot_kws: dict = {},
    **kwargs: Any,
) -> Any:
    if name is None:
        name = f"{key}_{type}_{mode}"
    if type == "hist":
        sharex, sharey = False, True
        W, H = 0.1, 0.3
    elif type == "box":
        sharex, sharey = True, False
        W, H = 0.5, 0.15

    if "subfolder" not in kwargs:
        if key == "end":
            kwargs["subfolder"] = "endpoint"
            ks = plot.define_end_ks(ks, mode)
        elif key == "step":
            kwargs["subfolder"] = "distro"
        else:
            raise
    if key == "end":
        ks = plot.define_end_ks(ks, mode)

    P = plot.AutoPlot(
        ks=ks,
        key=key,
        name=name,
        build_kws={
            "N": "Nks",
            "Ncols": Ncols,
            "wh": 7,
            "sharex": sharex,
            "sharey": sharey,
        },
        **kwargs,
    )
    if type == "hist":
        P.plot_hist(**plot_kws)
    elif type == "box":
        P.boxplots(**plot_kws)
    P.conf_fig(
        align=True, adjust_kws={"LR": (0.1, 0.95), "BT": (0.15, 0.9), "W": W, "H": H}
    )
    return P.get()
