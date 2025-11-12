"""
Behaviorl-epoch-related plotting
"""

from __future__ import annotations
from typing import Any, Optional, Sequence

import copy

from .. import plot, reg, util, funcs

__all__: list[str] = [
    "plot_single_bout",
    "plot_bouts",
    # 'plot_stridesNpauses',
]


def plot_single_bout(
    x0: Sequence[float],
    bout: str,
    color: str,
    label: str,
    ax: Any,
    fit_dic: Optional[dict] = None,
    plot_fits: str | Sequence[str] = "best",
    marker: str = ".",
    legend_outside: bool = False,
    xlabel: str = "time (sec)",
    xlim: Optional[Sequence[float]] = None,
    **kwargs: Any,
) -> None:
    """
    Plot single behavioral bout duration distribution with fitted curves.

    Creates log-log plot of bout duration probability distribution with
    optional fitted distribution curves (powerlaw, exponential, etc.).

    Args:
        x0: Bout duration data
        bout: Bout type label ('pauses', 'runs', etc.)
        color: Color for data points
        label: Legend label for this dataset
        ax: Matplotlib axes to plot on
        fit_dic: Pre-computed fit results. If None, computes fits
        plot_fits: Which fits to show ('best', 'all', or list). Defaults to 'best'
        marker: Marker style for data points. Defaults to '.'
        legend_outside: Place legend outside plot. Defaults to False
        xlabel: X-axis label. Defaults to 'time (sec)'
        xlim: X-axis limits. Defaults to None
        **kwargs: Additional arguments passed to fit_bout_distros

    Example:
        >>> plot_single_bout(pause_data, 'pauses', 'red', 'Control', ax)
    """
    distro_ls = [
        "powerlaw",
        "exponential",
        "lognormal",
        "lognorm-pow",
        "levy",
        "normal",
        "uniform",
    ]
    distro_cs = ["c", "g", "m", "k", "orange", "brown", "purple"]
    lws = [2] * len(distro_ls)

    if fit_dic is None:
        fit_dic = reg.fit_bout_distros(x0, bout=bout, **kwargs)
    idx_Kmax = fit_dic["idx_Kmax"]
    xrange, du2, c2, y = fit_dic["values"]
    lws[idx_Kmax] = 4

    ax.loglog(xrange, y, marker, color=color, alpha=0.7, label=label)
    ax.set_title(bout)
    ax.set_xlabel(xlabel)
    ax.set_ylim([10**-3.5, 10**0.2])
    if xlim is not None:
        ax.set_xlim(xlim)
    distro_ls0, distro_cs0 = [], []
    for z, (l, col, ddf) in enumerate(zip(distro_ls, distro_cs, fit_dic["cdfs"])):
        if ddf is None:
            continue
        else:
            ddf /= ddf[0]
        if plot_fits == "best" and z == idx_Kmax:
            cc = color
        elif plot_fits == "all" or l in plot_fits:
            distro_ls0.append(l)
            distro_cs0.append(col)
            cc = col
        else:
            continue
        ax.loglog(xrange, ddf, color=cc, lw=lws[z], label=l)
    if len(distro_ls0) > 1:
        if legend_outside:
            plot.dataset_legend(
                distro_ls0,
                distro_cs0,
                ax=ax,
                loc="center left",
                fontsize=25,
                anchor=(1.0, 0.5),
            )
        else:
            plot.dataset_legend(
                distro_ls0, distro_cs0, ax=ax, loc="lower left", fontsize=15
            )


@funcs.graph("sample_epochs", required={"dicts": ["pooled_epochs"]})
def plot_sample_bouts(mID: str, d: Any, **kwargs: Any) -> Any:
    d2 = copy.deepcopy(d)
    d2.config.dir = None
    d2.fitted_epochs = d.generate_pooled_epochs(mID=mID)
    kws = {
        "datasets": util.ItemList([d, d2]),
        "labels": ["experiment", "model"],
        "colors": ["red", "blue"],
    }
    return plot_bouts(**kws, **kwargs)


@funcs.graph("epochs", required={"dicts": ["fitted_epochs"]})
def plot_bouts(
    name: Optional[str] = None,
    plot_fits: str | Sequence[str] = "",
    print_fits: bool = False,
    turns: bool = False,
    stridechain_duration: bool = False,
    legend_outside: bool = False,
    **kwargs: Any,
) -> Any:
    """
    Plot behavioral bout distributions across datasets.

    Creates two-panel plots showing distributions of bout durations (runs/pauses
    or turn durations/amplitudes) with fitted distribution curves.

    Args:
        name: Plot name for saving. Auto-generated if None
        plot_fits: Which distribution fits to display ('best', 'all', or list). Defaults to ''
        print_fits: Whether to print fit parameters. Defaults to False
        turns: Plot turn epochs instead of run/pause. Defaults to False
        stridechain_duration: Use run duration instead of stride count. Defaults to False
        legend_outside: Place legend outside plots. Defaults to False
        **kwargs: Additional arguments passed to AutoPlot

    Returns:
        Plot output (figure object or None based on return_fig setting)

    Example:
        >>> fig = plot_bouts(datasets=[d1, d2], plot_fits='best', turns=False)
    """
    if name is None:
        if not turns:
            name = f"runsNpauses{plot_fits}"
        else:
            name = f"turn_epochs{plot_fits}"
    P = plot.AutoPlot(
        name=name, build_kws={"Ncols": 2, "sharey": True, "wh": 5}, **kwargs
    )
    ax0, ax1 = P.axs[0], P.axs[1]

    valid_labs = {}
    for l, d, c in P.data_palette:
        v = d.fitted_epochs
        if v is None:
            continue

        kws = util.AttrDict(
            {
                "marker": "o",
                "plot_fits": plot_fits,
                "print_fits": print_fits,
                "label": l,
                "color": c,
                "legend_outside": legend_outside,
                "x0": None,
            }
        )

        def try_bout(k: str, ax_idx: int, bout: str, **kws2: Any) -> None:
            if k in v and v[k] is not None:
                plot_single_bout(
                    fit_dic=v[k], bout=bout, ax=P.axs[ax_idx], **kws2, **kws
                )
                valid_labs[l] = kws.color

        if not turns:
            try_bout("pause_dur", 1, "pauses")
            if stridechain_duration:
                try_bout("run_dur", 0, "runs")
            else:
                try_bout(
                    "run_count", 0, "stridechains", xlabel="# strides", discrete=True
                )
        else:
            try_bout("turn_dur", 0, "turn duration")
            try_bout(
                "turn_amp",
                1,
                "turn amplitude",
                xlabel="angle (deg)",
                xlim=(10**-0.5, 10**3),
            )

    ax0.set_ylabel("probability")
    ax1.yaxis.set_visible(False)
    if P.Ndatasets > 1:
        P.data_leg(
            0,
            labels=valid_labs.keys(),
            colors=valid_labs.values(),
            loc="lower left",
            fontsize=15,
        )
    P.adjust((0.15, 0.95), (0.15, 0.92), 0.05, 0.005)
    return P.get()
