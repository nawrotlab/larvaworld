"""
Barplots
"""

from __future__ import annotations
from typing import Any, Dict, Optional, Sequence

import itertools

import numpy as np
from matplotlib.patches import Patch
from scipy.stats import ttest_ind

from .. import plot, util, funcs

__all__: list[str] = [
    "error_barplot",
    "intake_barplot",
    "barplot",
    "auto_barplot",
]


@funcs.graph("error barplot", required={"args": ["error_dict", "evaluation"]})
def error_barplot(
    error_dict: Dict[str, Any],
    evaluation: Dict[str, Any],
    labels: Optional[Dict[str, str]] = None,
    name: str = "error_barplots",
    titles: Sequence[str] = (
        r"$\bf{endpoint}$ $\bf{metrics}$",
        r"$\bf{timeseries}$ $\bf{metrics}$",
    ),
    **kwargs: Any,
) -> Any:
    """
    Create bar plots comparing error metrics across models.

    Generates multi-panel bar plots showing endpoint and timeseries error metrics
    for different model configurations with statistical comparisons.

    Args:
        error_dict: Dictionary of error DataFrames by metric type
        evaluation: Dictionary of evaluation metadata with group labels and colors
        labels: Custom labels for metric types. Defaults to metric keys
        name: Plot name for saving. Defaults to 'error_barplots'
        titles: Panel titles. Defaults to endpoint and timeseries labels
        **kwargs: Additional arguments passed to AutoBasePlot

    Returns:
        Plot output (figure object or None based on return_fig setting)

    Example:
        >>> fig = error_barplot(error_dict, evaluation, labels={'endpoint': 'Endpoints'})
    """

    def build_legend(ax: Any, eval_df: Any) -> None:
        h, l = ax.get_legend_handles_labels()
        empty = Patch(color="none")
        counter = 0
        for g in eval_df.index:
            h.insert(counter, empty)
            l.insert(counter, eval_df["group_label"].loc[g])
            counter += len(eval_df["shorts"].loc[g]) + 1
        ax.legend(h, l, loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=15)

    Nplots = len(error_dict)
    P = plot.AutoBasePlot(
        name=name, build_kws={"Nrows": Nplots, "w": 20, "h": 6}, **kwargs
    )

    P.adjust((0.07, 0.7), (0.05, 0.95), 0.05, 0.2)
    for ii, (k, eval_df) in enumerate(evaluation.items()):
        kws = {
            "ylabel": labels[k] if labels is not None else k,
            "color": util.flatten_list(eval_df["par_colors"].values.tolist()),
            "ax": P.axs[ii],
            "kind": "bar",
            "rot": 0,
            "legend": False,
            "width": 0.6,
        }

        df = error_dict[k][util.SuperList(eval_df["symbols"].values).flatten]
        df.plot(**kws)
        build_legend(P.axs[ii], eval_df)
        P.conf_ax(ii, title=titles[ii], xlab="", yMaxN=4)
    return P.get()


@funcs.graph("food intake (barplot)", required={"ks": ["f_am"]})
def intake_barplot(**kwargs: Any) -> Any:
    """
    Create bar plot of food intake amounts.

    Convenience wrapper for barplot() specifically for food amount ('f_am')
    parameter visualization across datasets.

    Args:
        **kwargs: Arguments passed to barplot()

    Returns:
        Plot output (figure object or None based on return_fig setting)

    Example:
        >>> fig = intake_barplot(datasets=[d1, d2], labels=['Control', 'Test'])
    """
    return barplot(ks=["f_am"], **kwargs)


@funcs.graph("barplot", required={"ks": []})
def barplot(
    ks: Sequence[str],
    coupled_labels: Optional[Sequence[str]] = None,
    xlabel: Optional[str] = None,
    leg_cols: Optional[Sequence[str]] = None,
    **kwargs: Any,
) -> Any:
    """
    Create bar plots with statistical comparisons across datasets.

    Generates bar plots for specified parameters showing mean values with
    error bars and p-values from t-tests between dataset pairs.

    Args:
        ks: Parameter keys to plot
        coupled_labels: Labels for paired comparisons. If provided, datasets
            are grouped into pairs
        xlabel: Custom x-axis label. Defaults to None
        leg_cols: Colors for legend items. Auto-generated if None
        **kwargs: Additional arguments passed to AutoPlot

    Returns:
        Plot output (figure object or None based on return_fig setting)

    Example:
        >>> fig = barplot(ks=['v', 'a'], datasets=[d1, d2], labels=['Control', 'Test'])
    """
    Nks = len(ks)

    P = plot.AutoPlot(
        ks=ks,
        name=ks[0],
        build_kws={"N": Nks, "Ncols": int(np.ceil(Nks / 3)), "w": 8, "h": 6},
        **kwargs,
    )
    Nds = P.Ndatasets

    w = 0.15

    if coupled_labels is not None:
        Npairs = len(coupled_labels)
        N = int(Nds / Npairs)
        if leg_cols is None:
            leg_cols = util.N_colors(N)
        colors = leg_cols * Npairs
        leg_ids = P.labels[:N]
        ind = np.hstack([np.linspace(0 + i / N, w + i / N, N) for i in range(Npairs)])
        new_ind = ind[::N] + (ind[N - 1] - ind[0]) / N
        xticks, xticklabels = new_ind, coupled_labels
    else:
        ind = np.arange(0, w * Nds, w)
        colors = P.colors
        leg_ids = P.labels
        xticks, xticklabels = ind, P.labels

    bar_kwargs = {
        "width": w,
        "color": colors,
        "linewidth": 2,
        "zorder": 5,
        "align": "center",
        "edgecolor": "black",
    }
    err_kwargs = {
        "zorder": 20,
        "fmt": "none",
        "linewidth": 4,
        "ecolor": "k",
        "barsabove": True,
        "capsize": 10,
    }

    for ii, k in enumerate(P.ks):
        ax = P.axs[ii]
        p = P.pdict[k]
        vs = P.vdict[k]
        means = [np.mean(v) for v in vs]
        stds = [np.std(v) for v in vs]

        ax.p1 = ax.bar(ind, means, **bar_kwargs)
        ax.errs = ax.errorbar(ind, means, yerr=stds, **err_kwargs)

        if not coupled_labels:
            for i, j in itertools.combinations(np.arange(Nds).tolist(), 2):
                st, pv = ttest_ind(vs[i], vs[j], equal_var=False)
                pv = np.round(pv, 4)
                plot.label_diff(i, j, f"p={pv}", ind, means, ax)
        else:
            for k in range(Npairs):
                i, j = k * N, k * N + 1
                st, pv = ttest_ind(list(vs[i]), list(vs[j]), equal_var=False)
                if pv <= 0.05:
                    ax.text(ind[i], means[i] + stds[i], "*", ha="center", fontsize=20)
            P.data_leg(
                ii,
                labels=leg_ids,
                colors=leg_cols,
                loc="upper left",
                handlelength=1,
                handleheight=1,
            )

        P.conf_ax(
            ii,
            xlab=xlabel if xlabel is not None else None,
            ylab=p.l,
            ylim=[0, None],
            yMaxN=4,
            ytickMath=(-3, 3),
            xticks=xticks,
            xticklabels=xticklabels,
        )
    P.adjust((0.15, 0.95), (0.15, 0.95), H=0.05)
    P.fig.align_ylabels(P.axs[:])
    return P.get()


@funcs.graph("auto_barplot", required={"ks": []})
def auto_barplot(
    ks: Sequence[str],
    coupled_labels: Optional[Sequence[str]] = None,
    xlabel: Optional[str] = None,
    leg_cols: Optional[Sequence[str]] = None,
    **kwargs: Any,
) -> Any:
    """
    Create automatic bar plots with enhanced statistical annotations.

    Similar to barplot() but with enhanced automatic layout and statistical
    significance markers for paired comparisons. Shows asterisks for p <= 0.05.

    Args:
        ks: Parameter keys to plot
        coupled_labels: Labels for paired comparisons. If provided, datasets
            are grouped into pairs
        xlabel: Custom x-axis label. Defaults to None
        leg_cols: Colors for legend items. Auto-generated if None
        **kwargs: Additional arguments passed to AutoPlot

    Returns:
        Plot output (figure object or None based on return_fig setting)

    Example:
        >>> fig = auto_barplot(ks=['run_count'], coupled_labels=['Control', 'Treatment'])
    """
    Nks = len(ks)

    P = plot.AutoPlot(
        ks=ks,
        name=ks[0],
        build_kws={"N": Nks, "Ncols": int(np.ceil(Nks / 3)), "w": 8, "h": 6},
        **kwargs,
    )
    Nds = P.Ndatasets

    w = 0.15

    if coupled_labels is not None:
        Npairs = len(coupled_labels)
        N = int(Nds / Npairs)
        if leg_cols is None:
            leg_cols = util.N_colors(N)
        colors = leg_cols * Npairs
        # leg_ids = P.labels[:N]
        leg_kws = {"labels": P.labels[:N], "colors": leg_cols}
        ind = np.hstack([np.linspace(0 + i / N, w + i / N, N) for i in range(Npairs)])
        new_ind = ind[::N] + (ind[N - 1] - ind[0]) / N
        # xticks, xticklabels = new_ind, coupled_labels
        ax_kws = {"xticks": new_ind, "xticklabels": coupled_labels}
        ijs = [(kk * N, kk * N + 1) for kk in range(Npairs)]

        ij_pairs = ijs
        # finfuncN = 2

    else:
        ind = np.arange(0, w * Nds, w)
        colors = P.colors
        # leg_ids = P.labels
        # xticks, xticklabels = ind, P.labels
        ax_kws = {"xticks": ind, "xticklabels": P.labels}
        ijs = []
        for i, j in itertools.combinations(np.arange(Nds).tolist(), 2):
            ijs.append((i, j))

        ij_pairs = ijs
        # finfuncN=1

    bar_kwargs = {
        "width": w,
        "color": colors,
        "linewidth": 2,
        "zorder": 5,
        "align": "center",
        "edgecolor": "black",
    }
    err_kwargs = {
        "zorder": 20,
        "fmt": "none",
        "linewidth": 4,
        "ecolor": "k",
        "barsabove": True,
        "capsize": 10,
    }

    for ii, k in enumerate(P.ks):
        ax = P.axs[ii]
        p = P.pdict[k]
        vs = P.vdict[k]
        means = [np.mean(v) for v in vs]
        stds = [np.std(v) for v in vs]
        ax.p1 = ax.bar(ind, means, **bar_kwargs)
        ax.errs = ax.errorbar(ind, means, yerr=stds, **err_kwargs)

        for i, j in ij_pairs:
            st, pv = ttest_ind(list(vs[i]), list(vs[j]), equal_var=False)
            pv = np.round(pv, 4)
            if coupled_labels is not None:
                if pv <= 0.05:
                    ax.text(ind[i], means[i] + stds[i], "*", ha="center", fontsize=20)
                P.data_leg(
                    ii, **leg_kws, loc="upper left", handlelength=1, handleheight=1
                )
            else:
                plot.label_diff(i, j, f"p={pv}", ind, means, ax)

        P.conf_ax(
            ii,
            xlab=xlabel if xlabel is not None else None,
            ylab=p.l,
            ylim=[0, None],
            yMaxN=4,
            ytickMath=(-3, 3),
            **ax_kws,
        )
    P.adjust((0.15, 0.95), (0.15, 0.95), W=0.1, H=0.1)
    P.fig.align_ylabels(P.axs[:])
    return P.get()
