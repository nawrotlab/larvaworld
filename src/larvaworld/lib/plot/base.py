"""
Basic plotting classes
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, Iterable, Optional, Sequence, Tuple

if TYPE_CHECKING:  # type-only imports to avoid heavy runtime deps
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

import itertools
import os

import numpy as np
import pandas as pd
from matplotlib import patches, ticker
from matplotlib.gridspec import GridSpec
from scipy.stats import ttest_ind

from ... import vprint
from .. import plot, reg, util
from ..process import LarvaDatasetCollection
from ..util import AttrDict

__all__: list[str] = [
    "BasePlot",
    "AutoBasePlot",
    "AutoPlot",
    "GridPlot",
]

__displayname__ = "Plotting template classes"

_MPL_CONFIGURED = False


def _ensure_matplotlib_config():
    global _MPL_CONFIGURED
    if not _MPL_CONFIGURED:
        from matplotlib import pyplot as plt  # local import

        plt_conf = {
            "axes.labelsize": 20,
            "axes.titlesize": 25,
            "figure.titlesize": 25,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "legend.fontsize": 20,
            "legend.title_fontsize": 20,
        }
        try:
            plt.rcParams.update(plt_conf)
        except Exception:
            pass
        _MPL_CONFIGURED = True


class BasePlot:
    """
    Base class for creating customizable matplotlib plots.

    Provides common functionality for plot generation, styling, and output
    management. Handles figure/axes creation, saving, and display options.
    Subclasses implement specific plot types by overriding plot methods.

    Attributes:
        filename: Output filename with extension
        fit_filename: Filename for fit data CSV
        save_to: Directory path for saving plots
        return_fig: Whether to return figure object
        show: Whether to display plot immediately
        build_kws: Keyword arguments for figure building

    Example:
        >>> plotter = BasePlot(name='myplot', save_to='./plots', suf='png')
        >>> plotter.build(nrows=2, ncols=2)
        >>> # ... add plot content ...
        >>> plotter.save()
    """

    def __init__(
        self,
        name: str = "larvaworld_plot",
        save_as: Optional[str] = None,
        pref: Optional[str] = None,
        suf: str = "pdf",
        save_to: Optional[str] = None,
        subfolder: Optional[str] = None,
        return_fig: bool = False,
        show: bool = False,
        subplot_kw: Dict[str, Any] = {},
        build_kws: Dict[str, Any] = {},
        **kwargs: Any,
    ) -> None:
        if save_as is None:
            if pref:
                name = f"{pref}_{name}"
            save_as = name
        self.filename = f"{save_as}.{suf}"
        self.fit_filename = f"{save_as}_fits.csv"
        self.fit_ind = None
        self.fit_df = None

        if save_to:
            if subfolder:
                save_to = f"{save_to}/{subfolder}"
            os.makedirs(save_to, exist_ok=True)
        self.save_to = save_to

        self.return_fig = return_fig
        self.show = show
        build_kws["subplot_kw"] = subplot_kw
        self.build_kws = build_kws
        for k, v in self.build_kws.items():
            if v == "Ndatasets" and hasattr(self, "Ndatasets"):
                self.build_kws[k] = self.Ndatasets
            elif v == "Nks" and hasattr(self, "Nks"):
                self.build_kws[k] = self.Nks

    def build(
        self,
        fig: Optional["Figure"] = None,
        axs: Optional["Axes | Sequence[Axes]"] = None,
        dim3: bool = False,
        azim: int = 115,
        elev: int = 15,
    ) -> None:
        """
        Method that defines the figure and axes on which to draw.
        These can be provided externally as arguments to create a composite figure. Otherwise they are created independently.

        Args:
            fig: The figure of the plot (optional)
            axs: The axes of the figure (optional)
            dim3: Whether the figure will be 3-dimensional. Default : False
            azim: The azimuth of a 3D figure. Default : 115
            elev: The elevation of a 3D figure. Default : 15

        """
        if fig is not None and axs is not None:
            self.fig = fig
            self.axs = axs if type(axs) in [list, np.ndarray] else [axs]

        else:
            if dim3:
                from mpl_toolkits.mplot3d import Axes3D
                from matplotlib import pyplot as plt

                _ensure_matplotlib_config()
                self.fig = plt.figure(figsize=(15, 10))
                ax = Axes3D(self.fig, azim=azim, elev=elev)
                self.axs = [ax]
            else:
                from matplotlib import pyplot as plt

                _ensure_matplotlib_config()
                self.fig, axs = plt.subplots(
                    **plot.configure_subplot_grid(**self.build_kws)
                )
                self.axs = axs.ravel() if isinstance(axs, np.ndarray) else [axs]

    @property
    def Naxs(self) -> int:
        return len(self.axs)

    @property
    def Ncols(self) -> int:
        return self.axs[0].get_gridspec().ncols

    @property
    def Nrows(self) -> int:
        return self.axs[0].get_gridspec().nrows

    def conf_ax(
        self,
        idx: int = 0,
        ax: Optional["Axes"] = None,
        xlab: Optional[str] = None,
        ylab: Optional[str] = None,
        zlab: Optional[str] = None,
        xlim: Optional[Sequence[float]] = None,
        ylim: Optional[Sequence[float]] = None,
        zlim: Optional[Sequence[float]] = None,
        xticks: Optional[Sequence[float]] = None,
        xticklabels: Optional[Sequence[str]] = None,
        yticks: Optional[Sequence[float]] = None,
        xticklabelrotation: Optional[int] = None,
        yticklabelrotation: Optional[int] = None,
        yticklabels: Optional[Sequence[str]] = None,
        zticks: Optional[Sequence[float]] = None,
        zticklabels: Optional[Sequence[str]] = None,
        xtickpos: Optional[str] = None,
        xtickpad: Optional[int] = None,
        ytickpad: Optional[int] = None,
        ztickpad: Optional[int] = None,
        xlabelfontsize: Optional[int] = None,
        ylabelfontsize: Optional[int] = None,
        xticklabelsize: Optional[int] = None,
        yticklabelsize: Optional[int] = None,
        zticklabelsize: Optional[int] = None,
        major_ticklabelsize: Optional[int] = None,
        minor_ticklabelsize: Optional[int] = None,
        xlabelpad: Optional[int] = None,
        ylabelpad: Optional[int] = None,
        zlabelpad: Optional[int] = None,
        equal_aspect: Optional[bool] = None,
        xMaxN: Optional[int] = None,
        yMaxN: Optional[int] = None,
        zMaxN: Optional[int] = None,
        yStrN: Optional[int] = None,
        xMath: Optional[bool] = None,
        yMath: Optional[bool] = None,
        tickMath: Optional[Tuple[int, int]] = None,
        ytickMath: Optional[Tuple[int, int]] = None,
        xMaxFix: bool = False,
        leg_loc: Optional[str] = None,
        leg_handles: Optional[Sequence[Any]] = None,
        leg_labels: Optional[Sequence[str]] = None,
        legfontsize: Optional[int] = None,
        xvis: Optional[bool] = None,
        yvis: Optional[bool] = None,
        zvis: Optional[bool] = None,
        title: Optional[str] = None,
        title_y: Optional[float] = None,
        titlefontsize: Optional[int] = None,
    ) -> None:
        """
        Helper method that configures an axis of the figure

        """
        if ax is None:
            ax = self.axs[idx]
        if equal_aspect is not None:
            ax.set_aspect("equal", adjustable="box")
        if xvis is not None:
            ax.xaxis.set_visible(xvis)
        if yvis is not None:
            ax.yaxis.set_visible(yvis)
        if zvis is not None:
            ax.zaxis.set_visible(zvis)
        if ylab is not None:
            if ylabelfontsize is not None:
                ax.set_ylabel(ylab, labelpad=ylabelpad, fontsize=ylabelfontsize)
            else:
                ax.set_ylabel(ylab, labelpad=ylabelpad)
        if xlab is not None:
            if xlabelfontsize is not None:
                ax.set_xlabel(xlab, labelpad=xlabelpad, fontsize=xlabelfontsize)
            else:
                ax.set_xlabel(xlab, labelpad=xlabelpad)
        if zlab is not None:
            ax.set_zlabel(zlab, labelpad=zlabelpad)
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        if zlim is not None:
            ax.set_zlim(zlim)
        if xticks is not None:
            ax.set_xticks(ticks=xticks)
        if xticklabelrotation is not None:
            ax.tick_params(axis="x", which="major", rotation=xticklabelrotation)
        if xticklabelsize is not None:
            ax.tick_params(axis="x", which="major", labelsize=xticklabelsize)
        if yticklabelsize is not None:
            ax.tick_params(axis="y", which="major", labelsize=yticklabelsize)
        if zticklabelsize is not None:
            ax.tick_params(axis="z", which="major", labelsize=zticklabelsize)
        if major_ticklabelsize is not None:
            ax.tick_params(axis="both", which="major", labelsize=major_ticklabelsize)
        if minor_ticklabelsize is not None:
            ax.tick_params(axis="both", which="minor", labelsize=minor_ticklabelsize)

        if xticklabels is not None:
            ax.set_xticklabels(labels=xticklabels, rotation=xticklabelrotation)
        if yticks is not None:
            ax.set_yticks(ticks=yticks)
        if yticklabels is not None:
            ax.set_yticklabels(labels=yticklabels, rotation=yticklabelrotation)
        if zticks is not None:
            ax.set_zticks(ticks=zticks)
        if zticklabels is not None:
            ax.set_zticklabels(labels=zticklabels)
        if tickMath is not None:
            ax.ticklabel_format(useMathText=True, scilimits=tickMath)
        if ytickMath is not None:
            ax.ticklabel_format(
                axis="y", useMathText=True, scilimits=ytickMath, useOffset=True
            )
        if xMaxFix:
            ticks_loc = ax.get_xticks().tolist()
            ax.xaxis.set_major_locator(ticker.FixedLocator(ticks_loc))
        if xMaxN is not None:
            ax.xaxis.set_major_locator(ticker.MaxNLocator(xMaxN))
        if yMaxN is not None:
            ax.yaxis.set_major_locator(ticker.MaxNLocator(yMaxN))
        if zMaxN is not None:
            ax.zaxis.set_major_locator(ticker.MaxNLocator(zMaxN))
        if yStrN is not None:
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter(f"%.{yStrN}f"))

        if xMath is not None:
            ax.xaxis.set_major_formatter(
                ticker.ScalarFormatter(useOffset=True, useMathText=True)
            )
        if yMath is not None:
            ax.yaxis.set_major_formatter(
                ticker.ScalarFormatter(useOffset=True, useMathText=True)
            )
        if xtickpos is not None:
            ax.xaxis.set_ticks_position(xtickpos)
        if title is not None:
            ax.set_title(title, fontsize=titlefontsize, y=title_y)
        if xtickpad is not None:
            ax.xaxis.set_tick_params(pad=xtickpad)
        if ytickpad is not None:
            ax.yaxis.set_tick_params(pad=ytickpad)
        if ztickpad is not None:
            ax.zaxis.set_tick_params(pad=ztickpad)

        if leg_loc is not None:
            kws = {
                "loc": leg_loc,
                "fontsize": legfontsize,
            }
            if leg_handles is not None:
                kws["handles"] = leg_handles
            if leg_labels is not None:
                kws["labels"] = leg_labels
            ax.legend(**kws)

    def conf_ax_3d(
        self,
        vars: Sequence[str],
        target: str,
        lims: Optional[Tuple[Sequence[float], Sequence[float], Sequence[float]]] = None,
        title: Optional[str] = None,
        maxN: int = 3,
        labelpad: int = 15,
        tickpad: int = 5,
        idx: int = 0,
    ) -> None:
        if lims is None:
            xlim, ylim, zlim = None, None, None
        else:
            xlim, ylim, zlim = lims
        self.conf_ax(
            idx=idx,
            xlab=vars[0],
            ylab=vars[1],
            zlab=target,
            xlim=xlim,
            ylim=ylim,
            zlim=zlim,
            xtickpad=tickpad,
            ytickpad=tickpad,
            ztickpad=tickpad,
            xlabelpad=labelpad,
            ylabelpad=labelpad,
            zlabelpad=labelpad,
            xMaxN=maxN,
            yMaxN=maxN,
            zMaxN=maxN,
            title=title,
        )

    def adjust(
        self,
        LR: Optional[Tuple[float, float]] = None,
        BT: Optional[Tuple[float, float]] = None,
        W: Optional[float] = None,
        H: Optional[float] = None,
    ) -> None:
        kws = {}
        if LR is not None:
            kws["left"] = LR[0]
            kws["right"] = LR[1]
        if BT is not None:
            kws["bottom"] = BT[0]
            kws["top"] = BT[1]
        if W is not None:
            kws["wspace"] = W
        if H is not None:
            kws["hspace"] = H
        self.fig.subplots_adjust(**kws)

    def set(self, fig: "Figure") -> None:
        self.fig = fig

    def get(self) -> Any:
        if self.fit_df is not None and self.save_to is not None:
            self.fit_df.to_csv(
                os.path.join(self.save_to, self.fit_filename), index=True, header=True
            )
        return plot.process_plot(
            self.fig, self.save_to, self.filename, self.return_fig, self.show
        )

    def conf_fig(
        self,
        adjust_kws: Optional[Dict[str, Any]] = None,
        align: Optional[Sequence["Axes"] | Sequence[Any]] = None,
        title: Optional[str] = None,
        title_kws: Dict[str, Any] = {},
    ) -> None:
        if title is not None:
            pairs = {
                # 't':'t',
                "w": "fontweight",
                "s": "fontsize",
                # 't':title_kws.t,
            }
            kws = AttrDict(title_kws).replace_keys(pairs)
            self.fig.suptitle(t=title, **kws)
        if adjust_kws is not None:
            self.adjust(**adjust_kws)
        if align is not None:
            if type(align) == list:
                ax_list = align
            else:
                ax_list = self.axs[:]
            self.fig.align_ylabels(ax_list)


class AutoBasePlot(BasePlot):
    """
    Automatic plot generation with immediate figure building.

    Extends BasePlot by automatically calling build() during initialization,
    creating the matplotlib figure and axes immediately. Supports both 2D
    and 3D plots with customizable viewing angles.

    Attributes:
        fig: Matplotlib Figure object
        ax: Matplotlib Axes object (or array of Axes for subplots)
        dim3: Whether plot is 3D

    Example:
        >>> plot = AutoBasePlot(nrows=2, ncols=2, dim3=False)
        >>> plot.ax[0, 0].plot(x, y)  # Use axes directly
        >>> plot.save()
    """

    def __init__(
        self,
        fig: Optional["Figure"] = None,
        axs: Optional["Axes | Sequence[Axes]"] = None,
        dim3: bool = False,
        azim: int = 115,
        elev: int = 15,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.build(fig=fig, axs=axs, dim3=dim3, azim=azim, elev=elev)


class AutoPlot(AutoBasePlot, LarvaDatasetCollection):
    """
    Automatic plot generation with larvaworld dataset integration.

    Combines AutoBasePlot functionality with LarvaDatasetCollection to
    enable direct plotting from larvaworld LarvaDataset objects. Handles
    multiple datasets with automatic labeling, coloring, and unit conversion.

    Attributes:
        datasets: Collection of LarvaDataset objects
        labels: Dataset labels for legend
        colors: Colors for each dataset
        ks: Parameter keys to plot
        key: Indexing key ('step', 'time', etc.)
        klabels: Custom labels for parameters
        Ndatasets: Number of datasets
        Nks: Number of parameters

    Example:
        >>> plot = AutoPlot(datasets=[d1, d2], labels=['Control', 'Test'])
        >>> plot.plot(ks=['v', 'a'])  # Plot velocity and acceleration
        >>> plot.save()
    """

    def __init__(
        self,
        ks: Sequence[str] = [],
        key: str = "step",
        klabels: Dict[str, str] = {},
        datasets: Sequence[Any] = [],
        labels: Optional[Sequence[str]] = None,
        colors: Optional[Sequence[Any]] = None,
        add_samples: bool = False,
        ranges: Optional[Sequence[Any]] = None,
        absolute: bool = False,
        rad2deg: bool = False,
        space_unit: str = "mm",
        **kwargs: Any,
    ) -> None:
        LarvaDatasetCollection.__init__(
            self,
            datasets=datasets,
            labels=labels,
            colors=colors,
            add_samples=add_samples,
        )
        self.key = key
        self.ks = []
        self.kkdict = AttrDict()
        self.pdict = AttrDict()
        self.vdict = AttrDict()
        reg.par.update_kdict(ks=ks)
        for k in util.SuperList(ks).existing(reg.par.ks):
            p = reg.par.kdict[k]
            if p.u == reg.units.m and space_unit == "mm":
                p.u = reg.units.millimeter
                coeff = 1000
            else:
                coeff = 1
            if k in klabels:
                p.disp = klabels[k]

            try:
                dfs = self.datasets.get_par(k=k, key=key) * coeff

                def get_vs_from_df(df: pd.DataFrame) -> np.ndarray:
                    assert df is not None
                    v = df.dropna().values
                    if absolute:
                        v = np.abs(v)
                    if rad2deg:
                        # if p.u == reg.units.rad:
                        #     p.u = reg.units.deg
                        v = np.rad2deg(v)
                    # else:
                    # if p.u == reg.units.deg:
                    #     p.u = reg.units.rad
                    #     v = np.deg2rad(v)
                    return v

                self.vdict[k] = [get_vs_from_df(df) for df in dfs]
                self.kkdict[k] = AttrDict(zip(self.labels, dfs))
                self.pdict[k] = p
                self.ks.append(k)
            except:
                vprint(f"Failed to retrieve key {k}", 1)
                pass
        self.dkdict = AttrDict(
            {l: {k: self.kkdict[k][l] for k in self.ks} for l in self.labels}
        )
        self.pars = reg.getPar(self.ks)
        self.Nks = len(self.ks)
        self.ranges = ranges
        self.absolute = absolute
        self.rad2deg = rad2deg

        AutoBasePlot.__init__(self, **kwargs)

    def comp_all_pvalues(self) -> None:
        if self.Ndatasets < 2:
            return
        columns = pd.MultiIndex.from_product(
            [self.ks, ["significance", "stat", "pvalue"]]
        )
        fit_ind = pd.MultiIndex.from_tuples(
            list(itertools.combinations(self.labels, 2))
        )
        self.fit_df = pd.DataFrame(index=fit_ind, columns=columns)

        for k in self.ks:
            for ind, (vv1, vv2) in zip(
                fit_ind, itertools.combinations(self.vdict[k], 2)
            ):
                v1, v2 = list(vv1), list(vv2)
                st, pv = ttest_ind(v1, v2, equal_var=False)
                if not pv <= 0.01:
                    t = 0
                elif np.nanmean(v1) < np.nanmean(v2):
                    t = 1
                else:
                    t = -1
                self.fit_df.loc[ind, k] = [t, st, np.round(pv, 11)]

    def plot_all_half_circles(self) -> None:
        if self.fit_df is None:
            return
        for i, k in enumerate(self.ks):
            df = self.fit_df[k]
            ii = 0
            for z, (l1, l2) in enumerate(df.index.values):
                col1, col2 = (
                    self.colors[self.labels.index(l1)],
                    self.colors[self.labels.index(l2)],
                )
                pv = df["pvalue"].loc[(l1, l2)]
                v = df["significance"].loc[(l1, l2)]
                res = self.plot_half_circle(
                    self.axs[i], col1, col2, v=v, pv=pv, coef=z - ii
                )
                if not res:
                    ii += 1
                    continue

    def plot_half_circle(
        self,
        ax: "Axes",
        col1: Any,
        col2: Any,
        v: int,
        pv: float,
        coef: int = 0,
    ) -> bool:
        res = True
        if v == 1:
            c1, c2 = col1, col2
        elif v == -1:
            c1, c2 = col2, col1
        else:
            res = False

        if res:
            rad = 0.04
            yy = 0.95 - coef * 0.08
            xx = 0.75
            plot.dual_half_circle(
                center=(xx, yy),
                radius=rad,
                ax=ax,
                colors=(c1, c2),
                transform=ax.transAxes,
            )
            if pv == 0:
                pvi = -9
            else:
                for pvi in np.arange(-1, -10, -1):
                    if np.log10(pv) > pvi:
                        pvi += 1
                        break
            ax.text(
                xx + 0.05,
                yy + rad / 1.5,
                f"p<10$^{{{pvi}}}$",
                ha="left",
                va="top",
                color="k",
                fontsize=15,
                transform=ax.transAxes,
            )
        return res

    def data_leg(
        self,
        idx: Optional[int] = None,
        labels: Optional[Sequence[str]] = None,
        colors: Optional[Sequence[Any]] = None,
        anchor: Optional[Tuple[float, float]] = None,
        handlelength: float = 0.5,
        handleheight: float = 0.5,
        Nagents_in_label: bool = True,
        **kwargs: Any,
    ) -> Any:
        if labels is None:
            if not Nagents_in_label:
                labels = self.labels
            else:
                labels = self.labels_with_N
        if colors is None:
            colors = self.colors
        kws = {
            "handles": [
                patches.Patch(facecolor=c, label=l, edgecolor="black")
                for c, l in zip(colors, labels)
            ],
            "handlelength": handlelength,
            "handleheight": handleheight,
            "labels": labels,
            "bbox_to_anchor": anchor,
            **kwargs,
        }
        if idx is None:
            from matplotlib import pyplot as plt

            leg = plt.legend(**kws)
        else:
            ax = self.axs[idx]
            leg = ax.legend(**kws)
            ax.add_artist(leg)
        return leg

    def plot_quantiles(
        self,
        k: Optional[str] = None,
        par: Optional[str] = None,
        idx: int = 0,
        ax: Optional["Axes"] = None,
        xlim: Optional[Sequence[float]] = None,
        ylim: Optional[Sequence[float]] = None,
        ylab: Optional[str] = None,
        unit: str = "sec",
        leg_loc: str = "upper left",
        coeff: float = 1,
        absolute: bool = False,
        individuals: bool = False,
        show_first: bool = False,
        Nagents_in_label: bool = True,
        **kwargs: Any,
    ) -> None:
        x = self.trange(unit)
        if ax is None:
            ax = self.axs[idx]

        try:
            if k is None:
                k = reg.getPar(d=par, to_return="k")
            reg.par.update_kdict(ks=[k])
            p = reg.par.kdict[k]

            if ylab is None:
                ylab = p.l
            if ylim is None:
                ylim = p.lim
        except:
            pass
        if xlim is None:
            xlim = [x[0], x[-1]]

        if not Nagents_in_label:
            data = self.data_palette
        else:
            data = self.data_palette_with_N

        for l, d, c in data:
            df = d.get_par(k=k, par=par) * coeff
            if absolute:
                df = df.abs()
            if individuals:
                # plot each timeseries individually
                for id in df.index.get_level_values("AgentID"):
                    df_single = df.xs(id, level="AgentID")
                    ax.plot(x, df_single, color=c, linewidth=1)
            else:
                # plot the shaded range between first and third quantile
                df_u = df.groupby(level="Step").quantile(q=0.75)
                df_b = df.groupby(level="Step").quantile(q=0.25)
                # print(df_u.shape,df_b.shape,x.shape,x[:df_u.shape[0]].shape,self.Nticks,d.Nticks)
                # if x.shape[0]!=df_u.shape[0]
                # x=x[:df_u.shape[0]]
                ax.fill_between(
                    x[: df_u.shape[0]], df_u, df_b, color=c, alpha=0.2, zorder=0
                )

                if show_first:
                    df_single = df.xs(
                        df.index.get_level_values("AgentID")[0], level="AgentID"
                    )
                    ax.plot(x, df_single, color=c, linestyle="dashed", linewidth=1)

            # plot the mean on top
            df_m = df.groupby(level="Step").quantile(q=0.5)
            ax.plot(
                x[: df_m.shape[0]], df_m, c, label=l, linewidth=2, alpha=1.0, zorder=10
            )
        self.conf_ax(
            ax=ax,
            xlab=f"time, ${unit}$",
            ylab=ylab,
            xlim=xlim,
            ylim=ylim,
            xMaxN=5,
            yMaxN=5,
            leg_loc=leg_loc,
            **kwargs,
        )

    def plot_hist(
        self,
        half_circles: bool = True,
        use_title: bool = False,
        par_legend: bool = False,
        nbins: int = 30,
        alpha: float = 0.5,
        ylim: Sequence[float] = [0, 0.2],
        Nagents_in_label: bool = True,
        **kwargs: Any,
    ) -> None:
        loc = "upper left" if half_circles else "upper right"
        for i, k in enumerate(self.ks):
            p = self.pdict[k]
            vs = self.vdict[k]
            if self.ranges:
                r = self.ranges[i]
                if isinstance(r, tuple):
                    r0, r1 = r
                else:
                    r0, r1 = -r, r
            else:
                r0, r1 = (
                    np.min([np.min(v) for v in vs]),
                    np.max([np.max(v) for v in vs]),
                )
            if self.absolute:
                r0 = 0
            bins = np.linspace(r0, r1, nbins)
            xlim = (r0, r1)
            plot.prob_hist(
                vs=vs,
                colors=self.colors,
                labels=self.labels,
                ax=self.axs[i],
                bins=bins,
                alpha=alpha,
                **kwargs,
            )
            self.conf_ax(
                i,
                ylab="probability",
                yvis=True if i % self.Ncols == 0 else False,
                xlab=p.l,
                xlim=xlim,
                ylim=ylim,
                xMaxN=4,
                yMaxN=4,
                xMath=True,
                title=p.disp if use_title else None,
                leg_loc=loc if par_legend else None,
            )

        self.comp_all_pvalues()
        if half_circles:
            self.plot_all_half_circles()
        self.data_leg(0, loc=loc, Nagents_in_label=Nagents_in_label)

    def boxplots(
        self,
        grouped: bool = False,
        annotation: bool = True,
        show_ns: bool = False,
        target_only: Any = None,
        stripplot: bool = False,
        ylims: Optional[Sequence[Sequence[float]]] = None,
        **kwargs: Any,
    ) -> None:
        if not grouped:
            hue = None
            palette = dict(zip(self.labels, self.colors))
        else:
            hue = "GroupID"
            palette = dict(zip(self.group_ids, util.N_colors(self.Ngroups)))
        kws0 = {
            "x": "DatasetID",
            "palette": palette,
            "hue": hue,
            "data": util.concat_datasets(
                dict(zip(self.labels, self.datasets)), key=self.key
            ),
        }

        for ii, k in enumerate(self.ks):
            p = self.pdict[k]
            kws = {"y": p.d, "ax": self.axs[ii], **kws0}
            plot.single_boxplot(
                stripplot=stripplot,
                annotation=annotation,
                show_ns=show_ns,
                target_only=target_only,
                **kws,
            )
            self.conf_ax(
                ii,
                xticklabelrotation=30,
                ylab=p.l,
                yMaxN=4,
                ylim=ylims[ii] if ylims is not None else None,
                xvis=False if ii < (self.Nrows - 1) * self.Ncols else True,
            )


class GridPlot(BasePlot):
    """
    Multi-panel grid layout for composite plots.

    Creates a grid-based figure layout using matplotlib GridSpec for
    organizing multiple subplots. Supports automatic subplot placement
    with optional lettering (A, B, C, ...) and flexible sizing.

    Attributes:
        width: Number of columns in grid
        height: Number of rows in grid
        fig: Matplotlib Figure object
        grid: GridSpec layout manager
        cur_w: Current column position
        cur_h: Current row position
        letters: List of panel labels
        letter_dict: Mapping of panel positions to letters

    Example:
        >>> grid = GridPlot(name='composite', width=3, height=2)
        >>> ax1 = grid.add()  # Add first panel
        >>> ax2 = grid.add(N=2)  # Add panel spanning 2 columns
        >>> grid.save()
    """

    def __init__(
        self,
        name: str,
        width: int,
        height: int,
        scale: Tuple[int, int] = (1, 1),
        **kwargs: Any,
    ) -> None:
        super().__init__(name, **kwargs)
        ws, hs = scale
        self.width, self.height = width, height
        figsize = (int(width * ws), int(height * hs))
        from matplotlib import pyplot as plt

        _ensure_matplotlib_config()
        self.fig = plt.figure(constrained_layout=False, figsize=figsize)
        self.grid = GridSpec(height, width, figure=self.fig)
        self.cur_w, self.cur_h = 0, 0

        self.cur_idx = 0
        self.letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
        self.letter_dict = {}
        self.x0s, self.y0s = [], []

    def add(
        self,
        N: int = 1,
        w: Optional[int] = None,
        h: Optional[int] = None,
        w0: Optional[int] = None,
        h0: Optional[int] = None,
        dw: int = 0,
        dh: int = 0,
        share_w: bool = False,
        share_h: bool = False,
        letter: bool = True,
        x0: bool = False,
        y0: bool = False,
        cols_first: bool = False,
        annotate_all: bool = False,
    ) -> "Axes | list[Axes]":
        if w0 is None:
            w0 = self.cur_w
        if h0 is None:
            h0 = self.cur_h

        if w is None:
            w = self.width - w0
        if h is None:
            h = self.height - h0

        if N == 1:
            axs = self.fig.add_subplot(self.grid[h0 : h0 + h, w0 : w0 + w])
            self.add_letter(axs, letter, x0=x0, y0=y0)
        else:
            if share_h and not share_w:
                ww = int((w - (N - 1) * dw) / N)
                axs = [
                    self.fig.add_subplot(
                        self.grid[
                            h0 : h0 + h,
                            w0 + dw * i + ww * i : w0 + dw * i + ww * (i + 1),
                        ]
                    )
                    for i in range(N)
                ]
            elif share_w and not share_h:
                hh = int((h - (N - 1) * dh) / N)
                axs = [
                    self.fig.add_subplot(
                        self.grid[
                            h0 + dh * i + hh * i : h0 + dh * i + hh * (i + 1),
                            w0 : w0 + w,
                        ]
                    )
                    for i in range(N)
                ]
            elif share_w and share_h:
                Nrows, Ncols = N, N
                hh = int((h - (Nrows - 1) * dh) / Nrows)
                ww = int((w - (Ncols - 1) * dw) / Ncols)
                axs = []
                if not cols_first:
                    for i in range(Nrows):
                        for j in range(Ncols):
                            ax = self.fig.add_subplot(
                                self.grid[
                                    h0 + dh * i + hh * i : h0 + dh * i + hh * (i + 1),
                                    w0 + dw * j + ww * j : w0 + dw * j + ww * (j + 1),
                                ]
                            )
                            axs.append(ax)
                else:
                    for j in range(Ncols):
                        for i in range(Nrows):
                            ax = self.fig.add_subplot(
                                self.grid[
                                    h0 + dh * i + hh * i : h0 + dh * i + hh * (i + 1),
                                    w0 + dw * j + ww * j : w0 + dw * j + ww * (j + 1),
                                ]
                            )
                            axs.append(ax)
            if annotate_all:
                for i, ax in enumerate(axs):
                    if i == 0:
                        self.add_letter(ax, letter, x0=x0, y0=y0)
                    else:
                        self.add_letter(ax, letter)
            else:
                self.add_letter(axs[0], letter, x0=x0, y0=y0)
            # ax_letter = axs[0]
        return axs

    def add_letter(
        self, ax: "Axes", letter: bool = True, x0: bool = False, y0: bool = False
    ) -> None:
        if letter:
            self.letter_dict[ax] = self.letters[self.cur_idx]
            self.cur_idx += 1
            if x0:
                self.x0s.append(ax)
            if y0:
                self.y0s.append(ax)

    def annotate(
        self, dx: float = -0.05, dy: float = 0.005, full_dict: bool = False
    ) -> None:
        text_x0, text_y0 = 0.05, 0.98

        if full_dict:
            for i, ax in enumerate(self.axs):
                self.letter_dict[ax] = self.letters[i]
        for i, (ax, text) in enumerate(self.letter_dict.items()):
            X = text_x0 if ax in self.x0s else ax.get_position().x0 + dx
            Y = text_y0 if ax in self.y0s else ax.get_position().y1 + dy
            self.fig.text(X, Y, text, size=30, weight="bold")

    def plot(
        self,
        func: str,
        kws: Dict[str, Any],
        axs: Optional[Sequence["Axes"]] = None,
        **kwargs: Any,
    ) -> Any:
        if axs is None:
            axs = self.add(**kwargs)
        _ = reg.graphs.run(ID=func, fig=self.fig, axs=axs, **kws)
