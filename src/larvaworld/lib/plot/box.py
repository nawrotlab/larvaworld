"""
Boxplots
"""

import itertools

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import ttest_ind

from .. import plot, reg, util, funcs

__all__ = [
    "boxplot",
    "boxplot_PI",
    "PIboxplot",
    "boxplot_double_patch",
    "plot_foraging",
    "lineplot",
]


@funcs.graph("boxplot (grouped)", required={"ks": []})
def boxplot(
    ks,
    sort_labels=False,
    name=None,
    xlabel=None,
    pair_ids=None,
    common_ids=None,
    coupled_labels=None,
    **kwargs,
):
    Npars = len(ks)
    if name is None:
        name = ks[0]
    P = plot.AutoPlot(
        name=name,
        build_kws={"N": Npars, "Nrows": int(np.ceil(Npars / 3)), "w": 8, "h": 7},
        **kwargs,
    )
    gIDs = P.group_ids
    grouped_ds = util.AttrDict(
        {
            gID: util.ItemList([d for d in P.datasets if d.config.group_id == gID])
            for gID in gIDs
        }
    )

    if common_ids is None:
        common_ids = util.unique_list([l.split("_")[-1] for l in gIDs])
    Ncommon = len(common_ids)
    if pair_ids is None:
        pair_ids = util.unique_list([l.split("_")[0] for l in gIDs])
    Npairs = len(pair_ids)

    if sort_labels:
        common_ids = sorted(common_ids)
        pair_ids = sorted(pair_ids)

    if coupled_labels is None:
        coupled_labels = True if P.Ngroups == Npairs * Ncommon else False

    for ii, k in enumerate(P.ks):
        p = P.pdict[k]
        if coupled_labels:
            palette = {id: c for id, c in zip(common_ids, util.N_colors(Ncommon))}
            cdf = pd.concat(
                [
                    pd.DataFrame(
                        util.boolean_indexing(
                            util.flatten_list(
                                [
                                    grouped_ds[id].get_par(k=k, key="end")
                                    for id in [
                                        f"{pair_id}_{common_id}"
                                        for common_id in common_ids
                                    ]
                                ]
                            )
                        ).T,
                        columns=common_ids,
                    ).assign(Trial=pair_id)
                    for pair_id in pair_ids
                ]
            )
        else:
            palette = {id: c for id, c in zip(gIDs, util.N_colors(P.Ngroups))}
            cdf = pd.concat(
                [
                    pd.DataFrame(
                        util.boolean_indexing(
                            util.flatten_list(P.datasets.get_par(k=k, key="end"))
                        ).T,
                        columns=gIDs,
                    ).assign(Trial=1)
                ]
            )  # CONCATENATE
        mdf = pd.melt(cdf, id_vars=["Trial"], var_name=["Group"])  # MELT

        kws = {
            "x": "Trial",
            "y": "value",
            "hue": "Group",
            "palette": palette,
            "data": mdf,
            "ax": P.axs[ii],
            "width": 0.5,
        }
        plot.single_boxplot(stripplot=False, annotation=False, **kws)
        P.conf_ax(ii, xlab=xlabel, ylab=p.l, ylim=p.lim)
    P.adjust((0.1, 0.95), (0.15, 0.9), 0.3, 0.3)
    return P.get()


@funcs.graph("PI (combo)")
def boxplot_PI(sort_labels=False, xlabel="Trials", **kwargs):
    P = plot.AutoPlot(name="PI_boxplot", figsize=(10, 5), **kwargs)

    group_ids = util.unique_list([d.config["group_id"] for d in P.datasets])
    Ngroups = len(group_ids)
    common_ids = util.unique_list([l.split("_")[-1] for l in group_ids])

    Ncommon = len(common_ids)
    pair_ids = util.unique_list([l.split("_")[0] for l in group_ids])

    Npairs = len(pair_ids)
    coupled_labels = True if Ngroups == Npairs * Ncommon else False

    if Npairs == 3 and all([l in pair_ids for l in ["Low", "Medium", "High"]]):
        pair_ids = ["Low", "Medium", "High"]
        xlabel = "Substate fructose concentration"
    elif Npairs == 3 and all([l in pair_ids for l in ["1:20", "1:200", "1:2000"]]):
        pair_ids = ["1:20", "1:200", "1:2000"]
        xlabel = "Odor concentration"
    if Ncommon == 2 and all([l in common_ids for l in ["AM", "EM"]]):
        common_ids = ["EM", "AM"]

    if sort_labels:
        common_ids = sorted(common_ids)
        pair_ids = sorted(pair_ids)

    all_PIs = []
    all_PIs_dict = {}
    for group_id in group_ids:
        group_ds = [d for d in P.datasets if d.config["group_id"] == group_id]
        PIdicts = [d.config["PI"] for d in group_ds]
        PIs = [dic["PI"] for dic in PIdicts]
        all_PIs.append(PIs)
        all_PIs_dict[group_id] = PIs

    if coupled_labels:
        colors = util.N_colors(Ncommon)
        palette = {id: c for id, c in zip(common_ids, colors)}
        pair_dfs = []
        for pair_id in pair_ids:
            paired_group_ids = [f"{pair_id}_{common_id}" for common_id in common_ids]
            pair_PIs = [all_PIs_dict[id] for id in paired_group_ids]
            pair_PI_array = util.boolean_indexing(pair_PIs).T
            pair_df = pd.DataFrame(pair_PI_array, columns=common_ids).assign(
                Trial=pair_id
            )
            pair_dfs.append(pair_df)
            cdf = pd.concat(pair_dfs)  # CONCATENATE

    else:
        colors = util.N_colors(Ngroups)
        palette = {id: c for id, c in zip(group_ids, colors)}
        PI_array = util.boolean_indexing(all_PIs).T
        df = pd.DataFrame(PI_array, columns=group_ids).assign(Trial=1)
        cdf = pd.concat([df])  # CONCATENATE
    mdf = pd.melt(cdf, id_vars=["Trial"], var_name=["Group"])  # MELT

    sns.boxplot(
        x="Trial",
        y="value",
        hue="Group",
        data=mdf,
        palette=palette,
        ax=P.axs[0],
        width=0.5,
        fliersize=3,
        linewidth=None,
        whis=1.0,
    )  # RUN PLOT
    P.conf_ax(xlab=xlabel, ylab="Odor preference", ylim=[-1, 1], leg_loc="lower left")
    P.adjust((0.2, 0.9), (0.15, 0.9), 0.05, 0.005)
    return P.get()


@funcs.graph("PI (simple)")
def PIboxplot(df, exp, save_to, ylabel, ylim=None, show=False, suf=""):
    f = f"{save_to}/{exp}{suf}.pdf"
    box = plt.boxplot(
        df,
        boxprops=dict(linestyle="-", linewidth=3),
        medianprops=dict(linestyle="-", linewidth=3),
        whiskerprops=dict(linestyle="-", linewidth=3),
        capprops=dict(linestyle="-", linewidth=3),
    )
    plt.suptitle(exp, fontsize=30)
    plt.xlabel("# training trials", fontsize=25)
    plt.ylabel(ylabel, fontsize=25)
    plt.ylim(ylim)
    plt.tick_params(labelsize=20)
    plt.tight_layout()
    plt.savefig(f, dpi=300)
    if show:
        plt.show()
    plt.close()


@funcs.graph("double patch", required={"ks": []})
def boxplot_double_patch(
    ks=None, xlabel="substrate", show_ns=False, stripplot=False, title=True, **kwargs
):
    if ks is None:
        ks = ["v_mu", "tur_N_mu", "pau_tr", "tur_H", "cum_d", "on_food_tr"]
    P = plot.AutoPlot(
        name="double_patch", Ncols=2, Nrows=3, figsize=(14 * 2, 8 * 3), **kwargs
    )
    RStexts = [r"$\bf{Rovers}$" + f" (N={P.N})", r"$\bf{Sitters}$" + f" (N={P.N})"]
    mIDs = ["rover", "sitter"]
    Cmods = dict(zip(mIDs, ["dark", "light"]))
    subIDs = util.unique_list([l.split("_")[0] for l in P.labels])
    Csubs = dict(zip(subIDs, ["green", "orange", "magenta"]))

    # ks = ['tur_tr', 'tur_N_mu', 'pau_tr', 'f_am', 'cum_d', 'on_food_tr']

    DataDic = util.AttrDict(
        {
            subID: {
                mID: {
                    "data": dict(P.data_dict)[f"{subID}_{mID}"],
                    "colors": [
                        f"xkcd:{Cmods[mID]} {Csubs[subID]}",
                        f"xkcd:{Cmods[mID]} cyan",
                    ],
                }
                for mID in mIDs
            }
            for subID in subIDs
        }
    )

    # Nmods = len(mIDs)
    # ks = ['v_mu', 'tur_N_mu', 'pau_tr', 'tur_H', 'cum_d', 'on_food_tr']
    def get_df(par, scale):
        pair_dfs = []
        for subID, RvSdic in DataDic.items():
            pair_vs = []
            for id, dic in RvSdic.items():
                vs = dic.data.endpoint_data[par].values * scale
                pair_vs.append(vs)
            pair_dfs.append(
                pd.DataFrame(util.boolean_indexing(pair_vs).T, columns=mIDs).assign(
                    Substrate=subID
                )
            )
        cdf = pd.concat(pair_dfs)  # CONCATENATE
        mdf = pd.melt(cdf, id_vars=["Substrate"], var_name=["Model"])  # MELT
        return mdf

    def get_df_onVSoff(par, scale):
        mdf_on = get_df(f"{par}_on_food", scale)
        mdf_off = get_df(f"{par}_off_food", scale)
        mdf_on["food"] = "on"
        mdf_off["food"] = "off"
        mdf = pd.concat([mdf_on, mdf_off])
        mdf.sort_index(inplace=True)
        mdf.sort_values(
            ["Model", "Substrate", "food"], ascending=[True, False, False], inplace=True
        )
        mdf["Substrate"] = mdf["Model"] + mdf["Substrate"]
        mdf.drop(["Model"], axis=1, inplace=True)
        return mdf

    def plot_p(data, ax, hue, agar=False):
        kws = {
            "x": "Substrate",
            "y": "value",
            "hue": hue,
            "data": data,
            "ax": ax,
            "color": "black",
        }
        plot.single_boxplot(stripplot=stripplot, show_ns=show_ns, width=0.5, **kws)
        cols = []
        if not agar:
            for subID, RvSdic in DataDic.items():
                for id, dic in RvSdic.items():
                    cols.append(dic.colors[0])
        else:
            for mID in mIDs:
                for subID, RvSdic in DataDic.items():
                    cols += RvSdic[mID].colors
            ax.set_xticklabels(subIDs * 2)
            ax.axvline(2.5, color="black", alpha=1.0, linestyle="dashed", linewidth=6)
            for x_text, text in zip([0.25, 0.75], RStexts):
                ax.text(
                    x_text,
                    1.1,
                    text,
                    ha="center",
                    va="top",
                    color="k",
                    fontsize=25,
                    transform=ax.transAxes,
                )
        for j, patch in enumerate(ax.artists):
            patch.set_facecolor(cols[j])

    reg.par.update_kdict(ks=ks)
    for ii, k in enumerate(ks):
        ax = P.axs[ii]
        p = reg.par.kdict[k]
        par = p.d
        ylab = p.label
        scale = 1
        if k in ["v_mu", "tur_N_mu", "pau_tr", "tur_H", "tur_tr"]:
            if k == "v_mu":
                ylab = "crawling speed (mm/s)"
                scale = 1000
            mdf = get_df_onVSoff(par, scale)
            plot_p(mdf, ax, "food", agar=True)
        else:
            if k == "cum_d":
                ylab = "pathlength (mm)"
                scale = 1000
            mdf = get_df(par, scale)
            plot_p(mdf, ax, "Model")
        P.conf_ax(ii, xlab=xlabel if ii > 3 else None, ylab=ylab, ylim=None)
    if title:
        dur = int(np.round(P.duration / 60))
        P.fig.suptitle(
            f"Double-patch experiment (duration = {dur} min)", size=40, weight="bold"
        )
    P.fig.align_ylabels(P.axs[:])
    P.adjust((0.1, 0.95), (0.15, 0.9), 0.3, 0.3)
    return P.get()


@funcs.graph("foraging", required={"ks": ["on_food_tr", "sf_am"]})
def plot_foraging(**kwargs):
    P = plot.AutoPlot(
        name="foraging",
        build_kws={"Ncols": 2, "w": 8, "h": 10, "sharex": True},
        **kwargs,
    )
    for j, k in enumerate(["on_food_tr", "sf_am"]):
        dfs = []
        for d in P.datasets:
            foodtypes = d.config["foodtypes"]
            dics = d.load_dicts("foraging")
            dic0 = {ft: [d[ft][k] for d in dics] for ft in foodtypes}
            df = pd.DataFrame.from_dict(dic0)
            df["Group"] = d.id
            dfs.append(df)
        df0 = pd.concat(dfs)
        par = reg.getPar(k, to_return="l")
        mdf = pd.melt(df0, id_vars=["Group"], var_name="foodtype", value_name=par)
        kws = {
            "x": "Group",
            "y": par,
            "hue": "foodtype",
            "palette": foodtypes,
            "data": mdf,
            "ax": P.axs[j],
            "width": 0.5,
        }
        plot.single_boxplot(stripplot=False, annotation=False, **kws)

        P.conf_ax(j, yMaxN=4, leg_loc="upper right")
    P.adjust((0.1, 0.95), (0.15, 0.92), 0.2, 0.005)
    P.get()


@funcs.graph("lineplot", required={"ks": []})
def lineplot(
    markers,
    ks=["f_am"],
    name=None,
    coupled_labels=None,
    xlabel=None,
    ylabel=None,
    leg_cols=None,
    scale=1.0,
    **kwargs,
):
    Npars = len(ks)
    if name is None:
        name = ks[0]

    P = plot.AutoPlot(
        name=name, build_kws={"N": Npars, "w": 8, "h": 7 / Npars}, **kwargs
    )
    Nds = P.Ndatasets

    if coupled_labels is not None:
        Npairs = len(coupled_labels)
        N = int(Nds / Npairs)
        if leg_cols is None:
            leg_cols = util.N_colors(N)
        leg_ids = P.labels[:N]
        ind = np.arange(Npairs)
        xticks, xticklabels = ind, coupled_labels
    else:
        ind = np.arange(Nds)
        leg_ids = P.labels
        N = Nds
        xticks, xticklabels = ind, P.labels

    # Pull the formatting out here
    plot_kws = {"linewidth": 2, "zorder": 5}
    err_kws = {
        "zorder": 2,
        "fmt": "none",
        "linewidth": 4,
        "ecolor": "k",
        "barsabove": True,
        "capsize": 10,
    }

    for ii, sh in enumerate(ks):
        ax = P.axs[ii]
        p, u = reg.getPar(sh, to_return=["d", "l"])
        vs = [d.e[p] * scale for d in P.datasets]
        means = [v.mean() for v in vs]
        stds = [v.std() for v in vs]
        for n, marker in zip(range(N), markers):
            ax.errs = ax.errorbar(ind, means[n::N], yerr=stds[n::N], **err_kws)
            ax.p1 = ax.plot(
                ind,
                means[n::N],
                marker=marker,
                label=leg_ids[n],
                markeredgecolor="black",
                markerfacecolor=leg_cols[n],
                markersize=8,
                **plot_kws,
            )

        if coupled_labels is None:
            for i, j in itertools.combinations(np.arange(Nds).tolist(), 2):
                st, pv = ttest_ind(vs[i], vs[j], equal_var=False)
                pv = np.round(pv, 4)
                plot.label_diff(i, j, f"p={pv}", ind, means, ax)
        else:
            for k in range(Npairs):
                i, j = k * N, k * N + 1
                st, pv = ttest_ind(vs[i], vs[j], equal_var=False)
                if pv <= 0.05:
                    ax.text(
                        ind[k],
                        np.max([means[i], means[j]]) + np.max([stds[i], stds[j]]),
                        "*",
                        ha="center",
                        fontsize=20,
                    )

        h = 2 * (np.nanmax(means) + np.nanmax(stds))
        P.conf_ax(
            ii,
            xlab=xlabel if xlabel is not None else None,
            ylab=u if ylabel is None else ylabel,
            ylim=[0, None],
            yMaxN=4,
            ytickMath=(-3, 3),
            leg_loc="upper right",
            xticks=xticks,
            xticklabels=xticklabels,
        )
    P.adjust((0.15, 0.95), (0.15, 0.95), H=0.05)
    return P.get()
