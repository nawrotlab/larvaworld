import itertools

import numpy as np
from matplotlib.patches import Patch
from scipy.stats import ttest_ind

from lib.aux import dictsNlists as dNl, colsNstr as cNs

from lib.plot.aux import label_diff
from lib.plot.base import BasePlot, Plot


def error_barplot(error_dict, evaluation, axs=None, fig=None, labels=None, name='error_barplots',
                  titles=[r'$\bf{endpoint}$ $\bf{metrics}$', r'$\bf{timeseries}$ $\bf{metrics}$'], **kwargs):
    def build_legend(ax, eval_df):
        h, l = ax.get_legend_handles_labels()
        empty = Patch(color='none')
        counter = 0
        for g in eval_df.index:
            h.insert(counter, empty)
            l.insert(counter, eval_df['group_label'].loc[g])
            counter += (len(eval_df['shorts'].loc[g]) + 1)
        ax.legend(h, l, loc='upper left', bbox_to_anchor=(1.0, 1.0), fontsize=15)

    P = BasePlot(name=name, **kwargs)
    Nplots = len(error_dict)
    P.build(Nplots, 1, figsize=(20, Nplots * 6), sharex=False, fig=fig, axs=axs)
    P.adjust((0.07, 0.7), (0.05, 0.95), 0.05, 0.2)
    for ii, (k, eval_df) in enumerate(evaluation.items()):
        lab = labels[k] if labels is not None else k
        # ax = P.axs[ii] if axs is None else axs[ii]
        df = error_dict[k]
        color = dNl.flatten_list(eval_df['par_colors'].values.tolist())
        df = df[dNl.flatten_list(eval_df['symbols'].values.tolist())]
        df.plot(kind='bar', ax=P.axs[ii], ylabel=lab, rot=0, legend=False, color=color, width=0.6)
        build_legend(P.axs[ii], eval_df)
        P.conf_ax(ii, title=titles[ii], xlab='', yMaxN=4)
    return P.get()


def intake_barplot(**kwargs):
    return barplot(par_shorts=['f_am'], **kwargs)


def barplot(par_shorts, coupled_labels=None, xlabel=None, ylabel=None, leg_cols=None, **kwargs):
    P = Plot(name=par_shorts[0], **kwargs)
    Nds = P.Ndatasets
    Npars = len(par_shorts)
    w = 0.15

    if coupled_labels is not None:
        Npairs = len(coupled_labels)
        N = int(Nds / Npairs)
        if leg_cols is None:
            leg_cols = cNs.N_colors(N)
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

    bar_kwargs = {'width': w, 'color': colors, 'linewidth': 2, 'zorder': 5, 'align': 'center', 'edgecolor': 'black'}
    err_kwargs = {'zorder': 20, 'fmt': 'none', 'linewidth': 4, 'ecolor': 'k', 'barsabove': True, 'capsize': 10}
    P.build(Npars, 1, figsize=(9, 6))
    for ii, sh in enumerate(par_shorts):
        ax = P.axs[ii]
        from lib.conf.pars.pars import getPar
        p, u = getPar(sh, to_return=['d', 'l'])
        vs = [d.get_par(key='end', par=p) for d in P.datasets]
        # vs = [d.endpoint_data[p] for d in P.datasets]
        means = [v.mean() for v in vs]
        stds = [v.std() for v in vs]
        ax.p1 = ax.bar(ind, means, **bar_kwargs)
        ax.errs = ax.errorbar(ind, means, yerr=stds, **err_kwargs)

        if not coupled_labels:
            for i, j in itertools.combinations(np.arange(Nds).tolist(), 2):
                st, pv = ttest_ind(vs[i], vs[j], equal_var=False)
                pv = np.round(pv, 4)
                label_diff(i, j, f'p={pv}', ind, means, ax)
        else:
            for k in range(Npairs):
                i, j = k * N, k * N + 1
                st, pv = ttest_ind(vs[i], vs[j], equal_var=False)
                if pv <= 0.05:
                    ax.text(ind[i], means[i] + stds[i], '*', ha='center', fontsize=20)
            P.data_leg(ii,labels=leg_ids, colors=leg_cols, loc='upper left', handlelength=1, handleheight=1)
            # dataset_legend(leg_ids, leg_cols, ax=ax, loc='upper left', handlelength=1, handleheight=1)

        h = 2 * (np.nanmax(means) + np.nanmax(stds))
        P.conf_ax(ii, xlab=xlabel if xlabel is not None else None, ylab=u if ylabel is None else ylabel,
                  ylim=[0, None], yMaxN=4, ytickMath=(-3, 3), xticks=xticks, xticklabels=xticklabels)
    P.adjust((0.15, 0.95), (0.15, 0.95), H=0.05)
    return P.get()




