import itertools
import time

import numpy as np
from matplotlib.patches import Patch
from scipy.stats import ttest_ind

from lib.aux import dictsNlists as dNl, colsNstr as cNs

from lib.plot.aux import label_diff
from lib.plot.base import BasePlot, Plot, AutoLoadPlot, AutoPlot


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
    Nks = len(par_shorts)
    maxNrows = 3
    if Nks > maxNrows:
        Ncols = int(np.ceil(Nks / maxNrows))
        Nrows = maxNrows
        figsize = (Ncols * 6, Nrows * 5)
    else:
        Nrows = Nks
        Ncols = 1
        figsize = (9, 6)


    P = AutoPlot(name=par_shorts[0],Nrows=Nrows, Ncols=Ncols, figsize=figsize, **kwargs)
    # P.build(Nrows=Nrows, Ncols=Ncols, figsize=figsize)
    Nds = P.Ndatasets
    # Npars = len(par_shorts)
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


    for ii, sh in enumerate(par_shorts):
        ax = P.axs[ii]
        from lib.registry.pars import preg
        p, u = preg.getPar(sh, to_return=['d', 'l'])
        vs = [d.get_par(key='end', par=p) for d in P.datasets]
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

        P.conf_ax(ii, xlab=xlabel if xlabel is not None else None, ylab=u if ylabel is None else ylabel,
                  ylim=[0, None], yMaxN=4, ytickMath=(-3, 3), xticks=xticks, xticklabels=xticklabels)
    P.adjust((0.15, 0.95), (0.15, 0.95), H=0.05)
    P.fig.align_ylabels(P.axs[:])
    return P.get()




def auto_barplot(par_shorts, coupled_labels=None, xlabel=None, ylabel=None, leg_cols=None, **kwargs):
    Nks = len(par_shorts)
    maxNrows=3
    if Nks>maxNrows:
        Ncols=int(np.ceil(Nks/maxNrows))
        Nrows = maxNrows
        figsize = (Ncols * 6, Nrows * 5)
    else :
        Nrows = Nks
        Ncols = 1
        figsize = (9, 6)
    P = AutoLoadPlot(ks=par_shorts,name=par_shorts[0],Nrows=Nrows,Ncols=Ncols, figsize=figsize, **kwargs)
    Nds = P.Ndatasets

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
        ijs=[(kk * N, kk * N + 1) for kk in range(Npairs)]

        ij_pairs = ijs
        finfuncN = 2

    else:
        ind = np.arange(0, w * Nds, w)
        colors = P.colors
        leg_ids = P.labels
        xticks, xticklabels = ind, P.labels
        ijs =[]
        for i, j in itertools.combinations(np.arange(Nds).tolist(), 2):
            ijs.append((i,j))

        ij_pairs=ijs
        finfuncN=1


    bar_kwargs = {'width': w, 'color': colors, 'linewidth': 2, 'zorder': 5, 'align': 'center', 'edgecolor': 'black'}
    err_kwargs = {'zorder': 20, 'fmt': 'none', 'linewidth': 4, 'ecolor': 'k', 'barsabove': True, 'capsize': 10}

    for ii,k in enumerate(P.ks) :
        ax = P.axs[ii]
        dic,p=P.kpdict[k]
        vs=[ddic.df for l,ddic in dic.items()]
        means = [v.mean() for v in vs]
        stds = [v.std() for v in vs]
        ax.p1 = ax.bar(ind, means, **bar_kwargs)
        ax.errs = ax.errorbar(ind, means, yerr=stds, **err_kwargs)

        for i, j in ij_pairs:
            st, pv = ttest_ind(vs[i], vs[j], equal_var=False)
            pv = np.round(pv, 4)


            if finfuncN==1:
                label_diff(i, j, f'p={pv}', ind, means, P.axs[ii])

            elif finfuncN==2:
                if pv <= 0.05:
                    P.axs[ii].text(ind[i], means[i] + stds[i], '*', ha='center', fontsize=20)
                P.data_leg(ii, labels=leg_ids, colors=leg_cols, loc='upper left', handlelength=1, handleheight=1)

        P.conf_ax(ii, xlab=xlabel if xlabel is not None else None, ylab=p.label if ylabel is None else ylabel,
                  ylim=[0, None], yMaxN=4, ytickMath=(-3, 3), xticks=xticks, xticklabels=xticklabels)
    P.adjust((0.15, 0.95), (0.15, 0.95), W=0.1,H=0.1)
    P.fig.align_ylabels(P.axs[:])
    return P.get()





