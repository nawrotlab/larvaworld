import itertools

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import ttest_ind

from lib.aux import dictsNlists as dNl, data_aux, colsNstr as cNs
from lib.registry.pars import preg
from lib.plot.aux import label_diff

from lib.plot.base import AutoPlot, Plot


def boxplots(shorts=['l', 'v_mu'], key='end', Ncols=4, annotation=True, show_ns=True, grouped=False, ylims=None,
             in_mm=[], target_only=None, **kwargs):
    pars, labs, units, symbols = preg.getPar(shorts, to_return=['d', 'lab', 'unit', 'symbol'])
    Npars = len(pars)
    Ncols = Ncols
    Nrows = int(np.ceil(Npars / Ncols))

    P = AutoPlot(name=f'boxplot_{Npars}_{key}_pars', Ncols=Ncols, Nrows=Nrows, figsize=(8 * Ncols, 6 * Nrows),
                 sharex=True, **kwargs)

    group_ids = dNl.unique_list([d.config['group_id'] for d in P.datasets])
    Ngroups = len(group_ids)
    data = data_aux.concat_datasets(P.datasets, key=key)
    if not grouped:
        x = "DatasetID"
        hue = None
        palette = dict(zip(P.labels, P.colors))
        data = data[pars + [x]]
    else:
        x = "DatasetID"
        hue = 'GroupID'
        palette = dict(zip(group_ids, cNs.N_colors(Ngroups)))
        data = data[pars + [x, hue]]
    for sh in in_mm:
        data[preg.getPar(sh)] *= 1000

    for ii in range(Npars):
        kws = {
            'x': x,
            'y': pars[ii],
            'palette': palette,
            'hue': hue,
            'data': data,
            'ax': P.axs[ii],
            'width': 0.8,
            'fliersize': 3,
            'whis': 1.5,
            'linewidth': None
        }
        g1 = sns.boxplot(**kws)  # RUN PLOT
        try:
            g1.get_legend().remove()
        except:
            pass
        # print(pars[ii])
        if annotation:
            from lib.plot.aux import annotate_plot
            annotate_plot(show_ns=show_ns, target_only=target_only, **kws)
        P.conf_ax(ii, xticklabelrotation=30, ylab=labs[ii], yMaxN=4, ylim=ylims[ii] if ylims is not None else None,
                  xvis=False if ii < (Nrows - 1) * Ncols else True)

    P.adjust((0.1, 0.95), (0.15, 0.9), 0.5, 0.05)
    return P.get()


def boxplot(par_shorts, sort_labels=False, xlabel=None, pair_ids=None, common_ids=None, coupled_labels=None, **kwargs):
    P = Plot(name=par_shorts[0], **kwargs)
    pars, sim_labels, exp_labels, labs, lims = preg.getPar(par_shorts, to_return=['d', 's', 's', 'l', 'lim'])
    Npars = len(pars)

    group_ids = dNl.unique_list([d.config['group_id'] for d in P.datasets])
    Ngroups = len(group_ids)
    if common_ids is None:
        common_ids = dNl.unique_list([l.split('_')[-1] for l in group_ids])

    Ncommon = len(common_ids)
    if pair_ids is None:
        pair_ids = dNl.unique_list([l.split('_')[0] for l in group_ids])
    Npairs = len(pair_ids)
    if coupled_labels is None:
        coupled_labels = True if Ngroups == Npairs * Ncommon else False
    if sort_labels:
        common_ids = sorted(common_ids)
        pair_ids = sorted(pair_ids)
    if Npars > 3:
        Ncols = int(Npars / 2)
        Nrows = int(Npars / Ncols)
        P.build(Ncols=Ncols, Nrows=Nrows, figsize=(8 * Ncols, 7 * Nrows))
    else:
        P.build(Ncols=Npars, figsize=(7 * Npars, 6))
    for ii in range(Npars):

        p = pars[ii]
        ylabel = labs[ii]
        ylim = lims[ii]
        all_vs = []
        all_vs_dict = {}
        for group_id in group_ids:
            group_ds = [d for d in P.datasets if d.config['group_id'] == group_id]
            vs = [d.get_par(key='end', par=p) for d in group_ds]
            # vs = [d.endpoint_data[p].values for d in group_ds]
            all_vs.append(vs)
            all_vs_dict[group_id] = vs
        all_vs = dNl.flatten_list(all_vs)
        if coupled_labels:
            colors = cNs.N_colors(Ncommon)
            palette = {id: c for id, c in zip(common_ids, colors)}
            pair_dfs = []
            for pair_id in pair_ids:
                paired_group_ids = [f'{pair_id}_{common_id}' for common_id in common_ids]
                pair_vs = [all_vs_dict[id] for id in paired_group_ids]
                pair_vs = dNl.flatten_list(pair_vs)
                pair_array = data_aux.boolean_indexing(pair_vs).T
                pair_df = pd.DataFrame(pair_array, columns=common_ids).assign(Trial=pair_id)
                pair_dfs.append(pair_df)
                cdf = pd.concat(pair_dfs)  # CONCATENATE

        else:
            colors = cNs.N_colors(Ngroups)
            palette = {id: c for id, c in zip(group_ids, colors)}
            array = data_aux.boolean_indexing(all_vs).T
            df = pd.DataFrame(array, columns=group_ids).assign(Trial=1)
            cdf = pd.concat([df])  # CONCATENATE
        mdf = pd.melt(cdf, id_vars=['Trial'], var_name=['Group'])  # MELT

        g1 = sns.boxplot(x="Trial", y="value", hue='Group', data=mdf, palette=palette, ax=P.axs[ii], width=0.5,
                         fliersize=3, linewidth=None, whis=1.5)  # RUN PLOT

        g2 = sns.stripplot(x="Trial", y="value", hue='Group', data=mdf, palette=palette, ax=P.axs[ii])  # RUN PLOT
        P.conf_ax(ii, xlab=xlabel, ylab=ylabel, ylim=ylim)
    P.adjust((0.1, 0.95), (0.15, 0.9), 0.3, 0.3)
    return P.get()


def boxplot_PI(sort_labels=False, xlabel='Trials', **kwargs):
    P = AutoPlot(name='PI_boxplot', figsize=(10, 5), **kwargs)

    group_ids = dNl.unique_list([d.config['group_id'] for d in P.datasets])
    Ngroups = len(group_ids)
    common_ids = dNl.unique_list([l.split('_')[-1] for l in group_ids])

    Ncommon = len(common_ids)
    pair_ids = dNl.unique_list([l.split('_')[0] for l in group_ids])

    Npairs = len(pair_ids)
    coupled_labels = True if Ngroups == Npairs * Ncommon else False

    if Npairs == 3 and all([l in pair_ids for l in ['Low', 'Medium', 'High']]):
        pair_ids = ['Low', 'Medium', 'High']
        xlabel = 'Substate fructose concentration'
    elif Npairs == 3 and all([l in pair_ids for l in ['1:20', '1:200', '1:2000']]):
        pair_ids = ['1:20', '1:200', '1:2000']
        xlabel = 'Odor concentration'
    if Ncommon == 2 and all([l in common_ids for l in ['AM', 'EM']]):
        common_ids = ['EM', 'AM']

    if sort_labels:
        common_ids = sorted(common_ids)
        pair_ids = sorted(pair_ids)

    all_PIs = []
    all_PIs_dict = {}
    for group_id in group_ids:
        group_ds = [d for d in P.datasets if d.config['group_id'] == group_id]
        PIdicts = [d.config['PI'] for d in group_ds]
        PIs = [dic['PI'] for dic in PIdicts]
        all_PIs.append(PIs)
        all_PIs_dict[group_id] = PIs

    if coupled_labels:
        colors = cNs.N_colors(Ncommon)
        palette = {id: c for id, c in zip(common_ids, colors)}
        pair_dfs = []
        for pair_id in pair_ids:
            paired_group_ids = [f'{pair_id}_{common_id}' for common_id in common_ids]
            pair_PIs = [all_PIs_dict[id] for id in paired_group_ids]
            pair_PI_array = data_aux.boolean_indexing(pair_PIs).T
            pair_df = pd.DataFrame(pair_PI_array, columns=common_ids).assign(Trial=pair_id)
            pair_dfs.append(pair_df)
            cdf = pd.concat(pair_dfs)  # CONCATENATE

    else:
        colors = cNs.N_colors(Ngroups)
        palette = {id: c for id, c in zip(group_ids, colors)}
        PI_array = data_aux.boolean_indexing(all_PIs).T
        df = pd.DataFrame(PI_array, columns=group_ids).assign(Trial=1)
        cdf = pd.concat([df])  # CONCATENATE
    mdf = pd.melt(cdf, id_vars=['Trial'], var_name=['Group'])  # MELT

    sns.boxplot(x="Trial", y="value", hue="Group", data=mdf, palette=palette, ax=P.axs[0], width=.5,
                fliersize=3, linewidth=None, whis=1.0)  # RUN PLOT
    P.conf_ax(xlab=xlabel, ylab='Odor preference', ylim=[-1, 1], leg_loc='lower left')
    P.adjust((0.2, 0.9), (0.15, 0.9), 0.05, 0.005)
    return P.get()


def PIboxplot(df, exp, save_to, ylabel, ylim=None, show=False, suf=''):
    f = f'{save_to}/{exp}{suf}.pdf'
    box = boxplot(figsize=(10, 7), grid=False,
                  color=dict(boxes='k', whiskers='k', medians='b', caps='k'),
                  boxprops=dict(linestyle='-', linewidth=3),
                  medianprops=dict(linestyle='-', linewidth=3),
                  whiskerprops=dict(linestyle='-', linewidth=3),
                  capprops=dict(linestyle='-', linewidth=3)
                  )
    box.set_title(exp, fontsize=35)
    box.set_xlabel('# training trials', fontsize=25)
    box.set_ylabel(ylabel, fontsize=25)
    box.set_ylim(ylim)
    plt.tick_params(labelsize=20)
    plt.tight_layout()
    plt.savefig(f, dpi=300)
    if show:
        plt.show()
    plt.close()


def boxplot_double_patch(xlabel='substrate', complex_colors=True, **kwargs):
    P = Plot(name='double_patch', **kwargs)
    DataIDs = dNl.unique_list([d.config['group_id'] for d in P.datasets])
    ModIDs = dNl.unique_list([l.split('_')[-1] for l in DataIDs])
    subIDs = dNl.unique_list([l.split('_')[0] for l in DataIDs])
    Nmods = len(ModIDs)
    Csubs = dict(zip(subIDs, ['green', 'orange', 'magenta']))
    if Nmods == 2:
        temp = ['dark', 'light']
    elif Nmods == 3:
        temp = ['dark', 'light', '']
    Cmods = dict(zip(ModIDs, temp))

    shorts = ['v_mu', 'tur_N_mu', 'pau_tr', 'tur_H', 'cum_d', 'on_food_tr']
    pars, labs, lims = preg.getPar(shorts, to_return=['d', 'l', 'lim'])
    Npars = len(pars)

    P.build(Ncols=2, Nrows=3, figsize=(14 * 2, 8 * 3))
    for ii in range(Npars):
        sh = shorts[ii]
        p = pars[ii]
        ylabel = labs[ii]
        ylim = lims[ii]
        scale = 1
        onVSoff = True if sh in ['v_mu', 'tur_N_mu', 'pau_tr', 'tur_H'] else False
        if sh == 'cum_d':
            ylabel = "Pathlength 5' (mm)"
            scale = 1000
        elif sh == 'v_mu':
            ylabel = "Crawling speed (mm/s)"
            scale = 1000
        elif sh == 'tur_N_mu':
            ylabel = "Avg. number turns per min"
            scale = 60
        elif sh == 'pau_tr':
            ylabel = "Fraction of pauses"

        def get_df(p):
            dic = {id: [d.endpoint_data[p].values * scale for d in P.datasets if d.config['group_id'] == id] for id in
                   DataIDs}

            pair_dfs = []
            for subID in subIDs:
                subModIDs = [f'{subID}_{ModID}' for ModID in ModIDs]
                pair_vs = dNl.flatten_list([dic[id] for id in subModIDs])
                pair_dfs.append(pd.DataFrame(data_aux.boolean_indexing(pair_vs).T, columns=ModIDs).assign(Substrate=subID))
                cdf = pd.concat(pair_dfs)  # CONCATENATE
            mdf = pd.melt(cdf, id_vars=['Substrate'], var_name=['Model'])  # MELT
            return mdf

        def plot_p(data, ii, hue, agar=False):

            with sns.plotting_context('notebook', font_scale=1.4):
                kws = {
                    'x': "Substrate",
                    'y': "value",
                    'hue': hue,
                    'data': data,
                    'ax': P.axs[ii],
                    'width': 0.5,
                }
                g1 = sns.boxplot(**kws)  # RUN PLOT
                g1.get_legend().remove()
                from lib.plot.aux import annotate_plot
                annotate_plot(**kws)
                g1.set(xlabel=None)
                g2 = sns.stripplot(x="Substrate", y="value", hue=hue, data=data, color='black',
                                   ax=P.axs[ii])  # RUN PLOT
                g2.get_legend().remove()
                g2.set(xlabel=None)

                if complex_colors:
                    cols = []
                    if not agar:
                        for pID, cID in itertools.product(subIDs, ModIDs):
                            cols.append(f'xkcd:{Cmods[cID]} {Csubs[pID]}')
                    else:
                        for cID, pID in itertools.product(ModIDs, subIDs):
                            cols.append(f'xkcd:{Cmods[cID]} {Csubs[pID]}')
                            cols.append(f'xkcd:{Cmods[cID]} cyan')
                        P.axs[ii].set_xticklabels(subIDs * 2)
                        P.axs[ii].axvline(2.5, color='black', alpha=1.0, linestyle='dashed', linewidth=6)
                        P.axs[ii].text(0.25, 1.1, r'$\bf{Rovers}$', ha='center', va='top', color='k',
                                       fontsize=25, transform=P.axs[ii].transAxes)
                        P.axs[ii].text(0.75, 1.1, r'$\bf{Sitters}$', ha='center', va='top', color='k',
                                       fontsize=25, transform=P.axs[ii].transAxes)
                    for j, patch in enumerate(P.axs[ii].artists):
                        patch.set_facecolor(cols[j])
                P.conf_ax(ii, xlab=xlabel, ylab=ylabel, ylim=ylim)

        if not onVSoff:
            mdf = get_df(p)
            plot_p(mdf, ii, 'Model')
        else:
            mdf_on = get_df(f'{p}_on_food')
            mdf_off = get_df(f'{p}_off_food')
            mdf_on['food'] = 'on'
            mdf_off['food'] = 'off'
            mdf = pd.concat([mdf_on, mdf_off])
            mdf.sort_index(inplace=True)
            mdf.sort_values(['Model', 'Substrate', 'food'], ascending=[True, False, False], inplace=True)
            mdf['Substrate'] = mdf['Model'] + mdf['Substrate']
            mdf.drop(['Model'], axis=1, inplace=True)
            plot_p(mdf, ii, 'food', agar=True)
    P.fig.subplots_adjust(top=0.9, bottom=0.15, left=0.1, right=0.95, hspace=0.3, wspace=0.3)
    return P.get()


def ggboxplot(shorts=['l', 'v_mu'], key='end', figsize=(12, 6), subfolder=None, **kwargs):
    pars, syms, labs, lims = preg.getPar(shorts, to_return=['d', 's', 'lab', 'lim'])
    from plotnine import ggplot, aes, geom_boxplot, scale_color_manual, theme
    Npars = len(pars)
    if Npars == 1:
        name = pars[0]
    else:
        name = f'ggboxplot_{len(pars)}_end_pars'
    P = Plot(name=name, subfolder=subfolder, **kwargs)
    e = data_aux.concat_datasets(P.datasets, key=key)
    Cdict = dict(zip(P.labels, P.colors))
    ggs = [ggplot(e, aes(x='DatasetID', y=p, color='DatasetID')) for p in pars]
    if Npars == 1:
        P.fig = (ggs[0] + geom_boxplot() + scale_color_manual(Cdict) + theme(
            figure_size=figsize)).draw()
    else:

        P.fig = (ggs[0] + geom_boxplot() + scale_color_manual(
            Cdict) + theme(
            figure_size=figsize)).draw()
    return P.get()


def plot_foraging(**kwargs):
    P = Plot(name='foraging', **kwargs)
    P.build(1, 2, figsize=(15, 10), sharex=True)
    for j, action in enumerate(['on_food_tr', 'sf_am']):
        dfs = []
        for i, d in enumerate(P.datasets):
            foodtypes = d.config['foodtypes']
            dics = d.load_dicts('foraging')
            dic0 = {ft: [d[ft][action] for d in dics] for ft in foodtypes.keys()}
            df = pd.DataFrame.from_dict(dic0)
            df['Group'] = d.id
            dfs.append(df)
        df0 = pd.concat(dfs)
        par = preg.getPar(action, to_return='lab')
        mdf = pd.melt(df0, id_vars=['Group'], var_name='foodtype', value_name=par)
        with sns.plotting_context('notebook', font_scale=1.4):
            kws = {
                'x': "Group",
                'y': par,
                'hue': 'foodtype',
                'palette': foodtypes,
                'data': mdf,
                'ax': P.axs[j],
                'width': 0.5,
            }
            g1 = sns.boxplot(**kws)
            P.conf_ax(yMaxN=4, leg_loc='upper right')
    P.adjust((0.1, 0.95), (0.15, 0.92), 0.2, 0.005)
    P.get()


def lineplot(markers, par_shorts=['f_am'], coupled_labels=None, xlabel=None, ylabel=None, leg_cols=None, scale=1.0,
             **kwargs):
    Npars = len(par_shorts)
    P = AutoPlot(name=par_shorts[0], Nrows=Npars, figsize=(8, 7), **kwargs)
    Nds = P.Ndatasets

    if coupled_labels is not None:
        Npairs = len(coupled_labels)
        N = int(Nds / Npairs)
        if leg_cols is None:
            leg_cols = cNs.N_colors(N)
        leg_ids = P.labels[:N]
        ind = np.arange(Npairs)
        xticks, xticklabels = ind, coupled_labels
    else:
        ind = np.arange(Nds)
        leg_ids = P.labels
        N = Nds
        xticks, xticklabels = ind, P.labels

    # Pull the formatting out here
    plot_kws = {'linewidth': 2, 'zorder': 5}
    err_kws = {'zorder': 2, 'fmt': 'none', 'linewidth': 4, 'ecolor': 'k', 'barsabove': True, 'capsize': 10}

    for ii, sh in enumerate(par_shorts):
        ax = P.axs[ii]
        p, u = preg.getPar(sh, to_return=['d', 'l'])
        vs = [d.endpoint_data[p] * scale for d in P.datasets]
        means = [v.mean() for v in vs]
        stds = [v.std() for v in vs]
        for n, marker in zip(range(N), markers):
            ax.errs = ax.errorbar(ind, means[n::N], yerr=stds[n::N], **err_kws)
            ax.p1 = ax.plot(ind, means[n::N], marker=marker, label=leg_ids[n],
                            markeredgecolor='black', markerfacecolor=leg_cols[n], markersize=8, **plot_kws)

        if coupled_labels is None:
            for i, j in itertools.combinations(np.arange(Nds).tolist(), 2):
                st, pv = ttest_ind(vs[i], vs[j], equal_var=False)
                pv = np.round(pv, 4)
                label_diff(i, j, f'p={pv}', ind, means, ax)
        else:
            for k in range(Npairs):
                i, j = k * N, k * N + 1
                st, pv = ttest_ind(vs[i], vs[j], equal_var=False)
                if pv <= 0.05:
                    ax.text(ind[k], np.max([means[i], means[j]]) + np.max([stds[i], stds[j]]), '*', ha='center',
                            fontsize=20)

        h = 2 * (np.nanmax(means) + np.nanmax(stds))
        P.conf_ax(ii, xlab=xlabel if xlabel is not None else None, ylab=u if ylabel is None else ylabel, ylim=[0, None],
                  yMaxN=4, ytickMath=(-3, 3), leg_loc='upper right', xticks=xticks, xticklabels=xticklabels)
    P.adjust((0.15, 0.95), (0.15, 0.95), H=0.05)
    return P.get()



