import itertools
import os
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import collections as mc, pyplot as plt, ticker, cm, patches

from scipy import signal
from scipy.stats import ttest_ind, multivariate_normal



from lib.aux import naming as nam, dictsNlists as dNl
from lib.aux.colsNstr import N_colors, col_range
from lib.conf.pars.pars import getPar, ParDict
from lib.conf.pars.units import ureg
from lib.plot.base import Plot, AutoPlot, ParPlot, BasePlot
from lib.plot.aux import dataset_legend, label_diff, annotate_plot, plot_quantiles, \
    plot_mean_and_range, process_plot, circNarrow, circular_hist, scatter_hist, suf
from lib.aux.data_aux import boolean_indexing, concat_datasets



def plot_ethogram(subfolder='timeplots', **kwargs):
    P = Plot(name='ethogram', subfolder=subfolder, **kwargs)
    P.build(P.Ndatasets, 2, sharex=True)
    Cbouts = {
        'lin': {'stridechain': 'green',
                'pause': 'red',
                'feedchain': 'blue'},
        'ang': {'Lturn': 'cyan',
                'Rturn': 'orange'}

    }
    for i, d in enumerate(P.datasets):
        N = d.config['N']
        try:
            s = d.step_data
        except:
            s = d.read('step')
        for k, (n, title) in enumerate(zip(['lin', 'ang'], [r'$\bf{runs & pauses}$', r'$\bf{left & right turns}$'])):
            idx = 2 * i + k
            ax = P.axs[idx]
            P.conf_ax(idx, xlab='time $(sec)$', ylab='Individuals $(idx)$', ylim=(0, N + 2),
                      xlim=(0, d.config['Nticks'] * d.dt), title=title if i == 0 else None)
            for b, c in Cbouts[n].items():
                bp0, bp1 = nam.start(b), nam.stop(b)
                if not {bp0, bp1}.issubset(s.columns.values):
                    continue
                for j, id in enumerate(s.index.unique('AgentID').values):
                    bbs = s[[bp0, bp1]].xs(id, level='AgentID')
                    b0s = bbs[bp0].dropna().index.values * d.dt
                    b1s = bbs[bp1].dropna().index.values * d.dt
                    lines = [[(b0, j + 1), (b1, j + 1)] for b0, b1 in zip(b0s, b1s)]
                    lc = mc.LineCollection(lines, colors=c, linewidths=2)
                    ax.add_collection(lc)
            dataset_legend(labels=list(Cbouts[n].keys()), colors=list(Cbouts[n].values()), ax=ax,
                           loc=None, anchor=None, fontsize=None, handlelength=0.5, handleheight=0.5)

    P.adjust((0.1, 0.95), (0.15, 0.92), 0.15, 0.1)
    return P.get()


def plot_2pars(shorts, subfolder='step', larva_legend=True, **kwargs):
    ypar, ylab, ylim = getPar(shorts[1], to_return=['d', 'l', 'lim'])
    xpar, xlab, xlim = getPar(shorts[0], to_return=['d', 'l', 'lim'])
    P = Plot(name=f'{ypar}_VS_{xpar}', subfolder=subfolder, **kwargs)
    P.build()
    ax = P.axs[0]
    if P.Ndatasets == 1 and larva_legend:
        d = P.datasets[0]
        Nids = len(d.agent_ids)
        cs = N_colors(Nids)
        s = d.read('step')
        for j, id in enumerate(d.agent_ids):
            ss = s.xs(id, level='AgentID', drop_level=True)
            ax.scatter(ss[xpar], ss[ypar], color=cs[j], marker='.', label=id)
            ax.legend()
    else:
        for d, c in zip(P.datasets, P.colors):
            s = d.read('step')
            ax.scatter(s[xpar], s[ypar], color=c, marker='.')
        dataset_legend(P.labels, P.colors, ax=ax, loc='upper left')
    P.conf_ax(xlab=xlab, ylab=ylab, xlim=xlim, ylim=ylim, xMaxN=4, yMaxN=4)
    P.adjust((0.15, 0.95), (0.15, 0.92), 0.05, 0.005)
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
            leg_cols = N_colors(N)
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
            dataset_legend(leg_ids, leg_cols, ax=ax, loc='upper left', handlelength=1, handleheight=1)

        h = 2 * (np.nanmax(means) + np.nanmax(stds))
        P.conf_ax(ii, xlab=xlabel if xlabel is not None else None, ylab=u if ylabel is None else ylabel,
                  ylim=[0, None], yMaxN=4, ytickMath=(-3, 3), xticks=xticks, xticklabels=xticklabels)
    P.adjust((0.15, 0.95), (0.15, 0.95), H=0.05)
    return P.get()


def odorscape_from_config(c, mode='2D', fig=None, axs=None, show=True, grid_dims=(201, 201), col_max=(0, 0, 0),
                          **kwargs):
    env = c.env_params
    source = list(env.food_params.source_units.values())[0]
    a0, b0 = source.pos
    oP, oS = source.odor.odor_intensity, source.odor.odor_spread
    oD = multivariate_normal([0, 0], [[oS, 0], [0, oS]])
    oM = oP / oD.pdf([0, 0])
    if col_max is None:
        col_max = source.default_color if source.default_color is not None else (0, 0, 0)
    if grid_dims is not None:
        X, Y = grid_dims
    else:
        X, Y = [51, 51] if env.odorscape.grid_dims is None else env.odorscape.grid_dims
    Xdim, Ydim = env.arena.arena_dims
    s = 1
    Xmesh, Ymesh = np.meshgrid(np.linspace(-Xdim * s / 2, Xdim * s / 2, X), np.linspace(-Ydim * s / 2, Ydim * s / 2, Y))

    @np.vectorize
    def func(a, b):
        return oD.pdf([a - a0, b - b0]) * oM

    grid = func(Xmesh, Ymesh)

    if mode == '2D':
        if fig is None and axs is None:
            fig, axs = plt.subplots(1, 1, figsize=(10, 10 * Ydim / Xdim))
        q = grid.flatten() - np.min(grid)
        q /= np.max(q)
        cols = col_range(q, low=(255, 255, 255), high=col_max, mul255=False)
        x, y = Xmesh * 1000 / s, Ymesh * 1000 / s,
        axs.scatter(x=x, y=y, color=cols)
        axs.set_aspect('equal', adjustable='box')
        axs.set_xlim([np.min(x), np.max(x)])
        axs.set_ylim([np.min(y), np.max(y)])
        axs.set_xlabel(r'X $(mm)$')
        axs.set_ylabel(r'Y $(mm)$')
        if show:
            plt.show()
    elif mode == '3D':
        return plot_surface(x=Xmesh * 1000 / s, y=Ymesh * 1000 / s, z=grid, vars=[r'X $(mm)$', r'Y $(mm)$'],
                            target=r'concentration $(μM)$', save_as=f'odorscape', show=show, fig=fig, ax=axs, azim=0,
                            elev=0)


def odorscape_with_sample_tracks(datasets, unit='mm', fig=None, axs=None, show=False, save_to=None, **kwargs):
    scale = 1000 if unit == 'mm' else 1
    if fig is None and axs is None:
        fig, axs = plt.subplots(1, 1, figsize=(20, 20))
    odorscape_from_config(datasets[0].config, mode='2D', fig=fig, axs=axs, show=False, **kwargs)
    for d in datasets:
        s, c = d.step_data, d.config
        xy = s[['x', 'y']].xs(c.agent_ids[0], level="AgentID").values * scale
        axs.plot(xy[:, 0], xy[:, 1], label=c.id, color=c.color)
    axs.legend(loc='upper left', fontsize=15)
    if show:
        plt.show()
    return fig


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
    pars, labs, lims = getPar(shorts, to_return=['d', 'l', 'lim'])
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
                pair_dfs.append(pd.DataFrame(boolean_indexing(pair_vs).T, columns=ModIDs).assign(Substrate=subID))
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


def plot_nengo_network(group=None, probes=None, same_plot=False, subfolder='nengo', **kwargs):
    probe_groups = {
        'anemotaxis': ['Ch', 'LNa', 'LNb', 'Ha', 'Hb', 'B1', 'B2', 'Bend', 'Hunch'],
        'frequency': ['linFrIn', 'angFrIn', 'linFr', 'angFr'],
        'frequency_x3': ['linFrIn', 'angFrIn', 'feeFrIn', 'linFr', 'angFr', 'feeFr'],
        'velocity': ['Vs', 'linV', 'angV'],
        'velocity_x3': ['Vs', 'linV', 'angV', 'feeV'],
        'interference': ['Vs', 'interference'],
        'crawler': ['linFrIn', 'linFr', 'linV'],
        'turner': ['angFrIn', 'angFr', 'angV'],
        'feeder': ['feeFrIn', 'feeFr', 'feeV'],
        'feeding': ['feeFrIn', 'feeFr', 'feeV', 'f_cur', 'f_suc'],
        'wind_effect_on_V': ['Bend', 'Hunch', 'linV', 'angV'],
        'wind_effect_on_Fr': ['Bend', 'Hunch', 'linFr', 'angFr'],
    }
    if group is not None:
        probes = probe_groups[group]
        name = f'{group}_network'
    elif probes is None:
        raise ValueError('Either a probe group or individual probes have to be defined')
    else:
        name = f'{probes[0]}_VS_{probes[1]}'
    N = len(probes)
    Cprobes = N_colors(N)
    P = Plot(name=name, subfolder=subfolder, **kwargs)
    Nds = P.Ndatasets
    Nids = np.max([len(d.agent_ids) for d in P.datasets])
    Nticks = np.max([d.num_ticks for d in P.datasets])
    dt = np.max([d.dt for d in P.datasets])
    x = np.linspace(0, (Nticks * dt) / 60, Nticks)
    if same_plot:
        Nrows = Nds
        sharey = False
        yMaxN = 8
    else:
        Nrows = N * Nds
        sharey = False
        yMaxN = 3
    P.build(Nrows, Nids, figsize=(Nids * 30, Nrows * 15), sharex=True, sharey=sharey)
    for i, d in enumerate(P.datasets):
        dics = d.load_dicts('nengo')
        for j, dic in enumerate(dics):
            for k, (p, c) in enumerate(zip(probes, Cprobes)):
                Nrow = i if same_plot else i * Nds + k
                idx = j + Nrow * Nids
                y = np.array(dic[p])
                dim = y.shape[1]
                if dim == 1:
                    P.axs[idx].plot(x, y, color=c, label=p)
                else:
                    for jj in range(dim):
                        P.axs[idx].plot(x, y[:, jj], label=f'{p}_{jj}')
                P.conf_ax(idx, xlab=r'time $min$' if Nrow == Nrows - 1 else None, ylab='activity' if j == 0 else None,
                          yticks=[] if j != 0 else None, yticklabels=[] if j != 0 else None, yMaxN=yMaxN,
                          leg_loc='upper right')
    P.adjust((0.1, 0.95), (0.1, 0.95), 0.01, 0.05)
    return P.get()


def ggboxplot(shorts=['l', 'v_mu'], key='end', figsize=(12, 6), subfolder=None, **kwargs):
    pars, syms, labs, lims = getPar(shorts, to_return=['d', 's', 'lab', 'lim'])
    from plotnine import ggplot, aes, geom_boxplot, scale_color_manual, theme
    Npars = len(pars)
    if Npars == 1:
        name = pars[0]
    else:
        name = f'ggboxplot_{len(pars)}_end_pars'
    P = Plot(name=name, subfolder=subfolder, **kwargs)
    e = concat_datasets(P.datasets, key=key)
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
        par = getPar(action, to_return='lab')
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


def timeplot(par_shorts=[], pars=[], same_plot=True, individuals=False, table=None, unit='sec', absolute=True,
             show_legend=True, show_first=False, subfolder='timeplots', legend_loc='upper left', leg_fontsize=15,
             figsize=(7.5, 5),
             **kwargs):
    unit_coefs = {'sec': 1, 'min': 1 / 60, 'hour': 1 / 60 / 60}
    if len(pars) == 0:
        if len(par_shorts) == 0:
            raise ValueError('Either parameter names or shortcuts must be provided')
        else:
            pars, symbols, ylabs, ylims, ylabs0 = getPar(par_shorts, to_return=['d', 's', 'l', 'lim', 'lab'])

            # ylabs=[]
            # for ii in range(len(pars)) :
            #     if ylabs0[ii] is not None :
            #         ylabs.append(ylabs0[ii])
            #     else :
            #         ylabs.append(ylabs1[ii])
            # print(ylabs1, ylabs0, ylabs)
    else:
        symbols = pars
        ylabs = pars
        ylabs0 = pars
        # ylabs0 = pars
        ylims = [None] * len(pars)
    # pars=[p for p in pars if all([p in ])]
    N = len(pars)
    cols = ['grey'] if N == 1 else N_colors(N)
    if not same_plot:
        raise NotImplementedError
    if N == 1:
        name = f'{pars[0]}'
    elif N == 2:
        name = f'{pars[0]}_VS_{pars[1]}'
    else:
        name = f'{N}_pars'
    P = AutoPlot(name=name, subfolder=subfolder, figsize=figsize, **kwargs)

    ax = P.axs[0]
    counter = 0
    for p, symbol, ylab0, ylab, ylim, c in zip(pars, symbols, ylabs0, ylabs, ylims, cols):
        if ylab0 is not None:
            ylab = ylab0
        P.conf_ax(xlab=f'time, ${unit}$' if table is None else 'timesteps', ylab=ylab, ylim=ylim, yMaxN=4)
        for d, d_col, d_lab in zip(P.datasets, P.colors, P.labels):
            if P.Ndatasets > 1:
                c = d_col
            try:
                if table is not None:
                    dc = d.load_table(table)[p]
                else:
                    dc = d.get_par(p, key='step')
                if absolute:
                    dc = dc.abs()
                    # dc = d.read('step')[p]
                dc_m = dc.groupby(level='Step').quantile(q=0.5)
                Nticks = len(dc_m)
                x = np.linspace(0, int(Nticks / d.fr) * unit_coefs[unit], Nticks) if table is None else np.arange(
                    Nticks)
                ax.set_xlim([x[0], x[-1]])

                if individuals:
                    for id in dc.index.get_level_values('AgentID'):
                        dc_single = dc.xs(id, level='AgentID')
                        ax.plot(x, dc_single, color=c, linewidth=1)
                    ax.plot(x, dc_m, color=c, linewidth=2)
                else:
                    plot_quantiles(df=dc, x=x, axis=ax, color_shading=c, label=symbol, linewidth=2)
                    if show_first:
                        dc0 = dc.xs(dc.index.get_level_values('AgentID')[0], level='AgentID')
                        ax.plot(x, dc0, color=c, linestyle='dashed', linewidth=1)
                counter += 1
            except:
                pass
    if counter == 0:
        raise ValueError('None of the parameters exist in any dataset')
    if N > 1:
        ax.legend()
    if P.Ndatasets > 1 and show_legend:
        dataset_legend(P.labels, P.colors, ax=ax, loc=legend_loc, fontsize=leg_fontsize)
    P.adjust((0.1, 0.95), (0.15, 0.95))
    return P.get()


def plot_odor_concentration(**kwargs):
    return timeplot(['c_odor1'], **kwargs)


def plot_sensed_odor_concentration(**kwargs):
    return timeplot(['dc_odor1'], **kwargs)


def plot_Y_pos(**kwargs):
    return timeplot(['y'], **kwargs)


def plot_pathlength(scaled=True, unit='mm', xlabel=None, **kwargs):
    lab = 'pathlength'
    if scaled:
        name = f'scaled_{lab}'
        ylab = f'scaled {lab} $(-)$'
    else:
        name = f'{lab}'
        ylab = f'{lab} $({unit})$'
    P = Plot(name=name, **kwargs)
    if xlabel is None:
        xlabel = 'time, $min$'
    P.build(figsize=(7, 6))

    dst_par, dst_u = getPar('cum_d', to_return=['d', 'u'])
    x = P.trange()
    for d, lab, c in zip(P.datasets, P.labels, P.colors):
        df = d.step_data[dst_par]
        if not scaled and unit == 'cm':

            if dst_u == ureg.m:
                df *= 100
        plot_quantiles(df=df, x=x, axis=P.axs[0], color_shading=c, label=lab)

    P.conf_ax(xlab=xlabel, ylab=ylab, xlim=(x[0], x[-1]), ylim=[0, None], xMaxN=5, leg_loc='upper left')
    P.adjust((0.2, 0.95), (0.15, 0.95), 0.05, 0.005)
    return P.get()


def plot_gut(**kwargs):
    P = AutoPlot(name='gut', **kwargs)
    x = P.trange()
    for d, l, c in zip(P.datasets, P.labels, P.colors):
        df = d.step_data['gut_occupancy'] * 100
        plot_quantiles(df=df, x=x, axis=P.axs[0], color_shading=c, label=l)
    P.conf_ax(xlab='time, $min$', ylab='% gut occupied',
              xlim=(x[0], x[-1]), ylim=[0, 100], xMaxN=5, yMaxN=5, leg_loc='upper left')
    P.adjust((0.1, 0.95), (0.15, 0.95), 0.05, 0.005)
    return P.get()


def plot_food_amount(filt_amount=False, scaled=False, **kwargs):
    name = 'food_intake'
    ylab = r'Cumulative food intake $(mg)$'
    par = 'ingested_food_volume'
    if scaled:
        name = f'scaled_{name}'
        ylab = r'Cumulative food intake as % larval mass'
        par = 'ingested_body_mass_ratio'
    if filt_amount:
        name = f'filtered_{name}'
        ylab = r'Food intake $(mg)$'
    if filt_amount and scaled:
        ylab = 'Food intake as % larval mass'
    P = AutoPlot(name=name, **kwargs)

    for d, lab, c in zip(P.datasets, P.labels, P.colors):
        dst_df = d.step_data[par]
        dst_m = dst_df.groupby(level='Step').quantile(q=0.5)
        dst_u = dst_df.groupby(level='Step').quantile(q=0.75)
        dst_b = dst_df.groupby(level='Step').quantile(q=0.25)
        if filt_amount:
            sos = signal.butter(N=1, Wn=0.1, btype='lowpass', analog=False, fs=P.Nticks / P.tlim[1], output='sos')
            dst_m = dst_m.diff()
            dst_m.iloc[0] = 0
            dst_m = signal.sosfiltfilt(sos, dst_m)
            dst_u = dst_u.diff()
            dst_u.iloc[0] = 0
            dst_u = signal.sosfiltfilt(sos, dst_u)
            dst_b = dst_b.diff()
            dst_b.iloc[0] = 0
            dst_b = signal.sosfiltfilt(sos, dst_b)
        x = P.trange()
        plot_mean_and_range(x=x, mean=dst_m, lb=dst_b, ub=dst_u, axis=P.axs[0], color_shading=c, label=lab)
    P.conf_ax(xlab='time, $min$', ylab=ylab, xlim=(x[0], x[-1]), xMaxN=5, leg_loc='upper left')
    P.adjust((0.1, 0.95), (0.15, 0.95), 0.05, 0.005)
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
        colors = N_colors(Ncommon)
        palette = {id: c for id, c in zip(common_ids, colors)}
        pair_dfs = []
        for pair_id in pair_ids:
            paired_group_ids = [f'{pair_id}_{common_id}' for common_id in common_ids]
            pair_PIs = [all_PIs_dict[id] for id in paired_group_ids]
            pair_PI_array = boolean_indexing(pair_PIs).T
            pair_df = pd.DataFrame(pair_PI_array, columns=common_ids).assign(Trial=pair_id)
            pair_dfs.append(pair_df)
            cdf = pd.concat(pair_dfs)  # CONCATENATE

    else:
        colors = N_colors(Ngroups)
        palette = {id: c for id, c in zip(group_ids, colors)}
        PI_array = boolean_indexing(all_PIs).T
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


def lineplot(markers, par_shorts=['f_am'], coupled_labels=None, xlabel=None, ylabel=None, leg_cols=None, scale=1.0,
             **kwargs):
    Npars = len(par_shorts)
    P = AutoPlot(name=par_shorts[0], Nrows=Npars, figsize=(8, 7), **kwargs)
    Nds = P.Ndatasets

    if coupled_labels is not None:
        Npairs = len(coupled_labels)
        N = int(Nds / Npairs)
        if leg_cols is None:
            leg_cols = N_colors(N)
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
        p, u = getPar(sh, to_return=['d', 'l'])
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


def plot_debs(deb_dicts=None, save_to=None, save_as=None, mode='full', roversVSsitters=False, include_egg=True,
              time_unit='hours', return_fig=False, sim_only=False, force_ymin=None, color_epoch_quality=True,
              datasets=None, labels=None, show=False, label_epochs=True, label_lifestages=True, **kwargs):
    warnings.filterwarnings('ignore')
    if save_to is None:
        from lib.conf.base import paths
        save_to = paths.path('DEB')
    os.makedirs(save_to, exist_ok=True)
    if save_as is None:
        save_as = f'debs.{suf}'
    if deb_dicts is None:
        deb_dicts = dNl.flatten_list([d.load_dicts('deb') for d in datasets])
    Ndebs = len(deb_dicts)
    ids = [d['id'] for d in deb_dicts]
    if Ndebs == 1:
        cols = [(0, 1, 0.1)]
        leg_ids = ids
        leg_cols = cols
    elif roversVSsitters:
        cols = []
        temp_id = None
        for id in ids:
            if str.startswith(id, 'Rover'):
                cols.append((0, 0, 1))
            elif str.startswith(id, 'Sitter'):
                cols.append((1, 0, 0))
            else:
                cols.append((0, 1, 0.1))
                temp_id = id
        if temp_id is not None:
            leg_cols = [(0, 1, 0.1), (0, 0, 1), (1, 0, 0)]
            leg_ids = [temp_id, 'Rovers', 'Sitters']
        else:
            leg_cols = [(0, 0, 1), (1, 0, 0)]
            leg_ids = ['Rovers', 'Sitters']
    else:
        cols = [(0.9 - i, 0.1 + i, 0.1) for i in np.linspace(0, 0.9, Ndebs)]
        leg_ids = ids
        leg_cols = cols

    labels0 = ['mass', 'length',
               'reserve', 'reserve_density', 'hunger',
               'pupation_buffer',
               'f', 'f_filt',
               'EEB',
               'M_gut', 'mol_ingested', 'M_absorbed', 'M_faeces', 'M_not_digested', 'M_not_absorbed',
               'R_faeces', 'R_absorbed', 'R_not_digested', 'gut_occupancy',
               'deb_p_A', 'sim_p_A', 'gut_p_A', 'gut_f', 'gut_p_A_deviation',
               'M_X', 'M_P', 'M_Pu', 'M_g', 'M_c', 'R_M_c', 'R_M_g', 'R_M_X_M_P', 'R_M_X', 'R_M_P'
               ]
    ylabels0 = ['wet weight $(mg)$', 'body length $(mm)$',
                r'reserve $(J)$', r'reserve density $(-)$', r'hunger drive $(-)$',
                r'pupation buffer $(-)$',
                r'f $^{sim}$ $(-)$', r'f $_{filt}^{sim}$ $(-)$',
                r'exploit VS explore $(-)$',
                'gut content $(mg)$', 'food intake $(C-mmole)$', 'food absorption $(mg)$',
                'faeces $(mg)$', 'food not digested $(mg)$', 'product not absorbed $(mg)$',
                'faeces fraction', 'absorption efficiency', 'fraction not digested', 'gut occupancy',
                r'[p$_{A}^{deb}$] $(microJ/cm^3)$', r'[p$_{A}^{sim}$] $(microJ/cm^3)$',
                r'[p$_{A}^{gut}$] $(microJ/cm^3)$',
                # r'[p$_{A}^{deb}$] $(\mu J/cm^3)$', r'[p$_{A}^{sim}$] $(\mu J/cm^3)$',r'[p$_{A}^{gut}$] $(\mu J/cm^3)$',
                r'f $^{gut}$ $(-)$', r'$\Delta$p$_{A}^{gut}$ $(-)$',
                r'Food in gut $(C-moles)$', r'Product in gut $(C-moles)$', r'Product absorbed $(C-mmoles)$',
                r'Active enzyme amount in gut $(-)$', r'Available carrier amount in gut surface $(-)$',
                r'Available carrier ratio in gut surface $(-)$', r'Active enzyme ratio in gut surface $(-)$',
                r'Food VS Product ratio in gut $(-)$',
                r'Ratio of Food in gut $(-)$', r'Ratio of Product in gut $(-)$'
                # r'(deb) assimilation energy $(J)$', r'(f) assimilation energy $(J)$', r'(gut) assimilation energy $(J)$'
                ]
    sharey = False
    if mode == 'energy':
        idx = [2, 3, 4, 5]
    elif mode == 'growth':
        idx = [0, 1, 5]
    elif mode == 'full':
        idx = [0, 1, 2, 3, 4, 5]
    elif mode == 'feeding':
        idx = [3, 4, 8]
    elif mode in labels0:
        idx = [labels0.index(mode)]
    elif mode == 'food_mass':
        idx = [10, 26]
        # idx = [9, 10, 11, 12, 13, 14]
        sharey = True
    elif mode == 'food_ratio':
        idx = [17, 15, 16, 18]
    elif mode == 'food_mass_1':
        idx = [9, 10, 11]
    elif mode == 'food_mass_2':
        idx = [12, 13, 14]
    elif mode == 'food_ratio_1':
        idx = [16]
        # idx = [18, 16]
    elif mode == 'food_ratio_2':
        idx = [17, 15]
    elif mode == 'assimilation':
        idx = [19, 20, 21]
        sharey = True
    elif mode == 'fs':
        idx = [6, 7, 22]
        sharey = True
    elif mode == 'plug_flow_food':
        idx = [24, 25, 32, 33, 31]
        # sharey = True
    elif mode == 'plug_flow_enzymes':
        idx = [29, 30]
        # sharey = True

    tickstep = 24
    time_unit_dict = {
        'hours': 1,
        'minutes': 60,
        'seconds': 60 * 60,
    }
    t_coef = time_unit_dict[time_unit]

    labels = [labels0[i] for i in idx]
    ylabels = [ylabels0[i] for i in idx]
    Npars = len(labels)
    fig, axs = plt.subplots(Npars, figsize=(20, 6 * Npars), sharex=True, sharey=sharey)
    axs = axs.ravel() if Npars > 1 else [axs]

    # rr0, gg0, bb0 = q_col1 = np.array([255, 0, 0]) / 255
    # rr1, gg1, bb1 = q_col2 = np.array([255, 255, 255]) / 255
    # quality_col_range = np.array([rr1 - rr0, gg1 - gg0, bb1 - bb0])

    t0s, t1s, t2s, t3s, max_ages = [], [], [], [], []
    for jj, (d, id, c) in enumerate(zip(deb_dicts, ids, cols)):
        t0_sim, t0, t1, t2, t3, age = d['sim_start'], d['birth'], d['pupation'], d['death'], d['hours_as_larva'] + d[
            'birth'], np.array(d['age'])
        t00 = 0
        epochs = np.array(d['epochs'])
        if 'epoch_qs' in d.keys():
            epoch_qs = np.array(d['epoch_qs'])
        else:
            epoch_qs = np.zeros(len(epochs))
        if sim_only:
            t0_sim -= t3
            t00 -= t3
            t0 -= t3
            t1 -= t3
            t2 -= t3
            age -= t3
            epochs -= t3
            t3 = 0
        elif not include_egg:
            t0_sim -= t0
            t00 -= t0
            t1 -= t0
            t2 -= t0
            t3 -= t0
            age -= t0
            epochs -= t0
            t0 = 0

        t0_sim *= t_coef
        t00 *= t_coef
        t0 *= t_coef
        t1 *= t_coef
        t2 *= t_coef
        t3 *= t_coef
        age *= t_coef
        epochs *= t_coef
        tickstep *= t_coef

        t0s.append(t0)
        t1s.append(t1)
        t2s.append(t2)
        t3s.append(t3)
        max_ages.append(age[-1])

        for j, (l, yl) in enumerate(zip(labels, ylabels)):
            if l == 'f_filt':
                P = d['f']
                sos = signal.butter(N=1, Wn=d['fr'] / 1000, btype='lowpass', analog=False, fs=d['fr'], output='sos')
                P = signal.sosfiltfilt(sos, P)
            else:
                P = d[l]
            ax = axs[j]
            ax.plot(age, P, color=c, label=id, linewidth=2, alpha=1.0)
            ax.axvline(t0, color=c, alpha=0.6, linestyle='dashdot', linewidth=3)
            ax.axvline(t1, color=c, alpha=0.6, linestyle='dashdot', linewidth=3)
            ax.axvline(t2, color=c, alpha=0.6, linestyle='dashdot', linewidth=3)
            ax.axvspan(t00, t0, color='darkgrey', alpha=0.5)
            ax.axvspan(t0, t0_sim, color='lightgrey', alpha=0.5)

            if d['simulation']:
                ax.axvspan(t0, t3, color='grey', alpha=0.05)
            for (st0, st1), qq in zip(epochs, epoch_qs):
                q_col = col_range(qq, low=(255, 0, 0), high=(255, 255, 255)) if color_epoch_quality else c
                ax.axvspan(st0, st1, color=q_col, alpha=0.2)

            ax.set_ylabel(yl, labelpad=15, fontsize=10)
            ax.yaxis.set_major_locator(ticker.MaxNLocator(3))
            ax.tick_params(axis='y', labelsize=10)
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

            if l in ['pupation_buffer', 'EEB', 'R_faeces', 'R_absorbed', 'R_not_digested',
                     'gut_occupancy']:
                ax.set_ylim([0, 1])
            if force_ymin is not None:
                ax.set_ylim(ymin=force_ymin)
            if not sim_only:
                ax.set_xlim(xmin=0)
            if l == 'f' or mode == 'fs':
                ax.axhline(np.nanmean(P), color=c, alpha=0.6, linestyle='dashed', linewidth=2)
            if mode == 'assimilation':
                ax.axhline(np.nanmean(P), color=c, alpha=0.6, linestyle='dashed', linewidth=2)
            if label_lifestages and not sim_only:
                y0, y1 = ax.get_ylim()
                x0, x1 = ax.get_xlim()
                if jj == Ndebs - 1:
                    try:
                        ytext = y0 + 0.5 * (y1 - y0)
                        xtext = t00 + 0.5 * (t0 - t00)
                        ax.annotate('$incubation$', rotation=90, fontsize=15, va='center', ha='center',
                                    xy=(xtext, ytext), xycoords='data',
                                    )
                    except:
                        pass

                try:

                    ytext = y0 + 0.5 * (y1 - y0)
                    if not np.isnan(t1) and x1 > t1:
                        xtext = t1 + 0.5 * (x1 - t1)
                        ax.annotate('$pupation$', rotation=90, fontsize=15, va='center', ha='center',
                                    xy=(xtext, ytext), xycoords='data',
                                    )
                except:
                    pass
            if label_epochs and Ndebs == 1:

                try:
                    y0, y1 = ax.get_ylim()
                    ytext = y0 + 0.8 * (y1 - y0)
                    xpre = t0 + 0.5 * (t0_sim - t0)
                    if t0_sim - t0 > 0.2 * (np.max(age) - t00):
                        ax.annotate('$prediction$', rotation=0, fontsize=15, va='center', ha='center',
                                    xy=(xpre, ytext), xycoords='data',
                                    )
                    xsim = t0_sim + 0.5 * (np.max(age) - t0_sim)
                    if np.max(age) - t0_sim > 0.2 * (np.max(age) - t00):
                        ax.annotate('$simulation$', rotation=0, fontsize=15, va='center', ha='center',
                                    xy=(xsim, ytext), xycoords='data',
                                    )
                except:
                    pass

        for t in [0, t0, t1, t2]:
            if not np.isnan(t):
                try:
                    y0, y1 = ax.get_ylim()
                    ytext = y0 - 0.2 * (y1 - y0)
                    ax.annotate('', xy=(t, y0), xycoords='data',
                                xytext=(t, ytext), textcoords='data',
                                arrowprops=dict(color='black', shrink=0.08, alpha=0.6)
                                )
                except:
                    pass

    ax.set_xlabel(f'time $({time_unit})$')
    T0 = np.nanmean(t0s)
    T1 = np.nanmean(t1s)
    T2 = np.nanmean(t2s)

    fontsize = 20
    y = -0.2
    # texts = ['hatch', 'pupation', 'death']
    texts = ['egg', 'hatch', 'pupation', 'death']
    # text_xs = [T0, T1, T2]
    text_xs = [0, T0, T1, T2]
    for text, x in zip(texts, text_xs):
        try:
            y0, y1 = ax.get_ylim()
            ytext = y0 - 0.2 * (y1 - y0)
            ax.annotate(text, xy=(x, y0), xycoords='data', fontsize=fontsize,
                        xytext=(x, ytext), textcoords='data',
                        horizontalalignment='center', verticalalignment='top')
        except:
            pass

    if sim_only:
        ax.set_xlim([0, np.max(max_ages)])
        ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
    else:
        # ax.set_xlim([0, np.max(max_ages)])
        for ax in axs:
            ax.set_xticks(ticks=np.arange(0, np.max(max_ages), tickstep))

    dataset_legend(leg_ids, leg_cols, ax=axs[0], loc='upper left', fontsize=20, prop={'size': 15})
    fig.subplots_adjust(top=0.95, bottom=0.15, left=0.15, right=0.93, hspace=0.15)
    return process_plot(fig, save_to, save_as, return_fig, show)


def plot_surface(x, y, z, vars, target, z0=None, ax=None, fig=None, title=None, lims=None, azim=115, elev=15, **kwargs):
    P = ParPlot(name='3d_surface', **kwargs)
    P.build(fig=fig, axs=ax, dim3=True, azim=azim, elev=elev)
    P.conf_ax_3d(vars, target, lims=lims, title=title)
    P.axs[0].plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=True)
    if z0 is not None:
        P.axs[0].plot_surface(x, y, np.ones(x.shape) * z0, alpha=0.5)
    return P.get()


def plot_heatmap(z, heat_kws={}, ax_kws={}, cbar_kws={}, **kwargs):
    base_heat_kws = {'annot': True, 'cmap': cm.coolwarm, 'vmin': None, 'vmax': None}
    base_heat_kws.update(heat_kws)
    base_cbar_kws = {"orientation": "vertical"}
    base_cbar_kws.update(cbar_kws)
    P = ParPlot(name='heatmap', **kwargs)
    P.build()
    sns.heatmap(z, ax=P.axs[0], **base_heat_kws, cbar_kws=base_cbar_kws)
    cax = plt.gcf().axes[-1]
    cax.tick_params(length=0)
    P.conf_ax(**ax_kws)
    P.adjust((0.15, 0.95), (0.15, 0.95))
    return P.get()


def plot_heatmap_PI(csv_filepath='PIs.csv', save_as='PI_heatmap.pdf', **kwargs):
    z = pd.read_csv(csv_filepath, index_col=0)
    Lgains = z.index.values.astype(int)
    Rgains = z.columns.values.astype(int)
    Ngains = len(Lgains)
    r = np.linspace(0.5, Ngains - 0.5, 5)
    ax_kws = {
        'xticklabels': Rgains[r.astype(int)],
        'yticklabels': Lgains[r.astype(int)],
        'xticklabelrotation': 0,
        'yticklabelrotation': 0,
        'xticks': r,
        'yticks': r,
        'xlab': r'Right odor gain, $G_{R}$',
        'ylab': r'Left odor gain, $G_{L}$',
        'xlabelpad': 20
    }
    heat_kws = {
        'annot': False,
        'vmin': -1,
        'vmax': 1,
        'cmap': 'RdYlGn',
    }

    cbar_kws = {
        'label': 'Preference for left odor',
        'ticks': [1, 0, -1]
    }

    return plot_heatmap(z, heat_kws=heat_kws, ax_kws=ax_kws, cbar_kws=cbar_kws, save_as=save_as, **kwargs)


def plot_3pars(df, vars, target, z0=None, **kwargs):
    figs = {}
    pr = f'{vars[0]}VS{vars[1]}'
    figs[f'{pr}_3d'] = plot_3d(df=df, vars=vars, target=target, **kwargs)
    try:
        x, y = np.unique(df[vars[0]].values), np.unique(df[vars[1]].values)
        X, Y = np.meshgrid(x, y)

        z = df[target].values.reshape(X.shape).T

        figs[f'{pr}_heatmap'] = plot_heatmap(z, ax_kws={'xticklabels': x.tolist(), 'yticklabels': y.tolist(),
                                                        'xlab': vars[0], 'ylab': vars[1]},
                                             cbar_kws={'label': target}, **kwargs)
        figs[f'{pr}_surface'] = plot_surface(X, Y, z, vars=vars, target=target, z0=z0, **kwargs)
    except:
        pass
    return figs


def plot_3d(df, vars, target, lims=None, title=None, surface=True, line=False, ax=None, fig=None, dfID=None,
            color='black', **kwargs):
    from statsmodels import api as sm
    P = ParPlot(name='3d_plot', **kwargs)
    P.build(fig=fig, axs=ax, dim3=True)
    P.conf_ax_3d(vars, target, lims=lims, title=title)

    l0, l1 = vars
    X = df[vars]
    y = df[target]

    X = sm.add_constant(X)

    # plot hyperplane
    if surface:
        est = sm.OLS(y, X).fit()

        xx1, xx2 = np.meshgrid(np.linspace(X[l0].min(), X[l0].max(), 100),
                               np.linspace(X[l1].min(), X[l1].max(), 100))
        # plot the hyperplane by evaluating the parameters on the grid
        Z = est.params[0] + est.params[1] * xx1 + est.params[2] * xx2
        surf = P.axs[0].plot_surface(xx1, xx2, Z, cmap=plt.cm.RdBu_r, alpha=0.6, linewidth=0)
        # plot data points - points over the HP are white, points below are black
        resid = y - est.predict(X)
        P.axs[0].scatter(X[resid >= 0][l0], X[resid >= 0][l1], y[resid >= 0], color='black', alpha=0.4,
                         facecolor='white')
        P.axs[0].scatter(X[resid < 0][l0], X[resid < 0][l1], y[resid < 0], color='black', alpha=0.4, facecolor=color)
    else:
        P.axs[0].scatter(X[l0], X[l1], y, color='black', alpha=0.4)

    return P.get()


def plot_3d_multi(dfs, dfIDs, df_colors=None, show=True, **kwargs):
    from mpl_toolkits.mplot3d import Axes3D
    if df_colors is None:
        df_colors = [None] * len(dfs)
    fig = plt.figure(figsize=(18, 12))
    ax = Axes3D(fig, azim=115, elev=15)
    for df, dfID, dfC in zip(dfs, dfIDs, df_colors):
        plot_3d(df, dfID=dfID, color=dfC, ax=ax, fig=fig, show=False, **kwargs)
    if show:
        plt.show()


def plot_2d(df, labels, **kwargs):
    P = ParPlot(name='2d_plot', **kwargs)
    par = labels[0]
    res = labels[1]
    p = df[par].values
    r = df[res].values
    P.build()
    P.axs[0].scatter(p, r)
    P.conf_ax(xlab=par, ylab=res)
    return P.get()


def plot_EEB_vs_food_quality(samples=None, dt=None, species_list=['rover', 'sitter', 'default'],
                             save_to=None, return_fig=False, show=False, **kwargs):
    if samples is None:
        raise ('No sample configurations provided')
    from lib.model.modules.intermitter import get_EEB_poly1d
    from lib.model.DEB.deb import DEB

    filename = f'EEB_vs_food_quality.{suf}'
    qs = np.arange(0.01, 1, 0.01)

    fig, axs = plt.subplots(3, len(samples), figsize=(10 * len(samples), 20))
    axs = axs.ravel()
    cols = N_colors(len(species_list))

    for i, sample in enumerate(samples):
        z = get_EEB_poly1d(sample=sample, dt=dt)
        for col, species in zip(cols, species_list):
            ss = []
            EEBs = []
            cc = {'color': col,
                  'label': species,
                  'marker': '.'}
            for q in qs:
                deb = DEB(substrate_quality=q, species=species, **kwargs)
                s = np.round(deb.fr_feed, 2)
                ss.append(s)
                EEBs.append(z(s))

            axs[3 * i].scatter(qs, ss, **cc)
            axs[3 * i + 1].scatter(qs, EEBs, **cc)
            axs[3 * i + 2].scatter(ss, EEBs, **cc)

        axs[3 * i + 0].set_xlabel('food quality')
        axs[3 * i + 1].set_xlabel('food quality')
        axs[3 * i + 2].set_xlabel(r'estimated feed freq $Hz$')
        axs[3 * i + 0].set_ylabel(r'estimated feed freq $Hz$')
        axs[3 * i + 1].set_ylabel('EEB')
        axs[3 * i + 2].set_ylabel('EEB')
        axs[3 * i + 1].set_ylim([0, 1])
        axs[3 * i + 2].set_ylim([0, 1])
    for ax in axs:
        ax.legend()
    return process_plot(fig, save_to, filename, return_fig, show)


def module_endpoint_hists(module, valid, e=None, refID=None, Nbins=None, show_median=True, fig=None, axs=None,
                          **kwargs):
    if e is None and refID is not None:
        from lib.conf.stored.conf import loadRef
        d = loadRef(refID)
        d.load(step=False)
        e = d.endpoint_data
    if Nbins is None:
        Nbins = int(e.index.values.shape[0] / 10)
    yy = int(e.index.values.shape[0] / 7)
    from lib.conf.base.dtypes import par
    # from lib.conf.base.init_pars import InitDict
    d0 = ParDict.init_dict[module]
    N = len(valid)

    P = BasePlot(name=f'{module}_endpoint_hists', **kwargs)
    P.build(1, N, figsize=(7 * N, 6), sharey=True, fig=fig, axs=axs)

    for i, n in enumerate(valid):
        ax = P.axs[i]
        p0 = par(n, **d0[n])[n]
        vs = e[p0['codename']]
        v_mu = vs.median()
        P.axs[i].hist(vs.values, bins=Nbins)
        P.conf_ax(i, xlab=p0['label'], ylab='# larvae' if i == 0 else None, xMaxN=3, xlabelfontsize=18,
                  xticklabelsize=18,
                  yvis=False if i != 0 else True)

        if show_median:
            text = p0['symbol'] + f' = {np.round(v_mu, 2)}'
            P.axs[i].axvline(v_mu, color='red', alpha=1, linestyle='dashed', linewidth=3)
            P.axs[i].annotate(text, rotation=0, fontsize=15, va='center', ha='left',
                              xy=(0.55, 0.8), xycoords='axes fraction',
                              )
    P.adjust((0.2, 0.9), (0.2, 0.9), 0.01)
    return P.get()


def plot_turns(absolute=True, subfolder='turn', **kwargs):
    P = Plot(name='turn_amplitude', subfolder=subfolder, **kwargs)
    P.build()
    p, xlab = getPar('tur_fou', to_return=['d', 'l'])
    bins, xlim = P.angrange(150, absolute, 30)
    P.plot_par(p, bins, i=0, absolute=absolute, alpha=1.0, histtype='step')
    P.conf_ax(xlab=xlab, ylab='probability, $P$', xlim=xlim, yMaxN=4, leg_loc='upper right')
    P.adjust((0.25, 0.95), (0.15, 0.92), 0.05, 0.005)
    return P.get()


def plot_turn_Dbearing(min_angle=30.0, max_angle=180.0, ref_angle=None, source_ID='Source',
                       Nplots=4, subfolder='turn', **kwargs):
    if ref_angle is None:
        name = f'turn_Dorient_to_center'
        ang0 = 0
        norm = False
        p = nam.bearing2(source_ID)
    else:
        ang0 = ref_angle
        norm = True
        name = f'turn_Dorient_to_{ang0}deg'
        p = nam.unwrap(nam.orient('front'))
    P = Plot(name=name, subfolder=subfolder, **kwargs)
    P.build(P.Ndatasets, Nplots, figsize=(5 * Nplots, 5 * P.Ndatasets), subplot_kw=dict(projection='polar'),
            sharey=True)



    for i, (d, c) in enumerate(zip(P.datasets, P.colors)):
        ii = Nplots * i
        for k, (chunk, side) in enumerate(zip(['Lturn', 'Rturn'], ['left', 'right'])):
            b0_par = nam.at(p, nam.start(chunk))
            b1_par = nam.at(p, nam.stop(chunk))
            bd_par = nam.chunk_track(chunk, p)
            # print(b0_par)
            b0 = d.get_par(b0_par).dropna().values.flatten() - ang0
            b1 = d.get_par(b1_par).dropna().values.flatten() - ang0
            db = d.get_par(bd_par).dropna().values.flatten()
            if norm:
                b0 %= 360
                b1 = b0 + db
                b0[b0 > 180] -= 360
                b1[b0 > 180] -= 360
            B0 = np.deg2rad(b0[(np.abs(db) > min_angle) & (np.abs(db) < max_angle)])
            B1 = np.deg2rad(b1[(np.abs(db) > min_angle) & (np.abs(db) < max_angle)])
            if Nplots == 2:
                for tt, BB, aa in zip(['start', 'stop'], [B0, B1], [0.3, 0.6]):
                    circNarrow(P.axs[ii + k], BB, aa, tt, c)
                P.axs[ii + 1].legend(bbox_to_anchor=(-0.7, 0.1), loc='center', fontsize=12)
            elif Nplots == 4:
                B00 = B0[B0 < 0]
                B10 = B1[B0 < 0]
                B01 = B0[B0 > 0]
                B11 = B1[B0 > 0]
                for tt, BB, aa in zip([r'$\theta^{init}_{or}$', r'$\theta^{fin}_{or}$'], [(B01, B00), (B11, B10)],
                                      [0.3, 0.6]):
                    for kk, ss, BBB in zip([0, 1], [r'$L_{sided}$', r'$R_{sided}$'], BB):
                        circNarrow(P.axs[ii + k + 2 * kk], BBB, aa, f'{ss} {tt}', c)
                        for iii in [ii + 1, ii + 2 + 1]:
                            P.axs[iii].legend(bbox_to_anchor=(-0.3, 0.1), loc='center', fontsize=12)
            if i == P.Ndatasets - 1:
                if Nplots == 2:
                    P.axs[ii + k].set_title(f'Bearing due to {side} turn.', y=-0.4)
                elif Nplots == 4:
                    P.axs[ii + k].set_title(fr'$L_{{sided}}$ {side} turn.', y=-0.4)
                    P.axs[ii + 2 + k].set_title(fr'$R_{{sided}}$ {side} turn.', y=-0.4)
    for ax in P.axs:
        ax.set_xticklabels([0, '', +90, '', 180, '', -90, ''], fontsize=15)
    dataset_legend(P.labels, P.colors, ax=P.axs[0], loc='upper center', anchor=(0.5, 0.99),
                   bbox_transform=P.fig.transFigure)
    P.adjust((0.0, 1.0), (0.15, 0.9), 0.0, 0.35)
    return P.get()


def plot_ang_pars(absolute=False, include_rear=False, half_circles=False, subfolder='turn', Npars=5, Nbins=100, **kwargs):
    if Npars == 5:
        shorts = ['b', 'bv', 'ba', 'fov', 'foa']
        rs = [100, 200, 2000, 200, 2000]
    elif Npars == 3:
        shorts = ['b', 'bv', 'fov']
        rs = [100, 200, 200]
    else:
        raise ValueError('3 or 5 pars allowed')

    if include_rear:
        shorts += ['rov', 'roa']
        rs += [200, 2000]

    Nps = len(shorts)
    P = AutoPlot(name='ang_pars', subfolder=subfolder, Ncols=Nps, figsize=(Nps * 8, 8), sharey=True, **kwargs)
    P.init_fits(getPar(shorts))
    for i, (k,r) in enumerate(zip(shorts, rs)):
        p=ParDict.dict[k]
        vs=[ParDict.get(k,d) for d in P.datasets]
        bins, xlim = P.angrange(r, absolute, Nbins)
        P.plot_par(vs=vs, bins=bins, i=i, absolute=absolute, labels=p.disp, alpha=0.8, histtype='step', linewidth=3,
                   pvalues=False, half_circles=half_circles)
        P.conf_ax(i, ylab='probability', yvis=True if i == 0 else False, xlab=p.label, ylim=[0, 0.1], yMaxN=3)
    dataset_legend(P.labels, P.colors, ax=P.axs[0], loc='upper left' if half_circles else 'upper right')
    P.adjust((0.3 / Nps, 0.99), (0.15, 0.95), 0.01)
    return P.get()


def plot_crawl_pars(subfolder='endpoint', par_legend=False, pvalues=False,type='sns.hist',
                    half_circles=False, kde=True, fig=None, axs=None,shorts=['str_N', 'run_tr', 'cum_d'], **kwargs):
    sns_kws={'kde' : kde, 'stat' : "probability", 'element': "step", 'fill':True, 'multiple' : "layer", 'shrink' :1}
    P = Plot(name='crawl_pars', subfolder=subfolder, **kwargs)
    Ncols=len(shorts)
    P.init_fits(getPar(shorts))
    P.build(1, Ncols, figsize=(Ncols * 5, 5), sharey=True, fig=fig, axs=axs)
    for i, k in enumerate(shorts):
        p=ParDict.dict[k]
        vs=[ParDict.get(k,d) for d in P.datasets]
        P.plot_par(vs=vs, bins='broad', nbins=40, labels=p.disp, i=i, sns_kws = sns_kws,
                   type=type, pvalues=pvalues, half_circles=half_circles, key='end')
        P.conf_ax(i, ylab='probability', yvis=True if i == 0 else False, xlab=p.label, xlim=p.lim, yMaxN=4,
                  leg_loc='upper right' if par_legend else None)
    dataset_legend(P.labels, P.colors, ax=P.axs[0], loc='upper left', fontsize=15)
    P.adjust((0.25 / Ncols, 0.99), (0.15, 0.95), 0.01)
    return P.get()


def plot_turn_duration(absolute=True, **kwargs):
    return plot_turn_amp(par_short='tur_t', mode='scatter', absolute=absolute, **kwargs)


def plot_turn_amp(par_short='tur_t', ref_angle=None, subfolder='turn', mode='hist', cumy=True, absolute=True, **kwargs):
    nn = 'turn_amp' if ref_angle is None else 'rel_turn_angle'
    name = f'{nn}_VS_{par_short}_{mode}'
    P = Plot(name=name, subfolder=subfolder, **kwargs)
    ypar, ylab, ylim = getPar('tur_fou', to_return=['d', 'l', 'lim'])

    if ref_angle is not None:
        A0 = float(ref_angle)
        p_ref = getPar(['tur_fo0', 'tur_fo1'])
        ys = []
        ylab = r'$\Delta\theta_{bearing} (deg)$'
        cumylab = r'$\bar{\Delta\theta}_{bearing} (deg)$'
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
        cumylab = r'$\bar{\Delta\theta}_{or} (deg)$'
        ys = [d.get_par(ypar).dropna().values.flatten() for d in P.datasets]
        if absolute:
            ys = [np.abs(y) for y in ys]
    xpar, xlab = getPar(par_short, to_return=['d', 'l'])
    xs = [d.get_par(xpar).dropna().values.flatten() for d in P.datasets]

    if mode == 'scatter':
        P.build(1, 1, figsize=(10, 10))
        ax = P.axs[0]
        for x, y, l, c in zip(xs, ys, P.labels, P.colors):
            ax.scatter(x=x, y=y, marker='o', s=5.0, color=c, alpha=0.5)
            m, k = np.polyfit(x, y, 1)
            ax.plot(x, m * x + k, linewidth=4, color=c, label=l)
            P.conf_ax(xlab=xlab, ylab=ylab, ylim=ylim, yMaxN=4, leg_loc='upper left')
            P.adjust((0.15, 0.95), (0.1, 0.95), 0.01)
    elif mode == 'hist':
        P.fig = scatter_hist(xs, ys, P.labels, P.colors, xlabel=xlab, ylabel=ylab, ylim=ylim, cumylabel=cumylab,
                             cumy=cumy)
    return P.get()


def plot_stride_Dbend(show_text=False, subfolder='stride', **kwargs):
    P = Plot(name='stride_bend_change', subfolder=subfolder, **kwargs)
    P.build()
    ax = P.axs[0]

    fits = {}
    for i, (d, l, c) in enumerate(zip(P.datasets, P.labels, P.colors)):
        b0 = d.get_par(nam.at('bend', nam.start('stride'))).dropna().values.flatten()[:500]
        b1 = d.get_par(nam.at('bend', nam.stop('stride'))).dropna().values.flatten()[:500]
        sign_b = np.sign(b0)
        b0 *= sign_b
        b1 *= sign_b
        db = b1 - b0
        ax.scatter(x=b0, y=db, marker='o', s=2.0, alpha=0.6, color=c, label=l)
        m, k = np.polyfit(b0, db, 1)
        m = np.round(m, 2)
        k = np.round(k, 2)
        fits[l] = [m, k]
        ax.plot(b0, m * b0 + k, linewidth=4, color=c)
        if show_text:
            ax.text(0.3, 0.9 - i * 0.1, rf'${l} : \Delta\theta_{{b}}={m} \cdot \theta_{{b}}$', fontsize=12,
                    transform=ax.transAxes)
            print(f'Bend correction during strides for {l} fitted as : db={m}*b + {k}')
    P.conf_ax(xlab=r'$\theta_{bend}$ at stride start $(deg)$', ylab=r'$\Delta\theta_{bend}$ over stride $(deg)$',
              xlim=[0, 85], ylim=[-60, 60], yMaxN=5)
    P.adjust((0.25, 0.95), (0.2, 0.95), 0.01)
    return P.get()


def plot_stride_Dorient(absolute=True, subfolder='stride', **kwargs):
    P = Plot(name='stride_orient_change', subfolder=subfolder, **kwargs)
    shorts = ['str_fo', 'str_ro']
    P.build(1, len(shorts))
    for i, sh in enumerate(shorts):
        p, sl, xlab = getPar(sh, to_return=['d', 's', 'l'])
        bins, xlim = P.angrange(80, absolute, 200)
        P.plot_par(p, bins, i=i, absolute=absolute, labels=[sl] * P.Ndatasets, alpha=0.5)
        P.conf_ax(i, ylab='probability' if i == 0 else None, xlab=xlab, yMaxN=4, leg_loc='upper left')
    P.adjust((0.12, 0.99), (0.2, 0.95), 0.01)
    return P.get()


def plot_interference(mode='orientation', agent_idx=None, subfolder='interference', **kwargs):
    name = f'interference_{mode}' if agent_idx is None else f'interference_{mode}_agent_idx_{agent_idx}'

    shorts = ['sv']
    if mode == 'orientation':
        shorts.append('fov')
    elif mode == 'orientation_x2':
        shorts.append('fov')
        shorts.append('rov')
    elif mode == 'bend':
        shorts.append('bv')
    elif mode == 'spinelength':
        shorts.append('l')
    Npars = len(shorts)

    pars, ylabs = getPar(shorts, to_return=['d', 'l'])
    P = AutoPlot(name=name, subfolder=subfolder, Nrows=Npars, figsize=(10, Npars * 5), sharex=True, **kwargs)

    ylim = [0, 60] if mode in ['bend', 'orientation', 'orientation_x2'] else None

    if agent_idx is not None:
        data = [[d.load_aux(type='stride', par=p).loc[d.agent_ids[agent_idx]].values for p in pars] for
                d in P.datasets]
    else:
        data = [[d.load_aux(type='stride', par=p).values for p in pars] for d in P.datasets]
    Npoints = data[0][0].shape[1] - 1
    for d0, c, l in zip(data, P.colors, P.labels):
        if mode in ['bend', 'orientation']:
            d0 = [np.abs(d) for d in d0]
        for i, (p, ylab, df) in enumerate(zip(pars, ylabs, d0)):
            plot_quantiles(df=df, from_np=True, axis=P.axs[i], color_shading=c, label=l)
            P.conf_ax(i, ylab=ylab, ylim=ylim if i != 0 else [0.0, 0.6], yMaxN=4, leg_loc='upper right')

    P.conf_ax(-1, xlab='$\phi_{stride}$', xlim=[0, Npoints], xticks=np.linspace(0, Npoints, 5),
              xticklabels=[r'$0$', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$'])
    P.adjust((0.12, 0.95), (0.2 / Npars, 0.97), 0.05, 0.1)
    return P.get()


def plot_dispersion(range=(0, 40), scaled=False, subfolder='dispersion', fig_cols=1, ymax=None,
                    **kwargs):
    from lib.process.store import get_dsp
    ylab = 'scaled dispersal' if scaled else r'dispersal $(mm)$'
    r0, r1 = range
    par = f'dispersion_{r0}_{r1}'
    name = f'scaled_dispersal_{r0}-{r1}' if scaled else f'dispersal_{r0}-{r1}'
    k = f'sdsp_{r0}_{r1}' if scaled else f'dsp_{r0}_{r1}'
    P = AutoPlot(name=name, subfolder=subfolder, **kwargs)
    t0, t1 = int(r0 * P.datasets[0].config.fr), int(r1 * P.datasets[0].config.fr)
    x = np.linspace(r0, r1, t1 - t0)


    for d, lab, c in zip(P.datasets, P.labels, P.colors):
        try :
            try:
                dsp = d.load_aux(type='dispersion', par=par if not scaled else nam.scal(par))
            except:
                dsp = get_dsp(d.step_data, par)
            mean = dsp['median'].values[t0:t1]
            lb = dsp['upper'].values[t0:t1]
            ub = dsp['lower'].values[t0:t1]
        except :
            dsp = ParDict.get(k, d)
            mean = dsp.groupby(level='Step').quantile(q=0.5).values[t0:t1]
            ub = dsp.groupby(level='Step').quantile(q=0.75).values[t0:t1]
            lb = dsp.groupby(level='Step').quantile(q=0.25).values[t0:t1]
        P.axs[0].fill_between(x, ub, lb, color=c, alpha=.2)
        P.axs[0].plot(x, mean, c, label=lab, linewidth=3 if lab != 'experiment' else 8, alpha=1.0)
    P.conf_ax(xlab='time, $sec$', ylab=ylab, xlim=[x[0], x[-1]], ylim=[0, ymax], xMaxN=4, yMaxN=4)
    P.axs[0].legend(loc='upper left', fontsize=15)
    return P.get()


def boxplots(shorts=['l', 'v_mu'], key='end', Ncols=4, annotation=True, show_ns=True, grouped=False, ylims=None,
             in_mm=[], target_only=None, **kwargs):
    pars, labs, units, symbols = getPar(shorts, to_return=['d', 'lab', 'unit', 'symbol'])
    Npars = len(pars)
    Ncols = Ncols
    Nrows = int(np.ceil(Npars / Ncols))

    P = AutoPlot(name=f'boxplot_{Npars}_{key}_pars', Ncols=Ncols, Nrows=Nrows, figsize=(8 * Ncols, 6 * Nrows),
                 sharex=True, **kwargs)

    group_ids = dNl.unique_list([d.config['group_id'] for d in P.datasets])
    Ngroups = len(group_ids)
    data = concat_datasets(P.datasets, key=key)
    if not grouped:
        x = "DatasetID"
        hue = None
        palette = dict(zip(P.labels, P.colors))
        data = data[pars + [x]]
    else:
        x = "DatasetID"
        hue = 'GroupID'
        palette = dict(zip(group_ids, N_colors(Ngroups)))
        data = data[pars + [x, hue]]
    for sh in in_mm:
        data[getPar(sh)] *= 1000

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
            annotate_plot(show_ns=show_ns, target_only=target_only, **kws)
        P.conf_ax(ii, xticklabelrotation=30, ylab=labs[ii], yMaxN=4, ylim=ylims[ii] if ylims is not None else None,
                  xvis=False if ii < (Nrows - 1) * Ncols else True)

    P.adjust((0.1, 0.95), (0.15, 0.9), 0.5, 0.05)
    return P.get()


def boxplot(par_shorts, sort_labels=False, xlabel=None, pair_ids=None, common_ids=None, coupled_labels=None, **kwargs):
    P = Plot(name=par_shorts[0], **kwargs)
    pars, sim_labels, exp_labels, labs, lims = getPar(par_shorts, to_return=['d', 's', 's', 'l', 'lim'])
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
            colors = N_colors(Ncommon)
            palette = {id: c for id, c in zip(common_ids, colors)}
            pair_dfs = []
            for pair_id in pair_ids:
                paired_group_ids = [f'{pair_id}_{common_id}' for common_id in common_ids]
                pair_vs = [all_vs_dict[id] for id in paired_group_ids]
                pair_vs = dNl.flatten_list(pair_vs)
                pair_array = boolean_indexing(pair_vs).T
                pair_df = pd.DataFrame(pair_array, columns=common_ids).assign(Trial=pair_id)
                pair_dfs.append(pair_df)
                cdf = pd.concat(pair_dfs)  # CONCATENATE

        else:
            colors = N_colors(Ngroups)
            palette = {id: c for id, c in zip(group_ids, colors)}
            array = boolean_indexing(all_vs).T
            df = pd.DataFrame(array, columns=group_ids).assign(Trial=1)
            cdf = pd.concat([df])  # CONCATENATE
        mdf = pd.melt(cdf, id_vars=['Trial'], var_name=['Group'])  # MELT

        g1 = sns.boxplot(x="Trial", y="value", hue='Group', data=mdf, palette=palette, ax=P.axs[ii], width=0.5,
                         fliersize=3, linewidth=None, whis=1.5)  # RUN PLOT

        g2 = sns.stripplot(x="Trial", y="value", hue='Group', data=mdf, palette=palette, ax=P.axs[ii])  # RUN PLOT
        P.conf_ax(ii, xlab=xlabel, ylab=ylabel, ylim=ylim)
    P.adjust((0.1, 0.95), (0.15, 0.9), 0.3, 0.3)
    return P.get()


def powerspectrum(par_shorts=['v', 'fov'], thr=0.2, pars=[], subfolder='powerspectrums', legend_loc='upper left',
                  Nids=None, **kwargs):
    from scipy.fft import fft, fftfreq
    from lib.process.aux import moving_average

    if len(pars) == 0:
        if len(par_shorts) == 0:
            raise ValueError('Either parameter names or shortcuts must be provided')
        else:
            pars, symbols, ylabs, ylims = getPar(par_shorts, to_return=['d', 's', 'l', 'lim'])
    else:
        symbols = pars
        ylabs = pars
        ylims = [None] * len(pars)
    N = len(pars)
    cols = ['grey'] if N == 1 else N_colors(N)
    if N == 1:
        name = f'{pars[0]}'
    elif N == 2:
        name = f'{pars[0]}_VS_{pars[1]}'
    else:
        name = f'{N}_pars'
    P = AutoPlot(name=name, subfolder=subfolder, figsize=(10, 8), **kwargs)

    ax = P.axs[0]
    counter = 0
    for p, symbol, ylab, ylim, c in zip(pars, symbols, ylabs, ylims, cols):
        P.conf_ax(xlab='Frequency in Hertz [Hz]', ylab='Frequency Domain (Spectrum) Magnitude', xlim=(0, 3.5),
                  ylim=(0, 5))
        for d, d_col, d_lab in zip(P.datasets, P.colors, P.labels):
            if P.Ndatasets > 1:
                c = d_col
            dc = d.get_par(p, key='step')
            Nticks = len(dc.index.get_level_values('Step').unique())
            xf = fftfreq(Nticks, 1 / d.fr)[:Nticks // 2]
            ids = dc.index.get_level_values('AgentID').unique()
            if Nids is not None:
                ids = ids[:Nids]
            yf0 = np.zeros(Nticks // 2)
            for id in ids:
                dc_single = dc.xs(id, level='AgentID').values
                dc_single = np.nan_to_num(dc_single)
                yf = fft(dc_single)
                yf = 2.0 / Nticks * np.abs(yf[0:Nticks // 2])
                yf = 1000 * yf / np.sum(yf)
                yf = moving_average(yf, n=21)
                ax.plot(xf, yf, color=c, alpha=0.2)
                yf0 += yf
            # xf=np.sort(xf)
            yf0 = 1000 * yf0 / np.sum(yf0)
            ax.plot(xf, yf0, color=c, label=symbol)
            ymax = np.max(yf0[xf > thr])
            xpos = np.argmax(yf0[xf > thr])
            xmax = xf[xf > thr][xpos]
            ax.plot(xmax, ymax, color=c, marker='o')
            ax.annotate(np.round(xmax, 2), xy=(xmax, ymax), xytext=(xmax + 0.2, ymax + 0.1), color=c, fontsize=25)
            # yf0 = moving_average(yf0, n=11)
            # ax.plot(xf, yf0, color=c, label=symbol)
            counter += 1

    if counter == 0:
        raise ValueError('None of the parameters exist in any dataset')
    if N > 1:
        ax.legend()
    if P.Ndatasets > 1:
        dataset_legend(P.labels, P.colors, ax=ax, loc=legend_loc, fontsize=15)
    P.adjust((0.2, 0.95), (0.15, 0.95))
    return P.get()


def plot_navigation_index(subfolder='source', **kwargs):
    P = AutoPlot(name='nav_index', subfolder=subfolder, Nrows=2, figsize=(20, 20), sharex=True, sharey=True, **kwargs)
    from lib.process.aux import compute_component_velocity, compute_velocity

    for d, c, g in zip(P.datasets, P.colors, P.labels):
        dt = 1 / d.fr
        Nticks = d.Nticks
        Nsec = int(Nticks * dt)
        s = d.read(key='trajectories', file='aux_h5')

        vxs = []
        vys = []
        for id in d.agent_ids:
            s0 = s.xs(id, level='AgentID').values
            v0 = compute_velocity(s0, dt=dt)
            vx = compute_component_velocity(s0, angles=np.zeros(Nticks), dt=dt)
            vy = compute_component_velocity(s0, angles=np.ones(Nticks) * -np.pi / 2, dt=dt)
            vx = np.divide(vx, v0, out=np.zeros_like(v0), where=v0 != 0)
            vy = np.divide(vy, v0, out=np.zeros_like(v0), where=v0 != 0)
            vxs.append(vx)
            vys.append(vy)
        vx0 = np.nanmean(np.array(vxs), axis=0)
        vy0 = np.nanmean(np.array(vys), axis=0)
        P.axs[0].plot(np.linspace(0, Nsec, Nticks), vx0, color=c, label=g)
        P.axs[1].plot(np.linspace(0, Nsec, Nticks), vy0, color=c, label=g)
    P.adjust((0.1, 0.95), (0.2, 0.98), H=0.15)
    P.conf_ax(0, ylab='X index', leg_loc='upper right')
    P.conf_ax(1, xlab='time (sec)', ylab='Y index', xlim=[0, Nsec], ylim=[-1.0, 1.0])
    P.axs[0].axhline(0.0, color='green', alpha=0.5, linestyle='dashed', linewidth=1)
    P.axs[1].axhline(0.0, color='green', alpha=0.5, linestyle='dashed', linewidth=1)
    return P.get()


def plot_stridesNpauses(stridechain_duration=False, time_unit='sec',
                        plot_fits='all', range='default', print_fits=False, only_fit_one=True, mode='cdf',
                        subfolder='bouts', refit_distros=False, test_detection=False, **kwargs):
    from lib.anal.fitting import compute_density, fit_bout_distros
    warnings.filterwarnings('ignore')
    nn = f'stridesNpauses_{mode}_{range}_{plot_fits}'
    name = nn if not only_fit_one else f'{nn}_0'
    P = AutoPlot(name=name, subfolder=subfolder, Ncols=2, figsize=(10, 5), sharey=True, **kwargs)
    pause_par = nam.dur('pause')
    if stridechain_duration:
        chain_par = nam.dur('run')  # nam.dur(nam.chain('stride'))
        chn_discr = False
        chain_xlabel = f'time $({time_unit})$'
        chn0 = 0.5
        chn1 = 50
        chn_t0, chn_t1 = 0, 10 ** 2
    else:
        chain_par = nam.length(nam.chain('stride'))
        chn_discr = True
        chain_xlabel = '# chained strides'
        chn0 = 1
        chn1 = 100
        chn_t0, chn_t1 = 10 ** 0, 10 ** 2.5

    pau_discr = False
    pau0 = 0.4
    pau1 = 20.0
    pau_t0, pau_t1 = 0, 10 ** 1.4
    pause_xlabel = f'time $({time_unit})$'

    pau_durs = []
    chn_durs = []

    frs = []
    for label, dataset in zip(P.labels, P.datasets):
        frs.append(dataset.fr)

        pau_dur = dataset.get_par(pause_par).dropna().values
        chn_dur = dataset.get_par(chain_par).dropna().values
        if time_unit == 'ms':
            pau_dur *= 1000
            pau0 *= 1000
            pau1 *= 1000
            pau_t0 *= 1000
            pau_t1 *= 1000
            if stridechain_duration:
                chn_dur *= 1000
                chn0 *= 1000
                chn1 *= 1000
                chn_t0 *= 1000
                chn_t1 *= 1000
        pau_durs.append(pau_dur)
        chn_durs.append(chn_dur)

    if test_detection:
        for l, d, col in zip(P.labels, P.datasets, P.colors):
            dic0 = d.load_dicts('bout_dicts')
            dic = {}
            for iid, ddd in dic0.items():
                df = pd.DataFrame.from_dict(ddd)
                df.index.set_names(0, inplace=True)
                dic[iid] = df

            pau_dur = np.array(dNl.flatten_list([ddic[pause_par] for ddic in dic.values()]))
            chn_dur = np.array(dNl.flatten_list([ddic[chain_par] for ddic in dic.values()]))
            pau_durs.append(pau_dur)
            chn_durs.append(chn_dur)
            P.labels.append(f'{l} truth')
            frs.append(d.fr)
            P.colors.append(f'dark{col}')

    min_pauses, max_pauses = [np.min(dur) for dur in pau_durs], [np.max(dur) for dur in pau_durs]
    min_chains, max_chains = [np.min(dur) for dur in chn_durs], [np.max(dur) for dur in chn_durs]

    if range == 'broad':
        pau0, pau1 = np.min(min_pauses), np.max(max_pauses)
        chn0, chn1 = np.min(min_chains), np.max(max_chains)
    elif range == 'restricted':
        pau0, pau1 = np.max(min_pauses), np.min(max_pauses)
        chn0, chn1 = np.max(min_chains), np.min(max_chains)
    elif range == 'default':
        pass
    fits = {l: {} for l in P.labels}

    distro_ls = ['powerlaw', 'exponential', 'lognormal', 'lognorm-pow', 'levy', 'normal', 'uniform']
    distro_cs = ['c', 'g', 'm', 'k', 'yellow', 'brown', 'purple']
    num_distros = len(distro_ls)

    for j, (pau_dur, chn_dur, c, label, fr) in enumerate(zip(pau_durs, chn_durs, P.colors, P.labels, frs)):
        try:
            from lib.conf.stored.conf import loadConf
            ref = loadConf(label, 'Ref')
        except:
            ref = None
        for i, (x0, discr, xmin, xmax) in enumerate(
                zip([chn_dur, pau_dur], [chn_discr, pau_discr], [chn0, pau0], [chn1, pau1])):
            bout = 'stride' if i == 0 else 'pause'
            lws = [2] * num_distros

            if not refit_distros and ref is not None:
                from lib.anal.fitting import BoutGenerator
                u2, du2, c2, c2cum = compute_density(x0, xmin, xmax)
                b = BoutGenerator(**ref[bout]['best'])
                pdfs = [b.get(x=du2, mode='pdf')] * num_distros
                cdfs = [1 - b.get(x=u2, mode='cdf')] * num_distros
                idx_Kmax = 0

            else:
                fit_dic = fit_bout_distros(x0, xmin, xmax, discr, dataset_id=label, bout=bout,
                                           print_fits=print_fits, combine=False)
                idx_Kmax = fit_dic['idx_Kmax']
                cdfs = fit_dic['cdfs']
                pdfs = fit_dic['pdfs']
                u2, du2, c2, c2cum = fit_dic['values']
                lws[idx_Kmax] = 4
                fits[label].update(fit_dic['res_dict'])
            if mode == 'cdf':
                ylabel = 'cumulative probability'
                xrange = u2
                y = c2cum
                ddfs = cdfs
                for ii in ddfs:
                    if ii is not None:
                        ii /= ii[0]

            elif mode == 'pdf':
                ylabel = 'probability'
                xrange = du2
                y = c2
                ddfs = pdfs
                for ii in ddfs:
                    if ii is not None:
                        ii /= sum(ii)

            P.axs[i].loglog(xrange, y, '.', color=c, alpha=0.7)
            for z, (l, col, lw, ddf) in enumerate(zip(distro_ls, distro_cs, lws, ddfs)):
                if ddf is None:
                    continue
                if plot_fits == 'best' and z == idx_Kmax:
                    cc = c
                elif plot_fits == 'all':
                    cc = col
                else:
                    continue
                P.axs[i].loglog(xrange, ddf, color=cc, lw=lw, label=l)

    for ii in [0, 1]:
        if plot_fits in ['all']:
            dataset_legend(distro_ls, distro_cs, ax=P.axs[ii], loc='lower left', fontsize=15)
        dataset_legend(P.labels, P.colors, ax=P.axs[ii], loc='upper right', fontsize=15)
    P.conf_ax(0, xlab=chain_xlabel, ylab=ylabel, xlim=[chn_t0, chn_t1], title=r'$\bf{stridechains}$')
    P.conf_ax(1, xlab=pause_xlabel, xlim=[pau_t0, pau_t1], ylim=[10 ** -3.5, 10 ** 0], title=r'$\bf{pauses}$')
    P.adjust((0.15, 0.95), (0.15, 0.92), 0.05, 0.005)
    fit_df = pd.DataFrame.from_dict(fits, orient="index")
    fit_df.to_csv(P.fit_filename, index=True, header=True)
    return P.get()


def plot_bout_ang_pars(absolute=True, include_rear=True, subfolder='turn', **kwargs):
    shorts = ['bv', 'fov', 'rov', 'ba', 'foa', 'roa'] if include_rear else ['bv', 'fov', 'ba', 'foa']
    ranges = [250, 250, 50, 2000, 2000, 500] if include_rear else [200, 200, 2000, 2000]

    pars, sim_ls, xlabels, disps = getPar(shorts, to_return=['d', 's', 'l', 'd'])
    Ncols = int(len(pars) / 2)
    chunks = ['stride', 'pause']
    chunk_cols = ['green', 'purple']
    P = AutoPlot(name='bout_ang_pars', subfolder=subfolder, Nrows=2, Ncols=Ncols, figsize=(Ncols * 7, 14), sharey=True,
                 **kwargs)
    p_labs = [[sl] * P.Ndatasets for sl in sim_ls]

    P.init_fits(pars, multiindex=False)

    for i, (p, r, p_lab, xlab, disp) in enumerate(zip(pars, ranges, p_labs, xlabels, disps)):
        bins, xlim = P.angrange(r, absolute, 200)
        ax = P.axs[i]
        for d, l in zip(P.datasets, P.labels):
            vs = []
            for c, col in zip(chunks, chunk_cols):
                v = d.step_data.dropna(subset=[nam.id(c)])[p].values
                if absolute:
                    v = np.abs(v)
                vs.append(v)
                ax.hist(v, color=col, bins=bins, label=c, weights=np.ones_like(v) / float(len(v)),
                        alpha=1.0, histtype='step', linewidth=2)
            P.comp_pvalue(l, vs[0], vs[1], p)
            P.plot_half_circle(p, ax, col1=chunk_cols[0], col2=chunk_cols[1], v=P.fit_df[p].loc[l], ind=l)

        P.conf_ax(i, xlab=xlab, xlim=xlim, yMaxN=3)
    P.conf_ax(0, ylab='probability', ylim=[0, 0.04], leg_loc='upper left')
    P.conf_ax(Ncols, ylab='probability', leg_loc='upper left')
    P.adjust((0.25 / Ncols, 0.95), (0.1, 0.9), 0.1, 0.3)
    return P.get()


def plot_endpoint_params(axs=None, fig=None, mode='basic', par_shorts=None, subfolder='endpoint',
                         plot_fit=True, nbins=20, Ncols=None, use_title=True, **kwargs):
    warnings.filterwarnings('ignore')
    P = Plot(name=f'endpoint_params_{mode}', subfolder=subfolder, **kwargs)
    ylim = [0.0, 0.25]
    nbins = nbins
    l_par = 'l'  # 'l_mu
    if par_shorts is None:
        dic = {
            'basic': [l_par, 'fsv', 'sv_mu', 'sstr_d_mu',
                      'str_tr', 'pau_tr', 'Ltur_tr', 'Rtur_tr',
                      'tor20_mu', 'dsp_0_40_fin', 'b_mu', 'bv_mu'],
            'minimal': [l_par, 'fsv', 'sv_mu', 'sstr_d_mu',
                        'cum_t', 'str_tr', 'pau_tr', 'tor',
                        'tor5_mu', 'tor20_mu', 'dsp_0_40_max', 'dsp_0_40_fin',
                        'b_mu', 'bv_mu', 'Ltur_tr', 'Rtur_tr'],
            'tiny': ['fsv', 'sv_mu', 'str_tr', 'pau_tr',
                     'b_mu', 'bv_mu', 'Ltur_tr', 'Rtur_tr'],
            'stride_def': [l_par, 'fsv', 'sstr_d_mu', 'sstr_d_std'],
            'reorientation': ['str_fo_mu', 'str_fo_std', 'tur_fou_mu', 'tur_fou_std'],
            'tortuosity': ['tor2_mu', 'tor5_mu', 'tor10_mu', 'tor20_mu'],
            'result': ['sv_mu', 'str_tr', 'pau_tr', 'pau_t_mu'],
            'limited': [l_par, 'fsv', 'sv_mu', 'sstr_d_mu',
                        'cum_t', 'str_tr', 'pau_tr', 'pau_t_mu',
                        'tor5_mu', 'tor5_std', 'tor20_mu', 'tor20_std',
                        'tor', 'sdsp_mu', 'sdsp_0_40_max', 'sdsp_0_40_fin',
                        'b_mu', 'b_std', 'bv_mu', 'bv_std',
                        'Ltur_tr', 'Rtur_tr', 'Ltur_fou_mu', 'Rtur_fou_mu'],
            'full': [l_par, 'str_N', 'fsv',
                     'cum_d', 'cum_sd', 'v_mu', 'sv_mu',
                     'str_d_mu', 'str_d_std', 'sstr_d_mu', 'sstr_d_std',
                     'str_std_mu', 'str_std_std', 'sstr_std_mu', 'sstr_std_std',
                     'str_fo_mu', 'str_fo_std', 'str_ro_mu', 'str_ro_std',
                     'str_b_mu', 'str_b_std', 'str_t_mu', 'str_t_std',
                     'cum_t', 'str_tr', 'pau_tr',
                     'pau_N', 'pau_t_mu', 'pau_t_std', 'tor',
                     'tor2_mu', 'tor5_mu', 'tor10_mu', 'tor20_mu',
                     'tor2_std', 'tor5_std', 'tor10_std', 'tor20_std',
                     'dsp_mu', 'dsp_fin', 'dsp_0_40_fin', 'dsp_0_40_max',
                     'sdsp_mu', 'sdsp_fin', 'sdsp_0_40_fin', 'sdsp_0_40_max',
                     'Ltur_t_mu', 'Ltur_t_std', 'cum_Ltur_t', 'Ltur_tr',
                     'Rtur_t_mu', 'Rtur_t_std', 'cum_Rtur_t', 'Rtur_tr',
                     'Ltur_fou_mu', 'Ltur_fou_std', 'Rtur_fou_mu', 'Rtur_fou_std',
                     'b_mu', 'b_std', 'bv_mu', 'bv_std',
                     ],
            'deb': [
                'deb_f_mu', 'hunger', 'reserve_density', 'puppation_buffer',
                'cum_d', 'cum_sd', 'str_N', 'fee_N',
                'str_tr', 'pau_tr', 'fee_tr', 'f_am',
                l_par, 'm'
                # 'tor2_mu', 'tor5_mu', 'tor10_mu', 'tor20_mu',
                # 'v_mu', 'sv_mu',

            ]
        }
        if mode in dic.keys():
            par_shorts = dic[mode]
        else:
            raise ValueError('Provide parameter shortcuts or define a mode')
    ends = [d.read('end', file='endpoint_h5') for d in P.datasets]
    pars = getPar(par_shorts)

    pars = [p for p in pars if all([p in e.columns for e in ends])]
    xlabels, xlims, disps = getPar(par_shorts, to_return=['l', 'lim', 'd'])

    if mode == 'stride_def':
        xlims = [[2.5, 4.8], [0.8, 2.0], [0.1, 0.25], [0.02, 0.09]]
    P.init_fits(pars)

    lw = 3
    Npars = len(pars)
    if Npars == 0:
        return None
    elif Ncols is not None:
        Nrows = int(np.ceil(Npars / Ncols))
    elif Npars == 4:
        Ncols = 2
        Nrows = 2
    else:
        Ncols = int(np.min([Npars, 4]))
        Nrows = int(np.ceil(Npars / Ncols))
    fig_s = 5

    P.build(Nrows, Ncols, figsize=(fig_s * Ncols, fig_s * Nrows), sharey=True, fig=fig, axs=axs)
    for i, (p, xlabel, xlim, disp) in enumerate(zip(pars, xlabels, xlims, disps)):
        bins = nbins if xlim is None else np.linspace(xlim[0], xlim[1], nbins)
        ax = P.axs[i]
        vs = [e[p].values for e in ends]
        P.comp_pvalues(vs, p)

        Nvalues = [len(i) for i in vs]
        a = np.empty((np.max(Nvalues), len(vs),)) * np.nan
        for k in range(len(vs)):
            a[:Nvalues[k], k] = vs[k]
        df = pd.DataFrame(a, columns=P.labels)
        for j, (col, lab) in enumerate(zip(df.columns, P.labels)):

            try:
                v = df[[col]].dropna().values
                y, x, patches = ax.hist(v, bins=bins, weights=np.ones_like(v) / float(len(v)),
                                        color=P.colors[j], alpha=0.5)
                if plot_fit:
                    x = x[:-1] + (x[1] - x[0]) / 2
                    y_smooth = np.polyfit(x, y, 5)
                    poly_y = np.poly1d(y_smooth)(x)
                    ax.plot(x, poly_y, color=P.colors[j], label=lab, linewidth=lw)
            except:
                pass
        P.conf_ax(i, ylab='probability' if i % Ncols == 0 else None, xlab=xlabel, xlim=xlim, ylim=ylim,
                  xMaxN=4, yMaxN=4, xMath=True, title=disp if use_title else None)
        P.plot_half_circles(p, i)
    P.adjust((0.1, 0.97), (0.17 / Nrows, 1 - (0.1 / Nrows)), 0.1, 0.2 * Nrows)
    dataset_legend(P.labels, P.colors, ax=P.axs[0], loc='upper right', fontsize=15)
    return P.get()


def plot_chunk_Dorient2source(source_ID, subfolder='bouts', chunk='stride', Nbins=16, min_dur=0.0, plot_merged=False,
                              **kwargs):
    P = Plot(name=f'{chunk}_Dorient_to_{source_ID}', subfolder=subfolder, **kwargs)

    if plot_merged:
        P.Ndatasets += 1
        P.colors.insert(0, 'black')
        P.labels.insert(0, 'merged')
    Ncols = int(np.ceil(np.sqrt(P.Ndatasets)))
    Nrows = Ncols - 1 if P.Ndatasets < Ncols ** 2 - Ncols else Ncols
    P.build(Nrows, Ncols, figsize=(8 * Ncols, 8 * Nrows), subplot_kw=dict(projection='polar'), sharey=True)

    durs = [d.get_par(nam.dur(chunk)).dropna().values for d in P.datasets]
    c0 = nam.start(chunk)
    c1 = nam.stop(chunk)
    b = nam.bearing2(source_ID)
    b0_par = nam.at(b, c0)
    b1_par = nam.at(b, c1)
    db_par = nam.chunk_track(chunk, b)
    b0s = [d.get_par(b0_par).dropna().values for d in P.datasets]
    b1s = [d.get_par(b1_par).dropna().values for d in P.datasets]
    dbs = [d.get_par(db_par).dropna().values for d in P.datasets]
    # print(len(durs[0]))
    # print(len(b0s[0]))
    # print(len(b1s[0]))
    # print(len(dbs[0]))
    # print(chunk)
    # print(P.datasets[0].step_data[nam.dur(chunk)])

    if plot_merged:
        b0s.insert(0, np.vstack(b0s))
        b1s.insert(0, np.vstack(b1s))
        dbs.insert(0, np.vstack(dbs))
        durs.insert(0, np.vstack(durs))

    for i, (b0, b1, db, dur, label, c) in enumerate(zip(b0s, b1s, dbs, durs, P.labels, P.colors)):
        ax = P.axs[i]
        b0 = b0[dur > min_dur]
        b1 = b1[dur > min_dur]
        db = db[dur > min_dur]
        b0m, b1m = np.mean(b0), np.mean(b1)
        dbm = np.round(np.mean(db), 2)
        if np.isnan([dbm, b0m, b1m]).any():
            continue
        circular_hist(ax, b0, bins=Nbins, alpha=0.3, label='start', color=c, offset=np.pi / 2)
        circular_hist(ax, b1, bins=Nbins, alpha=0.6, label='stop', color=c, offset=np.pi / 2)
        arrow0 = patches.FancyArrowPatch((0, 0), (np.deg2rad(b0m), 0.3), zorder=2, mutation_scale=30, alpha=0.3,
                                         facecolor=c, edgecolor='black', fill=True, linewidth=0.5)

        ax.add_patch(arrow0)
        arrow1 = patches.FancyArrowPatch((0, 0), (np.deg2rad(b1m), 0.3), zorder=2, mutation_scale=30, alpha=0.6,
                                         facecolor=c, edgecolor='black', fill=True, linewidth=0.5)
        ax.add_patch(arrow1)

        text_x = -0.3
        text_y = 1.2
        ax.text(text_x, text_y, f'Dataset : {label}', transform=ax.transAxes)
        ax.text(text_x, text_y - 0.1, f'Chunk (#) : {chunk} ({len(b0)})', transform=ax.transAxes)
        ax.text(text_x, text_y - 0.2, f'Min duration : {min_dur} sec', transform=ax.transAxes)
        ax.text(text_x, text_y - 0.3, fr'Correction $\Delta\theta_{{{"or"}}} : {dbm}^{{{"o"}}}$',
                transform=ax.transAxes)
        ax.legend(loc=[0.9, 0.9])
        ax.set_title(f'Bearing before and after a {chunk}.', fontsize=15, y=-0.2)
    for ax in P.axs:
        ticks_loc = ax.get_xticks().tolist()
        ax.xaxis.set_major_locator(ticker.FixedLocator(ticks_loc))
        ax.set_xticklabels([0, '', +90, '', 180, '', -90, ''])
    P.adjust((0.05 * Ncols / 2, 0.9), (0.2, 0.8), 0.8, 0.3)
    return P.get()


def plot_endpoint_scatter(subfolder='endpoint', keys=None, **kwargs):
    pairs = list(itertools.combinations(keys, 2))
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
        name = f'endpoint_scatterplot'
    else:
        name = f'{keys[1]}_vs_{keys[0]}'
    P = Plot(name=name, subfolder=subfolder, **kwargs)
    P.build(Nx, Ny, figsize=(10 * Ny, 10 * Nx))
    for i, (p0, p1) in enumerate(pairs):
        pars, labs = getPar([p0, p1], to_return=['d', 'l'])

        v0_all = [d.endpoint_data[pars[0]].values for d in P.datasets]
        v1_all = [d.endpoint_data[pars[1]].values for d in P.datasets]
        r0, r1 = 0.9, 1.1
        v0_r = [np.min(np.array(v0_all)) * r0, np.max(np.array(v0_all)) * r1]
        v1_r = [np.min(np.array(v1_all)) * r0, np.max(np.array(v1_all)) * r1]

        for v0, v1, l, c in zip(v0_all, v1_all, P.labels, P.colors):
            P.axs[i].scatter(v0, v1, color=c, label=l)
        P.conf_ax(i, xlab=labs[0], ylab=labs[1], xlim=v0_r, ylim=v1_r, tickMath=(0, 0),
                  title=f'{pars[1]}_vs_{pars[0]}', leg_loc='upper right')

    return P.get()


def plot_turn_Dorient2center(**kwargs):
    return plot_turn_Dbearing(ref_angle=None, **kwargs)
