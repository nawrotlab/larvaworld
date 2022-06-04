import itertools
import os
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import collections as mc, pyplot as plt, ticker, cm

from scipy import signal
from scipy.stats import ttest_ind, multivariate_normal



from lib.anal.plot_aux import Plot, dataset_legend, AutoPlot, plot_quantiles, plot_mean_and_range, process_plot, ParPlot
from lib.aux import naming as nam, dictsNlists as dNl
from lib.aux.colsNstr import N_colors, col_range
from lib.conf.base.opt_par import getPar

suf = 'pdf'

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
    from lib.anal.plot_aux import label_diff
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
    from lib.anal.plot_aux import boolean_indexing, annotate_plot
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
    from lib.anal.plot_aux import concat_datasets
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

    dst_par, dst_SI = getPar('cum_d', to_return=['d', 'u'])
    x = P.trange()
    for d, lab, c in zip(P.datasets, P.labels, P.colors):
        df = d.step_data[dst_par]
        if not scaled and unit == 'cm':
            import siunits

            if dst_SI.unit == siunits.m:
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
    from lib.anal.plot_aux import boolean_indexing
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
    boxplot = df.boxplot(figsize=(10, 7), grid=False,
                         color=dict(boxes='k', whiskers='k', medians='b', caps='k'),
                         boxprops=dict(linestyle='-', linewidth=3),
                         medianprops=dict(linestyle='-', linewidth=3),
                         whiskerprops=dict(linestyle='-', linewidth=3),
                         capprops=dict(linestyle='-', linewidth=3)
                         )
    boxplot.set_title(exp, fontsize=35)
    boxplot.set_xlabel('# training trials', fontsize=25)
    boxplot.set_ylabel(ylabel, fontsize=25)
    boxplot.set_ylim(ylim)
    plt.tick_params(labelsize=20)
    plt.tight_layout()
    plt.savefig(f, dpi=300)
    if show:
        plt.show()
    plt.close()


def lineplot(markers, par_shorts=['f_am'], coupled_labels=None, xlabel=None, ylabel=None, leg_cols=None, scale=1.0,
             **kwargs):
    from lib.anal.plot_aux import label_diff
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
