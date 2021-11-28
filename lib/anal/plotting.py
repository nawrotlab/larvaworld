import copy
import heapq
import itertools
import warnings
from matplotlib import collections  as mc
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import siunits
from matplotlib import cm, ticker, patches
from matplotlib.pyplot import bar
from matplotlib.ticker import FixedLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
import statsmodels.api as sm
from scipy import stats, signal
from scipy.stats import ttest_ind
from sklearn.linear_model import LinearRegression
from PIL import Image
import os

from lib.aux.dictsNlists import unique_list, flatten_list
from lib.anal.fitting import BoutGenerator
from lib.anal.plot_aux import plot_mean_and_range, circular_hist, confidence_ellipse, save_plot, \
    plot_config, dataset_legend, process_plot, label_diff, boolean_indexing, Plot, plot_quantiles, annotate_plot, \
    concat_datasets, ParPlot
from lib.aux import naming as nam
from lib.aux.colsNstr import N_colors, col_range

from lib.conf.base.par import getPar
from lib.model.DEB.deb import DEB

'''
Generic plot function. Uses the next two functions internally'''

plt_conf = {'axes.labelsize': 20,
            'axes.titlesize': 25,
            'figure.titlesize': 25,
            'xtick.labelsize': 20,
            'ytick.labelsize': 20,
            'legend.fontsize': 20,
            'legend.title_fontsize': 20}
plt.rcParams.update(plt_conf)
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
        try :
            s=d.step_data
        except :
            s = d.read('step')
        for k,(n,title) in enumerate(zip(['lin', 'ang'],[r'$\bf{runs & pauses}$', r'$\bf{left & right turns}$'])) :
            idx=2 * i+k
            ax=P.axs[idx]
            P.conf_ax(idx, xlab='time $(sec)$', ylab='Individuals $(idx)$', ylim=(0, N + 2), xlim=(0,d.config['Nticks']*d.dt), title=title if i==0 else None)
            for b,c in Cbouts[n].items() :
                bp0,bp1=nam.start(b), nam.stop(b)
                if not {bp0, bp1}.issubset(s.columns.values):
                    continue
                for j, id in enumerate(s.index.unique('AgentID').values) :
                    bbs=s[[bp0,bp1]].xs(id, level='AgentID')
                    b0s=bbs[bp0].dropna().index.values*d.dt
                    b1s=bbs[bp1].dropna().index.values*d.dt
                    lines = [[(b0, j+1), (b1, j+1)] for b0,b1 in zip(b0s, b1s)]
                    lc = mc.LineCollection(lines, colors=c, linewidths=2)
                    ax.add_collection(lc)
            dataset_legend(labels=list(Cbouts[n].keys()), colors=list(Cbouts[n].values()), ax=ax,
                           loc=None, anchor=None, fontsize=None, handlelength=0.5, handleheight=0.5)

    P.adjust((0.1, 0.95), (0.15, 0.92), 0.15, 0.1)
    return P.get()


def plot_2pars(shorts, subfolder='step',larva_legend=True, **kwargs):
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

    def circNarrow(ax, data, alpha, label, color):
        circular_hist(ax, data, bins=16, alpha=alpha, label=label, color=color, offset=np.pi / 2)
        arrow = patches.FancyArrowPatch((0, 0), (np.mean(data), 0.3), zorder=2, mutation_scale=30, alpha=alpha,
                                        facecolor=color, edgecolor='black', fill=True, linewidth=0.5)
        ax.add_patch(arrow)

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


def plot_ang_pars(absolute=True, include_rear=False, subfolder='turn', Npars=3, **kwargs):
    P = Plot(name='ang_pars', subfolder=subfolder, **kwargs)
    if Npars == 5:
        shorts = ['b', 'bv', 'ba', 'fov', 'foa']
        rs = [100, 200, 2000, 200, 2000]
        ylim = 0.05
    elif Npars == 3:
        shorts = ['b', 'bv', 'fov']
        rs = [100, 200, 200]
        ylim = 0.05
    else:
        raise ValueError('3 or 5 pars allowed')

    if include_rear:
        shorts += ['rov', 'roa']
        rs += [200, 2000]

    pars, sim_ls, xlabs = getPar(shorts, to_return=['d', 's', 'l'])
    p_ls = [[sl] * P.Ndatasets for sl in sim_ls]
    P.init_fits(pars)
    P.build(1, len(shorts), figsize=(len(shorts) * 5, 5), sharey=True)

    for i, (p, r, p_lab, xlab) in enumerate(zip(pars, rs, p_ls, xlabs)):
        bins, xlim = P.angrange(r, absolute, 200)
        P.plot_par(p, bins, i=i, absolute=absolute, labels=p_lab, alpha=0.8, histtype='step', linewidth=2,
                   pvalues=True, half_circles=True)
        P.conf_ax(i, ylab='probability' if i == 0 else None, xlab=xlab, ylim=[0, ylim], yMaxN=3)
    dataset_legend(P.labels, P.colors, ax=P.axs[0], loc='upper left')
    P.adjust((0.3 / len(pars), 0.99), (0.15, 0.95), 0.01)
    return P.get()


def plot_crawl_pars(subfolder='endpoint', par_legend=False, **kwargs):
    P = Plot(name='crawl_pars', subfolder=subfolder, **kwargs)
    pars, sim_ls, xlabs, xlims = getPar(['str_N', 'str_tr', 'cum_d'], to_return=['d', 's', 'l', 'lim'])
    p_ls = [[sl] * P.Ndatasets for sl in sim_ls]
    P.init_fits(pars)
    P.build(1, len(pars), figsize=(len(pars) * 5, 5), sharey=True)
    for i, (p, p_lab, xlab, xlim) in enumerate(zip(pars, p_ls, xlabs, xlims)):
        P.plot_par(p, bins='broad', nbins=40, labels=p_lab, i=i, kde=True, stat="probability", element="step",
                   type='sns.hist', pvalues=True, half_circles=True)
        P.conf_ax(i, ylab='probability' if i == 0 else None, xlab=xlab, xlim=xlim, yMaxN=4,
                  leg_loc='upper right' if par_legend else None)
    dataset_legend(P.labels, P.colors, ax=P.axs[0], loc='upper left', fontsize=15)
    P.adjust((0.25 / len(pars), 0.99), (0.15, 0.95), 0.01)
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
        p_ref, = getPar(['tur_fo0', 'tur_fo1'], to_return=['d'])
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


def plot_marked_strides(agent_idx=0, agent_id=None, slice=[20, 40], subfolder='individuals', **kwargs):
    temp = f'marked_strides_{slice[0]}-{slice[1]}' if slice is not None else f'marked_strides'
    name = f'{temp}_{agent_id}' if agent_id is not None else f'{temp}_{agent_idx}'
    P = Plot(name=name, subfolder=subfolder, **kwargs)
    Nds = P.Ndatasets

    chunks = ['stride', 'pause']
    chunk_cols = ['lightblue', 'grey']
    p, ylab = getPar('sv', to_return=['d', 'l'])
    figx = 15 * 6 * 3 if slice is None else int((slice[1] - slice[0]) / 3)
    figy = 5

    P.build(Nds, 1, figsize=(figx, figy * Nds), sharey=True, sharex=True)
    handles = [patches.Patch(color=col, label=n) for n, col in zip(['stride', 'pause'], chunk_cols)]

    for ii, (d, l) in enumerate(zip(P.datasets, P.labels)):
        ax = P.axs[ii]
        P.conf_ax(ii, xlab=r'time $(sec)$' if ii == Nds - 1 else None, ylab=ylab, ylim=[0, 1.0], xlim=slice,
                  leg_loc='upper right', leg_handles=handles)
        temp_id = d.agent_ids[agent_idx] if agent_id is None else agent_id
        s = copy.deepcopy(d.read('step').xs(temp_id, level='AgentID', drop_level=True))
        s.set_index(s.index * d.dt, inplace=True)
        ax.plot(s[p], color='blue')
        for i, (c, col) in enumerate(zip(chunks, chunk_cols)):
            s0s = s.index[s[nam.start(c)] == True]
            s1s = s.index[s[nam.stop(c)] == True]
            for s0, s1 in zip(s0s, s1s):
                ax.axvspan(s0, s1, color=col, alpha=1.0)
                ax.axvline(s0, color=f'{0.4 * (i + 1)}', alpha=0.6, linestyle='dashed', linewidth=1)
                ax.axvline(s1, color=f'{0.4 * (i + 1)}', alpha=0.6, linestyle='dashed', linewidth=1)

        ax.plot(s[p].loc[s[nam.max(p)] == True], linestyle='None', lw=10, color='green', marker='v')
        ax.plot(s[p].loc[s[nam.min(p)] == True], linestyle='None', lw=10, color='red', marker='^')
    P.adjust((0.08, 0.95), (0.15, 0.95), H=0.1)
    return P.get()


def plot_sample_tracks(mode='strides', agent_idx=0, agent_id=None, slice=[20, 40], subfolder='individuals', **kwargs):
    t0, t1 = slice
    temp = f'sample_marked_{mode}_{t0}-{t1}'
    name = f'{temp}_{agent_id}' if agent_id is not None else f'{temp}_{agent_idx}'
    P = Plot(name=name, subfolder=subfolder, **kwargs)
    Nds = P.Ndatasets

    if mode == 'strides':
        chunks = ['stride', 'pause']
        chunk_cols = ['lightblue', 'grey']

        p, ylab, ylim = getPar('sv', to_return=['d', 'l', 'lim'])
        ylim = (0.0,1.0)
    elif mode == 'turns':
        chunks = ['Rturn', 'Lturn']
        chunk_cols = ['lightgreen', 'orange']

        b = 'bend'
        bv = nam.vel(b)
        ho = nam.orient('front')
        hov = nam.vel(ho)
        p, ylab, ylim = getPar('fov', to_return=['d', 'l', 'lim'])

    figx = 15 * 6 * 3 if slice is None else int((t1 - t0) / 3)
    figy = 5

    P.build(Nds, 1, figsize=(figx, figy * Nds), sharey=True, sharex=True)
    handles = [patches.Patch(color=col, label=n) for n, col in zip(chunks, chunk_cols)]

    for ii, (d, l) in enumerate(zip(P.datasets, P.labels)):
        ax = P.axs[ii]

        P.conf_ax(ii, xlab=r'time $(sec)$' if ii == Nds - 1 else None, ylab=ylab, ylim=ylim, xlim=slice,
                  leg_loc='upper right', leg_handles=handles)

        temp_id = d.agent_ids[agent_idx] if agent_id is None else agent_id
        s = copy.deepcopy(d.read('step').xs(temp_id, level='AgentID', drop_level=True))
        s.set_index(s.index * d.dt, inplace=True)
        ax.plot(s[p], color='blue')
        for i, (c, col) in enumerate(zip(chunks, chunk_cols)):
            s0s = s.index[s[nam.start(c)] == True]
            s1s = s.index[s[nam.stop(c)] == True]
            for s0, s1 in zip(s0s, s1s):
                ax.axvspan(s0, s1, color=col, alpha=1.0)
                ax.axvline(s0, color=f'{0.4 * (i + 1)}', alpha=0.6, linestyle='dashed', linewidth=1)
                ax.axvline(s1, color=f'{0.4 * (i + 1)}', alpha=0.6, linestyle='dashed', linewidth=1)

        ax.plot(s[p].loc[s[nam.max(p)] == True], linestyle='None', lw=10, color='green', marker='v')
        ax.plot(s[p].loc[s[nam.min(p)] == True], linestyle='None', lw=10, color='red', marker='^')
    P.adjust((0.08, 0.95), (0.15, 0.95), H=0.1)
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
        vs = [d.endpoint_data[p] for d in P.datasets]
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


def lineplot(markers, par_shorts=['f_am'], coupled_labels=None, xlabel=None, ylabel=None, leg_cols=None, scale=1.0,
             **kwargs):
    P = Plot(name=par_shorts[0], **kwargs)
    Nds = P.Ndatasets
    Npars = len(par_shorts)
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

    P.build(Npars, 1, figsize=(8, 7))
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
    P = Plot(name=name, subfolder=subfolder, **kwargs)

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
    P.build(Npars, 1, figsize=(10, Npars * 5), sharex=True)

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


def plot_dispersion(range=(0, 40), scaled=False, subfolder='dispersion', fig_cols=1, ymax=None, **kwargs):
    ylab = 'scaled dispersion' if scaled else r'dispersion $(mm)$'
    r0, r1 = range
    par = f'dispersion_{r0}_{r1}'
    name = f'scaled_dispersion_{r0}-{r1}_{fig_cols}' if scaled else f'dispersion_{r0}-{r1}_{fig_cols}'
    P = Plot(name=name, subfolder=subfolder, **kwargs)
    t0, t1 = int(r0 * P.datasets[0].fr), int(r1 * P.datasets[0].fr)
    x = np.linspace(r0, r1, t1 - t0)
    P.build(figsize=(5 * fig_cols, 5))

    for d, lab, c in zip(P.datasets, P.labels, P.colors):
        dsp = d.load_aux(type='dispersion', par=par if not scaled else nam.scal(par))
        plot_mean_and_range(x=x,
                            mean=dsp['median'].values[t0:t1],
                            lb=dsp['upper'].values[t0:t1],
                            ub=dsp['lower'].values[t0:t1],
                            axis=P.axs[0], color_shading=c, label=lab)
    P.conf_ax(xlab='time, $sec$', ylab=ylab, xlim=[x[0], x[-1]], ylim=[0, ymax], xMaxN=4, yMaxN=4, leg_loc='upper left')
    P.adjust((0.2 / fig_cols, 0.95), (0.15, 0.95), 0.05, 0.005)
    return P.get()


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
            if dst_SI.unit == siunits.m:
                df *= 100
        plot_quantiles(df=df, x=x, axis=P.axs[0], color_shading=c, label=lab)

    P.conf_ax(xlab=xlabel, ylab=ylab, xlim=(x[0], x[-1]), ylim=[0, None], xMaxN=5, leg_loc='upper left')
    P.adjust((0.2, 0.95), (0.15, 0.95), 0.05, 0.005)
    return P.get()


def plot_gut(**kwargs):
    P = Plot(name='gut', **kwargs)
    P.build()
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
    P = Plot(name=name, **kwargs)
    P.build()

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
    P = Plot(name='PI_boxplot', **kwargs)

    group_ids = unique_list([d.config['group_id'] for d in P.datasets])
    Ngroups = len(group_ids)
    common_ids = unique_list([l.split('_')[-1] for l in group_ids])

    Ncommon = len(common_ids)
    pair_ids = unique_list([l.split('_')[0] for l in group_ids])

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

    P.build(figsize=(10, 5))
    sns.boxplot(x="Trial", y="value", hue="Group", data=mdf, palette=palette, ax=P.axs[0], width=.5,
                fliersize=3, linewidth=None, whis=1.0)  # RUN PLOT
    P.conf_ax(xlab=xlabel, ylab='Odor preference', ylim=[-1, 1], leg_loc='lower left')
    P.adjust((0.2, 0.9), (0.15, 0.9), 0.05, 0.005)
    return P.get()


def boxplot(par_shorts, sort_labels=False, xlabel=None, pair_ids=None, common_ids=None, **kwargs):
    P = Plot(name=par_shorts[0], **kwargs)
    pars, sim_labels, exp_labels, labs, lims = getPar(par_shorts, to_return=['d', 's', 's', 'l', 'lim'])
    Npars = len(pars)

    group_ids = unique_list([d.config['group_id'] for d in P.datasets])
    Ngroups = len(group_ids)
    if common_ids is None:
        common_ids = unique_list([l.split('_')[-1] for l in group_ids])

    Ncommon = len(common_ids)
    if pair_ids is None:
        pair_ids = unique_list([l.split('_')[0] for l in group_ids])
    Npairs = len(pair_ids)
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
            vs = [d.endpoint_data[p].values for d in group_ds]
            all_vs.append(vs)
            all_vs_dict[group_id] = vs
        all_vs = flatten_list(all_vs)
        if coupled_labels:
            colors = N_colors(Ncommon)
            palette = {id: c for id, c in zip(common_ids, colors)}
            pair_dfs = []
            for pair_id in pair_ids:
                paired_group_ids = [f'{pair_id}_{common_id}' for common_id in common_ids]
                pair_vs = [all_vs_dict[id] for id in paired_group_ids]
                pair_vs = flatten_list(pair_vs)
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
        # g1.get_legend().remove()

        g2 = sns.stripplot(x="Trial", y="value", hue='Group', data=mdf, palette=palette, ax=P.axs[ii])  # RUN PLOT
        # g2.get_legend().remove()
        P.conf_ax(ii, xlab=xlabel, ylab=ylabel, ylim=ylim)
    P.adjust((0.1, 0.95), (0.15, 0.9), 0.3, 0.3)
    return P.get()


def timeplot(par_shorts=[], pars=[], same_plot=True, individuals=False, table=None, unit='sec',absolute=True,
             show_first=False, subfolder='timeplots', legend_loc='upper left', **kwargs):
    unit_coefs = {'sec': 1, 'min': 1 / 60, 'hour': 1 / 60 / 60}
    if len(pars) == 0:
        if len(par_shorts) == 0:
            raise ValueError('Either parameter names or shortcuts must be provided')
        else:
            pars, symbols, ylabs, ylims = getPar(par_shorts, to_return=['d', 's', 'l', 'lim'])
    else:
        symbols = pars
        ylabs = pars
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
    P = Plot(name=name, subfolder=subfolder, **kwargs)

    P.build(figsize=(7.5, 5))
    ax = P.axs[0]
    counter = 0
    for p, symbol, ylab, ylim, c in zip(pars, symbols, ylabs, ylims, cols):
        P.conf_ax(xlab=f'time, ${unit}$' if table is None else 'timesteps', ylab=ylab, ylim=ylim, yMaxN=4)
        for d, d_col, d_lab in zip(P.datasets, P.colors, P.labels):
            if P.Ndatasets > 1:
                c = d_col
            try:
                if table is not None:
                    dc = d.load_table(table)[p]
                else:
                    dc = d.get_par(p, key='step')
                if absolute :
                    dc=dc.abs()
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
                    plot_quantiles(df=dc, x=x, axis=ax, color_shading=c, label=symbol)
                    if show_first:
                        dc0 = dc.xs(dc.index.get_level_values('AgentID')[0], level='AgentID')
                        ax.plot(x, dc0, color=c)
                counter += 1
            except:
                pass
    if counter == 0:
        raise ValueError('None of the parameters exist in any dataset')
    if N > 1:
        ax.legend()
    if P.Ndatasets > 1:
        dataset_legend(P.labels, P.colors, ax=ax, loc=legend_loc, fontsize=15)
    P.adjust((0.2, 0.95), (0.15, 0.95))
    return P.get()


def plot_navigation_index(subfolder='source', **kwargs):
    P = Plot(name='nav_index', subfolder=subfolder, **kwargs)
    from lib.process.aux import compute_component_velocity, compute_velocity
    P.build(2, 1, figsize=(20, 20), sharex=True, sharey=True)

    for d, c, g in zip(P.datasets, P.colors, P.labels):
        dt = 1 / d.fr
        Nticks = d.num_ticks
        Nsec = int(Nticks * dt)
        s, e = d.step_data, d.endpoint_data

        vxs = []
        vys = []
        for id in d.agent_ids:
            s0 = s.xs(id, level='AgentID')
            s0 = s0[['x', 'y']].values
            v0 = compute_velocity(s0, dt=dt)
            vx = compute_component_velocity(s0, angles=np.zeros(Nticks), dt=dt)
            vy = compute_component_velocity(s0, angles=np.ones(Nticks) * -np.pi / 2, dt=dt)
            vx = np.divide(vx, v0, out=np.zeros_like(v0), where=v0 != 0)
            vy = np.divide(vy, v0, out=np.zeros_like(v0), where=v0 != 0)
            vxs.append(vx)
            vys.append(vy)
        vx0 = np.nanmean(np.array(vxs), axis=0)
        vy0 = np.nanmean(np.array(vys), axis=0)
        P.axs[0].plot(np.linspace(0, Nsec, Nticks - 1), vx0, color=c, label=g)
        P.axs[1].plot(np.linspace(0, Nsec, Nticks - 1), vy0, color=c, label=g)
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
    P = Plot(name=name, subfolder=subfolder, **kwargs)
    pause_par = nam.dur('pause')
    if stridechain_duration:
        chain_par = nam.dur(nam.chain('stride'))
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

            pau_dur = np.array(flatten_list([ddic[pause_par] for ddic in dic.values()]))
            chn_dur = np.array(flatten_list([ddic[chain_par] for ddic in dic.values()]))
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

    P.build(1, 2, figsize=(10, 5), sharex=False, sharey=True)

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
        if plot_fits == 'all':
            dataset_legend(distro_ls, distro_cs, ax=P.axs[ii], loc='lower left', fontsize=15)
        dataset_legend(P.labels, P.colors, ax=P.axs[ii], loc='upper right', fontsize=15)
    P.conf_ax(0, xlab=chain_xlabel, ylab=ylabel, xlim=[chn_t0, chn_t1], title=r'$\bf{stridechains}$')
    P.conf_ax(1, xlab=pause_xlabel, xlim=[pau_t0, pau_t1], ylim=[10 ** -3.5, 10 ** 0], title=r'$\bf{pauses}$')
    P.adjust((0.15, 0.95), (0.15, 0.92), 0.05, 0.005)
    fit_df = pd.DataFrame.from_dict(fits, orient="index")
    fit_df.to_csv(P.fit_filename, index=True, header=True)
    return P.get()


def plot_bout_ang_pars(absolute=True, include_rear=True, subfolder='turn', **kwargs):
    P = Plot(name='bout_ang_pars', subfolder=subfolder, **kwargs)
    shorts = ['bv', 'fov', 'rov', 'ba', 'foa', 'roa'] if include_rear else ['bv', 'fov', 'ba', 'foa']
    ranges = [250, 250, 50, 2000, 2000, 500] if include_rear else [200, 200, 2000, 2000]

    pars, sim_ls, xlabels, disps = getPar(shorts, to_return=['d', 's', 'l', 'd'])

    chunks = ['stride', 'pause']
    chunk_cols = ['green', 'purple']

    p_labs = [[sl] * P.Ndatasets for sl in sim_ls]

    P.init_fits(pars, multiindex=False)

    Ncols = int(len(pars) / 2)
    P.build(2, Ncols, figsize=(Ncols * 7, 14), sharey=True)

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


def plot_endpoint_params(mode='basic', par_shorts=None, subfolder='endpoint', **kwargs):
    warnings.filterwarnings('ignore')
    P = Plot(name=f'endpoint_params_{mode}', subfolder=subfolder, **kwargs)

    ylim = [0.0, 0.25]
    nbins = 20
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
    ends = []
    for d in P.datasets:
        try:
            e = d.endpoint_data
        except:
            e = d.read('end')
        ends.append(e)
    pars, = getPar(par_shorts, to_return=['d'])

    pars = [p for p in pars if all([p in e.columns for e in ends])]
    xlabels, xlims, disps = getPar(par_shorts, to_return=['l', 'lim', 'd'])

    if mode == 'stride_def':
        xlims = [[2.5, 4.8], [0.8, 2.0], [0.1, 0.25], [0.02, 0.09]]
    P.init_fits(pars)

    lw = 3
    Npars = len(pars)
    if Npars == 0:
        return None
    elif Npars == 4:
        Ncols = 2
        Nrows = 2
    else:
        Ncols = int(np.min([Npars, 4]))
        Nrows = int(np.ceil(Npars / Ncols))
    fig_s = 5

    P.build(Nrows, Ncols, figsize=(fig_s * Ncols, fig_s * Nrows), sharey=True)
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
                weights = np.ones_like(v) / float(len(v))

                y, x, patches = ax.hist(v, bins=bins, weights=weights, color=P.colors[j], alpha=0.5)
                x = x[:-1] + (x[1] - x[0]) / 2
                y_smooth = np.polyfit(x, y, 5)
                poly_y = np.poly1d(y_smooth)(x)
                ax.plot(x, poly_y, color=P.colors[j], label=lab, linewidth=lw)
            except:
                pass
        P.conf_ax(i, ylab='probability' if i % Ncols == 0 else None, xlab=xlabel, xlim=xlim, ylim=ylim,
                  xMaxN=4, yMaxN=4, xMath=True, title=disp)
        P.plot_half_circles(p, i)

    P.adjust((0.1, 0.97), (0.17 / Nrows, 1 - (0.1 / Nrows)), 0.1, 0.2 * Nrows)
    P.axs[0].legend(loc='upper left', prop={'size': 15})
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
        ax.xaxis.set_major_locator(FixedLocator(ticks_loc))
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


def plot_odor_concentration(**kwargs):
    return timeplot(['c_odor1'], **kwargs)


def plot_sensed_odor_concentration(**kwargs):
    return timeplot(['dc_odor1'], **kwargs)


def plot_Y_pos(**kwargs):
    return timeplot(['y'], **kwargs)


def plot_stridechains(dataset, save_to=None):
    from lib.anal.fitting import powerlaw_cdf, exponential_cdf, lognorm_cdf
    d = dataset

    if save_to is None:
        save_to = os.path.join(d.plot_dir, 'plot_stridechains')
    if not os.path.exists(save_to):
        os.makedirs(save_to)
    filepath_MLE = os.path.join(save_to, f'stridechain_distribution_MLE.{suf}')
    filepath_r = os.path.join(save_to, f'stridechain_distribution_r.{suf}')

    s = d.step_data[nam.length(nam.chain('stride'))].dropna()
    u, c = np.unique(s, return_counts=True)
    c = c / np.sum(c)
    c = 1 - np.cumsum(c)

    alpha = 1 + len(s) / np.sum(np.log(s))
    beta = len(s) / np.sum(s - 1)
    mu = np.mean(np.log(s))
    sigma = np.std(np.log(s))

    fig, axs = plt.subplots(1, 1, figsize=(10, 10), sharex=True, sharey=True)
    fig.suptitle('Stridechain distribution', fontsize=25)

    axs.loglog(u, c, 'or', label='stridechains')
    # axs.loglog(u, 1 - powerlaw_cdf_2(u, P[0], P[1]), 'k', lw=2, label='powerlaw linear')
    axs.loglog(u, 1 - powerlaw_cdf(u, 1, alpha), 'r', lw=2, label='powerlaw MLE')
    axs.loglog(u, 1 - exponential_cdf(u, 1, beta), 'g', lw=2, label='exponential MLE')
    axs.loglog(u, 1 - lognorm_cdf(u, mu, sigma), 'b', lw=2, label='lognormal MLE')

    axs.legend(loc='lower left', fontsize=15)
    axs.axis([1, np.max(s), 10 ** -4.0, 10 ** 0])
    # axs.text(25, 10 ** - 1.5, r'$\alpha=' + str(np.round(alpha * 100) / 100) + '$',
    #          {'color': 'k', 'fontsize': 16})

    plt.xlabel(r'Stridechain  length, $l$', fontsize=15)
    plt.ylabel(r'Probability Density, $P_l$', fontsize=15)

    fig.savefig(filepath_MLE, dpi=300)
    print(f'Image saved as {filepath_MLE}')

    fig, axs = plt.subplots(1, 1, figsize=(10, 10), sharex=True, sharey=True)
    fig.suptitle('Stridechain distribution', fontsize=25)

    axs.loglog(u, c, 'or', label='stridechains')

    for r in np.round(np.arange(0.8, 1, 0.025), 3):
        x = np.arange(1, np.max(s), 1)
        y = (1 - r) * r ** (x - 1)
        y = 1 - np.cumsum(y)
        plt.plot(x, y)
        plt.loglog(x, y, label=r)
        # plt.ylim(10 ** -4.5, 10 ** -0.2)
    # axs.loglog(u, 1 - pareto.cdf(u, b=my_b, loc=loc, scale=scale), 'y', lw=2, label=my_label)

    axs.legend(loc='lower left', fontsize=15)
    axs.axis([1, np.max(s), 10 ** -4.0, 10 ** 0])
    # axs.text(25, 10 ** - 1.5, r'$\alpha=' + str(np.round(alpha * 100) / 100) + '$',
    #          {'color': 'k', 'fontsize': 16})

    plt.xlabel(r'Stridechain  length, $l$', fontsize=15)
    plt.ylabel(r'Probability Density, $P_l$', fontsize=15)

    fig.savefig(filepath_r, dpi=300)
    print(f'Image saved as {filepath_r}')


def plot_bend_pauses(dataset, save_to=None):
    from lib.anal.fitting import compute_density, powerlaw_cdf, exponential_cdf, lognorm_cdf
    d = dataset
    if save_to is None:
        save_to = os.path.join(d.plot_dir, 'plot_bend_pauses')
    if not os.path.exists(save_to):
        os.makedirs(save_to)
    filepath = os.path.join(save_to, f'bend_pause_distribution.{suf}')

    s = d.step_data[nam.dur('bend_pause')].dropna()
    durmin, durmax = np.min(s), np.max(s)
    u, uu, c, ccum = compute_density(s, durmin, durmax)
    alpha = 1 + len(s) / np.sum(np.log(s / durmin))
    beta = len(s) / np.sum(s - durmin)
    mu = np.mean(np.log(s))
    sigma = np.std(np.log(s))

    fig, axs = plt.subplots(1, 1, figsize=(10, 10), sharex=True, sharey=True)
    fig.suptitle('Bend-pause distribution', fontsize=25)

    axs.loglog(u, ccum, 'or', label='bend_pauses')
    axs.loglog(u, 1 - powerlaw_cdf(u, durmin, alpha), 'r', lw=2, label='powerlaw MLE')
    axs.loglog(u, 1 - exponential_cdf(u, durmin, beta), 'g', lw=2, label='exponential MLE')
    axs.loglog(u, 1 - lognorm_cdf(u, mu, sigma), 'b', lw=2, label='lognormal MLE')

    axs.legend(loc='lower left', fontsize=15)
    axs.axis([durmin, durmax, 10 ** -4.0, 10 ** 0])

    plt.xlabel(r'Bend pause duration, $(sec)$', fontsize=15)
    plt.ylabel(r'Probability Density, $P_d$', fontsize=15)

    fig.savefig(filepath, dpi=300)
    print(f'Image saved as {filepath}')


def plot_marked_turns(dataset, agent_ids=None, turn_epochs=['Rturn', 'Lturn'],
                      vertical_boundaries=False, min_turn_angle=0, slices=[], subfolder='individuals',
                      save_to=None, return_fig=False, show=False):
    Ndatasets, colors, save_to, labels = plot_config(datasets=[dataset], labels=[dataset.id], save_to=save_to,
                                                     subfolder=subfolder)
    # We plot the complete or a slice of the timeseries of scal centroid velocity. The grey areas are stridechains
    d = dataset

    if agent_ids is None:
        agent_ids = d.agent_ids

    xx = f'marked_turns_min_angle_{min_turn_angle}'
    filepath_full = f'{xx}_full.{suf}'
    filepath_full_long = f'{xx}_full_long.{suf}'
    filepath_slices = []
    for i, slice in enumerate(slices):
        filepath_slices.append(f'{xx}_slice_{i}.{suf}')
    generic_filepaths = [filepath_full_long, filepath_full] + filepath_slices

    figsize_short = (20, 5)
    figsize_long = (15 * 6, 5)
    figsizes = [figsize_long, figsize_short] + [figsize_short] * len(generic_filepaths)

    xlims = [None, None] + slices

    # ymax=1.0

    b = 'bend'
    bv = nam.vel(b)
    ho = nam.orient('front')
    hov = nam.vel(ho)
    fig_dict = {}
    for agent_id in agent_ids:
        filepaths = [f'{agent_id}_{f}' for f in generic_filepaths]

        s = d.step_data.xs(agent_id, level='AgentID', drop_level=True)
        # Nticks=len(sigma.index)
        # dur=Nticks/d.fr
        s.set_index(s.index.values / d.fr, inplace=True)

        b0 = s[b]
        bv0 = s[bv]
        ho0 = s[ho]
        hov0 = s[hov]

        for idx, (filepath, figsize, xlim) in enumerate(zip(filepaths, figsizes, xlims)):
            fig, axs = plt.subplots(1, 1, figsize=figsize)

            if turn_epochs is not None:
                cmap = cm.get_cmap('Pastel2')
                num_chunks = len(turn_epochs)
                colors = [cmap(i) for i in np.arange(num_chunks)]
                epoch_handles = []
                temp = None
                for i, (chunk, color) in enumerate(zip(turn_epochs, colors)):
                    start_flag = f'{chunk}_start'
                    stop_flag = f'{chunk}_stop'
                    stop_indexes = s.index[s[stop_flag] == True]
                    start_indexes = s.index[s[start_flag] == True]
                    if min_turn_angle > 0:
                        angle_flag = nam.chunk_track(chunk, nam.unwrap(nam.orient('front')))
                        angles = np.abs(s[angle_flag].dropna().values)
                        stop_indexes = stop_indexes[angles > min_turn_angle]
                        start_indexes = start_indexes[angles > min_turn_angle]

                    for start, stop in zip(start_indexes, stop_indexes):
                        temp = plt.axvspan(start, stop, color=color, alpha=1.0)

                        if vertical_boundaries:
                            plt.axvline(start, color=f'{0.4 * (i + 1)}', alpha=0.6, linestyle='dashed', linewidth=1)
                            plt.axvline(stop, color=f'{0.4 * (i + 1)}', alpha=0.6, linestyle='dashed', linewidth=1)
                    if temp is not None:
                        epoch_handles.append(temp)
            ax1 = b0.plot(label=r'$\theta_{b}$', lw=2, color='blue')
            ax1.set_ylabel(r'angle $(deg)$')
            ax1.set_xlabel(r'time $(sec)$')
            ax1.set_ylim([-100, 100])
            ax1.set_xlim(xlim)
            # plt.legend(loc= 'upper left')
            ax2 = bv0.plot(secondary_y=True, label=r'$\dot{\theta}_{b}$', lw=2, color='green')
            ax2.plot(hov0, label=r'$\dot{\theta}_{or}$', lw=3, color='black')
            ax2.set_ylabel(r'angular velocity $(deg/sec)$')
            ax2.set_ylim([-500, 500])

            plt.axhline(0, color='black', alpha=0.4, linestyle='dashed', linewidth=1)

            handles, labels = [], []
            for ax in fig.axes:
                for h, l in zip(*ax.get_legend_handles_labels()):
                    handles.append(h)
                    labels.append(l)

            # plt.legend(handles, labels, loc='upper left')

            par_legend = plt.legend(handles, labels, loc=2)
            plt.legend(epoch_handles, turn_epochs, loc=1)
            plt.gca().add_artist(par_legend)
            plt.subplots_adjust(hspace=0.05, top=0.95, bottom=0.2, left=0.08, right=0.92)
            filename = f'{save_to}/{filepath}'
            fig.savefig(filename, dpi=300)
            print(f'Image saved as {filename}')
            fig_dict[f'turns_{agent_id}_{i}'] = fig
    # return process_plot(fig, save_to, filename, return_fig, show)
    return fig_dict


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
        deb_dicts = flatten_list([d.load_dicts('deb') for d in datasets])
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
               'M_X', 'M_P', 'M_Pu','M_g', 'M_c','R_M_c','R_M_g','R_M_X_M_P','R_M_X','R_M_P'
               ]
    ylabels0 = ['wet weight $(mg)$', 'body length $(mm)$',
                r'reserve $(J)$', r'reserve density $(-)$', r'hunger drive $(-)$',
                r'pupation buffer $(-)$',
                r'f $^{sim}$ $(-)$', r'f $_{filt}^{sim}$ $(-)$',
                r'exploit VS explore $(-)$',
                'gut content $(mg)$', 'food intake $(C-mmole)$', 'food absorption $(mg)$',
                'faeces $(mg)$', 'food not digested $(mg)$', 'product not absorbed $(mg)$',
                'faeces fraction', 'absorption efficiency', 'fraction not digested', 'gut occupancy',
                r'[p$_{A}^{deb}$] $(microJ/cm^3)$', r'[p$_{A}^{sim}$] $(microJ/cm^3)$',r'[p$_{A}^{gut}$] $(microJ/cm^3)$',
                # r'[p$_{A}^{deb}$] $(\mu J/cm^3)$', r'[p$_{A}^{sim}$] $(\mu J/cm^3)$',r'[p$_{A}^{gut}$] $(\mu J/cm^3)$',
                r'f $^{gut}$ $(-)$', r'$\Delta$p$_{A}^{gut}$ $(-)$',
                r'Food in gut $(C-moles)$', r'Product in gut $(C-moles)$', r'Product absorbed $(C-mmoles)$',
                r'Active enzyme amount in gut $(-)$', r'Available carrier amount in gut surface $(-)$',
                r'Available carrier ratio in gut surface $(-)$',r'Active enzyme ratio in gut surface $(-)$',r'Food VS Product ratio in gut $(-)$',
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
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

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


def plot_surface(x, y, z, vars, target, z0=None, ax=None, fig=None, title=None, lims=None, **kwargs):
    P = ParPlot(name='3d_surface', **kwargs)
    P.build(fig=fig, axs=ax, dim3=True)
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


def plot_heatmap_PI(csv_filepath='PIs.csv', **kwargs):
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

    return plot_heatmap(z, heat_kws=heat_kws, ax_kws=ax_kws, cbar_kws=cbar_kws, save_as='PI_heatmap.pdf', **kwargs)


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


def plot_bend2orientation_analysis(dataset, save_to=None, save_as=f'bend2orientation.{suf}'):
    d = dataset
    s = d.step_data
    if save_to is None:
        save_to = dataset.plot_dir
    filepath = os.path.join(save_to, save_as)

    avels = nam.vel(d.angles)
    if not set(avels).issubset(s.columns.values):
        raise ValueError('Spineangle angular velocities do not exist in step')
    hov = nam.vel(nam.orient('front'))
    N = d.Nangles
    k = range(N)
    s = s.loc[s[avels].dropna().index.values].copy()
    target = s[hov].dropna()
    num_best = 5
    combos = []
    corrs = []
    ps = []
    for i in k[:-5]:
        for c in itertools.combinations(avels, i + 1):
            tseries = s.sum(axis=1)
            r, p = stats.pearsonr(target, tseries)
            combos.append(c)
            corrs.append(r)
            ps.append(p)
    max_corrs = heapq.nlargest(num_best, corrs)
    max_corrs_idx = heapq.nlargest(num_best, range(len(corrs)), key=corrs.__getitem__)
    best_combos = [combos[i] for i in max_corrs_idx]
    best_combos_ind = [np.sort([avels.index(x) + 1 for x in set(avels).intersection(c)]) for c in best_combos]
    best_combo = combos[heapq.nlargest(1, range(len(corrs)), key=corrs.__getitem__)[0]]

    # best_combos_ind[0]=[1,2,3,4,5]
    # best_combos_ind[1]=[1,2,3,4]
    # best_combos_ind[2]=[1,2,3,4,7]
    # best_combos_ind[3]=[1,3]
    # best_combos_ind[4]=[1,2,3]
    for i, (cor, combo) in enumerate(zip(max_corrs, best_combos_ind)):
        print(f'Combo number {i} has correlation {cor}')
        print(f'Includes {combo}')
        print()
    print(f'Best combo is : {best_combo}')

    X0 = s[avels].dropna().values
    y = target.values

    figsize = (15, 7)

    # Plot figure with subplots of different sizes
    fig = plt.figure(figsize=figsize)
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    axs = axs.ravel()

    scores1 = []
    coefs1 = []
    for i in k:
        X = X0[:, i:i + 1]
        reg = LinearRegression().fit(X, y)
        scores1.append(reg.score(X, y))
        coefs1.append(reg.coef_)
    axs[0].scatter(np.arange(1, N + 1), scores1, c='blue', alpha=1.0, marker=",", label='single', s=200)
    axs[0].plot(np.arange(1, N + 1), scores1, c='blue')
    axs[0].set_xticks(ticks=np.arange(1, N + 1))
    axs[0].set_xlabel(r'angular velocity, $\dot{\theta}_{i}$')
    axs[0].set_ylabel('regression score')

    # plt.subplot2grid((2, 2), (0, 1), colspan=1, rowspan=1)

    scores2 = []
    coefs2 = []
    for i in k:
        X = X0[:, 0:i + 1]
        reg = LinearRegression().fit(X, y)
        scores2.append(reg.score(X, y))
        coefs2.append(reg.coef_)
    # fig.suptitle('Reorientation prediction by each spineangle')
    axs[0].scatter(np.arange(1, N + 1), scores2, c='green', alpha=1.0, marker="o", label='cumulative', s=200)
    axs[0].plot(np.arange(1, N + 1), scores2, c='green')
    shape1 = patches.Circle((0, 0), 1, facecolor='blue')
    shape2 = patches.Rectangle((0, 0), 1, 1, facecolor='green')
    axs[0].legend(loc='lower left')
    # axs[0].legend((shape1, shape2), ('single', 'cumulative'), loc='lower left')
    axs[0].yaxis.set_major_locator(ticker.MaxNLocator(4))
    # r = np.arange(1, N + 1)
    # plt.xticks(ticks=r, labels=['1'] + [f'1-{i}' for i in r[1:]])
    # plt.xlabel(r'cumulative angular velocity, $\dot{\theta}_{1-i}$')
    # plt.ylabel('regression score')

    # plt.subplot2grid((2, 2), (1, 0), colspan=2, rowspan=1)

    ylim = [0.6, 1]
    bar(x=[','.join(map(str, c)) for c in best_combos_ind], height=max_corrs, width=0.8, color='black')
    # ax.set_xticks(best_combos_ind)
    axs[1].set_xlabel('combined angular velocities')
    axs[1].set_ylabel('Pearson correlation')
    axs[1].tick_params(axis='x', which='major', labelsize=15)
    axs[1].set_ylim(ylim)
    axs[1].yaxis.set_major_locator(ticker.MaxNLocator(4))
    plt.subplots_adjust(wspace=0.3, left=0.1, right=0.95, bottom=0.15, top=0.95)
    fig.savefig(filepath, dpi=300)
    print(f'Image saved as {filepath}')
    return best_combo


def plot_sliding_window_analysis(dataset, parameter, flag, radius_in_sec, save_to=None):
    d = dataset
    if save_to is None:
        save_to = os.path.join(d.plot_dir, 'plot_strides')
    if not os.path.exists(save_to):
        os.makedirs(save_to)
    filepath = os.path.join(save_to, f'{parameter}_around_{flag}_offset_analysis.{suf}')
    radius_in_ticks = np.ceil(radius_in_sec / d.dt)

    parsed_data_dir = f'{d.data_dir}/{parameter}_around_{flag}'
    file_description_path = os.path.join(parsed_data_dir, 'filename_description.csv')
    file_description = pd.read_csv(file_description_path, index_col=0, header=0)
    file_description = file_description[flag].dropna()
    offsets_in_ticks = file_description.index.values
    offsets_in_sec = np.round(offsets_in_ticks * d.dt, 3)
    means = []
    stds = []
    for offset in offsets_in_ticks:
        print(offset)
        data_filename = file_description.loc[offset]
        # print(data_filename)
        data_file_path = os.path.join(parsed_data_dir, data_filename)
        # print(data_file_path)

        segments = pd.read_csv(data_file_path, index_col=[0, 1], header=0)

        d = segments.droplevel('AgentID')
        # We plot distance so we prefer a cumulative plot
        d = d.T.cumsum().T
        tot_dsts = d.iloc[:, -1]
        mean = np.nanmean(tot_dsts)
        std = np.nanstd(tot_dsts)
        # print(f'mean : {mean}, std : {std}')
        means.append(mean)
        stds.append(std)

    fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    plt.rc('text', usetex=True)
    font = {'size': 15}
    plt.rc('font', **font)
    plt.rc('legend', **{'fontsize': 12})
    fig.suptitle('Scaled displacement per stride ', fontsize=25)
    fig.subplots_adjust(top=0.94, bottom=0.15, hspace=0.06)

    axs[0].scatter(np.arange(len(means)), means, marker='o', color='r', label='mean')
    # axs[0].set_title('Mean', fontsize=15)

    axs[1].scatter(np.arange(len(stds)), stds, marker='o', color='g', label='std')
    # axs[1].set_title('Standard deviation', fontsize=15)
    plt.xticks(ticks=np.arange(len(offsets_in_sec)), labels=offsets_in_sec)
    axs[1].set_xlabel('offset from velocity maximum, $sec$', fontsize=15)
    axs[0].set_ylabel('length fraction', fontsize=15)
    axs[1].set_ylabel('length fraction', fontsize=15)
    # axs[0].set_ylim([0.215, 0.235])
    # axs[1].set_ylim([0.04, 0.06])
    axs[0].legend(loc="upper right")
    axs[1].legend(loc="upper right")
    index_min_std = stds.index(min(stds))
    optimal_flag_phase_in_rad = 2 * np.pi * index_min_std / (len(stds) - 1)
    min_std = min(stds)
    mean_at_min_std = means[index_min_std]

    axs[1].annotate('min std', xy=(index_min_std, min_std + 0.0003), xytext=(-25, 25), textcoords='offset points',
                    arrowprops=dict(arrowstyle="-|>"))
    axs[0].annotate('', xy=(index_min_std, mean_at_min_std + 0.0005), xytext=(0, 25), textcoords='offset points',
                    arrowprops=dict(arrowstyle="-|>"))

    # plt.text(20, 2.5, rf'Distance mean', {'color': 'black', 'fontsize': 20})

    fig.savefig(filepath, dpi=300)
    print(f'Plot saved as {filepath}')
    return optimal_flag_phase_in_rad, mean_at_min_std


def plot_spatiotemporal_variation(dataset, spatial_cvs, temporal_cvs, sizes=None,
                                  save_to=None, save_as=f'velocity_flag.{suf}'):
    d = dataset
    Nvels = len(spatial_cvs)
    N_svels = int(Nvels / 2)
    N_lvels = int(Nvels / 2) - 1
    if save_to is None:
        save_to = d.plot_dir
    if not os.path.exists(save_to):
        os.makedirs(save_to)
    filepath = os.path.join(save_to, save_as)
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    # r'$\Theta_{bend}$'
    c = 'c'
    svel_num_strings = ['{' + str(i + 1) + '}' for i in range(N_svels)]
    lvel_num_strings = ['{' + str(i + 2) + '}' for i in range(N_lvels)]
    labels = [r'$v_{cen}$'] + \
             [rf'$v^{c}_{i}$' for i in lvel_num_strings] + [rf'$v_{i}$' for i in svel_num_strings]
    markers = ['sigma'] + ['o' for i in range(N_lvels)] + ['v' for i in range(N_svels)]
    cnum = 1 + N_svels
    cmap = plt.get_cmap('hsv')
    cmap = [cmap(1. * i / cnum) for i in range(cnum)]
    cmap = [cmap[0]] + cmap[2:] + cmap[1:]
    if sizes is None:
        for v, m, scv, tcv, c in zip(labels, markers, spatial_cvs, temporal_cvs, cmap):
            plt.scatter(scv, tcv, marker=m, c=c, label=v)
    else:
        for v, m, scv, tcv, c, s in zip(labels, markers, spatial_cvs, temporal_cvs, cmap, sizes):
            plt.scatter(scv, tcv, marker=m, c=c, label=v, s=s)
    plt.legend(loc='upper left', ncol=2, handleheight=2.4, labelspacing=0.05)
    # ax.legend(loc='upper left', fontsize=15, bbox_to_anchor=(1.03, 1))
    plt.ylabel(r'$\overline{cv}_{temporal}$')
    plt.xlabel(r'$\overline{cv}_{spatial}$')
    plt.tight_layout()
    fig.savefig(filepath, dpi=300)
    print(f'Image saved as {filepath}')


# def plot_2D_countour(x, y, z, dimensions, Cmax, filepath):
#     xmin, xmax = dimensions[0]
#     ymin, ymax = dimensions[1]
#     # define grid.
#     xi = np.linspace(xmin, xmax, 1000)
#     yi = np.linspace(ymin, ymax, 1000)
#     ## grid the data.
#     zi = interpolate.griddata((x, y), z, (xi[None, :], yi[:, None]), method='cubic')
#     levels = np.linspace(0.0, Cmax, 10000)
#     fig = plt.figure(figsize=(xmax - xmin, ymax - ymin))
#     # CS = plt.contour(xi, yi, zi, len(levels), linewidths=0.0, colors='k', levels=levels)
#     CS = plt.contourf(xi, yi, zi, len(levels), cmap=cm.Purples, levels=levels, alpha=0.9)
#     cbaxes = fig.add_axes([0.68, 0.93, 2.0, 0.2])
#     cbar = fig.colorbar(CS, cax=cbaxes, orientation="horizontal", ticks=[0, Cmax])
#     cbar.ax.set_xticklabels([0, f'${int(Cmax)} \mu$M'])
#
#     plt.xlim(xmin, xmax)
#     plt.ylim(ymin, ymax)
#     plt.locator_params(nbins=4)
#     fig.savefig(filepath, dpi=300)
#     print(f'Image saved as {filepath}')


# def gauss(x, y, Sigma, mu):
#     from lib.aux.par_aux import dot
#     X = np.vstack((x, y)).T
#     mat_multi = np.dot(dot(np.linalg.inv(Sigma)), (X - mu[None, ...]).T)
#     return np.diag(np.exp(-1 * (mat_multi)))


# def plot_2D_odorscape(dimensions, Cmax, Cstd, filepath, pos=None):
#     if pos is None:
#         pos = [0., 0.]
#     npts = 10000
#     x = np.random.uniform(dimensions[0][0], dimensions[0][1], npts)
#     y = np.random.uniform(dimensions[1][0], dimensions[1][1], npts)
#     z = gauss(x, y, Sigma=np.asarray([[Cstd, 0.0], [0.0, Cstd]]), mu=np.asarray(pos)) * Cmax
#     plot_2D_countour(x, y, z, dimensions=dimensions, Cmax=Cmax, filepath=filepath)


def plot_bend_change_over_displacement(dataset, return_fig=False):
    s = dataset.step_data
    save_to = os.path.join(dataset.plot_dir, 'plot_bend_change_over_displacement')
    if not os.path.exists(save_to):
        os.makedirs(save_to)
    figsize = (5, 5)

    b, o = 'bend', nam.unwrap(nam.orient('front'))
    bv, ov = nam.vel(b), nam.vel(nam.orient('front'))
    sd = nam.scal(dataset.distance)

    ind = s[sd].dropna().index
    b_data = s.loc[ind, b].values
    bv_data = s.loc[ind, bv].values
    ov_data = s.loc[ind, ov].values
    sd_data = s.loc[ind, sd].values

    bv_correction = bv_data / dataset.fr * np.sign(b_data)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.scatter(x=sd_data, y=bv_correction, marker='.')
    filename = f'bend_change_over_displacement.{suf}'
    return process_plot(fig, save_to, filename, return_fig)


def plot_EEB_vs_food_quality(samples=None, dt=None, species_list=['rover', 'sitter', 'default'],
                             save_to=None, return_fig=False, show=False, **kwargs):
    if samples is None:
        raise ('No sample configurations provided')
    from lib.model.modules.intermitter import get_EEB_poly1d
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


def plot_vel_during_strides(dataset, use_component=False, save_to=None, return_fig=False, show=False):
    chunk = 'stride'
    Npoints = 64
    d = dataset
    s = d.step_data

    if save_to is None:
        save_to = os.path.join(d.plot_dir, 'plot_vel_during_strides')
    if not os.path.exists(save_to):
        os.makedirs(save_to)
    save_as_lin = f'linear_velocities_during_strides.{suf}'
    save_as_ang = f'angular_velocity_during_strides.{suf}'
    filepath_lin = os.path.join(save_to, save_as_lin)
    filepath_ang = os.path.join(save_to, save_as_ang)
    filepaths = [filepath_lin, filepath_ang]

    svels = nam.scal(nam.vel(d.points))
    lvels = nam.scal(nam.lin(nam.vel(d.points[1:])))
    ids = d.agent_ids
    hov = nam.vel(nam.orient('front'))

    if use_component:
        lin_vels = lvels
    else:
        lin_vels = svels
    lin_vels = [lin_vels[0], lin_vels[int(len(lin_vels) / 2)], lin_vels[-1]]
    ang_vels = [hov]
    vels = [lin_vels, ang_vels]
    vels_list = lin_vels + ang_vels
    Nvels = len(vels_list)

    all_agents = [s.xs(id, level='AgentID', drop_level=True) for id in ids]
    all_flag_starts = [ag[ag[nam.start(chunk)] == True].index.values.astype(int) for ag in all_agents]
    all_flag_stops = [ag[ag[nam.stop(chunk)] == True].index.values.astype(int) for ag in all_agents]

    all_vel_timeseries = [[] for i in range(Nvels)]
    for agent_id, flag_starts, flag_stops in zip(ids, all_flag_starts, all_flag_stops):
        for start, stop in zip(flag_starts, flag_stops):
            for i, vel in enumerate(vels_list):
                vel_timeserie = s.loc[(slice(start, stop), agent_id), vel].values
                all_vel_timeseries[i].append(vel_timeserie)

    durations = [len(i) for i in all_vel_timeseries[0]]
    lin_vel_timeseries = all_vel_timeseries[:-1]
    ang_vel_timeseries = [[np.abs(a) for a in all_vel_timeseries[-1]]]
    vel_timeseries = [lin_vel_timeseries, ang_vel_timeseries]

    lin_cs = ['black', 'seagreen', 'mediumturquoise']
    ang_cs = ['black']
    cs = [lin_cs, ang_cs]
    lin_labels = [r'$\bf{head}$', r'$\bf{mid}$', r'$\bf{tail}$']
    ang_labels = [r'$\dot{\theta}_{or}$']
    labels = [lin_labels, ang_labels]
    ylabels = [r'scaled velocity $(sec^{-1})$', 'angular velocity $(deg/sec)$']

    for i in [0, 1]:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        for serie, vel, col, c, l in zip(vel_timeseries[i], vels[i], cs[i], cs[i], labels[i]):
            array = [np.interp(x=np.linspace(0, 2 * np.pi, Npoints), xp=np.linspace(0, 2 * np.pi, dur), fp=ts, left=0,
                               right=0) for dur, ts in zip(durations, serie)]
            plot_quantiles(df=array, from_np=True, axis=ax, color_mean=c, color_shading=col, label=l)

        Nticks = 5
        ticks = np.linspace(0, Npoints - 1, Nticks)
        ax.set_xticks(ticks=ticks)
        ax.set_xticklabels(labels=[r'$0$', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$'])
        ax.set_xlim([0, Npoints - 1])
        ax.set_ylabel(ylabels[i])
        ax.set_xlabel('$\phi_{stride}$')
        l = ax.legend(loc='upper right')
        for j, text in enumerate(l.get_texts()):
            text.set_color(cs[i][j])
        plt.subplots_adjust(bottom=0.2, top=0.95, left=0.1, right=0.95, wspace=0.01)
        fig.savefig(filepaths[i], dpi=300)
        print(f'Plot saved as {filepaths[i]}')


def plot_correlated_pars(dataset, pars, labels, save_to=None, save_as=f'correlated_pars.{suf}', return_fig=False):
    if len(pars) != 3:
        raise ValueError('Currently implemented only for 3 parameters')
    if save_to is None:
        save_to = dataset.plot_dir
    e = dataset.endpoint_data
    g = sns.PairGrid(e[pars])
    g.map_upper(sns.scatterplot)
    g.map_lower(sns.kdeplot, fill=True)
    g.map_diag(sns.histplot, kde=True, bins=20)

    for i, ax in enumerate(g.axes[-1, :]):
        ax.xaxis.set_label_text(labels[i])
    for j, ax in enumerate(g.axes[:, 0]):
        ax.yaxis.set_label_text(labels[j])
    for ax, (i, j) in zip([g.axes[0, 1], g.axes[0, 2], g.axes[1, 2]], [(1, 0), (2, 0), (2, 1)]):
        for std, a in zip([0.5, 1, 2, 3], [0.4, 0.3, 0.2, 0.1]):
            confidence_ellipse(x=e[pars[i]].values, y=e[pars[j]].values,
                               ax=ax, n_std=std, facecolor='red', alpha=a)
    return process_plot(g, save_to, save_as, return_fig)


def scatter_hist(xs, ys, labels, colors, Nbins=40, xlabel=None, ylabel=None, cumylabel=None, ylim=None, fig=None,
                 cumy=False):
    ticksize = 15
    labelsize = 15
    labelsize2 = 20
    # definitions for the axes
    left, width = 0.15, 0.6
    bottom, height = 0.12, 0.4
    dh = 0.01
    # dw = 0.01
    h = 0.2
    if not cumy:
        height += h
    h1 = bottom + dh + h
    h2 = h1 + height + dh
    w1 = left + width + dh

    y0, y1 = np.min([np.min(y) for y in ys]), np.max([np.max(y) for y in ys])
    ybins = np.linspace(y0, y1, Nbins)
    if ylim is None:
        ylim = (y0, y1)
    # ymax=0.4
    show_zero = True if ylim is not None and ylim[0] == -ylim[1] else False
    x0, x1 = np.min([np.min(x) for x in xs]), np.max([np.max(x) for x in xs])
    xbins = np.linspace(x0, x1, Nbins)
    dx = xbins[1] - xbins[0]
    xbin_mids = xbins[:-1] + dx / 2

    rect_scatter = [left, h1, width, height]
    rect_cumy = [left, h2, width, 1.1 * h]
    rect_histy = [w1 + dh, h1, h, height]
    rect_histx = [left, bottom, width, h]

    # start with a rectangular Figure
    if fig is None:
        fig = plt.figure(figsize=(10, 8))
    cc = {
        'left': True,
        'top': False,
        'bottom': True,
        'right': False,
        'labelsize': ticksize,
        'direction': 'in',
    }
    ax_scatter = plt.axes(rect_scatter)
    ax_scatter.tick_params(labelbottom=False, **cc)
    ax_histx = plt.axes(rect_histx)
    ax_histx.tick_params(**cc)
    ax_histy = plt.axes(rect_histy)
    ax_histy.tick_params(labelleft=False, **cc)

    ax_scatter.set_xlim([x0, x1])
    ax_scatter.set_ylim(ylim)
    ax_histx.set_xlim(ax_scatter.get_xlim())
    ax_histy.set_ylim(ax_scatter.get_ylim())
    ax_histy.set_xlabel('pdf', fontsize=labelsize)
    if xlabel is not None:
        ax_histx.set_xlabel(xlabel, fontsize=labelsize2)
    if ylabel is not None:
        ax_scatter.set_ylabel(ylabel, fontsize=labelsize2)

    if cumy:
        ax_cumy = plt.axes(rect_cumy)
        ax_cumy.tick_params(labelbottom=False, **cc)
        ax_cumy.set_xlim(ax_scatter.get_xlim())
    xmax_ps, ymax_ps = [], []
    for x, y, l, c in zip(xs, ys, labels, colors):
        ax_scatter.scatter(x, y, marker='.', color=c, alpha=1.0, label=l)
        if show_zero:
            ax_scatter.axhline(0.0, color='green', alpha=0.5, linestyle='dashed', linewidth=1)

        yw = np.ones_like(y) / float(len(y))
        y_vs0, y_vs1, y_patches = ax_histy.hist(y, bins=ybins, weights=yw, color=c, alpha=0.5, orientation='horizontal')

        y_vs1 = y_vs1[:-1] + (y_vs1[1] - y_vs1[0]) / 2
        y_smooth = np.polyfit(y_vs1, y_vs0, 5)
        poly_y = np.poly1d(y_smooth)(y_vs1)
        ax_histy.plot(poly_y, y_vs1, color=c, linewidth=2)

        xw = np.ones_like(x) / float(len(x))
        x_vs0, x_vs1, x_patches = ax_histx.hist(x, bins=xbins, weights=xw, color=c, alpha=0.5)
        x_vs1 = x_vs1[:-1] + (x_vs1[1] - x_vs1[0]) / 2
        x_smooth = np.polyfit(x_vs1, x_vs0, 5)
        poly_x = np.poly1d(x_smooth)(x_vs1)
        ax_histx.plot(x_vs1, poly_x, color=c, linewidth=2)

        xmax_ps.append(np.max(x_vs0))
        ymax_ps.append(np.max(y_vs0))
        ax_histx.set_ylabel('pdf', fontsize=labelsize)
        if cumy:
            xbinned_y = [y[(x0 <= x) & (x < x1)] for x0, x1 in zip(xbins[:-1], xbins[1:])]
            cum_y = np.array([np.sum(y) / len(y) for y in xbinned_y])
            ax_cumy.plot(xbin_mids, cum_y, color=c, alpha=0.5)
            if show_zero:
                ax_cumy.axhline(0.0, color='green', alpha=0.5, linestyle='dashed', linewidth=1)
            if cumylabel is not None:
                ax_cumy.set_ylabel(cumylabel, fontsize=labelsize)
    ax_histx.set_ylim([0.0, np.max(xmax_ps) + 0.05])
    ax_histy.set_xlim([0.0, np.max(ymax_ps) + 0.05])
    dataset_legend(labels, colors, ax=ax_scatter, loc='upper left', anchor=(1.0, 1.6) if cumy else None, fontsize=10)

    # plt.show()
    # raise
    return fig


# def plot_nengo(d, save_to=None):
#     if save_to is None:
#         save_to = d.plot_dir
#     s = d.step_data.xs(d.agent_ids[0], level='AgentID')
#     t = np.linspace(0, d.num_ticks * d.dt, d.num_ticks)
#     filename = f'nengo.{suf}'
#     filepath = os.path.join(save_to, filename)
#
#     pars = [['crawler_activity', 'turner_activity'], ['crawler_activity', 'feeder_motion']]
#     labels = [['crawler', 'turner'], ['crawler', 'feeder']]
#     colors = [['blue', 'red'], ['blue', 'green']]
#
#     try:
#         chunk1 = 'pause'
#         pau1s = s.index[s[f'{chunk1}_stop'] == True] * d.dt
#         pau0s = s.index[s[f'{chunk1}_start'] == True] * d.dt
#         pause = True
#     except:
#         pause = False
#     try:
#         chunk2 = 'stride'
#         str1s = s.index[s[f'{chunk2}_stop'] == True] * d.dt
#         str0s = s.index[s[f'{chunk2}_start'] == True] * d.dt
#         stride = True
#     except:
#         stride = False
#     fig, axs = plt.subplots(2, 1, figsize=(20, 5))
#     axs = axs.ravel()
#     for ax1, (p1, p2), (l1, l2), (c1, c2) in zip(axs, pars, labels, colors):
#         # ax1=axs[0]
#         ax2 = ax1.twinx()
#         ax1.plot(t, s[p1], color=c1, label=l1)
#         ax2.plot(t, s[p2], color=c2, label=l2)
#         ax1.legend(loc='upper left')
#         ax2.legend(loc='upper right')
#
#         if pause:
#             for start, stop in zip(pau0s, pau1s):
#                 plt.axvspan(start, stop, color='grey', alpha=0.3)
#         if stride:
#             for start, stop in zip(str0s, str1s):
#                 plt.axvspan(start, stop, color='blue', alpha=0.3)
#     plt.xlabel(r'time $(sec)$')
#     save_plot(fig, filepath, filename)


def calibration_plot(save_to=None, files=None):
    # plt.tick_params(
    tick_params = {
        'axis': 'both',  # changes apply to the x-axis
        'which': 'both',  # both major and minor ticks are affected
        'bottom': False,  # ticks along the bottom edge are off
        'top': False,  # ticks along the top edge are off
        'labelbottom': False,  # labels along the bottom edge are off
        'labeltop': False,
        'labelleft': False,
        'labelright': False,
    }

    filename = 'calibration.pdf'
    fig = plt.figure(constrained_layout=True, figsize=(6 * 5, 2 * 5))
    gs = fig.add_gridspec(2, 6)
    interference = fig.add_subplot(gs[:, :2])
    bouts = fig.add_subplot(gs[0, 2:4])
    orient = fig.add_subplot(gs[0, 4:])
    angular = fig.add_subplot(gs[1, 3:])
    bend = fig.add_subplot(gs[1, 2])

    if save_to is None:
        save_to = '.'
    if files is None:
        filenames = [
            'interference/interference_orientation.png',
            'bouts/stridesNpauses_cdf_restricted_0.png',
            'stride/stride_orient_change.png',
            'turn/angular_pars_3.png',
            'stride/stride_bend_change.png'
        ]
        files = [f'{save_to}/{f}' for f in filenames]
    images = [Image.open(f) for f in files]
    axes = [interference, bouts, orient, angular, bend]
    for ax, im in zip(axes, images):
        ax.tick_params(**tick_params)
        ax.axis('off')
        ax.imshow(im, cmap=None, aspect=None)
    filepath = os.path.join(save_to, filename)
    save_plot(fig, filepath, filename)
    return fig


def boxplot_double_patch(xlabel='substrate', complex_colors=True, **kwargs):
    P = Plot(name='double_patch', **kwargs)
    DataIDs = unique_list([d.config['group_id'] for d in P.datasets])
    ModIDs = unique_list([l.split('_')[-1] for l in DataIDs])
    subIDs = unique_list([l.split('_')[0] for l in DataIDs])
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
                pair_vs = flatten_list([dic[id] for id in subModIDs])
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
                y=np.array(dic[p])
                dim=y.shape[1]
                if dim==1:
                    P.axs[idx].plot(x, y, color=c, label=p)
                else :
                    for jj in range(dim) :
                        P.axs[idx].plot(x, y[:,jj], label=f'{p}_{jj}')
                P.conf_ax(idx, xlab=r'time $min$' if Nrow == Nrows - 1 else None, ylab='activity' if j == 0 else None,
                          yticks=[] if j != 0 else None, yticklabels=[] if j != 0 else None, yMaxN=yMaxN,
                          leg_loc='upper right')
    P.adjust((0.1, 0.95), (0.1, 0.95), 0.01, 0.05)
    return P.get()


def ggboxplot(p='length', subfolder='ggplot', **kwargs):
    from plotnine import ggplot, aes, geom_boxplot, scale_color_manual, theme
    P = Plot(name=p, subfolder=subfolder, **kwargs)
    e = concat_datasets(P.datasets, key='end')
    Cdict = dict(zip(P.labels, P.colors))
    P.fig = (ggplot(e, aes(x='GroupID', y=p, color='GroupID')) + geom_boxplot() + scale_color_manual(Cdict) + theme(
        figure_size=(12, 6))).draw()
    return P.get()


# def plot_foraging(**kwargs) :
#     P = Plot(name='foraging', **kwargs)
#     P.build(2, 1, figsize=(10, 10), sharex=True)
#     for i, d in enumerate(P.datasets):
#         dics = d.load_dicts('foraging')
#         for dic in dics :
#             for j,(action, vs) in enumerate(dic.items()):
#                 for k, (foodtype, timeseries) in enumerate(vs.items()) :
#                     P.axs[j].plot(timeseries, color='red')
#     P.get()

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
        par = getPar(action, to_return=['lab'])[0]
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
            # g1.get_legend().remove()

            # annotate_plot(**kws)
            P.conf_ax(yMaxN=4, leg_loc='upper right')
    # P.conf_ax(xlab=xlab, ylab='probability, $P$', xlim=xlim, yMaxN=4, leg_loc='upper right')
    P.adjust((0.1, 0.95), (0.15, 0.92), 0.2, 0.005)
    P.get()


graph_dict = {
    'crawl pars': plot_crawl_pars,
    'angular pars': plot_ang_pars,
    'endpoint params': plot_endpoint_params,
    'stride Dbend': plot_stride_Dbend,
    'stride Dor': plot_stride_Dorient,
    'interference': plot_interference,
    'dispersion': plot_dispersion,
    'runs & pauses': plot_stridesNpauses,
    'turn duration': plot_turn_duration,
    'turn amplitude': plot_turns,
    'marked_strides': plot_marked_strides,
    'turn amplitude VS Y pos': plot_turn_amp,
    'turn Dbearing to center': plot_turn_Dorient2center,
    'chunk Dbearing to source': plot_chunk_Dorient2source,
    'C odor (real)': plot_odor_concentration,
    'C odor (perceived)': plot_sensed_odor_concentration,
    'navigation index': plot_navigation_index,
    'Y pos': plot_Y_pos,
    'PI (boxplot)': boxplot_PI,
    'pathlength': plot_pathlength,
    'food intake (timeplot)': plot_food_amount,
    'gut': plot_gut,
    'food intake (barplot)': intake_barplot,
    'deb': plot_debs,
    'timeplot': timeplot,
    'ethogram': plot_ethogram,
    'foraging': plot_foraging,
    'barplot': barplot,
    'scatter': plot_2pars,
    'nengo': plot_nengo_network,
    'ggboxplot': ggboxplot
}
