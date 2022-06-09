import heapq
import itertools
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm, ticker, patches
from scipy import stats

import os

from lib.plot.base import AutoPlot, Plot, BasePlot
from lib.plot.aux import dataset_legend, plot_quantiles, confidence_ellipse, \
    save_plot, plot_config, process_plot
from lib.aux import naming as nam

from lib.conf.pars.pars import getPar
from lib.process.aux import comp_pooled_epochs, compute_interference

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


def plot_marked_turns(dataset, agent_ids=None, agent_idx=[0], turn_epochs=['Rturn', 'Lturn'],
                      vertical_boundaries=False, min_turn_angle=0, slices=[], subfolder='individuals',
                      save_to=None, return_fig=False, show=False, sizes=['short', 'long']):
    Ndatasets, colors, save_to, labels = plot_config(datasets=[dataset], labels=[dataset.id], save_to=save_to,
                                                     subfolder=subfolder)
    # We plot the complete or a slice of the timeseries of scal centroid velocity. The grey areas are stridechains
    d = dataset

    if agent_ids is None:
        if agent_idx is None:
            agent_ids = d.agent_ids
        else:
            agent_ids = [d.agent_ids[idx] for idx in agent_idx]

    xx = f'marked_turns_min_angle_{min_turn_angle}'
    filepath_full = f'{xx}_full.{suf}'
    filepath_full_long = f'{xx}_full_long.{suf}'
    filepath_slices = []
    for i, slice in enumerate(slices):
        filepath_slices.append(f'{xx}_slice_{i}.{suf}')
    generic_filepaths = [filepath_full_long] + filepath_slices

    figsize_short = (20, 5)
    figsize_long = (15 * 6, 5)
    figsizes = [figsize_long] + [figsize_short] * len(generic_filepaths)

    xlims = [(0, d.duration)] + slices

    # ymax=1.0

    b = 'bend'
    bv = nam.vel(b)
    ho = nam.orient('front')
    hov = nam.vel(ho)
    fig_dict = {}
    for agent_id in agent_ids:
        filepaths = [f'{agent_id}_{f}' for f in generic_filepaths]

        s = d.step_data.xs(agent_id, level='AgentID', drop_level=True)
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
                    try:
                        idx01 = d.load_chunk_dicts()[agent_id][chunk] / d.fr
                        if min_turn_angle > 0:
                            angles = np.abs([np.trapz(hov0[s0:s1 + 1], dx=d.dt) for s0, s1 in idx01])
                            idx01 = idx01[angles >= min_turn_angle]
                        start_indexes, stop_indexes = idx01[:, 0], idx01[:, 1]
                    except:
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
            # print(xlim)
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


def plot_bend2orientation_analysis(dataset, save_to=None, save_as=f'bend2orientation.{suf}'):
    from sklearn.linear_model import LinearRegression
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

    scores2 = []
    coefs2 = []
    for i in k:
        X = X0[:, 0:i + 1]
        reg = LinearRegression().fit(X, y)
        scores2.append(reg.score(X, y))
        coefs2.append(reg.coef_)
    axs[0].scatter(np.arange(1, N + 1), scores2, c='green', alpha=1.0, marker="o", label='cumulative', s=200)
    axs[0].plot(np.arange(1, N + 1), scores2, c='green')
    shape1 = patches.Circle((0, 0), 1, facecolor='blue')
    shape2 = patches.Rectangle((0, 0), 1, 1, facecolor='green')
    axs[0].legend(loc='lower left')
    axs[0].yaxis.set_major_locator(ticker.MaxNLocator(4))
    ylim = [0.6, 1]
    plt.bar(x=[','.join(map(str, c)) for c in best_combos_ind], height=max_corrs, width=0.8, color='black')
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
        data_filename = file_description.loc[offset]
        data_file_path = os.path.join(parsed_data_dir, data_filename)

        segments = pd.read_csv(data_file_path, index_col=[0, 1], header=0)

        d = segments.droplevel('AgentID')
        # We plot distance so we prefer a cumulative plot
        d = d.T.cumsum().T
        tot_dsts = d.iloc[:, -1]
        mean = np.nanmean(tot_dsts)
        std = np.nanstd(tot_dsts)
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
    axs[1].scatter(np.arange(len(stds)), stds, marker='o', color='g', label='std')
    plt.xticks(ticks=np.arange(len(offsets_in_sec)), labels=offsets_in_sec)
    axs[1].set_xlabel('offset from velocity maximum, $sec$', fontsize=15)
    axs[0].set_ylabel('length fraction', fontsize=15)
    axs[1].set_ylabel('length fraction', fontsize=15)
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


def plot_spatiotemporal_variation(spatial_cvs, temporal_cvs, sizes=None, dataset=None,
                                  save_to=None, save_as=f'velocity_flag.{suf}'):
    Nvels = len(spatial_cvs)
    N_svels = int(Nvels / 2)
    N_lvels = int(Nvels / 2) - 1
    if save_to is None:
        if dataset is not None:
            save_to = dataset.plot_dir
        else:
            raise ValueError('Provide "save_to" directory')
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


def plot_segmentation_definition(subfolder='metric_definition', axs=None, fig=None, **kwargs):
    P = Plot(name=f'segmentation_definition', subfolder=subfolder, **kwargs)
    P.build(1, P.Ndatasets * 2, figsize=(5 * P.Ndatasets, 5), sharex=False, sharey=False, fig=fig, axs=axs)
    Nbest = 5
    for ii, d in enumerate(P.datasets):
        ax1, ax2 = P.axs[ii * 2], P.axs[ii * 2 + 1]
        N = d.Nangles
        dic = d.load_vel_definition()
        df_reg = dic['/bend2or_regression']
        df_corr = dic['/bend2or_correlation']

        df_reg.sort_index(inplace=True)
        single_scores = df_reg['single_score'].values
        cum_scores = df_reg['cum_score'].values
        x = np.arange(1, N + 1)
        ax1.scatter(x, single_scores, c='blue', alpha=1.0, marker=",", label='single', s=200)
        ax1.plot(x, single_scores, c='blue')

        ax1.scatter(x, cum_scores, c='green', alpha=1.0, marker="o", label='cumulative', s=200)
        ax1.plot(x, cum_scores, c='green')

        P.conf_ax(ii * 2, xlab=r'angular velocity, $\dot{\theta}_{i}$', ylab='regression score',
                  xticks=x, yMaxN=4, leg_loc='lower left')

        df_corr.sort_values('corr', ascending=False, inplace=True)
        max_corrs = df_corr['corr'].values[:Nbest]
        best_combos = df_corr.index.values[:Nbest]
        xx = [','.join(map(str, cc)) for cc in best_combos]
        ax2.bar(x=xx, height=max_corrs, width=0.5, color='black')
        P.conf_ax(ii * 2 + 1, xlab='combined angular velocities', ylab='Pearson correlation', yMaxN=4, ylim=(0.5, 1))
        ax2.tick_params(axis='x', which='major', labelsize=20)
        P.adjust(LR=(0.1, 0.95), BT=(0.15, 0.95), W=0.3)
        return P.get()


def plot_stride_variability(component_vels=True, subfolder='metric_definition', axs=None, fig=None, **kwargs):
    P = Plot(name=f'stride_spatiotemporal_variation', subfolder=subfolder, **kwargs)
    P.build(1, P.Ndatasets, figsize=(5 * P.Ndatasets, 5), sharex=True, sharey=True, fig=fig, axs=axs)
    for ii, d in enumerate(P.datasets):
        ax = P.axs[ii]
        try:
            dic = d.load_vel_definition()
        except:
            d.save_vel_definition(component_vels=component_vels)
            dic = d.load_vel_definition()
        stvar = dic['/stride_variability']
        stvar.sort_values(by='idx', inplace=True)
        ps = stvar.index if component_vels else [p for p in stvar.index if 'lin' not in p]
        for p in ps:
            row = stvar.loc[p]
            ax.scatter(x=row['scaled_stride_dst_var'], y=row['stride_dur_var'], marker=row['marker'], s=200,
                       color=row['color'], label=row['symbol'])
        ax.legend(ncol=2, handleheight=1.7, labelspacing=0.01, loc='lower right')
        ax.set_ylabel(r'$\overline{cv}_{temporal}$')
        ax.set_xlabel(r'$\overline{cv}_{spatial}$')
    return P.get()


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
    from PIL import Image
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


def annotated_strideplot(a, dt, a2plot=None, ax=None, ylim=None, xlim=None, show_extrema=True, show_strides=True,
                         moving_average_interval=None, **kwargs):
    """
    Plots annotated strides-runs and pauses in timeseries.

    Extended description of function.

    Parameters
    ----------
    a : array
        1D np.array : velocity timeseries
    dt : float
        Timestep of the timeseries
    ax : obj
        The matplotlib axis on which to draw
    ylim : Tuple[float,float]
        The yaxis boundaries
    xlim : Tuple[float,float]
        The xaxis boundaries.Default is the whole a.
    show_extrema : bool
        Annotate minima & maxima. Default : True.
    show_strides : bool
        Annotate strides by vertical dashed lines. Default : True.
    moving_average_interval : float
        Plot moving average of velocity over a time interval instead of the actual. Default : None.
    **kwargs : dict
        Other arguments for bout annotation

    Returns
    -------
    ax : obj
        The drawn matplotlib axis

    """
    from lib.process.aux import detect_strides, detect_pauses, moving_average
    chunk_cols = ["lightblue", "grey"]
    trange = np.arange(0, a.shape[0] * dt, dt)
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(20, 5), sharex=True, sharey=True)
    if xlim is None:
        xlim = (0, trange[-1])

    i_min, i_max, strides, runs, run_counts = detect_strides(a=a, dt=dt, **kwargs)
    pauses = detect_pauses(a, dt, runs=runs)

    if moving_average_interval:
        a = moving_average(a, n=int(moving_average_interval / dt))
    if a2plot is not None:
        ax.plot(trange, a2plot)
    else:
        ax.plot(trange, a)
        ax.set_ylim(ylim)
        ax.set_ylabel("velocity (1/sec)")
        if show_extrema:
            ax.plot(trange[i_max], a[i_max], linestyle='None', lw=10, color='green', marker='v')
            ax.plot(trange[i_min], a[i_min], linestyle='None', lw=10, color='red', marker='^')
    ax.set_xlabel("time (sec)")
    ax.set_xlim(xlim)

    if show_strides:
        for s0, s1 in strides:
            # ax.axvspan(trange[s0], trange[s1], color=chunk_cols[0], alpha=1.0)
            ax.axvline(trange[s0], color=f'{0.4 * (0 + 1)}', alpha=0.3, linestyle='dashed', linewidth=1)
            ax.axvline(trange[s1], color=f'{0.4 * (0 + 1)}', alpha=0.3, linestyle='dashed', linewidth=1)
    for s0, s1 in runs:
        ax.axvspan(trange[s0], trange[s1], color=chunk_cols[0], alpha=1.0)
    for p0, p1 in pauses:
        ax.axvspan(trange[p0], trange[p1], color=chunk_cols[1], alpha=1.0)
    labels = ['runs', 'pauses']
    handles = [patches.Patch(color=col, label=n) for n, col in zip(labels, chunk_cols)]
    ax.legend(loc="upper right", handles=handles, labels=labels, fontsize=15)


def annotated_turnplot(a, dt, a2plot=None, ax=None, min_dur=None, min_amp=None, ylim=None, xlim=None,
                       moving_average_interval=None, **kwargs):
    """
    Plots annotated turnss in timeseries.

    Extended description of function.

    Parameters
    ----------
    a : array
        1D np.array : velocity timeseries
    dt : float
        Timestep of the timeseries
    ax : obj
        The matplotlib axis on which to draw
    ylim : Tuple[float,float]
        The yaxis boundaries
    xlim : Tuple[float,float]
        The xaxis boundaries.Default is the whole a.
    show_extrema : bool
        Annotate minima & maxima. Default : True.
    show_strides : bool
        Annotate strides by vertical dashed lines. Default : True.
    moving_average_interval : float
        Plot moving average of velocity over a time interval instead of the actual. Default : None.
    **kwargs : dict
        Other arguments for bout annotation

    Returns
    -------
    ax : obj
        The drawn matplotlib axis

    """
    from lib.process.aux import detect_turns, process_epochs, moving_average

    chunk_cols = ["lightgreen", "orange"]
    trange = np.arange(0, a.shape[0] * dt, dt)
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(20, 5), sharex=True, sharey=True)
    if xlim is None:
        xlim = (0, trange[-1])

    Lturns, Rturns = detect_turns(a, dt, min_dur=min_dur)
    if min_amp is not None:
        Lturns1, Ldurs, Lturn_slices, Lamps, Lturn_idx, Lmaxs = process_epochs(a, Lturns, dt)
        Rturns1, Rdurs, Rturn_slices, Ramps, Rturn_idx, Rmaxs = process_epochs(a, Rturns, dt)
        Lturns = Lturns[np.abs(Lamps) > min_amp]
        Rturns = Rturns[np.abs(Ramps) > min_amp]

    if moving_average_interval:
        a = moving_average(a, n=int(moving_average_interval / dt))
    if a2plot is not None:
        ax.plot(trange, a2plot)
    else:
        ax.plot(trange, a)

        ax.set_ylabel("angular velocity (deg/sec)")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("time (sec)")
    ax.axhline(0, color='black', alpha=1, linestyle='dashed', linewidth=1)
    for s0, s1 in Lturns:
        ax.axvspan(trange[s0], trange[s1], color=chunk_cols[0], alpha=1.0)
    for p0, p1 in Rturns:
        ax.axvspan(trange[p0], trange[p1], color=chunk_cols[1], alpha=1.0)
    labels = ['L turns', 'R turns']
    handles = [patches.Patch(color=col, label=n) for n, col in zip(labels, chunk_cols)]
    ax.legend(loc="upper right", handles=handles, labels=labels, fontsize=15)


def stride_cycle(shorts=['sv', 'fov', 'rov', 'foa'], modes=None, Nbins=64, individuals=False, pooled=True, **kwargs):
    x = np.linspace(0, 2 * np.pi, Nbins)
    Nsh = len(shorts)
    P = AutoPlot(name=f'pooled_norm_average_curves', Nrows=Nsh, sharex=True, figsize=(10, 4 * Nsh), **kwargs)
    for ii, sh in enumerate(shorts):
        par, lab, sym = getPar(sh, to_return=['d', 'lab', 'symbol'])
        if modes is None:
            mode = 'abs' if sh == 'sv' else 'norm'
        else:
            mode = modes[ii]

        for d in P.datasets:
            c = d.config
            col = c.color if 'color' in c.keys() else d.color
            if individuals:
                try:
                    cycle_curves = d.cycle_curves
                except:
                    cycle_curves = d.load_cycle_curves()
                if cycle_curves is None:
                    s, e = d.step_data, d.endpoint_data
                    cycle_curves = compute_interference(s, e, c=c)
                if cycle_curves is not None:
                    df = cycle_curves[sh][mode]
                    if pooled:
                        plot_quantiles(df=df, from_np=True, axis=P.axs[ii], color_shading=col, x=x, label=d.id)
                    else:
                        for j in range(df.shape[0]):
                            P.axs[ii].plot(x, df[j, :], color=col)
                        P.axs[ii].plot(x, np.nanquantile(df, q=0.5, axis=0), label=d.id, color=col)

            else:
                if 'pooled_cycle_curves' not in c.keys():
                    s, e = d.step_data, d.endpoint_data
                    compute_interference(s, e, c=c)
                # pooled_cycle_curves=c.pooled_cycle_curves
                P.axs[ii].plot(x, np.array(c.pooled_cycle_curves[sh][mode]), label=d.id, color=col)


        P.conf_ax(ii, xticks=np.linspace(0, 2 * np.pi, 5), xlim=[0, 2 * np.pi],
                  xticklabels=[r'$0$', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$'],
                  xlab='$\phi_{stride}$', ylab=sym, xvis=True if ii == Nsh - 1 else False)
    P.axs[0].legend(loc='upper left', fontsize=15)
    P.fig.subplots_adjust(hspace=0.01)
    P.fig.align_ylabels(P.axs[:])
    return P.get()


def stride_cycle_individual(s=None, e=None, c=None, ss=None, fr=None, dt=1 / 16, short='fov', idx=0, Nbins=64,
                            color_solo='grey', color='red',
                            absolute=False, save_to=None, pooled=False,
                            ylim=None, axs=None, fig=None, show=False):
    from lib.process.aux import detect_strides
    p, sv, fv = getPar([short, 'sv', 'fv'])
    if ss is None:
        id = c.agent_ids[idx]
        ee = e.loc[id]
        ss = s.xs(id, level='AgentID')
        fr = ee[fv]
        dt = c.dt
    ssp = ss[p].abs().values if absolute else ss[p].values
    strides = detect_strides(ss[sv], dt, fr=fr, return_runs=False, return_extrema=False)
    strides = strides.tolist()
    pi2 = 2 * np.pi
    x = np.linspace(0, pi2, Nbins)

    if axs is None and fig is None:
        fig, axs = plt.subplots(1, 1, figsize=(15, 6))

    aa = np.zeros([len(strides), Nbins])
    for ii, (s0, s1) in enumerate(strides):
        aa[ii, :] = np.interp(x, np.linspace(0, pi2, s1 - s0), ssp[s0:s1])
        if not pooled:
            axs.plot(x, aa[ii, :], color_solo, linewidth=1, alpha=0.5, zorder=2)

    if pooled:
        plot_quantiles(df=aa, from_np=True, axis=axs, color_shading=color, x=x)
    else:
        aa_mu = np.nanquantile(aa, q=0.5, axis=0)
        axs.plot(x, aa_mu, color, linewidth=5, alpha=1.0, zorder=10)
    axs.set_xlabel('$\phi_{stride}$')
    axs.yaxis.set_major_locator(ticker.MaxNLocator(5))
    axs.set_xlim([0, pi2])
    axs.set_ylim(ylim)
    axs.set_xticks(np.linspace(0, pi2, 5))
    axs.set_xticklabels([r'$0$', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$'])
    if save_to is not None:
        fig.savefig(f'{save_to}/stride_cycle_individual.pdf', dpi=300)
    if show:
        plt.show()


def stride_cycle_all_points(s, e, c, idx=0, Nbins=64, short=None, ang_absolute=True, maxNpoints=5, save_to=None,
                            axs=None, fig=None, axx=None):
    from lib.process.aux import detect_strides, stride_interp,compute_velocity
    l, sv, pau_fov_mu, fv, fov = getPar(['l', 'sv', 'pau_fov_mu', 'fv', 'fov'])
    att = 'attenuation'
    att_max, att_min, phi_att_max, phi_sv_max = nam.max(att), nam.min(att), nam.max(f'phi_{att}'), nam.max(f'phi_{sv}')

    points0 = nam.midline(c.Npoints, type='point')
    id = c.agent_ids[idx]
    ee = e.loc[id]
    ss = s.xs(id, level='AgentID')
    strides = detect_strides(ss[sv], c.dt, fr=ee[fv], return_runs=False, return_extrema=False)
    # strides = strides.tolist()
    pi2 = 2 * np.pi
    x = np.linspace(0, pi2, Nbins)

    if axs is None and fig is None and axx is None:
        Nrows = 2 if short else 1
        fig, axs = plt.subplots(Nrows, 1, figsize=(15, 6 * Nrows))
        axs = axs.ravel() if short else [axs]
        axx = fig.add_axes([0.64, 0.4, 0.25, 0.12])
        fig.subplots_adjust(hspace=0.1, left=0.15, right=0.9, bottom=0.2, top=0.9)
    if short is not None:
        par, lab = getPar(short, to_return=['d', 'lab'])
        a_sh = ss[par].values
        a_fov = ss[getPar('fov')].values
        da = np.array([np.trapz(a_fov[s0:s1]) for ii, (s0, s1) in enumerate(strides)])

        aa = stride_interp(a_sh, strides, Nbins)
        aa_minus = aa[da < 0]
        aa_plus = aa[da > 0]
        aa_norm = np.vstack([aa_plus, -aa_minus])

        plot_quantiles(df=aa_norm, from_np=True, axis=axs[1], color_shading='blue', x=x, label='experiment')

        axs[1].set_ylabel(lab)

    if c.Npoints > maxNpoints:
        points = [points0[0]] + [points0[2 + int(ii * (c.Npoints - 2) / (maxNpoints - 2))] for ii in
                                 range(maxNpoints - 2)] + [points0[-1]]
    else:
        points = points0
    if len(points) == 5:
        pointcols = ['black', 'darkblue', 'darkgreen', 'seagreen', 'mediumturquoise']
    else:
        pointcols = cm.rainbow(np.linspace(0, 1, len(points)))
    ymax = 0.7
    for p, col in zip(points, pointcols):
        v_p = nam.vel(p)
        a = ss[v_p] if v_p in ss.columns else compute_velocity(ss[nam.xy(p)].values, dt=c.dt)
        a = a / ee[l]
        aa = np.zeros([len(strides), Nbins])
        for ii, (s0, s1) in enumerate(strides):
            aa[ii, :] = np.interp(x, np.linspace(0, pi2, s1 - s0), a[s0:s1])
        aa_mu = np.nanquantile(aa, q=0.5, axis=0)
        aa_max = np.max(aa_mu)
        phi_max = x[np.argmax(aa_mu)]
        plot_quantiles(df=aa, from_np=True, axis=axs[0], color_shading=col, x=x, label=p)
        axs[0].axvline(phi_max, ymax=aa_max / ymax, color=col, alpha=1, linestyle='dashed', linewidth=2, zorder=20)
        axs[0].scatter(phi_max, aa_max + 0.02 * ymax, color=col, marker='v', linewidth=2, zorder=20)

    axs[0].set_ylabel(r'scaled velocity $(sec^{-1})$')
    axs[0].set_ylim([0, ymax])
    for ax in axs:
        ax.set_xlabel('$\phi_{stride}$')
        ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
        ax.set_xlim([0, pi2])
        ax.set_xticks(np.linspace(0, pi2, 5))
        ax.set_xticklabels([r'$0$', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$'])
        ax.legend(loc='upper left', fontsize=15)

    try:
        ps = [nam.max(f'phi_{nam.vel(p)}') for i, p in enumerate(points0)]
        aa = np.zeros([c.Npoints, c.N]) * np.nan
        for i, p in enumerate(ps):
            aa[i, :] = e[p].values - e[phi_att_max].values
        axx.violinplot(aa.T, widths=0.9)
        axx.set_ylabel(r'$\Delta\phi$')
        axx.set_xlabel('# point')
        axx.set_xticks(np.arange(c.Npoints + 1))
        axx.set_xticklabels([None] + np.arange(1, c.Npoints + 1, 1).tolist())
        axx.set_yticks([-np.pi / 2, 0, np.pi / 2, np.pi])
        axx.set_yticklabels([r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'])
        axx.tick_params(axis='both', which='minor', labelsize=12)
        axx.tick_params(axis='both', which='major', labelsize=12)
        axx.axhline(0, color='green', alpha=0.5, linestyle='dashed', linewidth=1)
    except:
        pass
    if save_to is not None:
        fig.savefig(f'{save_to}/stride_cycle_all_points.pdf', dpi=300)



def plot_fft(s, c, palette=None, axx=None, ax=None, fig=None, **kwargs):
    from scipy.fft import fftfreq
    from lib.aux.sim_aux import fft_max
    if palette is None:
        palette = {'v': 'red', 'fov': 'blue'}
    P = BasePlot(name=f'fft_powerspectrum', **kwargs)
    P.build(fig=fig, axs=ax, figsize=(15, 12))
    if axx is None:
        axx = P.fig.add_axes([0.64, 0.65, 0.25, 0.2])
    xf = fftfreq(c.Nticks, c.dt)[:c.Nticks // 2]

    l, v, fov = getPar(['l', 'v', 'fov'])
    fvs = np.zeros(c.N) * np.nan
    ffovs = np.zeros(c.N) * np.nan
    v_ys = np.zeros([c.N, c.Nticks // 2])
    fov_ys = np.zeros([c.N, c.Nticks // 2])

    for j, id in enumerate(c.agent_ids):
        ss = s.xs(id, level='AgentID')
        fvs[j], v_ys[j, :] = fft_max(ss[v], c.dt, fr_range=(1.0, 2.5), return_amps=True)
        ffovs[j], fov_ys[j, :] = fft_max(ss[fov], c.dt, fr_range=(0.1, 0.8), return_amps=True)
    plot_quantiles(v_ys, from_np=True, x=xf, axis=P.axs[0], label='forward speed', color_shading=palette['v'])
    plot_quantiles(fov_ys, from_np=True, x=xf, axis=P.axs[0], label='angular speed', color_shading=palette['fov'])
    xmax = 3.5
    P.conf_ax(0, ylim=(0, 4), xlim=(0, xmax), ylab='Amplitude', xlab='Frequency (Hz)',
              title='Fourier analysis', leg_loc='lower left', yMaxN=5)

    bins = np.linspace(0, 2, 40)

    v_weights = np.ones_like(fvs) / float(len(fvs))
    fov_weights = np.ones_like(ffovs) / float(len(ffovs))
    axx.hist(fvs, color=palette['v'], bins=bins, weights=v_weights)
    axx.hist(ffovs, color=palette['fov'], bins=bins, weights=fov_weights)
    axx.set_xlabel('Dominant frequency (Hz)')
    axx.set_ylabel('Probability')
    axx.tick_params(axis='both', which='minor', labelsize=12)
    axx.tick_params(axis='both', which='major', labelsize=12)
    axx.yaxis.set_major_locator(ticker.MaxNLocator(2))
    return P.get()


def plot_fft_multi(axx=None, ax=None, fig=None, **kwargs):
    P = Plot(name=f'fft_powerspectrum', **kwargs)
    P.build(fig=fig, axs=ax, figsize=(15, 12))
    if axx is None:
        axx = P.fig.add_axes([0.64, 0.65, 0.25, 0.2])
    for d in P.datasets:
        try:
            s = d.read(key='step', file='data_h5')
        except:
            s = d.step_data
        c = d.config
        _ = plot_fft(s, c, axx=axx, ax=P.axs[0], fig=P.fig, palette={'v': d.color, 'fov': d.color}, return_fig=True)
    return P.get()


def plot_trajectories_solo(s, c, unit='mm', fig=None, axs=None, **kwargs):
    scale = 1000 if unit == 'mm' else 1
    from lib.aux.sim_aux import get_tank_polygon
    P = BasePlot(name=f'trajectories', **kwargs)
    P.build(fig=fig, axs=axs)
    tank = get_tank_polygon(c, return_polygon=False) * scale
    for id in c.agent_ids:
        xy = s[['x', 'y']].xs(id, level="AgentID").values * scale
        P.axs[0].plot(xy[:, 0], xy[:, 1])
    P.axs[0].fill(tank[:, 0], tank[:, 1], fill=True, color='lightgrey', edgecolor='black', linewidth=4)
    P.conf_ax(xMaxN=3, yMaxN=3, title=c.id, xlab=f'X ({unit})', ylab=f'Y ({unit})', equal_aspect=True)
    return P.get()


def plot_trajectories(axs=None, fig=None, unit='mm', name=f'comparative_trajectories', subfolder='trajectories',
                      range=None, mode=None, **kwargs):
    def get_traj(d, mode=None):
        if mode == 'origin':
            try:
                s = d.get_traj_aligned(mode)[['x', 'y']]
                return s
            except:
                return get_traj(d, None)
        else:
            try:
                try:
                    s = d.read(key='trajectories', file='aux_h5')
                    return s
                except:
                    s = d.step_data[['x', 'y']]
                    return s
            except:
                s = d.step_data[['x', 'y']]
                return s

    P = Plot(name=name, subfolder=subfolder, **kwargs)
    P.build(1, P.Ndatasets, figsize=(5 * P.Ndatasets, 6), sharex=True, sharey=True, fig=fig, axs=axs)
    for ii, d in enumerate(P.datasets):
        print(d.id,mode,range, ii)
        s = get_traj(d, mode)
        c = d.config
        if range is not None:
            t0, t1 = range
            tick0, tick1 = int(t0 / c.dt), int(t1 / c.dt)
            s = s.loc[tick0:tick1]

        _ = plot_trajectories_solo(s, c, unit=unit, fig=P.fig, axs=P.axs[ii], save_to=None)
        if ii != 0:
            P.axs[ii].yaxis.set_visible(False)
    P.adjust((0.07, 0.95), (0.1, 0.95), 0.05, 0.005)
    return P.get()

def plot_single_bout(x0, discr, bout, i, color, label, axs, fit_dic=None, plot_fits='best',
                     marker='.', legend_outside=False,xlabel = 'time (sec)',xlim=None, **kwargs):
    distro_ls = ['powerlaw', 'exponential', 'lognormal', 'lognorm-pow', 'levy', 'normal', 'uniform']
    distro_cs = ['c', 'g', 'm', 'k', 'orange', 'brown', 'purple']
    num_distros = len(distro_ls)
    lws = [2] * num_distros

    if fit_dic is None:
        from lib.anal.fitting import fit_bout_distros
        xmin, xmax = np.min(x0), np.max(x0)
        fit_dic = fit_bout_distros(x0, xmin, xmax, discr, dataset_id='test', bout=bout, **kwargs)
    idx_Kmax = fit_dic['idx_Kmax']
    cdfs = fit_dic['cdfs']
    pdfs = fit_dic['pdfs']
    u2, du2, c2, c2cum = fit_dic['values']
    lws[idx_Kmax] = 4
    ylabel = 'probability'
    xlabel = xlabel
    xrange = u2
    y = c2cum
    ddfs = cdfs
    for ii in ddfs:
        if ii is not None:
            ii /= ii[0]
    axs[i].loglog(xrange, y, marker, color=color, alpha=0.7, label=label)
    axs[i].set_title(bout)
    axs[i].set_xlabel(xlabel)
    axs[i].set_ylim([10 ** -3.5, 10 ** 0.2])
    if xlim is not None :
        axs[i].set_xlim(xlim)
    distro_ls0, distro_cs0 = [], []
    for z, (l, col, lw, ddf) in enumerate(zip(distro_ls, distro_cs, lws, ddfs)):
        if ddf is None:
            continue
        if plot_fits == 'best' and z == idx_Kmax:
            cc = color
        elif plot_fits == 'all' or l in plot_fits:
            distro_ls0.append(l)
            distro_cs0.append(col)
            cc = col
        else:
            continue
        axs[i].loglog(xrange, ddf, color=cc, lw=lw, label=l)
    if len(distro_ls0) > 1:
        if legend_outside:
            dataset_legend(distro_ls0, distro_cs0, ax=axs[1], loc='center left', fontsize=25, anchor=(1.0, 0.5))
        else:
            for ax in axs:
                dataset_legend(distro_ls0, distro_cs0, ax=ax, loc='lower left', fontsize=15)
    # dataset_legend(gIDs, colors, ax=axs[1], loc='center left', fontsize=25, anchor=(1.0, 0.5))
    # fig.subplots_adjust(left=0.1, right=0.95, wspace=0.08, hspace=0.3, bottom=0.05)
    for jj in [0]:
        axs[jj].set_ylabel(ylabel)

def plot_bouts(plot_fits='', turns=False, stridechain_duration=False, legend_outside=False, **kwargs):
    if not turns:
        name = f'runsNpauses{plot_fits}'
    else:
        name = f'turn_epochs{plot_fits}'
    P = AutoPlot(name=name, sharey=True, Ncols=2, figsize=(10, 5), **kwargs)
    valid_labs = {}
    for j, d in enumerate(P.datasets):
        id = d.id
        try:
            v = d.pooled_epochs
        except:
            v = d.load_pooled_epochs()

        if v is None :
            v = comp_pooled_epochs(d)
        # print(id,v.keys())

        kws = {
            'marker': 'o',
            'plot_fits': plot_fits,
            'label': id,
            'color': d.color,
            'legend_outside': legend_outside,
            'axs': P.axs,
            'x0': None
        }
        if not turns:
            if v.pause_dur is not None:
                plot_single_bout(fit_dic=v.pause_dur, discr=False, bout='pauses', i=1, **kws)
                valid_labs[id] = kws['color']
            if stridechain_duration and v.run_dur is not None:
                plot_single_bout(fit_dic=v.run_dur, discr=False, bout='runs', i=0, **kws)
                valid_labs[id] = kws['color']
            elif not stridechain_duration and v.run_count is not None:
                plot_single_bout(fit_dic=v.run_count, discr=True, bout='stridechains', xlabel='# strides', i=0, **kws)
                valid_labs[id] = kws['color']
        else:
            if v.turn_dur is not None:
                plot_single_bout(fit_dic=v.turn_dur, discr=False, bout='turn duration', i=0, **kws)
                valid_labs[id] = kws['color']
            if v.turn_amp is not None:
                plot_single_bout(fit_dic=v.turn_amp, discr=False, bout='turn amplitude', xlabel='angle (deg)',
                                 xlim=(10 ** -0.5, 10 ** 3), i=1, **kws)
                valid_labs[id] = kws['color']
    P.axs[1].yaxis.set_visible(False)
    if P.Ndatasets > 1:
        dataset_legend(valid_labs.keys(), valid_labs.values(), ax=P.axs[0], loc='lower left', fontsize=15)
    P.adjust((0.15, 0.95), (0.15, 0.92), 0.05, 0.005)
    return P.get()
