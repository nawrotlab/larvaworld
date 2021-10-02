import copy
import heapq
import itertools
import warnings

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import siunits
from matplotlib import cm, transforms, ticker, patches
from matplotlib.ticker import FixedLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
import statsmodels.api as sm
from scipy import stats, signal, interpolate
from scipy.stats import ttest_ind
from sklearn.linear_model import LinearRegression
from PIL import Image
import os

from lib.anal.combining import combine_images, combine_pdfs
from lib.conf import conf
from lib.aux import naming as nam
from lib.aux import functions as fun
from lib.conf.par import getPar, chunk_dict
from lib.model.DEB.deb import DEB

from lib.stor import paths

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


def mean_confidence_interval(data, confidence=0.95):
    n = len(data)
    m = np.nanmean(data, axis=0)
    se = stats.sem(data, axis=0, nan_policy='omit')
    h = se * stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, m - h, m + h


def plot_mean_and_range(x, mean, lb, ub, axis, color_mean=None, color_shading=None, label=None):
    if x.shape[0] > mean.shape[0]:
        xx = x[:mean.shape[0]]
    elif x.shape[0] == mean.shape[0]:
        xx=x
    # plot the shaded range of e.g. the confidence intervals
    axis.fill_between(xx, ub, lb, color=color_shading, alpha=.2)
    # plot the mean on top
    if label is not None:
        axis.plot(xx, mean, color_mean, label=label, linewidth=2, alpha=1.0)
    else:
        axis.plot(xx, mean, color_mean, linewidth=2, alpha=1.0)

    # pass


def plot_stride_distribution(dataset, agent_id=None, save_to=None):
    d = dataset
    if agent_id is None:
        agent_id = d.agent_ids[0]

    if save_to is None:
        save_to = os.path.join(d.plot_dir, 'plot_stride_distribution')
    if not os.path.exists(save_to):
        os.makedirs(save_to)
    filepath = os.path.join(save_to, f'stride_distribution.{suf}')
    filepath_1 = os.path.join(save_to, f'stride_distribution_spatiotemporal.{suf}')
    filepath_2 = os.path.join(save_to, f'stride_distribution_spatial_hist.{suf}')

    agent_data = d.step_data.xs(agent_id, level='AgentID', drop_level=True)
    # l = d.end['length'].loc[agent_id]

    s = agent_data['scaled_stride_dst'].dropna()
    t = agent_data['stride_dur'].dropna()
    # print(sigma)
    # print(t)
    fig, axs = plt.subplots(1, 1, figsize=([5, 5]))
    axs.plot(s, t, '.')
    axs.set_ylabel(r'duration, $(sec)$', fontsize=15)
    axs.set_xlabel(r'scal displacement', fontsize=15)
    fig.savefig(filepath_1, dpi=300)
    print(f'Image saved as {filepath_1}')

    fig, axs = plt.subplots(1, 1, figsize=([5, 5]))
    axs.hist(s, bins=20)
    axs.axvline(s.median(), color='red', linestyle='dashed', linewidth=1)
    # plt.plot(mean_spatial_stds, mean_temporal_stds)
    axs.set_ylabel(r'# strides', fontsize=15)
    axs.set_xlabel(r'scal displacement', fontsize=15)
    fig.savefig(filepath_2, dpi=300)
    print(f'Image saved as {filepath_2}')

    fig, axs = plt.subplots(1, 2, sharex=True, figsize=([10, 5]))
    axs = axs.ravel()

    axs[0].plot(s, t, '.')
    # plt.plot(mean_spatial_stds, mean_temporal_stds)
    axs[0].set_ylabel(r'duration, $(sec)$', fontsize=15)
    axs[0].set_xlabel(r'scal displacement', fontsize=15)

    axs[1].hist(s, bins=20)
    axs[1].axvline(s.median(), color='red', linestyle='dashed', linewidth=1)
    # plt.plot(mean_spatial_stds, mean_temporal_stds)
    axs[1].set_ylabel(r'# strides', fontsize=15)
    axs[1].set_xlabel(r'scal displacement', fontsize=15)

    plt.tight_layout()

    fig.savefig(filepath, dpi=300)
    print(f'Image saved as {filepath}')


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


def plot_marked_strides(datasets, labels=None, agent_idx=0, agent_id=None, slice=[20, 40],
                        subfolder='individuals', save_as=None, save_to=None, return_fig=False, show=False):
    Ndatasets, colors, save_to, labels = plot_config(datasets, labels, save_to, subfolder)
    if save_as is None:
        temp = f'marked_strides_{slice[0]}-{slice[1]}' if slice is not None else f'marked_strides'
        filename = f'{temp}_{agent_id}.pdf' if agent_id is not None else f'{temp}_{agent_idx}.pdf'
    else:
        filename = save_as

    chunks = ['stride', 'pause']
    chunk_cols = ['lightblue', 'grey']

    p, ylab = getPar('sv', to_return=['d', 'l'])
    ylim = 1.0

    figx = 15 * 6 * 3 if slice is None else int((slice[1] - slice[0]) / 3)
    figy = 5

    fig, axs = plt.subplots(Ndatasets, 1, figsize=(figx, figy * Ndatasets), sharey=True, sharex=True)
    axs = axs.ravel() if Ndatasets > 1 else [axs]
    handles = [patches.Patch(color=col, label=n) for n, col in zip(['stride', 'pause'], chunk_cols)]

    for ii, (d, l) in enumerate(zip(datasets, labels)):
        ax = axs[ii]
        if ii == Ndatasets - 1:
            ax.set_xlabel(r'time $(sec)$')
        ax.set_ylabel(ylab)
        ax.set_ylim([0, ylim])
        ax.set_xlim(slice)
        ax.legend(handles=handles, loc='upper right')

        step_data=d.read('step')
        temp_id = d.agent_ids[agent_idx] if agent_id is None else agent_id
        s = copy.deepcopy(step_data.xs(temp_id, level='AgentID', drop_level=True))
        s.set_index(s.index * d.dt, inplace=True)
        ax.plot(s[p], color='blue')
        for i, (c, col) in enumerate(zip(chunks, chunk_cols)):
            s0s = s.index[s[nam.start(c)] == True]
            s1s = s.index[s[nam.stop(c)] == True]
            for s0, s1 in zip(s0s, s1s):
                # kkk=s['state'].loc[s0:s1].values
                # print(kkk)
                # if c=='pause' :
                #     if all([kkk[i]==1 for i in range(len(kkk))]) :
                #         col2='red'
                #     elif all([kkk[i]==2 for i in range(len(kkk))]) :
                #         col2='green'
                #     elif all([kkk[i]==3 for i in range(len(kkk))]) :
                #         col2='black'
                #     else :
                #         col2=col
                # else :
                #     col2=col
                ax.axvspan(s0, s1, color=col, alpha=1.0)
                ax.axvline(s0, color=f'{0.4 * (i + 1)}', alpha=0.6, linestyle='dashed', linewidth=1)
                ax.axvline(s1, color=f'{0.4 * (i + 1)}', alpha=0.6, linestyle='dashed', linewidth=1)

        ax.plot(s[p].loc[s[nam.max(p)] == True], linestyle='None', lw=10, color='green', marker='v')
        ax.plot(s[p].loc[s[nam.min(p)] == True], linestyle='None', lw=10, color='red', marker='^')

    plt.subplots_adjust(bottom=0.15, top=0.95, left=0.08, right=0.95, hspace=0.1)

    return process_plot(fig, save_to, filename, return_fig, show)


def plot_sample_tracks(datasets, labels=None, mode='strides', agent_idx=0, agent_id=None, slice=[20, 40],
                       subfolder='individuals', save_as=None, save_to=None, return_fig=False, show=False):
    Ndatasets, colors, save_to, labels = plot_config(datasets, labels, save_to, subfolder)
    # for d in datasets:
    #     if not hasattr(d, 'step'):
    #         d.load()
    if save_as is None:
        temp = f'sample_marked_{mode}_{slice[0]}-{slice[1]}'
        filename = f'{temp}_{agent_id}.pdf' if agent_id is not None else f'{temp}_{agent_idx}.pdf'
    else:
        filename = save_as

    if mode == 'strides':
        chunks = ['stride', 'pause']
        chunk_cols = ['lightblue', 'grey']

        p, ylab, ylim = getPar('sv', to_return=['d', 'l', 'lim'])
        ylim = 1.0
    elif mode == 'turns':
        chunks = ['Rturn', 'Lturn']
        chunk_cols = ['lightgreen', 'orange']

        b = 'bend'
        bv = nam.vel(b)
        ho = nam.orient('front')
        hov = nam.vel(ho)

    figx = 15 * 6 * 3 if slice is None else int((slice[1] - slice[0]) / 3)
    figy = 5

    fig, axs = plt.subplots(Ndatasets, 1, figsize=(figx, figy * Ndatasets), sharey=True, sharex=True)
    axs = axs.ravel() if Ndatasets > 1 else [axs]
    handles = [patches.Patch(color=col, label=n) for n, col in zip(chunks, chunk_cols)]

    for ii, (d, l) in enumerate(zip(datasets, labels)):
        ax = axs[ii]
        if ii == Ndatasets - 1:
            ax.set_xlabel(r'time $(sec)$')
        ax.set_ylabel(ylab)
        ax.set_ylim([0, ylim])
        ax.set_xlim(slice)
        ax.legend(handles=handles, loc='upper right')

        step_data = d.read('step')
        temp_id = d.agent_ids[agent_idx] if agent_id is None else agent_id
        s = copy.deepcopy(step_data.xs(temp_id, level='AgentID', drop_level=True))
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

    plt.subplots_adjust(bottom=0.15, top=0.95, left=0.08, right=0.95, hspace=0.1)
    return process_plot(fig, save_to, filename, return_fig, show)


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
                temp=None
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
            filename=f'{save_to}/{filepath}'
            fig.savefig(filename, dpi=300)
            print(f'Image saved as {filename}')
            fig_dict[f'turns_{agent_id}_{i}'] = fig
    # return process_plot(fig, save_to, filename, return_fig, show)
    return fig_dict


def plot_pauses(dataset, Npauses=10, save_to=None, plot_simulated=False, return_fig=False, show=False):
    if save_to is None:
        save_to = dataset.plot_dir
    dt = dataset.dt
    filename1 = f'pauses.{suf}'
    filepath1 = os.path.join(save_to, filename1)
    filename2 = f'pause_max_hist.{suf}'
    filepath2 = os.path.join(save_to, filename2)
    if not plot_simulated:
        exp_bends, exp_bendvels = dataset.load_pause_dataset(load_simulated=False)
        sim = False
    else:
        exp_bends, exp_bendvels, sim_bends, sim_bendvels, sim_acts = dataset.load_pause_dataset(load_simulated=True)
        sim = True

    bend_r, bendvel_r = 180, 600
    lengths = (~np.isnan(exp_bends)).sum(1)
    sel_ind = lengths.argsort()[-Npauses:][::-1]
    sel_exp_bends = [exp_bends[i] for i in sel_ind]
    sel_exp_bendvels = [exp_bendvels[i] for i in sel_ind]
    max_dur = np.max(lengths)
    x = np.arange(0, max_dur, int(0.5 / dt))
    if sim:
        sel_sim_bends = [sim_bends[i] for i in sel_ind]
        sel_sim_bendvels = [sim_bendvels[i] for i in sel_ind]
        sel_sim_acts = [sim_acts[i] for i in sel_ind]

    fig, axs = plt.subplots(Npauses, 1, figsize=(15, Npauses * 3), sharex=True, sharey=True)
    axs = axs.ravel()
    axs_l, axs_r = [], []
    for i in range(Npauses):
        ax_l = axs[i]
        ax_r = ax_l.twinx()
        axs_l.append(ax_l)
        axs_r.append(ax_r)

    # share the secondary axes
    for ax in axs_r[1:]:
        axs_r[0].get_shared_y_axes().join(axs_r[0], ax)

    for i in range(Npauses):
        eb = sel_exp_bends[i]
        ebv = sel_exp_bendvels[i]
        axs_l[i].plot(eb, 'r', label='observed bending angle')
        axs_r[i].plot(ebv, 'b', label='observed bending velocity')

        if sim:
            sb = sel_sim_bends[i]
            sbv = sel_sim_bendvels[i]
            sa = sel_sim_acts[i]
            axs_l[i].plot(sb, 'r', linestyle='dashed', label='simulated bending angle')
            axs_r[i].plot(sb, 'b', linestyle='dashed', label='simulated bending velocity')
        axs_l[i].axhline(0, color='green', alpha=0.5, linestyle='dashed', linewidth=1)
        axs_l[i].set_ylabel('angle (deg)')
        axs_l[i].set_ylim([-bend_r, bend_r])
        axs_r[i].set_ylabel('angular velocity (deg/sec)')
        axs_r[i].set_ylim([-bendvel_r, bendvel_r])
        axs_l[i].legend(loc='upper left')
        axs_r[i].legend(loc='upper right')
    axs_l[-1].set_xticks(x)
    axs_l[-1].set_xticklabels([i * dt for i in x])
    axs_l[-1].set_xlim([0, max_dur])
    axs_l[-1].set_xlabel('time(sec)', fontsize=15)
    plt.subplots_adjust(hspace=0.05, top=0.96, bottom=0.02, left=0.07, right=0.93)
    fig.suptitle('Individual pauses', fontsize=25)
    save_plot(fig, filepath1, filename1)

    # Plot max values histogram
    max_bends, max_bendvels = [np.round(np.nanmax(np.abs(p), axis=0)) for p in [exp_bends, exp_bendvels]]
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    axs = axs.ravel()
    Nbins = 50
    axs[0].hist(max_bends, color='g', bins=Nbins, label='maximum bend')
    axs[0].set_xlabel('angle (deg)')
    axs[0].set_ylabel('counts')
    axs[1].hist(max_bendvels, color='r', bins=Nbins, label='maximum bending velocity')
    axs[1].set_xlabel('angular velocity (deg/sigma)')
    axs[1].set_ylabel('counts')
    fig.suptitle('Maximum values during pauses', fontsize=20)
    save_plot(fig, filepath2, filename2)


def plot_debs(deb_dicts=None, save_to=None, save_as=None, mode='full', roversVSsitters=False,include_egg=True,
              time_unit='hours', return_fig=False, sim_only=False, force_ymin=None, color_epoch_quality=True,
              datasets=None, labels=None, show=False, label_epochs=True, label_lifestages=True,**kwargs):
    warnings.filterwarnings('ignore')
    if save_to is None:
        save_to = paths.DebFolder
    os.makedirs(save_to, exist_ok=True)
    if save_as is None:
        save_as = f'debs.{suf}'
    if deb_dicts is None:
        deb_dicts = []
        for d, l in zip(datasets, labels):
            dataset_deb_dicts = d.load_deb_dicts()
            deb_dicts.append(dataset_deb_dicts)
        deb_dicts = fun.flatten_list(deb_dicts)
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
               'M_gut', 'M_ingested', 'M_absorbed', 'M_faeces', 'M_not_digested', 'M_not_absorbed',
               'R_faeces', 'R_absorbed', 'R_not_digested', 'gut_occupancy',
               'deb_p_A', 'sim_p_A', 'gut_p_A', 'gut_f', 'gut_p_A_deviation'
               ]
    ylabels0 = ['wet weight $(mg)$', 'body length $(mm)$',
                r'reserve $(J)$', r'reserve density $(-)$', r'hunger drive $(-)$',
                r'pupation buffer $(-)$',
                r'f $^{sim}$ $(-)$', r'f $_{filt}^{sim}$ $(-)$',
                r'exploit VS explore $(-)$',
                'gut content $(mg)$', 'food intake $(mg)$', 'food absorption $(mg)$',
                'faeces $(mg)$', 'food not digested $(mg)$', 'product not absorbed $(mg)$',
                'faeces fraction', 'absorption efficiency', 'fraction not digested', 'gut occupancy',
                r'[p$_{A}^{deb}$] $(\mu J/cm^3)$', r'[p$_{A}^{sim}$] $(\mu J/cm^3)$',
                r'[p$_{A}^{gut}$] $(\mu J/cm^3)$', r'f $^{gut}$ $(-)$', r'$\Delta$p$_{A}^{gut}$ $(-)$'
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
        idx = [9, 10, 11, 12, 13, 14]
    elif mode == 'food_ratio':
        idx = [17, 15, 16, 18]
    elif mode == 'food_mass_1':
        idx = [9, 10, 11]
    elif mode == 'food_mass_2':
        idx = [12, 13, 14]
    elif mode == 'food_ratio_1':
        idx = [18, 16]
    elif mode == 'food_ratio_2':
        idx = [17, 15]
    elif mode == 'assimilation':
        idx = [19, 20, 21]
        sharey = True
    elif mode == 'fs':
        idx = [6, 7, 22]
        sharey = True

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
    figsize = (13, 4 * Npars)
    fig, axs = plt.subplots(Npars, figsize=figsize, sharex=True, sharey=sharey)
    axs = axs.ravel() if Npars > 1 else [axs]

    rr0, gg0, bb0 = q_col1 = np.array([255, 0, 0]) / 255
    rr1, gg1, bb1 = q_col2 = np.array([0, 255, 0]) / 255
    quality_col_range = np.array([rr1 - rr0, gg1 - gg0, bb1 - bb0])

    t0s, t1s, t2s, t3s, max_ages = [], [], [], [], []
    for d, id, c in zip(deb_dicts, ids, cols):
        t0_sim, t0, t1, t2, t3, age = d['sim_start'],d['birth'], d['pupation'], d['death'], d['hours_as_larva'] + d['birth'], np.array(d['age'])
        t00=0
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
        elif not include_egg :
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
            # print(l, yl, len(age))
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
                q_col = q_col1 + qq * quality_col_range if color_epoch_quality else c
                ax.axvspan(st0, st1, color=q_col, alpha=0.2)

            ax.set_ylabel(yl, labelpad=15, fontsize=15)
            ax.yaxis.set_major_locator(ticker.MaxNLocator(3))
            ax.tick_params(axis='y', labelsize=15)
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
            if label_lifestages:
                try:
                    y0, y1 = ax.get_ylim()
                    ytext = y0 + 0.5 * (y1 - y0)
                    xtext = t00 + 0.5 * (t0 - t00)
                    ax.annotate('$incubation$',rotation=90,fontsize=25,va='center',ha='center',
                                xy=(xtext, ytext), xycoords='data',
                                )
                except:
                    pass
                try:
                    y0, y1 = ax.get_ylim()
                    x0, x1 = ax.get_xlim()
                    ytext = y0 + 0.5 * (y1 - y0)
                    xtext = t3 + 0.5 * (x1 - t3)
                    ax.axvspan(t3, x1, color='darkgrey', alpha=0.5)
                    ax.annotate('$pupation$', rotation=90, fontsize=25, va='center', ha='center',
                                xy=(xtext, ytext), xycoords='data',
                                )
                except:
                    pass
            if label_epochs:

                try:
                    y0, y1 = ax.get_ylim()
                    ytext = y0 + 0.8 * (y1 - y0)
                    xpre = t0 + 0.5 * (t0_sim-t0)
                    if t0_sim-t0>0.2*(np.max(age)-t00) :
                        ax.annotate('$prediction$',rotation=0,fontsize=20,va='center',ha='center',
                                    xy=(xpre, ytext), xycoords='data',
                                    )
                    xsim = t0_sim + 0.5 * (np.max(age) - t0_sim)
                    if np.max(age) - t0_sim > 0.2 * (np.max(age) - t00):
                        ax.annotate('$simulation$', rotation=0, fontsize=20, va='center', ha='center',
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
    fig.subplots_adjust(top=0.95, bottom=0.25, left=0.15, right=0.93, hspace=0.02)
    return process_plot(fig, save_to, save_as, return_fig, show)


def plot_surface(x, y, z, labels, z0=None, title=None, save_to=None, save_as=None, pref=None, show=False):
    fig = plt.figure(figsize=(20, 10))
    if title is not None:
        fig.suptitle(title)
    # ax = fig.gca(projection='3d')
    # ax = fig.add_subplot(111, projection='3d')
    ax = Axes3D(fig)
    ax.plot_surface(x, y, z,
                    cmap=cm.coolwarm,
                    linewidth=0,
                    antialiased=True)
    if z0 is not None:
        ax.plot_surface(x, y, np.ones(x.shape) * z0, alpha=0.5)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])

    ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.zaxis.set_major_locator(ticker.MaxNLocator(5))
    if show:
        plt.show()
    if save_to is not None:
        os.makedirs(save_to, exist_ok=True)
        if save_as is None:
            save_as = f'surface.{suf}'
            if pref is not None:
                save_as = f'{pref}_{save_as}'
        filepath = os.path.join(save_to, save_as)
        fig.savefig(filepath, dpi=300)
        print(f'Surface saved as {save_as}')
    plt.close('all')
    return fig


def plot_heatmap(x, y, z, labels, title=None, save_to=None, save_as=None, pref=None, show=False):
    # fig = plt.figure(figsize=(10, 5))
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(z, annot=True, fmt="g", cmap=cm.coolwarm,
                xticklabels=x.tolist(), yticklabels=y.tolist(), ax=ax,
                cbar_kws={"orientation": "vertical",
                          'label': labels[2],
                          # 'ticks': [1, 0, -1]
                          })
    ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
    ax.xaxis.set_ticks_position('top')
    ax.set_ylabel(labels[1])
    ax.set_xlabel(labels[0])
    if title is not None:
        ax.set_suptitle(title, fontsize=20)
    if show:
        plt.show()
    if save_to is not None:
        os.makedirs(save_to, exist_ok=True)
        if save_as is None:
            save_as = f'heatmap.{suf}'
            if pref is not None:
                save_as = f'{pref}_{save_as}'
        filepath = os.path.join(save_to, save_as)
        plt.savefig(filepath, dpi=300)
        print(f'Heatmap saved as {filepath}')
    plt.close('all')
    return fig


def plot_3pars(df, labels, save_to, z0=None, pref=None, show=False):
    fig_dict = {}
    pr = f'{labels[0]}VS{labels[1]}'
    fig1 = plot_3d(df, labels, save_to, pref=pref, save_as=None, show=show)
    fig_dict[f'{pr}_3d'] = fig1
    try:
        x, y = np.unique(df[labels[0]].values), np.unique(df[labels[1]].values)
        X, Y = np.meshgrid(x, y)

        z = df[labels[2]].values.reshape(X.shape).T

        fig2 = plot_heatmap(x, y, z, labels, save_to=save_to, pref=pref, show=show)
        fig3 = plot_surface(X, Y, z, labels, save_to=save_to, z0=z0, pref=pref, show=show)
        fig_dict[f'{pr}_heatmap'] = fig2
        fig_dict[f'{pr}_surface'] = fig3
    except:
        pass
    return fig_dict


def plot_3d(df, labels, save_to, pref=None, save_as=None, show=False):
    l0, l1, l2 = labels
    X = df[[l0, l1]]
    y = df[l2]

    X = sm.add_constant(X)
    est = sm.OLS(y, X).fit()

    xx1, xx2 = np.meshgrid(np.linspace(X[l0].min(), X[l0].max(), 100),
                           np.linspace(X[l1].min(), X[l1].max(), 100))

    # plot the hyperplane by evaluating the parameters on the grid
    Z = est.params[0] + est.params[1] * xx1 + est.params[2] * xx2

    # create matplotlib 3d axes
    fig = plt.figure(figsize=(12, 8))
    ax = Axes3D(fig, azim=115, elev=15)

    # plot hyperplane
    surf = ax.plot_surface(xx1, xx2, Z, cmap=plt.cm.RdBu_r, alpha=0.6, linewidth=0)

    # plot data points - points over the HP are white, points below are black
    resid = y - est.predict(X)
    ax.scatter(X[resid >= 0][l0], X[resid >= 0][l1], y[resid >= 0], color='black', alpha=0.4, facecolor='white')
    ax.scatter(X[resid < 0][l0], X[resid < 0][l1], y[resid < 0], color='black', alpha=0.4)

    ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
    ax.zaxis.set_major_locator(ticker.MaxNLocator(4))

    ax.set_xlabel(l0, labelpad=20)
    ax.set_ylabel(l1, labelpad=20)
    ax.set_zlabel(l2, labelpad=20)
    if show:
        plt.show()
    if save_to is not None:
        os.makedirs(save_to, exist_ok=True)
        if save_as is None:
            save_as = f'3d_plot.{suf}'
            if pref is not None:
                save_as = f'{pref}_{save_as}'
        filepath = os.path.join(save_to, save_as)
        plt.savefig(filepath, dpi=300)
        print(f'3D plot saved as {filepath}')
    plt.close('all')
    return fig


def plot_2d(df, labels, save_to, pref=None, save_as=None, show=False):
    par = labels[0]
    res = labels[0]
    p = df[par].values
    r = df[res].values
    fig, axs = plt.subplots(1, 1, figsize=(10, 10))
    axs.scatter(p, r)
    axs.set_xlabel(par)
    axs.set_ylabel(res)
    if show:
        plt.show()
    if save_to is not None:
        os.makedirs(save_to, exist_ok=True)
        if save_as is None:
            save_as = f'2d_plot.{suf}'
            if pref is not None:
                save_as = f'{pref}_{save_as}'
        filepath = os.path.join(save_to, save_as)
        plt.savefig(filepath, dpi=300)
        print(f'3D plot saved as {filepath}')
    plt.close('all')
    return fig


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
            tseries = s[list(c)].dropna().sum(axis=1)
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
    axs[1].bar(x=[','.join(map(str, c)) for c in best_combos_ind], height=max_corrs, width=0.8, color='black')
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


def plot_2D_countour(x, y, z, dimensions, Cmax, filepath):
    xmin, xmax = dimensions[0]
    ymin, ymax = dimensions[1]
    # define grid.
    xi = np.linspace(xmin, xmax, 1000)
    yi = np.linspace(ymin, ymax, 1000)
    ## grid the data.
    zi = interpolate.griddata((x, y), z, (xi[None, :], yi[:, None]), method='cubic')
    levels = np.linspace(0.0, Cmax, 10000)
    fig = plt.figure(figsize=(xmax - xmin, ymax - ymin))
    # CS = plt.contour(xi, yi, zi, len(levels), linewidths=0.0, colors='k', levels=levels)
    CS = plt.contourf(xi, yi, zi, len(levels), cmap=cm.Purples, levels=levels, alpha=0.9)
    cbaxes = fig.add_axes([0.68, 0.93, 2.0, 0.2])
    cbar = fig.colorbar(CS, cax=cbaxes, orientation="horizontal", ticks=[0, Cmax])
    cbar.ax.set_xticklabels([0, f'${int(Cmax)} \mu$M'])

    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.locator_params(nbins=4)
    fig.savefig(filepath, dpi=300)
    print(f'Image saved as {filepath}')


def gauss(x, y, Sigma, mu):
    X = np.vstack((x, y)).T
    mat_multi = np.dot((X - mu[None, ...]).dot(np.linalg.inv(Sigma)), (X - mu[None, ...]).T)
    return np.diag(np.exp(-1 * (mat_multi)))


def plot_2D_odorscape(dimensions, Cmax, Cstd, filepath, pos=None):
    if pos is None:
        pos = [0., 0.]
    npts = 10000
    x = np.random.uniform(dimensions[0][0], dimensions[0][1], npts)
    y = np.random.uniform(dimensions[1][0], dimensions[1][1], npts)
    z = gauss(x, y, Sigma=np.asarray([[Cstd, 0.0], [0.0, Cstd]]), mu=np.asarray(pos)) * Cmax
    plot_2D_countour(x, y, z, dimensions=dimensions, Cmax=Cmax, filepath=filepath)


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


def plot_stride_Dbend(datasets, labels=None, show_text=False, subfolder='stride', save_to=None, return_fig=False,
                      show=False):
    Ndatasets, colors, save_to, labels = plot_config(datasets, labels, save_to, subfolder)
    filename = f'stride_bend_change.{suf}'

    p = 'bend'
    b0_par = f'{p}_at_stride_start'
    b1_par = f'{p}_at_stride_stop'

    b0s = []
    b1s = []
    dbs = []
    for d in datasets:
        b0 = d.get_par(b0_par).dropna().values.flatten()
        b1 = d.get_par(b1_par).dropna().values.flatten()
        sign_b = np.sign(b0)
        b0 *= sign_b
        b1 *= sign_b
        db = b1 - b0
        b0s.append(b0)
        b1s.append(b1)
        dbs.append(db)

    figsize = (6, 5)
    ylim = [-60, 60]
    xlim = [0, 85]

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
    fits = {}
    for i, (b0, b1, db, label, c) in enumerate(zip(b0s, b1s, dbs, labels, colors)):
        b0 = b0[:500]
        db = db[:500]
        ax.scatter(x=b0, y=db, marker='o', s=2.0, alpha=0.6, color=c, label=label)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel(r'$\theta_{bend}$ at stride start $(deg)$')
        ax.set_ylabel(r'$\Delta\theta_{bend}$ over stride $(deg)$')

        m, k = np.polyfit(b0, db, 1)
        m = np.round(m, 2)
        k = np.round(k, 2)
        fits[label] = [m, k]
        ax.plot(b0, m * b0 + k, linewidth=4, color=c)
        if show_text:
            ax.text(0.3, 0.9 - i * 0.1, rf'${label} : \Delta\theta_{{b}}={m} \cdot \theta_{{b}}$', fontsize=12,
                    transform=ax.transAxes)
            print(f'Bend correction during strides for {label} fitted as : db={m}*b + {k}')
    plt.subplots_adjust(bottom=0.2, top=0.95, left=0.25, right=0.95, wspace=0.01)
    return process_plot(fig, save_to, filename, return_fig, show)


def plot_EEB_vs_food_quality(samples=None, dt=None,species_list=['rover', 'sitter', 'default'],
                             save_to=None, return_fig=False,show=False, **kwargs):
    if samples is None :
        raise ('No sample configurations provided')
    from lib.model.modules.intermitter import get_EEB_poly1d
    filename = f'EEB_vs_food_quality.{suf}'
    qs = np.arange(0.01, 1, 0.01)
    # qs=[1.0,0.75,0.5,0.25,0.15]

    fig, axs = plt.subplots(3, len(samples), figsize=(10 * len(samples), 20))
    axs = axs.ravel()
    cols = fun.N_colors(len(species_list))

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


def plot_stride_Dorient(datasets, labels=None, simVSexp=False, absolute=True, subfolder='stride',
                        save_to=None, legend=False, return_fig=False, show=False):
    Ndatasets, colors, save_to, labels = plot_config(datasets, labels, save_to, subfolder)
    filename = f'stride_orient_change.{suf}'

    par_shorts = ['str_fo', 'str_ro']
    pars, sim_labels, exp_labels, xlabels = getPar(par_shorts, to_return=['d', 's', 's', 'l'])

    ranges = [80, 80]

    p_labels = [[sl, el] for sl, el in zip(sim_labels, exp_labels)] if simVSexp else [[sl] * Ndatasets for sl in
                                                                                      sim_labels]

    fig, axs = plt.subplots(1, len(pars), figsize=(10, 5), sharey=True)
    axs = axs.ravel() if len(pars) > 1 else [axs]

    nbins = 200

    for i, (p, r, p_lab, xlab) in enumerate(zip(pars, ranges, p_labels, xlabels)):
        for j, d in enumerate(datasets):
            v = d.get_par(p).dropna().values
            if absolute:
                v = np.abs(v)
                r1, r2 = 0, r
            else:
                r1, r2 = -r, r
            x = np.linspace(r1, r2, nbins)
            weights = np.ones_like(v) / len(v)
            axs[i].hist(v, color=colors[j], bins=x, label=p_lab[j], weights=weights, alpha=0.5)
        axs[i].set_xlabel(xlab)
        axs[i].yaxis.set_major_locator(ticker.MaxNLocator(4))
        if legend:
            axs[i].legend()
    axs[0].set_ylabel('probability')
    plt.subplots_adjust(bottom=0.2, top=0.95, left=0.12, right=0.99, wspace=0.01)
    return process_plot(fig, save_to, filename, return_fig, show)


def plot_interference(datasets, labels=None, mode='orientation', agent_idx=None, subfolder='interference',
                      save_to=None, save_as=None, return_fig=False, show=False):
    Ndatasets, colors, save_to, labels = plot_config(datasets, labels, save_to, subfolder)
    if save_as is None:
        save_as = f'interference_{mode}.{suf}' if agent_idx is None else f'interference_{mode}_agent_idx_{agent_idx}.{suf}'

    par_shorts = ['sv']
    if mode == 'orientation':
        par_shorts.append('fov')
    elif mode == 'orientation_x2':
        par_shorts.append('fov')
        par_shorts.append('rov')
    elif mode == 'bend':
        par_shorts.append('bv')
    elif mode == 'spinelength':
        par_shorts.append('l')

    pars, sim_labels, exp_labels, units = getPar(par_shorts, to_return=['d', 's', 's', 'l'])
    fig, axs = plt.subplots(len(pars), 1, figsize=(10, len(pars) * 5), sharex=True)
    axs = axs.ravel()

    ang_ylim = [0, 60] if mode in ['bend', 'orientation', 'orientation_x2'] else None

    if agent_idx is not None:
        data = [[d.load_aux(type='stride', pars=p).loc[d.agent_ids[agent_idx]].values for p in pars] for
                d in datasets]
    else:
        data = [[d.load_aux(type='stride', pars=p).values for p in pars] for d in datasets]
    Npoints = data[0][0].shape[1]
    for d0, c, color, label in zip(data, colors, colors, labels):
        if mode in ['bend', 'orientation']:
            d0 = [np.abs(d) for d in d0]
        for i, (p, u, pd) in enumerate(zip(pars, units, d0)):
            ts_m, ts_l, ts_h = [np.nanquantile(pd, q=q, axis=0) for q in [0.5, 0.25, 0.75]]
            plot_mean_and_range(x=np.arange(len(ts_m)), mean=ts_m, lb=ts_l, ub=ts_h, axis=axs[i],
                                color_mean=c, color_shading=color, label=label)
            axs[i].yaxis.set_major_locator(ticker.MaxNLocator(4))
            axs[i].legend(loc='upper right')
            axs[i].set_ylabel(u)
            ylim = ang_ylim if i != 0 else [0.0, 0.6]
            axs[i].set_ylim(ylim)

    Nticks = 5
    ticks = np.linspace(0, Npoints - 1, Nticks)
    axs[-1].set_xticks(ticks=ticks)
    axs[-1].set_xticklabels(labels=[r'$0$', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$'])
    axs[-1].set_xlim([0, Npoints - 1])
    axs[-1].set_xlabel('$\phi_{stride}$')

    fig.subplots_adjust(top=0.97, bottom=0.2 / len(pars), left=0.12, right=0.95, hspace=.1, wspace=0.05)
    return process_plot(fig, save_to, save_as, return_fig, show)


def plot_dispersion(datasets, labels=None, ranges=None, scaled=False, subfolder='dispersion', fig_cols=1, ymax=None,
                    save_as=None, save_to=None, return_fig=False, show=False):
    Ndatasets, colors, save_to, labels = plot_config(datasets, labels, save_to, subfolder)
    if ranges is None:
        ranges = [(0, 40)]
    for r0, r1 in ranges:
        if r0 == 0 and r1 == 40:
            par = f'dispersion'
        else:
            par = f'dispersion_{r0}_{r1}'

        if scaled:
            filename = f'scaled_dispersion_{r0}-{r1}_{fig_cols}.{suf}'
            ylab = 'scaled dispersion'
        else:
            filename = f'dispersion_{r0}-{r1}_{fig_cols}.{suf}'
            ylab = r'dispersion $(mm)$'
        if save_as is not None:
            filename = save_as
        t0, t1 = int(r0 * datasets[0].fr), int(r1 * datasets[0].fr)
        Nticks = t1 - t0
        trange = np.linspace(r0, r1, Nticks)
        fig, axs = plt.subplots(1, 1, figsize=(5 * fig_cols, 5))

        for d, lab, c in zip(datasets, labels, colors):
            dsp = d.load_aux(type='dispersion', pars=par if not scaled else nam.scal(par))
            plot_mean_and_range(x=trange,
                                mean=dsp['median'].values[t0:t1],
                                lb=dsp['upper'].values[t0:t1],
                                ub=dsp['lower'].values[t0:t1],
                                axis=axs, color_mean=c, color_shading=c, label=lab)
        if ymax is not None:
            axs.set_ylim(ymax=ymax)
        axs.set_ylabel(ylab)
        axs.set_xlabel('time, $sec$')
        axs.set_xlim([trange[0], trange[-1]])
        axs.yaxis.set_major_locator(ticker.MaxNLocator(4))
        axs.xaxis.set_major_locator(ticker.MaxNLocator(4))
        axs.legend(loc='upper left')
        fig.subplots_adjust(top=0.95, bottom=0.15, left=0.2 / fig_cols, right=0.95, hspace=.005, wspace=0.05)
        return process_plot(fig, save_to, filename, return_fig, show)


def plot_pathlength(datasets, labels=None, scaled=True, save_to=None, save_as=None, return_fig=False, unit='mm',
                    xlabel=None, show=False, **kwargs):
    Ndatasets, colors, save_to, labels = plot_config(datasets, labels, save_to)
    Nticks = len(datasets[0].step_data.index.unique('Step'))
    t0, t1 = 0, int(Nticks / datasets[0].fr / 60)

    lab = 'pathlength'
    if scaled:
        filename = f'scaled_{lab}.{suf}'
        ylab = f'scaled {lab} $(-)$'
    else:
        filename = f'{lab}.{suf}'
        ylab = f'{lab} $({unit})$'

    if save_as is not None:
        filename = save_as
    # colors=['blue', 'darkred', 'purple', 'red']
    trange = np.linspace(t0, t1, Nticks)
    fig, axs = plt.subplots(1, 1, figsize=(7, 6))
    dst_par, dst_SI = getPar('cum_d', to_return=['d', 'u'])
    for d, lab, c in zip(datasets, labels, colors):
        dst_df = d.step_data[dst_par]
        if not scaled and unit == 'cm':
            if dst_SI.unit == siunits.m:
                dst_df *= 100
        dst_m = dst_df.groupby(level='Step').quantile(q=0.5)
        dst_u = dst_df.groupby(level='Step').quantile(q=0.75)
        dst_b = dst_df.groupby(level='Step').quantile(q=0.25)
        plot_mean_and_range(x=trange, mean=dst_m, lb=dst_b, ub=dst_u, axis=axs, color_mean=c,
                            color_shading=c, label=lab)
    axs.set_ylabel(ylab)
    if xlabel is None:
        xlabel = 'time, $min$'
    axs.set_xlabel(xlabel)
    axs.set_xlim([trange[0], trange[-1]])
    axs.set_ylim(ymin=0)
    axs.xaxis.set_major_locator(ticker.MaxNLocator(5))
    axs.legend(loc='upper left')
    fig.subplots_adjust(top=0.95, bottom=0.15, left=0.2, right=0.95, hspace=.005, wspace=0.05)
    return process_plot(fig, save_to, filename, return_fig, show)


def plot_gut(datasets, labels=None, save_to=None, save_as=None, return_fig=False, show=False):
    Ndatasets, colors, save_to, labels = plot_config(datasets, labels, save_to)
    Nticks = len(datasets[0].step_data.index.unique('Step'))
    t0, t1 = 0, int(Nticks / datasets[0].fr / 60)

    lab = 'gut'

    filename = f'{lab}.{suf}'
    ylab = f'% {lab} occupied'

    if save_as is not None:
        filename = save_as

    trange = np.linspace(t0, t1, Nticks)
    fig, axs = plt.subplots(1, 1, figsize=(10, 5))
    for d, lab, c in zip(datasets, labels, colors):
        dst_df = d.step_data['gut_occupancy'] * 100
        dst_m = dst_df.groupby(level='Step').quantile(q=0.5)
        dst_u = dst_df.groupby(level='Step').quantile(q=0.75)
        dst_b = dst_df.groupby(level='Step').quantile(q=0.25)
        plot_mean_and_range(x=trange, mean=dst_m, lb=dst_b, ub=dst_u, axis=axs, color_mean=c,
                            color_shading=c, label=lab)
    axs.set_ylabel(ylab)
    axs.set_xlabel('time, $min$')
    axs.set_xlim([trange[0], trange[-1]])
    axs.set_ylim([0, 100])
    axs.xaxis.set_major_locator(ticker.MaxNLocator(5))
    axs.yaxis.set_major_locator(ticker.MaxNLocator(5))
    axs.legend(loc='upper left', fontsize=9)
    fig.subplots_adjust(top=0.95, bottom=0.15, left=0.1, right=0.95, hspace=.005, wspace=0.05)
    return process_plot(fig, save_to, filename, return_fig, show)


def plot_food_amount(datasets, labels=None, save_to=None, save_as=None, filt_amount=False, scaled=False,
                     return_fig=False, show=False, **kwargs):
    Ndatasets, colors, save_to, labels = plot_config(datasets, labels, save_to)
    Nticks = len(datasets[0].step_data.index.unique('Step'))
    t0, t1 = 0, int(Nticks / datasets[0].fr / 60)

    filename = f'food_intake.{suf}'
    ylab = r'Cumulative food intake $(mg)$'
    par = 'amount_eaten'
    if scaled:
        filename = f'scaled_{filename}'
        ylab = r'Cumulative food intake as % larval mass'
        par = 'ingested_body_mass_ratio'
    if filt_amount:
        filename = f'filtered_{filename}'
        ylab = r'Food intake $(mg)$'
    if filt_amount and scaled:
        ylab = 'Food intake as % larval mass'

    if save_as is not None:
        filename = save_as

    trange = np.linspace(t0, t1, Nticks)
    fig, axs = plt.subplots(1, 1, figsize=(10, 5))
    for d, lab, c in zip(datasets, labels, colors):
        dst_df = d.step_data[par]

        dst_m = dst_df.groupby(level='Step').quantile(q=0.5)
        dst_u = dst_df.groupby(level='Step').quantile(q=0.75)
        dst_b = dst_df.groupby(level='Step').quantile(q=0.25)
        # print(dst_m)
        # print(mode(dst_m))
        if filt_amount:
            sos = signal.butter(N=1, Wn=0.1, btype='lowpass', analog=False, fs=Nticks / (t1 - t0), output='sos')
            dst_m = dst_m.diff()
            dst_m.iloc[0] = 0
            dst_m = signal.sosfiltfilt(sos, dst_m)
            dst_u = dst_u.diff()
            dst_u.iloc[0] = 0
            dst_u = signal.sosfiltfilt(sos, dst_u)
            dst_b = dst_b.diff()
            dst_b.iloc[0] = 0
            dst_b = signal.sosfiltfilt(sos, dst_b)
        plot_mean_and_range(x=trange, mean=dst_m, lb=dst_b, ub=dst_u, axis=axs, color_mean=c,
                            color_shading=c, label=lab)
    axs.set_ylabel(ylab)
    axs.set_xlabel('time, $min$')
    axs.set_xlim([trange[0], trange[-1]])
    axs.xaxis.set_major_locator(ticker.MaxNLocator(5))
    axs.legend(loc='upper left', fontsize=9)
    fig.subplots_adjust(top=0.95, bottom=0.15, left=0.1, right=0.95, hspace=.005, wspace=0.05)
    # plt.show()
    # raise
    return process_plot(fig, save_to, filename, return_fig, show)


def boxplot_PI(datasets, labels=None, subfolder=None, save_as=None,sort_labels=False,
               save_to=None, return_fig=False, show=False, xlabel='Trials') :

    if save_to is None:
        save_to = paths.OdorPrefFigFolder
    if subfolder is not None:
        save_to = f'{save_to}/{subfolder}'
    if not os.path.exists(save_to):
        os.makedirs(save_to)
    filename = f'PI_boxplot.{suf}' if save_as is None else save_as

    group_ids=fun.unique_list([d.config['group_id'] for d in datasets])
    Ngroups=len(group_ids)
    common_ids = fun.unique_list([l.split('_')[-1] for l in group_ids])

    Ncommon = len(common_ids)
    pair_ids = fun.unique_list([l.split('_')[0] for l in group_ids])

    Npairs = len(pair_ids)
    coupled_labels=True if Ngroups==Npairs*Ncommon else False

    if Npairs==3 and all([l in pair_ids for l in ['Low','Medium', 'High']]) :
        pair_ids = ['Low','Medium', 'High']
        xlabel= 'Substate fructose concentration'
    elif Npairs==3 and all([l in pair_ids for l in ['1:20','1:200', '1:2000']]) :
        pair_ids = ['1:20','1:200', '1:2000']
        xlabel= 'Odor concentration'
    if Ncommon==2 and all([l in common_ids for l in ['AM','EM']]) :
        common_ids = ['EM','AM']

    if sort_labels :
        common_ids = sorted(common_ids)
        pair_ids = sorted(pair_ids)


    all_PIs = []
    all_PIs_dict = {}
    for group_id in group_ids :
        group_ds=[d for d in datasets if d.config['group_id']==group_id]
        PIdicts = [d.config['PI'] for d in group_ds]
        PIs = [dic['PI'] for dic in PIdicts]
        all_PIs.append(PIs)
        all_PIs_dict[group_id]=PIs

    if coupled_labels :
        colors = fun.N_colors(Ncommon)
        palette = {id: c for id, c in zip(common_ids, colors)}
        pair_dfs=[]
        for pair_id in pair_ids :
            paired_group_ids=[f'{pair_id}_{common_id}' for common_id in common_ids]
            pair_PIs=[all_PIs_dict[id] for id in paired_group_ids]
            pair_PI_array = fun.boolean_indexing(pair_PIs).T
            pair_df = pd.DataFrame(pair_PI_array, columns=common_ids).assign(Trial=pair_id)
            pair_dfs.append(pair_df)
            cdf = pd.concat(pair_dfs)  # CONCATENATE

    else :
        colors = fun.N_colors(Ngroups)
        palette = {id: c for id, c in zip(group_ids, colors)}
        PI_array = fun.boolean_indexing(all_PIs).T
        df = pd.DataFrame(PI_array, columns=group_ids).assign(Trial=1)
        cdf = pd.concat([df])  # CONCATENATE
    mdf = pd.melt(cdf, id_vars=['Trial'], var_name=['Group'])  # MELT

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    sns.boxplot(x="Trial", y="value", hue="Group", data=mdf, palette=palette, ax=ax, width=.5,
                fliersize=3, linewidth=None,whis=1.0)  # RUN PLOT
    ax.set_ylabel('Odor preference')
    ax.set_xlabel(xlabel)
    ax.set_ylim([-1, 1])
    ax.legend(loc='lower left', fontsize=12)
    fig.subplots_adjust(top=0.9, bottom=0.15, left=0.2, right=0.9, hspace=.005, wspace=0.05)
    return process_plot(fig, save_to, filename, return_fig, show)

def plot_heatmap_PI(save_to, csv_filepath='PIs.csv', return_fig=False, show=False):
    filename = 'PI_heatmap.pdf'
    print('Creating heatmap')
    new_data = pd.read_csv(csv_filepath, index_col=0)
    new_data.sort_index(ascending=True, inplace=True)
    new_data = new_data.reindex(sorted(new_data.columns, reverse=True), axis=1)
    Lgains = new_data.index.values.astype(int)
    Rgains = new_data.columns.values.astype(int)
    Ngains = len(Lgains)

    grid_kws = {"height_ratios": (.9, .05), "hspace": 0.4}
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(new_data, annot=False, fmt="g", cmap='RdYlGn', vmin=-1, vmax=1, ax=ax,
                cbar_kws={"orientation": "vertical",
                          'label': 'Preference for left odor',
                          'ticks': [1, 0, -1]})
    # ax.set_size_cm(3.5, 3.5)
    cax = plt.gcf().axes[-1]
    cax.tick_params(length=0)

    # ax.set_title('Preference index for variable odor gain combinations')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_xlabel(r'Left odor gain, $G_{L}$')
    ax.set_ylabel(r'Right odor gain, $G_{R}$')
    # ax.set_ylabel(r'$V_{left}$')
    # ax.set_ylabel(r'Valence$_{left}$')
    # ax.set_ylabel('Left odor valence')
    # ax.set_xlabel(r'$Gain_{right}$')
    # ax.set_xlabel(r'$V_{right}$')
    # ax.set_xlabel(r'Valence$_{right}$')
    # ax.set_xlabel('Right odor valence')
    # ax.xaxis.set_ticks_position('top')
    r = np.linspace(0.5, Ngains - 0.5, 5)
    ax.set_xticks(r)
    ax.set_yticks(r)
    # print(Rgains)
    # print(Lgains)

    # print(Lgains[r.astype(int)])
    # print(Rgains[r.astype(int)])
    ax.set_xticklabels(Lgains[r.astype(int)])
    ax.set_yticklabels(Rgains[r.astype(int)])
    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.95)
    return process_plot(fig, save_to, filename, return_fig, show)


def plot_odor_concentration(**kwargs):
    return plot_timeplot(['c_odor1'], **kwargs)


def plot_sensed_odor_concentration(**kwargs):
    return plot_timeplot(['dc_odor1'], **kwargs)


def plot_Y_pos(**kwargs):
    return plot_timeplot(['y'], **kwargs)


def plot_timeplot(par_shorts, datasets, labels=None, same_plot=True, individuals=False, table=None, show_first=True,
                  subfolder='timeplots', legend_loc='upper left', save_to=None, save_as=None, return_fig=False,
                  show=False, **kwargs):
    Ndatasets, colors, save_to, labels = plot_config(datasets, labels, save_to, subfolder)
    N = len(par_shorts)
    cols = ['grey'] if N == 1 else fun.N_colors(N)
    if not same_plot:
        raise NotImplementedError
    if N == 1:
        filename = f'{par_shorts[0]}.{suf}' if save_as is None else save_as
    elif N == 2:
        filename = f'{par_shorts[0]}_VS_{par_shorts[1]}.{suf}' if save_as is None else save_as
    else:
        filename = f'{N}_pars.{suf}' if save_as is None else save_as
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 5))
    for short, c in zip(par_shorts, cols):
        p, symbol, ylab, ylim = getPar(short, to_return=['d', 's', 'l', 'lim'])

        for d, d_col, d_lab in zip(datasets, colors, labels):
            if Ndatasets > 1:
                c = d_col
            s = d.load_table(table) if table is not None else d.step_data
            if p not in list(s.keys()):
                print(f'Parameter {p} does not exist in dataset')
                continue
            # print(s[p])
            dc = s[p]
            dc_m = dc.groupby(level='Step').quantile(q=0.5)
            Nticks = len(dc_m)
            if table is None:
                x = np.linspace(0, int(Nticks / d.fr), Nticks)
                xlab = 'time, $sec$'
            else:
                x = np.arange(Nticks)
                xlab = 'timesteps'

            if individuals:
                for id in dc.index.get_level_values('AgentID'):
                    dc_single = dc.xs(id, level='AgentID')
                    ax.plot(x, dc_single, color=c, linewidth=1)
                ax.plot(x, dc_m, 'r', linewidth=2)
            else:

                dc_u = dc.groupby(level='Step').quantile(q=0.75)
                dc_b = dc.groupby(level='Step').quantile(q=0.25)

                plot_mean_and_range(x=x, mean=dc_m, lb=dc_u, ub=dc_b, axis=ax, color_mean=c, color_shading=c,
                                    label=symbol)
                if show_first:
                    dc0 = dc.xs(dc.index.get_level_values('AgentID')[0], level='AgentID')
                    ax.plot(x, dc0, 'r')

    ax.set_ylabel(ylab)
    ax.set_xlabel(xlab)
    ax.set_xlim([x[0], x[-1]])
    if ylim is not None:
        ax.set_ylim(ylim)
    if N > 1:
        ax.legend()
    ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
    if Ndatasets > 1:
        dataset_legend(labels, colors, ax=ax, loc=legend_loc, fontsize=15)

    plt.subplots_adjust(bottom=0.15, left=0.2, right=0.95, top=0.95)
    return process_plot(fig, save_to, filename, return_fig, show)


def plot_navigation_index(datasets, labels=None, subfolder='source', save_as=None, save_to=None, return_fig=False,
                          show=False):
    Ndatasets, colors, save_to, labels = plot_config(datasets, labels, save_to, subfolder=subfolder)
    filename = f'nav_index.{suf}' if save_as is None else save_as

    fig, axs = plt.subplots(2, 1, figsize=(20, 20), sharex=True, sharey=True)

    for d, c, g in zip(datasets, colors, labels):
        dt = 1 / d.fr
        Nticks = d.num_ticks
        Nsec = int(Nticks * dt)
        s, e = d.step_data, d.endpoint_data

        vxs = []
        vys = []
        for id in d.agent_ids:
            s0 = s.xs(id, level='AgentID')
            s0 = s0[['x', 'y']].values
            v0 = fun.compute_velocity(s0, dt=dt)
            vx = fun.compute_component_velocity(s0, angles=np.zeros(Nticks), dt=dt)
            # vx = fun.compute_velocity(s0 * np.array([1, 0]), dt=dt)
            # vy = fun.compute_component_velocity(s0, dt=dt)
            vy = fun.compute_component_velocity(s0, angles=np.ones(Nticks) * -np.pi / 2, dt=dt)
            # vy = fun.compute_component_velocity(s0, angles=np.zeros(Nticks)*12, dt=dt)
            # vy = fun.compute_velocity(s0 * np.array([0, 1]), dt=dt)
            # print(s0* np.array([1, 0]))
            # print(s0* np.array([0, 1]))
            # raise
            vx = np.divide(vx, v0, out=np.zeros_like(v0), where=v0 != 0)
            vy = np.divide(vy, v0, out=np.zeros_like(v0), where=v0 != 0)
            vxs.append(vx)
            vys.append(vy)
        vx0 = np.nanmean(np.array(vxs), axis=0)
        vy0 = np.nanmean(np.array(vys), axis=0)
        axs[0].plot(np.linspace(0, Nsec, Nticks - 1), vx0, color=c, label=g)
        axs[1].plot(np.linspace(0, Nsec, Nticks - 1), vy0, color=c, label=g)

        # break
        # break
    plt.subplots_adjust(top=0.98, bottom=0.2, left=0.1, right=0.95, hspace=0.15)
    axs[0].legend()
    axs[1].set_xlabel('time (sec)')
    axs[0].set_ylabel('X index')
    axs[1].set_ylabel('Y index')
    axs[1].set_xlim([0, Nsec])
    axs[1].set_ylim([-1.0, 1.0])
    # axs[0].avhline(0.5)
    axs[0].axhline(0.0, color='green', alpha=0.5, linestyle='dashed', linewidth=1)
    axs[1].axhline(0.0, color='green', alpha=0.5, linestyle='dashed', linewidth=1)
    # plt.show()
    return process_plot(fig, save_to, filename, return_fig, show)


def plot_stridesNpauses(datasets, labels=None, stridechain_duration=False, pause_chunk='pause', time_unit='sec',
                        plot_fits='all', range='default', print_fits=False, only_fit_one=True, mode='cdf',
                        subfolder='bouts', refit_distros=False, test_detection=False,
                        save_to=None, save_as=None, save_fits_to=None, save_fits_as=None, return_fig=False, show=False):
    from lib.anal.fitting import compute_density,  get_distro, fit_bout_distros
    warnings.filterwarnings('ignore')
    Ndatasets, colors, save_to, labels = plot_config(datasets, labels, save_to, subfolder=subfolder)

    if save_as is None:
        base_file = f'stridesNpauses_{mode}_{range}_{plot_fits}'
        filename = f'{base_file}.{suf}' if not only_fit_one else f'{base_file}_0.{suf}'
    else:
        filename = save_as
    if save_fits_as is None:
        save_fits_as = f'bout_fits_{range}.csv'
    if save_fits_to is None:
        save_fits_to = save_to
    if not os.path.exists(save_fits_to):
        os.makedirs(save_fits_to)
    fit_filename = os.path.join(save_fits_to, save_fits_as)

    pause_par = nam.dur(pause_chunk)
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
    for label, dataset in zip(labels, datasets):
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
        for l, d, col in zip(labels, datasets, colors):
            dic = d.load_bout_dicts()
            pau_dur = np.array(fun.flatten_list([ddic[pause_par] for ddic in dic.values()]))
            chn_dur = np.array(fun.flatten_list([ddic[chain_par] for ddic in dic.values()]))
            pau_durs.append(pau_dur)
            chn_durs.append(chn_dur)
            labels.append(f'{l} truth')
            frs.append(d.fr)
            colors.append(f'dark{col}')

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
    fits = {l: {} for l in labels}

    fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharex=False, sharey=True)
    axs = axs.ravel()

    distro_ls = ['powerlaw', 'exponential', 'lognormal', 'lognorm-pow', 'levy', 'normal', 'uniform']
    distro_cs = ['c', 'g', 'm', 'k', 'yellow', 'brown', 'purple']
    num_distros=len(distro_ls)

    for j, (pau_dur, chn_dur, c, label, fr) in enumerate(zip(pau_durs, chn_durs, colors, labels, frs)):
        try:
            ref = conf.loadConf(label, 'Ref')
        except:
            ref = None
        for i, (x0, discr, xmin, xmax) in enumerate(
                zip([chn_dur, pau_dur], [chn_discr, pau_discr], [chn0, pau0], [chn1, pau1])):
            bout = 'stride' if i == 0 else 'pause'
            combine = False
            # combine=False if i == 0 else True
            lws = [2]*num_distros

            if not refit_distros and ref is not None:

                u2, du2, c2, c2cum = compute_density(x0, xmin, xmax)
                fitted = ref[bout]['best']
                pdfs = [get_distro(x=du2, **fitted, mode='pdf')] * num_distros
                cdfs = [1 - get_distro(x=u2, **fitted, mode='cdf')] * num_distros
                idx_Kmax = 0

            else:
                fit_dic=fit_bout_distros(x0, xmin, xmax, fr, discr,dataset_id=label, bout=bout,
                                         print_fits=print_fits,combine=combine)
                idx_Kmax=fit_dic['idx_Kmax']
                cdfs=fit_dic['cdfs']
                pdfs=fit_dic['pdfs']
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

            axs[i].loglog(xrange, y, '.', color=c, alpha=0.7)
            for z, (l, col, lw, ddf) in enumerate(zip(distro_ls, distro_cs, lws, ddfs)):
                if ddf is None:
                    continue
                if plot_fits == 'best' and z == idx_Kmax:
                    cc = c
                elif plot_fits == 'all':
                    cc = col
                else:
                    continue
                axs[i].loglog(xrange, ddf, color=cc, lw=lw, label=l)

    for ii in [0, 1]:
        if plot_fits == 'all':
            dataset_legend(distro_ls, distro_cs, ax=axs[ii], loc='lower left', fontsize=15)
        dataset_legend(labels, colors, ax=axs[ii], loc='upper right', fontsize=15)
    axs[0].set_ylabel(ylabel)
    axs[0].set_xlabel(chain_xlabel)
    axs[1].set_xlabel(pause_xlabel)
    axs[0].set_xlim([chn_t0, chn_t1])
    axs[1].set_xlim([pau_t0, pau_t1])
    axs[1].set_ylim([10 ** -3.5, 10 ** 0])
    axs[0].set_title(r'$\bf{stridechains}$')
    axs[1].set_title(r'$\bf{pauses}$')
    fig.subplots_adjust(top=0.92, bottom=0.15, left=0.15, right=0.95, hspace=.005, wspace=0.05)
    fit_df = pd.DataFrame.from_dict(fits, orient="index")
    fit_df.to_csv(fit_filename, index=True, header=True)
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

    point = d.point
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
    # lin_cs = ['darkred', 'red', 'lightsalmon']
    ang_cs = ['black']
    cs = [lin_cs, ang_cs]
    lin_labels = [r'$\bf{head}$', r'$\bf{mid}$', r'$\bf{tail}$']
    ang_labels = [r'$\dot{\theta}_{or}$']
    labels = [lin_labels, ang_labels]
    ylabels = [r'scaled velocity $(sec^{-1})$', 'angular velocity $(deg/sec)$']

    for i in [0, 1]:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        # print(vels[i], cs[i], cs[i], labels[i])
        for serie, vel, col, c, l in zip(vel_timeseries[i], vels[i], cs[i], cs[i], labels[i]):
            print(vel, col, c, l)
            array = [np.interp(x=np.linspace(0, 2 * np.pi, Npoints), xp=np.linspace(0, 2 * np.pi, dur), fp=ts, left=0,
                               right=0) for dur, ts in zip(durations, serie)]
            serie_m, serie_l, serie_h = np.nanquantile(array, q=0.5, axis=0), \
                                        np.nanquantile(array, q=0.25, axis=0), \
                                        np.nanquantile(array, q=0.75, axis=0)

            plot_mean_and_range(x=np.arange(len(serie_m)), mean=serie_m, lb=serie_l, ub=serie_h, axis=ax, color_mean=c,
                                color_shading=col, label=l)
            # break

        Nticks = 5
        ticks = np.linspace(0, Npoints - 1, Nticks)
        # ax.locator_params(axis='x', nbins=Nticks)
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


def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object_class to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse'sigma radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = patches.Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                              facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def plot_correlated_pars(dataset, pars, labels, save_to=None, save_as=f'correlated_pars.{suf}', return_fig=False):
    if len(pars) != 3:
        raise ValueError('Currently implemented only for 3 parameters')
    if save_to is None:
        save_to = dataset.plot_dir
    e = dataset.endpoint_data
    # e = e[e['length'] > 2.5]
    g = sns.PairGrid(e[pars])
    g.map_upper(sns.scatterplot)
    g.map_lower(sns.kdeplot, fill=True)
    g.map_diag(sns.histplot, kde=True, bins=20)

    for i, ax in enumerate(g.axes[-1, :]):
        ax.xaxis.set_label_text(labels[i])
    for j, ax in enumerate(g.axes[:, 0]):
        ax.yaxis.set_label_text(labels[j])
    # g.axes[0,1].scatter(x=samples[1], y=samples[0], marker='.', color='r', alpha=0.1)
    for ax, (i, j) in zip([g.axes[0, 1], g.axes[0, 2], g.axes[1, 2]], [(1, 0), (2, 0), (2, 1)]):
        for std, a in zip([0.5, 1, 2, 3], [0.4, 0.3, 0.2, 0.1]):
            confidence_ellipse(x=e[pars[i]].values, y=e[pars[j]].values,
                               ax=ax, n_std=std, facecolor='red', alpha=a)
    return process_plot(g, save_to, save_as, return_fig)


def plot_ang_pars(datasets, labels=None, simVSexp=False, absolute=True, include_turns=False, include_rear=False,
                  subfolder='turn',
                  save_fits_as=None, save_as=None, save_to=None, Npars=3, return_fig=False, show=False):
    Ndatasets, colors, save_to, labels = plot_config(datasets, labels, save_to, subfolder=subfolder)
    if Npars == 5:
        par_shorts = ['b', 'bv', 'ba', 'fov', 'foa']
        ranges = [100, 200, 2000, 200, 2000]
        ylim = 0.05
    elif Npars == 3:
        par_shorts = ['b', 'bv', 'fov']
        ranges = [100, 200, 200]
        ylim = 0.05

    if include_rear:
        par_shorts += ['rov', 'roa']
        ranges += [200, 2000]
    if include_turns:
        par_shorts += ['tur_fou']
        ranges += [100]

    pars, sim_labels, exp_labels, xlabels = getPar(par_shorts, to_return=['d', 's', 's', 'l'])

    filename = f'angular_pars_{len(pars)}.{suf}' if save_as is None else save_as
    fit_filename = 'ang_pars_ttest.csv' if save_fits_as is None else save_fits_as
    fit_filepath = os.path.join(save_to, fit_filename)

    p_labels = [[sl, el] for sl, el in zip(sim_labels, exp_labels)] if simVSexp else [[sl] * Ndatasets for sl in
                                                                                      sim_labels]

    if Ndatasets > 1:
        fit_ind = np.array([np.array([l1, l2]) for l1, l2 in itertools.combinations(labels, 2)])
        fit_ind = pd.MultiIndex.from_arrays([fit_ind[:, 0], fit_ind[:, 1]], names=('dataset1', 'dataset2'))
        fit_df = pd.DataFrame(index=fit_ind, columns=pars + [f'S_{p}' for p in pars] + [f'P_{p}' for p in pars])

    fig, axs = plt.subplots(1, len(pars), figsize=(len(pars) * 5, 5), sharey=True)
    axs = axs.ravel()
    nbins = 200

    # vs=[]
    for i, (p, r, p_lab, xlab) in enumerate(zip(pars, ranges, p_labels, xlabels)):
        vs = []
        for j, d in enumerate(datasets):
            v = d.get_par(p).dropna().values
            if absolute:
                v = np.abs(v)
                r1, r2 = 0, r
            else:
                r1, r2 = -r, r
            vs.append(v)
            x = np.linspace(r1, r2, nbins)
            weights = np.ones_like(v) / float(len(v))
            axs[i].hist(v, color=colors[j], bins=x, label=p_lab[j], weights=weights, alpha=0.8, histtype='step',
                        linewidth=2)
            axs[i].set_xlim(xmin=0)

        if Ndatasets > 1:
            for ind, (v1, v2) in zip(fit_ind, itertools.combinations(vs, 2)):
                st, pv = ttest_ind(v1, v2, equal_var=False)
                signif = pv <= 0.01
                temp = np.nanmean(v1) < np.nanmean(v2)
                if not signif:
                    fit_df[p].loc[ind] = 0
                else:
                    fit_df[p].loc[ind] = 1 if temp else -1
                fit_df[f'S_{p}'].loc[ind] = st
                fit_df[f'P_{p}'].loc[ind] = np.round(pv, 11)
            ii = 0
            for z, (l1, l2) in enumerate(fit_df.index.values):
                if fit_df[p].iloc[z] == 1:
                    c1, c2 = colors[labels.index(l1)], colors[labels.index(l2)]
                elif fit_df[p].iloc[z] == -1:
                    c1, c2 = colors[labels.index(l2)], colors[labels.index(l1)]
                else:
                    ii += 1
                    continue
                rad = 0.04
                yy = 0.95 - (z - ii) * 0.08
                xx = 0.75
                dual_half_circle(center=(xx, yy), radius=rad, angle=90, ax=axs[i], colors=(c1, c2),
                                 transform=axs[i].transAxes)
                pv = fit_df[f'P_{p}'].loc[(l1, l2)]
                if pv == 0:
                    pvi = -9
                else:
                    for pvi in np.arange(-1, -10, -1):
                        if np.log10(pv) > pvi:
                            pvi += 1
                            break
                axs[i].text(xx + 0.05, yy + rad / 1.5, f'p<10$^{{{pvi}}}$', ha='left', va='top', color='k', fontsize=15,
                            transform=axs[i].transAxes)

        axs[i].set_xlabel(xlab)
        axs[i].yaxis.set_major_locator(ticker.MaxNLocator(3))
    dataset_legend(labels, colors, ax=axs[0], loc='upper left')
    # axs[0].legend(handles=[patches.Patch(facecolor=c, label=id, edgecolor='black') for c, id in zip(colors, labels)],
    #               labels=labels, loc='upper left', handlelength=0.5, handleheight=0.5)
    axs[0].set_ylabel('probability')
    axs[0].set_ylim([0, ylim])
    plt.subplots_adjust(bottom=0.15, top=0.95, left=0.3 / len(pars), right=0.99, wspace=0.01)
    if Ndatasets > 1:
        fit_df.to_csv(fit_filepath, index=True, header=True)
    return process_plot(fig, save_to, filename, return_fig, show)


def plot_bout_ang_pars(datasets, labels=None, simVSexp=False, absolute=True, include_rear=True, subfolder='turn',
                       save_fits_as=None, save_as=None, save_to=None, return_fig=False, show=False):
    Ndatasets, colors, save_to, labels = plot_config(datasets, labels, save_to, subfolder=subfolder)
    par_shorts = ['bv', 'fov', 'rov', 'ba', 'foa', 'roa'] if include_rear else ['bv', 'fov', 'ba', 'foa']
    ranges = [250, 250, 50, 2000, 2000, 500] if include_rear else [200, 200, 2000, 2000]
    ylim = 0.04

    pars, sim_labels, exp_labels, xlabels, xlims, disps = getPar(par_shorts, to_return=['d', 's', 's', 'l', 'lim', 'd'])

    chunks = ['stride', 'pause']
    chunk_cols = ['green', 'purple']

    filename = f'bout_ang_pars.{suf}' if save_as is None else save_as
    fit_filename = 'bout_ang_pars_ttest.csv' if save_fits_as is None else save_fits_as
    fit_filepath = os.path.join(save_to, fit_filename)

    p_labels = [[sl, el] for sl, el in zip(sim_labels, exp_labels)] if simVSexp else [[sl] * Ndatasets for sl in
                                                                                      sim_labels]

    # fit_ind = np.array([np.array([l1, l2]) for l1, l2 in itertools.combinations(labels, 2)])
    # fit_ind = pd.MultiIndex.from_arrays([fit_ind[:, 0], fit_ind[:, 1]], names=('dataset1', 'dataset2'))
    fit_ind = labels
    fit_df = pd.DataFrame(index=fit_ind, columns=pars + [f'S_{p}' for p in pars] + [f'P_{p}' for p in pars])

    Ncols = int(len(pars) / 2)
    fig, axs = plt.subplots(2, Ncols, figsize=(Ncols * 7, 14), sharey=True)
    axs = axs.ravel()
    nbins = 200

    # vs=[]
    for i, (p, r, p_lab, xlab, disp) in enumerate(zip(pars, ranges, p_labels, xlabels, disps)):

        for j, (d, l) in enumerate(zip(datasets, labels)):
            vs = []
            s, e = d.step_data, d.endpoint_data
            for c, col in zip(chunks, chunk_cols):
                temp = s.dropna(subset=[nam.id(c)])
                # if c=='pause' and pause_min_dur!=0.0 :
                #     temp = temp[temp.groupby('AgentID')]
                v = temp[p].values
                if absolute:
                    v = np.abs(v)
                    r1, r2 = 0, r
                else:
                    r1, r2 = -r, r
                vs.append(v)
                x = np.linspace(r1, r2, nbins)
                weights = np.ones_like(v) / float(len(v))
                axs[i].hist(v, color=col, bins=x, label=c, weights=weights, alpha=1.0, histtype='step', linewidth=2)
                axs[i].set_xlim([r1, r2])

            # if Ndatasets > 1:
            #     for ind, (v1, v2) in zip(fit_ind, itertools.combinations(vs, 2)):
            st, pv = ttest_ind(vs[0], vs[1], equal_var=False)
            signif = pv <= 0.01
            temp = np.nanmean(vs[0]) < np.nanmean(vs[1])
            if not signif:
                fit_df[p].loc[l] = 0
            else:
                fit_df[p].loc[l] = 1 if temp else -1
            fit_df[f'S_{p}'].loc[l] = st
            fit_df[f'P_{p}'].loc[l] = np.round(pv, 11)
            # ii = 0
            # for z, (l1, l2) in enumerate(fit_df.index.values):
            if fit_df[p].loc[l] == 1:
                c1, c2 = chunk_cols[0], chunk_cols[1]
            elif fit_df[p].loc[l] == -1:
                c1, c2 = chunk_cols[1], chunk_cols[0]
            else:
                # ii += 1
                continue
            rad = 0.04
            yy = 0.95
            xx = 0.75
            dual_half_circle(center=(xx, yy), radius=rad, angle=90, ax=axs[i], colors=(c1, c2),
                             transform=axs[i].transAxes)
            pv = fit_df[f'P_{p}'].loc[l]
            if pv == 0:
                pvi = -9
            else:
                for pvi in np.arange(-1, -10, -1):
                    if np.log10(pv) > pvi:
                        pvi += 1
                        break
            axs[i].text(xx + 0.05, yy + rad / 1.5, f'p<10$^{{{pvi}}}$', ha='left', va='top', color='k', fontsize=15,
                        transform=axs[i].transAxes)

        axs[i].set_xlabel(xlab)
        axs[i].yaxis.set_major_locator(ticker.MaxNLocator(3))
    axs[0].set_ylabel('probability')
    axs[Ncols].set_ylabel('probability')
    axs[0].set_ylim([0, ylim])
    axs[0].legend(loc='upper left')
    axs[Ncols].legend(loc='upper left')
    plt.subplots_adjust(bottom=0.1, top=0.9, left=0.25 / Ncols, right=0.95, wspace=0.1, hspace=0.3)
    # plt.show()
    if Ndatasets > 1:
        fit_df.to_csv(fit_filepath, index=True, header=True)
    return process_plot(fig, save_to, filename, return_fig, show)


def plot_crawl_pars(datasets, labels=None, simVSexp=False, subfolder='endpoint',
                    save_as=None, save_fits_as=None, save_to=None, par_legend=False, return_fig=False, show=False):
    Ndatasets, colors, save_to, labels = plot_config(datasets, labels, save_to, subfolder=subfolder)
    filename = f'crawl_pars.{suf}' if save_as is None else save_as
    fit_filename = 'ang_pars_ttest.csv' if save_fits_as is None else save_fits_as
    fit_filepath = os.path.join(save_to, fit_filename)

    par_shorts = ['str_N', 'str_tr', 'cum_d']
    pars, sim_labels, exp_labels, xlabels, xlims = getPar(par_shorts, to_return=['d', 's', 's', 'l', 'lim'])
    # ranges = [(100, 300), (0.5, 1.0), (80, 320)]

    if simVSexp:
        p_labels = [[sl, el] for sl, el in zip(sim_labels, exp_labels)]
    else:
        p_labels = [[sl] * Ndatasets for sl in sim_labels]

    if Ndatasets > 1:
        fit_ind = np.array([np.array([l1, l2]) for l1, l2 in itertools.combinations(labels, 2)])
        fit_ind = pd.MultiIndex.from_arrays([fit_ind[:, 0], fit_ind[:, 1]], names=('dataset1', 'dataset2'))
        fit_df = pd.DataFrame(index=fit_ind, columns=pars + [f'S_{p}' for p in pars] + [f'P_{p}' for p in pars])

    fig, axs = plt.subplots(1, len(pars), figsize=(len(pars) * 5, 5), sharey=True)
    axs = axs.ravel()
    nbins = 40
    for i, (p, p_lab, xlab, xlim) in enumerate(zip(pars, p_labels, xlabels, xlims)):
        vs = [d.get_par(p).dropna().values for d in datasets]
        r1, r2 = np.min([np.min(v) for v in vs]), np.max([np.max(v) for v in vs])
        # r1, r2 = r[0], r[1]
        x = np.linspace(r1, r2, nbins)
        for j, d in enumerate(datasets):
            # v = d.get_par(p).dropna().values
            # axs[i].set_xlim([-ranges[i], ranges[i]])
            # statistic, pvalue = ks_2samp(exp, exp)

            # weights = np.ones_like(vs) / float(len(vs))
            # exp_weights = np.ones_like(exp) / float(len(exp))
            # sim_weights = np.ones_like(exp) / float(len(exp))
            # sns.distplot(exp, color="red", ax=axs[i], bins=x, hist=False, label=sim_labels[i],
            #              kde=True)
            # sns.distplot(exp, color="blue", ax=axs[i], bins=x, hist=False, label=exp_labels[i],
            #              kde=True)
            # axs[i].hist(exp, color="red", bins=x, label=sim_labels[i],
            #             weights=sim_weights, alpha=0.5, histtype='stepfilled')
            # axs[i].hist(exp, color="blue", bins=x, label=exp_labels[i],
            #             weights=exp_weights, alpha=0.5, histtype='stepfilled')
            sns.histplot(vs[j], color=colors[j], bins=x, kde=True, ax=axs[i], label=p_lab[j],
                         stat="probability", element="step")
        # sns.distplot(exp, color="red", bins=x, hist=False, ax=axs[i], label=sim_labels[i],
        #              norm_hist=True)
        # sns.distplot(exp, color="blue", bins=x, hist=False, ax=axs[i], label=exp_labels[i],
        #              norm_hist=True)
        axs[i].set_xlabel(xlab)
        # axs[i].set_ylim([0,0.2])
        axs[i].yaxis.set_major_locator(ticker.MaxNLocator(4))

        if Ndatasets > 1:
            for ind, (v1, v2) in zip(fit_ind, itertools.combinations(vs, 2)):
                st, pv = ttest_ind(v1, v2, equal_var=False)
                signif = pv <= 0.01
                temp = np.nanmean(v1) < np.nanmean(v2)
                if not signif:
                    fit_df[p].loc[ind] = 0
                else:
                    fit_df[p].loc[ind] = 1 if temp else -1
                fit_df[f'S_{p}'].loc[ind] = st
                fit_df[f'P_{p}'].loc[ind] = np.round(pv, 11)
            ii = 0
            for z, (l1, l2) in enumerate(fit_df.index.values):
                if fit_df[p].iloc[z] == 1:
                    c1, c2 = colors[labels.index(l1)], colors[labels.index(l2)]
                elif fit_df[p].iloc[z] == -1:
                    c1, c2 = colors[labels.index(l2)], colors[labels.index(l1)]
                else:
                    ii += 1
                    continue
                rad = 0.04
                yy = 0.95 - (z - ii) * 0.08
                xx = 0.75
                dual_half_circle(center=(xx, yy), radius=rad, angle=90, ax=axs[i], colors=(c1, c2),
                                 transform=axs[i].transAxes)
                pv = fit_df[f'P_{p}'].loc[(l1, l2)]
                if pv == 0:
                    pvi = -9
                else:
                    for pvi in np.arange(-1, -10, -1):
                        if np.log10(pv) > pvi:
                            pvi += 1
                            break
                axs[i].text(xx + 0.05, yy + rad / 1.5, f'p<10$^{{{pvi}}}$', ha='left', va='top', color='k', fontsize=15,
                            transform=axs[i].transAxes)
        if par_legend:
            axs[i].legend(loc='upper right')
        if xlim is not None:
            axs[i].set_xlim(xlim)
    dataset_legend(labels, colors, ax=axs[0], loc='upper left', fontsize=15)
    axs[0].set_ylabel('probability')
    plt.subplots_adjust(bottom=0.15, top=0.95, left=0.25 / len(pars), right=0.99, wspace=0.01)
    if Ndatasets > 1:
        fit_df.to_csv(fit_filepath, index=True, header=True)
    # plt.show()
    # raise
    return process_plot(fig, save_to, filename, return_fig, show)


def plot_endpoint_params(datasets, labels=None, mode='basic', par_shorts=None, subfolder='endpoint',
                         save_to=None, save_as=None, save_fits_as=None, return_fig=False, show=False):
    warnings.filterwarnings('ignore')
    Ndatasets, colors, save_to, labels = plot_config(datasets, labels, save_to, subfolder=subfolder)
    filename = f'endpoint_params_{mode}.{suf}' if save_as is None else save_as
    fit_filename = 'endpoint_ttest.csv' if save_fits_as is None else save_fits_as
    fit_filepath = os.path.join(save_to, fit_filename)

    ylim = [0.0, 0.25]
    nbins = 20
    l_par='l' # 'l_mu
    if par_shorts is None:
        if mode == 'basic':
            par_shorts = [l_par, 'fsv', 'sv_mu', 'sstr_d_mu',
                          'str_tr', 'pau_tr', 'Ltur_tr', 'Rtur_tr',
                          'tor20_mu', 'dsp_0_40_fin', 'b_mu', 'bv_mu']
        elif mode == 'minimal':
            par_shorts = [l_par, 'fsv', 'sv_mu', 'sstr_d_mu',
                          'cum_t', 'str_tr', 'pau_tr', 'tor',
                          'tor5_mu', 'tor20_mu', 'dsp_0_40_max', 'dsp_0_40_fin',
                          'b_mu', 'bv_mu', 'Ltur_tr', 'Rtur_tr']
        elif mode == 'stride_def':
            par_shorts = [l_par, 'fsv', 'sstr_d_mu', 'sstr_d_std']
        elif mode == 'reorientation':
            par_shorts = ['str_fo_mu', 'str_fo_std', 'tur_fou_mu', 'tur_fou_std']
        elif mode == 'tortuosity':
            par_shorts = ['tor2_mu', 'tor5_mu', 'tor10_mu', 'tor20_mu']
        elif mode == 'result':
            par_shorts = ['sv_mu', 'str_tr', 'pau_tr', 'pau_t_mu']
        elif mode == 'limited':
            par_shorts = [l_par, 'fsv', 'sv_mu', 'sstr_d_mu',
                          'cum_t', 'str_tr', 'pau_tr', 'pau_t_mu',
                          'tor5_mu', 'tor5_std', 'tor20_mu', 'tor20_std',
                          'tor', 'sdsp_mu', 'sdsp_0_40_max', 'sdsp_0_40_fin',
                          'b_mu', 'b_std', 'bv_mu', 'bv_std',
                          'Ltur_tr', 'Rtur_tr', 'Ltur_fou_mu', 'Rtur_fou_mu']
        elif mode == 'full':
            par_shorts = [l_par, 'str_N', 'fsv',
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
                          ]
        elif mode == 'deb':
            par_shorts = [
                'deb_f_mu', 'hunger', 'reserve_density', 'puppation_buffer',
                'cum_d', 'cum_sd', 'str_N', 'fee_N',
                'str_tr', 'pau_tr', 'fee_tr', 'f_am',
                l_par, 'm'
                # 'tor2_mu', 'tor5_mu', 'tor10_mu', 'tor20_mu',
                # 'v_mu', 'sv_mu',

            ]
        else:
            raise ValueError('Provide parameter shortcuts or define a mode')
    ends=[]
    for d in datasets :
        try :
            e=d.endpoint_data
        except :
            e=d.read('end')
        ends.append(e)
    pars, = getPar(par_shorts, to_return=['d'])
    pars = [p for p in pars if all([p in e.columns for e in ends])]
    symbols, exp_symbols, xlabels, xlims, disps = getPar(par_shorts, to_return=['s', 's', 'l', 'lim', 'd'])

    if mode == 'stride_def':
        xlims = [[2.5, 4.8], [0.8, 2.0], [0.1, 0.25], [0.02, 0.09]]
    if Ndatasets > 1:
        fit_ind = np.array([np.array([l1, l2]) for l1, l2 in itertools.combinations(labels, 2)])
        fit_ind = pd.MultiIndex.from_arrays([fit_ind[:, 0], fit_ind[:, 1]], names=('dataset1', 'dataset2'))
        fit_df = pd.DataFrame(index=fit_ind, columns=pars + [f'S_{p}' for p in pars] + [f'P_{p}' for p in pars])

    lw = 3
    Npars = len(pars)
    if Npars==4:
        Ncols=2
        Nrows=2
    else:
        Ncols = int(np.min([Npars, 4]))
        Nrows = int(np.ceil(Npars / Ncols))
    fig_s = 5

    fig, axs = plt.subplots(Nrows, Ncols, figsize=(fig_s * Ncols, fig_s * Nrows), sharey=True)
    axs = axs.ravel() if Nrows * Ncols > 1 else [axs]
    for i, (p, symbol, xlabel, xlim, disp) in enumerate(zip(pars, symbols, xlabels, xlims, disps)):
        # if xlim is not None :
        #     print(p, xlabel,xlim, mode(xlim), xlim[0])
        values = [e[p].values for e in ends]
        # print(p)
        if Ndatasets > 1:
            for ind, (v1, v2) in zip(fit_ind, itertools.combinations(values, 2)):
                st, pv = ttest_ind(v1, v2, equal_var=False)
                signif = pv <= 0.01
                temp = np.nanmean(v1) < np.nanmean(v2)
                if not signif:
                    ii = 0
                else:
                    if temp:
                        ii = 1
                    else:
                        ii = -1
                fit_df[f'S_{p}'].loc[ind] = st
                fit_df[f'P_{p}'].loc[ind] = np.round(pv, 11)
                fit_df[p].loc[ind] = ii

        Nvalues = [len(i) for i in values]
        a = np.empty((np.max(Nvalues), len(values),)) * np.nan
        for k in range(len(values)):
            a[:Nvalues[k], k] = values[k]
        df = pd.DataFrame(a, columns=labels)
        for j, (col, lab) in enumerate(zip(df.columns, labels)):
            try:
                v = df[[col]].dropna().values
                weights = np.ones_like(v) / float(len(v))
                bins = nbins if xlim is None else np.linspace(xlim[0], xlim[1], nbins)
                y, x, patches = axs[i].hist(v, bins=bins, weights=weights, color=colors[j], alpha=0.5)
                x = x[:-1] + (x[1] - x[0]) / 2
                y_smooth = np.polyfit(x, y, 5)
                poly_y = np.poly1d(y_smooth)(x)
                axs[i].plot(x, poly_y, color=colors[j], label=lab, linewidth=lw)
            except:
                pass
        if i % Ncols == 0:
            axs[i].set_ylabel('probability')
        axs[i].set_title(disp)
        axs[i].set_xlabel(xlabel)
        if xlim is not None:
            axs[i].set_xlim(xlim)
        axs[i].xaxis.set_major_locator(ticker.MaxNLocator(4))
        axs[i].yaxis.set_major_locator(ticker.MaxNLocator(4))
        axs[i].xaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=True, useMathText=True))
        # axs[i].ticklabel_format(axis='x', useMathText=True, scilimits=(-3, 3))
        # axs[i].ticklabel_format(axis='x', useMathText=True, scilimits=(-3, 3), useOffset=True)

        if Ndatasets > 1:
            ii = 0
            for z, (l1, l2) in enumerate(fit_df.index.values):
                if fit_df[p].iloc[z] == 1:
                    c1, c2 = colors[labels.index(l1)], colors[labels.index(l2)]
                elif fit_df[p].iloc[z] == -1:
                    c1, c2 = colors[labels.index(l2)], colors[labels.index(l1)]
                else:
                    ii += 1
                    continue
                rad = 0.04
                yy = 0.95 - (z - ii) * 0.08
                xx = 0.7
                dual_half_circle(center=(xx, yy), radius=rad, angle=90, ax=axs[i], colors=(c1, c2),
                                 transform=axs[i].transAxes)
                pv = fit_df[f'P_{p}'].loc[(l1, l2)]
                if pv == 0:
                    pvi = -9
                else:
                    for pvi in np.arange(-1, -10, -1):
                        if np.log10(pv) > pvi:
                            pvi += 1
                            break
                axs[i].text(xx + 0.05, yy + rad / 1.5, f'p<10$^{{{pvi}}}$', ha='left', va='top', color='k', fontsize=15,
                            transform=axs[i].transAxes)

    plt.subplots_adjust(wspace=0.1, hspace=0.2 * Nrows, left=0.1, right=0.97, top=1 - (0.1 / Nrows),
                        bottom=0.17 / Nrows)
    plt.ylim(ylim)
    axs[0].legend(loc='upper left', prop={'size': 15})
    if Ndatasets > 1:
        fit_df.to_csv(fit_filepath, index=True, header=True)
    return process_plot(fig, save_to, filename, return_fig, show)


def plot_turn_duration(datasets, labels=None, save_to=None, return_fig=False, legend=True, absolute=True):
    return plot_turn_amp(datasets, labels, par_short='tur_t', mode='scatter', legend=legend, absolute=absolute,
                         save_to=save_to, return_fig=return_fig)


def plot_turn_amp(datasets, labels=None, par_short='tur_t', ref_angle=None,
                  subfolder='turn', mode='hist', cumy=True,
                  save_to=None, legend=True, show=False, absolute=True, return_fig=False):
    Ndatasets, colors, save_to, labels = plot_config(datasets, labels, save_to, subfolder=subfolder)
    nn = 'turn_amp' if ref_angle is None else 'rel_turn_angle'
    filename = f'{nn}_VS_{par_short}_{mode}.{suf}'

    ypar, ylab, ylim = getPar('tur_fou', to_return=['d', 'l', 'lim'])

    if ref_angle is not None:
        A0 = float(ref_angle)
        # ylim = (-180, 180)
        pars_ref, = getPar(['tur_fo0', 'tur_fo1'], to_return=['d'])
        ys = []
        ylab = r'$\Delta\theta_{bearing} (deg)$'
        cumylab = r'$\bar{\Delta\theta}_{bearing} (deg)$'
        for d in datasets:
            y0 = d.get_par(pars_ref[0]).dropna().values.flatten() - A0
            y1 = d.get_par(pars_ref[1]).dropna().values.flatten() - A0
            y0 %= 360
            y1 %= 360
            y0[y0 > 180] -= 360
            y1[y1 > 180] -= 360
            y = np.abs(y0) - np.abs(y1)
            ys.append(y)
            # print(d.id, sum(y)/len(d.agent_ids))

    else:
        cumylab = r'$\bar{\Delta\theta}_{or} (deg)$'
        ys = [d.get_par(ypar).dropna().values.flatten() for d in datasets]
        if absolute:
            ys = [np.abs(y) for y in ys]
        # ylim=None

    xpar, xlab = getPar(par_short, to_return=['d', 'l'])

    xs = [d.get_par(xpar).dropna().values.flatten() for d in datasets]

    if mode == 'scatter':
        fig, axs = plt.subplots(1, 1, figsize=(10, 10))
        for x, y, l, c in zip(xs, ys, labels, colors):
            plt.scatter(x=x, y=y, marker='o', s=5.0, color=c, alpha=0.5)
            m, k = np.polyfit(x, y, 1)
            axs.plot(x, m * x + k, linewidth=4, color=c, label=l)
            plt.legend(loc='upper left')
            plt.xlabel(xlab)
            plt.ylabel(ylab)
            plt.ylim(ylim)
            axs.yaxis.set_major_locator(ticker.MaxNLocator(4))
            plt.subplots_adjust(bottom=0.1, top=0.95, left=0.15, right=0.95, wspace=0.01)
    elif mode == 'hist':
        fig = scatter_hist(xs, ys, labels, colors, xlabel=xlab, ylabel=ylab, ylim=ylim, cumylabel=cumylab, cumy=cumy)
    # plt.show()
    # raise
    return process_plot(fig, save_to, filename, return_fig, show)


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
    # ax_histy.xaxis.set_ticks([0,0.1,0.2])

    ax_scatter.set_xlim([x0, x1])
    ax_scatter.set_ylim(ylim)
    ax_histx.set_xlim(ax_scatter.get_xlim())
    ax_histy.set_ylim(ax_scatter.get_ylim())

    # ax_histy.set_xlim(xmin=0.0)

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


def dataset_legend(labels, colors, ax=None, loc=None, anchor=None, fontsize=None, handlelength=0.5, handleheight=0.5,
                   **kwargs):
    if ax is None:
        leg = plt.legend(
            bbox_to_anchor=anchor,
            handles=[patches.Patch(facecolor=c, label=l, edgecolor='black') for c, l in zip(colors, labels)],
            labels=labels, loc=loc, handlelength=handlelength, handleheight=handleheight, fontsize=fontsize, **kwargs)
    else:
        leg = ax.legend(
            bbox_to_anchor=anchor,
            handles=[patches.Patch(facecolor=c, label=l, edgecolor='black') for c, l in zip(colors, labels)],
            labels=labels, loc=loc, handlelength=handlelength, handleheight=handleheight, fontsize=fontsize, **kwargs)
        ax.add_artist(leg)
    return leg


def plot_turns(datasets, labels=None, absolute=True, save_to=None, subfolder='turn', return_fig=False, show=False):
    Ndatasets, colors, save_to, labels = plot_config(datasets, labels, save_to, subfolder=subfolder)
    filename = f'turns.{suf}'
    fig, axs = plt.subplots(1, 1, figsize=(6, 5))
    par, xlabel = getPar('tur_fou', to_return=['d', 'l'])

    ts = [d.get_par(par).dropna().values for d in datasets]

    r = 150
    Nbins = 30

    for data, col, l in zip(ts, colors, labels):
        if absolute:
            data = np.abs(data)
            r0, r1 = np.min(data), r

        else:
            r0, r1 = -r, r
            Nbins *= 2

        x = np.linspace(r0, r1, Nbins)
        weights = np.ones_like(data) / float(len(data))
        axs.hist(data, bins=x, weights=weights, label=l, color=col, alpha=1.0, histtype='step')

    axs.set_ylabel('probability, $P$')
    axs.set_xlabel(xlabel)
    axs.set_xlim([r0, r1])
    axs.yaxis.set_major_locator(ticker.MaxNLocator(4))
    axs.legend(loc='upper right', fontsize=10)
    fig.subplots_adjust(top=0.92, bottom=0.15, left=0.25, right=0.95, hspace=.005, wspace=0.05)
    return process_plot(fig, save_to, filename, return_fig, show)


def plot_turn_Dbearing(datasets, labels=None, min_angle=30.0, max_angle=180.0, ref_angle=None,
                       par=nam.bearing2('center'), Nplots=4, subfolder='turn', save_to=None, return_fig=False,
                       show=False):
    Nds, colors, save_to, labels = plot_config(datasets, labels, save_to, subfolder=subfolder)
    fig, axs = plt.subplots(Nds, Nplots, figsize=(5 * Nplots, 5 * Nds), subplot_kw=dict(projection='polar'),
                            sharey=True)
    axs = axs.ravel()

    if par == nam.bearing2('center'):
        filename = f'turn_Dorient_to_center.{suf}'
        ang0 = 0
        norm = False
    elif par is None and ref_angle is not None:
        ang0 = ref_angle
        norm = True
        filename = f'turn_Dorient_to_{ang0}deg.{suf}'
        par = nam.unwrap(nam.orient('front'))
    else:
        raise ValueError('No parameter or target angle has been provided.')

    def circNarrow(ax, data, alpha, label, color):
        circular_hist(ax, data, bins=16, alpha=alpha, label=label, color=color, offset=np.pi / 2)
        arrow = patches.FancyArrowPatch((0, 0), (np.mean(data), 0.3), zorder=2, mutation_scale=30, alpha=alpha,
                                        facecolor=color, edgecolor='black', fill=True, linewidth=0.5)
        ax.add_patch(arrow)

    for i, (d, label, c) in enumerate(zip(datasets, labels, colors)):
        ii = Nplots * i
        for k, (chunk, side) in enumerate(zip(['Lturn', 'Rturn'], ['left', 'right'])):
            b0_par = nam.at(par, nam.start(chunk))
            b1_par = nam.at(par, nam.stop(chunk))
            bd_par = nam.chunk_track(chunk, par)

            b0 = d.get_par(b0_par).dropna().values.flatten()
            b1 = d.get_par(b1_par).dropna().values.flatten()
            db = d.get_par(bd_par).dropna().values.flatten()

            b0 -= ang0
            b1 -= ang0
            if norm:
                b0 %= 360
                b1 = b0 + db
                b0[b0 > 180] -= 360
                b1[b0 > 180] -= 360
            B0 = np.deg2rad(b0[(np.abs(db) > min_angle) & (np.abs(db) < max_angle)])
            B1 = np.deg2rad(b1[(np.abs(db) > min_angle) & (np.abs(db) < max_angle)])
            # DB = np.deg2rad(db[(np.abs(db) > min_angle) & (np.abs(db) < max_angle)])
            if Nplots == 2:
                for tt, BB, aa in zip(['start', 'stop'], [B0, B1], [0.3, 0.6]):
                    circNarrow(axs[ii + k], BB, aa, tt, c)
                axs[ii + 1].legend(bbox_to_anchor=(-0.7, 0.1), loc='center', fontsize=12)
            elif Nplots == 4:
                B00 = B0[B0 < 0]
                B10 = B1[B0 < 0]
                # DB0=DB[B0<0]
                B01 = B0[B0 > 0]
                B11 = B1[B0 > 0]
                # DB1=DB[B0>0]
                for tt, BB, aa in zip([r'$\theta^{init}_{or}$', r'$\theta^{fin}_{or}$'], [(B01, B00), (B11, B10)],
                                      [0.3, 0.6]):
                    for kk, ss, BBB in zip([0, 1], [r'$L_{sided}$', r'$R_{sided}$'], BB):
                        circNarrow(axs[ii + k + 2 * kk], BBB, aa, f'{ss} {tt}', c)
                        axs[ii + 1].legend(bbox_to_anchor=(-0.3, 0.1), loc='center', fontsize=12)
                        axs[ii + 2 + 1].legend(bbox_to_anchor=(-0.3, 0.1), loc='center', fontsize=12)
            if i == Nds - 1:
                if Nplots == 2:
                    axs[ii + k].set_title(f'Bearing due to {side} turn.', y=-0.4)
                    # axs[ii+k].set_title(f'Bearing before and after a {side} turn.', fontsize=12, y=-0.4)
                elif Nplots == 4:
                    axs[ii + k].set_title(fr'$L_{{sided}}$ {side} turn.', y=-0.4)
                    # axs[ii+2*k].set_title(f'Bearing before and after a left-starting {side} turn.', fontsize=12, y=-0.4)
                    axs[ii + 2 + k].set_title(fr'$R_{{sided}}$ {side} turn.', y=-0.4)
                    # axs[ii+2*k+1].set_title(f'Bearing before and after a right-starting {side} turn.', fontsize=12, y=-0.4)
    for ax in axs:
        ax.set_xticklabels([0, '', +90, '', 180, '', -90, ''], fontsize=15)
    dataset_legend(labels, colors, ax=axs[0], loc='upper center', anchor=(0.5, 0.99),
                   bbox_transform=fig.transFigure)
    plt.subplots_adjust(bottom=0.15, top=1 - 0.1, left=0.0, right=1.0, wspace=0.0, hspace=0.35)
    # plt.show()
    # raise
    return process_plot(fig, save_to, filename, return_fig, show)


def plot_turn_Dorient2center(**kwargs):
    return plot_turn_Dbearing(ref_angle=None, par=nam.bearing2('center'), **kwargs)


def plot_chunk_Dorient2source(datasets, labels=None, chunk='stride', source=(0.0, 0.0), Nbins=16, min_dur=0.0,
                              plot_merged=False, save_to=None, return_fig=False, show=False):
    Ndatasets, colors, save_to, labels = plot_config(datasets, labels, save_to, subfolder=chunk)
    filename = f'{chunk}_Dorient2souce.{suf}'
    if plot_merged:
        Ndatasets += 1
        colors.insert(0, 'black')
        labels.insert(0, 'merged')
    Ncols = int(np.ceil(np.sqrt(Ndatasets)))
    Nrows = Ncols - 1 if Ndatasets < Ncols ** 2 - Ncols else Ncols
    fig, axs = plt.subplots(Nrows, Ncols, figsize=(8 * Ncols, 8 * Nrows),
                            subplot_kw=dict(projection='polar'),
                            # sharex=True,
                            sharey=True
                            )

    axs = axs.ravel() if Ndatasets > 1 else [axs]

    chunk_dur = nam.dur(chunk)
    durs = [d.get_par(chunk_dur) for d in datasets]

    temp = 'o_cent' if source == (0, 0) else 'o_chem'
    chunk_k = [k for k in chunk_dict if chunk_dict[k] == chunk][0]
    ks = [f'{chunk_k}_{temp}{i}' for i in [0, 1, '']]
    b0_par, b1_par, db_par = getPar(ks, to_return=['d'])[0]
    try:
        b0s = [d.get_par(b0_par).dropna().values for d in datasets]
        b1s = [d.get_par(b1_par).dropna().values for d in datasets]
        dbs = [d.get_par(db_par).dropna().values for d in datasets]
    except:
        ho = nam.unwrap(nam.orient('front'))
        ho0_par = f'{ho}_at_{chunk}_start'
        ho1_par = f'{ho}_at_{chunk}_stop'

        x0_par = f'x_at_{chunk}_start'
        x1_par = f'x_at_{chunk}_stop'

        y0_par = f'y_at_{chunk}_start'
        y1_par = f'y_at_{chunk}_stop'

        b0s = [fun.compute_bearing2source(d.get_par(x0_par).dropna().values, d.get_par(y0_par).dropna().values,
                                          d.get_par(ho0_par).dropna().values, loc=source, in_deg=True) for d in
               datasets]
        b1s = [fun.compute_bearing2source(d.get_par(x1_par).dropna().values, d.get_par(y1_par).dropna().values,
                                          d.get_par(ho1_par).dropna().values, loc=source, in_deg=True) for d in
               datasets]
        dbs = [np.abs(b0) - np.abs(b1) for b0, b1 in zip(b0s, b1s)]

    if plot_merged:
        b0s.insert(0, np.vstack(b0s))
        b1s.insert(0, np.vstack(b1s))
        dbs.insert(0, np.vstack(dbs))
        durs.insert(0, np.vstack(durs))

    for i, (b0, b1, db, dur, label, c) in enumerate(zip(b0s, b1s, dbs, durs, labels, colors)):
        b0 = b0[dur > min_dur]
        b1 = b1[dur > min_dur]
        db = db[dur > min_dur]
        b0m, b1m = np.mean(b0), np.mean(b1)
        dbm = np.round(np.mean(db), 2)
        if np.isnan([dbm, b0m, b1m]).any():
            continue
        circular_hist(axs[i], b0, bins=Nbins, alpha=0.3, label='start', color=c, offset=np.pi / 2)
        circular_hist(axs[i], b1, bins=Nbins, alpha=0.6, label='stop', color=c, offset=np.pi / 2)
        arrow0 = patches.FancyArrowPatch((0, 0), (np.deg2rad(b0m), 0.3), zorder=2, mutation_scale=30, alpha=0.3,
                                         facecolor=c,
                                         edgecolor='black', fill=True, linewidth=0.5)

        axs[i].add_patch(arrow0)
        arrow1 = patches.FancyArrowPatch((0, 0), (np.deg2rad(b1m), 0.3), zorder=2, mutation_scale=30, alpha=0.6,
                                         facecolor=c,
                                         edgecolor='black', fill=True, linewidth=0.5)
        axs[i].add_patch(arrow1)

        text_x = -0.3
        text_y = 1.2
        axs[i].text(text_x, text_y, f'Dataset : {label}', transform=axs[i].transAxes)
        axs[i].text(text_x, text_y - 0.1, f'Chunk (#) : {chunk} ({len(b0)})', transform=axs[i].transAxes)
        axs[i].text(text_x, text_y - 0.2, f'Min duration : {min_dur} sec', transform=axs[i].transAxes)
        axs[i].text(text_x, text_y - 0.3, fr'Correction $\Delta\theta_{{{"or"}}} : {dbm}^{{{"o"}}}$',
                    transform=axs[i].transAxes)
        axs[i].legend(loc=[0.9, 0.9])
        axs[i].set_title(f'Bearing before and after a {chunk}.', fontsize=15, y=-0.2)
    for ax in axs:
        ticks_loc = ax.get_xticks().tolist()
        ax.xaxis.set_major_locator(FixedLocator(ticks_loc))
        ax.set_xticklabels([0, '', +90, '', 180, '', -90, ''])
    plt.subplots_adjust(bottom=0.2, top=0.8, left=0.05 * Ncols / 2, right=0.9, wspace=0.8, hspace=0.3)
    return process_plot(fig, save_to, filename, return_fig, show)


def circular_hist(ax, x, bins=16, density=True, offset=0, gaps=True, **kwargs):
    """
    Produce a circular histogram of angles on ax.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.PolarAxesSubplot
        axis instance created with subplot_kw=dict(projection='polar').

    x : array
        Angles to plot, expected in units of radians.

    bins : int, optional
        Defines the number of equal-width bins in the range. The default is 16.

    density : bool, optional
        If True plot frequency proportional to area. If False plot frequency
        proportional to radius. The default is True.

    offset : float, optional
        Sets the offset for the location of the 0 direction in units of
        radians. The default is 0.

    gaps : bool, optional
        Whether to allow gaps between bins. When gaps = False the bins are
        forced to partition the entire [-pi, pi] range. The default is True.

    Returns
    -------
    n : array or list of arrays
        The number of values in each bin.

    bins : array
        The edges of the bins.

    patches : `.BarContainer` or list of a single `.Polygon`
        Container of individual artists used to create the histogram
        or list of such containers if there are multiple input datasets.
    """
    # Wrap angles to [-pi, pi)
    x = (x + np.pi) % (2 * np.pi) - np.pi

    # Force bins to partition entire circle
    if not gaps:
        bins = np.linspace(-np.pi, np.pi, num=bins + 1)

    # Bin data and record counts
    n, bins = np.histogram(x, bins=bins)

    # Compute width of each bin
    widths = np.diff(bins)

    # By default plot frequency proportional to area
    if density:
        # Area to assign each bin
        area = n / x.size
        # Calculate corresponding bin radius
        radius = (area / np.pi) ** .5
    # Otherwise plot frequency proportional to radius
    else:
        radius = n

    # Plot data on ax
    patches = ax.bar(bins[:-1], radius, zorder=1, align='edge', width=widths,
                     edgecolor='black', fill=True, linewidth=2, **kwargs)

    # Set the direction of the zero angle
    ax.set_theta_offset(offset)

    # Remove ylabels for area plots (they are mostly obstructive)
    if density:
        ax.set_yticks([])

    return n, bins, patches


def comparative_analysis(datasets, labels=None, simVSexp=False, save_to=None, **kwargs):
    fig_dict = {}
    warnings.filterwarnings('ignore')
    if save_to is None:
        save_to = datasets[0].dir_dict['comp_plot']
    if labels is None:
        labels = [d.id for d in datasets]
    cc = {'datasets': datasets,
          'labels': labels,
          'save_to': save_to}
    for r in ['default']:
        # for r in ['broad', 'default', 'restricted']:
        for m in ['cdf', 'pdf']:
            for f in ['best', 'all']:
                n = f'bout_{m}_fit_{f}_{r}'
                try:
                    fig_dict[n] = plot_stridesNpauses(**cc, plot_fits=f, range=r, only_fit_one=False, mode=m,
                                                      print_fits=False, **kwargs)
                except:
                    pass
    for m in ['minimal', 'limited', 'full']:
        fig_dict[f'endpoint_{m}'] = plot_endpoint_params(**cc, mode=m, **kwargs)
    for m in ['orientation', 'orientation_x2', 'bend', 'spinelength']:
        for agent_idx in [None, 0, 1]:
            i = '' if agent_idx is None else f'_{agent_idx}'
            try:
                fig_dict[f'interference_{m}{i}'] = plot_interference(**cc, mode=m, agent_idx=agent_idx, **kwargs)
            except:
                pass
    for scaled in [True, False]:
        for fig_cols in [1, 2]:
            for r0, r1 in itertools.product([0, 20], [40, 80, 120, 160, 200]):
                s = 'scaled_' if scaled else ''
                l = f'{s}dispersion_{r0}->{r1}_{fig_cols}'
                try:
                    fig_dict[l] = plot_dispersion(**cc, scaled=scaled, fig_cols=fig_cols, ranges=[(r0, r1)], **kwargs)
                except:
                    pass

    try:
        fig_dict['stride_Dbend'] = plot_stride_Dbend(**cc, show_text=False, **kwargs)
    except:
        pass
    try:
        fig_dict['stride_Dorient'] = plot_stride_Dorient(**cc, simVSexp=simVSexp, absolute=True, **kwargs)
    except:
        pass
    try:
        fig_dict['ang_pars'] = plot_ang_pars(**cc, simVSexp=simVSexp, absolute=True, include_turns=False, Npars=3,
                                             **kwargs)
    except:
        pass
    try:
        fig_dict['calibration'] = calibration_plot(save_to=save_to, **kwargs)
    except:
        pass
    fig_dict['crawl_pars'] = plot_crawl_pars(**cc, simVSexp=simVSexp, **kwargs)
    fig_dict['turns'] = plot_turns(**cc, **kwargs)
    fig_dict['turn_duration'] = plot_turn_amp(**cc, **kwargs)
    combine_pdfs(file_dir=save_to)
    return fig_dict


def targeted_analysis(datasets, labels=None, simVSexp=False, save_to=None, pref='', show=False, **kwargs):
    # with fun.suppress_stdout():
    if save_to is None:
        save_to = datasets[0].dir_dict['comp_plot']
    if labels is None:
        labels = [d.id for d in datasets]
    anal_kws = {'datasets': datasets,
                'labels': labels,
                'save_to': save_to,
                'subfolder': None,
                'show': show}
    # init_dir, res_dir = 'init', 'result'
    plot_stridesNpauses(**anal_kws, plot_fits='best', time_unit='sec', range='default', print_fits=False,
                        save_as=f'bouts{pref}.pdf', save_fits_as=f'bout_fits{pref}.csv', **kwargs)
    plot_endpoint_params(**anal_kws, mode='stride_def', save_as=f'stride_pars{pref}.pdf',
                         save_fits_as=f'stride_pars_ttest{pref}.csv', **kwargs)

    plot_interference(**anal_kws, mode='orientation', save_as=f'interference{pref}.pdf', **kwargs)
    plot_crawl_pars(**anal_kws, save_as=f'crawl_pars{pref}.pdf', save_fits_as=f'crawl_pars_ttest{pref}.csv', **kwargs)
    plot_ang_pars(**anal_kws, Npars=3, save_as=f'ang_pars{pref}.pdf', save_fits_as=f'ang_pars_ttest{pref}.csv',
                  **kwargs)
    plot_endpoint_params(**anal_kws, mode='result', save_as=f'results{pref}.pdf', **kwargs)
    plot_endpoint_params(**anal_kws, mode='reorientation', save_as=f'reorientation{pref}.pdf', **kwargs)
    plot_endpoint_params(**anal_kws, mode='tortuosity', save_as=f'tortuosity{pref}.pdf', **kwargs)
    plot_dispersion(**anal_kws, scaled=True, fig_cols=2, ranges=[(0, 80)], ymax=18, save_as=f'dispersion{pref}.pdf',
                    **kwargs)
    plot_marked_strides(**anal_kws, agent_idx=1, slice=[0, 180], save_as=f'sample_tracks{pref}.pdf', **kwargs)


def dual_half_circle(center, radius, angle=0, ax=None, colors=('W', 'k'), **kwargs):
    """
    Add two half circles to the axes *ax* (or the current axes) with the
    specified facecolors *colors* rotated at *angle* (in degrees).
    """
    if ax is None:
        ax = plt.gca()
    theta1, theta2 = angle, angle + 180
    w1 = patches.Wedge(center, radius, theta1, theta2, fc=colors[0], **kwargs)
    w2 = patches.Wedge(center, radius, theta2, theta1, fc=colors[1], **kwargs)
    for wedge in [w1, w2]:
        ax.add_artist(wedge)
    return [w1, w2]


def save_plot(fig, filepath, filename=None):
    fig.savefig(filepath, dpi=300, facecolor=None)
    # print(fig.get_size_inches(), filename)
    # fig.clear()
    plt.close(fig)
    if filename is not None:
        pass
        # print(f'Plot saved as {filename}')


def plot_config(datasets, labels, save_to, subfolder=None):
    if labels is None:
        labels = [d.id for d in datasets]
    Ndatasets = len(datasets)
    if Ndatasets != len(labels):
        raise ValueError(f'Number of labels {len(labels)} does not much number of datasets {Ndatasets}')
    colors = fun.N_colors(Ndatasets)
    if save_to is None:
        save_to = datasets[0].dir_dict['comp_plot']
    if subfolder is not None:
        save_to = f'{save_to}/{subfolder}'
    if not os.path.exists(save_to):
        os.makedirs(save_to)
    return Ndatasets, colors, save_to, labels


def plot_endpoint_scatter(datasets, labels=None, save_to=None, keys=None, return_fig=False, show=False):
    Ndatasets, colors, save_to, labels = plot_config(datasets, labels, save_to)

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
    fig, axs = plt.subplots(Nx, Ny, figsize=(10 * Ny, 10 * Nx))
    if Nx * Ny > 1:
        axs = axs.ravel()
        filename = f'endpoint_scatterplot.{suf}'
    else:
        axs = [axs]
        filename = f'{keys[1]}_vs_{keys[0]}.{suf}'
    # filepath = os.path.join(save_to, filename)
    for i, (p0, p1) in enumerate(pairs):
        ax = axs[i]
        pars, sim_labels, exp_labels, units = getPar([p0, p1], to_return=['d', 's', 's', 'l'])

        v0_all = [d.endpoint_data[pars[0]].values for d in datasets]
        v1_all = [d.endpoint_data[pars[1]].values for d in datasets]
        r0, r1 = 0.9, 1.1
        v0_r = [np.min(np.array(v0_all)) * r0, np.max(np.array(v0_all)) * r1]
        v1_r = [np.min(np.array(v1_all)) * r0, np.max(np.array(v1_all)) * r1]

        for v0, v1, l, c in zip(v0_all, v1_all, labels, colors):
            ax.scatter(v0, v1, color=c, label=l)
        ax.set_title(f'{pars[1]}_vs_{pars[0]}', fontsize=20)
        ax.legend()
        ax.set_xlabel(units[0])
        ax.set_ylabel(units[1])
        ax.set_xlim(v0_r)
        ax.set_ylim(v1_r)
        ax.ticklabel_format(useMathText=True, scilimits=(0, 0))
    return process_plot(fig, save_to, filename, return_fig, show)


def plot_nengo(d, save_to=None):
    if save_to is None:
        save_to = d.plot_dir
    s = d.step_data.xs(d.agent_ids[0], level='AgentID')
    t = np.linspace(0, d.num_ticks * d.dt, d.num_ticks)
    filename = f'nengo.{suf}'
    filepath = os.path.join(save_to, filename)

    pars = [['crawler_activity', 'turner_activity'], ['crawler_activity', 'feeder_motion']]
    labels = [['crawler', 'turner'], ['crawler', 'feeder']]
    colors = [['blue', 'red'], ['blue', 'green']]

    try:
        chunk1 = 'pause'
        pau1s = s.index[s[f'{chunk1}_stop'] == True] * d.dt
        pau0s = s.index[s[f'{chunk1}_start'] == True] * d.dt
        pause = True
    except:
        pause = False
    try:
        chunk2 = 'stride'
        str1s = s.index[s[f'{chunk2}_stop'] == True] * d.dt
        str0s = s.index[s[f'{chunk2}_start'] == True] * d.dt
        stride = True
    except:
        stride = False
    fig, axs = plt.subplots(2, 1, figsize=(20, 5))
    axs = axs.ravel()
    for ax1, (p1, p2), (l1, l2), (c1, c2) in zip(axs, pars, labels, colors):
        # ax1=axs[0]
        ax2 = ax1.twinx()
        ax1.plot(t, s[p1], color=c1, label=l1)
        ax2.plot(t, s[p2], color=c2, label=l2)
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')

        if pause:
            for start, stop in zip(pau0s, pau1s):
                plt.axvspan(start, stop, color='grey', alpha=0.3)
        if stride:
            for start, stop in zip(str0s, str1s):
                plt.axvspan(start, stop, color='blue', alpha=0.3)
    plt.xlabel(r'time $(sec)$')
    save_plot(fig, filepath, filename)


def process_plot(fig, save_to, filename, return_fig, show=False):
    if show:
        plt.show()
    fig.patch.set_visible(False)
    if return_fig:
        return fig, save_to, filename
    else:
        filepath = os.path.join(save_to, filename)
        save_plot(fig, filepath, filename)
        return fig


def barplot(datasets, labels=None, par_shorts=['f_am'], coupled_labels=None, xlabel=None, ylabel=None, save_to=None,
            save_as=None, return_fig=False, show=False, leg_cols=None, **kwargs):
    Ndatasets, colors, save_to, labels = plot_config(datasets, labels, save_to)
    w = 0.15

    if coupled_labels is not None:
        Npairs = len(coupled_labels)
        N = int(Ndatasets / Npairs)
        if leg_cols is None:
            leg_cols = fun.N_colors(N)
        colors = leg_cols * Npairs
        leg_ids = labels[:N]
        ind = np.hstack([np.linspace(0 + i / N, w + i / N, N) for i in range(Npairs)])
        new_ind = ind[::N] + (ind[N - 1] - ind[0]) / N
    else:
        ind = np.arange(0, w * Ndatasets, w)

    pars, sim_labels, exp_labels, units = getPar(par_shorts, to_return=['d', 's', 's', 'l'])

    # Pull the formatting out here
    bar_kwargs = {'width': w, 'color': colors, 'linewidth': 2, 'zorder': 5, 'align': 'center', 'edgecolor': 'black'}
    plot_kwargs = {'linewidth': 2, 'zorder': 5}
    err_kwargs = {'zorder': 20, 'fmt': 'none', 'linewidth': 4, 'ecolor': 'k', 'barsabove': True, 'capsize': 10}

    es = [d.endpoint_data for d in datasets]

    for p, u in zip(pars, units):
        filename = f'{p}.{suf}' if save_as is None else save_as
        values = [e[p] for e in es]
        means = [v.mean() for v in values]
        stds = [v.std() for v in values]
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.p1 = plt.bar(ind, means, **bar_kwargs)
        ax.errs = plt.errorbar(ind, means, yerr=stds, **err_kwargs)

        if not coupled_labels:
            for i, j in itertools.combinations(np.arange(Ndatasets).tolist(), 2):
                st, pv = ttest_ind(values[i], values[j], equal_var=False)
                pv = np.round(pv, 4)
                label_diff(i, j, f'p={pv}', ind, means, ax)
        else:
            for k in range(Npairs):
                i, j = k * N, k * N + 1
                st, pv = ttest_ind(values[i], values[j], equal_var=False)
                if pv <= 0.05:
                    ax.text(ind[i], means[i] + stds[i], '*', ha='center', fontsize=20)
                    # label_diff(i, j, '*', ind, means, ax)

        h = 2 * (np.nanmax(means) + np.nanmax(stds))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
        ax.ticklabel_format(axis='y', useMathText=True, scilimits=(-3, 3), useOffset=True)
        if coupled_labels is None:
            plt.xticks(ind, labels, color='k')
        else:
            plt.xticks(new_ind, coupled_labels, color='k')
            dataset_legend(leg_ids, leg_cols, ax=ax, loc='upper left', handlelength=1, handleheight=1)
        if ylabel is None:
            plt.ylabel(u)
        else:
            plt.ylabel(ylabel)
        try:
            plt.ylim(0, h)
        except:
            ax.set_ylim(ymin=0)
        if xlabel is not None:
            plt.xlabel(xlabel)
        plt.subplots_adjust(hspace=0.05, top=0.95, bottom=0.15, left=0.15, right=0.95)
        return process_plot(fig, save_to, filename, return_fig, show)


def lineplot(datasets, markers, labels=None, par_shorts=['f_am'], coupled_labels=None, xlabel=None, ylabel=None,
             save_to=None, save_as=None, return_fig=False, show=False, leg_cols=None,scale=1.0, **kwargs):
    Ndatasets, colors, save_to, labels = plot_config(datasets, labels, save_to)
    # w = 0.15

    if coupled_labels is not None:
        Npairs = len(coupled_labels)
        N = int(Ndatasets / Npairs)
        if leg_cols is None:
            leg_cols = fun.N_colors(N)
        colors = leg_cols * Npairs
        leg_ids = labels[:N]
        # print(leg_ids)
        ind = np.arange(Npairs)
        # ind = np.hstack([np.linspace(0 + i / N, w + i / N, N) for i in range(Npairs)])
        # new_ind = ind[::N] + (ind[N - 1] - ind[0]) / N
    else:
        ind = np.arange(Ndatasets)

    pars, sim_labels, exp_labels, units = getPar(par_shorts, to_return=['d', 's', 's', 'l'])

    # Pull the formatting out here
    plot_kwargs = {'linewidth': 2, 'zorder': 5}
    err_kwargs = {'zorder': 2, 'fmt': 'none', 'linewidth': 4, 'ecolor': 'k', 'barsabove': True, 'capsize': 10}

    es = [d.endpoint_data for d in datasets]

    for p, u in zip(pars, units):
        filename = f'{p}.{suf}' if save_as is None else save_as

        values = [e[p]*scale for e in es]

        # print(p, values)
        # values = [e[p] * 1000 for e in es]
        means = [v.mean() for v in values]
        stds = [v.std() for v in values]
        fig, ax = plt.subplots(figsize=(8, 7))
        for n, marker in zip(range(N), markers):
            ax.errs = plt.errorbar(ind, means[n::N], yerr=stds[n::N], **err_kwargs)
            ax.p1 = plt.plot(ind, means[n::N], marker=marker, label=leg_ids[n],
                             markeredgecolor='black', markerfacecolor=leg_cols[n], markersize=8, **plot_kwargs)

        if not coupled_labels:
            for i, j in itertools.combinations(np.arange(Ndatasets).tolist(), 2):
                st, pv = ttest_ind(values[i], values[j], equal_var=False)
                pv = np.round(pv, 4)
                label_diff(i, j, f'p={pv}', ind, means, ax)
        else:
            for k in range(Npairs):
                i, j = k * N, k * N + 1
                st, pv = ttest_ind(values[i], values[j], equal_var=False)
                if pv <= 0.05:
                    ax.text(ind[k], np.max([means[i], means[j]]) + np.max([stds[i], stds[j]]), '*', ha='center',
                            fontsize=20)
                    # label_diff(i, j, '*', ind, means, ax)

        h = 2 * (np.nanmax(means) + np.nanmax(stds))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
        ax.ticklabel_format(axis='y', useMathText=True, scilimits=(-3, 3), useOffset=True)
        if coupled_labels is None:
            plt.xticks(ind, labels, color='k')
        else:
            plt.xticks(ind, coupled_labels, color='k')
        plt.legend(loc='upper right')
        if ylabel is None:
            plt.ylabel(u)
        else:
            plt.ylabel(ylabel)
        try:
            plt.ylim(0, h)
        except:
            ax.set_ylim(ymin=0)
        if xlabel is not None:
            plt.xlabel(xlabel)
        plt.subplots_adjust(hspace=0.05, top=0.95, bottom=0.15, left=0.15, right=0.95)
        return process_plot(fig, save_to, filename, return_fig, show)


def label_diff(i, j, text, X, Y, ax):
    x = (X[i] + X[j]) / 2
    y = 1.5 * max(Y[i], Y[j])
    dx = abs(X[i] - X[j])

    props = {'connectionstyle': 'bar', 'arrowstyle': '-', \
             'shrinkA': 20, 'shrinkB': 20, 'linewidth': 2}
    ax.annotate(text, xy=(X[i], y), zorder=10)
    # ax.annotate(text, xy=(X[i], y), zorder=10)
    ax.annotate('', xy=(X[i], y), xytext=(X[j], y), arrowprops=props)


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
    'food intake (barplot)': barplot,
    'deb': plot_debs,
}
