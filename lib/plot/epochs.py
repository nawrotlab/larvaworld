import os

import numpy as np
from matplotlib import pyplot as plt, ticker

from lib.aux import naming as nam
from lib.conf.pars.pars import getPar
from lib.plot.aux import dataset_legend, plot_quantiles, suf
from lib.plot.base import AutoPlot, BasePlot, Plot
from lib.process.aux import comp_pooled_epochs


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

#
# def plot_stridechains(dataset, save_to=None):
#     d = dataset
#
#     if save_to is None:
#         save_to = os.path.join(d.plot_dir, 'plot_stridechains')
#     if not os.path.exists(save_to):
#         os.makedirs(save_to)
#     filepath_MLE = os.path.join(save_to, f'stridechain_distribution_MLE.{suf}')
#     filepath_r = os.path.join(save_to, f'stridechain_distribution_r.{suf}')
#
#     s = d.step_data[nam.length(nam.chain('stride'))].dropna()
#     u, c = np.unique(s, return_counts=True)
#     c = c / np.sum(c)
#     c = 1 - np.cumsum(c)
#
#     alpha = 1 + len(s) / np.sum(np.log(s))
#     beta = len(s) / np.sum(s - 1)
#     mu = np.mean(np.log(s))
#     sigma = np.std(np.log(s))
#
#     fig, axs = plt.subplots(1, 1, figsize=(10, 10), sharex=True, sharey=True)
#     fig.suptitle('Stridechain distribution', fontsize=25)
#
#     axs.loglog(u, c, 'or', label='stridechains')
#     axs.loglog(u, 1 - ParDict.dist_dict['powerlaw']['cdf'](u, 1, alpha), 'r', lw=2, label='powerlaw MLE')
#     axs.loglog(u, 1 - ParDict.dist_dict['exponential']['cdf'](u, 1, beta), 'g', lw=2, label='exponential MLE')
#     axs.loglog(u, 1 - ParDict.dist_dict['lognormal']['cdf'](u, mu, sigma), 'b', lw=2, label='lognormal MLE')
#
#     axs.legend(loc='lower left', fontsize=15)
#     axs.axis([1, np.max(s), 10 ** -4.0, 10 ** 0])
#     plt.xlabel(r'Stridechain  length, $l$', fontsize=15)
#     plt.ylabel(r'Probability Density, $P_l$', fontsize=15)
#
#     fig.savefig(filepath_MLE, dpi=300)
#     print(f'Image saved as {filepath_MLE}')
#
#     fig, axs = plt.subplots(1, 1, figsize=(10, 10), sharex=True, sharey=True)
#     fig.suptitle('Stridechain distribution', fontsize=25)
#
#     axs.loglog(u, c, 'or', label='stridechains')
#
#     for r in np.round(np.arange(0.8, 1, 0.025), 3):
#         x = np.arange(1, np.max(s), 1)
#         y = (1 - r) * r ** (x - 1)
#         y = 1 - np.cumsum(y)
#         plt.plot(x, y)
#         plt.loglog(x, y, label=r)
#
#     axs.legend(loc='lower left', fontsize=15)
#     axs.axis([1, np.max(s), 10 ** -4.0, 10 ** 0])
#
#     plt.xlabel(r'Stridechain  length, $l$', fontsize=15)
#     plt.ylabel(r'Probability Density, $P_l$', fontsize=15)
#
#     fig.savefig(filepath_r, dpi=300)
#     print(f'Image saved as {filepath_r}')

#
# def plot_bend_pauses(dataset, save_to=None):
#     from lib.anal.fitting import compute_density, powerlaw_cdf, exponential_cdf, lognorm_cdf
#     d = dataset
#     if save_to is None:
#         save_to = os.path.join(d.plot_dir, 'plot_bend_pauses')
#     if not os.path.exists(save_to):
#         os.makedirs(save_to)
#     filepath = os.path.join(save_to, f'bend_pause_distribution.{suf}')
#
#     s = d.step_data[nam.dur('bend_pause')].dropna()
#     durmin, durmax = np.min(s), np.max(s)
#     u, uu, c, ccum = compute_density(s, durmin, durmax)
#     alpha = 1 + len(s) / np.sum(np.log(s / durmin))
#     beta = len(s) / np.sum(s - durmin)
#     mu = np.mean(np.log(s))
#     sigma = np.std(np.log(s))
#
#     fig, axs = plt.subplots(1, 1, figsize=(10, 10), sharex=True, sharey=True)
#     fig.suptitle('Bend-pause distribution', fontsize=25)
#
#     axs.loglog(u, ccum, 'or', label='bend_pauses')
#     axs.loglog(u, 1 - powerlaw_cdf(u, durmin, alpha), 'r', lw=2, label='powerlaw MLE')
#     axs.loglog(u, 1 - exponential_cdf(u, durmin, beta), 'g', lw=2, label='exponential MLE')
#     axs.loglog(u, 1 - lognorm_cdf(u, mu, sigma), 'b', lw=2, label='lognormal MLE')
#
#     axs.legend(loc='lower left', fontsize=15)
#     axs.axis([durmin, durmax, 10 ** -4.0, 10 ** 0])
#
#     plt.xlabel(r'Bend pause duration, $(sec)$', fontsize=15)
#     plt.ylabel(r'Probability Density, $P_d$', fontsize=15)
#
#     fig.savefig(filepath, dpi=300)
#     print(f'Image saved as {filepath}')
#

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


