import time

import numpy as np
from matplotlib import ticker, cm

from lib.aux import colsNstr as cNs, data_aux, dictsNlists as dNl
from lib.registry.pars import preg
from lib.plot.aux import plot_quantiles
from lib.plot.base import BasePlot, Plot, AutoPlot, AutoLoadPlot


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

    l, v, fov = preg.getPar(['l', 'v', 'fov'])
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
            s = d.step_data
        except:
            s = d.read(key='step', file='data_h5')
        c = d.config
        _ = plot_fft(s, c, axx=axx, ax=P.axs[0], fig=P.fig, palette={'v': d.color, 'fov': d.color}, return_fig=True)
    return P.get()


def powerspectrum_old(par_shorts=['v', 'fov'], thr=0.2, pars=[], subfolder='powerspectrums', legend_loc='upper left',
                  Nids=None, **kwargs):
    from scipy.fft import fft, fftfreq

    if len(pars) == 0:
        if len(par_shorts) == 0:
            raise ValueError('Either parameter names or shortcuts must be provided')
        else:
            pars, symbols, ylabs, ylims = preg.getPar(par_shorts, to_return=['d', 's', 'l', 'lim'])
    else:
        symbols = pars
        ylabs = pars
        ylims = [None] * len(pars)
    N = len(pars)
    cols = ['grey'] if N == 1 else cNs.N_colors(N)
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
                yf = data_aux.moving_average(yf, n=21)
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
        P.data_leg(0,loc=legend_loc, fontsize=15)
        # dataset_legend(P.labels, P.colors, ax=ax, loc=legend_loc, fontsize=15)
    P.adjust((0.2, 0.95), (0.15, 0.95))
    return P.get()




#
def powerspectrum(ks=['v', 'fov'],name=None, thr=0.2, subfolder='powerspectrums', **kwargs):
    Nks=len(ks)

    if name is None :
        name=f'fft_powerspectrum_x{Nks}'
    P = AutoLoadPlot(ks=ks, name=name, subfolder=subfolder, figsize=(10, 8), **kwargs)
    P.conf_ax(xlab='Frequency in Hertz [Hz]', ylab='Frequency Domain (Spectrum) Magnitude', xlim=(0, 3.5),ylim=(0, 5))
    ax = P.axs[0]
    from scipy.fft import fft, fftfreq
    # Nticks=P.Nticks
    kcols=['Greens','Reds']

    #
    cols=[[cm.get_cmap(kcols[j])(i) for i in np.linspace(0.6, 0.9, P.Ndatasets)]for j,k in enumerate(ks)]
    # elif P.Ndatasets :

    # print(xf.shape)
    # print(P.Nticks, P.datasets[0].Nticks,P.datasets[1].Nticks)


    def proc(df, ids,ax, d_col, label) :
        Nticks = len(df.index.get_level_values('Step').unique())
        xf = fftfreq(Nticks, 1 / P.fr)[:Nticks // 2]
        yf0 = np.zeros(Nticks // 2)
        for id in ids:
            dc_single = df.xs(id, level='AgentID').values
            dc_single = np.nan_to_num(dc_single)
            yf = fft(dc_single)
            yf = 2.0 / Nticks * np.abs(yf[0:Nticks // 2])
            yf = 1000 * yf / np.sum(yf)
            yf = data_aux.moving_average(yf, n=21)
            ax.plot(xf, yf, color=d_col, alpha=0.2)
            yf0 += yf
        yf0 = 1000 * yf0 / np.sum(yf0)
        ax.plot(xf, yf0, color=d_col, label=label)
        ymax = np.max(yf0[xf > thr])
        xpos = np.argmax(yf0[xf > thr])
        xmax = xf[xf > thr][xpos]
        ax.plot(xmax, ymax, color=d_col, marker='o')
        ax.annotate(np.round(xmax, 2), xy=(xmax, ymax), xytext=(xmax + 0.2, ymax + 0.1), color=d_col, fontsize=25)

    for j,k in enumerate(P.ks) :
        dic,p=P.kpdict[k]
        for i,l in enumerate(P.labels) :
            ids=P.datasets[l].config.agent_ids
            df = dic[l].df
            col = cols[j][i]
            proc(df, ids, ax, col, p.disp)


    if P.Ndatasets > 1:
        P.data_leg(0,colors=[ii[0] for ii in cols],loc='upper left', fontsize=15)
    # elif Nks > 1:
    #     ax.legend()
    P.adjust((0.2, 0.95), (0.15, 0.95))
    return P.get()
