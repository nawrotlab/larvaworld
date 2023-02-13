import numpy as np
from matplotlib import ticker, cm
from scipy.fft import fft, fftfreq

from larvaworld.lib import reg, aux, plot



@reg.funcs.graph('fft')
def plot_fft(s, c, palette=None, axx=None, **kwargs):
    if palette is None:
        palette = {'v': 'red', 'fov': 'blue'}
    P = plot.AutoBasePlot(name=f'fft_powerspectrum',build_kws={'w': 15, 'h': 12}, **kwargs)
    if axx is None:
        axx = P.axs[0].inset_axes([0.64, 0.65, 0.35, 0.34])
    xf = fftfreq(c.Nticks, c.dt)[:c.Nticks // 2]

    l, v, fov = reg.getPar(['l', 'v', 'fov'])
    fvs = np.zeros(c.N) * np.nan
    ffovs = np.zeros(c.N) * np.nan
    v_ys = np.zeros([c.N, c.Nticks // 2])
    fov_ys = np.zeros([c.N, c.Nticks // 2])

    for j, id in enumerate(c.agent_ids):
        ss = s.xs(id, level='AgentID')
        fvs[j], v_ys[j, :] = aux.fft_max(ss[v], c.dt, fr_range=(1.0, 2.5), return_amps=True)
        ffovs[j], fov_ys[j, :] = aux.fft_max(ss[fov], c.dt, fr_range=(0.1, 0.8), return_amps=True)
    plot.plot_quantiles(v_ys, from_np=True, x=xf, axis=P.axs[0], label='forward speed', color_shading=palette['v'])
    plot.plot_quantiles(fov_ys, from_np=True, x=xf, axis=P.axs[0], label='angular speed', color_shading=palette['fov'])
    xmax = 3.5
    P.conf_ax(0, ylim=(0, 4), xlim=(0, xmax), ylab='Amplitude', xlab='Frequency (Hz)',
              title='Fourier analysis',titlefontsize=25, leg_loc='lower left', yMaxN=5)

    bins = np.linspace(0, 2, 40)

    v_weights = np.ones_like(fvs) / float(len(fvs))
    fov_weights = np.ones_like(ffovs) / float(len(ffovs))
    axx.hist(fvs, color=palette['v'], bins=bins, weights=v_weights)
    axx.hist(ffovs, color=palette['fov'], bins=bins, weights=fov_weights)
    axx.set_xlabel('Dominant frequency (Hz)')
    axx.set_ylabel('Probability')
    axx.tick_params(axis='both', which='minor', labelsize=10)
    axx.tick_params(axis='both', which='major', labelsize=10)
    axx.yaxis.set_major_locator(ticker.MaxNLocator(2))
    return P.get()

@reg.funcs.graph('fft multi')
def plot_fft_multi(axx=None, dataset_colors=False, **kwargs):
    P = plot.AutoPlot(name=f'fft_powerspectrum', build_kws={'w': 15, 'h': 12},**kwargs)
    if axx is None:
        axx = P.axs[0].inset_axes([0.64, 0.65, 0.3, 0.25])


    for d in P.datasets:
        if dataset_colors :
            palette = {'v': d.color, 'fov': d.color}
        else :
            palette = None
        try:
            s = d.step_data
        except:
            s = d.read(key='step')
        c = d.config
        _ = plot_fft(s, c, axx=axx, axs=P.axs[0], fig=P.fig, palette=palette, return_fig=True)
    return P.get()


def powerspectrum_old(par_shorts=['v', 'fov'], thr=0.2, pars=[], subfolder='powerspectrums', legend_loc='upper left',
                  Nids=None, **kwargs):


    if len(pars) == 0:
        if len(par_shorts) == 0:
            raise ValueError('Either parameter names or shortcuts must be provided')
        else:
            pars, symbols, ylabs, ylims = reg.getPar(par_shorts, to_return=['d', 's', 'l', 'lim'])
    else:
        symbols = pars
        ylabs = pars
        ylims = [None] * len(pars)
    N = len(pars)
    cols = ['grey'] if N == 1 else aux.N_colors(N)
    if N == 1:
        name = f'{pars[0]}'
    elif N == 2:
        name = f'{pars[0]}_VS_{pars[1]}'
    else:
        name = f'{N}_pars'
    P = plot.AutoPlot(name=name, subfolder=subfolder, figsize=(10, 8), **kwargs)

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
                yf = aux.moving_average(yf, n=21)
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




@reg.funcs.graph('powerspectrum')
def powerspectrum(ks=['v', 'fov'],name=None, thr=0.2, subfolder='powerspectrums', **kwargs):
    Nks=len(ks)
    if name is None :
        name=f'fft_powerspectrum_x{Nks}'
    P = plot.AutoLoadPlot(ks=ks, name=name, subfolder=subfolder, figsize=(10, 8), **kwargs)
    P.conf_ax(xlab='Frequency in Hertz [Hz]', ylab='Frequency Domain (Spectrum) Magnitude', xlim=(0, 3.5),ylim=(0, 5))
    ax = P.axs[0]
    from scipy.fft import fft, fftfreq
    kcols=['Greens','Reds']

    cols=[[cm.get_cmap(kcols[j])(i) for i in np.linspace(0.6, 0.9, P.Ndatasets)]for j,k in enumerate(ks)]

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
            yf = aux.moving_average(yf, n=21)
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
    P.adjust((0.2, 0.95), (0.15, 0.95))
    return P.get()
