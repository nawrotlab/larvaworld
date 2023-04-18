import numpy as np
from matplotlib import ticker, cm
from scipy.fft import fft, fftfreq

from larvaworld.lib import reg, aux, plot



@reg.funcs.graph('fft')
def plot_fft(s, c, name=f'fft_powerspectrum',palette=None, axx=None, **kwargs):
    if palette is None:
        palette = {'v': 'red', 'fov': 'blue'}
    P = plot.AutoBasePlot(name=name,build_kws={'w': 15, 'h': 12}, **kwargs)
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
    plot.plot_quantiles(v_ys, x=xf, axis=P.axs[0], label='forward speed', color_shading=palette['v'])
    plot.plot_quantiles(fov_ys, x=xf, axis=P.axs[0], label='angular speed', color_shading=palette['fov'])
    P.conf_ax(0, ylim=(0, 4), xlim=(0, 3.5), ylab='Amplitude (a.u.)', xlab='Frequency (Hz)',
              title='Fourier analysis',titlefontsize=25, leg_loc='lower left', yMaxN=5)

    bins = np.linspace(0, 2, 40)

    v_weights = np.ones_like(fvs) / float(len(fvs))
    fov_weights = np.ones_like(ffovs) / float(len(ffovs))
    axx.hist(fvs, color=palette['v'], bins=bins, weights=v_weights)
    axx.hist(ffovs, color=palette['fov'], bins=bins, weights=fov_weights)
    P.conf_ax(ax=axx, ylab='Probability', xlab='Dominant frequency (Hz)',yMaxN=2,
              major_ticklabelsize=10,minor_ticklabelsize=10)

    return P.get()

@reg.funcs.graph('fft multi')
def plot_fft_multi(name=f'fft_powerspectrum',axx=None, dataset_colors=False, **kwargs):
    P = plot.AutoPlot(name=name, build_kws={'w': 15, 'h': 12},**kwargs)
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



@reg.funcs.graph('powerspectrum')
def powerspectrum(ks=['v', 'fov'],name=None, thr=0.2, subfolder='powerspectrums', **kwargs):
    if name is None :
        name=f'fft_powerspectrum_x{len(ks)}'
    P = plot.AutoLoadPlot(ks=ks, name=name, subfolder=subfolder, build_kws={'w': 10, 'h': 8}, **kwargs)
    P.conf_ax(xlab='Frequency in Hertz [Hz]', ylab='Frequency Domain (Spectrum) Magnitude', xlim=(0, 3.5),ylim=(0, 5))
    from scipy.fft import fft, fftfreq
    def proc(df, ids,ax, d_col, label) :
        N = len(df.index.get_level_values('Step').unique())
        xf = fftfreq(N, 1 / P.fr)[:N // 2]
        yf0 = np.zeros(N // 2)
        for id in ids:
            dc_single = np.nan_to_num(df.xs(id, level='AgentID').values)
            yf = fft(dc_single)
            yf = 2.0 / N * np.abs(yf[0:N // 2])
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

    if P.Nks!=2 :
        raise
    for k, cmapID in zip(P.ks, ['Greens','Reds']) :
        cmap=cm.get_cmap(cmapID)
        dic, p = P.kpdict[k]
        cols=[cmap(i) for i in np.linspace(0.6, 0.9, P.Ndatasets)]
        for l, d, c in zip(P.labels, P.datasets, cols):
            proc(df=dic[l].df, ids=d.config.agent_ids, ax=P.axs[0], d_col=c, label=p.disp)


    if P.Ndatasets > 1:
        P.data_leg(0,colors=[ii[0] for ii in cols],loc='upper left', fontsize=15)
    P.adjust((0.2, 0.95), (0.15, 0.95))
    return P.get()
