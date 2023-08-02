import numpy as np
from matplotlib import ticker, cm
from scipy.fft import fft, fftfreq

from larvaworld.lib import reg, aux, plot

#
#
# @reg.funcs.graph('fft', required={'ks':['v', 'fov']})
# def plot_fft(dkdict, c, name=f'fft_powerspectrum',palette=None, axx=None,**kwargs):
#     if palette is None:
#         palette = {'v': 'red', 'fov': 'blue'}
#     P = plot.AutoBasePlot(name=name,build_kws={'w': 15, 'h': 12}, **kwargs)
#     if axx is None:
#         axx = P.axs[0].inset_axes([0.64, 0.65, 0.35, 0.34])
#     xf = fftfreq(c.Nticks, c.dt)[:c.Nticks // 2]
#
#     # fvs = np.zeros(c.N) * np.nan
#     # ffovs = np.zeros(c.N) * np.nan
#     # v_ys = np.zeros([c.N, c.Nticks // 2])
#     # fov_ys = np.zeros([c.N, c.Nticks // 2])
#
#     res_v=dkdict['v'].groupby('AgentID').apply(aux.fft_max,dt=c.dt, fr_range=(1.0, 2.5), return_amps=True)
#     res_fov=dkdict['fov'].groupby('AgentID').apply(aux.fft_max,dt=c.dt, fr_range=(0.1, 0.8), return_amps=True)
#
#     fvs = [r[0] for r in res_v.values]
#     ffovs = [r[0] for r in res_fov.values]
#     v_ys = np.array([r[1] for r in res_v.values])
#     fov_ys = np.array([r[1] for r in res_fov.values])
#
#
#     plot.plot_quantiles(v_ys, x=xf, axis=P.axs[0], label='forward speed', color_shading=palette['v'])
#     plot.plot_quantiles(fov_ys, x=xf, axis=P.axs[0], label='angular speed', color_shading=palette['fov'])
#     P.conf_ax(0, ylim=(0, 8), xlim=(0, 3.5), ylab='Amplitude (a.u.)', xlab='Frequency (Hz)',
#               title='Fourier analysis',titlefontsize=25, leg_loc='lower left', yMaxN=5)
#
#     bins = np.linspace(0, 2, 40)
#
#     v_weights = np.ones_like(fvs) / float(len(fvs))
#     fov_weights = np.ones_like(ffovs) / float(len(ffovs))
#     axx.hist(fvs, color=palette['v'], bins=bins, weights=v_weights)
#     axx.hist(ffovs, color=palette['fov'], bins=bins, weights=fov_weights)
#     P.conf_ax(ax=axx, ylab='Probability', xlab='Dominant frequency (Hz)',yMaxN=2,
#               major_ticklabelsize=10,minor_ticklabelsize=10)
#
#     return P.get()

@reg.funcs.graph('fft multi', required={'ks':[ 'v', 'fov']})
def plot_fft_multi(ks=['v', 'fov'],name=f'fft_powerspectrum',axx=None, dataset_colors=False, **kwargs):
    P = plot.AutoPlot(ks=ks,name=name, build_kws={'w': 15, 'h': 12},**kwargs)
    if axx is None:
        axx = P.axs[0].inset_axes([0.64, 0.65, 0.3, 0.25])

    palette = {'v': 'green', 'fov': 'blue'}
    data_palette_new= {k:[aux.mix2colors(c,palette[k]) for c in P.colors] for k in ks}

    fvs = []
    ffovs = []
    v_ys = []
    fov_ys = []
    for i,(l, d, col) in enumerate(P.data_palette) :
        c = d.config
        res_v = P.dkdict[l]['v'].groupby('AgentID').apply(aux.fft_max, dt=c.dt, fr_range=(1.0, 2.5), return_amps=True)
        res_fov = P.dkdict[l]['fov'].groupby('AgentID').apply(aux.fft_max, dt=c.dt, fr_range=(0.1, 0.8), return_amps=True)

        fvs.append([r[0] for r in res_v.values])
        ffovs.append( [r[0] for r in res_fov.values])
        v_ys.append(np.array([r[1] for r in res_v.values]))
        fov_ys.append(np.array([r[1] for r in res_fov.values]))


    for i,(l, d, col) in enumerate(P.data_palette) :
        c = d.config
        xf = fftfreq(c.Nticks, c.dt)[:c.Nticks // 2]


        plot.plot_quantiles(v_ys[i], x=xf, axis=P.axs[0], label='forward speed', color_shading=data_palette_new['v'][i])
        plot.plot_quantiles(fov_ys[i], x=xf, axis=P.axs[0], label='angular speed', color_shading=data_palette_new['fov'][i])
    P.conf_ax(0, ylim=(0, 8), xlim=(0, 3.5), ylab='Amplitude (a.u.)', xlab='Frequency (Hz)',
                  title='Fourier analysis', titlefontsize=25, yMaxN=5)
    P.data_leg(0, loc='lower left',labels=['forward speed','angular speed'],colors=list(palette.values()), fontsize=15)

    bins = np.linspace(0, 2, 40)

    plot.prob_hist(vs=fvs, colors=data_palette_new['v'], labels=P.labels, ax=axx, bins=bins, alpha=0.5)
    plot.prob_hist(vs=ffovs, colors=data_palette_new['fov'], labels=P.labels, ax=axx, bins=bins, alpha=0.5)


    P.conf_ax(ax=axx, ylab='Probability', xlab='Dominant frequency (Hz)', yMaxN=2,
                  major_ticklabelsize=10, minor_ticklabelsize=10)

    if P.Ndatasets > 1:
        P.data_leg(0,loc='lower right', fontsize=15)
    return P.get()


#
# @reg.funcs.graph('powerspectrum', required={'ks':['v', 'fov']})
# def powerspectrum(ks=['v', 'fov'],name=None, thr=0.2, subfolder='powerspectrums', **kwargs):
#     if name is None :
#         name=f'fft_powerspectrum_x{len(ks)}'
#     P = plot.AutoPlot(ks=ks, name=name, subfolder=subfolder, build_kws={'w': 10, 'h': 8}, **kwargs)
#     P.conf_ax(xlab='Frequency in Hertz [Hz]', ylab='Frequency Domain (Spectrum) Magnitude', xlim=(0, 3.5),ylim=(0, 5))
#     from scipy.fft import fft, fftfreq
#     def proc(df, ids,ax, d_col, label) :
#         N = len(df.index.get_level_values('Step').unique())
#         xf = fftfreq(N, 1 / P.fr)[:N // 2]
#         yf0 = np.zeros(N // 2)
#         for id in ids:
#             dc_single = np.nan_to_num(df.xs(id, level='AgentID').values)
#             yf = fft(dc_single)
#             yf = 2.0 / N * np.abs(yf[0:N // 2])
#             yf = 1000 * yf / np.sum(yf)
#             yf = aux.moving_average(yf, n=21)
#             ax.plot(xf, yf, color=d_col, alpha=0.2)
#             yf0 += yf
#         yf0 = 1000 * yf0 / np.sum(yf0)
#         ax.plot(xf, yf0, color=d_col, label=label)
#         ymax = np.max(yf0[xf > thr])
#         xpos = np.argmax(yf0[xf > thr])
#         xmax = xf[xf > thr][xpos]
#         ax.plot(xmax, ymax, color=d_col, marker='o')
#         ax.annotate(np.round(xmax, 2), xy=(xmax, ymax), xytext=(xmax + 0.2, ymax + 0.1), color=d_col, fontsize=25)
#
#     if P.Nks!=2 :
#         raise
#     for k, cmapID in zip(P.ks, ['Greens','Reds']) :
#         cmap=cm.get_cmap(cmapID)
#         dic = P.kkdict[k]
#         p = P.pdict[k]
#         cols=[cmap(i) for i in np.linspace(0.6, 0.9, P.Ndatasets)]
#         for l, d, c in zip(P.labels, P.datasets, cols):
#             proc(df=dic[l], ids=d.config.agent_ids, ax=P.axs[0], d_col=c, label=p.disp)
#
#
#     if P.Ndatasets > 1:
#         P.data_leg(0,loc='upper left', fontsize=15)
#     P.adjust((0.2, 0.95), (0.15, 0.95))
#     return P.get()
