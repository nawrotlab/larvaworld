import itertools
import time

import numpy as np
import pandas as pd
import seaborn as sns


from larvaworld.lib.aux import naming as nam
from larvaworld.lib import reg, aux, plot



@reg.funcs.graph('module hists')
def module_endpoint_hists(mkey='crawler', mode='realistic',e=None, refID=None, Nbins=None, show_median=True, fig=None, axs=None,
                          **kwargs):

    if e is None and refID is not None:
        d = reg.loadRef(refID)
        d.load(step=False)
        e = d.endpoint_data
    if Nbins is None:
        Nbins = int(e.index.values.shape[0] / 10)

    var_mdict = reg.MD.variable_mdict(mkey, mode=mode)
    N = len(list(var_mdict.keys()))

    P = plot.BasePlot(name=f'{mkey}_endpoint_hists', **kwargs)
    P.build(1, N, figsize=(7 * N, 6), sharey=True, fig=fig, axs=axs)

    for i, (k,p) in enumerate(var_mdict.items()):
        # p=d00.args[k]
        vs = e[p.codename]
        P.axs[i].hist(vs.values, bins=Nbins)
        P.conf_ax(i, xlab=p.label, ylab='# larvae' if i == 0 else None, xMaxN=3, xlabelfontsize=18,
                  xticklabelsize=18,yvis=False if i != 0 else True)

        if show_median:
            v_mu = vs.median()
            text = f'{p.symbol} = {np.round(v_mu, 2)}'
            P.axs[i].axvline(v_mu, color='red', alpha=1, linestyle='dashed', linewidth=3)
            P.axs[i].annotate(text, rotation=0, fontsize=15, va='center', ha='left',
                              xy=(0.55, 0.8), xycoords='axes fraction')
    P.adjust((0.2, 0.9), (0.2, 0.9), 0.01)
    return P.get()

@reg.funcs.graph('angular pars')
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
    P = plot.AutoPlot(name='ang_pars', subfolder=subfolder, build_kws={'N':Nps,'Nrows':1, 'wh':8, 'mode':'hist'}, **kwargs)

    P.init_fits(reg.getPar(shorts))
    for i, (k,r) in enumerate(zip(shorts, rs)):
        p=reg.par.kdict[k]
        vs=[reg.par.get(k,d) for d in P.datasets]
        bins, xlim = P.angrange(r, absolute, Nbins)
        P.plot_par(vs=vs, bins=bins, i=i, absolute=absolute, labels=p.disp, alpha=0.8, histtype='step', linewidth=3,
                   pvalues=False, half_circles=half_circles)
        P.conf_ax(i, ylab='probability', yvis=True if i == 0 else False, xlab=p.label, ylim=[0, 0.1], yMaxN=3)
    P.data_leg(0, loc='upper left' if half_circles else 'upper right')
    P.adjust((0.3 / Nps, 0.99), (0.15, 0.95), 0.01)
    return P.get()
# ks=['v', 'a','sv', 'sa', 'b', 'bv', 'ba', 'fov', 'foa']
@reg.funcs.graph('distros')
def plot_distros(name=None,ks=['v', 'a','sv', 'sa', 'b', 'bv', 'ba', 'fov', 'foa'],mode='hist',
                 pvalues=True, half_circles=True,annotation=False,target_only=None, show_ns=False, subfolder='distro', Nbins=100, **kwargs):
    Nps = len(ks)
    if name is None:
        name = f'distros_{mode}_{Nps}'
    legloc = 'upper left' if half_circles else 'upper right'

    P = plot.AutoPlot(name=name, subfolder=subfolder, build_kws={'N':Nps, 'wh':8, 'mode':mode}, **kwargs)
    P.init_fits(reg.getPar(ks))
    palette = dict(zip(P.labels, P.colors))
    Ddata = {}
    ps = reg.getPar(ks)
    lims={}
    parlabs={}
    for sh, par in zip(ks, ps):
        Ddata[par] = {}
        vs = []
        for d, l in zip(P.datasets, P.labels):
            x=d.get_par(par, key='distro').dropna().values
            Ddata[par][l] = x
            vs.append(x)
        if pvalues:
            P.comp_pvalues(vs, par)
        vvs = np.hstack(vs)
        vmin, vmax = np.quantile(vvs, 0.005), np.quantile(vvs, 0.995)
        lims[par]=(vmin, vmax)
        parlabs[par]=reg.par.kdict[sh].l
    for i,(par,dic) in enumerate(Ddata.items()):

        if mode == 'box':
            dfs = []
            for l,x in dic.items():
                df = pd.DataFrame(x, columns=[par])
                df['DatasetID'] = l
                dfs.append(df)
            df0 = pd.concat(dfs)
            kws = {
                'x': "DatasetID",
                'y': par,
                'palette': palette,
                'hue': None,
                'data': df0,
                'ax': P.axs[i],
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
            if annotation:
                try:
                    plot.annotate_plot(show_ns=show_ns, target_only=target_only, **kws)
                except:
                    pass

            P.conf_ax(i, xticklabelrotation=30, ylab=parlabs[par], yMaxN=4, ylim=lims[par]
                      # xvis=False if i < (Nrows - 1) * Ncols else True
                      )


        else:
            vmin, vmax = lims[par]
            bins = np.linspace(vmin, vmax, Nbins)
            dic = {}
            for l, x in dic.items():
                ws = np.ones_like(x) / float(len(x))
            # print(l,k,x.shape)
                dic[l] = {'weights': ws, 'color': palette[l], 'label': l, 'x': x, 'alpha': 0.6}
                P.axs[i].hist(bins=bins,**dic[l])
            # if pvalues:
            #     P.comp_pvalues(vs, par)
            if half_circles:
                P.plot_half_circles(par, i)

            # bins, xlim = P.angrange(r, absolute, Nbins)
            # P.plot_par(vs=vs, nbins=Nbins, i=i, labels=p.disp, alpha=0.8, histtype='step', linewidth=3,
            #            pvalues=False, half_circles=half_circles)
            P.conf_ax(i, ylab='probability',yvis=True if i%P.Ncols == 0 else False,  xlab=parlabs[par], yMaxN=3,xMaxN=5, leg_loc=legloc)
        # P.conf_ax(i, ylab='probability', yvis=True if i == 0 else False, xlab=p.l, yMaxN=3)

    # dataset_legend(P.labels, P.colors, ax=P.axs[0], loc='upper left' if half_circles else 'upper right')
    if mode == 'box':
        P.adjust((0.15, 0.95), (0.15, 0.95), 0.3, 0.01)
    else:
        P.data_leg(0, loc='upper left' if half_circles else 'upper right')
        P.adjust((0.15, 0.95), (0.15, 0.95), 0.01, 0.1)
    return P.get()

@reg.funcs.graph('crawl pars')
def plot_crawl_pars(shorts=['str_N', 'run_tr', 'cum_d'],subfolder='endpoint',name=None, par_legend=False, pvalues=False,type='sns.hist',
                    half_circles=False, kde=True,  **kwargs):
    sns_kws={'kde' : kde, 'stat' : "probability", 'element': "step", 'fill':True, 'multiple' : "layer", 'shrink' :1}
    Nps = len(shorts)
    if name is None:
        name = 'crawl_pars'
    P = plot.AutoPlot(name=name, subfolder=subfolder, build_kws={'N':Nps,'Nrows':1, 'wh':5, 'mode':'hist'}, **kwargs)
    P.init_fits(reg.getPar(shorts))
    for i, k in enumerate(shorts):
        p=reg.par.kdict[k]
        vs=[reg.par.get(k,d) for d in P.datasets]
        P.plot_par(vs=vs, bins='broad', nbins=40, labels=p.disp, i=i, sns_kws = sns_kws,
                   type=type, pvalues=pvalues, half_circles=half_circles, key='end')
        P.conf_ax(i, ylab='probability', yvis=True if i == 0 else False, xlab=p.label, xlim=p.lim, yMaxN=4,
                  leg_loc='upper right' if par_legend else None)
    P.data_leg(0, loc='upper left', fontsize=15)
    P.adjust((0.25 / Nps, 0.99), (0.15, 0.95), 0.01)
    return P.get()

@reg.funcs.graph('turn amplitude VS Y pos')
def plot_turn_amp_VS_Ypos(**kwargs):
    return plot_turn_amp(par_short='tur_y0', mode='scatter', **kwargs)

@reg.funcs.graph('turn duration')
def plot_turn_duration(absolute=True, **kwargs):
    return plot_turn_amp(par_short='tur_t', mode='scatter', absolute=absolute, **kwargs)

def plot_turn_amp(name=None,par_short='tur_t', ref_angle=None, subfolder='turn', mode='hist', cumy=True, absolute=True, **kwargs):
    if name is None:
        nn = 'turn_amp' if ref_angle is None else 'rel_turn_angle'
        name = f'{nn}_VS_{par_short}_{mode}'
    P = plot.Plot(name=name, subfolder=subfolder, **kwargs)
    ypar, ylab, ylim = reg.getPar('tur_fou', to_return=['d', 'l', 'lim'])

    if ref_angle is not None:
        A0 = float(ref_angle)
        p_ref = reg.getPar(['tur_fo0', 'tur_fo1'])
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
    xpar, xlab = reg.getPar(par_short, to_return=['d', 'l'])
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
        P.fig = plot.scatter_hist(xs, ys, P.labels, P.colors, xlabel=xlab, ylabel=ylab, ylim=ylim, cumylabel=cumylab,
                             cumy=cumy)
    return P.get()

@reg.funcs.graph('angular/epoch')
def plot_bout_ang_pars(name=None,absolute=True, include_rear=True, subfolder='turn', **kwargs):
    shorts = ['bv', 'fov', 'rov', 'ba', 'foa', 'roa'] if include_rear else ['bv', 'fov', 'ba', 'foa']
    Nps=len(shorts)
    if name is None:
        name = 'bout_ang_pars'
    P = plot.AutoPlot(name=name, subfolder=subfolder, build_kws={'N':Nps,'Nrows':2, 'wh':7, 'mode':'hist'}, **kwargs)

    ranges = [250, 250, 50, 2000, 2000, 500] if include_rear else [200, 200, 2000, 2000]

    pars, sim_ls, xlabels, disps = reg.getPar(shorts, to_return=['d', 's', 'l', 'd'])
    # Ncols = int(len(pars) / 2)
    chunks = ['stride', 'pause']
    chunk_cols = ['green', 'purple']
    # P = AutoPlot(name='bout_ang_pars', subfolder=subfolder, Nrows=2, Ncols=Ncols, figsize=(Ncols * 7, 14), sharey=True,
    #              **kwargs)
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
    P.conf_ax(ylab='probability', leg_loc='upper left')
    P.adjust((0.1, 0.95), (0.1, 0.9), 0.1, 0.3)
    return P.get()

@reg.funcs.graph('endpoint pars (scatter)')
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
    P = plot.Plot(name=name, subfolder=subfolder, **kwargs)
    P.build(Nx, Ny, figsize=(10 * Ny, 10 * Nx))
    for i, (p0, p1) in enumerate(pairs):
        pars, labs = reg.getPar([p0, p1], to_return=['d', 'l'])

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

@reg.funcs.graph('turn amplitude')
def plot_turns(name=None,absolute=True, subfolder='turn', **kwargs):
    if name is None:
        name = 'turn_amplitude'
    P = plot.Plot(name=name, subfolder=subfolder, **kwargs)
    P.build()
    p, xlab = reg.getPar('tur_fou', to_return=['d', 'l'])
    bins, xlim = P.angrange(150, absolute, 30)
    P.plot_par(p, bins, i=0, absolute=absolute, alpha=1.0, histtype='step')
    P.conf_ax(xlab=xlab, ylab='probability, $P$', xlim=xlim, yMaxN=4, leg_loc='upper right')
    P.adjust((0.25, 0.95), (0.15, 0.92), 0.05, 0.005)
    return P.get()

@reg.funcs.graph('endpoint pars (hist)')
def plot_endpoint_params(name=None,mode='basic', ks=None, subfolder='endpoint',
                         plot_fit=True, nbins=20, use_title=True, **kwargs):
    ylim = [0.0, 0.25]
    nbins = nbins
    l_par = 'l'
    if ks is None:
        dic = {
            'basic': [l_par, 'fsv', 'sv_mu', 'str_sd_mu',
                      'str_tr', 'pau_tr', 'Ltur_tr', 'Rtur_tr',
                      'tor20_mu', 'dsp_0_40_fin', 'b_mu', 'bv_mu'],
            'minimal': [l_par, 'fsv', 'sv_mu', 'str_sd_mu',
                        'cum_t', 'str_tr', 'pau_tr', 'tor5_std',
                        'tor5_mu', 'tor20_mu', 'dsp_0_40_max', 'dsp_0_40_fin',
                        'b_mu', 'bv_mu', 'Ltur_tr', 'Rtur_tr'],
            'tiny': ['fsv', 'sv_mu', 'str_tr', 'pau_tr',
                     'b_mu', 'bv_mu', 'Ltur_tr', 'Rtur_tr'],
            'stride_def': [l_par, 'fsv', 'str_sd_mu', 'str_sd_std'],
            'reorientation': ['str_fo_mu', 'str_fo_std', 'tur_fou_mu', 'tur_fou_std'],
            'tortuosity': ['tor2_mu', 'tor5_mu', 'tor10_mu', 'tor20_mu'],
            'result': ['sv_mu', 'str_tr', 'pau_tr', 'pau_t_mu'],
            'limited': [l_par, 'fsv', 'sv_mu', 'str_sd_mu',
                        'cum_t', 'str_tr', 'pau_tr', 'pau_t_mu',
                        'tor5_mu', 'tor5_std', 'tor20_mu', 'tor20_std',
                        'tor', 'sdsp_mu', 'sdsp_0_40_max', 'sdsp_0_40_fin',
                        'b_mu', 'b_std', 'bv_mu', 'bv_std',
                        'Ltur_tr', 'Rtur_tr', 'Ltur_fou_mu', 'Rtur_fou_mu'],

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
            ks = dic[mode]
        else:
            raise ValueError('Provide parameter shortcuts or define a mode')

    Nks = len(ks)
    Ncols = int(np.ceil(Nks / 3))


    if name is None:
        name = f'endpoint_params_{mode}'
    P = plot.AutoLoadPlot(ks=ks, name=name, subfolder=subfolder, build_kws={'N':Nks, 'Ncols':Ncols, 'w':7, 'h':5, 'mode': 'hist'}, **kwargs)

    def build_df(vs, ax, bins):
        Nvalues = [len(i) for i in vs]
        a = np.empty((np.max(Nvalues), len(vs),)) * np.nan
        for ll in range(len(vs)):
            a[:Nvalues[ll], ll] = vs[ll]
        df = pd.DataFrame(a, columns=P.labels)
        for j, (col, lab) in enumerate(zip(df.columns, P.labels)):

            v = df[[col]].dropna().values
            y, x, patches = ax.hist(v, bins=bins, weights=np.ones_like(v) / float(len(v)),
                                    color=P.colors[j], alpha=0.5)
            if plot_fit:
                x = x[:-1] + (x[1] - x[0]) / 2
                y_smooth = np.polyfit(x, y, 5)
                poly_y = np.poly1d(y_smooth)(x)
                ax.plot(x, poly_y, color=P.colors[j], label=lab, linewidth=3)

    P.init_fits(P.pars)
    for i,k in enumerate(P.ks) :
        dic,p=P.kpdict[k]
        par=p.d
        P.conf_ax(i, ylab='probability' if i % Ncols == 0 else None, xlab=p.label, xlim=p.lim, ylim=ylim,
                  xMaxN=4, yMaxN=4, xMath=True, title=p.disp if use_title else None)
        if p.lim is None or None in p.lim:
            bins = nbins
        else:
            bins = np.linspace(p.lim[0], p.lim[1], nbins)
        ax = P.axs[i]
        vs=[ddic.df.tolist() for l,ddic in dic.items()]
        P.comp_pvalues(vs, par)
        P.plot_half_circles(par, i)


        try :
            build_df(vs, ax, bins)
        except:
            pass
    P.adjust((0.1, 0.97), (0.1, 1 - (0.1 / 2)), 0.1, 0.2 * 2)
    P.data_leg(0, loc='upper right', fontsize=15)
    return P.get()



if __name__ == '__main__':

    ds=[]
    for refID in ['None.100controls','None.150controls' ] :

    # refID = 'None.100controls'
    # refID='None.Sims2019_controls'

        d = reg.loadRef(refID)
        d.load(contour=False,step=True)
        ds.append(d)
    # s, e, c = d.step_data, d.endpoint_data, d.config
    ks=['str_sd_mu','fv', 'v_mu', 'str_d_mu',
                        'cum_t', 'run_tr', 'pau_tr',
                        'tor5_mu', 'tor20_mu', 'dsp_0_40_max', 'dsp_0_40_fin',
                        'b_mu', 'bv_mu']

    # kws={'datasets' : ds, 'show':True, 'mode' : 'minimal'}
    kws={'datasets' : ds, 'show':True, 'ks' : ks}

    t0=time.time()
    # _ = plot_endpoint_params(**kws)
    t1 = time.time()
    _=plot_endpoint_params(**kws)
    t2= time.time()
    print(t1-t0,t2-t1)

