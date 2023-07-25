import itertools
import time

import numpy as np
import pandas as pd
import seaborn as sns


from larvaworld.lib.aux import nam
from larvaworld.lib import reg, aux, plot



@reg.funcs.graph('module hists')
def module_endpoint_hists(mkey='crawler', mode='realistic',e=None, refID=None, Nbins=None, show_median=True, **kwargs):

    if e is None and refID is not None:
        d = reg.loadRef(refID)
        d.load(step=False)
        e = d.endpoint_data
    if Nbins is None:
        Nbins = int(e.index.values.shape[0] / 10)

    var_mdict = reg.model.variable_mdict(mkey, mode=mode)
    N = len(list(var_mdict.keys()))

    P = plot.AutoBasePlot(name=f'{mkey}_endpoint_hists',build_kws={'Ncols':N,'w':7, 'h':6, 'sharey': True}, **kwargs)

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

@reg.funcs.graph('angular pars', required={'ks':['b', 'bv', 'ba', 'fov', 'foa','rov', 'roa']})
def plot_ang_pars(absolute=False, include_rear=False, name='ang_pars', half_circles=True,kde=True, subfolder='turn',
                  Npars=5, Nbins=100,type='sns.hist', **kwargs):
    if Npars == 5:
        ks = ['b', 'bv', 'ba', 'fov', 'foa']
        rs = [100, 200, 2000, 200, 2000]
    elif Npars == 3:
        ks = ['b', 'bv', 'fov']
        rs = [100, 200, 200]
    else:
        raise ValueError('3 or 5 pars allowed')

    if include_rear:
        ks += ['rov', 'roa']
        rs += [200, 2000]

    P = plot.AutoPlot(ks=ks,ranges=rs, absolute=absolute, name=name, subfolder=subfolder, build_kws={'Ncols':'Nks', 'wh':8, 'sharey': True}, **kwargs)
    P.plot_hist(type=type, kde=kde, half_circles=half_circles, alpha=0.8, linewidth=3,nbins=Nbins)
    P.adjust((0.1, 0.95), (0.15, 0.95), 0.01)
    return P.get()

# ks=['v', 'a','sv', 'sa', 'b', 'bv', 'ba', 'fov', 'foa']
@reg.funcs.graph('distros', required={'ks':[]})
def plot_distros(name=None,ks=['v', 'a','sv', 'sa', 'b', 'bv', 'ba', 'fov', 'foa'],mode='hist',
                 half_circles=True,annotation=False,target_only=None, show_ns=False, subfolder='distro', Nbins=100, **kwargs):
    Nps = len(ks)
    if name is None:
        name = f'distros_{mode}_{Nps}'
    legloc = 'upper left' if half_circles else 'upper right'
    build_kws = {'N': 'Nks', 'wh': 8}
    if mode == 'box':
        build_kws['sharex']=True
    elif mode == 'hist':
        build_kws['sharey']=True
    P = plot.AutoPlot(ks=ks, name=name, subfolder=subfolder, build_kws=build_kws, **kwargs)
    palette = dict(zip(P.labels, P.colors))
    Ddata = {}
    lims={}
    parlabs={}
    for sh, par in zip(ks, P.pars):
        Ddata[par] = {}
        vs = []
        for d, l in zip(P.datasets, P.labels):
            x=d.get_par(par).dropna().values
            Ddata[par][l] = x
            vs.append(x)
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
                dic[l] = {'weights': ws, 'color': palette[l], 'label': l, 'x': x, 'alpha': 0.6}
                P.axs[i].hist(bins=bins,**dic[l])
            P.conf_ax(i, ylab='probability',yvis=True if i%P.Ncols == 0 else False,  xlab=parlabs[par], yMaxN=3,xMaxN=5, leg_loc=legloc)

    if mode == 'box':
        P.adjust((0.15, 0.95), (0.15, 0.95), 0.3, 0.01)
    else:
        P.data_leg(0, loc='upper left' if half_circles else 'upper right')
        P.adjust((0.15, 0.95), (0.15, 0.95), 0.01, 0.1)
    return P.get()

@reg.funcs.graph('crawl pars', required={'ks':['str_N', 'run_tr', 'cum_sd']})
def plot_crawl_pars(ks=['str_N', 'run_tr', 'cum_sd'],subfolder='endpoint',name='crawl_pars',
                    type='sns.hist',kde=True,  **kwargs):
    P = plot.AutoPlot(ks=ks,key='end',name=name, subfolder=subfolder, build_kws={'Ncols':'Nks', 'wh':7, 'sharey': True}, **kwargs)
    P.plot_hist(type=type,kde=kde)
    P.adjust((0.1, 0.95), (0.15, 0.95), 0.1)
    return P.get()

@reg.funcs.graph('turn amplitude VS Y pos', required={'ks':['tur_y0']})
def plot_turn_amp_VS_Ypos(**kwargs):
    return plot_turn_amp(k='tur_y0', **kwargs)

@reg.funcs.graph('turn duration', required={'ks':['tur_t']})
def plot_turn_duration(absolute=True, **kwargs):
    return plot_turn_amp(k='tur_t', absolute=absolute, **kwargs)

def plot_turn_amp(name=None,k='tur_t', ref_angle=None, subfolder='turn', absolute=True, **kwargs):
    if name is None:
        nn = 'turn_amp' if ref_angle is None else 'rel_turn_angle'
        name = f'{nn}_VS_{k}_scatter'


    P = plot.AutoPlot(name=name, subfolder=subfolder, **kwargs)
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
    xpar, xlab = reg.getPar(k, to_return=['d', 'l'])
    xs = [d.get_par(xpar).dropna().values.flatten() for d in P.datasets]

    ax = P.axs[0]
    for x, y, l, c in zip(xs, ys, P.labels, P.colors):
        ax.scatter(x=x, y=y, marker='o', s=5.0, color=c, alpha=0.5)
        m, k = np.polyfit(x, y, 1)
        ax.plot(x, m * x + k, linewidth=4, color=c, label=l)
        P.conf_ax(xlab=xlab, ylab=ylab, ylim=ylim, yMaxN=4, leg_loc='upper left')
        P.adjust((0.15, 0.95), (0.1, 0.95), 0.01)
    return P.get()

@reg.funcs.graph('angular/epoch', required={'ks':['bv', 'fov', 'rov', 'ba', 'foa', 'roa']})
def plot_bout_ang_pars(name='bout_ang_pars',absolute=True, include_rear=True, subfolder='turn', **kwargs):
    ks = ['bv', 'fov', 'rov', 'ba', 'foa', 'roa'] if include_rear else ['bv', 'fov', 'ba', 'foa']
    Nps=len(ks)
    P = plot.AutoPlot(name=name, subfolder=subfolder, build_kws={'N':Nps,'Nrows':2, 'wh':7, 'sharey': True}, **kwargs)
    ranges = [250, 250, 50, 2000, 2000, 500] if include_rear else [200, 200, 2000, 2000]
    chunks = ['run', 'pause']
    chunk_cols = ['green', 'purple']
    for i, k in enumerate(ks):
        p=reg.par.kdict[k]
        r=ranges[i]
        xlim = (r0, r1) = (0, r) if absolute else (-r, r)
        bins = np.linspace(r0, r1, 200)

        P.conf_ax(i, xlab=p.disp, xlim=xlim, yMaxN=3)
        for d, l in zip(P.datasets, P.labels):
            vs = [d.get_chunk_par(chunk=c, par=p.d) for c in chunks]
            if absolute:
                vs = [np.abs(v) for v in vs]
            plot.prob_hist(vs, chunk_cols, chunks, ax=P.axs[i], bins=bins,alpha=1.0, histtype='step', linewidth=2)

            # vs = []
            # for c, col in zip(chunks, chunk_cols):
            #     v=d.get_chunk_par(chunk=c,par=p)
            #     v = d.step_data.dropna(subset=[nam.id(c)])[p].values
                # if absolute:
                #     v = np.abs(v)
                # vs.append(v)
                # ax.hist(v, color=col, bins=bins, label=c, weights=np.ones_like(v) / float(len(v)),
                #         alpha=1.0, histtype='step', linewidth=2)
            # if P.Ndatasets>1:
            #     P.comp_pvalue(l, vs[0], vs[1], p)
            #     P.plot_half_circle(p, ax, col1=chunk_cols[0], col2=chunk_cols[1], v=P.fit_df[p].loc[l], ind=l)


    P.conf_ax(0, ylab='probability', ylim=[0, 0.04], leg_loc='upper left')
    P.conf_ax(ylab='probability', leg_loc='upper left')
    P.adjust((0.1, 0.95), (0.1, 0.9), 0.1, 0.3)
    return P.get()

@reg.funcs.graph('endpoint pars (scatter)', required={'ks':[]})
def plot_endpoint_scatter(subfolder='endpoint', ks=None, **kwargs):
    pairs = list(itertools.combinations(ks, 2))
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
        name = f'{ks[1]}_vs_{ks[0]}'
    P = plot.AutoPlot(name=name, subfolder=subfolder,build_kws={'Nrows': Nx,'Ncols': Ny, 'wh' : 10}, **kwargs)
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

@reg.funcs.graph('turn amplitude', required={'ks':['tur_fou']})
def plot_turns(name='turn_amplitude',absolute=False, subfolder='turn', **kwargs):
    P = plot.AutoPlot(ks=['tur_fou'],ranges=[100],absolute=absolute, rad2deg=True, name=name, subfolder=subfolder, **kwargs)
    P.plot_hist(par_legend=True, nbins=30, alpha=1.0, histtype='step')
    P.adjust((0.25, 0.95), (0.15, 0.92), 0.05, 0.005)
    return P.get()

@reg.funcs.graph('endpoint hist', required={'ks':[]})
def plot_endpoint_hist(**kwargs):
    return plot_endpoint_params(type='hist',**kwargs)

@reg.funcs.graph('endpoint box', required={'ks':[]})
def plot_endpoint_box(**kwargs):
    return plot_endpoint_params(type='box',**kwargs)

def plot_endpoint_params(type, name=None,mode='basic', ks=None,Ncols=None, subfolder='endpoint',
                         **kwargs):
    ks=plot.define_end_ks(ks, mode)
    if name is None:
        name = f'endpoint_{type}_{mode}'
    if type=='hist' :
        sharex,sharey=False,True
        W,H=0.1, 0.3
    elif type=='box' :
        sharex,sharey=True,False
        W, H = 0.5, 0.15
    P = plot.AutoPlot(ks=ks,key='end', name=name, subfolder=subfolder,
                          build_kws={'N':'Nks','Ncols':Ncols, 'wh':7, 'sharex': sharex, 'sharey': sharey}, **kwargs)
    if type == 'hist':
        P.plot_hist(nbins=20)
    elif type == 'box':
        P.boxplots()
    P.conf_fig(align=True, adjust_kws={'LR': (0.1, 0.95), 'BT': (0.15, 0.9), 'W': W, 'H': H})
    return P.get()


