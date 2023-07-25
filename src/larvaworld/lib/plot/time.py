import numpy as np
from matplotlib import collections as mc

from larvaworld.lib import reg, aux, plot


@reg.funcs.graph('ethogram', required={'dicts':['chunk_dicts']})
def plot_ethogram(subfolder='timeplots', **kwargs):
    P = plot.AutoPlot(name='ethogram', subfolder=subfolder,build_kws={'Nrows': 'Ndatasets', 'Ncols': 2, 'sharex': True}, **kwargs)
    Cbouts = {
        # 'lin': {'stridechain': 'green',
        'lin': {'exec': 'green',
                'pause': 'red',
                'feedchain': 'blue'},
        'ang': {'Lturn': 'cyan',
                'Rturn': 'orange'}

    }
    for i, (d, dlab) in enumerate(zip(P.datasets, P.labels)):
        c=d.config
        d.chunk_dicts = aux.AttrDict(d.read('chunk_dicts'))
        dic0 = d.chunk_dicts
        for j, (id, dic) in enumerate(dic0.items()):
            for k, (n, title) in enumerate(zip(['lin', 'ang'], [r'$\bf{runs & pauses}$', r'$\bf{left & right turns}$'])):
                idx = 2 * i + k
                ax = P.axs[idx]

                for b, bcol in Cbouts[n].items():
                    try :
                        bbs = dic[b] * c.dt
                        b0s, b1s = bbs[:, 0], bbs[:, 1]

                        lines = [[(b0, j + 1), (b1, j + 1)] for b0, b1 in zip(b0s, b1s)]
                        lc = mc.LineCollection(lines, colors=bcol, linewidths=2)
                        ax.add_collection(lc)

                    except:
                        pass
                P.conf_ax(idx, xlab='time $(sec)$' if i == P.Ndatasets - 1 else None,
                          ylab=f'{dlab} Individuals $(idx)$' if k == 0 else None, ylim=(0, c.N + 2),
                          xlim=(0, c.Nticks * d.dt), title=title if i == 0 else None)
                P.data_leg(idx, labels=list(Cbouts[n].keys()), colors=list(Cbouts[n].values()))
    P.adjust((0.1, 0.95), (0.15, 0.92), 0.05, 0.05)
    return P.get()




@reg.funcs.graph('nengo')
def plot_nengo_network(datasets, group=None, probes=None, same_plot=False, subfolder='nengo', **kwargs):
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
    Cprobes = aux.N_colors(N)

    Nds = len(datasets)
    Nids = np.max([len(d.agent_ids) for d in datasets])
    if same_plot:
        Nrows = Nds
        yMaxN = 8
    else:
        Nrows = N * Nds
        yMaxN = 3

    P = plot.AutoPlot(name=name, subfolder=subfolder,datasets=datasets,
                  build_kws={'Nrows': Nrows,'Ncols': Nids, 'sharex': True, 'w' : 30, 'h' : 15}, **kwargs)

    for i, d in enumerate(P.datasets):
        dics = d.load_dicts('nengo')
        for j, dic in enumerate(dics):
            for k, (p, c) in enumerate(zip(probes, Cprobes)):
                Nrow = i if same_plot else i * P.Ndatasets + k
                idx = j + Nrow * Nids
                y = np.array(dic[p])
                dim = y.shape[1]
                if dim == 1:
                    P.axs[idx].plot(P.trange(), y, color=c, label=p)
                else:
                    for jj in range(dim):
                        P.axs[idx].plot(P.trange(), y[:, jj], label=f'{p}_{jj}')
                P.conf_ax(idx, xlab=r'time $min$' if Nrow == Nrows - 1 else None, ylab='activity' if j == 0 else None,
                          yticks=[] if j != 0 else None, yticklabels=[] if j != 0 else None, yMaxN=yMaxN,
                          leg_loc='upper right')
    P.adjust((0.1, 0.95), (0.1, 0.95), 0.01, 0.05)
    return P.get()

@reg.funcs.graph('timeplot', required={'ks':[]})
def timeplot(ks=[], pars=[], name=None, same_plot=True, individuals=False, table=None, unit='sec', absolute=True,
             show_legend=True, show_first=False, subfolder='timeplots', legend_loc='upper left', leg_fontsize=15,
             figsize=(7.5, 5),
             **kwargs):
    unit_coefs = {'sec': 1, 'min': 1 / 60, 'hour': 1 / 60 / 60}
    if len(pars) == 0:
        if len(ks) == 0:
            raise ValueError('Either parameter names or shortcuts must be provided')
        else:
            pars, symbols, ylabs, ylims = reg.getPar(ks, to_return=['d', 'disp', 'l', 'lim'])

    else:
        symbols = pars
        ylabs = pars
        ylims = [None] * len(pars)
    N = len(pars)
    cols = ['grey'] if N == 1 else aux.N_colors(N)
    if not same_plot:
        raise NotImplementedError
    if name is None:
        if N == 1:
            name = f'{pars[0]}'
        elif N == 2:
            name = f'{pars[0]}_VS_{pars[1]}'
        else:
            name = f'{N}_pars'
    P = plot.AutoPlot(name=name, subfolder=subfolder, figsize=figsize, **kwargs)

    ax = P.axs[0]
    counter = 0
    for p, symbol, ylab, ylim, c in zip(pars, symbols, ylabs, ylims, cols):

        P.conf_ax(xlab=f'time, ${unit}$' if table is None else 'timesteps', ylab=ylab, ylim=ylim, yMaxN=4)
        for d, d_col, d_lab in zip(P.datasets, P.colors, P.labels):
            if P.Ndatasets > 1:
                c = d_col
            try:
                dc = d.get_par(p)
                if absolute:
                    dc = dc.abs()
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
                    plot.plot_quantiles(df=dc, x=x, axis=ax, color_shading=c, label=symbol, linewidth=2)
                    if show_first:
                        cc='red' if P.Ndatasets == 1 else c
                        dc0 = dc.xs(dc.index.get_level_values('AgentID')[0], level='AgentID')
                        ax.plot(x, dc0, color=cc, linestyle='dashed', linewidth=1)
                counter += 1
            except:
                pass
    if counter == 0:
        raise ValueError('None of the parameters exist in any dataset')
    if N > 1:
        ax.legend()
    if P.Ndatasets > 1 and show_legend:
        P.data_leg(0, loc=legend_loc, fontsize=leg_fontsize)
    P.adjust((0.15, 0.95), (0.15, 0.95))
    return P.get()


@reg.funcs.graph('timeplots', required={'ks':[]})
def timeplots(ks,subfolder='timeplots',name=None, unit='sec',xlim=None,
              individuals=False,absolute=False,show_first=False,**kwargs):
    Nks=len(ks)
    if name is None :
        name=f'timeplots_x{Nks}'
    P = plot.AutoPlot(name=name, subfolder=subfolder,build_kws={'Nrows':Nks,'sharex':True, 'w' : 15, 'h' : 5}, **kwargs)

    for i, k in enumerate(ks):
        P.plot_quantiles(k=k, idx=i, unit=unit,xvis=True if i==Nks-1 else False,xlim=xlim,
                         individuals=individuals,absolute=absolute, show_first=show_first)
    P.adjust((0.1, 0.95), (0.15, 0.95), H=0.05)
    P.fig.align_ylabels(P.axs[:])
    return P.get()


@reg.funcs.graph('navigation index', required={'traj':['default']})
def plot_navigation_index(subfolder='source', **kwargs):
    P = plot.AutoPlot(name='nav_index', subfolder=subfolder, build_kws={'Nrows': 2, 'w': 20, 'h': 10,'sharex':True, 'sharey':True}, **kwargs)

    for l, d, c in P.data_palette:
        dt = 1 / d.fr
        Nticks = P.Nticks
        Nsec = int(Nticks * dt)
        s=d.load_traj(mode='default')

        vxs = []
        vys = []
        for id in d.agent_ids:
            s0 = s.xs(id, level='AgentID').values
            dxy = np.diff(s0, axis=0)
            rads = np.arctan2(dxy[:, 1], dxy[:, 0])
            rads = np.insert(rads, 0, 0)
            vxs.append(np.cos(rads))
            vys.append(-np.sin(rads))
        vx0 = np.nanmean(np.array(vxs), axis=0)
        vy0 = np.nanmean(np.array(vys), axis=0)
        P.axs[0].plot(np.linspace(0, Nsec, Nticks), vx0, color=c, label=l)
        P.axs[1].plot(np.linspace(0, Nsec, Nticks), vy0, color=c, label=l)
    P.adjust((0.1, 0.95), (0.2, 0.98), H=0.15)
    P.conf_ax(0, ylab='X index', leg_loc='upper right')
    P.conf_ax(1, xlab='time (sec)', ylab='Y index', xlim=[0, Nsec], ylim=[-1.0, 1.0])
    P.axs[0].axhline(0.0, color='green', alpha=0.5, linestyle='dashed', linewidth=1)
    P.axs[1].axhline(0.0, color='green', alpha=0.5, linestyle='dashed', linewidth=1)
    return P.get()


@reg.funcs.graph('pathlength', required={'ks':['cum_d','cum_sd']})
def plot_pathlength(scaled=False, **kwargs):
    k = 'cum_sd' if scaled else 'cum_d'
    return timeplots(ks=[k], **kwargs)


@reg.funcs.graph('dispersal')
def plot_dispersal(range=(0, 40), scaled=False, **kwargs):
    t0, t1 = range
    k = f'dsp_{int(t0)}_{int(t1)}'
    if scaled:
        k=f's{k}'
    return timeplots(name=reg.getPar(k),ks=[k],xlim=range, **kwargs)




