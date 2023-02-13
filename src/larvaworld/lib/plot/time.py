import numpy as np
from matplotlib import collections as mc

from larvaworld.lib import reg, aux, plot
from larvaworld.lib.process.spatial import get_disp_df, comp_dispersion


@reg.funcs.graph('ethogram')
def plot_ethogram(subfolder='timeplots', **kwargs):
    P = plot.Plot(name='ethogram', subfolder=subfolder,build_kws={'Nrows': 'Ndatasets', 'Ncols': 2, 'sharex': True}, **kwargs)
    P.build()
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

        dic0 = d.chunk_dicts
        # dic0 = d.load_chunk_dicts()
        for j, (id, dic) in enumerate(dic0.items()):
            for k, (n, title) in enumerate(zip(['lin', 'ang'], [r'$\bf{runs & pauses}$', r'$\bf{left & right turns}$'])):
                idx = 2 * i + k
                ax = P.axs[idx]

                # legdic=dNl.AttrDict({'labels': [], 'colors': []})
                for b, bcol in Cbouts[n].items():
                    try :
                        bbs = dic[b] * c.dt
                        # bbs*=c.dt
                        b0s, b1s = bbs[:, 0], bbs[:, 1]

                        lines = [[(b0, j + 1), (b1, j + 1)] for b0, b1 in zip(b0s, b1s)]
                        lc = mc.LineCollection(lines, colors=bcol, linewidths=2)
                        ax.add_collection(lc)

                        # legdic.labels.append(b)
                        # legdic.colors.append(bcol)
                    except:
                        pass
                P.conf_ax(idx, xlab='time $(sec)$' if i == P.Ndatasets - 1 else None,
                          ylab=f'{dlab} Individuals $(idx)$' if k == 0 else None, ylim=(0, c.N + 2),
                          xlim=(0, c.Nticks * d.dt), title=title if i == 0 else None)
                P.data_leg(idx, labels=list(Cbouts[n].keys()), colors=list(Cbouts[n].values()))
    P.adjust((0.1, 0.95), (0.15, 0.92), 0.05, 0.05)
    return P.get()

    #     # N = d.config['N']
    #     # try:
    #     #     s = d.step_data
    #     # except:
    #     #     s = d.read('step')
    #     for k, (n, title) in enumerate(zip(['lin', 'ang'], [r'$\bf{runs & pauses}$', r'$\bf{left & right turns}$'])):
    #         idx = 2 * i + k
    #         ax = P.axs[idx]
    #         P.conf_ax(idx, xlab='time $(sec)$', ylab='Individuals $(idx)$', ylim=(0, c.N + 2),
    #                   xlim=(0, c.Nticks * d.dt), title=title if i == 0 else None)
    #         for b, bcol in Cbouts[n].items():
    #             try :
    #
    #                 for j, (id, dic) in enumerate(dic0.items()):
    #                     bbs = dic[b]*c.dt
    #                     # bbs*=c.dt
    #                     b0s, b1s=bbs[:,0], bbs[:,1]
    #
    #                     lines = [[(b0, j + 1), (b1, j + 1)] for b0, b1 in zip(b0s, b1s)]
    #                     lc = mc.LineCollection(lines, colors=bcol, linewidths=2)
    #                     ax.add_collection(lc)
    #             except:
    #                 pass
    #         P.data_leg(idx, labels=list(Cbouts[n].keys()), colors=list(Cbouts[n].values()))
    # P.adjust((0.1, 0.95), (0.15, 0.92), 0.15, 0.1)
    # return P.get()



@reg.funcs.graph('nengo')
def plot_nengo_network(group=None, probes=None, same_plot=False, subfolder='nengo', **kwargs):
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
    P = plot.Plot(name=name, subfolder=subfolder, **kwargs)
    Nds = P.Ndatasets
    Nids = np.max([len(d.agent_ids) for d in P.datasets])
    Nticks = np.max([d.num_ticks for d in P.datasets])
    dt = np.max([d.dt for d in P.datasets])
    x = np.linspace(0, (Nticks * dt) / 60, Nticks)
    if same_plot:
        Nrows = Nds
        sharey = False
        yMaxN = 8
    else:
        Nrows = N * Nds
        sharey = False
        yMaxN = 3
    P.build(Nrows, Nids, figsize=(Nids * 30, Nrows * 15), sharex=True, sharey=sharey)
    for i, d in enumerate(P.datasets):
        dics = d.load_dicts('nengo')
        for j, dic in enumerate(dics):
            for k, (p, c) in enumerate(zip(probes, Cprobes)):
                Nrow = i if same_plot else i * Nds + k
                idx = j + Nrow * Nids
                y = np.array(dic[p])
                dim = y.shape[1]
                if dim == 1:
                    P.axs[idx].plot(x, y, color=c, label=p)
                else:
                    for jj in range(dim):
                        P.axs[idx].plot(x, y[:, jj], label=f'{p}_{jj}')
                P.conf_ax(idx, xlab=r'time $min$' if Nrow == Nrows - 1 else None, ylab='activity' if j == 0 else None,
                          yticks=[] if j != 0 else None, yticklabels=[] if j != 0 else None, yMaxN=yMaxN,
                          leg_loc='upper right')
    P.adjust((0.1, 0.95), (0.1, 0.95), 0.01, 0.05)
    return P.get()

@reg.funcs.graph('timeplot')
def timeplot(ks=[], pars=[], name=None, same_plot=True, individuals=False, table=None, unit='sec', absolute=True,
             show_legend=True, show_first=False, subfolder='timeplots', legend_loc='upper left', leg_fontsize=15,
             figsize=(7.5, 5),
             **kwargs):
    unit_coefs = {'sec': 1, 'min': 1 / 60, 'hour': 1 / 60 / 60}
    if len(pars) == 0:
        if len(ks) == 0:
            raise ValueError('Either parameter names or shortcuts must be provided')
        else:
            pars, symbols, ylabs, ylims, ylabs0 = reg.getPar(ks, to_return=['d', 's', 'l', 'lim', 'lab'])

    else:
        symbols = pars
        ylabs = pars
        ylabs0 = pars
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
    for p, symbol, ylab0, ylab, ylim, c in zip(pars, symbols, ylabs0, ylabs, ylims, cols):
        # print(p, symbol, ylab0, ylab, ylim, c)
        if ylab0 is not None:
            ylab = ylab0
        P.conf_ax(xlab=f'time, ${unit}$' if table is None else 'timesteps', ylab=ylab, ylim=ylim, yMaxN=4)
        for d, d_col, d_lab in zip(P.datasets, P.colors, P.labels):
            if P.Ndatasets > 1:
                c = d_col
            try:
                if table is not None:
                    dc=d.read(key=p, file='tables_h5')
                else:
                    dc = d.get_par(p, key='step')
                if absolute:
                    dc = dc.abs()
                    # dc = d.read('step')[p]
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
        # dataset_legend(P.labels, P.colors, ax=ax, loc=legend_loc, fontsize=leg_fontsize)
    P.adjust((0.15, 0.95), (0.15, 0.95))
    return P.get()

@reg.funcs.graph('autoplot')
def auto_timeplot(ks,subfolder='timeplots',name=None, unit='sec',show_first=True,individuals=True,**kwargs):
    Nks=len(ks)
    if name is None :
        name=f'timeplot_x{Nks}'
    P = plot.AutoLoadPlot(ks=ks,name=name, subfolder=subfolder,build_kws={'figsize':(15,5*Nks),'Ncols':1,'Nrows':Nks,'sharex':True}, **kwargs)
    x=P.trange(unit)
    for i,k in enumerate(P.ks) :
        dic,p=P.kpdict[k]
        ax=P.axs[i]
        P.conf_ax(i, xlab=f'time, ${unit}$', ylab=p.label, ylim=p.lim, yMaxN=4,xvis=False if i!=Nks-1 else True)
        for j,l in enumerate(P.labels):
            df=dic[l].df
            c=dic[l].col
            if individuals:
                df_m = df.groupby(level='Step').quantile(q=0.5)
                for id in df.index.get_level_values('AgentID').unique():
                    dc_single = df.xs(id, level='AgentID')
                    ax.plot(x, dc_single, color=c, linewidth=1)
                ax.plot(x, df_m, color=c, linewidth=2)
            else:
                plot.plot_quantiles(df=df, x=x, axis=ax, color_shading=c, linewidth=2)
                if show_first:
                    cc = 'red' if P.Ndatasets == 1 else c

                    dc0 = df.xs(df.index.get_level_values('AgentID')[0], level='AgentID')
                    ax.plot(x, dc0, color=cc, linestyle='dashed', linewidth=1)
    P.data_leg(0, loc='lower left')
    P.adjust((0.1, 0.95), (0.15, 0.95))
    P.fig.align_ylabels(P.axs[:])
    return P.get()

@reg.funcs.graph('C odor (real)')
def plot_odor_concentration(**kwargs):
    return timeplot(['c_odor1'], **kwargs)

@reg.funcs.graph('C odor (perceived)')
def plot_sensed_odor_concentration(**kwargs):
    return timeplot(['dc_odor1'], **kwargs)

@reg.funcs.graph('Y pos')
def plot_Y_pos(**kwargs):
    return timeplot(['y'], **kwargs)

@reg.funcs.graph('dispersal')
def plot_dispersion(range=(0, 40), scaled=False, subfolder='dispersion', ymax=None,
                    **kwargs):
    t0, t1 = range
    p0 = f'dispersion_{int(t0)}_{int(t1)}'

    if not scaled :
        par=p0
        ylab = r'dispersal $(mm)$'
    else :
        par=aux.nam.scal(p0)
        ylab = 'scaled dispersal'

    P = plot.AutoPlot(name=par, subfolder=subfolder, **kwargs)
    for d, lab, c in zip(P.datasets, P.labels, P.colors):
        try :
            dic=aux.load_dict(reg.datapath('dsp', d.config.dir))
            df=dic[par]
            reg.vprint(f'Dispersal {par} for dataset {d.id} loaded from stored dictionary')
        except :
            dt = 1 / d.config.fr
            s0, s1 = int(t0 / dt), int(t1 / dt)
            if hasattr(d, 'step_data') and par in d.step_data.columns:
                dsp=d.step_data[par]
                df = get_disp_df(dsp, s0, Nt=s1-s0)
                reg.vprint(f'Dispersal {par} for dataset {d.id} found in step data')
            else :
                try :
                    dsp=d.read(key='step')[par]
                    df = get_disp_df(dsp, s0, Nt=s1 - s0)
                    reg.vprint(f'Dispersal {par} for dataset {d.id} found in step h5 file')
                except :
                    try :
                        dsp = d.read(key='dspNtor')[par]
                        if dsp.dropna().values.shape[0]==0 :
                            raise
                        df = get_disp_df(dsp, s0, Nt=s1 - s0)
                        reg.vprint(f'Dispersal {par} for dataset {d.id} found in dspNtor h5 file')
                    except :
                        d.load()
                        comp_dispersion(d.step_data, d.endpoint_data, d.config, dsp_starts=[t0], dsp_stops=[t1], store=True)
                        dic = aux.load_dict(reg.datapath('dsp', d.config.dir))
                        df = dic[par]
                        reg.vprint(f'Dispersal {par} for dataset {d.id} computed and then loaded from stored dictionary')
        mean=df['median'].values
        ub=df['upper'].values
        lb=df['lower'].values
        trange=df.index.values*d.config.dt

        P.axs[0].fill_between(trange, ub, lb, color=c, alpha=.2)
        P.axs[0].plot(trange, mean, c, label=lab, linewidth=3 if lab != 'experiment' else 8, alpha=1.0)
    P.conf_ax(xlab='time, $sec$', ylab=ylab, xlim=range, ylim=[0, ymax], xMaxN=4, yMaxN=4, leg_loc='upper left', legfontsize=15)
    return P.get()

@reg.funcs.graph('navigation index')
def plot_navigation_index(subfolder='source', **kwargs):
    P = plot.AutoPlot(name='nav_index', subfolder=subfolder, build_kws={'Nrows': 2, 'Ncols': 1, 'w': 20, 'h': 10,'mode':'both'}, **kwargs)

    for d, c, g in zip(P.datasets, P.colors, P.labels):
        dt = 1 / d.fr
        Nticks = P.Nticks
        Nsec = int(Nticks * dt)
        # s = d.read(key='trajectories', file='aux_h5')
        s=d.load_traj(mode='default')[['x', 'y']]

        vxs = []
        vys = []
        for id in d.agent_ids:
            s0 = s.xs(id, level='AgentID').values
            v0 = aux.eudist(s0)/dt
            # v0 = aux.compute_velocity(s0, dt=dt)
            vx = aux.compute_component_velocity(s0, angles=np.zeros(Nticks), dt=dt)
            vy = aux.compute_component_velocity(s0, angles=np.ones(Nticks) * -np.pi / 2, dt=dt)
            vx = np.divide(vx, v0, out=np.zeros_like(v0), where=v0 != 0)
            vy = np.divide(vy, v0, out=np.zeros_like(v0), where=v0 != 0)
            vxs.append(vx)
            vys.append(vy)
        vx0 = np.nanmean(np.array(vxs), axis=0)
        vy0 = np.nanmean(np.array(vys), axis=0)
        P.axs[0].plot(np.linspace(0, Nsec, Nticks), vx0, color=c, label=g)
        P.axs[1].plot(np.linspace(0, Nsec, Nticks), vy0, color=c, label=g)
    P.adjust((0.1, 0.95), (0.2, 0.98), H=0.15)
    P.conf_ax(0, ylab='X index', leg_loc='upper right')
    P.conf_ax(1, xlab='time (sec)', ylab='Y index', xlim=[0, Nsec], ylim=[-1.0, 1.0])
    P.axs[0].axhline(0.0, color='green', alpha=0.5, linestyle='dashed', linewidth=1)
    P.axs[1].axhline(0.0, color='green', alpha=0.5, linestyle='dashed', linewidth=1)
    return P.get()

def plot_pathlength2(scaled=True, unit='mm',subfolder='timeplots', **kwargs):
    t_unit='min'
    ylab0 = 'pathlength'
    if scaled:
        name = f'scaled_{ylab0}'
        ylab = f'scaled {ylab0} $(-)$'
        k='cum_sd'
    else:
        name = f'{ylab0}'
        ylab = f'{ylab0} $({unit})$'
        k = 'cum_d'
    p=reg.par[k]
    P = plot.AutoPlot(ks=[k], name=name, subfolder=subfolder, figsize=(15, 5),**kwargs)
    x = P.trange(t_unit)
    # dic,p=P.kpdict[k]
    P.conf_ax(0, xlab=f'time, ${t_unit}$', ylab=ylab, xlim=(x[0], x[-1]), ylim=[0, None], xMaxN=5, leg_loc='upper left')
    for d, lab, c in zip(P.datasets, P.labels, P.colors):
        df = d.read(key='pathlength', file='aux')[p.d]
        plot.plot_quantiles(df=df, x=x, axis=P.axs[0], color_shading=c, label=lab)

    P.adjust((0.2, 0.95), (0.15, 0.95), 0.05, 0.005)
    return P.get()


@reg.funcs.graph('pathlength')
def plot_pathlength(scaled=True, unit='mm', xlabel=None, **kwargs):
    lab = 'pathlength'
    if scaled:
        name = f'scaled_{lab}'
        ylab = f'scaled {lab} $(-)$'
    else:
        name = f'{lab}'
        ylab = f'{lab} $({unit})$'
    if xlabel is None:
        xlabel = 'time, $min$'
    P = plot.AutoPlot(name=name,figsize=(7, 6), **kwargs)

    p=reg.par.kdict['cum_d']

    x = P.trange()
    for d, lab, c in zip(P.datasets, P.labels, P.colors):
        df = d.get_par(p.d, key='step')
        if not scaled and unit == 'cm':
            from larvaworld.lib.reg import units as ureg
            if p.u == ureg.m:
                df *= 100
        plot.plot_quantiles(df=df, x=x, axis=P.axs[0], color_shading=c, label=lab)

    P.conf_ax(xlab=xlabel, ylab=ylab, xlim=(x[0], x[-1]), ylim=[0, None], xMaxN=5, leg_loc='upper left')
    P.adjust((0.2, 0.95), (0.15, 0.95), 0.05, 0.005)
    return P.get()






