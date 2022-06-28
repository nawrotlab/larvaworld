import numpy as np
from matplotlib import collections as mc

from lib.aux import naming as nam, colsNstr as cNs, dictsNlists as dNl
from lib.registry.pars import preg

from lib.plot.base import Plot, AutoPlot, AutoLoadPlot
from lib.plot.aux import plot_quantiles


def plot_ethogram(subfolder='timeplots', **kwargs):
    P = Plot(name='ethogram', subfolder=subfolder, **kwargs)
    P.build(P.Ndatasets, 2, sharex=True)
    Cbouts = {
        'lin': {'stridechain': 'green',
                'pause': 'red',
                'feedchain': 'blue'},
        'ang': {'Lturn': 'cyan',
                'Rturn': 'orange'}

    }
    for i, d in enumerate(P.datasets):
        N = d.config['N']
        try:
            s = d.step_data
        except:
            s = d.read('step')
        for k, (n, title) in enumerate(zip(['lin', 'ang'], [r'$\bf{runs & pauses}$', r'$\bf{left & right turns}$'])):
            idx = 2 * i + k
            ax = P.axs[idx]
            P.conf_ax(idx, xlab='time $(sec)$', ylab='Individuals $(idx)$', ylim=(0, N + 2),
                      xlim=(0, d.config['Nticks'] * d.dt), title=title if i == 0 else None)
            for b, c in Cbouts[n].items():
                bp0, bp1 = nam.start(b), nam.stop(b)
                if not {bp0, bp1}.issubset(s.columns.values):
                    continue
                for j, id in enumerate(s.index.unique('AgentID').values):
                    bbs = s[[bp0, bp1]].xs(id, level='AgentID')
                    b0s = bbs[bp0].dropna().index.values * d.dt
                    b1s = bbs[bp1].dropna().index.values * d.dt
                    lines = [[(b0, j + 1), (b1, j + 1)] for b0, b1 in zip(b0s, b1s)]
                    lc = mc.LineCollection(lines, colors=c, linewidths=2)
                    ax.add_collection(lc)
            P.data_leg(idx, labels=list(Cbouts[n].keys()), colors=list(Cbouts[n].values()))
            # dataset_legend(labels=list(Cbouts[n].keys()), colors=list(Cbouts[n].values()), ax=ax,
            #                loc=None, anchor=None, fontsize=None, handlelength=0.5, handleheight=0.5)

    P.adjust((0.1, 0.95), (0.15, 0.92), 0.15, 0.1)
    return P.get()




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
    Cprobes = cNs.N_colors(N)
    P = Plot(name=name, subfolder=subfolder, **kwargs)
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


def timeplot(par_shorts=[], pars=[], same_plot=True, individuals=False, table=None, unit='sec', absolute=True,
             show_legend=True, show_first=False, subfolder='timeplots', legend_loc='upper left', leg_fontsize=15,
             figsize=(7.5, 5),
             **kwargs):
    unit_coefs = {'sec': 1, 'min': 1 / 60, 'hour': 1 / 60 / 60}
    if len(pars) == 0:
        if len(par_shorts) == 0:
            raise ValueError('Either parameter names or shortcuts must be provided')
        else:
            pars, symbols, ylabs, ylims, ylabs0 = preg.getPar(par_shorts, to_return=['d', 's', 'l', 'lim', 'lab'])

    else:
        symbols = pars
        ylabs = pars
        ylabs0 = pars
        ylims = [None] * len(pars)
    N = len(pars)
    cols = ['grey'] if N == 1 else cNs.N_colors(N)
    if not same_plot:
        raise NotImplementedError
    if N == 1:
        name = f'{pars[0]}'
    elif N == 2:
        name = f'{pars[0]}_VS_{pars[1]}'
    else:
        name = f'{N}_pars'
    P = AutoPlot(name=name, subfolder=subfolder, figsize=figsize, **kwargs)

    ax = P.axs[0]
    counter = 0
    for p, symbol, ylab0, ylab, ylim, c in zip(pars, symbols, ylabs0, ylabs, ylims, cols):
        if ylab0 is not None:
            ylab = ylab0
        P.conf_ax(xlab=f'time, ${unit}$' if table is None else 'timesteps', ylab=ylab, ylim=ylim, yMaxN=4)
        for d, d_col, d_lab in zip(P.datasets, P.colors, P.labels):
            if P.Ndatasets > 1:
                c = d_col
            try:
                if table is not None:
                    dc = d.load_table(table)[p]
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
                    plot_quantiles(df=dc, x=x, axis=ax, color_shading=c, label=symbol, linewidth=2)
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

def auto_timeplot(ks,subfolder='timeplots',name=None, unit='sec',show_first=True,individuals=True,**kwargs):
    Nks=len(ks)
    if name is None :
        name=f'timeplot_x{Nks}'
    P = AutoLoadPlot(ks=ks,name=name, subfolder=subfolder, figsize=(15,5*Nks),Ncols=1,Nrows=Nks,sharex=True, **kwargs)
    x=P.trange(unit)
    for i,k in enumerate(P.ks) :
        dic,p=P.kpdict[k]
    # for i,(k,(dic,p)) in enumerate(P.kpdict.items()) :
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
                plot_quantiles(df=df, x=x, axis=ax, color_shading=c, linewidth=2)
                if show_first:
                    cc = 'red' if P.Ndatasets == 1 else c

                    dc0 = df.xs(df.index.get_level_values('AgentID')[0], level='AgentID')
                    ax.plot(x, dc0, color=cc, linestyle='dashed', linewidth=1)
    P.data_leg(0)
    P.adjust((0.1, 0.95), (0.15, 0.95))
    P.fig.align_ylabels(P.axs[:])
    return P.get()


def plot_odor_concentration(**kwargs):
    return timeplot(['c_odor1'], **kwargs)


def plot_sensed_odor_concentration(**kwargs):
    return timeplot(['dc_odor1'], **kwargs)


def plot_Y_pos(**kwargs):
    return timeplot(['y'], **kwargs)


def plot_dispersion(range=(0, 40), scaled=False, subfolder='dispersion', fig_cols=1, ymax=None,
                    **kwargs):
    from lib.process.store import get_dsp
    ylab = 'scaled dispersal' if scaled else r'dispersal $(mm)$'
    r0, r1 = range
    par = f'dispersion_{r0}_{r1}'
    name = f'scaled_dispersal_{r0}-{r1}' if scaled else f'dispersal_{r0}-{r1}'
    k = f'sdsp_{r0}_{r1}' if scaled else f'dsp_{r0}_{r1}'
    P = AutoPlot(name=name, subfolder=subfolder, **kwargs)
    t0, t1 = int(r0 * P.datasets[0].config.fr), int(r1 * P.datasets[0].config.fr)
    x = np.linspace(r0, r1, t1 - t0)


    for d, lab, c in zip(P.datasets, P.labels, P.colors):
        try :
            try:
                dsp = d.load_aux(type='dispersion', par=par if not scaled else nam.scal(par))
            except:
                dsp = get_dsp(d.step_data, par)
            mean = dsp['median'].values[t0:t1]
            lb = dsp['upper'].values[t0:t1]
            ub = dsp['lower'].values[t0:t1]
        except :
            dsp = preg.get(k, d)
            mean = dsp.groupby(level='Step').quantile(q=0.5).values[t0:t1]
            ub = dsp.groupby(level='Step').quantile(q=0.75).values[t0:t1]
            lb = dsp.groupby(level='Step').quantile(q=0.25).values[t0:t1]
        P.axs[0].fill_between(x, ub, lb, color=c, alpha=.2)
        P.axs[0].plot(x, mean, c, label=lab, linewidth=3 if lab != 'experiment' else 8, alpha=1.0)
    P.conf_ax(xlab='time, $sec$', ylab=ylab, xlim=[x[0], x[-1]], ylim=[0, ymax], xMaxN=4, yMaxN=4)
    P.axs[0].legend(loc='upper left', fontsize=15)
    return P.get()


def plot_navigation_index(subfolder='source', **kwargs):
    P = AutoPlot(name='nav_index', subfolder=subfolder, Nrows=2, figsize=(20, 20), sharex=True, sharey=True, **kwargs)
    from lib.aux.vel_aux import compute_component_velocity,compute_velocity

    for d, c, g in zip(P.datasets, P.colors, P.labels):
        dt = 1 / d.fr
        Nticks = d.Nticks
        Nsec = int(Nticks * dt)
        s = d.read(key='trajectories', file='aux_h5')

        vxs = []
        vys = []
        for id in d.agent_ids:
            s0 = s.xs(id, level='AgentID').values
            v0 = compute_velocity(s0, dt=dt)
            vx = compute_component_velocity(s0, angles=np.zeros(Nticks), dt=dt)
            vy = compute_component_velocity(s0, angles=np.ones(Nticks) * -np.pi / 2, dt=dt)
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


def plot_pathlength(scaled=True, unit='mm', xlabel=None, **kwargs):
    lab = 'pathlength'
    if scaled:
        name = f'scaled_{lab}'
        ylab = f'scaled {lab} $(-)$'
    else:
        name = f'{lab}'
        ylab = f'{lab} $({unit})$'
    P = Plot(name=name, **kwargs)
    if xlabel is None:
        xlabel = 'time, $min$'
    P.build(figsize=(7, 6))

    dst_par, dst_u = preg.getPar('cum_d', to_return=['d', 'u'])
    x = P.trange()
    for d, lab, c in zip(P.datasets, P.labels, P.colors):
        df = d.step_data[dst_par]
        if not scaled and unit == 'cm':
            from lib.registry.units import ureg
            if dst_u == ureg.m:
                df *= 100
        plot_quantiles(df=df, x=x, axis=P.axs[0], color_shading=c, label=lab)

    P.conf_ax(xlab=xlabel, ylab=ylab, xlim=(x[0], x[-1]), ylim=[0, None], xMaxN=5, leg_loc='upper left')
    P.adjust((0.2, 0.95), (0.15, 0.95), 0.05, 0.005)
    return P.get()





