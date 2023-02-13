import os

import numpy as np
from matplotlib import pyplot as plt, ticker, cm

from larvaworld.lib.aux import naming as nam
from larvaworld.lib import reg, aux, plot




def plot_vel_during_strides(dataset, use_component=False, save_to=None, return_fig=False, show=False):
    chunk = 'stride'
    Npoints = 64
    d = dataset
    s = d.step_data

    if save_to is None:
        save_to = os.path.join(d.plot_dir, 'plot_vel_during_strides')
    if not os.path.exists(save_to):
        os.makedirs(save_to)
    save_as_lin = f'linear_velocities_during_strides.{plot.suf}'
    save_as_ang = f'angular_velocity_during_strides.{plot.suf}'
    filepath_lin = os.path.join(save_to, save_as_lin)
    filepath_ang = os.path.join(save_to, save_as_ang)
    filepaths = [filepath_lin, filepath_ang]

    svels = nam.scal(nam.vel(d.points))
    lvels = nam.scal(nam.lin(nam.vel(d.points[1:])))
    ids = d.agent_ids
    hov = nam.vel(nam.orient('front'))

    if use_component:
        lin_vels = lvels
    else:
        lin_vels = svels
    lin_vels = [lin_vels[0], lin_vels[int(len(lin_vels) / 2)], lin_vels[-1]]
    ang_vels = [hov]
    vels = [lin_vels, ang_vels]
    vels_list = lin_vels + ang_vels
    Nvels = len(vels_list)

    all_agents = [s.xs(id, level='AgentID', drop_level=True) for id in ids]
    all_flag_starts = [ag[ag[nam.start(chunk)] == True].index.values.astype(int) for ag in all_agents]
    all_flag_stops = [ag[ag[nam.stop(chunk)] == True].index.values.astype(int) for ag in all_agents]

    all_vel_timeseries = [[] for i in range(Nvels)]
    for agent_id, flag_starts, flag_stops in zip(ids, all_flag_starts, all_flag_stops):
        for start, stop in zip(flag_starts, flag_stops):
            for i, vel in enumerate(vels_list):
                vel_timeserie = s.loc[(slice(start, stop), agent_id), vel].values
                all_vel_timeseries[i].append(vel_timeserie)

    durations = [len(i) for i in all_vel_timeseries[0]]
    lin_vel_timeseries = all_vel_timeseries[:-1]
    ang_vel_timeseries = [[np.abs(a) for a in all_vel_timeseries[-1]]]
    vel_timeseries = [lin_vel_timeseries, ang_vel_timeseries]

    lin_cs = ['black', 'seagreen', 'mediumturquoise']
    ang_cs = ['black']
    cs = [lin_cs, ang_cs]
    lin_labels = [r'$\bf{head}$', r'$\bf{mid}$', r'$\bf{tail}$']
    ang_labels = [r'$\dot{\theta}_{or}$']
    labels = [lin_labels, ang_labels]
    ylabels = [r'scaled velocity $(sec^{-1})$', 'angular velocity $(deg/sec)$']

    for i in [0, 1]:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        for serie, vel, col, c, l in zip(vel_timeseries[i], vels[i], cs[i], cs[i], labels[i]):
            array = [np.interp(x=np.linspace(0, 2 * np.pi, Npoints), xp=np.linspace(0, 2 * np.pi, dur), fp=ts, left=0,
                               right=0) for dur, ts in zip(durations, serie)]
            plot.plot_quantiles(df=array, from_np=True, axis=ax, color_mean=c, color_shading=col, label=l)

        Nticks = 5
        ticks = np.linspace(0, Npoints - 1, Nticks)
        ax.set_xticks(ticks=ticks)
        ax.set_xticklabels(labels=[r'$0$', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$'])
        ax.set_xlim([0, Npoints - 1])
        ax.set_ylabel(ylabels[i])
        ax.set_xlabel('$\phi_{stride}$')
        l = ax.legend(loc='upper right')
        for j, text in enumerate(l.get_texts()):
            text.set_color(cs[i][j])
        plt.subplots_adjust(bottom=0.2, top=0.95, left=0.1, right=0.95, wspace=0.01)
        fig.savefig(filepaths[i], dpi=300)
        print(f'Plot saved as {filepaths[i]}')

@reg.funcs.graph('stride cycle')
def stride_cycle(name=None, shorts=['sv', 'fov', 'rov', 'foa', 'b'], modes=None, subfolder='stride',
                 Nbins=64, individuals=False, pooled=True, **kwargs):
    from larvaworld.lib.process.annotation import compute_interference
    if name is None:
        if individuals:
            name = f'stride_cycle_curves_all_larvae'
        else:
            name = f'stride_cycle_curves'
    Nsh = len(shorts)
    P = plot.AutoPlot(name=name, subfolder=subfolder, build_kws={'N': Nsh, 'Ncols': 1, 'w': 8, 'h': 5, 'mode': 'box'},
                 **kwargs)

    x = np.linspace(0, 2 * np.pi, Nbins)
    for ii, sh in enumerate(shorts):
        # par, lab,lab2, sym, symunit = preg.getPar(sh, to_return=['d', 'label','l', 'symbol', 'symunit'])
        if modes is None:
            mode = 'abs' if sh == 'sv' else 'norm'
        else:
            mode = modes[ii]

        for d in P.datasets:
            c = d.config
            col = c.color if 'color' in c.keys() else d.color
            if individuals:
                cycle_curves = d.cycle_curves
                if cycle_curves is None:
                    try:
                        s, e = d.step_data, d.endpoint_data
                    except:
                        d.load()
                        s, e = d.step_data, d.endpoint_data
                    cycle_curves = compute_interference(s, e, c=c)
                if cycle_curves is not None:
                    df = cycle_curves[sh][mode]
                    if pooled:
                        plot.plot_quantiles(df=df, from_np=True, axis=P.axs[ii], color_shading=col, x=x, label=d.id)
                    else:
                        for j in range(df.shape[0]):
                            P.axs[ii].plot(x, df[j, :], color=col)
                        P.axs[ii].plot(x, np.nanquantile(df, q=0.5, axis=0), label=d.id, color=col)

            else:
                if 'pooled_cycle_curves' not in c.keys():
                    s, e = d.step_data, d.endpoint_data
                    compute_interference(s, e, c=c)
                P.axs[ii].plot(x, np.array(c.pooled_cycle_curves[sh][mode]), label=d.id, color=col)

        P.conf_ax(ii, xticks=np.linspace(0, 2 * np.pi, 5), xlim=[0, 2 * np.pi],
                  xticklabels=[r'$0$', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$'],
                  xlab='$\phi_{stride}$', ylab=reg.par.kdict[sh].symunit, xvis=True if ii == Nsh - 1 else False)
    P.axs[0].legend(loc='upper left', fontsize=15)
    P.conf_fig(title='Stride cycle analysis', title_kws={'w': 'bold', 's': 20}, align=True,
               adjust_kws={'BT': (0.1, 0.9), 'LR': (0.2, 0.9), 'H': 0.01})
    return P.get()

def stride_cycle_individual(s=None, e=None, c=None, ss=None, fr=None, dt=1 / 16, short='fov', idx=0, Nbins=64,
                            color_solo='grey', color='red',
                            absolute=False, save_to=None, pooled=False,
                            ylim=None, axs=None, fig=None, show=False):
    from larvaworld.lib.process.annotation import detect_strides
    p, sv, fv = reg.getPar([short, 'sv', 'fv'])
    if ss is None:
        id = c.agent_ids[idx]
        ee = e.loc[id]
        ss = s.xs(id, level='AgentID')
        fr = ee[fv]
        dt = c.dt
    ssp = ss[p].abs().values if absolute else ss[p].values
    strides = detect_strides(ss[sv], dt, fr=fr, return_runs=False, return_extrema=False)
    strides = strides.tolist()
    pi2 = 2 * np.pi
    x = np.linspace(0, pi2, Nbins)

    if axs is None and fig is None:
        fig, axs = plt.subplots(1, 1, figsize=(15, 6))

    aa = np.zeros([len(strides), Nbins])
    for ii, (s0, s1) in enumerate(strides):
        aa[ii, :] = np.interp(x, np.linspace(0, pi2, s1 - s0), ssp[s0:s1])
        if not pooled:
            axs.plot(x, aa[ii, :], color_solo, linewidth=1, alpha=0.5, zorder=2)

    if pooled:
        plot.plot_quantiles(df=aa, from_np=True, axis=axs, color_shading=color, x=x)
    else:
        aa_mu = np.nanquantile(aa, q=0.5, axis=0)
        axs.plot(x, aa_mu, color, linewidth=5, alpha=1.0, zorder=10)
    axs.set_xlabel('$\phi_{stride}$')
    axs.yaxis.set_major_locator(ticker.MaxNLocator(5))
    axs.set_xlim([0, pi2])
    axs.set_ylim(ylim)
    axs.set_xticks(np.linspace(0, pi2, 5))
    axs.set_xticklabels([r'$0$', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$'])
    if save_to is not None:
        fig.savefig(f'{save_to}/stride_cycle_individual.pdf', dpi=300)
    if show:
        plt.show()

@reg.funcs.graph('stride cycle multi')
def stride_cycle_all_points(name=None,  idx=0, Nbins=64, short='fov',subfolder='stride', maxNpoints=5,
                            axx=None, **kwargs):
    if name is None:
        name = 'stride cycle multi'
    P = plot.AutoPlot(name=name, subfolder=subfolder, build_kws={'Nrows': 2, 'Ncols': 1, 'w': 15, 'h': 6, 'mode': 'box'},
                 **kwargs)




    pi2 = 2 * np.pi
    x = np.linspace(0, pi2, Nbins)

    from larvaworld.lib.process.annotation import stride_interp
    from larvaworld.lib.process.annotation import detect_strides
    l, sv, fv, fov = reg.getPar(['l', 'sv', 'fv', 'fov'])


    for d in P.datasets:
        s,e,c = d.step_data, d.endpoint_data, d.config
        id = c.agent_ids[idx]
        ee = e.loc[id]
        ss = s.xs(id, level='AgentID')
        strides = detect_strides(ss[sv], c.dt, fr=ee[fv], return_runs=False, return_extrema=False)


        if short is not None:
            par, lab = reg.getPar(short, to_return=['d', 'lab'])
            a_sh = ss[par].values
            a_fov = ss[fov].values
            da = np.array([np.trapz(a_fov[s0:s1]) for ii, (s0, s1) in enumerate(strides)])

            aa = stride_interp(a_sh, strides, Nbins)
            aa_minus = aa[da < 0]
            aa_plus = aa[da > 0]
            aa_norm = np.vstack([aa_plus, -aa_minus])

            plot.plot_quantiles(df=aa_norm, from_np=True, axis=P.axs[1], color_shading='blue', x=x, label='experiment')

            P.axs[1].set_ylabel(lab)


        points0 = nam.midline(c.Npoints, type='point')
        if c.Npoints > maxNpoints:
            points = [points0[0]] + [points0[2 + int(ii * (c.Npoints - 2) / (maxNpoints - 2))] for ii in
                                     range(maxNpoints - 2)] + [points0[-1]]
        else:
            points = points0
        if len(points) == 5:
            pointcols = ['black', 'darkblue', 'darkgreen', 'seagreen', 'mediumturquoise']
        else:
            pointcols = cm.rainbow(np.linspace(0, 1, len(points)))
        ymax = 0.7
        for p, col in zip(points, pointcols):
            v_p = nam.vel(p)
            a = ss[v_p] if v_p in ss.columns else aux.eudist(ss[nam.xy(p)].values)/c.dt
            a = a / ee[l]
            aa = np.zeros([len(strides), Nbins])
            for ii, (s0, s1) in enumerate(strides):
                aa[ii, :] = np.interp(x, np.linspace(0, pi2, s1 - s0), a[s0:s1])
            aa_mu = np.nanquantile(aa, q=0.5, axis=0)
            aa_max = np.max(aa_mu)
            phi_max = x[np.argmax(aa_mu)]
            plot.plot_quantiles(df=aa, from_np=True, axis=P.axs[0], color_shading=col, x=x, label=p)
            P.axs[0].axvline(phi_max, ymax=aa_max / ymax, color=col, alpha=1, linestyle='dashed', linewidth=2, zorder=20)
            P.axs[0].scatter(phi_max, aa_max + 0.02 * ymax, color=col, marker='v', linewidth=2, zorder=20)

    P.axs[0].set_ylabel(r'scaled velocity $(sec^{-1})$')
    P.axs[0].set_ylim([0, ymax])
    for ax in P.axs:
        ax.set_xlabel('$\phi_{stride}$')
        ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
        ax.set_xlim([0, pi2])
        ax.set_xticks(np.linspace(0, pi2, 5))
        ax.set_xticklabels([r'$0$', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$'])
        ax.legend(loc='upper left', fontsize=15)

    try:
        att = 'attenuation'
        att_max, att_min, phi_att_max, phi_sv_max = nam.max(att), nam.min(att), nam.max(f'phi_{att}'), nam.max(f'phi_{sv}')
        ps = [nam.max(f'phi_{nam.vel(p)}') for i, p in enumerate(points0)]
        aa = np.zeros([c.Npoints, c.N]) * np.nan
        for i, p in enumerate(ps):
            aa[i, :] = e[p].values - e[phi_att_max].values
        if axx is None:
            axx = P.axs[1].inset_axes([0.7, 0.7, 0.3, 0.25])
        axx.violinplot(aa.T, widths=0.9)
        axx.set_ylabel(r'$\Delta\phi$')
        axx.set_xlabel('# point')
        axx.set_xticks(np.arange(c.Npoints + 1))
        axx.set_xticklabels([None] + np.arange(1, c.Npoints + 1, 1).tolist())
        axx.set_yticks([-np.pi / 2, 0, np.pi / 2, np.pi])
        axx.set_yticklabels([r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'])
        axx.tick_params(axis='both', which='minor', labelsize=12)
        axx.tick_params(axis='both', which='major', labelsize=12)
        axx.axhline(0, color='green', alpha=0.5, linestyle='dashed', linewidth=1)
    except:
        pass
    P.adjust((0.15, 0.9), (0.2, 0.9), 0.1, 0.15)
    return P.get()

@reg.funcs.graph('stride Dbend')
def plot_stride_Dbend(show_text=False, subfolder='stride', **kwargs):
    P = plot.Plot(name='stride_bend_change', subfolder=subfolder, **kwargs)
    P.build()
    ax = P.axs[0]

    fits = {}
    for i, (d, l, c) in enumerate(zip(P.datasets, P.labels, P.colors)):
        b0 = d.get_par(nam.at('bend', nam.start('stride'))).dropna().values.flatten()[:500]
        b1 = d.get_par(nam.at('bend', nam.stop('stride'))).dropna().values.flatten()[:500]
        sign_b = np.sign(b0)
        b0 *= sign_b
        b1 *= sign_b
        db = b1 - b0
        ax.scatter(x=b0, y=db, marker='o', s=2.0, alpha=0.6, color=c, label=l)
        m, k = np.polyfit(b0, db, 1)
        m = np.round(m, 2)
        k = np.round(k, 2)
        fits[l] = [m, k]
        ax.plot(b0, m * b0 + k, linewidth=4, color=c)
        if show_text:
            ax.text(0.3, 0.9 - i * 0.1, rf'${l} : \Delta\theta_{{b}}={m} \cdot \theta_{{b}}$', fontsize=12,
                    transform=ax.transAxes)
            print(f'Bend correction during strides for {l} fitted as : db={m}*b + {k}')
    P.conf_ax(xlab=r'$\theta_{bend}$ at stride start $(deg)$', ylab=r'$\Delta\theta_{bend}$ over stride $(deg)$',
              xlim=[0, 85], ylim=[-60, 60], yMaxN=5)
    P.adjust((0.25, 0.95), (0.2, 0.95), 0.01)
    return P.get()

@reg.funcs.graph('stride Dor')
def plot_stride_Dorient(absolute=True, subfolder='stride', **kwargs):
    P = plot.Plot(name='stride_orient_change', subfolder=subfolder, **kwargs)
    shorts = ['str_fo', 'str_ro']
    P.build(1, len(shorts))
    for i, sh in enumerate(shorts):
        p, sl, xlab = reg.getPar(sh, to_return=['d', 's', 'l'])
        bins, xlim = P.angrange(80, absolute, 200)
        P.plot_par(p, bins, i=i, absolute=absolute, labels=[sl] * P.Ndatasets, alpha=0.5)
        P.conf_ax(i, ylab='probability' if i == 0 else None, xlab=xlab, yMaxN=4, leg_loc='upper left')
    P.adjust((0.12, 0.99), (0.2, 0.95), 0.01)
    return P.get()

@reg.funcs.graph('interference')
def plot_interference(mode='orientation', agent_idx=None, subfolder='interference', **kwargs):
    name = f'interference_{mode}' if agent_idx is None else f'interference_{mode}_agent_idx_{agent_idx}'

    shorts = ['sv']
    if mode == 'orientation':
        shorts.append('fov')
    elif mode == 'orientation_x2':
        shorts.append('fov')
        shorts.append('rov')
    elif mode == 'bend':
        shorts.append('bv')
    elif mode == 'spinelength':
        shorts.append('l')

    Npars = len(shorts)

    pars, ylabs = reg.getPar(shorts, to_return=['d', 'l'])
    P = plot.AutoPlot(name=name, subfolder=subfolder, Nrows=Npars, figsize=(10, Npars * 5), sharex=True, **kwargs)

    ylim = [0, 60] if mode in ['bend', 'orientation', 'orientation_x2'] else None

    for i, (p, ylab) in enumerate(zip(pars, ylabs)):
        for l, d, c in P.data_palette:
            df = d.read(key=f'stride.{p}', file='aux')
            if agent_idx is not None:
                df = df.loc[d.agent_ids[agent_idx]].values
            else:
                df = df.values
            if mode in ['bend', 'orientation']:
                df = np.abs(df)
            Npoints = df.shape[1] - 1
            plot.plot_quantiles(df=df, from_np=True, axis=P.axs[i], color_shading=c, label=l)
        P.conf_ax(i, ylab=ylab, ylim=ylim if i != 0 else [0.0, 0.6], yMaxN=4, leg_loc='upper right',
                  xlab='$\phi_{stride}$', xlim=[0, Npoints], xticks=np.linspace(0, Npoints, 5),
                  xticklabels=[r'$0$', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$'],
                  xvis=True if i == Npars - 1 else False)
    P.adjust((0.12, 0.95), (0.2 / Npars, 0.97), 0.05, 0.1)
    return P.get()
