import copy

import numpy as np
from matplotlib import pyplot as plt, patches

from lib.aux import naming as nam, dictsNlists as dNl
from lib.registry.pars import preg

from lib.plot.base import BasePlot, Plot
from lib.process.aux import detect_strides, detect_pauses, detect_turns, process_epochs


def traj_1group(s, c, unit='mm', fig=None, axs=None, **kwargs):
    scale = 1000 if unit == 'mm' else 1
    from lib.aux.sim_aux import get_tank_polygon
    P = BasePlot(name=f'trajectories', **kwargs)
    P.build(fig=fig, axs=axs)
    tank = get_tank_polygon(c, return_polygon=False) * scale
    for id in c.agent_ids:
        xy = s[['x', 'y']].xs(id, level="AgentID").values * scale
        P.axs[0].plot(xy[:, 0], xy[:, 1])
    P.axs[0].fill(tank[:, 0], tank[:, 1], fill=True, color='lightgrey', edgecolor='black', linewidth=4)
    P.conf_ax(xMaxN=3, yMaxN=3, title=c.id, titlefontsize = 25, xlab=f'X ({unit})', ylab=f'Y ({unit})', equal_aspect=True)
    return P.get()


def traj_grouped(axs=None, fig=None, unit='mm', name=f'comparative_trajectories', subfolder='trajectories',
                 range=None, mode=None, **kwargs):
    def get_traj(d, mode=None):
        if mode == 'origin':
            try:
                s = d.get_traj_aligned(mode)[['x', 'y']]
                return s
            except:
                return get_traj(d, None)
        else:
            try:
                try:
                    s = d.read(key='trajectories', file='aux_h5')
                    return s
                except:
                    s = d.step_data[['x', 'y']]
                    return s
            except:
                s = d.step_data[['x', 'y']]
                return s

    P = Plot(name=name, subfolder=subfolder, **kwargs)
    P.build(1, P.Ndatasets, figsize=(5 * P.Ndatasets, 6), sharex=True, sharey=True, fig=fig, axs=axs)
    for ii, d in enumerate(P.datasets):
        s = get_traj(d, mode)
        c = d.config
        if range is not None:
            t0, t1 = range
            tick0, tick1 = int(t0 / c.dt), int(t1 / c.dt)
            s = s.loc[tick0:tick1]

        _ = traj_1group(s, c, unit=unit, fig=P.fig, axs=P.axs[ii], save_to=None)
        if ii != 0:
            P.axs[ii].yaxis.set_visible(False)
    P.adjust((0.07, 0.95), (0.1, 0.95), 0.05, 0.005)
    return P.get()

def track_annotated(epoch='stride',a=None, dt=0.1, a2plot=None, fig=None, ax=None, ylab =None, ylim=None, xlim=None,slice=None,agent_idx=0, agent_id=None,
                         subfolder='tracks', moving_average_interval=None,epoch_boundaries=True, show_extrema=True, min_amp=None, **kwargs) :
    temp = f'track_{slice[0]}-{slice[1]}' if slice is not None else f'track'
    name = f'{temp}_{agent_id}' if agent_id is not None else f'{temp}_{agent_idx}'
    P = Plot(name=name, subfolder=subfolder, **kwargs)
    Nds = P.Ndatasets

    if epoch == 'stride':
        chunk_cols = ["lightblue", "grey"]
        labels = ['runs', 'pauses']
        ylab = "velocity (1/sec)" if ylab is None else ylab
        i_min, i_max, strides, runs, run_counts = detect_strides(a=a, dt=dt)
        pauses = detect_pauses(a, dt, runs=runs)
        epochs1, epochs2, epochs0 = runs, pauses, strides
    elif epoch == 'turn':
        ax.axhline(0, color='black', alpha=1, linestyle='dashed', linewidth=1)
        chunk_cols = ["lightgreen", "orange"]
        labels = ['L turns', 'R turns']
        ylab = "angular velocity (deg/sec)"  if ylab is None else ylab
        Lturns, Rturns = detect_turns(a, dt)
        if min_amp is not None:
            Lturns1, Ldurs, Lturn_slices, Lamps, Lturn_idx, Lmaxs = process_epochs(a, Lturns, dt)
            Rturns1, Rdurs, Rturn_slices, Ramps, Rturn_idx, Rmaxs = process_epochs(a, Rturns, dt)
            Lturns = Lturns[np.abs(Lamps) > min_amp]
            Rturns = Rturns[np.abs(Ramps) > min_amp]
        epochs1, epochs2, epochs0 = Lturns, Rturns, Lturns.tolist() + Rturns.tolist()

    handles = [patches.Patch(color=col, label=n) for n, col in zip(labels, chunk_cols)]
    # figx = 20 if slice is None else int((slice[1] - slice[0]) / 3)
    P.build(Nds, 1, figsize=(20, 5 * Nds), sharey=True, sharex=True,fig=fig, axs=ax)


    trange = np.arange(0, a.shape[0] * dt, dt)
    if xlim is None:
        xlim = (0, trange[-1])

    if moving_average_interval:
        from lib.aux.data_aux import moving_average
        a = moving_average(a, n=int(moving_average_interval / dt))

    ii=0
    P.conf_ax(ii, xlab=r'time $(sec)$' if ii == Nds - 1 else None, ylab=ylab, ylim=ylim, xlim=xlim)
    ax.legend(loc="upper right", handles=handles, labels=labels, fontsize=15)

    if a2plot is not None:
        ax.plot(trange, a2plot)
    else:
        ax.plot(trange, a)
        if show_extrema and epoch=='stride':
            ax.plot(trange[i_max], a[i_max], linestyle='None', lw=10, color='green', marker='v')
            ax.plot(trange[i_min], a[i_min], linestyle='None', lw=10, color='red', marker='^')

    if epoch_boundaries:
        for s0, s1 in epochs0:
            ax.axvline(trange[s0], color=f'{0.4 * (0 + 1)}', alpha=0.3, linestyle='dashed', linewidth=1)
            ax.axvline(trange[s1], color=f'{0.4 * (0 + 1)}', alpha=0.3, linestyle='dashed', linewidth=1)
    for s0, s1 in epochs1:
        ax.axvspan(trange[s0], trange[s1], color=chunk_cols[0], alpha=1.0)
    for s0, s1 in epochs2:
        ax.axvspan(trange[s0], trange[s1], color=chunk_cols[1], alpha=1.0)

    # P.adjust((0.08, 0.95), (0.15, 0.95), H=0.1)
    return P.get()


def annotated_strideplot(**kwargs):
    return track_annotated(epoch='stride', **kwargs)

def annotated_turnplot(**kwargs):
    return track_annotated(epoch='turn', **kwargs)


def plot_marked_strides(agent_idx=0, agent_id=None, slice=[20, 40], subfolder='individuals', **kwargs):
    temp = f'marked_strides_{slice[0]}-{slice[1]}' if slice is not None else f'marked_strides'
    name = f'{temp}_{agent_id}' if agent_id is not None else f'{temp}_{agent_idx}'
    P = Plot(name=name, subfolder=subfolder, **kwargs)
    Nds = P.Ndatasets

    chunks = ['stride', 'pause']
    chunk_cols = ['lightblue', 'grey']
    p, ylab = preg.getPar('sv', to_return=['d', 'l'])
    figx = 15 * 6 * 3 if slice is None else int((slice[1] - slice[0]) / 3)
    figy = 5

    P.build(Nds, 1, figsize=(figx, figy * Nds), sharey=True, sharex=True)
    handles = [patches.Patch(color=col, label=n) for n, col in zip(['stride', 'pause'], chunk_cols)]

    for ii, (d, l) in enumerate(zip(P.datasets, P.labels)):
        ax = P.axs[ii]
        P.conf_ax(ii, xlab=r'time $(sec)$' if ii == Nds - 1 else None, ylab=ylab, ylim=[0, 1.0], xlim=slice,
                  leg_loc='upper right', leg_handles=handles)
        temp_id = d.agent_ids[agent_idx] if agent_id is None else agent_id
        s = copy.deepcopy(d.read('step').xs(temp_id, level='AgentID', drop_level=True))
        s.set_index(s.index * d.dt, inplace=True)
        ax.plot(s[p], color='blue')
        for i, (c, col) in enumerate(zip(chunks, chunk_cols)):
            s0s = s.index[s[nam.start(c)] == True]
            s1s = s.index[s[nam.stop(c)] == True]
            for s0, s1 in zip(s0s, s1s):
                ax.axvspan(s0, s1, color=col, alpha=1.0)
                ax.axvline(s0, color=f'{0.4 * (i + 1)}', alpha=0.6, linestyle='dashed', linewidth=1)
                ax.axvline(s1, color=f'{0.4 * (i + 1)}', alpha=0.6, linestyle='dashed', linewidth=1)

        ax.plot(s[p].loc[s[nam.max(p)] == True], linestyle='None', lw=10, color='green', marker='v')
        ax.plot(s[p].loc[s[nam.min(p)] == True], linestyle='None', lw=10, color='red', marker='^')
    P.adjust((0.08, 0.95), (0.15, 0.95), H=0.1)
    return P.get()


def plot_sample_tracks(mode=['strides', 'turns'], agent_idx=0, agent_id=None, slice=[20, 40], subfolder='individuals',
                       **kwargs):
    Nrows = len(mode)
    if Nrows == 2:
        suf = 'stridesVSturns'
    else:
        suf = mode[0]
    t0, t1 = slice
    temp = f'sample_marked_{suf}_{t0}-{t1}'
    name = f'{temp}_{agent_id}' if agent_id is not None else f'{temp}_{agent_idx}'
    P = Plot(name=name, subfolder=subfolder, **kwargs)
    Nds = P.Ndatasets

    figx = 15 * 6 * 3 if slice is None else int((t1 - t0) / 3)
    figy = 5

    P.build(Nrows, Nds, figsize=(figx * Nds, figy * Nrows), sharey=False, sharex=True)

    for ii, (d, l) in enumerate(zip(P.datasets, P.labels)):
        for jj, key in enumerate(mode):
            kk = ii + Nrows * jj
            ax = P.axs[kk]
            if key == 'strides':
                chunks = ['stride', 'pause']
                chunk_cols = ['lightblue', 'grey']

                p, ylab, ylim = preg.getPar('sv', to_return=['d', 'l', 'lim'])
                ylim = (0.0, 1.0)
            elif key == 'turns':
                chunks = ['Rturn', 'Lturn']
                chunk_cols = ['lightgreen', 'orange']

                b = 'bend'
                bv = nam.vel(b)
                ho = nam.orient('front')
                hov = nam.vel(ho)
                p, ylab, ylim = preg.getPar('fov', to_return=['d', 'l', 'lim'])

            handles = [patches.Patch(color=col, label=n) for n, col in zip(chunks, chunk_cols)]
            P.conf_ax(kk, xlab=r'time $(sec)$' if jj == Nrows - 1 else None, ylab=ylab, ylim=ylim, xlim=slice,
                      leg_loc='upper right', leg_handles=handles)

            temp_id = d.agent_ids[agent_idx] if agent_id is None else agent_id
            s = copy.deepcopy(d.read('step').xs(temp_id, level='AgentID', drop_level=True))
            s.set_index(s.index * d.dt, inplace=True)
            ax.plot(s[p], color='blue')
            for i, (c, col) in enumerate(zip(chunks, chunk_cols)):
                s0s = s.index[s[nam.start(c)] == True]
                s1s = s.index[s[nam.stop(c)] == True]
                for s0, s1 in zip(s0s, s1s):
                    ax.axvspan(s0, s1, color=col, alpha=1.0)
                    ax.axvline(s0, color=f'{0.4 * (i + 1)}', alpha=0.6, linestyle='dashed', linewidth=1)
                    ax.axvline(s1, color=f'{0.4 * (i + 1)}', alpha=0.6, linestyle='dashed', linewidth=1)

            ax.plot(s[p].loc[s[nam.max(p)] == True], linestyle='None', lw=10, color='green', marker='v')
            ax.plot(s[p].loc[s[nam.min(p)] == True], linestyle='None', lw=10, color='red', marker='^')
    P.adjust((0.08, 0.95), (0.12, 0.95), H=0.2)
    return P.get()




