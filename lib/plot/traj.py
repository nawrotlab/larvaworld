import copy

import numpy as np
from matplotlib import pyplot as plt, patches

from lib.aux import naming as nam
from lib.conf.pars.pars import getPar

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
    P.conf_ax(xMaxN=3, yMaxN=3, title=c.id, xlab=f'X ({unit})', ylab=f'Y ({unit})', equal_aspect=True)
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
        print(d.id,mode,range, ii)
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

#
# def plot_marked_turns(dataset, agent_ids=None, agent_idx=[0], turn_epochs=['Rturn', 'Lturn'],
#                       vertical_boundaries=False, min_turn_angle=0, slices=[], subfolder='individuals',
#                       save_to=None, return_fig=False, show=False, sizes=['short', 'long']):
#     Ndatasets, colors, save_to, labels = plot_config(datasets=[dataset], labels=[dataset.id], save_to=save_to,
#                                                      subfolder=subfolder)
#     # We plot the complete or a slice of the timeseries of scal centroid velocity. The grey areas are stridechains
#     d = dataset
#
#     if agent_ids is None:
#         if agent_idx is None:
#             agent_ids = d.agent_ids
#         else:
#             agent_ids = [d.agent_ids[idx] for idx in agent_idx]
#
#     xx = f'marked_turns_min_angle_{min_turn_angle}'
#     filepath_full = f'{xx}_full.{suf}'
#     filepath_full_long = f'{xx}_full_long.{suf}'
#     filepath_slices = []
#     for i, slice in enumerate(slices):
#         filepath_slices.append(f'{xx}_slice_{i}.{suf}')
#     generic_filepaths = [filepath_full_long] + filepath_slices
#
#     figsize_short = (20, 5)
#     figsize_long = (15 * 6, 5)
#     figsizes = [figsize_long] + [figsize_short] * len(generic_filepaths)
#
#     xlims = [(0, d.duration)] + slices
#
#     # ymax=1.0
#
#     b = 'bend'
#     bv = nam.vel(b)
#     ho = nam.orient('front')
#     hov = nam.vel(ho)
#     fig_dict = {}
#     for agent_id in agent_ids:
#         filepaths = [f'{agent_id}_{f}' for f in generic_filepaths]
#
#         s = d.step_data.xs(agent_id, level='AgentID', drop_level=True)
#         s.set_index(s.index.values / d.fr, inplace=True)
#
#         b0 = s[b]
#         bv0 = s[bv]
#         ho0 = s[ho]
#         hov0 = s[hov]
#
#         for idx, (filepath, figsize, xlim) in enumerate(zip(filepaths, figsizes, xlims)):
#             fig, axs = plt.subplots(1, 1, figsize=figsize)
#
#             if turn_epochs is not None:
#                 cmap = cm.get_cmap('Pastel2')
#                 num_chunks = len(turn_epochs)
#                 colors = [cmap(i) for i in np.arange(num_chunks)]
#                 epoch_handles = []
#                 temp = None
#                 for i, (chunk, color) in enumerate(zip(turn_epochs, colors)):
#                     try:
#                         idx01 = d.load_chunk_dicts()[agent_id][chunk] / d.fr
#                         if min_turn_angle > 0:
#                             angles = np.abs([np.trapz(hov0[s0:s1 + 1], dx=d.dt) for s0, s1 in idx01])
#                             idx01 = idx01[angles >= min_turn_angle]
#                         start_indexes, stop_indexes = idx01[:, 0], idx01[:, 1]
#                     except:
#                         start_flag = f'{chunk}_start'
#                         stop_flag = f'{chunk}_stop'
#                         stop_indexes = s.index[s[stop_flag] == True]
#                         start_indexes = s.index[s[start_flag] == True]
#                         if min_turn_angle > 0:
#                             angle_flag = nam.chunk_track(chunk, nam.unwrap(nam.orient('front')))
#                             angles = np.abs(s[angle_flag].dropna().values)
#                             stop_indexes = stop_indexes[angles > min_turn_angle]
#                             start_indexes = start_indexes[angles > min_turn_angle]
#
#                     for start, stop in zip(start_indexes, stop_indexes):
#                         temp = plt.axvspan(start, stop, color=color, alpha=1.0)
#
#                         if vertical_boundaries:
#                             plt.axvline(start, color=f'{0.4 * (i + 1)}', alpha=0.6, linestyle='dashed', linewidth=1)
#                             plt.axvline(stop, color=f'{0.4 * (i + 1)}', alpha=0.6, linestyle='dashed', linewidth=1)
#                     if temp is not None:
#                         epoch_handles.append(temp)
#             ax1 = b0.plot(label=r'$\theta_{b}$', lw=2, color='blue')
#             ax1.set_ylabel(r'angle $(deg)$')
#             ax1.set_xlabel(r'time $(sec)$')
#             ax1.set_ylim([-100, 100])
#             ax1.set_xlim(xlim)
#             # print(xlim)
#             # plt.legend(loc= 'upper left')
#             ax2 = bv0.plot(secondary_y=True, label=r'$\dot{\theta}_{b}$', lw=2, color='green')
#             ax2.plot(hov0, label=r'$\dot{\theta}_{or}$', lw=3, color='black')
#             ax2.set_ylabel(r'angular velocity $(deg/sec)$')
#             ax2.set_ylim([-500, 500])
#
#             plt.axhline(0, color='black', alpha=0.4, linestyle='dashed', linewidth=1)
#
#             handles, labels = [], []
#             for ax in fig.axes:
#                 for h, l in zip(*ax.get_legend_handles_labels()):
#                     handles.append(h)
#                     labels.append(l)
#             par_legend = plt.legend(handles, labels, loc=2)
#             plt.legend(epoch_handles, turn_epochs, loc=1)
#             plt.gca().add_artist(par_legend)
#             plt.subplots_adjust(hspace=0.05, top=0.95, bottom=0.2, left=0.08, right=0.92)
#             filename = f'{save_to}/{filepath}'
#             fig.savefig(filename, dpi=300)
#             print(f'Image saved as {filename}')
#             fig_dict[f'turns_{agent_id}_{i}'] = fig
#     # return process_plot(fig, save_to, filename, return_fig, show)
#     return fig_dict

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
        from lib.process.aux import moving_average
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

def annotated_strideplot0(a, dt, a2plot=None, ax=None, ylim=None, xlim=None,
                         moving_average_interval=None,epoch_boundaries=True, show_extrema=True, min_amp=None, **kwargs):
    """
    Plots annotated strides-runs and pauses in timeseries.

    Extended description of function.

    Parameters
    ----------
    a : array
        1D np.array : velocity timeseries
    dt : float
        Timestep of the timeseries
    ax : obj
        The matplotlib axis on which to draw
    ylim : Tuple[float,float]
        The yaxis boundaries
    xlim : Tuple[float,float]
        The xaxis boundaries.Default is the whole a.
    show_extrema : bool
        Annotate minima & maxima. Default : True.
    epoch_boundaries : bool
        Annotate strides by vertical dashed lines. Default : True.
    moving_average_interval : float
        Plot moving average of velocity over a time interval instead of the actual. Default : None.
    **kwargs : dict
        Other arguments for bout annotation

    Returns
    -------
    ax : obj
        The drawn matplotlib axis

    """
    from lib.process.aux import detect_strides, detect_pauses, moving_average
    chunk_cols = ["lightblue", "grey"]
    labels = ['runs', 'pauses']
    trange = np.arange(0, a.shape[0] * dt, dt)
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(20, 5), sharex=True, sharey=True)
    if xlim is None:
        xlim = (0, trange[-1])

    i_min, i_max, strides, runs, run_counts = detect_strides(a=a, dt=dt, **kwargs)
    pauses = detect_pauses(a, dt, runs=runs)

    if moving_average_interval:
        a = moving_average(a, n=int(moving_average_interval / dt))
    if a2plot is not None:
        ax.plot(trange, a2plot)
    else:
        ax.plot(trange, a)
        ax.set_ylim(ylim)
        ax.set_ylabel("velocity (1/sec)")
        if show_extrema:
            ax.plot(trange[i_max], a[i_max], linestyle='None', lw=10, color='green', marker='v')
            ax.plot(trange[i_min], a[i_min], linestyle='None', lw=10, color='red', marker='^')
    ax.set_xlabel("time (sec)")
    ax.set_xlim(xlim)

    if epoch_boundaries:
        for s0, s1 in strides:
            # ax.axvspan(trange[s0], trange[s1], color=chunk_cols[0], alpha=1.0)
            ax.axvline(trange[s0], color=f'{0.4 * (0 + 1)}', alpha=0.3, linestyle='dashed', linewidth=1)
            ax.axvline(trange[s1], color=f'{0.4 * (0 + 1)}', alpha=0.3, linestyle='dashed', linewidth=1)
    for s0, s1 in runs:
        ax.axvspan(trange[s0], trange[s1], color=chunk_cols[0], alpha=1.0)
    for p0, p1 in pauses:
        ax.axvspan(trange[p0], trange[p1], color=chunk_cols[1], alpha=1.0)

    handles = [patches.Patch(color=col, label=n) for n, col in zip(labels, chunk_cols)]
    ax.legend(loc="upper right", handles=handles, labels=labels, fontsize=15)


def annotated_turnplot0(a, dt, a2plot=None, ax=None, ylim=None, xlim=None,
                       moving_average_interval=None,epoch_boundaries=False, show_extrema=False, min_amp=None, **kwargs):
    """
    Plots annotated turnss in timeseries.

    Extended description of function.

    Parameters
    ----------
    a : array
        1D np.array : velocity timeseries
    dt : float
        Timestep of the timeseries
    ax : obj
        The matplotlib axis on which to draw
    ylim : Tuple[float,float]
        The yaxis boundaries
    xlim : Tuple[float,float]
        The xaxis boundaries.Default is the whole a.
    show_extrema : bool
        Annotate minima & maxima. Default : True.
    epoch_boundaries : bool
        Annotate epochs by vertical dashed lines. Default : True.
    moving_average_interval : float
        Plot moving average of velocity over a time interval instead of the actual. Default : None.
    **kwargs : dict
        Other arguments for bout annotation

    Returns
    -------
    ax : obj
        The drawn matplotlib axis

    """
    from lib.process.aux import detect_turns, process_epochs, moving_average

    chunk_cols = ["lightgreen", "orange"]
    labels = ['L turns', 'R turns']
    trange = np.arange(0, a.shape[0] * dt, dt)
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(20, 5), sharex=True, sharey=True)
    if xlim is None:
        xlim = (0, trange[-1])

    Lturns, Rturns = detect_turns(a, dt, **kwargs)
    if min_amp is not None:
        Lturns1, Ldurs, Lturn_slices, Lamps, Lturn_idx, Lmaxs = process_epochs(a, Lturns, dt)
        Rturns1, Rdurs, Rturn_slices, Ramps, Rturn_idx, Rmaxs = process_epochs(a, Rturns, dt)
        Lturns = Lturns[np.abs(Lamps) > min_amp]
        Rturns = Rturns[np.abs(Ramps) > min_amp]

    if moving_average_interval:
        a = moving_average(a, n=int(moving_average_interval / dt))
    if a2plot is not None:
        ax.plot(trange, a2plot)
    else:
        ax.plot(trange, a)

        ax.set_ylabel("angular velocity (deg/sec)")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("time (sec)")
    ax.axhline(0, color='black', alpha=1, linestyle='dashed', linewidth=1)
    for s0, s1 in Lturns:
        ax.axvspan(trange[s0], trange[s1], color=chunk_cols[0], alpha=1.0)
        if epoch_boundaries:
            ax.axvline(trange[s0], color=f'{0.4 * (0 + 1)}', alpha=0.3, linestyle='dashed', linewidth=1)
            ax.axvline(trange[s1], color=f'{0.4 * (0 + 1)}', alpha=0.3, linestyle='dashed', linewidth=1)
    for s0, s1 in Rturns:
        ax.axvspan(trange[s0], trange[s1], color=chunk_cols[1], alpha=1.0)
        if epoch_boundaries:
            ax.axvline(trange[s0], color=f'{0.4 * (0 + 1)}', alpha=0.3, linestyle='dashed', linewidth=1)
            ax.axvline(trange[s1], color=f'{0.4 * (0 + 1)}', alpha=0.3, linestyle='dashed', linewidth=1)

    handles = [patches.Patch(color=col, label=n) for n, col in zip(labels, chunk_cols)]
    ax.legend(loc="upper right", handles=handles, labels=labels, fontsize=15)


def plot_marked_strides(agent_idx=0, agent_id=None, slice=[20, 40], subfolder='individuals', **kwargs):
    temp = f'marked_strides_{slice[0]}-{slice[1]}' if slice is not None else f'marked_strides'
    name = f'{temp}_{agent_id}' if agent_id is not None else f'{temp}_{agent_idx}'
    P = Plot(name=name, subfolder=subfolder, **kwargs)
    Nds = P.Ndatasets

    chunks = ['stride', 'pause']
    chunk_cols = ['lightblue', 'grey']
    p, ylab = getPar('sv', to_return=['d', 'l'])
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

                p, ylab, ylim = getPar('sv', to_return=['d', 'l', 'lim'])
                ylim = (0.0, 1.0)
            elif key == 'turns':
                chunks = ['Rturn', 'Lturn']
                chunk_cols = ['lightgreen', 'orange']

                b = 'bend'
                bv = nam.vel(b)
                ho = nam.orient('front')
                hov = nam.vel(ho)
                p, ylab, ylim = getPar('fov', to_return=['d', 'l', 'lim'])

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
