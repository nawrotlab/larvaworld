import copy

import numpy as np
from matplotlib import pyplot as plt, patches

from larvaworld.lib.aux import naming as nam, moving_average
from larvaworld.lib import reg, aux, plot



def traj_1group(xy, c, unit='mm', title=None, single_color=False, **kwargs):
    ids = xy.index.unique('AgentID').values
    color = c.color if single_color else None
    scale = 1000 if unit == 'mm' else 1
    P = plot.AutoBasePlot(name=f'trajectories', **kwargs)
    ax = P.axs[0]
    tank = aux.get_tank_polygon(c, return_polygon=False) * scale
    for id in ids:
        xy0 = xy.xs(id, level="AgentID").values * scale
        ax.plot(xy0[:, 0], xy0[:, 1], color=color)
    ax.fill(tank[:, 0], tank[:, 1], fill=True, color='lightgrey', edgecolor='black', linewidth=4)
    for sid, sdic in c.env_params.food_params.source_units.items():
        px, py = sdic['pos']
        circle = plt.Circle((px * scale, py * scale), sdic['radius'] * scale, color=sdic['default_color'])
        ax.add_patch(circle)
    P.conf_ax(xMaxN=3, yMaxN=3, title=title, titlefontsize=25, xlab=f'X ({unit})', ylab=f'Y ({unit})',
              equal_aspect=True)
    return P.get()


def get_traj(d, mode='default'):
    if mode=='default':
        return d.load_traj(mode)
    elif mode == 'origin':
        try:
            ss=d.load_traj(mode)
            return ss[['x', 'y']]
        except:
            s = d.load_step(h5_ks=['contour', 'midline'])
            from larvaworld.lib.process.spatial import align_trajectories
            ss=align_trajectories(s, c=d.config, store=True, replace=False, transposition='origin')
            return ss[['x', 'y']]

@reg.funcs.graph('trajectories')
def traj_grouped(unit='mm', name=None, subfolder='trajectories',
                 range=None, mode='default', single_color=False, **kwargs):
    if name is None:
        name = f'comparative_trajectories_{mode}'

    P = plot.AutoPlot(name=name, subfolder=subfolder,  # subplot_kw=dict(projection='polar'),
                 build_kws={'Nrows': 1, 'Ncols': 'Ndatasets', 'wh': 5, 'mode': 'both'}, **kwargs)
    for ii, (l, d) in enumerate(P.data_dict.items()):
        xy = get_traj(d, mode)
        c = d.config
        if range is not None:
            t0, t1 = range
            tick0, tick1 = int(t0 / c.dt), int(t1 / c.dt)
            xy = xy.loc[tick0:tick1]

        _ = traj_1group(xy, c, unit=unit, fig=P.fig, axs=P.axs[ii], title=l, single_color=single_color, save_to=None)
        if ii != 0:
            P.axs[ii].yaxis.set_visible(False)
    P.adjust((0.1, 0.9), (0.2, 0.9), 0.1, 0.01)
    return P.get()

def ax_conf_kws(kws, trange, Ndatasets,Nrows, i=0, ylab=None, ylim=None, xlim=None):
    conf_kws = {
        'ylab': kws.ylab if ylab is None else ylab,
        'ylim': kws.ylim if ylim is None else ylim,
        'xlim': (0, trange[-1]) if xlim is None else xlim,
        'xlab': r'time $(sec)$',
        'xvis' : True if i == Nrows - 1 else False,
    }

    leg_kws = {
        'leg_loc': "upper right",
        'leg_handles': [patches.Patch(color=col, label=l) for l, col in zip(kws.labels, kws.chunk_cols)],
        'leg_labels': kws.labels,
        'legfontsize': 15,
    }

    return {**conf_kws, **leg_kws}

def epoch_func(**kwargs):
    from larvaworld.lib.process.annotation import detect_turns, detect_strides, detect_pauses, process_epochs

    def stride_epochs(a, trange, show_extrema=True, a2plot=None, dt=0.1, **kwargs):

        if show_extrema and a2plot is None:
            i_min, i_max, strides, runs, run_counts = detect_strides(a=a, dt=dt, return_extrema=True)

            def func(ax):
                ax.plot(trange[i_max], a[i_max], linestyle='None', lw=10, color='green', marker='v')
                ax.plot(trange[i_min], a[i_min], linestyle='None', lw=10, color='red', marker='^')
        else:
            strides, runs, run_counts = detect_strides(a=a, dt=dt, return_extrema=False)

            def func(ax):
                pass
        pauses = detect_pauses(a, dt, runs=runs)
        # epochs1, epochs2, epochs0 = runs, pauses, strides
        epochs = [runs, pauses]
        epochs0 = strides
        return epochs, epochs0, func

    def turn_epochs(a, trange, min_amp=None, dt=0.1, **kwargs):
        def func(ax):
            ax.axhline(0, color='black', alpha=1, linestyle='dashed', linewidth=1)

        Lturns, Rturns = detect_turns(a, dt)
        if min_amp is not None:
            Lturns1, Ldurs, Lturn_slices, Lamps, Lturn_idx, Lmaxs = process_epochs(a, Lturns, dt)
            Rturns1, Rdurs, Rturn_slices, Ramps, Rturn_idx, Rmaxs = process_epochs(a, Rturns, dt)
            Lturns = Lturns[np.abs(Lamps) > min_amp]
            Rturns = Rturns[np.abs(Ramps) > min_amp]

        epochs = [Lturns, Rturns]
        epochs0 = Lturns.tolist() + Rturns.tolist()
        return epochs, epochs0, func

    epoch_dict = aux.AttrDict({
        'stride': {
            'ylab': "velocity (1/sec)",
            'ylim': (0.0,1.0),
            'labels': ['runs', 'pauses'],
            'chunk_cols': ["lightblue", "grey"],
            'func': stride_epochs,
            'k': 'sv'

        },
        'turn': {
            'ylab': "angular velocity (deg/sec)",
            'ylim': (-100.0,100.0),
            'labels': ['L turns', 'R turns'],
            'chunk_cols': ["lightgreen", "orange"],
            'func': turn_epochs,
            'k': 'fov'
        }
    })

    def epoch_f(epoch):
        kws = epoch_dict[epoch]

        def ss_f(ss, moving_average_interval=None, a2plot=None, dt=0.1, **kwargs):
            a = ss[reg.getPar(kws.k)].values
            trange = np.arange(0, a.shape[0] * dt, dt)
            if a2plot is not None:
                aa2plot = a2plot
            else:
                if moving_average_interval:
                    a = moving_average(a, n=int(moving_average_interval / dt))
                aa2plot = a

            epochs, epochs0, ax_func = kws.func(a, trange, a2plot=a2plot, dt=dt, **kwargs)

            def ax0_f(ax, epoch_boundaries=True):
                ax_func(ax)
                ax.plot(trange, aa2plot)

                if epoch_boundaries:
                    for s0, s1 in epochs0:
                        for s01 in [s0, s1]:
                            ax.axvline(trange[s01], color=f'{0.4 * (0 + 1)}', alpha=0.3, linestyle='dashed',
                                       linewidth=1)

                for color, epoch in zip(kws.chunk_cols, epochs):
                    for s0, s1 in epoch:
                        ax.axvspan(trange[s0], trange[s1], color=color, alpha=1.0)

            # return ax0_f

            def ax_conf0(P, **kwargs):
                ax_conf = ax_conf_kws(kws=kws, trange=trange, Ndatasets=P.Ndatasets,Nrows=P.Nrows, **kwargs)
                return ax_conf

            def P0_f(P, i=0,title=None,**kwargs):
                ax0_f(ax=P.axs[i], **kwargs)
                ax_conf = ax_conf0(P=P, i=i, **kwargs)
                P.conf_ax(i, title=title, **ax_conf)
                # for i,ax in enumerate(P.axs) :





            return P0_f

        return ss_f

    return epoch_f

def track_annotated(epoch='stride', a=None, dt=0.1, a2plot=None, ylab=None, ylim=None, xlim=None,
                    slice=None, agent_idx=0, agent_id=None,
                    subfolder='tracks', moving_average_interval=None, epoch_boundaries=True, show_extrema=True,
                    min_amp=None, **kwargs):
    from larvaworld.lib.process.annotation import detect_turns, detect_strides, detect_pauses, process_epochs
    temp = f'track_{slice[0]}-{slice[1]}' if slice is not None else f'track'
    name = f'{temp}_{agent_id}' if agent_id is not None else f'{temp}_{agent_idx}'
    P = plot.AutoPlot(name=name, subfolder=subfolder,
                 build_kws={'Nrows': 'Ndatasets', 'Ncols': 1, 'w': 20, 'h': 5, 'mode': 'both'}, ** kwargs)

    trange = np.arange(0, a.shape[0] * dt, dt)

    ax = P.axs[0]

    def stride_epochs(a, dt, ax):
        if show_extrema and a2plot is None:
            i_min, i_max, strides, runs, run_counts = detect_strides(a=a, dt=dt, return_extrema=True)
            ax.plot(trange[i_max], a[i_max], linestyle='None', lw=10, color='green', marker='v')
            ax.plot(trange[i_min], a[i_min], linestyle='None', lw=10, color='red', marker='^')
        else:
            strides, runs, run_counts = detect_strides(a=a, dt=dt, return_extrema=False)
        pauses = detect_pauses(a, dt, runs=runs)
        # epochs1, epochs2, epochs0 = runs, pauses, strides
        epochs = [runs, pauses]
        epochs0 = strides
        return epochs, epochs0

    def turn_epochs(a, dt, ax):
        ax.axhline(0, color='black', alpha=1, linestyle='dashed', linewidth=1)
        Lturns, Rturns = detect_turns(a, dt)
        if min_amp is not None:
            Ldurs, Lamps, Lmaxs = process_epochs(a, Lturns, dt, return_idx=False)
            Rdurs, Ramps, Rmaxs = process_epochs(a, Rturns, dt, return_idx=False)
            Lturns = Lturns[np.abs(Lamps) > min_amp]
            Rturns = Rturns[np.abs(Ramps) > min_amp]

        epochs = [Lturns, Rturns]
        epochs0 = Lturns.tolist() + Rturns.tolist()
        return epochs, epochs0

    epoch_dict = aux.AttrDict({
        'stride': {
            'ylab': "velocity (1/sec)",
            'labels': ['runs', 'pauses'],
            'chunk_cols': ["lightblue", "grey"],
            'func': stride_epochs,

        },
        'turn': {
            'ylab': "angular velocity (deg/sec)",
            'labels': ['L turns', 'R turns'],
            'chunk_cols': ["lightgreen", "orange"],
            'func': turn_epochs,
        }
    })

    kws = epoch_dict[epoch]

    epochs, epochs0 = kws.func(a, dt, ax=ax)

    conf_kws = {
        'ylab': kws.ylab if ylab is None else ylab,
        'ylim': ylim,
        'xlim': (0, trange[-1]) if xlim is None else xlim,
        'xlab': r'time $(sec)$' if 0 == P.Ndatasets - 1 else None,
    }

    if a2plot is not None:
        aa2plot = a2plot
    else:
        if moving_average_interval:
            a = aux.moving_average(a, n=int(moving_average_interval / dt))
        aa2plot = a

    ax.plot(trange, aa2plot)

    if epoch_boundaries:
        for s0, s1 in epochs0:
            for s01 in [s0, s1]:
                ax.axvline(trange[s01], color=f'{0.4 * (0 + 1)}', alpha=0.3, linestyle='dashed', linewidth=1)

    for color, epoch in zip(kws.chunk_cols, epochs):
        for s0, s1 in epoch:
            ax.axvspan(trange[s0], trange[s1], color=color, alpha=1.0)

    leg_kws = {
        'leg_loc': "upper right",
        'leg_handles': [patches.Patch(color=col, label=l) for l, col in zip(kws.labels, kws.chunk_cols)],
        'leg_labels': kws.labels,
        'legfontsize': 15,
    }

    P.conf_ax(0, **conf_kws, **leg_kws)
    return P.get()

def annotated_strideplot(**kwargs):
    return track_annotated(epoch='stride', **kwargs)

def annotated_turnplot(**kwargs):
    return track_annotated(epoch='turn', **kwargs)

def track_annotated_data(name=None, subfolder='tracks',
                         epoch='stride', a2plot_k=None, agent_idx=[3, 4, 5, 6, 7], dur=1, **kwargs):
    if name is None:
        name = f'annotated_{epoch}plot'
    Nidx = len(agent_idx)

    P = plot.AutoPlot(name=name, subfolder=subfolder,
                 build_kws={'Nrows': 'Ndatasets', 'Nrows_coef': Nidx, 'Ncols': 1, 'w': 15, 'h': 3, 'mode': 'both'},
                 **kwargs)
    epoch_kdic = {
        'stride': 'sv',
        'turn': 'fov'
    }
    apar = reg.getPar(epoch_kdic[epoch])

    def get_a(ss):
        return ss[apar].values

    if a2plot_k is not None:

        par, lab = reg.getPar(a2plot_k, to_return=['d', 'symbol'])
    else:
        par, lab = None, None

    def get_a2plot(ss):
        return ss[par].values if par is not None else None

    def get_title(idx,c,e, l):
        id = c.agent_ids[idx]
        ee = e.loc[id]

        length = np.round(ee['length'] * 1000, 2)
        cum_sd = np.round(ee[reg.getPar('cum_sd')], 2)
        run_tr = int(ee[reg.getPar('run_tr')] * 100)
        title = f'{l}  # {idx} track, l : {length} mm, pathlength {cum_sd}xl , {run_tr}% time crawling'
        return title

    for jj, (l, d) in enumerate(P.data_dict.items()):

        s, e, c = d.step_data, d.endpoint_data, d.config
        Nticks = int(dur * 60 / c.dt)
        kws0 = aux.AttrDict({
            'datasets': [d],
            'labels': [l],
            # 'agent_idx': idx,
            'slice': (0, dur * 60),
            'dt': c.dt,
            'fig': P.fig,
            'show': False,
            # 'a': a_dict[epoch],
            'epoch': epoch,
            'ylab': lab,
            # 'a2plot': ss[par].values if par is not None else None,
        })

        for i, idx in enumerate(agent_idx):
            ii = Nidx * jj + i
            id = c.agent_ids[idx]
            ss = s.xs(id, level='AgentID', drop_level=True).loc[:Nticks]
            # ee = e.loc[id]

            # length = np.round(ee['length'] * 1000, 2)
            # cum_sd = np.round(ee[preg.getPar('cum_sd')], 2)
            # run_tr = int(ee[preg.getPar('run_tr')] * 100)
            # title = f'{l}  # {idx} track, l : {length} mm, pathlength {cum_sd}xl , {run_tr}% time crawling'
            title=get_title(idx,c,e,l)
            kws1 = aux.AttrDict({
                'agent_idx': idx,
                'a': get_a(ss),
                'axs': P.axs[ii],
                'a2plot': get_a2plot(ss),
                **kws0
            })

            track_annotated(**kws1)
            P.conf_ax(ii, xvis=True if ii == P.Nrows - 1 else False, ylab=lab, title=title)
    P.adjust((0.1, 0.98), (0.05, 0.95), 0.001, 0.2)
    P.fig.align_ylabels(P.axs[:])
    return P.get()

@reg.funcs.graph('stride track')
def annotated_strideplot_data(**kwargs):
    return track_annotated_data(epoch='stride', **kwargs)

@reg.funcs.graph('turn track')
def annotated_turnplot_data(**kwargs):
    return track_annotated_data(epoch='turn', **kwargs)

@reg.funcs.graph('marked strides')
def plot_marked_strides(agent_idx=0, agent_id=None, slice=[20, 40], subfolder='individuals', **kwargs):
    temp = f'marked_strides_{slice[0]}-{slice[1]}' if slice is not None else f'marked_strides'
    name = f'{temp}_{agent_id}' if agent_id is not None else f'{temp}_{agent_idx}'
    P = plot.Plot(name=name, subfolder=subfolder, **kwargs)
    Nds = P.Ndatasets

    chunks = ['stride', 'pause']
    chunk_cols = ['lightblue', 'grey']
    p, ylab = reg.getPar('sv', to_return=['d', 'l'])
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

@reg.funcs.graph('sample tracks')
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
    P = plot.Plot(name=name, subfolder=subfolder, **kwargs)
    Nds = P.Ndatasets

    figx = 15 * 6 * 3 if slice is None else int((t1 - t0) / 3)
    figy = 5

    P.build(Nrows, Nds, figsize=(figx * Nds, figy * Nrows), sharey=False, sharex=True)
    for ii, (l, d) in enumerate(P.data_dict.items()):
        for jj, key in enumerate(mode):
            kk = ii + Nrows * jj
            ax = P.axs[kk]
            if key == 'strides':
                chunks = ['stride', 'pause']
                chunk_cols = ['lightblue', 'grey']

                p, ylab, ylim = reg.getPar('sv', to_return=['d', 'l', 'lim'])
                ylim = (0.0, 1.0)
            elif key == 'turns':
                chunks = ['Rturn', 'Lturn']
                chunk_cols = ['lightgreen', 'orange']

                b = 'bend'
                bv = nam.vel(b)
                ho = nam.orient('front')
                hov = nam.vel(ho)
                p, ylab, ylim = reg.getPar('fov', to_return=['d', 'l', 'lim'])

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


if __name__ == '__main__':





    refID = 'exploration.dish'
    d = reg.loadRef(refID)
    d.load()
    s, e,c = d.step_data,d.endpoint_data, d.config
    def get_title(idx):
        l=d.id

        id = c.agent_ids[idx]
        ss = s.xs(id, level='AgentID', drop_level=True)
        ee = e.loc[id]

        length = np.round(ee['length'] * 1000, 2)
        cum_sd = np.round(ee[reg.getPar('cum_sd')], 2)
        run_tr = int(ee[reg.getPar('run_tr')] * 100)
        title = f'{l}  # {idx} track, l : {length} mm, pathlength {cum_sd}xl , {run_tr}% time crawling'
        return title,ss

    P = plot.AutoPlot(name='name', subfolder=None, show=True,
                 build_kws={'Nrows': 5, 'Ncols': 2, 'w': 20, 'h':6, 'mode':'box'}, datasets=[d])

    # epoch = 'turn'

    f0 = epoch_func()
    # print(0, f0, type(f0))

    # f1 = f0(epoch=epoch)

    # print(1, f1, type(f1))


    for ii in range(P.Nrows):
        title, ss = get_title(idx=ii)
        for jj,ep in enumerate(['stride', 'turn']):
            f1 = f0(epoch=ep)
            f2 = f1(ss=ss, min_amp=30)
            f3 = f2(P=P, i=2*ii+jj, title=title)

     # print(3, f3, type(f3))

    P.get()
