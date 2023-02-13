import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from larvaworld.lib.aux import naming as nam
from larvaworld.lib import reg, aux, util

def register_bout_distros(c,e):
    from larvaworld.lib.model.modules.intermitter import get_EEB_poly1d
    try:
        c['intermitter'] = {
            nam.freq('crawl'): e[nam.freq(nam.scal(nam.vel('')))].mean(),
            nam.freq('feed'): e[nam.freq('feed')].mean() if nam.freq('feed') in e.columns else 2.0,
            'dt': c.dt,
            'crawl_bouts': True,
            'feed_bouts': True,
            'stridechain_dist': c.bout_distros.run_count,
            'pause_dist': c.bout_distros.pause_dur,
            'run_dist': c.bout_distros.run_dur,
            'feeder_reoccurence_rate': None,
        }
        c['EEB_poly1d'] = get_EEB_poly1d(**c['intermitter']).c.tolist()
    except :
        pass



def process_epochs(a, epochs, dt, return_idx=True):
    if epochs.shape[0] == 0:
        stops = []
        durs = np.array([])
        slices = []
        amps = np.array([])
        idx = []  # np.array([])
        maxs = np.array([])
        if return_idx:
            return stops, durs, slices, amps, idx, maxs
        else:
            return durs, amps, maxs

    else:
        if epochs.shape == (2,):
            epochs = np.array([epochs, ])
        durs = (np.diff(epochs).flatten()) * dt
        slices = [np.arange(r0, r1, 1) for r0, r1 in epochs]
        amps = np.array([np.trapz(a[p][~np.isnan(a[p])], dx=dt) for p in slices])
        maxs = np.array([np.max(a[p]) for p in slices])
        if return_idx:
            stops = epochs[:, 1]
            idx = np.concatenate(slices) if len(slices) > 1 else slices[0]
            return stops, durs, slices, amps, idx, maxs
        else:
            return durs, amps, maxs


def detect_pauses(a, dt, vel_thr=0.3, runs=None, min_dur=None):
    """
    Annotates crawl-pauses in timeseries.

    Extended description of function.

    Parameters
    ----------
    a : array
        1D np.array : forward velocity timeseries
    dt : float
        Timestep of the timeseries
    vel_thr : float
        Maximum velocity threshold
    runs : list
        A list of pairs of the start-end indices of the runs.
        If provided pauses that overlap with runs will be excluded.
    min_dur : float, optional
        The minimum required duration for a turn

    Returns
    -------
    pauses : list
        A list of pairs of the start-end indices of the pauses.

    """
    idx = np.where(a <= vel_thr)[0]
    if runs is not None:
        for r0, r1 in runs:
            idx = idx[(idx <= r0) | (idx >= r1)]
    pauses = detect_epochs(idx, dt, min_dur)
    return pauses


def detect_epochs(idx, dt, min_dur=None):
    if min_dur is None:
        min_dur = 2 * dt
    p0s = idx[np.where(np.diff(idx, prepend=[-np.inf]) != 1)[0]]
    p1s = idx[np.where(np.diff(idx, append=[np.inf]) != 1)[0]]
    epochs = np.vstack([p0s, p1s]).T
    durs = (np.diff(epochs).flatten()) * dt
    return epochs[durs >= min_dur]


def detect_runs(a, dt, vel_thr=0.3, min_dur=0.5):
    """
    Annotates crawl-runs in timeseries.

    Extended description of function.

    Parameters
    ----------
    a : array
        1D np.array : forward velocity timeseries
    dt : float
        Timestep of the timeseries
    vel_thr : float
        Maximum velocity threshold
     min_dur : float, optional
        The minimum required duration for a turn

    Returns
    -------
    runs : list
        A list of pairs of the start-end indices of the runs.


    """
    idx = np.where(a >= vel_thr)[0]
    runs = detect_epochs(idx, dt, min_dur)
    return runs


def detect_strides(a, dt, vel_thr=0.3, stretch=(0.75, 2.0), fr=None, return_extrema=True, return_runs=True):
    """
    Annotates strides-runs and pauses in timeseries.

    Extended description of function.

    Parameters
    ----------
    a : array
        1D np.array : forward velocity timeseries
    dt : float
        Timestep of the timeseries
    vel_thr : float
        Maximum velocity threshold
    stretch : Tuple[float,float]
        The min-max stretch of a stride relative to the default derived from the dominnt frequency
    fr : float, optional
        The dominant crawling frequency.
    return_extrema : boolean
        Whether to additionally return the stride extrema
    return_runs : boolean
        Whether to additionally return the runs (stridechains)

    Returns
    -------
    strides : list
        A list of pairs of the start-end indices of the strides.
    i_min : array
        Indices of the local minima.
    i_max : array
        Indices of the local maxima
    runs : list
         A list of pairs of the start-end indices of the runs/stridechains.
    run_counts : list
         Stride-counts of the runs/stridechains.

    """
    if fr is None:
        fr = aux.fft_max(a, dt, fr_range=(1, 2.5))
    tmin = stretch[0] // (fr * dt)
    tmax = stretch[1] // (fr * dt)
    i_min = find_peaks(-a, height=-3 * vel_thr, distance=tmin)[0]
    i_max = find_peaks(a, height=vel_thr, distance=tmin)[0]
    strides = []
    for m in i_max:
        try:
            s0, s1 = [i_min[i_min < m][-1], i_min[i_min > m][0]]
            if ((s1 - s0) <= tmax) and ([s0, s1] not in strides):
                strides.append([s0, s1])
        except:
            pass
    strides = np.array(strides)
    if not return_runs:
        if return_extrema:
            return i_min, i_max, strides
        else:
            return strides

    runs, run_counts = [], []
    s00, s11 = None, None

    count = 0
    for ii, (s0, s1) in enumerate(strides.tolist()):
        if ii == 0:
            s00, s11 = s0, s1
            count = 1
            continue
        if s11 == s0:
            s11 = s1
            count += 1
        else:
            runs.append([s00, s11])
            run_counts.append(count)
            count = 1
            s00, s11 = s0, s1
        if ii == len(strides) - 1:
            runs.append([s00, s11])
            run_counts.append(count)
            break
    runs = np.array(runs)
    if return_extrema:
        return i_min, i_max, strides, runs, run_counts
    else:
        return strides, runs, run_counts


def detect_turns(a, dt, min_dur=None):
    """
    Annotates turns in timeseries.

    Extended description of function.

    Parameters
    ----------
    a : array
        1D np.array : angular velocity timeseries
    dt : float
        Timestep of the timeseries
    min_dur : float, optional
        The minimum required duration for a turn

    Returns
    -------
    Lturns : list
        A list of pairs of the start-end indices of the Left turns.
    Rturns : list
        A list of pairs of the start-end indices of the Right turns.


    """
    if type(a) != pd.core.series.Series:
        a = pd.Series(a)
    if min_dur is None:
        min_dur = 2 * dt
    i_zeros = np.where(np.sign(a).diff().ne(0) == True)[0]
    Rturns, Lturns = [], []
    for s0, s1 in zip(i_zeros[:-1], i_zeros[1:]):
        if (s1 - s0) <= 2:
            continue
        elif np.isnan(np.sum(a[s0:s1])):
            continue
        else:
            if all(a[s0:s1] >= 0):
                Lturns.append([s0, s1])
            elif all(a[s0:s1] <= 0):
                Rturns.append([s0, s1])
    Lturns = np.array(Lturns)
    Rturns = np.array(Rturns)
    Ldurs = (np.diff(Lturns).flatten()) * dt
    Rdurs = (np.diff(Rturns).flatten()) * dt
    return Lturns[Ldurs >= min_dur], Rturns[Rdurs >= min_dur]


def weathervanesNheadcasts(run_idx, pause_idx, turn_slices, Tamps):
    wvane_idx = [ii for ii, t in enumerate(turn_slices) if all([tt in run_idx for tt in t])]
    cast_idx = [ii for ii, t in enumerate(turn_slices) if all([tt in pause_idx for tt in t])]
    wvane_amps = Tamps[wvane_idx]
    cast_amps = Tamps[cast_idx]
    wvane_min, wvane_max = np.nanquantile(wvane_amps, 0.25), np.nanquantile(wvane_amps, 0.75)
    cast_min, cast_max = np.nanquantile(cast_amps, 0.25), np.nanquantile(cast_amps, 0.75)
    return wvane_min, wvane_max, cast_min, cast_max


def comp_chunk_dicts(s, e, c, vel_thr=0.3, strides_enabled=True, store=False, **kwargs):
    aux.fft_freqs(s, e, c)
    turn_dict = turn_annotation(s, e, c, store=store)
    crawl_dict = crawl_annotation(s, e, c, strides_enabled=strides_enabled, vel_thr=vel_thr, store=store)
    chunk_dicts = aux.AttrDict({id: {**turn_dict[id], **crawl_dict[id]} for id in c.agent_ids})
    return chunk_dicts

@reg.funcs.annotation("bout_distribution")
def bout_distribution(s, e, c, d, **kwargs) :
    d.grouped_epochs = aux.group_epoch_dicts(d.chunk_dicts)
    d.pooled_epochs = util.fit_epochs(d.grouped_epochs)
    c.bout_distros = util.get_bout_distros(d.pooled_epochs)
    register_bout_distros(c, e)

@reg.funcs.annotation("bout_detection")
def bout_detection(s, e, c, d, store=False, **kwargs):
    d.chunk_dicts = comp_chunk_dicts(s, e, c, store=store, **kwargs)
    turn_mode_annotation(e, d.chunk_dicts)


def stride_interp(a, strides, Nbins=64):
    x = np.linspace(0, 2 * np.pi, Nbins)
    aa = np.zeros([strides.shape[0], Nbins])
    for ii, (s0, s1) in enumerate(strides):
        aa[ii, :] = np.interp(x, np.linspace(0, 2 * np.pi, s1 - s0), a[s0:s1])
    return aa


def mean_stride_curve(a, strides, da, Nbins=64):
    aa = stride_interp(a, strides, Nbins)
    aa_minus = aa[da < 0]
    aa_plus = aa[da > 0]
    aa_norm = np.vstack([aa_plus, -aa_minus])
    dic = aux.AttrDict({
        'abs': np.nanquantile(np.abs(aa), q=0.5, axis=0).tolist(),
        'plus': np.nanquantile(aa_plus, q=0.5, axis=0).tolist(),
        'minus': np.nanquantile(aa_minus, q=0.5, axis=0).tolist(),
        'norm': np.nanquantile(aa_norm, q=0.5, axis=0).tolist(),
    })

    return dic



def cycle_curve_dict(s, dt, shs=['sv', 'fov', 'rov', 'foa', 'b']):
    strides = detect_strides(s[reg.getPar('sv')], dt, return_extrema=False, return_runs=False)
    da = np.array([np.trapz(s[reg.getPar('fov')][s0:s1].dropna()) for ii, (s0, s1) in enumerate(strides)])
    dic = {sh: mean_stride_curve(s[reg.getPar(sh)], strides, da) for sh in shs}
    return aux.AttrDict(dic)


def cycle_curve_dict_multi(s, dt, shs=['sv', 'fov', 'rov', 'foa', 'b']):

    ids = s.index.unique('AgentID').values
    dic={}
    for id in ids:
        ss = s.xs(id, level="AgentID").dropna()
        dic[id]=cycle_curve_dict(ss, dt=dt, shs=shs)
    return aux.AttrDict(dic)

@reg.funcs.annotation("interference")
def compute_interference_data(s, e, c, d, Nbins=64, **kwargs) :
    d.cycle_curves = compute_interference(s=s, e=e, c=c, chunk_dicts=d.chunk_dicts, Nbins=Nbins)

@reg.funcs.annotation("interference2")
def compute_interference(s, e, c, Nbins=64, chunk_dicts=None):
    x = np.linspace(0, 2 * np.pi, Nbins)

    sss = {id: s.xs(id, level="AgentID") for id in c.agent_ids}

    if chunk_dicts is None:
        stride_dic = {}

        stride_dic_dfo = {}
        for jj, id in enumerate(c.agent_ids):
            ss = sss[id]
            stride_dic[id] = detect_strides(ss[reg.getPar('sv')].values, c.dt, return_runs=False, return_extrema=False)
            a_fov = ss[reg.getPar('fov')].values
            stride_dic_dfo[id] = np.array([np.trapz(a_fov[s0:s1]) for ii, (s0, s1) in enumerate(stride_dic[id])])
    else:
        stride_dic = {id: chunk_dicts[id]['stride'] for id in c.agent_ids}
        stride_dic_dfo = {id: chunk_dicts[id]['stride_Dor'] for id in c.agent_ids}

    pooled_curves = {}
    cycle_curves = {}
    mean_curves_abs = {}
    for sh in ['sv', 'fov', 'rov', 'foa', 'b']:
        par = reg.getPar(sh)
        curves_abs = np.zeros([c.N, Nbins]) * np.nan
        curves_plus = np.zeros([c.N, Nbins]) * np.nan
        curves_minus = np.zeros([c.N, Nbins]) * np.nan
        curves_norm = np.zeros([c.N, Nbins]) * np.nan
        for jj, id in enumerate(c.agent_ids):
            ss = sss[id]
            aa = stride_interp(ss[par].values, stride_dic[id], Nbins=64)
            aa_plus = aa[stride_dic_dfo[id] > 0]
            aa_minus = aa[stride_dic_dfo[id] < 0]
            aa_norm = np.vstack([aa_plus, -aa_minus])
            curves_abs[jj, :] = np.nanquantile(np.abs(aa), q=0.5, axis=0)
            curves_plus[jj, :] = np.nanquantile(aa_plus, q=0.5, axis=0)
            curves_minus[jj, :] = np.nanquantile(aa_minus, q=0.5, axis=0)
            curves_norm[jj, :] = np.nanquantile(aa_norm, q=0.5, axis=0)
        mean_curves_abs[sh] = curves_abs
        cycle_curves[sh] = aux.AttrDict({
            'abs': curves_abs,
            'plus': curves_plus,
            'minus': curves_minus,
            'norm': curves_norm,
        })
        pooled_curves[sh] = aux.AttrDict({
            'abs': np.nanquantile(curves_abs, q=0.5, axis=0).tolist(),
            'plus': np.nanquantile(curves_plus, q=0.5, axis=0).tolist(),
            'minus': np.nanquantile(curves_minus, q=0.5, axis=0).tolist(),
            'norm': np.nanquantile(curves_norm, q=0.5, axis=0).tolist(),
        })

    att0s, att1s = np.min(mean_curves_abs['fov'], axis=1), np.max(mean_curves_abs['fov'], axis=1)

    e[aux.nam.max('phi_attenuation')] = x[np.argmax(mean_curves_abs['fov'], axis=1)]
    e[aux.nam.max(f'phi_{reg.getPar("sv")}')] = x[np.argmax(mean_curves_abs['sv'], axis=1)]
    e[reg.getPar('str_sv_max')] = np.max(mean_curves_abs['sv'], axis=1)
    try:
        e['attenuation'] = att0s / e[reg.getPar('pau_fov_mu')]
        e[aux.nam.max('attenuation')] = (att1s - att0s) / e[reg.getPar('pau_fov_mu')]
    except:
        pass

    c.pooled_cycle_curves = pooled_curves
    return cycle_curves

@reg.funcs.annotation("turn_mode")
def turn_mode_annotation(e, chunk_dicts):
    wNh = {}
    wNh_ps = ['weathervane_q25_amp', 'weathervane_q75_amp', 'headcast_q25_amp', 'headcast_q75_amp']
    for jj, id in enumerate(e.index.values):
        dic = chunk_dicts[id]
        wNh[id] = dict(zip(wNh_ps, weathervanesNheadcasts(dic.run_idx, dic.pause_idx, dic.turn_slice, dic.turn_amp)))
    e[wNh_ps] = pd.DataFrame.from_dict(wNh).T

@reg.funcs.annotation("turn")
def turn_annotation(s, e, c, store=False):
    ids = s.index.unique('AgentID').values
    N = s.index.unique('Step').size

    # reg.vprint((len(ids), c.N),3)
    # reg.vprint([id for id in ids if id not in e.index.values.tolist()])


    fov, foa = reg.getPar(['fov', 'foa'])

    eTur_ps = reg.getPar( ['Ltur_N', 'Rtur_N', 'tur_N', 'tur_H'])
    eTur_vs = np.zeros([len(ids), len(eTur_ps)]) * np.nan
    turn_ps = reg.getPar(['tur_fou', 'tur_t','Ltur_t','Rtur_t', 'tur_fov_max'])
    turn_vs = np.zeros([N, len(ids), len(turn_ps)]) * np.nan
    turn_dict = {}

    for jj, id in enumerate(ids):
        a_fov = s[fov].xs(id, level="AgentID")
        Lturns, Rturns = detect_turns(a_fov, c.dt)

        Lturns1, Ldurs, Lturn_slices, Lamps, Lturn_idx, Lmaxs = process_epochs(a_fov.values, Lturns, c.dt, return_idx=True)
        Rturns1, Rdurs, Rturn_slices, Ramps, Rturn_idx, Rmaxs = process_epochs(a_fov.values, Rturns, c.dt, return_idx=True)
        Lturns_N,Rturns_N = Lturns.shape[0],Rturns.shape[0]
        turns_N=Lturns_N+Rturns_N
        tur_H=Lturns_N/turns_N if turns_N!=0 else 0
        Tamps = np.concatenate([Lamps, Ramps])
        Tdurs = np.concatenate([Ldurs, Rdurs])
        Tmaxs = np.concatenate([Lmaxs, Rmaxs])
        Tslices = Lturn_slices + Rturn_slices
        if Lturns_N > 0:
            turn_vs[Lturns[:, 1], jj, 0] = Lamps
            turn_vs[Lturns[:, 1], jj, 1] = Ldurs
            turn_vs[Lturns[:, 1], jj, 2] = Ldurs
            turn_vs[Lturns[:, 1], jj, 4] = Lmaxs
        if Rturns_N > 0:
            turn_vs[Rturns[:, 1], jj, 0] = Ramps
            turn_vs[Rturns[:, 1], jj, 1] = Rdurs
            turn_vs[Rturns[:, 1], jj, 3] = Rdurs
            turn_vs[Rturns[:, 1], jj, 4] = Rmaxs
        turn_dict[id] = {'Lturn': Lturns, 'Rturn': Rturns, 'turn_slice': Tslices, 'turn_amp': Tamps,
                         'turn_dur': Tdurs, 'Lturn_dur': Ldurs, 'Rturn_dur': Rdurs, 'turn_vel_max': Tmaxs}
        eTur_vs[jj, :] = [Lturns_N,Rturns_N, turns_N, tur_H]
    s[turn_ps] = turn_vs.reshape([N * len(ids), len(turn_ps)])
    e[eTur_ps] = eTur_vs
    if store:
        aux.store_distros(s, pars=reg.getPar(['tur_fou', 'tur_t', 'tur_fov_max']), parent_dir=c.dir)
    return turn_dict


@reg.funcs.annotation("crawl")
def crawl_annotation(s, e, c, strides_enabled=True, vel_thr=0.3, store=False):
    l, v, sv, dst, acc, fov, foa, b, bv, ba, fv = \
        reg.getPar(['l', 'v', 'sv', 'd', 'a', 'fov', 'foa', 'b', 'bv', 'ba', 'fv'])

    str_ps = reg.getPar(['str_d_mu', 'str_d_std', 'str_sv_mu', 'str_fov_mu', 'str_fov_std', 'str_N'])
    lin_ps = reg.getPar(
        ['run_v_mu', 'pau_v_mu', 'run_a_mu', 'pau_a_mu', 'run_fov_mu', 'run_fov_std', 'pau_fov_mu', 'pau_fov_std',
         'run_foa_mu', 'pau_foa_mu', 'pau_b_mu', 'pau_b_std', 'pau_bv_mu', 'pau_bv_std', 'pau_ba_mu', 'pau_ba_std',
         'cum_run_t', 'cum_pau_t', 'run_t_min', 'run_t_max', 'pau_t_min', 'pau_t_max'])

    dt = c.dt
    lin_vs = np.zeros([c.N, len(lin_ps)]) * np.nan
    str_vs = np.zeros([c.N, len(str_ps)]) * np.nan

    run_ps = reg.getPar(['pau_t', 'run_t', 'run_d', 'str_c_l', 'str_t', 'str_d', 'str_sd'])
    run_vs = np.zeros([c.Nticks, c.N, len(run_ps)]) * np.nan

    crawl_dict = {}

    for jj, id in enumerate(c.agent_ids):
        strides, str_chain_ls, stride_Dor, stride_durs, stride_dsts, stride_sdsts, strides1 = [], [], [], [], [], [], []
        a_v = s[v].xs(id, level="AgentID").values
        a_fov = s[fov].xs(id, level="AgentID").values

        if c.Npoints > 1:
            a_sv = s[sv].xs(id, level="AgentID").values
            if strides_enabled:
                strides, runs, str_chain_ls = detect_strides(a_sv, dt, fr=e[fv].loc[id], vel_thr=vel_thr,
                                                             return_extrema=False)
                strides1, stride_durs, stride_slices, stride_dsts, stride_idx, stride_maxs = process_epochs(a_v,
                                                                                                            strides, dt)
                stride_sdsts = stride_dsts / e[l].loc[id]
                stride_Dor = np.array([np.trapz(a_fov[s0:s1 + 1]) for s0, s1 in strides])
                str_fovs = np.abs(a_fov[stride_idx])
                str_vs[jj, :] = [np.nanmean(stride_dsts),
                                 np.nanstd(stride_dsts),
                                 np.nanmean(a_sv[stride_idx]),
                                 np.nanmean(str_fovs),
                                 np.nanstd(str_fovs),
                                 np.nansum(str_chain_ls),
                                 ]
            else:
                runs = detect_runs(a_sv, dt, vel_thr=vel_thr)
            pauses = detect_pauses(a_sv, dt, vel_thr=vel_thr, runs=runs)
        else:

            runs = detect_runs(a_v, dt, vel_thr=vel_thr)
            pauses = detect_pauses(a_v, dt, runs=runs, vel_thr=vel_thr)

        pauses1, pause_durs, pause_slices, pause_dsts, pause_idx, pause_maxs = process_epochs(a_v, pauses, dt)
        runs1, run_durs, run_slices, run_dsts, run_idx, run_maxs = process_epochs(a_v, runs, dt)

        run_vs[pauses1, jj, 0] = pause_durs
        run_vs[runs1, jj, 1] = run_durs
        run_vs[runs1, jj, 2] = run_dsts
        run_vs[runs1, jj, 3] = str_chain_ls
        run_vs[strides1, jj, 4] = stride_durs
        run_vs[strides1, jj, 5] = stride_dsts
        run_vs[strides1, jj, 6] = stride_sdsts

        if b in s.columns:
            pau_bs = s[b].xs(id, level="AgentID").abs().values[pause_idx]
            pau_bvs = s[bv].xs(id, level="AgentID").abs().values[pause_idx]
            pau_bas = s[ba].xs(id, level="AgentID").abs().values[pause_idx]
            pau_b_temp = [np.mean(pau_bs), np.std(pau_bs), np.mean(pau_bvs), np.std(pau_bvs), np.mean(pau_bas),
                          np.std(pau_bas)]
        else:
            pau_b_temp = [np.nan] * 6
        a_foa = s[foa].xs(id, level="AgentID").abs().values
        a_acc = s[acc].xs(id, level="AgentID").values
        pau_fovs = np.abs(a_fov[pause_idx])
        run_fovs = np.abs(a_fov[run_idx])
        pau_foas = a_foa[pause_idx]
        run_foas = a_foa[run_idx]
        lin_vs[jj, :] = [
            np.mean(a_v[run_idx]),
            np.mean(a_v[pause_idx]),
            np.mean(a_acc[run_idx]),
            np.mean(a_acc[pause_idx]),
            np.mean(run_fovs), np.std(run_fovs),
            np.mean(pau_fovs), np.std(pau_fovs),
            np.mean(run_foas),
            np.mean(pau_foas),
            *pau_b_temp,
            np.sum(run_durs),
            np.sum(pause_durs),
            np.nanmin(run_durs) if len(run_durs) > 0 else 1,
            np.nanmax(run_durs) if len(run_durs) > 0 else 100,
            np.nanmin(pause_durs) if len(pause_durs) > 0 else dt,
            np.nanmax(pause_durs) if len(pause_durs) > 0 else 100,
        ]
        crawl_dict[id] = {'stride': strides, 'stride_Dor': stride_Dor, 'exec': runs, 'pause': pauses,
                          'run_idx': run_idx, 'pause_idx': pause_idx, 'stride_dur': stride_durs,
                          'run_count': str_chain_ls, 'run_dur': run_durs, 'run_dst': run_dsts, 'pause_dur': pause_durs}
    s[run_ps] = run_vs.reshape([c.Nticks * c.N, len(run_ps)])
    e[lin_ps] = lin_vs

    str_d_mu, str_d_std, str_sd_mu, str_sd_std, run_tr, pau_tr, cum_run_t, cum_pau_t, cum_t = \
        reg.getPar(
            ['str_d_mu', 'str_d_std', 'str_sd_mu', 'str_sd_std', 'run_tr', 'pau_tr', 'cum_run_t', 'cum_pau_t', 'cum_t'])

    e[run_tr] = e[cum_run_t] / e[cum_t]
    e[pau_tr] = e[cum_pau_t] / e[cum_t]

    if c.Npoints > 1 and strides_enabled:
        e[str_ps] = str_vs
        e[str_sd_mu] = e[str_d_mu] / e[l]
        e[str_sd_std] = e[str_d_std] / e[l]
    if store:
        run_ps = reg.getPar(['pau_t', 'run_t', 'run_d', 'str_c_l', 'str_d', 'str_sd'])
        aux.store_distros(s, pars=run_ps, parent_dir=c.dir)
    return crawl_dict


def track_par_in_chunk(d, chunk, par):
    s, c = d.step_data, d.config
    A = np.zeros([c.Nticks, c.N, 3]) * np.nan
    for i, id in enumerate(c.agent_ids):
        epochs = d.chunk_dicts[id][chunk]
        ss = s[par].xs(id, level='AgentID')
        if epochs.shape[0] > 0:
            t0s, t1s = epochs[:, 0], epochs[:, 1]
            b0s = ss.loc[t0s].values
            b1s = ss.loc[t1s].values
            A[t0s, i, 0] = b0s
            A[t1s, i, 1] = b1s
            A[t1s, i, 2] = b1s - b0s
    s[aux.nam.at(par, aux.nam.start(chunk)), aux.nam.at(par, aux.nam.stop(chunk)), aux.nam.chunk_track(chunk, par)] = A.reshape([c.Nticks * c.N, 3])
