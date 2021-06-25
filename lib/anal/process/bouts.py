import itertools

from scipy.spatial.distance import euclidean
from sklearn.metrics.pairwise import nan_euclidean_distances
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema, spectrogram

import lib.aux.functions as fun
import lib.aux.naming as nam
import lib.conf.dtype_dicts as dtypes
from lib.anal.process.basic import compute_extrema, compute_freq
from lib.anal.process.store import create_par_distro_dataset, create_chunk_dataset
from lib.conf.par import load_ParDict
from lib.stor import paths


def detect_bouts(s,e,dt,Npoints,point, config=None, bouts=['stride', 'pause', 'turn'],
                 recompute=False, track_point=None, track_pars=None,chunk_pars=None,
                 vel_par=None, ang_vel_par=None, bend_vel_par=None,min_ang=5.0,
                 non_chunks=False,distro_dir=None,stride_p_dir=None,source=None, show_output=True, **kwargs):

    if paths.new_format :
        dic = load_ParDict()
        if vel_par is None :
            vel_par=dic['sv']['d']
        if ang_vel_par is None :
            ang_vel_par=dic['fov']['d']
        if bend_vel_par is None :
            bend_vel_par=dic['bv']['d']
        if track_pars is None:
            track_pars = [dic[k]['d'] for k in ['fou', 'rou','fo', 'ro', 'b', 'x', 'y', 'o_cent']]
        if chunk_pars is None :
            chunk_pars=[dic[k]['d'] for k in ['sv', 'fov', 'rov', 'bv', 'l']]
        # if get_unit(dic['fo'].d).name()=='radian' :
        min_ang=np.deg2rad(min_ang)
        # raise
    else :
        if track_pars is None:
            track_pars = [nam.unwrap(nam.orient('front')),nam.orient('front'), nam.unwrap(nam.orient('rear')),nam.orient('rear'), 'bend', 'x', 'y',
                          nam.bearing2('center')]
        if chunk_pars is None :
            chunk_pars=[vel_par, 'spinelength', nam.vel(nam.orient('front')),nam.vel(nam.orient('rear')),nam.vel('bend')]
    track_pars = [p for p in track_pars if p in s.columns]
    if track_point is None:
        track_point = point
    c = {
        's': s,
        'e': e,
        'dt': dt,
        'Npoints': Npoints,
        'track_point': track_point,
        'track_pars': track_pars,
        'config': config,
        'recompute': recompute,
        'distro_dir': distro_dir,
        'stride_p_dir': stride_p_dir,
    }
    with fun.suppress_stdout(show_output):
        if 'stride' in bouts:
            detect_strides(**c, non_chunks=non_chunks,vel_par=vel_par,chunk_pars=chunk_pars, **kwargs)
        if 'pause' in bouts:
            detect_pauses(**c,vel_par=vel_par, **kwargs)
        if 'turn' in bouts:
            detect_turns(**c,ang_vel_par=ang_vel_par, bend_vel_par=bend_vel_par,min_ang=min_ang, **kwargs)
        if source is not None :
            for b in bouts :
                compute_chunk_bearing2source(s, b, source=source, distro_dir=distro_dir)
    return s,e



def detect_turns(s, e, dt, track_pars, recompute=False, min_ang_vel=0.0, min_ang=5.0,
                 ang_vel_par=None, bend_vel_par=None,chunk_only=None,
                 constant_bend_chunks=False, distro_dir=None, **kwargs):
    if set(nam.num(['Lturn', 'Rturn'])).issubset(e.columns.values) and not recompute:
        print('Turns are already detected. If you want to recompute it, set recompute_turns to True')
        return
    ss = s.loc[s[nam.id(chunk_only)].dropna().index] if chunk_only is not None else s

    if ang_vel_par is None :
        ang_vel_par = nam.vel(nam.orient('front'))
    if bend_vel_par is None :
        bend_vel_par = nam.vel('bend')


    detect_chunks(s,e,dt,chunk_names=['Lturn', 'Rturn'], chunk_only=chunk_only, par=ang_vel_par,
                       ROU_ranges=[[min_ang, np.inf], [-np.inf, -min_ang]],
                       par_ranges=[[min_ang_vel, np.inf], [-np.inf, -min_ang_vel]], merged_chunk='turn',
                       store_max=[True, False], store_min=[False, True])
    track_pars_in_chunks(s,e,chunks=['Lturn', 'Rturn'], pars=track_pars, merged_chunk='turn', distro_dir=distro_dir)
    if constant_bend_chunks:
        print('Additionally detecting constant bend chunks.')
        detect_chunks(s,e,dt,chunk_names=['constant_bend'], chunk_only=chunk_only, par=bend_vel_par,
                           par_ranges=[[-min_ang_vel, min_ang_vel]])
    print('All turns detected')

def detect_pauses(s, e, dt, track_pars,config=None, recompute=False, stride_non_overlap=True, vel_par=None, min_dur=0.4,
                  distro_dir=None, **kwargs):
    c = 'pause'
    if nam.num(c) in e.columns.values and not recompute:
        print('Pauses are already detected. If you want to recompute it, set recompute to True')
        return

    sv_thr = config['scaled_vel_threshold'] if config is not None else 0.3
    par_range = [-np.inf, sv_thr]
    if vel_par is None:
        vel_par = nam.scal(nam.vel(''))
    non_overlap_chunk = 'stride' if stride_non_overlap else None

    detect_chunks(s, e, dt, chunk_names=[c], par=vel_par, par_ranges=[par_range],
                  non_overlap_chunk=non_overlap_chunk, min_dur=min_dur)

    track_pars_in_chunks(s, e, chunks=[c], pars=track_pars, distro_dir=distro_dir)

    if distro_dir is not None:
        create_par_distro_dataset(s, [nam.dur(c)], distro_dir)
    print('All crawl-pauses detected')


def detect_strides(s, e, dt, config=None, recompute=False, vel_par=None, track_point=None, track_pars=None,
                   chunk_pars=[],non_chunks=False, distro_dir=None, stride_p_dir=None, **kwargs):

    c = 'stride'
    if nam.num(c) in e.columns.values and not recompute:
        print('Strides are already detected. If you want to recompute it, set recompute to True')
        return

    sv_thr = config['scaled_vel_threshold'] if config is not None else 0.3
    if vel_par is None:
        vel_par = nam.scal(nam.vel(''))
    mid_flag = nam.max(vel_par)
    edge_flag = nam.min(vel_par)

    compute_extrema(s, dt, parameters=[vel_par], interval_in_sec=0.3, abs_threshold=[np.inf, sv_thr])
    compute_freq(s, e, dt, parameters=[vel_par], freq_range=[0.7, 1.8])
    detect_contacting_chunks(s, e, dt, chunk=c, mid_flag=mid_flag, edge_flag=edge_flag,
                             vel_par=vel_par, control_pars=track_pars,
                             track_point=track_point, distro_dir=distro_dir)
    if non_chunks:
        detect_non_chunks(s, e, dt, chunk_name=c, guide_parameter=vel_par)
    track_pars_in_chunks(s, e, chunks=[c], pars=track_pars,distro_dir=distro_dir)

    if stride_p_dir is not None:
        create_chunk_dataset(s, c, pars=chunk_pars, dir=stride_p_dir)
    if distro_dir is not None:
        create_par_distro_dataset(s, [nam.dur(c)], distro_dir)

    print('All strides detected')


def detect_chunks(s, e, dt, chunk_names, par, chunk_only=None, par_ranges=[[-np.inf, np.inf]],
                  ROU_ranges=[[-np.inf, np.inf]],
                  non_overlap_chunk=None, merged_chunk=None, store_min=[False], store_max=[False],
                  min_dur=0.0):
    ids = s.index.unique('AgentID').values
    Nids = len(ids)
    output = f'Detecting chunks-on-condition for {Nids} agents'
    N = len(s.index.unique('Step'))
    t0 = s.index.unique('Step').min()
    if min_dur == 0.0:
        min_dur = dt
    ss = s.loc[s[nam.id(chunk_only)].dropna().index] if chunk_only is not None else s
    if non_overlap_chunk is not None:
        non_ov_id = nam.id(non_overlap_chunk)
        data = [ss.xs(id, level='AgentID', drop_level=True) for id in ids]
        data = [d[d[non_ov_id].isna()] for d in data]
        data = [d[par] for d in data]
    else:
        data = [ss[par].xs(id, level='AgentID', drop_level=True) for id in ids]

    for c, (Vmin, Vmax), (Amin, Amax), storMin, storMax in zip(chunk_names, par_ranges, ROU_ranges, store_min,
                                                               store_max):
        S0 = np.zeros([N, Nids]) * np.nan
        S1 = np.zeros([N, Nids]) * np.nan
        Dur = np.zeros([N, Nids]) * np.nan
        Id = np.zeros([N, Nids]) * np.nan
        Max = np.zeros([N, Nids]) * np.nan
        Min = np.zeros([N, Nids]) * np.nan

        p_s0 = nam.start(c)
        p_s1 = nam.stop(c)
        p_id = nam.id(c)
        p_dur = nam.dur(c)
        p_max = nam.max(nam.chunk_track(c, par))
        p_min = nam.min(nam.chunk_track(c, par))
        for i, (id, d) in enumerate(zip(ids, data)):

            ii0 = d[(d < Vmax) & (d > Vmin)].index
            # ii0=np.unique(np.hstack([ii00,ii00[np.where(np.diff(ii00, prepend=[-np.inf]) == 2)[0]]+1]))
            s0s = ii0[np.where(np.diff(ii0, prepend=[-np.inf]) != 1)[0]]
            s1s = ii0[np.where(np.diff(ii0, append=[np.inf]) != 1)[0]]
            ROUs = np.array([np.trapz(d.loc[slice(s0, s1)].values) * dt for s0, s1 in zip(s0s, s1s)])
            s0s = s0s[(ROUs <= Amax) & (ROUs >= Amin)]
            s1s = s1s[(ROUs <= Amax) & (ROUs >= Amin)]

            ds = (s1s - s0s) * dt
            ii1 = np.where(ds >= min_dur)
            ds = ds[ii1]
            s0s = s0s[ii1].values.astype(int)
            s1s = s1s[ii1].values.astype(int)

            S0[s0s - t0, i] = True
            S1[s1s - t0, i] = True
            Dur[s1s - t0, i] = ds
            for j, (s0, s1) in enumerate(zip(s0s, s1s)):
                Id[s0 - t0:s1 + 1 - t0, i] = j
                if storMax:
                    Max[s1 - t0, i] = s.loc[(slice(s0, s1), id), par].max()
                if storMin:
                    Min[s1 - t0, i] = s.loc[(slice(s0, s1), id), par].min()

        for a, p in zip([S0, S1, Dur, Id, Max, Min], [p_s0, p_s1, p_dur, p_id, p_max, p_min]):
            a = a.flatten()
            # print(p,a)
            s[p] = a
    compute_chunk_metrics(s, e, chunk_names)
    if merged_chunk is not None:
        mc0, mc1, mcdur = nam.start(merged_chunk), nam.stop(merged_chunk), nam.dur(merged_chunk)
        s[mcdur] = s[[nam.dur(c) for c in chunk_names]].sum(axis=1, min_count=1)
        s[mc0] = s[[nam.start(c) for c in chunk_names]].sum(axis=1, min_count=1)
        s[mc1] = s[[nam.stop(c) for c in chunk_names]].sum(axis=1, min_count=1)
        compute_chunk_metrics(s, e, [merged_chunk])

    print(output)
    print('All chunks-on-condition detected')


def detect_non_chunks(s, e, dt, chunk_name, guide_parameter, non_chunk_name=None, min_dur=0.0):
    ids = s.index.unique('AgentID').values
    Nids = len(ids)
    Nticks = len(s.index.unique('Step'))
    t0 = s.index.unique('Step').min()
    min_dur = int(min_dur / dt)
    chunk_id = nam.id(chunk_name)
    if non_chunk_name is None:
        c = nam.non(chunk_name)
    else:
        c = non_chunk_name

    S0 = np.zeros([Nticks, Nids]) * np.nan
    S1 = np.zeros([Nticks, Nids]) * np.nan
    Dur = np.zeros([Nticks, Nids]) * np.nan
    Id = np.zeros([Nticks, Nids]) * np.nan

    p_s0 = nam.start(c)
    p_s1 = nam.stop(c)
    p_id = nam.id(c)
    p_dur = nam.dur(c)
    print(f'Detecting non-chunks for {Nids} agents')
    for j, id in enumerate(ids):
        d = s.xs(id, level='AgentID', drop_level=True)
        nonna = d[guide_parameter].dropna().index.values
        inval = d[chunk_id].dropna().index.values
        val = np.setdiff1d(nonna, inval, assume_unique=True)
        if len(val) == 0:
            print(f'No valid steps for {id}')
            continue
        s0s = np.sort([val[0]] + val[np.where(np.diff(val) > 1)[0] + 1].tolist())
        s1s = np.sort(val[np.where(np.diff(val) > 1)[0]].tolist() + [val[-1]])

        if len(s0s) != len(s1s):
            print('Number of start and stop indexes does not match')
            min = np.min([len(s0s), len(s1s)])
            s0s = s0s[:min]
            s1s = s1s[:min]
        v_s0s = []
        v_s1s = []
        for i, (s0, s1) in enumerate(zip(s0s, s1s)):
            if s0 > s1:
                print('Start index bigger than stop index.')
                continue
            elif s1 - s0 <= min_dur:
                continue
            elif d[guide_parameter].loc[slice(s0, s1)].isnull().values.any():
                continue
            else:
                v_s0s.append(s0)
                v_s1s.append(s1)
        v_s0s = np.array(v_s0s).astype(int)
        v_s1s = np.array(v_s1s).astype(int)
        S0[v_s0s - t0, j] = True
        S1[v_s1s - t0, j] = True
        Dur[v_s1s - t0, j] = (v_s1s - v_s0s) * dt
        for k, (v_s0, v_s1) in enumerate(zip(v_s0s, v_s1s)):
            Id[v_s0 - t0:v_s1 - t0, j] = k

    pars = [p_s0, p_s1, p_dur, p_id]
    arrays = [S0, S1, Dur, Id]
    for p, a in zip(pars, arrays):
        s[p] = a.flatten()

    compute_chunk_metrics(s, e, [c])

    print('All non-chunks detected')


def compute_chunk_metrics(s, e, chunks):
    for c in chunks:
        dur = nam.dur(c)
        e[nam.num(c)] = s[nam.stop(c)].groupby('AgentID').sum()
        e[nam.cum(dur)] = s[dur].groupby('AgentID').sum()
        e[nam.mean(dur)] = s[dur].groupby('AgentID').mean()
        e[nam.std(dur)] = s[dur].groupby('AgentID').std()
        e[nam.dur_ratio(c)] = e[nam.cum(dur)] / e[nam.cum('dur')]


def detect_contacting_chunks(s, e, dt, chunk='stride', track_point=None, mid_flag=None, edge_flag=None, control_pars=[],
                             vel_par=None, chunk_dur_in_sec=None, distro_dir=None):

    ids = s.index.unique('AgentID').values
    Nids = len(ids)
    t0 = s.index.unique('Step').min()
    N = len(s.index.unique('Step'))

    c_0 = nam.start(chunk)
    c_1 = nam.stop(chunk)
    c_id = nam.id(chunk)
    c_dur = nam.dur(chunk)
    c_or = nam.orient(chunk)
    c_chain = nam.chain(chunk)
    c_chain_dur = nam.dur(c_chain)
    c_chain_l = nam.length(c_chain)
    c_dst = nam.dst(chunk)
    c_sdst = nam.straight_dst(chunk)
    scaled_chunk_strdst = nam.scal(c_sdst)
    scaled_chunk_dst = nam.scal(c_dst)

    if track_point in [None, 'centroid']:
        track_xy = ['x', 'y']
        track_dst = 'dst' if 'dst' in s.columns else 'distance'
    else:
        track_xy = nam.xy(track_point)
        track_dst = nam.dst(track_point)

    if 'length' in e.columns.values:
        lengths = e['length'].values
    else:
        lengths = None

    cpars = [track_dst] + track_xy + control_pars
    cpars = [p for p in cpars if p in s.columns]

    start_array = np.zeros([N, Nids]) * np.nan
    stop_array = np.zeros([N, Nids]) * np.nan
    dur_array = np.zeros([N, Nids]) * np.nan
    orientation_array = np.zeros([N, Nids]) * np.nan
    id_array = np.zeros([N, Nids]) * np.nan
    chunk_chain_length_array = np.zeros([N, Nids]) * np.nan
    chunk_chain_dur_array = np.zeros([N, Nids]) * np.nan
    dst_array = np.zeros([N, Nids]) * np.nan
    straight_dst_array = np.zeros([N, Nids]) * np.nan
    scaled_dst_array = np.zeros([N, Nids]) * np.nan
    scaled_straight_dst_array = np.zeros([N, Nids]) * np.nan

    arrays = [start_array, stop_array, dur_array, id_array,
              chunk_chain_length_array, chunk_chain_dur_array,
              dst_array, straight_dst_array, scaled_dst_array, scaled_straight_dst_array, orientation_array]

    pars = [c_0, c_1, c_dur, c_id,
            c_chain_l, c_chain_dur,
            c_dst, c_sdst, scaled_chunk_dst, scaled_chunk_strdst, c_or]

    if vel_par:
        freqs = e[nam.freq(vel_par)]
        mean_freq = freqs.mean()
        print(f'Replacing {freqs.isna().sum()} nan values with population mean frequency : {mean_freq}')
        freqs.fillna(value=mean_freq, inplace=True)
        chunk_dur_in_ticks = {id: 1 / freqs[id] / dt for id in ids}
        cpars.append(vel_par)
        if edge_flag is None :
            edge_flag=nam.min(vel_par)
        if mid_flag is None :
            mid_flag=nam.max(vel_par)
    elif chunk_dur_in_sec:
        chunk_dur_in_ticks = {id: chunk_dur_in_sec / dt for id in ids}

    for i, id in enumerate(ids):
        t = chunk_dur_in_ticks[id]
        d = s.xs(id, level='AgentID', drop_level=True)
        edges = d[d[edge_flag] == True].index.values
        mids = d[d[mid_flag] == True].index.values
        valid = d[cpars].dropna().index.values

        d_dst = d[track_dst].values
        d_xy = d[track_xy].values

        chunks = np.array([[a, b] for a, b in zip(edges[:-1], edges[1:]) if (b - a >= 0.5 * t)
                           and (b - a <= 2.0 * t)
                           and set(np.arange(a, b + 1)) <= set(valid)
                           and (any((m > a) and (m < b) for m in mids))
                           ]).astype(int)
        Nchunks = len(chunks)
        if Nchunks != 0:
            durs = np.diff(chunks, axis=1)[:, 0] * dt
            contacts = [int(stop1 == start2) for (start1, stop1), (start2, stop2) in
                        zip(chunks[:-1, :], chunks[1:, :])] + [0]
            s0s = chunks[:, 0] - t0
            s1s = chunks[:, 1] - t0
            start_array[s0s, i] = True
            stop_array[s1s, i] = True
            dur_array[s1s, i] = durs
            chain_counter = 0
            chain_dur_counter = 0
            for j, (s0, s1, dur, contact) in enumerate(zip(s0s, s1s, durs, contacts)):
                if chain_counter > 0:
                    s0 += 1
                id_array[s0:s1 + 1, i] = j
                chain_counter += 1
                chain_dur_counter += dur
                if contact == 0:
                    chunk_chain_length_array[s1 + 1, i] = chain_counter
                    chunk_chain_dur_array[s1 + 1, i] = chain_dur_counter
                    chain_counter = 0
                    chain_dur_counter = 0

                dst_array[s1, i] = np.sum(d_dst[s0 + 1: s1])
                straight_dst_array[s1, i] = euclidean(tuple(d_xy[s1, :]), tuple(d_xy[s0, :]))
                orientation_array[s1, i] = fun.angle_to_x_axis(d_xy[s0], d_xy[s1])

            if lengths is not None:
                l = lengths[i]
                scaled_dst_array[s1s, i] = dst_array[s1s, i] / l
                scaled_straight_dst_array[s1s, i] = straight_dst_array[s1s, i] / l

    for array, par in zip(arrays, pars):
        s[par] = array.flatten()
    for pp in [c_dst, c_sdst]:
        e[nam.cum(pp)] = s[pp].groupby('AgentID').sum()
        e[nam.mean(pp)] = s[pp].groupby('AgentID').mean()
        e[nam.std(pp)] = s[pp].groupby('AgentID').std()

    e['stride_reoccurence_rate'] = 1 - 1 / s[c_chain_l].groupby('AgentID').mean()

    if 'length' in e.columns.values:
        for pp in [c_dst, c_sdst]:
            spp = nam.scal(pp)
            e[nam.cum(spp)] = e[nam.cum(pp)] / e['length']
            e[nam.mean(spp)] = e[nam.mean(pp)] / e['length']
            e[nam.std(spp)] = e[nam.std(pp)] / e['length']

    compute_chunk_metrics(s, e, [chunk])
    if distro_dir is not None:
        create_par_distro_dataset(s, [c_chain_dur, c_chain_l], dir=distro_dir)

    print('All chunks-around-flag detected')


def compute_chunk_overlap(s, e, base_chunk, overlapping_chunk):
    ids = s.index.unique('AgentID').values
    c0, c1 = base_chunk, overlapping_chunk
    print(f'Computing overlap of {c1} on {c0} chunks')
    p = nam.overlap_ratio(base_chunk=c0, overlapping_chunk=c1)
    s[p] = np.nan
    p0_id = nam.id(c0)
    p1_id = nam.id(c1)
    for id in ids:
        d = s.xs(id, level='AgentID', drop_level=True)
        inds = np.unique(d[p0_id].dropna().values)
        s0s = d[d[nam.stop(c0)] == True].index
        for i, s0 in zip(inds, s0s):
            d0 = d[d[p0_id] == i]
            d1 = d0[p1_id].dropna()
            s.loc[(s0, id), p] = len(d1) / len(d0)
    print('All chunk overlaps computed')
    return s[p].dropna().sum()


def track_pars_in_chunks(s, e, chunks, pars, mode='dif', merged_chunk=None, distro_dir=None):
    ids = s.index.unique('AgentID').values
    Nids = len(ids)
    Nticks = len(s.index.unique('Step'))
    t0 = s.index.unique('Step').min()
    all_d = [s.xs(id, level='AgentID', drop_level=True) for id in ids]

    for c in chunks:
        c0 = nam.start(c)
        c1 = nam.stop(c)

        all_0s = [d[d[c0] == True].index.values.astype(int) for d in all_d]
        all_1s = [d[d[c1] == True].index.values.astype(int) for d in all_d]

        if mode == 'dif':
            p_tracks = nam.chunk_track(c, pars)
        elif mode == 'max':
            p_tracks = nam.max(nam.chunk_track(c, pars))
        elif mode == 'min':
            p_tracks = nam.min(nam.chunk_track(c, pars))
        p0s = [nam.at(p, c0) for p in pars]
        p1s = [nam.at(p, c1) for p in pars]
        for p, p_track, p0, p1 in zip(pars, p_tracks, p0s, p1s):
            p_pars = [p_track, p0, p1]
            p_array = np.zeros([Nticks, Nids, len(p_pars)]) * np.nan
            ds = [d[p] for d in all_d]
            for k, (id, d, s0s, s1s) in enumerate(zip(ids, ds, all_0s, all_1s)):
                a0s = d[s0s].values
                a1s = d[s1s].values
                if mode == 'dif':
                    vs = a1s - a0s
                elif mode == 'max':
                    vs = [d[slice(s0, s1)].max() for s0, s1 in zip(s0s, s1s)]
                elif mode == 'min':
                    vs = [d[slice(s0, s1)].min() for s0, s1 in zip(s0s, s1s)]
                p_array[s1s - t0, k, 0] = vs
                p_array[s1s - t0, k, 1] = a0s
                p_array[s1s - t0, k, 2] = a1s
            for i, p in enumerate(p_pars):
                s[p] = p_array[:, :, i].flatten()
            e[nam.mean(p_track)] = s[p_track].groupby('AgentID').mean()
            e[nam.std(p_track)] = s[p_track].groupby('AgentID').std()

        if distro_dir is not None:
            create_par_distro_dataset(s, p_tracks + p0s + p1s, dir=distro_dir)
    if merged_chunk is not None:
        mc0, mc1, mcdur = nam.start(merged_chunk), nam.stop(merged_chunk), nam.dur(merged_chunk)
        for p in pars:
            p_mc0, p_mc1, p_mc = nam.at(p, mc0),nam.at(p, mc1), nam.chunk_track(merged_chunk, p)
            s[p_mc0] = s[[nam.at(p, nam.start(c)) for c in chunks]].sum(axis=1, min_count=1)
            s[p_mc1] = s[[nam.at(p, nam.stop(c)) for c in chunks]].sum(axis=1, min_count=1)
            s[p_mc] = s[[nam.chunk_track(c, p) for c in chunks]].sum(axis=1, min_count=1)

            if distro_dir is not None:
                create_par_distro_dataset(s, [p_mc0, p_mc1, mcdur, p_mc], dir=distro_dir)

            e[nam.mean(p_mc)] = s[[nam.chunk_track(c, p) for c in chunks]].abs().groupby('AgentID').mean().mean(
                axis=1)
            e[nam.std(p_mc)] = s[[nam.chunk_track(c, p) for c in chunks]].abs().groupby('AgentID').std().mean(
                axis=1)
    print('All parameters tracked')


def compute_chunk_bearing2source(s,chunk, source=(-50.0, 0.0), distro_dir=None):
    c0 = nam.start(chunk)
    c1 = nam.stop(chunk)
    ho = nam.unwrap(nam.orient('front'))
    ho0_par = f'{ho}_at_{c0}'
    ho1_par = f'{ho}_at_{c1}'

    x0_par = f'x_at_{c0}'
    x1_par = f'x_at_{c1}'

    y0_par = f'y_at_{c0}'
    y1_par = f'y_at_{c1}'

    b0_par = f'{nam.bearing2(source)}_at_{c0}'
    b1_par = f'{nam.bearing2(source)}_at_{c1}'
    db_par = f'{chunk}_{nam.bearing2(source)}_correction'

    b0 = fun.compute_bearing2source(s[x0_par].dropna().values, s[y0_par].dropna().values,
                                    s[ho0_par].dropna().values, loc=source, in_deg=True)
    b1 = fun.compute_bearing2source(s[x1_par].dropna().values, s[y1_par].dropna().values,
                                    s[ho1_par].dropna().values, loc=source, in_deg=True)
    s[b0_par] = np.nan
    s.loc[s[c0] == True, b0_par] = b0
    s[b1_par] = np.nan
    s.loc[s[c1] == True, b1_par] = b1
    s[db_par] = np.nan
    s.loc[s[c1] == True, db_par] = np.abs(b0) - np.abs(b1)
    if distro_dir is not None :
        create_par_distro_dataset(s,[b0_par, b1_par, db_par], dir=distro_dir)
    print(f'Bearing to source {source} during {chunk} computed')


if __name__ == '__main__':
    from lib.stor.managing import get_datasets
    d = get_datasets(datagroup_id='SimGroup', last_common='single_runs', names=['dish/wwr'], mode='load')[0]
    s = d.step_data
    d.detect_bouts(show_output=True)
