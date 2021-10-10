from scipy.spatial.distance import euclidean
import numpy as np

from lib.process.aux import suppress_stdout, comp_bearing
from lib.aux.ang_aux import angle_to_x_axis
import lib.aux.naming as nam
from lib.anal.fitting import fit_bouts
from lib.process.basic import comp_extrema, compute_freq
from lib.process.spatial import scale_to_length
from lib.process.store import store_aux_dataset


def annotate(s, e, config=None, bouts={'stride': True, 'pause': True, 'turn': True},
             recompute=False, track_point=None, track_pars=None, chunk_pars=None,
             vel_par=None, ang_vel_par=None, bend_vel_par=None, min_ang=5.0, min_ang_vel=100.0,
             non_chunks=False, source=None, show_output=True, **kwargs):
    from lib.conf.base.par import ParDict
    dic = ParDict(mode='load').dict
    if vel_par is None:
        vel_par = dic['sv']['d']
    if ang_vel_par is None:
        ang_vel_par = dic['fov']['d']
    if bend_vel_par is None:
        bend_vel_par = dic['bv']['d']
    if track_pars is None:
        track_pars = [dic[k]['d'] for k in ['fou', 'rou', 'fo', 'ro', 'b', 'x', 'y', 'o_cent', 'o_chem']]
    if chunk_pars is None:
        chunk_pars = [dic[k]['d'] for k in ['sv', 'fov', 'rov', 'bv', 'l']]
    track_pars = [p for p in track_pars if p in s.columns]
    if track_point is None:
        track_point = config['point']
    if min_ang is None:
        min_ang = 0.0
    if min_ang_vel is None:
        min_ang_vel = 0.0
    c = {
        's': s,
        'e': e,
        'dt': config['dt'],
        'Npoints': config['Npoints'],
        'track_point': track_point,
        'track_pars': track_pars,
        'config': config,
        'aux_dir': f'{config["dir"]}/data/aux.h5',
        'recompute': recompute,
    }
    with suppress_stdout(show_output):
        if bouts['stride']:
            detect_strides(**c, non_chunks=non_chunks, vel_par=vel_par, chunk_pars=chunk_pars, **kwargs)
        if bouts['pause']:
            detect_pauses(**c, vel_par=vel_par, **kwargs)
        if bouts['turn']:
            detect_turns(**c, ang_vel_par=ang_vel_par, bend_vel_par=bend_vel_par, min_ang=min_ang,
                         min_ang_vel=min_ang_vel, **kwargs)

        if bouts['stride'] and bouts['pause']:
            fit_bouts(**c, **kwargs)
        for b in bouts.keys():
            if bouts[b]:
                comp_chunk_bearing(**c, chunk=b, **kwargs)
                if b=='turn' :
                    comp_chunk_bearing(**c, chunk='Lturn', **kwargs)
                    comp_chunk_bearing(**c, chunk='Rturn', **kwargs)
    return s, e


def detect_turns(s, e, aux_dir, dt, track_pars, min_ang_vel, min_ang=30.0,
                 ang_vel_par=None, bend_vel_par=None, chunk_only=None, recompute=False,
                 constant_bend_chunks=False, **kwargs):
    if set(nam.num(['Lturn', 'Rturn'])).issubset(e.columns.values) and not recompute:
        print('Turns are already detected. If you want to recompute it, set recompute_turns to True')
        return
    ss = s.loc[s[nam.id(chunk_only)].dropna().index] if chunk_only is not None else s

    if ang_vel_par is None:
        ang_vel_par = nam.vel(nam.orient('front'))
    if bend_vel_par is None:
        bend_vel_par = nam.vel('bend')

    # comp_extrema(ss, dt, parameters=[ang_vel_par], interval_in_sec=0.3, abs_threshold=[-min_ang_vel, min_ang_vel])
    # detect_turn_bouts(ss, e, dt, par=ang_vel_par)
    print(min_ang)
    detect_chunks(ss, e, dt, chunk_names=['Lturn', 'Rturn'], chunk_only=chunk_only, par=ang_vel_par,
                  ROU_ranges=[[min_ang, np.inf], [-np.inf, -min_ang]],
                  par_ranges=[[min_ang_vel, np.inf], [-np.inf, -min_ang_vel]], merged_chunk='turn',
                  store_max=[True, False], store_min=[False, True])
    track_pars_in_chunks(ss, e, aux_dir, chunks=['Lturn', 'Rturn'], pars=track_pars, merged_chunk='turn')
    if constant_bend_chunks:
        print('Additionally detecting constant bend chunks.')
        detect_chunks(ss, e, dt, chunk_names=['constant_bend'], chunk_only=chunk_only, par=bend_vel_par,
                      par_ranges=[[-min_ang_vel, min_ang_vel]])
    print('All turns detected')


def detect_pauses(s, e, aux_dir, config, dt, track_pars, recompute=False, stride_non_overlap=True, vel_par=None,
                  min_dur=0.4, **kwargs):
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

    track_pars_in_chunks(s, e, aux_dir, chunks=[c], pars=track_pars)

    store_aux_dataset(s, pars=[nam.dur(c)], type='distro', file=aux_dir)
    print('All crawl-pauses detected')


def detect_strides(s, e, aux_dir, config, dt, recompute=False, vel_par=None, track_point=None, track_pars=None,
                   chunk_pars=[], non_chunks=False, **kwargs):
    c = 'stride'
    if nam.num(c) in e.columns.values and not recompute:
        print('Strides are already detected. If you want to recompute it, set recompute to True')
        return

    sv_thr = config['scaled_vel_threshold'] if config is not None else 0.3
    if vel_par is None:
        vel_par = nam.scal(nam.vel(''))
    mid_flag = nam.max(vel_par)
    edge_flag = nam.min(vel_par)

    comp_extrema(s, dt, parameters=[vel_par], interval_in_sec=0.3, abs_threshold=[np.inf, sv_thr])
    compute_freq(s, e, dt, parameters=[vel_par], freq_range=[0.7, 1.8])
    detect_contacting_chunks(s, e, aux_dir, dt, mid_flag=mid_flag, edge_flag=edge_flag,
                             vel_par=vel_par, control_pars=track_pars,
                             track_point=track_point)
    if non_chunks:
        detect_non_chunks(s, e, dt, chunk_name=c, guide_parameter=vel_par)
    track_pars_in_chunks(s, e, aux_dir, chunks=[c], pars=track_pars)

    store_aux_dataset(s, pars=chunk_pars, type='stride', file=aux_dir)
    print('All strides detected')


# def detect_turn_bouts(s, e, dt, par):
#     ids = s.index.unique('AgentID').values
#     Nids = len(ids)
#     output = f'Detecting chunks-on-condition for {Nids} agents'
#     N = len(s.index.unique('Step'))
#     t0 = int(s.index.unique('Step').min())
#     pos_flag = nam.max(par)
#     neg_flag = nam.min(par)
#     cs, c0 = ['Lturn', 'Rturn'], 'turn'
#     dic = {}
#     for c in cs:
#         c_ps = [nam.start(c), nam.stop(c), nam.id(c), nam.dur(c)]
#         dic[c] = {pp: np.zeros([N, Nids]) * np.nan for pp in c_ps}
#
#     for i, id in enumerate(ids):
#         sss = s.xs(id, level='AgentID', drop_level=True)
#         idx0 = np.unique(sign_changes(sss, par).index.values.astype(int))
#         for c, flag in zip(cs, [pos_flag, neg_flag]):
#             idxM = np.unique(sss[sss[flag] == True].index.values.astype(int))
#             s0s, s1s = [], []
#             for jj, ii in enumerate(idxM):
#                 try:
#                     s0, s1 = idx0[idx0 < ii][-1], idx0[idx0 > ii][0]
#                     if s0 not in s0s and s1 not in s1s:
#                         s0s.append(s0)
#                         s1s.append(s1)
#                 except:
#                     pass
#
#             s0s = np.array(s0s)
#             s1s = np.array(s1s)
#             ds = (s1s - s0s) * dt
#             dic[c][nam.start(c)][s0s - t0, i] = True
#             dic[c][nam.stop(c)][s1s - t0, i] = True
#             dic[c][nam.dur(c)][s1s - t0, i] = ds
#             for j, (s0, s1) in enumerate(zip(s0s, s1s)):
#                 dic[c][nam.id(c)][s0 - t0:s1 + 1 - t0, i] = j
#     for c in cs:
#         for p, a in dic[c].items():
#             s[p] = a.flatten()
#     comp_merged_chunk(s, c0, cs)
#     compute_chunk_metrics(s, e, cs + [c0])
#
#     print(output)
#     print('All chunks-on-condition detected')


def comp_merged_chunk(s, c0, cs, pars=[], e=None):
    ps = []
    mc0, mc1, mcdur = nam.start(c0), nam.stop(c0), nam.dur(c0)
    cs0, cs1, csdur = nam.start(cs), nam.stop(cs), nam.dur(cs)
    s[mcdur] = s[csdur].sum(axis=1, min_count=1)
    s[mc0] = s[cs0].sum(axis=1, min_count=1)
    s[mc1] = s[cs1].sum(axis=1, min_count=1)
    ps += [mc0, mc1, mcdur]
    for p in pars:
        p_mc0, p_mc1, p_mc = nam.at(p, mc0), nam.at(p, mc1), nam.chunk_track(c0, p)
        p_cs = [nam.chunk_track(c, p) for c in cs]
        s[p_mc0] = s[nam.at(p, cs0)].sum(axis=1, min_count=1)
        s[p_mc1] = s[nam.at(p, cs1)].sum(axis=1, min_count=1)
        s[p_mc] = s[p_cs].sum(axis=1, min_count=1)
        e[nam.mean(p_mc)] = s[p_cs].abs().groupby('AgentID').mean().mean(axis=1)
        e[nam.std(p_mc)] = s[p_cs].abs().groupby('AgentID').std().mean(axis=1)
        ps += [p_mc0, p_mc1, p_mc]
    return ps


def detect_chunks(s, e, dt, chunk_names, par, chunk_only=None, par_ranges=[[-np.inf, np.inf]],
                  ROU_ranges=[[-np.inf, np.inf]],
                  non_overlap_chunk=None, merged_chunk=None, store_min=[False], store_max=[False],
                  min_dur=0.0):
    cs, c0 = chunk_names, merged_chunk
    ids = s.index.unique('AgentID').values
    Nids = len(ids)
    output = f'Detecting chunks-on-condition for {Nids} agents'
    N = len(s.index.unique('Step'))
    t0 = int(s.index.unique('Step').min())
    if min_dur == 0.0:
        min_dur = dt
    ss = s.loc[s[nam.id(chunk_only)].dropna().index] if chunk_only is not None else s

    def agent_data(id):
        if non_overlap_chunk is None:
            return ss[par].xs(id, level='AgentID', drop_level=True)
        else:
            d = ss.xs(id, level='AgentID', drop_level=True)
            return d[d[nam.id(non_overlap_chunk)].isna()][par]

    for c, (Vmin, Vmax), (Amin, Amax), storMin, storMax in zip(cs, par_ranges, ROU_ranges, store_min, store_max):
        c_ps = [S0, S1, Id, Dur, Max, Min] = [nam.start(c), nam.stop(c), nam.id(c), nam.dur(c),
                                              nam.max(nam.chunk_track(c, par)), nam.min(nam.chunk_track(c, par))]
        dic = {pp: np.zeros([N, Nids]) * np.nan for pp in c_ps}

        for i, id in enumerate(ids):
            d = agent_data(id)
            ii0 = d[(d < Vmax) & (d > Vmin)].index
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

            dic[S0][s0s - t0, i] = True
            dic[S1][s1s - t0, i] = True
            dic[Dur][s1s - t0, i] = ds
            for j, (s0, s1) in enumerate(zip(s0s, s1s)):
                dic[Id][s0 - t0:s1 + 1 - t0, i] = j
                if storMax:
                    dic[Max][s1 - t0, i] = s.loc[(slice(s0, s1), id), par].max()
                if storMin:
                    dic[Min][s1 - t0, i] = s.loc[(slice(s0, s1), id), par].min()

        for p, a in dic.items():
            s[p] = a.flatten()
    compute_chunk_metrics(s, e, cs)
    if c0 is not None:
        comp_merged_chunk(s, c0, cs)
        compute_chunk_metrics(s, e, [c0])
    print(output)
    print('All chunks-on-condition detected')


def detect_non_chunks(s, e, dt, chunk_name, guide_parameter, non_chunk_name=None, min_dur=0.0):
    ids = s.index.unique('AgentID').values
    Nids = len(ids)
    Nticks = len(s.index.unique('Step'))
    t0 = int(s.index.unique('Step').min())
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
        e[nam.num(c)] = s[dur].groupby('AgentID').count()
        e[nam.cum(dur)] = s[dur].groupby('AgentID').sum()
        e[nam.mean(dur)] = s[dur].groupby('AgentID').mean()
        e[nam.std(dur)] = s[dur].groupby('AgentID').std()
        e[nam.dur_ratio(c)] = e[nam.cum(dur)] / e[nam.cum('dur')]


def detect_contacting_chunks(s, e, aux_dir, dt, vel_par, track_point, chunk='stride', mid_flag=None, edge_flag=None,
                             control_pars=[]):
    ids = s.index.unique('AgentID').values
    Nids = len(ids)
    t0 = int(s.index.unique('Step').min())
    N = len(s.index.unique('Step'))

    c_chain_dur = nam.dur(nam.chain(chunk))
    c_chain_l = nam.length(nam.chain(chunk))
    c_dst = nam.dst(chunk)
    c_sdst = nam.straight_dst(chunk)

    if track_point in [None, 'centroid', -1]:
        XY0 = ['x', 'y']
        D0 = 'dst' if 'dst' in s.columns else 'distance'
    else:
        XY0 = nam.xy(track_point)
        D0 = nam.dst(track_point)
    cpars = [D0] + XY0 + control_pars
    cpars = [p for p in cpars if p in s.columns]

    A_s0s = np.zeros([N, Nids]) * np.nan
    A_s1s = np.zeros([N, Nids]) * np.nan
    A_dur = np.zeros([N, Nids]) * np.nan
    A_or = np.zeros([N, Nids]) * np.nan
    A_id = np.zeros([N, Nids]) * np.nan
    A_cl = np.zeros([N, Nids]) * np.nan
    A_cdur = np.zeros([N, Nids]) * np.nan
    A_d = np.zeros([N, Nids]) * np.nan
    A_sd = np.zeros([N, Nids]) * np.nan

    arrays = [A_s0s, A_s1s, A_dur, A_id,
              A_cl, A_cdur,
              A_d, A_sd, A_or]

    pars = [nam.start(chunk), nam.stop(chunk), nam.dur(chunk), nam.id(chunk),
            c_chain_l, c_chain_dur,
            c_dst, c_sdst, nam.orient(chunk)]

    freqs = e[nam.freq(vel_par)]
    freqs.fillna(value=freqs.mean(), inplace=True)
    if edge_flag is None:
        edge_flag = nam.min(vel_par)
    if mid_flag is None:
        mid_flag = nam.max(vel_par)

    for i, id in enumerate(ids):
        t = 1 / freqs[id] / dt

        d = s.xs(id, level='AgentID', drop_level=True)
        edges = d[d[edge_flag] == True].index.values
        mids = d[d[mid_flag] == True].index.values
        valid = d[cpars].dropna().index.values
        # print(id, edges, mids, valid)
        d_dst = d[D0].values
        d_xy = d[XY0].values

        chunks = np.array([[a, b] for a, b in zip(edges[:-1], edges[1:]) if (b - a >= 0.5 * t)
                           and (b - a <= 2.0 * t)
                           and set(np.arange(a, b + 1)) <= set(valid)
                           and (any((m > a) and (m < b) for m in mids))
                           ]).astype(int)
        if len(chunks) == 0:
            continue

        durs = np.diff(chunks, axis=1)[:, 0] * dt
        contacts = [int(ss11 == ss02) for (ss01, ss11), (ss02, ss12) in zip(chunks[:-1, :], chunks[1:, :])] + [0]
        s0s = chunks[:, 0] - t0
        s1s = chunks[:, 1] - t0
        A_dur[s1s, i] = durs
        chain_counter = 0
        chain_dur_counter = 0
        for j, (s0, s1, dur, contact) in enumerate(zip(s0s, s1s, durs, contacts)):
            if chain_counter > 0:
                s0 += 1
            A_s0s[s0, i] = True
            A_s1s[s1, i] = True
            A_id[s0:s1 + 1, i] = j
            chain_counter += 1
            chain_dur_counter += dur
            if contact == 0:
                A_cl[s1 + 1, i] = chain_counter
                A_cdur[s1 + 1, i] = chain_dur_counter
                chain_counter = 0
                chain_dur_counter = 0

            A_d[s1, i] = np.sum(d_dst[s0 + 1: s1])
            A_sd[s1, i] = euclidean(tuple(d_xy[s1, :]), tuple(d_xy[s0, :]))
            A_or[s1, i] = angle_to_x_axis(d_xy[s0], d_xy[s1])
    for a, p in zip(arrays, pars):
        s[p] = a.flatten()
    for pp in [c_dst, c_sdst]:
        e[nam.cum(pp)] = s[pp].groupby('AgentID').sum()
        e[nam.mean(pp)] = s[pp].groupby('AgentID').mean()
        e[nam.std(pp)] = s[pp].groupby('AgentID').std()
    e['stride_reoccurence_rate'] = 1 - 1 / s[c_chain_l].groupby('AgentID').mean()
    pars = [c_dst, c_sdst]
    pars = pars + nam.cum(pars) + nam.mean(pars) + nam.std(pars)
    scale_to_length(s, e, pars=pars)
    compute_chunk_metrics(s, e, [chunk])

    store_aux_dataset(s, pars=[c_chain_dur, c_chain_l], type='distro', file=aux_dir)
    print('All chunks-around-flag detected')


def track_pars_in_chunks(s, e, aux_dir, chunks, pars, mode='dif', merged_chunk=None):
    ids = s.index.unique('AgentID').values
    Nids = len(ids)
    Nticks = len(s.index.unique('Step'))
    t0 = int(s.index.unique('Step').min())
    all_d = [s.xs(id, level='AgentID', drop_level=True) for id in ids]
    p_aux = []
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
        p_aux = p_aux + p_tracks + p0s + p1s

    if merged_chunk is not None:
        p_aux_m = comp_merged_chunk(s, merged_chunk, chunks, pars, e=e)
        p_aux += p_aux_m

    store_aux_dataset(s, pars=p_aux, type='distro', file=aux_dir)
    print('All parameters tracked')


def comp_chunk_bearing(s, config, aux_dir, chunk, **kwargs):
    c0 = nam.start(chunk)
    c1 = nam.stop(chunk)
    ho = nam.unwrap(nam.orient('front'))
    ho0s = s[nam.at(ho, c0)].dropna().values
    ho1s = s[nam.at(ho, c1)].dropna().values
    for n, pos in config['sources'].items():
        b = nam.bearing2(n)
        b0_par = nam.at(b, c0)
        b1_par = nam.at(b, c1)
        db_par = nam.chunk_track(chunk, b)
        b0 = comp_bearing(s[nam.at('x', c0)].dropna().values, s[nam.at('y', c0)].dropna().values, ho0s, loc=pos)
        b1 = comp_bearing(s[nam.at('x', c1)].dropna().values, s[nam.at('y', c1)].dropna().values, ho1s, loc=pos)
        s[b0_par] = np.nan
        s.loc[s[c0] == True, b0_par] = b0
        s[b1_par] = np.nan
        s.loc[s[c1] == True, b1_par] = b1
        s[db_par] = np.nan
        s.loc[s[c1] == True, db_par] = np.abs(b0) - np.abs(b1)
        store_aux_dataset(s, pars=[b0_par, b1_par, db_par], type='distro', file=aux_dir)
        print(f'Bearing to source {n} during {chunk} computed')
