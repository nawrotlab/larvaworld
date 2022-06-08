import pandas as pd
from scipy.spatial.distance import euclidean
import numpy as np

from lib.aux.ang_aux import angle_to_x_axis
import lib.aux.naming as nam
from lib.process.spatial import scale_to_length
from lib.process.store import store_aux_dataset

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

def compute_chunk_metrics(s, e, chunks):
    for c in chunks:
        dur = nam.dur(c)
        N = nam.num(c)
        e[N] = s[dur].groupby('AgentID').count()
        e[nam.cum(dur)] = s[dur].groupby('AgentID').sum()
        e[nam.mean(dur)] = s[dur].groupby('AgentID').mean()
        e[nam.std(dur)] = s[dur].groupby('AgentID').std()
        e[nam.dur_ratio(c)] = e[nam.cum(dur)] / e[nam.cum('dur')]
        e[nam.mean(N)] = e[N] / e[nam.cum('dur')]


def detect_contacting_chunks(s, e, c, aux_dir, dt, vel_par, track_point, chunk='stride', mid_flag=None, edge_flag=None,
                             control_pars=[], vel_threshold=0.0):
    ids = s.index.unique('AgentID').values
    Nids = len(ids)
    t0 = int(s.index.unique('Step').min())
    N = len(s.index.unique('Step'))


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
    A_c0s = np.zeros([N, Nids]) * np.nan
    A_c1s = np.zeros([N, Nids]) * np.nan
    A_cl = np.zeros([N, Nids]) * np.nan
    A_cdur = np.zeros([N, Nids]) * np.nan
    A_d = np.zeros([N, Nids]) * np.nan
    A_sd = np.zeros([N, Nids]) * np.nan

    arrays = [A_s0s, A_s1s, A_dur, A_id,
              A_c0s, A_c1s,A_cl, A_cdur,
              A_d, A_sd, A_or]

    chain=nam.chain(chunk)
    pars = [nam.start(chunk), nam.stop(chunk), nam.dur(chunk), nam.id(chunk),
            nam.start(chain), nam.stop(chain), nam.dur(chain),nam.length(chain),

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
        d_dst = d[D0].values
        d_xy = d[XY0].values
        guide=d[vel_par]
        v_edges=guide.loc[edges]
        v_mids=guide.loc[mids]

        chunks = np.array([[a, b] for a, b, va, vb in zip(edges[:-1], edges[1:],v_edges[:-1],v_edges[1:] ) if (b - a >= 0.5 * t)
                           and (b - a <= 2.0 * t)
                           and set(np.arange(a, b + 1)) <= set(valid)
                           and (any((m > a) and (m < b) and (np.abs(vm-va)>vel_threshold) and (np.abs(vm-vb)>vel_threshold) for m, vm in zip(mids, v_mids)))
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
            else :
                A_c0s[s0, i] = True
            A_s0s[s0, i] = True
            A_s1s[s1, i] = True
            A_id[s0:s1 + 1, i] = j
            chain_counter += 1
            chain_dur_counter += dur
            if contact == 0:
                A_cl[s1 + 1, i] = chain_counter
                A_cdur[s1 + 1, i] = chain_dur_counter
                A_c1s[s1 + 1, i] = True
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
    e['stride_reoccurence_rate'] = 1 - 1 / s[nam.length(chain)].groupby('AgentID').mean()
    pars = [c_dst, c_sdst]
    pars = pars + nam.cum(pars) + nam.mean(pars) + nam.std(pars)
    scale_to_length(s, e, c, pars=pars)
    compute_chunk_metrics(s, e, [chunk])

    store_aux_dataset(s, pars=[nam.dur(chain),nam.length(chain)], type='distro', file=aux_dir)
    print('All chunks-around-flag detected')


def comp_chunk_bearing(s, c, chunk, **kwargs):
    from lib.process.aux import comp_bearing

    c0 = nam.start(chunk)
    c1 = nam.stop(chunk)
    ho = nam.unwrap(nam.orient('front'))
    ho0s = s[nam.at(ho, c0)].dropna().values
    ho1s = s[nam.at(ho, c1)].dropna().values
    for n, pos in c.sources.items():
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
        store_aux_dataset(s, pars=[b0_par, b1_par, db_par], type='distro', file=c.aux_dir)
        print(f'Bearing to source {n} during {chunk} computed')


def comp_patch_metrics(s, e, **kwargs):
    cum_t = nam.cum('dur')
    on = 'on_food'
    off = 'off_food'
    on_tr = nam.dur_ratio(on)
    for c in ['Lturn', 'turn', 'pause']:
        dur = nam.dur(c)
        cdur = nam.cum(dur)
        N = nam.num(c)
        e[f'{N}_{on}'] = s[s[on] == True][dur].groupby('AgentID').count()
        e[f'{N}_{off}'] = s[s[on] == False][dur].groupby('AgentID').count()

        e[f'{cdur}_{on}'] = s[s[on] == True][dur].groupby('AgentID').sum()
        e[f'{cdur}_{off}'] = s[s[on] == False][dur].groupby('AgentID').sum()

        e[f'{nam.dur_ratio(c)}_{on}'] = e[f'{cdur}_{on}'] / e[cum_t] / e[on_tr]
        e[f'{nam.dur_ratio(c)}_{off}'] = e[f'{cdur}_{off}'] / e[cum_t] / (1 - e[on_tr])
        e[f'{nam.mean(N)}_{on}'] = e[f'{N}_{on}'] / e[cum_t] / e[on_tr]
        e[f'{nam.mean(N)}_{off}'] = e[f'{N}_{off}'] / e[cum_t] / (1 - e[on_tr])

    dst = nam.dst('')
    cdst = nam.cum(dst)
    v_mu = nam.mean(nam.vel(''))
    e[f'{cdst}_{on}'] = s[s[on] == True][dst].dropna().groupby('AgentID').sum()
    e[f'{cdst}_{off}'] = s[s[on] == False][dst].dropna().groupby('AgentID').sum()

    e[f'{v_mu}_{on}'] = e[f'{cdst}_{on}'] / e[cum_t] / e[on_tr]
    e[f'{v_mu}_{off}'] = e[f'{cdst}_{off}'] / e[cum_t] / (1 - e[on_tr])
    e['handedness_score'] = e[nam.num('Lturn')] / e[nam.num('turn')]
    e[f'handedness_score_{on}'] = e[f"{nam.num('Lturn')}_{on}"] / e[f"{nam.num('turn')}_{on}"]
    e[f'handedness_score_{off}'] = e[f"{nam.num('Lturn')}_{off}"] / e[f"{nam.num('turn')}_{off}"]

def get_stride_df(s,e,c,shorts=['sv', 'b','bv','fov','rov'], idx=0, Nbins=64):
    # from lib.process.aux import comp_bearing
    from lib.process.aux import detect_strides

    from lib.conf.base.pars import getPar, ParDict

    id = c.agent_ids[idx]
    ee = e.loc[id]
    ss = s.xs(id, level='AgentID')
    pars= getPar(shorts)
    Npars=len(pars)
    strides = detect_strides(ss[getPar('sv')], c.dt, fr=ee[getPar('fv')], return_runs=False, return_extrema=False)
    strides = strides.tolist()
    pi2 = 2 * np.pi
    x = np.linspace(0, pi2, Nbins)
    my_index = pd.MultiIndex.from_product([np.arange(len(strides)), np.arange(Nbins)], names=['Stride', 'Step'])
    df = pd.DataFrame(columns=pars,index=my_index)
    for j, par in enumerate(pars):
        aa = np.zeros([len(strides), Nbins]) * np.nan
        ssp=ss[par].values
        for ii, (s0, s1) in enumerate(strides):
            aa[ii, :] = np.interp(x, np.linspace(0, pi2, s1 - s0), ssp[s0:s1])
        df[par]=aa.flatten()
    return df