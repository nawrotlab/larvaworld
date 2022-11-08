import copy
import random

import numpy as np
import pandas as pd

from lib.aux import dictsNlists as dNl
# from lib.registry.pars import preg
from lib.registry import reg




def generate_larvae(N, sample_dict, base_model):
    if len(sample_dict) > 0:
        all_pars = []
        for i in range(N):
            dic= dNl.NestDict({p: vs[i] for p, vs in sample_dict.items()})
            all_pars.append(dNl.update_nestdict(dNl.copyDict(base_model), dic))
    else:
        all_pars = [base_model] * N
    return all_pars


def get_sample_bout_distros0(Im, bout_distros):
    dic = {
        'pause_dist': ['pause', 'pause_dur'],
        'stridechain_dist': ['stride', 'run_count'],
        'run_dist': ['exec', 'run_dur'],
    }

    ds = [ii for ii in ['pause_dist', 'stridechain_dist', 'run_dist'] if
          (ii in Im.keys()) and (Im[ii] is not None) and ('fit' in Im[ii].keys()) and (Im[ii]['fit'])]
    for d in ds:
        for sample_d in dic[d]:
            if sample_d in bout_distros.keys() and bout_distros[sample_d] is not None:
                Im[d] = bout_distros[sample_d]
    return Im


def get_sample_bout_distros(model, sample):
    if sample in [None, {}]:
        return model
    m = dNl.copyDict(model)
    if m.brain.intermitter_params:
        m.brain.intermitter_params = get_sample_bout_distros0(Im=m.brain.intermitter_params,
                                                              bout_distros=sample.bout_distros)

    return m


def sample_group(e, N=1, ps=[]):
    means = [e[p].mean() for p in ps]
    if len(ps) >= 2:
        base = e[ps].dropna().values.T
        cov = np.cov(base)
        vs = np.random.multivariate_normal(means, cov, N).T
    elif len(ps) == 1:
        std = np.std(e[ps].values)
        vs = np.atleast_2d(np.random.normal(means[0], std, N))
    else:
        return {}
    dic = {p: v for p, v in zip(ps, vs)}
    return dic


def get_sample_ks(m, sample_ks=None):
    if sample_ks is None:
        sample_ks = []
    modF = dNl.flatten_dict(m)
    sample_ks += [p for p in modF if modF[p] == 'sample']
    return sample_ks


def sampleRef(mID=None, m=None, refID=None, refDataset=None, sample_ks=None, Nids=1, parameter_dict={}):
    sample_dict = {}
    if m is None:
        m = preg.loadConf(id=mID, conftype="Model")
    ks = get_sample_ks(m, sample_ks=sample_ks)

    if len(ks) > 0:
        if refDataset is None:
            if refID is not None:
                refDataset = preg.loadRef(refID, load=True, step=False)
        if refDataset is not None:
            m = get_sample_bout_distros(m, refDataset.config)
            e = refDataset.endpoint_data if hasattr(refDataset, 'endpoint_data') else refDataset.read(key='end')
            Sinv=reg.SampleDic.inverse
            sample_ps=[]
            for k in ks:
                if k in Sinv.keys():
                    p=Sinv[k]
                    if p in e.columns :
                        sample_ps.append(p)

            if len(sample_ps) > 0:
                sample_dict_p = sample_group(N=Nids, ps=sample_ps, e=e)
                sample_dict={reg.SampleDic[p]:vs for p,vs in sample_dict_p.items()}
                refID = refDataset.refID
    sample_dict.update(parameter_dict)
    return generate_larvae(Nids, sample_dict, m), refID


def imitateRef(mID=None, m=None, refID=None, refDataset=None, Nids=1, parameter_dict={}):
    if refDataset is None:
        if refID is not None:
            refDataset = preg.loadRef(refID, load=True, step=False)
        else:
            raise
    else:
        refID = refDataset.refID
    if Nids is None:
        Nids = refDataset.config.N

    e = refDataset.endpoint_data if hasattr(refDataset, 'endpoint_data') else refDataset.read(key='end')
    ids = random.sample(e.index.values.tolist(), Nids)
    sample_dict = {}
    for p,k in reg.SampleDic.items():
        if p in e.columns:
            pmu = e[p].mean()
            vs = []
            for id in ids:
                v = e[p].loc[id]
                if np.isnan(v):
                    v = pmu
                vs.append(v)
            sample_dict[k] = vs


    sample_dict.update(parameter_dict)

    if m is None:
        m = preg.loadConf(id=mID, conftype="Model")
    m = get_sample_bout_distros(m, refDataset.config)
    ms = generate_larvae(Nids, sample_dict, m)
    ps = [tuple(e[['initial_x', 'initial_y']].loc[id].values) for id in ids]
    try:
        ors = [e['initial_front_orientation'].loc[id] for id in ids]
    except:
        ors = np.random.uniform(low=0, high=2 * np.pi, size=len(ids)).tolist()
    return ids, ps, ors, ms


def sim_models(mIDs, colors=None, dataset_ids=None, data_dir=None, **kwargs):
    N = len(mIDs)
    if colors is None:
        from lib.aux.colsNstr import N_colors
        colors = N_colors(N)
    if dataset_ids is None:
        dataset_ids = mIDs
    if data_dir is None:
        dirs = [None] * N
    else:
        dirs = [f'{data_dir}/{dID}' for dID in dataset_ids]
    ds = [sim_model(mID=mIDs[i], color=colors[i], dataset_id=dataset_ids[i], dir=dirs[i], **kwargs) for i in range(N)]
    return ds


def sim_single_agent(m, Nticks=1000, dt=0.1, df_columns=None, p0=None, fo0=None):
    from lib.model.modules.locomotor import DefaultLocomotor
    from lib.model.body.controller import PhysicsController
    from lib.aux.ang_aux import rear_orientation_change, wrap_angle_to_0
    if fo0 is None :
        fo0=0.0
    if p0 is None :
        p0=(0.0,0.0)
    x0,y0=p0
    if df_columns is None:
        df_columns = preg.getPar(['b', 'fo', 'ro', 'fov', 'I_T', 'x', 'y', 'd', 'v', 'A_T', 'c_CT'])
    AA = np.ones([Nticks, len(df_columns)]) * np.nan

    controller = PhysicsController(**m.physics)
    l = m.body.initial_length
    bend_errors = 0
    DL = DefaultLocomotor(dt=dt, conf=m.brain)
    for qq in range(100):
        if random.uniform(0, 1) < 0.5:
            DL.step(A_in=0, length=l)
    b, fo, ro, fov, x, y, dst, v = 0, fo0, 0, 0, x0, y0, 0, 0
    #print(fo0,x0,y0)
    for i in range(Nticks):
        lin, ang, feed = DL.step(A_in=0, length=l)
        v, fov = controller.get_vels(lin, ang, fov, v, b, dt=dt, ang_suppression=DL.cur_ang_suppression)

        d_or = fov * dt
        if np.abs(d_or) > np.pi:
            bend_errors += 1
        dst = v * dt
        d_ro = rear_orientation_change(b, dst, l, correction_coef=controller.bend_correction_coef)
        # if d_ro is None :
        #     print(i, lin, DL.crawler.stride_dst_mean, DL.crawler.stride_dst_std, DL.crawler.Act_coef, DL.crawler.new_stride, DL.crawler.Act)
        b = wrap_angle_to_0(b + d_or - d_ro)
        fo = (fo + d_or) % (2 * np.pi)
        ro = (ro + d_ro) % (2 * np.pi)
        x += dst * np.cos(fo)
        y += dst * np.sin(fo)

        AA[i, :] = [b, fo, ro, fov, DL.turner.input, x, y, dst, v, DL.turner.output, DL.cur_ang_suppression]

    AA[:, :4] = np.rad2deg(AA[:, :4])
    return AA


def sim_multi_agents(Nticks, Nids, ms, group_id, dt=0.1, ids=None, p0s=None, fo0s=None):
    df_columns = preg.getPar(['b', 'fo', 'ro', 'fov', 'I_T', 'x', 'y', 'd', 'v', 'A_T', 'c_CT'])
    if ids is None:
        ids = [f'{group_id}{j}' for j in range(Nids)]
    if p0s is None:
        p0s = [(0.0, 0.0) for j in range(Nids)]
    if fo0s is None:
        fo0s = [0.0 for j in range(Nids)]
    my_index = pd.MultiIndex.from_product([np.arange(Nticks), ids], names=['Step', 'AgentID'])
    AA = np.ones([Nticks, Nids, len(df_columns)]) * np.nan

    for j, id in enumerate(ids):
        m = ms[j]
        #x0, y0 = p0s[j]
        # mConf = mConfs[j]

        AA[:, j, :] = sim_single_agent(m, Nticks, dt=dt, df_columns=df_columns, p0=p0s[j], fo0=fo0s[j])

    AA = AA.reshape(Nticks * Nids, len(df_columns))
    s = pd.DataFrame(AA, index=my_index, columns=df_columns)
    s = s.astype(float)

    e = pd.DataFrame(index=ids)
    e['cum_dur'] = Nticks * dt
    e['num_ticks'] = Nticks
    e['length'] = [m.body.initial_length for m in ms]

    from lib.process.spatial import scale_to_length
    scale_to_length(s, e, keys=['v'])
    return s, e


def sim_model(mID, Nids=1, refID=None, refDataset=None, sample_ks=None, use_LarvaConfDict=False, imitation=False,
              **kwargs):
    if use_LarvaConfDict:
        pass
    if not imitation:
        ms, refID = sampleRef(mID=mID, refID=refID, Nids=Nids, refDataset=refDataset, sample_ks=sample_ks)
        ids, p0s, fo0s = None, None, None
    else:
        ids, p0s, fo0s, ms = imitateRef(mID=mID, refID=refID, Nids=Nids, refDataset=refDataset)
        if refID is None:
            refID = refDataset.refID
    d = sim_model_dataset(ms, mID=mID, Nids=Nids, refID=refID, ids=ids, p0s=p0s, fo0s=fo0s, **kwargs)
    return d


def sim_model_dataset(ms, mID, env_params={}, dir=None, dur=3, dt=1 / 16, color='blue', dataset_id=None, tor_durs=[],
                      dsp_starts=[0], dsp_stops=[40],
                      bout_annotation=True, enrichment=True, refID=None, Nids=1, ids=None, p0s=None, fo0s=None,
                      **kwargs):
    Nticks = int(dur * 60 / dt)
    if dataset_id is None:
        dataset_id = mID

    c_kws = {
        # 'load_data': False,
        'dir': dir,
        'id': dataset_id,
        # 'metric_definition': g.enrichment.metric_definition,
        'larva_groups': preg.lg(id=dataset_id, c=color, sample=refID, mID=mID, N=Nids, expand=True, **kwargs),
        'env_params': env_params,
        'Npoints': 3,
        'Ncontour': 0,
        'fr': 1 / dt,
        'mID': mID,
    }

    from lib.stor.larva_dataset import LarvaDataset
    d = LarvaDataset(**c_kws, load_data=False)
    s, e = sim_multi_agents(Nticks, Nids, ms, dataset_id, dt=dt, ids=ids, p0s=p0s, fo0s=fo0s)

    d.set_data(step=s, end=e)
    print(d.config.agent_ids)
    print(d.step_data.index.unique('AgentID').values)
    print(d.endpoint_data.index.values)
    if enrichment:
        d = d._enrich(proc_keys=['spatial', 'angular', 'dispersion', 'tortuosity'], bout_annotation=bout_annotation,
                      store=dir is not None,
                      dsp_starts=dsp_starts, dsp_stops=dsp_stops, tor_durs=tor_durs)

    return d
