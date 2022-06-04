import copy
import itertools
import random

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import ks_2samp
from shapely.geometry import Point


from lib.anal.fitting import fit_bouts
from lib.aux import naming as nam, dictsNlists as dNl
from lib.aux.ang_aux import rear_orientation_change, wrap_angle_to_0
from lib.aux.colsNstr import N_colors
from lib.aux.sim_aux import get_tank_polygon
from lib.conf.base.dtypes import null_dict
from lib.conf.stored.conf import expandConf

from lib.process.aux import annotation, fft_freqs, suppress_stdout, compute_interference
from lib.process.spatial import scale_to_length, comp_dispersion, comp_straightness_index, comp_spatial, store_spatial

from lib.conf.base.par import ParDict, getPar

dic = ParDict(mode='load').dict
dst, v, sv, acc, sa, fou, rou, fo, ro, b,fov, rov, bv,foa, roa, ba, x, y, l,dsp, dsp_0_40,dsp_0_40_mu,dsp_0_40_max,sdsp, sdsp_0_40,sdsp_0_40_mu,sdsp_0_40_max,str_fov_mu,run_fov_mu,pau_fov_mu,run_foa_mu,pau_foa_mu, str_fov_std,pau_fov_std,sstr_d_mu, sstr_d_std,str_d_mu, str_d_std, str_sv_mu, pau_sv_mu,str_v_mu,run_v_mu,run_sv_mu, pau_v_mu, str_tr,run_tr, pau_tr,Ltur_tr,Rtur_tr, Ltur_fou,Rtur_fou, run_t_min, cum_t, run_t, run_dst, pau_t= [dic[k]['d'] for k in ['d','v', 'sv','a','sa','fou', 'rou', 'fo', 'ro', 'b', 'fov', 'rov', 'bv', 'foa', 'roa', 'ba', 'x', 'y', 'l',"dsp", "dsp_0_40","dsp_0_40_mu","dsp_0_40_max","sdsp", "sdsp_0_40","sdsp_0_40_mu","sdsp_0_40_max",'str_fov_mu','run_fov_mu','pau_fov_mu','run_foa_mu','pau_foa_mu', 'str_fov_std','pau_fov_std', 'sstr_d_mu', 'sstr_d_std','str_d_mu', 'str_d_std', 'str_sv_mu', 'pau_sv_mu','str_v_mu', 'run_v_mu','run_sv_mu', 'pau_v_mu', 'str_tr','run_tr','pau_tr','Ltur_tr','Rtur_tr', 'Ltur_fou','Rtur_fou', 'run_t_min', 'cum_t',  'run_t', 'run_d', 'pau_t']]
l, v_mu,cum_d, sv_mu, fov_mu, b_mu = [dic[k]['d'] for k in ['l', 'v_mu', "cum_d", "sv_mu",'fov_mu','b_mu']]
tors=tor60,tor2,tor2_mu,tor2_std,tor5,tor5_mu,tor5_std,tor10,tor10_mu,tor10_std,tor20,tor20_mu,tor20_std= [dic[k]['d'] for k in ['tor60',"tor2","tor2_mu","tor2_std","tor5","tor5_mu","tor5_std","tor10","tor10_mu","tor10_std","tor20","tor20_mu","tor20_std"]]
fsv, ffov =  [dic[k]['d'] for k in ['fsv','ffov']]
run_t_max, pau_t_min, pau_t_max = getPar(['run_t_max', 'pau_t_min', 'pau_t_max'])

def eval_endpoint(ee,e,e_shorts=None,e_pars=None,e_labs=None, mode = 'pooled'):
    if e_pars is None :
        e_pars = getPar(e_shorts)
    if e_labs is None :
        e_labs = getPar(d=e_pars, to_return=['lab'])
    Eend= {}
    for p, pl in zip(e_pars, e_labs):
        Eend[pl] = None
        if p in e.columns and p in ee.columns:
            if mode == '1:1':
                Eend[pl] = ((e[p] - ee[p]) ** 2).mean() ** .5
            elif mode == 'pooled':
                Eend[pl] = ks_2samp(e[p].values, ee[p].values)[0]
    return Eend

def eval_end_fast(ee,e_data,e_sym, mode = 'pooled'):
    Eend= {}
    for p, sym in e_sym.items():
        e_vs=e_data[p]
        # sym=e_sym[p]
        Eend[sym] = None
        if p in ee.columns:
            if mode == '1:1':
                Eend[sym] = ((e_vs - ee[p]) ** 2).mean() ** .5
            elif mode == 'pooled':
                Eend[sym] = ks_2samp(e_vs.values, ee[p].values)[0]
    return Eend

def eval_distro(ss,s,s_shorts=None,s_pars=None,s_labs=None, mode = 'pooled', min_size=20):
    if s_pars is None :
        s_pars = getPar(s_shorts)
    if s_labs is None :
        s_labs = getPar(d=s_pars, to_return=['lab'])

    Edistro = {}
    for p, pl in zip(s_pars, s_labs):
        Edistro[pl] = None
        if p in s.columns and p in ss.columns:
            if mode == '1:1':
                pps = []
                for id in s.index.unique('AgentID').values:
                    sp, ssp = s[p].xs(id, level="AgentID").dropna().values, ss[p].xs(id,level="AgentID").dropna().values
                    if sp.shape[0] > min_size and ssp.shape[0] > min_size:
                        pps.append(ks_2samp(sp, ssp)[0])

                Edistro[pl] = np.median(pps)
            elif mode == 'pooled':
                spp, sspp = s[p].dropna().values, ss[p].dropna().values
                if spp.shape[0] > min_size and sspp.shape[0] > min_size:
                    Edistro[pl] = ks_2samp(spp, sspp)[0]
    return Edistro

def eval_distro_fast(ss,s_data,s_sym, mode = 'pooled', min_size=20):
    Edistro = {}
    for p, sym in s_sym.items():
        s_vs=s_data[p]
        # sym=s_sym[p]
        Edistro[sym] = None
        if p in ss.columns:
            if mode == '1:1':
                pps = []
                for id in s_data.index:
                    sp, ssp = s_data[p].loc[id].values, ss[p].xs(id,level="AgentID").dropna().values
                    if sp.shape[0] > min_size and ssp.shape[0] > min_size:
                        pps.append(ks_2samp(sp, ssp)[0])

                Edistro[sym] = np.median(pps)
            elif mode == 'pooled':
                spp, sspp = s_vs.dropna().values, ss[p].dropna().values
                if spp.shape[0] > min_size and sspp.shape[0] > min_size:
                    Edistro[sym] = ks_2samp(spp, sspp)[0]
    return Edistro

def eval_multi(datasets,s=None,e=None, s_shorts=None,e_shorts=None, mode='pooled', min_size=20):
    GEend, GEdistro = {}, {}
    if e is not None and e_shorts is not None:
        e_pars, e_labs = getPar(e_shorts, to_return=['d', 'lab'])
        GEend = {d.id : eval_endpoint(d.endpoint_data,e,e_pars=e_pars,e_labs=e_labs, mode = mode) for d in datasets}
    if s is not None and s_shorts is not None:
        s_pars, s_labs = getPar(s_shorts, to_return=['d', 'lab'])
        GEdistro= {d.id : eval_distro(d.step_data,s,s_pars=s_pars,s_labs=s_labs, mode = mode, min_size=min_size) for d in datasets}
    if mode == '1:1':
        labels = ['RSS error', r'median 1:1 distribution KS$_{D}$']
    elif mode == 'pooled':
        labels = ['pooled endpoint KS$_{D}$', 'pooled distribution KS$_{D}$']
    error_dict = {'endpoint': pd.DataFrame.from_records(GEend).T,
                  'distro': pd.DataFrame.from_records(GEdistro).T}

    return error_dict

def eval_fast(datasets,data,symbols,mode='pooled', min_size=20):
    GEend, GEdistro = {}, {}
    GEend = {d.id: eval_end_fast(d.endpoint_data, data.end, symbols.end, mode=mode) for d in datasets}
    GEdistro = {d.id: eval_distro_fast(d.step_data, data.step, symbols.step,mode=mode, min_size=min_size) for d
                    in datasets}
    if mode == '1:1':
        labels = ['RSS error', r'median 1:1 distribution KS$_{D}$']
    elif mode == 'pooled':
        labels = ['pooled endpoint KS$_{D}$', 'pooled distribution KS$_{D}$']
    error_dict = {'end': pd.DataFrame.from_records(GEend).T,
                  'step': pd.DataFrame.from_records(GEdistro).T}
    error_dict['end'].index.name = 'metric'
    error_dict['step'].index.name = 'metric'
    return error_dict







def adapt_conf(conf0, ee, cc):
    conf = copy.deepcopy(conf0)

    dic = {
        run_sv_mu: ee[str_sv_mu] if str_sv_mu in ee.index else ee[run_v_mu] / ee[l],
        run_v_mu: ee[run_v_mu],
        run_t_min: ee[run_t_min],
        run_t_max: ee[run_t_max],
        pau_t_min: ee[pau_t_min],
        pau_t_max: ee[pau_t_max],
        'ang_vel_headcast': np.deg2rad(ee[pau_fov_mu]),
        'theta_min_headcast': np.deg2rad(ee['headcast_q25_amp']),
        'theta_max_headcast': np.deg2rad(ee['headcast_q75_amp']),
        'theta_max_weathervane': np.deg2rad(ee['weathervane_q75_amp']),
        'ang_vel_weathervane': np.deg2rad(ee[run_fov_mu]),
        'turner_input_constant': ee['turner_input_constant'],
        'run_dist': cc.bout_distros.run_dur,
        'pause_dist': cc.bout_distros.pause_dur,
        'stridechain_dist': cc.bout_distros.run_count}

    if cc.Npoints > 1:
        ddic = {'initial_freq': ee[fsv],
                'step_mu': ee[sstr_d_mu],
                'step_std': ee[sstr_d_std],
                'attenuation': ee['attenuation'],
                'attenuation_max': ee['attenuation_max'],
                'max_vel_phase': ee['phi_scaled_velocity_max']}
        dic.update(**ddic)
    for kk, vv in dic.items():
        if kk in conf.keys():
            conf[kk] = vv

    return conf


def sim_locomotor(L, N, df_cols=None, tank_polygon=None, cur_x=0, cur_y=0, cur_fo=0, length=0.004):
    if df_cols is None:
        from lib.conf.base.par import getPar
        df_cols = getPar(['v', 'fov', 'd', 'fo', 'x', 'y', 'b'])
    aL = np.ones([N, len(df_cols)]) * np.nan
    for i in range(N):
        lin, ang, feed = L.step(A_in=0, length=length)
        cur_d = lin * L.dt
        if tank_polygon:
            if not tank_polygon.contains(Point(cur_x, cur_y)):
                cur_fo -= np.pi
        cur_fo += ang * L.dt
        cur_x += np.cos(cur_fo) * cur_d
        cur_y += np.sin(cur_fo) * cur_d
        aL[i, :7] = [lin, ang, cur_d, cur_fo, cur_x, cur_y, L.bend]
    for ii in [1, 3, 6]:
        aL[:, ii] = np.rad2deg(aL[:, ii])
    return aL


def sim_dataset(ee, cc, loco_func, loco_conf, adapted=False):
    df_cols = getPar(['v', 'fov', 'd', 'fo', 'x', 'y', 'b'])
    Ncols = len(df_cols)

    Ls = {}
    for jj, id in enumerate(cc.agent_ids):

        # Build locomotor instance of given model configuration adapted to the specific experimental larva
        if adapted == 'I':
            loco_conf = adapt_conf(loco_conf, ee.loc[id], cc)
        Ls[id] = loco_func(dt=cc.dt, **loco_conf)

    aaL = np.zeros([cc.Nticks, cc.N, Ncols]) * np.nan
    aaL[0, :, :] = 0
    try:
        aaL[0, :, 3:6] = ee[nam.initial([fo, x, y])]
    except:
        pass
    tank_polygon = get_tank_polygon(cc)
    for jj, id in enumerate(cc.agent_ids):
        # cur_x, cur_y, cur_fo = 0,0,0
        cur_fo, cur_x, cur_y = aaL[0, jj, 3:6]
        aaL[1:, jj, :] = sim_locomotor(Ls[id], cc.Nticks - 1, df_cols, tank_polygon, cur_x, cur_y, np.deg2rad(cur_fo),
                                       length=ee['length'].loc[id])

    df_index = pd.MultiIndex.from_product([np.arange(cc.Nticks), cc.agent_ids], names=['Step', 'AgentID'])
    ss = pd.DataFrame(aaL.reshape([cc.Nticks * cc.N, Ncols]), index=df_index, columns=df_cols)

    return ss


def enrich_dataset(ss, ee, cc, tor_durs=[2, 5, 10, 20], dsp_starts=[0], dsp_stops=[40]):
    strides_enabled = True if cc.Npoints > 1 else False
    vel_thr = cc.vel_thr if cc.Npoints == 1 else 0.2

    dt = cc.dt
    ss[bv] = ss[b].groupby('AgentID').diff() / dt
    ss[ba] = ss[bv].groupby('AgentID').diff() / dt
    ss[foa] = ss[fov].groupby('AgentID').diff() / dt
    ss[acc] = ss[v].groupby('AgentID').diff() / dt

    ee[v_mu] = ss[v].dropna().groupby('AgentID').mean()
    ee[cum_d] = ss[dst].dropna().groupby('AgentID').sum()

    scale_to_length(ss, ee, cc, keys=['d', 'v', 'a', 'v_mu'])

    comp_dispersion(ss, ee, cc,dsp_starts=dsp_starts, dsp_stops=dsp_stops)
    comp_straightness_index(ss,  ee,cc, dt,tor_durs=tor_durs)

    chunk_dicts= annotation(ss, ee, cc, strides_enabled=strides_enabled, vel_thr=vel_thr)
    cycle_curves = compute_interference(ss, ee, cc, chunk_dicts=chunk_dicts)
    pooled_epochs = fit_bouts(c=cc, chunk_dicts=chunk_dicts, s=ss, e=ee, id=cc.id)

    return pooled_epochs


def arrange_evaluation(d, evaluation_metrics):

    Edata, Ddata = {},{}
    dic = dNl.AttrDict.from_nested_dicts({'end': {'shorts': [], 'groups': []}, 'step': {'shorts': [], 'groups': []}})
    for g, shs in evaluation_metrics.items():
        Eshorts, Dshorts = [], []
        ps = getPar(shs)
        for sh, p in zip(shs, ps) :
            try:
                data=d.read(key='end', file='endpoint_h5')[p]
                if data is not None:
                    Edata[p]=data
                    Eshorts.append(sh)
            except :
                data = d.get_par(p, key='distro')
                if data is not None:
                    Ddata[p] = data.dropna()
                    Dshorts.append(sh)

        # Eshorts = [sh for sh, p in zip(shs, ps) if p in e.columns]
        # Dshorts = [sh for sh, p in zip(shs, ps) if p in s.columns]
        # Dshorts = [sh for sh in Dshorts if sh not in Eshorts]
        if len(Eshorts) > 0:
            dic.end.shorts.append(Eshorts)
            dic.end.groups.append(g)
        if len(Dshorts) > 0:
            dic.step.shorts.append(Dshorts)
            dic.step.groups.append(g)
    eval_data = dNl.AttrDict.from_nested_dicts({'step':Ddata, 'end': Edata})
    return dic,eval_data

def arrange_evaluation2(s, e, evaluation_metrics):
    d = dNl.AttrDict.from_nested_dicts({'end': {'shorts': [], 'groups': []}, 'step': {'shorts': [], 'groups': []}})
    for g, shs in evaluation_metrics.items():
        ps = getPar(shs)
        Eshorts = [sh for sh, p in zip(shs, ps) if p in e.columns]
        Dshorts = [sh for sh, p in zip(shs, ps) if p in s.columns]
        Dshorts = [sh for sh in Dshorts if sh not in Eshorts]
        if len(Eshorts) > 0:
            d.end.shorts.append(Eshorts)
            d.end.groups.append(g)
        if len(Dshorts) > 0:
            d.step.shorts.append(Dshorts)
            d.step.groups.append(g)
    return d


def prepare_sim_dataset(e, c, id, color):
    cc = dNl.AttrDict.from_nested_dicts({
        'env_params': c.env_params,
        'bout_distros': c.bout_distros,
        'id': id,
        'dt': c.dt,
        'fr': c.fr,
        'N': c.N,
        'agent_ids': c.agent_ids,
        'Nticks': c.Nticks,
        'Npoints': 3,
        'color': color,
        'point': '',
        'parent_plot_dir': c.parent_plot_dir,

    })

    e_ps = ['length', nam.initial(x), nam.initial(y), nam.initial(fo)]
    ee = pd.DataFrame(e[e_ps].loc[cc.agent_ids], index=cc.agent_ids, columns=e_ps)
    ee[cum_t] = cc.Nticks * cc.dt
    return ee, cc


def prepare_dataset(d, Nids, id='experiment', color='black'):
    s0, e0, c0 = d.step_data, d.endpoint_data, d.config
    if Nids is None:
        ids = c0.agent_ids
        Nids = len(ids)
    else:
        ids = e0.nlargest(Nids, 'cum_dur').index.values
    s = copy.deepcopy(s0.loc[(slice(None), ids), slice(None)])
    e = copy.deepcopy(e0.loc[ids])

    c = dNl.AttrDict.from_nested_dicts(c0)
    c.id = id
    c.agent_ids = ids
    c.N = Nids
    c.color = color
    if "length" not in e.columns:
        e["length"] = np.ones(c.N) * 0.004
    return s, e, c


def prepare_validation_dataset(d, Nids):
    s, e, c = d.step_data, d.endpoint_data, d.config
    ids = e.nlargest(Nids, 'cum_dur').index.values
    ids2 = e.nlargest(2 * Nids, 'cum_dur').index.values[Nids:]
    s_val = s.loc[(slice(None), ids2), slice(None)]
    e_val = e.loc[ids2]
    e_val.rename(index=dict(zip(ids2, ids)), inplace=True)
    s_val.reset_index(level='Step', drop=False, inplace=True)
    s_val.rename(index=dict(zip(ids2, ids)), inplace=True)
    s_val.reset_index(drop=False, inplace=True)
    s_val.set_index(keys=['Step', 'AgentID'], inplace=True, drop=True)

    s_val = copy.deepcopy(s.loc[(slice(None), ids2), slice(None)])
    e_val = copy.deepcopy(e.loc[ids2])
    c_val = dNl.AttrDict.from_nested_dicts(c)
    c_val.id = 'cross-val'
    c_val.agent_ids = ids2
    c_val.N = Nids
    c_val.color = 'grey'

    if "length" not in e_val.columns:
        e_val["length"] = np.ones(c_val.N) * 0.004

    return s_val, e_val, c_val


def torsNdsps(pars):
    tor_durs = [int(ii[len('tortuosity') + 1:]) for ii in pars if ii.startswith('tortuosity')]
    tor_durs=np.unique(tor_durs)
    tor_shorts = [f'tor{ii}' for ii in tor_durs]

    dsp_temp = [ii[len(dsp) + 1:].split('_') for ii in pars if ii.startswith(f'{dsp}_')]
    dsp_starts = np.unique([int(ii[0]) for ii in dsp_temp]).tolist()
    dsp_stops = np.unique([int(ii[1]) for ii in dsp_temp]).tolist()
    dsp_shorts0 = [f'dsp_{s0}_{s1}' for s0, s1 in itertools.product(dsp_starts, dsp_stops)]
    dsp_shorts = dNl.flatten_list([[f'{ii}_max', f'{ii}_mu', f'{ii}_fin'] for ii in dsp_shorts0])
    return tor_durs, dsp_starts, dsp_stops

def sim_models(mIDs,colors=None,dataset_ids=None,data_dir=None, **kwargs):
    N=len(mIDs)
    if colors is None :
        colors=N_colors(N)
    if dataset_ids is None :
        dataset_ids=mIDs
    if data_dir is None :
        dirs=[None]*N
    else :
        dirs=[f'{data_dir}/{dID}' for dID in dataset_ids]
    ds=[sim_model(mID=mIDs[i],color=colors[i],dataset_id=dataset_ids[i],dir=dirs[i], **kwargs) for i in range(N)]
    return ds

def sim_model(mID, dur=3, dt=1 / 16,Nids=1,color='blue',dataset_id=None,tor_durs=[],dsp_starts=[0],dsp_stops=[40],env_params={},dir=None,
              bout_annotation=True,refDataset=None,sample_ks=None,store=False, **kwargs):
    from lib.model.modules.locomotor import DefaultLocomotor
    from lib.conf.stored.conf import loadConf, kConfDict, loadRef, copyConf
    from lib.process.angular import angular_processing
    from lib.model.body.controller import PhysicsController


    if dataset_id is None:
        dataset_id = mID
    if refDataset is not None :
        refID = refDataset.refID
        ms = refDataset.sample_modelConf(N=Nids, mID=mID, sample_ks=sample_ks)
    else :
        refID =None
        m = loadConf(mID, "Model")
        ms = [m] * Nids

    Nticks = int(dur * 60 / dt)
    ids=[f'Agent{j}' for j in range(Nids)]

    larva_groups = {dataset_id: null_dict('LarvaGroup', sample=refID, model=expandConf(mID, 'Model'),
                                   default_color=color,
                                   distribution=null_dict('larva_distro', N=Nids))}

    c = dNl.AttrDict.from_nested_dicts(
        {'dir': dir, 'id': dataset_id,'larva_groups' : larva_groups, 'group_id': 'offline', 'dt': dt, 'fr': 1/dt, 'agent_ids': ids, 'duration': dur * 60,
         'Npoints': 3, 'Ncontour': 0, 'point': '', 'N': Nids, 'Nticks': Nticks, 'mID': mID, 'color': color, 'env_params': env_params})









    my_index = pd.MultiIndex.from_product([np.arange(Nticks), ids], names=['Step', 'AgentID'])
    e = pd.DataFrame(index=ids)
    e['cum_dur'] = dur*60
    e['num_ticks'] = Nticks
    e['length'] = [m.body.initial_length for m in ms]

    df_columns = getPar(['b', 'fo', 'ro', 'fov', 'Act_tur','x', 'y', 'd', 'v', 'A_tur']) + ['ang_suppression']
    AA = np.ones([Nticks, Nids, len(df_columns)]) * np.nan




    for j,id in enumerate(ids) :
        m=ms[j]
        controller = PhysicsController(**m.physics)
        l=e['length'].loc[id]
        bend_errors = 0
        DL = DefaultLocomotor(dt=dt, conf=m.brain)
        for qq in range(100):
            if random.uniform(0, 1) < 0.5:
                DL.step(A_in=0, length=l)
        b, fo, ro, fov, x, y, dst, v = 0,0,0,0,0,0,0,0


        for i in range(Nticks):
            lin, ang, feed = DL.step(A_in=0, length=l)
            v, fov = controller.get_vels(lin, ang, fov, v,b, dt=dt, ang_suppression=DL.cur_ang_suppression)

            d_or =fov * dt
            if np.abs(d_or) > np.pi:
                bend_errors += 1
            dst = lin * dt
            d_ro = rear_orientation_change(b, dst, l, correction_coef= controller.bend_correction_coef)
            b = wrap_angle_to_0(b + d_or - d_ro)
            fo=(fo+d_or)%(2*np.pi)
            ro=(ro+d_ro)%(2*np.pi)
            x+=dst*np.cos(fo)
            y+=dst*np.sin(fo)

            AA[i,j,:]=[b,fo,ro,fov,DL.turner.activity,x,y,dst, v,DL.turner.activation,DL.cur_ang_suppression]
    AA[:, :, :4] = np.rad2deg(AA[:, :, :4])
    AA=AA.reshape(Nticks*Nids, len(df_columns))
    s = pd.DataFrame(AA,index=my_index, columns=df_columns)
    s[getPar('v_in_mm')] = s[getPar('v')]*1000
    # s['torque'] = s[getPar('Act_tur')] * controller.torque_coef
    s=s.astype(float)

    if c.dir is not None :
        from lib.stor.larva_dataset import LarvaDataset
        d=LarvaDataset(**c, load_data=False)
        store=True
        d.set_data(step=s, end=e)
        c=d.config
    else :
        d = dNl.AttrDict.from_nested_dicts(
            {'id': c.id, 'group_id': c.group_id, 'step_data': s, 'endpoint_data': e, 'config': c, 'color': c.color})

    with suppress_stdout(False):
        comp_spatial(s, e, c, mode='minimal')
        store_spatial(s, e, c,store=store)
        angular_processing(s, e, c, store=store)
        comp_dispersion(s, e, c, dsp_starts=dsp_starts, dsp_stops=dsp_stops, store=store)
        comp_straightness_index(s, e=e, c=c, tor_durs=tor_durs, store=store)



    if bout_annotation:
        from lib.process import aux
        d.chunk_dicts = aux.annotation(s, e, c, store=store)
        d.cycle_curves = compute_interference(s=s, e=e, c=c, chunk_dicts=d.chunk_dicts, store=store)
        d.pooled_epochs = fit_bouts(c=c, chunk_dicts=d.chunk_dicts, s=s, e=e, id=c.id, store=store)

    return d


from lib.aux.dictsNlists import flatten_dict, AttrDict


def RSS(vs0, vs):
    er = (vs - vs0)

    r = np.abs(np.max(vs0) - np.min(vs0))

    ee=(er/r)**2

    MSE = np.mean(np.sum(ee))
    return np.round(np.sqrt(MSE), 2)




def RSS_dic(dd, d):
    f=d.config.pooled_cycle_curves
    ff=dd.config.pooled_cycle_curves

    def RSS0(ff, f, sh, mode):
        vs0 = np.array(f[sh][mode])
        vs = np.array(ff[sh][mode])
        return RSS(vs0, vs)

    def RSS1(ff, f, sh):
        dic = {}
        for mode in f[sh].keys():
            dic[mode] = RSS0(ff, f, sh, mode)
        return AttrDict.from_nested_dicts(dic)


    dic = {}
    for sh in f.keys():
        dic[sh] = RSS1(ff, f, sh)

    ers = AttrDict.from_nested_dicts(dic)

    # ang_ks=[sh for sh in f.keys() if sh!='sv']

    stat = np.round(np.mean([ers[sh]['norm'] for sh in f.keys() if sh!='sv']),2)
    dd.config.pooled_cycle_curves_errors=AttrDict.from_nested_dicts({'dict' : ers, 'stat':stat})
    return stat

if __name__ == '__main__':
    from lib.conf.base import paths
    dir=f"{paths.path('RUN')}/testing/the/sim_model/method"
    d=sim_model(mID='NEU_PHI', dur=1, dt=1 / 16,Nids=2,color='blue',tor_durs=[], dir=dir)

    raise

    s=d.step_data
    ss=s.xs(d.config.agent_ids[0], level='AgentID')
    plt.plot(ss[getPar('foa')].values)

    #plt.hist(s['bend'].dropna().values)
    plt.show()

    raise