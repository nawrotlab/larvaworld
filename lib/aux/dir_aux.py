import copy
import os

import numpy as np
import pandas as pd

from lib.aux import naming as nam, dictsNlists as dNl


from lib.stor.larva_dataset import LarvaDataset

from lib import reg

def detect_dataset(datagroup_id=None, folder_path=None, raw=True, **kwargs):
    dic = {}
    if folder_path in ['', None]:
        return dic
    if raw:
        conf = reg.loadConf(id=datagroup_id, conftype='Group').tracker.filesystem
        dF, df = conf.folder, conf.file
        dFp, dFs = dF.pref, dF.suf
        dfp, dfs, df_ = df.pref, df.suf, df.sep

        fn = folder_path.split('/')[-1]
        if dFp is not None:
            if fn.startswith(dFp):
                dic[fn] = folder_path
            else:
                ids, dirs = detect_dataset_in_subdirs(datagroup_id, folder_path, fn, **kwargs)
                for id, dr in zip(ids, dirs):
                    dic[id] = dr
        elif dFs is not None:
            if fn.startswith(dFs):
                dic[fn] = folder_path
            else:
                ids, dirs = detect_dataset_in_subdirs(datagroup_id, folder_path, fn, **kwargs)
                for id, dr in zip(ids, dirs):
                    dic[id] = dr
        elif dfp is not None:
            fs = os.listdir(folder_path)
            ids, dirs = [f.split(df_)[1:][0] for f in fs if f.startswith(dfp)], [folder_path]
            for id, dr in zip(ids, dirs):
                dic[id] = dr
        elif dfs is not None:
            fs = os.listdir(folder_path)
            ids = [f.split(df_)[:-1][0] for f in fs if f.endswith(dfs)]
            for id in ids:
                dic[id] = folder_path
        elif df_ is not None:
            fs = os.listdir(folder_path)
            ids = dNl.unique_list([f.split(df_)[0] for f in fs if df_ in f])
            for id in ids:
                dic[id] = folder_path
        return dic
    else:
        if os.path.exists(f'{folder_path}/data'):
            dd = LarvaDataset(dir=folder_path)
            dic[dd.id] = dd
        else:
            for ddr in [x[0] for x in os.walk(folder_path)]:
                if os.path.exists(f'{ddr}/data'):
                    dd = LarvaDataset(dir=ddr)
                    dic[dd.id] = dd
        return dic


def detect_dataset_in_subdirs(datagroup_id, folder_path, last_dir, full_ID=False):
    fn = last_dir
    ids, dirs = [], []
    if os.path.isdir(folder_path):
        fs = os.listdir(folder_path)
        for f in fs:
            dic = detect_dataset(datagroup_id, f'{folder_path}/{f}', full_ID=full_ID, raw=True)
            for id, dr in dic.items():
                if full_ID:
                    ids += [f'{fn}/{id0}' for id0 in id]
                else:
                    ids.append(id)
                dirs.append(dr)
    return ids, dirs

def split_dataset(step,end, food, larva_groups,dir, **kwargs):
    ds = []
    for gID, gConf in larva_groups.items():
        d = LarvaDataset(f'{dir}/{gID}', id=gID, larva_groups={gID: gConf}, load_data=False, **kwargs)
        d.set_data(step=step.loc[(slice(None), gConf.ids), :], end=end.loc[gConf.ids], food=food)
        ds.append(d)
    return ds

def smaller_dataset(d, track_point=None, ids=None, transposition=None, time_range=None, pars=None,env_params=None,close_view=False):


    c=d.config
    c0=dNl.copyDict(c)


    if track_point is None:
        track_point = c.point
    elif type(track_point) == int:
        track_point = 'centroid' if track_point == -1 else nam.midline(c.Npoints, type='point')[track_point]
    c0.point = track_point
    if ids is not None:
        if type(ids) == list and all([type(i) == int for i in ids]):
            ids = [c.agent_ids[i] for i in ids]
    else :
        ids = c.agent_ids
    c0.agent_ids = ids
    c0.N = len(ids)

    def get_data(d,ids) :
        if not hasattr(d, 'step_data'):
            d.load(h5_ks=['contour', 'midline'])
        s, e = d.step_data, d.endpoint_data
        e0=copy.deepcopy(e.loc[ids])
        s0=copy.deepcopy(s.loc[(slice(None), ids), :])
        return s0,e0

    s0,e0=get_data(d,ids)

    if pars is not None:
        s0 = s0.loc[(slice(None), slice(None)), pars]

    if env_params is not None:
        c0.env_params = env_params

    if transposition is not None:
        try:
            s_tr = d.load_traj(mode=transposition)
            s0.update(s_tr)

        except:
            from lib.process.spatial import align_trajectories
            s0 = align_trajectories(s0, c=c0, transposition=transposition,replace=True)

        xy_max=2*np.max(s0[nam.xy(c0.point)].dropna().abs().values.flatten())
        c0.env_params.arena = reg.get_null('arena', arena_dims=(xy_max, xy_max))

    if close_view:
        c0.env_params.arena = reg.get_null('arena', arena_dims=(0.01, 0.01))


    if time_range is not None:
        a, b = time_range
        a = int(a / c.dt)
        b = int(b / c.dt)
        s0 = s0.loc[(slice(a, b), slice(None)), :]

    c0.Nsteps = len(s0.index.unique('Step').values)
    return s0,e0, c0

def import_smaller_dataset(step, dt, max_Nagents=None, min_duration_in_sec=0.0,time_slice=None):
    min_ticks=min_duration_in_sec/dt


    if time_slice is not None :
        tmin,tmax=time_slice
        tickmin,tickmax=int(tmin/dt),int(tmax/dt)
        step = copy.deepcopy(step.loc[(slice(tickmin,tickmax), slice(None)),:])
    end = step['head_x'].dropna().groupby('AgentID').count().to_frame()
    end.columns = ['num_ticks']

    if max_Nagents is not None:
        selected = end.nlargest(max_Nagents, 'num_ticks').index.values
        step = step.loc[(slice(None), selected), :]
        end = end.loc[selected]
    selected = end[end['num_ticks'] > min_ticks].index.values
    step = step.loc[(slice(None), selected), :]
    end = end.loc[selected]
    end['cum_dur'] = end['num_ticks'] * dt
    return step, end

def reset_AgentIDs(s):
    s.reset_index(level='Step', drop=False, inplace=True)
    trange = np.arange(int(s['Step'].max())).astype(int)
    old_ids = s.index.unique().tolist()
    new_ids = [f'Larva_{100 + i}' for i in range(len(old_ids))]
    new_pairs = dict(zip(old_ids, new_ids))
    s.rename(index=new_pairs, inplace=True)

    s.reset_index(drop=False, inplace=True)
    s.set_index(keys=['Step', 'AgentID'], inplace=True, drop=True)
    s.sort_index(level=['Step', 'AgentID'], inplace=True)
    s.drop_duplicates(inplace=True)

    my_index = pd.MultiIndex.from_product([trange, new_ids], names=['Step', 'AgentID'])
    return s, my_index

def reset_MultiIndex(s, columns=None) :
    s, my_index=reset_AgentIDs(s)
    if columns is None :
        columns = s.columns.values
    step = pd.DataFrame(s, index=my_index, columns=columns)
    return step


def get_traj(d, mode='default'):
    if mode=='default':
        return d.load_traj(mode)
    elif mode == 'origin':
        try:
            ss=d.load_traj(mode)
            return ss[['x', 'y']]
        except:
            s = d.load_step(h5_ks=['contour', 'midline'])
            from lib.process.spatial import align_trajectories
            ss=align_trajectories(s, c=d.config, store=True, replace=False, transposition='origin')
            return ss[['x', 'y']]


