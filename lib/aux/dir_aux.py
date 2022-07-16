import copy
import os

from lib.aux import dictsNlists as dNl
from lib.registry.pars import preg
from lib.stor.larva_dataset import LarvaDataset

def detect_dataset(datagroup_id=None, folder_path=None, raw=True, **kwargs):
    dic = {}
    if folder_path in ['', None]:
        return dic
    if raw:
        conf = preg.loadConf(id=datagroup_id, conftype='Group').tracker.filesystem
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

def split_dataset(step,end, food, larva_groups,dir, id,plot_dir,  show_output=False, **kwargs):
    agent_ids = end.index.values
    ds = []
    for gID, gConf in larva_groups.items():
        new_dir=f'{dir}/{gID}'
        valid_ids = [id for id in agent_ids if str.startswith(id, f'{gID}_')]
        d = LarvaDataset(new_dir, id=gID, larva_groups={gID: gConf}, load_data=False, **kwargs)
        d.set_data(step=step.loc[(slice(None), valid_ids), :], end=end.loc[valid_ids], food=food)
        d.config.parent_plot_dir = plot_dir
        # if is_last:
        #     d.save()
        ds.append(d)
    if show_output:
        print(f'Dataset {id} splitted in {[d.id for d in ds]}')
    return ds

def smaller_dataset(d, track_point=None, ids=None, time_range=None, pars=None,env_params=None,close_view=False):
    from lib.aux import naming as nam, dictsNlists as dNl
    c=d.config
    c0=dNl.copyDict(c)
    if env_params is None:
        env_params = c.env_params
    if close_view:
        env_params.arena = preg.get_null('arena', arena_dims=(0.01, 0.01))
    c0.env_params = env_params

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
        if len(ids)==1:
            id=ids[0]
            s0, e0 = d.load_agent(id)
            if s0 is not None and e0 is not None :
                return s0,e0

        if not hasattr(d, 'step_data'):
            d.load(h5_ks=['contour', 'midline', 'base_spatial'])
        s, e = d.step_data, d.endpoint_data
        e0=copy.deepcopy(e.loc[ids])
        s0=copy.deepcopy(s.loc[(slice(None), ids), :])
        return s0,e0

    s0,e0=get_data(d,ids)

    if pars is not None:
        s0 = s0.loc[(slice(None), slice(None)), pars]

    if time_range is not None:
        a, b = time_range
        a = int(a / c.dt)
        b = int(b / c.dt)
        s0 = s0.loc[(slice(a, b), slice(None)), :]

    c0.Nsteps = len(s0.index.unique('Step').values)
    return s0,e0, c0

