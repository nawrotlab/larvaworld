import copy
import os
import os.path
import numpy as np
import pandas as pd
import shutil
import warnings

from scipy import interpolate

import larvaworld
from larvaworld.lib import reg, aux

from larvaworld.lib.aux import nam
from larvaworld.lib.process.build_aux import df_from_csvs, match_larva_ids


def interpolate_step_data(df, dt):
    t = 't'
    step='Step'
    aID = 'AgentID'



    s = copy.deepcopy(df)
    s[step] = s[t]/dt
    Nticks = int(np.ceil(s[step].max()))
    s.reset_index(drop=False, inplace=True)
    s.set_index(keys=[step, aID], inplace=True, drop=True, verify_integrity=False)
    s.sort_index(level=[step, aID], inplace=True)
    ids = aux.index_unique(s, level=aID)

    ticks = np.arange(0, Nticks, 1).astype(int)

    my_index = pd.MultiIndex.from_product([ticks, ids], names=[step, aID])
    ps = s.columns
    A = np.zeros([Nticks, ids.shape[0], len(ps)]) * np.nan

    for j, id in enumerate(ids):
        dff = s.xs(id, level=aID, drop_level=True)
        float_ticks_j = dff.index
        ticks_j = np.arange(int(np.floor(float_ticks_j.min())), int(np.ceil(float_ticks_j.max())), 1)
        for i, p in enumerate(ps):
            f = interpolate.interp1d(x=float_ticks_j.values, y=dff[p].loc[float_ticks_j].values, fill_value='extrapolate',
                                     assume_sorted=True)
            A[ticks_j, j, i] = f(ticks_j)
    A = A.reshape([-1, len(ps)])
    df_new = pd.DataFrame(A, index=my_index, columns=ps)
    df_new.sort_index(level=[step, aID], inplace=True)
    return df_new

def build_Jovanic(dataset,  source_id,source_dir,
                  max_Nagents=None, min_duration_in_sec=0.0,time_slice=None,
                  match_ids=True,**kwargs):
    d = dataset

    def init_endpoint_data(df, dt):
        g = df['t'].groupby(level='AgentID')
        t0, t1,Nts, cum_t = reg.getPar(['t0', 't_fin','N_ts', 'cum_t'])
        tick0, tick1, Nticks = reg.getPar(['tick0', 'tick_fin', 'N_ticks'])
        e = pd.concat(dict(zip([t0, t1, Nts], [g.first(), g.last(), g.count()])), axis=1)
        e[cum_t] = e[t1] - e[t0]
        e['dt'] = e[cum_t] / (e[Nts] - 1)
        e[tick1] = np.ceil(e[t1] / dt).astype(int)
        e[tick0] = np.floor(e[t0] / dt).astype(int)
        e[Nticks] = e[tick1] - e[tick0]
        e.sort_index(inplace=True)
        return e

    def comp_length(df, e):
        xys = d.config.midline_xy
        xy2 = df[xys].values.reshape(-1, d.config.Npoints, 2)
        xy3 = np.sum(np.diff(xy2, axis=1) ** 2, axis=2)
        df['length'] = np.sum(np.sqrt(xy3), axis=1)
        e['length'] = df['length'].groupby('AgentID').quantile(q=0.5)
        print(f'----- Body lengths computed for group "{d.id}" of experiment "{d.group_id}".')

    print(f'*---- Buiding dataset {d.id} of group {d.group_id}!-----')

    df = df_from_csvs(pref=f'{source_dir}/{source_id}',Npoints =d.config.Npoints, Ncontour =d.config.Ncontour,
                      max_Nagents=max_Nagents, min_duration_in_sec=min_duration_in_sec,time_slice=time_slice)

    e = init_endpoint_data(df=df, dt=d.dt)

    if match_ids :
        comp_length(df, e)
        df = match_larva_ids(s=df, e=e, pars=['head_x', 'head_y'])
        e = init_endpoint_data(df=df, dt=d.dt)

    s = interpolate_step_data(df=df, dt=d.dt)
    print(f'----- Timeseries data for group "{d.id}" of experiment "{d.group_id}" generated in full mode ')

    return s, e


def build_Schleyer(dataset, source_dir,save_mode='semifull',
                   max_Nagents=None, min_end_time_in_sec=0, min_duration_in_sec=0, start_time_in_sec=0, **kwargs):

    def read_Schleyer_metadata(dir):
        meta_filename = os.path.join(dir, 'vidAndLogs/metadata.txt')
        dictionary = {}
        with open(meta_filename) as f:
            for j, line in enumerate(f):
                try:
                    nb, list = line.rstrip('\n').split('=')
                    dictionary[nb] = list
                except:
                    pass
        return dictionary

    def get_invert_x_array(meta_dict, Nfiles):
        try:
            odor_side = meta_dict['OdorA_Side']
            if odor_side == 'right':
                invert_x_array = [True for i in range(Nfiles)]
            elif odor_side == 'left':
                invert_x_array = [False for i in range(Nfiles)]
            else:
                raise ValueError(f'Odor side found in metadata is not consistent : {odor_side}')
            return invert_x_array
        except:
            print('Odor side not found in metadata. Assuming left side')
            invert_x_array = [False for i in range(Nfiles)]
            return invert_x_array

    def get_odor_pos(meta_dict, arena_dims):
        ar_x, ar_y = arena_dims
        try:
            odor_side = meta_dict['OdorA_Side']
            x, y = meta_dict['OdorALocation'].split(',')
            x, y = float(x), float(y)
            # meta_dict.pop('OdorALocation', None)
            # meta_dict['OdorPos']=[x,y]
            x, y = 2 * x / ar_x, 2 * y / ar_y
            if odor_side == 'left':
                return [x, y]
            elif odor_side == 'right':
                return [-x, y]
        except:
            return [-0.8, 0]

    g = reg.conf.LabFormat.getID('Schleyer')
    build_conf = g.filesystem

    d = dataset
    dt=d.dt
    cols0 = build_conf.read_sequence
    raw_fs = []
    inv_xs = []
    if type(source_dir)==str :
        source_dir=[source_dir]
    for i, f in enumerate(source_dir):

        fs = [os.path.join(f, n) for n in os.listdir(f) if n.endswith('.csv')]
        raw_fs += fs

        if build_conf.read_metadata :
            try :
                inv_xs += get_invert_x_array(read_Schleyer_metadata(f), len(fs))
            except :
                pass
    if len(inv_xs) == 0:
        inv_xs = [False] * len(raw_fs)

    if save_mode == 'full':
        cols1 = cols0[1:]
    elif save_mode == 'minimal':
        cols1 = nam.xy(d.point)
    elif save_mode == 'semifull':
        N,Nc=d.Npoints, d.Ncontour
        cols1 = nam.midline_xy(N, flat=True) + nam.contour_xy(Nc, flat=True)+ ['collision_flag']

    elif save_mode == 'points':
        cols1 = nam.xy(d.points, flat=True) + ['collision_flag']

    end = pd.DataFrame(columns=['AgentID', 'num_ticks', 'cum_dur'])
    Nvalid = 0
    dfs = []
    ids = []
    for f, inv_x in zip(raw_fs, inv_xs):
        df = pd.read_csv(f, header=None, index_col=0, names=cols0)

        # If indexing is in strings replace with ascending floats
        if all([type(ii)==str for ii in df.index.values]) :
            df.reset_index(inplace=True,drop=True)
        if len(df) >= int(min_duration_in_sec / dt) and df.index.max() >= int(min_end_time_in_sec / dt):
            df = df[df.index >= int(start_time_in_sec / dt)]
            df = df[cols1]
            df = df.apply(pd.to_numeric, errors='coerce')
            if inv_x:
                for x_par in [p for p in cols1 if p.endswith('x')]:
                    df[x_par] *= -1
            Nvalid += 1
            dfs.append(df)
            ids.append(f'Larva_{Nvalid}')
            if max_Nagents is not None and Nvalid >= max_Nagents:
                break
    if len(dfs) == 0:
        return None, None
    t0, t1 = np.min([df.index.min() for df in dfs]), np.max([df.index.max() for df in dfs])
    df0 = pd.DataFrame(np.nan, index=np.arange(t0, t1 + 1).tolist(), columns=cols1)
    df0.index.name = 'Step'

    for i, (df, id) in enumerate(zip(dfs, ids)):
        ddf = df0.copy(deep=True)
        end = end.append({'AgentID': id,
                          'num_ticks': len(df),
                          'cum_dur': len(df) * dt}, ignore_index=True)
        ddf.update(df)
        ddf = ddf.assign(AgentID=id).set_index('AgentID', append=True)
        step = ddf if i == 0 else step.append(ddf)

    end.set_index('AgentID', inplace=True)

    # I add this because some 'na' values were found
    step = step.mask(step == 'na', np.nan)
    return step, end



def build_Berni(dataset, source_files,  max_Nagents=None, min_duration_in_sec=0.0,min_end_time_in_sec=0, **kwargs):
    g = reg.conf.LabFormat.getID('Berni')
    cols0 = g.filesystem.read_sequence
    cols1 = cols0[1:]

    d = dataset
    dt = d.dt
    end = pd.DataFrame(columns=['AgentID', 'num_ticks', 'cum_dur'])
    Nvalid = 0
    dfs = []
    ids = []
    fs = source_files
    # fs = [os.path.join(source_dir, n) for n in os.listdir(source_dir) if n.startswith(dataset.id)]
    for f in fs:
        df = pd.read_csv(f, header=0, index_col=0, names=cols0)
        df.reset_index(drop=True,inplace=True)
        if len(df) >= int(min_duration_in_sec / dt) and len(df) >= int(min_end_time_in_sec / dt):
            # df = df[df.index >= int(start_time_in_sec / dt)]
            df = df[cols1]
            df = df.apply(pd.to_numeric, errors='coerce')
            Nvalid += 1
            dfs.append(df)
            ids.append(f'Larva_{Nvalid}')
            if max_Nagents is not None and Nvalid >= max_Nagents:
                break
        if len(dfs) == 0:
            return None, None
    Nticks=np.max([len(df) for df in dfs])
    df0 = pd.DataFrame(np.nan, index=np.arange(Nticks).tolist(), columns=cols1)
    df0.index.name = 'Step'

    for i, (df, id) in enumerate(zip(dfs, ids)):
        ddf = df0.copy(deep=True)
        end = end.append({'AgentID': id,
                          'num_ticks': len(df),
                          'cum_dur': len(df) * dt}, ignore_index=True)
        ddf.update(df)
        ddf = ddf.assign(AgentID=id).set_index('AgentID', append=True)
        step = ddf if i == 0 else step.append(ddf)
    end.set_index('AgentID', inplace=True)
    return step, end


def build_Arguello(dataset, source_files, max_Nagents=None, min_duration_in_sec=0.0,
                  min_end_time_in_sec=0, **kwargs):
    g = reg.conf.LabFormat.getID('Arguello')
    cols0 = g.filesystem.read_sequence
    cols1 = cols0[1:]
    d = dataset
    dt = d.dt
    end = pd.DataFrame(columns=['AgentID', 'num_ticks', 'cum_dur'])
    Nvalid = 0
    dfs = []
    ids = []

    fs = source_files
    # fs = [os.path.join(source_dir, n) for n in os.listdir(source_dir) if n.startswith(dataset.id)]
    for f in fs:
        df = pd.read_csv(f, header=0, index_col=0, names=cols0)
        df.reset_index(drop=True,inplace=True)
        if len(df) >= int(min_duration_in_sec / dt) and len(df) >= int(min_end_time_in_sec / dt):
            # df = df[df.index >= int(start_time_in_sec / dt)]
            df = df[cols1]
            df = df.apply(pd.to_numeric, errors='coerce')
            Nvalid += 1
            dfs.append(df)
            ids.append(f'Larva_{Nvalid}')
            if max_Nagents is not None and Nvalid >= max_Nagents:
                break
        if len(dfs) == 0:
            return None, None
    Nticks=np.max([len(df) for df in dfs])
    df0 = pd.DataFrame(np.nan, index=np.arange(Nticks).tolist(), columns=cols1)
    df0.index.name = 'Step'

    for i, (df, id) in enumerate(zip(dfs, ids)):
        ddf = df0.copy(deep=True)
        # ddf = ddf.interpolate() # DEALING WITH MISSING DATA? DROP OR INTERPOLATE? DEFAULT IS LINEAR.
        end = end.append({'AgentID': id,
                          'num_ticks': len(df),
                          'cum_dur': len(df) * dt}, ignore_index=True)
        ddf.update(df)
        ddf = ddf.assign(AgentID=id).set_index('AgentID', append=True)
        step = ddf if i == 0 else step.append(ddf)
    end.set_index('AgentID', inplace=True)
    return step, end









def import_datasets(source_ids, ids=None, colors=None, refIDs=None, **kwargs):
    if colors is None:
        colors = aux.N_colors(len(source_ids))
    if ids is None:
        ids = source_ids
    ds = []
    for i, source_id in enumerate(source_ids):
        refID = None if refIDs is None else refIDs[i]

        d = import_dataset(id=ids[i], color=colors[i], source_id=source_id, refID=refID, **kwargs)
        ds.append(d)

    return ds


def import_dataset(labID, parent_dir, group_id=None, N=None, id=None, merged=False,
                   refID=None, enrich_conf=None, **kwargs):
    print()
    print(f'----- Initializing {labID} format-specific dataset import. -----')

    if id is None:
        id = f'{N}controls'
    if group_id is None:
        group_id = parent_dir


    g = reg.conf.LabFormat.get(labID)
    group_dir = g.path
    raw_folder = f'{group_dir}/raw'
    proc_folder = f'{group_dir}/processed'
    source_dir = f'{raw_folder}/{parent_dir}'
    if merged:
        source_dir = [f'{source_dir}/{f}' for f in os.listdir(source_dir)]
    kws = {
        'labID': labID,
        'group_id': group_id,
        'Ν': N,
        'target_dir': f'{proc_folder}/{group_id}/{id}',
        'source_dir': source_dir,
        'max_Nagents': N,
        **kwargs
    }
    d = build_dataset(id=id, **kws)
    if d is not None:
        print(f'***-- Dataset {d.id} created with {len(d.config.agent_ids)} larvae! -----')
        print(f'****- Processing dataset {d.id} to derive secondary metrics -----')
        if enrich_conf is None:
            enrich_conf=reg.gen.EnrichConf(proc_keys =[], anot_keys =[])
        enrich_conf.pre_kws = g.preprocess
        d = d.enrich(**enrich_conf.nestedConf, is_last=False)

        d.save(refID=refID)
        if refID is not None :
            print(f'***** Dataset stored under the reference ID : {refID} -----')
    else:
        print(f'xxxxx Failed to create dataset {id}! -----')
    return d


def build_dataset(labID, id, target_dir, group_id, N=None, sample=None,
                  color='black', epochs={},age=0.0, **kwargs):
    print(f'*---- Building dataset {id} under the {labID} format. -----')

    func_dict = {
        'Jovanic': build_Jovanic,
        'Berni': build_Berni,
        'Schleyer': build_Schleyer,
        'Arguello': build_Arguello,
    }

    warnings.filterwarnings('ignore')

    shutil.rmtree(target_dir, ignore_errors=True)
    g = reg.conf.LabFormat.getID(labID)

    conf = {
        'load_data': False,
        'dir': target_dir,
        'id': id,
        'larva_groups': reg.lg(id=group_id, c=color, sample=sample, mID= None, N=N,epochs=epochs,age=age),
        'env_params': g.env_params,
        **g.tracker
    }
    d = larvaworld.lib.LarvaDataset(**conf)
    kws0 = {
        'dataset': d,
        # **g.filesystem
        **kwargs
    }

    # try:

    step, end = func_dict[labID](**kws0)
    d.set_data(step=step, end=end)
    # print(f'***-- Dataset {d.id} created with {len(d.agent_ids)} larvae! -----')
    return d
    # except:
        # print(f'xxxxx Failed to create dataset {id}! -----')
        # d.delete()
        # return None



