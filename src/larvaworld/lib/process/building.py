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


def match_larva_ids(s, e, pars=None, wl=100, wt=0.1, ws=0.5, max_error=600, Nidx=20, verbose=1, **kwargs):
    pairs = {}

    def common_member(a, b):
        a_set = set(a)
        b_set = set(b)
        return a_set & b_set

    def eval(t0, xy0, l0, t1, xy1, l1):
        tt = t1 - t0
        if tt <= 0:
            return max_error * 2
        ll = np.abs(l1 - l0)
        dd = np.sqrt(np.sum((xy1 - xy0) ** 2))
        # print(tt,ll,dd)
        return wt * tt + wl * ll + ws * dd

    def get_extrema(ss, pars):
        ids = ss.index.unique().tolist()

        mins = ss['Step'].groupby('AgentID').min()
        maxs = ss['Step'].groupby('AgentID').max()
        durs = ss['Step'].groupby('AgentID').count()
        first_xy, last_xy = {}, {}
        for id in ids:
            first_xy[id] = ss[pars].xs(id).dropna().values[0, :]
            last_xy[id] = ss[pars].xs(id).dropna().values[-1, :]
        return ids, mins, maxs, first_xy, last_xy, durs

    def update_extrema(id0, id1, ids, mins, maxs, first_xy, last_xy):
        mins[id1], first_xy[id1] = mins[id0], first_xy[id0]
        del mins[id0]
        del maxs[id0]
        del first_xy[id0]
        del last_xy[id0]
        ids.remove(id0)
        return ids, mins, maxs, first_xy, last_xy

    ls = e['length']
    if pars is None:
        pars = s.columns.values.tolist()
    s.reset_index(level='Step', drop=False, inplace=True)
    s['Step'] = s['Step'].values.astype(int)
    ids, mins, maxs, first_xy, last_xy, durs = get_extrema(s, pars)
    Nids0 = len(ids)
    while Nidx <= len(ids):
        cur_er, id0, id1 = max_error, None, None
        t0s = maxs.nsmallest(Nidx)
        t1s = mins.loc[mins > t0s.min()].nsmallest(Nidx)
        if len(t1s) > 0:
            for i in range(Nidx):
                cur_id0, t0 = t0s.index[i], t0s.values[i]
                xy0, l0 = last_xy[cur_id0], ls[cur_id0]
                ee = [eval(t0, xy0, l0, mins[id], first_xy[id], ls[id]) for id in t1s.index]
                temp_err = np.min(ee)
                if temp_err < cur_er:
                    cur_er, id0, id1 = temp_err, cur_id0, t1s.index[np.argmin(ee)]
        if id0 is not None:
            pairs[id0] = id1
            ls[id1] = (ls[id0] * durs[id0] + ls[id1] * durs[id1]) / (durs[id0] + durs[id1])
            durs[id1] += durs[id0]
            del durs[id0]
            ls.drop([id0], inplace=True)
            ids, mins, maxs, first_xy, last_xy = update_extrema(id0, id1, ids, mins, maxs, first_xy, last_xy)
            if verbose >= 2:
                print(len(ids), int(cur_er))
        else:
            Nidx += 1
    Nids1 = len(ids)
    if verbose >= 2:
        print('Finalizing dataset')
    while len(common_member(list(pairs.keys()), list(pairs.values()))) > 0:
        for id0, id1 in pairs.items():
            if id1 in pairs.keys():
                pairs[id0] = pairs[id1]
                break
    s.rename(index=pairs, inplace=True)
    s.reset_index(drop=False, inplace=True)
    s.set_index(keys=['Step', 'AgentID'], inplace=True, drop=True)
    if verbose >= 1:
        print(f'**--- Track IDs reduced from {Nids0} to {Nids1} by the matchIDs algorithm -----')
    return s

def build_Schleyer(dataset, build_conf,  source_dir,source_files=None, save_mode='semifull',
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


def build_Jovanic(dataset, build_conf, source_id,source_dir, source_files=None, max_Nagents=None, min_duration_in_sec=10.0,time_slice=None,
                  match_ids=True,**kwargs):
    d = dataset

    def df_from_csvs(d, pref, max_Nagents=None, time_slice=(0, 60), min_duration_in_sec=10.0):
        kws = {'header': None, 'sep': '\t'}
        par_list = [pd.read_csv(f'{pref}_{suf}.txt', **kws) for suf in ['larvaid', 't', 'x_spine', 'y_spine']]
        columns = ['AgentID', 'Step'] + aux.nam.xy(d.points, xsNys=True, flat=True)
        try:
            states = pd.read_csv(f'{pref}_state.txt', **kws)
            par_list.append(states)
            columns.append('state')
        except:
            pass

        if d.Ncontour > 0:
            try:
                # xcps, ycps = aux.nam.xy(d.contour, xsNys=True, flat=True)

                xcs = pd.read_csv(f'{pref}_x_contour.txt', **kws)
                ycs = pd.read_csv(f'{pref}_y_contour.txt', **kws)
                xcs, ycs = aux.convex_hull(xs=xcs.values, ys=ycs.values, N=d.Ncontour)
                xcs = pd.DataFrame(xcs, index=None)
                ycs = pd.DataFrame(ycs, index=None)
                par_list += [xcs, ycs]
                columns+=aux.nam.xy(d.contour, xsNys=True, flat=True)
            except:
                pass

        df = pd.concat(par_list, axis=1, sort=False)
        df.columns=columns
        df.set_index(keys=['AgentID'], inplace=True, drop=True)

        if time_slice is not None:
            tmin, tmax = time_slice
            df = df[df['Step'] < tmax]
            df = df[df['Step'] >= tmin]
        df = df.loc[df['Step'].groupby('AgentID').last() - df['Step'].groupby('AgentID').first() > min_duration_in_sec]
        if max_Nagents is not None:
            df = df.loc[df['head_x'].dropna().groupby('AgentID').count().nlargest(max_Nagents).index]
        df.sort_index(inplace=True)
        return df



    def interpolate_step_data(df, e, dt):
        s = copy.deepcopy(df)
        s['Step'] /= dt
        s.reset_index(drop=False, inplace=True)
        s.set_index(keys=['Step', 'AgentID'], inplace=True, drop=True, verify_integrity=False)
        s.sort_index(level=['Step', 'AgentID'], inplace=True)
        ids = aux.index_unique(s, level='AgentID')

        tmax = int(np.ceil(e['t1'].max()))
        Nticks = int(np.ceil(tmax / dt))
        ticks = np.arange(0, Nticks, 1).astype(int)

        my_index = pd.MultiIndex.from_product([ticks, ids], names=['Step', 'AgentID'])
        ps = s.columns
        A = np.zeros([Nticks, ids.shape[0], len(ps)]) * np.nan

        for j, id in enumerate(ids):
            dff = s.xs(id, level='AgentID', drop_level=True)
            ticks = np.arange(int(e['tick0'].loc[id]), int(e['tick1'].loc[id]), 1)
            for i, p in enumerate(ps):
                f = interpolate.interp1d(x=dff.index.values, y=dff[p].values, fill_value='extrapolate',
                                         assume_sorted=True)
                A[ticks, j, i] = f(ticks)
        A = A.reshape([-1, len(ps)])
        df_new = pd.DataFrame(A, index=my_index, columns=ps)
        df_new.sort_index(level=['Step', 'AgentID'], inplace=True)
        print(f'----- Timeseries data for group "{d.id}" of experiment "{d.group_id}" generated in full mode ')
        return df_new

    def init_endpoint_data(df, dt):
        g = df['Step'].groupby(level='AgentID')
        e = pd.concat(dict(zip(['t0', 't1', 'Nts'], [g.first(), g.last(), g.count()])), axis=1)
        e['cum_dur'] = e['t1'] - e['t0']
        e['dt'] = e['cum_dur'] / (e['Nts'] - 1)
        e['tick1'] = np.ceil(e['t1'] / dt).astype(int)
        e['tick0'] = np.floor(e['t0'] / dt).astype(int)
        e['Nticks'] = e['tick1'] - e['tick0']
        e.sort_index(inplace=True)

        return e

    def comp_length(df, e):
        xys = nam.xy(d.points, flat=True)
        xy2 = df[xys].values.reshape(-1, d.Npoints, 2)
        xy3 = np.sum(np.diff(xy2, axis=1) ** 2, axis=2)
        df['length'] = np.sum(np.sqrt(xy3), axis=1)
        e['length'] = df['length'].groupby('AgentID').quantile(q=0.5)
        print(f'----- Body lengths computed for group "{d.id}" of experiment "{d.group_id}".')

    print(f'*---- Buiding dataset {d.id} of group {d.group_id}!-----')
    df = df_from_csvs(d, pref=f'{source_dir}/{source_id}',
                      max_Nagents=max_Nagents, min_duration_in_sec=min_duration_in_sec,time_slice=time_slice,**kwargs)
    # df.sort_index(inplace=True)

    e = init_endpoint_data(df=df, dt=d.dt)

    if match_ids :
        comp_length(df, e)
        df = match_larva_ids(s=df, e=e, pars=['head_x', 'head_y'])

    s = interpolate_step_data(df=df, e=e, dt=d.dt)

    # e = pd.DataFrame({}, index=df.index.unique('AgentID').values)
    # reg.funcs.processing['length'](df, e, N=11)


    # df = temp_build(df, fr=d.fr)


    #     df = match_larva_ids(s=df, e=e, pars=['head_x', 'head_y'], **kwargs)
    # step = reset_MultiIndex(df)
    # end = pd.DataFrame({}, index=step.index.unique('AgentID').values)
    # reg.funcs.processing['length'](df, end, N=11)
    #
    # end['num_ticks'] = step['head_x'].dropna().groupby('AgentID').count()
    # end['cum_dur'] = end['num_ticks'] * d.dt
    return s, e

def build_Berni(dataset, build_conf, source_files, source_dir=None, max_Nagents=None, min_duration_in_sec=0.0,
                  match_ids=True,min_end_time_in_sec=0, start_time_in_sec=0,**kwargs):
    d = dataset
    dt = d.dt
    end = pd.DataFrame(columns=['AgentID', 'num_ticks', 'cum_dur'])
    Nvalid = 0
    dfs = []
    ids = []
    cols0 = build_conf['read_sequence']
    cols1=cols0[1:]
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


def build_Arguello(dataset, build_conf, source_files, source_dir=None, max_Nagents=None, min_duration_in_sec=0.0,
                  match_ids=True,min_end_time_in_sec=0, start_time_in_sec=0,**kwargs):
    d = dataset
    dt = d.dt
    end = pd.DataFrame(columns=['AgentID', 'num_ticks', 'cum_dur'])
    Nvalid = 0
    dfs = []
    ids = []
    cols0 = build_conf['read_sequence']
    cols1=cols0[1:]
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


def import_dataset(datagroup_id, parent_dir, group_id=None, N=None, id=None, merged=False, enrich=True,
                   refID=None, enrich_conf=None, **kwargs):
    print()
    print(f'----- Initializing {datagroup_id} format-specific dataset import. -----')

    if id is None:
        id = f'{N}controls'
    if group_id is None:
        group_id = parent_dir


    g = reg.loadGroup(datagroup_id)
    group_dir = g.path
    raw_folder = f'{group_dir}/raw'
    proc_folder = f'{group_dir}/processed'
    source_dir = f'{raw_folder}/{parent_dir}'
    if merged:
        source_dir = [f'{source_dir}/{f}' for f in os.listdir(source_dir)]
    kws = {
        'datagroup_id': datagroup_id,
        'group_id': group_id,
        'Ν': N,
        'target_dir': f'{proc_folder}/{group_id}/{id}',
        'source_dir': source_dir,
        'max_Nagents': N,
        **kwargs
    }
    d = build_dataset(id=id, **kws)
    if d is not None:
        print(f'***-- Dataset {d.id} created with {len(d.agent_ids)} larvae! -----')
        if enrich:
            print(f'****- Processing dataset {d.id} to derive secondary metrics -----')
            if enrich_conf is None:
                enrich_conf = g.enrichment
            d = d.enrich(**enrich_conf, is_last=False)
        d.save(refID=refID)
        if refID is not None :
            print(f'***** Dataset stored under the reference ID : {refID} -----')
    else:
        print(f'xxxxx Failed to create dataset {id}! -----')
    return d


def build_dataset(datagroup_id, id, target_dir, group_id, N=None, sample=None,
                  color='black', epochs={},age=0.0, **kwargs):
    print(f'*---- Building dataset {id} under the {datagroup_id} format. -----')

    func_dict = {
        'Jovanic lab': build_Jovanic,
        'Berni lab': build_Berni,
        'Schleyer lab': build_Schleyer,
        'Arguello lab': build_Arguello,
    }

    warnings.filterwarnings('ignore')

    shutil.rmtree(target_dir, ignore_errors=True)
    g = reg.loadGroup(datagroup_id)

    conf = {
        'load_data': False,
        'dir': target_dir,
        'id': id,
        'metric_definition': g.enrichment.metric_definition,
        'larva_groups': reg.lg(id=group_id, c=color, sample=sample, mID= None, N=N,epochs={},age=0.0),
        'env_params': reg.get_null('Env', arena=g.Tracker.arena),
        **g.Tracker.resolution
    }
    d = larvaworld.LarvaDataset(**conf)
    kws0 = {
        'dataset': d,
        'build_conf': g.Tracker.filesystem,
        **kwargs
    }

    try:

        step, end = func_dict[datagroup_id](**kws0)
        d.set_data(step=step, end=end)
        # print(f'***-- Dataset {d.id} created with {len(d.agent_ids)} larvae! -----')
        return d
    except:
        # print(f'xxxxx Failed to create dataset {id}! -----')
        # d.delete()
        return None



