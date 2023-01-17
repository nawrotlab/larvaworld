import copy
import os
import os.path
import numpy as np
import pandas as pd
import shutil
import warnings

from lib import reg, aux

from lib.aux import naming as nam
from lib.process.dataset import LarvaDataset


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


def build_Jovanic(dataset, build_conf, source_id,source_dir, source_files=None, max_Nagents=None, min_duration_in_sec=0.0,time_slice=None,
                  match_ids=True,**kwargs):
    d = dataset
    pref=f'{source_dir}/{source_id}'
    temp_step_path = f'{pref}_step.csv'
    temp_length_path = f'{pref}_length.csv'




    def temp_build(d, pref):
        # fr = d.fr
        # x_pars = [x for x, y in d.points_xy]
        # y_pars = [y for x, y in d.points_xy]
        # xc_pars = [x for x, y in d.contour_xy]
        # yc_pars = [y for x, y in d.contour_xy]
        x_pars = [x for x, y in d.midline_xy]
        y_pars = [y for x, y in d.midline_xy]
        columns = x_pars + y_pars
        xy_pars = nam.xy(d.points, flat=True)

        print(f'*---- Buiding temporary data files for dataset {d.id} of group {d.group_id}!-----')
        xs = pd.read_csv(f'{pref}_x_spine.txt', header=None, sep='\t', names=x_pars)
        ys = pd.read_csv(f'{pref}_y_spine.txt', header=None, sep='\t', names=y_pars)
        ts = pd.read_csv(f'{pref}_t.txt', header=None, sep='\t', names=['Step'])
        ids = pd.read_csv(f'{pref}_larvaid.txt', header=None, sep='\t', names=['AgentID'])
        ids['AgentID'] = [f'Larva_{10000 + i[0]}' for i in ids.values]
        par_list = [ids, ts, xs, ys]
        try:
            states = pd.read_csv(f'{pref}_state.txt', header=None, sep='\t', names=['state'])
            par_list.append(states)
            columns.append('state')

        except:
            states = None

        if d.Ncontour > 0:
            try :
                xc_pars = [x for x, y in d.contour_xy]
                yc_pars = [y for x, y in d.contour_xy]

                xcs = pd.read_csv(f'{pref}_x_contour.txt', header=None, sep='\t')
                ycs = pd.read_csv(f'{pref}_y_contour.txt', header=None, sep='\t')
                xcs, ycs = aux.convex_hull(xs=xcs.values, ys=ycs.values, N=d.Ncontour)
                xcs = pd.DataFrame(xcs, columns=xc_pars, index=None)
                ycs = pd.DataFrame(ycs, columns=yc_pars, index=None)
                par_list += [xcs, ycs]
                columns = columns + xc_pars + yc_pars
            except :
                pass

        min_t, max_t = float(ts.min()), float(ts.max())
        step = pd.concat(par_list, axis=1, sort=False)
        step.set_index(keys=['AgentID'], inplace=True, drop=True)
        step['spinelength'] = np.nan
        agent_ids = np.sort(step.index.unique())

        durs = []
        starts = []
        stops = []
        ls = []
        for id in agent_ids:
            data = step.loc[id]
            t = data['Step'].values
            t0 = int((t[0] - min_t) * d.fr)
            t1 = t0 + len(t)
            t = np.arange(t0, t1)
            step.loc[id, 'Step'] = t
            durs.append(len(t))
            starts.append(t0)
            stops.append(t1)

            xy = data[xy_pars].values
            spinelength = np.zeros(len(data)) * np.nan
            for j in range(xy.shape[0]):
                k = np.sum(np.diff(np.array(aux.group_list_by_n(xy[j, :], 2)), axis=0) ** 2, axis=1).T
                if not np.isnan(np.sum(k)):
                    spinelength[j] = np.sum([np.sqrt(kk) for kk in k])
                # else:
                #     sp_l = np.nan
                # spinelength[j] = sp_l
            step['spinelength'].loc[id] = spinelength
            ls.append(np.nanmean(spinelength))
        step['Step'] = step['Step'].values.astype(int)
        e = pd.DataFrame({'length': ls}, index=agent_ids)
        step.reset_index(drop=False, inplace=True)
        step.set_index(keys=['Step', 'AgentID'], inplace=True, drop=True)
        temp_save(step, e)
        return step, e, columns


    def temp_save(step, length):
        step.to_csv(temp_step_path, index=True, header=True)
        length.to_csv(temp_length_path, index=True, header=True)
        # print(f'**---- Saved temporary dataset {dataset.id} successfully!----- ')

    def temp_load():
        step = pd.read_csv(temp_step_path, index_col=['Step', 'AgentID'])
        e = pd.read_csv(temp_length_path, index_col=0)
        columns=step.columns.values
        return step, e, columns

    def import_smaller_dataset(step, dt, max_Nagents=None, min_duration_in_sec=0.0, time_slice=None):
        min_ticks = min_duration_in_sec / dt

        if time_slice is not None:
            tmin, tmax = time_slice
            tickmin, tickmax = int(tmin / dt), int(tmax / dt)
            step = copy.deepcopy(step.loc[(slice(tickmin, tickmax), slice(None)), :])
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

    def reset_MultiIndex(s, columns=None):
        s, my_index = reset_AgentIDs(s)
        if columns is None:
            columns = s.columns.values
        step = pd.DataFrame(s, index=my_index, columns=columns)
        return step


    try:
        temp, e, columns = temp_load()
        print('**--- Loaded temporary data successfully!----- ')
    except:
        temp, e, columns = temp_build(d, pref)

    if match_ids :
        temp = match_larva_ids(s=temp, e=e, pars=['head_x', 'head_y'], **kwargs)


    step=reset_MultiIndex(temp, columns=columns)
    step, end=import_smaller_dataset(step, dt=d.dt,
                                             max_Nagents=max_Nagents, min_duration_in_sec=min_duration_in_sec,time_slice=time_slice)

    return step, end

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


    g = reg.loadConf(id=datagroup_id, conftype='Group')
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
            d = d.enrich(**enrich_conf, store=True, is_last=False)
        d.save(food=False, refID=refID)
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
    g = reg.loadConf(id=datagroup_id, conftype='Group')

    conf = {
        'load_data': False,
        'dir': target_dir,
        'id': id,
        'metric_definition': g.enrichment.metric_definition,
        'larva_groups': reg.lg(id=group_id, c=color, sample=sample, mID= None, N=N,epochs={},age=0.0),
        'env_params': reg.get_null('Env', arena=g.tracker.arena),
        **g.tracker.resolution
    }
    # print(conf)
    from lib.process.dataset import LarvaDataset
    d = LarvaDataset(**conf)
    kws0 = {
        'dataset': d,
        'build_conf': g.tracker.filesystem,
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


def split_dataset(step,end, food, larva_groups,dir, **kwargs):
    ds = []
    for gID, gConf in larva_groups.items():
        d = LarvaDataset(f'{dir}/{gID}', id=gID, larva_groups={gID: gConf}, load_data=False, **kwargs)
        d.set_data(step=step.loc[(slice(None), gConf.ids), :], end=end.loc[gConf.ids], food=food)
        ds.append(d)
    return ds
