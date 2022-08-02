import copy
import os.path
import numpy as np
import pandas as pd

from lib.stor.match_ids import match_larva_ids
from lib.aux import naming as nam, dictsNlists as dNl, xy_aux, dir_aux


def build_Schleyer(dataset, build_conf,  source_dir,source_files=None, save_mode='semifull',
                   max_Nagents=None, min_end_time_in_sec=0, min_duration_in_sec=0, start_time_in_sec=0, **kwargs):
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
        cols1 = nam.xy(d.points, flat=True) + nam.xy(d.contour, flat=True) + [
            'collision_flag']
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

    # fr = d.fr



    def temp_build(d, pref):
        # fr = d.fr
        # x_pars = [x for x, y in d.points_xy]
        # y_pars = [y for x, y in d.points_xy]
        # xc_pars = [x for x, y in d.contour_xy]
        # yc_pars = [y for x, y in d.contour_xy]
        x_pars = [x for x, y in d.points_xy]
        y_pars = [y for x, y in d.points_xy]
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
            states = pd.read_csv(f'{pref}_global_state_large_state.txt', header=None, sep='\t', names=['state'])
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
                xcs, ycs = xy_aux.convex_hull(xs=xcs.values, ys=ycs.values, N=d.Ncontour)
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
                k = np.sum(np.diff(np.array(dNl.group_list_by_n(xy[j, :], 2)), axis=0) ** 2, axis=1).T
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




    try:
        temp, e, columns = temp_load()
        print('**--- Loaded temporary data successfully!----- ')
    except:
        temp, e, columns = temp_build(d, pref)

    if match_ids :
        temp = match_larva_ids(s=temp, e=e, pars=['head_x', 'head_y'], **kwargs)


    step=dir_aux.reset_MultiIndex(temp, columns=columns)
    step, end=dir_aux.import_smaller_dataset(step, dt=d.dt,
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
