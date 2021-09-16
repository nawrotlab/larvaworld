import os.path

import pandas as pd

from lib.conf.conf import *
from lib.aux.functions import match_larva_ids, convex_hull
from lib.aux import functions as fun
from lib.aux import naming as nam


def build_Schleyer(dataset, build_conf, raw_folders, save_mode='semifull',
                   use_tick_index=True, max_Nagents=None, complete_ticks=True,
                   min_end_time_in_sec=0, min_duration_in_sec=0, start_time_in_sec=0, **kwargs):
    d = dataset
    dt=d.dt
    cols0 = build_conf['read_sequence']
    raw_fs = []
    inv_xs = []
    for i, f in enumerate(raw_folders):
        fs = [os.path.join(f, n) for n in os.listdir(f) if n.endswith('.csv')]
        raw_fs += fs
        if build_conf['read_metadata']:
            inv_xs += get_invert_x_array(read_Schleyer_metadata(f), len(fs))
    if len(inv_xs) == 0:
        inv_xs = [False] * len(raw_fs)

    if save_mode == 'full':
        cols1 = cols0[1:]
    elif save_mode == 'minimal':
        cols1 = nam.xy(d.point)
    elif save_mode == 'semifull':
        cols1 = nam.xy(d.points, flat=True) + nam.xy(d.contour, flat=True) + [
            'collision_flag']
    # elif save_mode == 'spinepointsNcollision':
    #     cols1 = nam.xy(d.points, flat=True) + ['collision_flag']
    elif save_mode == 'points':
        cols1 = nam.xy(d.points, flat=True) + ['collision_flag']

    end = pd.DataFrame(columns=['AgentID', 'num_ticks', 'cum_dur'])
    Nvalid = 0
    dfs = []
    ids = []
    for f, inv_x in zip(raw_fs, inv_xs):
        df = pd.read_csv(f, header=None, index_col=0, names=cols0)
        # FIXME This has been added because some csv in Schleyer datasets have index=NA. This happens if a larva is lost and refound by tracker
        df = df.dropna()
        if use_tick_index == False:
            df.index = np.arange(len(df))

        if len(df) >= int(min_duration_in_sec / dt) and df.index.max() >= int(min_end_time_in_sec / dt):
            df = df[df.index >= int(start_time_in_sec / dt)]
            df = df[cols1]
            df = df.apply(pd.to_numeric, errors='coerce')
            if inv_x:
                for x_par in [p for p in cols1 if p.endswith('x')]:
                    df[x_par] *= -1
            # # This scales mm to meters
            # for p in cols1 :
            #     if p.endswith('x') or p.endswith('y') :
            #         df[p] *= 0.001

            Nvalid += 1
            dfs.append(df)
            ids.append(f'Larva_{Nvalid}')
            if max_Nagents is not None and Nvalid >= max_Nagents:
                break
    if len(dfs) == 0:
        return None, None
    if complete_ticks:
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

    else:
        for i, (df, id) in enumerate(zip(dfs, ids)):
            end = end.append({'AgentID': id,
                              'num_ticks': len(df),
                              'cum_dur': len(df) * dt}, ignore_index=True)
            df = df.assign(AgentID=id).set_index('AgentID', append=True)
            step = df if i == 0 else step.append(df)
    end.set_index('AgentID', inplace=True)

    # I add this because some 'na' values were found
    step = step.mask(step == 'na', np.nan)
    return step, end


def build_Jovanic(dataset, build_conf, source_dir, max_Nagents=None, complete_ticks=True, min_duration_in_sec=0.0,
                  match_ids=True,**kwargs):
    pref=f'{source_dir}/{dataset.id}'
    temp_step_path = f'{pref}_step.csv'
    temp_length_path = f'{pref}_length.csv'

    def temp_save(step, length):
        step.to_csv(temp_step_path, index=True, header=True)
        length.to_csv(temp_length_path, index=True, header=True)
        print(f'Saved temporary dataset {dataset.id} successfully!')

    def temp_load():
        step = pd.read_csv(temp_step_path, index_col=['Step', 'AgentID'])
        e = pd.read_csv(temp_length_path, index_col=0)
        return step, e

    d = dataset
    fr = d.fr
    x_pars = [x for x, y in d.points_xy]
    y_pars = [y for x, y in d.points_xy]
    xc_pars = [x for x, y in d.contour_xy]
    yc_pars = [y for x, y in d.contour_xy]

    try:
        temp, e = temp_load()
        print('Loaded temporary data successfully!')
    except:

        t_file = f'{pref}_t.txt'
        id_file = f'{pref}_larvaid.txt'
        x_file = f'{pref}_x_spine.txt'
        y_file = f'{pref}_y_spine.txt'
        state_file = f'{pref}_global_state_large_state.txt'

        x_contour_file = f'{pref}_x_contour.txt'
        y_contour_file = f'{pref}_y_contour.txt'




        xs = pd.read_csv(x_file, header=None, sep='\t', names=x_pars)
        ys = pd.read_csv(y_file, header=None, sep='\t', names=y_pars)
        ts = pd.read_csv(t_file, header=None, sep='\t', names=['Step'])

        xcs = pd.read_csv(x_contour_file, header=None, sep='\t')
        ycs = pd.read_csv(y_contour_file, header=None, sep='\t')
        xcs,ycs=fun.convex_hull(xs=xcs.values,ys=ycs.values, N=d.Ncontour)
        xcs=pd.DataFrame(xcs, columns=xc_pars, index=None)
        ycs=pd.DataFrame(ycs, columns=yc_pars, index=None)
        # print(xs.values.shape, ys.values.shape, xcs.values.shape, ycs.values.shape)
        # xcs = xcs.iloc[:, :d.Ncontour]
        # ycs = ycs.iloc[:, :d.Ncontour]
        # xcs.set_axis(xc_pars, axis=1, inplace=True)
        # ycs.set_axis(yc_pars, axis=1, inplace=True)

        try:
            states = pd.read_csv(state_file, header=None, sep='\t', names=['state'])
        except:
            states = None

        ids = pd.read_csv(id_file, header=None, sep='\t', names=['AgentID'])
        ids['AgentID'] = [f'Larva_{10000 + i[0]}' for i in ids.values]

        min_t, max_t = float(ts.min()), float(ts.max())

        # par_list = [ids, ts, xs, ys]
        par_list = [ids, ts, xs, ys, xcs,ycs]

        if states is not None:
            par_list.append(states)

        temp = pd.concat(par_list, axis=1, sort=False)
        temp.set_index(keys=['AgentID'], inplace=True, drop=True)
        agent_ids = np.sort(temp.index.unique())

        durs = []
        starts = []
        stops = []
        for id in agent_ids:
            data = temp.xs(id)
            t = data['Step'].values
            start_t = int((t[0] - min_t) * fr)
            stop_t = start_t + len(t)
            t = np.arange(start_t, stop_t)
            temp.loc[id, 'Step'] = t
            durs.append(len(t))
            starts.append(start_t)
            stops.append(stop_t)
        temp.reset_index(drop=False, inplace=True)
        temp.set_index(keys=['Step', 'AgentID'], inplace=True, drop=True)

        temp['spinelength'] = np.nan
        temp.reset_index(level='Step', drop=False, inplace=True)
        temp_ids = temp.index.unique().tolist()
        ls = []
        for id in temp_ids:
            ag_temp = temp.loc[id]
            xy = ag_temp[nam.xy(d.points, flat=True)].values
            spinelength = np.zeros(len(ag_temp)) * np.nan
            for j in range(xy.shape[0]):
                k = np.sum(np.diff(np.array(fun.group_list_by_n(xy[j, :], 2)), axis=0) ** 2, axis=1).T
                if not np.isnan(np.sum(k)):
                    sp_l = np.sum([np.sqrt(kk) for kk in k])
                else:
                    sp_l = np.nan
                spinelength[j] = sp_l
            temp['spinelength'].loc[id] = spinelength
            ls.append(np.nanmean(spinelength))
        e = pd.DataFrame({'length': ls}, index=temp_ids)
        temp.reset_index(drop=False, inplace=True)
        temp.set_index(keys=['Step', 'AgentID'], inplace=True, drop=True)
        temp_save(temp, e)
    if match_ids :
        temp = match_larva_ids(s=temp, e=e, pars=['head_x', 'head_y'], **kwargs)
    temp.reset_index(level='Step', drop=False, inplace=True)
    old_ids = temp.index.unique().tolist()
    new_ids = [f'Larva_{100 + i}' for i in range(len(old_ids))]
    new_pairs = dict(zip(old_ids, new_ids))
    temp.rename(index=new_pairs, inplace=True)
    temp.reset_index(drop=False, inplace=True)
    max_step = int(temp['Step'].max())
    temp.set_index(keys=['Step', 'AgentID'], inplace=True, drop=True)
    temp.sort_index(level=['Step', 'AgentID'], inplace=True)
    temp.drop_duplicates(inplace=True)
    if complete_ticks:
        trange = np.arange(max_step).astype(int)
        my_index = pd.MultiIndex.from_product([trange, new_ids], names=['Step', 'AgentID'])
        # columns = x_pars + y_pars
        columns = x_pars + y_pars + xc_pars + yc_pars
        if 'state' in temp.columns:
            columns.append('state')

        step_data = pd.DataFrame(index=my_index, columns=columns)
        step_data.update(temp)
    else:
        step_data = temp
    endpoint_data = temp['head_x'].dropna().groupby('AgentID').count().to_frame()
    endpoint_data.columns = ['num_ticks']
    endpoint_data['cum_dur'] = endpoint_data['num_ticks'] / fr

    if max_Nagents is not None:
        selected = endpoint_data.nlargest(max_Nagents, columns='num_ticks').index.values
        step_data = step_data.loc[(slice(None), selected), :]
        endpoint_data = endpoint_data.loc[selected]

    if min_duration_in_sec > 0:
        selected = endpoint_data[endpoint_data['cum_dur'] >= min_duration_in_sec].index.values
        step_data = step_data.loc[(slice(None), selected), :]
        endpoint_data = endpoint_data.loc[selected]

    return step_data, endpoint_data


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
