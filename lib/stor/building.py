import numpy as np
import pandas as pd

from lib.stor.datagroup import *
from lib.aux.functions import match_larva_ids
from lib.aux import functions as fun
from lib.aux import naming as nam

def build_Schleyer(dataset, build_conf, raw_folders, save_mode='semifull',
          use_tick_index=True, max_Nagents=np.inf, complete_ticks=True,
          min_end_time_in_sec=0, min_duration_in_sec=0, start_time_in_sec=0):
    d=dataset
    raw_cols=build_conf['read_sequence']
    raw_files=[]
    all_invert_x = []
    for i, f in enumerate(raw_folders):
        files = [os.path.join(f, n) for n in os.listdir(f) if n.endswith('.csv')]
        raw_files+=files
        if build_conf['read_metadata'] :
            all_invert_x+= get_invert_x_array(read_Schleyer_metadata(f), len(files))
    if len(all_invert_x) == 0:
        all_invert_x = [False] *len(raw_files)
    min_duration_in_ticks = int(min_duration_in_sec / d.dt)
    min_end_time_in_ticks = int(min_end_time_in_sec / d.dt)
    start_time_in_ticks = int(start_time_in_sec / d.dt)

    if save_mode == 'full':
        save_sequence = raw_cols[1:]
    elif save_mode == 'minimal':
        save_sequence = nam.xy(d.point)
    elif save_mode == 'semifull':
        save_sequence = nam.xy(d.points, flat=True) + nam.xy(d.contour, flat=True) + [
            'collision_flag']
    elif save_mode == 'spinepointsNcollision':
        save_sequence = nam.xy(d.points, flat=True) + ['collision_flag']
    elif save_mode == 'points':
        save_sequence = nam.xy(d.points, flat=True)

    x_pars = [p for p in save_sequence if p.endswith('x')]

    endpoint_data = pd.DataFrame(columns=['AgentID', 'num_ticks', 'cum_dur'])
    appropriate_recordings_counter = 0
    dfs = []
    agent_ids = []
    for filename, invert_x in zip(raw_files, all_invert_x):
        df = pd.read_csv(filename, header=None, index_col=0, names=raw_cols)
        # FIXME This has been added because some csv in Schleyer datasets have index=NA. This happens if a larva is lost and refound by tracker
        df = df.dropna()
        if use_tick_index == False:
            df.index = np.arange(len(df))

        if len(df) >= min_duration_in_ticks and df.index.max() >= min_end_time_in_ticks:
            df = df[df.index >= start_time_in_ticks]
            df = df[save_sequence]
            df = df.apply(pd.to_numeric, errors='coerce')
            if invert_x:
                for x_par in x_pars:
                    df[x_par] *= -1

            appropriate_recordings_counter += 1
            agent_id = f'Larva_{appropriate_recordings_counter}'
            dfs.append(df)
            agent_ids.append(agent_id)
            if appropriate_recordings_counter >= max_Nagents:
                break
    if complete_ticks:
        min_tick, max_tick = np.min([df.index.min() for df in dfs]), np.max([df.index.max() for df in dfs])
        trange = np.arange(min_tick, max_tick + 1).tolist()

        df_empty = pd.DataFrame(np.nan, index=trange, columns=save_sequence)
        df_empty.index.name = 'Step'

        for i, (df, agent_id) in enumerate(zip(dfs, agent_ids)):
            ddf = df_empty.copy(deep=True)
            endpoint_data = endpoint_data.append({'AgentID': agent_id,
                                                  'num_ticks': len(df),
                                                  'cum_dur': len(df) * d.dt}, ignore_index=True)
            ddf.update(df)
            ddf = ddf.assign(AgentID=agent_id).set_index('AgentID', append=True)
            if i == 0:
                step_data = ddf
            else:
                step_data = step_data.append(ddf)
    else:
        for i, (df, agent_id) in enumerate(zip(dfs, agent_ids)):
            endpoint_data = endpoint_data.append({'AgentID': agent_id,
                                                  'num_ticks': len(df),
                                                  'cum_dur': len(df) * d.dt}, ignore_index=True)
            df = df.assign(AgentID=agent_id).set_index('AgentID', append=True)
            if i == 0:
                step_data = df
            else:
                step_data = step_data.append(df)
    endpoint_data.set_index('AgentID', inplace=True)

    # I add this because some 'na' values were found
    step_data = step_data.mask(step_data == 'na', np.nan)
    return step_data, endpoint_data

def build_Jovanic(dataset, build_conf, source_dir, max_Nagents=None, complete_ticks=True,
                  min_Nids=None, dl=None,**kwargs):
    d=dataset
    fr = d.fr
    x_pars = [x for x, y in d.points_xy]
    y_pars = [y for x, y in d.points_xy]

    t_file = os.path.join(source_dir, 't.txt')
    id_file = os.path.join(source_dir, 'larvaid.txt')
    x_file = os.path.join(source_dir, 'x_spine.txt')
    y_file = os.path.join(source_dir, 'y_spine.txt')
    xs = pd.read_csv(x_file, header=None, sep='\t', names=x_pars)
    ys = pd.read_csv(y_file, header=None, sep='\t', names=y_pars)
    ts = pd.read_csv(t_file, header=None, sep='\t', names=['Step'])

    ids = pd.read_csv(id_file, header=None, sep='\t', names=['AgentID'])
    ids['AgentID'] = [f'Larva_{10000 + i[0]}' for i in ids.values]

    min_t, max_t = float(ts.min()), float(ts.max())
    trange = np.arange(np.ceil(max_t * fr))
    temp = pd.concat([ids, ts, xs, ys], axis=1, sort=False)
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

    if min_Nids is not None:
        if dl is not None :
            temp['spinelength']=np.nan
            temp.reset_index(level='Step', drop=False, inplace=True)
            temp_ids = temp.index.unique().tolist()
            ls=[]
            for id in temp_ids :
                ag_temp=temp.loc[id]
                xy = ag_temp[nam.xy(d.points, flat=True)].values
                spinelength = np.zeros(len(ag_temp)) * np.nan
                for j in range(xy.shape[0]):
                    k = np.sum(np.diff(np.array(fun.group_list_by_n(xy[j, :], 2)), axis=0) ** 2, axis=1).T
                    if not np.isnan(np.sum(k)):
                        sp_l = np.sum([np.sqrt(kk) for kk in k])
                    else:
                        sp_l = np.nan
                    spinelength[j] = sp_l
                temp['spinelength'].loc[id]=spinelength
                ls.append(np.nanmean(spinelength))
            e = pd.DataFrame({'length' : ls}, index=temp_ids)
            # import matplotlib.pyplot as plt
            # e.hist()
            # plt.show()
            # raise
            temp.reset_index(drop=False, inplace=True)
            temp.set_index(keys=['Step', 'AgentID'], inplace=True, drop=True)
        else :
            e=None

        temp = match_larva_ids(s=temp, pars=['head_x', 'head_y'], e=e,
                               min_Nids=min_Nids, dl=dl, **kwargs)
        temp.reset_index(level='Step', drop=False, inplace=True)
        old_ids = temp.index.unique().tolist()
        new_ids = [f'Larva_{100 + i}' for i in range(len(old_ids))]
        new_pairs = dict(zip(old_ids, new_ids))
        temp.rename(index=new_pairs, inplace=True)
        temp.reset_index(drop=False, inplace=True)
        temp.set_index(keys=['Step', 'AgentID'], inplace=True, drop=True)

    if complete_ticks:
        my_index = pd.MultiIndex.from_product([trange, new_ids], names=['Step', 'AgentID'])
        step_data = pd.DataFrame(index=my_index, columns=x_pars + y_pars)
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