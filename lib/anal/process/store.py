from distutils.dir_util import copy_tree

import numpy as np
import pandas as pd
import os

import lib.aux.dictsNlists
import lib.aux.colsNstr as fun
import lib.aux.naming as nam
import lib.conf.dtype_dicts as dtypes

from lib.stor import paths


def store_aux_dataset(s, pars, type, file):
    store = pd.HDFStore(file)
    ps = [p for p in pars if p in s.columns]
    if type == 'distro':
        for p in ps:
            # print(p)
            d = s[p].dropna().reset_index(level=0, drop=True)
            d.sort_index(inplace=True)
            store[f'{type}.{p}'] = d
            # if p=='turn_front_orientation_unwrapped' :
            #     print(d)
            #     print(store[f'{type}.{p}'])
                # raise
    elif type == 'dispersion':
        for p in ps:
            dsp = s[p]
            steps = s.index.unique('Step')
            Nticks = len(steps)
            dsp_ar = np.zeros([Nticks, 3]) * np.nan
            dsp_m = dsp.groupby(level='Step').quantile(q=0.5)
            dsp_u = dsp.groupby(level='Step').quantile(q=0.75)
            dsp_b = dsp.groupby(level='Step').quantile(q=0.25)
            dsp_ar[:, 0] = dsp_m
            dsp_ar[:, 1] = dsp_u
            dsp_ar[:, 2] = dsp_b
            d = pd.DataFrame(dsp_ar, index=steps, columns=['median', 'upper', 'lower'])
            store[f'{type}.{p}'] = d
    elif type == 'stride':
        Npoints = 32
        ids = s.index.unique('AgentID').values
        all_data = [s.xs(id, level='AgentID', drop_level=True) for id in ids]
        all_starts = [d[d[nam.start(type)] == True].index.values.astype(int) for d in all_data]
        all_stops = [d[d[nam.stop(type)] == True].index.values.astype(int) for d in all_data]

        p_timeseries = [[] for p in ps]
        p_chunk_ids = [[] for p in ps]
        for id, data, starts, stops in zip(ids, all_data, all_starts, all_stops):
            for start, stop in zip(starts, stops):
                for i, p in enumerate(ps):
                    timeserie = data.loc[slice(start, stop), p].values
                    p_timeseries[i].append(timeserie)
                    p_chunk_ids[i].append(id)
        p_durations = [[len(i) for i in t] for t in p_timeseries]

        p_chunks = [
            [np.interp(x=np.linspace(0, 2 * np.pi, Npoints), xp=np.linspace(0, 2 * np.pi, dur), fp=ts, left=0,
                       right=0) for dur, ts in zip(durations, timeseries)] for durations, timeseries in
            zip(p_durations, p_timeseries)]
        for chunks, chunk_ids, p in zip(p_chunks, p_chunk_ids, ps):
            d = pd.DataFrame(np.array(chunks), index=chunk_ids, columns=np.arange(Npoints).tolist())
            store[f'{type}.{p}'] = d
    store.close()
    print(f'{len(ps)} aux parameters saved')


def create_reference_dataset(config, dataset_id='reference', Nstd=3, overwrite=False):
    from lib.stor.larva_dataset import LarvaDataset
    from lib.anal.fitting import fit_bouts
    from lib.conf.conf import saveConf
    from lib.model.modules.intermitter import get_EEB_poly1d

    path_dir = f'{paths.RefFolder}/{dataset_id}'
    path_data = f'{path_dir}/data/reference.csv'
    path_fits = f'{path_dir}/data/bout_fits.csv'
    if not os.path.exists(path_dir) or overwrite:
        copy_tree(config['dir'], path_dir)
    new_d = LarvaDataset(path_dir)
    new_d.set_id(dataset_id)

    pars = lib.aux.dictsNlists.load_dict(paths.RefParsFile, use_pickle=False)

    pars= {p:pp for p,pp in pars.items() if p in new_d.endpoint_data.columns}

    df = new_d.endpoint_data[list(pars.keys())].rename(columns=pars, inplace=True)
    df.to_csv(path_data)

    fit_bouts(dataset=new_d,config=new_d.config,e=new_d.endpoint_data, store=True, bouts=['stride', 'pause'])

    dic = {
        nam.freq('crawl'): df['brain.crawler_params.initial_freq'].mean(),
        nam.freq('feed'): df['brain.feeder_params.initial_freq'].mean() if 'brain.feeder_params.initial_freq' in df.columns else 2.0,
        'feeder_reoccurence_rate': None,
        'dt': 1 / config['fr'],
    }
    saveConf(dic, conf_type='Ref', id=dataset_id, mode='update')
    z = get_EEB_poly1d(dataset_id)
    saveConf({'EEB_poly1d': z.c.tolist()}, conf_type='Ref', id=dataset_id, mode='update')

    print(f'Reference dataset {dataset_id} saved.')
