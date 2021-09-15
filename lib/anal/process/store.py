from distutils.dir_util import copy_tree

import numpy as np
import pandas as pd
import os
from scipy.signal import argrelextrema, spectrogram

import lib.aux.functions as fun
import lib.aux.naming as nam
import lib.conf.dtype_dicts as dtypes

from lib.stor import paths



def create_par_distro_dataset(s, pars, dir):
    ps = [p for p in pars if p in s.columns]
    fs = [f'{dir}/{p}.csv' for p in ps]
    for p, f in zip(ps, fs):
        d = s[p].dropna().reset_index(level=0, drop=True)
        d.sort_index(inplace=True)
        d.to_csv(f, index=True, header=True)
    print(f'{len(ps)} parameters saved in distro-datasets ')


def create_dispersion_dataset(s, par='dispersion', scaled=True, dir=None):
    p = nam.scal(par) if scaled else par
    f=f'{p}.csv'
    filepath = f'{dir}/{f}'
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
    dsp_df = pd.DataFrame(dsp_ar, index=steps, columns=['median', 'upper', 'lower'])
    dsp_df.to_csv(filepath, index=True, header=True)
    print(f'Dataset saved as {f}')

def create_chunk_dataset(s, chunk, pars, Npoints=32, dir=None):
    ids = s.index.unique('AgentID').values
    pars_to_store = [p for p in pars if p in s.columns]
    if chunk == 'stride':
        filenames = [f'{p}.csv' for p in pars_to_store]
    else:
        raise ValueError('Only stride chunks allowed')

    all_data = [s.xs(id, level='AgentID', drop_level=True) for id in ids]
    all_starts = [d[d[nam.start(chunk)] == True].index.values.astype(int) for d in all_data]
    all_stops = [d[d[nam.stop(chunk)] == True].index.values.astype(int) for d in all_data]

    p_timeseries = [[] for p in pars_to_store]
    p_chunk_ids = [[] for p in pars_to_store]
    for id, d, starts, stops in zip(ids, all_data, all_starts, all_stops):
        for start, stop in zip(starts, stops):
            for i, p in enumerate(pars_to_store):
                timeserie = d.loc[slice(start, stop), p].values
                p_timeseries[i].append(timeserie)
                p_chunk_ids[i].append(id)
    p_durations = [[len(i) for i in t] for t in p_timeseries]

    p_chunks = [[np.interp(x=np.linspace(0, 2 * np.pi, Npoints), xp=np.linspace(0, 2 * np.pi, dur), fp=ts, left=0,
                           right=0) for dur, ts in zip(durations, timeseries)] for durations, timeseries in
                zip(p_durations, p_timeseries)]
    chunk_dfs = []
    for chunks, chunk_ids, f in zip(p_chunks, p_chunk_ids, filenames):
        chunk_df = pd.DataFrame(np.array(chunks), index=chunk_ids, columns=np.arange(Npoints).tolist())
        chunk_df.to_csv(f'{dir}/{f}', index=True, header=True)
        chunk_dfs.append(chunk_df)
        print(f'Dataset saved as {f}')
    return chunk_dfs

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
    pars = ['length', nam.freq(nam.scal(nam.vel(''))),
            'stride_reoccurence_rate',
            nam.mean(nam.scal(nam.chunk_track('stride', nam.dst('')))),
            nam.std(nam.scal(nam.chunk_track('stride', nam.dst(''))))]
    sample_pars = ['body.initial_length', 'brain.crawler_params.initial_freq',
                   'brain.intermitter_params.crawler_reoccurence_rate',
                   'brain.crawler_params.step_to_length_mu',
                   'brain.crawler_params.step_to_length_std'
                   ]

    v = new_d.endpoint_data[pars]
    v['length'] = v['length'] / 1000
    df = pd.DataFrame(v.values, columns=sample_pars)
    df.to_csv(path_data)

    fit_bouts(new_d, store=True, bouts=['stride', 'pause'])

    dic = {
        nam.freq('crawl'): v[nam.freq(nam.scal(nam.vel('')))].mean(),
        nam.freq('feed'): v[nam.freq('feed')].mean() if nam.freq('feed') in v.columns else 2.0,
        'feeder_reoccurence_rate': None,
        'dt': 1/config['fr'],
    }
    saveConf(dic, conf_type='Ref', id=dataset_id, mode='update')
    z = get_EEB_poly1d(dataset_id)
    saveConf({'EEB_poly1d': z.c.tolist()}, conf_type='Ref', id=dataset_id, mode='update')

    print(f'Reference dataset {dataset_id} saved.')


if __name__ == '__main__':
    from lib.stor.managing import get_datasets

    d = get_datasets(datagroup_id='SimGroup', last_common='single_runs', names=['dish/ppp'], mode='load')[0]
    s = d.step_data
    d.perform_angular_analysis(show_output=True)
