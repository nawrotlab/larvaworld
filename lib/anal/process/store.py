import numpy as np
import pandas as pd
from scipy.signal import argrelextrema, spectrogram

import lib.aux.functions as fun
import lib.aux.naming as nam
import lib.conf.dtype_dicts as dtypes


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


if __name__ == '__main__':
    from lib.stor.managing import get_datasets

    d = get_datasets(datagroup_id='SimGroup', last_common='single_runs', names=['dish/ppp'], mode='load')[0]
    s = d.step_data
    d.perform_angular_analysis(show_output=True)
