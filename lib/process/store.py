import numpy as np
import pandas as pd

import lib.aux.naming as nam

def get_dsp(s, p):
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
    return pd.DataFrame(dsp_ar, index=steps, columns=['median', 'upper', 'lower'])



def store_aux_dataset(s, pars, type, file, verbose=0):
    store = pd.HDFStore(file)
    ps = [p for p in pars if p in s.columns]
    if type == 'distro':
        for p in ps:
            d = s[p].dropna().reset_index(level=0, drop=True)
            d.sort_index(inplace=True)
            store[f'{type}.{p}'] = d

    elif type == 'trajectories':
        store['trajectories'] = s[['x', 'y']]
    elif type == 'dispersion':
        for p in ps:
            d=get_dsp(s, p)
            store[f'{type}.{p}'] = d
    elif type == 'stride':
        Npoints = 32
        x = np.linspace(0, 2 * np.pi, Npoints)
        columns = np.arange(Npoints).tolist()
        pID=nam.id(type)
        ss=s[s[pID].notnull()][ps+[pID]].reset_index(level='Step', drop=True)
        if len(ss)>0 :
            sss=ss.groupby(['AgentID', pID])
            def func(ts) :
                return np.interp(x=x, xp=np.linspace(0, 2 * np.pi, ts.shape[0]), fp=ts, left=0,right=0)
            for p in ps :
                df=sss[p].apply(func).reset_index(level=pID, drop=True)
                df0 = pd.DataFrame(df.values.tolist(),index=df.index, columns=columns)
                store[f'{type}.{p}'] = df0
    store.close()
    if verbose>1 :
        print(f'{len(ps)} aux parameters saved')