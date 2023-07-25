import os
import pandas as pd




def storeH5(df, path=None, key=None, mode=None, **kwargs):

    if path is not None :
        if mode is None:
            if os.path.isfile(path):
                mode = 'a'
            else:
                mode = 'w'

        if key is not None:

            try:
                store = pd.HDFStore(path, mode=mode)
                store[key] = df
                store.close()
            except:
                if mode == 'a':
                    storeH5(df, path=path, key=key, mode='w', **kwargs)
        elif key is None and isinstance(df, dict):
            store = pd.HDFStore(path, mode=mode)
            for k, v in df.items():
                store[k] = v
            store.close()
        else:
            raise ValueError('H5key not provided.')



def retrieve_results(experiment, id):
    from larvaworld.lib import reg
    f=f'{reg.SIM_DIR}/batch_runs/{experiment}/{id}/results.h5'
    try:
        df =  pd.read_hdf(f, key='results')
    except:
        df =  None
    figs={}
    return df,figs


def delete_traj(experiment, key):
    from larvaworld.lib import reg
    path = f'{reg.SIM_DIR}/batch_runs/{experiment}/{experiment}.hdf5'
    store = pd.HDFStore(path, mode='a')
    del store[key]
    store.close()

