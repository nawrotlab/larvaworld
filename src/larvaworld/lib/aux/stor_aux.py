import os
import pandas as pd


from larvaworld.lib import aux




def get_distros(s, pars):
    ps = [p for p in pars if p in s.columns]
    dic = {}
    for p in ps:
        df = s[p].dropna().reset_index(level=0, drop=True)
        df.sort_index(inplace=True)
        dic[p] = df
    return dic


def store_distros(s, pars, parent_dir):
    dic = get_distros(s, pars=pars)
    storeH5(dic, filepath_key='distro', parent_dir=parent_dir)



def get_path(path=None, key=None, filepath_key=None, filepath_dic=None, parent_dir=None):
    if path is not None:
        return path
    if filepath_key is None:
        if key is not None:
            filepath_key = key
        else:
            print('None of file or key are provided.Returning None')
            return None

    if filepath_key == 'data_h5':
        filepath_key = 'step'
        print('Change this')
    if filepath_key == 'endpoint_h5':
        filepath_key = 'end'
        print('Change this')

    if filepath_dic is not None:
        path = filepath_dic[filepath_key]
    else:
        if parent_dir is not None:
            from larvaworld.lib import reg
            path = reg.datapath(filepath_key, parent_dir)
        else:
            print('Filepath not provided.Returning None')
            return None
    return path


def read(path=None, key=None, mode='r',**kwargs):
    path = get_path(path=path, key=key, **kwargs)
    if path is None :
        return None
    elif not os.path.isfile(path):
        return None
    if key is not None:
        try :
            return pd.read_hdf(path, key=key, mode=mode)
        except :
            return None
    else:
        store = pd.HDFStore(path)
        ks = list(store.keys())
        if len(ks) == 1:
            df = store[ks[0]]
            store.close()
            return df
        else :
            dd=aux.AttrDict()
            for k in ks :
                dd[k]=store[k]
            store.close()
            return dd





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



def retrieve_results(batch_type, id):
    from larvaworld.lib import reg
    f=f'{reg.SIM_DIR}/batch_runs/{batch_type}/{id}/results.h5'
    df = read(path=f, key='results')
    figs={}
    return df,figs


def delete_traj(batch_type, key):
    from larvaworld.lib import reg
    path = f'{reg.SIM_DIR}/batch_runs/{batch_type}/{batch_type}.hdf5'
    store = pd.HDFStore(path, mode='a')
    del store[key]
    store.close()

