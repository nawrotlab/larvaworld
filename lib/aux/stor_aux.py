import os
import shutil

import pandas as pd
from lib.aux import dictsNlists as dNl
from lib.registry import reg



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
    storeH5(dic, key=None, filepath_key='distro', parent_dir=parent_dir)



def get_path(path=None, key=None, filepath_key=None, filepath_dic=None, parent_dir=None):
    if path is not None:
        return path
    if filepath_key is None:
        if key is not None:
            filepath_key = key
        else:
            print('None of file or key are provided.Returning None')
            return None

    # h5_key = key

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
            path = datapath(filepath_key, parent_dir)
        else:
            print('Filepath not provided.Returning None')
            return None
    return path


def read(path=None, key=None, mode='r',**kwargs):
    path = get_path(path=path, key=key, **kwargs)
    if path is None :
        # print(f'Filepath not provided for key {key} H5.Returning None')
        return None
    elif not os.path.isfile(path):
        # print(f'File H5 does not exist at {path}.Returning None')
        return None
    if key is not None:
        return pd.read_hdf(path, key=key, mode=mode)
    else:
        store = pd.HDFStore(path)
        ks = list(store.keys())
        if len(ks) == 1:
            df = store[ks[0]]
            store.close()
            return df

        # if not all_keys :
        #     return pd.read_hdf(path, key=key, mode='r')
        else:
            dic = {k: store[k] for k in store.keys()}
            store.close()
            return dic


def storeH5(df, path=None, key=None, mode=None, **kwargs):
    if path is not None :
        if mode is None:
            if os.path.isfile(path):
                mode = 'a'
            else:
                mode = 'w'

        if key is not None:
            store = pd.HDFStore(path, mode=mode)
            store[key] = df
            store.close()
        elif key is None and isinstance(df, dict):
            store = pd.HDFStore(path, mode=mode)
            for k, v in df.items():
                store[k] = v
            store.close()
        else:
            raise ValueError('H5key not provided.')





def loadSoloDics(agent_ids, path=None, use_pickle=False):
    if os.path.isdir(path) :
        files = [f'{id}.txt' for id in agent_ids]
        return dNl.load_dicts(files=files, folder=path, use_pickle=use_pickle)
        # except:
        #     ds = dNl.load_dicts(files=files, folder=path, use_pickle=True)
        # return ds


def storeSoloDics(agent_dics, path=None, use_pickle=False):
    if path is not None :
        os.makedirs(path, exist_ok=True)
        for id, dic in agent_dics.items():
            dNl.save_dict(dic, f'{path}/{id}.txt', use_pickle=use_pickle)
        # except:
        #     dNl.save_dict(dic, filepath, use_pickle=not use_pickle)


#
# class C(object):
#     def __init__(self, path):
#         self.path=path
#         #self._x = None
#
#     @property
#     def x(self):
#         """I'm the 'x' property."""
#         print("getter of x called")
#         return loadDic(path=self.path)
#
#     @x.setter
#     def x(self, d):
#         print("setter of x called")
#         storeDic(d, path=self.path)
#
#     # @x.deleter
#     # def x(self):
#     #     print("deleter of x called")
#     #     del self._x
# if __name__ == '__main__':
#     print(datapath('dsp', 'my/ddd'))