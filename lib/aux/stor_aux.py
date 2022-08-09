import os
import shutil

import pandas as pd
from lib.aux import dictsNlists as dNl




def data_key_dict():
    kd = dNl.NestDict()
    kd.solo_dicts = ['bouts', 'foraging', 'deb', 'nengo']

    kd.folders = {
        'parent': ['data', 'plots', 'visuals', 'aux', 'model'],
        'data': ['individuals'],
        'individuals': kd.solo_dicts,
        'plots': ['model_tables', 'model_summaries'],
        'model': ['GAoptimization', 'evaluation'],

    }

    h5base = ['end', 'step']
    kd.h5step = ['contour', 'midline', 'epochs', 'base_spatial', 'angular', 'dspNtor']
    h5aux = ['derived', 'traj', 'aux', 'vel_definition', 'tables', 'food', 'distro']

    kd.h5 = h5base + kd.h5step + h5aux

    confs = ['conf', 'sim_conf', 'log']
    dics1 = ['chunk_dicts','grouped_epochs', 'pooled_epochs', 'cycle_curves', 'dsp', 'fit']
    dics2 = ['ExpFitter']

    kd.dic = dics1 + dics2 + confs

    return kd


kd = data_key_dict()


def DataFunc(filepath_key, mode='load'):
    func_dic = {'h5':
                    {'load': read, 'save': storeH5},
                'dic': {'load': loadDic, 'save': storeDic},

                'solo_dicts': {'load': loadSoloDics, 'save': storeSoloDics}
                }
    for k in func_dic.keys():
        if filepath_key in kd[k]:
            return func_dic[k][mode]
    raise ValueError(f'Filepath key {filepath_key}  does not exist')


def define_filepath_dict():
    d = dNl.NestDict()
    d.parent = ''
    for k0, ks in kd.folders.items():
        for k in ks:
            d[k] = f'{d[k0]}/{k}'

    for k in kd.h5:
        d[k] = f'{d.data}/{k}.h5'
    for k in kd.dic:
        d[k] = f'{d.data}/{k}.txt'
    return d


filepath_dict = define_filepath_dict()

print(filepath_dict.deb)


def datapath(filepath_key, dir=None):
    if dir is not None :
        v = filepath_dict[filepath_key]
        return f'{dir}{v}'
    else :
        return None


def get_dir_dict(dir):
    return dNl.NestDict({k: f'{dir}{v}' for k, v in filepath_dict.items()})


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


def read(path=None, key=None, **kwargs):
    path = get_path(path=path, key=key, **kwargs)
    if path is None :
        # print(f'Filepath not provided for key {key} H5.Returning None')
        return None
    elif not os.path.isfile(path):
        # print(f'File H5 does not exist at {path}.Returning None')
        return None
    if key is not None:
        return pd.read_hdf(path, key=key, mode='r')
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
    path = get_path(path=path, key=key, **kwargs)
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


def loadDic(path=None, key=None, use_pickle=True, **kwargs):
    path = get_path(path=path, key=key, **kwargs)
    if path is None :
        # print(f'Filepath not provided for key {key} Dict.Returning None')
        return None
    elif not os.path.isfile(path):
        # print(f'File of type Dict does not exist at {path}.Returning None')
        return None
    else:
        return dNl.load_dict(path, use_pickle=use_pickle)


def storeDic(d, path=None, key=None, use_pickle=True, **kwargs):
    path = get_path(path=path, key=key, **kwargs)
    if path is not None :
        # shutil.rmtree(path,ignore_errors=True)
    # os.makedirs(path, exist_ok=True)
        dNl.save_dict(d, path, use_pickle=use_pickle)


def loadSoloDics(agent_ids, path=None, key=None, **kwargs):
    path = get_path(path=path, key=key, **kwargs)
    if path is None :
        # print(f'Filepath not provided for key {key} SoloDicts.Returning None')
        return None
    elif not os.path.isfile(path):
        # print(f'File of type SoloDicts does not exist at {path}.Returning None')
        return None
    else:
        files = [f'{id}.txt' for id in agent_ids]
        try:
            ds = dNl.load_dicts(files=files, folder=path, use_pickle=False)
        except:
            ds = dNl.load_dicts(files=files, folder=path, use_pickle=True)
        return ds


def storeSoloDics(agent_dics, path=None, key=None, **kwargs):
    path = get_path(path=path, key=key, **kwargs)
    os.makedirs(path, exist_ok=True)
    for id, dic in agent_dics.items():
        filepath = f'{path}/{id}.txt'
        try:
            dNl.save_dict(dic, filepath, use_pickle=False)
        except:
            dNl.save_dict(dic, filepath, use_pickle=True)


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
