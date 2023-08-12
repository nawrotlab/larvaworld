import copy
import json
import os
import pickle
import numpy as np
import typing

import pandas as pd



class AttrDict(dict):
    '''
    Dictionary subclass whose entries can be accessed as attributes (as well as normally).
    '''

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self.autonest(d=self)


    def autonest(self, d):
        for k, data in d.items():
            d[k] = self.from_nested_dicts(data)
        return d

    @classmethod
    def from_nested_dicts(cls, data):
        """ Construct nested NestDicts from nested dictionaries. """
        if not isinstance(data, dict):
            return data
        else:
            return cls(data)

    def replace_keys(self, pairs={}):
        dic = {}
        for k, v in self.items():
            if k in list(pairs.keys()):
                dic[pairs[k]] = v
            else:
                dic[k] = v
        return AttrDict(dic)

    def get_copy(self):
        return AttrDict(copy.deepcopy(self))

    def flatten(self, parent_key='', sep='.'):
        items = []
        for k, v in self.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, typing.MutableMapping):
                if len(v) > 0:
                    items.extend(AttrDict(v).flatten(new_key, sep=sep).items())
                    # items.extend(flatten_dict(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, 'empty_dict'))

            else:
                items.append((new_key, v))
        return AttrDict(dict(items))

    def unflatten(self, sep='.'):
        dic = {}
        for k, v in self.items():
            if v == 'empty_dict':
                v = {}
            parts = k.split(sep)
            dd = dic
            for part in parts[:-1]:
                if part not in dd:
                    dd[part] = {}
                dd = dd[part]
            dd[parts[-1]] = v
        return AttrDict(dic)

    def update_existingdict(self, dic):
        for k, v in dic.items():
            if k in list(self.keys()):
                self[k] = v

    def update_existingdict_by_suffix(self, dic):
        for k, v in dic.items():
            k1s=[k0 for k0 in self.keylist if k0.endswith(k)]
            if len(k1s)==1:
                k1=k1s[0]
                self[k1] = v
            elif len(k1s) > 1:
                print(f'Non unique suffix : {k}')

    def update_nestdict(self, dic):
        dic0_f = self.flatten()
        dic0_f.update(dic)
        return dic0_f.unflatten()

    def update_existingnestdict(self, dic):
        dic0_f = self.flatten()
        dic0_f.update_existingdict(dic)
        return dic0_f.unflatten()

    def update_existingnestdict_by_suffix(self, dic):
        dic0_f = self.flatten()
        dic0_f.update_existingdict_by_suffix(dic)
        return dic0_f.unflatten()

    def save(self, file):
        save_dict(self,file)

    @classmethod
    def load(cls, file):
        return load_dict(file)

    def print(self, flat=False):
        if flat :
            for k, v in self.flatten().items():
                print(f'      {k} : {v}')
        else:
            pref0='     '
            def print_nested_level(d, pref=pref0):
                for k, v in d.items():
                    if not isinstance(v, dict):
                        print(f'{pref}{k} : {v}')
                    else:
                        print(f'{pref}{k} : ')
                        print_nested_level(v, pref=f'{pref}{pref0}')

            print_nested_level(self, pref=pref0)

    @property
    def keylist(self):
        return SuperList(self.keys())

# class StoreDict(AttrDict):
#
#
#     def __new__(cls,path=None):
#         cls.path = path
#         if path is not None :
#             return cls.retrieve(path)
#         else :
#             return super().__new__(cls)
#
#
#
#
#     # def __init__(self,*args, **kwargs):
#     #     super().__init__(*args, **kwargs)
#
#     def store(self):
#         super().save(self.path)
#
#     @classmethod
#     def retrieve(cls,path):
#         d=super().load(path)
#         return super().__new__(cls,**d)




def load_dict(file):
    try:
        with open(file, 'rb') as tfp:
            d = pickle.load(tfp)
            # print('pickle load')
    except:
        try:
            with open(file) as tfp:
                d = json.load(tfp)
                # print('json load')
        except:

            d= {}
    return AttrDict(d)


def save_dict(d, file):
    if file is not None :
        try:
            with open(file, 'wb') as fp:
                pickle.dump(d, fp, protocol=pickle.HIGHEST_PROTOCOL)
                # print('pickle save')
        except:
            try:
                with open(file, "w") as fp:
                    json.dump(d, fp)
                    # print('json save')
            except:

                raise
        return True
    else :
        return False


def merge_dicts(dict_list):
    super_dict = {}
    for d in dict_list:
        for k, v in d.items():
            super_dict[k] = v
    return super_dict


def load_dicts(files=None, pref=None, suf=None, folder=None, extension='txt'):
    if files is None:
        files = os.listdir(folder)
        suf = extension if suf is None else f'{suf}.{extension}'
        files = [f for f in files if str.endswith(f, suf)]
        if pref is not None:
            files = [f for f in files if str.startswith(f, pref)]
    ds = []
    for f in files:
        n = f'{folder}/{f}' if folder is not None else f
        d = load_dict(n)
        ds.append(d)
    return ds

def loadSoloDics(agent_ids, path=None):
    if os.path.isdir(path) :
        files = [f'{id}.txt' for id in agent_ids]
        return load_dicts(files=files, folder=path)


def storeSoloDics(agent_dics, path=None):
    if path is not None :
        os.makedirs(path, exist_ok=True)
        for id, dic in agent_dics.items():
            save_dict(dic, f'{path}/{id}.txt')

def group_epoch_dicts(individual_epochs):
    keys = ['turn_dur', 'turn_amp', 'turn_vel_max', 'run_dur', 'run_dst', 'pause_dur', 'run_count']
    return {k: np.array(flatten_list([dic[k] for id,dic in individual_epochs.items()])) for k in keys}

class bidict(dict):
    def __init__(self, *args, **kwargs):
        super(bidict, self).__init__(*args, **kwargs)
        self.inverse = {}
        for key, value in self.items():
            self.inverse.setdefault(value,[]).append(key)

    def __setitem__(self, key, value):
        if key in self:
            self.inverse[self[key]].remove(key)
        super(bidict, self).__setitem__(key, value)
        self.inverse.setdefault(value,[]).append(key)

    def __delitem__(self, key):
        self.inverse.setdefault(self[key],[]).remove(key)
        if self[key] in self.inverse and not self.inverse[self[key]]:
            del self.inverse[self[key]]
        super(bidict, self).__delitem__(key)

class SuperList(list) :

    @property
    def N(self):
        return len(self)

    @property
    def sorted(self):
        return sorted(self)

    @property
    def flatten(self):
        return SuperList([item for sublist in self for item in sublist])

    @property
    def unique(self):
        if len(self) == 0:
            return SuperList()
        elif len(self) == 1:
            return self
        else:
            seen = set()
            seen_add = seen.add
            return SuperList([x for x in self if not (x in seen or seen_add(x))])

    def group_by_n(self, n=2):
        Nmore = int(len(self) % n)
        N = int((len(self) - Nmore) / n)
        g = [self[i * n:(i + 1) * n] for i in range(N)]
        if Nmore != 0:
            g.append(self[-Nmore:])
        return SuperList(g)

    @property
    def in_pairs(self):
        return self.group_by_n(n=2)


    def existing(self, df):
        return SuperList(existing_cols(self,df))

    def nonexisting(self, df):
        return SuperList(nonexisting_cols(self,df))

    def exist_in(self, df):
        return cols_exist(self,df)

    def __add__(self, *args, **kwargs): # real signature unknown
        """ Return self+value. """
        return SuperList(super().__add__(*args, **kwargs))



def existing_cols(cols,df) :
    if isinstance(df, pd.DataFrame):
        df=df.columns.values
    return [col for col in cols if col in df]

def nonexisting_cols(cols,df) :
    if isinstance(df, pd.DataFrame):
        df=df.columns.values
    return [col for col in cols if col not in df]

def cols_exist(cols,df) :
    if isinstance(df, pd.DataFrame):
        df=df.columns.values
    return set(cols).issubset(df)


def flatten_list(l):
    return [item for sublist in l for item in sublist]

def group_list_by_n(l, n):
    Nmore = int(len(l) % n)
    N = int((len(l) - Nmore) / n)
    g = [l[i * n:(i + 1) * n] for i in range(N)]
    if Nmore != 0:
        g.append(l[-Nmore:])
    return g



def unique_list(l):
    if len(l) == 0:
        return []
    elif len(l) == 1:
        return l
    else:
        seen = set()
        seen_add = seen.add
        return [x for x in l if not (x in seen or seen_add(x))]


def checkEqual(L1, L2):
    return len(L1) == len(L2) and sorted(L1) == sorted(L2)

def np2Dtotuples(a):
    if isinstance(a, list) and all([isinstance(aa, tuple) for aa in a]):
        return a
    else:
        return list(zip(a[:,0], a[:,1]))

