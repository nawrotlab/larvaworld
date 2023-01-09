import copy
import json
import os
import pickle
import numpy as np
import typing

class AttrDict(dict):
    '''
    Dictionary subclass whose entries can be accessed by attributes (as well as normally).
    '''

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    @classmethod
    def from_nested_dicts(cls, data):
        """ Construct nested AttrDicts from nested dictionaries. """
        if not isinstance(data, dict):
            return data
        else:
            return cls({key: cls.from_nested_dicts(data[key]) for key in data})


def NestDict(data=None):
    if data is None:
        return AttrDict()
    else:
        return AttrDict.from_nested_dicts(data)

def copyDict(d):
    return NestDict(copy.deepcopy(d))



def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, typing.MutableMapping):
            if len(v) > 0:
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key,'empty_dict'))

        else:
            items.append((new_key, v))
    return NestDict(dict(items))


def unflatten_dict(d, sep='.'):
    resultDict = NestDict()
    for key, value in d.items():
        if value=='empty_dict' :
            value={}
        parts = key.split(sep)
        dd = resultDict
        for part in parts[:-1]:
            if part not in dd:
                dd[part] = NestDict()
            dd = dd[part]
        dd[parts[-1]] = value
    return resultDict


def merge_dicts(dict_list):
    super_dict = {}
    for d in dict_list:
        for k, v in d.items():
            super_dict[k] = v
    return super_dict


def load_dicts(files=None, pref=None, suf=None, folder=None, extension='txt', use_pickle=True):
    if files is None:
        files = os.listdir(folder)
        suf = extension if suf is None else f'{suf}.{extension}'
        files = [f for f in files if str.endswith(f, suf)]
        if pref is not None:
            files = [f for f in files if str.startswith(f, pref)]
    ds = []
    for f in files:
        n = f'{folder}/{f}' if folder is not None else f
        d = load_dict(n, use_pickle=use_pickle)
        ds.append(d)
    return ds


def load_dict(file, use_pickle=True):
    if use_pickle:
        with open(file, 'rb') as tfp:
            d = pickle.load(tfp)
    else:
        with open(file) as tfp:
            d = json.load(tfp)
    return NestDict(d)

def load_dict2(file):
    try:
        with open(file) as tfp:
            d = json.load(tfp)
    except:
        try:
            with open(file, 'rb') as tfp:
                d = pickle.load(tfp)
        except:
            d= {}
    return NestDict(d)


def save_dict(d, file, use_pickle=True):
    if file is not None :
        if use_pickle:
            with open(file, 'wb') as fp:
                pickle.dump(d, fp, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(file, "w") as fp:
                json.dump(d, fp)
        return True
    else :
        return False





def replace_in_dict(d0, d1, inv_d0=False, inv_d1=False, replace_key=False):
    if inv_d0 :
        d0 = {v0: k0 for k0, v0 in d0.items()}

    if inv_d1:
        d1 = {v0: k0 for k0, v0 in d1.items()}

    if replace_key :
        d=NestDict()
        for k, v in d0.items():
            if k in list(d1.keys()):
                d[d1[k]] = v

    else :
        d = copy.deepcopy(d0)
        for k, v in d.items():  # for each elem in the list datastreams
            if type(v) == dict:
                d[k] = replace_in_dict(v, d1)
            elif v in list(d1.keys()):
                d[k] = d1[v]
    return NestDict(d)


def update_existingdict(dic0,dic):
    dic0.update((k, dic[k]) for k in set(dic).intersection(dic0))
    return dic0


def update_nestdict(dic0, dic):
    dic0_f = flatten_dict(dic0)
    dic0_f.update(dic)
    for k,v in dic0_f.items():
        if v=='empty_dict':
            dic0_f[k]={}
    return NestDict(unflatten_dict(dic0_f))

def update_existingnestdict(dic0, dic):
    dic0_f = flatten_dict(dic0)
    dic0_f = update_existingdict(dic0_f, dic)
    return NestDict(unflatten_dict(dic0_f))


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


def loadSoloDics(agent_ids, path=None, use_pickle=False):
    if os.path.isdir(path) :
        files = [f'{id}.txt' for id in agent_ids]
        return load_dicts(files=files, folder=path, use_pickle=use_pickle)


def storeSoloDics(agent_dics, path=None, use_pickle=False):
    if path is not None :
        os.makedirs(path, exist_ok=True)
        for id, dic in agent_dics.items():
            save_dict(dic, f'{path}/{id}.txt', use_pickle=use_pickle)
