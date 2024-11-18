"""
Classes and methods for managing nested dictionaries and lists
"""

import copy
import json
import pickle
import typing

import agentpy.sequences
import pandas as pd

__all__ = [
    "AttrDict",
    "load_dict",
    "save_dict",
    "bidict",
    "SuperList",
    "ItemList",
    "existing_cols",
    "nonexisting_cols",
    "cols_exist",
    "flatten_list",
    "unique_list",
    "checkEqual",
    "np2Dtotuples",
]


class AttrDict(dict):
    """
    Dictionary subclass whose entries can be accessed as attributes (as well as normally).
    """

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self.autonest(d=self)

    def autonest(self, d):
        for k, data in d.items():
            d[k] = self.from_nested_dicts(data)
        return d

    @classmethod
    def from_nested_dicts(cls, data):
        """Construct nested NestDicts from nested dictionaries."""
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

    def flatten(self, parent_key="", sep="."):
        items = []
        for k, v in self.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, typing.MutableMapping):
                if len(v) > 0:
                    items.extend(AttrDict(v).flatten(new_key, sep=sep).items())
                else:
                    items.append((new_key, "empty_dict"))

            else:
                items.append((new_key, v))
        return AttrDict(dict(items))

    def unflatten(self, sep="."):
        dic = {}
        for k, v in self.items():
            if v == "empty_dict":
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
            k1s = [k0 for k0 in self.keylist if k0.endswith(k)]
            if len(k1s) == 1:
                k1 = k1s[0]
                self[k1] = v
            elif len(k1s) > 1:
                print(f"Non unique suffix : {k}")

    def update_nestdict(self, dic):
        dic0_f = self.flatten()
        dic0_f.update(dic)
        return dic0_f.unflatten()

    def update_nestdict_copy(self, dic):
        return self.get_copy().update_nestdict(dic)

    def new_dict(self, dic):
        return self.get_copy().update_nestdict(dic)

    def update_existingnestdict(self, dic):
        dic0_f = self.flatten()
        dic0_f.update_existingdict(dic)
        return dic0_f.unflatten()

    def update_existingnestdict_by_suffix(self, dic):
        dic0_f = self.flatten()
        dic0_f.update_existingdict_by_suffix(dic)
        return dic0_f.unflatten()

    def save(self, file):
        save_dict(self, file)

    @classmethod
    def load(cls, file):
        return load_dict(file)

    def print(self, flat=False):
        if flat:
            for k, v in self.flatten().items():
                print(f"      {k} : {v}")
        else:
            pref0 = "     "

            def print_nested_level(d, pref=pref0):
                for k, v in d.items():
                    if not isinstance(v, dict):
                        print(f"{pref}{k} : {v}")
                    else:
                        print(f"{pref}{k} : ")
                        print_nested_level(v, pref=f"{pref}{pref0}")

            print_nested_level(self, pref=pref0)

    @property
    def keylist(self):
        return SuperList(self.keys())

    @classmethod
    def merge_dicts(cls, l):
        D = {}
        for d in l:
            for k, v in d.items():
                D[k] = v
        return cls(D)

    @classmethod
    def merge_nestdicts(cls, l):
        D = {}
        for d in l:
            for k, v in AttrDict(d).flatten().items():
                D[k] = v
        return cls(D).unflatten()

    # def keys_on_condition(self, conditions={}):
    #     D=[]
    #     ids=self.keylist
    #     d=AttrDict({id:self[id].flatten() for id in ids})
    #     for id in ids :
    #         for k,v in conditions.items():
    #             try:
    #                 if not self[id][k]==v:
    #                     break
    #             except
    #                 pass


def load_dict(file):
    try:
        with open(file, "rb") as tfp:
            d = pickle.load(tfp)
    except:
        try:
            with open(file) as tfp:
                d = json.load(tfp)
        except:
            d = {}
    return AttrDict(d)


def save_dict(d, file):
    if file is not None:
        try:
            with open(file, "wb") as fp:
                pickle.dump(d, fp, protocol=pickle.HIGHEST_PROTOCOL)
        except:
            with open(file, "w") as fp:
                json.dump(d, fp)
        finally:
            pass
            # print('Failed to save dict')


class bidict(dict):
    def __init__(self, *args, **kwargs):
        super(bidict, self).__init__(*args, **kwargs)
        self.inverse = {}
        for key, value in self.items():
            self.inverse.setdefault(value, []).append(key)

    def __setitem__(self, key, value):
        if key in self:
            self.inverse[self[key]].remove(key)
        super(bidict, self).__setitem__(key, value)
        self.inverse.setdefault(value, []).append(key)

    def __delitem__(self, key):
        self.inverse.setdefault(self[key], []).remove(key)
        if self[key] in self.inverse and not self.inverse[self[key]]:
            del self.inverse[self[key]]
        super(bidict, self).__delitem__(key)


class SuperList(list):
    @property
    def N(self):
        return len(self)

    @property
    def sorted(self):
        return SuperList(sorted(self))

    @property
    def flatten(self):
        l = SuperList()
        for a in self:
            if not isinstance(a, list):
                l.append(a)
            else:
                aa = SuperList(a).flatten
                l.extend(aa)
        return l

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
        g = [self[i * n : (i + 1) * n] for i in range(N)]
        if Nmore != 0:
            g.append(self[-Nmore:])
        return SuperList(g)

    @property
    def in_pairs(self):
        return self.group_by_n(n=2)

    def existing(self, df):
        return SuperList(existing_cols(self, df))

    def nonexisting(self, df):
        return SuperList(nonexisting_cols(self, df))

    def exist_in(self, df):
        return cols_exist(self, df)

    def __add__(self, *args, **kwargs):  # real signature unknown
        """Return self+value."""
        return SuperList(super().__add__(*args, **kwargs))

    def suf(self, suf=""):
        return SuperList([i for i in self if i.endswith(suf)])

    def pref(self, pref=""):
        return SuperList([i for i in self if i.startswith(pref)])

    def contains(self, l=""):
        return SuperList([i for i in self if l in i])


class ItemList(agentpy.sequences.AgentSequence, list):
    def __init__(self, objs=(), cls=None, *args, **kwargs):
        if isinstance(objs, int):
            objs = self._obj_gen(objs, cls, *args, **kwargs)
        super().__init__(objs)

    @staticmethod
    def _obj_gen(n, cls, *args, **kwargs):
        """Generate objects for sequence."""
        for i in range(n):
            # AttrIter values get broadcasted among agents
            i_kwargs = {
                k: arg[i] if isinstance(arg, list) else arg for k, arg in kwargs.items()
            }
            yield cls(**i_kwargs)

    def __setattr__(self, name, value):
        if isinstance(value, list):
            # Apply each value to each agent
            for obj, v in zip(self, value):
                setattr(obj, name, v)
        else:
            # Apply single value to all agents
            for obj in self:
                setattr(obj, name, value)


def existing_cols(cols, df):
    if isinstance(df, pd.DataFrame):
        df = df.columns.values
    return [col for col in cols if col in df]


def nonexisting_cols(cols, df):
    if isinstance(df, pd.DataFrame):
        df = df.columns.values
    return [col for col in cols if col not in df]


def cols_exist(cols, df):
    if isinstance(df, pd.DataFrame):
        df = df.columns.values
    return set(cols).issubset(df)


def flatten_list(l):
    return [item for sublist in l for item in sublist]


def checkEqual(l1, l2):
    for a in l1:
        if a not in l2:
            return False
    for a in l2:
        if a not in l1:
            return False
    return True


def unique_list(l):
    if len(l) == 0:
        return SuperList()
    elif len(l) == 1:
        return l
    else:
        seen = set()
        seen_add = seen.add
        return SuperList([x for x in l if not (x in seen or seen_add(x))])


def np2Dtotuples(a):
    if isinstance(a, list) and all([isinstance(aa, tuple) for aa in a]):
        return a
    else:
        return list(zip(a[:, 0], a[:, 1]))
