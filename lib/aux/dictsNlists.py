import collections
import copy
import json
import os
import pickle
import sys
from collections import deque
from itertools import groupby

from pypet import ParameterGroup, Parameter


def flatten_tuple(test_tuple):
    res = []
    if isinstance(test_tuple, tuple):
        for i in test_tuple:
            if isinstance(i, tuple):
                for j in i:
                    res.append(j)
            else:
                res.append(i)
        return tuple(res)

    # res = []
    # for sub in test_tuple:
    #     res += flatten_tuple(sub)
    # return tuple(res)


def flatten_list(l):
    return [item for sublist in l for item in sublist]


def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            if len(v) > 0:
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, 'empty_dict'))
        else:
            items.append((new_key, v))
    return dict(items)


def reconstruct_dict(param_group, **kwargs):
    dict = {}
    for p in param_group:
        if type(p) == ParameterGroup:
            d = reconstruct_dict(p)
            dict.update({p.v_name: d})
        elif type(p) == Parameter:
            if p.f_is_empty():
                dict.update({p.v_name: None})
            else:
                v = p.f_get()
                if v == 'empty_dict':
                    v = {}
                dict.update({p.v_name: v})
    dict.update(**kwargs)
    return dict


def group_list_by_n(l, n):
    Nmore=int(len(l) % n)
    N=int((len(l)-Nmore) / n)
    # if not len(l) % n == 0.0:
    #     raise ValueError('List length must be multiple of n')
    g= [l[i * n:(i + 1) * n] for i in range(N)]
    if Nmore!=0 :
        g.append(l[-Nmore:])
    return g


def common_member(a, b):
    a_set = set(a)
    b_set = set(b)
    return a_set & b_set


def merge_dicts(dict_list) :
    # import collections
    super_dict = {}
    # super_dict = collections.defaultdict(set)
    for d in dict_list:
        for k, v in d.items():  # d.items() in Python 3+
            super_dict[k]=v
            # super_dict[k].add(v)
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
        d=load_dict(n, use_pickle=use_pickle)
        # if use_pickle :
        #     with open(n, 'rb') as tfp:
        #         d = pickle.load(tfp)
        # else :
        #     with open(n) as tfp:
        #         d = json.load(tfp)
        ds.append(d)
    return ds


def load_dict(file, use_pickle=True) :
    if use_pickle:
        with open(file, 'rb') as tfp:
            d = pickle.load(tfp)
    else:
        with open(file) as tfp:
            d = json.load(tfp)
    return d


def save_dict(d, file, use_pickle=True) :
    if use_pickle :
        with open(file, 'wb') as fp:
            pickle.dump(d, fp, protocol=pickle.HIGHEST_PROTOCOL)
    else :
        with open(file, "w") as fp:
            json.dump(d, fp)


def depth(d):
    queue = deque([(id(d), d, 1)])
    memo = set()
    while queue:
        id_, o, level = queue.popleft()
        if id_ in memo:
            continue
        memo.add(id_)
        if isinstance(o, dict):
            queue += ((id(v), v, level + 1) for v in o.values())
    return level


def print_dict(d):
    l = depth(d)
    for k, v in d.items():
        if isinstance(v, dict):
            print('----' * l, k, '----' * l)
            print_dict(v)
        else:
            print(k, ':', v)
    print()


def dict_to_file(dictionary, filename):
    orig_stdout = sys.stdout
    f = open(filename, 'w')
    sys.stdout = f
    print_dict(dictionary)
    sys.stdout = orig_stdout
    f.close()
    # sys.stdout = open(filename, 'W')
    # sys.stdout = stdout
    # with open(filename, 'W') as sys.stdout: print_dict(dictionary)


def unique_list(l):
    if len(l)==0:
        return []
    elif len(l)==1:
        return l
    else :
        seen = set()
        seen_add = seen.add
        return [x for x in l if not (x in seen or seen_add(x))]


def replace_in_dict(d0, replace_d, inverse=False) :
    d = copy.deepcopy(d0)
    if inverse :
        replace_d={v0:k0 for k0,v0 in replace_d.items()}
    for k,v in d.items():  # for each elem in the list datastreams
        if type(v)==dict :
            d[k]=replace_in_dict(v, replace_d, inverse=False)
        elif v in list(replace_d.keys()):
            d[k] = replace_d[v]
    return d

def group_dicts(dics):
  if all(not isinstance(i, dict) for i in dics):
    return [i for b in dics for i in b]
  r = [i for b in dics for i in b.items()]
  _d = [[a, [c for _, c in b]] for a, b in groupby(sorted(r, key=lambda x:x[0]), key=lambda x:x[0])]
  return {a:b[0] if len(b) == 1 else group_dicts(b) for a, b in _d}

def merge(item):
  from collections import defaultdict
  merged = defaultdict(list)
  for ref in item.get('ref', []):
    for key, val in ref.items():
      merged[key].append(val)
  return {**item, 'ref': dict(merged)}

class AttrDict(dict):
    '''
    Dictionary subclass whose entries can be accessed by attributes (as well
        as normally).
    # >>> obj = AttrDict()
    # >>> obj['test'] = 'hi'
    # >>> print obj.test
    hi
    # >>> del obj.test
    # >>> obj.test = 'bye'
    # >>> print obj['test']
    bye
    # >>> print len(obj)
    1
    # >>> obj.clear()
    # >>> print len(obj)
    0
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


if __name__ == '__main__':

    data = {
        "a": "aval",
        "b": {
            "b1": {
                "b2b": "b2bval",
                "b2a": {
                    "b3a": "b3aval",
                    "b3b": "b3bval"
                }
            }
        }
    }

    data1 = AttrDict.from_nested_dicts(data)
    print(data1.b.b1.b2a.b3b)  # -> b3bval

