import time
from typing import List



import lib.aux.dictsNlists as dNl
# from lib.conf.stored.conf import loadConfDict
from lib.registry.base import BaseConfDict

from lib.registry.pars import preg
from lib.registry import reg

class ParsArg:
    """
    Create a single parser argument
    This is a class used to populate a parser with arguments and get their values.
    """

    def __init__(self, short, key, **kwargs):
        self.key = key
        self.args = [f'-{short}', f'--{key}']
        self.kwargs = kwargs

    def add(self, p):
        p.add_argument(*self.args, **self.kwargs)
        return p

    def get(self, input):
        return getattr(input, self.key)


def build_ParsArg(name, k=None, h='', dtype=float, v=None, vs=None, **kwargs):
    if k is None:
        k = name
    d = {
        'key': name,
        'short': k,
        'help': h,
    }
    if dtype == bool:
        d['action'] = 'store_true' if not v else 'store_false'
    elif dtype == List[str]:
        d['type'] = str
        d['nargs'] = '+'
        if vs is not None:
            d['choices'] = vs
    elif dtype == List[int]:
        d['type'] = int
        d['nargs'] = '+'
        if vs is not None:
            d['choices'] = vs
    else:
        d['type'] = dtype
        if vs is not None:
            d['choices'] = vs
        if v is not None:
            d['default'] = v
            d['nargs'] = '?'
    return {name: d}


def get_ParsDict(d0):
    dic = {}
    for n, ndic in d0.items():
        entry = build_ParsArg(name=n, **ndic)
        dic.update(entry)
    return dic


def build_ParsDict(dic):
    return dNl.NestDict({k: ParsArg(**v) for k, v in dic.items()})
    # return parsargs


def get_ParsDict2(d0):
    def par(name, dtype=float, v=None, vs=None, h='', k=None, **kwargs):
        return build_ParsArg(name, k, h, dtype, v, vs)

    def par_dict(d0, **kwargs):
        if d0 is None:
            return None
        d = {}
        for n, v in d0.items():
            if 'v' in v.keys() or 'k' in v.keys() or 'h' in v.keys():
                entry = par(n, **v, **kwargs)
            else:
                entry = {n: {'dtype': dict, 'content': par_dict(d0=v, **kwargs)}}
            d.update(entry)
        return d

    dic = par_dict(d0=d0)
    return dic


def build_ParsDict2(dic):
    try:
        parsargs = {k: ParsArg(**v) for k, v in dic.items()}
    except:
        parsargs = {}
        for k, v in dic.items():
            for kk, vv in v['content'].items():
                parsargs[kk] = ParsArg(**vv)
    return parsargs



class ParserDict(BaseConfDict):

    def __init__(self, mode='load',
                 names=['sim_params', 'batch_setup', 'eval_conf', 'visualization', 'ga_select_kws', 'replay']):
        self.names = names
        super().__init__(mode=mode)
        self.parser_dict = self.build_parser_dict()

    def build(self):
        init_dict = preg.init_dict.dict
        d = dNl.NestDict()
        for name in self.names:
            d0 = init_dict[name]
            try:
                d[name] = get_ParsDict2(d0)
            except:
                d[name] = get_ParsDict(d0)
        return d

    def prepare(self, dic):
        try:
            d = build_ParsDict2(dic)
        except:
            d = build_ParsDict(dic)
        return d

    def build_parser_dict(self):
        d = {}
        for name, dic in self.dict.items():
            d[name] = self.prepare(dic)
        return dNl.NestDict(d)

    def retrieve(self, name):
        if name in self.parser_dict.keys():
            return self.parser_dict[name]
        elif name in self.dict.keys():
            self.parser_dict[name] = self.prepare(self.dict[name])
            return self.parser_dict[name]
        else:
            raise


ParsD = ParserDict()
# print('tt')