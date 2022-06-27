from typing import List
import lib.aux.dictsNlists as dNl


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


def build_ParsArg(name, k=None, h='', t=float, v=None, vs=None, **kwargs):
    if k is None:
        k = name
    d = {
        'key': name,
        'short': k,
        'help': h,
    }
    if t == bool:
        d['action'] = 'store_true' if not v else 'store_false'
    elif t == List[str]:
        d['type'] = str
        d['nargs'] = '+'
        if vs is not None:
            d['choices'] = vs
    else:
        d['type'] = t
        if vs is not None:
            d['choices'] = vs
        if v is not None:
            d['default'] = v
            d['nargs'] = '?'
    return {name: d}

def build_ParsDict(d0):
    dic = {}
    for n, ndic in d0.items():
        entry = build_ParsArg(name=n, **ndic)
        dic.update(entry)
    return {k: ParsArg(**v) for k, v in dic.items()}
    # return parsargs

def build_ParsDict2(d0):
    from lib.registry.dtypes import par_dict
    dic = par_dict(d0=d0, argparser=True)
    try:
        parsargs = {k: ParsArg(**v) for k, v in dic.items()}
    except:
        parsargs = {}
        for k, v in dic.items():
            for kk, vv in v['content'].items():
                parsargs[kk] = ParsArg(**vv)
    return parsargs


class ParserDict:
    def __init__(self, init_dict=None, names=['sim_params', 'batch_setup','eval_conf','visualization','ga_select_kws'] ):
        if init_dict is None :
            from lib.registry.pars import preg
            init_dict=preg.init_dict
        self.init_dict=init_dict
        self.dict=self.build_parser_dict(names)

    def build_parser_dict(self,names) :
        d=dNl.NestDict()
        for name in names:
            d0 = self.init_dict[name]
            try:
                parsargs = build_ParsDict2(d0)
            except:
                parsargs = build_ParsDict(d0)
            d[name]=parsargs
        return d
