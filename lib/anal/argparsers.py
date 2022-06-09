import copy
from argparse import ArgumentParser
from typing import List

from lib.aux.colsNstr import N_colors
from lib.aux.dictsNlists import AttrDict


from lib.conf.stored.conf import kConfDict
from lib.conf.base.dtypes import null_dict

def build_ParsArg(name,k=None,h='',t=float,v=None,vs=None, **kwargs):
    if k is None :
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

def build_ParsDict(dic_name) :
    from lib.conf.base.init_pars import init_pars
    d0 = init_pars().get(dic_name, None)
    dic = {}
    for n, ndic in d0.items():
        entry = build_ParsArg(name=n, **ndic)
        dic.update(entry)
    parsargs = {k: ParsArg(**v) for k, v in dic.items()}
    return parsargs

def build_ParsDict2(dic_name) :
    from lib.conf.base.dtypes import par_dict
    dic = par_dict(dic_name, argparser=True)
    try:
        parsargs = {k: ParsArg(**v) for k, v in dic.items()}
    except:
        parsargs = {}
        for k, v in dic.items():
            for kk, vv in v['content'].items():
                parsargs[kk] = ParsArg(**vv)
    return parsargs

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


class Parser:
    """
    Create an argument parser for a group of arguments (normally from a dict).
    """

    def __init__(self, name):
        self.name = name
        try :
            self.parsargs =build_ParsDict2(name)
        except :
            self.parsargs = build_ParsDict(name)


    def add(self, parser=None):
        if parser is None:
            parser = ArgumentParser()
        for k, v in self.parsargs.items():
            parser = v.add(parser)
        return parser

    def get(self, input):
        dic = {k: v.get(input) for k, v in self.parsargs.items()}
        # print(dic)
        return null_dict(self.name, **dic)


class MultiParser:
    """
    Combine multiple parsers under a single multi-parser
    """

    def __init__(self, names):
        self.parsers = {n: Parser(n) for n in names}

    def add(self, parser=None):
        if parser is None:
            parser = ArgumentParser()
        for k, v in self.parsers.items():
            parser = v.add(parser)
        return parser

    def get(self, input):
        return AttrDict.from_nested_dicts({k: v.get(input) for k, v in self.parsers.items()})

def update_exp_conf(exp, d=None, N=None, models=None, arena=None, conf_type='Exp', **kwargs):
    from lib.conf.stored.conf import expandConf
    from lib.conf.pars.opt_par import SimParConf
    try :
        exp_conf = expandConf(exp, conf_type)
    except :
        exp_conf = expandConf(exp, conf_type='Exp')
    if arena is not None :
        exp_conf.env_params.arena = arena
    if d is None:
        d = {'sim_params': null_dict('sim_params')}
    exp_conf.sim_params = SimParConf(exp=exp, conf_type=conf_type, **d['sim_params']).dict
    if models is not None:
        if conf_type in ['Exp', 'Eval'] :
            exp_conf = update_exp_models(exp_conf, models)
    if N is not None:
        if conf_type == 'Exp':
            for gID, gConf in exp_conf.larva_groups.items():
                gConf.distribution.N = N
    exp_conf.update(**kwargs)
    return exp_conf


def update_exp_models(exp_conf, models, N=None):
    from lib.conf.stored.conf import expandConf
    larva_groups = {}
    Nmodels = len(models)
    colors = N_colors(Nmodels)
    gConf0 = list(exp_conf.larva_groups.values())[0]
    if isinstance(models, dict):
        for i, ((gID, m), col) in enumerate(zip(models.items(), colors)):
            gConf = AttrDict.from_nested_dicts(copy.deepcopy(gConf0))
            gConf.default_color = col
            gConf.model = m
            larva_groups[gID] = gConf
    elif isinstance(models, list):
        for i, (m, col) in enumerate(zip(models, colors)):
            gConf = AttrDict.from_nested_dicts(copy.deepcopy(gConf0))
            gConf.default_color = col
            if isinstance(m, dict):
                gConf.model = m
                larva_groups[f'LarvaGroup{i}'] = gConf
            elif m in kConfDict('Model'):
                gConf.model = expandConf(m, 'Model')
                larva_groups[m] = gConf
            elif m in kConfDict('Brain'):
                gConf.model = expandConf(m, 'Brain')
                larva_groups[m] = gConf
            else:
                raise ValueError(f'{m} larva-model or brain-model does not exist!')
    if N is not None:
        for gID, gConf in larva_groups.items():
            gConf.distribution.N = N
    exp_conf.larva_groups = larva_groups
    return exp_conf


if __name__ == '__main__':
    conf = update_exp_conf(exp='chemorbit', d=None, N=None, models=None, arena=None, conf_type='Eval')

    print(conf.sim_params)

    # raise