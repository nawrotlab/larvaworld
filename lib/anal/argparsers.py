import copy
from argparse import ArgumentParser

from lib.aux.colsNstr import N_colors
import lib.aux.dictsNlists as dNl
from lib.registry.pars import preg


#
class Parser:
    """
    Create an argument parser for a group of arguments (normally from a dict).
    """

    # def __init__(self, name):
    #     self.name = name
    #     d0 = ParDict.init_dict[name]
    #     try :
    #         self.parsargs =build_ParsDict2(d0)
    #     except :
    #         self.parsargs = build_ParsDict(d0)

    def __init__(self, name):

        self.name = name
        self.parsargs = preg.parser_dict[name]

    def add(self, parser=None):
        if parser is None:
            parser = ArgumentParser()
        for k, v in self.parsargs.items():
            parser = v.add(parser)
        return parser

    def get(self, input):
        dic = {k: v.get(input) for k, v in self.parsargs.items()}
        #print(self.parsargs)
        d=preg.get_null(self.name, **dic)
        # print(dic)
        return d


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
        return dNl.NestDict({k: v.get(input) for k, v in self.parsers.items()})


def adjust_sim(exp, conf_type, sim):
    if exp is not None and conf_type is not None:
        from lib.conf.stored.conf import next_idx
        if sim.duration is None:
            try:
                exp_conf = preg.loadConf(id=exp, conftype=conf_type)
                sim.duration = exp_conf.sim_params.duration
            except:
                sim.duration = 3.0
        if sim.sim_ID is None:
            sim.sim_ID = f'{exp}_{next_idx(exp, conf_type)}'
        if sim.path is None:
            if conf_type == 'Exp':
                sim.path = f'single_runs/{exp}'
            elif conf_type == 'Ga':
                sim.path = f'ga_runs/{exp}'
            elif conf_type == 'Batch':
                sim.path = f'batch_runs/{exp}'
            elif conf_type == 'Eval':
                sim.path = f'eval_runs/{exp}'
        return sim


def update_exp_conf(exp, d=None, N=None, models=None, arena=None, conf_type='Exp', **kwargs):
    from lib.conf.stored.conf import expandConf
    try:
        exp_conf = expandConf(exp, conf_type)
    except:
        exp_conf = expandConf(exp, conf_type='Exp')
    if arena is not None:
        exp_conf.env_params.arena = arena
    if d is None:
        d = {'sim_params': preg.get_null('sim_params')}

    exp_conf.sim_params = adjust_sim(exp=exp, conf_type=conf_type, sim=dNl.NestDict(d['sim_params']))

    if models is not None:
        if conf_type in ['Exp', 'Eval']:
            exp_conf = update_exp_models(exp_conf, models)
    if N is not None:
        if conf_type == 'Exp':
            for gID, gConf in exp_conf.larva_groups.items():
                gConf.distribution.N = N
    exp_conf.update(**kwargs)
    return exp_conf


def update_exp_models(exp_conf, models, N=None):
    from lib.conf.stored.conf import expandConf, kConfDict
    larva_groups = {}
    Nmodels = len(models)
    colors = N_colors(Nmodels)
    gConf0 = list(exp_conf.larva_groups.values())[0]
    if isinstance(models, dict):
        for i, ((gID, m), col) in enumerate(zip(models.items(), colors)):
            gConf = dNl.NestDict(copy.deepcopy(gConf0))
            gConf.default_color = col
            gConf.model = m
            larva_groups[gID] = gConf
    elif isinstance(models, list):
        for i, (m, col) in enumerate(zip(models, colors)):
            # print(i,m,col)
            gConf = dNl.NestDict(copy.deepcopy(gConf0))
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
