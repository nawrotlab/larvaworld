from typing import List
from argparse import ArgumentParser


from lib import reg, aux, sim



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


def build_ParsArg(name, k=None, h='', dtype=float, v=None, vs=None):
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


def par_dict(d0):
    if d0 is None:
        return None

    def par(name, dtype=float, v=None, vs=None, h='', k=None, **kwargs):
        return build_ParsArg(name, k, h, dtype, v, vs)

    d = {}
    for n, v in d0.items():
        if 'v' in v.keys() or 'k' in v.keys() or 'h' in v.keys():
            entry = par(n, **v)
        else:
            entry = {n: {'dtype': dict, 'content': par_dict(d0=v)}}
        d.update(entry)
    return d


def parser_dict(name):
    dic = par_dict(reg.par.PI[name])
    try:
        parsargs = {k: ParsArg(**v) for k, v in dic.items()}
    except:
        parsargs = {}
        for k, v in dic.items():
            for kk, vv in v['content'].items():
                parsargs[kk] = ParsArg(**vv)
    return aux.AttrDict(parsargs)


class Parser:
    """
    Create an argument parser for a group of arguments (normally from a dict).
    """

    def __init__(self, name):
        self.name = name
        self.parsargs = parser_dict(name)
        # self.parsargs = reg.parsers.parser_dict[name]

    def add(self, parser=None):
        if parser is None:
            parser = ArgumentParser()
        for k, v in self.parsargs.items():
            parser = v.add(parser)
        return parser

    def get(self, input):
        dic = {k: v.get(input) for k, v in self.parsargs.items()}
        d = reg.get_null(name=self.name, **dic)
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
        return aux.AttrDict({k: v.get(input) for k, v in self.parsers.items()})


def update_exp_conf(exp, N=None, mIDs=None):
    conf = reg.expandConf(id=exp, conftype='Exp')
    conf.experiment = exp

    if mIDs is not None:
        Nm = len(mIDs)
        gConfs = list(conf.larva_groups.values())
        if len(conf.larva_groups) != Nm:
            gConfs = [gConfs[0]] * Nm
            for gConf, col in zip(gConfs, aux.N_colors(Nm)):
                gConf.default_color = col
        conf.larva_groups = aux.AttrDict({mID: {} for mID in mIDs})
        for mID, gConf in zip(mIDs, gConfs):
            conf.larva_groups[mID] = gConf
            conf.larva_groups[mID].model = reg.loadConf('Model', mID)

    if N is not None:
        for gID, gConf in conf.larva_groups.items():
            gConf.distribution.N = N

    return conf

def run_template(sim_mode, args, d):
    kws={'id' : args.id}
    if sim_mode == 'Replay':
        run = sim.ReplayRun(**d['Replay'], **kws)
        run.run()
    elif sim_mode == 'Batch':
        conf = reg.loadConf(conftype='Batch', id=args.experiment)
        conf.batch_type = args.experiment
        conf.exp = update_exp_conf(conf.exp, N=args.Nagents, mIDs=args.models)
        if args.duration is not None:
            conf.exp.sim_params.duration = args.duration
        conf.exp.sim_params.store_data = args.store_data
        conf.exp.sim_params.Box2D = args.Box2D
        exec = sim.Exec(mode='batch', conf=conf, run_externally=False, **kws)
        exec.run()
    elif sim_mode == 'Exp':
        conf = update_exp_conf(exp=args.experiment, N=args.Nagents, mIDs=args.models)
        if args.duration is not None:
            conf.sim_params.duration = args.duration
        conf.sim_params.store_data = args.store_data
        conf.sim_params.Box2D = args.Box2D


        run = sim.ExpRun(parameters=conf,
                     screen_kws={'vis_kwargs': d['visualization']}, **kws)
        ds = run.simulate()
        if args.analysis:
            run.analyze(show=args.show)

    elif sim_mode == 'Ga':
        conf = reg.expandConf(id=args.experiment, conftype='Ga')
        conf.experiment = args.experiment
        conf.offline=args.offline
        conf.show_screen=args.show_screen
        if args.duration is not None:
            conf.sim_params.duration = args.duration
        conf.sim_params.store_data = args.store_data
        conf.ga_select_kws = d['ga_select_kws']

        if args.base_model is not None:
            conf.ga_build_kws.base_model = args.base_model
        if args.bestConfID is not None:
            conf.ga_build_kws.bestConfID = args.bestConfID
        GA = sim.GAlauncher(parameters=conf, **kws)
        best_genome = GA.simulate()
    elif sim_mode == 'Eval':
        evrun = sim.EvalRun(**d.Eval, show=args.show_screen, **kws)
        evrun.simulate()
        evrun.plot_results()
        evrun.plot_models()


def get_parser(sim_mode, parser=None):
    dic = aux.AttrDict({
        'Batch': [[], ['e','t','Box2D', 'N', 'ms']],
        'Eval': [['Eval'], ['hide']],
        'Exp': [['visualization'], ['e','t','Box2D', 'N', 'ms', 'a']],
        'Ga': [['ga_select_kws'], ['e','t', 'mID0', 'mID1', 'offline', 'hide']],
        'Replay': [['Replay'], []]
    })
    mks, ks = dic[sim_mode]

    MP = MultiParser(mks)
    p = MP.add(parser)
    p.add_argument('-id', '--id', type=str, help='The simulation ID. If not provided a default is generated')
    p.add_argument('-no_store', '--store_data', action="store_false", help='Whether to store the simulation data or not')
    for k in ks:
        if k == 'e':
            p.add_argument('experiment', choices=reg.storedConf(sim_mode), help='The experiment mode')
        elif k == 't':
            p.add_argument('-t', '--duration', type=float, help='The duration of the simulation in minutes')
        elif k == 'Box2D':
            p.add_argument('-Box2D', '--Box2D', action="store_true", help='Whether to use the Box2D physics engine or not')
        elif k == 'N':
            p.add_argument('-N', '--Nagents', type=int, help='The number of simulated larvae in each larva group')
        elif k == 'ms':
            p.add_argument('-ms', '--models', type=str, nargs='+',
                           help='The larva models to use for creating the simulation larva groups')
        elif k == 'mID0':
            p.add_argument('-mID0', '--base_model', choices=reg.storedConf('Model'),
                           help='The model configuration to optimize')
        elif k == 'mID1':
            p.add_argument('-mID1', '--bestConfID', type=str,
                           help='The model configuration ID to store the best genome')
        elif k == 'a':
            p.add_argument('-a', '--analysis', action="store_true", help='Whether to exec analysis')
            p.add_argument('-show', '--show', action="store_true", help='Whether to show the analysis plots')
        elif k == 'offline':
            p.add_argument('-offline', '--offline', action="store_true",
                           help='Whether to exec a full LarvaworldSim environment')
        elif k == 'hide':
            p.add_argument('-hide', '--show_screen', action="store_false",
                           help='Whether to render the screen visualization')

    return MP, p

