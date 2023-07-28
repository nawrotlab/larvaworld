from typing import List
from argparse import ArgumentParser

import param

from larvaworld.lib import reg, aux, sim




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

    @ classmethod
    def from_dict(cls, name, **kwargs):
        entry = build_ParsArg(name, **kwargs)
        return cls(**entry)

    @classmethod
    def from_param(cls, k, p):
        entry = build_ParamArg(k, p)
        return cls(**entry)


def build_ParamArg(k, p):
    c = p.__class__
    # dtype=aux.param_dtype(c)
    v = p.default
    d = aux.AttrDict({
        'key': k,
        'short': k,
        'help': p.doc,
    })
    if v is not None:
        d.default = v
    if c == param.Boolean:
        d.action = 'store_true' if not v else 'store_false'
    elif c == param.String:
        d.type = str
    elif c in param.Integer.__subclasses__():
        d.type = int
    elif c in param.Number.__subclasses__():
        d.type = float
    elif c in param.Tuple.__subclasses__():
        d.type = tuple

    if hasattr(p, 'objects'):
        d.choices = p.objects
        if c in param.List.__subclasses__():
            d.nargs = '+'
        if hasattr(p, 'item_type'):
            d.type = p.item_type
    return d



def parser_Paramdict(d0):
    d = aux.AttrDict()
    for k, p in d0.param.objects().items():

        if p.__class__ not in param.Parameterized.__subclasses__():
            d[k] = ParsArg.from_param(k, p)
        else:
            d[k] = parser_Paramdict(p)
    return d.flatten()






class ParsDict:
    def __init__(self, parsargs):
        self.parsargs = parsargs

    @classmethod
    def from_param(cls, d0):
        parsargs = parser_Paramdict(d0)
        return cls(parsargs)

    @classmethod
    def from_dict(cls, d0):
        parsargs = parser_dict(d0)
        return cls(parsargs)

    def add(self, parser=None):
        if parser is None:
            parser = ArgumentParser()
        for k, v in self.parsargs.items():
            parser = v.add(parser)
        return parser

    def get(self, input):
        dic = aux.AttrDict({k: v.get(input) for k, v in self.parsargs.items()})
        return dic.unflatten()

class Parser:
    """
    Create an argument parser for a group of arguments (normally from a dict).
    """

    def __init__(self, name):
        self.name = name
        self.parsargs = parser_dict(reg.par.PI[name])


    def add(self, parser=None):
        if parser is None:
            parser = ArgumentParser()
        for k, v in self.parsargs.items():
            parser = v.add(parser)
        return parser

    def get(self, input):
        dic = aux.AttrDict({k: v.get(input) for k, v in self.parsargs.items()})
        return dic.unflatten()

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
    return d

def parser_dict(d0):
    p = aux.AttrDict()
    for n, v in d0.items():
        if 'v' in v.keys() or 'k' in v.keys() or 'h' in v.keys():
            p[n] = ParsArg.from_dict(n, **v)
        else:
            p[n] = parser_dict(v)
    return p.flatten()





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


class SimModeParser :
    def __init__(self):
        self.dict = aux.AttrDict({
        'Batch': [],
        'Eval': ['Eval'],
        'Exp': ['visualization'],
        'Ga': ['ga_select_kws', 'ga_eval_kws', 'reference_dataset'],
        'Replay': ['Replay']
    })
        self.parser_keys=aux.unique_list(aux.flatten_list(list(self.dict.values())))
        self.parsers =self.init_parsers()
        self.cli_parser = self.build_cli_parser()
        self.mode=None
        self.run=None
        self.args = aux.AttrDict()
        # self.default_args = aux.AttrDict()
        self.parser_args = aux.AttrDict()
        self.run_kws = aux.AttrDict()



    def parse_args(self):

        self.args = aux.AttrDict(vars(self.cli_parser.parse_args()))

    def init_parsers(self):
        parsers = aux.AttrDict()
        parsers.sim_params = Parser('sim_params')
        for p_key in self.parser_keys:
            parsers[p_key] = Parser(p_key)
        return parsers

    def populate_mode_subparser(self, sp, m):
        if m not in ['Replay', 'Eval']:
            sp = self.parsers.sim_params.add(sp)
        for k in self.dict[m]:
            sp = self.parsers[k].add(sp)
        return sp

    def init_mode_subparser(self, sp, m):
        sp.add_argument('-id', '--id', type=str, help='The simulation ID. If not provided a default is generated')
        if m in ['Exp', 'Batch', 'Ga']:
            sp.add_argument('experiment', choices=list(reg.CONFTREE[m].keys()), help='The experiment mode')
            # p.add_argument('experiment', choices=reg.stored.confIDs(sim_mode), help='The experiment mode')
        # elif k == 't':
        #     p.add_argument('-t', '--duration', type=float, help='The duration of the simulation in minutes')
        # elif k == 'Box2D':
        #     p.add_argument('-Box2D', '--Box2D', action="store_true", help='Whether to use the Box2D physics engine or not')
        if m in ['Exp', 'Batch']:
            sp.add_argument('-N', '--Nagents', type=int, help='The number of simulated larvae in each larva group')
            sp.add_argument('-ms', '--models', type=str, nargs='+',
                            help='The larva models to use for creating the simulation larva groups')
        # elif k == 'mID0':
        #     p.add_argument('-mID0', '--base_model', choices=reg.storedConf('Model'),
        #                    help='The model configuration to optimize')
        # elif k == 'mID1':
        #     p.add_argument('-mID1', '--bestConfID', type=str,
        #                    help='The model configuration ID to store the best genome')
        if m in ['Exp']:
            sp.add_argument('-a', '--analysis', action="store_true", help='Whether to exec analysis')
            sp.add_argument('-show', '--show', action="store_true", help='Whether to show the analysis plots')
        # elif k == 'offline':
        #     p.add_argument('-offline', '--offline', action="store_true",
        #                    help='Whether to exec a full LarvaworldSim environment')
        # elif k == 'hide':
        #     p.add_argument('-hide', '--show_display', action="store_false",

        #                    help='Whether to render the screen visualization')
        return sp

    def build_cli_parser(self):
        p = ArgumentParser()


        subparsers = p.add_subparsers(dest='sim_mode', help='The simulation mode to launch')
        for m in reg.SIMTYPES:
            sp = subparsers.add_parser(m)
            sp = self.init_mode_subparser(sp, m)
            sp = self.populate_mode_subparser(sp, m)
        return p

    def eval_parser(self,p_key):
        return self.parsers[p_key].get(self.args)

    def eval_parsers(self):
        parser_args = aux.AttrDict()
        if self.mode not in ['Replay', 'Eval']:
            parser_args.sim_params = self.eval_parser('sim_params')
        for k in self.dict[self.mode]:
            parser_args[k] = self.eval_parser(k)
        return parser_args

    def configure(self, show_args=False):
        a=self.args

        # self.default_args =aux.AttrDict(vars(self.cli_parser.parse_args([])))
        self.mode= m =a.sim_mode
        self.parser_args=sp=self.eval_parsers()
        kw = aux.AttrDict({'id': a.id})

        if m not in ['Replay', 'Eval']:
            kw.update(**sp.sim_params)
            kw.experiment = a.experiment
        if m == 'Batch':
            kw.mode='batch'
            kw.run_externally=False
            kw.conf = reg.conf.Batch.getID(a.experiment)
            kw.conf.exp = update_exp_conf(kw.conf.exp, N=a.Nagents, mIDs=a.models)
            if kw.duration is None:
                kw.duration = kw.conf.exp.sim_params.duration
            self.run = sim.Exec(**kw)
        elif m == 'Exp':
            kw.parameters = update_exp_conf(kw.experiment, N=a.Nagents, mIDs=a.models)
            if kw.duration is None:
                kw.duration = kw.parameters.sim_params.duration

            kw.screen_kws = {'vis_kwargs': sp.visualization}
            self.run = sim.ExpRun(**kw)
        elif m == 'Ga':


            p = reg.conf.Ga.expand(kw.experiment)

            # p.ga_space_kws.init_mode = sp.ga_space_kws.init_mode
            for k in ['base_model','bestConfID', 'space_mkeys']:
                if sp.ga_select_kws[k] is None:
                    sp.ga_select_kws.pop(k)
            p.ga_select_kws.update(**sp.ga_select_kws)
             
            if sp.reference_dataset.refID is not None:
                # print(sp.reference_dataset.refID,p.refID)
                # raise
                p.refID = sp.reference_dataset.refID


            if kw.duration is None:
                kw.duration = p.sim_params.duration
            kw.parameters = p
            self.run = sim.GAlauncher(**kw)
        elif m == 'Eval':
            # kw.parameters = sp.Eval
            self.run = sim.EvalRun(**sp.Eval)
        elif m == 'Replay':
            kw.parameters = sp.Replay
            self.run = sim.ReplayRun(**kw)
        self.run_kws=kw
        if show_args:
            self.show_args()
        return self.run

    def show_args(self, nested=True, flat_nested=False,input=False, default=False, run_args=True):
        print(f'Simulation mode : {self.mode}')
        if nested :
            print(f'Simulation args as nested dictionary: ')
            self.run_kws.print(flat=False)
        if flat_nested :
            print(f'Simulation args as flattened nested dictionary: ')
            self.run_kws.print(flat=True)
        if input:
            print(f'Input args : ')
            self.args.print(flat=True)
        # if default:
        #     print(f'Default args : ')
        #     self.default_args.print(flat=True)
        if run_args:
            if hasattr(self.run, 'configuration_text'):
                print(self.run.configuration_text)


    def launch(self):
        m=self.mode
        r=self.run
        if m == 'Batch':
            r.run()
        elif m == 'Exp':
            _ = r.simulate()
            if self.args.analysis:
                r.analyze(show=self.args.show)
        elif m == 'Ga':
            _ = r.simulate()
        elif m == 'Eval':
            r.simulate()
            r.plot_results()
            r.plot_models()
        elif m == 'Replay':
            r.run()



def update_exp_conf(type, N=None, mIDs=None):
    '''
    Loads the configuration of an experiment based on its ID.
    Modifies the included larvagroups
    Args:
        type: The experiment type
        N: Overwrite the number of agents per larva group
        mIDs: Overwrite the larva models used in the experiment.If not None a larva group per model ID will be simulated.

    Returns:
        The experiment's configuration
    '''
    conf = reg.conf.Exp.expand(type)
    conf.experiment = type

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
            conf.larva_groups[mID].model = reg.conf.Model.getID(mID)

    if N is not None:
        for gID, gConf in conf.larva_groups.items():
            gConf.distribution.N = N

    return conf
