from typing import List
from argparse import ArgumentParser
import param
from larvaworld.lib import reg, aux, sim

__all__ = [
    'SingleParserArgument',
    'ParserArgumentDict',
    'SimModeParser',
    'update_larva_groups',
]

__displayname__ = 'CLI argument parsing classes'

class SingleParserArgument:
    """
    Create a single parser argument.

    This class is used to populate a parser with arguments and get their values.

    Parameters
    ----------
    short : str
        The short argument name.
    key : str
        The argument key.
    **kwargs : dict
        Additional keyword arguments.

    Attributes
    ----------
    key : str
        The argument key.
    args : list of str
        A list containing the short and long argument names.
    kwargs : dict
        Additional keyword arguments.
    """

    def __init__(self, short, key, **kwargs):
        self.key = key
        self.args = [f'-{short}', f'--{key}']
        self.kwargs = kwargs

    def add(self, p):
        """
        Add the argument to a parser.

        Parameters
        ----------
        p : argparse.ArgumentParser
            The argument parser.

        Returns
        -------
        argparse.ArgumentParser
            The modified parser.
        """
        p.add_argument(*self.args, **self.kwargs)
        return p

    def get(self, input):
        """
        Get the value of the argument from parsed input.

        Parameters
        ----------
        input : argparse.Namespace
            The parsed input.

        Returns
        -------
        Any
            The value of the argument.
        """
        return getattr(input, self.key)

    @classmethod
    def from_dict(cls, name, **kwargs):
        """
        Create a SingleParserArgument from a dictionary.

        Parameters
        ----------
        name : str
            The argument name.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        SingleParserArgument
            A SingleParserArgument instance.
        """
        return cls(**parser_entry_from_dict(name, **kwargs))

    @classmethod
    def from_param(cls, k, p):
        """
        Create a SingleParserArgument from a parameter.

        Parameters
        ----------
        k : str
            The parameter key.
        p : param.Parameter
            The parameter instance.

        Returns
        -------
        SingleParserArgument
            A SingleParserArgument instance.
        """
        return cls(**parser_entry_from_param(k, p))


def parser_entry_from_param(k, p):
    """
    Create a dictionary entry for a parser argument from a parameter.

    Parameters
    ----------
    k : str
        The parameter key.
    p : param.Parameter
        The parameter instance.

    Returns
    -------
    dict
        A dictionary entry for the parser argument.
    """
    c = p.__class__
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


def parser_entry_from_dict(name, k=None, h='', dtype=float, v=None, vs=None, **kwargs):
    """
    Create a dictionary entry for a parser argument from a dictionary.

    Parameters
    ----------
    name : str
        The argument name.
    k : str, optional
        The argument key. If not provided, it defaults to the name.
    h : str, optional
        The help text.
    dtype : type, optional
        The data type.
    v : Any, optional
        The default value.
    vs : list of Any, optional
        List of choices.
    **kwargs : dict
        Additional keyword arguments.

    Returns
    -------
    dict
        A dictionary entry for the parser argument.
    """
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


class ParserArgumentDict:
    """
    Create a dictionary of parser arguments.

    This class can be instantiated either by a dictionary of param.Parameters or by a dictionary existing in the registry parameter Database.
    """

    def __init__(self, parsargs):
        """
        Initialize a ParserArgumentDict.

        Parameters
        ----------
        parsargs : dict
            A dictionary of parser arguments.
        """
        self.parsargs = parsargs

    @classmethod
    def from_param(cls, d0):
        """
        Create a ParserArgumentDict from a parameter dictionary.

        Parameters
        ----------
        d0 : dict
            A dictionary of parameters.

        Returns
        -------
        ParserArgumentDict
            A ParserArgumentDict instance.
        """
        return cls(parser_dict_from_param(d0))

    @classmethod
    def from_dict(cls, d0):
        """
        Create a ParserArgumentDict from a dictionary.

        Parameters
        ----------
        d0 : dict
            A dictionary of parser arguments.

        Returns
        -------
        ParserArgumentDict
            A ParserArgumentDict instance.
        """
        return cls(parser_dict_from_dict(d0))

    def add(self, parser=None):
        """
        Add parser arguments to an ArgumentParser.

        Parameters
        ----------
        parser : argparse.ArgumentParser, optional
            The ArgumentParser to add the arguments to.

        Returns
        -------
        argparse.ArgumentParser
            The modified ArgumentParser.
        """
        if parser is None:
            parser = ArgumentParser()
        for k, v in self.parsargs.items():
            parser = v.add(parser)
        return parser

    def get(self, input):
        """
        Get parser argument values from parsed input.

        Parameters
        ----------
        input : argparse.Namespace
            The parsed input.

        Returns
        -------
        dict
            A dictionary of argument values.
        """
        dic = aux.AttrDict({k: v.get(input) for k, v in self.parsargs.items()})
        return dic.unflatten()


def parser_dict_from_param(d0):
    """
    Create a dictionary of parser arguments from a parameter dictionary.

    Parameters
    ----------
    d0 : dict
        A dictionary of parameters.

    Returns
    -------
    dict
        A dictionary of parser arguments.
    """
    d = aux.AttrDict()
    for k, p in d0.param.objects().items():
        if p.__class__ not in param.Parameterized.__subclasses__():
            d[k] = SingleParserArgument.from_param(k, p)
        else:
            d[k] = parser_dict_from_param(p)
    return d.flatten()


def parser_dict_from_dict(d0):
    """
    Create a dictionary of parser arguments from a dictionary.

    Parameters
    ----------
    d0 : dict
        A dictionary of parser arguments.

    Returns
    -------
    dict
        A dictionary of parser arguments.
    """
    p = aux.AttrDict()
    for n, v in d0.items():
        if 'v' in v.keys() or 'k' in v.keys() or 'h' in v.keys():
            p[n] = SingleParserArgument.from_dict(n, **v)
        else:
            p[n] = parser_dict_from_dict(v)
    return p.flatten()


class SimModeParser:
    """
    Parser for simulation modes and arguments.
    """

    def __init__(self):
        """
        Initialize the SimModeParser.
        """
        self.dict = aux.AttrDict({
            'Batch': [],
            'Eval': ['Eval'],
            'Exp': ['visualization'],
            'Ga': ['ga_select_kws', 'ga_eval_kws', 'reference_dataset'],
            'Replay': ['Replay']
        })
        self.parser_keys = aux.unique_list(aux.flatten_list(list(self.dict.values())))
        self.parsers = self.init_parsers()
        self.cli_parser = self.build_cli_parser()
        self.mode = None
        self.run = None
        self.args = aux.AttrDict()
        self.parser_args = aux.AttrDict()
        self.run_kws = aux.AttrDict()

    def parse_args(self):
        """
        Parse command line arguments.
        """
        self.args = aux.AttrDict(vars(self.cli_parser.parse_args()))

    def init_parsers(self):
        """
        Initialize parsers for different simulation modes.
        """
        d = aux.AttrDict()
        d.sim_params = ParserArgumentDict.from_dict(reg.par.PI['sim_params'])
        for k in self.parser_keys:
            d[k] = ParserArgumentDict.from_dict(reg.par.PI[k])
        return d

    def populate_mode_subparser(self, sp, m):
        """
        Populate a subparser with arguments for a specific simulation mode.

        :param sp: The subparser.
        :param m: The simulation mode.
        :return: The modified subparser.
        """
        if m not in ['Replay', 'Eval']:
            sp = self.parsers.sim_params.add(sp)
        for k in self.dict[m]:
            sp = self.parsers[k].add(sp)
        return sp

    def init_mode_subparser(self, sp, m):
        """
        Initialize a subparser with common arguments for a specific simulation mode.

        :param sp: The subparser.
        :param m: The simulation mode.
        :return: The modified subparser.
        """
        sp.add_argument('-id', '--id', type=str, help='The simulation ID. If not provided a default is generated')
        if m in ['Exp', 'Batch', 'Ga']:
            sp.add_argument('experiment', choices=reg.conf[m].confIDs, help='The experiment mode')
        if m in ['Exp', 'Batch']:
            sp.add_argument('-N', '--Nagents', type=int, help='The number of simulated larvae in each larva group')
            sp.add_argument('-mIDs', '--group_model_ids', type=str, nargs='+',
                            help='The larva models to use for creating the simulation larva groups')
            sp.add_argument('-dIDs', '--group_disp_ids', type=str, nargs='+',
                            help='The displayed IDs of the simulation larva groups')
        return sp

    def build_cli_parser(self):
        """
        Build the command line argument parser.
        """
        p = ArgumentParser()
        subparsers = p.add_subparsers(dest='sim_mode', help='The simulation mode to launch')
        for m in reg.SIMTYPES:
            sp = subparsers.add_parser(m)
            sp = self.init_mode_subparser(sp, m)
            sp = self.populate_mode_subparser(sp, m)
        return p

    def eval_parser(self, p_key):
        """
        Evaluate a parser argument.

        :param p_key: The argument key.
        :return: The parsed value of the argument.
        """
        return self.parsers[p_key].get(self.args)

    def eval_parsers(self):
        """
        Evaluate all parser arguments.

        :return: A dictionary of parsed values.
        """
        d = aux.AttrDict()
        if self.mode not in ['Replay', 'Eval']:
            d.sim_params = self.eval_parser('sim_params')
        for k in self.dict[self.mode]:
            d[k] = self.eval_parser(k)
        return d

    def configure(self, show_args=False):
        """
        Configure the simulation run based on parsed arguments.

        :param show_args: Whether to show parsed arguments.
        :return: The configured simulation run.
        """
        a = self.args
        self.mode = m = a.sim_mode
        self.parser_args = sp = self.eval_parsers()
        kw = aux.AttrDict({'id': a.id})

        if m not in ['Replay', 'Eval']:
            kw.update(**sp.sim_params)
            kw.experiment = a.experiment
        if m == 'Batch':
            kw.mode = 'batch'
            kw.run_externally = False
            kw.conf = reg.conf.Batch.getID(a.experiment)
            kw.conf.exp = reg.conf.Exp.expand(kw.conf.exp)
            kw.conf.exp.experiment = kw.conf.exp
            kw.conf.exp.larva_groups = update_larva_groups(
                kw.conf.exp.larva_groups, N=a.Nagents, mIDs=a.group_model_ids, dIDs=a.group_disp_ids)

            if kw.duration is None:
                kw.duration = kw.conf.exp.sim_params.duration
            self.run = sim.Exec(**kw)
        elif m == 'Exp':
            kw.parameters = reg.conf.Exp.expand(kw.experiment)
            kw.parameters.experiment = kw.experiment
            kw.parameters.larva_groups = update_larva_groups(
                kw.parameters.larva_groups, N=a.Nagents, mIDs=a.group_model_ids, dIDs=a.group_disp_ids)
            if kw.duration is None:
                kw.duration = kw.parameters.sim_params.duration

            kw.screen_kws = {'vis_kwargs': sp.visualization}
            self.run = sim.ExpRun(**kw)
        elif m == 'Ga':
            p = reg.conf.Ga.expand(kw.experiment)

            for k in ['base_model', 'bestConfID', 'space_mkeys']:
                if sp.ga_select_kws[k] is None:
                    sp.ga_select_kws.pop(k)
            p.ga_select_kws.update(**sp.ga_select_kws)

            if sp.reference_dataset.refID is not None:
                p.refID = sp.reference_dataset.refID

            if kw.duration is None:
                kw.duration = p.sim_params.duration
            kw.parameters = p
            self.run = sim.GAlauncher(**kw)
        elif m == 'Eval':
            self.run = sim.EvalRun(**sp.Eval)
        elif m == 'Replay':
            kw.parameters = sp.Replay
            self.run = sim.ReplayRun(**kw)
        self.run_kws = kw
        if show_args:
            self.show_args()
        return self.run

    def show_args(self, nested=True, flat_nested=False, input=False, default=False, run_args=True):
        """
        Show parsed arguments.

        :param nested: Whether to show arguments as a nested dictionary.
        :param flat_nested: Whether to show arguments as a flattened nested dictionary.
        :param input: Whether to show input arguments.
        :param default: Whether to show default arguments.
        :param run_args: Whether to show run arguments.
        """
        print(f'Simulation mode : {self.mode}')
        if nested:
            print(f'Simulation args as nested dictionary: ')
            self.run_kws.print(flat=False)
        if flat_nested:
            print(f'Simulation args as flattened nested dictionary: ')
            self.run_kws.print(flat=True)
        if input:
            print(f'Input args : ')
            self.args.print(flat=True)
        if run_args:
            if hasattr(self.run, 'configuration_text'):
                print(self.run.configuration_text)

    def launch(self):
        """
        Launch the simulation run.
        """
        m = self.mode
        r = self.run
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



def update_larva_groups(lgs, N=None, mIDs=None, dIDs=None, sample=None):
    """
    Modifies the experiment's configuration larvagroups.

    Args:
        lgs (dict): The existing larvagroups in the experiment configuration.
        N (int):: Overwrite the number of agents per larva group.
        mIDs (list): Overwrite the larva models used in the experiment. If not None, a larva group per model ID will be simulated.
        dIDs (list): The displayed IDs of the groups. If None, the model IDs (mIDs) are used.
        sample: The reference dataset.

    Returns:
        The experiment's configuration larvagroups.
    """
    if mIDs is not None:
        if dIDs is None:
            dIDs = mIDs
        Nm = len(mIDs)
        gConfs = list(lgs.values())
        if len(lgs) != Nm:
            gConfs = [gConfs[0]] * Nm
            for gConf, col in zip(gConfs, aux.N_colors(Nm)):
                gConf.default_color = col
        lgs = aux.AttrDict({dID: {} for dID in dIDs})
        for dID, mID, gConf in zip(dIDs, mIDs, gConfs):
            lgs[dID] = gConf
            lgs[dID].model = reg.conf.Model.getID(mID)

    if N is not None:
        for gID, gConf in lgs.items():
            gConf.distribution.N = N

    if sample is not None:
        for gID, gConf in lgs.items():
            gConf.sample = sample

    return lgs



#
#
# class SingleParserArgument:
#     """
#     Create a single parser argument
#     This is a class used to populate a parser with arguments and get their values.
#     """
#
#     def __init__(self, short, key, **kwargs):
#         self.key = key
#         self.args = [f'-{short}', f'--{key}']
#         self.kwargs = kwargs
#
#     def add(self, p):
#         p.add_argument(*self.args, **self.kwargs)
#         return p
#
#     def get(self, input):
#         return getattr(input, self.key)
#
#     @ classmethod
#     def from_dict(cls, name, **kwargs):
#         return cls(**parser_entry_from_dict(name, **kwargs))
#
#     @classmethod
#     def from_param(cls, k, p):
#         return cls(**parser_entry_from_param(k, p))
#
# def parser_entry_from_param(k, p):
#     c = p.__class__
#     v = p.default
#     d = aux.AttrDict({
#         'key': k,
#         'short': k,
#         'help': p.doc,
#     })
#     if v is not None:
#         d.default = v
#     if c == param.Boolean:
#         d.action = 'store_true' if not v else 'store_false'
#     elif c == param.String:
#         d.type = str
#     elif c in param.Integer.__subclasses__():
#         d.type = int
#     elif c in param.Number.__subclasses__():
#         d.type = float
#     elif c in param.Tuple.__subclasses__():
#         d.type = tuple
#
#     if hasattr(p, 'objects'):
#         d.choices = p.objects
#         if c in param.List.__subclasses__():
#             d.nargs = '+'
#         if hasattr(p, 'item_type'):
#             d.type = p.item_type
#     return d
#
# def parser_entry_from_dict(name, k=None, h='', dtype=float, v=None, vs=None, **kwargs):
#     if k is None:
#         k = name
#     d = {
#         'key': name,
#         'short': k,
#         'help': h,
#     }
#     if dtype == bool:
#         d['action'] = 'store_true' if not v else 'store_false'
#     elif dtype == List[str]:
#         d['type'] = str
#         d['nargs'] = '+'
#         if vs is not None:
#             d['choices'] = vs
#     elif dtype == List[int]:
#         d['type'] = int
#         d['nargs'] = '+'
#         if vs is not None:
#             d['choices'] = vs
#     else:
#         d['type'] = dtype
#         if vs is not None:
#             d['choices'] = vs
#         if v is not None:
#             d['default'] = v
#             d['nargs'] = '?'
#     return d
#
#
# class ParserArgumentDict:
#     """
#         Create a dictionary of parser arguments
#         This can be instantiated either by a dictionary of param.Parameters or by a dictionary existing in the registry parameter Database
#     """
#
#
#     def __init__(self, parsargs):
#         self.parsargs = parsargs
#
#     @classmethod
#     def from_param(cls, d0):
#         return cls(parser_dict_from_param(d0))
#
#     @classmethod
#     def from_dict(cls, d0):
#         return cls(parser_dict_from_dict(d0))
#
#     def add(self, parser=None):
#         if parser is None:
#             parser = ArgumentParser()
#         for k, v in self.parsargs.items():
#             parser = v.add(parser)
#         return parser
#
#     def get(self, input):
#         dic = aux.AttrDict({k: v.get(input) for k, v in self.parsargs.items()})
#         return dic.unflatten()
#
# def parser_dict_from_param(d0):
#     d = aux.AttrDict()
#     for k, p in d0.param.objects().items():
#         if p.__class__ not in param.Parameterized.__subclasses__():
#             d[k] = SingleParserArgument.from_param(k, p)
#         else:
#             d[k] = parser_dict_from_param(p)
#     return d.flatten()
#
# def parser_dict_from_dict(d0):
#     p = aux.AttrDict()
#     for n, v in d0.items():
#         if 'v' in v.keys() or 'k' in v.keys() or 'h' in v.keys():
#             p[n] = SingleParserArgument.from_dict(n, **v)
#         else:
#             p[n] = parser_dict_from_dict(v)
#     return p.flatten()
#
#
# class SimModeParser :
#     def __init__(self):
#         self.dict = aux.AttrDict({
#         'Batch': [],
#         'Eval': ['Eval'],
#         'Exp': ['visualization'],
#         'Ga': ['ga_select_kws', 'ga_eval_kws', 'reference_dataset'],
#         'Replay': ['Replay']
#     })
#         self.parser_keys=aux.unique_list(aux.flatten_list(list(self.dict.values())))
#         self.parsers =self.init_parsers()
#         self.cli_parser = self.build_cli_parser()
#         self.mode=None
#         self.run=None
#         self.args = aux.AttrDict()
#         self.parser_args = aux.AttrDict()
#         self.run_kws = aux.AttrDict()
#
#
#
#     def parse_args(self):
#
#         self.args = aux.AttrDict(vars(self.cli_parser.parse_args()))
#
#     def init_parsers(self):
#         d = aux.AttrDict()
#         d.sim_params = ParserArgumentDict.from_dict(reg.par.PI['sim_params'])
#         # d.sim_params = Parser('sim_params')
#         for k in self.parser_keys:
#             d[k] = ParserArgumentDict.from_dict(reg.par.PI[k])
#             # d[p_key] = Parser(p_key)
#         return d
#
#     def populate_mode_subparser(self, sp, m):
#         if m not in ['Replay', 'Eval']:
#             sp = self.parsers.sim_params.add(sp)
#         for k in self.dict[m]:
#             sp = self.parsers[k].add(sp)
#         return sp
#
#     def init_mode_subparser(self, sp, m):
#         sp.add_argument('-id', '--id', type=str, help='The simulation ID. If not provided a default is generated')
#         if m in ['Exp', 'Batch', 'Ga']:
#             sp.add_argument('experiment', choices=reg.conf[m].confIDs, help='The experiment mode')
#             # p.add_argument('experiment', choices=reg.stored.confIDs(sim_mode), help='The experiment mode')
#         # elif k == 't':
#         #     p.add_argument('-t', '--duration', type=float, help='The duration of the simulation in minutes')
#         # elif k == 'Box2D':
#         #     p.add_argument('-Box2D', '--Box2D', action="store_true", help='Whether to use the Box2D physics engine or not')
#         if m in ['Exp', 'Batch']:
#             sp.add_argument('-N', '--Nagents', type=int, help='The number of simulated larvae in each larva group')
#             sp.add_argument('-mIDs', '--group_model_ids', type=str, nargs='+', help='The larva models to use for creating the simulation larva groups')
#             sp.add_argument('-dIDs', '--group_disp_ids', type=str, nargs='+', help='The displayed IDs of the simulation larva groups')
#         # elif k == 'mID0':
#         #     p.add_argument('-mID0', '--base_model', choices=reg.storedConf('Model'),
#         #                    help='The model configuration to optimize')
#         # elif k == 'mID1':
#         #     p.add_argument('-mID1', '--bestConfID', type=str,
#         #                    help='The model configuration ID to store the best genome')
#         if m in ['Exp']:
#             sp.add_argument('-a', '--analysis', action="store_true", help='Whether to exec analysis')
#             sp.add_argument('-show', '--show', action="store_true", help='Whether to show the analysis plots')
#         # elif k == 'offline':
#         #     p.add_argument('-offline', '--offline', action="store_true",
#         #                    help='Whether to exec a full LarvaworldSim environment')
#         # elif k == 'hide':
#         #     p.add_argument('-hide', '--show_display', action="store_false",
#
#         #                    help='Whether to render the screen visualization')
#         return sp
#
#     def build_cli_parser(self):
#         p = ArgumentParser()
#         subparsers = p.add_subparsers(dest='sim_mode', help='The simulation mode to launch')
#         for m in reg.SIMTYPES:
#             sp = subparsers.add_parser(m)
#             sp = self.init_mode_subparser(sp, m)
#             sp = self.populate_mode_subparser(sp, m)
#         return p
#
#     def eval_parser(self,p_key):
#         return self.parsers[p_key].get(self.args)
#
#     def eval_parsers(self):
#         d = aux.AttrDict()
#         if self.mode not in ['Replay', 'Eval']:
#             d.sim_params = self.eval_parser('sim_params')
#         for k in self.dict[self.mode]:
#             d[k] = self.eval_parser(k)
#         return d
#
#     def configure(self, show_args=False):
#         a=self.args
#         self.mode= m =a.sim_mode
#         self.parser_args=sp=self.eval_parsers()
#         kw = aux.AttrDict({'id': a.id})
#
#         if m not in ['Replay', 'Eval']:
#             kw.update(**sp.sim_params)
#             kw.experiment = a.experiment
#         if m == 'Batch':
#             kw.mode='batch'
#             kw.run_externally=False
#             kw.conf = reg.conf.Batch.getID(a.experiment)
#             kw.conf.exp = reg.conf.Exp.expand(kw.conf.exp)
#             kw.conf.exp.experiment = kw.conf.exp
#             kw.conf.exp.larva_groups = update_larva_groups(kw.conf.exp.larva_groups, N=a.Nagents, mIDs=a.group_model_ids, dIDs=a.group_disp_ids)
#
#
#             if kw.duration is None:
#                 kw.duration = kw.conf.exp.sim_params.duration
#             self.run = sim.Exec(**kw)
#         elif m == 'Exp':
#             kw.parameters = reg.conf.Exp.expand(kw.experiment)
#             kw.parameters.experiment = kw.experiment
#             kw.parameters.larva_groups = update_larva_groups(kw.parameters.larva_groups, N=a.Nagents, mIDs=a.group_model_ids, dIDs=a.group_disp_ids)
#             if kw.duration is None:
#                 kw.duration = kw.parameters.sim_params.duration
#
#             kw.screen_kws = {'vis_kwargs': sp.visualization}
#             self.run = sim.ExpRun(**kw)
#         elif m == 'Ga':
#
#
#             p = reg.conf.Ga.expand(kw.experiment)
#
#             # p.ga_space_kws.init_mode = sp.ga_space_kws.init_mode
#             for k in ['base_model','bestConfID', 'space_mkeys']:
#                 if sp.ga_select_kws[k] is None:
#                     sp.ga_select_kws.pop(k)
#             p.ga_select_kws.update(**sp.ga_select_kws)
#
#             if sp.reference_dataset.refID is not None:
#                 # print(sp.reference_dataset.refID,p.refID)
#                 # raise
#                 p.refID = sp.reference_dataset.refID
#
#
#             if kw.duration is None:
#                 kw.duration = p.sim_params.duration
#             kw.parameters = p
#             self.run = sim.GAlauncher(**kw)
#         elif m == 'Eval':
#             # kw.parameters = sp.Eval
#             self.run = sim.EvalRun(**sp.Eval)
#         elif m == 'Replay':
#             kw.parameters = sp.Replay
#             self.run = sim.ReplayRun(**kw)
#         self.run_kws=kw
#         if show_args:
#             self.show_args()
#         return self.run
#
#     def show_args(self, nested=True, flat_nested=False,input=False, default=False, run_args=True):
#         print(f'Simulation mode : {self.mode}')
#         if nested :
#             print(f'Simulation args as nested dictionary: ')
#             self.run_kws.print(flat=False)
#         if flat_nested :
#             print(f'Simulation args as flattened nested dictionary: ')
#             self.run_kws.print(flat=True)
#         if input:
#             print(f'Input args : ')
#             self.args.print(flat=True)
#         # if default:
#         #     print(f'Default args : ')
#         #     self.default_args.print(flat=True)
#         if run_args:
#             if hasattr(self.run, 'configuration_text'):
#                 print(self.run.configuration_text)
#
#
#     def launch(self):
#         m=self.mode
#         r=self.run
#         if m == 'Batch':
#             r.run()
#         elif m == 'Exp':
#             _ = r.simulate()
#             if self.args.analysis:
#                 r.analyze(show=self.args.show)
#         elif m == 'Ga':
#             _ = r.simulate()
#         elif m == 'Eval':
#             r.simulate()
#             r.plot_results()
#             r.plot_models()
#         elif m == 'Replay':
#             r.run()
#
#
#
# def update_larva_groups(lgs, N=None, mIDs=None, dIDs=None,sample=None):
#     '''
#     Modifies the experiment's configuration larvagroups
#     Args:
#         sample : The reference dataset
#         lgs: The existing larvagroups in the experiment configuration
#         N: Overwrite the number of agents per larva group
#         mIDs: Overwrite the larva models used in the experiment.If not None a larva group per model ID will be simulated.
#         dIDs: The displayed ids of the groups. If None the model IDs (mIDs) are used
#
#     Returns:
#         The experiment's configuration larvagroups
#     '''
#
#     if mIDs is not None:
#         if dIDs is None:
#             dIDs=mIDs
#         Nm = len(mIDs)
#         gConfs = list(lgs.values())
#         if len(lgs) != Nm:
#             gConfs = [gConfs[0]] * Nm
#             for gConf, col in zip(gConfs, aux.N_colors(Nm)):
#                 gConf.default_color = col
#         lgs = aux.AttrDict({dID: {} for dID in dIDs})
#         for dID, mID, gConf in zip(dIDs, mIDs, gConfs):
#             lgs[dID] = gConf
#             lgs[dID].model = reg.conf.Model.getID(mID)
#
#     if N is not None:
#         for gID, gConf in lgs.items():
#             gConf.distribution.N = N
#
#     if sample is not None:
#         for gID, gConf in lgs.items():
#             gConf.sample = sample
#
#     return lgs
