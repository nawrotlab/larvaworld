"""
Classes that build the larvaworld CLI argument parser.

Parser arguments are derived automatically from the canonical
:class:`param.Parameterized` configuration classes (or from registry parameter
dictionaries), so that the CLI surface stays in sync with the underlying
parameter definitions instead of duplicating them.
"""

from __future__ import annotations
from argparse import ArgumentParser, Namespace
from typing import Any, List
import param

from .. import SIMTYPES
from ..lib.util import AttrDict
from ..lib.screen import ScreenOps
from ..lib import reg, sim
from ..lib.param import RuntimeOps, SimOps

__all__: list[str] = ["SingleParserArgument", "ParserArgumentDict", "SimModeParser"]

__displayname__ = "CLI argument parsing classes"


class SingleParserArgument:
    """
    Create a single parser argument.

    This class is used to populate a parser with arguments and get their values.

    Args:
        short (str): The short argument name.
        key (str): The argument key.
        **kwargs (dict): Additional keyword arguments.

    Attributes:
        key (str): The argument key.
        args (list of str): A list containing the short and long argument names.
        kwargs (dict): Additional keyword arguments.
    """

    def __init__(self, short: str, key: str, **kwargs: Any) -> None:
        """Initialize the argument.

        Args:
            short: The short argument name.
            key: The argument key.
            **kwargs: Additional keyword arguments forwarded to
                :meth:`argparse.ArgumentParser.add_argument`.
        """
        self.key = key
        self.args = [f"-{short}", f"--{key}"]
        self.kwargs = kwargs

    def add(self, p: ArgumentParser) -> ArgumentParser:
        """
        Add the argument to a parser.

        Args:
            p (argparse.ArgumentParser): The argument parser.

        Returns:
            argparse.ArgumentParser: The modified parser.
        """
        p.add_argument(*self.args, **self.kwargs)
        return p

    def get(self, input: Namespace) -> Any:
        """
        Get the value of the argument from parsed input.

        Args:
            input (argparse.Namespace): The parsed input.

        Returns:
            Any: The value of the argument.
        """
        return getattr(input, self.key)

    @classmethod
    def from_dict(cls, name: str, **kwargs: Any) -> SingleParserArgument:
        """
        Create a SingleParserArgument from a dictionary.

        Args:
            name (str): The argument name.
            **kwargs (dict): Additional keyword arguments.

        Returns:
            SingleParserArgument: A SingleParserArgument instance.
        """
        return cls(**parser_entry_from_dict(name, **kwargs))

    @classmethod
    def from_param(cls, k: str, p: param.Parameter) -> SingleParserArgument:
        """
        Create a SingleParserArgument from a parameter.

        Args:
            k (str): The parameter key.
            p (param.Parameter): The parameter instance.

        Returns:
            SingleParserArgument: A SingleParserArgument instance.
        """
        return cls(**parser_entry_from_param(k, p))


def parser_entry_from_param(k: str, p: param.Parameter) -> AttrDict:
    """
    Create a dictionary entry for a parser argument from a parameter.

    Args:
        k (str): The parameter key.
        p (param.Parameter): The parameter instance.

    Returns:
        dict A dictionary entry for the parser argument.
    """
    c = p.__class__
    v = p.default
    d = AttrDict(
        {
            "key": k,
            "short": k,
            "help": p.doc,
            "default": p.default,
        }
    )
    # if v is not None:
    #     d.default = v
    if c == param.Boolean:
        d.action = "store_true" if not v else "store_false"
    elif c == param.String:
        d.type = str
    elif c in [param.Integer] + param.Integer.__subclasses__():
        d.type = int
    elif c in [param.Number] + param.Number.__subclasses__():
        d.type = float
    elif c in [param.Tuple] + param.Tuple.__subclasses__():
        d.type = tuple

    if hasattr(p, "objects"):
        d.choices = p.objects
        if (
            c
            in [param.List, param.ListSelector]
            + param.List.__subclasses__()
            + param.ListSelector.__subclasses__()
        ):
            d.nargs = "+"
        if hasattr(p, "item_type"):
            d.type = p.item_type
    return d


def parser_entry_from_dict(
    name: str,
    k: str | None = None,
    h: str = "",
    dtype: type = float,
    v: Any | None = None,
    vs: List[Any] | None = None,
    **kwargs: Any,
) -> AttrDict:
    """
    Create a dictionary entry for a parser argument from a dictionary.

    Args:
        name (str): The argument name.
        k (str): The argument key. If not provided, it defaults to the name.
        h (str): The help text.
        dtype (type): The data type.
        v (Any): The default value.
        vs (list of Any): List of choices.
        **kwargs (dict): Additional keyword arguments.

    Returns:
        dict A dictionary entry for the parser argument.
    """
    if k is None:
        k = name
    d = AttrDict(
        {
            "key": name,
            "short": k,
            "help": h,
        }
    )
    if dtype == bool:
        d.action = "store_true" if not v else "store_false"
    elif dtype == List[str]:
        d.type = str
        d.nargs = "+"
        if vs is not None:
            d.choices = vs
    elif dtype == List[int]:
        d.type = int
        d.nargs = "+"
        if vs is not None:
            d.choices = vs
    else:
        d.type = dtype
        if vs is not None:
            d.choices = vs
        if v is not None:
            d.default = v
            d.nargs = "?"
    return d


class ParserArgumentDict:
    """
    Create a dictionary of parser arguments.

    This class can be instantiated either by a dictionary of param.Parameters or by a dictionary existing in the registry parameter Database.
    """

    def __init__(self, parsargs: AttrDict) -> None:
        """
        Initialize a ParserArgumentDict.

        Args:
            parsargs (dict): A dictionary of parser arguments.
        """
        self.parsargs = parsargs

    @classmethod
    def from_param(cls, d0: param.Parameterized) -> ParserArgumentDict:
        """
        Create a ParserArgumentDict from a parameter dictionary.

        Args:
            d0 (dict): A dictionary of parameters.

        Returns:
            ParserArgumentDict: A ParserArgumentDict instance.
        """
        return cls(parser_dict_from_param(d0))

    @classmethod
    def from_dict(cls, d0: dict) -> ParserArgumentDict:
        """
        Create a ParserArgumentDict from a dictionary.

        Args:
            d0 (dict): A dictionary of parser arguments.

        Returns:
            ParserArgumentDict: A ParserArgumentDict instance.
        """
        return cls(parser_dict_from_dict(d0))

    def add(self, parser: ArgumentParser | None = None) -> ArgumentParser:
        """
        Add parser arguments to an ArgumentParser.

        Args:
            parser (argparse.ArgumentParser): The ArgumentParser to add the arguments to.

        Returns:
            argparse.ArgumentParser: The modified ArgumentParser.
        """
        if parser is None:
            parser = ArgumentParser()
        for k, v in self.parsargs.items():
            parser = v.add(parser)
        return parser

    def get(self, input: Namespace) -> AttrDict:
        """
        Get parser argument values from parsed input.

        Args:
            input (argparse.Namespace): The parsed input.

        Returns:
            dict: A dictionary of argument values.
        """
        dic = AttrDict({k: v.get(input) for k, v in self.parsargs.items()})
        return dic.unflatten()


def parser_dict_from_param(d0: param.Parameterized) -> AttrDict:
    """
    Create a dictionary of parser arguments from a parameter dictionary.

    Args:
        d0 (dict): A dictionary of parameters.

    Returns:
        dict A dictionary of parser arguments.
    """
    # dv0 = aux.AttrDict(d0.param.values())

    d = AttrDict()
    for k, p in d0.param.objects().items():
        # print(k)
        if k == "name" or p.readonly:
            continue
        elif p.__class__ in param.Parameterized.__subclasses__():
            d[k] = parser_dict_from_param(p)
        # elif k in dv0 and dv0[k] is not None:
        # elif type(p) == ClassAttr:
        #     d[k] = parser_dict_from_param(p.class_)
        # elif type(p) == ClassDict:
        #     d[k] = parser_dict_from_param(p.item_type)
        else:
            d[k] = SingleParserArgument.from_param(k, p)
    return d.flatten()


def parser_dict_from_dict(d0: dict) -> AttrDict:
    """
    Create a dictionary of parser arguments from a dictionary.

    Args:
        d0 (dict): A dictionary of parser arguments.

    Returns:
        dict A dictionary of parser arguments.
    """
    p = AttrDict()
    for n, v in d0.items():
        if "v" in v.keys() or "k" in v.keys() or "h" in v.keys():
            p[n] = SingleParserArgument.from_dict(n, **v)
        else:
            p[n] = parser_dict_from_dict(v)
    return p.flatten()


class SimModeParser(ArgumentParser):
    """
    Parser for simulation modes and arguments.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Build the top-level parser and one subparser per simulation mode.

        Args:
            *args: Positional arguments forwarded to
                :class:`argparse.ArgumentParser`.
            **kwargs: Keyword arguments forwarded to
                :class:`argparse.ArgumentParser`.
        """
        self.parser_dicts = AttrDict(
            {
                "screen_kws": ParserArgumentDict.from_param(d0=ScreenOps),
                "SimOps": ParserArgumentDict.from_param(d0=SimOps),
                "RuntimeOps": ParserArgumentDict.from_param(d0=RuntimeOps),
                "Replay": ParserArgumentDict.from_param(d0=reg.gen.Replay),
                "Eval": ParserArgumentDict.from_param(d0=reg.gen.Eval),
                "GAselector": ParserArgumentDict.from_param(d0=reg.gen.GAselector),
                "GAevaluation": ParserArgumentDict.from_param(d0=reg.gen.GAevaluation),
            }
        )
        from .. import __version__

        super().__init__(
            prog="larvaworld",
            description="CLI for running larvaworld simulations",
            *args,
            **kwargs,
        )
        self.add_argument(
            "--version",
            action="version",
            version=f"%(prog)s {__version__}",
            help="Show the version number and exit",
        )
        self.add_argument(
            "-verbose",
            "--VERBOSE",
            type=int,
            default=2,
            help="Level of verbosity in the output",
        )
        self.add_argument(
            "-parsargs",
            "--show_parser_args",
            action="store_true",
            default=False,
            help="Whether to show the parser argument namespace",
        )
        subparsers = self.add_subparsers(
            dest="sim_mode",
            help="The simulation mode to launch",
            parser_class=ArgumentParser,
        )
        for m in SIMTYPES:
            sp = subparsers.add_parser(m)
            sp = self.init_mode_subparser(sp, m)

    def init_mode_subparser(self, sp: ArgumentParser, m: str) -> ArgumentParser:
        """
                Initialize a subparser with common arguments for a specific simulation mode.

        Args:
            sp: The subparser.
            m: The simulation mode.

        Returns:
            The modified subparser.
        """
        if m in ["Exp", "Batch", "Ga"]:
            sp.add_argument(
                "experiment", choices=reg.conf[m].confIDs, help="The experiment mode"
            )
            sp = self.parser_dicts.SimOps.add(sp)
        if m in ["Exp", "Batch"]:
            sp.add_argument(
                "-N",
                "--Nagents",
                type=int,
                help="The number of simulated larvae in each larva group",
            )
        if m == "Batch":
            sp.add_argument(
                "-mIDs",
                "--modelIDs",
                type=str,
                nargs="+",
                help="The larva models to use for creating the simulation larva groups",
            )
            sp.add_argument(
                "-gIDs",
                "--groupIDs",
                type=str,
                nargs="+",
                help="The displayed IDs of the simulation larva groups",
            )
        if m in ["Exp", "Batch", "Eval"]:
            sp.add_argument(
                "-a",
                "--analysis",
                action="store_true",
                default=False,
                help="Whether to run data-analysis after the simulation",
            )
            sp.add_argument(
                "-show",
                "--show",
                action="store_true",
                default=False,
                help="Whether to show the plots generated during data-analysis",
            )

        sp = self.parser_dicts.screen_kws.add(sp)
        sp = self.parser_dicts.RuntimeOps.add(sp)
        if m == "Replay":
            sp = self.parser_dicts.Replay.add(sp)
        elif m == "Eval":
            sp = self.parser_dicts.Eval.add(sp)
        elif m == "Ga":
            sp = self.parser_dicts.GAselector.add(sp)
            sp = self.parser_dicts.GAevaluation.add(sp)
        return sp

    def eval_parser(self, p_key: str, args: Namespace) -> AttrDict:
        """
                Evaluate a parser argument.

        Args:
            p_key: The argument key.

        Returns:
            The parsed value of the argument.
        """
        return self.parser_dicts[p_key].get(args)

    def configure(self, args: Namespace) -> tuple[Any, AttrDict]:
        """
                Configure the simulation run based on parsed arguments.

        Args:
            show_args: Whether to show parsed arguments.

        Returns:
            The configured simulation run.
        """
        m = args.sim_mode
        # Propagate CLI verbosity to global vprint setting
        import larvaworld

        larvaworld.VERBOSE = args.VERBOSE
        kw = AttrDict(
            {
                "screen_kws": self.eval_parser("screen_kws", args),
                **self.eval_parser("RuntimeOps", args),
            }
        )
        if m not in ["Replay", "Eval"]:
            kw.update(**self.eval_parser("SimOps", args))
            kw.experiment = args.experiment
        if m == "Batch":
            kw.mode = "batch"
            kw.experiment = args.experiment
            conf = reg.conf.Batch.getID(args.experiment)
            Nagents = getattr(args, "Nagents", None)
            modelIDs = getattr(args, "modelIDs", None)
            groupIDs = getattr(args, "groupIDs", None)
            if Nagents is not None:
                conf.N = args.Nagents
            if modelIDs is not None:
                conf.modelIDs = modelIDs
            if groupIDs is not None:
                conf.groupIDs = groupIDs
            run = sim.BatchRun(**kw, **conf)
        elif m == "Exp":
            kw.N = args.Nagents
            modelIDs = getattr(args, "modelIDs", None)
            groupIDs = getattr(args, "groupIDs", None)
            if groupIDs is None and modelIDs is not None:
                groupIDs = modelIDs
            if modelIDs is not None:
                kw.modelIDs = modelIDs
            if groupIDs is not None:
                kw.groupIDs = groupIDs
            run = sim.ExpRun(**kw)
        elif m == "Ga":
            p = reg.conf.Ga.expand(kw.experiment)
            ev = self.eval_parser("GAevaluation", args)
            sel = self.eval_parser("GAselector", args)

            for k in ["base_model", "bestConfID", "space_mkeys"]:
                if sel[k] is None:
                    sel.pop(k)
            p.ga_select_kws.update(**sel)

            if ev.refID is not None:
                p.refID = ev.refID
            kw.parameters = p
            # Set GA-specific screen defaults if values match ScreenOps defaults
            # ScreenOps defaults are black_background=False and panel_width=0
            # For GA simulations, we want panel_width=600 (but keep black_background=False for white background)
            # If the values are the ScreenOps defaults (not explicitly set to different values),
            # replace them with GA defaults
            if kw.screen_kws.get("panel_width") == 0:
                kw.screen_kws["panel_width"] = 600
            run = sim.GAlauncher(**kw)
        elif m == "Eval":
            run = sim.EvalRun(**self.eval_parser("Eval", args), **kw)
        elif m == "Replay":
            kw.parameters = self.eval_parser("Replay", args)
            run = sim.ReplayRun(**kw)
        else:
            raise ValueError(f"Simulation mode {m} not recognized")
        return run, kw

    def show_args(
        self,
        args: Namespace,
        run_kws: AttrDict,
        nested: bool = True,
        flat_nested: bool = False,
        input: bool = False,
    ) -> None:
        """
                Show parsed arguments.

        Args:
            nested: Whether to show arguments as a nested dictionary.
            flat_nested: Whether to show arguments as a flattened nested dictionary.
            input: Whether to show input arguments.
        """
        print(f"Simulation mode : {args.sim_mode}")
        if nested:
            print("Simulation args as nested dictionary: ")
            run_kws.print(flat=False)
        if flat_nested:
            print("Simulation args as flattened nested dictionary: ")
            run_kws.print(flat=True)
        if input:
            print("Input args : ")
            AttrDict(vars(args)).print(flat=True)

    def launch(self, run: Any, args: Namespace) -> None:
        """
        Launch the simulation run.
        """
        m = args.sim_mode
        if m == "Batch":
            run.run()
        elif m == "Exp":
            _ = run.simulate()
            if args.analysis:
                run.analyze(show=args.show)
        elif m == "Ga":
            _ = run.simulate()
        elif m == "Eval":
            _ = run.simulate()
            if args.analysis:
                run.plot_results(show=args.show)
                run.plot_models(show=args.show)
        elif m == "Replay":
            run.run()
