from argparse import ArgumentParser

from cli.conf_aux import update_exp_conf
from lib import reg, aux


class Parser:
    """
    Create an argument parser for a group of arguments (normally from a dict).
    """

    def __init__(self, name):
        self.name = name
        self.parsargs = reg.parsers.parser_dict[name]

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
        return aux.NestDict({k: v.get(input) for k, v in self.parsers.items()})


def run_template(sim_mode, args, d):
    # MP, p = get_parser(conftype)

    if sim_mode == 'Rep':
        from lib.sim.replay import ReplayRun
        run = ReplayRun(**d['replay'])
        run.run()
    elif sim_mode == 'Batch':
        from lib.sim.exec_run import Exec
        conf = update_exp_conf(exp=args.experiment, d=d, N=args.Nagents, models=args.models, conf_type='Batch')
        exec = Exec(mode='batch', conf=conf, run_externally=False)
        exec.run()
    elif sim_mode == 'Exp':
        from lib.sim.single_run import SingleRun
        conf = update_exp_conf(exp=args.experiment, d=d, N=args.Nagents, models=args.models, conf_type='Exp')


        run = SingleRun(**conf, vis_kwargs=d['visualization'])



        ds = run.run()

        if args.analysis:
            fig_dict, results = run.analyze(show=args.show)
    elif sim_mode == 'Ga':
        from lib.sim.ga_launcher import GAlauncher
        conf = update_exp_conf(exp=args.experiment, d=d, offline=args.offline, show_screen=args.show_screen,
                               conf_type='Ga')

        conf.ga_select_kws = d['ga_select_kws']

        if args.base_model is not None:
            conf.ga_build_kws.base_model = args.base_model
        if args.bestConfID is not None:
            conf.ga_build_kws.bestConfID = args.bestConfID
        GA = GAlauncher(**conf)
        best_genome = GA.run()
    elif sim_mode == 'Eval':
        from lib.sim.evaluation import EvalRun
        evrun = EvalRun(**d.eval_conf, show=args.show_screen)
        evrun.simulate()
        evrun.analyze()
        evrun.plot_results()
        evrun.plot_models()


def get_parser(sim_mode, parser=None):
    dic = aux.NestDict({
        'Batch': [['sim_params', 'batch_setup'], ['e', 'N', 'ms']],
        'Eval': [['eval_conf'], ['hide']],
        'Exp': [['sim_params', 'visualization'], ['e', 'N', 'ms', 'a']],
        'Ga': [['sim_params', 'ga_select_kws'], ['e', 'mID0', 'mID1', 'offline', 'hide']],
        'Rep': [['replay'], []]
    })
    mks, ks = dic[sim_mode]

    MP = MultiParser(mks)
    p = MP.add(parser)
    for k in ks:
        if k == 'e':
            p.add_argument('experiment', choices=reg.conf0.dict[sim_mode].ConfIDs, help='The experiment mode')
        elif k == 'N':
            p.add_argument('-N', '--Nagents', type=int, help='The number of simulated larvae in each larva group')
        elif k == 'ms':
            p.add_argument('-ms', '--models', type=str, nargs='+',
                           help='The larva models to use for creating the simulation larva groups')
        elif k == 'mID0':
            p.add_argument('-mID0', '--base_model', choices=reg.conf0.dict['Model'].ConfIDs,
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
