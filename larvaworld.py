from argparse import ArgumentParser


from lib import reg, aux, plot
from cli.cli_argparsers import run_template, get_parser

p=ArgumentParser()
subps = p.add_subparsers(dest='sim_mode', help='The simulation mode')

MPs=dict()
ps=dict()
for mode in ['Exp','Batch', 'Ga', 'Eval', 'Rep'] :
    subp = subps.add_parser(mode)
    MPs[mode], ps[mode] = get_parser(mode,subp)


if __name__ == "__main__":
    args = p.parse_args()
    sim_mode = args.sim_mode
    d = MPs[sim_mode].get(args)
    kwargs = aux.NestDict(vars(args))
    kwargs.pop('sim_mode')
    run_template(sim_mode, kwargs, d)

