from argparse import ArgumentParser


from lib import aux
from cli.parser import run_template, get_parser

p=ArgumentParser()
subps = p.add_subparsers(dest='sim_mode', help='The simulation mode to launch')

MPs=dict()
ps=dict()
for mode in ['Exp','Batch', 'Ga', 'Eval', 'Replay'] :
    subp = subps.add_parser(mode)
    MPs[mode], ps[mode] = get_parser(mode,subp)


if __name__ == "__main__":
    kwargs = aux.AttrDict(vars(p.parse_args()))
    sim_mode = kwargs.sim_mode
    d = MPs[sim_mode].get(kwargs)
    kwargs.pop('sim_mode')
    run_template(sim_mode, kwargs, d)

