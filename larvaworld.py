from argparse import ArgumentParser


from larvaworld import aux
from cli.argparsing import run_template, get_parser

p=ArgumentParser()
subps = p.add_subparsers(dest='sim_mode', help='The simulation mode to launch')

MPs=dict()
# ps=dict()
for mode in ['Exp','Batch', 'Ga', 'Eval', 'Replay'] :
    subp = subps.add_parser(mode)
    MPs[mode] = get_parser(mode,subp)


if __name__ == "__main__":
    kws = aux.AttrDict(vars(p.parse_args()))
    # print(kws)
    # raise
    sim_mode = kws.sim_mode
    kw_dicts = MPs[sim_mode].get(kws)
    kws.pop('sim_mode')
    run_template(sim_mode, kws, kw_dicts)

