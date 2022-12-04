from argparse import ArgumentParser

from lib.aux import dictsNlists as dNl, colsNstr as cNs
from lib.registry import reg
reg.init()



from lib.anal.argparsers import get_parser, run_template

dest='sim_mode'

p=ArgumentParser()
subps = p.add_subparsers(dest=dest, help='The simulation mode')

MPs=dict()
ps=dict()
for mode in ['Exp','Batch', 'Ga', 'Eval', 'Rep'] :
    subp = subps.add_parser(mode)
    MPs[mode], ps[mode] = get_parser(mode,subp)


if __name__ == "__main__":
    args = p.parse_args()


    sim_mode = args.sim_mode
    d = MPs[sim_mode].get(args)

    kwargs = dNl.NestDict(vars(args))
    kwargs.pop(dest)

    run_template(sim_mode, kwargs, d)

