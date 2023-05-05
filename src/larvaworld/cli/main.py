from argparse import ArgumentParser


from larvaworld.lib import aux
from larvaworld.cli.argparsing import run_template, subparser_dict


def main():
    p=ArgumentParser()
    # subps = p.add_subparsers(dest='sim_mode', help='The simulation mode to launch')
    p,MPs=subparser_dict(p)
    # MPs=dict()
    # ps=dict()
    # for mode in ['Exp','Batch', 'Ga', 'Eval', 'Replay'] :
    #     subp = subps.add_parser(mode)
    #     MPs[mode] = get_parser(mode,subp)
    args = aux.AttrDict(vars(p.parse_args()))
    # print(args)
    # raise
    sim_mode = args.sim_mode
    kw_dicts = MPs[sim_mode].get(args)
    args.pop('sim_mode')
    # kw_dicts['sim_params']=MPs['sim_params'].get(args)
    kws = aux.AttrDict({'id': args.id, **kw_dicts['sim_params']})
    run_template(sim_mode, args, kw_dicts, kws)


if __name__ == '__main__':
    main()