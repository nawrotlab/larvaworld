import argparse
import sys
import time
import numpy as np



sys.path.insert(0, '..')
from lib.sim.single_run import run_sim, get_exp_conf
from lib.sim.analysis import sim_analysis
from lib.conf.conf import loadConfDict
from lib.aux import argparsers as prs

s = time.time()

parser = argparse.ArgumentParser(description="Run given experiments")
parser.add_argument('experiment', choices=list(loadConfDict('Exp').keys()), help='The experiment mode')
parser.add_argument('-a', '--analysis', action="store_true", help='Whether to run analysis')
parser.add_argument('-no_save', '--no_save', action="store_true", help='Whether to run analysis')

parser = prs.add_sim_kwargs(parser)
parser = prs.add_life_kwargs(parser)
parser = prs.add_vis_kwargs(parser)
parser = prs.add_place_kwargs(parser)

args = parser.parse_args()

exp_type = args.experiment
analysis = args.analysis
sim_kwargs = prs.get_sim_kwargs(args)
life_kwargs = prs.get_life_kwargs(args)
vis_kwargs = prs.get_vis_kwargs(args)
place_kwargs = prs.get_place_kwargs(args)



exp_conf = get_exp_conf(exp_type,  sim_kwargs, life_kwargs, **place_kwargs)
kws={'vis_kwargs':vis_kwargs, 'save_data_flag' : not args.no_save, **exp_conf}

d = run_sim(**kws)

if analysis:
    fig_dict, results=sim_analysis(d, exp_type)

e = time.time()
if d is not None:
    print(f'    Single run completed in {np.round(e - s).astype(int)} seconds!')
