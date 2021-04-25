import argparse
import sys
import time

sys.path.insert(0, '..')

from lib.sim.single_run import run_sim, sim_analysis, get_exp_conf
from lib.conf.conf import next_idx, loadConfDict
from lib.aux import argparsers as prs

s = time.time()

parser = argparse.ArgumentParser(description="Run given experiments")
parser.add_argument('experiment', choices=list(loadConfDict('Exp').keys()), help='The experiment type')
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

if sim_kwargs['sim_id'] is None:
    idx = next_idx(exp_type)
    sim_kwargs['sim_id'] = f'{exp_type}_{idx}'
if sim_kwargs['path'] is None:
    sim_kwargs['path'] = f'single_runs/{exp_type}'

exp_conf = get_exp_conf(exp_type,  sim_kwargs, life_kwargs, enrich=True, **place_kwargs)
d = run_sim(**exp_conf, vis_kwargs=vis_kwargs, save_data_flag=not args.no_save)

if analysis:
    fig_dict, results=sim_analysis(d, exp_type)

e = time.time()
if d is not None:
    print(f'Simulation completed in {e - s}')
