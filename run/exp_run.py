import argparse
import sys
import time

sys.path.insert(0, '..')

from lib.conf import exp_types
from lib.sim.single_run import generate_config, run_sim, sim_analysis
from lib.conf.conf import next_idx
from lib.aux import argparsers as prs

s = time.time()

parser = argparse.ArgumentParser(description="Run given experiments")
parser.add_argument('experiment', choices=list(exp_types.keys()), help='The experiment type')
parser.add_argument('-a', '--analysis', action="store_true", help='Whether to run analysis')

parser = prs.add_sim_kwargs(parser)
parser = prs.add_life_kwargs(parser)
parser = prs.add_vis_kwargs(parser)
parser = prs.add_place_kwargs(parser)

args = parser.parse_args()

exp = args.experiment
analysis = args.analysis
sim_kwargs = prs.get_sim_kwargs(args)
life_kwargs = prs.get_life_kwargs(args)
vis_kwargs = prs.get_vis_kwargs(args)
place_kwargs = prs.get_place_kwargs(args)

if sim_kwargs['sim_id'] is None:
    idx = next_idx(exp)
    sim_kwargs['sim_id'] = f'{exp}_{idx}'
if sim_kwargs['path'] is None:
    sim_kwargs['path'] = f'single_runs/{exp}'

sim_config = generate_config(exp, **place_kwargs, sim_params=sim_kwargs, life_params=life_kwargs)

# print(list(sim_config.keys()))

d = run_sim(**sim_config, **vis_kwargs, enrich=True)

if analysis:
    sim_analysis(d, exp)

e = time.time()
if d is not None:
    print(f'Simulation completed in {e - s}')
