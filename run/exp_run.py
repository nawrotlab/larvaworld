import sys
import argparse
import time



sys.path.insert(0, '..')
from lib.conf import exp_types
from lib.sim.single_run import generate_config, run_sim, sim_analysis
from lib.aux import argparsers as prs

s=time.time()

parser = argparse.ArgumentParser(description="Run given experiments")
parser.add_argument('experiment', choices=list(exp_types.keys()), help='The experiment type')
parser.add_argument('-a', '--analysis', action="store_true", help='Whether to run analysis')

parser = prs.add_sim_kwargs(parser)
parser = prs.add_vis_kwargs(parser)

args = parser.parse_args()

exp = args.experiment
analysis = args.analysis
sim_kwargs = prs.get_sim_kwargs(args)
vis_kwargs = prs.get_vis_kwargs(args)

sim_config = generate_config(exp, **sim_kwargs)
exp_config = {
              'common_folder': f'single_runs/{exp}',
              **sim_config}
d=run_sim(**exp_config, **vis_kwargs)

if analysis:
    sim_analysis(d, exp)

e=time.time()
print(f'Simulation completed in {e-s}')