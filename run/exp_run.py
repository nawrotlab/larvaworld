import argparse
import sys
import time
import numpy as np

sys.path.insert(0, '..')
from lib.sim.single.single_run import SingleRun
from lib.sim.single.analysis import sim_analysis
from lib.conf.stored.conf import loadConfDict
from lib.anal.argparsers import MultiParser, update_exp_conf

s = time.time()
MP = MultiParser(['visualization', 'sim_params'])
p = MP.add()
p.add_argument('experiment', choices=list(loadConfDict('Exp').keys()), help='The experiment mode')
p.add_argument('-a', '--analysis', action="store_true", help='Whether to run analysis')
p.add_argument('-N', '--Nagents', type=int, help='The number of simulated larvae in each larva group')

args = p.parse_args()
d = MP.get(args)
exp = args.experiment
N = args.Nagents


exp_conf = update_exp_conf(exp, d, N)
ds = SingleRun(**exp_conf, vis_kwargs=d['visualization']).run()

if args.analysis:
    fig_dict, results = sim_analysis(ds, exp)

e = time.time()
if d is not None:
    print(f'    Single run completed in {np.round(e - s).astype(int)} seconds!')
