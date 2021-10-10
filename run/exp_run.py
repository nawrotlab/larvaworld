import argparse
import sys
import time
import numpy as np

sys.path.insert(0, '..')
from lib.sim.single.single_run import SingleRun
from lib.sim.single.analysis import sim_analysis
from lib.conf.stored.conf import expandConf, loadConfDict
from lib.anal.argparsers import MultiParser

MP = MultiParser(['visualization', 'sim_params'])
p = MP.add()
p.add_argument('experiment', choices=list(loadConfDict('Exp').keys()), help='The experiment mode')
p.add_argument('-a', '--analysis', action="store_true", help='Whether to run analysis')

args = p.parse_args()
d = MP.get(args)
exp = args.experiment

s = time.time()
exp_conf = expandConf(exp, 'Exp')
exp_conf['sim_params'] = d['sim_params']
ds = SingleRun(**exp_conf, vis_kwargs=d['visualization']).run()

if args.analysis:

    fig_dict, results=sim_analysis(ds, exp)

e = time.time()
if d is not None:
    print(f'    Single run completed in {np.round(e - s).astype(int)} seconds!')
