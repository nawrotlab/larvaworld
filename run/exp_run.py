import argparse
import sys
import time
import numpy as np

sys.path.insert(0, '..')
from lib.sim.single_run import run_sim, SingleRun
from lib.conf.conf import expandConf, loadConfDict
from lib.anal.argparsers import MultiParser, add_exp_kwargs

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
# d = run_sim(**exp_conf, vis_kwargs = d['visualization'])
ds = SingleRun(**exp_conf, vis_kwargs=d['visualization']).run()
# ds = run_sim(**exp_conf, vis_kwargs=d['visualization'])

if args.analysis:
    from lib.sim.analysis import sim_analysis
    fig_dict, results=sim_analysis(ds, exp)

e = time.time()
if d is not None:
    print(f'    Single run completed in {np.round(e - s).astype(int)} seconds!')
