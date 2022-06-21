# import argparse
import sys
import time
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


sys.path.insert(0, '..')
# from run.exec_run import Exec
from lib.sim.single.single_run import SingleRun
# from lib.sim.single.analysis import sim_analysis
from lib.conf.stored.conf import kConfDict
from lib.anal.argparsers import MultiParser, update_exp_conf

s = time.time()
MP = MultiParser(['visualization', 'sim_params'])
p = MP.add()
p.add_argument('experiment', choices=kConfDict('Exp'), help='The experiment mode')
p.add_argument('-a', '--analysis', action="store_true", help='Whether to run analysis')
p.add_argument('-show', '--show', action="store_true", help='Whether to show the analysis plots')
p.add_argument('-N', '--Nagents', type=int, help='The number of simulated larvae in each larva group')
p.add_argument('-ms', '--models', type=str, nargs='+', help='The larva models to use for creating the simulation larva groups')

args = p.parse_args()
d = MP.get(args)
exp = args.experiment
N = args.Nagents
models = args.models


exp_conf = update_exp_conf(exp, d, N, models)

# exec = Exec(mode='sim', conf=exp_conf, run_externally=False)
# exec.run()
# while not exec.check() :
#     pass
# fig_dict, results = exec.results
run = SingleRun(**exp_conf, vis_kwargs=d['visualization'])
ds=run.run()

if args.analysis:
    fig_dict, results = run.analyze(show=args.show)

e = time.time()
if d is not None:
    print(f'    Single run completed in {np.round(e - s).astype(int)} seconds!')
