import argparse
import sys
import time
import numpy as np
import warnings



warnings.simplefilter(action='ignore', category=FutureWarning)


sys.path.insert(0, '..')
from lib.ga.util.ga_engine import GAlauncher
from lib.conf.stored.conf import kConfDict
from lib.anal.argparsers import MultiParser, update_exp_conf

s = time.time()
MP = MultiParser(['sim_params'])
# MP = MultiParser(['visualization', 'sim_params'])
p = MP.add()
p.add_argument('experiment', choices=kConfDict('Ga'), help='The experiment mode')
p.add_argument('-hide', '--show_screen', action="store_false", help='Whether to render the screen visualization')

# p.add_argument('-a', '--analysis', action="store_true", help='Whether to run analysis')
# p.add_argument('-show', '--show', action="store_true", help='Whether to show the analysis plots')
p.add_argument('-N', '--Nagents', type=int, help='The number of simulated larvae in each generation')
p.add_argument('-ms', '--models', type=str, nargs='+', help='The base model and storage model for the GA run')

args = p.parse_args()
d = MP.get(args)
exp = args.experiment
N = args.Nagents
models = args.models
show_screen = args.show_screen


exp_conf = update_exp_conf(exp, d, N, models, conf_type='Ga')
exp_conf.show_screen = show_screen

GAlauncher(**exp_conf)
# run = SingleRun(**exp_conf, vis_kwargs=d['visualization'])
# ds=run.run()

# if args.analysis:
#     fig_dict, results = run.analyze(show=args.show)

e = time.time()
if d is not None:
    print(f'   Genetic Algorithm run completed in {np.round(e - s).astype(int)} seconds!')
