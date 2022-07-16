# import argparse
import sys
import time
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


sys.path.insert(0, '..')
#
# from lib.registry.pars import ParDict
# print(ParDict.dict['b'].d)
# raise
# from run.exec_run import Exec
from lib.sim.replay.replay import ReplayRun
# from lib.sim.single.analysis import sim_analysis
from lib.conf.stored.conf import kConfDict
from lib.anal.argparsers import MultiParser, update_exp_conf

s = time.time()
MP = MultiParser(['replay'])
p = MP.add()
# p.add_argument('experiment', choices=kConfDict('Exp'), help='The experiment mode')
# p.add_argument('-a', '--analysis', action="store_true", help='Whether to run analysis')
# p.add_argument('-show', '--show', action="store_true", help='Whether to show the analysis plots')
# p.add_argument('-N', '--Nagents', type=int, help='The number of simulated larvae in each larva group')
# p.add_argument('-ms', '--models', type=str, nargs='+', help='The larva models to use for creating the simulation larva groups')

args = p.parse_args()
d = MP.get(args)
replay_kws=d['replay']




run = ReplayRun(**replay_kws)
run.run()


e = time.time()
if d is not None:
    print()
    print(f'Replay completed in {np.round(e - s).astype(int)} seconds!')
