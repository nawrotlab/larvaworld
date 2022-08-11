# import argparse
import sys
import time
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


sys.path.insert(0, '..')
from lib.registry import reg
reg.init(['DEF', 'ParsD'])
#
# from lib.registry.pars import ParDict
# print(ParDict.dict['b'].d)
# raise
# from run.exec_run import Exec
from lib.sim.replay.replay import ReplayRun
from lib.anal.argparsers import MultiParser

s = time.time()
MP = MultiParser(['replay'])
p = MP.add()

args = p.parse_args()
d = MP.get(args)
replay_kws=d['replay']




run = ReplayRun(**replay_kws)
run.run()


e = time.time()
if d is not None:
    print()
    print(f'Replay completed in {np.round(e - s).astype(int)} seconds!')

'''
python replay_run.py -refID exploration.dish -id replay_dish -ids 1 -fix_p 6 -vis0
'''
