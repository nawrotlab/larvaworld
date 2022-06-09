

import sys
import time
import numpy as np
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


sys.path.insert(0, '..')
from lib.ga.util.ga_launcher import GAlauncher
print('dd')

from lib.conf.stored.conf import kConfDict
from lib.anal.argparsers import MultiParser, update_exp_conf
print('dd')

s = time.time()
MP = MultiParser(['sim_params', 'ga_select_kws'])
p = MP.add()
p.add_argument('experiment', choices=kConfDict('Ga'), help='The experiment mode')
p.add_argument('-hide', '--show_screen', action="store_false", help='Whether to render the screen visualization')
p.add_argument('-offline', '--offline', action="store_true", help='Whether to run a full LarvaworldSim environment')
print('dd')

p.add_argument('-mID0', '--base_model', choices=kConfDict('Model'), help='The model configuration to optimize')
p.add_argument('-mID1', '--bestConfID', type=str, help='The model configuration ID to store the best genome')

args = p.parse_args()
d = MP.get(args)
exp = args.experiment
base_model = args.base_model
bestConfID = args.bestConfID
show_screen = args.show_screen
offline = args.offline

print('dd')


exp_conf = update_exp_conf(exp, d, conf_type='Ga', ga_select_kws = d['ga_select_kws'],
                           offline=offline, show_screen=show_screen)
print('dd')

if base_model is not None :
    exp_conf.ga_build_kws.base_model=base_model
if bestConfID is not None :
    exp_conf.ga_build_kws.bestConfID=bestConfID

print('dd')


GA=GAlauncher(**exp_conf)
best_genome = GA.run()
print('dd')

e = time.time()
if d is not None:
    print(f'   Genetic Algorithm run completed in {np.round(e - s).astype(int)} seconds!')
