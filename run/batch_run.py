import sys
import argparse

sys.path.insert(0, '..')
from lib.conf.batch_modes import batch_types
from lib.sim.batch_lib import *
from lib.sim.single_run import generate_config, next_idx
import lib.aux.functions as fun
import lib.aux.argparsers as prs

parser = argparse.ArgumentParser(description="Run given batch-run")
parser.add_argument('experiment', choices=list(batch_types.keys()), help='The experiment type')
parser = prs.add_sim_kwargs(parser)
parser = prs.add_batch_kwargs(parser)
parser = prs.add_space_kwargs(parser)

args = parser.parse_args()
# print(args)
exp = args.experiment
sim_kwargs = prs.get_sim_kwargs(args)
sim_config = generate_config(exp, **sim_kwargs)

setup = batch_types[exp]
space_method = setup['space_method']
space_kwargs = prs.get_space_kwargs(args)

if space_kwargs['pars'] is None:
    space_kwargs['pars'] = setup['pars']
if space_kwargs['ranges'] is None:
    space_kwargs['ranges'] = setup['ranges']
else:
    space_kwargs['ranges'] = np.array(fun.group_list_by_n(space_kwargs['ranges'], 2))
space = space_method(**space_kwargs)

batch_config = setup['batch_config']
batch_kwargs = prs.get_batch_kwargs(args)

if batch_config is not None:
    batch_config['max_Nsims'] = batch_kwargs['max_Nsims']
    batch_config['Nbest'] = batch_kwargs['Nbest']
    batch_config['ranges'] = space_kwargs['ranges']

if batch_kwargs['batch_id'] is None :
    idx = next_idx(exp, type='batch')
    batch_id = f'{exp}_{idx}'
else :
    batch_id = batch_kwargs['batch_id']


batch_run(dir=exp,
          batch_id=batch_id,
          space=space,
          process_method=setup['process_method'],
          post_process_method=setup['post_process_method'],
          final_process_method=setup['final_process_method'],
          sim_config=sim_config,
          config=batch_config,
          post_kwargs=setup['post_kwargs'],
          run_kwargs=setup['run_kwargs']
          )

'''
python batch_run.py odor_pref -N 25 -t 3.0 -rng -200.0 200.0 -Ngrd 5
'''
