import sys
import argparse



sys.path.insert(0, '..')
from lib.conf.stored.conf import next_idx, loadConfDict, loadConf, get_exp_conf
import lib.anal.argparsers as prs
from run.exec_run import Exec

parser = argparse.ArgumentParser(description="Run given batch-run")
parser.add_argument('batch', choices=list(loadConfDict('Batch').keys()), help='The batch-run mode')
parser = prs.add_sim_kwargs(parser)
parser = prs.add_life_kwargs(parser)
parser = prs.add_batch_kwargs(parser)
parser = prs.add_space_kwargs(parser)
parser = prs.add_optimization_kwargs(parser)
parser = prs.add_place_kwargs(parser)
args = parser.parse_args()
batch_type = args.batch
sim_kwargs = prs.get_sim_kwargs(args)
life_kwargs = prs.get_life_kwargs(args)
place_kwargs = prs.get_place_kwargs(args)

batch_kwargs = prs.get_batch_kwargs(args)
space_kwargs = prs.get_space_kwargs(args)
optimization_kwargs = prs.get_optimization_kwargs(args)

batch_conf = loadConf(batch_type, 'Batch')
if batch_kwargs['batch_id'] is None:
    idx = next_idx(batch_type, type='batch')
    batch_id = f'{batch_type}_{idx}'
else:
    batch_id = batch_kwargs['batch_id']

if batch_conf['optimization'] is not None:
    for k in ['fit_par', 'minimize', 'threshold']:
        if optimization_kwargs[k] is None:
            optimization_kwargs[k] = batch_conf['optimization'][k]
    batch_conf['optimization'] = optimization_kwargs
for k in ['pars', 'ranges', 'Ngrid']:
    if space_kwargs[k] is None:
        space_kwargs[k] = batch_conf['space_search'][k]
batch_conf['space_search'] = space_kwargs

batch_conf['exp'] = get_exp_conf(batch_conf['exp'], sim_kwargs, life_kwargs, **place_kwargs)
batch_conf['batch_id'] = batch_id
batch_conf['batch_type'] = batch_type
batch_conf['optimization']['operations']={'mean':True, 'abs':False, 'std':False}

exec = Exec(mode='batch', conf=batch_conf, run_externally=False)
exec.run()

'''
python batch_run.py odor_pref -N 5 -t 1.0 -id_b test
'''
