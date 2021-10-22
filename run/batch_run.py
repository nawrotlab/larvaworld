import sys
import argparse

sys.path.insert(0, '..')
from lib.conf.stored.conf import next_idx, loadConfDict, loadConf
import lib.anal.argparsers as prs
from run.exec_run import Exec

MP = prs.MultiParser(['sim_params', 'batch_setup'])
p = MP.add()
p.add_argument('batch', choices=list(loadConfDict('Batch').keys()), help='The batch-run mode')
p.add_argument('-N', '--Nagents', type=int, help='The number of simulated larvae in each larva group')

args = p.parse_args()
d = MP.get(args)
batch_type = args.batch
N = args.Nagents

batch_conf = loadConf(batch_type, 'Batch')

batch_id = d['batch_setup']['batch_id']
if batch_id is None:
    idx = next_idx(batch_type, type='batch')
    batch_id = f'{batch_type}_{idx}'

batch_conf['exp'] = prs.update_exp_conf(batch_conf['exp'], d, N)
batch_conf['batch_id'] = batch_id
batch_conf['batch_type'] = batch_type

exec = Exec(mode='batch', conf=batch_conf, run_externally=False)
exec.run()
