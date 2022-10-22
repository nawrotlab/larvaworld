import sys
# import argparse


sys.path.insert(0, '..')
# from lib.registry.pars import preg

from lib.registry import reg
reg.init()

from lib.anal.argparsers import MultiParser, update_exp_conf
from run.exec_run import Exec
from lib.conf.stored.conf import kConfDict
MP = MultiParser(['sim_params', 'batch_setup'])
p = MP.add()
# p.add_argument(**preg.conftype_dict.conf_parsarg('Batch'))
p.add_argument('experiment', choices=reg.conF.Batch.keys(), help='The experiment mode')
# p.add_argument('experiment', choices=kConfDict('Batch'), help='The experiment mode')
p.add_argument('-N', '--Nagents', type=int, help='The number of simulated larvae in each larva group')
p.add_argument('-ms', '--models', type=str, nargs='+', help='The larva models to use for creating the simulation larva groups')



args = p.parse_args()
d = MP.get(args)
exp = args.experiment
N = args.Nagents
models = args.models

conf=reg.conF.Batch[exp]


batch_conf = update_exp_conf(exp, d, N, models, conf_type='Batch')


exec = Exec(mode='batch', conf=batch_conf, run_externally=False)
exec.run()
