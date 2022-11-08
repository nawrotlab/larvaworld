# import argparse
import sys
import time
import numpy as np
import warnings



warnings.simplefilter(action='ignore', category=FutureWarning)


sys.path.insert(0, '..')


from lib.registry import reg
reg.init()
from lib.anal.argparsers import MultiParser, update_exp_models

s = time.time()
MP = MultiParser(['visualization', 'sim_params'])
p = MP.add()

# p.add_argument(**preg.conftype_dict.conf_parsarg('Exp'))
p.add_argument('experiment', choices=reg.conF.Exp.keys(), help='The experiment mode')
# p.add_argument('experiment', choices=kConfDict('Exp'), help='The experiment mode')
p.add_argument('-a', '--analysis', action="store_true", help='Whether to run analysis')
p.add_argument('-show', '--show', action="store_true", help='Whether to show the analysis plots')
p.add_argument('-N', '--Nagents', type=int, help='The number of simulated larvae in each larva group')
p.add_argument('-ms', '--models', type=str, nargs='+', help='The larva models to use for creating the simulation larva groups')

args = p.parse_args()
d = MP.get(args)
exp = args.experiment
N = args.Nagents
models = args.models

conf0=reg.conF.Exp[exp]
conf=update_exp_models(conf0,models,N)

# conf=dNl.NestDict({k:conf0[k] for k in ['env_params', 'larva_groups', 'trials']})
# sim0,sim=conf0,d['sim_params']
# conf.dt=sim['timestep']
# conf.dur=sim['duration'] if sim['duration'] is not None else sim0.duration
# conf.Box2D=sim['Box2D']
# conf.vis_kwargs=d['visualization']
#
# kws=

for k,v in d['sim_params'].items() :
    if v is not None :
        conf.sim_params[k]=v

conf.analysis=args.analysis
conf.show=args.show

# exp_conf = preg.loadConf(id=exp, conftype='Exp')
# print(exp, exp_conf.enrichment.preprocessing)

# print(preg.enr_dict(proc=['angular', 'spatial', 'dispersion', 'tortuosity'],
#                                                           bouts=['stride', 'pause', 'turn']).preprocessing)
# raise
# exec = Exec(mode='sim', conf=exp_conf, run_externally=False)
# exec.run()
# while not exec.check() :
#     pass
# fig_dict, results = exec.results
from lib.sim.single.exp_run import ExpRun
run = ExpRun(**conf, vis_kwargs=d['visualization'])
ds=run.run()

e = time.time()
if d is not None:
    print(f'    Single run completed in {np.round(e - s).astype(int)} seconds!')
