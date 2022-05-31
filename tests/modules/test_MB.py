
import copy
import sys
from os import listdir

import numpy as np
import os
import pandas as pd
from multiprocessing import Pool
import matplotlib.pyplot as plt

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# Default code to locate to the correct working directory


sys.path.insert(0, '../..')
# os.chdir('../..')
#
# print(os.getcwd())


# for i in range(len(fs)):
#     print(ns[i])
#     for BB_key,BBs in fs[i].items():
#         print(BB_key,  '....', np.median(BBs))
#     print()
from lib.anal.argparsers import MultiParser, update_exp_conf
# from lib.conf.stored.conf import loadConf, kConfDict, expandConf
from lib.sim.single.single_run import SingleRun
from lib.conf.base.dtypes import null_dict

# from lib.conf.stored.larva_conf import OD
from lib.aux.dictsNlists import load_dict, flatten_dict, save_dict, AttrDict
# from lib.conf.base import paths



def get_PI_single(run_kws) :
    run = SingleRun(**run_kws)
    ds = run.run()
    fig_dict, results = run.analyze()
    return [results['PIs']['Larva'],results['PI2s']['Larva']]


def get_PI(exp='test', N=5,memory_mode='MB', video=True):
    vis_kwargs = null_dict('visualization', mode='video', video_speed=60) if video else null_dict('visualization', mode=None)
    # parameter_dict = {
    #     'brain.olfactor_params.odor_dict.CS.mean': BBs*G,
    #     'brain.olfactor_params.odor_dict.UCS.mean': np.zeros(BBs.shape[0])
    #                  }
    sim = null_dict('sim_params',
                    sim_ID=f'{N}models.sim_RemoteMB',
                    path=f'single_runs/MB_model/{exp}',
                    duration=3,
                    timestep=0.1,
                    store_data=False)
    exp_conf = update_exp_conf('PItest_off', d={'sim_params': sim}, N=N, models=None)

    # print(exp_conf.enrichment)
    # raise

    exp_conf.enrichment.bout_annotation = False
    # print(exp_conf.larva_groups.Larva.model.brain.memory_params)
    # raise
    exp_conf.larva_groups.Larva.model.brain.modules.memory=True
    exp_conf.larva_groups.Larva.model.brain.memory_params =null_dict('memory', mode=memory_mode)
    # print( exp_conf.larva_groups.Larva.model.brain.memory_params)
    exp_conf.larva_groups.Larva.model.brain.turner_params.activation_noise = 0.0
    exp_conf.larva_groups.Larva.model.brain.turner_params.noise = 0.0
    exp_conf.update(
        {
            'show_output': False,
            # 'parameter_dict': parameter_dict
        }
    )
    run_kws=AttrDict.from_nested_dicts({**exp_conf, 'vis_kwargs': vis_kwargs})

    return get_PI_single(run_kws)

    # with Pool(Ncores) as p:
    #     temp=p.map(get_PI_single, run_kws_list)
    #     temp=np.array(temp)
    #     return temp[:, 0], temp[:, 1]


if __name__ == '__main__':
    res=get_PI()
    print(res)