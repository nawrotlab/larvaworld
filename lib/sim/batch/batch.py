'''
This file is the template for a batch run of simulations.
Simulations are managed through a pypet trajectory.
Results are saved in hdf5 format.
CAUTION : save_hdf5 parameters whether step and end pandas dataframes are saved (in the hdf5 not as csvs). This takes LONG!!!
Created by bagjohn on April 5th 2020
'''

# !/usr/bin/python
import logging
import os
import time
import numpy as np
from pypet import Environment, load_trajectory, pypetconstants

from lib.conf.init_dtypes import null_dict
from lib.sim.batch.aux import config_traj, prepare_traj
from lib.sim.batch.functions import single_run

import lib.stor.paths as paths

''' Default batch run.
Arguments :
- Experiment mode eg 'chemorbit'
- Batchrun configuration as a dict eg :
                {'fit_par': 'final_scaled_dst_to_center',
                'minimize': True,
                'threshold': 0.1,
                'max_Nsims': 1,
                'Nbest': 4,
                'ranges': ranges}
where ranges is a np.array of shape (Npars,2)
- Number of larvae
- Simulation time per run
- Parameters to perform space search
- Values of the parameters to combine. 
- par_space_density : If values is None then the space to search will be a grid. 
Each parameter will be sampled at a given number of equally-distanced values within the provided range.
This number can be the same for all parameters (par_space_steps is an int) or different for each parameter (par_space_steps is a list of ints)

Examples of this default batch run are given in  :
- chemo_batchrun.py for chemorbit and chemotax experiments
- feed_scatter_batchrun.py for feed_scatter_experiment
'''


def batch_run(**kwargs):
    return _batch_run(**kwargs)



def get_batch_env(batch_id, batch_type, dir_path, parent_dir_path, exp, params, optimization, space,batch_methods, **env_kws):
    traj_name = batch_id
    filename = f'{parent_dir_path}/{batch_type}.hdf5'
    env_kws['overwrite_file'] = True
    env = Environment(trajectory=traj_name, filename=filename, **env_kws)
    print('Created novel environment')
    traj = prepare_traj(env.traj, exp, params, batch_id, dir_path)
    traj = config_traj(traj, optimization, batch_methods)
    traj.f_explore(space)
    if os.path.exists(dir_path) :
        if env_kws['resumable']:
            try:
                env = Environment(continuable=True)
                env.resume(trajectory_name=traj_name, resume_folder=dir_path)
                print('Resumed existing trajectory')
                return env
            except:
                pass
        try:
            traj = load_trajectory(filename=filename, name=traj_name, load_all=0)
            env = Environment(trajectory=traj, **env_kws)
            traj = config_traj(traj, optimization, batch_methods)
            traj.f_load(index=None, load_parameters=2, load_results=0)
            traj.f_expand(space)
            print('Loaded existing trajectory')
            return env
        except:
            pass
    for v in [False, True] :
        try :
            env_kws['overwrite_file'] = v
            env = Environment(trajectory=traj_name, filename=filename, **env_kws)
            print('Created novel environment')
            traj = prepare_traj(env.traj, exp, params, batch_id, dir_path)
            traj = config_traj(traj, optimization, batch_methods)
            traj.f_explore(space)
            return env
        except:
            pass
        # try:
        #     env_kws['overwrite_file']=True
        #     env = Environment(trajectory=traj_name, filename=filename, **env_kws)
        #     print('Created novel environment overwriting existing')
        #     traj = prepare_traj(env.traj, exp, params, batch_id, dir_path)
        #     traj = config_traj(traj, optimization)
        #     traj.f_explore(space)
        #     return env
        # except:
    raise ValueError('Loading, resuming or creating a new environment failed')


def _batch_run(batch_type='unnamed', batch_id='template', space=None, exp=None, params=None, post_kws={}, exp_kws={},
               runfunc=single_run, procfunc=None, postfunc=None, finfunc=None, optimization=None,ncores=8,proc_kws={},
               multiproc=True, resumable=False, overwrite_file=False, save_hdf5=False, batch_methods=None):

    s0 = time.time()
    parent_dir_path = f'{paths.BatchRunFolder}/{batch_type}'
    dir_path = f'{parent_dir_path}/{batch_id}'
    env_kws = {
        'file_title': batch_type,
        # 'file_title': batch_id,
        'comment': f'{batch_type} batch run!',
        # 'comment': f'{batch_id} batch run!',
        'multiproc': multiproc,
        'resumable': resumable,
        'small_overview_tables': True,
        'large_overview_tables': True,
        'summary_tables': True,
        'overwrite_file': overwrite_file,
        'resume_folder': dir_path,
        'ncores': ncores,
        'wrap_mode': pypetconstants.WRAP_MODE_QUEUE,
        'report_progress' : (20, 'pypet', logging.CRITICAL),
        # 'ncores': os.cpu_count(),
        'use_pool': True,  # Our runs are inexpensive we can get rid of overhead by using a pool
        'freeze_input': True,  # We can avoid some overhead by freezing the input to the pool
        # 'wrap_mode': pypetconstants.WRAP_MODE_LOCK,
        # wrap_mode=pypetconstants.WRAP_MODE_QUEUE if multiproc else pypetconstants.WRAP_MODE_LOCK,
        'graceful_exit': True,
    }
    run_kws = {
        'runfunc': runfunc,
        'procfunc': procfunc,
        'save_hdf5': save_hdf5,
        'exp_kws': {**exp_kws,
                    'save_to': dir_path,
                    'vis_kwargs': null_dict('visualization', mode=None),
                    'collections': exp['collections']
                    },
        'proc_kws':proc_kws
    }

    def test_batch():
        env = get_batch_env(batch_id, batch_type, dir_path, parent_dir_path, exp=exp, params=params,
                            optimization=optimization,batch_methods = batch_methods,
                            space=space, **env_kws)
        if postfunc is not None:
            env.add_postprocessing(postfunc, **post_kws)
        env.run(**run_kws)
        env.disable_logging()
        print('Batch run complete')
        return env
    env =test_batch()
    # except :
    #     env_kws['overwrite_file'] = True
    #     env =test_batch()
    #     print('Overwritten existing file')

    if finfunc is not None:
        res = finfunc(env.traj)
    s1 = time.time()
    print(f'Batch-run completed in {np.round(s1 - s0).astype(int)} seconds!')
    return res


if __name__ == "__main__":
    batch_type = 'odor-preference'
    from lib.conf.conf import expandConf

    # conf = expandConf(batch_type, 'Batch')
    #
    # batch_kwargs = prepare_batch(conf, 'odor_preference_49', batch_type)
    # df, fig_dict = batch_run(**batch_kwargs)
    # pass
