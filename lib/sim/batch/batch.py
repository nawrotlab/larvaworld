'''
This file is the template for a batch exec of simulations.
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

from lib.aux import dictsNlists as dNl, colsNstr as cNs, naming as nam
from lib.registry.pars import preg
from lib.sim.batch.aux import grid_search_dict, delete_traj
from lib.sim.batch.functions import single_run, batch_method_unpack
from lib.registry import reg
''' Default batch exec.
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
- Simulation time per exec
- Parameters to perform space search
- Values of the parameters to combine. 
- par_space_density : If values is None then the space to search will be a grid. 
Each parameter will be sampled at a given number of equally-distanced values within the provided range.
This number can be the same for all parameters (par_space_steps is an int) or different for each parameter (par_space_steps is a list of ints)

Examples of this default batch exec are given in  :
- chemo_batchrun.py for chemorbit and chemotax experiments
- feed_scatter_batchrun.py for feed_scatter_experiment
'''


class BatchRun:
    def __init__(self, batch_type='unnamed', batch_id='template', space_search=None, exp=None, params=None, post_kws={},
                 exp_kws={}, proc_kws={}, multiproc=True, resumable=False, save_hdf5=False, batch_methods=None,
                 optimization=None, ncores=8):
        self.s0 = time.time()
        self.id = batch_id
        self.type = batch_type
        self.parent_dir = f'{reg.Path.BATCH}/{self.type}'
        self.dir = f'{self.parent_dir}/{self.id}'
        self.filename = f'{self.parent_dir}/{self.type}.hdf5'
        bm=batch_method_unpack(**batch_methods)
        self.finfunc = bm['finfunc']
        self.procfunc = bm['procfunc']
        self.postfunc = bm['postfunc']

        self.batch_methods = batch_methods

        if optimization is not None:
            optimization['ranges'] = np.array([space_search[k]['range'] for k in space_search.keys() if 'range' in space_search[k].keys()])
        self.space = grid_search_dict(space_search)
        self.optimization = optimization
        self.exp = exp
        self.params = params
        self.run_kws = {
            'runfunc': single_run,
            'procfunc': self.procfunc,
            'save_hdf5': save_hdf5,
            'exp_kws': {**exp_kws,
                        'save_to': self.dir,
                        'vis_kwargs': preg.get_null('visualization', mode=None),
                        'collections': exp['collections']
                        },
            'proc_kws': proc_kws
        }
        self.resumable = resumable
        self.env2_kws = {
            'file_title': self.type,
            'small_overview_tables': True,
            'large_overview_tables': True,
            'summary_tables': True,
        }

        self.env1_kws = {
            'comment': f'{self.type} batch exec!',
            'multiproc': multiproc,
            'resumable': self.resumable,
            'resume_folder': self.dir,
            'ncores': ncores,
            'wrap_mode': pypetconstants.WRAP_MODE_QUEUE,
            'report_progress': (20, 'pypet', logging.CRITICAL),
            'use_pool': True,  # Our runs are inexpensive we can get rid of overhead by using a pool
            'freeze_input': True,  # We can avoid some overhead by freezing the input to the pool
            # wrap_mode=pypetconstants.WRAP_MODE_QUEUE if multiproc else pypetconstants.WRAP_MODE_LOCK,
            'graceful_exit': True,
        }

        self.env = self.get_env()
        if self.postfunc is not None:
            self.env.add_postprocessing(self.postfunc, **post_kws)

    def resume(self):
        env = Environment(continuable=True)
        env.resume(trajectory_name=self.id, resume_folder=self.dir)
        print('Resumed existing trajectory')
        return env

    def load(self):
        traj = load_trajectory(filename=self.filename, name=self.id, load_all=0)
        env = Environment(trajectory=traj, **self.env1_kws)
        traj = self.config(traj, self.optimization, self.batch_methods)
        traj.f_load(index=None, load_parameters=2, load_results=0)
        traj.f_expand(self.space)
        print('Loaded existing trajectory')
        return env

    def build(self, overwrite_file):
        env = Environment(trajectory=self.id, filename=self.filename, overwrite_file=overwrite_file,
                          **self.env1_kws, **self.env2_kws)
        traj = self.prepare(env.traj, self.exp, self.params, self.id, self.dir)
        traj = self.config(traj, self.optimization, self.batch_methods)
        traj.f_explore(self.space)
        return env

    def run(self):
        self.env.run(**self.run_kws)
        self.env.disable_logging()
        print('Batch exec complete')
        if self.finfunc is not None:
            res = self.finfunc(self.env.traj)
        print(f'Batch-exec completed in {np.round(time.time() - self.s0).astype(int)} seconds!')
        return res

    def get_env(self):
        if self.resumable:
            try:
                return self.resume()
            except:
                pass
        try:
            return self.load()
        except:
            try:
                delete_traj(self.type, self.id)
            except :
                pass
            try:
                env = self.build(overwrite_file=False)
                print('Created novel environment without overwriting file')
            except:
                try:
                    env = self.build(overwrite_file=True)
                    print('Created novel environment by overwriting file')
                except:
                    raise ValueError('Loading, resuming or creating a new environment failed')
        return env

    def config(self, traj, optimization, batch_methods):
        if optimization is not None:
            opt_dict = dNl.flatten_dict(optimization, parent_key='optimization', sep='.')
            for k, v in opt_dict.items():
                traj.f_aconf(k, v)
        if batch_methods is not None:
            opt_dict = dNl.flatten_dict(batch_methods, parent_key='batch_methods', sep='.')
            for k, v in opt_dict.items():
                traj.f_aconf(k, v)
        return traj

    def prepare(self, traj, exp, params, batch_id, dir_path):
        traj = self.load_exp(traj, exp)
        if params is not None:
            for p in params:
                traj.f_apar(p, 0.0)

        traj.f_aconf('dir_path', dir_path, comment='Directory for saving data')
        traj.f_aconf('plot_path', f'{dir_path}/{batch_id}.pdf', comment='File for saving plot')
        traj.f_aconf('data_path', f'{dir_path}/{batch_id}.csv', comment='File for saving data')
        traj.f_aconf('dataset_path', f'{dir_path}/datasets', comment='Directory for saving datasets')
        return traj

    def load_exp(self, traj, exp):
        for k0 in ['env_params', 'sim_params', 'trials', 'enrichment', 'larva_groups']:
            dic = dNl.flatten_dict(exp[k0], parent_key=k0, sep='.')
            for k, v in dic.items():
                if type(v) == list:
                    if len(v) == 0:
                        v = None
                    elif type(v[0]) == list:
                        v = np.array(v)
                traj.f_apar(k, v)
        return traj
