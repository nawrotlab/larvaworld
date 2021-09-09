'''
This file is the template for a batch run of simulations.
Simulations are managed through a pypet trajectory.
Results are saved in hdf5 format.
CAUTION : save_hdf5 parameters whether step and end pandas dataframes are saved (in the hdf5 not as csvs). This takes LONG!!!
Created by bagjohn on April 5th 2020
'''
import itertools
import json
import os
import random
import time
import numpy as np
import pandas as pd
from pypet import Environment, cartesian_product, load_trajectory, pypetconstants
from pypet.parameter import ObjectTable
import matplotlib.pyplot as plt

from lib.anal.plotting import plot_heatmap_PI, plot_endpoint_scatter, plot_debs, plot_3pars, plot_endpoint_params, \
    plot_2d

from lib.sim.single_run import run_sim
import lib.aux.functions as fun
import lib.stor.paths as paths
from lib.stor.larva_dataset import LarvaDataset
import lib.conf.dtype_dicts as dtypes

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


def prepare_batch(batch, batch_id, batch_type):
    space = grid_search_dict(**batch['space_search'])
    if batch['optimization'] is not None:
        batch['optimization']['ranges'] = np.array(batch['space_search']['ranges'])
    # print(list(batch.keys()))
    # exp_conf['sim_params']['path'] = batch_type
    prepared_batch = {
        'batch_type': batch_type,
        'batch_id': batch_id,
        'exp': batch['exp'],
        'space': space,
        **batch_methods(**batch['batch_methods']),
        'optimization': batch['optimization'],
        'exp_kws': batch['exp_kws'],
        'post_kws': {},
    }

    return prepared_batch


def get_Nbest(traj, fit_par, ranges, Nbest=20, minimize=True, mutate=True, recombine=True):
    traj.f_load(index=None, load_parameters=2, load_results=2)
    p_n0s = [traj.f_get(p).v_full_name for p in traj.f_get_explored_parameters()]
    p_vs = [traj.f_get(p).f_get_range() for p in traj.f_get_explored_parameters()]
    fits = np.array([traj.f_get(run).f_get(fit_par).f_get() for run in traj.f_get_run_names(sort=True)])
    idx0 = np.argpartition(fits, Nbest)
    idx = idx0[:Nbest] if minimize else idx0[-Nbest:]
    V0s = np.array([[np.round(np.array(v)[i], 2) for v in p_vs] for i in idx]).T
    if mutate:
        space = []
        for v0s, r in zip(V0s, ranges):
            vs = [fun.mutate_value(v0, r, scale=0.1) for v0 in v0s]
            if recombine:
                random.shuffle(vs)
            space.append(vs)
    else:
        space = list(V0s)
    best_pop = dict(zip(p_n0s, space))
    return best_pop


def get_results(traj, res_names=None):
    p_vs = [traj.f_get(p).f_get_range() for p in traj.f_get_explored_parameters()]
    p_ns = [traj.f_get(p).v_name for p in traj.f_get_explored_parameters()]
    p_n0s = [traj.f_get(p).v_full_name for p in traj.f_get_explored_parameters()]
    runs = traj.f_get_run_names(sort=True)
    runs_idx = [int(i) for i in traj.f_iter_runs(yields='idx')]
    if res_names is None:
        res_names = np.unique([traj.f_get(r).v_name for r in traj.f_get_results()])

    r_vs = [[traj.f_get(run).f_get(r_n).f_get() for run in runs] for r_n in res_names]
    # r_vs = []
    # for res in res_names:
    #     try:
    #         r_vs.append([traj.f_get(run).f_get(res).f_get() for run in runs])
    #     except:
    #         res_names = [r for r in res_names if r != res]
    return runs_idx, p_ns, p_n0s, p_vs, res_names, r_vs


def save_results_df(traj):
    runs_idx, p_ns, p_n0s, p_vs, r_ns, r_vs = get_results(traj, res_names=None)
    cols = list(p_ns) + list(r_ns)
    df = pd.DataFrame(np.array(p_vs + r_vs).T, index=runs_idx, columns=cols)
    df.index.name = 'run_idx'
    try:
        fit_par, minimize = traj.config.fit_par, traj.config.minimize
        df.sort_values(by=fit_par, ascending=minimize, inplace=True)
    except:
        pass
    df.to_csv(os.path.join(traj.config.dir_path, 'results.csv'), index=True, header=True)
    return df


def load_default_configuration(traj, exp):
    for k0 in ['env_params', 'sim_params', 'life_params', 'enrichment']:
        # print(k0)
        dic = fun.flatten_dict(exp[k0], parent_key=k0, sep='.')
        for k, v in dic.items():
            # print(k,v,type(v))
            if type(v) == list and type(v[0]) == list:
                v = np.array(v)
            traj.f_apar(k, v)
    return traj


def default_processing(traj, d=None):
    p = traj.config.fit_par
    s, e = d.step_data, d.endpoint_data
    if p in e.columns:
        vals = e[p].values
    elif p in s.columns:
        vals = s[p].groupby('AgentID').mean()
    else:
        # d.process(types='source',source=(0.04,0), show_output=True)
        # s, e = d.step_data, d.endpoint_data
        # vals = e[p].values
        # # try :
        # from lib.conf.par import post_get_par
        # vals=post_get_par(d,p)
        # except :
        raise ValueError('Could not retrieve fit parameter from dataset')

    ops_mean = traj.config.operations.mean
    ops_std = traj.config.operations.std
    ops_abs = traj.config.operations.abs
    if ops_abs:
        vals = np.abs(vals)
    if ops_mean:
        fit = np.mean(vals)
    elif ops_std:
        fit = np.std(vals)
    traj.f_add_result(p, fit, comment='The fit')
    return d, fit


def null_processing(traj, d=None):
    return d, np.nan


def deb_processing(traj, d=None):
    # dataset.deb_analysis()
    e = d.endpoint_data
    deb_f_mean = e['deb_f_mean'].mean()
    traj.f_add_result('deb_f_mean', deb_f_mean, comment='The average mean deb functional response')
    deb_f_deviation_mean = e['deb_f_deviation_mean'].mean()
    traj.f_add_result('deb_f_deviation_mean', deb_f_deviation_mean,
                      comment='The deviation of average mean deb functional response from 1')
    hunger = e['hunger'].mean()
    traj.f_add_result('hunger', hunger, comment='The average final hunger')
    reserve_density = e['reserve_density'].mean()
    traj.f_add_result('reserve_density', reserve_density, comment='The average final reserve density')

    return d, np.nan


def null_post_processing(traj, result_tuple):
    traj.f_load(index=None, load_parameters=2, load_results=2)


def plot_results(traj, df):
    fig_dict = {}
    filepath = traj.config.dir_path
    p_ns = [traj.f_get(p).v_name for p in traj.f_get_explored_parameters()]
    r_ns = np.unique([traj.f_get(r).v_name for r in traj.f_get_results()])
    kwargs = {'df': df,
              'save_to': filepath,
              'show': False}
    for r_n in r_ns:
        if len(p_ns) == 1:
            fig = plot_2d(labels=p_ns + [r_n], pref=r_n, **kwargs)
            fig_dict[f'{p_ns[0]}VS{r_n}'] = fig
        elif len(p_ns) == 2:
            dic = plot_3pars(labels=p_ns + [r_n], pref=r_n, **kwargs)
            fig_dict.update(dic)
        elif len(p_ns) > 2:
            for i, pair in enumerate(itertools.combinations(p_ns, 2)):
                dic = plot_3pars(labels=list(pair) + [r_n], pref=f'{i}_{r_n}', **kwargs)
                fig_dict.update(dic)
    return fig_dict


def null_final_processing(traj):
    df = save_results_df(traj)
    plots = plot_results(traj, df)
    return df, plots


def end_scatter_generation(traj):
    df = save_results_df(traj)
    data_dir = traj.config.dataset_path
    dirs = [f'{data_dir}/{d}' for d in os.listdir(data_dir)]
    dirs.sort()
    ds = [LarvaDataset(dir) for dir in dirs]
    fig_dict = {}
    kwargs = {'datasets': ds,
              'labels': [d.id for d in ds],
              'save_to': traj.config.dir_path}
    for i in [1, 2, 3]:
        l = f'end_parshorts_{i}'
        par_shorts = getattr(traj.config, l)
        f = plot_endpoint_scatter(**kwargs, keys=par_shorts)
        p1, p2 = par_shorts
        fig_dict[f'{p1}VS{p2}'] = f
    return df, fig_dict


def deb_analysis(traj):
    data_dir = traj.config.dataset_path
    save_to = traj.config.dir_path
    df = save_results_df(traj)

    p_vs = [traj.f_get(p).f_get_range() for p in traj.f_get_explored_parameters()]
    p_ns = [traj.f_get(p).v_name for p in traj.f_get_explored_parameters()]
    # r_ns = np.unique([traj.f_get(r).v_name for r in traj.f_get_results()])

    # if len(p_ns) == 2:
    #     for i in range(len(r_ns)):
    #         r = r_ns[i]
    #         labels = p_ns + [r]
    #         plot_3pars(df, labels, save_to=traj.config.dir_path, pref=r)
    dirs = [f'{data_dir}/{dir}' for dir in os.listdir(data_dir)]
    dirs.sort()
    ds = [LarvaDataset(dir) for dir in dirs]
    if len(ds) == 1:
        new_ids = [None]
    else:
        new_ids = [f'{p_ns[0]} : {v}' for v in p_vs[0]] if len(p_ns) == 1 else [d.id for d in ds]
        plot_endpoint_params(ds, new_ids, mode='deb', save_to=save_to)
    # deb_dicts = fun.flatten_list(
    #     [[deb_dict(d, id, new_id=new_id) for id in d.agent_ids] for d, new_id in zip(ds, new_ids)])
    deb_dicts = fun.flatten_list([d.load_deb_dicts() for d in ds])
    fig_dict = {}
    for m in ['energy', 'growth', 'full']:
        f = plot_debs(deb_dicts=deb_dicts, save_to=save_to, save_as=f'deb_{m}.pdf', mode=m)
        fig_dict[f'deb_{m}'] = f
    return df, fig_dict


def post_processing(traj, result_tuple):
    fit_par, minimize, thr, max_Nsims, Nbest, ranges = traj.config.fit_par, traj.config.minimize, traj.config.threshold, traj.config.max_Nsims, traj.config.Nbest, traj.config.ranges
    traj.f_load(index=None, load_parameters=2, load_results=2)
    runs = traj.f_get_run_names()
    Nruns = len(runs)
    fits = [traj.res.runs.f_get(run).f_get(fit_par).f_get() for run in runs]
    if minimize:
        best = min(fits)
        best_idx = np.argmin(fits)
        thr_reached = best <= thr
    else:
        best = max(fits)
        best_idx = np.argmax(fits)
        thr_reached = best >= thr
    best_run = runs[best_idx]
    print(f'Best result out of {Nruns} runs : {best} in run {best_run}')
    maxNreached = Nruns >= max_Nsims
    if not thr_reached and not maxNreached:
        space = get_Nbest(traj, ranges=ranges, fit_par=fit_par, Nbest=Nbest, minimize=minimize, mutate=True)
        traj.f_expand(space)
        print(f'Continuing expansion with another {Nbest} configurations')
    else:
        p_vs = [traj.f_get(p).f_get_range() for p in traj.f_get_explored_parameters()]
        p_ns = [traj.f_get(p).v_name for p in traj.f_get_explored_parameters()]
        best_config = {}
        for l, p in zip(p_vs, p_ns):
            best_config.update({p: l[best_idx]})
        if maxNreached:
            print(f'Maximum number of simulations reached. Halting search')
        else:
            print(f'Best result reached threshold. Halting search')
        print(f'Best configuration is {best_config} with result {best}')
    # try:
    #     traj.f_remove_items(['best_run_name'])
    # except:
    #     pass
    # traj.f_add_result('best_run_name', best_run, comment=f'The run with the best result')
    traj.f_store()


def single_run(traj, procfunc=None, save_hdf5=True, exp_kws={}):
    sim = fun.reconstruct_dict(traj.f_get('sim_params'))
    sim['sim_ID'] = f'run_{traj.v_idx}'
    sim['path'] = traj.config.dataset_path
    # print(sim['sim_ID'])
    with fun.suppress_stdout(True):
        d = run_sim(
            env_params=fun.reconstruct_dict(traj.f_get('env_params')),
            sim_params=sim,
            life_params=fun.reconstruct_dict(traj.f_get('life_params')),
            **exp_kws)

        if procfunc is None:
            results = np.nan
        else:
            d, results = procfunc(traj, d)

    if save_hdf5:
        s, e = [ObjectTable(data=k, index=k.index, columns=k.columns.values, copy=True) for k in
                [d.step_data.reset_index(level='Step'), d.endpoint_data]]
        traj.f_add_result('end', endpoint_data=e, comment='The simulation endpoint data')
        traj.f_add_result('step', step_data=s, comment='The simulation step-by-step data')
    return d, results


def batch_run(*args, **kwargs):
    return _batch_run(*args, **kwargs)


def get_batch_env(batch_id, batch_type, dir_path, parent_dir_path, exp, params, optimization, space, **env_kws):
    traj_name = batch_id
    filename = f'{parent_dir_path}/{batch_type}.hdf5'
    if os.path.exists(dir_path):
        # if os.path.exists(dir_path) and overwrite == False:
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
        traj = config_traj(traj, optimization)
        traj.f_load(index=None, load_parameters=2, load_results=0)
        traj.f_expand(space)
        print('Loaded existing trajectory')
        return env
    except:
        try:
            env = Environment(trajectory=traj_name, filename=filename, **env_kws)
            print('Created novel environment')
            traj = prepare_traj(env.traj, exp, params, batch_id, dir_path)
            traj = config_traj(traj, optimization)
            traj.f_explore(space)
            return env
        except:
            raise ValueError('Loading, resuming or creating a new environment failed')


def _batch_run(
        batch_type='unnamed',
        batch_id='template',
        space=None,
        save_hdf5=False,
        runfunc=single_run,
        procfunc=None,
        postfunc=None,
        finfunc=None,
        multiproc=True,
        resumable=True,
        exp=None,
        params=None,
        optimization=None,
        post_kws={},
        exp_kws={}
):
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
        'overwrite_file': False,
        'resume_folder': dir_path,
        'ncores': 4,
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
                    'vis_kwargs': dtypes.get_dict('visualization'),
                    'collections': exp['collections']
                    }
    }
    env = get_batch_env(batch_id, batch_type, dir_path, parent_dir_path,
                        exp=exp,
                        params=params,
                        optimization=optimization,
                        space=space,
                        **env_kws)
    if postfunc is not None:
        env.add_postprocessing(postfunc, **post_kws)
    env.run(**run_kws)
    env.disable_logging()
    print('Batch run complete')
    if finfunc is not None:
        res = finfunc(env.traj)
    s1 = time.time()
    print(f'Batch-run completed in {np.round(s1 - s0).astype(int)} seconds!')
    return res


def config_traj(traj, optimization):
    if optimization is not None:
        opt_dict = fun.flatten_dict(optimization, parent_key='optimization', sep='.')
        for k, v in opt_dict.items():
            traj.f_aconf(k, v)
    return traj


def prepare_traj(traj, exp, params, batch_id, dir_path):
    traj = load_default_configuration(traj, exp)
    if params is not None:
        for p in params:
            traj.f_apar(p, 0.0)

    traj.f_aconf('dir_path', dir_path, comment='Directory for saving data')
    traj.f_aconf('plot_path', f'{dir_path}/{batch_id}.pdf', comment='File for saving plot')
    traj.f_aconf('data_path', f'{dir_path}/{batch_id}.csv', comment='File for saving data')
    traj.f_aconf('dataset_path', f'{dir_path}/datasets', comment='Directory for saving datasets')
    return traj


def grid_search_dict(pars, ranges, Ngrid, values=None):
    if values is not None:
        values_dict = dict(zip(pars, values))
    else:
        Npars, Nsteps = len(pars), len(Ngrid)
        if any([type(s) != int for s in Ngrid]):
            raise ValueError('Parameter space steps are not integers')
        if Npars != Nsteps:
            if Nsteps == 1:
                Ngrid = [Ngrid[0]] * Npars
                print('Using the same step for all parameters')
            else:
                raise ValueError('Number of parameter space steps does not match number of parameters and is not one')
        if np.isnan(ranges).any():
            raise ValueError('Ranges of parameters not provided')
        values_dict = {}
        for par, (low, high), s in zip(pars, ranges, Ngrid):
            range = np.linspace(low, high, s)
            if type(low) == int and type(high) == int:
                range = range.astype(int)
            values_dict.update({par: range.tolist()})
    space = cartesian_product(values_dict)
    return space


def get_space_from_file(space_filepath=None, params=None, space_pd=None, returned_params=None, flag=None,
                        flag_range=[0, +np.inf], ranges=None,
                        par4ranges=None, additional_params=None, additional_values=None):
    if space_pd is None:
        space_pd = pd.read_csv(space_filepath, index_col=0)
    if params is None:
        params = space_pd.columns.values.tolist()
    if returned_params is None:
        returned_params = params
    if ((ranges is not None) and (par4ranges is not None)):
        for p, r in zip(par4ranges, ranges):
            space_pd = space_pd[(space_pd[p] >= r[0]) & (space_pd[p] <= r[1])].copy(deep=True)
            print('Ranges found. Selecting combinations within range')
    if flag:
        r0, r1 = flag_range
        space_pd = space_pd[space_pd[flag].dropna() > r0].copy(deep=True)
        space_pd = space_pd[space_pd[flag].dropna() < r1].copy(deep=True)
        print(f'Using {flag} to select suitable parameter combinations')

    values = [space_pd[p].values.tolist() for p in params]
    values = [[float(b) for b in a] for a in values]
    if additional_params is not None and additional_values is not None:
        for p, vs in zip(additional_params, additional_values):
            Nspace = len(values[0])
            Nv = len(vs)
            values = [a * Nv for a in values] + fun.flatten_list([[v] * Nspace for v in vs])
            returned_params += [p]

    space = dict(zip(returned_params, values))
    return space


def PI_computation(traj, dataset):
    ind = dataset.compute_preference_index()
    traj.f_add_result('PI', ind, comment=f'The preference index')
    return dataset, ind


def heat_map_generation(traj):
    csv_filepath = f'{traj.config.dir_path}/PIs.csv'
    p_vs = [traj.f_get(p).f_get_range() for p in traj.f_get_explored_parameters()]
    PIs = [traj.f_get(run).f_get('PI').f_get() for run in traj.f_get_run_names(sort=True)]
    Lgains = np.array(p_vs[0]).astype(int)
    Rgains = np.array(p_vs[1]).astype(int)
    Lgain_range = pd.Series(np.unique(Lgains), name="left_gain")
    Rgain_range = pd.Series(np.unique(Rgains), name="right_gain")
    df = pd.DataFrame(index=Lgain_range, columns=Rgain_range, dtype=float)
    for Lgain, Rgain, PI in zip(Lgains, Rgains, PIs):
        df[Rgain].loc[Lgain] = PI
    df.to_csv(csv_filepath, index=True, header=True)
    fig = plot_heatmap_PI(save_to=traj.config.dir_path, csv_filepath=csv_filepath)
    fig_dict = {'PI_heatmap': fig}
    return df, fig_dict


def load_traj(batch_type, batch_id):
    parent_dir_path = f'{paths.BatchRunFolder}/{batch_type}'
    filename = f'{parent_dir_path}/{batch_type}.hdf5'
    traj = load_trajectory(filename=filename, name=batch_id, load_all=2)
    return traj


def existing_trajs(batch_type):
    import h5py
    filename = f'{paths.BatchRunFolder}/{batch_type}/{batch_type}.hdf5'
    try:
        f = h5py.File(filename, 'r')
        return list(f.keys())
    except:
        return []


def existing_trajs_dict(batch_type):
    import h5py
    filename = f'{paths.BatchRunFolder}/{batch_type}/{batch_type}.hdf5'
    try:
        f = h5py.File(filename, 'r')
        return f
    except:
        return {}


def delete_traj(batch_type, traj_name):
    import h5py
    filename = f'{paths.BatchRunFolder}/{batch_type}/{batch_type}.hdf5'
    with h5py.File(filename, 'r+') as f:
        del f[traj_name]


procfunc_dict = {
    'null': null_processing,
    'default': default_processing,
    'deb': deb_processing,
    'odor_preference': PI_computation,
}

postfunc_dict = {
    'null': null_post_processing,
    'default': post_processing,
}

finfunc_dict = {
    'null': null_final_processing,
    'deb': deb_analysis,
    'scatterplots': end_scatter_generation,
    'odor_preference': heat_map_generation,
}


def batch_methods(run='default', post='default', final='null'):
    return {'procfunc': procfunc_dict[run],
            'postfunc': postfunc_dict[post],
            'finfunc': finfunc_dict[final], }


if __name__ == "__main__":
    batch_type = 'odor-preference'
    from lib.conf.conf import loadConf, expandConf

    conf = expandConf(batch_type, 'Batch')

    batch_kwargs = prepare_batch(conf, 'odor_preference_xxx', batch_type)

    df, fig_dict = batch_run(**batch_kwargs)
