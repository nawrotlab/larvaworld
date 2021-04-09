'''
This file is the template for a batch run of simulations.
Simulations are managed through a pypet trajectory.
Results are saved in hdf5 format.
CAUTION : save_data_in_hdf5 parameters whether step_data and endpoint_data pandas dataframes are saved (in the hdf5 not as csvs). This takes LONG!!!
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

from lib.anal.plotting import plot_heatmap_PI, plot_endpoint_scatter, plot_debs, plot_3pars, \
    plot_endpoint_params, plot_3d, plot_2d
from lib.model.agents.deb import deb_dict
from lib.sim.single_run import run_sim
import lib.aux.functions as fun
import lib.stor.paths as paths
from lib.stor.larva_dataset import LarvaDataset

''' Default batch run.
Arguments :
- Experiment type eg 'chemorbit'
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


# def batch_default(experiment, batch_config, Nagents, sim_time, params, values=None, par_space_steps=3):
#     batch_id = f'{experiment}_batchrun'
#     batch_idx = 0
#
#     sim_config = generate_config(experiment=experiment, Nagents=Nagents, sim_time=sim_time, config=None)
#
#     # Collect only the endpoint parameter required for fitting
#     sim_config['sim_params']['collect_effectors'] = ['']
#     sim_config['sim_params']['collected_step_parameters'] = ['']
#     sim_config['sim_params']['collected_endpoint_parameters'] = [batch_config['fit_par']]
#
#     if values is None:
#         space = grid_search_dict(params, batch_config['ranges'], Ngrid=par_space_steps)
#     else:
#         values_dict = dict(zip(params, values))
#         space = cartesian_product(values_dict)
#
#     batch_run(batch_id=batch_id,
#               batch_idx=batch_idx,
#               space=space,
#               save_data_in_hdf5=False,
#               process_method=default_processing,
#               post_process_method=post_processing,
#               final_process_method=null_final_processing,
#               sim_config=sim_config,
#               config=batch_config
#               )
def prepare_batch(batch, batch_id, sim_config):
    space = grid_search_dict(**batch['space_search'])
    batch['optimization']['ranges']=np.array(batch['space_search']['ranges'])
    prepared_batch = {
        'space': space,
        'dir': batch['exp'],
        'batch_id': batch_id,
        'sim_config': sim_config,
        # 'pars': batch['space_search']['pars'],
        # 'ranges': np.array(batch['space_search']['ranges']),
        'process_method': default_processing,
        'post_process_method': post_processing,
        'final_process_method': null_final_processing,
        # 'space_method': grid_search_dict,
        'optimization': batch['optimization'],
        'post_kwargs': {},
        'run_kwargs': {}
    }
    return prepared_batch

def get_best_individuals(traj, ranges=np.nan, fit_par='global_fit', num_individuals=20, minimize=True,
                         mutate=True, recombine=True):
    traj.f_load(index=None, load_parameters=2, load_results=2)
    runs_idx, runs, par_names, par_full_names, par_values, res_names, global_fits = get_results(traj,
                                                                                                res_names=[fit_par])
    global_fits = np.array(global_fits)
    idx = np.argpartition(global_fits, num_individuals)[0]
    inds = []
    if minimize:
        selected_idx = idx[:num_individuals]
    else:
        selected_idx = idx[-num_individuals:]
    # print(par_values)
    for i in selected_idx:
        ps = []
        for a in par_values:
            p0 = np.array(a)[i]
            p = np.round(p0, 2)
            ps.append(p)
        inds.append(ps)
    # print(selected_idx)
    # print(inds)
    pars = np.array(inds).T
    # print(pars)
    if mutate:
        mutation_ratio = 0.1
        exp_space = []
        if np.isnan(ranges).any():
            ranges = np.array([[np.min(p), np.max(p)] for p in pars])
        spaces = [np.abs(r[1] - r[0]) for r in ranges]
        # print(ranges)

        # print(pars)
        # print(spaces)
        # print(ranges)
        # print([len(pp) for pp in [pars, spaces, ranges]])
        for p, s, r in zip(pars, spaces, ranges):
            noisy_p = []
            for i in p:
                # if type(i) is np.ndarray :
                #     t = np.random.normal(loc=i[0], scale=mutation_ratio * s)
                #     t = np.array(float(np.round(np.clip(t, a_min=r[0], a_max=r[1]), 2)))
                # else :
                t = np.random.normal(loc=i, scale=mutation_ratio * s)
                t = float(np.round(np.clip(t, a_min=r[0], a_max=r[1]), 2))
                # print(i,t)
                noisy_p.append(t)
            # noisy_p = [float(np.round(np.clip(np.random.normal(loc=i, scale=mutation_ratio * s), a_min=r[0], a_max=r[1]),2)) for i in p]
            # print(noisy_p)
            if recombine:
                random.shuffle(noisy_p)
            # print(noisy_p)
            exp_space.append(noisy_p)
    else:
        exp_space = list(pars)
    best_pop = dict(zip(par_full_names, exp_space))
    # print(best_pop)
    return best_pop


def get_results(traj, res_names=None):
    par_values = []
    par_names = []
    par_full_names = []
    for i, p in enumerate(traj.f_get_explored_parameters()):
        par_values.append(traj.f_get(p).f_get_range())
        par_names.append(traj.f_get(p).v_name)
        par_full_names.append(traj.f_get(p).v_full_name)
    runs = traj.f_get_run_names(sort=True)
    runs_idx = [int(i) for i in traj.f_iter_runs(yields='idx')]
    if res_names is None:
        res_names = np.unique([traj.f_get(r).v_name for r in traj.f_get_results()])

    res_values = []
    for res in res_names:
        try:
            res_values.append([traj.f_get(run).f_get(res).f_get() for run in runs])
        except:
            res_names = [r for r in res_names if r != res]
    par_names = [str(p) for p in par_names]
    res_names = [str(p) for p in res_names]
    return runs_idx, runs, par_names, par_full_names, par_values, res_names, res_values


def save_results_df(traj, save_to=None, save_as='results.csv'):
    if save_to is None:
        save_to = traj.config.dir_path
    runs_idx, runs, par_names, par_full_names, par_values, res_names, res_values = get_results(traj, res_names=None)
    # print(runs_idx, runs, par_names, par_full_names, par_values, res_names, res_values)
    data = np.array(par_values + res_values).T
    cols = par_names + res_names
    df = pd.DataFrame(data, index=runs_idx, columns=cols)
    df.index.name = 'run_idx'
    file_path = os.path.join(save_to, save_as)
    try:
        fit_par, minimize = traj.config.fit_par, traj.config.minimize
        df.sort_values(by=fit_par, ascending=minimize, inplace=True)
    except:
        pass
    df.to_csv(file_path, index=True, header=True)
    return df


def save_results_dict(traj, save_to=None, save_as='results_dict.csv'):
    if save_to is None:
        save_to = traj.config.dir_path
    file_path = os.path.join(save_to, save_as)
    runs_idx, runs, par_names, par_full_names, par_values, res_names, res_values = get_results(traj, res_names=None)
    all = {}
    for idx, pvs, rvs in zip(runs_idx, par_values, res_values):
        all[idx] = {}
        for p, pv in zip(par_names, pvs):
            all[idx][p] = pv
        for r, rv in zip(res_names, rvs):
            all[idx][r] = list(rv)
    with open(file_path, "w") as fp:
        json.dump(all, fp)


def load_default_configuration(traj, sim_params=None, env_params=None, life_params=None, collections=[]):
    if sim_params is not None:
        env_dict = fun.flatten_dict(sim_params, parent_key='sim_params', sep='.')
        for k, v in env_dict.items():
            traj.f_aconf(k, v)

    if env_params is not None:
        env_dict = fun.flatten_dict(env_params, parent_key='env_params', sep='.')
        for k, v in env_dict.items():
            # print(k,v)
            traj.f_apar(k, v)
    # if larva_pars is not None:
    #     fly_dict = fun.flatten_dict(larva_pars, parent_key='larva_pars', sep='.')
    #     for k, v in fly_dict.items():
    #         traj.f_apar(k, v)

    if life_params is not None:
        life_dict = fun.flatten_dict(life_params, parent_key='life_params', sep='.')
        for k, v in life_dict.items():
            traj.f_apar(k, v)

    traj.f_aconf('collections', collections)
    return traj


def default_processing(traj, dataset=None):
    # print(dataset.endpoint_data)
    # raise
    fit_par = traj.config.fit_par
    try:
        fit = dataset.endpoint_data[fit_par].mean()
    except:
        fit = np.mean(dataset.step_data[fit_par].groupby('AgentID').mean())
    traj.f_add_result(fit_par, fit, comment='The fit')
    return dataset, fit


# def end_collection_processing(traj, dataset=None):
#     end_pars = traj.config.end_pars
#     for p in end_pars:
#         vs = dataset.endpoint_data[p].values
#         traj.f_add_result(p, vs, comment=f'Endpoint par {p}')
#     return dataset, None


def null_processing(traj, dataset=None):
    return dataset, np.nan


def deb_processing(traj, dataset=None):
    dataset.deb_analysis()
    deb_f_mean = dataset.endpoint_data['deb_f_mean'].mean()
    traj.f_add_result('deb_f_mean', deb_f_mean, comment='The average mean deb functional response')
    deb_f_mean_deviation = np.abs(dataset.endpoint_data['deb_f_mean'].mean() - 1)
    traj.f_add_result('deb_f_mean_deviation', deb_f_mean_deviation,
                      comment='The deviation of average mean deb functional response from 1')
    hunger = dataset.endpoint_data['hunger'].mean()
    traj.f_add_result('hunger', hunger, comment='The average final hunger')
    reserve_density = dataset.endpoint_data['reserve_density'].mean()
    traj.f_add_result('reserve_density', reserve_density, comment='The average final reserve density')

    return dataset, np.nan


def null_post_processing(traj, result_tuple):
    traj.f_load(index=None, load_parameters=2, load_results=2)


def plot_results(traj, df):
    filepath = traj.config.dir_path
    runs_idx, runs, par_names, par_full_names, par_values, res_names, res_values = get_results(traj, res_names=None)
    kwargs = {'df': df,
              # 'labels': par_names + res_names,
              'save_to': filepath,
              # 'pref': None,
              'show': False}
    if len(res_names) == 1:
        if len(par_names) == 1:
            plot_2d(labels= par_names + res_names, pref=None, **kwargs)
        elif len(par_names) == 2:
            plot_3pars(labels= par_names + res_names,pref=None,**kwargs)
        elif len(par_names) >2 :
            for i,pair in enumerate(itertools.combinations(par_names, 2)) :
                plot_3pars(labels=list(pair) + res_names,pref=i, **kwargs)




def null_final_processing(traj):
    df = save_results_df(traj)
    plot_results(traj, df)
    # d = save_results_dict(traj)
    # print(df)
    return df


def end_scatter_generation(traj):
    d = save_results_df(traj)
    parent_dir = traj.config.dataset_path
    dirs = [f'{parent_dir}/{d}' for d in os.listdir(parent_dir)]
    dirs.sort()
    ds = [LarvaDataset(dir) for dir in dirs]
    kwargs = {'datasets': ds,
              'labels': [d.id for d in ds],
              'save_to': traj.config.dir_path}
    plot_endpoint_scatter(**kwargs, par_shorts=traj.config.end_parshorts_1)
    plot_endpoint_scatter(**kwargs, par_shorts=traj.config.end_parshorts_2)
    plot_endpoint_scatter(**kwargs, par_shorts=traj.config.end_parshorts_3)
    return d


def deb_analysis(traj):
    data_dir = traj.config.dataset_path
    parent_dir = traj.config.dir_path
    df = save_results_df(traj)
    runs_idx, runs, par_names, par_full_names, par_values, res_names, res_values = get_results(traj, res_names=None)
    if len(par_names) == 2:
        # z0s=[1.0,0.5,1.0]
        for i in range(len(res_names)):
            r = res_names[i]
            labels = par_names + [r]
            plot_3pars(df, labels, save_to=traj.config.dir_path, pref=r)
            # plot_3pars(df, labels, z0=z0s[i], save_to = traj.config.dir_path, pref=r)

    dirs = [f'{data_dir}/{dir}' for dir in os.listdir(data_dir)]
    dirs.sort()
    ds = [LarvaDataset(dir) for dir in dirs]
    if len(ds) == 1:
        new_ids = [None]
    else:
        if len(par_names) == 1:
            new_ids = [f'{par_names[0]} : {v}' for v in par_values[0]]
        else:
            new_ids = [d.id for d in ds]
        plot_endpoint_params(ds, new_ids, mode='deb', save_to=parent_dir)
    # print(new_ids,[d.id for d in ds])
    # raise
    deb_dicts = fun.flatten_list(
        [[deb_dict(d, id, new_id=new_id) for id in d.agent_ids] for d, new_id in zip(ds, new_ids)])
    plot_debs(deb_dicts=deb_dicts, save_to=parent_dir, save_as='deb_f.pdf', mode='f')
    plot_debs(deb_dicts=deb_dicts, save_to=parent_dir, save_as='deb.pdf')
    plot_debs(deb_dicts=deb_dicts, save_to=parent_dir, save_as='deb_minimal.pdf', mode='minimal')

    return df


def post_processing(traj, result_tuple):
    fit_par, minimize, threshold, max_Nsims, Nbest, ranges = traj.config.fit_par, traj.config.minimize, traj.config.threshold, traj.config.max_Nsims, traj.config.Nbest, traj.config.ranges
    traj.f_load(index=None, load_parameters=2, load_results=2)
    run_names = traj.f_get_run_names()
    Nruns = len(run_names)
    fits = []
    for run_name in run_names:
        fit = traj.res.runs.f_get(run_name).f_get(fit_par).f_get()
        fits.append(fit)
    if minimize:
        best = min(fits)
        best_idx = np.argmin(fits)
        thr_reached = best <= threshold
    else:
        best = max(fits)
        best_idx = np.argmax(fits)
        thr_reached = best >= threshold
    best_run_name = run_names[best_idx]
    print(f'Best result out of {Nruns} runs : {best} in run {best_run_name}')
    maxNreached = Nruns >= max_Nsims
    if not thr_reached and not maxNreached:
        space = get_best_individuals(traj, ranges=ranges, fit_par=fit_par,
                                     num_individuals=Nbest,
                                     minimize=minimize, mutate=True)
        traj.f_expand(space)
        print(f'Continuing expansion with another {Nbest} configurations')
    else:
        par_values = []
        par_names = []
        for i, p in enumerate(traj.f_get_explored_parameters()):
            par_values.append(traj.f_get(p).f_get_range())
            par_names.append(traj.f_get(p).v_name)
        best_config = {}
        for l, p in zip(par_values, par_names):
            best_config.update({p: l[best_idx]})
        if maxNreached:
            print(f'Maximum number of simulations reached. Halting search')
        else:
            print(f'Best result reached threshold. Halting search')
        print(f'Best configuration is {best_config} with result {best}')
    try:
        traj.f_remove_items(['best_run_name'])
    except:
        pass
    traj.f_add_result('best_run_name', best_run_name, comment=f'The run with the best result')
    traj.f_store()


def single_run(traj, process_method=None, save_data_in_hdf5=True, save_data_flag=False, **kwargs):
    start = time.time()
    env_params = fun.reconstruct_dict(traj.f_get('env_params'))
    sim_params = fun.reconstruct_dict(traj.f_get('sim_params'))
    life_params = fun.reconstruct_dict(traj.f_get('life_params'))

    sim_params['sim_id'] = f'run_{traj.v_idx}'

    d = run_sim(
        env_params=env_params,
        sim_params=sim_params,
        life_params=life_params,
        collections=traj.collections,
        mode=None,
        save_data_flag=save_data_flag,
        **kwargs)

    if process_method is None:
        results = np.nan
    else:
        d, results = process_method(traj, d)

    # FIXME  For some reason the multiindex dataframe cannot be saved as it is.
    #  So I have to drop a level (and it does not work if I add it again).
    #  Also if I drop the AgentID, it has a problem because it does not find Step 0
    if save_data_in_hdf5 == True:
        temp = d.step_data.reset_index(level='Step', drop=False, inplace=False)
        e = ObjectTable(data=d.endpoint_data, index=d.endpoint_data.index, columns=d.endpoint_data.columns.values,
                        copy=True)
        s = ObjectTable(data=temp, index=temp.index, columns=temp.columns.values, copy=True)
        traj.f_add_result('endpoint_data', endpoint_data=e, comment='The simulation endpoint data')
        traj.f_add_result('step_data', step_data=s, comment='The simulation step-by-step data')
    end = time.time()
    print(f'Single run {traj.v_idx} complete in {end - start} seconds')
    return d, results


def batch_run(*args, **kwargs):
    return _batch_run(*args, **kwargs)


def _batch_run(dir='unnamed',
               batch_id='template',
               space=None,
               save_data_in_hdf5=False,
               single_method=single_run,
               process_method=null_processing,
               post_process_method=None,
               final_process_method=None,
               multiprocessing=True,
               resumable=True,
               overwrite=False,
               sim_config=None,
               params=None,
               optimization=None,
               post_kwargs={},
               run_kwargs={}
               ):
    saved_args = locals()
    traj_name = f'{batch_id}_traj'
    parent_dir_path = f'{paths.BatchRunFolder}/{dir}'
    dir_path = os.path.join(parent_dir_path, batch_id)

    filename = f'{dir_path}/{batch_id}.hdf5'
    build_new = True
    if os.path.exists(parent_dir_path) and os.path.exists(dir_path) and overwrite == False:
        build_new = False
        try:
            # print('Trying to resume existing trajectory')
            env = Environment(continuable=True)
            env.resume(trajectory_name=traj_name, resume_folder=dir_path)
            print('Resumed existing trajectory')
            build_new = False
        except:
            try:
                # print('Trying to load existing trajectory')
                traj = load_trajectory(filename=filename, name=traj_name, load_all=0)
                env = Environment(trajectory=traj, multiproc=True, ncores=4)

                traj = config_traj(traj, optimization)

                traj.f_load(index=None, load_parameters=2, load_results=0)
                traj.f_expand(space)
                print('Loaded existing trajectory')
                build_new = False
            except:
                print('Neither of resuming or expanding of existing trajectory worked')

    if build_new:
        if multiprocessing:
            multiproc = True
            resumable = False
            wrap_mode = pypetconstants.WRAP_MODE_QUEUE
        else:
            multiproc = False
            resumable = True
            wrap_mode = pypetconstants.WRAP_MODE_LOCK
        # print('Trying to create novel environment')
        env = Environment(trajectory=traj_name,
                          filename=filename,
                          file_title=batch_id,
                          comment=f'{batch_id} batch run!',
                          large_overview_tables=True,
                          overwrite_file=True,
                          resumable=False,
                          resume_folder=dir_path,
                          multiproc=multiproc,
                          ncores=4,
                          use_pool=True,  # Our runs are inexpensive we can get rid of overhead by using a pool
                          freeze_input=True,  # We can avoid some overhead by freezing the input to the pool
                          wrap_mode=wrap_mode,
                          graceful_exit=True)
        print('Created novel environment')
        traj = prepare_traj(env.traj, sim_config, params, batch_id, parent_dir_path, dir_path)
        traj = config_traj(traj, optimization)
        traj.f_explore(space)

    if post_process_method is not None:
        env.add_postprocessing(post_process_method, **post_kwargs)
    env.run(single_method, process_method, save_data_in_hdf5=save_data_in_hdf5, save_to=dir_path,
            **run_kwargs)
    env.disable_logging()
    print('Batch run complete')
    if final_process_method is not None:
        results= final_process_method(env.traj)
        # print(results)
        return results


def config_traj(traj, optimization):
    if optimization is not None:
        for k, v in optimization.items():
            # print(k,v)
            traj.f_aconf(k, v)
        # raise
    return traj


def prepare_traj(traj, sim_config, params, batch_id, parent_dir_path, dir_path):
    env_params, sim_params, life_params, collections = sim_config['env_params'], sim_config[
        'sim_params'], sim_config['life_params'], sim_config['collections']

    traj = load_default_configuration(traj, sim_params=sim_params, env_params=env_params,
                                      life_params=life_params, collections=collections)
    if params is not None:
        for p in params:
            traj.f_apar(p, 0.0)

    plot_path = os.path.join(dir_path, f'{batch_id}.pdf')
    data_path = os.path.join(dir_path, f'{batch_id}.csv')

    traj.f_aconf('parent_dir_path', parent_dir_path, comment='Parent directory')
    traj.f_aconf('dir_path', dir_path, comment='Directory for saving data')
    traj.f_aconf('plot_path', plot_path, comment='File for saving plot')
    traj.f_aconf('data_path', data_path, comment='File for saving data')
    traj.f_aconf('dataset_path', f'{dir_path}/{batch_id}', comment='Directory for saving datasets')
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
            range = np.linspace(low, high, s).tolist()
            values_dict.update({par: range})
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


if __name__ == '__main__':
    # This will execute the main function in case the script is called from the one true
    # main process and not from a child processes spawned by your environment.
    # Necessary for multiprocessing under Windows.
    batch_id = 'template'
    batch_idx = 0
    space = cartesian_product({'larva_pars.sensorimotor_params.torque_coef': [0.08, 0.09],
                               'larva_pars.sensorimotor_params.ang_damping': [0.6, 0.3]})
    batch_run(batch_id=batch_id, batch_idx=batch_idx, space=space)


def PI_computation(traj, dataset):
    arena_xdim_in_mm = traj.parameters.env_params.arena_params.arena_xdim * 1000
    ind = dataset.compute_preference_index(arena_diameter_in_mm=arena_xdim_in_mm)
    traj.f_add_result('PI', ind, comment=f'The preference index')
    return dataset, ind


def heat_map_generation(traj):
    path = traj.config.dir_path
    csv_filepath = f'{path}/PIs.csv'
    runs_idx, runs, par_names, par_full_names, par_values, res_names, res_values = get_results(traj, res_names=['PI'])
    inds = res_values[0]
    Lgains = np.array(par_values[0]).astype(int)
    Rgains = np.array(par_values[1]).astype(int)
    left_gain = pd.Series(np.unique(Lgains), name="left_gain")
    right_gain = pd.Series(np.unique(Rgains), name="right_gain")
    df = pd.DataFrame(index=left_gain, columns=right_gain, dtype=float)
    for Lgain, Rgain, ind in zip(Lgains, Rgains, inds):
        df[Rgain].loc[Lgain] = ind
    df.to_csv(csv_filepath, index=True, header=True)
    plot_heatmap_PI(save_to=traj.config.dir_path, csv_filepath=csv_filepath)

#
# def generate_gain_space(pars, ranges, Ngrid, values=None):
#     if len(pars) != 1 or len(Ngrid) != 1 or len(ranges) != 1:
#         raise ValueError('There must be a single parameter, range and space step')
#     r, s = ranges[0], Ngrid[0]
#     if values is None:
#         values = np.linspace(r[0], r[1], s)
#     values = [fun.flatten_list([[[a, b] for a in values] for b in values])]
#     values_dict = dict(zip(pars, values))
#     space = cartesian_product(values_dict)
#     return space
