import itertools
import os
import random

import numpy as np
import pandas as pd
from pypet import ObjectTable

from lib.process.aux import suppress_stdout
from lib.aux.dictsNlists import reconstruct_dict
import lib.aux.sim_aux
from lib.anal.plotting import plot_2d, plot_3pars, plot_endpoint_scatter, plot_endpoint_params, plot_debs, \
    plot_heatmap_PI
from lib.sim.batch.aux import grid_search_dict, load_traj
from lib.sim.single.single_run import SingleRun
from lib.stor.larva_dataset import LarvaDataset


def prepare_batch(batch, **kwargs):
    # print(batch.keys())
    space = grid_search_dict(batch['space_search'])
    if batch['optimization'] is not None:
        batch['optimization']['ranges'] = np.array(
            [batch['space_search'][k]['range'] for k in batch['space_search'].keys()])
    prepared_batch = {
        'batch_type': batch['batch_type'],
        'batch_id': batch['batch_id'],
        'exp': batch['exp'],
        'space': space,
        'batch_methods': batch['batch_methods'],
        **batch_methods(**batch['batch_methods']),
        'optimization': batch['optimization'],
        'exp_kws': batch['exp_kws'],
        'post_kws': {},
        'proc_kws': batch['proc_kws'],
        **kwargs
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
            vs = [lib.aux.sim_aux.mutate_value(v0, r, scale=0.1) for v0 in v0s]
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


def exp_fit_processing(traj, d, exp_fitter):
    from lib.anal.comparing import ExpFitter
    p = traj.config.fit_par
    fit = exp_fitter.compare(d)
    traj.f_add_result(p, fit, comment='The fit')
    return d, fit


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
    deb_dicts = lib.aux.dictsNlists.flatten_list([d.load_deb_dicts() for d in ds])
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


def single_run(traj, procfunc=None, save_hdf5=True, exp_kws={}, proc_kws={}):
    with suppress_stdout(True):
        try :
            trials=reconstruct_dict(traj.f_get('trials'))
        except :
            trials={}
        ds = SingleRun(
            env_params=reconstruct_dict(traj.f_get('env_params')),
            sim_params=reconstruct_dict(traj.f_get('sim_params'),
                                        sim_ID=f'run_{traj.v_idx}', path=traj.config.dataset_path,
                                        save_data=False),
            trials=trials,
            larva_groups=reconstruct_dict(traj.f_get('larva_groups')),
            **exp_kws).run()

        if procfunc is None:
            results = np.nan
        else:
            if len(ds)==1:
                d=ds[0]
                d, results = procfunc(traj, d, **proc_kws)
            else :
                raise ValueError (f'Splitting resulting dataset yielded {len(ds)} datasets but the batch-run is configured for a single one.')

    if save_hdf5:
        s, e = [ObjectTable(data=k, index=k.index, columns=k.columns.values, copy=True) for k in
                [d.step_data.reset_index(level='Step'), d.endpoint_data]]
        traj.f_add_result('end', endpoint_data=e, comment='The simulation endpoint data')
        traj.f_add_result('step', step_data=s, comment='The simulation step-by-step data')
    return d, results


def PI_computation(traj, dataset):
    ind = dataset.config['PI']['PI']
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


procfunc_dict = {
    'null': null_processing,
    'default': default_processing,
    'deb': deb_processing,
    'odor_preference': PI_computation,
    'exp_fit': exp_fit_processing
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


def retrieve_results(batch_type, batch_id):
    traj = load_traj(batch_type, batch_id)
    func = finfunc_dict[traj.config.batch_methods.final]
    return func(traj)
