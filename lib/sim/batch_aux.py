import itertools
import os
import random
import numpy as np
import pandas as pd
import pypet



from lib import reg, aux

from lib.plot.hist import plot_endpoint_params, plot_endpoint_scatter
from lib.plot.deb import plot_debs
from lib.plot.scape import plot_3pars, plot_heatmap_PI, plot_2d
from lib.sim.single_run import SingleRun
from lib.process.larva_dataset import LarvaDataset


def get_Nbest(traj, mutate=True, recombine=False):
    def mutate_value(v, range, scale=0.01):
        r0, r1 = range
        return np.clip(np.random.normal(loc=v, scale=scale * np.abs(r1 - r0)), a_min=r0, a_max=r1).astype(float)

    N = traj.config.Nbest
    df=traj_df(traj)
    p_n0s = df.columns[:-1]
    V0s = df[p_n0s].iloc[np.arange(N)].values.T
    if mutate:
        space = []
        for v0s, r in zip(V0s, traj.config.ranges):
            vs = mutate_value(v0s, r, scale=0.01)
            if recombine:
                random.shuffle(vs)
            vs=[float(v) for v in vs]
            space.append(vs)
    else:
        space = list(V0s)
    best_pop = dict(zip(p_n0s, space))
    return best_pop


def save_results_df(traj):
    df=traj_df(traj)
    pairs={traj.f_get(p).v_full_name : traj.f_get(p).v_name for p in traj.f_get_explored_parameters()}
    df.rename(pairs, axis=1, inplace=True)
    df.to_csv(os.path.join(traj.config.dir_path, 'results.csv'), index=True, header=True)
    try:
        for p in df.columns.values[:-1]:
            print(p, df[p].corr(df[traj.config.fit_par]))
    except:
        pass
    return df


def exp_fit_processing(traj, d, exp_fitter):
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
        raise ValueError('Could not retrieve fit parameter from dataset')

    ops = traj.config.operations
    if ops.abs:
        vals = np.abs(vals)
    if ops.mean:
        fit = np.mean(vals)
    elif ops.std:
        fit = np.std(vals)
    traj.f_add_result(p, fit, comment='The fit')
    return d, fit


def null_processing(traj, d=None):
    return d, np.nan


def deb_processing(traj, d=None):
    e = d.endpoint_data
    for p in ['deb_f_mean', 'deb_f_deviation_mean', 'hunger', 'reserve_density']:
        try:
            traj.f_add_result(p, e[p].mean())
        except:
            pass
    return d, np.nan


def null_post_processing(traj, result_tuple):
    traj.f_load(index=None, load_parameters=2, load_results=2)


def plot_results(traj, df):
    figs = {}
    p_ns = [traj.f_get(p).v_name for p in traj.f_get_explored_parameters()]
    r_ns = np.unique([traj.f_get(r).v_name for r in traj.f_get_results()])
    kws = {'df': df,
           'save_to': traj.config.dir_path,
           'show': False}
    for r_n in r_ns:
        if len(p_ns) == 1:
            figs[f'{p_ns[0]}VS{r_n}'] = plot_2d(labels=p_ns + [r_n], pref=r_n, **kws)
        elif len(p_ns) == 2:
            figs.update(plot_3pars(vars=p_ns, target=r_n, pref=r_n, **kws))
        elif len(p_ns) > 2:
            for i, pair in enumerate(itertools.combinations(p_ns, 2)):
                figs.update(plot_3pars(vars=list(pair), target=r_n, pref=f'{i}_{r_n}', **kws))
    return figs


def null_final_processing(traj):
    df = save_results_df(traj)
    try:
        plots = plot_results(traj, df)
    except:
        pass
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
        p1, p2 = par_shorts
        fig_dict[f'{p1}VS{p2}'] = plot_endpoint_scatter(**kwargs, keys=par_shorts)

        # fig_dict[f'{p1}VS{p2}'] = f
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
    deb_dicts = aux.flatten_list([d.load_dicts('deb') for d in ds])
    fig_dict = {}
    for m in ['energy', 'growth', 'full']:
        f = plot_debs(deb_dicts=deb_dicts, save_to=save_to, save_as=f'deb_{m}.pdf', mode=m)
        fig_dict[f'deb_{m}'] = f
    return df, fig_dict

def traj_df(traj):
    dics=[]
    for idx in traj.f_iter_runs(yields='idx'):
        dic={'idx' : idx, **traj.f_get_explored_parameters(fast_access=True), **traj.runs[idx].f_to_dict(short_names=True, fast_access=True)}
        dics.append(dic)
    df=pd.DataFrame.from_records(dics, index='idx')
    try:
        fit_par, minimize = traj.config.fit_par, traj.config.minimize
        df.sort_values(by=fit_par, ascending=minimize, inplace=True)
    except:
        pass
    return df



def post_processing(traj, result_tuple):
    traj.f_load(index=None, load_parameters=2, load_results=2)
    def threshold_reached(traj):
        fits = list(traj.f_get_from_runs(traj.config.fit_par, use_indices=True, fast_access=True).values())
        if traj.config.minimize:
            return np.nanmin(fits) <= traj.config.threshold
        else:
            return np.nanmax(fits) >= traj.config.threshold
    if len(traj) >= traj.config.max_Nsims:
        print(f'Maximum number of simulations reached. Halting search')
    elif threshold_reached(traj):
        print(f'Best result reached threshold. Halting search')
    else:
        traj.f_expand(get_Nbest(traj))
    traj.f_store()


def reconstruct_dict(param_group, **kwargs):


    dict = {}
    for p in param_group:
        if type(p) == pypet.ParameterGroup:
            d = reconstruct_dict(p)
            dict.update({p.v_name: d})
        elif type(p) == pypet.Parameter:
            if p.f_is_empty():
                dict.update({p.v_name: None})
            else:
                v = p.f_get()
                if v == 'empty_dict':
                    v = {}
                dict.update({p.v_name: v})
    dict.update(**kwargs)
    return dict


def retrieve_exp_conf(traj):
    d={}
    for k0 in ['env_params', 'sim_params', 'trials', 'larva_groups']:
        kws={'sim_ID':f'run_{traj.v_idx}', 'path':traj.config.dataset_path,'store_data':False} if k0=='sim_params' else {}
        try :
            c=traj.f_get(k0)
            d[k0]=reconstruct_dict(c, **kws)
        except:
            d[k0]={}
    return d


def single_run(traj, procfunc=None, save_hdf5=True, exp_kws={}, proc_kws={}):
    with aux.suppress_stdout(False):
        ds = SingleRun(**retrieve_exp_conf(traj), **exp_kws).run()
        if procfunc is None:
            results = np.nan
        else:
            if len(ds) == 1:
                d = ds[0]
                d, results = procfunc(traj, d, **proc_kws)
            else:
                raise ValueError(
                    f'Splitting resulting dataset yielded {len(ds)} datasets but the batch-run is configured for a single one.')

    if save_hdf5:
        from pypet import ObjectTable
        s, e = [ObjectTable(data=k, index=k.index, columns=k.columns.values, copy=True) for k in
                [d.step_data.reset_index(level='Step'), d.endpoint_data]]
        traj.f_add_result('end', endpoint_data=e, comment='The simulation endpoint data')
        traj.f_add_result('step', step_data=s, comment='The simulation step-by-step data')
    return d, results


def PI_computation(traj, dataset):
    ind = dataset.config['PI']['PI']
    traj.f_add_result('PI', ind, comment=f'The preference index')
    try :
        ind2 = dataset.config['PI2']
        traj.f_add_result('PI2', ind2, comment=f'The preference index, version 2')
    except :
        pass
    return dataset, ind


def heat_map_generation(traj):
    # df0 = save_results_df(traj)
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
    try:
        csv2_filepath = f'{traj.config.dir_path}/PI2s.csv'
        PI2s = [traj.f_get(run).f_get('PI2').f_get() for run in traj.f_get_run_names(sort=True)]
        df2 = pd.DataFrame(index=Lgain_range, columns=Rgain_range, dtype=float)
        for Lgain, Rgain, PI2 in zip(Lgains, Rgains, PI2s):
            df2[Rgain].loc[Lgain] = PI2
        df2.to_csv(csv2_filepath, index=True, header=True)
        fig2 = plot_heatmap_PI(save_to=traj.config.dir_path, csv_filepath=csv2_filepath, save_as='PI2_heatmap.pdf')
        fig_dict = {'PI2_heatmap': fig2}
    except :
        pass
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


def batch_method_unpack(run='default', post='default', final='null'):
    return {'procfunc': procfunc_dict[run],
            'postfunc': postfunc_dict[post],
            'finfunc': finfunc_dict[final], }


def load_traj(batch_type, batch_id):

    parent_dir_path = f'{reg.BATCH_DIR}/{batch_type}'
    filename = f'{parent_dir_path}/{batch_type}.hdf5'
    traj = pypet.load_trajectory(filename=filename, name=batch_id, load_all=2)
    return traj


def retrieve_results(batch_type, batch_id):
    traj = load_traj(batch_type, batch_id)
    func = finfunc_dict[traj.config.batch_methods.final]
    return func(traj)


def delete_traj(batch_type, traj_name):
    import h5py
    filename = f'{reg.BATCH_DIR}/{batch_type}/{batch_type}.hdf5'
    with h5py.File(filename, 'r+') as f:
        del f[traj_name]
