import numpy as np
import pandas as pd
from pypet import cartesian_product, load_trajectory

import lib.aux.dictsNlists
from lib.aux import colsNstr as fun
from lib.stor import paths as paths


def load_default_configuration(traj, exp):
    for k0 in ['env_params', 'sim_params', 'life_params', 'enrichment']:
        dic = lib.aux.dictsNlists.flatten_dict(exp[k0], parent_key=k0, sep='.')
        for k, v in dic.items():
            if type(v) == list and type(v[0]) == list:
                v = np.array(v)
            traj.f_apar(k, v)
    return traj


def config_traj(traj, optimization, batch_methods):
    if optimization is not None:
        opt_dict = lib.aux.dictsNlists.flatten_dict(optimization, parent_key='optimization', sep='.')
        for k, v in opt_dict.items():
            traj.f_aconf(k, v)
    if batch_methods is not None:
        opt_dict = lib.aux.dictsNlists.flatten_dict(batch_methods, parent_key='batch_methods', sep='.')
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

def grid_search_dict(space_dict):
    dic={}
    for p, args in space_dict.items() :
        if args['values'] is not None :
            dic[p]=args['values']
        else :
            r0,r1=args['range']
            vs = np.linspace(r0, r1, args['Ngrid'])
            if type(r0) == int and type(r1) == int:
                vs = vs.astype(int)
            dic[p] = vs.tolist()
    return cartesian_product(dic)

# def grid_search_dict2(pars, ranges, Ngrid, values=None):
#     if values is not None:
#         dic = dict(zip(pars, values))
#     else:
#         Npars, Nsteps = len(pars), len(Ngrid)
#         if any([type(s) != int for s in Ngrid]):
#             raise ValueError('Parameter space steps are not integers')
#         if Npars != Nsteps:
#             if Nsteps == 1:
#                 Ngrid = [Ngrid[0]] * Npars
#                 print('Using the same step for all parameters')
#             else:
#                 raise ValueError('Number of parameter space steps does not match number of parameters and is not one')
#         if np.isnan(ranges).any():
#             raise ValueError('Ranges of parameters not provided')
#         dic = {}
#         for p, (r0, r1), s in zip(pars, ranges, Ngrid):
#             vs = np.linspace(r0, r1, s)
#             if type(r0) == int and type(r1) == int:
#                 vs = vs.astype(int)
#             dic[p]=vs.tolist()
#     return cartesian_product(dic)


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
            values = [a * Nv for a in values] + lib.aux.dictsNlists.flatten_list([[v] * Nspace for v in vs])
            returned_params += [p]

    space = dict(zip(returned_params, values))
    return space


def load_traj(batch_type, batch_id):
    parent_dir_path = f'{paths.BatchRunFolder}/{batch_type}'
    filename = f'{parent_dir_path}/{batch_type}.hdf5'
    traj = load_trajectory(filename=filename, name=batch_id, load_all=2)
    return traj



def stored_trajs(batch_type):
    import h5py
    filename = f'{paths.BatchRunFolder}/{batch_type}/{batch_type}.hdf5'
    try:
        f = h5py.File(filename, 'r')
        return {k:f for k in f.keys()}
    except:
        return {}


def delete_traj(batch_type, traj_name):
    import h5py
    filename = f'{paths.BatchRunFolder}/{batch_type}/{batch_type}.hdf5'
    with h5py.File(filename, 'r+') as f:
        del f[traj_name]


