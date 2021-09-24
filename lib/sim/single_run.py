""" Run a simulation and save the parameters and data to files."""
import copy
import datetime
import time
import pickle
import os
import numpy as np
from operator import attrgetter

import lib.conf.data_conf as dat

from lib.aux.collecting import output_dict, midline_xy_pars
from lib.envs._larvaworld_sim import LarvaWorldSim
from lib.conf.conf import loadConf, next_idx, expandConf
import lib.aux.functions as fun
from lib.stor.larva_dataset import LarvaDataset
from lib.stor import paths
import lib.conf.dtype_dicts as dtypes


def _run_sim(
        sim_params,
        env_params,
        life_params,
        enrichment,
        collections,
        # vis_kwargs=None,
        save_to=None,
        save_data_flag=True,
        # experiment=None,
        par_config=dat.SimParConf,
        seed=None,
        **kwargs):
    # print(vis_kwargs)
    np.random.seed(seed)
    id = sim_params['sim_ID']
    dt = sim_params['timestep']
    Nsec = sim_params['duration'] * 60
    path = sim_params['path']
    Box2D = sim_params['Box2D']
    sample = sim_params['sample']

    if save_to is None:
        save_to = paths.SimFolder
    if path is not None:
        save_to = os.path.join(save_to, path)
    dir_path = os.path.join(save_to, id)

    # Store the parameters so that we can save them in the results folder
    sim_date = datetime.datetime.now()
    param_dict = locals()
    start = time.time()
    Nsteps = int(Nsec / dt)

    try :
        # FIXME This only takes the first configuration into account
        Npoints = list(env_params['larva_groups'].values())[0]['model']['body']['Nsegs'] + 1
    except :
        Npoints=3
    d = LarvaDataset(dir=dir_path, id=id, fr=1 / dt,
                     Npoints=Npoints, Ncontour=0, sample_dataset=sample,env_params=env_params,
                     par_conf=par_config, save_data_flag=save_data_flag, load_data=False,
                     life_params=life_params
                     )

    output = collection_conf(dataset=d, collections=collections)
    env = LarvaWorldSim(id=id, dt=dt, Box2D=Box2D, sample_dataset=sample,
                        env_params=env_params, output=output,
                        life_params=life_params, Nsteps=Nsteps,
                        save_to=d.vis_dir, **kwargs)
    print()
    print(f'---- Simulation {id} ----')
    # Run the simulation
    completed = env.run()
    print()
    if not completed:
        d.delete()
        print('    Simulation aborted!')
        res = None
    else:
        end = time.time()
        dur = end - start
        param_dict['duration'] = np.round(dur, 2)
        print(f'    Simulation completed in {np.round(dur).astype(int)} seconds!')
        res = store_sim_data(env, d, save_data_flag, enrichment, param_dict)
    env.close()
    return res


ser = pickle.dumps(_run_sim)
run_sim = pickle.loads(ser)


def store_sim_data(env, d, save_data_flag, enrichment, param_dict):
    # Read the data collected during the simulation
    if env.larva_step_col is not None:
        step = env.larva_step_col.get_agent_vars_dataframe()
    else:
        step = None
    if env.larva_end_col is not None:
        env.larva_end_col.collect(env)
        end = env.larva_end_col.get_agent_vars_dataframe().droplevel('Step')
    else:
        end = None
    if env.food_end_col is not None:
        env.food_end_col.collect(env)
        food = env.food_end_col.get_agent_vars_dataframe().droplevel('Step')
    else:
        food = None

    d.set_data(step=step,end=end,food=food)

    d.enrich(**enrichment, is_last=False)
    # Save simulation data and parameters
    if save_data_flag:
        d.save()
        d.save_dicts(env)
        fun.dict_to_file(param_dict, d.dir_dict['sim'])
        if env.table_collector is not None:
            d.save_tables(env.table_collector.tables)

            # except:
            #     pass
    return d


def collection_conf(dataset, collections):
    if not paths.new_format:
        if collections is None:
            collections = ['pose']
        cd = output_dict
        d = dataset
        step_pars = []
        end_pars = []
        tables = {}
        for c in collections:
            if c == 'midline':
                step_pars += list(midline_xy_pars(N=d.Nsegs).keys())
            elif c == 'contour':
                step_pars += fun.flatten_list(d.contour_xy)
            else:
                step_pars += cd[c]['step']
                end_pars += cd[c]['endpoint']
                if 'tables' in list(cd[c].keys()):
                    tables.update(cd[c]['tables'])
        step = fun.unique_list(step_pars)
        end = fun.unique_list(end_pars)
        output = {'step': step,
                  'end': end,
                  'tables': tables,
                  'step_groups': [],
                  'end_groups': [],
                  }

    else:
        cd = combo_collection_dict
        cs = [cd[c] for c in collections if c in cd.keys()]
        output = {'step': [],
                  'end': [],
                  'tables': {},
                  'step_groups': fun.flatten_list([c['step'] for c in cs]),
                  'end_groups': fun.flatten_list([c['end'] for c in cs])}
    return output


def load_reference_dataset(dataset_id='reference', load=False):
    d = LarvaDataset(dir=f'{paths.RefFolder}/{dataset_id}', load_data=load)
    if not load:
        d.load(step=False)
    return d


def get_exp_conf(exp_type, sim_params, life_params=None, N=None, larva_model=None):
    conf = copy.deepcopy(expandConf(exp_type, 'Exp'))

    for k in list(conf['env_params']['larva_groups'].keys()):
        if N is not None:
            conf['env_params']['larva_groups'][k]['N'] = N
        if larva_model is not None:
            conf['env_params']['larva_groups'][k]['model'] = loadConf(larva_model, 'Model')
    if life_params is not None:
        conf['life_params'] = life_params

    if sim_params['sim_ID'] is None:
        idx = next_idx(exp_type)
        sim_params['sim_ID'] = f'{exp_type}_{idx}'
    if sim_params['path'] is None:
        sim_params['path'] = f'single_runs/{exp_type}'
    if sim_params['duration'] is None:
        sim_params['duration'] = conf['sim_params']['duration']
    conf['sim_params'] = sim_params
    conf['experiment'] = exp_type
    return conf


def run_essay(id,path, exp_types,durations, vis_kwargs, **kwargs):
    ds = []
    for i, (exp_type, dur) in enumerate(zip(exp_types, durations)):
        sim=dtypes.get_dict('sim_params', duration=dur, sim_ID=f'{id}_{i}', path=path)
        conf = get_exp_conf(exp_type = exp_type, sim_params=sim, **kwargs)
        d = run_sim(**conf, vis_kwargs=vis_kwargs)
        ds.append(d)
    return ds

# def mimic_dataset(dataset=None, dir=None, idx=0, model='imitation', exp='dish', group_id='Mockers', default_color='black', **kwargs):
#     d=dataset
#     if d is None :
#         d=LarvaDataset(dir)
#
#
#     def imitate_group(d, model) :
#         s, e, c = d.step_data, d.endpoint_data, d.config
#         xy=s[nam.xy(c['point'])].dropna().groupby('AgentID').first()
#         os=np.deg2rad(s['front_orientation'].dropna().groupby('AgentID').first())
#         larva_conf=expandConf(model, 'Model')
#
#         try:
#             dic=loadConf(d.id, 'Ref')
#         except:
#             create_reference_dataset(d.config, dataset_id=d.id, Nstd=3, overwrite=False)
#             dic = loadConf(d.id, 'Ref')
#
#         if larva_conf['brain']['intermitter_params']:
#             for bout, dist in zip(['pause', 'stride'], ['pause_dist', 'stridechain_dist']):
#                 larva_conf['brain']['intermitter_params'][dist] = dic[bout]['best']
#
#
#         group_pars=[]
#         # print(os)
#         for id in d.agent_ids :
#             pars=copy.deepcopy(larva_conf)
#             pars['body']['initial_length']=e['length'].loc[id]
#             pars['body']['length_std']=0.0
#             pars['body']['Nsegs']=2
#             pars['body']['seg_ratio']=None
#             pars['brain']['crawler_params']['initial_freq']=e[nam.scal(nam.freq('velocity'))].loc[id]
#             pars['brain']['crawler_params']['step_to_length_mu']=e[nam.scal(nam.std(nam.dst('stride')))].loc[id]
#             pars['brain']['crawler_params']['step_to_length_std']=e[nam.scal(nam.mean(nam.dst('stride')))].loc[id]
#             # print(pars['brain']['crawler_params'])
#
#
#
#             kws={
#                 # 'position' : (0.0,0.0),
#                 'position' : tuple(xy.loc[id].values),
#                 'orientation' : os.loc[id],
#                 'id' : id,
#                 'group' : group_id,
#                 'default_color' : default_color,
#                 'pars' : pars}
#             # print(kws['position'])
#             # print(kws['orientation'])
#             group_pars.append(kws)
#         return {group_id : group_pars}
#
#     def imitate_exp(d, model, exp, **kwargs) :
#         s, e, c = d.step_data, d.endpoint_data, d.config
#
#         sim_params = {
#             'timestep': c['dt'],
#             'duration': e['cum_dur'].mean() / 60,
#             'path': 'single_runs/imitation',
#             'sim_ID': f'{d.id}_imitation_{idx}',
#             'sample': d.id,
#             'Box2D': False
#         }
#         exp_conf = expandConf(exp, 'Exp')
#         exp_conf['env_params']['larva_groups']=imitate_group(d, model)
#         exp_conf['env_params']['arena']=c['arena_pars']
#         # exp_conf['env_params']['arena']['arena_dims']=(100.0,100.0)
#         exp_conf['sim_params']=sim_params
#
#         return exp_conf
#
#     exp_conf=imitate_exp(d, model, exp)
#     exp_conf['experiment'] = exp
#     exp_conf['save_data_flag'] = True
#     exp_conf.update(kwargs)
#     dd= run_sim(**exp_conf)
#     from lib.sim.analysis import sim_analysis
#     fig_dict, results = sim_analysis(dd, exp)
#     return fig_dict, results


combo_collection_dict = {
    'pose': {'step': ['basic', 'bouts', 'spatial', 'angular'], 'end': ['e_basic', 'e_dispersion']},
    'source_vincinity': {'step': ['chemorbit'], 'end': ['e_chemorbit']},
    'source_approach': {'step': ['chemotax'], 'end': ['e_chemotax']},
    'olfactor': {'step': ['odors', 'olfactor'], 'end': []},
}