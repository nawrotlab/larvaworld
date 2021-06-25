""" Run a simulation and save the parameters and data to files."""
import datetime
import time
import pickle
from operator import attrgetter

import lib.conf.data_conf as dat
from lib.anal.plotting import *

from lib.aux.collecting import output, midline_xy_pars
from lib.conf.par import build_par_dict, GroupCollector, collection_dict, combo_collection_dict
from lib.envs._larvaworld import LarvaWorldSim
from lib.conf.conf import loadConf, next_idx
from lib.sim.enrichment import sim_enrichment
from lib.stor.larva_dataset import LarvaDataset
import lib.conf.dtype_dicts as dtypes





def run_sim_basic(
        sim_params,
        env_params,
        vis_kwargs=dtypes.get_dict('visualization'),
        life_params=dtypes.get_dict('life'),
        collections=None,
        save_to=None,
        save_data_flag=True,
        enrich=False,
        experiment=None,
        par_config=dat.SimParConf,
        seed=1,
        **kwargs):
    # print(life_params)
    # print(sim_params['sim_dur'])


    np.random.seed(seed)
    id = sim_params['sim_id']
    dt = sim_params['dt']
    Nsec = sim_params['sim_dur'] * 60
    path = sim_params['path']
    Box2D = sim_params['Box2D']
    sample_dataset = sim_params['sample_dataset']

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

    # FIXME This only takes the first configuration into account
    Npoints = list(env_params['larva_params'].values())[0]['model']['body']['Nsegs'] + 1

    d = LarvaDataset(dir=dir_path, id=id, fr=1 / dt,
                     Npoints=Npoints, Ncontour=0, sample_dataset=sample_dataset,
                     arena_pars=env_params['arena_params'],
                     par_conf=par_config, save_data_flag=save_data_flag, load_data=False,
                     life_params=life_params
                     )

    collected_pars = collection_conf(dataset=d, collections=collections)
    env = LarvaWorldSim(id=id, dt=dt, Box2D=Box2D, sample_dataset=sample_dataset,
                        env_params=env_params, collected_pars=collected_pars,
                        life_params=life_params, Nsteps=Nsteps,
                        save_to=d.vis_dir, experiment=experiment,
                        **kwargs, vis_kwargs=vis_kwargs)
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
        res=store_sim_data(env, d, save_data_flag, enrich, experiment, param_dict)
    env.close()
    return res


ser = pickle.dumps(run_sim_basic)
run_sim = pickle.loads(ser)

def store_sim_data(env, d, save_data_flag, enrich, experiment, param_dict):
    # Read the data collected during the simulation
    if not paths.new_format :
        # old format
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
        if env.table_collector is not None:
            d.save_tables(env.table_collector.tables)
        d.set_data(step=step,
                   end=end,
                   food=food)

    else :
        # new format
        save_to = d.dir_dict['table'] if save_data_flag else None

        df = env.step_group_collector.save(save_to=save_to)

        # print(df.columns)

        env.end_group_collector.collect(df=df)
        df0 = env.end_group_collector.save(save_to=save_to)
        df0=df0.droplevel('Step')
        d.set_data(step=df, end=df0)
    if enrich and experiment is not None:
        d = sim_enrichment(d, experiment)
    # Save simulation data and parameters
    if save_data_flag:
        d.save()
        fun.dict_to_file(param_dict, d.dir_dict['sim'])
        for l in env.get_flies():
            try:
                l.deb.save_dict(d.dir_dict['deb'])
            except:
                pass
    return d


def collection_conf(dataset, collections):
    if not paths.new_format :
        if collections is None:
            collections = ['pose']
        cd=output
        d = dataset
        step_pars = []
        end_pars = []
        tables = {}
        # groups = []
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
            # if 'groups' in list(cd[c].keys()):
            #     groups += cd[c]['groups']

        collected_pars = {'step': fun.unique_list(step_pars),
                          'endpoint': fun.unique_list(end_pars),
                          'tables': tables,
                          'step_groups': [],
                          'end_groups': [],
                          }

    else :
        cd = combo_collection_dict
        cs=[cd[c] for c in collections if c in cd.keys()]
        collected_pars = {'step': [],
                          'endpoint': [],
                          'tables': {},
                          'step_groups': fun.flatten_list([c['step'] for c in cs]),
                          'end_groups': fun.flatten_list([c['end'] for c in cs])}
    return collected_pars


def load_reference_dataset(dataset_id='reference', load_data=False):
    reference_dataset = LarvaDataset(dir=f'{paths.RefFolder}/{dataset_id}', load_data=load_data)
    if not load_data:
        reference_dataset.load(step=False)
    return reference_dataset


def get_exp_conf(exp_type, sim_params, life_params=None, enrich=True, N=None, larva_model=None):
    exp_conf = loadConf(exp_type, 'Exp')
    env = exp_conf['env_params']
    if type(env) == str:
        env = loadConf(env, 'Env')

    if N is not None:
        for k in list(env['larva_params'].keys()):
            env['larva_params'][k]['N'] = N
    if larva_model is not None:
        for k in list(env['larva_params'].keys()):
            env['larva_params'][k]['model'] = larva_model

    for k, v in env['larva_params'].items():
        if type(v['model']) == str:
            v['model'] = loadConf(v['model'], 'Model')

    exp_conf['env_params'] = env
    if 'life_params' not in list(exp_conf.keys()):
        if life_params is None:
            life_params = dtypes.get_dict('life')
        exp_conf['life_params'] = life_params

    if sim_params['sim_id'] is None:
        idx = next_idx(exp_type)
        sim_params['sim_id'] = f'{exp_type}_{idx}'
    if sim_params['path'] is None:
        sim_params['path'] = f'single_runs/{exp_type}'
    if sim_params['sim_dur'] is None:
        sim_params['sim_dur'] = exp_conf['sim_params']['sim_dur']
    exp_conf['sim_params'] = sim_params
    exp_conf['experiment'] = exp_type
    exp_conf['enrich'] = enrich
    return exp_conf
