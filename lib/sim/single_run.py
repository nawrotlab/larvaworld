""" Run a simulation and save the parameters and data to files."""
import datetime
import time
import pickle

import lib.conf.data_conf as dat
from lib.anal.plotting import *
from lib.aux.collecting import output, midline_xy_pars
from lib.conf.par import build_par_dict, Collection, AgentCollector, GroupCollector
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
    if collections is None:
        collections = ['pose']
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
    pargroup_names = []
    # pargroup_names = ['stride']
    # pargroup_names = ['pose', 'angular', 'dispersion']
    par_dict = build_par_dict(dt=env.dt)
    env.group_collectors = [GroupCollector(objects=env.get_flies(), name=n, par_dict=par_dict, common=True,
                                           save_to=d.dir_dict['table']) for n in pargroup_names]
    # save_to=f'{d.table_dir}/{n}') for n in pargroup_names]
    # Prepare the odor layer for a number of timesteps
    # odor_prep_time = 0.0
    # larva_prep_time = 0.5
    # env.prepare_odor_layer(int(odor_prep_time * 60 / env.dt))
    # Prepare the flies for a number of timesteps
    # env.prepare_flies(int(larva_prep_time * 60 / env.dt))
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
        # Read the data collected during the simulation
        env.larva_end_col.collect(env)
        env.food_end_col.collect(env)

        d.set_data(step=env.larva_step_col.get_agent_vars_dataframe(),
                   end=env.larva_end_col.get_agent_vars_dataframe().droplevel('Step'),
                   food=env.food_end_col.get_agent_vars_dataframe().droplevel('Step'))

        end = time.time()
        dur = end - start
        param_dict['duration'] = np.round(dur, 2)

        print(f'    Simulation completed in {np.round(dur).astype(int)} seconds!')
        # Save simulation data and parameters
        if save_data_flag:
            if enrich and experiment is not None:
                d = sim_enrichment(d, experiment)
            d.save()
            if env.table_collector is not None:
                d.save_tables(env.table_collector.tables)
            fun.dict_to_file(param_dict, d.dir_dict['sim'])
            for l in env.get_flies():
                try:
                    l.deb.save_dict(d.dir_dict['deb'])
                except:
                    pass
            for c in env.group_collectors:
                c.save()
        res = d
    env.close()
    # k=res.load_table('pose')
    # k['dst']=k['d']
    #
    # # print([par_dict[v].unit for v in k.columns])
    # print(k.columns)
    # res.detect_strides(table_name='stride')
    # print(k.columns)
    # k = res.load_table('stride')
    # print(k.columns)
    return res


ser = pickle.dumps(run_sim_basic)
run_sim = pickle.loads(ser)


def collection_conf(dataset, collections):
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
            step_pars += output[c]['step']
            end_pars += output[c]['endpoint']
        if 'tables' in list(output[c].keys()):
            tables.update(output[c]['tables'])

    collected_pars = {'step': fun.unique_list(step_pars),
                      'endpoint': fun.unique_list(end_pars),
                      'tables': tables}
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
