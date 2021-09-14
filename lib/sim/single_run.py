""" Run a simulation and save the parameters and data to files."""
import copy
import datetime
import time
import pickle
from operator import attrgetter

import lib.conf.data_conf as dat
from lib.anal.plotting import *

from lib.aux.collecting import output_dict, midline_xy_pars
from lib.conf.par import combo_collection_dict
from lib.envs._larvaworld import LarvaWorldSim
from lib.conf.conf import loadConf, next_idx, expandConf
from lib.stor.larva_dataset import LarvaDataset
import lib.conf.dtype_dicts as dtypes


def run_sim_basic(
        sim_params,
        env_params,
        vis_kwargs,
        life_params,
        enrichment,
        collections,
        save_to=None,
        save_data_flag=True,
        experiment=None,
        par_config=dat.SimParConf,
        seed=1,
        **kwargs):

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

    # FIXME This only takes the first configuration into account
    Npoints = list(env_params['larva_groups'].values())[0]['model']['body']['Nsegs'] + 1
    d = LarvaDataset(dir=dir_path, id=id, fr=1 / dt,
                     Npoints=Npoints, Ncontour=0, sample_dataset=sample,
                     arena_pars=env_params['arena'],
                     par_conf=par_config, save_data_flag=save_data_flag, load_data=False,
                     life_params=life_params
                     )

    output = collection_conf(dataset=d, collections=collections)
    env = LarvaWorldSim(id=id, dt=dt, Box2D=Box2D, sample_dataset=sample,
                        env_params=env_params, output=output,
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
        res = store_sim_data(env, d, save_data_flag, enrichment, param_dict)
    env.close()
    return res


ser = pickle.dumps(run_sim_basic)
run_sim = pickle.loads(ser)


def store_sim_data(env, d, save_data_flag, enrichment, param_dict):
    # Read the data collected during the simulation
    if not paths.new_format:
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

    else:
        # new format
        save_to = d.dir_dict['table'] if save_data_flag else None
        df = env.step_group_collector.save(save_to=save_to)
        env.end_group_collector.collect(df=df)
        df0 = env.end_group_collector.save(save_to=save_to)
        if df0 is not None:
            df0 = df0.droplevel('Step')
        d.set_data(step=df, end=df0)

    d.enrich(**enrichment)
    # Save simulation data and parameters
    if save_data_flag:
        d.save()
        fun.dict_to_file(param_dict, d.dir_dict['sim'])
        for l in env.get_flies():
            if hasattr(l, 'deb') and l.deb is not None:
                l.deb.finalize_dict(d.dir_dict['deb'])
            if l.brain.intermitter is not None:
                l.brain.intermitter.save_dict(d.dir_dict['bouts'])

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


