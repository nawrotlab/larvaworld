""" Run a simulation and save the parameters and data to files."""
import datetime
import time
import pickle
import os
import numpy as np

from lib.model.envs._larvaworld_sim import LarvaWorldSim

from lib.conf.base import paths
from lib.stor.larva_dataset import LarvaDataset


class SingleRun:
    def __init__(self,
                 sim_params,
                 env_params,
                 larva_groups,
                 enrichment,
                 collections,
                 experiment,
                 trials={},
                 save_to=None,
                 seed=None,
                 **kwargs):
        np.random.seed(seed)
        id = sim_params['sim_ID']
        # if id is None:
        #     from lib.conf.stored.conf import next_idx
        #     id = f'{experiment}_{next_idx(experiment)}'
        self.id = id
        dt = sim_params['timestep']
        path = sim_params['path']
        self.store_data = sim_params['store_data']
        # analysis = sim_params['analysis']
        self.enrichment = enrichment
        if save_to is None:
            save_to = paths.path("SIM")
        # if path is None:
        #     path = f'single_runs/{experiment}'
        save_to = os.path.join(save_to, path)
        dir_path = os.path.join(save_to, self.id)
        self.param_dict = locals()
        self.start = time.time()

        self.d = LarvaDataset(dir=dir_path, id=self.id, fr=1 / dt, Ncontour=0,
                              env_params=env_params, larva_groups=larva_groups, load_data=False)

        output = set_output(dataset=self.d, collections=collections)
        self.env = LarvaWorldSim(id=self.id, dt=dt, Box2D=sim_params['Box2D'],
                                 env_params=env_params,
                                 larva_groups=larva_groups,
                                 output=output,
                                 experiment=experiment,
                                 trials=trials,
                                 Nsteps=int(sim_params['duration'] * 60 / dt),
                                 save_to=self.d.vis_dir, **kwargs)

    def run(self):
        print()
        print(f'---- Simulation {self.id} ----')
        # Run the simulation
        completed = self.env.run()
        print()
        if not completed:
            self.d.delete()
            print('    Simulation aborted!')
            res = None
            # ds, fig_dict, results = None, None, None
        else:
            end = time.time()
            dur = end - self.start
            self.param_dict['date'] = datetime.datetime.now()
            self.param_dict['duration'] = np.round(dur, 2)
            print(f'    Simulation completed in {np.round(dur).astype(int)} seconds!')
            res = store_data(self.env, self.d, self.store_data, self.enrichment, self.param_dict)
            # if analysis and ds is not None :
            #     from lib.sim.analysis import sim_analysis
            #     fig_dict, results = sim_analysis(ds, env.experiment)
            # else :
            #     fig_dict, results = None, None
        self.env.close()
        return res

    def terminate(self):
        self.env.close()
        self.d.delete()


# def _run_sim(
#         sim_params,
#         env_params,
#         larva_groups,
#         trials,
#         enrichment,
#         collections,
#         experiment,
#         save_to=None,
#         seed=None,
#         **kwargs):
#     np.random.seed(seed)
#     id = sim_params['sim_ID']
#     dt = sim_params['timestep']
#     # Nsec = sim_params['duration'] * 60
#     path = sim_params['path']
#     # Box2D = sim_params['Box2D']
#     # save_data_flag = sim_params['store_data']
#     # analysis = sim_params['analysis']
#
#     if save_to is None:
#         save_to = paths.path("SIM")
#     if path is not None:
#         save_to = os.path.join(save_to, path)
#     dir_path = os.path.join(save_to, id)
#
#     # Store the parameters so that we can save them in the results folder
#     # sim_date = datetime.datetime.now()
#     param_dict = locals()
#     start = time.time()
#     # Nsteps = int(sim_params['duration'] * 60 / dt)
#
#     d = LarvaDataset(dir=dir_path, id=id, fr=1 / dt, Ncontour=0,
#                      env_params=env_params, larva_groups=larva_groups, load_data=False)
#
#     output = set_output(dataset=d, collections=collections)
#     env = LarvaWorldSim(id=id, dt=dt, Box2D=sim_params['Box2D'],
#                         env_params=env_params,
#                         larva_groups=larva_groups,
#                         output=output,
#                         experiment=experiment,
#                         trials=trials,
#                         Nsteps=int(sim_params['duration'] * 60 / dt),
#                         save_to=d.vis_dir, **kwargs)
#     print()
#     print(f'---- Simulation {id} ----')
#     # Run the simulation
#     completed = env.run()
#     print()
#     if not completed:
#         d.delete()
#         print('    Simulation aborted!')
#         res = None
#         # ds, fig_dict, results = None, None, None
#     else:
#         end = time.time()
#         dur = end - start
#         param_dict['date'] = datetime.datetime.now()
#         param_dict['duration'] = np.round(dur, 2)
#         print(f'    Simulation completed in {np.round(dur).astype(int)} seconds!')
#         res = store_data(env, d, sim_params['store_data'], enrichment, param_dict)
#         # if analysis and ds is not None :
#         #     from lib.sim.analysis import sim_analysis
#         #     fig_dict, results = sim_analysis(ds, env.experiment)
#         # else :
#         #     fig_dict, results = None, None
#     env.close()
#     return res


# ser = pickle.dumps(_run_sim)
# run_sim = pickle.loads(ser)


def store_data(env, d, save_data_flag, enrichment, param_dict, split_groups=True):
    # Read the data collected during the simulation
    step = env.larva_step_col.get_agent_vars_dataframe() if env.larva_step_col else None
    if env.larva_end_col is not None:
        env.larva_end_col.collect(env)
        end = env.larva_end_col.get_agent_vars_dataframe().droplevel('Step')
        # print(end)
        # raise
    else:
        end = None
    if env.food_end_col is not None:
        env.food_end_col.collect(env)
        food = env.food_end_col.get_agent_vars_dataframe().droplevel('Step')
    else:
        food = None
    d.set_data(step=step, end=end, food=food)
    if split_groups:
        ds = d.split_dataset()
    else:
        ds = [d]
    for dd in ds:
        dd.enrich(**enrichment, is_last=False)
        if save_data_flag:
            from lib.aux.dictsNlists import dict_to_file
            dd.save()
            dd.save_dicts(env)
            dict_to_file(param_dict, dd.dir_dict['sim'])
            if env.table_collector is not None:
                dd.save_tables(env.table_collector.tables)
    return ds


def set_output(dataset, collections):
    from lib.aux.dictsNlists import unique_list, flatten_list
    from lib.aux.collecting import output_dict

    if collections is None:
        collections = ['pose']
    cd = output_dict
    d = dataset
    step = []
    end = []
    tables = {}
    for c in collections:
        if c == 'midline':
            from lib.aux.collecting import midline_xy_pars
            step += list(midline_xy_pars(N=d.Nsegs).keys())
        elif c == 'contour':
            step += flatten_list(d.contour_xy)
        else:
            step += cd[c]['step']
            end += cd[c]['endpoint']
            if 'tables' in list(cd[c].keys()):
                tables.update(cd[c]['tables'])
    output = {'step': unique_list(step),
              'end': unique_list(end),
              'tables': tables,
              'step_groups': [],
              'end_groups': [],
              }

    # else:
    #     cd = combo_collection_dict
    #     cs = [cd[c] for c in collections if c in cd.keys()]
    #     output = {'step': [],
    #               'end': [],
    #               'tables': {},
    #               'step_groups': flatten_list([c['step'] for c in cs]),
    #               'end_groups': flatten_list([c['end'] for c in cs])}
    # # print(output)
    # # raise
    return output


def load_reference_dataset(dataset_id='reference', load=False):
    d = LarvaDataset(dir=f'{paths.path("REF")}/{dataset_id}', load_data=load)
    if not load:
        d.load(step=False)
    return d


def run_essay(id, path, exp_types, durations, vis_kwargs, **kwargs):
    from lib.conf.stored.conf import expandConf
    from lib.conf.base.dtypes import null_dict
    ds = []
    for i, (exp, dur) in enumerate(zip(exp_types, durations)):
        sim = null_dict('sim_params', duration=dur, sim_ID=f'{id}_{i}', path=path)
        conf = expandConf(exp, 'Exp')
        conf['sim_params'] = sim
        conf['experiment'] = exp
        conf.update(**kwargs)
        d = SingleRun(**conf, vis_kwargs=vis_kwargs).run()
        ds.append(d)
    return ds


combo_collection_dict = {
    'pose': {'step': ['basic', 'bouts', 'spatial', 'angular'], 'end': ['e_basic', 'e_dispersion']},
    'source_vincinity': {'step': ['chemorbit'], 'end': ['e_chemorbit']},
    'source_approach': {'step': ['chemotax'], 'end': ['e_chemotax']},
    'olfactor': {'step': ['odors', 'olfactor'], 'end': []},
}
