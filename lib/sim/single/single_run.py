""" Run a simulation and save the parameters and data to files."""
import copy
import datetime
import time
import pickle
import os
import numpy as np

from lib.aux.dictsNlists import unique_list
from lib.conf.stored.conf import loadRef
from lib.model.envs._larvaworld_sim import LarvaWorldSim
from lib.conf.base import paths
from lib.sim.single.analysis import targeted_analysis


class SingleRun:
    def __init__(self, sim_params, env_params, larva_groups, enrichment, collections, experiment,
                 trials={}, save_to=None, seed=None, analysis=False, **kwargs):
        np.random.seed(seed)
        self.id = sim_params['sim_ID']
        self.sim_params = sim_params
        self.experiment = experiment
        dt = sim_params['timestep']
        self.store_data = sim_params['store_data']
        # analysis = sim_params['analysis']
        self.enrichment = enrichment
        self.analysis = analysis
        if save_to is None:
            save_to = paths.path("SIM")
        self.save_to = save_to
        self.storage_path = f'{sim_params["path"]}/{self.id}'
        self.dir_path = f'{save_to}/{self.storage_path}'
        self.plot_dir = f'{self.dir_path}/plots'
        self.param_dict = locals()
        self.start = time.time()
        self.source_xy = get_source_xy(env_params['food_params'])

        # self.d = LarvaDataset(dir=dir_path, id=self.id, fr=1 / dt,
        #                       env_params=env_params, larva_groups=larva_groups, load_data=False)
        output = set_output(collections=collections, Nsegs=list(larva_groups.values())[0]['model']['body']['Nsegs'])
        self.env = LarvaWorldSim(id=self.id, dt=dt, Box2D=sim_params['Box2D'], output=output,
                                 env_params=env_params, larva_groups=larva_groups, trials=trials,
                                 experiment=self.experiment, Nsteps=int(sim_params['duration'] * 60 / dt),
                                 save_to=f'{self.dir_path}/visuals', configuration_text=self.configuration_text,
                                 **kwargs)

    def run(self):
        print()
        print(f'---- Simulation {self.id} ----')
        # Run the simulation
        completed = self.env.run()
        print()
        if not completed:
            # self.d.delete()
            print('    Simulation aborted!')
            self.datasets = None
        else:
            self.datasets = self.retrieve()
            end = time.time()
            dur = end - self.start

            if self.store_data:
                self.param_dict['date'] = datetime.datetime.now()
                self.param_dict['duration'] = np.round(dur, 2)
                self.store()

            # res = store_data(self.env, self.d, self.store_data, self.enrichment, self.param_dict)
            # if analysis and ds is not None :
            #     from lib.sim.analysis import sim_analysis
            #     fig_dict, results = sim_analysis(ds, env.experiment)
            # else :
            #     fig_dict, results = None, None
            print(f'    Simulation {self.id} completed in {np.round(dur).astype(int)} seconds!')
        self.env.close()
        return self.datasets

    def terminate(self):
        self.env.close()
        # self.d.delete()

    @property
    def configuration_text(self):
        sim = self.sim_params
        text = f"Simulation configuration : \n" \
               "\n" \
               f"Experiment : {self.experiment}\n" \
               f"Simulation ID : {self.id}\n" \
               f"Duration (min) : {sim['duration']}\n" \
               f"Timestep (sec) : {sim['timestep']}\n" \
               f"Parent path : {self.save_to}\n" \
               f"Dataset path : {self.storage_path}"
        return text

    def retrieve(self):
        from lib.stor.managing import split_dataset
        env = self.env
        # Read the data collected during the simulation
        step = env.larva_step_col.get_agent_vars_dataframe() if env.larva_step_col else None
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

        ds = split_dataset(step, end, food, env_params=self.env.env_pars, larva_groups=self.env.larva_groups,
                           source_xy=self.source_xy,
                           fr=1 / self.env.dt, dir=self.dir_path, id=self.id, plot_dir=self.plot_dir, show_output=False)
        for d in ds:
            d.enrich(**self.enrichment, is_last=False)
            d.get_larva_dicts(env)
            d.get_larva_tables(env)
        return ds

    def store(self):
        for d in self.datasets:
            from lib.aux.dictsNlists import dict_to_file
            d.save()
            d.save_larva_dicts()
            d.save_larva_tables()
            dict_to_file(self.param_dict, d.dir_dict.sim)

    def analyze(self, save_to=None, **kwargs):
        exp=self.experiment
        from lib.sim.single.analysis import source_analysis, deb_analysis, comparative_analysis, foraging_analysis
        from lib.conf.stored.analysis_conf import analysis_dict
        if 'tactile' in exp:
            anal_params = analysis_dict.tactile
        elif 'RvsS' in exp or 'growth' in exp:
            anal_params = analysis_dict.intake
        elif 'anemo' in exp:
            anal_params = analysis_dict.anemotaxis
        elif 'puff' in exp:
            anal_params = analysis_dict.puff
        elif 'chemo' in exp:
            anal_params = analysis_dict.chemo
        elif 'RL' in exp:
            anal_params = analysis_dict.RL
        elif exp in ['food_at_bottom']:
            anal_params = ['foraging_analysis']
        elif exp in ['random_food']:
            anal_params = analysis_dict.survival
        elif 'dispersion' in exp :
            anal_params = ['comparative_analysis']
        elif 'dish' in exp:
            anal_params = ['targeted_analysis']
        # elif self.experiment in ['growth', 'RvsS'] :
        #     anal_params = analysis_dict['DEB']
        else:
            return None, None

        kws = {'datasets': self.datasets, 'save_to': save_to if save_to is not None else self.plot_dir, **kwargs}
        from lib.anal.plotting import graph_dict
        figs, results = {}, {}
        for entry in anal_params:
            if entry == 'source_analysis':
                figs.update(**source_analysis(self.source_xy, **kws))
            elif entry == 'foraging_analysis':
                figs.update(**foraging_analysis(self.source_xy, **kws))
            elif entry == 'deb_analysis':
                figs.update(**deb_analysis(**kws))
            elif entry == 'targeted_analysis':
                figs.update(**targeted_analysis(**kws))
            elif entry == 'comparative_analysis':
                samples = unique_list([d.config.sample for d in self.datasets])
                targets = [loadRef(sd) for sd in samples]
                kkws = copy.deepcopy(kws)
                kkws['datasets'] = self.datasets + targets
                figs.update(**comparative_analysis(**kkws))
            else:
                figs[entry['title']] = graph_dict[entry['plotID']](**entry['args'], **kws)
        return figs, results


def set_output(collections, Nsegs=2, Ncontour=0):
    from lib.aux.dictsNlists import unique_list, flatten_list
    from lib.aux.collecting import output_dict

    if collections is None:
        collections = ['pose']
    cd = output_dict
    # d = dataset
    step = []
    end = []
    tables = {}
    for c in collections:
        if c == 'midline':
            from lib.aux.collecting import midline_xy_pars
            # Nsegs=np.clip(Npoints - 1, a_min=0, a_max=None)
            step += list(midline_xy_pars(N=Nsegs).keys())
        elif c == 'contour':
            from lib.aux import naming as nam
            nam.contour(Ncontour)
            contour_xy = nam.xy(nam.contour(Ncontour))
            step += flatten_list(contour_xy)
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


def run_essay(id, path, exp_types, durations, vis_kwargs, **kwargs):
    from lib.conf.stored.conf import expandConf
    from lib.conf.base.dtypes import null_dict
    ds = []
    for i, (exp, dur) in enumerate(zip(exp_types, durations)):
        conf = expandConf(exp, 'Exp')
        conf.sim_params = null_dict('sim_params', duration=dur, sim_ID=f'{id}_{i}', path=path)
        conf.experiment = exp
        conf.update(**kwargs)
        d = SingleRun(**conf, vis_kwargs=vis_kwargs).run()
        ds.append(d)
    return ds


def get_source_xy(food_params):
    sources_u = {k: v['pos'] for k, v in food_params['source_units'].items()}
    sources_g = {k: v['distribution']['loc'] for k, v in food_params['source_groups'].items()}
    return {**sources_u, **sources_g}


combo_collection_dict = {
    'pose': {'step': ['basic', 'bouts', 'spatial', 'angular'], 'end': ['e_basic', 'e_dispersion']},
    'source_vincinity': {'step': ['chemorbit'], 'end': ['e_chemorbit']},
    'source_approach': {'step': ['chemotax'], 'end': ['e_chemotax']},
    'olfactor': {'step': ['odors', 'olfactor'], 'end': []},
}
