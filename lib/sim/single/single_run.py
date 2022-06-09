""" Run a simulation and save the parameters and data to files."""
import copy
import datetime
import random
import time
import numpy as np
from lib.aux import naming as nam,dictsNlists as dNl
from lib.aux.sim_aux import get_source_xy
from lib.model.envs._larvaworld_sim import LarvaWorldSim
from lib.conf.base import paths


class SingleRun:
    def __init__(self, sim_params, env_params, larva_groups, enrichment, collections, experiment,
                 trials={}, save_to=None, seed=None, analysis=False, show_output=False, **kwargs):
        np.random.seed(seed)
        random.seed(seed)
        self.show_output = show_output
        self.id = sim_params.sim_ID
        self.sim_params = sim_params
        self.experiment = experiment
        dt = sim_params.timestep
        self.enrichment = enrichment
        self.analysis = analysis
        if save_to is None:
            save_to = paths.path("SIM")
        self.save_to = save_to
        self.storage_path = f'{sim_params.path}/{self.id}'
        self.dir_path = f'{save_to}/{self.storage_path}'
        self.plot_dir = f'{self.dir_path}/plots'
        self.data_dir = f'{self.dir_path}/data'
        self.param_dict = locals()
        self.start = time.time()
        self.source_xy = get_source_xy(env_params['food_params'])
        output = set_output(collections=collections, Nsegs=list(larva_groups.values())[0]['model']['body']['Nsegs'])
        self.env = LarvaWorldSim(id=self.id, dt=dt, Box2D=sim_params.Box2D, output=output,
                                 env_params=env_params, larva_groups=larva_groups, trials=trials,
                                 experiment=self.experiment, Nsteps=int(sim_params.duration * 60 / dt),
                                 save_to=f'{self.dir_path}/visuals', configuration_text=self.configuration_text,
                                 **kwargs)

    def run(self):
        if self.show_output :
            print()
            print(f'---- Simulation {self.id} ----')
        # Run the simulation
        completed = self.env.run()
        if not completed:
            print('    Simulation aborted!')
            self.datasets = None
        else:
            self.datasets = self.retrieve()
            end = time.time()
            dur = end - self.start
            if self.sim_params.store_data:
                self.param_dict['date'] = datetime.datetime.now()
                self.param_dict['duration'] = np.round(dur, 2)
                self.store()

            # res = store_data(self.env, self.d, self.store_data, self.enrichment, self.param_dict)
            # if analysis and ds is not None :
            #     from lib.sim.analysis import sim_analysis
            #     fig_dict, results = sim_analysis(ds, env.experiment)
            # else :
            #     fig_dict, results = None, None
            if self.show_output :
                print(f'    Simulation {self.id} completed in {np.round(dur).astype(int)} seconds!')
        self.env.close()
        return self.datasets

    def terminate(self):
        self.env.close()

    @property
    def configuration_text(self):
        text = f"Simulation configuration : \n" \
               "\n" \
               f"Experiment : {self.experiment}\n" \
               f"Simulation ID : {self.id}\n" \
               f"Duration (min) : {self.sim_params.duration}\n" \
               f"Timestep (sec) : {self.sim_params.timestep}\n" \
               f"Parent path : {self.save_to}\n" \
               f"Dataset path : {self.storage_path}"
        return text

    def retrieve(self):
        from lib.stor.managing import split_dataset
        env = self.env
        # Read the data collected during the simulation
        step = env.step_collector.get_agent_vars_dataframe() if env.step_collector else None
        if env.end_collector is not None:
            env.end_collector.collect(env)
            end = env.end_collector.get_agent_vars_dataframe().droplevel('Step')
        else:
            end = None
        if env.food_collector is not None:
            env.food_collector.collect(env)
            food = env.food_collector.get_agent_vars_dataframe().droplevel('Step')
        else:
            food = None

        ds = split_dataset(step, end, food, env_params=self.env.env_pars, larva_groups=self.env.larva_groups,
                           source_xy=self.source_xy,
                           fr=1 / self.env.dt, dir=self.data_dir, id=self.id, plot_dir=self.plot_dir, show_output=self.show_output)
        for d in ds:
            if self.show_output :
                print()
                print(f'--- Enriching dataset {self.id} with derived parameters ---')
            if self.enrichment:
                d.enrich(**self.enrichment, is_last=False, show_output=self.show_output, store=self.sim_params.store_data)
            d.get_larva_dicts(env)
            d.get_larva_tables(env)
        return ds

    def store(self):
        for d in self.datasets:
            d.save()
            d.save_larva_dicts()
            d.save_larva_tables()
            dNl.dict_to_file(self.param_dict, d.dir_dict.sim)

    def analyze(self, **kwargs):
        exp=self.experiment

        from lib.conf.stored.analysis_conf import analysis_dict
        dic={
            'tactile' : analysis_dict.tactile,
            'RvsS' : analysis_dict.intake,
            'growth' : analysis_dict.intake,
            'anemo' : analysis_dict.anemotaxis,
            'puff' : analysis_dict.puff,
            'chemo' : analysis_dict.chemo,
            'RL' : analysis_dict.RL,
            'dispersion' :  ['comparative_analysis'],
            'dish' :  ['targeted_analysis'],
        }
        for k,v in dic.items() :
            if k in exp :
                return self.run_analysis(v, **kwargs)
        if exp in ['food_at_bottom']:
            return self.run_analysis(['foraging_analysis'], **kwargs)
        elif exp in ['random_food']:
            return self.run_analysis(analysis_dict.survival, **kwargs)
        elif 'PI' in exp:
            PIs = {}
            PI2s = {}
            for d in self.datasets :
                PIs[d.id]=d.config.PI["PI"]
                PI2s[d.id]=d.config.PI2
                if self.show_output :
                    print(f'Group {d.id} -> PI : {PIs[d.id]}')
                    print(f'Group {d.id} -> PI2 : {PI2s[d.id]}')
            return None, {'PIs': PIs, 'PI2s': PI2s}
        else:
            return None, None

    def run_analysis(self, anal_params,save_to=None,**kwargs) :
        from lib.sim.single.analysis import targeted_analysis,source_analysis, deb_analysis, comparative_analysis, foraging_analysis
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
                from lib.conf.stored.conf import loadRef

                samples = dNl.unique_list([d.config.sample for d in self.datasets])
                targets = [loadRef(sd) for sd in samples]
                kkws = copy.deepcopy(kws)
                kkws['datasets'] = self.datasets + targets
                figs.update(**comparative_analysis(**kkws))
            else:
               try :
                    figs[entry['title']] = graph_dict[entry['plotID']](**entry['args'], **kws)
               except :
                   pass
        return figs, results


def set_output(collections, Nsegs=2, Ncontour=0):
    from lib.aux.collecting import output_dict
    if collections is None:
        collections = ['pose']
    cd = output_dict
    step = []
    end = []
    tables = {}
    for c in collections:
        if c == 'midline':
            from lib.aux.collecting import midline_xy_pars
            step += list(midline_xy_pars(N=Nsegs).keys())
        elif c == 'contour':
            step += dNl.flatten_list(nam.xy(nam.contour(Ncontour)))
        else:
            step += cd[c]['step']
            end += cd[c]['endpoint']
            if 'tables' in list(cd[c].keys()):
                tables.update(cd[c]['tables'])
    output = {'step': dNl.unique_list(step),
              'end': dNl.unique_list(end),
              'tables': tables,
              }
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

