""" Run a simulation and save the parameters and data to files."""
import copy
import datetime
import random
import time
import numpy as np


from lib.registry import reg
from lib.aux import naming as nam, dictsNlists as dNl, sim_aux, dir_aux

from lib.model.envs._larvaworld_sim import LarvaWorldSim
from lib.registry.output import set_output
from lib.registry.pars import preg


class SingleRun:
    def __init__(self, sim_params, env_params, larva_groups, enrichment, collections, experiment,store_data=True,id=None,
                 trials={}, save_to=None, seed=None, analysis=False, show_output=False, **kwargs):
        np.random.seed(seed)
        random.seed(seed)
        self.show_output = show_output
        if id is None :
            id = sim_params.sim_ID
        self.id = id
        self.sim_params = sim_params
        self.experiment = experiment
        self.store_data = store_data
        dt = sim_params.timestep
        self.enrichment = enrichment
        self.analysis = analysis
        if save_to is None:
            # if sim_params.path is None :
            from lib.registry.pars import preg
            save_to = reg.Path.SIM
        self.save_to = save_to
        # self.storage_path = f'{sim_params.path}/{self.id}'
        self.storage_path = f'{self.save_to}/{self.id}'
        self.plot_dir = f'{self.storage_path}/plots'
        self.data_dir = f'{self.storage_path}/data'
        # self.vis_dir = f'{self.storage_path}/visuals'
        self.start = time.time()
        self.source_xy = sim_aux.get_source_xy(env_params['food_params'])

        output = set_output(collections=collections, Npoints=np.min([lg.model.body.Nsegs+1 for id, lg in larva_groups.items()]))
        # print(output)
        # raise
        self.env = LarvaWorldSim(id=self.id, dt=dt, Box2D=sim_params.Box2D, output=output,
                                 env_params=env_params, larva_groups=larva_groups, trials=trials,dur=sim_params.duration,
                                 experiment=self.experiment,
                                 save_to=f'{self.storage_path}/visuals',
                                 **kwargs)

    def run(self):
        self.datasets = self.env.simulate()
        if self.datasets is not None:
            for d in self.datasets:
                if self.enrichment:
                    reg.vprint()
                    reg.vprint(f'--- Enriching dataset {self.id} with derived parameters ---')
                    d.enrich(**self.enrichment, is_last=False, store=self.store_data)
            if self.sim_params.store_data and self.store_data:
                self.store()


    # def run2(self):
    #     if self.show_output:
    #         print()
    #         print(f'---- Simulation {self.id} ----')
    #     # Run the simulation
    #     completed = self.env.run()
    #     if not completed:
    #         print('    Simulation aborted!')
    #         self.datasets = None
    #     else:
    #         self.datasets = self.retrieve()
    #         end = time.time()
    #         dur = end - self.start
    #         if self.sim_params.store_data and self.store_data:
    #             self.store()
    #         if self.show_output:
    #             print(f'    Simulation {self.id} completed in {np.round(dur).astype(int)} seconds!')
    #     self.env.close()
    #     return self.datasets

    def terminate(self):
        self.env.close()



    # def retrieve(self):
    #     env = self.env
    #     # Read the data collected during the simulation
    #     step = env.step_collector.get_agent_vars_dataframe() if env.step_collector else None
    #     if env.end_collector is not None:
    #         env.end_collector.collect(env)
    #         end = env.end_collector.get_agent_vars_dataframe().droplevel('Step')
    #     else:
    #         end = None
    #     if env.food_collector is not None:
    #         env.food_collector.collect(env)
    #         food = env.food_collector.get_agent_vars_dataframe().droplevel('Step')
    #     else:
    #         food = None
    #
    #     ds = dir_aux.split_dataset(step, end, food, env_params=self.env.env_pars, larva_groups=self.env.larva_groups,
    #                                source_xy=self.source_xy,
    #                                fr=1 / self.env.dt, dir=self.data_dir, id=self.id)
    #     for d in ds:
    #         if self.show_output:
    #             print()
    #             print(f'--- Enriching dataset {self.id} with derived parameters ---')
    #         if self.enrichment:
    #             d.enrich(**self.enrichment, is_last=False, show_output=self.show_output,
    #                      store=self.sim_params.store_data)
    #         d.larva_dicts = env.get_larva_dicts(ids=d.agent_ids)
    #         d.larva_tables = env.get_larva_tables()
    #     return ds

    def store(self):
        from lib.aux.stor_aux import storeSoloDics,storeH5
        for d in self.datasets:
            d.save()
            for type, vs in d.larva_dicts.items():
                storeSoloDics(vs, path=reg.datapath(type, d.dir), use_pickle=False)
            storeH5(df=d.larva_tables, key=None, path=reg.datapath('tables', d.dir))


    def analyze(self, save_to=None, **kwargs):
        kws = {'datasets': self.datasets, 'save_to': save_to if save_to is not None else self.plot_dir, **kwargs}
        exp = self.experiment
        if 'PI' in exp:
            PIs = {}
            PI2s = {}
            for d in self.datasets:
                PIs[d.id] = d.config.PI["PI"]
                PI2s[d.id] = d.config.PI2
                if self.show_output:
                    print(f'Group {d.id} -> PI : {PIs[d.id]}')
                    print(f'Group {d.id} -> PI2 : {PI2s[d.id]}')
            return None, {'PIs': PIs, 'PI2s': PI2s}

        entry_list = []
        sources = self.source_xy
        if len(sources) > 0 and len(sources) < 10:
            from lib.conf.stored.analysis_conf import source_anal_list
            entry_list += source_anal_list(sources=sources)

        from lib.conf.stored.analysis_conf import analysis_dict
        if exp in ['random_food']:
            entry_list += analysis_dict.survival
        else:
            dic = {
                'patch': analysis_dict.patch,
                'tactile': analysis_dict.tactile,
                'thermo': analysis_dict.thermo,
                'RvsS': analysis_dict.deb,
                'RvsS': analysis_dict.intake,
                'growth': analysis_dict.deb,
                'growth': analysis_dict.intake,
                'anemo': analysis_dict.anemotaxis,
                'puff': analysis_dict.puff,
                'chemo': analysis_dict.chemo,
                'RL': analysis_dict.RL,
                # 'dispersion': ['comparative_analysis'],
                # 'dish': ['targeted_analysis'],
            }
            for k, v in dic.items():
                if k in exp:
                    entry_list += v

        return self.run_analysis(entry_list, **kws)

    def run_analysis(self, entry_list, **kws):
        exp = self.experiment
        figs, results = {}, {}
        if len(entry_list) > 0:
            from lib.plot.dict import graph_dict
            graph_entries = graph_dict.eval(entries=entry_list, **kws)
            figs.update(graph_entries)

        if 'disp' in exp:
            from lib.sim.single.analysis import comparative_analysis
            samples = dNl.unique_list([d.config.sample for d in self.datasets])
            targets = [preg.loadRef(sd) for sd in samples]
            kkws = copy.deepcopy(kws)
            kkws['datasets'] = self.datasets + targets
            figs.update(**comparative_analysis(**kkws))
        if 'dish' in exp:
            from lib.sim.single.analysis import targeted_analysis
            figs.update(**targeted_analysis(**kws))
        # if 'RvsS' in exp or 'growth' in exp:
        #     from lib.sim.single.analysis import deb_analysis
        #     figs.update(**deb_analysis(**kws))
        if len(figs) == 0 and len(results) == 0:
            return None, None
        else:
            return figs, results




def run_essay(id, path, exp_types, durations, vis_kwargs, **kwargs):
    from lib.conf.stored.conf import expandConf
    # from lib.registry.dtypes import null_dict
    ds = []
    for i, (exp, dur) in enumerate(zip(exp_types, durations)):
        conf = expandConf(exp, 'Exp')
        conf.sim_params = preg.init_dict.get_null('sim_params', duration=dur, sim_ID=f'{id}_{i}', path=path)
        conf.experiment = exp
        conf.update(**kwargs)
        d = SingleRun(**conf, vis_kwargs=vis_kwargs).run()
        ds.append(d)
    return ds
