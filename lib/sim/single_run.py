""" Run a simulation and save the parameters and data to files."""
import random
import time
import numpy as np

from lib import reg, aux

from lib.model.envs.world_sim import WorldSim


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
            save_to = reg.Path.SIM
        self.save_to = save_to
        # self.storage_path = f'{sim_params.path}/{self.id}'
        self.storage_path = f'{self.save_to}/{self.id}'
        self.plot_dir = f'{self.storage_path}/plots'
        self.data_dir = f'{self.storage_path}/data'
        # self.vis_dir = f'{self.storage_path}/visuals'
        self.start = time.time()
        self.source_xy = aux.get_source_xy(env_params['food_params'])
        Npoints = np.min([lg.model.body.Nsegs + 1 for id, lg in larva_groups.items()])
        output = reg.set_output(collections=collections, Npoints=Npoints)

        self.env = WorldSim(id=self.id, dt=dt, Box2D=sim_params.Box2D, output=output,
                            env_params=env_params, larva_groups=larva_groups, trials=trials, dur=sim_params.duration,
                            experiment=self.experiment,
                            save_to=f'{self.storage_path}/visuals',
                            **kwargs)

    def run(self):
        self.datasets = self.env.simulate()
        if self.datasets is not None:
            for d in self.datasets:
                if self.enrichment:
                    reg.vprint()
                    reg.vprint(f'--- Enriching dataset {self.id} with derived parameters ---', 1)
                    d.enrich(**self.enrichment, is_last=False, store=self.store_data)
            if self.sim_params.store_data and self.store_data:
                self.store()



    def terminate(self):
        self.env.close()



    def store(self):
        from lib.aux.stor_aux import storeH5
        from lib.aux.dictsNlists import storeSoloDics
        for d in self.datasets:
            d.save()
            for type, vs in d.larva_dicts.items():
                storeSoloDics(vs, path=reg.datapath(type, d.dir), use_pickle=False)
            storeH5(df=d.larva_tables, key=None, path=reg.datapath('tables', d.dir))


    def analyze(self, save_to=None, **kwargs):
        exp = self.experiment
        ds = self.datasets

        if ds is None or any([d is None for d in ds]):
            return None, None

        if 'PI' in exp:
            PIs = {}
            PI2s = {}
            for d in ds:
                PIs[d.id] = d.config.PI["PI"]
                PI2s[d.id] = d.config.PI2
                # if self.show_output:
                #     print(f'Group {d.id} -> PI : {PIs[d.id]}')
                #     print(f'Group {d.id} -> PI2 : {PI2s[d.id]}')
            return None, {'PIs': PIs, 'PI2s': PI2s}

        if 'disp' in exp:
            samples = aux.unique_list([d.config.sample for d in ds])
            ds += [reg.loadRef(sd) for sd in samples]

        kws = {'datasets': ds, 'save_to': save_to if save_to is not None else self.plot_dir, **kwargs}
        sources = self.source_xy
        # from lib.conf.stored.analysis_conf import get_analysis_graphgroups

        graphgroups=reg.graphs.get_analysis_graphgroups(exp, sources)
        figs=reg.graphs.eval_graphgroups(graphgroups, **kws)
        return figs, None

    # def run_analysis(self, entry_list, **kws):
    #     # exp = self.experiment
    #     figs, results = {}, {}
    #     if len(entry_list) > 0:
    #         graph_entries = reg.graphs.eval(entries=entry_list, **kws)
    #         figs.update(graph_entries)
    #     # FIXME Substituted "comparative analysis" of dispersion simulation to automatize analysis. Probably will fail
    #     # if 'disp' in exp:
    #     #     from lib.sim.single.analysis import comparative_analysis
    #     #     samples = aux.unique_list([d.config.sample for d in self.datasets])
    #     #     targets = [reg.loadRef(sd) for sd in samples]
    #     #     kkws = copy.deepcopy(kws)
    #     #     kkws['datasets'] = self.datasets + targets
    #     #     figs.update(**comparative_analysis(**kkws))
    #     # if 'dish' in exp:
    #     #     from lib.sim.single.analysis import targeted_analysis
    #     #     figs.update(**targeted_analysis(**kws))
    #     if len(figs) == 0 and len(results) == 0:
    #         return None, None
    #     else:
    #         return figs, results

