""" Run a simulation and save the parameters and data to files."""
import copy
import numpy as np
from lib.aux import naming as nam, dictsNlists as dNl, sim_aux, dir_aux

from lib.model.envs.world_sim import WorldSim
from lib.registry.output import set_output
from lib.registry import reg, base


class ExpRun(base.BaseRun):
    def __init__(self, sim_params, enrichment, collections, larva_groups,progress_bar=False, save_to=None, store_data=True,
                 analysis=True, show=False, **kwargs):

        kws = {
            # 'dt': dt,
            'model_class': WorldSim,
            'progress_bar': progress_bar,
            'save_to': save_to,
            'store_data': store_data,
            'analysis': analysis,
            'show': show,
            # 'Nsteps': int(sim_params.duration * 60 / dt),
            # 'output': output,
            'id': sim_params.sim_ID,
            # 'Box2D': sim_params.Box2D,
            # 'larva_groups': larva_groups,
            # **kwargs
        }
        super().__init__(runtype='exp', **kws)
        self.enrichment = enrichment
        self.sim_params = sim_params
        dt = sim_params.timestep
        output = set_output(collections=collections,
                            Npoints=np.min([lg.model.body.Nsegs + 1 for id, lg in larva_groups.items()]))
        self.model_conf = {
            'dt': dt,
            # 'model_class': WorldSim,
            'dur': sim_params.duration,
            'output': output,
            'id': self.id,
            'experiment': self.experiment,
            'save_to': self.data_dir,
            'Box2D': sim_params.Box2D,
            'larva_groups': larva_groups,
            **kwargs
        }



        self.model = self.model_class(**self.model_conf)
        self.datasets = None
        # self.data=dNl.NestDict({'datas'})



    def simulate(self):
        self.datasets=self.model.simulate()
        if self.datasets is not None :
            for d in self.datasets:
                if self.enrichment:
                    reg.vprint()
                    reg.vprint(f'--- Enriching dataset {self.id} with derived parameters ---')
                    d.enrich(**self.enrichment, is_last=False, store=self.store_data)

        return self.datasets


    def terminate(self):
        self.model.close()

    @property
    def configuration_text(self):
        text = f"Simulation configuration : \n" \
               "\n" \
               f"Experiment : {self.experiment}\n" \
               f"Simulation ID : {self.id}\n" \
               f"Duration (min) : {self.sim_params.duration}\n" \
               f"Timestep (sec) : {self.sim_params.timestep}\n" \
               f"Plot path : {self.plot_dir}\n" \
               f"Parent path : {self.storage_path}"
        return text



    def store(self):
        from lib.aux.stor_aux import storeSoloDics, storeH5
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
                # if self.show_output:
                #     print(f'Group {d.id} -> PI : {PIs[d.id]}')
                #     print(f'Group {d.id} -> PI2 : {PI2s[d.id]}')
            return None, {'PIs': PIs, 'PI2s': PI2s}

        entry_list = []
        sources = self.datasets[0].config.source_xy
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

        self.figs,results= self.run_analysis(entry_list, **kws)

    def run_analysis(self, entry_list, **kws):
        exp = self.experiment
        figs, results = {}, {}
        if len(entry_list) > 0:
            graph_entries = reg.GD.eval(entries=entry_list, **kws)
            figs.update(graph_entries)

        if 'disp' in exp:
            from lib.sim.single.analysis import comparative_analysis
            samples = dNl.unique_list([d.config.sample for d in self.datasets])
            targets = [reg.loadRef(sd) for sd in samples]
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
        conf.sim_params = reg.get_null('sim_params', duration=dur, sim_ID=f'{id}_{i}', path=path)
        conf.experiment = exp
        conf.update(**kwargs)
        d = ExpRun(**conf, vis_kwargs=vis_kwargs).run()
        ds.append(d)
    return ds
