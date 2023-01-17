""" Run a simulation and save the parameters and data to files."""

from lib import reg, aux
from lib.model.envs2.world import Larvaworld
from lib.sim.base import BaseRun


# TODO Make SingleRun also a subclass of BaseRun
class ExpRun(BaseRun):
    def __init__(self, sim_params,env_params, enrichment, collections, larva_groups,trials,
                 parameter_dict={}, **kwargs):
        super().__init__(runtype='exp',id=sim_params.sim_ID, **kwargs)
        self.enrichment = enrichment
        self.sim_params = sim_params

        dt = sim_params.timestep
        dur = sim_params.duration
        steps = int(dur * 60 / dt) if dur is not None else None


        # Npoints = np.min([lg.model.body.Nsegs + 1 for id, lg in larva_groups.items()])
        # output = reg.set_output(collections=collections,Npoints=Npoints)
        self.model_conf = {


            'parameters':{
                'steps': steps,
                'dt': dt,
                'trials': trials,
                'collections': collections,
                'id': self.id,
                'experiment': self.experiment,
                'save_to': self.data_dir,
                'Box2D': sim_params.Box2D,
                'larva_groups': larva_groups,
                "parameter_dict": parameter_dict,
                'env_params': env_params,
                # 'screen': {},
                **kwargs
            }

        }

        self.model = Larvaworld(**self.model_conf)


    def simulate(self):
        self.datasets=self.model.simulate()
        if self.datasets is not None :
            for d in self.datasets:
                if self.enrichment:
                    reg.vprint()
                    reg.vprint(f'--- Enriching dataset {self.id} with derived parameters ---')
                    d.enrich(**self.enrichment, is_last=False, store=self.store_data)

        return self.datasets


    # def terminate(self):
    #     self.model.close()

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

        for d in self.datasets:
            d.save()
            for type, vs in d.larva_dicts.items():
                aux.storeSoloDics(vs, path=reg.datapath(type, d.dir), use_pickle=False)
            aux.storeH5(df=d.larva_tables, key=None, path=reg.datapath('tables', d.dir))

    def analyze(self, save_to=None, **kwargs):
        exp = self.experiment
        ds = self.datasets

        if ds is None or any([d is None for d in ds]):
            return

        if 'PI' in exp:
            PIs = {}
            PI2s = {}
            for d in ds:
                PIs[d.id] = d.config.PI["PI"]
                PI2s[d.id] = d.config.PI2
                # if self.show_output:
                #     print(f'Group {d.id} -> PI : {PIs[d.id]}')
                #     print(f'Group {d.id} -> PI2 : {PI2s[d.id]}')
            self.results={'PIs': PIs, 'PI2s': PI2s}
            return

        if 'disp' in exp:
            samples = aux.unique_list([d.config.sample for d in ds])
            ds += [reg.loadRef(sd) for sd in samples]

        kws = {'datasets': ds, 'save_to': save_to if save_to is not None else self.plot_dir, **kwargs}
        sources = self.model.source_xy
        graphgroups = reg.graphs.get_analysis_graphgroups(exp, sources)
        self.figs = reg.graphs.eval_graphgroups(graphgroups, **kws)


if __name__ == "__main__":
    from lib import reg
    conf=reg.expandConf(conftype='Exp', id='chemorbit')
    exp_run=ExpRun(**conf)
    ds=exp_run.simulate()
    exp_run.analyze()