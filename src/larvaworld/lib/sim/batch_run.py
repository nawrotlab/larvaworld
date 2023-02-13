import copy
import itertools
import logging
import os
import time

import agentpy
import numpy as np
import agentpy as ap
import pandas as pd

from larvaworld.lib import reg, aux
from larvaworld.lib.plot.scape import plot_heatmap_PI, plot_3d, plot_3pars, plot_2d
from larvaworld.lib.sim import ExpRun
# from lib.sim.base_run import BaseRun

class BatchRun(ap.Experiment):
    def __init__(self, batch_type, save_to=None, id=None,space_search=None,space_kws={},optimization=None,
                 record=True, exp=None, exp_kws={}, store_data=False,Box2D=False,  **kwargs):

        runtype = 'Batch'

        self.batch_type=batch_type

        if id is None:
            idx = reg.next_idx(self.batch_type, conftype=runtype)
            id = f'{self.batch_type}_{idx}'
        self.id = id
        # Define directories
        path=f'{reg.SIM_DIR}/{runtype.lower()}_runs'
        if save_to is None:
            save_to = path
        self.h5_path=f'{path}/{self.batch_type}/{self.batch_type}.h5'

        self.dir = f'{save_to}/{self.batch_type}/{self.id}'
        self.df_path = f'{self.dir}/results.h5'
        self.plot_dir = f'{self.dir}/plots'
        self.data_dir = f'{self.dir}/data'
        self.save_to = self.dir
        self.store_data = store_data
        self.Box2D = Box2D

        self.exp_conf = reg.expandConf('Exp', exp) if isinstance(exp, str) else exp
        self.exp_conf.update(**exp_kws)
        if optimization is not None:
            optimization['ranges'] = np.array([space_search[k]['range'] for k in space_search.keys() if 'range' in space_search[k].keys()])
        self.optimization = optimization
        super().__init__(model_class=ExpRun, sample = space_search_sample(space_search, **space_kws),
                         record=record, **kwargs)

        self.datasets = {}
        self.results = None
        self.figs = {}

    def _single_sim(self, run_id):
        """ Perform a single simulation."""
        sample_id = 0 if run_id[0] is None else run_id[0]
        parameters = self.exp_conf.update_existingnestdict_by_suffix(self.sample[sample_id])
        model = self.model(parameters=parameters, _run_id=run_id,store_data = self.store_data,
                           Box2D = self.Box2D, **self._model_kwargs)

        if self._random:
            ds = model.simulate(display=False, seed=self._random[run_id])
        else:
            ds = model.simulate(display=False)
        self.datasets[sample_id]=ds
        if 'variables' in model.output and self.record is False:
            del model.output['variables']  # Remove dynamic variables from record
        return model.output

    def default_processing(self, d=None):
        p = self.optimization.fit_par
        s, e = d.step_data, d.endpoint_data
        if p in e.columns:
            vals = e[p].values
        elif p in s.columns:
            vals = s[p].groupby('AgentID').mean()
        else:
            raise ValueError('Could not retrieve fit parameter from dataset')

        ops = self.optimization.operations
        if ops.abs:
            vals = np.abs(vals)
        if ops.mean:
            fit = np.mean(vals)
        elif ops.std:
            fit = np.std(vals)
        return fit

    @property
    def threshold_reached(self):
        fits = self.par_df['fit'].values
        if self.optimization.minimize:
            return np.nanmin(fits) <= self.optimization.threshold
        else:
            return np.nanmax(fits) >= self.optimization.threshold

    def end(self):
        self.par_df = self.output._combine_pars()
        self.par_names = self.par_df.columns.values.tolist()

        if self.optimization is not None :
            self.par_df['fit'] = [self.default_processing(self.datasets[i][0]) for i in self.par_df.index]


            if self.par_df.index.size >= self.optimization.max_Nsims:
                reg.vprint(f'Maximum number of simulations reached. Halting search',2)
            elif self.threshold_reached:
                reg.vprint(f'Best result reached threshold. Halting search',2)
            else:
                reg.vprint(f'Not reached threshold. Expanding space search', 2)
                pass
            # traj.f_expand(get_Nbest(traj))
            # for sample_id, ds in self.datasets.items():
            #     for d in ds:
            #         if d is not None:

        #     dics=[]
        #     for d in ds :
        #         dic = agentpy.DataDict()
        #         dic['id']=d.id
        #         dic['step'] = d.step_data
        #         dic['end'] = d.endpoint_data
        #         dic['config'] = d.config
        #         dics.append(dic)
        #     if len(dics)==1:
        #         dics=dics[0]
        #     self.output['datasets'][sample_id] = dics

    def simulate(self, **kwargs):
        self.run(**kwargs)
        os.makedirs(self.plot_dir, exist_ok=True)
        if 'PI' in self.batch_type :
            self.PI_heatmap()
        # if 'chemo' in self.batch_type :
            # p='final_dst_to_Source'
            # self.par_df[p] = [self.datasets[i][0].endpoint_data[p].mean() for i in self.par_df.index]
        self.plot_results()
        # self.par_df.to_csv(os.path.join(self.data_dir, 'results.csv'), index=True, header=True)
        aux.storeH5(self.par_df, key='results', path=self.df_path, mode='w')
        return self.par_df, self.figs

    def plot_results(self):
        p_ns = self.par_names
        target_ns = [p for p in self.par_df.columns if p not in p_ns]
        kws = {'df': self.par_df,
               'save_to': self.plot_dir,
               'show': True}
        for target in target_ns:
            if len(p_ns) == 1:
                self.figs[f'{p_ns[0]}VS{target}'] = plot_2d(labels=p_ns + [target], pref=target, **kws)
            elif len(p_ns) == 2:
                self.figs.update(plot_3pars(vars=p_ns, target=target, pref=target, **kws))
            elif len(p_ns) > 2:
                for i, pair in enumerate(itertools.combinations(p_ns, 2)):
                    self.figs.update(plot_3pars(vars=list(pair), target=target, pref=f'{i}_{target}', **kws))

    def PI_heatmap(self, **kwargs):
        PIs=[self.datasets[i][0].config.PI['PI'] for i in self.par_df.index]
        Lgains = self.par_df.values[:,0].astype(int)
        Rgains = self.par_df.values[:,1].astype(int)
        df = pd.DataFrame(index=pd.Series(np.unique(Lgains), name="left_gain"),
                          columns=pd.Series(np.unique(Rgains), name="right_gain"), dtype=float)
        for Lgain, Rgain, PI in zip(Lgains, Rgains, PIs):
            df[Rgain].loc[Lgain] = PI
        df.to_csv(f'{self.plot_dir}/PIs.csv', index=True, header=True)
        self.figs['PI_heatmap'] = plot_heatmap_PI(save_to=self.plot_dir, z=df, **kwargs)

def space_search_sample(space_dict,n=1, **kwargs):
    dic={}
    for p, args in space_dict.items() :
        if not isinstance(args, dict) or ('values' not in args.keys() and 'range' not in args.keys()) :
            dic[p] = args
        elif args['values'] is not None :
            dic[p]=ap.Values(*args['values'])
        else :
            r0,r1=args['range']
            if 'Ngrid' in args.keys() :
                vs = np.linspace(r0, r1, args['Ngrid'])
                if type(r0) == int and type(r1) == int:
                    vs = vs.astype(int)
                dic[p] = ap.Values(*vs.tolist())
            else :
                if type(r0) == int and type(r1) == int:
                    dic[p] = ap.IntRange(r0, r1)
                elif type(r0) == float and type(r1) == float:
                    dic[p] = ap.Range(r0, r1)
    return ap.Sample(dic,n=n,**kwargs)


if __name__ == "__main__":
    batch_type = 'chemorbit'
    batch_conf = reg.loadConf('Batch', batch_type)

    m = BatchRun(batch_type=batch_type, **batch_conf)
    m.simulate(n_jobs=1)
    # m.PI_heatmap(show=True)
