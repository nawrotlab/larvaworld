import os
import numpy as np
import random

from lib import reg, aux




class BaseRun:
    def __init__(self, runtype, experiment=None, id=None, progress_bar=False, save_to=None, store_data=True,
                 analysis=True, show=False, seed=None, model_class=None, graph_entries=[]):
        if model_class is None :
            from lib.model.envs.world_sim import WorldSim
            model_class = WorldSim

        np.random.seed(seed)
        random.seed(seed)
        self.runtype = runtype
        self.parent_storage_path = f'{reg.SIM_DIR}/{self.runtype.lower()}_runs'
        self.SimIdx_path = reg.SimIdx_PATH

        if experiment is None:
            experiment = self.runtype
        self.experiment = experiment
        if id is None:
            idx = self.next_idx(self.experiment)
            id = f'{self.runtype}_{self.experiment}_{idx}'
        self.id = id
        if save_to is None:
            save_to = self.parent_storage_path
        self.storage_path = f'{save_to}/{self.id}'
        self.anal_dir = f'{self.storage_path}/analysis'
        self.plot_dir = f'{self.storage_path}/plots'
        self.data_dir = f'{self.storage_path}/data'
        self.store_data = store_data
        self.model_class = model_class
        self.analysis = analysis
        self.show = show
        # self.data = None
        self.datasets = None
        self.figs = aux.NestDict()
        self.results = {}
        self.progress_bar = progress_bar

        self.graph_entries=graph_entries
        self.analysis_kws={}

        # self.is_running=False
        # self.completed=False
        # self.aborted=False


    def retrieve(self):
        pass

    def store(self):
        pass

    def next_idx(self, exp):
        f = self.SimIdx_path
        runtype = self.runtype
        if not os.path.isfile(f):
            d = aux.NestDict()
        else:
            d = aux.dNl.load_dict(f, use_pickle=False)
        if not runtype in d.keys():
            d[runtype] = aux.NestDict()
        if not exp in d[runtype].keys():
            d[runtype][exp] = 0
        d[runtype][exp] += 1
        aux.dNl.save_dict(d, f, use_pickle=False)
        return d[runtype][exp]

    def simulate(self):
        pass
        # reg.vprint()
        # reg.vprint(f'---- {self.runtype} exec : {self.id} ----')
        # # Run the simulation
        # start = time.time()
        # is_running = True
        # while is_running:
        # #     if self.aborted:
        # #         reg.vprint(f'---- {self.runtype} exec : {self.id} ----')
        # #     elif self.completed:
        # #         self.retrieve()
        # #         end = time.time()
        # #         dur = np.round(end - start).astype(int)
        # #         reg.vprint(f'    {self.runtype} exec : {self.id} completed in {dur} seconds!')
        # #         self.is_running = False
        # # return self.data

    def run(self):

        self.simulate()

        if self.store_data:
            os.makedirs(self.data_dir, exist_ok=True)
            self.store()

        if self.analysis:
            os.makedirs(self.plot_dir, exist_ok=True)
            self.analyze()
        return self.datasets

    def analyze(self):
        self.auto_anal()
        self.manual_anal()

    def auto_anal(self):
        if self.datasets is not None :
            if len(self.graph_entries)>0 :
                kwargs = {
                    'datasets': self.datasets,
                    'save_to': self.plot_dir,
                    'show': self.show,
                    # 'title': f"DOUBLE PATCH ESSAY (N={self.N}, duration={self.dur}')",
                    # 'mdiff_df': self.mdiff_df
                    **self.analysis_kws
                }
                self.figs.auto=reg.graphs.eval(entries=self.graph_entries, **kwargs)
        pass

    def manual_anal(self):
        pass

