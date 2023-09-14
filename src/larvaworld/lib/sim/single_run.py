import copy
import os
import time
import agentpy
import numpy as np
import pandas as pd
from pandas import to_datetime

from larvaworld.lib import reg, aux, util, plot
from larvaworld.lib.screen.drawing import ScreenManager
from larvaworld.lib.model.envs.conditions import get_exp_condition
from larvaworld.lib.sim.base_run import BaseRun
import larvaworld

__all__ = [
    'ExpRun',
]

class ExpRun(BaseRun):
    def __init__(self,experiment=None,parameters=None, screen_kws={},parameter_dict={}, **kwargs):
        '''
        Simulation mode 'Exp' launches a single simulation of a specified experiment type.

        Args:
            **kwargs: Arguments passed to the setup method

        '''

        super().__init__(runtype = 'Exp',experiment=experiment,parameters=parameters, **kwargs)

        self.screen_kws = screen_kws
        self.parameter_dict = parameter_dict

    @property
    def configuration_text(self):
        c = self.p
        pref0 = '     '
        text = f"Simulation configuration : \n" \
               f"{pref0}Simulation mode : {c.runtype}\n" \
               f"{pref0}Experiment : {c.experiment}\n" \
               f"{pref0}Simulation ID : {c.id}\n" \
               f"{pref0}Duration (min) : {c.duration}\n" \
               f"{pref0}Timestep (sec) : {c.dt}\n" \
               f"{pref0}Ticks (#) : {c.Nsteps}\n" \
               f"{pref0}Box2D active : {c.Box2D}\n" \
 \
               f"{pref0}Offline mode : {c.offline}\n" \
               f"{pref0}Data storage : {c.store_data}\n" \
               f"{pref0}Parent path : {c.dir}"
        return text

    def setup(self):

        self.sim_epochs = self.p.trials
        for idx, ep in self.sim_epochs.items():
            ep['start'] = int(ep['start'] * 60 / self.dt)
            ep['stop'] = int(ep['stop'] * 60 / self.dt)

        self.build_env(self.p.env_params)

        self.build_agents(self.p.larva_groups, self.parameter_dict)
        self.set_collectors(self.p.collections)
        self.accessible_sources = None

        self.screen_manager = ScreenManager(model=self, **self.screen_kws)


        if not self.larva_collisions:
            self.eliminate_overlap()

        k = get_exp_condition(self.experiment)
        self.exp_condition = k(self) if k is not None else None

        # self.report(['source_xy'])


    @property
    def end_condition_met(self):
        if self.exp_condition is not None:
            return self.exp_condition.check(self)
        return False

    # @profile
    def sim_step(self):
        """ Proceeds the simulation by one step, incrementing `Model.t` by 1
        and then calling :func:`Model.step` and :func:`Model.update`."""
        if not self.is_paused:
            self.t += 1
            self.step()
            self.update()
            if self.t >= self._steps or self.end_condition_met:
                self.running = False

    # @profile
    def step(self):
        """ Defines the models' events per simulation step. """
        if not self.larva_collisions:
            self.larva_bodies = self.get_larva_bodies()
        for id, layer in self.odor_layers.items():
            layer.update_values()  # Currently doing something only for the DiffusionValueLayer
        if self.windscape is not None:
            self.windscape.update()
        if len(self.sources)>10 :
            self.space.accessible_sources_multi(self.agents)
        self.agents.step()
        if self.Box2D:
            self.space.Step(self.dt, 6, 2)
            self.agents.updated_by_Box2D()

        self.screen_manager.step()

    def update(self):
        """ Record a dynamic variable. """
        self.agents.nest_record(self.collectors['step'])
        # print(self.t)

    def end(self):
        """ Repord an evaluation measure. """
        self.screen_manager.finalize()
        self.agents.nest_record(self.collectors['end'])

    def simulate(self, **kwargs):
        reg.vprint(f'--- Simulation {self.id} initialized!--- ', 1)
        start = time.time()
        self.run(**kwargs)
        self.data_collection = larvaworld.lib.LarvaDatasetCollection.from_agentpy_output(self.output)
        self.datasets=self.data_collection.datasets
        # self.datasets = self.retrieve()
        end = time.time()
        dur = np.round(end - start).astype(int)
        reg.vprint(f'--- Simulation {self.id} completed in {dur} seconds!--- ', 1)
        if self.p.enrichment:
            for d in self.datasets:
                # print(d.step_data)
                # raise
                reg.vprint(f'--- Enriching dataset {d.id} ---', 1)
                d.enrich(**self.p.enrichment, is_last=False)
                reg.vprint(f'--- Dataset {d.id} enriched ---', 1)
                reg.vprint(f'--------------------------------', 1)
        if self.store_data:
            self.store()
        return self.datasets



    # def retrieve(self):
    #     ds = []
    #     for gID, df in self.output.variables.items():
    #         # print(df)
    #         # raise
    #         assert 'sample_id' not in df.index.names
    #         kws = {
    #             'larva_groups': {gID: self.p.larva_groups[gID]},
    #             # 'df': df,
    #             'id': gID,
    #             'dir': f'{self.data_dir}/{gID}'
    #         }
    #         d = self.convert_output_to_dataset(df, **kws)
    #
    #         ds.append(d)
    #
    #
    #     return ds



    def build_agents(self, larva_groups, parameter_dict={}):
        reg.vprint(f'--- Simulation {self.id} : Generating agent groups!--- ', 1)
        confs = util.generate_agentConfs(larva_groups=larva_groups, parameter_dict=parameter_dict)

        self.place_agents(confs)


    def eliminate_overlap(self):
        scale = 3.0
        while self.collisions_exist(scale=scale):
            self.larva_bodies = self.get_larva_bodies(scale=scale)
            for l in self.agents:
                dx, dy = np.random.randn(2) * l.sim_length / 10
                overlap = True
                while overlap:
                    ids = self.detect_collisions(l.unique_id)
                    if len(ids) > 0:
                        l.move_body(dx, dy)
                        self.larva_bodies[l.unique_id] = l.get_polygon(scale=scale)
                    else:
                        break

    def collisions_exist(self, scale=1.0):
        self.larva_bodies = self.get_larva_bodies(scale=scale)
        for l in self.agents:
            ids = self.detect_collisions(l.unique_id)
            if len(ids) > 0:
                return True
        return False

    def detect_collisions(self, id):
        ids = []
        for id0, body0 in self.larva_bodies.items():
            if id0==id :
                continue
            if self.larva_bodies[id].intersects(body0):
                ids.append(id0)
        return ids

    def get_larva_bodies(self, scale=1.0):
        return {l.unique_id: l.get_shape(scale=scale) for l in self.agents}

    def analyze(self, **kwargs):
        os.makedirs(self.plot_dir, exist_ok=True)
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
            self.results = {'PIs': PIs, 'PI2s': PI2s}
            return

        if 'disp' in exp:
            samples = aux.unique_list([d.config.sample for d in ds])
            ds += [reg.loadRef(sd) for sd in samples if sd is not None]
        graphgroups = reg.graphs.get_analysis_graphgroups(exp, self.p.source_xy)
        self.figs = reg.graphs.eval_graphgroups(graphgroups, datasets=ds, save_to=self.plot_dir, **kwargs)

    def store(self):
        self.output.save(**self.p.agentpy_output_kws)
        os.makedirs(self.data_dir, exist_ok=True)
        for d in self.datasets:
            d.save()
            for type, vs in d.larva_dicts.items():
                aux.storeSoloDics(vs, path=f'{d.dir}/data/individuals/{type}.txt')

    def load_agentpy_output(self):
        df=agentpy.DataDict.load(**self.p.agentpy_output_kws)
        df1 = pd.concat(df.variables, axis=0).droplevel(1, axis=0)
        df1.index.rename('Model', inplace=True)
        return df1

