import os
import time
import agentpy
import numpy as np
import pandas as pd


from larvaworld.lib import reg, aux, util, plot
from larvaworld.lib.screen.drawing import ScreenManager
from larvaworld.lib.model import envs, agents
from larvaworld.lib.model.envs.conditions import get_exp_condition
from larvaworld.lib.sim.run_template import BaseRun


class ExpRun(BaseRun):
    def __init__(self, **kwargs):
        super().__init__(runtype = 'Exp', **kwargs)

    def setup(self, screen_kws={}, parameter_dict={}):

        self.sim_epochs = self.p.trials
        for idx, ep in self.sim_epochs.items():
            ep['start'] = int(ep['start'] * 60 / self.dt)
            ep['stop'] = int(ep['stop'] * 60 / self.dt)



        self.odor_ids = aux.get_all_odors(self.p.larva_groups, self.p.env_params.food_params)
        self.build_env(self.p.env_params)

        self.place_agents(self.p.larva_groups, parameter_dict)
        self.collectors = reg.get_reporters(collections=self.p.collections, agents=self.agents)


        self.screen_manager = ScreenManager(model=self, **screen_kws)


        if not self.larva_collisions:
            self.eliminate_overlap()

        k = get_exp_condition(self.experiment)
        self.exp_condition = k(self) if k is not None else None

        self.report(['source_xy'])


    @property
    def end_condition_met(self):
        if self.exp_condition is not None:
            return self.exp_condition.check(self)
        return False

    def sim_step(self):
        """ Proceeds the simulation by one step, incrementing `Model.t` by 1
        and then calling :func:`Model.step` and :func:`Model.update`."""
        if not self.is_paused:
            self.t += 1
            self.step()
            self.update()
            if self.t >= self._steps or self.end_condition_met:
                self.running = False

    def step(self):
        """ Defines the models' events per simulation step. """
        if not self.larva_collisions:
            self.larva_bodies = self.get_larva_bodies()
        for id, layer in self.odor_layers.items():
            layer.update_values()  # Currently doing something only for the DiffusionValueLayer
        if self.windscape is not None:
            self.windscape.update()

        self.agents.step()
        if self.Box2D:
            self.space.Step(self.dt, 6, 2)
            self.agents.updated_by_Box2D()

        self.screen_manager.step(self.t)

    def update(self):
        """ Record a dynamic variable. """
        self.agents.nest_record(self.collectors['step'])

    def end(self):
        """ Repord an evaluation measure. """
        self.screen_manager.finalize(self.t)
        self.agents.nest_record(self.collectors['end'])

    def simulate(self, **kwargs):
        reg.vprint(f'--- Simulation {self.id} initialized!--- ', 1)
        start = time.time()
        self.run(**kwargs)
        self.datasets = self.retrieve()
        end = time.time()
        dur = np.round(end - start).astype(int)
        reg.vprint(f'--- Simulation {self.id} completed in {dur} seconds!--- ', 1)
        if self.p.enrichment:
            for d in self.datasets:
                reg.vprint(f'--- Enriching dataset {d.id} ---', 1)
                d.enrich(**self.p.enrichment, is_last=False, store=self.store_data)
                reg.vprint(f'--- Dataset {d.id} enriched ---', 1)
                reg.vprint(f'--------------------------------', 1)
        if self.store_data:

            self.store()
        return self.datasets

    def retrieve(self):

        dkws=[]
        for gID, df in self.output.variables.items():
            if 'sample_id' in df.index.names :
                sIDs=df.index.get_level_values('sample_id').unique()
                if len(sIDs)>1 :
                    dkws+=[{'gID':gID, 'df':df.xs(sID, level='sample_id'), 'id':f'{gID}_{sID}'} for sID in sIDs]
                else :
                    dkws += [{'gID': gID, 'df': df.xs(sIDs[0], level='sample_id')}]

            else :
                dkws += [{'gID': gID, 'df': df}]
        ds = [self.convert_output_to_dataset(**kws) for kws in dkws]
        return ds

    def convert_output_to_dataset(self,gID, df, id=None):
        if id is None :
            id = gID
        from larvaworld.lib.process.dataset import LarvaDataset
        df.index.set_names(['AgentID', 'Step'], inplace=True)
        df = df.reorder_levels(order=['Step', 'AgentID'], axis=0)
        df.sort_index(level=['Step', 'AgentID'], inplace=True)

        end = df[list(self.collectors['end'].keys())].xs(df.index.get_level_values('Step').max(), level='Step')
        step = df[list(self.collectors['step'].keys())]
        d = LarvaDataset(f'{self.data_dir}/{id}', id=id, larva_groups={gID: self.p.larva_groups[gID]},
                         load_data=False, env_params=self.p.env_params,
                         source_xy=self.source_xy,
                         fr=1 / self.dt)
        d.set_data(step=step, end=end, food=None)
        d.larva_dicts = self.get_larva_dicts(ids=d.agent_ids)
        return d

    def get_larva_dicts(self, ids=None):

        ls = aux.AttrDict({l.unique_id: l for l in self.get_flies(ids=ids)})


        deb_dicts = {}
        nengo_dicts = {}
        bout_dicts = {}
        # foraging_dicts = {}
        for id, l in ls.items():
            if hasattr(l, 'deb') and l.deb is not None:
                deb_dicts[id] = l.deb.finalize_dict()
            try :
                from larvaworld.lib.model.modules.nengobrain import NengoBrain
                if isinstance(l.brain, NengoBrain):
                    if l.brain.dict is not None:
                        nengo_dicts[id] = l.brain.dict
            except :
                pass
            if l.brain.locomotor.intermitter is not None:
                bout_dicts[id] = l.brain.locomotor.intermitter.build_dict()
            # if len(self.foodtypes) > 0:
            #     foraging_dicts[id] = l.finalize_foraging_dict()

        dic0 = aux.AttrDict({'deb': deb_dicts,
                             'nengo': nengo_dicts, 'bouts': bout_dicts,
                             # 'foraging': foraging_dicts
                             })

        dic = aux.AttrDict({k: v for k, v in dic0.items() if len(v) > 0})
        return dic



    def place_agents(self, larva_groups, parameter_dict={}):
        reg.vprint(f'--- Simulation {self.id} : Generating agent groups!--- ', 1)
        agentConfs = util.generate_agentConfs(larva_groups=larva_groups, parameter_dict=parameter_dict)
        if not self.Box2D :
            from larvaworld.lib.model.agents._larva_sim import LarvaSim
            agent_list = [LarvaSim(model=self, **conf) for conf in agentConfs]
        else :
            from larvaworld.lib.model.agents._larva_box2d import LarvaBox2D
            agent_list = [LarvaBox2D(model=self, **conf) for conf in agentConfs]
        self.space.add_agents(agent_list, positions=[a.pos for a in agent_list])
        self.agents = agentpy.AgentList(model=self, objs=agent_list)

    def get_food(self):
        return self.sources

    def get_flies(self, ids=None, group=None):
        ls = self.agents
        if ids is not None:
            ls = [l for l in ls if l.unique_id in ids]
        if group is not None:
            ls = [l for l in ls if l.group == group]
        return ls

    def get_all_objects(self):
        return self.get_food() + self.get_flies() + self.borders

    def eliminate_overlap(self):
        scale = 3.0
        while self.collisions_exist(scale=scale):
            self.larva_bodies = self.get_larva_bodies(scale=scale)
            for l in self.get_flies():
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
        for l in self.get_flies():
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
                # if self.show_output:
                #     print(f'Group {d.id} -> PI : {PIs[d.id]}')
                #     print(f'Group {d.id} -> PI2 : {PI2s[d.id]}')
            self.results = {'PIs': PIs, 'PI2s': PI2s}
            return

        if 'disp' in exp:
            samples = aux.unique_list([d.config.sample for d in ds])
            ds += [reg.loadRef(sd) for sd in samples if sd is not None]
        graphgroups = reg.graphs.get_analysis_graphgroups(exp, self.source_xy)
        self.figs = reg.graphs.eval_graphgroups(graphgroups, datasets=ds, save_to=self.plot_dir, **kwargs)

    def store(self):
        self.output.save(**self.agentpy_output_kws)
        os.makedirs(self.data_dir, exist_ok=True)
        for d in self.datasets:
            d.save()
            for type, vs in d.larva_dicts.items():
                aux.storeSoloDics(vs, path=reg.datapath(type, d.dir))

    def load_agentpy_output(self):
        df=agentpy.DataDict.load(**self.agentpy_output_kws)
        df1 = pd.concat(df.variables, axis=0).droplevel(1, axis=0)
        df1.index.rename('Model', inplace=True)
        return df1