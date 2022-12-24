import time

import numpy as np
import pandas as pd
from mesa.datacollection import DataCollector


from lib import reg
from lib.aux import dictsNlists as dNl, sim_aux, xy_aux, dir_aux, sample_aux
from lib.model.envs.world import World
from lib.model.agents._larva_sim import LarvaSim
from lib.model.envs.collecting import TargetedDataCollector


class WorldSim(World):
    def __init__(self, output=None,  parameter_dict={},larva_groups={}, **kwargs):
        super().__init__(**kwargs)




        self.larva_groups = dNl.NestDict(larva_groups)
        self.odor_ids = get_all_odors(self.larva_groups, self.env_pars.food_params)
        self.create_larvae(larva_groups=self.larva_groups, parameter_dict=parameter_dict)
        self.create_collectors(output)
        self._create_odor_layers(self.get_flies(), self.env_pars.odorscape)




        if not self.larva_collisions:
            self.eliminate_overlap()




    def create_larvae(self, larva_groups, parameter_dict={}):
        for gID, gConf in larva_groups.items():
            d = gConf.distribution
            kws = {
                'm': gConf.model,
                'refID': gConf.sample,
                'Nids': d.N,
                'parameter_dict': parameter_dict,
            }

            if not gConf.imitation:

                ps, ors = xy_aux.generate_xyNor_distro(d)
                ids = [f'{gID}_{i}' for i in range(d.N)]
                all_pars, refID = sample_aux.sampleRef(**kws)






            else:
                ids, ps, ors, all_pars = sample_aux.imitateRef(**kws)
            gConf.ids=ids
            for id, p, o, pars in zip(ids, ps, ors, all_pars):
                conf = {
                    'pos': p,
                    'orientation': o,
                    'unique_id': id,
                    'larva_pars': pars,
                    'group': gID,
                    'odor': gConf.odor,
                    'default_color': gConf.default_color,
                    'life_history': gConf.life_history
                }

                l = self.add_larva(**conf)

    def _add_larva(self, pos):
        gID, gConf = list(self.larva_groups.items())[0]
        kws = {
            'm': gConf.model,
            'refID': gConf.sample,
            'Nids': 1,
        }
        pars, refID = sample_aux.sampleRef(**kws)

        conf = {
            'pos': pos,
            # 'orientation': o,
            'unique_id': f'{gID}_{self.all_larva_schedule.get_agent_count()+1}',
            'larva_pars': pars,
            'group': gID,
            'odor': gConf.odor,
            'default_color': gConf.default_color,
            'life_history': gConf.life_history
        }

        l = self.add_larva(**conf)


    def add_larva(self, pos, **kwargs):
        while not sim_aux.inside_polygon([pos], self.tank_polygon):
            pos = tuple(np.array(pos) * 0.999)


        l = LarvaSim(model=self, pos=pos,**kwargs)
        self.active_larva_schedule.add(l)
        self.all_larva_schedule.add(l)

        return l

    def add_agent(self, agent_class=None, p0=None, p1=None):
        try:
            if agent_class == 'Food':
                f = self.add_food(p0)
            elif agent_class == 'Larva':
                f = self._add_larva(p0)
            elif agent_class == 'Border':
                from lib.model.envs.obstacle import Border
                b = Border(model=self, points=[p1, p0])
                self.add_border(b)
        except:
            pass

    # def step(self):
    #     self.sim_clock.tick_clock()
    #     if not self.larva_collisions:
    #         self.larva_bodies = self.get_larva_bodies()
    #     # Update value_layers
    #     for id, layer in self.odor_layers.items():
    #         layer.update_values()  # Currently doing something only for the DiffusionValueLayer
    #     if self.windscape is not None:
    #         self.windscape.update()
    #     self.active_larva_schedule.step()
    #     self.active_food_schedule.step()
    #     if self.Box2D:
    #         self.space.Step(self.dt, self._sim_velocity_iterations, self._sim_position_iterations)
    #         for fly in self.get_flies():
    #             fly.update_trajectory()
    #     if self.step_collector is not None:
    #         self.step_collector.collect(self)
    #     # if self.step_group_collector is not None:
    #     #     self.step_group_collector.collect()
    #     self.Nticks += 1
    #     if self.exp_condition is not None:
    #         self.end_condition_met=self.exp_condition.check(self)
    #     if self.Nsteps is not None and self.Nticks >= self.Nsteps :
    #         self.end_condition_met = True


    def get_larva_bodies(self, scale=1.0):
        return {l.unique_id: l.get_shape(scale=scale) for l in self.get_flies()}



    def detect_collisions(self, id):

        body = self.larva_bodies[id]
        # body = self.larva_bodies[id]
        ids = []
        for id0, body0 in self.larva_bodies.items():
            if id0==id :
                continue
            if body.intersects(body0):
                ids.append(id0)
        return ids

    def collisions_exist(self, scale=1.0):
        self.larva_bodies = self.get_larva_bodies(scale=scale)
        for l in self.get_flies():
            ids = self.detect_collisions(l.unique_id)
            if len(ids) > 0:
                return True
        return False

    def create_collectors(self, output):


        if output is None:
            output = {'step': [], 'end': [], 'tables': {}}
        s, e, t = output['step'], output['end'], output['tables']

        f = []  # ['initial_amount', 'final_amount']
        self.step_collector = TargetedDataCollector(self.active_larva_schedule, pars=s) if len(
            s) > 0 else None
        self.end_collector = TargetedDataCollector(self.active_larva_schedule, pars=e) if len(
            e) > 0 else None
        self.food_collector = TargetedDataCollector(self.all_food_schedule, pars=f) if len(
            f) > 0 else None
        self.table_collector = DataCollector(tables=t) if len(t) > 0 else None

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

    # def refresh_odor_dicts(self, odor_ids):
    #     for l in self.get_flies():
    #         for id in odor_ids:
    #             try:
    #                 if id not in l.brain.olfactor.gain_ids:
    #                     l.brain.olfactor.add_novel_gain(id)
    #             except:
    #                 pass



    def get_larva_dicts(self, ids=None):

        ls=dNl.NestDict({l.unique_id:l for l in self.get_flies(ids=ids)})

        from lib.model.modules.nengobrain import NengoBrain
        deb_dicts = {}
        nengo_dicts = {}
        bout_dicts = {}
        foraging_dicts = {}
        for id, l in ls.items():
            # id = l.unique_id
            if hasattr(l, 'deb') and l.deb is not None:
                deb_dicts[id] = l.deb.finalize_dict()
            if isinstance(l.brain, NengoBrain):
                if l.brain.dict is not None:
                    nengo_dicts[id] = l.brain.dict
            if l.brain.locomotor.intermitter is not None:
                bout_dicts[id] = l.brain.locomotor.intermitter.build_dict()
            if len(self.foodtypes) > 0:
                foraging_dicts[id] = l.finalize_foraging_dict()
            # self.config.foodtypes = env.foodtypes

        dic0=dNl.NestDict({'deb': deb_dicts,
                      'nengo': nengo_dicts, 'bouts': bout_dicts,
                      'foraging': foraging_dicts})

        dic=dNl.NestDict({k:v for k,v in dic0.items() if len(v)>0})

        # print({k:len(v) for k,v in dic0.items()})
        return dic

    def get_larva_tables(self):
        dic = {}
        if self.table_collector is not None:

            for name, table in self.table_collector.tables.items():
                df = pd.DataFrame(table)
                if 'unique_id' in df.columns:
                    df.rename(columns={'unique_id': 'AgentID'}, inplace=True)
                    N = len(df['AgentID'].unique().tolist())
                    if N > 0:
                        Nrows = int(len(df.index) / N)
                        df['Step'] = np.array([[i] * N for i in range(Nrows)]).flatten()
                        df.set_index(['Step', 'AgentID'], inplace=True)
                        df.sort_index(level=['Step', 'AgentID'], inplace=True)
                        dic[name] = df
        return dNl.NestDict(dic)

    def simulate(self):
        reg.vprint()
        reg.vprint(f'---- Simulating {self.id} ----')
        # Run the simulation
        start = time.time()
        completed = self.run()
        if not completed:
            reg.vprint(f'---- Simulation {self.id} aborted!---- ')
            datasets=None
        else:
            datasets = self.retrieve()
            self.close()
            end = time.time()
            dur = np.round(end - start).astype(int)
            reg.vprint(f'---- Simulation {self.id} completed in {dur} seconds!---- ')
        # self.data=self.datasets
        return datasets

    def retrieve(self):
        # Read the data collected during the simulation
        step = self.step_collector.get_agent_vars_dataframe() if self.step_collector else None
        if self.end_collector is not None:
            self.end_collector.collect(self)
            end = self.end_collector.get_agent_vars_dataframe().droplevel('Step')
        else:
            end = None
        if self.food_collector is not None:
            self.food_collector.collect(self)
            food = self.food_collector.get_agent_vars_dataframe().droplevel('Step')
        else:
            food = None

        ds = dir_aux.split_dataset(step, end, food, env_params=self.env_pars, larva_groups=self.larva_groups,
                                   source_xy=self.source_xy,
                                   fr=1 / self.dt, dir=f'{self.save_to}/{self.id}')
        for d in ds:


            d.larva_dicts = self.get_larva_dicts(ids=d.agent_ids)
            d.larva_tables = self.get_larva_tables()
        return ds


def get_all_odors(larva_groups, food_params):
    lg = [conf.odor.odor_id for conf in larva_groups.values()]
    su = [conf.odor.odor_id for conf in food_params.source_units.values()]
    sg = [conf.odor.odor_id for conf in food_params.source_groups.values()]
    ids = dNl.unique_list([id for id in lg + su + sg if id is not None])
    return ids