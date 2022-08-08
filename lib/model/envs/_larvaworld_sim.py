import random
import time

import numpy as np
import pandas as pd
from mesa.datacollection import DataCollector

from lib.aux.sim_aux import get_sample_ks
from lib.registry.pars import preg

from lib.aux import naming as nam, dictsNlists as dNl, colsNstr as cNs, sim_aux, xy_aux
from lib.model.envs._larvaworld import LarvaWorld
from lib.sim.single.conditions import get_exp_condition


class LarvaWorldSim(LarvaWorld):
    def __init__(self, output=None, trials={}, parameter_dict={}, **kwargs):
        super().__init__(**kwargs)

        self.sim_epochs = trials
        for idx, ep in self.sim_epochs.items():
            ep['start'] = int(ep['start'] * 60 / self.dt)
            ep['stop'] = int(ep['stop'] * 60 / self.dt)

        self._place_food(self.env_pars.food_params)
        self.create_larvae(larva_groups=self.larva_groups, parameter_dict=parameter_dict)
        if self.env_pars.odorscape is not None:
            self.Nodors, self.odor_layers = self._create_odor_layers(self.env_pars.odorscape,
                                                                     sources=self.get_food() + self.get_flies())

        if 'thermoscape' in self.env_pars.keys() and self.env_pars.thermoscape is not None:
            self.Ntemps, self.thermo_layers = self._create_thermo_layers(self.env_pars.thermoscape)

        self.add_screen_texts(list(self.odor_layers.keys()), color=self.scale_clock_color)

        self.create_collectors(output)
        if not self.larva_collisions:
            self.eliminate_overlap()

        k = get_exp_condition(self.experiment)
        self.exp_condition = k(self) if k is not None else None

    def _create_odor_layers(self, pars, sources):
        Xdim, Ydim = self.arena_dims
        s = self.scaling_factor
        dt = self.dt
        from lib.model.envs._space import DiffusionValueLayer, GaussianValueLayer
        # sources = self.get_food() + self.get_flies()
        ids = dNl.unique_list([s.odor_id for s in sources if s.odor_id is not None])
        N = len(ids)
        cols = cNs.N_colors(N, as_rgb=True)
        layers = {}
        for i, (id, c) in enumerate(zip(ids, cols)):
            od_sources = [f for f in sources if f.odor_id == id]
            temp = dNl.unique_list([s.default_color for s in od_sources])
            if len(temp) == 1:
                c0 = temp[0]
            elif len(temp) == 3 and all([type(k) == float] for k in temp):
                c0 = temp
            else:
                c0 = c
            kwargs = {
                'model': self,
                'unique_id': id,
                'sources': od_sources,
                'default_color': c0,
                'space_range': np.array([-Xdim * s / 2, Xdim * s / 2, -Ydim * s / 2, Ydim * s / 2]),
            }
            if pars.odorscape == 'Diffusion':
                layers[id] = DiffusionValueLayer(dt=dt, scaling_factor=s,
                                                 grid_dims=pars['grid_dims'],
                                                 evap_const=pars['evap_const'],
                                                 gaussian_sigma=pars['gaussian_sigma'],
                                                 **kwargs)
            elif pars.odorscape == 'Gaussian':
                layers[id] = GaussianValueLayer(**kwargs)
        # self.refresh_odor_dicts(ids)
        return N, layers

    def create_larvae(self, larva_groups, parameter_dict={}):
        for gID, gConf in larva_groups.items():
            for conf in sim_aux.larvaConfs(gID, gConf, parameter_dict=parameter_dict) :
                l = self.add_larva(**conf)

    def step(self):
        self.sim_clock.tick_clock()
        if not self.larva_collisions:
            self.larva_bodies = self.get_larva_bodies()
        # Update value_layers
        for id, layer in self.odor_layers.items():
            layer.update_values()  # Currently doing something only for the DiffusionValueLayer
        if self.windscape is not None:
            self.windscape.update()
        self.active_larva_schedule.step()
        self.active_food_schedule.step()
        if self.Box2D:
            self.space.Step(self.dt, self._sim_velocity_iterations, self._sim_position_iterations)
            for fly in self.get_flies():
                fly.update_trajectory()
        if self.step_collector is not None:
            self.step_collector.collect(self)
        # if self.step_group_collector is not None:
        #     self.step_group_collector.collect()

        if self.exp_condition is not None:
            self.exp_condition.check(self)
        self.Nticks += 1

    def get_larva_bodies(self, scale=1.0):
        return {l.unique_id: l.get_shape(scale=scale) for l in self.get_flies()}

    def larva_bodies_except(self, id):
        return {k: v for k, v in self.larva_bodies.items() if k != id}

    def detect_collisions(self, id):
        body = self.larva_bodies[id]
        ids = []
        for id0, body0 in self.larva_bodies_except(id).items():
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
        from lib.aux.collecting import TargetedDataCollector
        # kws0 = {'par_dict': preg.dict}
        if output is None:
            output = {'step': [], 'end': [], 'tables': {}}
        s, e, t = output['step'], output['end'], output['tables']

        f = []  # ['initial_amount', 'final_amount']
        self.step_collector = TargetedDataCollector(schedule=self.active_larva_schedule, pars=s) if len(
            s) > 0 else None
        self.end_collector = TargetedDataCollector(schedule=self.active_larva_schedule, pars=e) if len(
            e) > 0 else None
        self.food_collector = TargetedDataCollector(schedule=self.all_food_schedule,
                                                    pars=f) if len(
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

    def refresh_odor_dicts(self, odor_ids):
        for l in self.get_flies():
            for id in odor_ids:
                try:
                    if id not in l.brain.olfactor.gain_ids:
                        l.brain.olfactor.add_novel_gain(id)
                except:
                    pass

    # @todo use _create_thermo_layers
    def _create_thermo_layers(self, pars):
        from lib.model.envs._space import ThermoScape
        # print(pars['thermo_sources'])
        sources = pars['thermo_sources']  # dictionary

        N = 1;
        id = 'temp'
        cols = cNs.N_colors(N, as_rgb=True)
        layers = {}
        plate_temp = pars['plate_temp']  # int/float
        source_temp_diff = pars['thermo_source_dTemps']  # dict
        kwargs = {
            'model': self,
            'unique_id': id,
            'default_color': 'green',
            'space_range': self.space_edges_for_screen,
        }
        kwargs = {}
        tlayers = ThermoScape(pTemp=plate_temp, spread=None, origins=sources, tempDiff=source_temp_diff, **kwargs)
        tlayers.generate_thermoscape()
        return N, tlayers

    def get_larva_dicts(self, ids):
        from lib.model.modules.nengobrain import NengoBrain
        deb_dicts = {}
        nengo_dicts = {}
        bout_dicts = {}
        foraging_dicts = {}
        for l in self.get_flies():
            if l.unique_id in ids:
                if hasattr(l, 'deb') and l.deb is not None:
                    deb_dicts[l.unique_id] = l.deb.finalize_dict()
                elif isinstance(l.brain, NengoBrain):
                    if l.brain.dict is not None:
                        nengo_dicts[l.unique_id] = l.brain.dict
                if l.brain.locomotor.intermitter is not None:
                    bout_dicts[l.unique_id] = l.brain.locomotor.intermitter.build_dict()
                if len(self.foodtypes) > 0:
                    foraging_dicts[l.unique_id] = l.finalize_foraging_dict()
                # self.config.foodtypes = env.foodtypes
        return dNl.NestDict({'deb': deb_dicts, 'nengo': nengo_dicts, 'bouts': bout_dicts,
                             'foraging': foraging_dicts})

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
