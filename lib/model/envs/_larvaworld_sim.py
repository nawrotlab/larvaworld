import random
import time

import numpy as np
from mesa.datacollection import DataCollector

import lib.aux.dictsNlists as dNl
from lib.aux.xy_aux import generate_xy_distro
from lib.aux.colsNstr import N_colors

from lib.conf.base.par import CompGroupCollector
from lib.model.envs._larvaworld import LarvaWorld, generate_larvae, get_sample_bout_distros, sample_group
from lib.sim.single.conditions import get_exp_condition
from lib.conf.base import paths


class LarvaWorldSim(LarvaWorld):
    def __init__(self, output=None, trials={}, parameter_dict={}, **kwargs):
        super().__init__(**kwargs)
        self.sim_epochs = trials
        for idx, ep in self.sim_epochs.items():
            ep['start'] = int(ep['start'] * 60 / self.dt)
            ep['stop'] = int(ep['stop'] * 60 / self.dt)

        self.odor_ids = get_all_odors(self.larva_groups, self.env_pars.food_params)
        self.foodtypes = get_all_foodtypes(self.env_pars.food_params)
        self._place_food(self.env_pars.food_params)
        self.create_larvae(larva_groups=self.larva_groups, parameter_dict=parameter_dict)
        if self.env_pars.odorscape is not None:
            self.Nodors, self.odor_layers = self._create_odor_layers(self.env_pars.odorscape)

        self.add_screen_texts(list(self.odor_layers.keys()), color=self.scale_clock_color)

        self.create_collectors(output)

        if not self.larva_collisions:
            self.eliminate_overlap()

        k = get_exp_condition(self.experiment)
        self.exp_condition = k(self) if k is not None else None

    def _create_odor_layers(self, pars):
        from lib.model.envs._space import DiffusionValueLayer, GaussianValueLayer
        sources = self.get_food() + self.get_flies()
        ids = dNl.unique_list([s.odor_id for s in sources if s.odor_id is not None])
        N = len(ids)
        cols = N_colors(N, as_rgb=True)
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
                'space_range': self.space_edges_for_screen,
            }
            if pars.odorscape == 'Diffusion':
                layers[id] = DiffusionValueLayer(dt=self.dt, scaling_factor=self.scaling_factor,
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
            mod, sample = gConf['model'], gConf['sample']
            if type(sample) == str:
                from lib.conf.stored.conf import loadConf
                sample = loadConf(sample, 'Ref')
            mod = get_sample_bout_distros(mod, sample)

            modF = dNl.flatten_dict(mod)
            sample_ks = [p for p in modF if modF[p] == 'sample']
            RefPars = dNl.load_dict(paths.path('ParRef'), use_pickle=False)
            invRefPars = {v: k for k, v in RefPars.items()}

            if gConf['imitation'] and sample != {}:
                self.sample_ps = list(invRefPars.values())
                ids, ps, ors, sample_dict = imitate_group(sample, self.sample_ps, N=gConf['distribution']['N'])
                ids = [f'{gID}_{id}' for id in ids]
                N = len(ids)
            else:
                self.sample_ps = [invRefPars[p] for p in sample_ks]
                d = gConf['distribution']
                N = d['N']
                ids = [f'{gID}_{i}' for i in range(N)]
                a1, a2 = np.deg2rad(d['orientation_range'])
                ors = np.random.uniform(low=a1, high=a2, size=N).tolist()
                ps = generate_xy_distro(N=N, **{k: d[k] for k in ['mode', 'shape', 'loc', 'scale']})
                sample_dict = sample_group(sample, N, self.sample_ps) if len(self.sample_ps) > 0 else {}
            sample_dict.update(parameter_dict)
            all_pars = generate_larvae(N, sample_dict, mod, RefPars)
            for id, p, o, pars in zip(ids, ps, ors, all_pars):
                l = self.add_larva(pos=p, orientation=o, id=id, pars=pars, group=gID, odor=gConf['odor'],
                                   default_color=gConf['default_color'], life_history=gConf['life_history'])

    def step(self):
        # t0=[]
        # t0.append(time.time())
        self.sim_clock.tick_clock()
        if not self.larva_collisions:
            self.larva_bodies = self.get_larva_bodies()
        # Update value_layers
        for id, layer in self.odor_layers.items():
            layer.update_values()  # Currently doing something only for the DiffusionValueLayer
        if self.windscape is not None:
            self.windscape.update()
        # t0.append(time.time())
        self.active_larva_schedule.step()
        self.active_food_schedule.step()
        if self.Box2D:
            self.space.Step(self.dt, self._sim_velocity_iterations, self._sim_position_iterations)
            for fly in self.get_flies():
                fly.update_trajectory()
        # t0.append(time.time())
        if self.larva_step_col is not None:
            self.larva_step_col.collect(self)
        if self.step_group_collector is not None:
            self.step_group_collector.collect()

        if self.exp_condition is not None:
            self.exp_condition.check(self)
        self.Nticks += 1

        # t0.append(time.time())
        # print(np.array(np.diff(t0)*100000).astype(int))

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
        from lib.conf.base.par import ParDict
        from lib.aux.collecting import TargetedDataCollector, collection_dict
        self.par_dict = ParDict(mode='load').dict
        kws0 = {'par_dict': self.par_dict}
        kws = {
            'objects': self.get_flies(),
            'common': True,
            'collection_dict': collection_dict,
            **kws0
        }

        if output is None:
            output = {'step': [], 'end': [], 'tables': {}, 'step_groups': [], 'end_groups': []}
        s, e, t = output['step'], output['end'], output['tables']
        sg, eg = output['step_groups'], output['end_groups']
        self.larva_step_col = TargetedDataCollector(schedule=self.active_larva_schedule, pars=s, **kws0) if len(
            s) > 0 else None
        self.larva_end_col = TargetedDataCollector(schedule=self.active_larva_schedule, pars=e, **kws0) if len(
            e) > 0 else None
        self.food_end_col = TargetedDataCollector(schedule=self.all_food_schedule,
                                                  pars=['initial_amount', 'final_amount'], **kws0)
        self.table_collector = DataCollector(tables=t) if len(t) > 0 else None
        self.step_group_collector = CompGroupCollector(names=sg, save_as='step.csv', **kws) if len(sg) > 0 else None
        self.end_group_collector = CompGroupCollector(names=eg, save_as='end.csv', **kws) if len(eg) > 0 else None
        # print(e)

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


def imitate_group(config, sample_pars=[], N=None):
    from lib.stor.larva_dataset import LarvaDataset
    d = LarvaDataset(config['dir'], load_data=False)
    e = d.read('end')
    ids = e.index.values.tolist()
    sample_pars = [p for p in sample_pars if p in e.columns]

    if N is not None:
        ids = random.sample(ids, N)
    ps = [tuple(e[['initial_x', 'initial_y']].loc[id].values) for id in ids]
    try:
        ors = [e['initial_front_orientation'].loc[id] for id in ids]
    except:
        ors = np.random.uniform(low=0, high=2 * np.pi, size=len(ids)).tolist()
    dic = {p: [e[p].loc[id] for id in ids] for p in sample_pars}
    return ids, ps, ors, dic


def get_all_odors(larva_groups, food_params):
    lg = [conf.odor.odor_id for conf in larva_groups.values()]
    su = [conf.odor.odor_id for conf in food_params.source_units.values()]
    sg = [conf.odor.odor_id for conf in food_params.source_groups.values()]
    ids = dNl.unique_list([id for id in lg + su + sg if id is not None])
    return ids


def get_all_foodtypes(food_params):
    sg = {k: v.default_color for k, v in food_params.source_groups.items()}
    su = {conf.group: conf.default_color for conf in food_params.source_units.values()}
    gr = {
        food_params.food_grid.unique_id: food_params.food_grid.default_color} if food_params.food_grid is not None else {}
    ids = {**gr, **su, **sg}
    ks = dNl.unique_list(list(ids.keys()))
    try:
        ids = {k: np.array(ids[k]) / 255 for k in ks}
    except:
        ids = {k: ids[k] for k in ks}
    return ids
