import numpy as np
# print('0')
from mesa.datacollection import DataCollector

import lib.aux.dictsNlists
import lib.aux.xy_aux
from lib.aux import colsNstr as fun
from lib.aux.collecting import TargetedDataCollector
# print('1')
from lib.conf.init_dtypes import null_dict
# print('2')
from lib.conf.par import CompGroupCollector
from lib.envs._larvaworld import LarvaWorld, generate_larvae, get_sample_bout_distros, sample_group
# print('3')
from lib.envs._space import DiffusionValueLayer, GaussianValueLayer
from lib.sim.conditions import get_exp_condition
# print('4')

# print('5')
from lib.stor import paths
# print('6')


class LarvaWorldSim(LarvaWorld):
    def __init__(self, output=None, id='Unnamed_Simulation', larva_collisions=True, count_bend_errors=False,
                 life_params=None, sample_dataset='reference', parameter_dict=None, **kwargs):
        super().__init__(id=id, **kwargs)

        if parameter_dict is None:
            parameter_dict = {}

        if life_params is None:
            life_params = null_dict('life')
        elif type(life_params)==str :
            from lib.conf.conf import loadConf
            life_params=loadConf(life_params, 'Life')
        self.epochs = life_params['epochs']
        if self.epochs is None:
            self.epochs = []
        self.hours_as_larva = life_params['hours_as_larva']
        self.substrate_quality = life_params['substrate_quality']
        self.sim_epochs = [
            [np.clip(s0 - self.hours_as_larva, a_min=0, a_max=+np.inf), s1 - self.hours_as_larva] for
            [s0, s1] in self.epochs if s1 > self.hours_as_larva]
        if len(self.sim_epochs) > 0:
            on_ticks = [int(s0 * 60 * 60 / self.dt) for [s0, s1] in self.sim_epochs]
            off_ticks = [int(s1 * 60 * 60 / self.dt) for [s0, s1] in self.sim_epochs]
            self.sim_clock.set_timer(on_ticks, off_ticks)
        self.starvation = self.sim_clock.timer_on
        self.count_bend_errors = count_bend_errors
        self.sample_dataset = sample_dataset

        self.larva_collisions = larva_collisions

        self._place_food(self.env_pars['food_params'])
        self.create_larvae(larva_pars=self.env_pars['larva_groups'], parameter_dict=parameter_dict)
        if self.env_pars['odorscape'] is not None:
            self.Nodors, self.odor_layers = self._create_odor_layers(self.env_pars['odorscape'])
        self.add_screen_texts(list(self.odor_layers.keys()), color=self.scale_clock_color)

        self.create_collectors(output)

        if not self.larva_collisions:
            self.eliminate_overlap()

        k = get_exp_condition(self.experiment)
        self.exp_condition = k(self) if k is not None else None

    def prepare_odor_layer(self, timesteps):
        for i in range(timesteps):
            for id, layer in self.odor_layers.items():
                layer.update_values()  # Currently doing something only for the DiffusionValueLayer

    def _create_odor_layers(self, pars):
        sources = self.get_food() + self.get_flies()
        odor_ids = lib.aux.dictsNlists.unique_list([s.odor_id for s in sources if s.odor_id is not None])
        Nodors = len(odor_ids)
        odor_colors = fun.N_colors(Nodors, as_rgb=True)
        layers = {}
        odorscape = pars['odorscape']
        for i, (id, color) in enumerate(zip(odor_ids, odor_colors)):
            od_sources = [f for f in sources if f.odor_id == id]
            temp = list(set([s.default_color for s in od_sources]))
            default_color = temp[0] if len(temp) == 1 else color
            kwargs = {
                'unique_id': id,
                'sources': od_sources,
                'default_color': default_color,
                'space_range': self.space_edges_for_screen,
            }
            if odorscape == 'Diffusion':
                layers[id] = DiffusionValueLayer(dt=self.dt, scaling_factor=self.scaling_factor,
                                                 grid_dims=pars['grid_dims'],
                                                 evap_const=pars['evap_const'],
                                                 gaussian_sigma=pars['gaussian_sigma'],
                                                 **kwargs)
            elif odorscape == 'Gaussian':
                layers[id] = GaussianValueLayer(**kwargs)
        self.refresh_odor_dicts(odor_ids)
        return Nodors, layers

    def create_larvae(self, larva_pars, parameter_dict={}):
        for gID, gConf in larva_pars.items():

            mod, sample=gConf['model'], gConf['sample']
            if type(sample) == str:
                from lib.conf.conf import loadConf
                sample = loadConf(sample, 'Ref')
            mod=get_sample_bout_distros(mod, sample)

            modF = lib.aux.dictsNlists.flatten_dict(mod)
            sample_ks = [p for p in modF if modF[p] == 'sample']
            RefPars = lib.aux.dictsNlists.load_dict(paths.RefParsFile, use_pickle=False)
            invRefPars = {v: k for k, v in RefPars.items()}
            self.sample_ps=[invRefPars[p] for p in sample_ks]
            if gConf['imitation'] and sample!={}:
                ids, ps, ors, sample_dict = imitate_group(sample, self.sample_ps)
                N=len(ids)
            else:
                d=gConf['distribution']
                N = d['N']
                ids = [f'{gID}_{i}' for i in range(N)]
                a1, a2 = np.deg2rad(d['orientation_range'])
                ors = np.random.uniform(low=a1, high=a2, size=N).tolist()
                ps = lib.aux.xy_aux.generate_xy_distro(N=N, **{k: d[k] for k in ['mode', 'shape', 'loc', 'scale']})
                sample_dict = sample_group(sample, N, self.sample_ps)
            sample_dict.update(parameter_dict)
            all_pars= generate_larvae(N, sample_dict, mod, RefPars)



            for id, p, o, pars in zip(ids, ps, ors, all_pars):
                l = self.add_larva(pos=p, orientation=o, id=id, pars=pars, group=gID,odor=gConf['odor'],
                                   default_color=gConf['default_color'], life=gConf['life'])

    def step(self):
        # Tick sim_clock
        self.sim_clock.tick_clock()
        self.Nticks += 1
        self.resolve_epochs()

        if not self.larva_collisions:
            self.larva_bodies = self.get_larva_bodies()
        # Update value_layers
        for id, layer in self.odor_layers.items():
            layer.update_values()  # Currently doing something only for the DiffusionValueLayer

        for l in self.get_flies():
            # print(l.unique_id)
            l.compute_next_action()
        self.active_larva_schedule.step()
        self.active_food_schedule.step()
        # print('ss')
        if self.Box2D:
            self.space.Step(self.dt, self._sim_velocity_iterations, self._sim_position_iterations)
            for fly in self.get_flies():
                fly.update_trajectory()
        if self.larva_step_col is not None:
            self.larva_step_col.collect(self)
        self.step_group_collector.collect()

        if self.exp_condition is not None:
            self.exp_condition.check(self)

    def space_to_mm(self, array):
        return array * 1000 / self.scaling_factor

    def plot_odorscape(self, save_to=None, show=False):
        from lib.anal.plotting import plot_surface
        for id, layer in self.odor_layers.items():
            title = f'{id} odorscape'
            X, Y = layer.meshgrid
            V = layer.get_grid()
            x = self.space_to_mm(X)
            y = self.space_to_mm(Y)
            plot_surface(x=x, y=y, z=V,
                         labels=[r'x $(mm)$', r'y $(mm)$', r'concentration $(μM)$'], title=title,
                         save_to=save_to, save_as=f'{id}_odorscape_{self.odorscape_counter}', show=show)

    def get_larva_bodies(self, scale=1.0):
        larva_bodies = {}
        for l in self.get_flies():
            larva_bodies[l.unique_id] = l.get_polygon(scale=scale)
        return larva_bodies

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
        if output is None:
            output = {'step': [], 'end': [], 'tables': {}}
        s, e, t = output['step'], output['end'], output['tables']
        sg, eg = output['step_groups'], output['end_groups']
        self.larva_step_col = TargetedDataCollector(schedule=self.active_larva_schedule, pars=s) if len(s) > 0 else None
        self.larva_end_col = TargetedDataCollector(schedule=self.active_larva_schedule, pars=e) if len(e) > 0 else None
        self.food_end_col = TargetedDataCollector(schedule=self.all_food_schedule,
                                                  pars=['initial_amount', 'final_amount'])
        self.table_collector = DataCollector(tables=t) if len(t) > 0 else None
        self.step_group_collector = CompGroupCollector(objects=self.get_flies(), names=sg,
                                                       save_units=True, common=True, save_as='step.csv')
        self.end_group_collector = CompGroupCollector(objects=self.get_flies(), names=eg,
                                                      save_units=True, common=True, save_as='end.csv')

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
                        # overlap=False
                        break

    def resolve_epochs(self):
        if len(self.sim_epochs) > 0:
            self.starvation = self.sim_clock.timer_on
            if self.sim_clock.timer_opened:
                if self.food_grid is not None:
                    self.food_grid.empty_grid()
            if self.sim_clock.timer_closed:
                if self.food_grid is not None:
                    self.food_grid.reset()

    def refresh_odor_dicts(self, odor_ids):
        for l in self.get_flies() :
            for id in odor_ids :
                try :
                    if id not in l.brain.olfactor.odor_ids :
                        l.brain.olfactor.add_novel_odor(id)
                except :
                    pass


def imitate_group(config, sample_pars=[]):
    from lib.stor.larva_dataset import LarvaDataset
    d = LarvaDataset(config['dir'], load_data=False)
    e = d.read('end')
    ids = e.index.values.tolist()
    ps = [tuple(e[['initial_x', 'initial_y']].loc[id].values) for id in ids]
    try:
        ors = [e['initial_front_orientation'].loc[id] for id in ids]
    except:
        ors = np.random.uniform(low=0, high=2 * np.pi, size=len(ids)).tolist()
    dic={p : [e[p].loc[id] for id in ids] for p in sample_pars}
    return ids, ps, ors, dic

