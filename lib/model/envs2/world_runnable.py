import time
import agentpy
import numpy as np

from lib import reg, aux, util, plot
from lib.screen.drawing import ScreenManager
from lib.model import envs2, agents


class Larvaworld(agentpy.Model):
    def setup(self, screen_kws={}, parameter_dict={}, larva_collisions=True, save_to=None):
        """ Initializes the agents and network of the model. """

        id = self.p.sim_params.sim_ID
        if id is None:
            idx = reg.next_idx(self.p.experiment, conftype='Exp')
            id = f'{self.p.experiment}_{idx}'
        self.id = id
        if save_to is None:
            save_to = f'{reg.SIM_DIR}/exp_runs'
        self.dir = f'{save_to}/{self.id}'
        self.plot_dir = f'{self.dir}/plots'
        self.data_dir = f'{self.dir}/data'
        self.save_to = f'{self.dir}/visuals'

        self.is_paused = False
        self.larva_collisions = larva_collisions

        self.dt = self.p.sim_params.timestep
        self.duration = self.p.sim_params.duration
        self.Nsteps = int(self.duration * 60 / self.dt) if self.duration is not None else None
        self._steps = self.Nsteps

        self.experiment = self.p.experiment
        self.env_pars = self.p.env_params
        self.Box2D = self.p.sim_params.Box2D
        self.scaling_factor = 1000.0 if self.Box2D else 1.0

        self.sim_epochs = self.p.trials
        for idx, ep in self.sim_epochs.items():
            ep['start'] = int(ep['start'] * 60 / self.dt)
            ep['stop'] = int(ep['stop'] * 60 / self.dt)

        self.space = envs2.Arena(self, **self.p.env_params.arena)
        self.arena_dims = self.space.dims

        self.place_obstacles(self.p.env_params.border_list)
        self.place_food(**self.p.env_params.food_params)

        '''
        Sensory landscapes of the simulation environment arranged per modality
        - Olfactory landscapes : odorscape
        - Wind landscape : windscape
        - Temperature landscape : thermoscape
        '''
        self.odor_ids = get_all_odors(self.p.larva_groups, self.p.env_params.food_params)
        self.odor_layers = create_odor_layers(model=self, sources=self.sources, pars=self.p.env_params.odorscape)
        self.windscape = envs2.WindScape(model=self,
                                         **self.p.env_params.windscape) if self.p.env_params.windscape else None
        self.thermoscape = envs2.ThermoScape(**self.p.env_params.thermoscape) if self.p.env_params.thermoscape else None

        self.place_agents(self.p.larva_groups, parameter_dict)
        self.collectors = reg.get_reporters(collections=self.p.collections, agents=self.agents)
        self.datasets = None
        self.results = None
        self.figs = {}

        self.screen_manager = ScreenManager(model=self, **screen_kws)

    def step(self):
        """ Defines the models' events per simulation step. """
        for id, layer in self.odor_layers.items():
            layer.update_values()  # Currently doing something only for the DiffusionValueLayer
        if self.windscape is not None:
            self.windscape.update()
        self.agents.step()
        self.screen_manager.step()

    def update(self):
        """ Record a dynamic variable. """
        self.agents.nest_record(self.collectors['step'])

    def end(self):
        """ Repord an evaluation measure. """
        self.agents.nest_record(self.collectors['end'])

    def simulate(self):
        start = time.time()
        self.run()
        self.datasets = self.retrieve()
        end = time.time()
        dur = np.round(end - start).astype(int)
        reg.vprint(f'--- Simulation {self.id} completed in {dur} seconds!--- ', 2)
        if self.p.enrichment:
            for d in self.datasets:
                # reg.vprint()
                # reg.vprint(f'--- Enriching dataset {d.id} with derived parameters ---', 2)
                d.enrich(**self.p.enrichment, is_last=False, store=self.p.sim_params.store_data)
                reg.vprint(f'--- Dataset {d.id} enriched ---', 2)
        return self.datasets

    def retrieve(self):
        from lib.process.dataset import LarvaDataset
        ds = []
        for gID, df in self.output.variables.items():
            df.index.set_names(['AgentID', 'Step'], inplace=True)
            df = df.reorder_levels(order=['Step', 'AgentID'], axis=0)
            df.sort_index(level=['Step', 'AgentID'], inplace=True)

            end = df[list(self.collectors['end'].keys())].xs(df.index.get_level_values('Step').max(), level='Step')
            step = df[list(self.collectors['step'].keys())]
            d = LarvaDataset(f'{self.data_dir}/{gID}', id=gID, larva_groups={gID: self.p.larva_groups[gID]},
                             load_data=False, env_params=self.p.env_params,
                             source_xy=self.source_xy,
                             fr=1 / self.dt)
            d.set_data(step=step, end=end, food=None)
            d.larva_dicts = self.get_larva_dicts(ids=d.agent_ids)
            ds.append(d)
        return ds

    def get_larva_dicts(self, ids=None):

        ls = aux.AttrDict({l.unique_id: l for l in self.get_flies(ids=ids)})

        from lib.model.modules.nengobrain import NengoBrain
        deb_dicts = {}
        nengo_dicts = {}
        bout_dicts = {}
        foraging_dicts = {}
        for id, l in ls.items():
            if hasattr(l, 'deb') and l.deb is not None:
                deb_dicts[id] = l.deb.finalize_dict()
            if isinstance(l.brain, NengoBrain):
                if l.brain.dict is not None:
                    nengo_dicts[id] = l.brain.dict
            if l.brain.locomotor.intermitter is not None:
                bout_dicts[id] = l.brain.locomotor.intermitter.build_dict()
            if len(self.foodtypes) > 0:
                foraging_dicts[id] = l.finalize_foraging_dict()

        dic0 = aux.AttrDict({'deb': deb_dicts,
                             'nengo': nengo_dicts, 'bouts': bout_dicts,
                             'foraging': foraging_dicts})

        dic = aux.AttrDict({k: v for k, v in dic0.items() if len(v) > 0})
        return dic

    def place_obstacles(self, barriers={}):
        self.borders, self.border_lines = [], []
        for id, pars in barriers.items():
            b = envs2.Border(unique_id=id, **pars)
            self.borders.append(b)
            self.border_lines += b.border_lines

    def place_food(self, food_grid=None, source_groups={}, source_units={}):
        self.food_grid = envs2.FoodGrid(**food_grid, model=self) if food_grid else None
        sourceConfs = util.generate_sourceConfs(source_groups, source_units)
        source_list = [agents.Food(model=self, **conf) for conf in sourceConfs]
        self.space.add_agents(source_list, positions=[a.pos for a in source_list])
        self.sources = agentpy.AgentList(model=self, objs=source_list)
        self.foodtypes = get_all_foodtypes(food_grid, source_groups, source_units)
        self.source_xy = get_source_xy(source_groups, source_units)

    def place_agents(self, larva_groups, parameter_dict={}):
        agentConfs = util.generate_agentConfs(larva_groups=larva_groups, parameter_dict=parameter_dict)
        agent_list = [agents.LarvaSim(model=self, **conf) for conf in agentConfs]
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

    def analyze(self, **kwargs):
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


def get_all_foodtypes(grid, groups, units):
    sg = {k: v.default_color for k, v in groups.items()}
    su = {conf.group: conf.default_color for conf in units.values()}
    gr = {
        grid.unique_id: grid.default_color} if grid is not None else {}
    ids = {**gr, **su, **sg}
    ks = aux.unique_list(list(ids.keys()))
    try:
        ids = {k: list(np.array(ids[k]) / 255) for k in ks}
    except:
        ids = {k: ids[k] for k in ks}
    return ids


def get_source_xy(groups, units):
    sources_u = {k: v['pos'] for k, v in units.items()}
    sources_g = {k: v['distribution']['loc'] for k, v in groups.items()}
    return {**sources_u, **sources_g}


def get_all_odors(larva_groups, food_params):
    lg = [conf.odor.odor_id for conf in larva_groups.values()]
    su = [conf.odor.odor_id for conf in food_params.source_units.values()]
    sg = [conf.odor.odor_id for conf in food_params.source_groups.values()]
    ids = aux.unique_list([id for id in lg + su + sg if id is not None])
    return ids


def create_odor_layers(model, sources, pars=None):
    odor_layers = {}
    ids = aux.unique_list([s.odor_id for s in sources if s.odor_id is not None])
    for id in ids:
        od_sources = [f for f in sources if f.odor_id == id]
        temp = aux.unique_list([s.default_color for s in od_sources])
        if len(temp) == 1:
            c0 = temp[0]
        elif len(temp) == 3 and all([type(k) == float] for k in temp):
            c0 = temp
        else:
            c0 = aux.random_colors(1)[0]
        kwargs = {
            'model': model,
            'unique_id': id,
            'sources': od_sources,
            'default_color': c0,
        }
        if pars.odorscape == 'Diffusion':
            odor_layers[id] = envs2.DiffusionValueLayer(grid_dims=pars['grid_dims'],
                                                        evap_const=pars['evap_const'],
                                                        gaussian_sigma=pars['gaussian_sigma'],
                                                        **kwargs)
        elif pars.odorscape == 'Gaussian':
            odor_layers[id] = envs2.GaussianValueLayer(**kwargs)
    return odor_layers


if __name__ == "__main__":
    exp = 'chemorbit'
    conf = reg.expandConf('Exp', exp)
    conf.experiment = exp
    screen_kws = {'vis_kwargs': reg.get_null('visualization', mode=None)}

    m = Larvaworld(parameters=conf, screen_kws=screen_kws)
    ds = m.simulate()
    m.analyze(show=True)