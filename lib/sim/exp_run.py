import os
import time
import agentpy
import numpy as np

from lib import reg, aux, util, plot
from lib.screen.drawing import ScreenManager
from lib.model import envs, agents
from lib.model.envs.conditions import get_exp_condition
from lib.sim.base_run import BaseRun


class ExpRun(BaseRun):
    def __init__(self, **kwargs):
        super().__init__(runtype = 'Exp', **kwargs)

    def setup(self, screen_kws={}, parameter_dict={}, larva_collisions=True):
        self._steps = self.Nsteps




        self.sim_epochs = self.p.trials
        for idx, ep in self.sim_epochs.items():
            ep['start'] = int(ep['start'] * 60 / self.dt)
            ep['stop'] = int(ep['stop'] * 60 / self.dt)


        self.build_env(self.p.env_params)



        self.place_agents(self.p.larva_groups, parameter_dict)
        self.collectors = reg.get_reporters(collections=self.p.collections, agents=self.agents)
        # self.experiment = self.p.experiment


        self.screen_manager = ScreenManager(model=self, **screen_kws)

        self.larva_collisions = larva_collisions
        if not self.larva_collisions:
            self.eliminate_overlap()

        k = get_exp_condition(self.experiment)
        self.exp_condition = k(self) if k is not None else None

    def build_env(self, env_params):
        # Define environment
        self.env_pars = env_params

        self.space = envs.Arena(self, **env_params.arena)
        self.arena_dims = self.space.dims

        self.place_obstacles(env_params.border_list)
        self.place_food(**env_params.food_params)

        '''
        Sensory landscapes of the simulation environment arranged per modality
        - Olfactory landscapes : odorscape
        - Wind landscape : windscape
        - Temperature landscape : thermoscape
        '''
        self.odor_ids = aux.get_all_odors(self.p.larva_groups, env_params.food_params)
        self.odor_layers = envs.create_odor_layers(model=self, sources=self.sources, pars=env_params.odorscape)
        self.windscape = envs.WindScape(model=self, **env_params.windscape) if env_params.windscape else None
        self.thermoscape = envs.ThermoScape(**env_params.thermoscape) if env_params.thermoscape else None


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
            for fly in self.agents:
                fly.update_trajectory()
        self.screen_manager.step()

    def update(self):
        """ Record a dynamic variable. """
        self.agents.nest_record(self.collectors['step'])

    def end(self):
        """ Repord an evaluation measure. """
        self.screen_manager.finalize(self.t)
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
                d.enrich(**self.p.enrichment, is_last=False, store=self.store_data)
                reg.vprint(f'--- Dataset {d.id} enriched ---', 2)
        if self.store_data:
            os.makedirs(self.data_dir, exist_ok=True)
            self.store()
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
            b = envs.Border(unique_id=id, **pars)
            self.borders.append(b)
            self.border_lines += b.border_lines

    def place_food(self, food_grid=None, source_groups={}, source_units={}):
        self.food_grid = envs.FoodGrid(**food_grid, model=self) if food_grid else None
        sourceConfs = util.generate_sourceConfs(source_groups, source_units)
        source_list = [agents.Food(model=self, **conf) for conf in sourceConfs]
        self.space.add_agents(source_list, positions=[a.pos for a in source_list])
        self.sources = agentpy.AgentList(model=self, objs=source_list)
        self.foodtypes = aux.get_all_foodtypes(self.env_pars.food_params)
        self.source_xy = aux.get_source_xy(self.env_pars.food_params)

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

        for d in self.datasets:
            d.save()
            for type, vs in d.larva_dicts.items():
                aux.storeSoloDics(vs, path=reg.datapath(type, d.dir))

    @property
    def Nticks(self):
        return self.t

    @property
    def configuration_text(self):
        text = f"Simulation configuration : \n" \
               "\n" \
               f"Experiment : {self.experiment}\n" \
               f"Simulation ID : {self.id}\n" \
               f"Duration (min) : {self.duration}\n" \
               f"Timestep (sec) : {self.dt}\n" \
               f"Plot path : {self.plot_dir}\n" \
               f"Parent path : {self.path}"
        return text



if __name__ == "__main__":
    exp = 'chemorbit'

    m = ExpRun(parameters=reg.expandConf('Exp', exp),
               screen_kws={'vis_kwargs': reg.get_null('visualization', mode=None)})
    ds = m.simulate()
    m.analyze(show=True)