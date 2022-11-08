import random
import numpy as np
import os
from typing import Any
from shapely.geometry import Polygon


os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

from mesa.space import ContinuousSpace


from lib.registry import reg
from lib.aux import naming as nam, dictsNlists as dNl, colsNstr as cNs, sim_aux, xy_aux, dir_aux, sample_aux
from lib.model.agents._larva_sim import LarvaSim
from lib.aux.collecting import NamedRandomActivation
# import lib.aux.sim_aux
# import lib.aux.xy_aux


class BaseWorld:
    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        """Create a new model object_class and instantiate its RNG automatically."""
        cls._seed = kwargs.get("seed", None)
        # print(cls._seed)
        # raise
        cls.random = random.Random(cls._seed)
        return object.__new__(cls)

    def __init__(self, env_params,  id='unnamed', dt=0.1, save_to='.',trials={},Nsteps=None,
                 Box2D=False, experiment=None, larva_collisions=True, dur=None):

        self.experiment = experiment
        self.Box2D = Box2D


        self.is_running = False
        self.is_paused = False
        self.dt = dt
        self.id = id
        self.Nticks = 0

        if Nsteps is None and dur is not None :
            Nsteps=int(dur * 60 / self.dt)
        elif Nsteps is not None and dur is None :
            dur=np.round(Nsteps*self.dt/60,2)
        self.duration = dur
        self.Nsteps = Nsteps
        self.save_to = save_to
        self.larva_collisions = larva_collisions

        #
        os.makedirs(save_to, exist_ok=True)
        self.sim_epochs = trials
        for idx, ep in self.sim_epochs.items():
            ep['start'] = int(ep['start'] * 60 / self.dt)
            ep['stop'] = int(ep['stop'] * 60 / self.dt)
        self.env_pars = dNl.NestDict(env_params)

        self.create_arena(**self.env_pars.arena)
        self.space = self.create_space()
        self.borders, self.border_walls, self.border_lines = [], [], []
        self.create_borders(self.env_pars.border_list)


        if 'windscape' in self.env_pars.keys() and self.env_pars.windscape not in [None,{}] :
            from lib.model.envs._space import WindScape
            self.windscape = WindScape(model=self, **self.env_pars.windscape)
        else:
            self.windscape = None
        if 'thermoscape' in self.env_pars.keys() and self.env_pars.thermoscape not in [None,{}]:
            self.Ntemps, self.thermo_layers = self._create_thermo_layers(self.env_pars.thermoscape)



        self.odor_aura = False
        self.odor_layers = {}

        self.food_grid = None
        self.foodtypes = get_all_foodtypes(self.env_pars.food_params)
        self.source_xy = sim_aux.get_source_xy(self.env_pars.food_params)

        self.create_schedules()
        self._place_food(self.env_pars.food_params)
        self._create_odor_layers(self.get_food(), self.env_pars.odorscape)



    def create_arena(self, arena_dims, arena_shape):
        self.arena_dims = X, Y = np.array(arena_dims)

        self.unscaled_space_edges = np.array([(-X / 2, -Y / 2),
                                              (-X / 2, Y / 2),
                                              (X / 2, Y / 2),
                                              (X / 2, -Y / 2)])
        if arena_shape == 'circular':
            # This is a circle_to_polygon shape from the function
            self.unscaled_tank_shape = sim_aux.circle_to_polygon(60, X / 2)
        elif arena_shape == 'rectangular':
            # This is a rectangular shape
            self.unscaled_tank_shape = self.unscaled_space_edges
        # print(self.screen_width, self.screen_height)

    def create_space(self):
        s = self.scaling_factor = 1000.0 if self.Box2D else 1.0
        X, Y = self.space_dims = self.arena_dims * s
        self.space_edges = [(x * s, y * s) for (x, y) in self.unscaled_space_edges]
        self.space_edges_for_screen = np.array([-X / 2, X / 2, -Y / 2, Y / 2])
        self.tank_shape = self.unscaled_tank_shape * s
        k = 0.96
        self.tank_polygon = Polygon(self.tank_shape * k)

        if self.Box2D:
            from Box2D import b2World, b2ChainShape, b2EdgeShape
            self._sim_velocity_iterations = 6
            self._sim_position_iterations = 2

            # create the space in Box2D
            space = b2World(gravity=(0, 0), doSleep=True)

            # create a static body for the space borders
            self.tank = space.CreateStaticBody(position=(.0, .0))
            self.tank.CreateFixture(shape=b2ChainShape(vertices=self.tank_shape.tolist()))
            #     create second static body to attach friction
            self.friction_body = space.CreateStaticBody(position=(.0, .0))
            self.friction_body.CreateFixture(shape=b2ChainShape(vertices=self.space_edges))
        else:
            space = ContinuousSpace(x_min=-X / 2, x_max=X / 2, y_min=-Y / 2, y_max=Y / 2, torus=False)
        return space

    def add_border(self, b):
        b.border_bodies=self.Box2D_border_bodies(b.border_xy)
        self.borders.append(b)
        self.border_lines += b.border_lines
        self.border_walls += b.border_walls

    def create_borders(self, border_list=None):
        if border_list is not None and len(border_list)>0:
            for id, pars in self.env_pars.border_list.items():
                from lib.model.envs._maze import Border
                b = Border(model=self, unique_id=id,scaling_factor=self.scaling_factor, **pars)
                self.add_border(b)
        pass


    def Box2D_border_bodies(self, border_xy):
        from Box2D import b2EdgeShape
        if self.Box2D:
            bs = []
            for xy in border_xy:
                b = self.space.CreateStaticBody(position=(.0, .0))
                b.CreateFixture(shape=b2EdgeShape(vertices=xy.tolist()))
                bs.append(b)
            return bs
        else:
            return []

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

    def _place_food(self, food_pars):
        if food_pars is not None:
            if food_pars.food_grid is not None :
                from lib.model.envs._space import FoodGrid
                self.food_grid = FoodGrid(**food_pars.food_grid, space_range=self.space_edges_for_screen, model=self)
                # self._create_food_grid(space_range=self.space_edges_for_screen,grid_pars=food_pars.food_grid)
            for gID, gConf in food_pars.source_groups.items():
                ps = xy_aux.generate_xy_distro(**gConf.distribution)


                for i, p in enumerate(ps):
                    self.add_food(id=f'{gID}_{i}', pos=p, group=gID, **gConf)

            for id, f_pars in food_pars.source_units.items():
                self.add_food(id=id, **f_pars)

    def add_food(self, pos, id=None, **food_pars):
        from lib.model.agents._source import Food
        f = Food(unique_id=self.next_id(type='Food') if id is None else id, pos=pos, model=self, **food_pars)
        self.active_food_schedule.add(f)
        self.all_food_schedule.add(f)
        return f

    def create_schedules(self):
        self.active_larva_schedule = NamedRandomActivation('active_larva_schedule', self)
        self.all_larva_schedule = NamedRandomActivation('all_larva_schedule', self)
        self.active_food_schedule = NamedRandomActivation('active_food_schedule', self)
        self.all_food_schedule = NamedRandomActivation('all_food_schedule', self)

    def delete_agent(self, agent):
        from lib.model.envs._maze import Border
        from lib.model.agents._source import Food
        if type(agent) is LarvaSim:
            self.active_larva_schedule.remove(agent)
        elif type(agent) is Food:
            self.active_food_schedule.remove(agent)
        elif type(agent) is Border:
            self.borders.remove(agent)
            for l in agent.border_lines:
                self.border_lines.remove(l)
            if len(agent.border_bodies) > 0:
                for b in agent.border_bodies:
                    self.space.delete(b)
            del agent

    def get_flies(self, ids=None, group=None):
        ls = self.active_larva_schedule.agents
        if ids is not None:
            ls = [l for l in ls if l.unique_id in ids]
        if group is not None:
            ls = [l for l in ls if l.group == group]
        return ls

    def get_food(self):
        return self.active_food_schedule.agents

    def get_agents(self, agent_class):
        if agent_class == 'Food':
            return self.get_food()
        elif agent_class == 'Larva':
            return self.get_flies()

    def get_all_objects(self):
        return self.get_food() + self.get_flies() + self.borders

    def next_id(self, type='Food'):
        if type == 'Food':
            N = self.all_food_schedule.get_agent_count()
            return f'Food_{N}'
        elif type == 'Larva':
            N = self.all_larva_schedule.get_agent_count()
            return f'Larva_{N}'

    def _create_odor_layers(self, sources, pars=None):
        if pars is None :
            return
        Xdim, Ydim = self.arena_dims
        s = self.scaling_factor
        dt = self.dt
        from lib.model.envs._space import DiffusionValueLayer, GaussianValueLayer
        # sources = self.get_food() + self.get_flies()
        ids = dNl.unique_list([s.odor_id for s in sources if s.odor_id is not None])
        # layers = {}
        for id in ids:
            od_sources = [f for f in sources if f.odor_id == id]
            temp = dNl.unique_list([s.default_color for s in od_sources])
            if len(temp) == 1:
                c0 = temp[0]
            elif len(temp) == 3 and all([type(k) == float] for k in temp):
                c0 = temp
            else:
                c0 = cNs.random_colors(1)[0]
            kwargs = {
                'model': self,
                'unique_id': id,
                'sources': od_sources,
                'default_color': c0,
                'space_range': np.array([-Xdim * s / 2, Xdim * s / 2, -Ydim * s / 2, Ydim * s / 2]),
            }
            if pars.odorscape == 'Diffusion':
                self.odor_layers[id] = DiffusionValueLayer(dt=dt, scaling_factor=s,
                                                 grid_dims=pars['grid_dims'],
                                                 evap_const=pars['evap_const'],
                                                 gaussian_sigma=pars['gaussian_sigma'],
                                                 **kwargs)
            elif pars.odorscape == 'Gaussian':
                self.odor_layers[id] = GaussianValueLayer(**kwargs)
        # self.refresh_odor_dicts(ids)
        # return layers

    @property
    def configuration_text(self):
        text = f"Simulation configuration : \n" \
               "\n" \
               f"Experiment : {self.experiment}\n" \
               f"Simulation ID : {self.id}\n" \
               f"Duration (min) : {self.duration}\n" \
               f"Timestep (sec) : {self.dt}\n" \
               f"Parent path : {self.save_to}"
        return text

    def move_larvae_to_center(self):
        N = len(self.get_flies())
        orientations = np.random.uniform(low=0.0, high=np.pi * 2, size=N).tolist()
        positions = xy_aux.generate_xy_distro(N=N, mode='uniform', scale=(0.005, 0.015), loc=(0.0, 0.0),
                                              shape='oval')

        for l, p, o in zip(self.get_flies(), positions, orientations):
            temp = np.array([-np.cos(o), -np.sin(o)])
            head = l.head
            head.set_pose(p, o)
            head.update_vertices(p, o)
            for i, seg in enumerate(l.segs[1:]):
                prev_p = l.get_global_rear_end_of_seg(seg_index=i)
                new_p = prev_p + temp * l.seg_lengths[i + 1] / 2
                seg.set_pose(new_p, o)

                seg.set_lin_vel(0.0)
                seg.set_ang_vel(0.0)
            l.pos = l.global_midspine_of_body
            self.space.move_agent(l, l.pos)


def get_all_foodtypes(food_params):
    sg = {k: v.default_color for k, v in food_params.source_groups.items()}
    su = {conf.group: conf.default_color for conf in food_params.source_units.values()}
    gr = {
        food_params.food_grid.unique_id: food_params.food_grid.default_color} if food_params.food_grid is not None else {}
    ids = {**gr, **su, **sg}
    ks = dNl.unique_list(list(ids.keys()))
    try:
        ids = {k: list(np.array(ids[k]) / 255) for k in ks}
    except:
        ids = {k: ids[k] for k in ks}
    return ids