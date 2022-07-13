import random
import numpy as np
import os
from typing import Any
from shapely.geometry import Polygon

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

from mesa.space import ContinuousSpace
import lib.aux.dictsNlists as dNl
import lib.aux.sim_aux
import lib.aux.xy_aux


class BaseLarvaWorld:
    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        pygame.init()
        W, H = pygame.display.Info().current_w, pygame.display.Info().current_h
        cls.screen_dim_W, cls.screen_dim_H = int(W * 2 / 3/16)*16, int(H * 2 / 3/16)*16
        """Create a new model object_class and instantiate its RNG automatically."""
        cls._seed = kwargs.get("seed", None)
        # print(cls._seed)
        # raise
        cls.random = random.Random(cls._seed)
        return object.__new__(cls)

    def __init__(self, env_params,  id='unnamed', dt=0.1, Nsteps=None, save_to='.',
                 Box2D=False, experiment=None, larva_collisions=True,larva_groups={}):

        self.experiment = experiment
        self.Box2D = Box2D

        self.is_running = False
        self.is_paused = False
        self.dt = dt
        self.id = id
        self.Nticks = 0
        self.Nsteps = Nsteps
        self.save_to = save_to
        self.larva_collisions = larva_collisions
        self.borders, self.border_xy, self.border_lines, self.border_bodies, self.border_walls = [], [], [], [], []
        os.makedirs(save_to, exist_ok=True)
        self.env_pars = dNl.NestDict(env_params)
        self.larva_groups = dNl.NestDict(larva_groups)
        self.create_arena(**self.env_pars.arena)
        self.space = self.create_space()

        self.odor_aura = False
        self.Nodors, self.odor_layers = 0, {}
        self.food_grid = None
        self.foodtypes = get_all_foodtypes(self.env_pars.food_params)
        self.odor_ids = get_all_odors(self.larva_groups, self.env_pars.food_params)

    def create_arena(self, arena_dims, arena_shape):
        self.arena_dims = X, Y = np.array(arena_dims)
        W0, H0 = self.screen_dim_W, self.screen_dim_H
        R0, R = W0 / H0, X / Y
        self.screen_width, self.screen_height = (W0, int(W0 / R/16)*16) if R0 < R else (int(H0 * R/16)*16, H0)
        self.unscaled_space_edges = np.array([(-X / 2, -Y / 2),
                                              (-X / 2, Y / 2),
                                              (X / 2, Y / 2),
                                              (X / 2, -Y / 2)])
        if arena_shape == 'circular':
            # This is a circle_to_polygon shape from the function
            self.unscaled_tank_shape = lib.aux.sim_aux.circle_to_polygon(60, X / 2)
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
        k = 0.9
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
        ids = {k: list(np.array(ids[k]) / 255) for k in ks}
    except:
        ids = {k: ids[k] for k in ks}
    return ids