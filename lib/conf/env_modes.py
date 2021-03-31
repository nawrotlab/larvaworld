import numpy as np
import sys

from shapely.geometry import Polygon, LineString

import lib.aux.functions as fun

######## FOOD PARAMETERS ###########

# -------------------------------------------SPACE MODES----------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
from lib.conf import odor

arena_shapes = ['circular', 'rectangular']

larva_place_modes = [
    'normal',
    'defined',
    'identical',
    'uniform',
    'uniform_circ',
    'spiral',
    'facing_right'
]

food_place_modes = [
    'normal',
    'defined',
    'uniform'
]


def food_grid(dim, amount=0.00001):
    return {'food_grid_dims': [dim, dim],
            'food_grid_amount': amount}


def food(r=0.1, amount=0.000001, quality=1.0, **odor_args):
    return {'radius': r,
            'amount': amount,
            'quality': quality,
            **odor_args
            }


def food_distro(N, mode='normal', loc=(0.0, 0.0), scale=0.1, pars={}):
    if N > 0:
        return {'N': N,
                'mode': mode,
                'loc': loc,
                'scale': scale,
                'pars': pars}
    else:
        return None


def food_param_conf(distro=None, list={}, grid=None):
    return {'food_distro': distro,
            'food_grid': grid,
            'food_list': list}


def larva_distro(N, mode='normal', loc=(0.0, 0.0), scale=0.1, orientation=None, group_id=''):
    if N > 0:
        return {group_id: {'N': N,
                           'mode': mode,
                           'loc': loc,
                           'scale': scale,
                           'orientation': orientation}}
    else:
        return None


set_on_xaxis_one_food = {'initial_num_flies': 1 * 8 * 20,
                         'initial_fly_positions': {'mode': 'defined',
                                                   'loc': fun.generate_positions_on_xaxis(num_identical=1 * 8,
                                                                                          num_starting_points=20,
                                                                                          step=0.05, starting_x=-0.5),
                                                   'orientation': fun.generate_orientations(num_identical=1,
                                                                                            circle_parsing=8,
                                                                                            iterations=20)},
                         'initial_num_food': 1,
                         'initial_food_positions': {'mode': 'defined',
                                                    'loc': np.array([(0.5, 0.0)])}}

food_patches = np.array([
    (0.70, 0.07), (0.50, -0.43),
    (0.04, -0.63), (-0.41, -0.46),
    (-0.66, 0.01), (-0.45, 0.50),
    (-0.00, 0.69), (0.45, 0.54)
])

one_diffusion_odor = {'odor_landscape': 'Diffusion',
                      'odor_layer_grid_resolution': [100, 100],
                      'odor_evaporation_rate': 0.9,
                      'odor_diffusion_rate': 0.8,
                      'odor_id_list': ['Default_odor_ID'],
                      'odor_carriers': 'food',
                      'odor_intensity_list': [1],
                      'odor_spread_list': [0.5],
                      'odor_source_allocation': 'iterative'
                      }


def dish(r):
    return {'arena_xdim': r,
            'arena_ydim': r,
            'arena_shape': 'circular'}


def arena(x, y):
    return {'arena_xdim': x,
            'arena_ydim': y,
            'arena_shape': 'rectangular'}


def maze(nx=15, ny=15, ix=0, iy=0, h=0.1):
    from lib.model.envs._maze import Maze
    m = Maze(nx, ny, ix, iy, height=h)
    m.make_maze()
    lines = m.maze_lines()
    return lines


def odor_source(id, pos=(0.0, 0.0), r=0.003, odor_id=None, odor_intensity=2, odor_spread=0.0002, can_be_carried=False):
    if odor_id is None:
        odor_id = f'{id} odor'
    return {id: {'pos': pos,
                 **food(r=r, amount=0.0, **odor(odor_id, odor_intensity, odor_spread), can_be_carried=can_be_carried)}}


CS_UCS_odors = {**odor_source(id='CS_source', pos=(-0.04, 0.0), odor_id='CS'),
                **odor_source(id='UCS_source', pos=(0.04, 0.0), odor_id='UCS')}


def gaussian_odor():
    return {'odor_landscape': 'Gaussian',
            # 'odor_id_list': odor_id_list,
            }


def game_conf(dim=0.1, N=10, x=0.8, y=0.0, scale=0.03):
    x0, y0 = x * dim / 2, y * dim / 2
    return {'arena_params': arena(dim, dim),
            'food_params': food_param_conf(list={
                **odor_source('Flag', can_be_carried=True),
                **odor_source('Left base', (-x0, y0)),
                **odor_source('Right base', (+x0, y0))}),
            'place_params': {
                **larva_distro(N, loc=(-x, y), scale=scale, group_id='Left'),
                **larva_distro(N, loc=(+x, y), scale=scale, group_id='Right')
            },
            'odor_params': gaussian_odor()
            }


game_exp = game_conf()


def maze_conf(N, n):
    conf = {'arena_params': arena(0.1, 0.1),
            'border_list': {
                'Maze': {
                    'lines': maze(nx=n, ny=n, h=0.1),
                    'from_screen': False}
            },
            'food_params': food_param_conf(list={**odor_source('Target')}),
            'place_params': larva_distro(N, mode='facing_right', loc=(-0.8, 0.0), scale=0.3),
            'odor_params': gaussian_odor()}
    return conf


pref_exp_np = {'arena_params': dish(0.1),
               'food_params': food_param_conf(list=CS_UCS_odors),
               'place_params': larva_distro(25),
               'odor_params': gaussian_odor()}

chemotax_exp_np = {'arena_params': arena(0.1, 0.06),
                   'food_params': food_param_conf(list={**odor_source(id='Odor source', pos=(0.04, 0.0),
                                                                      odor_id='Odor', odor_intensity=8,
                                                                      odor_spread=0.0004)}),
                   'place_params': larva_distro(30, mode='facing_right', loc=(-0.8, 0.0), scale=0.05),
                   'odor_params': gaussian_odor()}

chemorbit_exp_np = {'arena_params': arena(0.1, 0.06),
                    'food_params': food_param_conf(list={**odor_source(id='Odor source', odor_id='Odor')}),
                    'place_params': larva_distro(30, scale=0.0),
                    'odor_params': gaussian_odor()
                    }

maze_exp_np = maze_conf(1, 15)

disp_exp_np = {'arena_params': dish(0.2),
               'food_params': food_param_conf(),
               'place_params': larva_distro(30, scale=0.0),
               'odor_params': None}

dish_exp_np = {'arena_params': dish(0.15),
               'food_params': food_param_conf(),
               'place_params': larva_distro(25),
               'odor_params': None}

reorientation_exp_np = {'arena_params': dish(0.1),
                        'food_params': food_param_conf(list={**odor_source(id='Odor source', odor_id='Odor')}),
                        'place_params': larva_distro(200, 'uniform_circ', scale=None),
                        'odor_params': gaussian_odor()}

imitation_exp_p = {'arena_params': dish(0.15),
                   'food_params': food_param_conf(),
                   'place_params': larva_distro(25),
                   'odor_params': None}

focus_exp_np = {'arena_params': dish(0.02),
                'food_params': food_param_conf(),
                'place_params': larva_distro(1, 'normal', scale=0.3, orientation=np.pi / 2),
                'odor_params': None}

feed_grid_exp_np = {'arena_params': arena(0.05, 0.05),
                    'food_params': food_param_conf(grid=food_grid(50)),
                    'place_params': larva_distro(1),
                    'odor_params': None}

feed_scatter_exp_np = {'arena_params': arena(0.05, 0.05),
                       'food_params': food_param_conf(distro=food_distro(10000, 'uniform', None, None, food(0.0003,**odor()))),
                       'place_params': larva_distro(20, scale=0.3),
                       'odor_params': None}

feed_patchy_exp_np = {'arena_params': arena(0.2, 0.2),
                      'food_params': food_param_conf(distro=food_distro(8, 'circle', None, 0.7,
                                                     food(0.0025, amount=0.001, odor_id='Odor', odor_intensity=8,
                                                          odor_spread=0.0004))),
                      'place_params': larva_distro(20),
                      'odor_params': gaussian_odor()}

growth_exp_np = {'arena_params': arena(0.015, 0.015),  # dish(0.006),
                 'food_params': food_param_conf(grid=food_grid(50, 10 ** -3)),
                 'place_params': larva_distro(1),
                 'odor_params': None}

mock_env = {'arena_params': dish(0.1),
            'food_params': {
                'food_distro': food_distro(10, 'normal', pars=food(**odor(id='CS', intensity=2.0))),
                'food_grid': food_grid(50),
                'food_list': CS_UCS_odors
            },
            'place_params': larva_distro(25, orientation=np.pi),
            'odor_params': gaussian_odor()}
