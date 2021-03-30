import numpy as np
import sys

from shapely.geometry import Polygon, LineString

import lib.aux.functions as fun
from lib.model.envs._maze import Maze

######## FOOD PARAMETERS ###########

# -------------------------------------------SPACE MODES----------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------

mesa_space = {'physics_engine': False,
              'scaling_factor': 1.0}

mesa_space_in_mm = {'physics_engine': False,
                    'scaling_factor': 1000.0}

box2d_space = {'physics_engine': True,
               'scaling_factor': 100.0}

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


# --------------------------------------------FOOD PARAMETERS----------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------

def food_grid(dim, amount=0.00001):
    return {'food_grid_dims': [dim, dim],
            'food_grid_amount': amount}


def food(r=0.1, amount=0.000001, quality=1.0, odor_id=None, odor_intensity=0.0, odor_spread=0.1):
    return {'radius': r,
            'amount': amount,
            'quality': quality,
            'odor_id': odor_id,
            'odor_intensity': odor_intensity,
            'odor_spread': odor_spread,
            }


########## PLACEMENT PARAMETERS #############
# -----------------------------------LARVA AND FOOD PLACEMENT PARAMETERS------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# def larva_mid(N, s=0.1):
#     return {'initial_num_flies': N,
#             'initial_fly_positions': {'mode': 'normal',
#                                       'loc': (0.0, 0.0),
#                                       'scale': s}}


def food_place(N, place='mid', r=0.7):
    if N == 0:
        mode = None
        loc = None
        scale = None
    if place == 'mid':
        mode = 'defined'
        # loc = (0.0, 0.0)
        loc = [(0.0, 0.0)] * N
        # loc = np.array([(0.0, 0.0)] * N)
        scale = 0.1
    elif place == 'circle':
        mode = 'defined'
        loc = fun.positions_in_circle(r, N)
        scale = None
    elif place == 'uniform':
        mode = 'uniform'
        loc = None
        scale = None
    a = {'initial_num_food': N,
         'initial_food_positions': {'mode': mode,
                                    'loc': loc,
                                    'scale': scale}}
    return a


def food_distro(N, mode='uniform', loc=(0.0, 0.0), scale=0.1, pars={}):
    if N > 0:
        return {'N': N,
                'mode': mode,
                'loc': loc,
                'scale': scale,
                'pars': pars}
    else:
        return None


def larva_distro(N, mode='normal', loc=(0.0, 0.0), scale=0.1, orientation=None):
    if N > 0:
        return {'N': N,
                'mode': mode,
                'loc': loc,
                'scale': scale,
                'orientation': orientation}
    else:
        return None


def no_food():
    return {
        'food_list': {},
        'food_distro': None,
        'food_grid': None,
    }


def chemotax_place(N):
    return {'N': N,
            'mode': 'facing_right',
            'loc': (-0.8, 0.0),
            'scale': 0.03,
            'orientation' : None}


spiral_around_food = {'initial_num_flies': 32,
                      'initial_fly_positions': {'mode': 'spiral'},
                      **food_place(1)}

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

# larva1_food0_facing_up = {'initial_num_flies': 1,
#                           'initial_fly_positions': {'mode': 'identical',
#                                                     'loc': (0.0, 0.0),
#                                                     'orientation': np.pi / 2,
#                                                     'scale': 0.0}}

food_patches = np.array([
    (0.70, 0.07), (0.50, -0.43),
    (0.04, -0.63), (-0.41, -0.46),
    (-0.66, 0.01), (-0.45, 0.50),
    (-0.00, 0.69), (0.45, 0.54)
])
# larva0_food_patchy_8_exp = {**larva_mid(0),
#                             'initial_num_food': 8,
#                             'initial_food_positions': {'mode': 'defined',
#                                                        'loc': food_patches * (1, -1) + (-0.08, 0.08)}}

# larva0_food_patchy_8_exp_inverted_x = {**larva_distro(0),
#                                        'initial_num_food': 8,
#                                        'initial_food_positions': {'mode': 'defined',
#                                                                   'loc': food_patches * (-1, 1)}}

# larva0_food_patchy_8_exp_inverted_xy = {**larva_mid(0),
#                                         'initial_num_food': 8,
#                                         'initial_food_positions': {'mode': 'defined',
#                                                                    'loc': food_patches * (-1, -1)}}
#
# larva0_food_patchy_8_exp_inverted_y = {**larva_mid(0),
#                                        'initial_num_food': 8,
#                                        'initial_food_positions': {'mode': 'defined',
#                                                                   'loc': food_patches * (1, -1)}}

# larva20_food_patchy_9 = {**larva_mid(20),
#                          'initial_num_food': 9,
#                          'initial_food_positions': {'mode': 'defined',
#                                                     'loc': np.array([(-0.7, -0.7), (0.0, -0.7), (0.7, -0.7),
#                                                                      (-0.7, 0.7), (0.0, 0.7), (0.7, 0.7),
#                                                                      (-0.7, 0.0), (0.0, 0.0), (0.7, 0.0)])}}

# def odor(id, intensity=2, spread=0.0002):
#     return {
#         'odor_id': id,
#         'odor_intensity': intensity,
#         'odor_spread': spread,
#     }


####### ODORSCAPE PARAMETERS #################
# ---------------------------------------ODOR LANDSCAPE MODES------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
two_gaussian_odors = {'odor_landscape': 'Gaussian',
                      'odor_id_list': ['CS', 'UCS'],
                      'odor_carriers': 'food',
                      'odor_intensity_list': [1, 2],
                      'odor_spread_list': [0.5, 0.1],
                      'odor_source_allocation': 'iterative'}

one_gaussian_odor = {'odor_landscape': 'Gaussian',
                     'odor_id_list': ['Default_odor_ID'],
                     'odor_carriers': 'food',
                     'odor_intensity_list': [1],
                     'odor_spread_list': [0.5],
                     'odor_source_allocation': 'iterative'}

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

no_odor = {'odor_id_list': None}

chemotax_odor_p = {'odor_landscape': 'Gaussian',
                   'odor_id_list': ['Odor'],
                   'odor_carriers': 'food',
                   # 'odor_intensity_list': [3],
                   # 'odor_spread_list': [0.04],
                   'odor_source_allocation': 'iterative'}

chemotax_odor_np = {'odor_landscape': 'Gaussian',
                    'odor_id_list': ['Odor'],
                    'odor_carriers': 'food',
                    # 'odor_intensity_list': [8],
                    # 'odor_spread_list': [0.0004],
                    'odor_source_allocation': 'iterative'}

chemorbit_odor_p = {'odor_landscape': 'Gaussian',
                    'odor_id_list': ['Odor'],
                    'odor_carriers': 'food',
                    # 'odor_intensity_list': [3 / 16],
                    # 'odor_spread_list': [0.04 / 4],
                    'odor_source_allocation': 'iterative'}

chemorbit_odor_np = {'odor_landscape': 'Gaussian',
                     'odor_id_list': ['Odor'],
                     'odor_carriers': 'food',
                     # 'odor_intensity_list': [2],
                     # 'odor_spread_list': [0.0002],
                     'odor_source_allocation': 'iterative'}

patchy_odor_np = {'odor_landscape': 'Gaussian',
                  'odor_id_list': ['Odor'],
                  'odor_carriers': 'food',
                  # 'odor_intensity_list': [0.5],
                  # 'odor_spread_list': [0.00005],
                  'odor_source_allocation': 'iterative'}

pref_odors = {'odor_landscape': 'Gaussian',
              'odor_id_list': ['CS', 'UCS'],
              'odor_carriers': 'food',
              'odor_source_allocation': 'iterative'}

reorientation_odor_p = {'odor_landscape': 'Gaussian',
                        'odor_id_list': ['Odor'],
                        'odor_carriers': 'food',
                        # 'odor_intensity_list': [3 / 16],
                        # 'odor_spread_list': [0.04 / 4],
                        'odor_source_allocation': 'iterative'}

reorientation_odor_np = {'odor_landscape': 'Gaussian',
                         'odor_id_list': ['Odor'],
                         'odor_carriers': 'food',
                         # 'odor_intensity_list': [0.03 / 16],
                         # 'odor_spread_list': [0.0004 / 4],
                         'odor_source_allocation': 'iterative'}


# def odor(id, intensity=2, spread=0.0002):
#     return {'id': id,
#             'intensity': intensity,
#             'spread': spread}


########### ARENA PARAMETERS ################

def dish(r):
    return {'arena_xdim': r,
            'arena_ydim': r,
            'arena_shape': 'circular'}


def arena(x, y):
    return {'arena_xdim': x,
            'arena_ydim': y,
            'arena_shape': 'rectangular'}


def maze(nx=15, ny=15, ix=0, iy=0, h=0.1):
    # Maze dimensions (ncols, nrows)
    # nx, ny = 15, 15
    # Maze entry pos
    # ix, iy = 0, 0

    m = Maze(nx, ny, ix, iy, height=h)
    m.make_maze()
    lines = m.maze_lines()
    return lines


# maze0=[LineString([(0, 0.01), (0.02, 0.01), (0.02, 0.03), (-0.01, 0.03), (-0.01, -0.01)]),
#        LineString([(0, 0.01), (0.02, 0.01), (0.02, 0.03), (-0.01, 0.03), (-0.01, -0.01)]),
#        LineString([(0, 0.01), (0.02, 0.01), (0.02, 0.03), (-0.01, 0.03), (-0.01, -0.01)])]
# maze_polygons=[Polygon([(0, 0), (0, 0.01), (0.02, 0.01), (0.02, 0.03), (-0.01, 0.03), (-0.01, -0.01),])]

########## FULL EXPERIMENT PARAMETERS #########################

def pref_conf(N, dish_r=0.1):
    conf = {'arena_params': dish(dish_r),
            'food_params': {
                'food_distro': None,
                'food_grid': None,
                'food_list': {'CS_source': {
                    # 'unique_id': 'CS+ source',
                    'pos': (-0.04, 0.0),
                    'amount': 0.0,
                    'quality': 1.0,
                    'radius': 0.003,
                    'odor_id': 'CS',
                    'odor_intensity': 2,
                    'odor_spread': 0.001},
                    'UCS': {
                        # 'unique_id': 'CS- source',
                        'pos': (0.04, 0.0),
                        'amount': 0.0,
                        'quality': 1.0,
                        'radius': 0.003,
                        'odor_id': 'UCS',
                        'odor_intensity': 2,
                        'odor_spread': 0.001}}},
            'place_params': larva_distro(N),
            'odor_params': pref_odors}
    return conf


def chemotax_conf(N):
    conf = {'arena_params': arena(0.1, 0.06),
            'food_params': {
                'food_distro': None,
                'food_grid': None,
                'food_list': {'Odor source': {
                    'pos': (0.04, 0.0),
                    'amount': 0.0,
                    'quality': 1.0,
                    'radius': 0.002,
                    'odor_id': 'Odor',
                    'odor_intensity': 8,
                    'odor_spread': 0.0004}}},
            'place_params': chemotax_place(N),
            'odor_params': chemotax_odor_np}
    return conf


def chemorbit_conf(N):
    conf = {'arena_params': arena(0.1, 0.06),
            'food_params': {
                'food_distro': None,
                'food_grid': None,
                'food_list': {'Odor source': {
                    'pos': (0.0, 0.0),
                    'amount': 0.0,
                    'quality': 1.0,
                    'radius': 0.002,
                    'odor_id': 'Odor',
                    'odor_intensity': 2,
                    'odor_spread': 0.0002}}},
            # 'food_params': food(0.002, amount=0, odor_intensity=2, odor_spread=0.0002),
            'place_params': larva_distro(N, scale=0.0),
            'odor_params': chemorbit_odor_np}
    return conf


def maze_conf(N, n):
    conf = {'arena_params': arena(0.1, 0.1),
            'border_list': {
                'Maze' : {
                    # 'unique_id': 'Maze',
                    'lines': maze(nx=n, ny=n, h=0.1),
                    'from_screen': False}
            },
            'food_params': {
                'food_list': {},
                'food_grid': None,
                'food_distro': food_distro(1, mode='normal', loc=(0.0, 0.0), scale=0.1, pars=food(0.002))},
            'place_params': chemotax_place(N),
            'odor_params': chemorbit_odor_np}
    return conf


def disp_conf(N, dish_r=0.3):
    conf = {'arena_params': dish(dish_r),
            'food_params': no_food(),
            'place_params': larva_distro(N, scale=0.0),
            'odor_params': None}
    return conf


pref_exp_np = pref_conf(25)

chemotax_exp_np = chemotax_conf(30)

chemorbit_exp_np = chemorbit_conf(30)

maze_exp_np = maze_conf(1, 15)

disp_exp_np = disp_conf(30, 0.2)

dish_exp_np = {'arena_params': dish(0.15),
               'food_params': no_food(),
               'place_params': larva_distro(25),
               'odor_params': None}

# reorientation_exp_p = {'arena_params': dish(0.2),
#                        'food_params': food(0.01, amount=0, odor_intensity=8, odor_spread=0.0004),
#                        'place_params': reorientation_place_params,
#                        'odor_params': reorientation_odor_p}

reorientation_exp_np = {'arena_params': dish(0.1),
                        'food_params': {'food_distro': None,
                                        'food_grid': None,
                                        'food_list': {'Odor_source': {'pos': (0.0, 0.0),
                                                                      **food(0.01, amount=0, odor_intensity=8,
                                                                             odor_spread=0.0004)}}
                                        },
                        'place_params': larva_distro(200, 'uniform_circ', scale=None),
                        'odor_params': reorientation_odor_np}

imitation_exp_p = {'arena_params': dish(0.15),
                   'food_params': no_food(),
                   'place_params': larva_distro(25),
                   'odor_params': None}

focus_exp_np = {'arena_params': dish(0.02),
                'food_params': no_food(),
                'place_params': larva_distro(1, 'identical', scale=0.0),
                'odor_params': None}

focus_exp_p = {'arena_params': dish(0.02),
               'food_params': no_food(),
               'place_params': larva_distro(1, 'identical', scale=0.0),
               'odor_params': None}

feed_grid_exp_np = {'arena_params': arena(0.05, 0.05),
                    'food_params': {
                        'food_distro': None,
                        'food_list': {},
                        'food_grid': food_grid(50)},
                    'place_params': larva_distro(1),
                    'odor_params': None}

feed_scatter_exp_np = {'arena_params': arena(0.05, 0.05),
                       'food_params': {
                           'food_distro': food_distro(10000, 'uniform', None, None, food(0.0003)),
                           'food_list': {},
                           'food_grid': None},
                       'place_params': larva_distro(20, scale=0.3),
                       'odor_params': None}

feed_patchy_exp_np = {'arena_params': arena(0.2, 0.2),
                      'food_params': {
                          'food_distro': food_distro(8, 'circle', None, 0.7,
                                                     food(0.0025, amount=0.001, odor_intensity=8, odor_spread=0.0004)),
                          'food_list': {},
                          'food_grid': None},
                      'place_params': larva_distro(20),
                      'odor_params': chemorbit_odor_np}

# feed_patchy_empirical = {
#     # 'arena_params': arena(0.2, 0.2),
#     'arena_params': arena(0.192512, 0.192512),
#     'food_params': {**no_food(), **food(0.0025, amount=0.001, odor_intensity=8, odor_spread=0.0004)},
#     'place_params': larva0_food_patchy_8_exp,
#     'odor_params': chemorbit_odor_np}

growth_exp_np = {'arena_params': arena(0.015, 0.015),  # dish(0.006),
                 'food_params': {
                     'food_distro': None,
                     'food_list': {},
                     'food_grid': food_grid(50, 10 ** -3)},
                 'place_params': larva_distro(1),
                 'odor_params': None}

mock_env = {'arena_params': dish(0.1),
            'food_params': {
                'food_distro': food_distro(10, 'normal', pars=food()),
                'food_grid': food_grid(50),
                'food_list': {'CS_source': {
                    'pos': (-0.04, 0.0),
                    'amount': 0.0,
                    'quality': 1.0,
                    'radius': 0.003,
                    'odor_id': 'CS',
                    'odor_intensity': 2,
                    'odor_spread': 0.001},
                    'UCS_source': {
                        # 'unique_id': 'CS- source',
                        'pos': (0.04, 0.0),
                        'amount': 0.0,
                        'quality': 1.0,
                        'radius': 0.003,
                        'odor_id': 'UCS',
                        'odor_intensity': 2,
                        'odor_spread': 0.001}}},
            # 'food_params': food(0.005, amount=0, odor_intensity=2, odor_spread=0.001),
            'place_params': larva_distro(25, orientation=np.pi),
            'odor_params': pref_odors}
