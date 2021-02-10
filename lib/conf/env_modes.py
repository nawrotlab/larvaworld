import numpy as np
import sys

sys.path.insert(0, '../../..')
import lib.aux.functions as fun

######## FOOD PARAMETERS ###########

# -------------------------------------------SPACE MODES----------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------

mesa_space = {'physics_engine': False,
              'scaling_factor': 1.0}

mesa_space_in_mm = {'physics_engine': False,
                    'scaling_factor': 1000.0}

box2d_space = {'physics_engine': True,
               'scaling_factor': 100.0}


# --------------------------------------------FOOD PARAMETERS----------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------

def food_grid(dim, amount=0.00001):
    return {'grid_resolution': [dim, dim],
            'initial_value': amount}


def food(r, amount=0.000001):
    return {'shape_radius': r,
            'amount': amount}


########## PLACEMENT PARAMETERS #############
# -----------------------------------LARVA AND FOOD PLACEMENT PARAMETERS------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

def larva_mid(N, s=0.1):
    return {'initial_num_flies': N,
            'initial_fly_positions': {'mode': 'normal',
                                      'loc': (0.0, 0.0),
                                      'scale': s}}

def food_place(N, place='mid', r=0.7):
    if place=='mid' :
        mode='defined'
        loc=np.array([(0.0, 0.0)]*N)
    elif place=='circle' :
        mode = 'defined'
        loc=fun.positions_in_circle(r, N)
    elif place=='uniform' :
        mode = 'uniform'
        loc=None
    a = {'initial_num_food': N,
         'initial_food_positions': {'mode': mode,
                                    'loc': loc}}
    return a


def pref_place(N):
    return {**larva_mid(N),
            'initial_num_food': 2,
            'initial_food_positions': {'mode': 'defined',
                                       'loc': np.array([(-0.8, 0), (0.8, 0)])}}


def chemotax_place(N):
    return {'initial_num_flies': N,
            'initial_fly_positions': {'mode': 'facing_right',
                                      'loc': (-0.8, 0.0),
                                      'scale': 0.05},
            'initial_num_food': 1,
            'initial_food_positions': {'mode': 'defined',
                                       'loc': np.array([(0.8, 0.0)])}}


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







larva1_food0_facing_up = {'initial_num_flies': 1,
                          'initial_fly_positions': {'mode': 'identical',
                                                    'loc': (0.0, 0.0),
                                                    'orientation': np.pi / 2,
                                                    'scale': 0.0},
                          **food_place(0)}









food_patches = np.array([
    (0.70, 0.07), (0.50, -0.43),
    (0.04, -0.63), (-0.41, -0.46),
    (-0.66, 0.01), (-0.45, 0.50),
    (-0.00, 0.69), (0.45, 0.54)
])
larva0_food_patchy_8_exp = {**larva_mid(0),
                            'initial_num_food': 8,
                            'initial_food_positions': {'mode': 'defined',
                                                       'loc': food_patches* (1, -1) + (-0.08, 0.08)}}

larva0_food_patchy_8_exp_inverted_x = {**larva_mid(0),
                                       'initial_num_food': 8,
                                       'initial_food_positions': {'mode': 'defined',
                                                                  'loc': food_patches * (-1, 1)}}

larva0_food_patchy_8_exp_inverted_xy = {**larva_mid(0),
                                        'initial_num_food': 8,
                                        'initial_food_positions': {'mode': 'defined',
                                                                   'loc': food_patches * (-1, -1)}}

larva0_food_patchy_8_exp_inverted_y = {**larva_mid(0),
                                       'initial_num_food': 8,
                                       'initial_food_positions': {'mode': 'defined',
                                                                  'loc': food_patches * (1, -1)}}

larva20_food_patchy_9 = {**larva_mid(20),
                         'initial_num_food': 9,
                         'initial_food_positions': {'mode': 'defined',
                                                    'loc': np.array([(-0.7, -0.7), (0.0, -0.7), (0.7, -0.7),
                                                                     (-0.7, 0.7), (0.0, 0.7), (0.7, 0.7),
                                                                     (-0.7, 0.0), (0.0, 0.0), (0.7, 0.0)])}}

reorientation_place_params = {'initial_num_flies': 200,
                              'initial_fly_positions': {'mode': 'uniform_circ'},
                              **food_place(1)}

####### ODORSCAPE PARAMETERS #################
# ---------------------------------------ODOR LANDSCAPE MODES------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
two_gaussian_odors = {'odor_landscape': 'Gaussian',
                      'odor_id_list': ['CS+', 'CS-'],
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
                   'odor_id_list': ['Random_odor_ID'],
                   'odor_carriers': 'food',
                   'odor_intensity_list': [3],
                   'odor_spread_list': [0.04],
                   'odor_source_allocation': 'iterative'}

chemotax_odor_np = {'odor_landscape': 'Gaussian',
                    'odor_id_list': ['Random_odor_ID'],
                    'odor_carriers': 'food',
                    'odor_intensity_list': [8],
                    'odor_spread_list': [0.0004],
                    'odor_source_allocation': 'iterative'}

chemorbit_odor_p = {'odor_landscape': 'Gaussian',
                    'odor_id_list': ['Random_odor_ID'],
                    'odor_carriers': 'food',
                    'odor_intensity_list': [3 / 16],
                    'odor_spread_list': [0.04 / 4],
                    'odor_source_allocation': 'iterative'}

chemorbit_odor_np = {'odor_landscape': 'Gaussian',
                     'odor_id_list': ['Random_odor_ID'],
                     'odor_carriers': 'food',
                     'odor_intensity_list': [2],
                     'odor_spread_list': [0.0002],
                     'odor_source_allocation': 'iterative'}

patchy_odor_np = {'odor_landscape': 'Gaussian',
                  'odor_id_list': ['Random_odor_ID'],
                  'odor_carriers': 'food',
                  'odor_intensity_list': [0.5],
                  'odor_spread_list': [0.00005],
                  'odor_source_allocation': 'iterative'}

pref_odors = {'odor_landscape': 'Gaussian',
              'odor_id_list': ['CS+', 'CS-'],
              'odor_carriers': 'food',
              'odor_intensity_list': [2, 2],
              'odor_spread_list': [0.001, 0.001],
              'odor_source_allocation': 'iterative'}

reorientation_odor_p = {'odor_landscape': 'Gaussian',
                        'odor_id_list': ['Odor'],
                        'odor_carriers': 'food',
                        'odor_intensity_list': [3 / 16],
                        'odor_spread_list': [0.04 / 4],
                        'odor_source_allocation': 'iterative'}

reorientation_odor_np = {'odor_landscape': 'Gaussian',
                         'odor_id_list': ['Odor'],
                         'odor_carriers': 'food',
                         'odor_intensity_list': [0.03 / 16],
                         'odor_spread_list': [0.0004 / 4],
                         'odor_source_allocation': 'iterative'}


########### ARENA PARAMETERS ################

def dish(r):
    return {'arena_xdim': r,
            'arena_ydim': r,
            'arena_shape': 'circular'}


def arena(x, y):
    return {'arena_xdim': x,
            'arena_ydim': y,
            'arena_shape': 'rectangular'}


########## FULL EXPERIMENT PARAMETERS #########################

def pref_conf(N, dish_r=0.1):
    conf = {'arena_params': dish(dish_r),
            'food_params': food(0.005),
            'place_params': pref_place(N),
            'odor_params': pref_odors}
    return conf


def chemotax_conf(N):
    conf = {'arena_params': arena(0.1, 0.06),
            'food_params': food(0.002),
            'place_params': chemotax_place(N),
            'odor_params': chemotax_odor_np}
    return conf


def chemorbit_conf(N):
    conf = {'arena_params': arena(0.1, 0.06),
            'food_params': food(0.002),
            'place_params': {**larva_mid(N, s=0), **food_place(1)},
            'odor_params': chemorbit_odor_np}
    return conf


def disp_conf(N, dish_r=0.3):
    conf = {'arena_params': dish(dish_r),
            'food_params': None,
            'place_params': {**larva_mid(N, s=0.0), **food_place(0)},
            'odor_params': None}
    return conf


def exp_conf(exp, physics=False):
    new = exp.copy()
    if physics:
        new['space_params'] = box2d_space
    else:
        new['space_params'] = mesa_space
    return new


pref_exp_np = exp_conf(pref_conf(25))

chemotax_exp_np = exp_conf(chemotax_conf(30))

chemorbit_exp_np = exp_conf(chemorbit_conf(30))

disp_exp_np = exp_conf(disp_conf(30, 0.2))

dish_exp_np = {'arena_params': dish(0.1),
               'space_params': mesa_space,
               'food_params': None,
               'place_params': {**larva_mid(25), **food_place(0)},
               'odor_params': None}

reorientation_exp_p = {'arena_params': dish(0.2),
                       'space_params': box2d_space,
                       'food_params': food(0.01),
                       'place_params': reorientation_place_params,
                       'odor_params': reorientation_odor_p}

reorientation_exp_np = {'arena_params': dish(0.2),
                        'space_params': mesa_space,
                        'food_params': food(0.01),
                        'place_params': reorientation_place_params,
                        'odor_params': reorientation_odor_np}

imitation_exp_p = {'arena_params': dish(0.15),
                   'space_params': box2d_space,
                   'food_params': None,
                   'place_params': {**larva_mid(25), **food_place(0)},
                   'odor_params': None}

focus_exp_np = {'arena_params': dish(0.02),
                'space_params': mesa_space,
                'food_params': None,
                'place_params': larva1_food0_facing_up,
                'odor_params': None}

focus_exp_p = {'arena_params': dish(0.02),
               'space_params': box2d_space,
               'food_params': None,
               'place_params': larva1_food0_facing_up,
               'odor_params': None}

feed_grid_exp = {'arena_params': arena(0.05, 0.05),
                 'food_params': food_grid(50),
                 'place_params': {**larva_mid(1), **food_place(0)},
                 'odor_params': None}

feed_grid_exp_np = exp_conf(feed_grid_exp)

feed_grid_exp_p = exp_conf(feed_grid_exp, physics=True)

feed_scatter_exp_np = {'arena_params': arena(0.05, 0.05),
                       'space_params': mesa_space,
                       'food_params': food(0.0003),
                       'place_params': {**larva_mid(10, s=0.3), **food_place(10000, 'uniform')},
                       'odor_params': None}

feed_patchy_exp_np = {'arena_params': arena(0.2, 0.2),
                      'space_params': mesa_space,
                      'food_params': food(0.0025, amount=0.001),
                      'place_params': {**larva_mid(20, s=0.1),**food_place(8, 'circle', 0.7)},
                      'odor_params': chemorbit_odor_np}

feed_patchy_empirical = {
    # 'arena_params': arena(0.2, 0.2),
    'arena_params': arena(0.192512, 0.192512),
    'space_params': mesa_space_in_mm,
    'food_params': food(0.0025),
    'place_params': larva0_food_patchy_8_exp,
    'odor_params': chemorbit_odor_np}



growth_exp_np = {'arena_params': arena(0.01, 0.01),  # dish(0.006),
                 'space_params': mesa_space,
                 'food_params': food_grid(50,10**-5),  # food(0.0002),
                 'place_params': {**larva_mid(1), **food_place(0)},  # larva1_food_uniform,
                 'odor_params': None}

growth_exp_np_small = {'arena_params': arena(0.01, 0.01),  # dish(0.006),
                 'space_params': mesa_space,
                 'food_params': food_grid(20,10**-7),  # food(0.0002),
                 'place_params': {**larva_mid(1), **food_place(0)},  # larva1_food_uniform,
                 'odor_params': None}

growth_exp_np_old = {'arena_params': dish(0.006),
                     'space_params': mesa_space,
                     'food_params': food(0.0002),
                     'place_params': {**larva_mid(1), **food_place(300, 'uniform')},
                     'odor_params': None}
