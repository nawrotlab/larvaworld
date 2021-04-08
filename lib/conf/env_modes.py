import numpy as np

######## FOOD PARAMETERS ###########

# -------------------------------------------SPACE MODES----------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
import lib.aux.functions as fun
from lib.conf import odor


def food_grid(dim, amount=0.00001):
    return {'unique_id': 'Food_grid',
            'grid_dims': [dim, dim],
            'initial_value': amount,
            'distribution': 'uniform'}


def food(r=0.1, amount=0.000001, quality=1.0, **odor_args):
    return {'radius': r,
            'amount': amount,
            'quality': quality,
            **odor_args
            }


def food_distro(N, mode='normal', loc=(0.0, 0.0), scale=0.1, pars={}, group_id='Food', default_color=None):
    if N > 0:
        return {group_id: {'N': N,
                           'mode': mode,
                           'loc': loc,
                           'scale': scale,
                           **pars,
                           'default_color': default_color,
                           }}
    else:
        return {}


def food_param_conf(distro={}, list={}, grid=None):
    return {'source_groups': distro,
            'food_grid': grid,
            'source_units': list}


def larva_distro(N=1, mode='normal', loc=(0.0, 0.0), scale=0.1, orientation=None, group_id='Larva', model={},
                 default_color=None):
    if N > 0:
        return {group_id: {'N': N,
                           'mode': mode,
                           'loc': loc,
                           'scale': scale,
                           'orientation': orientation,
                           'model': model,
                           'default_color': default_color,
                           }}
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


def maze(nx=15, ny=15, ix=0, iy=0, h=0.1, return_points=False):
    from lib.model.envs._maze import Maze
    m = Maze(nx, ny, ix, iy, height=h)
    m.make_maze()
    lines = m.maze_lines()
    if return_points:
        ps = []
        for l in lines:
            ps.append(l.coords[0])
            ps.append(l.coords[1])
        return ps
    else:
        return lines


def odor_source(id, pos=(0.0, 0.0), r=0.003, odor_id=None, odor_intensity=2, odor_spread=0.0002, can_be_carried=False,
                default_color=None):
    if odor_id is None:
        odor_id = f'{id} odor'
    return {id: {'pos': pos,
                 **food(r=r, amount=0.0, **odor(odor_id, odor_intensity, odor_spread),
                        default_color=default_color, can_be_carried=can_be_carried)}}


def foodNodor_source(id, pos=(0.0, 0.0), r=0.003, odor_id=None, odor_intensity=2, odor_spread=0.0002,
                     can_be_carried=False, default_color=None):
    if odor_id is None:
        odor_id = f'{id} odor'
    return {id: {'pos': pos,
                 **food(r=r, amount=0.01, **odor(odor_id, odor_intensity, odor_spread),
                        default_color=default_color, can_be_carried=can_be_carried)}}


CS_UCS_odors = {**odor_source(id='CS_source', pos=(-0.04, 0.0), odor_id='CS', default_color='red'),
                **odor_source(id='UCS_source', pos=(0.04, 0.0), odor_id='UCS', default_color='blue')}


def gaussian_odor():
    return {'odorscape': 'Gaussian',
            # 'odor_id_list': odor_id_list,
            }


def diffusion_odor():
    return {'odorscape': 'Diffusion',
            'grid_dims': [100, 100],
            'evap_const': 0.9,
            'gaussian_sigma': (7, 7)
            }


def game_env_conf(dim=0.1, N=10, x=0.8, y=0.0, scale=0.03, mode='king'):
    x0, y0 = np.round(x * dim / 2, 3), np.round(y * dim / 2, 3)
    if mode == 'king':
        modL, modR = 'gamer_L', 'gamer_R'
    elif mode == 'flag':
        modL, modR = 'gamer', 'gamer'
    env = {'arena_params': arena(dim, dim),
           'food_params': food_param_conf(list={
               **odor_source('Flag', odor_intensity=8, odor_spread=0.0004, default_color='green', can_be_carried=True),
               **odor_source('Left base', (-x0, y0), default_color='red'),
               **odor_source('Right base', (+x0, y0), default_color='blue')}),
           'larva_params': {
               **larva_distro(N, loc=(-x, y), scale=scale, group_id='Left', model=modL, default_color='darkred'),
               **larva_distro(N, loc=(+x, y), scale=scale, group_id='Right', model=modR, default_color='darkblue')
           },
           'odorscape': gaussian_odor()
           }
    return env


king_env = game_env_conf(mode='king')
flag_env = game_env_conf(mode='flag')


def maze_conf(n):
    conf = {'arena_params': arena(0.1, 0.1),
            'border_list': {
                'Maze': {
                    'points': maze(nx=n, ny=n, h=0.1, return_points=True),
                    # 'lines': maze(nx=n, ny=n, h=0.1),
                    'from_screen': False}
            },
            'food_params': food_param_conf(list={**odor_source('Target')}),
            'larva_params': larva_distro(5, mode='facing_right', loc=(-0.8, 0.0), scale=0.03, model='navigator'),
            'odorscape': gaussian_odor()}
    return conf


pref_env = {'arena_params': dish(0.1),
            'food_params': food_param_conf(list=CS_UCS_odors),
            'larva_params': larva_distro(25, model='navigator_x2'),
            'odorscape': gaussian_odor()}

chemotax_env = {'arena_params': arena(0.1, 0.06),
                'food_params': food_param_conf(list={**odor_source(id='Odor source', pos=(0.04, 0.0),
                                                                   odor_id='Odor', odor_intensity=8,
                                                                   odor_spread=0.0004, default_color='blue')}),
                'larva_params': larva_distro(30, mode='facing_right', loc=(-0.8, 0.0), scale=0.05, model='navigator'),
                'odorscape': gaussian_odor()}

chemorbit_env = {'arena_params': arena(0.1, 0.06),
                 'food_params': food_param_conf(list={**odor_source(id='Odor source', odor_id='Odor',
                                                                    default_color='blue')}),
                 'larva_params': larva_distro(30, scale=0.0, model='navigator'),
                 'odorscape': gaussian_odor()
                 }

chemorbit_diffusion_env = {'arena_params': arena(0.3, 0.3),
                           'food_params': food_param_conf(
                               list={**odor_source(id='Odor source', odor_id='Odor',
                                                   odor_intensity=350.0, default_color='blue')}),
                           'larva_params': larva_distro(30, scale=0.0, model='navigator'),
                           'odorscape': diffusion_odor()
                           }

RL_chemorbit_env = {'arena_params': arena(0.1, 0.1),
                    'food_params': food_param_conf(
                        list={**foodNodor_source(id='Odor source', pos=(0.0, 0.0), odor_id='Odor',
                                                 odor_intensity=300.0, default_color='blue')}),
                    'larva_params': larva_distro(30, scale=0.0, model='RL_learner'),
                    'odorscape': diffusion_odor()
                    }

maze_env = maze_conf(15)

dispersion_env = {'arena_params': dish(0.2),
                  'food_params': food_param_conf(),
                  'larva_params': larva_distro(30, scale=0.0, model='explorer'),
                  'odorscape': None}

dish_env = {'arena_params': dish(0.1),
            'food_params': food_param_conf(),
            'larva_params': larva_distro(25, scale=0.05, model='explorer'),
            'odorscape': None}

reorientation_env = {'arena_params': dish(0.1),
                     'food_params': food_param_conf(list={**odor_source(id='Odor source', odor_id='Odor')}),
                     'larva_params': larva_distro(200, 'uniform_circ', scale=None, model='navigator'),
                     'odorscape': gaussian_odor()}

imitation_env_p = {'arena_params': dish(0.15),
                   'food_params': food_param_conf(),
                   'larva_params': larva_distro(25, model='imitation'),
                   'odorscape': None}

focus_env = {'arena_params': dish(0.02),
             'food_params': food_param_conf(),
             'larva_params': larva_distro(1, 'normal', scale=0.3, orientation=np.pi / 2, model='explorer'),
             'odorscape': None}

uniform_food_env = {'arena_params': arena(0.05, 0.05),
                    'food_params': food_param_conf(
                        distro=food_distro(10000, 'uniform', None, None, food(0.0003, **odor()))),
                    'larva_params': larva_distro(20, scale=0.3, model='feeder'),
                    'odorscape': None}

patchy_food_env = {'arena_params': arena(0.2, 0.2),
                   'food_params': food_param_conf(distro=food_distro(8, 'circle', None, 0.7,
                                                                     pars=food(0.0025, amount=0.001, odor_id='Odor',
                                                                               odor_intensity=8,
                                                                               odor_spread=0.0004), group_id='Food')),
                   'larva_params': larva_distro(25, model='feeder-navigator'),
                   'odorscape': gaussian_odor()}

food_grid_env = {'arena_params': arena(0.03, 0.03),  # dish(0.006),
                 'food_params': food_param_conf(grid=food_grid(50, 10 ** -9)),
                 'larva_params': larva_distro(25, model='feeder'),
                 'odorscape': None}

growth_env = {'arena_params': arena(0.03, 0.03),  # dish(0.006),
              'food_params': food_param_conf(grid=food_grid(50, 10 ** -3)),
              'larva_params': larva_distro(5, model='rover'),
              'odorscape': None}

growth_2x_env = {'arena_params': arena(0.02, 0.02),  # dish(0.006),
                 'food_params': food_param_conf(grid=food_grid(50, 10 ** -3)),
                 'larva_params': {
                     **larva_distro(1, group_id='Rover', model='rover', default_color='blue'),
                     **larva_distro(1, group_id='Sitter', model='sitter', default_color='red')
                 },
                 'odorscape': None}

test_env = {'arena_params': dish(0.1),
            'food_params': {
                'source_groups': food_distro(10, 'normal', pars=food(**odor(id='CS', intensity=2.0)), group_id='Food'),
                'food_grid': food_grid(50),
                'source_units': CS_UCS_odors
            },
            'larva_params': larva_distro(25, model='feeder'),
            'odorscape': gaussian_odor()}
