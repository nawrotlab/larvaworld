import itertools

import numpy as np

import lib.conf.dtype_dicts as dtypes
from lib.conf.init_dtypes import null_dict


def odor(i, s, id='Odor'):
    return null_dict('odor', odor_id=id, odor_intensity=i, odor_spread=s)


def odorG(c=1, id='Odor'):
    return odor(i=2.0 * c, s=0.0002 * np.sqrt(c), id=id)


def odorD(c=1, id='Odor'):
    return odor(i=300.0 * c, s=0.1 * np.sqrt(c), id=id)


def source(id='Source', group='Source', c='green', **kwargs):
    return {id: null_dict('source', default_color=c, group=group, **kwargs)}


def sg(id='Source', c='green', d={}, **kwargs):
    return {id: null_dict('SourceGroup', default_color=c, distribution=null_dict('spatial_distro', **d), **kwargs)}


def dish(r):
    return {'arena_dims': (r, r),
            'arena_shape': 'circular'}


def arena(x, y):
    return {'arena_dims': (x, y),
            'arena_shape': 'rectangular'}


def gaussian_odor():
    return {'odorscape': 'Gaussian',
            'grid_dims': None,
            'evap_const': None,
            'gaussian_sigma': None
            }


def diffusion_odor():
    return {'odorscape': 'Diffusion',
            'grid_dims': [100, 100],
            'evap_const': 0.9,
            'gaussian_sigma': (7, 7)
            }


def f_pars(sg={}, su={}, grid=None):
    return {'source_groups': sg,
            'food_grid': grid,
            'source_units': su}


def lg(group='Larva', c='black', N=1, mode='uniform', shape='circle', loc=(0.0, 0.0), ors=(0.0, 360.0),
       s=(0.0, 0.0),m='explorer', **kwargs):
    if type(s)==float :
        s=(s,s)
    dist = null_dict('larva_distro', N=N, mode=mode, shape=shape, loc=loc, orientation_range=ors,scale=s)
    g = null_dict('LarvaGroup', distribution=dist, default_color=c,model=m, **kwargs)
    return {group: g}



def foodNodor_4corners(d=0.05):
    l = [source(f'Source_{i}', pos=p,amount=0.01, odor=odorD(id=f'Odor_{i}'),c=c, radius=0.01) for i, (c, p) in
         enumerate(zip(['blue', 'red', 'green', 'magenta'],[(-d, -d), (-d, d), (d, -d), (d, d)]))]
    dic = {**l[0], **l[1], **l[2], **l[3]}
    return dic


def env(a, l={}, f=None, o=None):
    return null_dict('env_conf', arena=a, larva_groups=l, food_params=f, odorscape=o)


dish_env = env(dish(0.1), l=lg(m='explorer', N=25, s=0.02))
nengo_dish_env = env(dish(0.1), l=lg(m='nengo_explorer', N=25, s=0.02))
dispersion_env = env(dish(0.2), l=lg(m='explorer', N=25))
focus_env = env(arena(0.01, 0.01), l=lg(m='explorer', N=1, ors=[90.0, 90.0]))
imitation_env_p = env(dish(0.15), l=lg(m='imitation', N=25))

chemotax_env = env(arena(0.1, 0.06),l=lg(m='navigator', N=8, loc=(-0.04, 0.0), s=(0.005, 0.02), ors=(-30.0, 30.0)),
                   f=f_pars(su=source(pos=(0.04, 0.0), odor=odorG(2))),o=gaussian_odor())

chemorbit_env = env(arena(0.1, 0.06), l=lg(m='navigator', N=3),f=f_pars(su=source(odor=odorG())), o=gaussian_odor())

chemorbit_diffusion_env = env(arena(0.3, 0.3), l=lg(m='navigator', N=30),
                              f=f_pars(su=source(radius=0.03, odor=odorD())), o=diffusion_odor())

RL_chemorbit_env = env(dish(0.1), l=lg(m='RL-learner', N=10, mode='periphery', s=0.04),
                       f=f_pars(su=source(radius=0.01, odor=odorD())), o=diffusion_odor())

reorientation_env = env(dish(0.1), l=lg(m='immobile', N=200, s=0.05),
                        f=f_pars(su=source(odor=odorG())), o=gaussian_odor())

RL_4corners_env = env(arena(0.2, 0.2), l=lg(m='RL-learner', N=10, s=0.04),
                      f=f_pars(su=foodNodor_4corners()), o=diffusion_odor())

uniform_food_env = env(dish(0.05), l=lg(m='feeder-explorer', N=5, s=0.005),
                       f=f_pars(sg=sg(d={'N': 2000, 'scale': (0.025, 0.025)}, amount=0.01, radius=0.0001)))

patchy_food_env = env(arena(0.2, 0.2), l=lg(m='feeder-navigator', N=25),
                      f=f_pars(sg=sg(d={'N': 8, 'scale': (0.07, 0.07), 'mode': 'periphery'},amount=0.001, odor=odorG(2))),
                      o=gaussian_odor())
food_grid_env = env(arena(0.2, 0.2), l=lg(m='feeder-explorer', N=25),
                    f=null_dict('food_params', food_grid=null_dict('food_grid')))

growth_env = env(arena(0.2, 0.2), l=lg(m='sitter', N=1),f=f_pars(grid=null_dict('food_grid')))

single_patch_env = env(arena(0.1, 0.1),
                       l={**lg('Orco', m='feeder-explorer', c='red', N=20, mode='periphery', s=0.03),
                          **lg('control', m='feeder-navigator', c='blue', N=20, mode='periphery',s=0.03)},
                       f=f_pars(su=source('Patch', amount=0.1, radius=0.02)))

food_at_bottom_env = env(arena(0.2, 0.2),
                         l={**lg('Orco', m='feeder-explorer', c='red', N=20, shape='oval', loc=(0.0, 0.04),s=(0.04, 0.01)),
                            **lg('control', m='feeder-navigator', c='blue', N=20, shape='oval', loc=(0.0, 0.04),s=(0.04, 0.01))},
                         f=f_pars(sg=sg('FoodLine', odor=odorG(), amount=0.002, radius=0.001,
                                        d={'N': 20, 'shape': 'oval', 'scale': (0.01, 0.0), 'mode': 'periphery'})),
                         o=gaussian_odor())

catch_me_env = env(dish(0.05), l={**lg('Left', N=1, loc=(-0.01, 0.0), m='follower-L', c='darkblue', odor=odorD(id='Left_Odor')),
                                  **lg('Right', N=1, loc=(+0.01, 0.0), m='follower-R', c='darkred', odor=odorD(id='Right_Odor'))},
                   o=diffusion_odor())

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


def maze(nx=15, ny=15, ix=0, iy=0, h=0.1, return_points=False):
    from lib.envs._maze import Maze
    m = Maze(nx, ny, ix, iy, height=h)
    m.make_maze()
    lines = m.maze_lines()
    if return_points:
        ps = []
        for l in lines:
            ps.append(l.coords[0])
            ps.append(l.coords[1])
        ps = [(np.round(x - h / 2, 3), np.round(y - h / 2, 3)) for x, y in ps]
        return ps
    else:
        return lines


def CS_UCS(N=2, x=0.04):
    if N == 1:
        return {**source('CS', pos=(-x, 0.0), odor=odorG(id='CS'), c='red'),
                **source('UCS', pos=(x, 0.0), odor=odorG(id='UCS'), c='blue')}
    elif N == 2:
        return {
            **source('CS_l', pos=(-x, 0.0), odor=odorG(id='CS'), c='red'),
            **source('CS_r', pos=(x, 0.0), odor=odorG(id='CS'), c='red'),
            **source('UCS_l', pos=(-x, 0.0), odor=odorG(id='UCS'), c='blue'),
            **source('UCS_r', pos=(x, 0.0), odor=odorG(id='UCS'), c='blue')
        }


def game_env_conf(dim=0.1, N=10, x=0.4, y=0.0, mode='king'):
    x = np.round(x * dim, 3)
    y = np.round(y * dim, 3)
    if mode == 'king':
        modL, modR = 'gamer-L', 'gamer-R'
    elif mode == 'flag':
        modL, modR = 'gamer', 'gamer'

    lgs = {**lg('Left', N=N, loc=(-x, y), m=modL, c='darkblue'),
           **lg('Right', N=N, loc=(+x, y), m=modR, c='darkred')
           }
    # env=env(arena(dim, dim))
    env = {'arena': arena(dim, dim),
           'border_list': {},
           'food_params': f_pars(su={
               **source('Flag', c='green',can_be_carried=True, amount=0.01, odor=odorG(2, id='Flag_odor')),
               **source('Left_base', pos=(-x, y), c='blue', odor=odorG(id='Left_base_odor')),
               **source('Right_base', pos=(+x, y), c='red', odor=odorG(id='Right_base_odor'))}),
           'larva_groups': lgs,
           'odorscape': gaussian_odor()
           }
    return env


king_env = game_env_conf(mode='king')
flag_env = game_env_conf(mode='flag')


def maze_conf(n, h):
    conf = {'arena': arena(h, h),
            'border_list': {
                'Maze': {
                    'points': maze(nx=n, ny=n, h=h, return_points=True),
                    'default_color': 'black',
                    'width': 0.001}
            },
            'food_params': f_pars(su=source('Target', odor=odorG(), c='blue')),
            'larva_groups': lg(N=5, loc=(-0.4 * h, 0.0), ors=(-60.0, 60.0), m='navigator'),
            'odorscape': gaussian_odor()}
    return conf


def odor_pref_env(N, m, arena=dish(0.1), Nsources=2, grid=null_dict('food_grid')):
    return null_dict('env_conf', arena=arena,
                     food_params=f_pars(grid=grid, su=CS_UCS(Nsources)),
                     larva_groups=lg(N=N, scale=(0.005, 0.02), m=m),
                     odorscape=gaussian_odor())


pref_test_env = odor_pref_env(N=25, m='navigator-x2', Nsources=1, grid=None)
pref_test_env_on_food = odor_pref_env(N=25, m='feeder-navigator-x2', Nsources=1)
pref_train_env = odor_pref_env(N=25, m='RL-feeder')
pref_env_RL = odor_pref_env(N=25, m='RL-learner', Nsources=1, grid=None, arena=arena(0.2, 0.1))

maze_env = maze_conf(15, 0.1)


def RvsS_groups(N=1, age=72.0, q=1.0, sub='standard', h_starved=0.0,
                R_kws={'EEB': 0.37, 'absorption': 0.5},
                S_kws={'EEB': 0.67, 'absorption': 0.15},
                RS_kws={'hunger_gain': 2.0, 'DEB_dt': 1.0},
                **kwargs):
    from lib.conf.larva_conf import RvsS_larva
    group_kws = {
        'sample': 'AttP2.Fed',
        'life': null_dict('life', hours_as_larva=age, substrate_quality=q, substrate_type=sub,
                          epochs=[(age - h_starved, age)], epoch_qs=[0.0]),
        'distribution': null_dict('larva_distro', N=N),
        **kwargs
    }

    mod_r = RvsS_larva(**R_kws, **RS_kws)
    mod_s = RvsS_larva(**S_kws, **RS_kws)

    return {'Rover': null_dict('LarvaGroup', model=mod_r, default_color='blue', **group_kws),
            'Sitter': null_dict('LarvaGroup', model=mod_s, default_color='red', **group_kws)}


def RvsS_env(on_food=True, **kwargs):
    grid = null_dict('food_grid') if on_food else None
    return null_dict('env_conf', arena=arena(0.02, 0.02),  # dish(0.006),
                     food_params=null_dict('food_params', food_grid=grid),
                     larva_groups=RvsS_groups(**kwargs)
                     )


RvsS_agar = RvsS_env(on_food=False)
RvsS_food = RvsS_env()
RvsS_food_q75 = RvsS_env(q=0.75)
RvsS_food_q50 = RvsS_env(q=0.50)
RvsS_food_q25 = RvsS_env(q=0.25)
RvsS_food_q15 = RvsS_env(q=0.15)
RvsS_food_1h_prestarved = RvsS_env(h_starved=1.0)
RvsS_food_2h_prestarved = RvsS_env(h_starved=2.0)
RvsS_food_3h_prestarved = RvsS_env(h_starved=3.0)
RvsS_food_4h_prestarved = RvsS_env(h_starved=4.0)
