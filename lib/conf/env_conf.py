import numpy as np
from lib.conf.init_dtypes import null_dict
import lib.aux.colsNstr as fun

def odor(i, s, id='Odor'):
    return null_dict('odor', odor_id=id, odor_intensity=i, odor_spread=s)


def oG(c=1, id='Odor'):
    return odor(i=2.0 * c, s=0.0002 * np.sqrt(c), id=id)


def oD(c=1, id='Odor'):
    return odor(i=300.0 * c, s=0.1 * np.sqrt(c), id=id)


def su(id='Source', group='Source', c='green', r=0.003, a=0.0, o=null_dict('odor'), **kwargs):
    return {id: null_dict('source', default_color=c, group=group, radius=r, amount=a, odor=o, **kwargs)}


def sg(id='Source', c='green', r=0.003, a=0.0, o=null_dict('odor'), N=1, s=(0.0, 0.0), loc=(0.0, 0.0), sh='circle',
       m='uniform', **kwargs):
    if type(s) == float:
        s = (s, s)
    d = null_dict('spatial_distro', N=N, loc=loc, scale=s, shape=sh, mode=m)
    return {id: null_dict('SourceGroup', default_color=c, distribution=d, radius=r, amount=a, odor=o, **kwargs)}


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


def lg(group='Larva', c='black', N=1, mode='uniform', sh='circle', p=(0.0, 0.0), ors=(0.0, 360.0),
       s=(0.0, 0.0), m='explorer', o=null_dict('odor'), **kwargs):
    if type(s) == float:
        s = (s, s)
    dist = null_dict('larva_distro', N=N, mode=mode, shape=sh, loc=p, orientation_range=ors, scale=s)
    g = null_dict('LarvaGroup', distribution=dist, default_color=c, model=m, odor=o, **kwargs)
    return {group: g}


def foodNodor_4corners(d=0.05):
    l = [su(f'Source_{i}', pos=p, a=0.01, o=oD(id=f'Odor_{i}'), c=c, r=0.01) for i, (c, p) in
         enumerate(zip(['blue', 'red', 'green', 'magenta'], [(-d, -d), (-d, d), (d, -d), (d, d)]))]
    dic = {**l[0], **l[1], **l[2], **l[3]}
    return dic


def env(a, l={}, f=None, o=None, bl={}):
    if o == 'D':
        o = diffusion_odor()
    elif o == 'G':
        o = gaussian_odor()
    return null_dict('env_conf', arena=a, larva_groups=l, food_params=f, odorscape=o, border_list=bl)

def lgs(models,ids=None, **kwargs) :
    if ids is None :
        ids=models
    N=len(models)
    cols=fun.N_colors(N)
    lgs={}
    for m,c, id in zip(models, cols, ids) :
        lg0=lg(id, c=c, m=m, **kwargs)
        lgs.update(lg0)
    return lgs



expl_envs = {
    'focus': env(arena(0.01, 0.01), lg(m='explorer', N=1, ors=[90.0, 90.0])),
    'dish': env(dish(0.1), lg(m='explorer', N=25, s=0.02)),
    'nengo_dish': env(dish(0.1), lg(m='nengo_explorer', N=25, s=0.02)),
    'dispersion': env(dish(0.2), lg(m='explorer', N=25)),
    'dispersion_x2': env(dish(0.2),
                         lgs(models=['explorer', 'Levy-walker', 'explorer_3con', 'nengo_explorer'],
                             ids=['CoupledOsc', 'Levy', '3con', 'nengo'], N=5)
                         ),
    'realistic_imitation': env(dish(0.15), lg(m='imitator', N=25))
}

chemo_envs = {
    'chemotaxis_approach': env(arena(0.1, 0.06),
                               lg(m='navigator', N=8, p=(-0.04, 0.0), s=(0.005, 0.02), ors=(-30.0, 30.0)),
                               f_pars(su=su(pos=(0.04, 0.0), o=oG(2))),
                               'G'),
    'chemotaxis_local': env(arena(0.1, 0.06),
                            lgs(models=['navigator', 'RL_navigator', 'basic_navigator'],
                                ids=['CoupledOsc', 'RL', 'basic'], N=10),
                            # lg(m='navigator', N=3),
                            f_pars(su=su(o=oG())),
                            'G'),
    'chemotaxis_diffusion': env(arena(0.3, 0.3),
                                lg(m='navigator', N=30),
                                f_pars(su=su(r=0.03, o=oD())),
                                'D'),
    'chemotaxis_RL': env(dish(0.1),
                         lg(m='RL_navigator', N=10, mode='periphery', s=0.04),
                         f_pars(su=su(r=0.01, o=oD())),
                         'D'),
    'food_at_bottom': env(arena(0.2, 0.2),
                          {**lg('Orco', m='Orco_forager', c='red', N=20, sh='oval', p=(0.0, 0.04),s=(0.04, 0.01)),
                           **lg('control', m='forager', c='blue', N=20, sh='oval', p=(0.0, 0.04),s=(0.04, 0.01))},
                          f_pars(sg=sg('FoodLine', o=oG(), a=0.002, r=0.001, N=20, sh='oval', s=(0.01, 0.0),m='periphery')),
                          'G'),
    '4corners': env(arena(0.2, 0.2),
                    lg(m='RL_forager', N=10, s=0.04),
                    f_pars(su=foodNodor_4corners()),
                    'D'),
    'reorientation': env(dish(0.1),
                         lg(m='immobile', N=200, s=0.05),
                         f_pars(su=su(o=oG())),
                         'G')
}

feed_envs = {
    'patchy_food': env(arena(0.2, 0.2),
                       lg(m='forager', N=25),
                       f_pars(sg=sg(N=8, s=0.07, m='periphery', a=0.001, o=oG(2))),
                       'G'),
    'uniform_food': env(dish(0.05),
                        lg(m='Orco_forager', N=5, s=0.005),
                        f_pars(sg=sg(N=2000, s=0.025, a=0.01, r=0.0001))),
    'food_grid': env(arena(0.2, 0.2),
                     lg(m='Orco_forager', N=25),
                     f_pars(grid=null_dict('food_grid'))),
    'single_patch': env(arena(0.1, 0.1),
                        {**lg('Orco', m='Orco_forager', c='red', N=20, mode='periphery', s=0.03),
                           **lg('control', m='forager', c='blue', N=20, mode='periphery', s=0.03)},
                        f_pars(su=su('Patch', a=0.1, r=0.02))),
    'growth': env(arena(0.2, 0.2),
                  lg(m='sitter', N=1),
                  f_pars(grid=null_dict('food_grid'))),
}

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


def maze_conf(n, h):
    return env(arena(h, h), l=lg(N=5, p=(-0.4 * h, 0.0), ors=(-60.0, 60.0), m='navigator'),
               f=f_pars(su=su('Target', o=oG(), c='blue')), o='G',
               bl={'Maze': {
                   'points': maze(nx=n, ny=n, h=h, return_points=True),
                   'default_color': 'black',
                   'width': 0.001}})


def game_env_conf(dim=0.1, N=10, x=0.4, y=0.0, mode='king'):
    x = np.round(x * dim, 3)
    y = np.round(y * dim, 3)
    if mode == 'king':
        l = {**lg('Left', N=N, p=(-x, y), m='gamer-5x', c='darkblue', o=oG(id='Left_Odor')),
             **lg('Right', N=N, p=(+x, y), m='gamer-5x', c='darkred', o=oG(id='Right_Odor'))}
    elif mode == 'flag':
        l = {**lg('Left', N=N, p=(-x, y), m='gamer', c='darkblue'),
             **lg('Right', N=N, p=(+x, y), m='gamer', c='darkred')}

    return env(arena(dim, dim),
               l=l,
               f=f_pars(su={
                   **su('Flag', c='green', can_be_carried=True, a=0.01, o=oG(2, id='Flag_odor')),
                   **su('Left_base', pos=(-x, y), c='blue', o=oG(id='Left_base_odor')),
                   **su('Right_base', pos=(+x, y), c='red', o=oG(id='Right_base_odor'))}),
               o='G')


game_envs = {
    'maze': maze_conf(15, 0.1),
    'keep_the_flag': game_env_conf(mode='king'),
    'capture_the_flag': game_env_conf(mode='flag'),
    'catch_me': env(dish(0.05),
                    l={**lg('Left', N=1, p=(-0.01, 0.0), m='follower-L', c='darkblue', o=oD(id='Left_Odor')),
                       **lg('Right', N=1, p=(+0.01, 0.0), m='follower-R', c='darkred', o=oD(id='Right_Odor'))},
                    o='D'),

}


def CS_UCS(N=2, x=0.04):
    if N == 1:
        return {**su('CS', pos=(-x, 0.0), o=oG(id='CS'), c='red'),
                **su('UCS', pos=(x, 0.0), o=oG(id='UCS'), c='blue')}
    elif N == 2:
        return {
            **su('CS_l', pos=(-x, 0.0), o=oG(id='CS'), c='red'),
            **su('CS_r', pos=(x, 0.0), o=oG(id='CS'), c='red'),
            **su('UCS_l', pos=(-x, 0.0), o=oG(id='UCS'), c='blue'),
            **su('UCS_r', pos=(x, 0.0), o=oG(id='UCS'), c='blue')
        }


pref_envs = {
    'odor_pref_test': env(dish(0.1),
                          lg(N=25, s=(0.005, 0.02), m='navigator_x2'),
                          f_pars(su=CS_UCS(1)),
                          'G'),
    'odor_pref_test_on_food': env(dish(0.1),
                                  lg(N=25, s=(0.005, 0.02), m='forager_x2'),
                                  f_pars(grid=null_dict('food_grid'), su=CS_UCS(1)),
                                  'G'),
    'odor_pref_train': env(dish(0.1),
                           lg(N=25, s=(0.005, 0.02), m='RL_forager'),
                           f_pars(grid=null_dict('food_grid'), su=CS_UCS(2)),
                           'G'),
    'odor_pref_RL': env(arena(0.2, 0.1),
                        lg(N=25, s=(0.005, 0.02), m='RL_navigator'),
                        f_pars(su=CS_UCS(1)),
                        'G'),
}


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
        # 'distribution': null_dict('larva_distro', N=N),
        **kwargs
    }

    mod_r = RvsS_larva(**R_kws, **RS_kws)
    mod_s = RvsS_larva(**S_kws, **RS_kws)
    return {
        **lg('Rover', m=mod_r, c='blue', N=N, **group_kws),
        **lg('Sitter', m=mod_s, c='red', N=N, **group_kws)}


def RvsS_env(on_food=True, **kwargs):
    grid = null_dict('food_grid') if on_food else None
    return null_dict('env_conf', arena=arena(0.02, 0.02),
                     food_params=null_dict('food_params', food_grid=grid),
                     larva_groups=RvsS_groups(**kwargs)
                     )


RvsS_envs = {
    'rovers_sitters_on_food': RvsS_env(),
    'rovers_sitters_on_food_q75': RvsS_env(q=0.75),
    'rovers_sitters_on_food_q50': RvsS_env(q=0.50),
    'rovers_sitters_on_food_q25': RvsS_env(q=0.25),
    'rovers_sitters_on_food_q15': RvsS_env(q=0.15),
    'rovers_sitters_on_food_1h_prestarved': RvsS_env(h_starved=1.0),
    'rovers_sitters_on_food_2h_prestarved': RvsS_env(h_starved=2.0),
    'rovers_sitters_on_food_3h_prestarved': RvsS_env(h_starved=3.0),
    'rovers_sitters_on_food_4h_prestarved': RvsS_env(h_starved=4.0),
    'rovers_sitters_on_agar': RvsS_env(on_food=False),
}

env_dict = {
    **expl_envs,
    **chemo_envs,
    **pref_envs,
    **feed_envs,
    **RvsS_envs,
    **game_envs
}
