import numpy as np
from lib.conf.dtypes import null_dict, arena, oG, oD


def su(id='Source', group='Source', c='green', r=0.003, a=0.0, o=null_dict('odor'), **kwargs):
    return {id: null_dict('source', default_color=c, group=group, radius=r, amount=a, odor=o, **kwargs)}


def sg(id='Source', c='green', r=0.003, a=0.0, o=null_dict('odor'), N=1, s=(0.0, 0.0), loc=(0.0, 0.0), sh='circle',
       m='uniform', **kwargs):
    if type(s) == float:
        s = (s, s)
    d = null_dict('spatial_distro', N=N, loc=loc, scale=s, shape=sh, mode=m)
    return {id: null_dict('SourceGroup', default_color=c, distribution=d, radius=r, amount=a, odor=o, **kwargs)}


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


def foodNodor_4corners(d=0.05):
    l = [su(f'Source_{i}', pos=p, a=0.01, o=oD(id=f'Odor_{i}'), c=c, r=0.01) for i, (c, p) in
         enumerate(zip(['blue', 'red', 'green', 'magenta'], [(-d, -d), (-d, d), (d, -d), (d, d)]))]
    dic = {**l[0], **l[1], **l[2], **l[3]}
    return dic


def env(a, f=f_pars(), o=None, bl={}):
    if o == 'D':
        o = diffusion_odor()
    elif o == 'G':
        o = gaussian_odor()
    return null_dict('env_conf', arena=a, food_params=f, odorscape=o, border_list=bl)


expl_envs = {
    'focus': env(arena(0.01, 0.01)),
    'dish': env(arena(0.1)),
    'arena_200mm': env(arena(0.2, 0.2)),
}

chemo_envs = {
    'odor_gradient': env(arena(0.1, 0.06),
                         f_pars(su=su(pos=(0.04, 0.0), o=oG(2))),
                         'G'),
    'mid_odor_gaussian': env(arena(0.1, 0.06),
                             f_pars(su=su(o=oG())),
                             'G'),

    'mid_odor_diffusion': env(arena(0.3, 0.3),
                              f_pars(su=su(r=0.03, o=oD())),
                              'D'),
    # 'chemotaxis_RL': env(arena(0.1),
    #                      f_pars(su=su(r=0.01, o=oD())),
    #                      'D'),
    'food_at_bottom': env(arena(0.2, 0.2),
                          f_pars(sg=sg('FoodLine', o=oG(), a=0.002, r=0.001, N=20, sh='oval', s=(0.01, 0.0),
                                       m='periphery')),
                          'G'),
    '4corners': env(arena(0.2, 0.2),
                    f_pars(su=foodNodor_4corners()),
                    'D'),
    # 'reorientation': env(arena(0.1),
    #                      f_pars(su=su(o=oG())),
    #                      'G')
}

feed_envs = {
    'patchy_food': env(arena(0.2, 0.2),
                       f_pars(sg=sg(N=8, s=0.07, m='periphery', a=0.001, o=oG(2))),
                       'G'),
    'uniform_food': env(arena(0.05),
                        f_pars(sg=sg(N=2000, s=0.025, a=0.01, r=0.0001))),
    'food_grid': env(arena(0.2, 0.2),
                     f_pars(grid=null_dict('food_grid'))),
    'single_patch': env(arena(0.1, 0.1),
                        f_pars(su=su('Patch', a=0.1, r=0.02))),
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
    return env(arena(h, h),

               f=f_pars(su=su('Target', o=oG(), c='blue')), o='G',
               bl={'Maze': {
                   'points': maze(nx=n, ny=n, h=h, return_points=True),
                   'default_color': 'black',
                   'width': 0.001}})


def game_env(dim=0.1, N=10, x=0.4, y=0.0, mode='king'):
    x = np.round(x * dim, 3)
    y = np.round(y * dim, 3)

    return env(arena(dim, dim),
               f=f_pars(su={
                   **su('Flag', c='green', can_be_carried=True, a=0.01, o=oG(2, id='Flag_odor')),
                   **su('Left_base', pos=(-x, y), c='blue', o=oG(id='Left_base_odor')),
                   **su('Right_base', pos=(+x, y), c='red', o=oG(id='Right_base_odor'))}),
               o='G')




game_envs = {
    'maze': maze_conf(15, 0.1),
    'keep_the_flag': game_env(mode='king'),
    'capture_the_flag': game_env(mode='flag'),
    'arena_50mm_diffusion': env(arena(0.05),
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
    'CS_UCS_on_food': env(arena(0.1),
                          f_pars(grid=null_dict('food_grid'), su=CS_UCS(2)),
                          'G'),
    'CS_UCS_off_food': env(arena(0.1),
                           f_pars(su=CS_UCS(2)),
                           'G'),
}

env_dict = {
    **expl_envs,
    **chemo_envs,
    **pref_envs,
    **feed_envs,
    **game_envs
}
