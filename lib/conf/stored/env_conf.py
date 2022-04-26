import numpy as np

from lib.aux.colsNstr import col_range, N_colors
from lib.conf.base.dtypes import null_dict, arena, oG, oD, border, vborder


def su(id='Source', group='Source', c='green', r=0.003, a=0.0, o=null_dict('odor'), **kwargs):
    return {id: null_dict('source', default_color=c, group=group, radius=r, amount=a, odor=o, **kwargs)}


def sg(id='Source', c='green', r=0.003, a=0.0, o=null_dict('odor'), N=1, s=(0.0, 0.0), loc=(0.0, 0.0), sh='circle',
       m='uniform', **kwargs):
    if type(s) == float:
        s = (s, s)
    d = null_dict('spatial_distro', N=N, loc=loc, scale=s, shape=sh, mode=m)
    return {id: null_dict('SourceGroup', default_color=c, distribution=d, radius=r, amount=a, odor=o, **kwargs)}


def sgs(Ngs, ids=None, cs=None, rs=None, ams=None, os=None, qs=None, **kwargs):
    if ids is None:
        ids = [f'Source{i}' for i in range(Ngs)]

    if ams is None:
        ams = np.random.uniform(0.002, 0.01, Ngs)
    if rs is None:
        rs = ams
    if qs is None:
        qs = np.linspace(0.1, 1, Ngs)
    if cs is None:
        cs = [tuple(col_range(q, low=(255, 0, 0), high=(0, 128, 0))) for q in qs]
    if os is None:
        os = [oG(id=f'Odor{i}') for i in range(Ngs)]
    l = [sg(id=ids[i], c=cs[i], r=rs[i], a=ams[i], o=os[i], quality=qs[i], **kwargs) for i in range(Ngs)]
    result = {}
    for d in l:
        result.update(d)
    return result


def f_pars(sg={}, su={}, grid=None):
    return {'source_groups': sg,
            'food_grid': grid,
            'source_units': su}


def foodNodor_4corners(d=0.05):
    l = [su(f'Source_{i}', pos=p, a=0.01, o=oD(id=f'Odor_{i}'), c=c, r=0.01) for i, (c, p) in
         enumerate(zip(['blue', 'red', 'green', 'magenta'], [(-d, -d), (-d, d), (d, -d), (d, d)]))]
    dic = {**l[0], **l[1], **l[2], **l[3]}
    return dic


def env(a, f=f_pars(), o=None, bl={}, w=None, t=None):
    if o == 'D':
        o = {'odorscape': 'Diffusion',
             'grid_dims': [51, 51],
             'evap_const': 0.9,
             'gaussian_sigma': (7, 7)
             }
    elif o == 'G':
        o = {'odorscape': 'Gaussian',
             'grid_dims': None,
             'evap_const': None,
             'gaussian_sigma': None
             }
    if w is not None:
        if 'puffs' in w.keys():
            for id, args in w['puffs'].items():
                w['puffs'][id] = null_dict('air_puff', **args)
        else:
            w['puffs'] = {}
        w = null_dict('windscape', **w)
    return null_dict('env_conf', arena=a, food_params=f, odorscape=o, border_list=bl, windscape=w, thermoscape=t)


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


def double_patches(type='standard'):
    return {**su('Left_patch', pos=(-0.06, 0.0), o=oG(id='Odor'), c='green', r=0.025, a=0.1, type=type),
            **su('Right_patch', pos=(0.06, 0.0), o=oG(id='Odor'), c='green', r=0.025, a=0.1, type=type)}


def maze_conf(n, h):
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
            ps = [(np.round(x - h / 2, 3), np.round(y - h / 2, 3)) for x, y in ps]
            return ps
        else:
            return lines

    return env(arena(h, h),

               f=f_pars(su=su('Target', o=oG(), c='blue')), o='G',
               bl={'Maze': {
                   'points': maze(nx=n, ny=n, h=h, return_points=True),
                   'default_color': 'black',
                   'width': 0.001}})


def game_env(dim=0.1, x=0.4, y=0.0):
    x = np.round(x * dim, 3)
    y = np.round(y * dim, 3)

    return env(arena(dim, dim),
               f=f_pars(su={
                   **su('Flag', c='green', can_be_carried=True, a=0.01, o=oG(2, id='Flag_odor')),
                   **su('Left_base', pos=(-x, y), c='blue', o=oG(id='Left_base_odor')),
                   **su('Right_base', pos=(+x, y), c='red', o=oG(id='Right_base_odor'))}),
               o='G')


env_dict = {
    'focus': env(arena(0.01, 0.01)),
    'dish': env(arena(0.1)),
    'arena_200mm': env(arena(0.2, 0.2)),

    'odor_gradient': env(arena(0.1, 0.06), f_pars(su=su(pos=(0.04, 0.0), o=oG(2))), 'G'),
    'mid_odor_gaussian': env(arena(0.1, 0.06), f_pars(su=su(o=oG())), 'G'),
    'mid_odor_diffusion': env(arena(0.3, 0.3), f_pars(su=su(r=0.03, o=oD())), 'D'),
    '4corners': env(arena(0.2, 0.2), f_pars(su=foodNodor_4corners()), 'D'),
    'food_at_bottom': env(arena(0.2, 0.2),
                          f_pars(sg=sg('FoodLine', o=oG(), a=0.002, r=0.001, N=20, sh='oval', s=(0.01, 0.0),
                                       m='periphery')), 'G'),

    'windy_arena': env(arena(0.3, 0.3), w={'wind_speed': 10.0}),
    'windy_blob_arena': env(arena(0.128, 0.014),
                            f_pars(sg=sgs(1, qs=np.ones(4),cs=N_colors(4), N=1, s=(0.0, 0.0),loc=(0.005,0.0), m='uniform', shape='rectangular', can_be_displaced=True,
                                          regeneration=True, regeneration_pos={'loc' : (0.005,0.0), 'scale': (0.0, 0.0)})),
                            w={'wind_speed': 1.0}),
    'windy_arena_bordered': env(arena(0.3, 0.3), w={'wind_speed': 10.0},
                                bl={'Border': vborder(-0.03, [-0.01, -0.06], w=0.005)}),
    'puff_arena_bordered': env(arena(0.3, 0.3), w={'puffs': {'PuffGroup': {}}},
                               bl={'Border': vborder(-0.03, [-0.01, -0.06], w=0.005)}),
    'single_puff': env(arena(0.3, 0.3),
                       w={'puffs': {'Puff': {'N': 1, 'duration': 30.0, 'start_time': 55, 'speed': 100}}}),

    'CS_UCS_on_food': env(arena(0.1), f_pars(grid=null_dict('food_grid'), su=CS_UCS(1)), 'G'),
    'CS_UCS_on_food_x2': env(arena(0.1), f_pars(grid=null_dict('food_grid'), su=CS_UCS(2)), 'G'),
    'CS_UCS_off_food': env(arena(0.1), f_pars(su=CS_UCS(1)), 'G'),

    'patchy_food': env(arena(0.2, 0.2), f_pars(sg=sg(N=8, s=0.07, m='periphery', a=0.001, o=oG(2))), 'G'),
    'random_food': env(arena(0.1, 0.1), f_pars(sg=sgs(4, N=1, s=0.04, m='uniform', shape='rectangular')), 'G'),
    'uniform_food': env(arena(0.05), f_pars(sg=sg(N=2000, s=0.025, a=0.01, r=0.0001))),
    'food_grid': env(arena(0.02, 0.02), f_pars(grid=null_dict('food_grid'))),
    'single_odor_patch': env(arena(0.05, 0.05), f_pars(su=su('Patch', a=0.1, r=0.01, o=oG())), 'G'),
    'single_patch': env(arena(0.05, 0.05), f_pars(su=su('Patch', a=0.1, r=0.01))),
    'multi_patch': env(arena(0.02, 0.02), f_pars(sg=sg(N=8, s=0.007, m='periphery', a=0.1, r=0.0015))),
    'double_patch': env(arena(0.24, 0.24),
                        f_pars(su=double_patches()),
                        'G'),

    'maze': maze_conf(15, 0.1),
    'game': game_env(),
    # 'capture_the_flag': game_env(),
    'arena_50mm_diffusion': env(arena(0.05), o='D'),
    'thermo_gradient': env(arena(0.17,0.17), t={"plate_temp": 22, "thermo_sources": [[0.5,0.05], [0.05,0.5], [0.5,0.95], [0.95,0.5]], "thermo_source_dTemps" : [8,-8,8,-8]}),
}

# if __name__ == '__main__':
#     a=env_dict['mid_odor_gaussian']['food_params']['source_units']
#     sources={k: v['pos'] for k,v in a.items()}
#     for k,v in a.items() :
#         print(k, v['pos'])
