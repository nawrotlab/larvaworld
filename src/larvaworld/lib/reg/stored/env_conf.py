import numpy as np

from larvaworld.lib import reg, aux


@reg.funcs.stored_conf("Env")
def Env_dict() :




    def oG(c=1, id='Odor'):
        return reg.get_null('odor', odor_id=id, odor_intensity=2.0 * c, odor_spread=0.0002 * np.sqrt(c))


    def oD(c=1, id='Odor'):
        return reg.get_null('odor', odor_id=id, odor_intensity=300.0 * c, odor_spread=0.1 * np.sqrt(c))
        # return self.odor(i=300.0 * c, s=0.1 * np.sqrt(c), id=id)

    def border(ps, c='black', w=0.01, id=None):
        b = reg.get_null('Border', points=ps, default_color=c, width=w)
        if id is not None:
            return {id: b}
        else:
            return b


    def hborder(y, xs, **kwargs):
        ps = [(x, y) for x in xs]
        return border(ps, **kwargs)


    def vborder(x, ys, **kwargs):
        ps = [(x, y) for y in ys]
        return border(ps, **kwargs)


    def su(id='Source', group='Source', c='green', r=0.003, a=0.0, o=reg.get_null('odor'), **kwargs):
        return {id: reg.get_null('source', default_color=c, group=group, radius=r, amount=a, odor=o, **kwargs)}


    def sg(id='Source', c='green', r=0.003, a=0.0, o=None, N=1, s=(0.0, 0.0), loc=(0.0, 0.0), sh='circle',
           m='uniform', **kwargs):
        if type(s) == float:
            s = (s, s)

        kws = {
            'kwdic': {'distribution': {'N': N, 'scale': s,  'loc': loc, 'shape': sh, 'mode': m}},
            'default_color': c, 'radius': r,'amount': a, **kwargs}
        if o is not None:
            kws['odor'] = o

        return reg.group.SourceGroup.entry(id=id, **kws)


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
            cs = [tuple(aux.col_range(q, low=(255, 0, 0), high=(0, 128, 0))) for q in qs]
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


    def env(arenaXY, f=f_pars(), o=None, bl={}, w=None, th=None, torus=False):
        if type(arenaXY) == float:
            arena = reg.get_null('arena', shape='circular', dims=(arenaXY, arenaXY), torus=torus)
        elif type(arenaXY) == tuple:
            arena = reg.get_null('arena', shape='rectangular', dims=arenaXY, torus=torus)
        else:
            raise
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
                    w['puffs'][id] = reg.get_null('air_puff', **args)
            else:
                w['puffs'] = {}
            w = reg.get_null('windscape', **w)
        if th is not None:
            th = reg.get_null('thermoscape', **th)
        return reg.get_null('Env', arena=arena, food_params=f, odorscape=o, border_list=bl, windscape=w,
                             thermoscape=th)


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
        return {
                'Left_patch': reg.get_null('source', pos=(-0.06, 0.0), default_color='green', group='Source', radius=0.025,
                                            amount=0.1, odor=oG(id='Odor'), type=type),
                'Right_patch': reg.get_null('source', pos=(0.06, 0.0), default_color='green', group='Source', radius=0.025,
                                             amount=0.1, odor=oG(id='Odor'), type=type)
                }



    def maze_conf(n, h):
        def maze(nx=15, ny=15, ix=0, iy=0, h=0.1, return_points=False):
            from larvaworld.lib.model.envs.maze import Maze
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

        return env((h, h),
                   f={'source_groups': {},
                      'food_grid': None,
                      'source_units': su('Target', o=oG(), c='blue')},
                   # f=f_pars(su=su('Target', o=preg.oG(), c='blue')),
                   o='G',
                   bl={'Maze': {
                       'points': maze(nx=n, ny=n, h=h, return_points=True),
                       'default_color': 'black',
                       'width': 0.001}})


    def game_env(dim=0.1, x=0.4, y=0.0):
        x = np.round(x * dim, 3)
        y = np.round(y * dim, 3)

        sus = {**su('Flag', c='green', can_be_carried=True, a=0.01, o=oG(2, id='Flag_odor')),
               **su('Left_base', pos=(-x, y), c='blue', o=oG(id='Left_base_odor')),
               **su('Right_base', pos=(+x, y), c='red', o=oG(id='Right_base_odor'))}

        return env((dim, dim),
                   f={'source_groups': {},
                      'food_grid': None,
                      'source_units': sus},
                   o='G')





    d = {
    'focus': env((0.01, 0.01)),
    'dish': env(0.1),
    'dish_40mm': env(0.04),
    'arena_200mm': env((0.2, 0.2)),
    'arena_500mm': env((0.5, 0.5)),
    'arena_1000mm': env((1.0, 1.0)),
    'odor_gradient': env((0.1, 0.06), f_pars(su=su(pos=(0.04, 0.0), o=oG(2))), 'G'),
    'mid_odor_gaussian': env((0.1, 0.06), f_pars(su=su(o=oG())), 'G'),
    'mid_odor_gaussian_square': env((0.2, 0.2), f_pars(su=su(o=oG())), 'G'),
    'mid_odor_diffusion': env((0.3, 0.3), f_pars(su=su(r=0.03, o=oD())), 'D'),
    '4corners': env((0.2, 0.2), f_pars(su=foodNodor_4corners()), 'D'),
    'food_at_bottom': env((0.2, 0.2),
                          f_pars(sg=sg('FoodLine', o=oG(), a=0.002, r=0.001, N=20, sh='oval', s=(0.01, 0.0),
                                       m='periphery')), 'G'),
    'thermo_arena': env((0.3, 0.3), th={}),
    'windy_arena': env((0.3, 0.3), w={'wind_speed': 10.0}),
    'windy_blob_arena': env((0.128, 0.014),
                            f_pars(sg=sgs(1, qs=np.ones(4), cs=aux.N_colors(4), N=1, s=(0.0, 0.0), loc=(0.005, 0.0),
                                          m='uniform', shape='rectangular', can_be_displaced=True,
                                          regeneration=True,
                                          regeneration_pos={'loc': (0.005, 0.0), 'scale': (0.0, 0.0)})),
                            w={'wind_speed': 1.0}),
    'windy_arena_bordered': env((0.3, 0.3), w={'wind_speed': 10.0},
                                bl={'Border': vborder(-0.03, [-0.01, -0.06], w=0.005)}),
    'puff_arena_bordered': env((0.3, 0.3), w={'puffs': {'PuffGroup': {}}},
                               bl={'Border': vborder(-0.03, [-0.01, -0.06], w=0.005)}),
    'single_puff': env((0.3, 0.3),
                       w={'puffs': {'Puff': {'N': 1, 'duration': 30.0, 'start_time': 55, 'speed': 100}}}),

    'CS_UCS_on_food': env(0.1, f_pars(grid=reg.get_null('food_grid'), su=CS_UCS(1)), 'G'),
    'CS_UCS_on_food_x2': env(0.1, f_pars(grid=reg.get_null('food_grid'), su=CS_UCS(2)), 'G'),
    'CS_UCS_off_food': env(0.1, f_pars(su=CS_UCS(1)), 'G'),

    'patchy_food': env((0.2, 0.2), f_pars(sg=sg(N=8, s=0.07, m='periphery', a=0.001, o=oG(2))), 'G'),
    'random_food': env((0.1, 0.1), f_pars(sg=sgs(4, N=1, s=0.04, m='uniform', shape='rectangular')), 'G'),
    'uniform_food': env(0.05, f_pars(sg=sg(N=2000, s=0.025, a=0.01, r=0.0001))),
    'patch_grid': env((0.2, 0.2), f_pars(sg=sg(N=5*5, s=0.2, a=0.01, r=0.007, m='grid', shape='rectangular', o=oG(0.2))), 'G', torus=True),

    'food_grid': env((0.02, 0.02), f_pars(grid=reg.get_null('food_grid'))),
    'single_odor_patch': env((0.1, 0.1), f_pars(su=su('Patch', a=0.1, r=0.01, o=oG())), 'G'),
    'single_patch': env((0.05, 0.05), f_pars(su=su('Patch', a=0.1, r=0.01))),
    'multi_patch': env((0.02, 0.02), f_pars(sg=sg(N=8, s=0.007, m='periphery', a=0.1, r=0.0015))),
    'double_patch': env((0.24, 0.24),
                        f_pars(su=double_patches()),
                        'G'),

    'maze': maze_conf(15, 0.1),
    'game': game_env(),
    'arena_50mm_diffusion': env(0.05, o='D'),
}
    return d