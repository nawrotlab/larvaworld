import numpy as np
from matplotlib import colors

from larvaworld.lib import reg, aux



@reg.funcs.stored_conf("Env")
def Env_dict() :
    from larvaworld.lib.reg import gen

    def oG(c=1, id='Odor'):
        return gen.Odor(id=id, intensity=2.0 * c, spread=0.0002 * np.sqrt(c)).nestedConf
        # return reg.get_null('odor', id=id, intensity=2.0 * c, spread=0.0002 * np.sqrt(c))


    def oD(c=1, id='Odor'):
        return gen.Odor(id=id, intensity=300.0 * c, spread=0.1 * np.sqrt(c)).nestedConf

    def border(ps, c='black', w=0.01, id=None):
        b = gen.Border(vertices=ps, default_color=c, width=w)
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

    def su2(id='Source',**kwargs):
        return gen.Food(**kwargs).entry(id)


    def sg2(id='Source', **kwargs):

        return gen.FoodGroup(**kwargs).entry(id)


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
        l = [sg2(id=ids[i], c=colors.rgb2hex(cs[i]), r=rs[i], a=ams[i], odor=os[i], sub=[qs[i], 'standard'], **kwargs) for i in range(Ngs)]
        result = {}
        for d in l:
            result.update(d)
        return result


    def f_pars(sg={}, su={}, grid=None):
        return gen.FoodConf(source_units =su, source_groups =sg,food_grid=grid)



    def foodNodor_4corners(d=0.05):
        l = [su2(f'Source_{i}', pos=p, a=0.01, odor=oD(id=f'Odor_{i}'), c=c, r=0.01) for i, (c, p) in
             enumerate(zip(['blue', 'red', 'green', 'magenta'], [(-d, -d), (-d, d), (d, -d), (d, d)]))]
        dic = {**l[0], **l[1], **l[2], **l[3]}
        return dic


    def env(arenaXY, f=f_pars(), o=None, bl={}, w=None, th=None, torus=False):
        if type(arenaXY) == float:
            arena = gen.Arena(geometry='circular', dims=(arenaXY, arenaXY), torus=torus)
            # arena = reg.get_null('arena', geometry='circular', dims=(arenaXY, arenaXY), torus=torus)
        elif type(arenaXY) == tuple:
            arena = gen.Arena(geometry='rectangular', dims=arenaXY, torus=torus)
            # arena = reg.get_null('arena', geometry='rectangular', dims=arenaXY, torus=torus)
        else:
            raise
        if o == 'D':
            o = gen.DiffusionValueLayer()
        elif o == 'G':
            o = gen.GaussianValueLayer()
        if w is not None:
            if 'puffs' in w.keys():
                for id, args in w['puffs'].items():
                    w['puffs'][id] = reg.get_null('air_puff', **args)
            else:
                w['puffs'] = {}
            w = gen.WindScape(**w)
        if th is not None:
            th = gen.ThermoScape(**th)
        return gen.Env(arena=arena, food_params=f, odorscape=o, border_list=bl, windscape=w,thermoscape=th).nestedConf


    def CS_UCS(N=2, x=0.04):
        if N == 1:
            return {**su2('CS', pos=(-x, 0.0), odor=oG(id='CS'), c='red'),
                    **su2('UCS', pos=(x, 0.0), odor=oG(id='UCS'), c='blue')}
        elif N == 2:
            return {
                **su2('CS_l', pos=(-x, 0.0), odor=oG(id='CS'), c='red'),
                **su2('CS_r', pos=(x, 0.0), odor=oG(id='CS'), c='red'),
                **su2('UCS_l', pos=(-x, 0.0), odor=oG(id='UCS'), c='blue'),
                **su2('UCS_r', pos=(x, 0.0), odor=oG(id='UCS'), c='blue')
            }


    def double_patches(type='standard'):
        return {**su2('Left_patch', pos=(-0.06, 0.0),c='green',group='Source',r=0.025,a=0.1,odor=oG(id='Odor'), sub=[1.0,type]),
                **su2('Right_patch', pos=(0.06, 0.0), c='green',group='Source',r=0.025,a=0.1, odor=oG(id='Odor'),sub=[1.0,type])}


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

        return env((h, h),f=f_pars(su =su2('Target', odor=oG(), c='blue')),
                   o='G',
                   bl=border(maze(nx=n, ny=n, h=h, return_points=True), c='black', w=0.001, id='Maze'))


    def game_env(dim=0.1, x=0.4, y=0.0):
        x = np.round(x * dim, 3)
        y = np.round(y * dim, 3)

        sus = {**su2('Flag', c='green', can_be_carried=True, a=0.01, odor=oG(2, id='Flag_odor')),
               **su2('Left_base', pos=(-x, y), c='blue', odor=oG(id='Left_base_odor')),
               **su2('Right_base', pos=(+x, y), c='red', odor=oG(id='Right_base_odor'))}

        return env((dim, dim),f=f_pars(su =sus),o='G')





    d = {
    'focus': env((0.01, 0.01)),
    'dish': env(0.1),
    'dish_40mm': env(0.04),
    'arena_200mm': env((0.2, 0.2)),
    'arena_500mm': env((0.5, 0.5)),
    'arena_1000mm': env((1.0, 1.0)),
    'odor_gradient': env((0.1, 0.06), f_pars(su=su2(pos=(0.04, 0.0), odor=oG(2))), 'G'),
    'mid_odor_gaussian': env((0.1, 0.06), f_pars(su=su2(odor=oG())), 'G'),
    'mid_odor_gaussian_square': env((0.2, 0.2), f_pars(su=su2(odor=oG())), 'G'),
    'mid_odor_diffusion': env((0.3, 0.3), f_pars(su=su2(r=0.03, odor=oD())), 'D'),
    '4corners': env((0.2, 0.2), f_pars(su=foodNodor_4corners()), 'D'),
    'food_at_bottom': env((0.2, 0.2),
                          f_pars(sg=sg2('FoodLine', odor=oG(), a=0.002, r=0.001, N=20, shape='oval', scale=(0.01, 0.0),
                                       mode='periphery')), 'G'),
    'thermo_arena': env((0.3, 0.3), th={}),
    'windy_arena': env((0.3, 0.3), w={'wind_speed': 10.0}),
    'windy_blob_arena': env((0.128, 0.014),
                            f_pars(sg=sgs(1, qs=np.ones(4), cs=aux.N_colors(4), N=1, scale=(0.0, 0.0), loc=(0.005, 0.0),
                                          mode='uniform', shape='rectangular', can_be_displaced=True,
                                          regeneration=True,
                                          regeneration_pos={'loc': (0.005, 0.0), 'scale': (0.0, 0.0)})),
                            w={'wind_speed': 1.0}),
    'windy_arena_bordered': env((0.3, 0.3), w={'wind_speed': 10.0},
                                bl={'Border': vborder(-0.03, [-0.01, -0.06], w=0.005)}),
    'puff_arena_bordered': env((0.3, 0.3), w={'puffs': {'PuffGroup': {}}},
                               bl={'Border': vborder(-0.03, [-0.01, -0.06], w=0.005)}),
    'single_puff': env((0.3, 0.3),
                       w={'puffs': {'Puff': {'N': 1, 'duration': 30.0, 'start_time': 55, 'speed': 100}}}),

    'CS_UCS_on_food': env(0.1, f_pars(grid=gen.FoodGrid(), su=CS_UCS(1)), 'G'),
    'CS_UCS_on_food_x2': env(0.1, f_pars(grid=gen.FoodGrid(), su=CS_UCS(2)), 'G'),
    'CS_UCS_off_food': env(0.1, f_pars(su=CS_UCS(1)), 'G'),

    'patchy_food': env((0.2, 0.2), f_pars(sg=sg2(N=8, scale=(0.07,0.07), mode='periphery', a=0.001, odor=oG(2))), 'G'),
    'random_food': env((0.1, 0.1), f_pars(sg=sgs(4, N=1, scale=(0.04,0.04), mode='uniform', shape='rectangular')), 'G'),
    'uniform_food': env(0.05, f_pars(sg=sg2(N=2000, scale=(0.025,0.025), a=0.01, r=0.0001))),
    'patch_grid': env((0.2, 0.2), f_pars(sg=sg2(N=5*5, scale=(0.2,0.2), a=0.01, r=0.007, mode='grid', shape='rectangular', odor=oG(0.2))), 'G', torus=False),

    'food_grid': env((0.02, 0.02), f_pars(grid=gen.FoodGrid())),
    'single_odor_patch': env((0.1, 0.1), f_pars(su=su2('Patch', a=0.1, r=0.01, odor=oG())), 'G'),
    'single_patch': env((0.05, 0.05), f_pars(su=su2('Patch', a=0.1, r=0.01))),
    'multi_patch': env((0.02, 0.02), f_pars(sg=sg2(N=8, scale=(0.007,0.007), mode='periphery', a=0.1, r=0.0015))),
    'double_patch': env((0.24, 0.24),
                        f_pars(su=double_patches()),
                        'G'),

    'maze': maze_conf(15, 0.1),
    'game': game_env(),
    'arena_50mm_diffusion': env(0.05, o='D'),
}
    return d