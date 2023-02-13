import copy
import random

import numpy as np
import PySimpleGUI as sg

import larvaworld.lib.aux.xy
from larvaworld.lib import reg, aux
from larvaworld.gui import gui_aux



class DrawEnvTab(gui_aux.DrawTab):
    def __init__(self,canvas_size = (800, 800), **kwargs):
        super().__init__(canvas_size = canvas_size,**kwargs)
        self.S, self.L, self.B = 'Source', 'Larva', 'Border'
        self.S0, self.L0, self.B0=self.S.lower(),self.L.lower(), self.B.lower()
        self.Su, self.Sg, self.SD = f'{self.S0}_units', f'{self.S0}_groups', f'{self.S0}_distro'
        self.Lu, self.Lg, self.LD = f'{self.L0}_units', f'{self.L0}_groups', f'{self.L0}_distro'
        self.Bg = f'{self.B0}_list'

    # @property
    def food_on(self, w):
        n = self.S0
        o = f'{n}_food'
        return w[f'TOGGLE_{o}'].get_state()

    def odor_on(self, w, n):
        o = f'{n}_odor'
        To = f'TOGGLE_{o}'
        return w[To].get_state()

    @property
    def food_ks(self):
        i = self.S0
        k_f = f'{i}_food'
        k_fM = f'{k_f}_amount'
        return k_f, k_fM

    def odor_ks(self, i):
        o = f'{i}_odor'
        o0 = f'{o}_odor_id'
        oM = f'{o}_odor_intensity'
        oS = f'{o}_odor_spread'
        # To=f'TOGGLE_{o}'
        return o, o0, oM, oS

    def group_ks(self, i):
        # if i == self.S:
        #     D = 'spatial_distro'
        # elif i==self.L :
        #     D ='larva_distro'

        s = f'{i}_single'
        g = f'{i}_group'
        s0 = f'{i}_id'
        g0 = f'{g}_id'
        D = f'{i.lower()}_distro'
        DN = f'{D}_N'
        Dm = f'{D}_mode'
        Ds = f'{D}_shape'
        # return g, g0, s, s0
        return g, g0, D, DN, Dm, Ds, s, s0

    def arena_pars(self, v, w, c):
        dic = c['arena'].get_dict(v, w)
        return dic['shape'], dic['dims']



    def add_agent_layout(self, o, color, c):
        kws={
            'toggle' : False,
            'value_kws' : gui_aux.t_kws(12),
            'text_kws' : gui_aux.t_kws(8),
            'header_width' : 14
        }

        U,G=f'{o}_unit',f'{o}_group'
        Uid,Gid=f'{U}_id', f'{G}_id'
        od, D=f'{o}_odor', f'{o}_distro'
        sh,fo=f'{o}_shape',f'{o}_food'
        # o=n0.lower()

        s1 = gui_aux.PadDict(D, disabled=True, disp_name='Distribution', **kws)
        s2 = gui_aux.PadDict(od, disp_name='Odor', dict_name='odor', **kws)
        s_all = [s1, s2]

        kws2 = {
            'group_id': 2,
            'disabled': True,
            'enable_events': True,
            **gui_aux.t_kws(5)
        }


        l = [[sg.R(f'Add {o}', 1, k=o, enable_events=True, **gui_aux.t_kws(10)), *gui_aux.color_pick_layout(o, color)],
             [sg.T('', **gui_aux.t_kws(2)), sg.R('single ID',k=U, **kws2),sg.In(o, k=Uid)],
             [sg.T('', **gui_aux.t_kws(2)), sg.R('group ID',k=G, **kws2),sg.In(G,k=Gid)],


             ]
        if o==self.S0 :
            l.append([sg.T('', **gui_aux.t_kws(2)), sg.T('shape', **gui_aux.t_kws(5)),
                      sg.Combo(['rect', 'circle'], default_value='circle', k=sh, enable_events=True,
                               readonly=True)])
            ss = gui_aux.PadDict(fo,disp_name='Food',dict_name='nutrient', **kws)
            s_all.append(ss)

        for ss in s_all:
            c.update(ss.get_subdicts())
            l.append([sg.T('', **gui_aux.t_kws(2)), *ss.get_layout()[0]])

        return l, c

    def build(self):
        S0, L0, B0 = self.S0, self.L0, self.B0
        dic = {
            'env_db': self.set_env_db(store=False),
            's': None,
            'arena': None,
            'sample_fig': None,
            'sample_pars': {},
            'dragging': None,
            'drag_figures': [],
            'current': {},
            'last_xy': (0, 0),
            'start_point': None,
            'end_point': None,
            'prior_rect': None,
            'P1': None,
            'P2': None,
        }
        c = {}
        lS, c = self.add_agent_layout(S0, 'green', c)
        lL, c = self.add_agent_layout(L0, 'black', c)

        lI = [[sg.R('Erase item', 1, k='-ERASE-', enable_events=True)],
              [sg.R('Move item', 1, True, k='-MOVE-', enable_events=True)],
              [sg.R('Inspect item', 1, True, k='-INSPECT-', enable_events=True)]]
        lB = [[sg.R(f'Add {B0}', 1, k=B0, enable_events=True, **gui_aux.t_kws(10)), *gui_aux.color_pick_layout(B0, 'black')],
              [sg.T('', **gui_aux.t_kws(2)), sg.T('id', **gui_aux.t_kws(5)), sg.In(B0, k=f'{B0}_id')],
              [sg.T('', **gui_aux.t_kws(2)), sg.T('width', **gui_aux.t_kws(5)),
               sg.Spin(values=np.arange(0.1, 1000, 0.1).tolist(), initial_value=0.001, k=f'{B0}_width')],
              ]

        col2 = sg.Col([[sg.Pane([sg.Col(ll, **gui_aux.col_kws)], border_width=8, pad=(10,10))] for ll in [lL, lB, lI]])
        col3=sg.Col([[sg.Pane([sg.Col(lS, **gui_aux.col_kws)], border_width=8, pad=(10,10))]], **gui_aux.col_kws)
        g1 = gui_aux.GraphList(self.name, tab=self, graph=True, canvas_size=self.canvas_size, canvas_kws={
            'graph_bottom_left': (0, 0),
            'graph_top_right': self.canvas_size,
            'change_submits': True,
            'drag_submits': True,
            'background_color': 'black',
        })

        col1 = [
            g1.canvas.get_layout(as_pane=True, pad=(0,10))[0],
            [sg.T('Hints : '), sg.T('', k='info', **gui_aux.t_kws(40))],
            [sg.T('Actions : '), sg.T('', k='out', **gui_aux.t_kws(40))],
        ]
        l = [[sg.Col(col1,**gui_aux.col_kws), col2, col3]]

        self.graph = g1.canvas_element

        return l, c, {g1.name: g1}, {self.name: dic}

    def eval(self, e, v, w, c, d, g):

        S0, L0, B0, S, L, B = self.S0, self.L0,self.B0,self.S, self.L, self.B
        gg = self.graph
        gg.bind('<Button-3>', '+RIGHT+')
        dic = self.base_dict

        info = w["info"]
        if e == 'RESET_ARENA':
            self.reset_arena(v, w, c)
        elif e == 'NEW_ARENA':
            w['out'].update(value='New arena initialized. All items erased.')
            self.draw_arena(v, w, c)
            self.set_env_db()

        if e == '-MOVE-':
            gg.Widget.config(cursor='fleur')
        elif not e.startswith('-GRAPH-'):
            gg.Widget.config(cursor='left_ptr')
        db = self.base_dict['env_db']
        if e == self.canvas_k:
            x, y = v[self.canvas_k]
            if not dic['dragging']:
                self.set_drag_ps(p1=(x, y))
                dic['dragging'] = True
                dic['drag_figures'] = [f for f in gg.get_figures_at_location((x, y)) if f != dic['arena']]
                dic['last_xy'] = x, y
            else:
                self.set_drag_ps(p2=(x, y))
            self.delete_prior()
            delta_x, delta_y = x - dic['last_xy'][0], y - dic['last_xy'][1]
            dic['last_xy'] = x, y
            if None not in self.get_drag_ps():
                if v['-MOVE-']:
                    delta_X, delta_Y = delta_x / self.s, delta_y / self.s
                    for fig in dic['drag_figures']:
                        for k in list(db.keys()):
                            if fig in list(db[k]['figs'].keys()):
                                id = db[k]['figs'][fig]
                                w['out'].update(value=f"Item {id} moved by ({delta_X}, {delta_Y})")
                                figs = [k for k, v in db[k]['figs'].items() if v == id]
                                for f in figs:
                                    if k == self.Su:
                                        X0, Y0 = db[k]['items'][id]['pos']
                                        db[k]['items'][id]['pos'] = (X0 + delta_X, Y0 + delta_Y)
                                    elif k in [self.Sg, self.Lg]:
                                        X0, Y0 = db[k]['items'][id]['loc']
                                        db[k]['items'][id]['loc'] = (X0 + delta_X, Y0 + delta_Y)
                                    elif k == self.Bg:
                                        db[k]['items'][id]['points'] = [(X0 + delta_X, Y0 + delta_Y) for X0, Y0 in
                                                                        db[k]['items'][id]['points']]
                                    gg.move_figure(f, delta_x, delta_y)
                        gg.update()
                elif v['-ERASE-']:
                    for fig in dic['drag_figures']:
                        for k in list(db.keys()):
                            if fig in list(db[k]['figs'].keys()):
                                id = db[k]['figs'][fig]
                                w['out'].update(value=f"Item {id} erased")
                                figs = [k for k, v in db[k]['figs'].items() if v == id]
                                for f in figs:
                                    gg.delete_figure(f)
                                    db[k]['figs'].pop(f)
                                db[k]['items'].pop(id)
                elif v['-INSPECT-']:
                    for fig in dic['drag_figures']:
                        for k in list(db.keys()):
                            if fig in list(db[k]['figs'].keys()):
                                id = db[k]['figs'][fig]
                                w['out'].update(value=f"Inspecting item {id} ")
                                if k in [self.Sg, self.Lg]:
                                    figs = self.inspect_distro(**db[k]['items'][id])
                                    for f in figs:
                                        db[k]['figs'][f] = id
                elif v[S0] or v[B0] or v[L0]:
                    P1, P2 = self.get_drag_ps(scaled=True)
                    p1, p2 = self.get_drag_ps(scaled=False)
                    if any([self.out_of_bounds(P, v, w, c) for P in [P1, P2]]):
                        dic['current'] = {}
                    else:
                        if v[S0] and not self.check_abort(S0, w, v, db[self.Su]['items'], db[self.Sg]['items']):
                            o = S0
                            U, G = f'{o}_unit', f'{o}_group'
                            Uid, Gid = f'{U}_id', f'{G}_id'
                            od, D = f'{o}_odor', f'{o}_distro'
                            sh, fo = f'{o}_shape', f'{o}_food'
                            color = v[f'{o}_color']
                            if v[Uid] or (v[Gid] and dic['sample_fig'] is None):
                                fill_color = color if self.food_on(w) else None
                                dic['prior_rect'] = self.draw_shape(shape=v[sh], p1=p1, p2=p2,
                                                                    line_color=color, fill_color=fill_color)
                                temp = np.max(np.abs(np.array(p2) - np.array(p1)))
                                w[f'{fo}_radius'].update(value=temp / self.s)
                                dic['sample_pars'] = {'default_color': color,
                                                      **c[fo].get_dict(v, w),
                                                      'odor': c[od].get_dict(v, w),
                                                      }
                                if v[Uid]:
                                    dic['current'] = {v[Uid]: {
                                        'group': v[Gid],
                                        'pos': P1,
                                        **dic['sample_pars']
                                    }}
                                    dic['sample_fig'], dic['sample_pars'] = None, {}
                                else:
                                    info.update(value=f"Draw a sample item for the distribution")
                            elif v[G]:

                                self.update_window_distro(v, w, o)
                                temp_dic = {
                                    'distribution': c[D].get_dict(v, w),
                                    **dic['sample_pars']
                                }
                                dic['current'] = {v[Gid]: reg.get_null('SourceGroup', **temp_dic)}
                                dic['prior_rect'] = self.draw_shape(shape=v[f'{D}_shape'], p1=p1,
                                                                    p2=p2, line_color=color)
                        elif v[L0] and not self.check_abort(L0, w, v, db[self.Lu]['items'], db[self.Lg]['items']):
                            o = S0
                            U, G = f'{o}_unit', f'{o}_group'
                            Uid, Gid = f'{U}_id', f'{G}_id'
                            od, D = f'{o}_odor', f'{o}_distro'
                            # sh, fo = f'{o}_shape', f'{o}_food'
                            color = v[f'{o}_color']
                            sample_larva_pars = {'default_color': color,
                                                 'odor': c[od].get_dict(v, w),
                                                 }
                            if v[G]:
                                self.update_window_distro(v, w, o)
                                temp = c[D].get_dict(v, w)
                                model = temp['model']
                                temp.pop('model')
                                temp_dic = {
                                    'model': model,
                                    'distribution': temp,
                                    **sample_larva_pars
                                }
                                dic['current'] = {v[Gid]: reg.get_null('LarvaGroup', **temp_dic)}
                                dic['prior_rect'] = self.draw_shape(shape=v[f'{D}_shape'], p1=p1,
                                                                    p2=p2, line_color=color)

                        elif v[B]:
                            id = v[f'{B}_id']
                            if id in list(db[self.Bg]['items'].keys()) or id == '':
                                info.update(value=f"{B} id {id} already exists or is empty")
                            else:
                                dic['current'] = {id:reg.get_null('Border', points=[P1, P2], default_color=v[f'{B}_color'], width=float(v[f'{B}_width']))}
                                # dic['current'] = border(ps=[P1, P2], c=v[f'{B}_color'],w=float(v[f'{B}_width']), id=id)
                                dic['prior_rect'] = self.graph.draw_line(p1, p2, color=v[f'{B}_color'],
                                                                         width=int(float(v[f'{B}_width']) * self.s))
        elif e.endswith('+UP'):  # The drawing has ended because mouse up
            if dic['arena'] is None:
                return d, g
            P1, P2 = self.get_drag_ps(scaled=True)
            current, prior_rect, sample_pars = dic['current'], dic['prior_rect'], dic['sample_pars']
            if v[B0] and current != {}:
                o = B0
                units = db[self.Bg]
                id = v[f'{o}_id']
                w['out'].update(value=f"{B} {id} placed from {P1} to {P2}")
                units['figs'][prior_rect] = id
                units['items'].update(current)
                w[f'{o}_id'].update(value=f"{B}_{len(units['items'].keys())}")
                c[self.Bg].update(w, units['items'])
            elif v[S0]:
                o = S0
                U, G = f'{o}_unit', f'{o}_group'
                Uid, Gid = f'{U}_id', f'{G}_id'
                od, D = f'{o}_odor', f'{o}_distro'
                # sh, fo = f'{o}_shape', f'{o}_food'
                # color = v[f'{o}_color']

                # oG = f'{o}_group'
                units, groups = db[self.Su], db[self.Sg]
                if v[U] and current != {}:
                    id = v[Uid]
                    w['out'].update(value=f'{S} {id} placed at {P1}')
                    units['figs'][prior_rect] = id
                    units['items'].update(current)
                    w[Uid].update(value=f'{S}_{len(units["items"].keys())}')
                    w[f'{od}_odor_id'].update(value='')
                    c[self.Su].update(w, units['items'])
                elif v[G] and sample_pars != {}:
                    id = v[Gid]
                    if current == {}:
                        info.update(value=f"Sample item for source group {id} detected." \
                                          "Now draw the distribution'sigma space")
                        dic['sample_fig'] = prior_rect
                    else:
                        w['out'].update(value=f'{o} group {id} placed at {P1}')
                        groups['items'].update(current)
                        w[Gid].update(value=f'{G}_{len(groups["items"].keys())}')
                        w[f'{od}_odor_id'].update(value='')
                        figs = self.inspect_distro(**groups['items'][id], item=o)
                        for f in figs:
                            groups['figs'][f] = id
                        self.delete_prior()
                        self.delete_prior(dic['sample_fig'])
                        dic['sample_fig'], dic['sample_pars'] = None, {}
                        c[self.Sg].update(w, groups['items'])
            elif v[L0] and current != {}:
                o = L0
                U, G = f'{o}_unit', f'{o}_group'
                Uid, Gid = f'{U}_id', f'{G}_id'
                od, D = f'{o}_odor', f'{o}_distro'
                units, groups = db[self.Lu], db[self.Lg]
                if v[U]:
                    pass
                elif v[G]:
                    id = v[Gid]
                    w['out'].update(value=f"{o} group {id} placed at {P1}")
                    groups['items'].update(current)
                    w[Gid].update(value=f"{G}_{len(groups['items'].keys())}")
                    w[f'{od}_odor_id'].update(value='')
                    figs = self.inspect_distro(**groups['items'][id], item=o)
                    for f in figs:
                        groups['figs'][f] = id
                    self.delete_prior()
                    c[self.Lg].update(w, groups['items'])
            else:
                self.delete_prior()
            self.aux_reset()

        for o in [S0, L0]:
            U, G = f'{o}_unit', f'{o}_group'
            Uid, Gid = f'{U}_id', f'{G}_id'
            od, D = f'{o}_odor', f'{o}_distro'
            w[U].update(disabled=not v[o])
            w[G].update(disabled=not v[o])
            c[D].disable(w) if not v[G] else c[D].enable(w)
            if e==G:
                w[Uid].update(value='')
            elif e==U:
                w[Gid].update(value='')


        return d, g

    def update_window_distro(self, v, w, name):
        D = f'{name.lower()}_distro'
        P1, P2 = self.get_drag_ps(scaled=True)
        s = np.abs(np.array(P2) - np.array(P1))
        if v[f'{D}_shape'] == 'circle':
            s = tuple([np.max(s), np.max(s)])
        else:
            s = tuple(s / 2)
        w[f'{D}_scale'].update(value=s)
        w[f'{D}_loc'].update(value=P1)

    def draw_shape(self, p1, p2, shape, **kwargs):
        g = self.graph
        if p2 == p1:
            return None
        pp1, pp2 = np.array(p1), np.array(p2)
        dpp = np.abs(pp2 - pp1)
        if shape in ['rect', 'oval']:
            p1 = tuple(pp1 - dpp / 2)
            p2 = tuple(pp1 + dpp / 2)
            if shape == 'rect':
                f = g.draw_rectangle(p1, p2, line_width=5, **kwargs)
            elif shape == 'oval':
                f = g.draw_oval(p1, p2, line_width=5, **kwargs)
        elif shape == 'circle':
            f = g.draw_circle(p1, np.max(dpp), line_width=5, **kwargs)
        else:
            f = None
        return f

    def draw_arena(self, v, w, c):
        W, H = self.graph_list.canvas_size
        g = self.graph
        g.erase()
        shape, (X, Y) = self.arena_pars(v, w, c)
        kws = {'fill_color': 'white',
               'line_color': 'black'}
        if shape == 'circular' and X is not None:
            arena = g.draw_circle((int(W / 2), int(H / 2)), int(W / 2), line_width=5, **kws)
            s = W / X
        elif shape == 'rectangular' and X is not None and Y is not None:
            if X >= Y:
                dif = (X - Y) / X
                arena = g.draw_rectangle((0, int(H * dif / 2)), (W, H - int(H * dif / 2)), line_width=5, **kws)
                s = W / X
            else:
                dif = (Y - X) / Y
                arena = g.draw_rectangle((int(W * dif / 2), 0), (W - int(W * dif / 2), H), **kws)
                s = H / Y
        self.base_dict['s'] = s
        self.base_dict['arena'] = arena

    def reset_arena(self, v, w, c):
        db = copy.deepcopy(self.base_dict['env_db'])
        self.draw_arena(v, w, c)
        for id, ps in db[self.Su]['items'].items():
            f = self.draw_source(P0=self.scale_xy(ps['pos'], reverse=True), **ps)
            db[self.Su]['figs'][f] = id
        for id, ps in db[self.Sg]['items'].items():
            figs = self.inspect_distro(item=self.S0, **ps)
            for f in figs:
                db[self.Sg]['figs'][f] = id
        for id, ps in db[self.Lg]['items'].items():
            figs = self.inspect_distro(item=self.L0, **ps)
            for f in figs:
                db[self.Lg]['figs'][f] = id
        for id, ps in db[self.Bg]['items'].items():
            points = [self.scale_xy(p) for p in ps['points']]
            f = self.graph.draw_lines(points=points, color=ps['default_color'],
                                      width=int(ps['width'] * self.s))
            db[self.Bg]['figs'][f] = id
        w['out'].update(value='Arena has been reset.')
        self.base_dict['env_db'] = db



    def out_of_bounds(self, xy, v, w, c):
        shape, (X, Y) = self.arena_pars(v, w, c)
        x, y = xy
        if shape == 'circular':
            return np.sqrt(x ** 2 + y ** 2) > X / 2
        elif shape == 'rectangular':
            return not (-X / 2 < x < X / 2 and -Y / 2 < y < Y / 2)

    def delete_prior(self, fig=None):
        if fig is None:
            fig = self.base_dict['prior_rect']
        g = self.graph
        if fig is None:
            pass
        elif type(fig) == list:
            for f in fig:
                g.delete_figure(f)
        else:
            g.delete_figure(fig)

    def inspect_distro(self, item, default_color, mode=None, shape=None, N=None, loc=None, scale=None,
                       orientation_range=None, distribution=None, **kwargs):
        if distribution is not None:
            mode = distribution['mode']
            shape = distribution['shape']
            N = distribution['N']
            loc = distribution['loc']
            scale = distribution['scale']

        Ps = aux.generate_xy_distro(mode, shape, N, loc=self.scale_xy(loc, reverse=True),
                                               scale=np.array(scale) * self.s)
        group_figs = []
        for i, P0 in enumerate(Ps):
            if item == self.S0:
                temp = self.draw_source(P0, default_color, **kwargs)
            elif item == self.L0:
                if distribution is not None:
                    orientation_range = distribution['orientation_range']
                temp = self.draw_larva(P0, default_color, orientation_range, **kwargs)
            group_figs.append(temp)
        return group_figs

    def draw_source(self, P0, default_color, amount, radius, **kwargs):
        fill_color = default_color if amount > 0 else None
        temp = self.graph.draw_circle(P0, radius * self.s, line_width=3, line_color=default_color,
                                      fill_color=fill_color)
        return temp

    def draw_larva(self, P0, color, orientation_range, **kwargs):
        xy0 = larvaworld.aux.xy.body_contour() - np.array([0.5, 0.0])
        xy0 = aux.rotate_points_around_point(xy0, random.uniform(*np.deg2rad(orientation_range)), origin=[0, 0])
        xy0 = xy0 * self.s / 250 + np.array(P0)
        temp = self.graph.draw_polygon(xy0, line_width=3, line_color=color, fill_color=color)
        return temp

    def check_abort(self, n, w, v, units, groups):
        S0, L0 = self.S0, self.L0



        o, o0, oM, oS = self.odor_ks(n)
        f, fM = self.food_ks
        info = w['info']
        abort = True
        odor_on = self.odor_on(w, n)
        food_on = self.food_on(w)
        D = f'{n}_distro'
        DN = f'{D}_N'
        Dm = f'{D}_mode'
        Ds = f'{D}_shape'
        U, G = f'{n}_unit', f'{n}_group'
        Uid, Gid = f'{U}_id', f'{G}_id'

        if not odor_on:
            w[o0].update(value=None)
            w[oM].update(value=0.0)

        if n == S0:
            if not odor_on and not food_on:
                info.update(value=f"Assign food and/or odor to the drawn source")
                return True
            elif food_on and float(v[fM]) == 0.0:
                w[fM].update(value=10 ** -3)
                info.update(value=f"{S0} food amount set to default")
                return True
            elif not food_on and float(v[fM]) != 0.0:
                w[fM].update(value=0.0)
                # t = f"{S} food amount set to 0"
        elif n == L0:
            if v[f'{D}_model'] == '':
                info.update(value="Assign a larva-model for the larva group")
                return True

        if v[Gid] == '' and v[Uid] == '':
            t = f"Both {n} single id and group id are empty"
        elif not v[G] and not v[U]:
            t = f"Select to add a single or a group of {n}s"
        elif v[U] and (v[Uid] in list(units.keys()) or v[Uid] == ''):
            t = f"{n} id {v[Uid]} already exists or is empty"
        elif odor_on and v[o0] == '':
            t = "Default odor id automatically assigned to the odor"
            id = v[Gid] if v[Gid] != '' else v[Uid]
            w[o0].update(value=f'{id}_odor')
        elif odor_on and not float(v[oM]) > 0:
            t = "Assign positive odor intensity to the drawn odor source"
        elif odor_on and (v[oS] == '' or not float(v[oS]) > 0):
            t = "Assign positive spread to the odor"
        elif v[G] and (v[Gid] in list(groups.keys()) or v[Gid] == ''):
            t = f"{n} group id {v[Gid]} already exists or is empty"
        elif v[G] and v[Dm] in ['', None]:
            t = "Define a distribution mode"
        elif v[G] and v[Ds] in ['', None]:
            t = "Define a distribution shape"
        elif v[G] and not int(v[DN]) > 0:
            t = "Assign a positive integer number of items for the distribution"
        else:
            t = "Valid item added!"
            abort = False
        info.update(value=t)
        return abort

    def set_env_db(self, env=None, lg={}, store=True):
        if env is None:

            env = {self.Bg: {},
                   'arena': reg.get_null('arena'),
                   'food_params': {self.Su: {}, self.Sg: {}, 'food_grid': None},
                   }
        items = [env[self.Bg],
                 env['food_params'][self.Su], env['food_params'][self.Sg],
                 {}, lg]
        dic = {k: {'items': ii, 'figs': {}} for k, ii in zip([self.Bg, self.Su, self.Sg, self.Lu, self.Lg], items)}
        if store:
            self.base_dict['env_db'] = dic
        else:
            return dic


