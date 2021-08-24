import copy
import random

import numpy as np
import PySimpleGUI as sg
import lib.conf.dtype_dicts as dtypes
import lib.aux.functions as fun
from lib.gui.gui_lib import CollapsibleDict, Collapsible, CollapsibleTable, col_kws, col_size, t40_kws, b_kws, t5_kws, \
    color_pick_layout, graphic_button, t2_kws, GraphList, retrieve_dict
from lib.gui.draw_env import draw_env, build_draw_env
from lib.conf.conf import loadConf
from lib.gui.tab import GuiTab, SelectionList


class EnvTab(GuiTab):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.canvas_size=(800,800)

    # @property
    def food_on(self,w):
        return w['TOGGLE_food'].get_state()

    def odor_on(self,w, n):
        o = f'{n}_ODOR'
        To = f'TOGGLE_{o}'
        return w[To].get_state()

    @property
    def food_ks(self):
        k_f = 'food'
        k_fM = f'{k_f}_amount'
        return k_f, k_fM

    def odor_ks(self, i):
        o=f'{i}_ODOR'
        o0=f'{o}_odor_id'
        oM=f'{o}_odor_intensity'
        oS=f'{o}_odor_spread'
        # To=f'TOGGLE_{o}'
        return o, o0,oM,oS

    def group_ks(self, i):
        s=f'{i}_single'
        g=f'{i}_group'
        s0=f'{i}_id'
        g0=f'{g}_id'
        D=f'{i}_DISTRO'
        DN=f'{D}_N'
        Dm=f'{D}_mode'
        Ds=f'{D}_shape'
        return g,g0,D,DN,Dm,Ds, s,s0


    def arena_pars(self, v, w, c):
        dic=c['arena'].get_dict(v, w)
        shape = dic['arena_shape']
        X, Y = dic['arena_xdim'], dic['arena_ydim']
        return shape, (X,Y)

    @property
    def s(self):
        return self.aux_dict['s']

    # @property
    def get_drag_ps(self, scaled=False):
        d=self.aux_dict
        p1,p2= d['start_point'], d['end_point']
        return [self.scale_xy(p1), self.scale_xy(p2)] if scaled else [p1,p2]

    def set_drag_ps(self, p1=None,p2=None):
        d=self.aux_dict
        if p1 is not None :
            d['start_point']=p1
        if p2 is not None :
            d['end_point']=p2
        # return self.scale_xy(p1), self.scale_xy(p2) if scaled else p1,p2



    def update(self, w, c, conf, id=None):
        for n in ['border_list', 'larva_groups', 'arena', 'odorscape']:
            c[n].update(w, conf[n] if n in conf.keys() else {})
        for n in ['source_groups', 'source_units', 'food_grid']:
            c[n].update(w, conf['food_params'][n])
        self.aux_dict['env_db'] = self.set_env_db(env=conf)
        w.write_event_value('RESET_ARENA', 'Draw the initial arena')

    def get(self, w, v, c, as_entry=False):
        env = {
            'food_params': {n: c[n].get_dict(v, w) for n in ['source_groups', 'source_units', 'food_grid']},
            **{n: c[n].get_dict(v, w) for n in ['larva_groups', 'border_list', 'arena', 'odorscape']}
        }
        print(env['food_params']['food_grid'])

        env0 = copy.deepcopy(env)
        if not as_entry:
            for n, gr in env0['larva_groups'].items():
                if type(gr['model']) == str:
                    gr['model'] = loadConf(gr['model'], 'Model')
        return env0

    def build_conf_env(self):
        s1 = CollapsibleTable('larva_groups', False, headings=['group', 'N', 'color', 'model'],
                              type_dict=dtypes.get_dict_dtypes('distro', class_name='Larva', basic=False))
        s2 = CollapsibleTable('source_groups', False, headings=['group', 'N', 'color', 'amount', 'odor_id'],
                              type_dict=dtypes.get_dict_dtypes('distro', class_name='Source', basic=False))
        s3 = CollapsibleTable('source_units', False, headings=['id', 'color', 'amount', 'odor_id'],
                              type_dict=dtypes.get_dict_dtypes('agent', class_name='Source'))
        s4 = CollapsibleTable('border_list', False, headings=['id', 'color', 'points'],
                              type_dict=dtypes.get_dict_dtypes('agent', class_name='Border'))
        c = {}
        for s in [s1, s2, s3, s4]:
            c.update(**s.get_subdicts())
        c1 = [CollapsibleDict(n, False, default=True, **kw)
              for n, kw in zip(['arena', 'food_grid', 'odorscape'], [{'next_to_header':[
                                 graphic_button('burn', 'RESET_ARENA',
                                                tooltip='Reset to the initial arena. All drawn items will be erased.'),
                                 graphic_button('globe_active', 'NEW_ARENA',
                                                tooltip='Create a new arena.All drawn items will be erased.'),
                             ]}, {'toggle': True}, {}])]
        for s in c1:
            c.update(s.get_subdicts())
        l1 = [c[n].get_layout() for n in ['source_groups', 'source_units', 'food_grid']]
        c2 = Collapsible('Sources', True, l1)
        c.update(c2.get_subdicts())
        l2 = [c[n].get_layout() for n in ['arena', 'larva_groups', 'Sources', 'border_list', 'odorscape']]
        l1 = SelectionList(tab=self, actions=['load', 'save', 'delete'])
        self.selectionlists = {sl.conftype : sl for sl in [l1]}
        l = sg.Col([l1.l, *l2], **col_kws, size=col_size(0.25))
        return l, c, {}, {}

    def add_agent_layout(self, n0, color, collapsibles):
        n = n0.upper()
        g, g0, D, DN, Dm, Ds, s, s0 = self.group_ks(n)
        o, o0, oM, oS = self.odor_ks(n)

        s1 = CollapsibleDict(D, False, dict=dtypes.get_dict('distro', class_name=n0),
                             type_dict=dtypes.get_dict_dtypes('distro', class_name=n0),
                             toggle=False, disabled=True, disp_name='distribution')

        s2 = CollapsibleDict(o, False, dict=dtypes.get_dict('odor'),
                                                       type_dict=dtypes.get_dict_dtypes('odor'),
                                                       toggle=False, disp_name='odor')

        for ss in [s1,s2]:
            collapsibles.update(ss.get_subdicts())

        l = [[sg.R(f'Add {n0}', 1, k=n, enable_events=True)],
             [sg.T('', **t2_kws),sg.R('single id', 2, disabled=True, k=s, enable_events=True, **t5_kws),sg.In(n, k=s0)],
             [sg.T('', **t2_kws), sg.R('group id', 2, disabled=True, k=g, enable_events=True, **t5_kws),sg.In(k=g0)],
             color_pick_layout(n, color),
             [sg.T('', **t5_kws), *s1.get_layout()],
             [sg.T('', **t5_kws), *s2.get_layout()]]
        return l, collapsibles

    def build_draw_env(self):
        dic={
            'env_db': self.set_env_db(),
            's': None,
            'arena': None,
            'sample_fig': None,
            'sample_pars': {},
            'dragging': None,
            'drag_figures': [],
            'current': {},
            'last_xy': (0,0),
            'start_point': None,
            'end_point': None,
            'prior_rect': None,
            'P1': None,
            'P2': None,
        }


        c = {}
        s2 = CollapsibleDict('food', False, default=True, toggle=False)
        c.update(s2.get_subdicts())
        source_l, c = self.add_agent_layout('Source', 'green', c)
        larva_l, c = self.add_agent_layout('Larva', 'black', c)

        b='BORDER'
        b0=f'{b}_id'
        bw=f'{b}_width'
        col2 = [
            *larva_l, *source_l,
            [sg.T('', **t5_kws), *s2.get_layout()],
            [sg.T('', **t5_kws), sg.T('shape', **t5_kws),
             sg.Combo(['rect', 'circle'], default_value='circle', k='SOURCE_shape', enable_events=True, readonly=True)],

            [sg.R('Add Border', 1, k=b, enable_events=True)],
            [sg.T('', **t5_kws), sg.T('id', **t5_kws), sg.In(b, k=b0)],
            [sg.T('', **t5_kws), sg.T('width', **t5_kws), sg.In(0.001, k=bw)],
            color_pick_layout(b, 'black'),

            [sg.R('Erase item', 1, k='-ERASE-', enable_events=True)],
            [sg.R('Move item', 1, True, k='-MOVE-', enable_events=True)],
            [sg.R('Inspect item', 1, True, k='-INSPECT-', enable_events=True)],
        ]
        g1 = GraphList(self.name, graph=True,canvas_size=self.canvas_size, canvas_kws={
            'graph_bottom_left': (0, 0),
            'graph_top_right': self.canvas_size,
            'change_submits': True,
            'drag_submits': True,
            'background_color': 'black',
        })



        col1 = [
            # s1.get_layout(),
            [g1.canvas],
            [sg.T('Hints : '), sg.T('', k='info', **t40_kws)],
            [sg.T('Actions : '), sg.T('', k='out', **t40_kws)],
        ]
        l = sg.Col([[sg.Col(col1, **col_kws), sg.Col(col2, **col_kws)]], **col_kws, size=col_size(0.75))

        g = {g1.name: g1}
        self.graph_list=g1
        self.graph = g1.canvas_element

        d = {self.name: dic}
        return l, c, g, d

    def build(self):
        l1, c1, g1, d1 = self.build_conf_env()
        l2, c2, g2, d2 = self.build_draw_env()
        c = {**c1, **c2}
        g = {**g1, **g2}
        d = {**d1, **d2}
        l = [[l1, l2]]
        return l, c, g, d


    def eval(self,e, v, w, c, d, g):
        S,L,B='SOURCE','LARVA', 'BORDER'
        gg=self.graph
        gg.bind('<Button-3>', '+RIGHT+')
        dic = d[self.name]
        info = w["info"]
        if e == 'RESET_ARENA':
            self.reset_arena(v, w, c)
        elif e == 'NEW_ARENA':
            w['out'].update(value='New arena initialized. All items erased.')
            self.draw_arena(v, w, c)
            self.aux_dict['env_db'] = self.set_env_db()

        if e == '-MOVE-':
            gg.Widget.config(cursor='fleur')
        elif not e.startswith('-GRAPH-'):
            gg.Widget.config(cursor='left_ptr')
        db = self.aux_dict['env_db']
        if e == self.graph_list.canvas_key:
            x, y = v[self.graph_list.canvas_key]
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
                                    if k == 's_u':
                                        X0, Y0 = db[k]['items'][id]['pos']
                                        db[k]['items'][id]['pos'] = (X0 + delta_X, Y0 + delta_Y)
                                    elif k in ['s_g', 'l_g']:
                                        X0, Y0 = db[k]['items'][id]['loc']
                                        db[k]['items'][id]['loc'] = (X0 + delta_X, Y0 + delta_Y)
                                    elif k == 'b':
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
                                if k in ['s_g', 'l_g']:
                                    figs = self.inspect_distro(**db[k]['items'][id])
                                    for f in figs:
                                        db[k]['figs'][f] = id
                elif v[S] or v[B] or v[L]:
                    P1, P2 = self.get_drag_ps(scaled=True)
                    p1, p2 = self.get_drag_ps(scaled=False)
                    if any([self.out_of_bounds(P, v, w, c) for P in [P1, P2]]):
                        dic['current'] = {}
                    else:
                        if v[S] and not self.check_abort(S, w, v, db['s_u']['items'], db['s_g']['items']):
                            o = S
                            color = v[f'{o}_color']
                            if v['SOURCE_single'] or (v['SOURCE_group'] and dic['sample_fig'] is None):
                                fill_color = color if self.food_on(w) else None
                                dic['prior_rect'] = self.draw_shape(shape=v[f'{o}_shape'], p1=p1, p2=p2,
                                                               line_color=color, fill_color=fill_color)
                                temp = np.max(np.abs(np.array(p2) - np.array(p1)))
                                w['food_radius'].update(value=temp / self.s)
                                dic['sample_pars'] = {'default_color': color,
                                                      **c['food'].get_dict(v, w, check_toggle=False),
                                                      **c['SOURCE_ODOR'].get_dict(v, w, check_toggle=False),
                                                      }
                                if v['SOURCE_single']:
                                    dic['current'] = {v['SOURCE_id']: {
                                        'group': v['SOURCE_group_id'],
                                        'pos': P1,
                                        **dic['sample_pars']
                                    }}
                                    dic['sample_fig'], dic['sample_pars'] = None, {}
                                else:
                                    info.update(value=f"Draw a sample item for the distribution")
                            elif v[f'{o}_group']:
                                self.update_window_distro(v, w, o)
                                dic['current'] = {v['SOURCE_group_id']: {
                                    **c['SOURCE_DISTRO'].get_dict(v, w, check_toggle=False),
                                    **dic['sample_pars']
                                }}
                                dic['prior_rect'] = self.draw_shape(shape=v[f'{o}_DISTRO_shape'], p1=p1,
                                                               p2=p2, line_color=color)
                        elif v[L] and not self.check_abort(L, w, v, db['l_u']['items'], db['l_g']['items']):
                            o = L
                            color = v[f'{o}_color']
                            sample_larva_pars = {'default_color': color,
                                                 **c[f'{o}_ODOR'].get_dict(v, w, check_toggle=False),
                                                 }
                            if v[f'{o}_group']:
                                self.update_window_distro(v, w, o)
                                dic['current'] = {v[f'{o}_group_id']: {
                                    **c[f'{o}_DISTRO'].get_dict(v, w, check_toggle=False),
                                    **sample_larva_pars
                                }}
                                dic['prior_rect'] = self.draw_shape(shape=v[f'{o}_DISTRO_shape'], p1=p1,
                                                               p2=p2, line_color=color)

                        elif v[B]:
                            id = v['BORDER_id']
                            if id in list(db['b']['items'].keys()) or id == '':
                                info.update(value=f"Border id {id} already exists or is empty")
                            else:
                                dic0 = {'unique_id': id,
                                        'default_color': v['BORDER_color'],
                                        'width': v['BORDER_width'],
                                        'points': [P1, P2]}
                                dic['current'] = fun.agent_list2dict(
                                    [retrieve_dict(dic0, dtypes.get_dict_dtypes('agent', class_name='Border'))])

                                dic['prior_rect'] = self.graph.draw_line(p1, p2, color=v['BORDER_color'],
                                                                    width=int(float(v['BORDER_width']) * self.s))

        elif e.endswith('+UP'):  # The drawing has ended because mouse up
            P1, P2 = self.get_drag_ps(scaled=True)
            current, prior_rect, sample_pars = dic['current'], dic['prior_rect'], dic['sample_pars']
            if v[B] and current != {}:
                o = B
                units = db['b']
                id = v[f'{o}_id']
                w['out'].update(value=f"Border {id} placed from {P1} to {P2}")
                units['figs'][prior_rect] = id
                units['items'].update(current)
                w[f'{o}_id'].update(value=f"BORDER_{len(units['items'].keys())}")
                c['border_list'].update(w, units['items'])
            elif v[S]:
                o = S
                units, groups = db['s_u'], db['s_g']
                if v[f'{o}_single'] and current != {}:
                    id = v[f'{o}_id']
                    w['out'].update(value=f"Source {id} placed at {P1}")
                    units['figs'][prior_rect] = id
                    units['items'].update(current)
                    w[f'{o}_id'].update(value=f"SOURCE_{len(units['items'].keys())}")
                    w[f'{o}_ODOR_odor_id'].update(value='')
                    c['source_units'].update(w, units['items'])
                elif v[f'{o}_group'] and sample_pars != {}:
                    id = v[f'{o}_group_id']
                    if current == {}:
                        info.update(value=f"Sample item for source group {id} detected." \
                                          "Now draw the distribution'sigma space")

                        dic['sample_fig'] = prior_rect
                    else:
                        w['out'].update(value=f"Source group {id} placed at {P1}")
                        groups['items'].update(current)
                        w[f'{o}_group_id'].update(value=f"SOURCE_GROUP_{len(groups['items'].keys())}")
                        w[f'{o}_ODOR_odor_id'].update(value='')
                        figs = self.inspect_distro(**groups['items'][id], item=o)
                        for f in figs:
                            groups['figs'][f] = id
                        self.delete_prior()
                        self.delete_prior(dic['sample_fig'])
                        dic['sample_fig'], dic['sample_pars'] = None, {}
                        c['source_groups'].update(w, groups['items'])
            elif v[L] and current != {}:
                o = L
                units, groups = db['l_u'], db['l_g']
                if v[f'{o}_single']:
                    pass
                elif v[f'{o}_group']:
                    id = v[f'{o}_group_id']
                    w['out'].update(value=f"{o} group {id} placed at {P1}")
                    groups['items'].update(current)
                    w[f'{o}_group_id'].update(value=f"{o}_GROUP_{len(groups['items'].keys())}")
                    w[f'{o}_ODOR_odor_id'].update(value='')
                    figs = self.inspect_distro(**groups['items'][id], item=o)
                    for f in figs:
                        groups['figs'][f] = id
                    self.delete_prior()
                    sample_larva_pars = {}
                    c['larva_groups'].update(w, groups['items'])
            else:
                self.delete_prior()
            self.aux_reset()

        for o in [S, L]:
            w[f'{o}_single'].update(disabled=not v[o])
            w[f'{o}_group'].update(disabled=not v[o])
            c[f'{o}_DISTRO'].disable(w) if not v[f'{o}_group'] else c[f'{o}_DISTRO'].enable(w)
            if v[f'{o}_group']:
                w[f'{o}_id'].update(value='')

        return d, g

    def update_window_distro(self, v, w, name):
        P1, P2 = self.get_drag_ps(scaled=True)
        s = np.abs(np.array(P2) - np.array(P1))
        if v[f'{name}_DISTRO_shape'] == 'circle':
            s = tuple([np.max(s), np.max(s)])
        else:
            s = tuple(s / 2)
        w[f'{name}_DISTRO_scale'].update(value=s)
        w[f'{name}_DISTRO_loc'].update(value=P1)

    def draw_shape(self, p1, p2, shape, **kwargs):
        g=self.graph
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
        shape, (X,Y)=self.arena_pars(v, w, c)
        kws={'fill_color' : 'white',
             'line_color' : 'black'}
        if shape == 'circular' and X is not None:
            arena = g.draw_circle((int(W / 2), int(H / 2)), int(W / 2),line_width=5, **kws)
            s = W / X
        elif shape == 'rectangular' and not None in (X, Y):
            if X >= Y:
                dif = (X - Y) / X
                arena = g.draw_rectangle((0, int(H * dif / 2)), (W, H - int(H * dif / 2)), line_width=5, **kws)
                s = W / X
            else:
                dif = (Y - X) / Y
                arena = g.draw_rectangle((int(W * dif / 2), 0), (W - int(W * dif / 2), H), **kws)
                s = H / Y
        self.aux_dict['s']=s
        self.aux_dict['arena']=arena

    def reset_arena(self, v, w, c):
        db = copy.deepcopy(self.aux_dict['env_db'])
        self.draw_arena(v, w, c)
        for id, ps in db['s_u']['items'].items():
            f = self.draw_source(P0=self.scale_xy(ps['pos'], reverse=True), **ps)
            db['s_u']['figs'][f] = id
        for id, ps in db['s_g']['items'].items():
            figs = self.inspect_distro(item='SOURCE', **ps)
            for f in figs:
                db['s_g']['figs'][f] = id
        for id, ps in db['l_g']['items'].items():
            figs = self.inspect_distro(item='LARVA', **ps)
            for f in figs:
                db['l_g']['figs'][f] = id
        for id, ps in db['b']['items'].items():
            points = [self.scale_xy(p) for p in ps['points']]
            f = self.graph.draw_lines(points=points, color=ps['default_color'],
                                    width=int(ps['width'] * self.s))
            db['b']['figs'][f] = id
        w['out'].update(value='Arena has been reset.')
        self.aux_dict['env_db']=db

    def scale_xy(self, xy, reverse=False):
        if xy is None :
            return None
        W, H = self.graph_list.canvas_size
        s=self.s
        x,y=xy
        if reverse :
            return x * s + W / 2, y * s + H / 2
        else :
            return (x - W / 2) / s, (y - H / 2) / s


    def out_of_bounds(self, xy, v, w, c):
        shape, (X,Y)=self.arena_pars(v, w, c)
        x, y = xy
        if shape == 'circular':
            return np.sqrt(x ** 2 + y ** 2) > X / 2
        elif shape == 'rectangular':
            return not (-X / 2 < x < X / 2 and -Y / 2 < y < Y / 2)

    def delete_prior(self, fig=None):
        if fig is None :
            fig=self.aux_dict['prior_rect']
        g = self.graph
        if fig is None :
            pass
        elif type(fig) == list:
            for f in fig:
                g.delete_figure(f)
        else:
            g.delete_figure(fig)

    def inspect_distro(self, default_color, mode, shape, N, loc, scale, item='LARVA', **kwargs):
        Ps = fun.generate_xy_distro(mode, shape, N, loc=self.scale_xy(loc, reverse=True), scale=np.array(scale) * self.s)
        group_figs = []
        for i, P0 in enumerate(Ps):
            if item == 'SOURCE':
                temp = self.draw_source(P0, default_color, **kwargs)
            elif item == 'LARVA':
                temp = self.draw_larva(P0, default_color, **kwargs)
            group_figs.append(temp)
        return group_figs

    def draw_source(self, P0, default_color, amount, radius, **kwargs):
        fill_color = default_color if amount > 0 else None
        temp = self.graph.draw_circle(P0, radius *self.s, line_width=3, line_color=default_color, fill_color=fill_color)
        return temp

    def draw_larva(self, P0, color, orientation_range, **kwargs):
        points = np.array([[0.9, 0.1], [0.05, 0.1]])
        xy0 = fun.body(points) - np.array([0.5, 0.0])
        xy0 = fun.rotate_multiple_points(xy0, random.uniform(*np.deg2rad(orientation_range)), origin=[0, 0])
        xy0 = xy0 * self.s / 250 + np.array(P0)
        temp = self.graph.draw_polygon(xy0, line_width=3, line_color=color, fill_color=color)
        return temp

    def check_abort(self, name, w, v, units, groups):
        n=name
        n0=n.lower()
        g,g0,D,DN,Dm,Ds, s,s0=self.group_ks(n)
        o, o0, oM, oS=self.odor_ks(n)
        f, fM=self.food_ks
        info = w['info']
        abort = True
        O = self.odor_on(w,name)
        F = self.food_on(w)

        if not O:
            w[o0].update(value=None)
            w[oM].update(value=0.0)

        if n == 'SOURCE':
            if not O and not F:
                info.update(value=f"Assign food and/or odor to the drawn source")
                return True
            elif F and float(v[fM]) == 0.0:
                w[fM].update(value=10 ** -3)
                info.update(value=f"Source food amount set to default")
                return True
            elif not F and float(v[fM]) != 0.0:
                w[fM].update(value=0.0)
                info.update(value=f"Source food amount set to 0")

        if v[g0] == '' and v[s0] == '':
            info.update(value=f"Both {n0} single id and group id are empty")
        elif not v[g] and not v[s]:
            info.update(value=f"Select to add a single or a group of {n0}s")
        elif v[s] and (v[s0] in list(units.keys()) or v[s0] == ''):
            info.update(value=f"{n0} id {v[s0]} already exists or is empty")
        elif O and v[o0] == '':
            info.update(value=f"Default odor id automatically assigned to the odor")
            id = v[g0] if v[g0] != '' else v[s0]
            w[o0].update(value=f'{id}_odor')
        elif O and not float(v[oM]) > 0:
            info.update(value=f"Assign positive odor intensity to the drawn odor source")
        elif O and (v[oS] == '' or not float(v[oS]) > 0):
            info.update(value=f"Assign positive spread to the odor")
        elif v[g] and (v[g0] in list(groups.keys()) or v[g0] == ''):
            info.update(value=f"{n0} group id {v[g0]} already exists or is empty")
        elif v[g] and v[Dm] in ['', None]:
            info.update(value=f"Define a distribution mode")
        elif v[g] and v[Ds] in ['', None]:
            info.update(value=f"Define a distribution shape")
        elif v[g] and not int(v[DN]) > 0:
            info.update(value=f"Assign a positive integer number of items for the distribution")
        else:
            abort = False
        return abort

    def set_env_db(self, env=None):
        if env is None:
            env = {'border_list': {},
                   'arena': dtypes.get_dict('arena'),
                   'food_params': {'source_units': {}, 'source_groups': {}, 'food_grid': None},
                   'larva_groups': {}
                   }
        items = [env['border_list'],
                 env['food_params']['source_units'], env['food_params']['source_groups'],
                 {}, env['larva_groups']]
        env_db = {k: {'items': ii, 'figs': {}} for k, ii in zip(['b', 's_u', 's_g', 'l_u', 'l_g'], items)}
        return env_db

    def aux_reset(self):
        dic=self.aux_dict
        dic['dragging'], dic['current'] = False, {}
        dic['start_point'], dic['end_point'], dic['prior_rect'] = None, None, None


if __name__ == "__main__":
    from lib.gui.gui import LarvaworldGui
    larvaworld_gui = LarvaworldGui(tabs=['environment'])
    larvaworld_gui.run()
