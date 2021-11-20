import copy
import random

import numpy as np
import PySimpleGUI as sg

import lib.aux.ang_aux
import lib.aux.sim_aux
import lib.aux.xy_aux
import lib.conf.base.dtypes
import lib.gui.aux.functions
from lib.conf.base.dtypes import null_dict
from lib.gui.aux.elements import CollapsibleDict, GraphList, PadDict, SelectionList
from lib.gui.aux.functions import t_kws, gui_col, gui_cols, col_size, default_list_width, col_kws
from lib.gui.aux.buttons import color_pick_layout, GraphButton
from lib.gui.tabs.tab import GuiTab, DrawTab


class DrawBodyTab(DrawTab):
    def __init__(self,canvas_size = (1200, 500), **kwargs):
        super().__init__(canvas_size = canvas_size, **kwargs)
        self.p_radius = self.canvas_size[0] / 240

    def build(self):
        c = {}

        dic = {
            # 'env_db': self.set_env_db(store=False),
            's': self.canvas_size[0] * 0.7,
            'sample_fig': None,
            'sample_pars': {},
            'contour': {},
            'dragging': None,
            'drag_figures': {},
            'current': {},
            'last_xy': (0, 0),
            'start_point': None,
            'end_point': None,
            'prior_rect': None,
            'P1': None,
            'P2': None,
        }

        sl = SelectionList(tab=self, disp='Body', buttons=['load', 'save', 'delete', 'run'],
                           width=30, text_kws=t_kws(12))

        c1 = PadDict('body_shape', disp_name='Configuration', text_kws=t_kws(10), header_width=30,
                     subconfs={'points' : {'Nspins' : 8}},
                     after_header=[GraphButton('Button_Burn', 'RESET_BODY',
                                               tooltip='Reset to the initial body.'),
                                   GraphButton('Globe_Active', 'NEW_BODY',
                                               tooltip='Create a new body.All drawn items will be erased.')])
        c.update(c1.get_subdicts())

        col1 = gui_col([sl, c1], x_frac=0.3, as_pane=True, pad=(10, 10))

        g1 = GraphList(self.name, tab=self, graph=True, canvas_size=self.canvas_size, canvas_kws={
            'graph_bottom_left': (0, 0),
            'graph_top_right': self.canvas_size,
            'change_submits': True,
            'drag_submits': True,
            'background_color': 'white',
        })
        col2 = sg.Col([g1.canvas.get_layout(as_pane=True, pad=(0, 10))[0]], **col_kws)

        l = [[col1, col2]]
        self.graph = g1.canvas_element

        return l, c, {g1.name: g1}, {self.name: dic}

    def update(self, w, c, conf, id):
        c['body_shape'].update(w, conf)
        # self.draw_tab.set_env_db(env=expandConf(conf['env_params'], 'Env'), lg=conf['larva_groups'])
        # w.write_event_value('RESET_ARENA', 'Draw the initial arena')
        self.set_contour(conf)
        self.draw_body()

    def get(self, w, v, c):
        conf = c['body_shape'].get_dict(v, w)
        return conf

    def eval(self, e, v, w, c, d, g):
        r=self.p_radius
        gg = self.graph
        gg.bind('<Button-3>', '+RIGHT+')
        # gg.Widget.config(cursor='fleur')
        dic = self.base_dict
        if e == 'RESET_BODY':
            conf = self.get(w, v, c)
            self.set_contour(conf)
            self.draw_body()
        elif e == 'NEW_BODY':
            gg.erase()
        if e == self.canvas_k:
            x, y = v[self.canvas_k]
            if not dic['dragging']:
                self.set_drag_ps(p1=(x, y))
                dic['dragging'] = True
                for idx, entry in dic['contour'].items():
                    if entry['fig'] in gg.get_figures_at_location((x, y)) :
                        dic['drag_figures'][idx] = entry
                    else :
                        try :
                            del dic['drag_figures'][idx]
                        except :
                            pass
                dic['last_xy'] = x, y
            else:
                self.set_drag_ps(p2=(x, y))
            delta_x, delta_y = x - dic['last_xy'][0], y - dic['last_xy'][1]
            dic['last_xy'] = x, y
            if None not in self.get_drag_ps():
                # if v['-MOVE-']:
                delta_X, delta_Y = delta_x / self.s, delta_y / self.s
                for idx,entry in dic['drag_figures'].items() :
                    X0, Y0 =p= entry['pos']
                    new_p=(X0 + delta_X, Y0 + delta_Y)

                    # conf=c['body_shape'].get_dict(v, w)
                    # try :
                    #     conf['points'].remove(p)
                    # except:
                    #     pass
                    # conf['points'].append(new_p)
                    # c['body_shape'].update(w, conf)
                    # self.update(w, c, conf, id=None)
                    dic['contour'][idx]['pos'] = new_p
                    gg.move_figure(entry['fig'], delta_x, delta_y)
                    gg.update()
                    self.draw_body()
        elif e.endswith('+UP'):  # The drawing has ended because mouse up
            self.aux_reset()

        return d, g

    def draw_body(self, color='black', **kwargs):
        self.graph.erase()
        cc=self.base_dict['contour']
        segs={}
        for (i,j),entry in cc.items() :
            p0 = self.scale_xy(entry['pos'] - np.array([0.5, 0.0]), reverse=True)
            entry['fig']=self.graph.draw_circle(p0, self.p_radius, line_width=1, line_color=color, fill_color=color)
            if i not in segs.keys() :
                segs[i]=[]
            segs[i].append(p0)
        for vs in segs.values() :
            vs.append(vs[0])
            self.graph.draw_lines(vs, width=2, color=color)

    def set_contour(self, conf, **kwargs):
        self.base_dict['contour']={}
        if conf['symmetry'] == 'bilateral':
            segs = lib.aux.sim_aux.generate_seg_shapes(centered=False, closed=False, **conf)
        else:
            raise NotImplementedError
        for i, seg in enumerate(segs):
            for j, p in enumerate(seg[0]):
                self.base_dict['contour'][(i, j)] = {'fig' : None, 'pos' : p}
