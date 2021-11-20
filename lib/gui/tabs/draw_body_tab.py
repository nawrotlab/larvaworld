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
from lib.gui.aux.buttons import color_pick_layout
from lib.gui.tabs.tab import GuiTab


class DrawBodyTab(GuiTab):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.canvas_size = (1200, 500)
        self.p_radius = self.canvas_size[0] / 240

    def build(self):
        c = {}

        dic = {
            # 'env_db': self.set_env_db(store=False),
            's': self.canvas_size[0] * 0.7,
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

        sl = SelectionList(tab=self, disp='Body', buttons=['load', 'save', 'delete', 'run'],
                           width=30, text_kws=t_kws(12))

        c1 = PadDict('body_shape', disp_name='Configuration', text_kws=t_kws(10), header_width=30)
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
        self.draw_body(conf)

    def eval(self, e, v, w, c, d, g):

        return d, g

    def draw_body(self, conf, color='black', **kwargs):
        self.graph.erase()
        if conf['symmetry'] == 'bilateral':
            segs = lib.aux.sim_aux.generate_seg_shapes(centered=False,closed=True, **conf)
        else:
            raise NotImplementedError
        for seg in segs:
            seg = [self.scale_xy(vs - np.array([0.5, 0.0]), reverse=True) for vs in seg[0]]
            temp = self.graph.draw_lines(seg, width=2, color=color)
            for p in seg:
                self.graph.draw_circle(tuple(p), self.p_radius, line_width=1, line_color=color, fill_color=color)

    def scale_xy(self, xy, reverse=False):
        if xy is None:
            return None
        W, H = self.graph_list.canvas_size
        s = self.s
        x, y = xy
        if reverse:
            return x * s + W / 2, y * s + H / 2
        else:
            return (x - W / 2) / s, (y - H / 2) / s

    @property
    def s(self):
        return self.base_dict['s']
