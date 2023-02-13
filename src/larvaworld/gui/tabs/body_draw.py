import numpy as np
import PySimpleGUI as sg

from larvaworld.gui import gui_aux

class DrawBodyTab(gui_aux.DrawTab):
    def __init__(self, canvas_size=(1200, 800), **kwargs):
        super().__init__(canvas_size=canvas_size, **kwargs)
        self.p_radius = self.canvas_size[0] / 240
        self.c_key='Body'
        self.P, self.S, self.O, self.T='points','segs','olfaction_sensors','touch_sensors'
        self.Cdict = {
            self.P: 'black',
            self.S: 'black',
            self.T: 'green',
            self.O: 'magenta',
        }

    def build(self):
        c = {}

        dic = {
            's': self.canvas_size[0] * 0.7,
            'dragging': None,
            'drag_figures': {},
            'current': {},
            self.S: [],
            'last_xy': (0, 0),
            'start_point': None,
            'end_point': None,
            'prior_rect': None,
            'P1': None,
            'P2': None,
        }

        sl = gui_aux.SelectionList(tab=self, disp='Body', buttons=['load', 'save', 'delete', 'exec'],
                                   width=30, text_kws=gui_aux.t_kws(12))

        c1 = gui_aux.PadDict(self.c_key, disp_name='Configuration', text_kws=gui_aux.t_kws(8), header_width=25,
                             background_color='orange',
                             subconfs={self.P: {'Nspins': 12, 'indexing': True, 'group_by_N': 2, 'text_kws': {'text_color': self.Cdict[self.P]}},
                               self.T: {'Nspins': 8, 'group_by_N': 4, 'text_kws': {'text_color': self.Cdict[self.T]}},
                               self.O: {'Nspins': 4, 'text_kws': {'text_color': self.Cdict[self.O]}}
                               },
                             after_header=[gui_aux.GraphButton('Button_Burn', 'RESET_BODY',
                                                               tooltip='Reset to the initial body.'),
                                           gui_aux.GraphButton('Globe_Active', 'NEW_BODY',
                                                               tooltip='Create a new body.All drawn items will be erased.')])
        c.update(c1.get_subdicts())
        col1 = gui_aux.gui_col([sl, c1], x_frac=0.3, as_pane=True, pad=(10, 10))
        g1 = gui_aux.GraphList(self.name, tab=self, graph=True, canvas_size=self.canvas_size, canvas_kws={
            'graph_bottom_left': (0, 0),
            'graph_top_right': self.canvas_size,
            'change_submits': True,
            'drag_submits': True,
            'background_color': 'white',
        })
        col2 = sg.Col([g1.canvas.get_layout(as_pane=True, pad=(0, 10))[0]], **gui_aux.col_kws)
        l = [[col1, col2]]
        self.graph = g1.canvas_element

        return l, c, {g1.name: g1}, {self.name: dic}

    def update(self, w, c, conf, id):
        from larvaworld.lib.aux.sim_aux import rearrange_contour

        if conf[self.P] is not None:
            conf[self.P] = rearrange_contour(conf[self.P])
            self.c.update(w, conf)

            self.draw_body(conf)

    def get(self, w, v, c, as_entry=True):
        conf = self.c.get_dict(v, w)
        return conf

    def eval(self, e, v, w, c, d, g):
        gg = self.graph
        gg.bind('<Button-3>', '+RIGHT+')
        # gg.Widget.config(cursor='fleur')
        dic = self.base_dict
        if e == 'RESET_BODY':
            conf = self.get(w, v, c)
            self.update(w, c, conf, id=None)
        elif e == 'NEW_BODY':
            gg.erase()
        if e == self.canvas_k:
            x, y = v[self.canvas_k]
            if not dic['dragging']:
                self.set_drag_ps(p1=(x, y))
                dic['dragging'] = True
                for idx, entry in dic[self.P].items():
                    if entry['fig'] in gg.get_figures_at_location((x, y)):
                        dic['drag_figures'][idx] = entry
                    else:
                        try:
                            del dic['drag_figures'][idx]
                        except:
                            pass
                dic['last_xy'] = x, y
            else:
                self.set_drag_ps(p2=(x, y))
            delta_x, delta_y = x - dic['last_xy'][0], y - dic['last_xy'][1]
            dic['last_xy'] = x, y
            if None not in self.get_drag_ps():
                delta_X, delta_Y = delta_x / self.s, delta_y / self.s
                for idx, entry in dic['drag_figures'].items():
                    X0, Y0 = entry['pos']
                    new_p = (X0 + delta_X, Y0 + delta_Y)
                    conf = self.get(w, v, c)
                    conf[self.P][idx] = new_p
                    self.update(w, c, conf, id=None)
        elif e.endswith('+UP'):  # The drawing has ended because mouse up
            self.aux_reset()

        return d, g

    def draw_body(self, conf, **kwargs):
        self.graph.erase()
        self.xy_reset()
        self.draw_segs(conf)
        self.draw_points(conf)

    def draw_points(self, conf, **kwargs):
        Cps = self.Cdict[self.P]
        dic = self.base_dict
        r = self.p_radius
        gg = self.graph
        ts = conf[self.T]
        os = conf[self.O]

        for i, p0 in enumerate(conf[self.P]):
            p = self.scale_xy(p0 - np.array([0.5, 0.0]), reverse=True)
            f = gg.draw_circle(p, r, line_width=1, line_color=Cps, fill_color=Cps)
            dic[self.P][i] = {'fig': f, 'pos': p}
            if ts is not None and i in ts:
                f = gg.draw_circle(p, r * 1.8, line_width=6, line_color= self.Cdict[self.T])
                dic[self.T][i] = {'fig': f, 'pos': p}
            if os is not None and i in os:
                f = gg.draw_circle(p, r * 2.6, line_width=5, line_color=self.Cdict[self.O])
                dic[self.O][i] = {'fig': f, 'pos': p}

    def draw_segs(self, conf, **kwargs):
        from larvaworld.lib.aux.sim_aux import generate_seg_shapes

        dic = self.base_dict
        gg = self.graph
        for i, ps0 in enumerate(generate_seg_shapes(centered=False, **conf)):
            ps = [self.scale_xy(p0 - np.array([0.5, 0.0]), reverse=True) for p0 in ps0[0]]
            ps.append(ps[0])
            f = gg.draw_lines(ps, width=2, color=self.Cdict[self.S])
            dic[self.S][i] = {'fig': f, 'pos': ps}

    def xy_reset(self):
        dic = self.base_dict
        dic[self.P] = {}
        dic[self.T] = {}
        dic[self.O] = {}
        dic[self.S] = {}

    @property
    def c(self):
        return self.gui.collapsibles[self.c_key]
