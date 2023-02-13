import PySimpleGUI as sg

from larvaworld.lib import reg
from larvaworld.gui import gui_aux

class LifeTab(gui_aux.GuiTab):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.Sq, self.Sa = 'SLIDER_quality', 'SLIDER_age'
        self.s0, self.s1 = 'start', 'stop'
        self.S0, self.S1 = [f'SLIDER_epoch_{s}' for s in [self.s0, self.s1]]
        self.ep = 'rearing epoch'
        self.K = 'EPOCHS'

    def build(self):
        from larvaworld.lib.plot.deb import plot_debs
        from larvaworld.lib.model.deb.substrate import substrate_dict
        sl1_kws = {
            'size': (30, 20),
            'enable_events': True,
            'orientation': 'h'
        }
        deb_modes = ['mass', 'length',
                     'reserve',

                     'reserve_density',
                     'hunger',
                     'f',
                     'pupation_buffer',
                     # 'explore2exploit_balance',
                     #     'f_filt'
                     ]
        sg.theme('LightGreen')
        ep = self.ep
        y = 0.55
        x1 = 0.2
        x2 = 0.8
        r1_size = gui_aux.col_size(x_frac=1 - x1 - 0.05, y_frac=y - 0.05)
        sl0 = gui_aux.SelectionList(tab=self, buttons=['load', 'save', 'delete'])
        sub = gui_aux.CollapsibleDict('substrate', dict_name='substrate_composition', header_dict=substrate_dict,
                                      header_value='standard', state=True, value_kws=gui_aux.t_kws(8))
        l1 = [[sg.T('Epoch start (hours) : ', **gui_aux.t_kws(24))],
              [sg.Slider(range=(0, 150), default_value=0, k=self.S0,
                         tick_interval=24, resolution=1, trough_color='green', **sl1_kws)],
              [sg.T('Epoch stop (hours) : ', **gui_aux.t_kws(24))],
              [sg.Slider(range=(0, 150), default_value=0, k=self.S1,
                         tick_interval=24, resolution=1, trough_color='red', **sl1_kws)]]
        l2 = [[sg.T('Food quality : ', **gui_aux.t_kws(24))],
              [sg.Slider(range=(0.0, 1.0), default_value=1.0, k=self.Sq,
                         tick_interval=0.25, resolution=0.01, **sl1_kws)],
              [sg.T('Starting age (hours post-hatch): ', **gui_aux.t_kws(24))],
              [sg.Slider(range=(0, 150), default_value=0, k=self.Sa,
                         tick_interval=24, resolution=1, **sl1_kws)]]
        after_header = [gui_aux.GraphButton('Button_Add', f'ADD {ep}', tooltip=f'Add a new {ep}.'),
                        gui_aux.GraphButton('Button_Remove', f'REMOVE {ep}', tooltip=f'Remove an existing {ep}.')]
        content = [gui_aux.Table(headings=[self.s0, self.s1, 'quality', 'type'], col_widths=[5, 5, 6, 7], key=self.K, num_rows=8)]
        l_tab = gui_aux.Header('Epochs', text=f'{ep.capitalize()}s (h)', text_kws=gui_aux.t_kws(18),
                               after_header=after_header, single_line=False, content=content)
        g1 = gui_aux.GraphList(self.name, tab=self, fig_dict={m: plot_debs for m in deb_modes}, default_values=['reserve'],
                               canvas_size=r1_size, list_header='DEB parameters', auto_eval=False)
        pane_kws={'as_pane': True, 'pad': (20,20)}
        l0 = gui_aux.gui_col([sl0, g1], x1, y, **pane_kws)
        l3 = gui_aux.gui_col([g1.canvas], 1 - x1, y, **pane_kws)
        l = [
            [l0, l3],
            gui_aux.gui_row([l_tab, l1, l2, sub], y_frac=1 - y, x_fracs=[1 - x2, x2 / 3, x2 / 3, x2 / 3], **pane_kws),
        ]
        return l, {sub.name: sub}, {g1.name: g1}, {}

    def update(self, w, c, conf, id=None):
        w.Element(self.Sa).Update(value=conf['age'])
        eps=conf['epochs']
        rows = [[ep['start'], ep['stop'], ep['substrate']['quality'], ep['substrate']['type']] for ep in eps.values()]
        w.Element(self.K).Update(values=rows, num_rows=len(rows))
        w.write_event_value('Draw', 'Draw the initial plot')

    def get(self, w, v, c, as_entry=False):
        rows = w.Element(self.K).get()
        return {
            'epochs': {i : {'start': r[0], 'stop': r[1], 'substrate': reg.get_null('substrate', type=r[3], quality=r[2])} for
                        i, r in enumerate(rows)},
            'age': v[self.Sa],
        }

    def eval(self, e, v, w, c, d, g):
        S0, S1, Sa, Sq, K, ep = self.S0, self.S1, self.Sa, self.Sq, self.K, self.ep
        v0, v1, q, sub = row = v[S0], v[S1], v[Sq], c['substrate'].header_value
        Ks = v[K]
        if e == self.graphlist_k:
            w.write_event_value('Draw', 'Draw the initial plot')
        elif e == f'ADD {ep}':
            if v1 > v0:
                w.Element(K).add_row(w, row, sort_idx=0)
                w.Element(S1).Update(value=0.0)
                w.Element(S0).Update(value=0.0)
                w.Element(Sq).Update(value=1.0)
                w.write_event_value('Draw', 'Draw the initial plot')
        elif e == f'REMOVE {ep}':
            if len(Ks) > 0:
                w.Element(K).remove_row(w, Ks[0])
                w.write_event_value('Draw', 'Draw the initial plot')
        elif e == 'Draw':
            if q > 0:
                from larvaworld.lib.model.deb.deb import deb_default
                from larvaworld.lib.plot.deb import plot_debs

                D = deb_default(**self.get(w, v, c))
                for Sii in [S0, S1, Sa]:
                    w.Element(Sii).Update(range=(0.0, D['pupation'] - D['birth']))
                fig, save_to, filename = plot_debs(deb_dicts=[D], mode=v[self.graphlist_k][0], return_fig=True)
                self.graph_list.draw_fig(w, fig)
        elif e in [S0, S1]:
            if e == S0 and v0 > v1:
                w.Element(S1).Update(value=v0)
            elif e == S1 and v1 < v0:
                w.Element(S0).Update(value=v1)
            for t1, t2, q, sub in w.Element(K).get():
                if t1 < v0 < t2:
                    w.Element(S0).Update(value=t2)
                elif v0 < t1 and v1 > t1:
                    w.Element(S1).Update(value=t1)
                if t1 < v1 < t2:
                    w.Element(S1).Update(value=t1)
                elif v1 > t2 and v0 < t2:
                    w.Element(S0).Update(value=t2)


if __name__ == "__main__":
    from larvaworld.gui.tabs.larvaworld_gui import LarvaworldGui
    larvaworld_gui = LarvaworldGui(tabs=['life'])
    larvaworld_gui.run()
