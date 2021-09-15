import PySimpleGUI as sg

import lib.conf.conf
from lib.anal.plotting import plot_debs
from lib.conf.init_dtypes import substrate_dict
from lib.gui.aux.elements import CollapsibleDict, Table, GraphList, SelectionList, Header
from lib.gui.aux.functions import col_size, t_kws, gui_col, gui_row
from lib.gui.aux.buttons import GraphButton
from lib.gui.tabs.tab import GuiTab
from lib.model.DEB.deb import deb_default


class LifeTab(GuiTab):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.Sq, self.Sa = 'SLIDER_quality', 'SLIDER_age'
        self.s0, self.s1 = 'start', 'stop'
        self.S0, self.S1 = [f'SLIDER_epoch_{s}' for s in [self.s0, self.s1]]
        self.ep = 'rearing epoch'
        self.K = 'EPOCHS'

    # def build(self):
    #     return [], {}, {}, {}
    def build(self):
        sl1_kws = {
            'size': (40, 20),
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

        y = 0.5
        x1 = 0.2
        x2 = 0.8
        r1_size = col_size(x_frac=1 - x1, y_frac=y)

        sl0 = SelectionList(tab=self, actions=['load', 'save', 'delete'])

        sub = CollapsibleDict('substrate', default=True, header_dict=substrate_dict, header_value='standard')
        l1 = [[sg.T('Epoch start (hours) : ', **t_kws(24))],
              [sg.Slider(range=(0, 150), default_value=0, k=self.S0,
                         tick_interval=24, resolution=1, trough_color='green', **sl1_kws)],
              [sg.T('Epoch stop (hours) : ', **t_kws(24))],
              [sg.Slider(range=(0, 150), default_value=0, k=self.S1,
                         tick_interval=24, resolution=1, trough_color='red', **sl1_kws)]]
        l2 = [[sg.T('Food quality : ', **t_kws(24))],
              [sg.Slider(range=(0.0, 1.0), default_value=1.0, k=self.Sq,
                         tick_interval=0.25, resolution=0.01, **sl1_kws)],
              [sg.T('Starting age (hours post-hatch): ', **t_kws(24))],
              [sg.Slider(range=(0, 150), default_value=0, k=self.Sa,
                         tick_interval=24, resolution=1, **sl1_kws)]]

        after_header = [GraphButton('Button_Add', f'ADD {ep}', tooltip=f'Add a new {ep}.'),
                        GraphButton('Button_Remove', f'REMOVE {ep}', tooltip=f'Remove an existing {ep}.')]
        content = [Table(headings=[self.s0, self.s1, 'quality'], def_col_width=7, key=self.K, num_rows=0)]
        l_tab = Header('Epochs', text=f'{ep.capitalize()}s (h)', header_text_kws=t_kws(18),
                       after_header=after_header, single_line=False, content=content).layout

        g1 = GraphList(self.name, fig_dict={m: plot_debs for m in deb_modes}, default_values=['reserve'],
                       canvas_size=r1_size, list_header='DEB parameters', auto_eval=False)

        l0 = gui_col([sl0, sub, g1], x1, y)
        l3 = gui_col([g1.canvas], 1 - x1, y)

        l = [
            [l0, l3],
            gui_row([l_tab, l1, l2], y_frac=1 - y, x_fracs=[1 - x2, x2 / 2, x2 / 2]),
        ]

        c = {sub.name: sub}
        g = {g1.name: g1}
        return l, c, g, {}

    def update(self, w, c, conf, id=None):
        c['substrate'].update_header(w, conf['substrate_type'])

        w.Element(self.Sq).Update(value=conf['substrate_quality'])
        w.Element(self.Sa).Update(value=conf['hours_as_larva'])
        if conf['epochs'] is not None:
            epochs = [[t0, t1, q] for (t0, t1), q in zip(conf['epochs'], conf['epoch_qs'])]
            w.Element(self.K).Update(values=epochs, num_rows=len(epochs))
        else:
            w.Element(self.K).Update(values=[], num_rows=0)

        w.write_event_value('Draw', 'Draw the initial plot')

    def get(self, w, v, c, as_entry=False):
        epochs = w.Element(self.K).get()
        life = {
            'epochs': [(t1, t2) for t1, t2, q in epochs] if len(epochs) > 0 else None,
            'epoch_qs': [q for t1, t2, q in epochs] if len(epochs) > 0 else None,
            'hours_as_larva': v[self.Sa],
            'substrate_quality': v[self.Sq],
            'substrate_type': c['substrate'].header_value,
        }
        return life

    def eval(self, e, v, w, c, d, g):
        S0, S1, Sa, Sq, K, ep = self.S0, self.S1, self.Sa, self.Sq, self.K, self.ep
        v0, v1, q = row = v[S0], v[S1], v[Sq]
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
        # elif e in [Sq]:
        #     w.write_event_value('Draw', 'Draw the initial plot')

        elif e == 'Draw':
            if q > 0:
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
            for t1, t2, q in w.Element(K).get():
                if t1 < v0 < t2:
                    w.Element(S0).Update(value=t2)
                elif v0 < t1 and v1 > t1:
                    w.Element(S1).Update(value=t1)
                if t1 < v1 < t2:
                    w.Element(S1).Update(value=t1)
                elif v1 > t2 and v0 < t2:
                    w.Element(S0).Update(value=t2)


if __name__ == "__main__":
    from lib.gui.tabs.gui import LarvaworldGui

    larvaworld_gui = LarvaworldGui(tabs=['life-history'])
    larvaworld_gui.run()
