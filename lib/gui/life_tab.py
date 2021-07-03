import PySimpleGUI as sg
import lib.conf.dtype_dicts as dtypes
from lib.anal.plotting import plot_debs

from lib.gui.gui_lib import CollapsibleDict, col_kws, col_size, t12_kws, graphic_button, \
    t18_kws, b_kws, t24_kws, Table, GraphList
from lib.gui.tab import GuiTab, SelectionList
from lib.model.DEB.deb import deb_default


class LifeTab(GuiTab):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.Sq, self.Sa = 'SLIDER_quality', 'SLIDER_age'
        self.s0, self.s1 = 'start', 'stop'
        self.S0, self.S1 = [f'SLIDER_epoch_{s}' for s in [self.s0, self.s1]]
        self.ep = 'rearing epoch'
        self.K = 'EPOCHS'



    def update(self,w, c, conf, id=None):
        c['substrate'].update_header(w, conf['substrate_type'])

        w.Element(self.Sq).Update(value=conf['substrate_quality'])
        w.Element(self.Sa).Update(value=conf['hours_as_larva'])
        if conf['epochs'] is not None :
            epochs=[[t0,t1,q] for (t0,t1),q in zip(conf['epochs'], conf['epoch_qs'])]
            w.Element(self.K).Update(values=epochs, num_rows=len(epochs))
        else :
            w.Element(self.K).Update(values=[], num_rows=0)

        w.write_event_value('Draw', 'Draw the initial plot')



    def get(self,w, v, c, as_entry=False):
        epochs=w.Element(self.K).get()
        life = {
            'epochs': [(t1, t2) for t1, t2, q in epochs] if len(epochs) > 0 else None,
            'epoch_qs': [q for t1, t2, q in epochs] if len(epochs) > 0 else None,
            'hours_as_larva': v[self.Sa],
            'substrate_quality': v[self.Sq],
            'substrate_type': c['substrate'].header_value,
        }
        return life



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
        x2 = 0.2
        l1_size = col_size(x_frac=x1, y_frac=y)
        l2_size = col_size(x_frac=x2, y_frac=1 - y)
        r1_size = col_size(x_frac=1 - x1, y_frac=y)
        r2_size = col_size(x_frac=(1 - x2)/2, y_frac=1 - y)

        sl0 = SelectionList(tab=self, conftype='Life', actions=['load', 'save', 'delete'])
        self.selectionlists = [sl0]

        sub = CollapsibleDict('substrate', False, default=True, header_dict=dtypes.substrate_dict,
                              header_value='standard')

        l1 = sg.Col([[sg.Text('Epoch start (hours) : ', **t24_kws)],
                     [sg.Slider(range=(0, 150), default_value=0, key=self.S0,
                                tick_interval=24, resolution=1,trough_color='green', **sl1_kws)],
                     [sg.Text('Epoch stop (hours) : ', **t24_kws)],
                     [sg.Slider(range=(0, 150), default_value=0, key=self.S1,
                                tick_interval=24, resolution=1,trough_color='red', **sl1_kws)],
                     ], size=r2_size, **col_kws)

        l2 = sg.Col([[sg.Text('Food quality : ', **t24_kws)],
              [sg.Slider(range=(0.0, 1.0), default_value=1.0, key=self.Sq,
                         tick_interval=0.25, resolution=0.01, **sl1_kws)],
              [sg.Text('Starting age (hours post-hatch): ', **t24_kws)],
              [sg.Slider(range=(0, 150), default_value=0, key=self.Sa,
                         tick_interval=24, resolution=1, **sl1_kws)],
              ], size=r2_size, **col_kws)

        l_tab = sg.Col([[
            sg.Text(f'{ep.capitalize()}s (h)', **t18_kws),
            graphic_button('add', f'ADD {ep}', tooltip=f'Add a new {ep}.'),
            graphic_button('remove', f'REMOVE {ep}', tooltip=f'Remove an existing {ep}.'),
            # graphic_button('data_add', f'ADD life', tooltip=f'Add a life history configuration.')
        ],
            [Table(values=[], headings=[self.s0, self.s1, 'quality'], def_col_width=7,
                      max_col_width=24, background_color='lightblue',
                      header_background_color='lightorange',
                      auto_size_columns=False,
                      # display_row_numbers=True,
                      justification='center',
                      # font=w_kws['font'],
                      num_rows=0,
                      alternating_row_color='lightyellow',
                      key=self.K
                      )],
            # [sg.B('Remove', **b_kws, **pad2), sg.B('Add', **b_kws, **pad2)]
        ], size=l2_size, **col_kws)

        g1=GraphList(self.name, fig_dict={m : plot_debs for m in deb_modes}, default_values=['reserve'],
                 canvas_size=r1_size, list_header='DEB parameters', auto_eval=False)

        l0 = sg.Col([
            sl0.l,
            sub.get_layout(),
            [g1.get_layout()]
        ], size=l1_size)

        l = [
            [l0, g1.canvas],
            [l_tab, l1,l2],
        ]

        c = {sub.name: sub}
        g = {g1.name: g1}
        return l, c, g, {}

    def eval(self, e, v, w, c, d, g):
        S0,S1,Sa,Sq,K,ep=self.S0,self.S1,self.Sa,self.Sq,self.K,self.ep
        if e == g[self.name].list_key:
            w.write_event_value('Draw', 'Draw the initial plot')
        elif e == f'ADD {ep}':
            t1, t2, q =row= v[S0], v[S1], v[Sq]
            if t2 > t1:
                w.Element(K).add_row(w,row, sort_idx=0)
                w.Element(S1).Update(value=0.0)
                w.Element(S0).Update(value=0.0)
                w.Element(Sq).Update(value=1.0)
                w.write_event_value('Draw', 'Draw the initial plot')
        elif e == f'REMOVE {ep}':
            if len(v[K]) > 0:
                w.Element(K).remove_row(w, v[K][0])
                w.write_event_value('Draw', 'Draw the initial plot')
        # elif e in [Sq]:
        #     w.write_event_value('Draw', 'Draw the initial plot')

        elif e == 'Draw':
            if v[Sq]>0 :
                D = deb_default(**self.get(w, v, c))
                for Sii in [S0, S1, Sa]:
                    w.Element(Sii).Update(range=(0.0, D['pupation'] - D['birth']))
                fig, save_to, filename = plot_debs(deb_dicts=[D], mode=v[g[self.name].list_key][0], return_fig=True)
                g[self.name].draw_fig(w,fig)

        elif e in [S0, S1]:
            if e == S0 and v[S0] > v[S1]:
                w.Element(S1).Update(value=v[S0])
            elif e == S1 and v[S1] < v[S0]:
                w.Element(S0).Update(value=v[S1])
            for t1, t2, q in w.Element(K).get():
                if t1 < v[S0] < t2:
                    w.Element(S0).Update(value=t2)
                elif v[S0] < t1 and v[S1] > t1:
                    w.Element(S1).Update(value=t1)
                if t1 < v[S1] < t2:
                    w.Element(S1).Update(value=t1)
                elif v[S1] > t2 and v[S0] < t2:
                    w.Element(S0).Update(value=t2)




if __name__ == "__main__":
    from lib.gui.gui import LarvaworldGui
    larvaworld_gui = LarvaworldGui(tabs=['life-history'])
    larvaworld_gui.run()

