import PySimpleGUI as sg
from PySimpleGUI import RELIEF_RAISED, RELIEF_SUNKEN, RELIEF_SOLID

import lib.conf.dtype_dicts as dtypes
from lib.anal.plotting import plot_debs
from lib.gui.gui_lib import default_run_window, named_list_layout, graphic_button, save_conf_window
from lib.gui.gui_lib import check_toggles, b_kws, w_kws, delete_figure_agg, draw_canvas, t24_kws, t10_kws, col_kws, \
    t12_kws, t18_kws, col_size, t14_kws, Collapsible, CollapsibleDict
from lib.gui.tab import SelectionList
from lib.model.DEB.deb import deb_default

W, H = 1600, 800
k1 = 20
pad = {'pad': ((k1, k1), (k1, k1))}
k2 = 20
pad2 = {'pad': ((k2, k2), (k2, k2))}
k = 2
sl1_kws = {
    'size': (30, 10),
    'enable_events': True,
    'orientation': 'h',
    'pad': ((k, k), (k, k)),
    # 'relief': RELIEF_SOLID,

}

sl2_kws = {
    'size': (30, 10),
    'enable_events': True,
    'orientation': 'h',
    'pad': ((k, k), (10 * k, 10 * k)),
    # 'background_color' : 'lightblue'
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


def life_conf():
    sg.theme('LightGreen')
    life = dtypes.get_dict('life')
    epochs = []
    epoch_qs = []
    Sq, Sa = 'SLIDER_quality', 'SLIDER_age'
    s0, s1 = 'start', 'stop'
    st0, st1 = [f'starvation_{s}' for s in [s0, s1]]
    S0, S1 = [f'Slider_{s}' for s in [st0, st1]]
    K = 'STARVATION'
    ep = 'rearing epoch'

    y = 0.5
    x1 = 0.2
    x2 = 0.7
    l1_size = col_size(x_frac=x1, y_frac=y, win_size=(W, H))
    l2_size = col_size(x_frac=x2, y_frac=1 - y, win_size=(W, H))
    r1_size = col_size(x_frac=1 - x1, y_frac=y, win_size=(W, H))
    r2_size = col_size(x_frac=1 - x2, y_frac=1 - y, win_size=(W, H))

    sub = CollapsibleDict('substrate', False, default=True, header_dict=dtypes.substrate_dict,
                          header_value='standard')
    collapsibles = {sub.name: sub}

    l_DEB = named_list_layout(text='DEB Parameter : ', key='deb_mode', choices=deb_modes, default_value='reserve',
                              drop_down=False, single_line=False)

    l0 = sg.Col([
        sub.get_layout(),
        [l_DEB]
    ], size=l1_size)

    l2 = [[sg.Text('Food quality : ', **t24_kws)],
          [sg.Slider(range=(0.0, 1.0), default_value=1.0, key=Sq,
                     tick_interval=0.25, resolution=0.01, **sl1_kws)],
          [sg.Text('', **t24_kws)],
          [sg.Text('', size=(10, 1)), sg.Ok(**b_kws), sg.Cancel(**b_kws)],
          ]

    starvation_table = sg.Col([[
        sg.Text(f'{ep.capitalize()}s (h)', **t18_kws),
        graphic_button('add', f'ADD {ep}', tooltip=f'Add a new {ep}.'),
        graphic_button('remove', f'REMOVE {ep}', tooltip=f'Remove an existing {ep}.'),
        graphic_button('data_add', f'ADD life', tooltip=f'Add a life history configuration.')
    ],
        [sg.Table(values=epochs[:][:], headings=[s0, s1, 'quality'], def_col_width=7,
                  max_col_width=24, background_color='lightblue',
                  header_background_color='lightorange',
                  auto_size_columns=False,
                  # display_row_numbers=True,
                  justification='center',
                  # font=w_kws['font'],
                  num_rows=len(epochs),
                  alternating_row_color='lightyellow',
                  key=K
                  )],
        # [sg.B('Remove', **b_kws, **pad2), sg.B('Add', **b_kws, **pad2)]
    ], **col_kws)

    l1 = [
        [starvation_table,
         sg.Col([*[[sg.Text(f'{i} (h): ', **t12_kws),
                    sg.Slider(range=(0, 250), default_value=0, key=k,
                              tick_interval=24, resolution=1, **sl2_kws)] for i, j, k in
                   zip([s0, s1, 'Starting age'], [st0, st1, 'age'], [S0, S1, Sa])],
                 # [sg.Text('')],
                 ])]
    ]
    l = [
        [l0, sg.Col([[sg.Canvas(size=r1_size, key='-CANVAS-')]])],
        [sg.Col(l1, size=l2_size), sg.Col(l2, size=r2_size)],
    ]
    w = sg.Window('Life history', l, location=(0, 0), size=(W, H), **w_kws)
    canvas_elem = w.FindElement('-CANVAS-')
    canvas = canvas_elem.TKCanvas
    fig_agg = None
    w.write_event_value('Draw', 'Draw the initial plot')
    # w[Sq].bind('<Button-3>', '+RIGHT+')
    # w[Sa].bind('<Button-3>', '+RIGHT+')
    # w[S0].bind('<Button-3>', '+RIGHT+')
    # w[S1].bind('<Button-3>', '+RIGHT+')

    while True:
        e, v = w.read()

        # print(e)

        def deb_model():
            dic = deb_default(epochs=[(t1, t2) for t1, t2, q in epochs], epoch_qs=[q for t1, t2, q in epochs],
                              substrate_quality=v[Sq], hours_as_larva=v[Sa],
                              substrate_type=collapsibles['substrate'].header_value)
            return dic

        def get_life_dict():
            life = {
                'epochs': [(t1, t2) for t1, t2, q in epochs] if len(epochs) > 0 else None,
                'epoch_qs': [q for t1, t2, q in epochs] if len(epochs) > 0 else None,
                'hours_as_larva': v[Sa],
                'substrate_quality': v[Sq],
                'substrate_type': collapsibles['substrate'].header_value,
            }
            return life

        if e in [None, 'Cancel']:
            break
        elif e == 'Ok':
            life = get_life_dict()
            print(life)
            break  # exit
        elif e == 'deb_mode':
            w.write_event_value('Draw', 'Draw the initial plot')
        elif e == f'ADD {ep}':
            t1, t2, q = v[S0], v[S1], v[Sq]
            if t2 > t1:
                epochs.append([t1, t2, q])
                epochs.sort(key=lambda x: x[0])
                w.Element(K).Update(values=epochs, num_rows=len(epochs))
                w.Element(S1).Update(value=0.0)
                w.Element(S0).Update(value=0.0)
                w.Element(Sq).Update(value=1.0)
                w.write_event_value('Draw', 'Draw the initial plot')
        elif e == f'REMOVE {ep}':
            if len(v[K]) > 0:
                epochs.remove(epochs[v[K][0]])
                w.Element(K).Update(values=epochs, num_rows=len(epochs))
                w.write_event_value('Draw', 'Draw the initial plot')
        elif e == 'ADD life':
            life = get_life_dict()
            id = save_conf_window(life, 'Life', disp='life history')
        elif e in [Sq]:
        # elif e in [Sq, Sa]:
            # elif e == f'{Sq}+UP':
            w.write_event_value('Draw', 'Draw the initial plot')

        elif e == 'Draw':
            D = deb_model()
            for Sii in [S0, S1, Sa]:
                w.Element(Sii).Update(range=(0.0, D['pupation'] - D['birth']))
            fig, save_to, filename = plot_debs(deb_dicts=[D], mode=v['deb_mode'][0], return_fig=True)
            if fig_agg:
                delete_figure_agg(fig_agg)
            fig_agg = draw_canvas(canvas, fig)

        elif e in [S0, S1]:
            if e == S0 and v[S0] > v[S1]:
                w.Element(S1).Update(value=v[S0])
            elif e == S1 and v[S1] < v[S0]:
                w.Element(S0).Update(value=v[S1])
            for t1, t2, q in epochs:
                if t1 < v[S0] < t2:
                    w.Element(S0).Update(value=t2)
                elif v[S0] < t1 and v[S1] > t1:
                    w.Element(S1).Update(value=t1)
                if t1 < v[S1] < t2:
                    w.Element(S1).Update(value=t1)
                elif v[S1] > t2 and v[S0] < t2:
                    w.Element(S0).Update(value=t2)
        else:
            default_run_window(w, e, v, collapsibles)
    w.close()
    return life


if __name__ == '__main__':
    life_conf()
