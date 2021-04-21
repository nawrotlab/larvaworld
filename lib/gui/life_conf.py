import copy
import random
from typing import Tuple

import PySimpleGUI as sg
import numpy as np

import lib.aux.functions as fun
import lib.conf.dtype_dicts as dtypes
from lib.anal.plotting import plot_debs
from lib.gui.gui_lib import CollapsibleDict, check_collapsibles, check_toggles, \
    retrieve_dict, t5_kws, t2_kws, color_pick_layout, popup_color_chooser, b_kws, t40_kws, b_kws, w_kws, t8_kws, \
    delete_figure_agg, draw_canvas, retrieve_value, t24_kws, t10_kws
from lib.model.deb import deb_dict, deb_default

W, H = 1400, 700
k = 2
sl1_kws = {
    'size': (30, 10),
    'enable_events': True,
    'orientation': 'h',
    'pad': ((k, 40 * k), (k, k))}

sl2_kws = {
    'size': (30, 10),
    'enable_events': True,
    'orientation': 'h',
    'pad': ((k, k), (3 * k, 3 * k)),
# 'background_color' : 'lightblue'
}

deb_modes=['mass', 'length',
               'reserve',
           # 'f',
               'reserve_density',
           'hunger',
               'puppation_buffer',
           # 'explore2exploit_balance',
           #     'f_filt'
           ]

def life_conf():
    sg.theme('LightGreen')
    # sg.theme('Dark Blue 3')
    life = dtypes.get_dict('life')
    starvation_hours = []

    l0=[[sg.Text('Parameter : ', **t10_kws)],
        [sg.Listbox(deb_modes, size=(14, len(deb_modes)),default_values=['reserve'], k='deb_mode', enable_events=True)]
    ]

    l1 = [[sg.Text('Food quality : ', **t24_kws)],
          [sg.Slider(range=(0.0, 1.0), default_value=1.0, key='SLIDER_quality',
                     tick_interval=0.25, resolution=0.01, **sl1_kws)],
          [sg.Text('', **t24_kws)],
          [sg.Text('Starting age (hours post-hatch): ', **t24_kws)],
          [sg.Slider(range=(0, 250), default_value=0, key=f'SLIDER_age',
                     tick_interval=25, resolution=0.1, **sl1_kws)]]

    l2 = [
        [sg.Col([[sg.Text('Starvation periods (h)')],
                 [sg.Table(values=starvation_hours[:][:], headings=['start', 'stop'], def_col_width=6,
                           max_col_width=20, background_color='lightblue', header_background_color='lightorange',
                           auto_size_columns=False,
                           # display_row_numbers=True,
                           justification='center',
                           # font=w_kws['font'],
                           num_rows=len(starvation_hours),
                           alternating_row_color='lightyellow',
                           key='STARVATION'
                           )],
                 # [sg.Listbox(values=[], change_submits=False, size=(22, 0), key='STARVATION',enable_events=True)],
                 [sg.B('Remove', **b_kws), sg.B('Add', **b_kws)]]),
         sg.Col([*[[sg.Text(f'{i} : ', size=(5, 1)),
                  sg.Slider(range=(0, 250), default_value=0, key=f'SLIDER_starvation_{i}',
                            tick_interval=25, resolution=0.1, **sl2_kws)] for i in ['start', 'stop']],
                 [sg.Text('')],
                 [sg.Text('', size=(25, 1)), sg.Ok(**b_kws), sg.Cancel(**b_kws)]])]
        # [sg.Text(' ' * 12)]
    ]

    graph_layout = [
        [sg.Col(l0),sg.Col([[sg.Canvas(size=(W, int(H * 2 / 3)), key='-CANVAS-')]])],
        [sg.Col(l1), sg.Col(l2)],
        # [sg.B('Draw', **b_kws)]
    ]
    w = sg.Window('Life history', graph_layout, location=(0, 0), size=(W, H), **w_kws)
    canvas_elem = w.FindElement('-CANVAS-')
    canvas = canvas_elem.TKCanvas
    fig_agg = None
    w.write_event_value('Draw', 'Draw the initial plot')

    while True:
        e, v = w.read()
        check_toggles(w, e)
        # info = w["info"]
        if e in [None, 'Cancel']:
            break
        elif e == 'Ok':
            life = {'starvation_hours': starvation_hours,
                    'hours_as_larva': v['SLIDER_age'],
                    'deb_base_f': v['SLIDER_quality']
                    }
            break  # exit
        elif e == 'Add':
            t1, t2 = v['SLIDER_starvation_start'], v['SLIDER_starvation_stop']
            if t2 > t1:
                starvation_hours.append([t1, t2])
                w.Element('STARVATION').Update(values=starvation_hours, num_rows=len(starvation_hours))
                w.Element('SLIDER_starvation_stop').Update(value=0.0)
                w.Element('SLIDER_starvation_start').Update(value=0.0)
                w.write_event_value('Draw', 'Draw the initial plot')
        elif e == 'Remove':
            if len(v['STARVATION']) > 0:
                starvation_hours.remove(starvation_hours[v['STARVATION'][0]])
                w.Element('STARVATION').Update(values=starvation_hours, num_rows=len(starvation_hours))
                w.write_event_value('Draw', 'Draw the initial plot')
        elif e == 'SLIDER_quality':
            w.write_event_value('Draw', 'Draw the initial plot')
        elif e == 'deb_mode':
            w.write_event_value('Draw', 'Draw the initial plot')
        elif e == 'Draw':
            deb_model = deb_default(starvation_hours=starvation_hours, base_f=v['SLIDER_quality'])
            fig, save_to, filename = plot_debs(deb_dicts=[deb_model], mode=v['deb_mode'][0], return_fig=True)
            if fig_agg:
                delete_figure_agg(fig_agg)
            fig_agg = draw_canvas(canvas, fig)
        if e == 'SLIDER_starvation_start' and v['SLIDER_starvation_start'] > v['SLIDER_starvation_stop']:
            w.Element('SLIDER_starvation_start').Update(value=v['SLIDER_starvation_stop'])
        if e == 'SLIDER_starvation_stop' and v['SLIDER_starvation_stop'] < v['SLIDER_starvation_start']:
            w.Element('SLIDER_starvation_stop').Update(value=v['SLIDER_starvation_start'])
        for ii in ['starvation_start', 'starvation_stop', 'age']:
            w.Element(f'SLIDER_{ii}').Update(range=(0.0, deb_model['puppation'] - deb_model['birth']))
    w.close()
    return life


if __name__ == '__main__':
    life_conf()
