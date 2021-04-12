#!/usr/bin/env python
# !/usr/bin/env python
import copy
import PySimpleGUI as sg
import matplotlib
from tkinter import *



sys.path.insert(0, '..')
from lib.gui.analysis_tab import build_analysis_tab, eval_analysis
from lib.gui.batch_tab import build_batch_tab, eval_batch
from lib.gui.gui_lib import SYMBOL_DOWN, SYMBOL_UP, on_image, off_image
from lib.gui.model_tab import build_model_tab, eval_model
from lib.gui.simulation_tab import build_sim_tab, eval_sim

matplotlib.use('TkAgg')
sg.theme('LightGreen')


def run_gui():
    collapsibles={}
    l_anal, graph_dict, data, func, func_kwargs, fig, save_to, save_as, figure_agg = build_analysis_tab()
    l_sim, sim_datasets, collapsibles, source_units, border_list, larva_groups, source_groups = build_sim_tab(collapsibles)
    l_mod, collapsibles, odor_gains = build_model_tab(collapsibles)
    l_batch, collapsibles, space_search = build_batch_tab(collapsibles)

    l_gui = [
        [sg.TabGroup([[
            sg.Tab('Model', l_mod, background_color='darkseagreen', key='MODEL_TAB'),
            sg.Tab('Simulation', l_sim, background_color='darkseagreen', key='SIMULATION_TAB'),
            sg.Tab('Batch run', l_batch, background_color='darkseagreen', key='BATCH_TAB'),
            sg.Tab('Analysis', l_anal, background_color='darkseagreen', key='ANALYSIS_TAB')]],
            key='ACTIVE_TAB', tab_location='top', selected_title_color='purple')]
    ]

    w = sg.Window('Larvaworld gui', l_gui, resizable=True, finalize=True, size=(2000, 1200))

    while True:
        e, v = w.read()
        if e in (None, 'Exit'):
            break
        if e.startswith('OPEN SEC'):
            sec = e.split()[-1]
            s=collapsibles[sec].state
            if s is not None:
                s = not s
                w[e].update(SYMBOL_DOWN if s else SYMBOL_UP)
                w[f'SEC {sec}'].update(visible=s)
        elif 'TOGGLE' in e:
            s=w[e].metadata.state
            if s is not None:
                s = not s
                w[e].update(image_data=on_image if s else off_image)

        tab = v['ACTIVE_TAB']
        if tab == 'ANALYSIS_TAB':
            w, func, func_kwargs, data, figure_agg, fig, save_to, save_as = eval_analysis(e, v, w,
                                                                                          func, func_kwargs, data,
                                                                                          figure_agg, fig, save_to,
                                                                                          save_as, graph_dict)
        elif tab == 'MODEL_TAB':
            odor_gains = eval_model(e, v, w, collapsibles, odor_gains)
        elif tab == 'BATCH_TAB':
            space_search = eval_batch(e, v, w, collapsibles, space_search)
        elif tab == 'SIMULATION_TAB':
            source_units, border_list, larva_groups, source_groups = eval_sim(e, v, w, sim_datasets, collapsibles,
                                                                              source_units, border_list, larva_groups,
                                                                              source_groups)
    w.close()
    return


if __name__ == "__main__":
    run_gui()
