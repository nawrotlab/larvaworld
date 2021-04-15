#!/usr/bin/env python
# !/usr/bin/env python
import copy
import threading

import PySimpleGUI as sg
import matplotlib
from tkinter import *

from lib.anal.combining import render_mpl_table
from lib.sim.batch_lib import batch_run

sys.path.insert(0, '..')

from lib.gui.batch_tab import build_batch_tab, eval_batch, get_batch
from lib.gui.gui_lib import SYMBOL_DOWN, SYMBOL_UP, on_image, off_image
from lib.gui.model_tab import build_model_tab, eval_model
from lib.gui.simulation_tab import build_sim_tab, eval_sim, get_exp
from lib.gui.analysis_tab import build_analysis_tab, eval_analysis
matplotlib.use('TkAgg')

# sg.change_look_and_feel('Dark Blue 3')
sg.theme('LightGreen')

# def batch_thread(kwargs, window, dicts):
#     """
#     A worker thread that communicates with the GUI through a global message variable
#     This thread can block for as long as it wants and the GUI will not be affected
#     :param seconds: (int) How long to sleep, the ultimate blocking call
#     """
#     # progress = 0
#     # print('Thread started - will sleep for {} seconds'.format(seconds))
#     # for i in range(int(seconds * 10)):
#     #     time.sleep(.1)  # sleep for a while
#     #     progress += 100 / (seconds * 10)
#     #     window.write_event_value('-PROGRESS-', progress)
#
#     df, fig_dict = batch_run(**kwargs)
#     df_ax, df_fig = render_mpl_table(df)
#     fig_dict['dataframe'] = df_fig
#     dicts['batch_results']['df'] = df
#     dicts['batch_results']['fig_dict'] = fig_dict
#
#     window.write_event_value('-THREAD-', '*** The thread says.... "I am finished" ***')

def run_gui():
    thread=None
    collapsibles={}
    graph_lists={}
    dicts = {}
    dicts['batch_kwargs']=None
    l_anal, collapsibles, graph_lists, dicts = build_analysis_tab(collapsibles,graph_lists, dicts)
    l_mod, collapsibles, dicts = build_model_tab(collapsibles, dicts)
    l_sim, collapsibles, graph_lists, dicts = build_sim_tab(collapsibles, graph_lists, dicts)
    l_batch, collapsibles, graph_lists, dicts = build_batch_tab(collapsibles, graph_lists, dicts)

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
            if collapsibles[sec].state is not None:
                collapsibles[sec].state = not collapsibles[sec].state
                w[e].update(SYMBOL_DOWN if collapsibles[sec].state else SYMBOL_UP)
                w[f'SEC {sec}'].update(visible=collapsibles[sec].state)
        elif 'TOGGLE' in e:
            if w[e].metadata.state is not None:
                w[e].metadata.state = not w[e].metadata.state
                w[e].update(image_data=on_image if w[e].metadata.state else off_image)

        else :
            for name,graph_list in graph_lists.items() :
                if e==graph_list.list_key :
                    graph_list.evaluate(w, v[graph_list.list_key])


        tab = v['ACTIVE_TAB']
        if tab == 'ANALYSIS_TAB':
            graph_lists, dicts = eval_analysis(e, v, w,collapsibles,graph_lists, dicts)
                                                                                          # func, func_kwargs,
                                                                                          # figure_agg, fig, save_to,
                                                                                          # save_as,

        elif tab == 'MODEL_TAB':
            dicts = eval_model(e, v, w, collapsibles, dicts)
        elif tab == 'BATCH_TAB':
            dicts, graph_lists = eval_batch(e, v, w, collapsibles, dicts, graph_lists)
        elif tab == 'SIMULATION_TAB':
            dicts, graph_lists = eval_sim(e, v, w, collapsibles, dicts, graph_lists)


        # if dicts['batch_kwargs'] :
        #     thread = threading.Thread(target=batch_thread, args=(dicts['batch_kwargs'], w, dicts),daemon=True)
        #     thread.start()
        #     dicts['batch_kwargs'] = None
        #
        #
        # elif e == '-THREAD-':  # Thread has completed
        #     thread.join(timeout=0)
        #     # print('Thread finished')
        #     # sg.popup_animated(None)  # stop animination in case one is running
        #     thread = None  # reset variables for next run
        #     # thread, message, progress, timeout = None, '', 0, None  # reset variables for next run
        #     graph_lists['BATCH'].update(w, dicts['batch_results']['fig_dict'])
    w.close()
    return


if __name__ == "__main__":
    run_gui()
