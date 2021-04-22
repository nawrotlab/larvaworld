#!/usr/bin/env python
# !/usr/bin/env python
import copy
import threading

import PySimpleGUI as sg
import matplotlib
from tkinter import *



sys.path.insert(0, '..')

from lib.gui.batch_tab import build_batch_tab, eval_batch, get_batch
from lib.gui.gui_lib import check_collapsibles, check_toggles, w_kws
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
    dicts = {
        'sim_results' : {'datasets' : []},
        'batch_kwargs' : None,
        'batch_results' : {},
        'analysis_data' : {},
    }


    l_mod, collapsibles = build_model_tab(collapsibles)
    l_sim, collapsibles, graph_lists = build_sim_tab(collapsibles, graph_lists)
    l_batch, collapsibles, graph_lists = build_batch_tab(collapsibles, graph_lists)
    l_anal, collapsibles, graph_lists = build_analysis_tab(collapsibles, graph_lists)

    l_gui = [
        [sg.TabGroup([[
            sg.Tab('Model', l_mod, background_color='darkseagreen', key='MODEL_TAB'),
            sg.Tab('Simulation', l_sim, background_color='darkseagreen', key='SIMULATION_TAB'),
            sg.Tab('Batch run', l_batch, background_color='darkseagreen', key='BATCH_TAB'),
            sg.Tab('Analysis', l_anal, background_color='darkseagreen', key='ANALYSIS_TAB')]],
            key='ACTIVE_TAB', tab_location='top', selected_title_color='purple')]
    ]

    w = sg.Window('Larvaworld gui', l_gui, size=(1800, 1200), **w_kws,location=(300, 100))

    while True:
        e, v = w.read()
        if e in (None, 'Exit'):
            break
        check_collapsibles(w,e, collapsibles)
        check_toggles(w,e)
        for name,graph_list in graph_lists.items() :
            if e==graph_list.list_key :
                graph_list.evaluate(w, v[graph_list.list_key])

        if e.startswith('EDIT_TABLE'):
            collapsibles[e.split()[-1]].edit_table(w)

        tab = v['ACTIVE_TAB']
        if tab == 'ANALYSIS_TAB':
            graph_lists, dicts = eval_analysis(e, v, w,collapsibles,graph_lists, dicts)
        elif tab == 'MODEL_TAB':
            eval_model(e, v, w, collapsibles)
        elif tab == 'BATCH_TAB':
            dicts, graph_lists = eval_batch(e, v, w, collapsibles, dicts, graph_lists)
        elif tab == 'SIMULATION_TAB':
            dicts, graph_lists = eval_sim(e, v, w, collapsibles, dicts, graph_lists)


        # if dicts['batch_kwargs'] :
        #     thread = threading.Thread(target=batch_thread, args=(dicts['batch_kwargs'], W, dicts),daemon=True)
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
        #     graph_lists['BATCH'].update(W, dicts['batch_results']['fig_dict'])
    w.close()
    return


if __name__ == "__main__":
    run_gui()
