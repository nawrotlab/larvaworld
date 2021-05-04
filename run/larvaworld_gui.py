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





class LarvaworldGui :
    def __init__(self):
        # sg.change_look_and_feel('Dark Blue 3')
        sg.theme('LightGreen')
        self.background_color = 'darkblue'
        collapsibles = {}
        graph_lists = {}
        dicts = {
            'sim_results': {'datasets': []},
            'batch_kwargs': None,
            'batch_results': {},
            'analysis_data': {},
        }

        l_mod, col_mod = build_model_tab()
        l_sim, col_sim, graph_lists_sim = build_sim_tab()
        l_batch, col_batch, graph_lists_batch = build_batch_tab()
        l_anal, col_anal, graph_lists_anal, dicts_anal = build_analysis_tab()
        for dic in [col_mod, col_sim, col_batch, col_anal]:
            collapsibles.update(dic)
        for dic in [graph_lists_sim, graph_lists_batch, graph_lists_anal]:
            graph_lists.update(dic)
        for dic in [dicts_anal]:
            dicts.update(dic)

        l_gui = [
            [sg.TabGroup([[
                sg.Tab('Model', l_mod, background_color=self.background_color,key='MODEL_TAB'),
                sg.Tab('Simulation', l_sim, background_color=self.background_color,key='SIMULATION_TAB'),
                sg.Tab('Batch run', l_batch, background_color=self.background_color,key='BATCH_TAB'),
                sg.Tab('Analysis', l_anal, background_color=self.background_color, key='ANALYSIS_TAB')]],
                key='ACTIVE_TAB', tab_location='top', selected_title_color='purple')]
        ]
        # self.layout=l_gui
        self.collapsibles=collapsibles
        self.graph_lists=graph_lists
        self.dicts=dicts

        self.window = sg.Window('Larvaworld gui', l_gui, size=(1800, 1200), **w_kws, location=(300, 100))

    def run(self):
        while True:

            e, v = self.window.read()
            if e in (None, 'Exit'):
                break
            check_collapsibles(self.window, e, self.collapsibles)
            check_toggles(self.window, e)
            for name, graph_list in self.graph_lists.items():
                if e == graph_list.list_key:
                    graph_list.evaluate(self.window, v[graph_list.list_key])

            if e.startswith('EDIT_TABLE'):
                self.collapsibles[e.split()[-1]].edit_table(self.window)

            tab = v['ACTIVE_TAB']
            if tab == 'ANALYSIS_TAB':
                self.graph_lists, self.dicts = eval_analysis(e, v, self.window, self.collapsibles, self.graph_lists, self.dicts)
            elif tab == 'MODEL_TAB':
                eval_model(e, v, self.window, self.collapsibles)
            elif tab == 'BATCH_TAB':
                self.dicts, self.graph_lists = eval_batch(e, v, self.window, self.collapsibles, self.dicts, self.graph_lists)
            elif tab == 'SIMULATION_TAB':
                self.dicts, self.graph_lists = eval_sim(e, v, self.window, self.collapsibles, self.dicts, self.graph_lists)

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
            # print(v)
        self.window.close()

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



if __name__ == "__main__":
    gui=LarvaworldGui()
    gui.run()
