import PySimpleGUI as sg
import matplotlib

from lib.gui.analysis_tab import AnalysisTab
from lib.gui.batch_tab import BatchTab
from lib.gui.env_tab import EnvTab
from lib.gui.sim_tab import SimTab
from lib.gui.tab import IntroTab, VideoTab
from lib.gui.model_tab import ModelTab
from lib.gui.settings_tab import SettingsTab
import lib.gui.gui_lib as gui

matplotlib.use('TkAgg')


class LarvaworldGui:

    def __init__(self, tabs=None):
        self.tab_dict = {'introduction': (IntroTab,None),
                         'larva-model': (ModelTab,'Model'),
                         'environment': (EnvTab,'Env'),
                         'simulation': (SimTab,'Exp'),
                         'batch-run': (BatchTab,'Batch'),
                         'analysis': (AnalysisTab,None),
                         'videos': (VideoTab,None),
                         'settings': (SettingsTab,None)
                         }

        if tabs is None:
            tabs = list(self.tab_dict.keys())
        # sg.change_look_and_feel('Dark Blue 3')
        sg.theme('LightGreen')
        self.background_color = None
        layout,self.collapsibles, self.graph_lists, self.dicts, self.tabs = self.build(tabs)
        c = {'layout': layout, 'size': gui.window_size, 'location': (300, 100), **gui.w_kws}
        self.window = sg.Window('Larvaworld gui', **c)


    def run(self):
        while True:
            e, v = self.window.read()
            if e in (None, 'Exit'):
                break
            gui.default_run_window(self.window, e, v, self.collapsibles, self.graph_lists)

            n = v['ACTIVE_TAB'].split()[0]
            self.tabs[n].eval0(e=e, v=v)
            # self.dicts, self.graph_lists = self.tabs[n].eval0(e=e, v=v)


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

    def build(self, tabs):
        ls, cs, ds, gs, ts = [], {}, {}, {},{}
        for n in tabs:
            ts[n]=self.tab_dict[n][0](name=n,gui=self)
            l, c, d, g = ts[n].build()
            cs.update(c)
            ds.update(d)
            gs.update(g)
            ls.append(sg.Tab(n, l, background_color=self.background_color, key=f'{n} TAB', ))
        l0 = [[sg.TabGroup([ls], key='ACTIVE_TAB', tab_location='topleft', selected_title_color='darkblue',
                           font=("Helvetica", 13, "normal"),
                           title_color='grey', selected_background_color=None,
                           tab_background_color='lightgrey', background_color=None)]]
        return l0,cs, ds, gs, ts

