import PySimpleGUI as sg
import matplotlib

from lib.gui.analysis_tab import AnalysisTab
from lib.gui.batch_tab import BatchTab
from lib.gui.env_tab import EnvTab
from lib.gui.essay_tab import EssayTab
from lib.gui.life_tab import LifeTab
from lib.gui.preprocess_tab import PreprocessTab
from lib.gui.sim_tab import SimTab
from lib.gui.tab import IntroTab, VideoTab, TutorialTab
from lib.gui.model_tab import ModelTab
from lib.gui.settings_tab import SettingsTab
import lib.gui.gui_lib as gui
import lib.aux.functions as fun
import lib.conf.dtype_dicts as dtypes

matplotlib.use('TkAgg')


class LarvaworldGui:

    def __init__(self, tabs=None):
        self.tab_dict = {
            'introduction': (IntroTab, None),
            'tutorials': (TutorialTab, None),
            'larva-model': (ModelTab, 'Model', 'model_conf'),
            'environment': (EnvTab, 'Env', 'env_conf'),
            'life-history': (LifeTab, 'Life', 'life'),
            'simulation': (SimTab, 'Exp', 'exp_conf'),
            'essay': (EssayTab, 'Essay', 'essay_conf'),
            'batch-run': (BatchTab, 'Batch', 'batch_conf'),
            'analysis': (AnalysisTab, None, None),
            'groups': (PreprocessTab, 'Group', None),
            'videos': (VideoTab, None, None),
            'settings': (SettingsTab, None, None)
        }

        if tabs is None:
            tabs = list(self.tab_dict.keys())
        # sg.change_look_and_feel('Dark Blue 3')
        sg.theme('LightGreen')
        self.background_color = None
        self.terminal = gui.gui_terminal()
        layout, self.collapsibles, self.graph_lists, self.dicts, self.tabs = self.build(tabs)

        c = {'layout': layout, 'size': gui.window_size, 'location': (300, 100), **gui.w_kws}
        self.window = sg.Window('Larvaworld gui', **c)

    def run(self):

        while True:

            e, v = self.window.read()
            if e in (None, 'Exit'):
                self.window.close()
                break
            else:
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
        # self.window.close()

    def build(self, tabs):
        ls, cs, ds, gs, ts = [], {}, {}, {}, {}
        for n in tabs:
            ii=self.tab_dict[n]
            ts[n] = ii[0](name=n, gui=self, conftype=ii[1])
            l, c, d, g = ts[n].build()
            cs.update(c)
            ds.update(d)
            gs.update(g)
            ls.append(sg.Tab(n, l, background_color=self.background_color, key=f'{n} TAB', ))

        l_tabs = sg.TabGroup([ls], key='ACTIVE_TAB', tab_location='topleft', selected_title_color='darkblue',
                             font=("Helvetica", 13, "normal"),
                             title_color='grey', selected_background_color=None,
                             tab_background_color='lightgrey', background_color=None)

        l0 = [[sg.Pane([sg.vtop(l_tabs), sg.vbottom(self.terminal)], handle_size=30)]]
        # l0 = [[sg.Pane([sg.Col([l_tabs]), sg.Col([[self.terminal]])], handle_size=30)]]
        return l0, cs, ds, gs, ts

    def get_vis_kwargs(self, v):
        c=self.collapsibles
        w=self.window
        vis_kwargs=c['Visualization'].get_dict(v, w) if 'Visualization' in list(
            c.keys()) else dtypes.get_dict('visualization')
        return vis_kwargs

    def get_replay_kwargs(self, v):
        c=self.collapsibles
        w=self.window
        replay_kwargs=c['Replay'].get_dict(v, w) if 'Replay' in list(
            c.keys()) else dtypes.get_dict('replay', arena_pars=None)
        return replay_kwargs
