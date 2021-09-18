import PySimpleGUI as sg
import matplotlib
import time

import pandas as pd

import lib.gui.aux.functions
from lib.stor import paths
from lib.gui.tabs.analysis_tab import AnalysisTab
from lib.gui.aux.functions import col_size, window_size, w_kws
from lib.gui.tabs.batch_tab import BatchTab
from lib.gui.tabs.env_tab import EnvTab
from lib.gui.tabs.essay_tab import EssayTab
from lib.gui.tabs.life_tab import LifeTab
from lib.gui.tabs.import_tab import ImportTab
from lib.gui.tabs.sim_tab import SimTab
from lib.gui.tabs.tab import IntroTab, VideoTab, TutorialTab
from lib.gui.tabs.model_tab import ModelTab
from lib.gui.tabs.settings_tab import SettingsTab
import lib.conf.dtype_dicts as dtypes

matplotlib.use('TkAgg')


class LarvaworldGui:

    def __init__(self, tabs=None):
        # s0 = time.time()
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
            'import': (ImportTab, 'Group', None),
            'videos': (VideoTab, None, None),
            'settings': (SettingsTab, None, None)
        }

        if tabs is None:
            tabs = list(self.tab_dict.keys())
        # sg.change_look_and_feel('Dark Blue 3')
        sg.theme('LightGreen')
        self.background_color = None
        self.terminal = gui_terminal()
        # s1 = time.time()
        layout, self.collapsibles, self.graph_lists, self.dicts, self.tabs = self.build(tabs)
        # s2 = time.time()
        c = {'layout': layout, 'size': window_size, **w_kws}
        # c = {'layout': layout, 'size': window_size, 'location': (300, 100), **w_kws}
        self.window = sg.Window('Larvaworld gui', **c)
        # s3 = time.time()
        # print(s3-s2)
        # print(s2-s1)
        # print(s1-s0)

    def run(self):

        while True:

            e, v = self.window.read()
            if e in (None, 'Exit'):
                self.window.close()
                break
            else:
                self.run0(e, v)

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
                             # size=gui.col_size(y_frac=0.7),
                             title_color='grey', selected_background_color=None,
                             tab_background_color='lightgrey', background_color=None)

        l0 = [[sg.Pane([sg.vtop(l_tabs), sg.vbottom(self.terminal)], handle_size=30)]]
        return l0, cs, ds, gs, ts

    def get_vis_kwargs(self, v, **kwargs):
        c=self.collapsibles
        w=self.window
        vis_kwargs=c['Visualization'].get_dict(v, w) if 'Visualization' in list(
            c.keys()) else dtypes.get_dict('visualization', **kwargs)
        return vis_kwargs

    def get_replay_kwargs(self, v):
        c=self.collapsibles
        w=self.window
        replay_kwargs=c['Replay'].get_dict(v, w) if 'Replay' in list(
            c.keys()) else dtypes.get_dict('replay', arena_pars=None)
        return replay_kwargs

    def run0(self, e, v):
        w=self.window
        check_togglesNcollapsibles(w, e, v, self.collapsibles)
        check_multispins(w, e)
        # for g in self.graph_lists.values():
        #     if e == g.list_key:
        #         g.evaluate(w, v[g.list_key])

        if e.startswith('EDIT_TABLE'):
            self.collapsibles[e.split()[-1]].edit_table(w)


def check_multispins(w, e):
    if e.startswith('SPIN+'):
        k = e.split()[-1]
        w.Element(k).add_spin(w)
    elif e.startswith('SPIN-'):
        k = e.split()[-1]
        w.Element(k).remove_spin(w)


def check_collapsibles(w, e, v, c):
    if e.startswith('OPEN SEC'):
        sec = e.split()[-1]
        c[sec].click(w)
    elif e.startswith('SELECT LIST'):
        sec = e.split()[-1]
        c[sec].update_header(w, v[e])


def check_toggles(w, e):
    if 'TOGGLE' in e:
        w[e].toggle()


def check_togglesNcollapsibles(w, e, v, c):
    toggled = []
    check_collapsibles(w, e, v, c)
    if 'TOGGLE' in e:
        w[e].toggle()
        name = e[7:]
        if name in list(c.keys()):
            c[name].toggle = not c[name].toggle
        toggled.append(name)
    return toggled





def gui_terminal(size=col_size(y_frac=0.3)):
    return sg.Output(size=size, key='Terminal', background_color='black', text_color='white',
                     echo_stdout_stderr=True, font=('Helvetica', 8, 'normal'),
                     tooltip='Terminal output')

def speed_test() :
    import numpy as np

    # ns0=['introduction']
    ns0=['introduction', 'tutorials', 'larva-model', 'environment', 'life-history', 'simulation', 'essay', 'batch-run', 'analysis', 'import', 'videos', 'settings']
    ns=[[n] for n in ns0]
    ns=[None]+ns+[None]
    ns0=['Total_1']+ns0+['Total_2']
    res=[]
    for n, n0 in zip(ns, ns0) :
        s0=time.time()
        larvaworld_gui = LarvaworldGui(tabs=n)
        s1 = time.time()
        larvaworld_gui.window.close()
        s2 = time.time()
        r=[n0, np.round(s1 - s0,1), np.round(s2 - s1,1)]
        res.append(r)
    df=pd.DataFrame(res)
    df.to_csv(paths.GuiTest, index=0, header=['Tabs', 'Open', 'Close'])