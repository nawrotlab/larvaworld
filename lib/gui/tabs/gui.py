from typing import Any

import PySimpleGUI as sg
import matplotlib
import time

import pandas as pd

from lib.conf.pars.pars import ParDict
from lib.gui.aux.functions import col_size, window_size, w_kws

matplotlib.use('TkAgg')


class LarvaworldGui:
    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        from lib.gui.tabs import intro_tab, life_tab, sim_tab, batch_tab, essay_tab, import_tab, \
            analysis_tab, video_tab, tutorial_tab, settings_tab, model_tab
        cls.tab_dict = {
            'introduction': (intro_tab.IntroTab, None, None),
            'larva-model': (model_tab.ModelTab, 'Model', 'model_conf'),
            'life-history': (life_tab.LifeTab, 'Life', 'life'),
            # 'environment': (env_tab.EnvTab, 'Env', 'env_conf'),
            'simulation': (sim_tab.SimTab, 'Exp', 'exp_conf'),
            'batch-run': (batch_tab.BatchTab, 'Batch', 'batch_conf'),
            'essay': (essay_tab.EssayTab, 'Essay', 'essay_conf'),
            'import': (import_tab.ImportTab, 'Group', None),
            'analysis': (analysis_tab.AnalysisTab, None, None),
            'videos': (video_tab.VideoTab, None, None),
            'tutorials': (tutorial_tab.TutorialTab, None, None),
            'settings': (settings_tab.SettingsTab, None, None)
        }
        cls.tabgroups = {
            'introduction': ['introduction'],
            'models': ['larva-model', 'life-history'],
            # 'environment': ['environment'],
            'data': ['import', 'analysis'],
            'simulations': ['simulation', 'batch-run', 'essay'],
            'resources': ['tutorials', 'videos'],
            'settings': ['settings'],
        }
        return object.__new__(cls)

    def __init__(self, tabs=None):
        self.run_externally = {'sim': False, 'batch': True}
        if tabs is None:
            tabs = list(self.tab_dict.keys())
        sg.theme('LightGreen')
        self.background_color = None
        self.terminal = gui_terminal()
        layout, self.collapsibles, self.graph_lists, self.dicts, self.tabs = self.build(tabs)
        c = {'layout': layout, 'size': window_size, **w_kws}
        self.window = sg.Window('Larvaworld gui', **c)

    def run(self):
        while True:
            e, v = self.window.read(10000)
            if e in (None, 'Exit'):
                self.window.close()
                break
            else:
                self.run0(e, v)
                n = v['ACTIVE_TAB'].split()[0]
                self.tabs[n].eval0(e=e, v=v)

    def build(self, tabs):
        ls, cs, ds, gs, ts = [], {}, {}, {}, {}
        dic = {}
        for n in tabs:
            func, conftype, dtype = self.tab_dict[n]
            ts[n] = func(name=n, gui=self, conftype=conftype, dtype=dtype)
            l, c, d, g = ts[n].build()
            cs.update(c)
            ds.update(d)
            gs.update(g)
            dic[n] = sg.Tab(n, l, background_color=self.background_color, key=f'{n} TAB')
            ls.append(dic[n])
        tab_kws = {'font': ("Helvetica", 14, "normal"), 'selected_title_color': 'darkblue', 'title_color': 'grey',
                   'tab_background_color': 'lightgrey'}
        l_tabs = sg.TabGroup([ls], key='ACTIVE_TAB', tab_location='topleft', **tab_kws)
        l0 = [[sg.Pane([sg.vtop(l_tabs), sg.vbottom(self.terminal)], handle_size=30)]]
        return l0, cs, ds, gs, ts

    def get_vis_kwargs(self, v, **kwargs):
        from lib.conf.base.dtypes import null_dict
        c = self.collapsibles
        w = self.window
        return c['visualization'].get_dict(v, w) if 'visualization' in c.keys() else null_dict('visualization',
                                                                                                     **kwargs)

    def get_replay_kwargs(self, v):
        from lib.conf.base.dtypes import null_dict
        c = self.collapsibles
        w = self.window
        return c['replay'].get_dict(v, w) if 'replay' in c.keys() else null_dict('replay')

    def run0(self, e, v):
        w = self.window
        check_togglesNcollapsibles(w, e, v, self.collapsibles)
        check_multispins(w, e)
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


def speed_test():
    import numpy as np
    from lib.conf.base import paths
    ns0 = ['introduction', 'tutorials', 'larva-model', 'environment', 'life-history', 'simulation', 'essay',
           'batch-run', 'analysis', 'import', 'videos', 'settings']
    ns = [[n] for n in ns0]
    ns = [None] + ns + [None]
    ns0 = ['Total_1'] + ns0 + ['Total_2']
    res = []
    for n, n0 in zip(ns, ns0):
        s0 = time.time()
        larvaworld_gui = LarvaworldGui(tabs=n)
        s1 = time.time()
        larvaworld_gui.window.close()
        s2 = time.time()
        r = [n0, np.round(s1 - s0, 1), np.round(s2 - s1, 1)]
        res.append(r)
    df = pd.DataFrame(res)
    # fdir = ParDict.path_dict["GUITEST"]
    df.to_csv(ParDict.path_dict["GUITEST"], index=0, header=['Tabs', 'Open', 'Close'])
