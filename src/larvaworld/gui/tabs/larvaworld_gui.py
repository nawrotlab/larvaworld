from typing import Any
import numpy as np
import PySimpleGUI as sg
import time

import pandas as pd

from larvaworld.lib import reg
from larvaworld.gui import gui_aux, tabs

def build_tab_dict():
    tab_dict = {
        'intro': (tabs.IntroTab, None, None, 'introduction'),
        'model': (tabs.ModelTab, 'Model', 'Model', 'larva-model'),
        'life': (tabs.LifeTab, 'Life', 'Life', 'life-history'),
        'sim': (tabs.SimTab, 'Exp', 'Exp', 'simulation'),
        'batch': (tabs.BatchTab, 'Batch', 'Batch', 'batch-exec'),
        'essay': (tabs.EssayTab, 'Essay', 'essay_conf', 'essay'),
        'import': (tabs.ImportTab, 'Group', None, 'import'),
        'anal': (tabs.AnalysisTab, None, None, 'analysis'),
        'vid': (tabs.VideoTab, None, None, 'videos'),
        'tutor': (tabs.TutorialTab, None, None, 'tutorials'),
        'set': (tabs.SettingsTab, None, None, 'settings')
    }
    return tab_dict


class LarvaworldGui:
    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        # reg.init(ks=['CT','PI','DEF','GT','GD'])
        cls.tab_dict = build_tab_dict()
        cls.tab_keys = list(cls.tab_dict.keys())
        return object.__new__(cls)

    def __init__(self, tabs=None, add_terminal=True):
        # print("a")
        self.run_externally = {'sim': False, 'batch': True}
        if tabs is None:
            tabs = self.tab_keys
        for t in tabs :
            if t not in self.tab_keys:
                raise ValueError(f'{t} not in tab keys')
        if 'sim' in tabs :
            tabs=[t for t in tabs if t!='env']
        # print(tabs)
        sg.theme('LightGreen')
        self.background_color = None

        # raise
        l_tabs, self.collapsibles, self.graph_lists, self.dicts, self.tabs = self.build(tabs)
        if add_terminal:
            self.terminal = gui_terminal()

            layout = [[sg.Pane([sg.vtop(l_tabs), sg.vbottom(self.terminal)], handle_size=30)]]
        else:
            self.terminal=None
            layout=l_tabs
        c = {'layout': layout, 'size': gui_aux.window_size, **gui_aux.w_kws}

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

        for t in tabs:

            func, conftype, dtype,n = self.tab_dict[t]
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

        # self.terminal = gui_terminal()

        # l0 = [[sg.Pane([sg.vtop(l_tabs), sg.vbottom(self.terminal)], handle_size=30)]]
        return l_tabs, cs, ds, gs, ts

    def get_vis_kwargs(self, v, **kwargs):
        c = self.collapsibles
        w = self.window
        return c['visualization'].get_dict(v, w) if 'visualization' in c.keys() else reg.get_null('visualization',
                                                                                                     **kwargs)

    def get_replay_kwargs(self, v):
        c = self.collapsibles
        w = self.window
        return c['Replay'].get_dict(v, w) if 'Replay' in c.keys() else reg.get_null('Replay')

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


def gui_terminal(size=gui_aux.col_size(y_frac=0.3)):
    return sg.Output(size=size, key='Terminal', background_color='black', text_color='white',
                     echo_stdout_stderr=True, font=('Helvetica', 8, 'normal'),
                     tooltip='Terminal output')


def speed_test():

    ns0 = ['introduction', 'tutorials', 'larva-model', 'environment', 'life-history', 'simulation', 'essay',
           'batch-exec', 'analysis', 'import', 'videos', 'settings']
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
    df.to_csv(f'{reg.ROOT_DIR}/gui/gui_speed_test.csv', index=0, header=['Tabs', 'Open', 'Close'])

