import copy
import random

import numpy as np
import PySimpleGUI as sg

import lib.aux.ang_aux
import lib.aux.sim_aux
import lib.aux.xy_aux
import lib.conf.dtypes
import lib.gui.aux.functions
from lib.conf.dtypes import null_dict
from lib.gui.aux.elements import CollapsibleDict, Collapsible, CollapsibleTable, GraphList, SelectionList
from lib.gui.aux.functions import col_size, col_kws, t_kws, retrieve_dict, gui_col
from lib.gui.aux.buttons import color_pick_layout, GraphButton
from lib.conf.conf import loadConf
from lib.gui.tabs.tab import GuiTab
from lib.sim.single_run import run_sim


class EnvTab(GuiTab):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.canvas_size=(800,800)
        self.S, self.L, self.B = 'Source', 'Larva', 'Border'
        self.Su, self.Sg=f'{self.S.lower()}_units', f'{self.S.lower()}_groups'
        # self.Lu, self.Lg=f'{self.L.lower()}_units', f'{self.L.lower()}_groups'
        self.Bg = f'{self.B.lower()}_list'


    def update(self, w, c, conf, id=None):
        for n in [self.Bg, 'arena', 'odorscape']:
            c[n].update(w, conf[n] if n in conf.keys() else {})
        for n in [self.Sg, self.Su, 'food_grid']:
            c[n].update(w, conf['food_params'][n])
        # self.base_dict['env_db'] = self.set_env_db(env=conf)
        # w.write_event_value('RESET_ARENA', 'Draw the initial arena')

    def get(self, w, v, c, as_entry=False):
        return {
            'food_params': {n: c[n].get_dict(v, w) for n in [self.Sg, self.Su, 'food_grid']},
            **{n: c[n].get_dict(v, w) for n in [self.Bg, 'arena', 'odorscape']}
            # **{n: c[n].get_dict(v, w) for n in [self.Lg, self.Bg, 'arena', 'odorscape']}
        }

    def build(self):
        s2 = CollapsibleTable(self.Sg, index='Group ID',heading_dict={'N':'distribution.N', 'color': 'default_color', 'odor_id' : 'odor.odor_id', 'amount' : 'amount'},dict_name='SourceGroup', state=True)
        s3 = CollapsibleTable(self.Su, index='ID', heading_dict={'color': 'default_color', 'odor_id' : 'odor.odor_id', 'amount' : 'amount'},dict_name='source', state=True)
        s4 = CollapsibleTable(self.Bg, index='ID', heading_dict={'color': 'default_color', 'points' : 'points' },dict_name='border_list')

        c1 = [CollapsibleDict(n, **kw)
              for n, kw in zip(['arena', 'food_grid', 'odorscape'], [{'next_to_header':[
                                 GraphButton('Button_Burn', 'RESET_ARENA',
                                                tooltip='Reset to the initial arena. All drawn items will be erased.'),
                                 GraphButton('Globe_Active', 'NEW_ARENA',
                                                tooltip='Create a new arena.All drawn items will be erased.'),
                             ]}, {'toggle': True}, {}])]
        c = {}
        for s in c1 + [s2, s3, s4]:
            c.update(s.get_subdicts())
        l1 = [c[n].get_layout() for n in [self.Sg, self.Su, 'food_grid']]
        c2 = Collapsible(self.S, content=l1, state=True)
        c.update(c2.get_subdicts())
        l2 = [c[n] for n in ['arena', self.S, self.Bg, 'odorscape']]
        # print(self.name)
        sl1 = SelectionList(tab=self, buttons=['save', 'delete'], disp=self.name)
        l = [[gui_col([sl1,*l2], 0.25)]]
        self.layout=l
        return l, c, {}, {}


    # def run(self, v, w,c,d,g, conf,id):
    #     sim=null_dict('sim_params', sim_ID='env_test', duration=0.5)
    #     exp_conf=null_dict('exp_conf', env_params=conf, sim_params=sim)
    #     exp_conf['life_params']=loadConf(exp_conf['life_params'], 'Life')
    #     # p = self.base_list.progressbar
    #     # p.run(w, max=N)
    #     exp_conf['experiment'] = 'test'
    #     exp_conf['save_data_flag'] = False
    #     exp_conf['vis_kwargs'] = null_dict('visualization', mode='video', video_speed=60)
    #     res = run_sim(**exp_conf)
    #     return d, g

if __name__ == "__main__":
    from lib.gui.tabs.gui import LarvaworldGui
    larvaworld_gui = LarvaworldGui(tabs=['environment'])
    larvaworld_gui.run()
