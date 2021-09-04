import os
import PySimpleGUI as sg
import numpy as np
# from tkinter import *
from PySimpleGUI import LISTBOX_SELECT_MODE_EXTENDED

from lib.gui.gui_lib import ButtonGraphList, graphic_button, named_list, col_size, col_kws, change_dataset_id, \
    browse_button, remove_button, sel_all_button, changeID_button, replay_button, add_ref_button, DataList
from lib.gui.tab import GuiTab
from lib.stor import paths
from lib.anal.plotting import graph_dict
from lib.stor.larva_dataset import LarvaDataset
import lib.conf.dtype_dicts as dtypes
from lib.stor.managing import detect_dataset


class AnalysisTab(GuiTab):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        col_frac = 0.25
        self.canvas_size = col_size(x_frac=1 - 2 * col_frac, y_frac=0.8)
        self.canvas_col_size = col_size(x_frac=1 - 2 * col_frac)
        self.col_size = col_size(col_frac)
        self.data_key = 'Datasets'

    # def change_dataset_id(self,w, v, dic, k0):
    #     # k0 = 'DATASET_IDS'
    #     v0 = v[k0]
    #     # data=self.base_dict
    #     if len(v0) > 0:
    #         old_id = v0[0]
    #         l = [[sg.Text('Enter new dataset ID', size=(20, 1)), sg.In(k='NEW_ID', size=(10, 1))],
    #              [sg.Button('Store'), sg.Ok(), sg.Cancel()]]
    #         e1, v1 = sg.Window('Change dataset ID', l).read(close=True)
    #         new_id=v1['NEW_ID']
    #         if e1 == 'Ok':
    #             dic[new_id] = dic.pop(old_id)
    #             w.Element(k0).Update(values=list(dic.keys()))
    #         elif e1 == 'Store':
    #             d = dic[old_id]
    #             d.set_id(new_id)
    #             dic[new_id] = dic.pop(old_id)
    #             w.Element(k0).Update(values=list(dic.keys()))
    # return data

    def build(self):
        # initial_folder=f'{paths.DataFolder}/SchleyerGroup/processed'
        initial_folder = paths.SingleRunFolder
        dicts = {self.name: {}}

        lA = DataList(name=self.data_key, tab=self, dict=dicts[self.name],
                      buttons=['replay', 'add_ref', 'select_all', 'remove', 'changeID', 'browse'],
                      button_args={'browse': {'initial_folder' : initial_folder, 'target': (0, -1)}})
        self.datalists = {dl.name: dl for dl in [lA]}

        # bs = [
        #     sel_all_button(self.data_key),
        #     remove_button(self.data_key),
        #     replay_button(self.data_key),
        #     add_ref_button(self.data_key),
        #     changeID_button(self.data_key),
        #     browse_button(self.data_key, initial_folder, target=(0, -1))
        # ]
        # data_list = named_list('Datasets', f'{self.data_key}_IDS', list(dicts[self.name].keys()),
        #                        drop_down=False, list_width=25, list_height=10,
        #                        single_line=False, next_to_header=bs, as_col=False,
        #                        list_kws={'select_mode': LISTBOX_SELECT_MODE_EXTENDED})

        g = ButtonGraphList(name=self.name, fig_dict=graph_dict, canvas_size=self.canvas_size,
                            canvas_col_kws={'size': self.canvas_col_size, 'scrollable': True, **col_kws})

        l0=lA.get_layout(size=self.col_size, **col_kws)
        lc=[g.canvas]
        lg=[g.get_layout(size=self.col_size, **col_kws)]
        l = [l0+lc+lg]
        gs = {g.name: g}
        return l, {}, gs, dicts

    def eval(self, e, v, w, c, d, g):
        if e == f'{self.name}_REFRESH_FIGS':
            self.graph_list.refresh_figs(w, self.base_dict)
        elif e == f'{self.name}_SAVE_FIG':
            self.graph_list.save_fig()
        elif e == f'{self.name}_FIG_ARGS':
            self.graph_list.set_fig_args()
        elif e == f'{self.name}_DRAW_FIG':
            self.graph_list.generate(w, self.base_dict)
        return d, g


if __name__ == "__main__":
    from lib.gui.gui import LarvaworldGui

    larvaworld_gui = LarvaworldGui(tabs=['analysis'])
    larvaworld_gui.run()
