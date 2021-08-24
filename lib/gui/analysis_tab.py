import os
import PySimpleGUI as sg
import numpy as np
# from tkinter import *

from lib.gui.gui_lib import t8_kws, ButtonGraphList, b6_kws, graphic_button, t10_kws, t16_kws, default_run_window, \
    w_kws, named_list_layout, col_size, col_kws
from lib.gui.tab import GuiTab
from lib.stor import paths
from lib.anal.plotting import graph_dict
from lib.stor.larva_dataset import LarvaDataset
import lib.conf.dtype_dicts as dtypes

class AnalysisTab(GuiTab):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def change_dataset_id(self,w, values):
        k0 = 'DATASET_IDS'
        v0 = values[k0]
        data=self.base_dict
        if len(v0) > 0:
            old_id = v0[0]
            l = [[sg.Text('Enter new dataset ID', size=(20, 1)), sg.In(k='NEW_ID', size=(10, 1))],
                 [sg.Button('Store'), sg.Ok(), sg.Cancel()]]
            e, v = sg.Window('Change dataset ID', l).read(close=True)
            new_id=v['NEW_ID']
            if e == 'Ok':
                data[new_id] = data.pop(old_id)
                w.Element(k0).Update(values=list(data.keys()))
            elif e == 'Store':
                d = data[old_id]
                d.set_id(new_id)
                data[new_id] = data.pop(old_id)
                w.Element(k0).Update(values=list(data.keys()))
        # return data


    def build(self):
        dicts = {self.name: {}}
        data_list = [
            [sg.Text('Datasets', **t8_kws),
             graphic_button('remove', 'Remove', tooltip='Remove a dataset from the analysis list.'),
             graphic_button('play', 'Replay', tooltip='Replay/Visualize the dataset.'),
             graphic_button('box_add', 'Add ref', tooltip='Add the reference experimental dataset to the analysis list.'),
             graphic_button('edit', 'Change ID', tooltip='Change the dataset ID transiently or permanently.'),
             graphic_button('search_add', 'DATASET_DIR', initial_folder=paths.SingleRunFolder, change_submits=True,
                            enable_events=True, target=(0, -1), button_type=sg.BUTTON_TYPE_BROWSE_FOLDER,
                            tooltip='Browse to add datasets to the analysis list.\n Either directly select a dataset directory or a parent directory containing multiple datasets.')
             ],

            [sg.Col([[sg.Listbox(values=list(dicts[self.name].keys()), size=(25, 5), key='DATASET_IDS', enable_events=True)]])]
        ]



        g = ButtonGraphList(name=self.name, fig_dict=graph_dict, canvas_col_kws={'size' : col_size(0.6), 'scrollable':True, **col_kws})
        l = [[sg.Col(data_list, size=col_size(0.2), **col_kws),
              g.canvas,
              sg.Col(g.get_layout(as_col=False), size=col_size(0.2), **col_kws)]]
        graph_lists = {g.name: g}
        return l, {}, graph_lists, dicts


    def eval(self, e, v, w, c, d, g):
        k0,kD='DATASET_IDS', 'DATASET_DIR'
        v0,vD=v[k0], v[kD]
        if e == kD:
            if vD != '':
                if os.path.exists(f'{vD}/data'):
                    dd = LarvaDataset(dir=vD)
                    self.base_dict[dd.id] = dd
                else:
                    for ddr in [x[0] for x in os.walk(vD)]:
                        if os.path.exists(f'{ddr}/data'):
                            dd = LarvaDataset(dir=ddr)
                            self.base_dict[dd.id] = dd
                w.Element(k0).Update(values=list(self.base_dict.keys()))

        elif e == 'Add ref':
            dd = LarvaDataset(dir=f'{paths.RefFolder}/reference')
            self.base_dict[dd.id] = dd
            w.Element(k0).Update(values=list(self.base_dict.keys()))
        elif e == 'Remove':
            if len(v0) > 0:
                self.base_dict.pop(v0[0], None)
                w.Element(k0).Update(values=list(self.base_dict.keys()))
        elif e == 'Change ID':
            self.change_dataset_id(w, v)
        elif e == 'Replay':
            if len(v0) > 0:
                dd = self.base_dict[v0[0]]
                dd.visualize(vis_kwargs=self.gui.get_vis_kwargs(v), **self.gui.get_replay_kwargs(v))

        elif e == 'ANALYSIS_SAVE_FIG':
            self.graph_list.save_fig()
        elif e == 'ANALYSIS_FIG_ARGS':
            self.graph_list.set_fig_args()
        elif e == 'ANALYSIS_DRAW_FIG':
            self.graph_list.generate(w, self.base_dict)
        return d, g

if __name__ == "__main__":
    from lib.gui.gui import LarvaworldGui
    larvaworld_gui = LarvaworldGui(tabs=['analysis'])
    larvaworld_gui.run()