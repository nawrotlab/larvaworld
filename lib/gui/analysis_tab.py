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


    def change_dataset_id(self,window, values, data):
        if len(values['DATASET_IDS']) > 0:
            old_id = values['DATASET_IDS'][0]
            l = [[sg.Text('Enter new dataset ID', size=(20, 1)), sg.In(k='NEW_ID', size=(10, 1))],
                 [sg.Button('Store'), sg.Ok(), sg.Cancel()]]
            e, v = sg.Window('Change dataset ID', l).read(close=True)
            if e == 'Ok':
                data[v['NEW_ID']] = data.pop(old_id)
                window.Element('DATASET_IDS').Update(values=list(data.keys()))
            elif e == 'Store':
                d = data[old_id]
                d.set_id(v['NEW_ID'])
                data[v['NEW_ID']] = data.pop(old_id)
                window.Element('DATASET_IDS').Update(values=list(data.keys()))
        return data


    def build(self):
        graph_lists = {}
        dicts = {'analysis_data': {}}
        data = dicts['analysis_data']
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

            [sg.Col([[sg.Listbox(values=list(data.keys()), size=(25, 5), key='DATASET_IDS', enable_events=True)]])]
        ]



        graph_lists['ANALYSIS'] = g = ButtonGraphList(name='ANALYSIS', fig_dict=graph_dict, canvas_col_kws={'size' : col_size(0.6), 'scrollable':True, **col_kws})
        l = [[sg.Col(data_list, size=col_size(0.2), **col_kws),
              g.canvas,
              sg.Col(g.get_layout(as_col=False), size=col_size(0.2), **col_kws)]]
        return l, {}, graph_lists, dicts


    def eval(self, e, v, w, c, d, g):
        if e == 'DATASET_DIR':
            dr = v['DATASET_DIR']
            if dr != '':
                if os.path.exists(f'{dr}/data'):
                    dd = LarvaDataset(dir=dr)
                    d['analysis_data'][dd.id] = dd
                else:
                    for ddr in [x[0] for x in os.walk(dr)]:
                        if os.path.exists(f'{ddr}/data'):
                            dd = LarvaDataset(dir=ddr)
                            d['analysis_data'][dd.id] = dd
                w.Element('DATASET_IDS').Update(values=list(d['analysis_data'].keys()))

        elif e == 'Add ref':
            sample_dataset='reference'
            dd = LarvaDataset(dir=f'{paths.RefFolder}/{sample_dataset}')
            d['analysis_data'][dd.id] = dd
            w.Element('DATASET_IDS').Update(values=list(d['analysis_data'].keys()))
        elif e == 'Remove':
            if len(v['DATASET_IDS']) > 0:
                d['analysis_data'].pop(v['DATASET_IDS'][0], None)
                w.Element('DATASET_IDS').Update(values=list(d['analysis_data'].keys()))
        elif e == 'Change ID':
            d['analysis_data'] = self.change_dataset_id(w, v, d['analysis_data'])
        elif e == 'Replay':
            if len(v['DATASET_IDS']) > 0:
                dd = d['analysis_data'][v['DATASET_IDS'][0]]
                if 'Visualization' in list(c.keys()):
                    vis_kwargs = c['Visualization'].get_dict(v, w)
                else:
                    vis_kwargs = dtypes.get_dict('visualization', mode='video', video_speed=60)

                if 'Replay' in list(c.keys()):
                    replay_kwargs = c['Replay'].get_dict(v, w)
                else:
                    replay_kwargs = dtypes.get_dict('replay', arena_pars=None)
                dd.visualize(vis_kwargs=vis_kwargs, **replay_kwargs)

        elif e == 'ANALYSIS_SAVE_FIG':
            g['ANALYSIS'].save_fig()
        elif e == 'ANALYSIS_FIG_ARGS':
            g['ANALYSIS'].set_fig_args()
        elif e == 'ANALYSIS_DRAW_FIG':
            g['ANALYSIS'].generate(w, d['analysis_data'])
        return d, g

if __name__ == "__main__":
    from lib.gui.gui import LarvaworldGui
    larvaworld_gui = LarvaworldGui(tabs=['analysis'])
    larvaworld_gui.run()