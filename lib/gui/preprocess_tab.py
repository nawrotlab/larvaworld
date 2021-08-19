import os
import PySimpleGUI as sg
import numpy as np
# from tkinter import *

from lib.gui.gui_lib import t8_kws, ButtonGraphList, b6_kws, graphic_button, t10_kws, t16_kws, default_run_window, \
    w_kws, named_list_layout, col_size, col_kws, get_disp_name
from lib.gui.tab import GuiTab, SelectionList
from lib.stor import paths
from lib.anal.plotting import graph_dict
from lib.stor.larva_dataset import LarvaDataset
import lib.conf.dtype_dicts as dtypes
from lib.stor.managing import detect_dataset, build_datasets
import lib.aux.functions as fun

class PreprocessTab(GuiTab):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.raw_key='raw_data'
        self.proc_key='processed_data'
        self.raw_ids_key=f'{self.raw_key}_IDS'
        self.proc_ids_key=f'{self.proc_key}_IDS'
        self.raw_folder=None

    def datagroup_id(self,v):
        return v[self.selectionlists[0].k]

    def change_dataset_id(self,window, values, data):
        if len(values[self.raw_ids_key]) > 0:
            old_id = values[self.raw_ids_key][0]
            l = [[sg.Text('Enter new dataset ID', size=(20, 1)), sg.In(k='NEW_ID', size=(10, 1))],
                 [sg.Button('Store'), sg.Ok(), sg.Cancel()]]
            e, v = sg.Window('Change dataset ID', l).read(close=True)
            if e == 'Ok':
                data[v['NEW_ID']] = data.pop(old_id)
                window.Element(self.raw_ids_key).Update(values=list(data.keys()))
            elif e == 'Store':
                d = data[old_id]
                d.set_id(v['NEW_ID'])
                data[v['NEW_ID']] = data.pop(old_id)
                window.Element(self.raw_ids_key).Update(values=list(data.keys()))
        return data

    def update(self,w, c, conf, id=None):
        p=conf['path']
        path=os.path.normpath(f'{paths.DataFolder}/{p}/raw')
        w[self.raw_key].InitialFolder=path
        self.raw_folder = path
        # w.Element(self.raw_key).InitialFolder=path
        # w.Element(self.raw_key).Update(initial_folder=path)
        # c['substrate'].update_header(w, conf['substrate_type'])
        #
        # w.Element(self.Sq).Update(value=conf['substrate_quality'])
        # w.Element(self.Sa).Update(value=conf['hours_as_larva'])
        # if conf['epochs'] is not None :
        #     epochs=[[t0,t1,q] for (t0,t1),q in zip(conf['epochs'], conf['epoch_qs'])]
        #     w.Element(self.K).Update(values=epochs, num_rows=len(epochs))
        # else :
        #     w.Element(self.K).Update(values=[], num_rows=0)
        #
        # w.write_event_value('Draw', 'Draw the initial plot')


    def build(self):

        dicts = {}
        dicts = {self.raw_key: {},self.proc_key: {}}
        raw_list = [
            [sg.Text(get_disp_name(self.raw_key), **t8_kws),
             graphic_button('burn', f'BUILD_{self.raw_key}', tooltip='Build a dataset from raw files.'),
             graphic_button('remove', f'REMOVE_{self.raw_key}', tooltip='Remove a dataset from the analysis list.'),
             # graphic_button('play', 'Replay', tooltip='Replay/Visualize the dataset.'),
             # graphic_button('box_add', 'Add ref', tooltip='Add the reference experimental dataset to the analysis list.'),
             graphic_button('edit', f'CHANGE_ID_{self.raw_key}', tooltip='Change the dataset ID transiently or permanently.'),
             graphic_button('search_add', key=self.raw_key, initial_folder=paths.SingleRunFolder, change_submits=True,
                            enable_events=True,
                            target=(1, -1), button_type=sg.BUTTON_TYPE_BROWSE_FOLDER,
                            tooltip='Browse to add datasets to the list.\n Either directly select a dataset directory or a parent directory containing multiple datasets.')
             ],

            [sg.Col([[sg.Listbox(values=list(dicts[self.raw_key].keys()), size=(25, 5), key=self.raw_ids_key, enable_events=True)]])]
        ]

        proc_list = [
            [sg.Text(get_disp_name(self.proc_key), **t8_kws),
             # graphic_button('burn', f'BUILD_{self.raw_key}', tooltip='Build a dataset from raw files.'),
             # graphic_button('remove', f'REMOVE_{self.raw_key}', tooltip='Remove a dataset from the analysis list.'),
             # graphic_button('play', 'Replay', tooltip='Replay/Visualize the dataset.'),
             # graphic_button('box_add', 'Add ref', tooltip='Add the reference experimental dataset to the analysis list.'),
             # graphic_button('edit', f'CHANGE_ID_{self.raw_key}',
             #                tooltip='Change the dataset ID transiently or permanently.'),
             # graphic_button('search_add', key=self.raw_key, initial_folder=paths.SingleRunFolder, change_submits=True,
             #                enable_events=True,
             #                target=(1, -1), button_type=sg.BUTTON_TYPE_BROWSE_FOLDER,
             #                tooltip='Browse to add datasets to the list.\n Either directly select a dataset directory or a parent directory containing multiple datasets.')
             ],

            [sg.Col([[sg.Listbox(values=list(dicts[self.proc_key].keys()), size=(25, 5), key=self.proc_ids_key,
                                 enable_events=True)]])]
        ]

        l_group = SelectionList(tab=self, conftype='Group', disp='Data group', actions=['load'])
        # l_group = SelectionList(tab=self, conftype='Group', actions=['load', 'save', 'delete'])
        self.selectionlists = [l_group]


        g = ButtonGraphList(name=self.name, fig_dict={})
        # l = [[sg.Col(data_list, size=col_size(0.2), **col_kws)]]
        l = [[sg.Col([l_group.get_layout()]+raw_list+proc_list, size=col_size(0.2), **col_kws)]]
        # l = [[sg.Col([l_group.get_layout(), [g.get_layout(as_col=True)]], size=col_size(0.2), **col_kws), g.canvas]]
        graph_lists = {g.name : g}
        # l = [[sg.Col(data_list + g.get_layout(as_col=False), size=col_size(0.2), **col_kws), g.canvas]]
        return l, {}, graph_lists, dicts


    def eval(self, e, v, w, c, d, g):
        if e == self.raw_key and self.datagroup_id(v)!='':
            dr = v[self.raw_key]
            if dr != '':
                ids=detect_dataset(self.datagroup_id(v), dr)
                if len(ids)>0 :
                    for id in ids :
                        d[self.raw_key][id] = dr
                    w.Element(self.raw_ids_key).Update(values=list(d[self.raw_key].keys()))
        elif e == f'REMOVE_{self.raw_key}':
            if len(v[self.raw_ids_key]) > 0:
                d[self.raw_key].pop(v[self.raw_ids_key][0], None)
                w.Element(self.raw_ids_key).Update(values=list(d[self.raw_key].keys()))
        elif e == f'CHANGE_ID_{self.raw_key}':
            d[self.raw_key] = self.change_dataset_id(w, v, d[self.raw_key])
        elif e == f'BUILD_{self.raw_key}':
            for id,dir in d[self.raw_key].items() :
                fdir=fun.remove_prefix(dir, f'{self.raw_folder}/')
                # print(dir)
                # print(self.raw_folder)
                # print(fdir)
                # fdir.removeprefix(f'{self.raw_folder}/')
                # fdir=dir.replace(f'{self.raw_folder}/', '')
                dd = build_datasets(datagroup_id=self.datagroup_id(v), folders=[id], ids=[id],names=[fdir], raw_folders=[f'{fdir}/{id}'])[0]
                d[self.proc_key][dd.id]=dd
                w.Element(self.proc_ids_key).Update(values=list(d[self.proc_key].keys()))
        return d, g

if __name__ == "__main__":
    from lib.gui.gui import LarvaworldGui
    larvaworld_gui = LarvaworldGui(tabs=['groups'])
    larvaworld_gui.run()