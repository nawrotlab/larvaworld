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
from lib.stor.managing import detect_dataset, build_datasets, enrich_datasets
import lib.aux.functions as fun

class PreprocessTab(GuiTab):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.raw_key='raw_data'
        self.proc_key='processed_data'
        self.raw_ids_key=f'{self.raw_key}_IDS'
        self.proc_ids_key=f'{self.proc_key}_IDS'
        self.raw_folder=None
        self.proc_folder=None

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
        pp=os.path.normpath(f'{paths.DataFolder}/{p}')
        # path=os.path.normpath(f'{paths.DataFolder}/{p}/raw')
        w[self.raw_key].InitialFolder=f'{pp}/raw'
        self.raw_folder = f'{pp}/raw'
        w[self.proc_key].InitialFolder = f'{pp}/processed'
        self.proc_folder = f'{pp}/processed'
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
             graphic_button('remove', f'REMOVE{self.raw_key}', tooltip='Remove a dataset from the analysis list.'),
             # graphic_button('play', 'Replay', tooltip='Replay/Visualize the dataset.'),
             # graphic_button('box_add', 'Add ref', tooltip='Add the reference experimental dataset to the analysis list.'),
             graphic_button('edit', f'CHANGE_ID {self.raw_key}', tooltip='Change the dataset ID transiently or permanently.'),
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
             graphic_button('remove', f'REMOVE {self.proc_key}', tooltip='Remove a dataset from the analysis list.'),
             # graphic_button('play', 'Replay', tooltip='Replay/Visualize the dataset.'),
             graphic_button('data_add', 'Enrich', tooltip='Enrich the dataset.'),
             graphic_button('edit', f'CHANGE_ID {self.proc_key}',
                            tooltip='Change the dataset ID transiently or permanently.'),
             graphic_button('search_add', key=self.proc_key, initial_folder=paths.SingleRunFolder, change_submits=True,
                            enable_events=True,
                            target=(1, -1), button_type=sg.BUTTON_TYPE_BROWSE_FOLDER,
                            tooltip='Browse to add datasets to the list.\n Either directly select a dataset directory or a parent directory containing multiple datasets.')
             ],

            [sg.Col([[sg.Listbox(values=list(dicts[self.proc_key].keys()), size=(25, 5), key=self.proc_ids_key,
                                 enable_events=True)]])]
        ]

        l_group = SelectionList(tab=self, disp='Data group', actions=['load'])
        self.selectionlists = {sl.conftype : sl for sl in [l_group]}


        g = ButtonGraphList(name=self.name, fig_dict={})
        # l = [[sg.Col(data_list, size=col_size(0.2), **col_kws)]]
        l = [[sg.Col([l_group.get_layout()]+raw_list+proc_list, size=col_size(0.2), **col_kws)]]
        # l = [[sg.Col([l_group.get_layout(), [g.get_layout(as_col=True)]], size=col_size(0.2), **col_kws), g.canvas]]
        graph_lists = {g.name : g}
        # l = [[sg.Col(data_list + g.get_layout(as_col=False), size=col_size(0.2), **col_kws), g.canvas]]
        return l, {}, graph_lists, dicts


    def eval(self, e, v, w, c, d, g):
        id0=self.current_ID(v)
        if e in [self.raw_key, self.proc_key] and id0!= '':
            k = e
            k0 = f'{k}_IDS'
            dr = v[k]
            if dr != '':
                ids=detect_dataset(id0, dr)
                if len(ids)>0 :
                    for id in ids :
                        d[k][id] = dr
                    w.Element(k0).Update(values=list(d[k].keys()))
        elif e.startswith('REMOVE'):
            k = e.split()[-1]
            k0=f'{k}_IDS'
            if len(v[k0]) > 0:
                d[k].pop(v[k0][0], None)
                w.Element(k0).Update(values=list(d[k].keys()))
        elif e.startswith('CHANGE_ID'):
            k=e.split()[-1]
            d[k] = self.change_dataset_id(w, v, d[k])
        elif e == f'BUILD_{self.raw_key}':
            for id,dir in d[self.raw_key].items() :
                fdir=fun.remove_prefix(dir, f'{self.raw_folder}/')
                raw_folders = [fdir]
                dd = build_datasets(datagroup_id=id0, folders=None, ids=[id], names=[fdir], raw_folders=raw_folders)[0]
                d[self.proc_key][dd.id]=dd
                w.Element(self.proc_ids_key).Update(values=list(d[self.proc_key].keys()))
        elif e == 'Enrich':
            for id,dir in d[self.proc_key].items() :
                fdir=fun.remove_prefix(dir, f'{self.raw_folder}/')
                dd = enrich_datasets(datagroup_id=id0, names=[fdir])[0]
                d[self.proc_key][id]=dd
        return d, g

if __name__ == "__main__":
    from lib.gui.gui import LarvaworldGui
    larvaworld_gui = LarvaworldGui(tabs=['groups'])
    larvaworld_gui.run()