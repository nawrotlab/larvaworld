import os
import PySimpleGUI as sg
import numpy as np
# from tkinter import *
from PySimpleGUI import LISTBOX_SELECT_MODE_SINGLE, LISTBOX_SELECT_MODE_MULTIPLE, LISTBOX_SELECT_MODE_BROWSE, \
    LISTBOX_SELECT_MODE_EXTENDED

from lib.gui.gui_lib import t8_kws, ButtonGraphList, b6_kws, graphic_button, col_size, col_kws, get_disp_name, \
    change_dataset_id, named_list_layout, build_datasets_window
from lib.gui.tab import GuiTab, SelectionList
from lib.stor import paths
import lib.conf.dtype_dicts as dtypes
from lib.stor.larva_dataset import LarvaDataset
from lib.stor.managing import detect_dataset, build_datasets, enrich_datasets
import lib.aux.functions as fun


class PreprocessTab(GuiTab):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.raw_key = 'raw_data'
        self.proc_key = 'processed_data'
        self.raw_ids_key = f'{self.raw_key}_IDS'
        self.proc_ids_key = f'{self.proc_key}_IDS'
        self.raw_folder = None
        self.proc_folder = None

    # def change_dataset_id(self,w, v, data):
    #     k = 'NEW_ID'
    #     kR0=self.raw_ids_key
    #     if len(v[kR0]) > 0:
    #         old_id = v[kR0][0]
    #         l = [[sg.Text('Enter new dataset ID', size=(20, 1)), sg.In(k=k, size=(10, 1))],
    #              [sg.Button('Store'), sg.Ok(), sg.Cancel()]]
    #         e, v2 = sg.Window('Change dataset ID', l).read(close=True)
    #         if e == 'Ok':
    #             data[v2[k]] = data.pop(old_id)
    #             w.Element(kR0).Update(values=list(data.keys()))
    #         elif e == 'Store':
    #             d = data[old_id]
    #             d.set_id(v2[k])
    #             data[v2[k]] = data.pop(old_id)
    #             w.Element(kR0).Update(values=list(data.keys()))
    #     return data

    def update(self, w, c, conf, id=None):
        p = conf['path']
        pp = os.path.normpath(f'{paths.DataFolder}/{p}')
        w[self.raw_key].InitialFolder = f'{pp}/raw'
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
        kR, kP = self.raw_key, self.proc_key
        dicts = {kR: {}, kP: {}}

        raw_bs = [graphic_button('burn', f'BUILD {kR}', tooltip='Build a dataset from raw files.'),
                  graphic_button('remove', f'REMOVE {kR}', tooltip='Remove a dataset from the analysis list.'),
                  graphic_button('search_add', key=kR, initial_folder=paths.SingleRunFolder, change_submits=True,
                                 enable_events=True,
                                 # size=(200,500),
                                 # auto_size_button=True,
                                 target=(1, -1), button_type=sg.BUTTON_TYPE_BROWSE_FOLDER,
                                 tooltip='Browse to add datasets to the list.\n Either directly select a dataset directory or a parent directory containing multiple datasets.')
                  ]

        raw_list = named_list_layout(get_disp_name(kR), self.raw_ids_key, list(dicts[kR].keys()),
                                     drop_down=False, list_width=25, list_height=5,
                                     single_line=False, next_to_header=raw_bs, as_col=False,
                                     list_kws={'select_mode': LISTBOX_SELECT_MODE_EXTENDED})

        proc_bs = [
            graphic_button('play', f'REPLAY {kP}', tooltip='Replay/Visualize the dataset.'),
            graphic_button('remove', f'REMOVE {kP}', tooltip='Remove a dataset from the analysis list.'),
                   # graphic_button('play', 'Replay', tooltip='Replay/Visualize the dataset.'),
                   graphic_button('data_add', f'ENRICH {kP}', tooltip='Enrich the dataset.'),
                   graphic_button('edit', f'CHANGE_ID {kP}',
                                  tooltip='Change the dataset ID transiently or permanently.'),
                   graphic_button('search_add', key=kP, initial_folder=paths.SingleRunFolder, change_submits=True,
                                  enable_events=True,
                                  target=(3, -1), button_type=sg.BUTTON_TYPE_BROWSE_FOLDER,
                                  tooltip='Browse to add datasets to the list.\n Either directly select a dataset directory or a parent directory containing multiple datasets.')
                   ]

        proc_list = named_list_layout(get_disp_name(kP), self.proc_ids_key, list(dicts[kP].keys()),
                                      drop_down=False, list_width=25, list_height=5,
                                      single_line=False, next_to_header=proc_bs, as_col=False,
                                     list_kws={'select_mode': LISTBOX_SELECT_MODE_EXTENDED})

        l_group = SelectionList(tab=self, disp='Data group', actions=['load'])
        self.selectionlists = {sl.conftype: sl for sl in [l_group]}

        g = ButtonGraphList(name=self.name, fig_dict={})
        l = [[sg.Col([l_group.get_layout()] + raw_list + proc_list, size=col_size(0.2), **col_kws)]]
        graph_lists = {g.name: g}
        return l, {}, graph_lists, dicts

    def eval(self, e, v, w, c, d, g):
        id0 = self.current_ID(v)
        kR, kP = self.raw_key, self.proc_key
        fR = self.raw_folder
        if e in [kR, kP] and id0 != '':
            k = e
            k0 = f'{k}_IDS'
            dr0 = v[k]
            if dr0 != '':
                ids, dirs = detect_dataset(id0, dr0)
                if len(ids) > 0:
                    for id, dr in zip(ids, dirs):
                        if e == kR:
                            d[k][id] = dr
                        elif e == kP:
                            dd=LarvaDataset(dir=dr)
                            d[k][dd.id] = dd
                    w.Element(k0).Update(values=list(d[k].keys()))
        elif e.startswith('REMOVE'):
            k = e.split()[-1]
            k0 = f'{k}_IDS'
            ids=v[k0]
            for i in range(len(ids)) :
                d[k].pop(ids[i], None)
            w.Element(k0).Update(values=list(d[k].keys()))
        elif e.startswith('CHANGE_ID'):
            k = e.split()[-1]
            d[k] = change_dataset_id(w, v, d[k], k0=f'{k}_IDS')
        elif e == f'BUILD {kR}':
            raw_dic={id:dir for id, dir in d[kR].items() if id in v[f'{kR}_IDS']}
            proc_dir = build_datasets_window(datagroup_id=id0, raw_folder=fR, raw_dic=raw_dic)
            d[kP].update(proc_dir)
            w.Element(self.proc_ids_key).Update(values=list(d[kP].keys()))
        elif e == f'ENRICH {kP}':
            # print(d[kP])
            for id, dd in d[kP].items():
                if id in v[f'{kP}_IDS']:

                    dd = enrich_datasets(datagroup_id=id0, datasets=[dd])[0]
                    # print(dd)
                    d[kP][id] = dd
            # print(d[kP])
        elif e == f'REPLAY {kP}':
            ids = v[f'{kP}_IDS']
            if len(ids) > 0:
                dd = d[kP][ids[0]]
                dd.visualize(vis_kwargs=self.gui.get_vis_kwargs(v, mode='video'), **self.gui.get_replay_kwargs(v))
        return d, g


if __name__ == "__main__":
    from lib.gui.gui import LarvaworldGui

    larvaworld_gui = LarvaworldGui(tabs=['groups'])
    larvaworld_gui.run()
