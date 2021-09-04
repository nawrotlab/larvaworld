import os
import PySimpleGUI as sg
import numpy as np
from PySimpleGUI import LISTBOX_SELECT_MODE_SINGLE, LISTBOX_SELECT_MODE_MULTIPLE, LISTBOX_SELECT_MODE_BROWSE, \
    LISTBOX_SELECT_MODE_EXTENDED

from lib.gui.gui_lib import ButtonGraphList, graphic_button, col_size, col_kws, get_disp_name, \
    change_dataset_id, named_list, import_window, CollapsibleDict, browse_button, remove_button, sel_all_button, \
    changeID_button, replay_button, import_button, enrich_button, DataList
from lib.gui.tab import GuiTab, SelectionList
from lib.stor import paths
import lib.conf.dtype_dicts as dtypes
from lib.stor.datagroup import LarvaDataGroup
from lib.stor.larva_dataset import LarvaDataset
from lib.stor.managing import detect_dataset, build_datasets, enrich_datasets
import lib.aux.functions as fun


class ImportTab(GuiTab):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.raw_key = 'raw_data'
        self.proc_key = 'imported_data'
        self.raw_ids_key = f'{self.raw_key}_IDS'
        self.proc_ids_key = f'{self.proc_key}_IDS'
        self.raw_folder = None
        self.proc_folder = None

    def update(self, w, c, conf, id=None):
        datagroup = LarvaDataGroup(id)
        w[f'BROWSE {self.raw_key}'].InitialFolder = datagroup.raw_dir
        # w[self.raw_key].InitialFolder = datagroup.raw_dir
        self.raw_folder = datagroup.raw_dir
        w[f'BROWSE {self.proc_key}'].InitialFolder = datagroup.proc_dir
        # w[self.proc_key].InitialFolder = datagroup.proc_dir
        self.proc_folder = datagroup.proc_dir
        c['enrichment'].update(w, datagroup.get_conf()['enrich'])

    def build(self):
        kR, kP = self.raw_key, self.proc_key
        dicts = {kR: {}, kP: {}}

        lR = DataList(name=self.raw_key, tab=self, dict=dicts[kR], buttons=['import', 'select_all', 'remove', 'changeID', 'browse'],
                            button_args={'browse' : {'target' : (0, -1)}}, raw=True)
        lP = DataList(name=self.proc_key, tab=self, dict=dicts[kP],
                       buttons=['replay', 'enrich', 'select_all', 'remove', 'changeID', 'browse'],
                       button_args={'browse': {'target': (0, -1)}})
        self.datalists = {dl.name: dl for dl in [lR, lP]}
        lG = SelectionList(tab=self, disp='Data group', actions=['load'])
        self.selectionlists = {sl.conftype: sl for sl in [lG]}



        g = ButtonGraphList(name=self.name, fig_dict={})
        s1 = CollapsibleDict('enrichment', False, default=True, toggled_subsections=None)

        l3=[sg.Col([
            lG.get_layout(),
            lR.get_layout(),
            lP.get_layout()
        ], size=col_size(0.25), **col_kws)]
        l4=s1.get_layout(size=col_size(0.25), **col_kws)

        c = {}
        for s in [s1]:
            c.update(**s.get_subdicts())

        l = [l3 + l4]
        graph_lists = {g.name: g}
        return l, c, graph_lists, dicts

    def eval(self, e, v, w, c, d, g):

        # kR, kP = self.raw_key, self.proc_key
        # fR = self.raw_folder
        # if e == f'BUILD {kR}':
        #     id0 = self.current_ID(v)
        #     raw_dic = {id: dir for id, dir in d[kR].items() if id in v[f'{kR}_IDS']}
        #     proc_dir = import_window(datagroup_id=id0, raw_folder=fR, raw_dic=raw_dic)
        #     d[kP].update(proc_dir)
        #     w.Element(self.proc_ids_key).Update(values=list(d[kP].keys()))
        return d, g


if __name__ == "__main__":
    from lib.gui.gui import LarvaworldGui

    # larvaworld_gui = LarvaworldGui(tabs=['import'])
    larvaworld_gui = LarvaworldGui(tabs=['import', 'analysis'])
    larvaworld_gui.run()
