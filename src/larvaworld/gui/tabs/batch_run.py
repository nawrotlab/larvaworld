#!/usr/bin/python

import copy
import PySimpleGUI as sg
import pandas as pd

from larvaworld.lib import reg, aux,sim
import larvaworld.lib.util.data_aux
from larvaworld.gui import gui_aux
from larvaworld.lib.plot.table import mpl_table

class BatchTab(gui_aux.GuiTab):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.batch_id_key = f'{self.name}_id'
        self.batch_path_key = f'{self.name}_path'
        self.k_stored = f'{self.name}_stored'
        self.k_active = f'{self.name}_active'
        self.k_stored_ids = f'{self.k_stored}_IDS'
        self.k_active_ids = f'{self.k_active}_IDS'

    @property
    def DL0(self):
        return self.datalists[self.k_active]

    @property
    def DL1(self):
        return self.datalists[self.k_stored]

    def update(self, w, c, conf, id):
        w.Element(self.batch_id_key).Update(value=f'{id}_{reg.next_idx(id=id, conftype="Batch")}')
        for n in ['optimization', 'space_search']:
            c[n].update(w, conf[n])
        self.DL1.add(w, stored_trajs(id), replace=True)

    def get(self, w, v, c, **kwargs):
        try:
            enrichment = self.current_conf(v)['exp_kws']['enrichment']
        except:
            enrichment = reg.loadConf(id=v[self.selectionlists['Exp'].k],conftype= 'Exp')['enrichment']
        conf = reg.get_null('Batch',
                         exp_kws={'enrichment': enrichment, 'experiment': self.current_conf(v)['exp_kws']['experiment']},
                         optimization=c['optimization'].get_dict(v, w),
                         space_search=c['space_search'].get_dict(v, w),
                         )
        return copy.deepcopy(conf)

    def build(self):
        kws = {'background_color': 'pink', 'text_kws' : gui_aux.t_kws(10)}
        kA, kS = self.k_active, self.k_stored
        d = {kA: {}, kS: {}}
        sl1 = gui_aux.SelectionList(tab=self, conftype='Exp', idx=1)
        sl2 = gui_aux.SelectionList(tab=self, buttons=['load', 'save', 'delete', 'exec'], sublists={'exp': sl1})
        batch_conf = [[sg.T('Batch id:', **gui_aux.t_kws(8)), sg.In('unnamed_batch_0', k=self.batch_id_key, **gui_aux.t_kws(16))],
                      ]
        s0 = gui_aux.PadDict(f'{self.name}_CONFIGURATION', content=batch_conf, disp_name='Configuration', dict_name='batch_setup', **kws)
        s2 = gui_aux.PadDict('optimization', toggle=True, disabled=True, **kws)
        s3 = gui_aux.CollapsibleTable('space_search', index='Parameter', heading_dict={'Range': 'range', 'N': 'Ngrid'},
                                      dict_name='space_search_par', state=True, col_widths=[12,8,4], num_rows=5)
        g1 = gui_aux.GraphList(self.name, tab=self, canvas_size=gui_aux.col_size(0.5, 0.8))
        dl1 = gui_aux.DataList(kS, dict=d[kS], tab=self, buttons=['select_all', 'remove'], disp='Stored batch-runs', size=(gui_aux.w_list, 5))
        dl2 = gui_aux.DataList(kA, dict=d[kA], tab=self, buttons=['select_all', 'stop'], disp='Active batch-runs', size=(gui_aux.w_list, 5))

        l = gui_aux.gui_cols(cols=[[sl2, sl1, dl2, dl1, g1], [s0, s2, s3], [g1.canvas]], x_fracs=[0.20, 0.22, 0.55], as_pane=True, pad=(20, 10))
        c = {}
        for s in [s0, s2, s3]:
            c.update(s.get_subdicts())
        return l, c, {g1.name: g1}, {self.name: {'df': None, 'fig_dict': None}}

    def run(self, v, w, c, d, g, conf, id):
        batch_id = v[self.batch_id_key]
        conf['id'] = batch_id
        conf['batch_type'] = id
        exec = sim.Exec(mode='batch', conf=conf, run_externally=self.gui.run_externally['batch'])
        self.DL0.add(w, {batch_id: exec})
        exec.run()
        return d, g

    def eval(self, e, v, w, c, d, g):
        self.check_subprocesses(w)
        id0 = self.current_ID(v)
        stor_ids = v[self.k_stored_ids]
        active_ids = v[self.k_active_ids]

        if e == self.k_stored_ids:
            if len(stor_ids) > 0:
                traj0 = stor_ids[0]
                w.Element(self.batch_id_key).Update(value=traj0)
                df, fig_dict = aux.retrieve_results(id0, traj0)
                self.draw(df, fig_dict, w)

        elif e == f'REMOVE {self.k_stored}':
            for stor_id in stor_ids:
                aux.delete_traj(id0, stor_id)
            self.DL1.add(w, stored_trajs(id0), replace=True)

        if e == f'STOP {self.k_active}':
            for act_id in active_ids:
                self.DL0.dict[act_id].terminate()
            self.DL0.remove(w, active_ids)

    def check_subprocesses(self, w):
        complete = []
        for batch_id, ex in self.DL0.dict.items():
            if ex.check():
                df, fig_dict = ex.results
                self.draw(df, fig_dict, w)
                self.DL1.add(w, stored_trajs(ex.type), replace=True)
                complete.append(batch_id)
        self.DL0.remove(w, complete)

    def draw(self, df, fig_dict, w):
        fig_dict['dataframe'] = mpl_table(df)
        self.base_dict['df'] = df
        self.base_dict['fig_dict'] = fig_dict
        self.graph_list.update(w, fig_dict)


def stored_trajs(batch_type):
    path = f'{reg.BATCH_DIR}/{batch_type}/{batch_type}.hdf5'
    try:
        store = pd.HDFStore(path, mode='r')
        dic = {k:store for k in store.keys()}
        store.close()
        return dic
    except:
        return {}


if __name__ == "__main__":
    from larvaworld.gui.tabs.larvaworld_gui import LarvaworldGui
    larvaworld_gui = LarvaworldGui(tabs=['batch-exec'])
    larvaworld_gui.run()
