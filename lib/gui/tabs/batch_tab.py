#!/usr/bin/python

import copy
import PySimpleGUI as sg
from lib.gui.tabs.tab import GuiTab
from lib.gui.aux import buttons as gui_but, functions as gui_fun, elements as gui_el


class BatchTab(GuiTab):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.batch_id_key = f'{self.name}_id'
        self.batch_path_key = f'{self.name}_path'
        self.k_stored = f'{self.name}_stored'
        self.k_active = f'{self.name}_active'
        self.k_stored_ids = f'{self.k_stored}_IDS'
        self.k_active_ids = f'{self.k_active}_IDS'
        # self.tree = GuiTreeData('batch_conf')

    @property
    def DL0(self):
        return self.datalists[self.k_active]

    @property
    def DL1(self):
        return self.datalists[self.k_stored]

    def update(self, w, c, conf, id):
        from lib.sim.batch.aux import stored_trajs
        from lib.conf.stored.conf import next_idx
        w.Element(self.batch_id_key).Update(value=f'{id}_{next_idx(id, type="Batch")}')
        for n in ['batch_methods', 'optimization', 'space_search']:
            c[n].update(w, conf[n])
        self.DL1.add(w, stored_trajs(id), replace=True)

    def get(self, w, v, c, **kwargs):
        try:
            enrichment = self.current_conf(v)['exp_kws']['enrichment']
        except:
            from lib.conf.stored.conf import loadConf
            enrichment = loadConf(v[self.selectionlists['Exp'].k], 'Exp')['enrichment']
        from lib.conf.base.dtypes import null_dict
        conf = null_dict('batch_conf',
                         save_hdf5=w['TOGGLE_save_hdf5'].metadata.state,
                         exp_kws={'enrichment': enrichment, 'experiment': self.current_conf(v)['exp_kws']['experiment']},
                         batch_methods=c['batch_methods'].get_dict(v, w),
                         optimization=c['optimization'].get_dict(v, w),
                         space_search=c['space_search'].get_dict(v, w),
                         )
        return copy.deepcopy(conf)

    def build(self):
        kws = {'background_color': 'pink', 'text_kws' : gui_fun.t_kws(10)}
        kA, kS = self.k_active, self.k_stored
        d = {kA: {}, kS: {}}
        sl1 = gui_el.SelectionList(tab=self, conftype='Exp', idx=1)
        sl2 = gui_el.SelectionList(tab=self, buttons=['load', 'save', 'delete', 'run'], sublists={'exp': sl1})
        batch_conf = [[sg.T('Batch id:', **gui_fun.t_kws(8)), sg.In('unnamed_batch_0', k=self.batch_id_key, **gui_fun.t_kws(16))],
                      gui_but.named_bool_button('Save data', False, toggle_name='save_hdf5'),
                      ]
        s0 = gui_el.PadDict(f'{self.name}_CONFIGURATION', content=batch_conf, disp_name='Configuration',dict_name='batch_setup', **kws)
        s1 = gui_el.PadDict('batch_methods',value_kws=gui_fun.t_kws(13), **kws)
        s2 = gui_el.PadDict('optimization', toggle=True, disabled=True, **kws)
        s3 = gui_el.CollapsibleTable('space_search', index='Parameter', heading_dict={'Range': 'range', 'N': 'Ngrid'},
                              dict_name='space_search_par', state=True, col_widths=[12,8,4], num_rows=5)
        g1 = gui_el.GraphList(self.name, tab=self, canvas_size=gui_fun.col_size(0.5,0.8))
        dl1 = gui_el.DataList(kS, dict=d[kS], tab=self, buttons=['select_all', 'remove'], disp='Stored batch-runs', size=(gui_fun.default_list_width, 5))
        dl2 = gui_el.DataList(kA, dict=d[kA], tab=self, buttons=['select_all', 'stop'], disp='Active batch-runs', size=(gui_fun.default_list_width, 5))

        l = gui_fun.gui_cols(cols=[[sl2, sl1, dl2, dl1, g1],[s0, s1, s2, s3], [g1.canvas]], x_fracs=[0.20, 0.22, 0.55], as_pane=True, pad=(20,10))
        c = {}
        for s in [s0, s1, s2, s3]:
            c.update(s.get_subdicts())
        return l, c, {g1.name: g1}, {self.name: {'df': None, 'fig_dict': None}}

    def run(self, v, w, c, d, g, conf, id):
        from run.exec_run import Exec
        batch_id = v[self.batch_id_key]
        conf['batch_id'] = batch_id
        conf['batch_type'] = id
        exec = Exec(mode='batch', conf=conf, run_externally=self.gui.run_externally['batch'])
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
                from lib.sim.batch.functions import retrieve_results
                df, fig_dict = retrieve_results(id0, traj0)
                self.draw(df, fig_dict, w)

        elif e == f'REMOVE {self.k_stored}':
            from lib.sim.batch.aux import delete_traj, stored_trajs
            for stor_id in stor_ids:
                delete_traj(id0, stor_id)
            self.DL1.add(w, stored_trajs(id0), replace=True)

        if e == f'STOP {self.k_active}':
            for act_id in active_ids:
                self.DL0.dict[act_id].terminate()
            self.DL0.remove(w, active_ids)

    def check_subprocesses(self, w):
        complete = []
        for batch_id, ex in self.DL0.dict.items():
            if ex.check():
                from lib.sim.batch.aux import stored_trajs
                df, fig_dict = ex.results
                self.draw(df, fig_dict, w)
                self.DL1.add(w, stored_trajs(ex.type), replace=True)
                complete.append(batch_id)
        self.DL0.remove(w, complete)

    def draw(self, df, fig_dict, w):
        from lib.plot.table import render_mpl_table
        df_fig = render_mpl_table(df)
        fig_dict['dataframe'] = df_fig
        self.base_dict['df'] = df
        self.base_dict['fig_dict'] = fig_dict
        self.graph_list.update(w, fig_dict)


if __name__ == "__main__":
    from lib.gui.tabs.gui import LarvaworldGui
    larvaworld_gui = LarvaworldGui(tabs=['batch-run'])
    larvaworld_gui.run()
