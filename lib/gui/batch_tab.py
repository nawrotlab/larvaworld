import copy
import threading
import PySimpleGUI as sg

from lib.anal.combining import render_mpl_table
from lib.gui.gui_lib import CollapsibleDict, Collapsible, GraphList, CollapsibleTable, SelectionList, DataList
from lib.gui.aux import t_kws, gui_col
from lib.gui.buttons import named_bool_button
from lib.conf.conf import loadConf, next_idx
import lib.conf.dtype_dicts as dtypes
from lib.gui.tab import GuiTab
from lib.sim.batch_lib import existing_trajs, finfunc_dict, load_traj, prepare_batch, batch_run, delete_traj, \
    existing_trajs_dict


class BatchTab(GuiTab):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.batch_id_key=f'{self.name}_id'
        self.batch_path_key=f'{self.name}_path'
        self.batch_trajs_key=f'{self.name}_IDS'

    def update(self, w, c, conf, id):
        w.Element(self.batch_id_key).Update(value=f'{id}_{next_idx(id, type="batch")}')
        w.Element(self.batch_path_key).Update(value=id)
        for n in ['batch_methods', 'optimization', 'space_search']:
            c[n].update(w, conf[n])
        w['TOGGLE_save_data_flag'].set_state(state=conf['exp_kws']['save_data_flag'])

        w.Element(self.batch_trajs_key).Update(values=existing_trajs(id))
        self.datalist.dict=existing_trajs_dict(id)
        self.datalist.update_window(w)

    def get(self, w, v, c, **kwargs):
        try :
            enrichment=self.current_conf(v)['exp_kws']['enrichment']
        except :
            enrichment=loadConf(v[self.selectionlists['Exp'].k], 'Exp')['enrichment']
        conf = {
            **{n: c[n].get_dict(v, w) for n in ['batch_methods', 'optimization', 'space_search']},
            'exp_kws': {
                'save_data_flag': w['TOGGLE_save_data_flag'].metadata.state,
                'enrichment': enrichment,
                # 'enrichment': self.current_conf(v)['exp_kws']['enrichment'],
                           }
        }
        return copy.deepcopy(conf)

    def build(self):
        sl1 = SelectionList(tab=self, conftype='Exp', idx=1)
        sl2 = SelectionList(tab=self, actions=['load', 'save', 'delete', 'run'],
                                sublists={'exp': sl1})
        batch_conf = [[sg.Text('Batch id:', **t_kws(8)), sg.In('unnamed_batch_0', key=self.batch_id_key, **t_kws(16))],
                      [sg.Text('Path:', **t_kws(8)), sg.In('unnamed_batch', key=self.batch_path_key, **t_kws(16))],
                      named_bool_button('Save data', False, toggle_name='save_data_flag'),
                      ]
        s0 = Collapsible(f'{self.name}_CONFIGURATION', True, batch_conf, disp_name='Configuration')
        s1 = CollapsibleDict('batch_methods', False, default=True)

        s2 = CollapsibleDict('optimization', False, default=True,
                             toggle=True, disabled=True, toggled_subsections=None)
        s3 = CollapsibleTable('space_search', False, headings=['pars', 'ranges', 'Ngrid'], dict={},
                              type_dict=dtypes.get_dict_dtypes('space_search'))
        g1 = GraphList(self.name)

        dl1 = DataList(name=self.name, tab=self, buttons=['select_all', 'remove'])

        l = [[
            gui_col([sl2, sl1, s0, s1, s2, s3, dl1], 0.2),
            gui_col([g1.canvas], 0.6),
            gui_col([g1], 0.2)
        ]]

        c = {}
        for s in [s0, s1, s2, s3]:
            c.update(s.get_subdicts())
        g = {g1.name: g1}
        d ={self.name:{'df' : None, 'fig_dict' : None}}
        return l, c, g, d

    def run(self, v, w, c, d, g, conf, id):
        batch_id=v[self.batch_id_key]
        batch_kwargs = prepare_batch(conf, batch_id, id)
        # print(batch_kwargs['space'])
        # print(batch_kwargs['optimization'])
        # raise
        # thread = threading.Thread(target=batch_run, kwargs=batch_kwargs, daemon=True)
        # thread.start()

        df, fig_dict = batch_run(**batch_kwargs)
        self.draw(df, fig_dict,w)
        w.Element(self.batch_trajs_key).Update(values=existing_trajs(id))

        self.datalist.dict = existing_trajs_dict(id)
        self.datalist.update_window(w)
        return d, g

    def eval(self, e, v, w, c, d, g):
        id0=self.current_ID(v)
        k_trajs=self.batch_trajs_key
        trajs=v[k_trajs]
        if len(trajs) > 0:
            traj0 = trajs[0]
            if e==k_trajs :
                w.Element(self.batch_id_key).Update(value=traj0)
                traj=load_traj(id0, traj0)
                func=finfunc_dict[c['batch_methods'].get_dict(v, w)['final']]
                df, fig_dict = func(traj)
                self.draw(df, fig_dict,w)
            elif e==f'REMOVE {self.name}' :
                delete_traj(id0, traj0)
                w.Element(k_trajs).Update(values=existing_trajs(id0))

                self.datalist.dict = existing_trajs_dict(id0)
                self.datalist.update_window(w)



    def draw(self, df, fig_dict,w):
        df_ax, df_fig = render_mpl_table(df)
        fig_dict['dataframe'] = df_fig
        self.base_dict['df'] = df
        self.base_dict['fig_dict'] = fig_dict
        self.graph_list.update(w, fig_dict)


if __name__ == "__main__":
    from lib.gui.gui import LarvaworldGui
    larvaworld_gui = LarvaworldGui(tabs=['batch-run'])
    larvaworld_gui.run()
