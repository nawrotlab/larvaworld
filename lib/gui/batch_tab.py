import copy
import threading
import PySimpleGUI as sg

from lib.anal.combining import render_mpl_table
from lib.gui.gui_lib import CollapsibleDict, Collapsible, named_bool_button, GraphList, t10_kws, t18_kws, \
    CollapsibleTable, col_kws, col_size, named_list_layout, graphic_button, t16_kws, t8_kws
from lib.conf.conf import loadConf, next_idx
import lib.conf.dtype_dicts as dtypes
from lib.gui.tab import GuiTab, SelectionList
from lib.sim.batch_lib import existing_trajs, finfunc_dict, load_traj, prepare_batch, batch_run, delete_traj


class BatchTab(GuiTab):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.batch_id_key=f'{self.name}_id'
        self.batch_path_key=f'{self.name}_path'
        self.batch_trajs_key=f'{self.name}_trajs'

    @ property
    def batch_type_key(self):
        return self.selectionlists[0].k

    def update(self, w, c, conf, id):
        w.Element(self.batch_id_key).Update(value=f'{id}_{next_idx(id, type="batch")}')
        w.Element(self.batch_path_key).Update(value=id)
        for n in ['batch_methods', 'optimization', 'space_search']:
            c[n].update(w, conf[n])
        w['TOGGLE_save_data_flag'].set_state(state=conf['exp_kws']['save_data_flag'])

        w.Element(self.batch_trajs_key).Update(values=existing_trajs(id))

    def get(self, w, v, c, **kwargs):
        conf = {
            **{n: c[n].get_dict(v, w) for n in ['batch_methods', 'optimization', 'space_search']},
            'exp_kws': {
                'save_data_flag': w['TOGGLE_save_data_flag'].metadata.state,
                'enrichment': loadConf(v[self.batch_type_key], 'Batch')['exp_kws']['enrichment'],
                           }
        }
        return copy.deepcopy(conf)

    def build(self):
        l_sim = SelectionList(tab=self, conftype='Exp', idx=1)
        l_batch = SelectionList(tab=self, conftype='Batch', actions=['load', 'save', 'delete', 'run'],
                                sublists={'exp': l_sim})
        self.selectionlists = [l_batch, l_sim]
        batch_conf = [[sg.Text('Batch id:', **t8_kws), sg.In('unnamed_batch_0', key=self.batch_id_key, **t16_kws)],
                      [sg.Text('Path:', **t8_kws), sg.In('unnamed_batch', key=self.batch_path_key, **t16_kws)],
                      named_bool_button('Save data', False, toggle_name='save_data_flag'),
                      ]
        s0 = Collapsible(f'{self.name}_CONFIGURATION', True, batch_conf, disp_name='Configuration')
        s1 = CollapsibleDict('batch_methods', False, default=True)

        s2 = CollapsibleDict('optimization', False, default=True,
                             toggle=True, disabled=True, toggled_subsections=None)
        s3 = CollapsibleTable('space_search', False, headings=['pars', 'ranges', 'Ngrid'], dict={},
                              type_dict=dtypes.get_dict_dtypes('space_search'))
        g1 = GraphList(self.name)

        traj_l = named_list_layout(f'{self.name.capitalize()}s', key=self.batch_trajs_key, choices=[],
                                   default_value=None, drop_down=False, list_width=24,
                                   readonly=True, enable_events=True, single_line=False,
                                   next_to_header=[graphic_button('remove', 'REMOVE_traj', tooltip='Remove a batch-run trajectory.')])

        l_batch0 = sg.Col([l_batch.l,
                           l_sim.l,
                           *[s.get_layout() for s in [s0, s1, s2, s3]],
                           # [g1.get_layout()],
                           [traj_l],
                           ], **col_kws, size=col_size(0.2))

        l = [[l_batch0, g1.canvas,sg.Col(g1.get_layout(as_col=False), size=col_size(0.2))]]
        # l = [[l_batch0, g1.canvas]]

        c = {}
        for s in [s0, s1, s2, s3]:
            c.update(s.get_subdicts())
        g = {g1.name: g1}
        d ={'batch_results':{'df' : None, 'fig_dict' : None}}
        return l, c, g, d

    def run(self, v, w, c, d, g, conf, id):

        batch_id=v[self.batch_id_key]
        batch_kwargs = prepare_batch(conf, batch_id, id)

        # thread = threading.Thread(target=batch_run, kwargs=batch_kwargs, daemon=True)
        # thread.start()

        df, fig_dict = batch_run(**batch_kwargs)
        self.draw(df, fig_dict,w,d,g)
        w.Element(self.batch_trajs_key).Update(values=existing_trajs(id))
        return d, g

    def eval(self, e, v, w, c, d, g):
        if e==self.batch_trajs_key :
            traj_name=v[self.batch_trajs_key][0]
            w.Element(self.batch_id_key).Update(value=traj_name)
            traj=load_traj(v[self.batch_type_key], traj_name)
            func=finfunc_dict[c['batch_methods'].get_dict(v, w)['final']]
            df, fig_dict = func(traj)
            self.draw(df, fig_dict,w,d,g)
        elif e=='REMOVE_traj' :
            if len(v[self.batch_trajs_key])>0 :
                traj_name = v[self.batch_trajs_key][0]
                delete_traj(v[self.batch_type_key], traj_name)
                w.Element(self.batch_trajs_key).Update(values=existing_trajs(v[self.batch_type_key]))


    def draw(self, df, fig_dict,w,d,g):
        df_ax, df_fig = render_mpl_table(df)
        fig_dict['dataframe'] = df_fig
        d['batch_results']['df'] = df
        d['batch_results']['fig_dict'] = fig_dict
        g[self.name].update(w, d['batch_results']['fig_dict'])


if __name__ == "__main__":
    from lib.gui.gui import LarvaworldGui
    larvaworld_gui = LarvaworldGui(tabs=['batch-run'])
    larvaworld_gui.run()
