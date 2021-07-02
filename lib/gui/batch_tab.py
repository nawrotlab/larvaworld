import copy

import threading

import PySimpleGUI as sg

from lib.anal.combining import render_mpl_table

from lib.gui.gui_lib import CollapsibleDict, Collapsible, named_bool_button, GraphList, b12_kws, b_kws, \
    graphic_button, t10_kws, t12_kws, t18_kws, t8_kws, t6_kws, CollapsibleTable, w_kws, col_kws, \
    col_size, t24_kws
from lib.conf.conf import loadConf, next_idx
import lib.conf.dtype_dicts as dtypes
import lib.sim.single_run as run
from lib.gui.tab import GuiTab, SelectionList


class BatchTab(GuiTab):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update(self, w, c, conf, id):
        w.Element(f'{self.name}_id').Update(value=f'{id}_{next_idx(id, type="batch")}')
        w.Element(f'{self.name}_path').Update(value=id)
        for n in ['batch_methods', 'optimization', 'space_search']:
            c[n].update(w, conf[n])
        w['TOGGLE_save_data_flag'].set_state(state=conf['run_kwargs']['save_data_flag'])
        # for i in range(2) :
        #     k=f'simulation_CONF{i}'
        #     w.Element(k, silent_on_error=True).Update(value=conf['exp'])

    def get(self, w, v, c, **kwargs):
        conf = {
            **{n: c[n].get_dict(v, w) for n in ['batch_methods', 'optimization', 'space_search']},
            'run_kwargs': {
                'save_data_flag': w['TOGGLE_save_data_flag'].metadata.state,
                'enrichment': loadConf(v[self.selectionlists[0].k], 'Batch')['run_kwargs']['enrichment'],
                           }
        }
        return copy.deepcopy(conf)

    def build(self):
        l_sim = SelectionList(tab=self, conftype='Exp', idx=1)
        l_batch = SelectionList(tab=self, conftype='Batch', actions=['load', 'save', 'delete', 'run'],
                                sublists={'exp': l_sim})
        self.selectionlists = [l_batch, l_sim]
        batch_conf = [[sg.Text('Batch id:', **t10_kws), sg.In('unnamed_batch_0', key=f'{self.name}_id', **t18_kws)],
                      [sg.Text('Path:', **t10_kws), sg.In('unnamed_batch', key=f'{self.name}_path', **t18_kws)],
                      named_bool_button('Save data', False, toggle_name='save_data_flag'),
                      ]
        s0 = Collapsible(f'{self.name}_CONFIGURATION', True, batch_conf, disp_name='Configuration')
        s1 = CollapsibleDict('batch_methods', False, default=True)

        s2 = CollapsibleDict('optimization', False, default=True,
                             toggle=True, disabled=True, toggled_subsections=None)
        s3 = CollapsibleTable('space_search', False, headings=['pars', 'ranges', 'Ngrid'], dict={},
                              type_dict=dtypes.get_dict_dtypes('space_search'))
        g1 = GraphList(self.name)
        l_batch0 = sg.Col([l_batch.l,
                           l_sim.l,
                           *[s.get_layout() for s in [s0, s1, s2, s3]],
                           [g1.get_layout()]
                           ], **col_kws, size=col_size(0.3))

        l = [[l_batch0, g1.canvas]]

        c = {}
        for s in [s0, s1, s2, s3]:
            c.update(s.get_subdicts())
        g = {g1.name: g1}
        d ={'batch_results':{'df' : None, 'fig_dict' : None}}
        # print(l)
        return l, c, g, d

    def run(self, v, w, c, d, g, conf, id):
        from lib.sim.batch_lib import prepare_batch, batch_run
        conf['exp']['sim_params']['path'] = id
        batch_kwargs = prepare_batch(conf, v[f'{self.name}_id'])
        # print(list(batch_kwargs.keys()))
        # batch_kwargs = prepare_batch(conf, batch_id, exp_conf)
        # dicts['batch_kwargs']=batch_kwargs
        #
        # thread = threading.Thread(target=batch_run, kwargs=batch_kwargs, daemon=True)
        # thread.start()

        # df, fig_dict = self.fork(batch_run, batch_kwargs)
        df, fig_dict = batch_run(**batch_kwargs)
        df_ax, df_fig = render_mpl_table(df)
        fig_dict['dataframe'] = df_fig
        d['batch_results']['df'] = df
        d['batch_results']['fig_dict'] = fig_dict
        g[self.name].update(w, d['batch_results']['fig_dict'])
        return d, g





if __name__ == "__main__":
    from lib.gui.gui import LarvaworldGui

    larvaworld_gui = LarvaworldGui(tabs=['batch-run'])
    larvaworld_gui.run()
