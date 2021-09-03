import copy

import PySimpleGUI as sg
import numpy as np

import lib.conf.dtype_dicts as dtypes

from lib.aux.collecting import output_keys
from lib.gui.gui_lib import CollapsibleDict, GraphList, col_kws, col_size, t_kws
from lib.gui.tab import GuiTab, SelectionList
from lib.sim.single_run import run_sim
from lib.sim.analysis import sim_analysis
from lib.conf.conf import next_idx


class SimTab(GuiTab):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self):
        l_env = SelectionList(tab=self, conftype='Env', idx=1)
        l_life = SelectionList(tab=self, conftype='Life', idx=1, with_dict=True, header_value='default',
                           text_kws=t_kws(14), value_kws=t_kws(10), width=12, header_text_kws=t_kws(9))
        l_sim = SelectionList(tab=self, actions=['load', 'save', 'delete', 'run'], progress=True,
                              sublists={'env_params': l_env, 'life_params' : l_life})
        s1 = CollapsibleDict('sim_params', True, default=True, disp_name='Configuration', text_kws=t_kws(8))
        output_dict = dict(zip(output_keys, [False] * len(output_keys)))
        s2 = CollapsibleDict('Output', False, dict=output_dict, auto_open=False)

        self.selectionlists = {sl.conftype : sl for sl in [l_env, l_sim, l_life]}
        g1 = GraphList(self.name)
        l_conf = [[sg.Col([
            *[i.get_layout() for i in [l_sim, l_env,s1, s2, l_life]],
        ])]]
        l = [[sg.Col(l_conf, **col_kws, size=col_size(0.2)), g1.canvas, sg.Col(g1.get_layout(as_col=False), size=col_size(0.2))]]
        # l = [[sg.Col(l_conf, **col_kws, size=col_size(0.2)), g1.canvas]]

        c = {}
        for i in [s1, s2, l_life]:
            c.update(i.get_subdicts())
        g = {g1.name: g1}
        d={}
        d[self.name]={'datasets' : [], 'fig_dict' : None}
        return l, c, g, d

    def run(self, v, w,c,d,g, conf,id):
        N=conf['sim_params']['duration'] * 60 / conf['sim_params']['timestep']
        p=self.base_list.progressbar
        p.run(w, max=N)
        conf['experiment'] = id
        kws={**conf,
             'vis_kwargs' : self.gui.get_vis_kwargs(v),
             'progress_bar' : w[p.k]
             }
        dd = run_sim(**kws)
        if dd is not None:
            w[p.k_complete].update(visible=True)
            if 'analysis' in d.keys() :
                d['analysis'][dd.id] = dd
            if 'DATASET_IDS' in w.AllKeysDict.keys():
                w.Element('DATASET_IDS').Update(values=list(d['analysis'].keys()))
            self.base_dict['datasets'].append(dd)
            fig_dict, results = sim_analysis(dd, conf['experiment'])
            self.base_dict['fig_dict'] = fig_dict
            self.graph_list.update(w, fig_dict)
        else:
            p.reset(w)
        return d,g

    def update(self, w,  c, conf, id):
        output_dict = dict(zip(output_keys, [True if k in conf['collections'] else False for k in output_keys]))
        c['Output'].update(w, output_dict)
        sim=copy.deepcopy(conf['sim_params'])
        sim.update({'sim_ID' : f'{id}_{next_idx(id)}', 'path' : f'single_runs/{id}'})
        c['sim_params'].update(w, sim)

    def get(self, w, v, c, as_entry=True):
        conf = {
                'sim_params': c['sim_params'].get_dict(v, w),
                'collections': [k for k in output_keys if c['Output'].get_dict(v, w)[k]],
                'enrichment': self.current_conf(v)['enrichment'],
                }
        return conf


if __name__ == "__main__":
    from lib.gui.gui import LarvaworldGui
    larvaworld_gui = LarvaworldGui(tabs=['simulation'])
    larvaworld_gui.run()
