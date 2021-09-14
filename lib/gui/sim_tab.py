import copy

import PySimpleGUI as sg
import numpy as np

import lib.conf.dtype_dicts as dtypes

from lib.aux.collecting import output_keys
from lib.gui.gui_lib import CollapsibleDict, GraphList, SelectionList
from lib.gui.aux import t_kws, gui_col
from lib.gui.tab import GuiTab
from lib.sim.single_run import run_sim
from lib.sim.analysis import sim_analysis
from lib.conf.conf import next_idx


class SimTab(GuiTab):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self):
        sl1 = SelectionList(tab=self, conftype='Env', idx=1)
        sl2 = SelectionList(tab=self, conftype='Life', idx=1, with_dict=True, header_value='default',
                           text_kws=t_kws(14), value_kws=t_kws(10), width=12, header_text_kws=t_kws(9))
        sl3 = SelectionList(tab=self, actions=['load', 'save', 'delete', 'run'], progress=True,
                              sublists={'env_params': sl1, 'life_params' : sl2})
        c1 = CollapsibleDict('sim_params', True, default=True, disp_name='Configuration', text_kws=t_kws(8))
        output_dict = dict(zip(output_keys, [False] * len(output_keys)))
        c2 = CollapsibleDict('Output', False, dict=output_dict, auto_open=False)

        g1 = GraphList(self.name)

        l = [[
            gui_col([sl3, sl1,c1, c2, sl2], 0.25),
            gui_col([g1.canvas], 0.55),
            gui_col([g1], 0.2)
        ]]

        c = {}
        for i in [c1, c2, sl2]:
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
                # 'life_params': c['sim_params'].get_dict(v, w),
                'collections': [k for k in output_keys if c['Output'].get_dict(v, w)[k]],
                'enrichment': self.current_conf(v)['enrichment'],
                }
        return conf


if __name__ == "__main__":
    from lib.gui.gui import LarvaworldGui
    larvaworld_gui = LarvaworldGui(tabs=['simulation'])
    larvaworld_gui.run()
