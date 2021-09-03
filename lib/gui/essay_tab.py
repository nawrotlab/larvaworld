import copy
import os

import PySimpleGUI as sg
import numpy as np

import lib.conf.dtype_dicts as dtypes

from lib.gui.gui_lib import CollapsibleDict, GraphList, graphic_button, col_kws, col_size, named_list_layout, t_kws
from lib.gui.tab import GuiTab, SelectionList
from lib.sim.single_run import run_essay
from lib.sim.analysis import essay_analysis
from lib.conf.conf import loadConf, next_idx
from lib.stor import paths


class EssayTab(GuiTab):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.essay_exps_key = 'Essay_exps'
        self.exp_figures_key = 'Essay_exp_figures'
        self.canvas_size = (1000, 500)

    def build(self):
        s1 = CollapsibleDict('essay_params', True, default=True, disp_name='Configuration', text_kws=t_kws(8))
        l_essay = SelectionList(tab=self, actions=['load', 'save', 'delete', 'run'],
                                # progress=True,
                                # sublists={'env_params': l_env, 'life_params' : l_life}
                                )
        next_to_header = [
            graphic_button('play', f'RUN_{self.essay_exps_key}', tooltip='Run the selected essay experiment.')]
        l_exps = named_list_layout(text='Experiments', key=self.essay_exps_key, choices=[], drop_down=False,
                                   single_line=False, next_to_header=next_to_header)
        self.selectionlists = {sl.conftype : sl for sl in [l_essay]}
        g1 = GraphList(self.name, list_header='Simulated', canvas_size=self.canvas_size)
        g2 = GraphList(self.exp_figures_key, list_header='Observed', canvas_size=self.canvas_size,fig_dict={})
        l_conf = [[sg.Col([
            *[i.get_layout() for i in [l_essay, s1]],
            [l_exps]
        ])]]
        gg = sg.Col([
            [g1.canvas, g1.get_layout()],
            [g2.canvas, g2.get_layout()]
        ],
            size=col_size(0.8), **col_kws)
        l = [[sg.Col(l_conf, **col_kws, size=col_size(0.2)),
              gg
              ]]

        c = {}
        for i in [s1]:
            c.update(i.get_subdicts())
        g = {g1.name: g1, g2.name: g2}
        d = {self.name: {'fig_dict': {}}}
        return l, c, g, d

    def run(self, v, w, c, d, g, conf, id):
        conf = loadConf(id, self.conftype)
        for essay_exp in list(conf['experiments'].keys()):
            d, g = self.run_essay_exp(v, w, c, d, g, essay_exp)
        return d, g

    def update(self, w, c, conf, id):
        exps = list(conf['experiments'].keys())
        w.Element(self.essay_exps_key).Update(values=exps)
        essay = dtypes.get_dict('essay_params', essay_ID=f'{id}_{next_idx(id)}', path=f'essays/{id}')
        c['essay_params'].update(w, essay)

        fdir=conf['exp_fig_folder']

        temp = {f.split('.')[0]: f'{fdir}/{f}' for f in os.listdir(fdir)}
        self.gui.graph_lists[self.exp_figures_key].update(w, temp)

    def get(self, w, v, c, as_entry=True):
        conf = {
            # 'exp_types' :
            # 'essay_params': c['essay_params'].get_dict(v, w),
            # # 'sim_params': sim,
            # 'collections': [k for k in output_keys if c['Output'].get_dict(v, w)[k]],
            # # 'life_params': c['life'].get_dict(v, w),
            # 'enrichment': loadConf(v[self.selectionlists[0].k], 'Exp')['enrichment'],
        }
        return conf

    def eval(self, e, v, w, c, d, g):
        if e == f'RUN_{self.essay_exps_key}':
            essay_exp = v[self.essay_exps_key][0]
            if essay_exp not in [None, '']:
                d, g = self.run_essay_exp(v, w, c, d, g, essay_exp)
        return d, g

    def run_essay_exp(self, v, w, c, d, g, essay_exp):
        pars = c['essay_params'].get_dict(v, w)
        id = pars['essay_ID']
        essay_type = self.current_ID(v)
        essay = loadConf(essay_type, self.conftype)[essay_exp]
        kws = {
            'id': f'{id}_{essay_exp}',
            'path': pars['path'],
            'vis_kwargs': self.gui.get_vis_kwargs(v),
            'exp_types': essay['exp_types'],
            'durations': essay['durations'],
            'N': pars['N'],
        }
        ds0 = run_essay(**kws)
        if ds0 is not None:
            fig_dict, results = essay_analysis(essay_type, essay_exp, ds0)
            self.base_dict[essay_exp] = {'exp_fig_dict': fig_dict, 'results': results}
            self.base_dict['fig_dict'].update(fig_dict)
            self.graph_list.update(w, self.base_dict['fig_dict'])
        return d, g


if __name__ == "__main__":
    from lib.gui.gui import LarvaworldGui

    larvaworld_gui = LarvaworldGui(tabs=['essay'])
    larvaworld_gui.run()
