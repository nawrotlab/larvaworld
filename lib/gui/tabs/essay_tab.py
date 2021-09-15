import os

import lib.conf.dtype_dicts as dtypes

from lib.gui.aux.elements import CollapsibleDict, GraphList, SelectionList, DataList
from lib.gui.aux.functions import t_kws, gui_col
from lib.gui.tabs.tab import GuiTab
from lib.sim.single_run import run_essay
from lib.sim.analysis import essay_analysis
from lib.conf.conf import loadConf, next_idx


class EssayTab(GuiTab):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.essay_exps_key = 'Essay_experiments'
        self.exp_figures_key = 'Essay_exp_figures'
        self.canvas_size = (1000, 500)

    def build(self):
        s1 = CollapsibleDict('essay_params', default=True, disp_name='Configuration', text_kws=t_kws(8))
        sl1 = SelectionList(tab=self, actions=['load', 'save', 'delete', 'run'])

        dl1 = DataList(name=self.essay_exps_key, tab=self, buttons=['run'], select_mode=None)

        g1 = GraphList(self.name, list_header='Simulated', canvas_size=self.canvas_size)
        g2 = GraphList(self.exp_figures_key, list_header='Observed', canvas_size=self.canvas_size,fig_dict={})

        l = [[
            gui_col([sl1, s1, dl1], 0.2),
            gui_col([g1.canvas, g2.canvas], 0.6),
            gui_col([g1, g2], 0.2),
              # gg
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
        # exps = list(conf['experiments'].keys())
        self.datalists[self.essay_exps_key].dict = conf['experiments']
        self.datalists[self.essay_exps_key].update_window(w)
        # w.Element(self.essay_exps_key).Update(values=exps)

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
        k=self.essay_exps_key
        k0=self.datalists[k].list_key
        if e == f'RUN {k}':
            essay_exp = v[k0][0]
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
    from lib.gui.tabs.gui import LarvaworldGui

    larvaworld_gui = LarvaworldGui(tabs=['essay'])
    larvaworld_gui.run()
