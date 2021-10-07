import os

from lib.conf.dtypes import null_dict

from lib.gui.aux.elements import CollapsibleDict, GraphList, SelectionList, DataList, ButtonGraphList
from lib.gui.aux.functions import gui_cols
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
        s1 = CollapsibleDict('essay_params', disp_name='Configuration')
        sl1 = SelectionList(tab=self, buttons=['load', 'save', 'delete', 'run'])
        dl1 = DataList(self.essay_exps_key, tab=self, buttons=['run'], select_mode=None)
        g1 = GraphList(self.name, tab=self, list_header='Simulated', canvas_size=self.canvas_size)
        g2 = ButtonGraphList(self.exp_figures_key, tab=self, list_header='Observed',
                             canvas_size=self.canvas_size, fig_dict={},
                             buttons=['browse_figs'],button_args={'browse_figs': {'target': (2, -1)}}
                             )

        l = gui_cols(cols=[[sl1, s1, dl1], [g1.canvas, g2.canvas], [g1, g2]], x_fracs=[0.2, 0.6, 0.2])

        return l, s1.get_subdicts(), {g1.name: g1, g2.name: g2}, {self.name: {'fig_dict': {}}}

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

        essay = null_dict('essay_params', essay_ID=f'{id}_{next_idx(id)}', path=f'essays/{id}')
        c['essay_params'].update(w, essay)

        fdir = conf['exp_fig_folder']

        temp = {f.split('.')[0]: f'{fdir}/{f}' for f in os.listdir(fdir)}
        temp = dict(sorted(temp.items()))
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
        k = self.essay_exps_key
        k0 = self.datalists[k].list_key
        if e == f'RUN {k}':
            essay_exp = v[k0][0]
            if essay_exp not in [None, '']:
                d, g = self.run_essay_exp(v, w, c, d, g, essay_exp)
        return d, g

    def run_essay_exp(self, v, w, c, d, g, essay_exp):
        pars = c['essay_params'].get_dict(v, w)
        essay_type = self.current_ID(v)
        essay = loadConf(essay_type, self.conftype)['experiments'][essay_exp]
        kws = {
            'id': essay_exp,
            'path': f"{pars['path']}/{pars['essay_ID']}",
            'vis_kwargs': self.gui.get_vis_kwargs(v),
            'exp_types': essay['exp_types'],
            'durations': essay['durations'],
            # 'N': pars['N'],
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
