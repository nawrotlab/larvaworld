import os

from lib.gui.tabs.tab import GuiTab
from lib.gui.aux import functions as gui_fun, elements as gui_el
from lib.registry.pars import preg


class EssayTab(GuiTab):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.essay_exps_key = 'Essay_experiments'
        self.exp_figures_key = 'Essay_exp_figures'

    def build(self):
        kws={'list_size' : (25,15), 'canvas_size' : gui_fun.col_size(x_frac=0.5, y_frac=0.4), 'tab' : self}
        s1 = gui_el.PadDict('essay_params', disp_name='Configuration', background_color='orange', text_kws=gui_fun.t_kws(10),
                     header_width=25)
        sl1 = gui_el.SelectionList(tab=self, buttons=['load', 'save', 'delete', 'run'])
        dl1 = gui_el.DataList(self.essay_exps_key, tab=self, buttons=['run'], select_mode=None, size=(24,10))
        g1 = gui_el.GraphList(self.name, list_header='Simulated', **kws)
        g2 = gui_el.ButtonGraphList(self.exp_figures_key, list_header='Observed',fig_dict={}, **kws,
                             buttons=['browse_figs'],button_args={'browse_figs': {'target': (2, -1)}}
                             )
        l = gui_fun.gui_cols(cols=[[sl1, s1, dl1], [g1.canvas, g2.canvas], [g1, g2]], x_fracs=[0.2, 0.55, 0.25],
                     as_pane=True, pad=(20,10))
        return l, s1.get_subdicts(), {g1.name: g1, g2.name: g2}, {self.name: {'fig_dict': {}}}

    def run(self, v, w, c, d, g, conf, id):
        conf = preg.loadConf(id=id, conftype=self.conftype)
        for essay_exp in list(conf['experiments'].keys()):
            d, g = self.run_essay_exp(v, w, c, d, g, essay_exp)
        return d, g

    def update(self, w, c, conf, id):
        # from lib.registry.dtypes import null_dict
        from lib.conf.stored.conf import next_idx
        self.datalists[self.essay_exps_key].dict = conf['experiments']
        self.datalists[self.essay_exps_key].update_window(w)
        essay = preg.get_null('essay_params', essay_ID=f'{id}_{next_idx(id, "Essay")}', path=f'essays/{id}')
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
        k0 = self.datalists[k].key
        if e == f'RUN {k}':
            essay_exp = v[k0][0]
            if essay_exp not in [None, '']:
                d, g = self.run_essay_exp(v, w, c, d, g, essay_exp)
        return d, g

    def run_essay_exp(self, v, w, c, d, g, essay_exp):
        from lib.sim.single.single_run import run_essay
        from lib.sim.single.analysis import essay_analysis
        pars = c['essay_params'].get_dict(v, w)
        essay_type = self.current_ID(v)
        essay = preg.loadConf(id=essay_type, conftype=self.conftype)['experiments'][essay_exp]
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
