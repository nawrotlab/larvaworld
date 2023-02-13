import os


from larvaworld.lib import reg
from larvaworld.gui import gui_aux

class EssayTab(gui_aux.GuiTab):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.essay_exps_key = 'Essay_experiments'
        self.exp_figures_key = 'Essay_exp_figures'

    def build(self):
        kws={'list_size' : (25,15), 'canvas_size' : gui_aux.col_size(x_frac=0.5, y_frac=0.4), 'tab' : self}
        s1 = gui_aux.PadDict('Essay', disp_name='Configuration', background_color='orange', text_kws=gui_aux.t_kws(10),
                             header_width=25)
        sl1 = gui_aux.SelectionList(tab=self, buttons=['load', 'save', 'delete', 'exec'])
        dl1 = gui_aux.DataList(self.essay_exps_key, tab=self, buttons=['exec'], select_mode=None, size=(24, 10))
        g1 = gui_aux.GraphList(self.name, list_header='Simulated', **kws)
        g2 = gui_aux.ButtonGraphList(self.exp_figures_key, list_header='Observed', fig_dict={}, **kws,
                                     buttons=['browse_figs'], button_args={'browse_figs': {'target': (2, -1)}}
                                     )
        l = gui_aux.gui_cols(cols=[[sl1, s1, dl1], [g1.canvas, g2.canvas], [g1, g2]], x_fracs=[0.2, 0.55, 0.25],
                             as_pane=True, pad=(20,10))
        return l, s1.get_subdicts(), {g1.name: g1, g2.name: g2}, {self.name: {'fig_dict': {}}}

    def run(self, v, w, c, d, g, conf, id):
        conf = reg.loadConf(id=id, conftype=self.conftype)
        for essay_exp in list(conf['experiments'].keys()):
            d, g = self.run_essay_exp(v, w, c, d, g, essay_exp)
        return d, g

    def update(self, w, c, conf, id):
        self.datalists[self.essay_exps_key].dict = conf['experiments']
        self.datalists[self.essay_exps_key].update_window(w)
        essay = reg.get_null('Essay', essay_ID=f'{id}_{reg.next_idx(id=id, conftype="Essay")}', path=f'essays/{id}')
        c['Essay'].update(w, essay)
        fdir = conf['exp_fig_folder']
        temp = {f.split('.')[0]: f'{fdir}/{f}' for f in os.listdir(fdir)}
        temp = dict(sorted(temp.items()))
        self.gui.graph_lists[self.exp_figures_key].update(w, temp)

    def get(self, w, v, c, as_entry=True):
        conf = {
            # 'exp_types' :
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
        from larvaworld.lib.reg.stored.essay_conf import RvsS_Essay, DoublePatch_Essay, Chemotaxis_Essay
        type = self.current_ID(v)
        pars = c['Essay'].get_dict(v, w)
        kws = {
            'essay_id': essay_exp,
            # 'path': f"{pars['path']}/{pars['essay_ID']}",
            # 'vis_kwargs': self.gui.get_vis_kwargs(v),
            # 'exp_types': essay['exp_types'],
            # 'durations': essay['durations'],
            # 'N': pars['N'],
            **pars
        }
        if type == 'RvsS':
            essay=RvsS_Essay(**kws)
        elif type == 'DoublePatch':
            essay=DoublePatch_Essay(**kws)
        elif type == 'Chemotaxis':
            essay=Chemotaxis_Essay(**kws)
        else :
            raise ValueError (f"Essay {type} not implemented")

        ds0 = essay.run()
        essay.anal()
        self.base_dict[essay_exp] = {'exp_fig_dict': essay.figs, 'results': essay.results}
        self.base_dict['fig_dict'].update(essay.figs)
        self.graph_list.update(w, self.base_dict['fig_dict'])
        return d, g


if __name__ == "__main__":
    from larvaworld.gui.tabs.larvaworld_gui import LarvaworldGui
    larvaworld_gui = LarvaworldGui(tabs=['essay'])
    larvaworld_gui.run()
