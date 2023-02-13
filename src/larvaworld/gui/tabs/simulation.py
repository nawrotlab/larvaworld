import copy
import PySimpleGUI as sg


from larvaworld.lib import reg, aux, sim

from larvaworld.gui import gui_aux
from larvaworld.gui.tabs import DrawEnvTab, EnvTab

class SimTab(gui_aux.GuiTab):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.canvas_size = gui_aux.col_size(0.5, 0.8)
        self.k_stored = f'{self.name}_stored'
        self.k_active = f'{self.name}_active'

    @property
    def DL0(self):
        return self.datalists[self.k_active]

    @property
    def DL1(self):
        return self.datalists[self.k_stored]

    def build(self):
        l0, l1, c1, g1, d1 = self.build_conf()
        l2, c2, g2, d2 = self.build_RUN()
        self.draw_tab = DrawEnvTab(name='draw', gui=self.gui)
        l3, c3, g3, d3 = self.draw_tab.build()

        tab_kws = {'font': ("Helvetica", 13, "normal"), 'selected_title_color': 'darkblue', 'title_color': 'grey',
                   'tab_background_color': 'lightgrey'}

        ts = [sg.Tab(n, nl, key=f'{n}_SIM TAB') for n, nl in zip(['draw','setup', 'exec'], [l3,l1, l2])]
        l_tabs = sg.TabGroup([ts], key='ACTIVE_SIM_TAB', tab_location='topleft', **tab_kws)
        l = [[l0, l_tabs]]
        return l, {**c1, **c2, **c3}, {**g1, **g2, **g3}, {**d1, **d2, **d3}

    def build_RUN(self):
        kA, kS = self.k_active, self.k_stored
        d = {kA: {}, kS: {}}
        g1 = gui_aux.GraphList(self.name, tab=self, canvas_size=self.canvas_size, list_size = (gui_aux.w_list, 15))
        dl1 = gui_aux.DataList(kA, dict=d[kA], tab=self, buttons=['select_all', 'stop'], disp='Active simulations', size=(gui_aux.w_list, 6))
        dl2 = gui_aux.DataList(kS, dict=d[kS], tab=self, buttons=['select_all', 'remove'], disp='Completed simulations', size=(gui_aux.w_list, 6))
        l = gui_aux.gui_cols(cols=[[g1.canvas], [dl1, dl2, g1]], x_fracs=[0.53, 0.25], as_pane=True, pad=(10, 10))
        return l, {}, {g1.name: g1}, d

    def build_conf(self):
        kws = {'background_color': 'lightgreen'}
        s1 = gui_aux.PadTable('larva_groups', buttons=['add', 'remove'], index='Group ID', col_widths=[10, 3, 7, 10],
                              heading_dict={'N': 'distribution.N', 'color': 'default_color', 'model': 'model'},
                              dict_name='LarvaGroup')
        self.envtab = EnvTab(name='environment', gui=self.gui, conftype='Env')
        tab1_l, tab1_c, tab1_g, tab1_d = self.envtab.build()
        sl1 = self.envtab.selectionlists[self.envtab.conftype]
        self.selectionlists[sl1.conftype] = sl1

        s2 = gui_aux.PadTable('trials', buttons=['add', 'remove'], index='idx', col_widths=[3, 4, 4, 5, 8],
                              heading_dict={'start': 'start', 'stop': 'stop', 'quality': 'substrate.quality',
                                            'type': 'substrate.type'}, dict_name='epoch')
        sl3 = gui_aux.SelectionList(tab=self, buttons=['load', 'save', 'delete', 'exec', 'tree', 'conf_tree'], progress=True, root_key='Exp',
                                    sublists={'env_params': sl1, 'larva_groups': s1}, text_kws=gui_aux.t_kws(10), width=28)
        sl4 = gui_aux.SelectionList(tab=self, conftype='ExpGroup', disp='Behavior/field :', buttons=[], single_line=True,
                                    width=15, text_kws=gui_aux.t_kws(12), sublists={'simulations': sl3})

        c1 = gui_aux.PadDict('sim_params', disp_name='Configuration', text_kws= gui_aux.t_kws(10), header_width=30, **kws)
        c2 = gui_aux.PadDict('output', text_kws= gui_aux.t_kws(7), Ncols=2, header_width=30, **kws)

        ll3 = gui_aux.gui_col([c1, c2, s2], x_frac=0.25, as_pane=True)
        l1 = [tab1_l[0]+[ll3]]
        # l1 = [self.envtab.layout[0]+[ll3]]

        c = {}
        for i in [c1, c2, s2, s1]:
            c.update(i.get_subdicts())
        c.update(**tab1_c)
        l0 = gui_aux.gui_col([sl4, sl3, s1, c['arena']], x_frac=0.2, as_pane=True, pad=(10, 10))
        return l0, l1, c, tab1_g, tab1_d

    def update(self, w, c, conf, id):

        output_keys=list(reg.output_dict.keys())
        c['output'].update(w, dict(zip(output_keys, [True if k in conf['collections'] else False for k in output_keys])))
        sim = copy.deepcopy(conf['sim_params'])
        sim.update({'sim_ID': f'{id}_{reg.next_idx(id=id, conftype="Exp")}', 'path': f'single_runs/{id}'})
        c['sim_params'].update(w, sim)
        c['trials'].update(w, reg.loadConf(id=conf['trials'], conftype='Trial'))
        self.draw_tab.set_env_db(env=reg.expandConf(id=conf['env_params'], conftype='Env'), lg=conf['larva_groups'])
        w.write_event_value('RESET_ARENA', 'Draw the initial arena')

    def get(self, w, v, c, as_entry=True):
        conf = {

            'sim_params': c['sim_params'].get_dict(v, w),
            'collections': [k for k in list(reg.output_dict.keys()) if c['output'].get_dict(v, w)[k]],
            # 'collections': [k for k in output_keys if c['output'].get_dict(v, w)[k]],
            'enrichment': self.current_conf(v)['enrichment'],
            'trials': c['trials'].get_dict(v, w),
        }
        return conf

    def run(self, v, w, c, d, g, conf, id):
        conf['env_params'] = self.envtab.get(w, v, c, as_entry=False)
        N = conf['sim_params']['duration'] * 60 / conf['sim_params']['timestep']
        p = self.base_list.progressbar
        p.run(w, max=N)
        conf['experiment'] = id
        conf['vis_kwargs'] = self.gui.get_vis_kwargs(v)
        self.active_id = conf['sim_params']['sim_ID']
        exec = sim.Exec(mode='sim', conf=conf, progressbar=p, w_progressbar=w[p.k],
                    run_externally=self.gui.run_externally['sim'])
        self.DL0.add(w, {self.active_id: exec})
        exec.run()

        return d, g

    def eval(self, e, v, w, c, d, g):
        if e==self.DL1.key :
            ks=v[self.DL1.key]
            if len(ks)>0:
                self.graph_list.update(w, self.DL1.dict[ks[0]]['figs'])

        # print(self.base_list.progressbar.k_incomplete)
        # active_ids = v[self.k_active_ids]
        # if e == f'STOP {self.k_active}':
        #     for act_id in active_ids:
        #         # self.DL0.dict[act_id]['process'].kill()
        #         self.DL0.dict[act_id].terminate()
        #     self.DL0.remove(w, active_ids)

        if self.base_list.progressbar is not None and e == self.base_list.progressbar.k_incomplete:
            pass
            # self.DL0.dict[self.active_id].terminate()
        self.draw_tab.eval(e, v, w, c, d, g)
        # self.env_tab.eval(e, v, w, c, d, g)
        self.check_subprocesses(w)



    def check_subprocesses(self, w):
        complete = []
        for sim_id, ex in self.DL0.dict.items():
            if ex.check():
                entries, fig_dict = ex.results
                if entries is not None:
                    ex.progressbar.done(w)

                    self.graph_list.update(w, fig_dict)
                    self.DL1.add(w, entries)
                else:
                    ex.progressbar.reset(w)
                complete.append(sim_id)
        self.DL0.remove(w, complete)


if __name__ == "__main__":
    from larvaworld.gui.tabs.larvaworld_gui import LarvaworldGui
    # larvaworld_gui = LarvaworldGui()
    larvaworld_gui = LarvaworldGui(tabs=['sim'])
    larvaworld_gui.run()
