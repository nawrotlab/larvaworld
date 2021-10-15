import copy
import PySimpleGUI as sg

from lib.aux.collecting import output_keys
from lib.gui.aux.elements import CollapsibleDict, GraphList, SelectionList, DataList, CollapsibleTable
from lib.gui.aux.functions import t_kws, gui_col, gui_cols
from lib.gui.tabs.draw_tab import DrawTab
from lib.gui.tabs.env_tab import EnvTab
from lib.gui.tabs.tab import GuiTab
from lib.conf.stored.conf import next_idx, expandConf, loadConf
from run.exec_run import Exec


class SimTab(GuiTab):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.canvas_size = (800, 800)
        self.k_stored = f'{self.name}_stored'
        self.k_active = f'{self.name}_active'
        self.k_stored_ids = f'{self.k_stored}_IDS'
        self.k_active_ids = f'{self.k_active}_IDS'

    @property
    def DL0(self):
        return self.datalists[self.k_active]

    @property
    def DL1(self):
        return self.datalists[self.k_stored]

    def build(self):
        l1, c1, g1, d1 = self.build_conf()
        l2, c2, g2, d2 = self.build_RUN()
        self.draw_tab = DrawTab(name='draw', gui=self.gui)
        l3, c3, g3, d3 = self.draw_tab.build()

        tab_kws = {'font': ("Helvetica", 13, "normal"), 'selected_title_color': 'darkblue', 'title_color': 'grey',
                   'tab_background_color': 'lightgrey'}

        ts = [sg.Tab(n, nl, key=f'{n}_SIM TAB') for n, nl in zip(['DRAW', 'RUN'], [l3, l2])]
        l_tabs = sg.TabGroup([ts], key='ACTIVE_SIM_TAB', tab_location='topleft', **tab_kws)
        l = [[l1[0][0], l_tabs]]
        return l, {**c1, **c2, **c3}, {**g1, **g2, **g3}, {**d1, **d2, **d3}

    def build_RUN(self):
        kA, kS = self.k_active, self.k_stored
        d = {kA: {}, kS: {}}
        g1 = GraphList(self.name, tab=self, canvas_size=self.canvas_size)
        dl1 = DataList(kA, dict=d[kA], tab=self, buttons=['select_all', 'stop'], disp='Active simulations')
        dl2 = DataList(kS, dict=d[kS], tab=self, buttons=['select_all', 'remove'], disp='Completed simulations')
        l = gui_cols(cols=[[g1.canvas], [dl1, dl2, g1]], x_fracs=[0.5, 0.2])
        return l, {}, {g1.name: g1}, d

    def build_conf(self):
        s1 = CollapsibleTable('larva_groups', buttons=['add', 'remove'], index='Group ID', col_widths=[10, 4, 8, 12],
                              heading_dict={'N': 'distribution.N', 'color': 'default_color', 'model': 'model'},
                              dict_name='LarvaGroup', state=True)
        tab1 = EnvTab(name='environment', gui=self.gui, conftype='Env')
        tab1_l, tab1_c, tab1_g, tab1_d = tab1.build()
        sl1 = tab1.selectionlists[tab1.conftype]

        s2 = CollapsibleTable('trials', buttons=['add', 'remove'], index='idx', col_widths=[3, 5, 5, 6, 10],
                              heading_dict={'start': 'start', 'stop': 'stop', 'quality': 'substrate.quality',
                                            'type': 'substrate.type'},
                              dict_name='epoch', state=True)
        sl3 = SelectionList(tab=self, buttons=['load', 'save', 'delete', 'run'], progress=True,
                            sublists={'env_params': sl1, 'larva_groups': s1})
        sl4 = SelectionList(tab=self, conftype='ExpGroup', disp='Simulation type :', buttons=[],single_line=True,
                            width=16, sublists={'simulations': sl3})

        c1 = CollapsibleDict('sim_params', disp_name='Configuration')
        c2 = CollapsibleDict('output')
        l = gui_cols(cols=[[sl4, sl3, s1, c1, c2, s2, tab1]], x_fracs=[0.3])

        c = {}
        for i in [c1, c2, s2, s1]:
            c.update(i.get_subdicts())
        c.update(**tab1_c)
        return l, c, {}, {}

    def update(self, w, c, conf, id):
        c['output'].update(w, dict(zip(output_keys, [True if k in conf['collections'] else False for k in output_keys])))
        sim = copy.deepcopy(conf['sim_params'])
        sim.update({'sim_ID': f'{id}_{next_idx(id)}', 'path': f'single_runs/{id}'})
        c['sim_params'].update(w, sim)
        c['trials'].update(w, loadConf(conf['trials'], 'Trial'))
        self.draw_tab.set_env_db(env=expandConf(conf['env_params'], 'Env'), lg=conf['larva_groups'])
        w.write_event_value('RESET_ARENA', 'Draw the initial arena')

    def get(self, w, v, c, as_entry=True):
        conf = {
            'sim_params': c['sim_params'].get_dict(v, w),
            'collections': [k for k in output_keys if c['output'].get_dict(v, w)[k]],
            'enrichment': self.current_conf(v)['enrichment'],
            'trials': c['trials'].get_dict(v, w),
        }
        return conf

    def run(self, v, w, c, d, g, conf, id):
        N = conf['sim_params']['duration'] * 60 / conf['sim_params']['timestep']
        p = self.base_list.progressbar
        p.run(w, max=N)
        conf['experiment'] = id
        conf['vis_kwargs'] = self.gui.get_vis_kwargs(v)
        self.active_id = conf['sim_params']['sim_ID']
        exec = Exec(mode='sim', conf=conf, progressbar=p, w_progressbar=w[p.k],
                    run_externally=self.gui.run_externally['sim'])
        self.DL0.add(w, {self.active_id: exec})
        exec.run()

        return d, g

    def eval(self, e, v, w, c, d, g):
        # print(e)
        # print(self.base_list.progressbar.k_incomplete)
        if e == self.base_list.progressbar.k_incomplete:
            pass
            # self.DL0.dict[self.active_id].terminate()
        self.draw_tab.eval(e, v, w, c, d, g)
        self.check_subprocesses(w)

    def check_subprocesses(self, w):
        complete = []
        for sim_id, ex in self.DL0.dict.items():
            if ex.check():
                entry, fig_dict = ex.results
                if entry is not None:
                    ex.progressbar.done(w)

                    self.graph_list.update(w, fig_dict)
                    self.DL1.add(w, entry)
                else:
                    ex.progressbar.reset(w)
                complete.append(sim_id)
        self.DL0.remove(w, complete)


if __name__ == "__main__":
    from lib.gui.tabs.gui import LarvaworldGui

    larvaworld_gui = LarvaworldGui(tabs=['simulation', 'settings'])
    larvaworld_gui.run()
