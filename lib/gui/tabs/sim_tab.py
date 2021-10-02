import copy

from lib.aux.collecting import output_keys
from lib.gui.aux.elements import CollapsibleDict, GraphList, SelectionList, DataList
from lib.gui.aux.functions import t_kws, gui_col, gui_cols
from lib.gui.tabs.tab import GuiTab
from lib.conf.conf import next_idx
from run.exec_run import Exec
from lib.stor import paths


class SimTab(GuiTab):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.k_stored = f'{self.name}_stored'
        self.k_active = f'{self.name}_active'
        self.k_stored_ids = f'{self.k_stored}_IDS'
        self.k_active_ids = f'{self.k_active}_IDS'

    @property
    def DL0(self):
        return self.datalists[self.k_active]

    @ property
    def DL1(self):
        return self.datalists[self.k_stored]

    def build(self):
        kA,kS=self.k_active, self.k_stored
        d = {kA: {}, kS: {}}

        sl1 = SelectionList(tab=self, conftype='Env', idx=1)
        sl2 = SelectionList(tab=self, conftype='Life', idx=1, with_dict=True, header_value='default',
                           text_kws=t_kws(14), value_kws=t_kws(10), width=12, header_text_kws=t_kws(9))
        sl3 = SelectionList(tab=self, buttons=['load', 'save', 'delete', 'run'], progress=True,
                            sublists={'env_params': sl1, 'life_params' : sl2})
        sl4 = SelectionList(tab=self, conftype='ExpGroup', disp='Simulation types', buttons=[], sublists={'simulations': sl3})
        c1 = CollapsibleDict('sim_params', disp_name='Configuration')
        c2 = CollapsibleDict('output')
        g1 = GraphList(self.name, tab=self)
        dl1 = DataList(kA, dict=d[kA], tab=self, buttons=['select_all', 'stop'], disp= 'Active simulations')
        dl2 = DataList(kS, dict=d[kS], tab=self, buttons=['select_all', 'remove'], disp= 'Completed simulations')

        l = gui_cols(cols=[[sl4, sl3, sl1,c1, c2, sl2], [g1.canvas], [dl1, dl2, g1]], x_fracs=[0.25, 0.55, 0.2])

        c = {}
        for i in [c1, c2, sl2]:
            c.update(i.get_subdicts())
        return l, c, {g1.name: g1}, d

    def update(self, w,  c, conf, id):
        output_dict = dict(zip(output_keys, [True if k in conf['collections'] else False for k in output_keys]))
        c['output'].update(w, output_dict)
        sim=copy.deepcopy(conf['sim_params'])
        sim.update({'sim_ID' : f'{id}_{next_idx(id)}', 'path' : f'single_runs/{id}'})
        c['sim_params'].update(w, sim)

    def get(self, w, v, c, as_entry=True):
        conf = {
                'sim_params': c['sim_params'].get_dict(v, w),
                # 'life_params': c['sim_params'].get_dict(v, w),
                'collections': [k for k in output_keys if c['output'].get_dict(v, w)[k]],
                'enrichment': self.current_conf(v)['enrichment'],
                }
        return conf

    def run(self, v, w,c,d,g, conf,id):
        N=conf['sim_params']['duration'] * 60 / conf['sim_params']['timestep']
        p=self.base_list.progressbar
        p.run(w, max=N)
        conf['experiment'] = id
        conf['vis_kwargs'] = self.gui.get_vis_kwargs(v)
        sim_id=conf['sim_params']['sim_ID']
        exec = Exec(mode='sim', conf=conf, progressbar = p,w_progressbar = w[p.k], run_externally=self.gui.run_externally['sim'])
        self.DL0.add(w, {sim_id: exec})
        exec.run()

        return d,g


    def eval(self, e, v, w, c, d, g):
        self.check_subprocesses(w)

    def check_subprocesses(self, w):
        complete = []
        for sim_id, ex in self.DL0.dict.items():
            if ex.check():
                entry, fig_dict = ex.results
                if entry is not None:
                    w[ex.progressbar.k_complete].update(visible=True)
                    self.graph_list.update(w, fig_dict)
                    self.DL1.add(w, entry)
                else:
                    ex.progressbar.reset(w)
                complete.append(sim_id)
        self.DL0.remove(w, complete)


if __name__ == "__main__":
    from lib.gui.tabs.gui import LarvaworldGui
    larvaworld_gui = LarvaworldGui(tabs=['simulation', 'analysis','settings'])
    larvaworld_gui.run()
